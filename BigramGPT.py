import torch
import torch.nn as nn
from torch.nn import functional as F
import gpt_tokenizers
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import bisect
from itertools import accumulate

batch_size = 64  # Number of sequences processed in parallel
block_size = 256  # Max content length for predictions
max_iters = 6000  # was 3000 with lr=1e-2
eval_interval = 1000  # how often we check train/val loss and generate autocompleted tokens.
learning_rate = 1e-3  # was 1e-2 then 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # try to use pytorch's CUDA for GPU parallel processing
print("device:", device)
eval_iters = 200
num_embeddings = 384  # this number was chosen because 384/6 = 64 (standard)
num_heads = 6
num_layers = 6
bpe_vocab_size = 1000

# dropout is a way to prevent overfitting in large neural networks. it works by having every forward-backward pass
# randomly shut off a subset of neurons(set to 0). It basically splits training into multiple sub-networks then
# re-merges them at testing time.
dropout = 0.2  # link here: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf


# load data
def load_batch(split):
    data_loader = train_loader if split == 'train' else val_loader
    for batch in data_loader:

        x, y = batch

        x = torch.stack(x)

        y = torch.stack(y)
        x, y = x.to(device), y.to(device)
        yield x, y

# gets rid of noise when getting loss. Averages the splits instead of randomly sampling(which creates noise)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    print("looping splits")
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(load_batch(split))
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    print("done looping splits")
    model.train()
    return out


class TextFileDataset(Dataset):
    def __init__(self, file_paths, block_size, byte_pair_encoder, chunk_size=1024 * 1024):
        self.file_paths = file_paths
        self.block_size = block_size
        self.byte_pair_encoder = byte_pair_encoder
        self.chunk_size = chunk_size
        self.lengths = []

        for file_path in file_paths:
            print(f"Processing {file_path}")
            total_length = 0
            for chunk in self.read_file_in_chunks(file_path):
                encoded_data = self.byte_pair_encoder.encode(chunk)
                total_length += len(encoded_data)
            self.lengths.append(total_length)

        self.cumulative_lengths = [0] + list(accumulate(self.lengths))

    def __len__(self):
        return sum(self.lengths) - self.block_size * len(self.file_paths)

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        file_offset = idx - self.cumulative_lengths[file_idx]
        file_path = self.file_paths[file_idx]

        encoded_data = []
        with open(file_path, "r", encoding="utf8") as f:
            while len(encoded_data) < file_offset + self.block_size + 1:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                encoded_data.extend(self.byte_pair_encoder.encode(chunk))

        seq = encoded_data[file_offset:file_offset + self.block_size + 1]
        return seq[:-1], seq[1:]

    def read_file_in_chunks(self, file_path):
        with open(file_path, "r", encoding="utf8") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk


def collate_fn(batch):
    lengths = torch.tensor([len(seq) for seq in batch])
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return padded_batch, lengths


'''
# Original loss estimation code. not as accurate as it doesnt use the monte carlo method, but it runs faster.
@torch.no_grad()  # tell's pytorch we AREN'T using backpropagation, saving memory
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = load_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
'''
# Self-Attention model
# The reason this is self-attention is because the keys/queries/values all come from the same source
# (source = x) <- see forward function

# the line where we apply masking to the weights is what makes this self-attention model a decoder.
# not allowing it to communicate with future nodes.

# if we wanted to perform something like sentiment analysis, where nodes need to all talk to each
# other(including future nodes) then we can simply delete the line where we use masking w/ tril
class Head(nn.Module):
    # ONE head of self-attention

    def __init__(self, head_size, num_embeddings, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input = (Batch, Time-Step, Channels)
        # output = (Batch, Time-Step, Head size)
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores using openAI's described "Scaled Dot-Product attention" formula
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,hs) @ (B,hs,T) = (B,T,T)

        # we only want current nodes using themselves and past nodes to make guesses for future nodes.
        # we DONT want current nodes to talk to future nodes for info. we use pytorch's tril to achieve this.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)

        weights = F.softmax(weights, dim=-1)  # (B,T,T)

        weights = self.dropout(weights)

        # perform weighted aggregation of values
        v = self.value(x)  # (B,T,C)
        out = weights @ v  # (B,T,T) @ (B,T,C) = (B,T,C)
        return out


# Run multiple heads in parallel
class MultipleHeads(nn.Module):

    def __init__(self, num_heads, head_size, num_embeddings, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, num_embeddings, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # concatenating over the channel dimension(C)
        out = self.dropout(self.projection(out))  # apply projection
        return out


# feedforward consisting of a linear layer, follow by a ReLu nonlinear function
class FeedForward(nn.Module):

    def __init__(self, num_embeddings, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),  # the default activation when developing our multilayer Perceptron
            nn.Linear(4 * num_embeddings, num_embeddings),  # projection layer going back into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer block: communication followed by computation
class Block(nn.Module):

    def __init__(self, num_embeddings, num_heads, block_size, dropout):
        super().__init__()
        head_size = num_embeddings // num_heads
        self.sa = MultipleHeads(num_heads, head_size, num_embeddings, block_size, dropout)
        self.ffwd = FeedForward(num_embeddings, dropout)

        # Pytorch's pre-norm formulation
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Bigram language model
class BigramLanguageModel(nn.Module):

    # dropout is a way to prevent overfitting in large neural networks. it works by having every forward-backward pass
    # randomly shut off a subset of neurons(set to 0). It basically splits training into multiple sub-networks then
    # re-merges them at testing time.
    # link here: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

    #block_size: Max content length for predictions
    #num_embeddings: this number was chosen because 384/6 = 64 (standard)
    def __init__(self, bpe_vocab_size=1000, num_embeddings=384, block_size=256, num_heads=6, num_layers=6, dropout=0.2):
        super().__init__()

        # each token reads off the logits for the next token using lookup table
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(bpe_vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.blocks = nn.Sequential(*[Block(num_embeddings, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(num_embeddings)
        self.lm_head = nn.Linear(num_embeddings, bpe_vocab_size)

    # forward feeding
    def forward(self, input_ids: torch.Tensor = None, labels=None, attention_mask=None):
        # The attention mask isnt used here, but is needed to not crash
        B, T = input_ids.shape
        device = input_ids.device

        # idx and targets are (B,T) tensors of integers
        token_embeddings = self.token_embedding_table(input_ids)  # (B,T,embeddings)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_embeddings + positional_embeddings  # encode info w/ tok & pos embeddings(B,T,C)
        x = self.blocks(x)  # apply multiple heads of self-attention(feed x into head). (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            labels = labels.view(B * T)
            loss = F.cross_entropy(logits, labels)

        return loss, logits  # Swapped these because thats the way hf expects the output.
        # return CausalLMOutput(
        #         loss=loss,
        #         logits=logits
        #     )

    def generate(self, idx, max_new_tokens, eos_token_id=417):
        #  idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens if necessary
            if idx.size(1) > block_size:
                idx = idx[:, :block_size]

            # get predictions
            logits, loss = self(idx)
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # get predictions
            loss, logits = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # Transforms from (B, T) to (B, C)

            # applying softmax to get probablilities
            probs = F.softmax(logits, dim=-1)  # Also (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Check if the EOS token was generated
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

            # Add the newly generated token to the sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


if __name__ == '__main__':
    # Grab data from datasets
    print("Loading dataset")
    train_files = ["input_data_files\\train_TinyStoriesV2-GPT4-valid.txt"]
    val_files = ["input_data_files\\valid_TinyStoriesV2-GPT4-valid.txt"]

    # Statistics about imported dataset(s)
    # print(len(dataset))
    # chars = sorted(list(set(dataset)))
    # print("token size char:", len(chars))

    # Use Byte-Pair Encoder
    print("training BPE")
    byte_pair_encoder = gpt_tokenizers.BytePairEncoder(bpe_vocab_size, 2)
    byte_pair_encoder.train(train_files + val_files)

    # Use Character decoder
    # use character_tokenizer.decode and character_tokenizer.encode for encoding/decoding
    # character_tokenizer_vocab_size = len(chars)  # Vocab size specifically used with character tokenizer
    # character_tokenizer = gpt_tokenizers.CharacterTokenizer(chars)

    # Use custom SentencePiece tokenizer
    # sp_vocab_size = 5000
    # sentencepiece_tokenizer = gpt_tokenizers.SentencePieceTokenizer(vocab_size=sp_vocab_size)
    # sentencepiece_tokenizer.fit(dataset)

    # Use Google SentencePiece
    # if you have multiple text files for your dataset, you can do something like:
    # data='openai_generated_text.txt, file2.txt, file3.txt' etc.
    # google_sentencepiece_tokenizer = gpt_tokenizers.SentencePieceTokenizerGoogle(vocab_size=sp_vocab_size,
    # data='openai_generated_text.txt')

    # Create datasets and data loaders
    print("Importing datasets")
    train_dataset = TextFileDataset(train_files, block_size, byte_pair_encoder)
    val_dataset = TextFileDataset(val_files, block_size, byte_pair_encoder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Loading model")
    model = BigramLanguageModel()
    print("move to cuda")
    m = model.to(device)  # CUDA!!1!1

    # Model's parameter count
    print("Loading hyperparameter count")
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # using Pytorch's adamW optimizer
    print("Loading optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    accumulation_steps = 4
    scaler = GradScaler()
    print("Looping over iterations")
    for iter in range(max_iters):
        print(iter)
        # Reset gradients at the beginning of the loop
        optimizer.zero_grad(set_to_none=True)

        for step in range(accumulation_steps):
            print("accumulation step:", step)
            # Sample a batch of data
            batch = next(load_batch('train'))
            xb, yb = batch

            # Forward pass under autocast
            with autocast():
                logits, loss = model(xb, yb)
                # Normalize the loss to account for accumulation
                loss = loss / accumulation_steps

            # Backward pass (accumulate gradients)
            scaler.scale(loss).backward()

        # Step with optimizer
        print("step optimizer")
        scaler.step(optimizer)
        print("update scaler")
        scaler.update()
        print("zero optimizer")
        optimizer.zero_grad(set_to_none=True)  # Ensure optimizer is zeroed at the end


        # Periodically evaluate the model and print out the loss and generated text
        if iter % eval_interval == 0 and iter != 0 or iter == max_iters - 1:
            print("estimating loss")
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated_text = m.generate(context, max_new_tokens=2500)[0].tolist()
            decoded_text = byte_pair_encoder.decode(generated_text)
            print(decoded_text)
            print("------------------------------------------------------")

    # Save pretrained model
    torch.save(model.state_dict(), 'Tinystories_last_ten_percent_4000epochsv2.pth')

    # Create target directory & all intermediate directories if don't exists
    # Then save the encoder
    encoder_dir = 'encoder_directory'
    tokenizer_name = 'Tinystories_last_ten_percent_4000epochsv2'
    byte_pair_encoder.save(encoder_dir, tokenizer_name)

    # Generate from the model
    print("GENERATING SAMPLE TEXT")
    for _ in range(5):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(byte_pair_encoder.decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
        print("------------------------------------------------------")
