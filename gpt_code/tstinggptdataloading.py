import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import gpt_tokenizers
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import mmap

batch_size = 64  # Number of sequences processed in parallel
block_size = 128  # Max content length for predictions
max_iters = 900  # was 3000 with lr=1e-2
eval_interval = 150  # how often we check train/val loss and generate autocompleted tokens.
learning_rate = 1e-3  # was 1e-2 then 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # try to use pytorch's CUDA for GPU parallel processing
eval_iters = 200
num_embeddings = 384  # this number was chosen because 384/6 = 64 (standard)
num_heads = 16
num_layers = 16
bpe_vocab_size = 25000
accumulation_steps = 1
loss_tracker = []
# dropout is a way to prevent overfitting in large neural networks. it works by having every forward-backward pass
# randomly shut off a subset of neurons(set to 0). It basically splits training into multiple sub-networks then
# re-merges them at testing time.
dropout = 0.2  # link here: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

dataset_loading_list = ["normal", "mmap", "chunking"]
dataset_loading_type = dataset_loading_list[0]

def get_model_info():
    # convert list to string
    loss_tracker_str = '\n'.join(str(loss) for loss in loss_tracker)

    # Gather model info in a string
    model_info = f"""
    Model Name: Tinystories_CustomStories
    Hyperparameter Count: {hyperparameter_count}
    Batch Size: {batch_size}
    Block Size: {block_size}
    Max Iterations: {max_iters}
    Eval Interval: {eval_interval}
    Learning Rate: {learning_rate}
    Device: {device}
    Eval Iters: {eval_iters}
    Num Embeddings: {num_embeddings}
    Num Heads: {num_heads}
    Num Layers: {num_layers}
    BPE Vocab Size: {bpe_vocab_size}
    Accumulation Steps: {accumulation_steps}
    Dropout: {dropout}
    Training time: {runtime}
    loss tracker: {loss_tracker_str}
    """
    return model_info.strip()  # Remove leading/trailing newlines

def load_dataset_chunking(file_path, chunk_size):
    with open(file_path, "r", encoding="utf8") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

def load_dataset_mmap(file_path):
    with open(file_path, "rb") as file:
        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_file:
            data = mmap_file.read().decode("utf-8")
            for line in data.split("\n"):
                if line:
                    yield line


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + block_size]
        y = self.data[idx + 1:idx + block_size + 1]
        return x, y


# load data

def load_batch(split):
    loader = train_loader if split == 'train' else val_loader
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yield x, y


# gets rid of noise when getting loss. Averages the splits instead of randomly sampling(which creates noise)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(len(loader))
        for k, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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

    def __init__(self, head_size):
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

        # we only want current nodes using themselves and past nodes to make guesses fofuture nodes.
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

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # concatenating over the channel dimension(C)
        out = self.dropout(self.projection(out))  # apply projection
        return out


# feedforward consisting of a linear layer, follow by a ReLu nonlinear function
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),

            # NOTE: look into replacing with GELU or preferably Mish (Mish is an evolution of GELU that is less
            # computationally expensive and relatively smoother)
            nn.ReLU(),  # using ReLU as the default activation

            nn.Linear(4 * n_embd, n_embd),  # projection layer going back into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer block: communication followed by computation
class Block(nn.Module):

    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultipleHeads(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)

        # Pytorch's pre-norm formulation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # each token reads off the logits for the next token using lookup table
        self.token_embedding_table = nn.Embedding(bpe_vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.blocks = nn.Sequential(*[Block(num_embeddings, num_heads=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(num_embeddings)
        self.lm_head = nn.Linear(num_embeddings, bpe_vocab_size)

    # forward feeding
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are (B,T) tensors of integers
        token_embeddings = self.token_embedding_table(idx)  # (B,T,embeddings)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_embeddings + positional_embeddings  # encode info w/ tok & pos embeddings(B,T,C)
        x = self.blocks(x)  # apply multiple heads of self-attention(feed x into head). (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, eos_token_id=-1):
        #  idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens if necessary
            if idx.size(1) > block_size:
                idx = idx[:, :block_size]

            # get predictions
            logits, loss = self(idx)

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

    def generate_top_k(self, idx, max_new_tokens, eos_token_id=-1, top_k=50):

        is_first_iteration = True
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # Crop idx to the last block_size tokens if necessary
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :]  # Transforms from (B, T, C) to (B, C)

            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)  # Also (B, C)

            if not is_first_iteration:
                # Top-K filtering
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1, sorted=True)
                top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)  # Re-normalize

                # Sample from the top k
                idx_next = torch.multinomial(top_k_probs, num_samples=1)  # (B, 1)

                # Map sampled indices back to the original indices
                idx_next = torch.gather(top_k_indices, -1, idx_next)
            else:
                is_first_iteration = False
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Check for the end of sequence token
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

            # Concatenate the next index
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


if __name__ == '__main__':
    start_time = time.time()
    print("device:", device)

    # Grab data from datasets
    print("Loading dataset")
    dataset = ""
    # files = ["../input_data_files/TinyStoriesV2-GPT4-train.txt", "../input_data_files/TinyStoriesV2-GPT4-valid.txt"]
    files = ["../input_data_files/Shellcode_IA32.txt", "../input_data_files/Extendend_Shellcode_IA32.txt"]
    for file in files:
        if dataset_loading_type == "normal":
            with open(file, "r", encoding="utf8") as f:
                dataset = dataset + f.read() + '\n'
        elif dataset_loading_type == "mmap":
            line_count = sum(1 for line in load_dataset_mmap(file))
            print(f"File: {file}, Line count: {line_count}")
            line_counter = 0
            prev_percentage = 0
            test_st = time.time()
            for line in load_dataset_mmap(file):
                line_counter += 1
                dataset += line

                # Loading stats
                completion_percentage = (line_counter / line_count) * 100
                if completion_percentage >= prev_percentage + 1:
                    test_et = time.time()
                    print("% completion time:", test_et - test_st)
                    print(f"{int(completion_percentage)}% completed")
                    prev_percentage = int(completion_percentage)
        elif dataset_loading_type == "chunking":
            chunk_size = 1024 * 1024
            file_size = os.path.getsize(file)
            total_chunks = (file_size + chunk_size - 1) // chunk_size
            loaded_chunks = 0
            print_interval = total_chunks // 100  # Print progress every 1%

            for chunk in load_dataset_chunking(file, chunk_size):
                dataset += chunk
                loaded_chunks += 1

                if loaded_chunks % print_interval == 0:
                    progress = (loaded_chunks / total_chunks) * 100
                    print(f"Loading progress: {progress:.2f}%")


    # Statistics about imported dataset(s)
    print(len(dataset))
    chars = sorted(list(set(dataset)))
    print("token size char:", len(chars))

    # Use Byte-Pair Encoder
    print("training BPE")
    byte_pair_encoder = gpt_tokenizers.BytePairEncoder(bpe_vocab_size, 2)
    byte_pair_encoder.train(files)

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

    # Split input data into train/test data - uses a 90%/10% split
    print("Splitting data")

    chunk_size = 100000  # Adjust the chunk size according to your memory capacity
    train_chunks = []
    val_chunks = []

    total_chunks = (len(dataset) + chunk_size - 1) // chunk_size
    processed_chunks = 0
    print_interval = total_chunks // 10  # Print progress every 10%

    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i + chunk_size]
        encoded_chunk = byte_pair_encoder.encode(chunk)
        encoded_chunk_tensor = torch.tensor(encoded_chunk, dtype=torch.long)
        n = int(0.9 * len(encoded_chunk_tensor))
        train_chunks.append(encoded_chunk_tensor[:n])
        val_chunks.append(encoded_chunk_tensor[n:])

        processed_chunks += 1
        if processed_chunks % print_interval == 0:
            progress = (processed_chunks / total_chunks) * 100
            print(f"Loading progress: {progress:.2f}%")

    train_data = torch.cat(train_chunks)
    train_dataset = TextDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=16, prefetch_factor=64, pin_memory=True)
    val_data = torch.cat(val_chunks)
    val_dataset = TextDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=16, prefetch_factor=64, pin_memory=True)
    print("tokens in dataset", len(train_data) + len(val_data))
    print("Done splitting")

    print("Loading model")
    model = BigramLanguageModel()
    print("move to cuda")
    m = model.to(device)  # CUDA!!1!1

    # Model's parameter count
    print("Loading hyperparameter count")
    hyperparameter_count = (sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
    print(hyperparameter_count)

    # using Pytorch's adamW optimizer
    print("Loading optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scaler = GradScaler()  # Setting up GradScaler for mixed precision
    # Training loop
    print("Looping over iterations")
    for iter in range(max_iters):
        start_iter_time = time.time()
        print(iter)

        # Reset gradients at the beginning of the loop
        optimizer.zero_grad(set_to_none=True)

        for _ in range(accumulation_steps):
            print(accumulation_steps)
            step = 0
            total_steps = len(train_loader)  # Total number of steps in the training loader
            print(total_steps)
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                print("loading step:", step)
                step += 1
                with autocast():
                    logits, loss = model(xb, yb)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                # Print progress every 10%
                if (i + 1) % (total_steps // 10) == 0:  # Print every 10%
                    progress_percentage = ((i + 1) / total_steps) * 100
                    print(f"Progress: {progress_percentage:.2f}%")

        # Step with optimizer
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # Ensure optimizer is zeroed at the end
        end_iter_time = time.time()
        print("iteration time:", end_iter_time - start_iter_time)
        # Periodically evaluate the model and print out the loss and generated text
        if iter % eval_interval == 0 and iter != 0 or iter == max_iters - 1:
            losses = estimate_loss()
            log_string = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            loss_tracker.append(log_string)
            print(log_string)
            if iter != 0:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated_text = m.generate(context, max_new_tokens=2500)[0].tolist()
                decoded_text = byte_pair_encoder.decode(generated_text)
                print(decoded_text)
            print("------------------------------------------------------")

    print("saving model and info")
    # Save pretrained model
    torch.save(model.state_dict(), '../shellcode_gpt_example/shellcode_v2.6.pth')

    # Create target directory & all intermediate directories if don't exists
    # Then save the encoder
    encoder_dir = '../encoder_directory'
    tokenizer_name = 'shellcode_v2.6'
    byte_pair_encoder.save(encoder_dir, tokenizer_name)

    # Record runtime
    end_time = time.time()
    runtime = end_time - start_time

    # Create target directory for model_info if it doesn't exist
    model_info_path = "../model_info"
    if not os.path.exists(model_info_path):
        os.makedirs(model_info_path)

    # Call get_model_info to get the model configuration and parameters
    model_info = get_model_info()

    file_name = tokenizer_name + '.txt'
    model_info_path = "../model_info"
    file_path = os.path.join(model_info_path, file_name)
    with open(file_path, 'w') as file:
        file.write(model_info)

    # Generate from the model
    print("GENERATING SAMPLE TEXT")
    for _ in range(3):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(byte_pair_encoder.decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
        print("------------------------------------------------------")
