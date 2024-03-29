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
import optuna

batch_size = 32  # Number of sequences processed in parallel
block_size = 128  # Max content length for predictions
max_iters = 2400  # was 3000 with lr=1e-2
eval_interval = 100  # how often we check train/val loss and generate autocompleted tokens.
learning_rate = 1e-4  # was 1e-2 then 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # try to use pytorch's CUDA for GPU parallel processing
eval_iters = 200
num_embeddings = 384  # this number was chosen because 384/6 = 64 (standard)
num_heads = 16
num_layers = 16
bpe_vocab_size = 25000
accumulation_steps = 4
loss_tracker = []
# dropout is a way to prevent overfitting in large neural networks. it works by having every forward-backward pass
# randomly shut off a subset of neurons(set to 0). It basically splits training into multiple sub-networks then
# re-merges them at testing time.
dropout = 0.2  # link here: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

def get_best_hyperparameters_bayesian_optimization(trial):
    # Before i implement this i need to package the tokenizer training neatly into a class/def, then do the same for
    # pretrianing.
    test_batch_size = 32
    test_block_size = 128
    test_max_iters = 1800
    test_eval_interval = 200
    test_learning_rate = 1e-4
    test_eval_iters = 200
    test_num_embeddings = 384  # this number was chosen because 384/6 = 64 (standard)
    test_num_heads = 16
    test_num_layers = 16
    test_accumulation_steps = 4
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


# load data
def load_batch(split):
    data = train_data if split == 'train' else val_data
    data_len = len(data)
    ix = torch.randint(data_len - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix if i + block_size <= data_len])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix if i + block_size + 1 <= data_len])
    x, y = x.to(device), y.to(device)
    return x, y


# gets rid of noise when getting loss. Averages the splits instead of randomly sampling(which creates noise)
# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = load_batch(split)
#             batch_losses = torch.zeros(batch_size)  # To store loss for each sample in the batch
#             for i in range(batch_size):  # <-- where monte carlo sampling comes in
#                 # Generate a single sample (sequence) for each input in the batch
#                 logits, loss = model(X[i:i + 1], Y[i:i + 1])  # Use X[i:i+1] to keep the dimensions
#                 batch_losses[i] = loss.item()
#             losses[k] = batch_losses.mean()  # Average the losses for the batch samples
#         out[split] = losses.mean()  # Average the losses over the iterations
#     model.train()
#     return out


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

    def generate_top_p(self, idx, max_new_tokens, eos_token_id=-1, top_p=0.9, temperature=1.0):
        is_first_iteration = True
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # Crop idx to the last block_size tokens if necessary
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :] / temperature  # Apply temperature
            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)  # Also (B, C)

            if not is_first_iteration:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Find indices where cumulative probability exceeds top_p
                indices_to_remove = cumulative_probs > top_p
                indices_to_remove[:, 1:] = indices_to_remove[:, :-1].clone()
                indices_to_remove[:, 0] = False

                # Mask out tokens
                sorted_probs[indices_to_remove] = 0
                probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # Sample from the modified distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

                # Map sampled indices back to the original indices
                idx_next = torch.gather(sorted_indices, -1, idx_next)
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
    files = ["../input_data_files/Shellcode_IA32.txt", "../input_data_files/Extendend_Shellcode_IA32.txt"]
    for file in files:
        with open(file, "r", encoding="utf8") as f:
            dataset = dataset + f.read() + '\n'

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

    chunk_size = 60000  # Adjust the chunk size according to your memory capacity
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
    val_data = torch.cat(val_chunks)
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
        print(iter)
        # Reset gradients at the beginning of the loop
        optimizer.zero_grad(set_to_none=True)

        for _ in range(accumulation_steps):
            # Sample a batch of data
            xb, yb = load_batch('train')

            # Forward pass under autocast
            with autocast():
                logits, loss = model(xb, yb)
                # Normalize the loss to account for accumulation
                loss = loss / accumulation_steps

            # Backward pass (accumulate gradients)
            scaler.scale(loss).backward()

        # Step with optimizer
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # Ensure optimizer is zeroed at the end

        # Periodically evaluate the model and print out the loss and generated text
        if iter % eval_interval == 0 or iter == max_iters - 1:
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
    torch.save(model.state_dict(), '../models/shellcode_v2.7.pth')

    # Create target directory & all intermediate directories if don't exists
    # Then save the encoder
    encoder_dir = '../encoder_directory'
    tokenizer_name = 'shellcode_v2.7'
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
