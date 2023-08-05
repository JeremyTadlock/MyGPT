import torch
import torch.nn as nn
from torch.nn import functional as F
import gpt_tokenizers
import os

batch_size = 64  # Number of sequences processed in parallel
block_size = 256  # Max content length for predictions
max_iters = 1  # was 3000 with lr=1e-2
eval_interval = 500  # how often we check train/val loss and generate autocompleted tokens.
learning_rate = 1e-3  # was 1e-2 then 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # try to use pytorch's CUDA for GPU parallel processing
eval_iters = 200
num_embeddings = 384  # this number was chosen because 384/6 = 64 (standard)
num_heads = 6
num_layers = 6
bpe_vocab_size = 7500

# dropout is a way to prevent overfitting in large neural networks. it works by having every forward-backward pass
# randomly shut off a subset of neurons(set to 0). It basically splits training into multiple sub-networks then
# re-merges them at testing time.
dropout = 0.2  # link here: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

# Grab data from datasets
dataset = ""
files = ["input_data_files/openai_generated_text.txt", "input_data_files/openai_generated_text_3000_0.txt",
         "input_data_files/openai_generated_text_3000_1.txt", "input_data_files/openai_generated_text_3000_spooky.txt"]
for file in files:
    with open(file, "r", encoding="utf8") as f:
        dataset = dataset + f.read() + '\n'

# Statistics about imported dataset(s)
print(len(dataset))
chars = sorted(list(set(dataset)))
print("token size char:", len(chars))

# Use Byte-Pair Encoder
byte_pair_encoder = gpt_tokenizers.BytePairEncoder(bpe_vocab_size, 2)
byte_pair_encoder.train(files)

# Use Character decoder
# use character_tokenizer.decode and character_tokenizer.encode for encoding/decoding
#character_tokenizer_vocab_size = len(chars)  # Vocab size specifically used with character tokenizer
#character_tokenizer = gpt_tokenizers.CharacterTokenizer(chars)

# Use custom SentencePiece tokenizer
sp_vocab_size = 5000
#sentencepiece_tokenizer = gpt_tokenizers.SentencePieceTokenizer(vocab_size=sp_vocab_size)
#sentencepiece_tokenizer.fit(dataset)

# Use Google SentencePiece
# if you have multiple text files for your dataset, you can do something like:
# data='openai_generated_text.txt, file2.txt, file3.txt' etc.
#google_sentencepiece_tokenizer = gpt_tokenizers.SentencePieceTokenizerGoogle(vocab_size=sp_vocab_size,
                                                                             #data='openai_generated_text.txt')




# Split input data into train/test data - uses a 90%/10% split
data = torch.tensor(byte_pair_encoder.encode(dataset), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# load data
def load_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# gets rid of noise when getting loss. Averages the splits instead of randomly sampling(which creates noise)
# Later I can replace this loss estimation with a potentially better one. Monte Carlo sampling.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = load_batch(split)
            batch_losses = torch.zeros(batch_size)  # To store loss for each sample in the batch
            for i in range(batch_size): # <-- where monte carlo sampling comes in
                # Generate a single sample (sequence) for each input in the batch
                logits, loss = model(X[i:i+1], Y[i:i+1])  # Use X[i:i+1] to keep the dimensions
                batch_losses[i] = loss.item()
            losses[k] = batch_losses.mean()  # Average the losses for the batch samples
        out[split] = losses.mean()  # Average the losses over the iterations
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
            nn.ReLU(),  # the default activation when developing our multilayer Perceptron
            nn.Linear(4 * n_embd, n_embd),  # projection layer going back into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer block: communication followed by computation
class Block(nn.Module):

    def __init__(self, n_embd, num_head):
        super().__init__()
        head_size = n_embd // num_head
        self.sa = MultipleHeads(num_head, head_size)
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
        self.blocks = nn.Sequential(*[Block(num_embeddings, num_head=num_heads) for _ in range(num_layers)])
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

    def generate(self, idx, max_new_tokens):
        #  idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # Transforms from (B, T) to (B, C)

            # applying softmax to get probablilities
            probs = F.softmax(logits, dim=-1)  # Also (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Add sample to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)  # CUDA!!1!1

# Model's parameter count
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# using Pytorch's adamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(iter)
    # For every 'eval_interval', evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(byte_pair_encoder.decode(m.generate(context, max_new_tokens=2500)[0].tolist()))
        print("------------------------------------------------------")

    # Sample a batch of data
    xb, yb = load_batch('train')

    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'BPE_8000iter_12mil-tokens_montecarlo_v1.pth')

# Create target directory & all intermediate directories if don't exists
# Then save the encoder
if not os.path.exists('encoder_directory'):
    os.makedirs('encoder_directory')

byte_pair_encoder.tokenizer.save_model('encoder_directory', 'encoder')

# this is the code to load that a model. use relavant .pth file name.
#model = BigramLanguageModel()
#m = model.to(device)
#m.load_state_dict(torch.load('BPE_5000_2_12m.pth'))
#m.eval()
# then generate like normal using m.generate. if it doesnt work, delete m and just use model with no cuda.

# Generate from the model
print("GENERATING SAMPLE TEXT")
for _ in range(10):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(byte_pair_encoder.decode(m.generate(context, max_new_tokens=2500)[0].tolist()))
    print("------------------------------------------------------")
