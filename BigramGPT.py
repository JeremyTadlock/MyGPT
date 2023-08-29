import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import PreTrainedModel
#from transformers.modeling_outputs import CausalLMOutput


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

    def generate(self, idx, max_new_tokens):
        #  idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
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

            # Add sample to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx