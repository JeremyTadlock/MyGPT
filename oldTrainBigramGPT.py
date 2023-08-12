

from BigramGPT import *

import torch
import torch.nn as nn
from torch.nn import functional as F
import gpt_tokenizers
# import os
# import json
# from datasets import load_dataset
# from torch.utils.data import Dataset, DataLoader


max_iters = 6500  # was 3000 with lr=1e-2
eval_interval = 500  # how often we check train/val loss and generate autocompleted tokens.

device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'  # try to use pytorch's CUDA for GPU parallel processing
eval_iters = 200
#num_embeddings this number was chosen because 384/6 = 64 (standard)
bpe_vocab_size = 1000  #!!!!!
batch_size = 16 #Number of sequences processed in parallel
learning_rate=1e-3  #was 1e-2 then 1e-3

# Grab data from datasets
print("Loading dataset")
dataset = ""
openAIFiles = ["input_data_files/openai_generated_text.txt", "input_data_files/openai_generated_text_3000_0.txt",
         "input_data_files/openai_generated_text_3000_1.txt", "input_data_files/openai_generated_text_3000_spooky.txt"]

allFiles = ["input_data_files/openai_generated_text.txt", "input_data_files/openai_generated_text_3000_0.txt",
         "input_data_files/openai_generated_text_3000_1.txt", "input_data_files/openai_generated_text_3000_spooky.txt"
         "cleaned_orca_dataset.txt"]

files = openAIFiles
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


# We just love loading things in chunks because our PCs cannot handle the insane memory requirements!
# This function will let us load our dataset into a tensor in chunks, so we don't have to spend thousands of extra
# dollars on hardware instead.
def encode_dataset_in_chunks(encoder, dataset, chunk_size=1000000):
    encoded_chunks = []
    start_idx = 0
    while start_idx < len(dataset):
        end_idx = min(start_idx + chunk_size, len(dataset))
        chunk = dataset[start_idx:end_idx]
        encoded_chunk = encoder.encode(chunk)
        encoded_chunk_tensor = torch.tensor(encoded_chunk, dtype=torch.long)  # Convert list to tensor
        encoded_chunks.append(encoded_chunk_tensor)
        start_idx = end_idx

    return torch.cat(encoded_chunks)  # Concatenate tensors before returning


chunk_size = 1000000  # Adjust the chunk size according to your memory capacity
data = encode_dataset_in_chunks(byte_pair_encoder, dataset, chunk_size=chunk_size)
print("more splitting")
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print("done splitting")


# load data
def load_batch(split, block_size=256):  # !!! block_size is hardset here, if someone changes it stuff will crash
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# gets rid of noise when getting loss. Averages the splits instead of randomly sampling(which creates noise)
# Later I can replace this loss estimation with a potentially better one. Monte Carlo sampling.
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

print("Loading model")
model = BigramLanguageModel(num_heads=3, num_layers=3)
print("move to cuda")
m = model.to(device)  # CUDA!!1!1

# Model's parameter count
print("Loading hyperparameter count")
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# using Pytorch's adamW optimizer
print("Loading optimizer")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Looping over iterations")
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

# Save pretrained model
torch.save(model.state_dict(), 'Orca_BPE_8000iter_1_6bil-tokens_montecarlo_v1.pth')

# Create target directory & all intermediate directories if don't exists
# Then save the encoder
encoder_dir = 'encoder_directory'
tokenizer_name = 'Orca_encoder'
byte_pair_encoder.save(encoder_dir, tokenizer_name)

# Generate from the model
print("GENERATING SAMPLE TEXT")
for _ in range(10):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(byte_pair_encoder.decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
    print("------------------------------------------------------")
