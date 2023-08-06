import BigramGPT
import torch
import gpt_tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Create bigram gpt model
model_path = 'BPE_8000iter_12mil-tokens_montecarlo_v1.pth'
print("Loading model")
model = BigramGPT.BigramLanguageModel()
m = model.to(BigramGPT.device)  # CUDA!!1!1
m.load_state_dict(torch.load(model_path))
model.eval()

# Load model tokenizer
print("Loading tokenizer")
byte_pair_encoder = gpt_tokenizers.BytePairEncoder()
byte_pair_encoder.load('encoder_directory/encoder-vocab.json', 'encoder_directory/encoder-merges.txt')

# Generate some text using loaded model & tokenizer
print("Generating text")
context = torch.zeros((1, 1), dtype=torch.long, device=BigramGPT.device)
print(byte_pair_encoder.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
















