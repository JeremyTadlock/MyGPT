import torch
import gpt_tokenizers
from original_bigram_gpt import BigramLanguageModel, device, block_size  # Assuming 'device' is defined and exported in BigramGPT module

# Paths for model and tokenizer
model_path = 'Tinystories_CustomStories_6000epochs_v2.pth'
vocab_path = 'encoder_directory/Tinystories_CustomStories_6000epochs_v2-vocab.json'
merges_path = 'encoder_directory/Tinystories_CustomStories_6000epochs_v2-merges.txt'

# Load the model
print("Loading model")
model = BigramLanguageModel().to(device)  # Use the device from BigramGPT
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the tokenizer
print("Loading tokenizer")
byte_pair_encoder = gpt_tokenizers.BytePairEncoder()
byte_pair_encoder.load(vocab_path, merges_path)

# Your question
question = "sophie went outside to"

# Encoding the question and converting to tensor
encoded_question = byte_pair_encoder.encode(question)

# truncate input if needed
if len(encoded_question) > block_size:
    encoded_question = encoded_question[:block_size]

question_tensor = torch.tensor([encoded_question], dtype=torch.long).to(device)  # Use the device from BigramGPT

# Generating text based on the question
print("Generating text")

for _ in range(3):
    generated_text_ids = model.generate(question_tensor, max_new_tokens=500, eos_token_id=417)[0].tolist()
    generated_text = byte_pair_encoder.decode(generated_text_ids)

    print(generated_text + '\n\n')

print ("-------------------------------------------")
print("Generating text using top k sampling")
for _ in range(3):
    generated_text_ids = model.generate_top_k(question_tensor, max_new_tokens=500, eos_token_id=417)[0].tolist()
    generated_text = byte_pair_encoder.decode(generated_text_ids)
    print(generated_text + '\n\n')