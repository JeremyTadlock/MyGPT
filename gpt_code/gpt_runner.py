import torch
import gpt_tokenizers
from original_bigram_gpt import BigramLanguageModel, device, block_size  # Assuming 'device' is defined and exported in BigramGPT module

# Paths for model and tokenizer
model_path = '../models/shellcode_v2.6.pth'
vocab_path = '../encoder_directory/shellcode_v2.6-vocab.json'
merges_path = '../encoder_directory/shellcode_v2.6-merges.txt'

# Load the model
print("Loading model")
model = BigramLanguageModel().to(device)  # Use the device from BigramGPT
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the tokenizer
print("Loading tokenizer")
byte_pair_encoder = gpt_tokenizers.BytePairEncoder()
byte_pair_encoder.load(vocab_path, merges_path)

# Question list
questions = ["<question> define the call_egghunter label.",
             "<question> push the dword 0x68732f6e onto the stack.",
             "<question> jump to 0x8 if not zero.", "<question> push 0x0 onto the stack.",
             "<question> increment the contents of the esi register.",
             "<question> declare the up label.", "<question> jump to 0x26 if not zero.",
             "<question> I have a stack. i want you to push the doubleword named potato onto the stack.",
             "<question> I have a stack. i want you to push the doubleword ebp onto the stack."]

more_questions = ["<question> Decrement eax by 1.",
                  "<question> Decrease ebp.",
                  "<question> Define the decode function and negate the byte in esi.",
                  "<question> decode: \n xor byte [esi], 0xaa",
                  "<question> Define decoder function and store the encoded shellcode pointer in the esi register."
                  ]

even_more_questions = ["<question> Write assembly code to decrement the contents of the ebx register by 1, and then perform a specific operation if the result is negative."]

# Encoding the question and converting to tensor
encoded_questions = [byte_pair_encoder.encode(question) for question in even_more_questions]



# Generating text based on the question and generation type
print("Generating text")
for encoded_question in encoded_questions:
    # truncate input if needed
    if len(encoded_question) > block_size:
        encoded_question = encoded_question[:block_size]

    question_tensor = torch.tensor([encoded_question], dtype=torch.long).to(device)  # Use the device from BigramGPT
    print("-------------------------------------------")
    print("Generating text using normal generation")
    for _ in range(3):

        generated_text_ids = model.generate(question_tensor, max_new_tokens=500, eos_token_id=2)[0].tolist()
        generated_text = byte_pair_encoder.decode(generated_text_ids)

        print(generated_text + '\n\n')

    print ("-------------------------------------------")
    print("Generating text using top k sampling")
    for _ in range(3):
        generated_text_ids = model.generate_top_k(question_tensor, max_new_tokens=500, eos_token_id=2, top_k=50)[0].tolist()
        generated_text = byte_pair_encoder.decode(generated_text_ids)
        print(generated_text + '\n\n')


    print ("-------------------------------------------")
    print("Generating text using top p sampling")
    for _ in range(5):
        generated_text_ids = \
        model.generate_top_p(question_tensor, max_new_tokens=500, eos_token_id=2, top_p=0.4, temperature=0.5)[
            0].tolist()
        generated_text = byte_pair_encoder.decode(generated_text_ids)
        print(generated_text + '\n\n')