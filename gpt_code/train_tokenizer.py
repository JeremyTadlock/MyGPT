import gpt_tokenizers

vocab_size = 25000

if __name__ == '__main__':

    print("Loading dataset")
    dataset = ""
    files = ["../input_data_files/train_TinyStoriesV2-GPT4-valid.txt"]
    for file in files:
        with open(file, "r", encoding="utf8") as f:
            dataset = dataset + f.read() + '\n'

    # Statistics about imported dataset(s)
    print(len(dataset))
    chars = sorted(list(set(dataset)))
    print("token size char:", len(chars))

    # Use Byte-Pair Encoder
    print("training BPE")
    byte_pair_encoder = gpt_tokenizers.BytePairEncoder(vocab_size, 2)
    byte_pair_encoder.train(files)

    # Create target directory & all intermediate directories if don't exists
    # Then save the encoder
    encoder_dir = '../encoder_directory'
    tokenizer_name = 'train_TinyStoriesV2-GPT-last10percent'
    byte_pair_encoder.save(encoder_dir, tokenizer_name)