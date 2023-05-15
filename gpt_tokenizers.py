import tokenizers


# Encoder/Decoder | Character tokenizer
# Map unique character dictionary to integers
# Input: lisy of unique characters in dataset
# Output: if encoded, change list of characters(dataset) to a list of ints.
# if decoded, returns a String based on given integer list.
class CharacterTokenizer:
    def __init__(self, chars):
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        print("normal encoding")
        sample_input = s[0:70]
        print(sample_input)
        encoded = []
        for c in s:
            if c in self.stoi:
                encoded.append(self.stoi[c])

        sample_encode = encoded[0:70]
        print(sample_encode)
        return encoded

    def decode(self, l):
        decoded = ''.join([self.itos[i] for i in l])
        return decoded


# Sub-Word Byte-Pair Encoder
# Input: vocab list size, minimum frequency, and your dataset
# Output: either encoded string to list or decoded list to string
class BytePairEncoder:
    def __init__(self, vocab_size=32000, min_frequency=2):
        self.tokenizer = tokenizers.ByteLevelBPETokenizer()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    # Train tokenizer on dataset
    def train(self, files):
        self.tokenizer.train(files=files, vocab_size=self.vocab_size, min_frequency=self.min_frequency)

    def encode(self, s):
        encoded = self.tokenizer.encode(s).ids
        return encoded

    def decode(self, l):
        decoded = self.tokenizer.decode(l)
        return decoded
