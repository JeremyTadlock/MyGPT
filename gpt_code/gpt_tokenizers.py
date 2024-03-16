from typing import Any
import tokenizers
import re
from collections import Counter
import os
import warnings

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


# Sub-Word tokenizer Byte-Pair Encoder
# Input: vocab list size, minimum frequency, and your dataset
# Output: either encoded string to list or decoded list to string
class BytePairEncoder:
    def __init__(self, vocab_size=5000, min_frequency=2):
        self.tokenizer = tokenizers.ByteLevelBPETokenizer()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens=[
            "<question>",
            "<answer>",
            "<eot>",
        ]

    def __call__(self, s):
        encoded = self.tokenizer.encode(s).ids
        return(encoded)

    # Train tokenizer on dataset
    def train(self, files):
        self.tokenizer.train(files=files, vocab_size=self.vocab_size, min_frequency=self.min_frequency, special_tokens=self.special_tokens)

    # Save the tokenizer
    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.tokenizer.save_model(path, name)

    # Load tokenizer from files
    def load(self, vocab_path, merges_path):
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_path, merges_path)

    def encode(self, s):
        encoded = self.tokenizer.encode(s).ids
        return encoded

    def decode(self, l):
        decoded = self.tokenizer.decode(l)
        return decoded


# Sub-word tokenizer SentencePiece
class SentencePieceTokenizer:

    def __init__(self, vocab_size=5000, unk_token='<UNK>', pad_token='<PAD>', bos_token='<BOS>', eos_token='<EOS>'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.vocab = []
        self.vocab_freq = Counter()

    def fit(self, texts):
        # Preprocess the texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]

        # Build the vocabulary
        self.vocab_freq = self.count_tokens(preprocessed_texts)
        self.vocab = self.build_vocab()

    def preprocess_text(self, text):
        # Apply necessary text preprocessing (e.g., lowercasing)
        # Return the preprocessed text
        return text.lower()

    def count_tokens(self, texts):
        # Count the frequency of each token in the texts
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        return counter

    def build_vocab(self):
        # Select the most common tokens as the vocabulary
        vocab = [self.unk_token, self.pad_token, self.bos_token, self.eos_token]
        vocab.extend([token for token, _ in self.vocab_freq.most_common(self.vocab_size - len(vocab))])
        return vocab

    def tokenize(self, text):
        # Implement the tokenization logic
        # Split the text into tokens
        tokens = text.split()
        return tokens

    def encode_token(self, token):
        # Encode a single token into its corresponding ID
        if token in self.vocab:
            return self.vocab.index(token)
        else:
            return self.vocab.index(self.unk_token)

    def decode_token(self, token_id):
        # Decode a single token ID into its corresponding token
        if token_id < len(self.vocab):
            return self.vocab[token_id]
        else:
            return self.unk_token

    def encode(self, text):
        # Encode a text into a list of token IDs
        tokens = self.tokenize(self.preprocess_text(text))
        token_ids = [self.encode_token(token) for token in tokens]
        return token_ids

    def decode(self, token_ids):
        # Decode a list of token IDs into a text
        tokens = [self.decode_token(token_id) for token_id in token_ids]
        text = ' '.join(tokens)
        return text


class SentencePieceTokenizerGoogle:
    def __init__(self, model_path=None, data=None, vocab_size=5000):
        self.tokenizer = spm.SentencePieceProcessor()
        self.vocab_size = vocab_size
        if model_path is None:
            if data is None:
                raise ValueError("If no model path is provided to load, one must be trained. Please include your data "
                                 "to train on.")
            else:
                print("training model")
                self.model_path = self.train(data)
        else:
            if data is None:
                self.model_path = model_path
            else:
                warnings.warn("A model path was provided, but an unneseccary data also was provided. if you want to "
                              "train a new model on your data text, please do not include a model path.")
        self.load_sp_model()

    def load_sp_model(self):
        self.tokenizer.load(self.model_path)

    def train(self, train_data):
        spm.SentencePieceTrainer.train(input=train_data, model_prefix='tokenizer', vocab_size=self.vocab_size)
        # Save the trained model
        tokenizer_path = 'tokenizer.model'
        serialized_model_path = 'sentencepiece.model'
        os.rename(tokenizer_path, serialized_model_path)
        path = os.path.abspath(serialized_model_path)
        print("sentencepiece model path:", path)
        return path

    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)

    def decode(self, tokens):
        return self.tokenizer.decode_ids(tokens)