# **GPT Model**
This project contains an implementation of a GPT (Generative Pre-trained Transformer) deep learning language model. The model uses self-attention to allow each word to attend to every other word in the input dataset. The model is implemented in PyTorch, and can be used to generate text based on a given input. 

# **Setup**
To run this code, you must have
* Python 3.6 or higher
* PyTorch 1.9 or higher

# **Usage**
To use this GPT model, run the BigramGPT class to start training on your input dataset based on the given data and hyperparameters. you can use the gpt_tokenizers class to switch between a character tokenizer and a Byte-pair(subword) tokenizer.

# **Hyperparameters**
The following hyperparameters can be set when using the BigramGPT class:

* batch_size: The number of sequences processed in parallel during training.
* block_size: The maximum content length for predictions.
* max_iters: The maximum number of iterations to train for.
* eval_interval: The number of iterations between evaluations.
* learning_rate: The learning rate used during training.
* device: The device used for processing (CPU or GPU).
* eval_iters: The number of iterations to use for estimating loss during evaluation.
* num_embeddings: The number of embeddings to use.
* num_heads: The number of attention heads to use.
* num_layers: The number of layers in the transformer.
* bpe_vocab_size: The size of the BPE vocabulary used.

# **Data**
This model requires a dataset of text to train on. Two example datasets are provided in this repository (openai_generated_text.txt and openai_generated_text_800.txt), but you can use any text dataset you like.

# **Acknowledgements**
This project was inspired by the GPT-2 model developed by OpenAI & the famous paper "Attention is all you need" by Google. For more information, see the link(s)
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762v5.pdf)