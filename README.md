# GPT Model Implementation

## **Project Overview**

This project represents a deep dive into the realm of Natural Language Processing (NLP) by designing and building a Generative Pre-trained Transformer (GPT) model from scratch. The primary goal is to construct a GPT model capable of understanding and generating human-like text, aiming to achieve performance on par with early versions of OpenAI's GPT models, such as GPT-1 or GPT-2, but on a smaller scale. This initiative covers a comprehensive range of NLP tasks including data preprocessing, tokenization, pretraining, and post-processing analysis. By undertaking this project, I aim to not only contribute a capable model to the NLP community but also deepen my understanding of the intricate workings of transformer-based language models.

## **Key Features**

- **Self-Attention Mechanism**: Implementing the core idea behind the Transformer architecture, focusing on the "attention is all you need" concept to process text in a non-sequential manner.
- **Custom Tokenization and Preprocessing**: Developing a tailored preprocessing pipeline that includes Byte Pair Encoding (BPE) for efficient text tokenization.
- **Model Training and Optimization**: Leveraging PyTorch for the development and training of the model, incorporating advanced techniques like mixed precision training and gradient accumulation to enhance performance.
- **Hyperparameter Tuning**: Utilizing Optuna for systematic hyperparameter optimization to fine-tune the model's performance.
- **Versatile Application**: Designed to be adaptable to various datasets, allowing for training on specific corpora and generating contextually relevant text.

## **Technology Stack**

- **Programming Language**: Python
- **Machine Learning Framework**: PyTorch
- **Optimization Tools**: Optuna, PyTorch CUDA for GPU acceleration

## **Setup and Usage**

### **Prerequisites**

- Python 3.8 or higher
- PyTorch 1.8.1 or higher
- Optuna for hyperparameter optimization
- Access to GPU (recommended for training)

### **Installation**

Clone the repository to your local machine:
git clone https://github.com/JeremyTadlock/MyGPT
cd https://github.com/JeremyTadlock/MyGPT

Install Dependencies:
pip install -r requirements.txt

Running the Model
To start pretraining, first train the tokenizer on your training data using train_tokenizer.py
Then, use bigram_gpt_dataloading.py to run the pretraining using your tokenized data.
Note: make sure to set your parameters, and input files including your training and validation data paths, and output model name when training is done.

After running the programs, the trained tokenizer and pretrained GPT will be saved, including information such as training time and hyperparameters.

You can then run post-processing text analysis on the model using its Generate function. see gpt_inference_testing.py for an example on how to do this easily.

## **Goals and Future Work**

The immediate goal is to ensure the model can undergo effective training and generation tasks with the current dataset(which has been accomplished!). The longer-term aspiration is to refine the model to achieve competitive performance against more advanced GPT models, focusing on scalability and efficiency. Continuous improvement in model architecture, training processes, and tokenization techniques will be pursued, alongside exploring novel applications of the GPT model in various domains(which im hopefully very close to completing!).
