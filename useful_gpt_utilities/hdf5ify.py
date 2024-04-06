import h5py
import numpy as np


# Function to load and split the data, appending the delimiter back to each paragraph
def load_paragraphs(text_file, delimiter='<|endoftext|>'):
    with open(text_file, 'r', encoding='utf-8') as file:
        content = file.read()
    paragraphs = content.split(delimiter)
    # Append the delimiter back to each paragraph, except for the last one if it's empty
    paragraphs = [para.strip() + delimiter for para in paragraphs if para.strip()]
    return paragraphs


# Convert dataset to HDF5, including the delimiter in each paragraph
def convert_to_hdf5(text_file, hdf5_file, delimiter='<|endoftext|>'):
    paragraphs = load_paragraphs(text_file, delimiter)

    with h5py.File(hdf5_file, 'w') as h5f:
        dt = h5py.special_dtype(vlen=str)  # Define datatype for variable-length strings
        dataset = h5f.create_dataset('paragraphs', (len(paragraphs),), dtype=dt)

        # Store paragraphs, with delimiter, in the dataset
        for i, para in enumerate(paragraphs):
            dataset[i] = para


# Example usage
text_file = '../input_data_files/TinyStoriesV2-GPT4-valid.txt'
hdf5_file = '../input_data_files/TinyStoriesV2-GPT4-valid.hdf5'
convert_to_hdf5(text_file, hdf5_file)