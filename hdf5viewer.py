import h5py

# Replace 'your_file.hdf5' with the path to your HDF5 file
file_path = 'input_data_files/TinyStoriesV2-GPT4-valid.hdf5'

# Replace 'your_dataset' with the name of your dataset
dataset_name = 'paragraphs'

with h5py.File(file_path, 'r') as file:
    # Access the dataset
    dataset = file[dataset_name]

    # Read the first 5 entries
    first_five_entries = dataset[:5]

    # Decode each entry if it's a byte string
    for entry in first_five_entries:
        if isinstance(entry, bytes):
            print(entry.decode('utf-8'))
            print("----------------------")
        else:
            print(entry)