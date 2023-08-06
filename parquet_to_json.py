import dask.dataframe as dd

# Converts a parquet raw pointer to a json file. uses DASK to avoid memory allocation issues so that any pc can run it
# regardless of memory amount.

print("Loading with Dask")
parquet_path = r"E:\huggingface_home\datasets\3_5M-GPT3_5-Augmented.parquet"
df = dd.read_parquet(parquet_path)

print("Converting with Dask")
json_path = r"E:\huggingface_home\datasets\3_5M-GPT3_5-Augmented.json"
df.to_json(json_path, orient='records', lines=True)