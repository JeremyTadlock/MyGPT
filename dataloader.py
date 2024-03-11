from math import ceil, floor
from multiprocess import set_start_method
from typing import List
from functools import partial

import pyarrow as pa
from datasets import Dataset, load_dataset
import torch

from  tokenizers.pre_tokenizers import BertPreTokenizer

#set_start_method("fork")  # for multiprocessing I think you need to set this on windows to fork

class chunkyText(Dataset):
    #preT = BertPreTokenizer()

    def __init__(self, files, chunk_size):#, block_size, tokenizer):
        # chunks is the strings to break the dataset into for better multiprocessing
        # add an option with chunk_size=0 for full data

        # self.block_size = block_size
        # self.tokenizer = tokenizer
        self.chunks = []
        if type(files) != list:
            files = [files]
        
        for file in files:
            with open(file, encoding="utf-8") as f:
                text = f.read()
                start_idx = 0
                while start_idx < len(text):
                    end_idx = min(start_idx + chunk_size, len(text))
                    chunk = text[start_idx:end_idx]
                    self.chunks.append(chunk)
                    start_idx = end_idx


        # sets all of the needed vars underthehood for more complex dataset features
        chunks = pa.array(self.chunks, type=pa.string())
        arrow_table = pa.table([chunks], names=["chunks"])
        Dataset.__init__(self, arrow_table)

        # this might not multiprocess well
        # self.map(self._processShard, remove_columns="chunks", num_proc=12)

    def __len__(self):
        return len(self.chunks)  # this wont ever update

    def __getitem__(self, idx):  
        return(self.chunks[idx])  # only needs one sequence
    
    def __getitems__(self, keys: List) -> List:
        """This is used by the trainer"""
        return([self.chunks[x] for x in keys])
    
def processShard(shard, block_size, tokenizer, rowname=None):
    if rowname:
        shard = shard[rowname][0]
    preT = BertPreTokenizer()
    """Takes in a shard and outputs as many valid token vectors as possible"""
    tokenpos = preT.pre_tokenize_str(shard)
    validBlockCount = floor((len(tokenpos)-2)/block_size)  # the last and first token have to be droped
    # if chunksize is too small you will lose alot of data
    # wastedTokens = (validBlockCount*self.block_size)-len(tokenpos)
    if not validBlockCount > 0:
        return({"input_ids":[]})  # no blocks from chunk
    
    tokenized = []
    for startidx in range(1, validBlockCount*block_size, block_size):  # skips the first and last tokens
        firsttok = tokenpos[startidx]
        lasttok = tokenpos[startidx+block_size]
        firstidx = firsttok[-1][0]
        lastidx = lasttok[-1][-1]
        assert type(firstidx) == type(lastidx) == int
        tokens = tokenizer(shard[firstidx: lastidx], add_special_tokens=True, truncation=True, max_length=block_size)
        if "input_ids" in tokens:
            tokenized.append(tokens['input_ids'])  # make sure output is right here
        #add_special_tokens=True, truncation=True, max_length=256
        else:
            print("bad tokens")

    data = {"input_ids":[torch.tensor(e, dtype=torch.long) for e in tokenized]}
    # data = [{"input_ids":torch.tensor(e, dtype=torch.long)} for e in tokenized]
    return(data)

def build_dataset(files, tokenizer, block_size=125, chunk_size=10000, num_proc=1):

    if chunk_size:
        dataset = chunkyText(files, chunk_size)
        processingPartial = partial(processShard, block_size=block_size, tokenizer=tokenizer, rowname="chunks")
        encodedDataset = dataset.map(processingPartial, num_proc=num_proc, remove_columns=dataset.column_names, batched=True, batch_size=1)
    # else:
    #     dataset = load_dataset("text", data_files=[file])
    #     processingPartial = partial(processShard, block_size=block_size, tokenizer=tokenizer, rowname="chunks")
    #     encodedDataset = dataset.map(processingPartial, num_proc=num_proc)


    return(encodedDataset)


# if __name__ == "__main__":
#     from transformers import RobertaTokenizer
#     # like could you just use the old tokenizer here? Or is Roberta adding some new magic that the old one didnt have that we just didnt use earlier
#     tokenizer = RobertaTokenizer.from_pretrained("./tests/KantaiBERT", max_length=512)
#     test = processShard("This is a string with alot of words, to test the chunking", 5, tokenizer=tokenizer)
#     print(test)
