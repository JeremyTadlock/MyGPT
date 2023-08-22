from math import ceil, floor
from multiprocess import set_start_method
set_start_method("fork")  # for multiprocessing I think you need to set this on windows to fork
from typing import List
from functools import partial

import pyarrow as pa
from datasets import Dataset
import torch

from  tokenizers.pre_tokenizers import BertPreTokenizer

class chunkyText(Dataset):
    #preT = BertPreTokenizer()

    def __init__(self, file, chunk_size):#, block_size, tokenizer):
        # chunks is the strings to break the dataset into for better multiprocessing
        # add an option with chunk_size=0 for full data

        # self.block_size = block_size
        # self.tokenizer = tokenizer
        self.chunks = []
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
        shard = shard[rowname]
    preT = BertPreTokenizer()
    """Takes in a shard and outputs as many valid token vectors as possible"""
    tokenpos = preT.pre_tokenize_str(shard)
    validBlockCount = floor((len(tokenpos)-2)/block_size)  # the last and first token have to be droped
    # if chunksize is too small you will lose alot of data
    # wastedTokens = (validBlockCount*self.block_size)-len(tokenpos)
    if not validBlockCount > 0:
        return({"input_ids":[]})  # no blocks from chunk
    
    tokenized = []
    for startidx in range(1, validBlockCount*block_size, block_size):
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

    # datadict = {key:[x[key] for x in tokenized if key in x] for key in tokenized[0].keys()}
    datadict = {"input_ids": torch.tensor(e, dtype=torch.long) for e in tokenized}
    return(datadict)

def build_dataset(file, tokenizer, block_size=125, chunk_size=10000):

    dataset = chunkyText(file, chunk_size)
    processingPartial = partial(processShard, block_size=block_size, tokenizer=tokenizer, rowname="chunks")
    encodedDataset = dataset.map(processingPartial, num_proc=12)

    return(encodedDataset)