import numpy as np
from typing import List, Union, Optional

# cpp extension (named in setup.py and PYBIND11_MODULE)
import _cpe

import kiui

class CPETokenizer:
    def __init__(self, num_basic_tokens, vocab_size, verbose=False):
        self.num_basic_tokens = num_basic_tokens
        self.vocab_size = vocab_size
        self.verbose = verbose
        # the cpp impl
        self.impl = _cpe.CPETokenizer(num_basic_tokens, vocab_size, verbose)
    
    def train(self, dataset: List[Union[List[int], np.ndarray]]) -> List[List[int]]:
        # dataset: list of list of int, or list of 1d np.ndarray. (each entry may have different length)
        self.impl.import_dataset(dataset)
        self.impl.train()
        
        # export and save merged dataset, so we don't need to encode them again (which is slow!)
        merged_dataset = self.impl.export_dataset() # list of list of int
        
        # delete dataset after finish training to save memory
        self.impl.clear_dataset()

        return merged_dataset

    def encode(self, x: Union[List[int], np.ndarray]) -> List[int]:
        # x: list of int, or 1d np.ndarray
        return self.impl.encode(x)

    def decode(self, tokens: Union[List[int], np.ndarray]) -> List[int]:
        # tokens: list of int, or 1d np.ndarray
        return self.impl.decode(tokens)

    def save(self, path: str):
        # export merges from cpp impl and sort it by idx
        merges = self.impl.export_merges() # dict of (int, int) -> int
        sorted_merges = [k for k, v in sorted(merges.items(), key=lambda item: item[1])] # sort by idx, list of (int, int)
        # write the model file
        with open(path, 'w') as f:
            # assume merges is already sorted by idx!
            for idx1, idx2 in sorted_merges:
                f.write(f"{idx1} {idx2}\n")


    def load(self, path: str):
        # read the model file
        sorted_merges = []
        with open(path, 'r', encoding="utf-8") as f:
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                sorted_merges.append((idx1, idx2))
        self.impl.import_merges(sorted_merges)