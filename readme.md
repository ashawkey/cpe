# Coordinate Pair Encoding Tokenizer

CPE Tokenizer for compressing mesh coordinate sequence.

The core functions are implemented in c++ and binded to python.

### Install

```bash
pip install git+https://github.com/ashawkey/cpe

# or locally
git clone --recursive https://github.com/ashawkey/cpe
cd cpe
pip install . 
```

### Usage

```python
import numpy as np
from cpe import CPETokenizer

num_basic_tokens = 128 # like the 256 byte in BPE
vocab_size = 1024 # targeted vocab size (including basic tokens)
verbose = True # if you want to see the training progress

tokenizer = CPETokenizer(num_basic_tokens, vocab_size, verbose) 

# construct dataset
dataset = ... # list of list of int

# train the tokenizer
encoded_dataset = tokenizer.train(dataset) # list of list of int, dataset after encoding (can be cached to avoid encoding again during training)

# encode and decode
seq = ... # list of int
encoded = tokenizer.encode(seq)
decoded = tokenizer.decode(encoded)
assert np.allclose(seq, decoded)
print(f'compressing {len(seq)} --> {len(encoded)} --> {len(decoded)}')

# save and load
tokenizer.save(f'cpe.model')

tokenizer2 = CPETokenizer(num_basic_tokens, vocab_size, verbose)
tokenizer2.load(f'cpe.model')
```

An example can be found in [test.py](tests/test.py), where we also compare the speed with a pure-python implementation:
```bash
# train on 1000 sequences with max length of 1000, vocab size 1024
[INFO] python implementation took 124.384 seconds
[INFO] cpp implementation took 1.164 seconds
```

### Note
* We use linked list to make token merging and counting faster, but it's not data-paralleled for now.
* We load all data into memory, which could require very high memory for large dataset.