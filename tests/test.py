import sys
sys.path.append('.')

import time
import numpy as np
import itertools

from cpe import CPETokenizer

### hyper parameters
num_basic_tokens = 128
vocab_size = 1024
dataset_size = 1000
min_seq_len = 8
max_seq_len = 1000
verbose = False

### pure-python reference implementation (from https://github.com/karpathy/minbpe)

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def get_stats_batch(data_ids):
    counts = None
    for ids in data_ids:
        counts = get_stats(ids, counts)
    # sort by pair to match cpp impl
    counts = {k: v for k, v in sorted(counts.items(), key=lambda x: x[0])}
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def merge_batch(data_ids, pair, idx):
    return [merge(ids, pair, idx) for ids in data_ids]

class PyCPETokenizer:

    def __init__(self, num_basic_tokens, vocab_size, verbose):
        self.num_basic_tokens = num_basic_tokens
        self.vocab_size = vocab_size
        self.verbose = verbose

        self.merges = {} # (int, int) -> int
        self.vocab = self._build_vocab() # int in vocab_size -> list of int in num_basic_tokens

    def train(self, data_ids):
        # data_ids: list of ids to train the vocab
        assert self.vocab_size >= self.num_basic_tokens
        num_merges = self.vocab_size - self.num_basic_tokens

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: [idx] for idx in range(self.num_basic_tokens)}
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats_batch(data_ids)
            # find the pair with the highest count (if multiple, take the first one)
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = self.num_basic_tokens + i
            # replace all occurrences of pair in ids with idx
            data_ids = merge_batch(data_ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] # list
            # prints
            if self.verbose:
                print(f"[INFO] merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

        return data_ids

    def encode(self, ids):
        # ids: list of ints in [0, num_basic_tokens)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        ids = [self.vocab[idx] for idx in ids]
        ids = list(itertools.chain.from_iterable(ids))
        return ids

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: [idx] for idx in range(self.num_basic_tokens)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save(self, path):
        with open(path, 'w') as f:
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

    def load(self, path):
        merges = {}
        idx = self.num_basic_tokens
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()

tokenizer_ref = PyCPETokenizer(num_basic_tokens, vocab_size, verbose=verbose)

### cpp implementation

tokenizer = CPETokenizer(num_basic_tokens, vocab_size, verbose=verbose)

# construct random coordinate sequences dataset
# since it's totally random, there are not many patterns to learn, and the compression ratio is not high...
def random_sequence():
    seq_len = np.random.randint(min_seq_len, max_seq_len)
    seq = np.random.randint(0, num_basic_tokens, seq_len)
    return seq

print(f'[INFO] constructing dataset...')
dataset = [random_sequence() for _ in range(dataset_size)]

# train the tokenizer
print(f'[INFO] training tokenizer...')
t0 = time.time()
encoded_dataset_ref = tokenizer_ref.train(dataset)
t1 = time.time()
print(f'[INFO] python implementation took {t1 - t0:.3f} seconds')
t0 = time.time()
encoded_dataset = tokenizer.train(dataset)
t1 = time.time()
print(f'[INFO] cpp implementation took {t1 - t0:.3f} seconds')

# make sure the two implementations are consistent
for i in range(len(encoded_dataset)):
    assert np.allclose(encoded_dataset_ref[i], encoded_dataset[i])

# test encode
print(f'[INFO] testing encode...')
for _ in range(3):
    seq = random_sequence()
    t0 = time.time()
    encoded_ref = tokenizer_ref.encode(seq)
    t1 = time.time()
    print(f'python implementation took {t1 - t0:.3f} seconds')
    t0 = time.time()
    encoded = tokenizer.encode(seq)
    t1 = time.time()
    print(f'cpp implementation took {t1 - t0:.3f} seconds')
    assert np.allclose(encoded_ref, encoded)
    decoded = tokenizer.decode(encoded)
    assert np.allclose(seq, decoded)
    print(f'compressing {len(seq)} --> {len(encoded)} --> {len(decoded)}')

# test save and load
print(f'[INFO] testing save and load...')
tokenizer.save(f'cpe.model')

tokenizer2 = CPETokenizer(num_basic_tokens, vocab_size, verbose=True)
tokenizer2.load(f'cpe.model')
for _ in range(3):
    seq = random_sequence()
    encoded = tokenizer.encode(seq)
    encoded2 = tokenizer2.encode(seq)
    assert np.allclose(encoded, encoded2)
    decoded = tokenizer.decode(encoded)
    decoded2 = tokenizer2.decode(encoded2)
    assert np.allclose(decoded, decoded2)
    print(f'compressing {len(seq)} --> {len(encoded)} --> {len(decoded)}')
