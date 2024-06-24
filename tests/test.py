import sys
sys.path.append('.')
import numpy as np

from cpe import CPETokenizer

num_basic_tokens = 128
vocab_size = 1024
dataset_size = 1000
min_seq_len = 8
max_seq_len = 1000

tokenizer = CPETokenizer(num_basic_tokens, vocab_size, verbose=True)

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
encoded_dataset = tokenizer.train(dataset)

# test encode
print(f'[INFO] testing encode...')
for _ in range(3):
    seq = random_sequence()
    encoded = tokenizer.encode(seq)
    decoded = tokenizer.decode(encoded)
    assert np.allclose(seq, decoded)
    print(f'compressing {len(seq)} --> {len(encoded)} --> {len(decoded)}')

# test save and load
print(f'[INFO] testing save and load...')
tokenizer.save(f'cpe')

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
