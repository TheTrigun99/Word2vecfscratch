def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

import numpy as np

A = np.random.randn(3, 5)
b = np.random.randn(5)

print(A.dot(b))