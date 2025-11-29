class Word2VecDataset:
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        return self.pairs[i]

import numpy as np
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
#ici on fait du negativ sampling skip-gram
class Word2Vec:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # W_in and W_out are your two matrices
        self.W_in = np.random.randn(vocab_size, embed_dim)
        self.W_out = np.random.randn(embed_dim, vocab_size)

    def forward(self, center_id):
        x = np.zeros(self.vocab_size)
        x[center_id] = 1

        #Hidden layer: h = x @ W_in 
        h = x @ self.W_in 
        return h #ici on fait pas tout le forward car SGNS on a pas de softmax sur tout l'ouyput

    def loss(self, center_id, context_id):
        pass

    def train(self, pairs, lr, epochs):
        # looping on pairs, gradient descent
        pass

