class Word2VecDataset:
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        return self.pairs[i]

import numpy as np
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
#ici on fait du negativ sampling skip-gram
class Word2Vec:
    def __init__(self, vocab_size, embed_dim,neg_dist,K):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.neg_dist=neg_dist
        self.W_in = np.random.randn(vocab_size, embed_dim)
        self.W_out = np.random.randn(vocab_size, embed_dim)
        self.K=K
    def forward(self, center_id):
        x = np.zeros(self.vocab_size)
        x[center_id] = 1

        #Hidden layer: h = x @ W_in 
        h = x @ self.W_in 
        return h #ici on fait pas tout le forward car SGNS on a pas de softmax sur tout l'ouyput
   
    def loss(self, center_id, context_id):
        v_c = self.W_in[center_id]
        v_o = self.W_out[context_id]

        # Positif, on cherche à ce que sigmoid(v_o t * v_c)= 1
        pos_loss = -np.log(sigmoid(np.dot(v_c, v_o)))

        # Négatifs, ici on veut sigmoid(neg_vecs t* v_c)=0 ie les vecteurs sont orthogonaux et du coup indépendant et du coup ils sont éloignés 
        neg_ids = np.random.choice(self.vocab_size, size=self.K, p=self.neg_dist)
        neg_vecs = self.W_out[neg_ids]             
        neg_scores = neg_vecs.dot(v_c)              

        neg_loss = -np.sum(np.log(sigmoid(-neg_scores)))
        return pos_loss + neg_loss

    def train(self, pairs, lr, epochs):

        return

