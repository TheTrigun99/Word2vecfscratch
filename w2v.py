

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
        neg_scores = np.dot(neg_vecs,v_c)              

        neg_loss = -np.sum(np.log(sigmoid(-neg_scores)))
        return pos_loss + neg_loss

    def train(self, pairs, lr, epochs, progress=False):
        for epoch in range(epochs):
            if progress:
                total = len(pairs)
                step = max(1, total // 50)  # update  tous les 2%
                print(f"Epoch {epoch+1}/{epochs}")

            for idx, (center_id,context_id) in enumerate(pairs):
                
                #vecteurs center et context
                v_c = self.W_in[center_id]
                v_o = self.W_out[context_id]

                #vecteurs négatifs
                neg_ids=np.random.choice(self.vocab_size, size=self.K, p=self.neg_dist)
                neg_vecs=self.W_out[neg_ids]
                
                #score
                pos = np.dot(v_c, v_o)           # scalaire
                neg = neg_vecs.dot(v_c)

                #calcul des gradients
                grad_c= ( sigmoid(pos)-1 )*v_o + np.sum(    sigmoid(neg)[:, None] *  neg_vecs ,axis=0 )
                grad_o= (sigmoid(pos)-1)*v_c
                grad_negs = sigmoid(neg)[:, None] * v_c[None, :]

                
                self.W_in[center_id]      -= lr * grad_c
                self.W_out[context_id]    -= lr * grad_o
                self.W_out[neg_ids]       -= lr * grad_negs

                if progress and (idx + 1) % step == 0:
                    pct = 100.0 * (idx + 1) / total
                    print(f"\r  {idx+1}/{total} ({pct:5.1f}%)", end="")

            if progress:
                print("\r  done".ljust(24))

        return

