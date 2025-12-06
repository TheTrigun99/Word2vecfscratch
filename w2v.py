import numpy as np
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def build_unigram_table(prob_dist, table_size=10_000_000, seed=42):
    # prob_dist: array de taille vocab, somme = 1
    rng = np.random.default_rng(seed)
    # poids cumulés puis réplication proportionnelle
    # on normalise sur table_size pour éviter les erreurs d'arrondi
    counts = np.rint(prob_dist * table_size).astype(int)
    # s'assurer que la table ne soit pas vide
    if counts.sum() == 0:
        counts[0] = table_size
    else:
        # ajuster si la somme diffère de table_size
        diff = table_size - counts.sum()
        if diff > 0:
            counts[np.argmax(prob_dist)] += diff
        elif diff < 0:
            counts[np.argmax(counts)] += diff  # diff est négatif
    # construire la table
    table = np.repeat(np.arange(len(prob_dist)), counts)
    rng.shuffle(table)
    return table

#here we do du negativ sampling skip-gram
class Word2Vec:
    
    
    def __init__(self, vocab_size, embed_dim,neg_dist,K):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.neg_dist=neg_dist
        self.W_in = np.random.randn(vocab_size, embed_dim).astype(np.float32)*0.01
        self.W_out = np.random.randn(vocab_size, embed_dim).astype(np.float32)*0.01 #np.zeros((vocab_size, embed_dim)).astype(np.float32)
        self.K=K
        self.neg_table = build_unigram_table(self.neg_dist, table_size=10_000_000, seed=42)
    
    
    def forward(self, center_id):
        x = np.zeros(self.vocab_size)
        x[center_id] = 1

        #Hidden layer: h = x @ W_in 
        h = x @ self.W_in 
        return h #ici on fait pas tout le forward car SGNS on a pas de softmax sur tout l'ouyput
   
    def loss(self, center_id, context_id):
        v_c = self.W_in[center_id]
        v_o = self.W_out[context_id]

        # Positiv, on want that sigmoid(v_o t * v_c)= 1
        pos_loss = -np.log(sigmoid(np.dot(v_c, v_o)))

        # Négatifs, ici on veut sigmoid(neg_vecs t* v_c)=0 ie les vecteurs sont orthogonaux et du coup indépendant et du coup ils sont éloignés 
        neg_ids = self.sample_negatives(self.K, rng)
        neg_vecs = self.W_out[neg_ids]             
        neg_scores = np.dot(neg_vecs,v_c)              

        neg_loss = -np.sum(np.log(sigmoid(-neg_scores)))
        return pos_loss + neg_loss
    
    def sample_negatives(self, size, rng):
        idx = rng.integers(0, len(self.neg_table), size=size)
        return self.neg_table[idx]
    
    def train(self, pairs, lr, epochs, progress=False):
        total_steps = len(pairs) * epochs
        step_counter = 0
        rng = np.random.default_rng(0)
        min_lr=1e-4 #valeur copiée de Gensim 
        for epoch in range(epochs):
            rng.shuffle(pairs)
            if progress:
                total = len(pairs)
                progress_step = max(1, total // 50)  # update every ~2%
                print(f"Epoch {epoch+1}/{epochs}")

            for idx, (center_id, context_id) in enumerate(pairs):
                # décroissance linéaire du LR
                progress_frac = step_counter / total_steps  
                lr_t = lr * (1-progress_frac)
                if lr_t < min_lr:
                    lr_t = min_lr

                # vectors center and context
                v_c = self.W_in[center_id]
                v_o = self.W_out[context_id]
                # negative vectors
                neg_ids = self.sample_negatives(self.K, rng)
                neg_vecs = self.W_out[neg_ids]
                

                pos = np.dot(v_c, v_o)
                pos = np.clip(pos, -10, 10) ##on clip pour garder de la convergence, 10 -10 c'est suffisant pour les sigmoid

                neg = neg_vecs.dot(v_c)
                neg = np.clip(neg, -10, 10) #on clip pour garder de la convergence, 10 -10 c'est suffisant pour les sigmoid
                # scores
                #pos = np.dot(v_c, v_o)       
                #neg = neg_vecs.dot(v_c)      

                # gradients
                sig_pos = sigmoid(pos)
                sig_neg = sigmoid(neg)

                grad_c = (sig_pos - 1) * v_o + np.sum(sig_neg[:, None] * neg_vecs, axis=0)
                grad_o = (sig_pos - 1) * v_c
                grad_negs = sig_neg[:, None] * v_c[None, :]

                # updates
                self.W_in[center_id]   -= lr_t * grad_c
                self.W_out[context_id] -= lr_t * grad_o
                np.add.at(self.W_out, neg_ids, -lr_t * grad_negs)

                step_counter += 1

                if progress and (idx + 1) % progress_step == 0:
                    pct = 100.0 * (idx + 1) / total
                    print(f"\r  {idx+1}/{total} ({pct:5.1f}%)", end="", flush=True)
            print(np.max(np.linalg.norm(self.W_in, axis=1)))
            print(np.max(np.linalg.norm(self.W_out, axis=1)))
            if progress:
                print("\r  done".ljust(24))

        return

