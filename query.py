import numpy as np


def load_model(path="w2v_model.npz"):
    data = np.load(path, allow_pickle=True)
    W_in = data["W_in"]
    W_out = data["W_out"]
    vocab = data["vocab"].tolist()
    w2id = data["w2id"].item()
    id2w = data["id2w"].item()
    return W_in, W_out, vocab, w2id, id2w

def topk(word, W, w2id, id2w, k=10):
    if word not in w2id:
        return []
    vec = W[w2id[word]]
    # Normalisation L2
    vn = vec / (np.linalg.norm(vec) + 1e-9)
    scores = W @ vn
    scores[w2id[word]] = -1e9  
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(id2w[i], float(scores[i])) for i in idx]

if __name__ == "__main__":
    W_in, W_out, vocab, w2id, id2w = load_model("w2v_model.npz")
    V = W_out + W_out
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    print("Top neighbors with normalized W_in for 'king':")
    for word, score in topk("football", V, w2id, id2w, k=40):
        print(f"{word}\t{score:.4f}")
