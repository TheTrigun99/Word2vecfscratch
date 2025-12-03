import numpy as np


def load_model(path="w2v_model.npz"):
    data = np.load(path, allow_pickle=True)
    W_in = data["W_in"]
    W_out = data["W_out"]
    vocab = data["vocab"].tolist()
    w2id = data["w2id"].item()
    id2w = data["id2w"].item()
    return W_in, W_out, vocab, w2id, id2w


def combined_embeddings(W_in, W_out):
    """Combine W_in and W_out then L2-normalize row-wise."""
    V = W_in + W_out
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V / norms


def normalized_embeddings(W_in):
    """L2-normalize W_in row-wise (gensim-like)."""
    norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-9
    return W_in / norms


def normalize_both(W_in, W_out):
    """Return L2-normalized versions of W_in and W_out."""
    return normalized_embeddings(W_in), normalized_embeddings(W_out)


def topk(word, W, w2id, id2w, k=10):
    if word not in w2id:
        return []
    vec = W[w2id[word]]
    # Normalisation L2
    vn = vec / (np.linalg.norm(vec) + 1e-9)
    scores = W @ vn
    scores[w2id[word]] = -1e9  # �viter le mot lui-m�me
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
