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
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    vn = vec / (np.linalg.norm(vec) + 1e-9)
    scores = Wn @ vn
    scores[w2id[word]] = -1e9  # �viter le mot lui-m�me
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(id2w[i], float(scores[i])) for i in idx]


if __name__ == "__main__":
    W_in, W_out, vocab, w2id, id2w = load_model("w2v_model.npz")
    for word, score in topk("king", W_in, w2id, id2w, k=10):
        print(f"{word}\t{score:.4f}")
