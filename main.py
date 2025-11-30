import numpy as np

from w2v import Word2Vec


def prepare_data(window_size=2, max_pairs=20000, max_sentences=None):
    """Charge les données wikitext, construit les paires et renvoie la distribution de bruit."""
    # Import paresseux pour éviter le coût à l'import du module.
    import traitement as tr

    sent, vocab, w2id, id2w, counts = tr.load_data(max_sentences=max_sentences)
    all_pairs = tr.pairs(sent, window_size, w2id, counts)

    if max_pairs and len(all_pairs) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in idx]

    noise_dist = tr.build_noise(np.array(counts))
    return all_pairs, noise_dist, vocab, w2id, id2w


def main():
    window_size = 2
    embed_dim = 50
    negatives = 5
    lr = 0.025
    epochs = 8
    max_pairs = 200000
    max_sentences = None  # pour tester vite, mettez None pour tout le corpus

    print("Preparing data from WikiText-2...")
    pairs, noise_dist, vocab, w2id, id2w = prepare_data(
        window_size, max_pairs, max_sentences=max_sentences
    )
    print(
        f"Training on {len(pairs)} (center, context) pairs with "
        f"window={window_size}, sentences={max_sentences or 'all'}."
    )

    model = Word2Vec(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        neg_dist=noise_dist,
        K=negatives,
    )

    model.train(pairs=pairs, lr=lr, epochs=epochs, progress=True)
    print("Training complete. Sample embeddings:")

    for word in ["king", "queen", "science", "football"]:
        if word in w2id:
            idx = w2id[word]
            vec = model.W_in[idx][:5]
            print(f"{word}: {vec}")
        else:
            print(f"{word}: not in vocabulary")

    # Test analogie : king - man + woman ≈ queen
    needed = ["king", "man", "woman", "queen"]
    if all(w in w2id for w in needed):
        vk = model.W_in[w2id["king"]]
        vm = model.W_in[w2id["man"]]
        vw = model.W_in[w2id["woman"]]
        vq = model.W_in[w2id["queen"]]

        analogy = vk - vm + vw

        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

        score_queen = cosine(analogy, vq)
        print(f"Analogie king - man + woman vs queen (cosine): {score_queen:.4f}")
    else:
        missing = [w for w in needed if w not in w2id]
        print(f"Analogie impossible, mots manquants: {missing}")


if __name__ == "__main__":
    main()
