import os
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


DATA_DIR = Path(__file__).resolve().parent / "wikitext2"
PARQUETS = ["train-00000-of-00001.parquet"]


def load_sentences():
    sentences = []
    for name in PARQUETS:
        df = pd.read_parquet(DATA_DIR / name)
        for line in df["text"].dropna():
            text = line.strip()
            if not text:
                continue
            tokens = simple_preprocess(text, deacc=True)
            if tokens:
                sentences.append(tokens)
    return sentences


def train_gensim_w2v(sentences):
    return Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=3,
        min_count=3,
        sg=1,  # skip-gram
        negative=5,  # negative sampling
        hs=0,
        sample=1e-3,
        workers=os.cpu_count() or 1,
        epochs=5,
        seed=42,
    )


def cosine(u, v):
    u = np.asarray(u)
    v = np.asarray(v)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def main():
    sentences = load_sentences()
    print(f"Loaded {len(sentences)} sentences from WikiText-2.")
    
    model = train_gensim_w2v(sentences)
    print("Training finished.")

    needed = ["king", "man", "woman", "queen"]
    missing = [w for w in needed if w not in model.wv]
    if missing:
        print(f"Missing words after training: {missing}")
        return

    analogy_vec = model.wv["king"] - model.wv["man"] + model.wv["woman"]
    queen_vec = model.wv["queen"]
    score = cosine(analogy_vec, queen_vec)
    print(f"cosine(king - man + woman, queen) = {score:.4f}")

    print("Most similar to king - man + woman:")
    for word, sim in model.wv.most_similar(
        positive=["king", "woman"], negative=["man"], topn=5
    ):
        print(f"  {word}\t{sim:.4f}")

    print("\nTop 10 neighbors of 'king':")
    for word, sim in model.wv.most_similar("football", topn=30):
        print(f"  {word}\t{sim:.4f}")


if __name__ == "__main__":
    main()
