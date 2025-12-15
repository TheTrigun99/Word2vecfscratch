# Word2Vec Skip-Gram (negative sampling) from scratch

Project implementing Word2Vec SGNS (skip-gram with negative sampling), no gensim or pytorch for the main model.

## Layout
- `main.py`: full pipeline (load WikiText-2, build pairs, train, save model).
- `traitement.py`: dataset download/tokenization and pair construction (with subsampling).
- `w2v.py`: numpy implementation of SGNS and the negative sampling table.
- `query.py`: load the `.npz` model and fetch nearest neighbors.
- `testgensim.py`: reference run with `gensim.Word2Vec` to compare results.

## Quick setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install numpy datasets pandas gensim pyarrow
```

## Train the scratch model
```bash
python main.py
```
- Downloads WikiText-2 via `datasets`, builds the vocab (`min_count=3`), applies subsampling, then trains a 300-dim model with 5 negatives and window 3.
- Main hyperparameters live at the top of `main.py` (`window_size`, `embed_dim`, `negatives`, `lr`, `epochs`, `max_pairs`, `max_sentences`) so you can shorten runs easily.
- Weights and vocab are saved to `w2v_model.npz` (`W_in`, `W_out`, `vocab`, `w2id`, `id2w`).

## Query embeddings
```bash
python query.py
```
- Loads `w2v_model.npz` and prints sample neighbors. To look up another word, change the `topk` call at the bottom of the file.

## Compare with gensim 
```bash
python testgensim.py
```
- `testgensim.py` trains a gensim Word2Vec with similar hyperparameters to serve as a quick baseline.

## Notes
- The default training builds ~1.3M pairs; this can take a few minutes on CPU. Lower `max_pairs` or `max_sentences` in `main.py` for faster experiments.
