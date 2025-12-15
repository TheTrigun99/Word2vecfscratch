from datasets import load_dataset
import re
from collections import Counter
import random as rd
import numpy as np


def load_corpus():
    data = load_dataset("wikitext", "wikitext-2-raw-v1")
    train = data["train"]["text"]
    return "\n".join(train)

def build_sentences(corpus):
    return sentences(corpus)

def build_vocab(sent):
    return tokenisation(sent)


def sentences(corpus):
    """ We extract sentences from the corpus delimited by "?" , "." and "!" """
    # Lowercase
    text = corpus.lower()
    # Remove only exotic characters
    text = re.sub(r"[^a-z0-9.,!?;:'\-()\s]", " ", text)

    # Tokenize words + punctuation
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[.,!?;:()\-]", text)

    # Split on . ! ?
    sentences = []
    current = []
    for tok in tokens:
        if tok in {".", "!", "?"}:
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        sentences.append(current)
    return sentences
#TODO: Mettre sentences directement en ids




def tokenisation(sent,min_count=3):
    """We take a sentence and return the total tokenisation (vocab) + w2id et id2w"""
    counter=Counter()
    for s in sent:
        counter.update(s)

    vocab = [w for w in counter if counter[w] >= min_count]
    w2id = {w: i for i, w in enumerate(vocab)}
    id2w = {i: w for w, i in w2id.items()}

    counts = [counter[w] for w in vocab]
    total = sum(counts)
    f_relg = np.array([c / total for c in counts], dtype=float)
    return vocab,w2id,id2w,counts,f_relg


def build_noise(count,alpha=0.75):
    """Making the unigram distribution """
    noise= count**alpha
    noise_dist=noise/noise.sum()
    return noise_dist

def pair(s, w, w2id):
    """Creation of pairs (center,context) for 1 sentence with dynamique window (random)."""
    p = []
    for word in range(len(s)):
        c = word
        # fenêtre tirée aléatoirement dans [1, w]
        w_cur = rd.randint(1, w)
        # fen <-
        for i in range(1, w_cur + 1):
            k = c - i
            if k < 0:
                break
            p.append((w2id[s[word]], w2id[s[k]]))
        # fen ->
        for j in range(1, w_cur + 1):
            k = c + j
            if k >= len(s):
                break
            p.append((w2id[s[word]], w2id[s[k]]))
    return p

def filtre_s(s, w2id, f_relg, t_s, drop_counter=None):
    """Subsampling"""
    filtered = []
    for w in s:
        if w not in w2id:
            continue
        f = f_relg[w2id[w]]          # relativ frequence of the word
        P_keep = min((np.sqrt(f/t_s) + 1) * t_s/f, 1)  # proba of keeping the word
        
        if rd.random()< P_keep:   # we keep the word
            filtered.append(w)
        elif drop_counter is not None:
            drop_counter[w] += 1

    return filtered

def pairs(sent,w,w2id,f_relg,id2w,t=1e-3):  #t is a sensible parameter
    """We create the pairs (center,context) that will be our training samples""" 
    p=[]
    t_s=t
    drop_counter = Counter()
    kept_tokens = 0
    for s in sent:
        s_filt=filtre_s(s, w2id, f_relg, t_s, drop_counter=drop_counter)
        kept_tokens += len(s_filt)
        p.extend(pair(s_filt,w,w2id))
    dropped_tokens = sum(drop_counter.values())
    if dropped_tokens:
        print(f"Subsampling: kept {kept_tokens} tokens, dropped {dropped_tokens} tokens.")
        top_dropped = drop_counter.most_common(20)
        print("Top dropped words:", top_dropped)
    # to see what pairs our model dropped the most
    cnt_ctx = Counter()
    if "paris" in w2id:
        kid = w2id["paris"]
        for c, ctx in p:
            if c == kid:
                cnt_ctx[ctx] += 1
        print("Contexts for 'paris' (after subsampling):")
        print([(id2w[i], n) for i, n in cnt_ctx.most_common(20)])
    return p


def load_data(max_sentences=None):
    """load wikitext-2 and prepare the vocab/sentences ."""
    data = load_dataset("wikitext", "wikitext-2-raw-v1")
    train = data["train"]["text"]
    corpus = "\n".join(train)
    sent = sentences(corpus)

    if max_sentences is not None:
        sent = sent[:max_sentences]

    vocab, w2id, id2w, counts,f_relg = tokenisation(sent,min_count=3)
    return sent, vocab, w2id, id2w, counts,f_relg
"""
sent=sentences(corpus)
vocab,w2id,id2w,counts=tokenisation(sent)
total = sum(counts)
print(total)
f_rel = [c / total for c in counts]
for w in ['valkyria', 'chronicles', 'iii', 'senj', 'no', 'valkyria', 'unrecorded', 'chronicles', 'japanese', 'lit']:
    f = f_rel[w2id[w]]
    print(counts[w2id[w]])
    print(w, f, "P_drop =", 1 - np.sqrt(1e-5 / f)) 

print(pairs(sent,2,w2id,counts)[:10])
print(sent[0])"""



