from datasets import load_dataset
import re
from collections import Counter
import random as rd
import numpy as np
data = load_dataset("wikitext", "wikitext-2-raw-v1")
train = data["train"]["text"]
valid = data["validation"]["text"]
test = data["test"]["text"]
print(test[0])

corpus = "\n".join(train)



def sentences(corpus):
    """on nettoie le texte et on extrait les phrases du dataset"""
    text = corpus.lower() #on met tout en minuscule (word2vec s'en fout des maj)
    text = re.sub(r"<unk>", " ", text)
    text = re.sub(r"[^a-z0-9.\s]", " ", text)   # on garde uniquement . + lettres + chiffres

    #tokenisation anglaise 
    raw_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\.", text)

    #on coupe sur les points "." mais on ne les garde pas (sinon problématique pour tokeniser après)
    sentences = []
    current = []

    for tok in raw_tokens:
        if tok == ".":
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(tok)

    if current:   #au cas où si j'ai juste un texte avec aucun points
        sentences.append(current)

    return sentences
#TODO: Mettre sentences directement en ids
def tokenisation(sent):
    """On prend une phrase et on renvoie la tokenisation totale (vocab) + w2id et id2w"""
    token=set()
    w2id={}
    id2w={}
    i=0
    counter=Counter()
    for s in sent:
        counter.update(s)
        for word in s:
            if word not in token:
                token.add(word)
                w2id[word]=i
                id2w[i]=word
                i=i+1
    counts = [counter[word] for word in w2id]
    return list(w2id.keys()),w2id,id2w,counts

sent=sentences(corpus)
vocab,w2id,id2w,counts=tokenisation(sent)

def build_noise(count,alpha=0.75):
    noise= count**alpha
    noise_dist=noise/noise.sum()
    return noise_dist

def pair(s,w,w2id):
    """Création des pairs (center,context) poir 1 phrase"""
    p=[]
    for word in range(len(s)):
        c=word
        #fen <-
        for i in range(1,w+1):
            k=c-i
            if k<0:
                break
            p.append((   w2id[s[word]],  w2id[s[k]]))
        #fen ->
        for j in range(1,w+1):
            k=c+j
            if k >=len(s):
                 break
            p.append((  w2id[s[word]] ,w2id[s[k]] ))
    return p

def filtre_s(s, w2id, freq_rel, t_s):
    filtered = []
    for w in s:
        f = freq_rel[w2id[w]]          # fréquence relative du mot
        P_drop = 1 - t_s/np.sqrt(f)  # proba de discard
        
        if rd.random() > P_drop:   # on garde le mot
            filtered.append(w)

    return filtered

def pairs(sent,w,w2id,counts,t=1e-5):  #t est un paramètres ultra sensible (j'ai essayé plusieurs valeurs et 1e-5 est pas mal)
    """On crée les pair (center,context) qui nous servent de données de training""" 
    p=[]
    total = sum(counts)
    freq_rel = [c / total for c in counts]
    t_s=np.sqrt(t)
    for s in sent:
        s_filt=filtre_s(s, w2id, freq_rel, t_s)
        p.extend(pair(s_filt,w,w2id))
    
    return p

"""total = sum(counts)
print(total)
f_rel = [c / total for c in counts]
for w in ['valkyria', 'chronicles', 'iii', 'senj', 'no', 'valkyria', 'unrecorded', 'chronicles', 'japanese', 'lit']:
    f = f_rel[w2id[w]]
    print(counts[w2id[w]])
    print(w, f, "P_drop =", 1 - np.sqrt(1e-5 / f)) 

print(pairs(sent,2,w2id,counts)[:10])
print(sent[0])"""



