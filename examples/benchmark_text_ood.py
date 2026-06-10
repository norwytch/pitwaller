"""Text near-OOD benchmark (no images): does the embedding-space OOD score flag
documents from unseen categories better than a max-softmax baseline?

    pip install -e '.[text]'
    python examples/benchmark_text_ood.py

Trains a classifier on five 20-Newsgroups categories, embeds every document with
a sentence-transformer (the substrate the OOD stack scores in), then on a test
batch of in-category + *unseen-category* documents measures:

  1. OOD AUROC -- does the kNN-distance OOD score separate unseen-category docs
     from in-category ones, vs a max-softmax baseline?
  2. Accuracy by confidence tier on the in-category split (labels valid there).

Raw text can't be scored directly, so the Embedder is load-bearing here -- unlike
a tabular setup where the features already are the input.
"""

from __future__ import annotations

import os

# On macOS, multiple OpenMP runtimes in one process (faiss-cpu, torch, and an
# Anaconda-MKL numpy) clash and segfault under multithreading. Skip faiss (brute
# kNN is used here) and serialize OpenMP. Must be set before importing numpy/torch.
os.environ.setdefault("PITWALLER_NO_FAISS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pitwaller import ConfidencePipeline, Tier
from pitwaller.embeddings import SentenceTransformerEmbedder

IN_CATEGORIES = [
    "comp.graphics", "rec.sport.baseball", "sci.med",
    "talk.politics.guns", "rec.autos",
]
OOD_CATEGORIES = [
    "sci.space", "soc.religion.christian", "misc.forsale", "talk.politics.mideast",
]
_STRIP = ("headers", "footers", "quotes")  # don't let the classifier cheat on metadata
MAX_DOCS = 800  # cap per split so CPU embedding stays quick


def _load(subset: str, categories: list[str]):
    data = fetch_20newsgroups(subset=subset, categories=categories, remove=_STRIP)
    docs, y = data.data, data.target
    if len(docs) > MAX_DOCS:
        idx = np.random.default_rng(0).choice(len(docs), MAX_DOCS, replace=False)
        docs, y = [docs[i] for i in idx], y[idx]
    return docs, y


def main() -> None:
    emb = SentenceTransformerEmbedder()

    train_docs, train_y = _load("train", IN_CATEGORIES)
    in_docs, in_y = _load("test", IN_CATEGORIES)
    ood_docs, _ = _load("test", OOD_CATEGORIES)

    # Embed once; the classifier and the OOD model share this embedding space.
    # (pipe.score(raw_docs) would embed internally -- we reuse the vectors here.)
    print(f"Embedding {len(train_docs)}+{len(in_docs)}+{len(ood_docs)} docs on CPU "
          "(one-time)...", flush=True)
    E_tr = emb.embed(train_docs)
    E_in = emb.embed(in_docs)
    E_ood = emb.embed(ood_docs)

    clf = LogisticRegression(max_iter=1000).fit(E_tr, train_y)
    # brute-force kNN (exact, instant at this scale) avoids loading faiss's OpenMP
    # alongside torch; "auto" (FAISS) is the production default for large indexes.
    pipe = ConfidencePipeline(emb, k=10, contamination=0.05, index_backend="brute").fit(E_tr)

    scored_in = pipe.score(E_in)
    scored_ood = pipe.score(E_ood)

    # 1. OOD AUROC: kNN-distance vs max-softmax.
    is_ood = np.r_[np.zeros(len(in_docs)), np.ones(len(ood_docs))]
    knn = np.array([s.ood.knn_distance for s in scored_in]
                   + [s.ood.knn_distance for s in scored_ood])
    msp = 1.0 - clf.predict_proba(np.r_[E_in, E_ood]).max(axis=1)
    print(f"In-distribution categories : {IN_CATEGORIES}")
    print(f"Unseen (OOD) categories    : {OOD_CATEGORIES}\n")
    print("OOD detection (AUROC, unseen-category vs in-category):")
    print(f"  pitwaller kNN-distance : {roc_auc_score(is_ood, knn):.3f}")
    print(f"  max-softmax baseline   : {roc_auc_score(is_ood, msp):.3f}")

    # 2. Accuracy by tier on the in-category split (labels are valid there).
    correct = clf.predict(E_in) == in_y
    tiers = [s.tier for s in scored_in]
    print("\nAccuracy by confidence tier (in-category test docs):")
    for t in Tier:
        idx = [i for i, ti in enumerate(tiers) if ti is t]
        if idx:
            frac, acc = len(idx) / len(tiers), correct[idx].mean()
            print(f"  {t.value:<4} n={len(idx):>4} ({frac:.0%})  accuracy {acc:.1%}")
        else:
            print(f"  {t.value:<4} n=   0")
    print(f"\n  overall in-category accuracy = {correct.mean():.1%}")


if __name__ == "__main__":
    main()
