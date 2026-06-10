"""Covariate-shift text benchmark (the method's intended strength).

Trains a sentiment classifier on movie reviews (Rotten Tomatoes), then feeds it
product reviews (Amazon) -- a shifted input domain with the *same* pos/neg labels.
Under covariate shift max-softmax tends to stay overconfident while accuracy
falls, which is exactly the failure an embedding-space OOD score should catch.

    pip install -e '.[text]'
    python examples/benchmark_covariate_shift.py

Reports, on a mixed in-domain + shifted batch:
  1. AUROC for flagging the shifted domain -- kNN-distance vs max-softmax.
  2. The accuracy drop under shift, and that softmax confidence stays high anyway.
  3. Accuracy by confidence tier (labels valid in both domains, so it's real).
"""

from __future__ import annotations

import itertools
import os

# Multiple OpenMP runtimes (torch + Anaconda-MKL numpy) clash and segfault under
# multithreading on macOS; skip faiss and serialize OpenMP before importing them.
os.environ.setdefault("PITWALLER_NO_FAISS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pitwaller import ConfidencePipeline, Tier
from pitwaller.embeddings import SentenceTransformerEmbedder

MAX_DOCS = 1500   # cap movie splits so CPU embedding stays quick
N_SHIFT = 2000    # shifted-domain (product) reviews to stream
_AMAZON_IDS = ("amazon_polarity", "fancyzhx/amazon_polarity")  # HF id has moved over time


def _subsample(docs, y, n, seed=0):
    if len(docs) <= n:
        return docs, y
    idx = np.random.default_rng(seed).choice(len(docs), n, replace=False)
    return [docs[i] for i in idx], y[idx]


def load_data():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("This benchmark needs `datasets`: pip install -e '.[text]'") from exc

    movie = load_dataset("rotten_tomatoes")  # small; label 0=neg, 1=pos
    train_docs, train_y = list(movie["train"]["text"]), np.array(movie["train"]["label"])
    in_docs, in_y = list(movie["test"]["text"]), np.array(movie["test"]["label"])

    rows = None
    for name in _AMAZON_IDS:  # product reviews; same 0=neg, 1=pos convention
        try:
            stream = load_dataset(name, split="test", streaming=True)
            rows = list(itertools.islice(stream, N_SHIFT))
            break
        except Exception:
            continue
    if not rows:
        raise SystemExit("could not stream a product-review dataset for the shifted domain")
    shift_docs = [r["content"] for r in rows]
    shift_y = np.array([r["label"] for r in rows])
    return train_docs, train_y, in_docs, in_y, shift_docs, shift_y


def main() -> None:
    emb = SentenceTransformerEmbedder()
    train_docs, train_y, in_docs, in_y, shift_docs, shift_y = load_data()
    train_docs, train_y = _subsample(train_docs, train_y, MAX_DOCS)
    in_docs, in_y = _subsample(in_docs, in_y, MAX_DOCS)

    print(f"Embedding {len(train_docs)}+{len(in_docs)}+{len(shift_docs)} docs "
          "(movie train/test + shifted product reviews)...", flush=True)
    E_tr, E_in, E_sh = emb.embed(train_docs), emb.embed(in_docs), emb.embed(shift_docs)

    clf = LogisticRegression(max_iter=1000).fit(E_tr, train_y)
    pipe = ConfidencePipeline(emb, k=10, contamination=0.05, index_backend="brute").fit(E_tr)
    sc_in, sc_sh = pipe.score(E_in), pipe.score(E_sh)

    # 1. Flag the shifted domain: kNN-distance vs max-softmax.
    is_shift = np.r_[np.zeros(len(in_docs)), np.ones(len(shift_docs))]
    knn = np.array([s.ood.knn_distance for s in sc_in] + [s.ood.knn_distance for s in sc_sh])
    msp = 1.0 - clf.predict_proba(np.r_[E_in, E_sh]).max(axis=1)
    print("\nDetecting the shifted domain (AUROC, product vs movie):")
    print(f"  pitwaller kNN-distance : {roc_auc_score(is_shift, knn):.3f}")
    print(f"  max-softmax baseline   : {roc_auc_score(is_shift, msp):.3f}")

    # 2. Accuracy drops under shift, but softmax confidence stays high.
    acc_in = float((clf.predict(E_in) == in_y).mean())
    acc_sh = float((clf.predict(E_sh) == shift_y).mean())
    conf_in = float(clf.predict_proba(E_in).max(axis=1).mean())
    conf_sh = float(clf.predict_proba(E_sh).max(axis=1).mean())
    print(f"\nAccuracy   in-domain {acc_in:.1%}  ->  shifted {acc_sh:.1%}")
    print(f"Mean softmax conf.  in-domain {conf_in:.1%}  shifted {conf_sh:.1%} "
          "(stays high while accuracy falls)")

    # 3. Accuracy by tier on the combined batch (labels valid in both domains).
    correct = clf.predict(np.r_[E_in, E_sh]) == np.r_[in_y, shift_y]
    tiers = [s.tier for s in (sc_in + sc_sh)]
    print("\nAccuracy by confidence tier (movie test + shifted product reviews):")
    for t in Tier:
        idx = [i for i, ti in enumerate(tiers) if ti is t]
        if idx:
            frac, acc = len(idx) / len(tiers), correct[idx].mean()
            print(f"  {t.value:<4} n={len(idx):>4} ({frac:.0%})  accuracy {acc:.1%}")
        else:
            print(f"  {t.value:<4} n=   0")
    print(f"\n  overall accuracy = {correct.mean():.1%}")


if __name__ == "__main__":
    main()
