"""Retrieval benchmark: dense (sentence-transformer) vs BM25 vs hybrid (RRF).

    pip install -e '.[text]'
    python examples/benchmark_retrieval.py

Indexes 20-Newsgroups documents and queries with held-out documents; a retrieved
item is relevant when it shares the query's newsgroup. Reports recall@k /
precision@k / MAP / MRR for dense embedding search, BM25, and their
reciprocal-rank fusion.
"""

from __future__ import annotations

import os

# macOS: clashing OpenMP runtimes (torch + MKL numpy) segfault, so skip faiss and
# serialize OpenMP before importing them.
os.environ.setdefault("PITWALLER_NO_FAISS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from pitwaller.embeddings import SentenceTransformerEmbedder
from pitwaller.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    evaluate_retrieval,
)

CATEGORIES = [
    "comp.graphics", "rec.sport.baseball", "sci.med",
    "talk.politics.guns", "rec.autos", "sci.space",
]
_STRIP = ("headers", "footers", "quotes")
MAX_CORPUS = 2000
MAX_QUERIES = 500
K = 10


def _subsample(docs, labels, n, seed):
    if len(docs) <= n:
        return docs, labels
    idx = np.random.default_rng(seed).choice(len(docs), n, replace=False)
    return [docs[i] for i in idx], labels[idx]


def main() -> None:
    train = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=_STRIP)
    test = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=_STRIP)
    corpus, corpus_labels = _subsample(train.data, train.target, MAX_CORPUS, 0)
    queries, query_labels = _subsample(test.data, test.target, MAX_QUERIES, 1)

    print(f"corpus={len(corpus)}  queries={len(queries)}  "
          f"categories={len(CATEGORIES)}  (relevant = same newsgroup)\n", flush=True)

    print("Indexing dense (MiniLM) + BM25 ...", flush=True)
    dense = DenseRetriever(SentenceTransformerEmbedder()).index(corpus, labels=corpus_labels)
    sparse = BM25Retriever().index(corpus, labels=corpus_labels)
    hybrid = HybridRetriever(dense, sparse)  # reuses the already-indexed retrievers

    results = {
        "dense (MiniLM)": evaluate_retrieval(dense, queries, query_labels, corpus_labels, K),
        "BM25 (sparse)": evaluate_retrieval(sparse, queries, query_labels, corpus_labels, K),
        "hybrid (RRF)": evaluate_retrieval(hybrid, queries, query_labels, corpus_labels, K),
    }

    cols = ["recall@k", "precision@k", "map", "mrr"]
    print(f"\n{'retriever':<16}" + "".join(f"{c:>13}" for c in cols))
    for name, metrics in results.items():
        print(f"{name:<16}" + "".join(f"{metrics[c]:>13.3f}" for c in cols))
    print(f"\n(k = {K})")


if __name__ == "__main__":
    main()
