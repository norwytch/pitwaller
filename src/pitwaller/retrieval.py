"""Embedding-based retrieval over the same vector index the OOD detector uses.

The OOD core already builds a FAISS HNSW index over training embeddings and runs
kNN search against it. This module surfaces that as similarity search and adds a
sparse (BM25) retriever, a hybrid fusion of the two, and the standard retrieval
metrics (recall@k, precision@k, MAP, MRR).

* Dense retrieval reuses :class:`~pitwaller.index.VectorIndex` (FAISS HNSW, with a
  numpy brute-force fallback).
* BM25 is Okapi BM25 implemented on a scikit-learn ``CountVectorizer`` -- no extra
  dependency.
* Hybrid fuses the two ranked lists by reciprocal rank fusion (Cormack et al.,
  2009), which needs only the ranks, so dense distances and BM25 scores never
  have to be put on a common scale.

Relevance for evaluation is by label: a retrieved item is relevant to a query
when they share a label (e.g. same class / category).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .index import HNSWConfig, VectorIndex

# --------------------------------------------------------------------------- #
# Retrieval metrics (binary relevance over a ranked top-k list)               #
# --------------------------------------------------------------------------- #


def recall_at_k(relevance: np.ndarray, n_relevant: int, k: int) -> float:
    """Fraction of all relevant items recovered in the top ``k``."""
    if n_relevant <= 0:
        return 0.0
    return float(np.asarray(relevance)[:k].sum()) / n_relevant


def precision_at_k(relevance: np.ndarray, k: int) -> float:
    """Fraction of the top ``k`` that are relevant."""
    return float(np.asarray(relevance)[:k].sum()) / k


def average_precision(relevance: np.ndarray, n_relevant: int) -> float:
    """Average precision over the ranking, normalised by ``min(n_relevant, len)``."""
    rel = np.asarray(relevance, dtype=float)
    denom = min(n_relevant, rel.size)
    if denom <= 0:
        return 0.0
    precision_at_hit = np.cumsum(rel) / np.arange(1, rel.size + 1)
    return float((precision_at_hit * rel).sum() / denom)


def reciprocal_rank(relevance: np.ndarray) -> float:
    """1 / rank of the first relevant item (0 if none retrieved)."""
    hits = np.flatnonzero(np.asarray(relevance))
    return 1.0 / (hits[0] + 1) if hits.size else 0.0


# --------------------------------------------------------------------------- #
# Retrievers                                                                  #
# --------------------------------------------------------------------------- #


class DenseRetriever:
    """Embedding similarity search backed by :class:`VectorIndex` (FAISS HNSW).

    ``embedder`` turns items into vectors. :meth:`index` embeds and indexes a
    corpus; :meth:`retrieve` returns, per query, the ranked corpus indices of its
    nearest neighbours.
    """

    def __init__(self, embedder, index_config: HNSWConfig | None = None, index_backend: str = "auto"):
        self.embedder = embedder
        self.index_config = index_config
        self.index_backend = index_backend
        self._index: VectorIndex | None = None
        self.labels: np.ndarray | None = None

    def index(self, corpus, labels=None) -> "DenseRetriever":
        feats = np.ascontiguousarray(self.embedder.embed(corpus), dtype=np.float32)
        self._index = VectorIndex(feats.shape[1], self.index_config, self.index_backend).build(feats)
        self.labels = None if labels is None else np.asarray(labels)
        return self

    def retrieve(self, queries, k: int = 10) -> list[np.ndarray]:
        if self._index is None:
            raise RuntimeError("DenseRetriever is not indexed; call index() first")
        q = np.ascontiguousarray(self.embedder.embed(queries), dtype=np.float32)
        _, idx = self._index.search(q, k)  # search returns neighbours sorted by distance
        return [row[row >= 0] for row in idx]  # drop FAISS -1 padding if k > corpus


class BM25Retriever:
    """Okapi BM25 sparse retrieval over text, on a scikit-learn tokenizer."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def index(self, corpus, labels=None) -> "BM25Retriever":
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer()
        counts = self.vectorizer.fit_transform(corpus)  # (N, V) term counts
        self.counts = counts.tocsc()  # column slices for per-term frequencies
        self.n_docs = counts.shape[0]
        doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        self.doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.labels = None if labels is None else np.asarray(labels)
        return self

    def retrieve(self, queries, k: int = 10) -> list[np.ndarray]:
        q_counts = self.vectorizer.transform(queries).tocsr()
        # Length-normalisation denominator is the same for every query term.
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        out = []
        for i in range(q_counts.shape[0]):
            scores = np.zeros(self.n_docs)
            for term in q_counts[i].indices:
                tf = self.counts[:, term].toarray().ravel()
                scores += self.idf[term] * (tf * (self.k1 + 1.0)) / (tf + norm)
            out.append(np.argsort(-scores)[:k])
        return out


def reciprocal_rank_fusion(rankings: list[np.ndarray], k: int = 10, c: int = 60) -> np.ndarray:
    """Fuse ranked index lists by RRF: score(d) = sum_r 1 / (c + rank_r(d))."""
    scores: dict[int, float] = defaultdict(float)
    for ranked in rankings:
        for rank, idx in enumerate(ranked):
            scores[int(idx)] += 1.0 / (c + rank + 1)
    fused = sorted(scores, key=lambda d: -scores[d])
    return np.array(fused[:k], dtype=np.int64)


class HybridRetriever:
    """Dense + sparse, fused per query by reciprocal rank fusion."""

    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever, c: int = 60):
        self.dense = dense
        self.sparse = sparse
        self.c = c

    def index(self, corpus, labels=None) -> "HybridRetriever":
        self.dense.index(corpus, labels=labels)
        self.sparse.index(corpus, labels=labels)
        return self

    def retrieve(self, queries, k: int = 10) -> list[np.ndarray]:
        pool = max(5 * k, 50)  # fuse over a deeper pool than we return
        dense = self.dense.retrieve(queries, pool)
        sparse = self.sparse.retrieve(queries, pool)
        return [
            reciprocal_rank_fusion([dense[i], sparse[i]], k=k, c=self.c)
            for i in range(len(dense))
        ]


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #


def evaluate_retrieval(retriever, queries, query_labels, corpus_labels, k: int = 10) -> dict[str, float]:
    """Mean recall@k / precision@k / MAP / MRR over ``queries``.

    An item is relevant to a query when ``corpus_labels[item] == query_label``.
    Queries should be held out of the indexed corpus.
    """
    corpus_labels = np.asarray(corpus_labels)
    ranked = retriever.retrieve(queries, k)
    agg: dict[str, list[float]] = defaultdict(list)
    for retrieved, q_label in zip(ranked, np.asarray(query_labels)):
        relevance = (corpus_labels[retrieved] == q_label).astype(float)
        n_relevant = int((corpus_labels == q_label).sum())
        agg["recall@k"].append(recall_at_k(relevance, n_relevant, k))
        agg["precision@k"].append(precision_at_k(relevance, k))
        agg["map"].append(average_precision(relevance, n_relevant))
        agg["mrr"].append(reciprocal_rank(relevance))
    return {metric: float(np.mean(values)) for metric, values in agg.items()}
