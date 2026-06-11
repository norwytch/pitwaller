import numpy as np
import pytest

from pitwaller import ConfidencePipeline, MockEmbedder
from pitwaller.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    average_precision,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)


class VecEmbedder:
    """Items are already vectors; embed() just stacks them."""

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def embed(self, batch):
        return np.asarray(list(batch), dtype=np.float32)


# --------------------------------------------------- metrics vs hand-computed


def test_metrics_match_reference_values():
    rel = np.array([1, 0, 1, 0])
    assert recall_at_k(rel, n_relevant=3, k=4) == pytest.approx(2 / 3)
    assert precision_at_k(rel, k=2) == pytest.approx(0.5)
    # AP = (P@1 + P@3) / min(n_relevant, len) = (1 + 2/3) / 3
    assert average_precision(rel, n_relevant=3) == pytest.approx((1.0 + 2 / 3) / 3)
    assert reciprocal_rank(np.array([0, 0, 1, 0])) == pytest.approx(1 / 3)
    assert reciprocal_rank(np.array([0, 0, 0])) == 0.0


# --------------------------------------------------------------- retrievers


def test_dense_retrieves_self_first():
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(120, 16)).astype(np.float32)
    dr = DenseRetriever(VecEmbedder(16)).index(corpus)
    for i, ranked in enumerate(dr.retrieve(corpus[:5], k=3)):
        assert ranked[0] == i  # a point's nearest neighbour is itself


def test_bm25_ranks_keyword_matches_first():
    corpus = [
        "the cat sat on the mat",
        "dogs are loyal companions",
        "a feline cat purrs softly",
        "stock markets fell sharply today",
    ]
    ranked = BM25Retriever().index(corpus).retrieve(["cat"], k=4)[0]
    assert set(ranked[:2]) == {0, 2}  # only docs 0 and 2 mention "cat"


def test_rrf_promotes_item_both_lists_rank_high():
    fused = reciprocal_rank_fusion([np.array([5, 1, 2]), np.array([5, 3, 4])], k=3)
    assert fused[0] == 5


def test_hybrid_fuses_both_retrievers():
    class Fixed:
        def __init__(self, ranking):
            self.ranking = np.asarray(ranking)

        def index(self, *a, **k):
            return self

        def retrieve(self, queries, k=10):
            return [self.ranking[:k] for _ in queries]

    hybrid = HybridRetriever(Fixed([0, 1, 2, 3]), Fixed([3, 2, 1, 0]))
    fused = hybrid.retrieve(["q"], k=4)[0]
    assert set(fused) == {0, 1, 2, 3}  # contributions from both rankings


# --------------------------------------------------------------- evaluation


def test_evaluate_retrieval_on_clustered_corpus():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(3, 16))
    labels = np.repeat([0, 1, 2], 40)
    corpus = np.vstack([centers[c] + 0.1 * rng.normal(size=16) for c in labels]).astype(np.float32)
    q_labels = np.array([0, 1, 2, 0, 1, 2])
    queries = np.vstack([centers[c] + 0.1 * rng.normal(size=16) for c in q_labels]).astype(np.float32)

    dr = DenseRetriever(VecEmbedder(16)).index(corpus, labels=labels)
    metrics = evaluate_retrieval(dr, queries, q_labels, labels, k=10)
    assert set(metrics) == {"recall@k", "precision@k", "map", "mrr"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["precision@k"] > 0.8  # tight clusters -> same-label neighbours


# --------------------------------------------------- OOD-core connection


def test_pipeline_neighbors_returns_self():
    emb = MockEmbedder(dim=32, seed=1)
    feats = emb.embed([(i % 8, 0.3, i) for i in range(200)])
    pipe = ConfidencePipeline(emb).fit(feats)
    dist, idx = pipe.neighbors(feats[:5], k=3)
    assert idx.shape == (5, 3)
    assert (idx[:, 0] == np.arange(5)).all()
    assert np.allclose(dist[:, 0], 0.0, atol=1e-4)
