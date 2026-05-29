import numpy as np
import pytest

from pitwaller.index import HNSWConfig, VectorIndex, _HAVE_FAISS


def _data(n=500, d=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_build_and_search_brute():
    X = _data()
    idx = VectorIndex(dim=16, backend="brute").build(X)
    assert idx.size == 500
    dist, ind = idx.search(X[:5], k=3)
    assert dist.shape == (5, 3) and ind.shape == (5, 3)
    # Nearest neighbour of a training point is itself, distance ~0.
    assert np.allclose(dist[:, 0], 0.0, atol=1e-4)
    assert (ind[:, 0] == np.arange(5)).all()


def test_neighbours_sorted_by_distance():
    X = _data()
    idx = VectorIndex(dim=16, backend="brute").build(X)
    dist, _ = idx.search(X[:10], k=5)
    assert (np.diff(dist, axis=1) >= -1e-5).all()  # non-decreasing


def test_dim_mismatch_raises():
    idx = VectorIndex(dim=16, backend="brute")
    with pytest.raises(ValueError):
        idx.build(_data(d=8))


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        VectorIndex(dim=16, backend="brute").search(_data(n=1), k=1)


@pytest.mark.skipif(not _HAVE_FAISS, reason="faiss not installed")
def test_faiss_matches_brute_on_top1():
    X = _data(n=300, d=32)
    q = _data(n=20, d=32, seed=99)
    faiss_idx = VectorIndex(32, HNSWConfig(ef_search=200), backend="faiss").build(X)
    brute_idx = VectorIndex(32, backend="brute").build(X)
    _, fi = faiss_idx.search(q, k=1)
    _, bi = brute_idx.search(q, k=1)
    # HNSW is approximate, but with a wide beam top-1 recall should be high.
    agreement = (fi[:, 0] == bi[:, 0]).mean()
    assert agreement >= 0.8
