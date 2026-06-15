import numpy as np
import pytest

from pitwaller.embeddings import MockEmbedder
from pitwaller.ood import OODModel, _knn_excluding_self


def test_knn_excluding_self_masks_by_id_not_column():
    # Self id (0) lands in column 1 at distance 0, as an approximate index can
    # return it; genuine neighbours are 10 (col 0) and 20 (col 2).
    out = _knn_excluding_self(np.array([[10.0, 0.0, 20.0]]), np.array([[1, 0, 2]]), k=2)
    assert out[0] == pytest.approx(15.0)  # (10+20)/2, not (0+20)/2 from a blind col-0 drop


def test_knn_excluding_self_exact_case_drops_column_zero():
    # Self at column 0 (exact index): identical to dropping column 0.
    out = _knn_excluding_self(np.array([[0.0, 3.0, 5.0]]), np.array([[0, 4, 7]]), k=2)
    assert out[0] == pytest.approx(4.0)  # (3+5)/2


def test_knn_excluding_self_keeps_true_duplicate_at_zero():
    # A genuine duplicate (different id) at distance 0 is a real neighbour, kept.
    out = _knn_excluding_self(np.array([[0.0, 0.0, 4.0]]), np.array([[0, 5, 2]]), k=2)
    assert out[0] == pytest.approx(2.0)  # self (id 0) masked; dup 0 and 4 kept


def _train_features(embedder, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    specs = [(int(rng.integers(0, 8)), 0.4, i) for i in range(n)]
    return embedder.embed(specs)


def test_thresholds_ordered():
    emb = MockEmbedder(dim=64, seed=1)
    model = OODModel(k=10).fit(_train_features(emb))
    assert model.p50 is not None and model.p90 is not None
    assert 0 <= model.p50 < model.p90


def test_core_samples_in_core_band():
    emb = MockEmbedder(dim=64, seed=1)
    model = OODModel(k=10).fit(_train_features(emb))
    # Fresh tight in-distribution samples should mostly land in the core.
    rng = np.random.default_rng(123)
    core = emb.embed([(int(rng.integers(0, 8)), 0.4, 10_000 + i) for i in range(200)])
    bands = [r.band for r in model.score(core)]
    core_frac = bands.count("core") / len(bands)
    assert core_frac > 0.4


def test_ood_samples_score_higher_distance():
    emb = MockEmbedder(dim=64, seed=1)
    model = OODModel(k=10).fit(_train_features(emb))
    rng = np.random.default_rng(7)
    indist = emb.embed([(int(rng.integers(0, 8)), 0.4, 20_000 + i) for i in range(150)])
    ood = emb.embed([(99, 0.5, 30_000 + i) for i in range(150)])
    d_in = np.mean([r.knn_distance for r in model.score(indist)])
    d_ood = np.mean([r.knn_distance for r in model.score(ood)])
    assert d_ood > d_in


def test_ood_samples_flagged_outlier_band():
    emb = MockEmbedder(dim=64, seed=1)
    model = OODModel(k=10).fit(_train_features(emb))
    ood = emb.embed([(99, 0.5, 40_000 + i) for i in range(150)])
    results = model.score(ood)
    outlier_frac = np.mean([r.band == "outlier" for r in results])
    if_frac = np.mean([r.if_outlier for r in results])
    # Far off-manifold points should be overwhelmingly flagged by both detectors.
    assert outlier_frac > 0.7
    assert if_frac > 0.5


def test_score_one_roundtrip():
    emb = MockEmbedder(dim=64, seed=1)
    model = OODModel(k=10).fit(_train_features(emb))
    one = emb.embed([(0, 0.4, 1)])[0]
    r = model.score_one(one)
    assert r.band in {"core", "margin", "outlier"}
    assert isinstance(r.if_outlier, bool)
