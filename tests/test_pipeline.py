import numpy as np

from pitwaller.confidence import Tier
from pitwaller.embeddings import MockEmbedder
from pitwaller.pipeline import ConfidencePipeline


def _specs(n, cluster_fn, jitter, offset):
    return [(cluster_fn(i), jitter, offset + i) for i in range(n)]


def test_pipeline_end_to_end_separates_confidence():
    rng = np.random.default_rng(0)
    emb = MockEmbedder(dim=64, seed=2)
    train = _specs(2000, lambda i: int(rng.integers(0, 8)), 0.4, 0)

    pipe = ConfidencePipeline(emb, k=10, contamination=0.05).fit(train)

    indist = emb.embed(_specs(200, lambda i: int(rng.integers(0, 8)), 0.4, 100_000))
    ood = emb.embed(_specs(200, lambda i: 99, 0.5, 200_000))

    high_frac = np.mean([s.tier == Tier.HIGH for s in pipe.score(indist)])
    low_frac = np.mean([s.tier == Tier.LOW for s in pipe.score(ood)])

    assert high_frac > 0.4   # most clean in-distribution samples are HIGH
    assert low_frac > 0.7    # most off-manifold samples are LOW


def test_pipeline_accepts_precomputed_features():
    emb = MockEmbedder(dim=64, seed=2)
    feats = emb.embed([(i % 8, 0.4, i) for i in range(500)])
    pipe = ConfidencePipeline(emb).fit(feats)  # pass features directly
    scored = pipe.score(feats[:10])
    assert len(scored) == 10


def test_score_before_fit_raises():
    import pytest

    pipe = ConfidencePipeline(MockEmbedder(dim=64))
    with pytest.raises(RuntimeError):
        pipe.score([(0, 0.4, 0)])
