"""Minimal quickstart: fit, score, tier, diagnose, recommend.

    python examples/quickstart.py
"""

import numpy as np

from pitwaller import (
    ConfidencePipeline,
    MockEmbedder,
    PredictionRecord,
    aggregate,
)
from pitwaller.experimental import recommend

rng = np.random.default_rng(0)
embedder = MockEmbedder(dim=64, n_clusters=8, seed=1)

# Training inputs: tight samples from the known clusters.
train = [(int(rng.integers(0, 8)), 0.4, i) for i in range(2000)]
pipe = ConfidencePipeline(embedder, k=10, contamination=0.05).fit(train)

# Production batch with some drift mixed in.
prod = [(int(rng.integers(0, 8)), 0.6, 100_000 + i) for i in range(120)]
prod += [(99, 0.5, 200_000 + i) for i in range(30)]  # off-manifold
scored = pipe.score(prod)

for s in scored[:8]:
    print(f"{s.tier.value:<4} band={s.ood.band:<7} "
          f"if_outlier={s.ood.if_outlier} dist={s.ood.knn_distance:.4f}")

# Pretend we got labels back and aggregate into diagnostics.
records = [
    PredictionRecord(ood=s.ood, tier=s.tier, pred_label=0,
                     true_label=0 if s.ood.band == "core" else 1)
    for s in scored
]
diag = aggregate(records, baseline_high_rate=0.85, baseline_accuracy=0.95)

print("\nRecommendations:")
for r in recommend(diag):
    print(f"  [{r.severity.value}] {r.action.value}: {r.rationale}")
