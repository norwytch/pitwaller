"""End-to-end demo on synthetic data -- runs with no weights, no dataset.

    python -m pitwaller.demo

Walks through the whole system: fit the OOD reference model on a synthetic
"training manifold", score a production batch that deliberately includes
in-distribution, margin, and out-of-distribution samples, tier them, aggregate
into diagnostics, and print the remediation recommendation.
"""

from __future__ import annotations

import numpy as np

from .confidence import Tier
from .decisions import PolicyThresholds, group_by_effort, heaviest_tier, recommend
from .embeddings import MockEmbedder
from .monitoring import PredictionRecord, aggregate
from .pipeline import ConfidencePipeline


def _make_inputs(specs):
    """specs: list of (cluster_id, jitter); returns embedder-ready tuples."""
    return [(cid, jit, i) for i, (cid, jit) in enumerate(specs)]


def main() -> None:
    rng = np.random.default_rng(42)
    embedder = MockEmbedder(dim=64, n_clusters=8, seed=1)

    # --- Training set: tight samples drawn from the 8 known clusters ----------
    train_specs = [(int(rng.integers(0, 8)), 0.4) for _ in range(2000)]
    train_inputs = _make_inputs(train_specs)

    pipe = ConfidencePipeline(embedder, k=10, contamination=0.05).fit(train_inputs)
    print(f"Fitted OOD model on {len(train_inputs)} samples "
          f"(p50={pipe.ood.p50:.4f}, p90={pipe.ood.p90:.4f})\n")

    # --- Production batch: mostly in-distribution + a slug of drift/OOD -------
    prod_specs = (
        [(int(rng.integers(0, 8)), 0.4) for _ in range(140)]    # core
        + [(int(rng.integers(0, 8)), 1.6) for _ in range(40)]   # margin (noisy)
        + [(99, 0.5) for _ in range(20)]                        # off-manifold OOD
    )
    prod_inputs = _make_inputs(prod_specs)
    scored = pipe.score(prod_inputs)

    counts = {t: 0 for t in Tier}
    for s in scored:
        counts[s.tier] += 1
    print("Tier distribution on production batch:")
    for t in Tier:
        print(f"  {t.value:<4} {counts[t]:>3}  ({counts[t] / len(scored):.0%})")
    print()

    # --- Synthesise records with labels: accuracy degrades with OOD distance --
    records = []
    for spec, s in zip(prod_specs, scored):
        cluster_id = spec[0]
        # In-distribution predictions are usually right; OOD ones often wrong --
        # this is the monotonic OOD-vs-accuracy relationship the system exploits.
        p_correct = {"core": 0.97, "margin": 0.80, "outlier": 0.45}[s.ood.band]
        true_label = cluster_id if 0 <= cluster_id < 8 else int(rng.integers(0, 8))
        correct = rng.random() < p_correct
        pred_label = true_label if correct else (true_label + 1) % 8
        records.append(
            PredictionRecord(ood=s.ood, tier=s.tier, pred_label=pred_label,
                             true_label=true_label)
        )

    diag = aggregate(records, baseline_high_rate=0.85, baseline_accuracy=0.95)
    print("Diagnostics:")
    print(f"  n={diag.n}  OOD={diag.ood_rate:.0%}  margin={diag.margin_rate:.0%}  "
          f"IF={diag.if_outlier_rate:.0%}")
    print(f"  accuracy overall = {diag.accuracy_overall:.0%} "
          f"(baseline 95%, drop {diag.accuracy_drop:.1%})")
    print("  accuracy by tier = "
          + ", ".join(f"{k}:{'-' if v is None else f'{v:.0%}'}"
                      for k, v in diag.accuracy_by_tier.items()))
    print()

    recs = recommend(diag, PolicyThresholds())
    print(f"Remediation -- biggest job required: {heaviest_tier(recs).value}\n")
    for tier, group in group_by_effort(recs).items():
        print(f"[{tier.value}]")
        for r in group:
            e = r.effort
            live = "stays live" if e.stays_live else "redeploy"
            labels = "labels" if e.needs_labels else "no labels"
            print(f"  - {r.action.value}  ({e.typical_duration}, "
                  f"gpu:{e.gpu_intensity}, {labels}, {live})")
            print(f"      {r.rationale}")


if __name__ == "__main__":
    main()
