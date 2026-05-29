"""Production monitoring.

Turns a stream of per-prediction records into the aggregate **diagnostics**
that the remediation policy consumes. This is the layer that watches the model
in production and decides *what is wrong*, separately from *what to do about it*
(that's :mod:`pitwaller.decisions`).

A ``PredictionRecord`` is what you log per inference. Ground truth is optional
because most of it arrives late (or never); the diagnostics degrade gracefully
when labels are sparse, falling back on the unsupervised OOD signals.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .confidence import Tier
from .ood import OODResult


@dataclass
class PredictionRecord:
    """One production inference."""

    ood: OODResult
    tier: Tier
    pred_label: int
    true_label: int | None = None  # may arrive later, or never


@dataclass
class Diagnostics:
    """Aggregate health signals over a monitoring window.

    All rates are in ``[0, 1]``. ``*_baseline`` fields, when set, let the policy
    reason about *drift* rather than absolute level.
    """

    n: int
    ood_rate: float                       # fraction in the "outlier" band
    margin_rate: float                    # fraction in the "margin" band
    if_outlier_rate: float
    tier_distribution: dict[str, float]   # HIGH/MED/LOW -> fraction
    accuracy_overall: float | None
    accuracy_by_tier: dict[str, float | None]
    per_class_recall: dict[int, float] = field(default_factory=dict)
    per_class_support: dict[int, int] = field(default_factory=dict)  # labelled count per class
    labelled_fraction: float = 0.0

    # Optional baselines captured at validation/deploy time.
    baseline_high_rate: float | None = None
    baseline_accuracy: float | None = None

    @property
    def high_rate(self) -> float:
        return self.tier_distribution.get(Tier.HIGH.value, 0.0)

    @property
    def high_rate_drop(self) -> float | None:
        """How far the HIGH-confidence fraction has fallen vs baseline."""
        if self.baseline_high_rate is None:
            return None
        return self.baseline_high_rate - self.high_rate

    @property
    def accuracy_drop(self) -> float | None:
        if self.baseline_accuracy is None or self.accuracy_overall is None:
            return None
        return self.baseline_accuracy - self.accuracy_overall


def aggregate(
    records: list[PredictionRecord],
    baseline_high_rate: float | None = None,
    baseline_accuracy: float | None = None,
) -> Diagnostics:
    """Collapse a window of records into :class:`Diagnostics`."""
    n = len(records)
    if n == 0:
        raise ValueError("cannot aggregate an empty window")

    ood_rate = np.mean([r.ood.band == "outlier" for r in records])
    margin_rate = np.mean([r.ood.band == "margin" for r in records])
    if_rate = np.mean([r.ood.if_outlier for r in records])

    tier_counts = Counter(r.tier.value for r in records)
    tier_dist = {t.value: tier_counts.get(t.value, 0) / n for t in Tier}

    labelled = [r for r in records if r.true_label is not None]
    labelled_fraction = len(labelled) / n

    if labelled:
        accuracy_overall = float(
            np.mean([r.pred_label == r.true_label for r in labelled])
        )
        acc_by_tier: dict[str, float | None] = {}
        for t in Tier:
            sub = [r for r in labelled if r.tier == t]
            acc_by_tier[t.value] = (
                float(np.mean([r.pred_label == r.true_label for r in sub])) if sub else None
            )
        # Per-class recall.
        hit = defaultdict(int)
        total = defaultdict(int)
        for r in labelled:
            total[r.true_label] += 1
            if r.pred_label == r.true_label:
                hit[r.true_label] += 1
        per_class_recall = {c: hit[c] / total[c] for c in total}
        per_class_support = dict(total)
    else:
        accuracy_overall = None
        acc_by_tier = {t.value: None for t in Tier}
        per_class_recall = {}
        per_class_support = {}

    return Diagnostics(
        n=n,
        ood_rate=float(ood_rate),
        margin_rate=float(margin_rate),
        if_outlier_rate=float(if_rate),
        tier_distribution=tier_dist,
        accuracy_overall=accuracy_overall,
        accuracy_by_tier=acc_by_tier,
        per_class_recall=per_class_recall,
        per_class_support=per_class_support,
        labelled_fraction=labelled_fraction,
        baseline_high_rate=baseline_high_rate,
        baseline_accuracy=baseline_accuracy,
    )
