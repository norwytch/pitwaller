"""Label-calibrated confidence tiers.

Given a labelled calibration set, defines HIGH/MED/LOW tiers by their selective
risk rather than by training-distance percentiles. Two steps:

1. Fuse the OOD signals into one reliability score: a monotonic map
   ``[kNN distance, IF score, ...] -> P(correct)`` (standardise + logistic
   regression), audited with a reliability diagram / ECE.
2. Place the tier cuts at risk targets on that score: HIGH keeps selective risk
   (error rate among accepted samples) under ``risk_high`` (default 1%), MED
   under ``risk_med`` (default 5%), everything else LOW. With ``delta`` set each
   cut carries a finite-sample guarantee (see :func:`risk_targeted_threshold`).

Opt-in: :class:`~pitwaller.pipeline.ConfidencePipeline` stays on p50/p90 tiering
until ``calibrate`` is called. Pure NumPy + scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .confidence import Tier
from .ood import OODResult


def ood_features(results: list[OODResult]) -> np.ndarray:
    """Default fusion features: ``(N, 2)`` array of ``[knn_distance, if_score]``.

    Pass your own ``feature_fn`` to :meth:`TierCalibrator.fit` (and the pipeline)
    to add per-sample signals such as max-softmax, logit margin, or ensemble
    disagreement.
    """
    return np.array([[r.knn_distance, r.if_score] for r in results], dtype=float)


# --------------------------------------------------------------------------- #
# 1. Reliability map: signals -> calibrated P(correct)                         #
# --------------------------------------------------------------------------- #


class ReliabilityModel:
    """Monotonic map from fusion features to ``P(correct)``.

    Standardises the features and fits a logistic regression; the predicted
    probability of correctness is the reliability score the tiers are cut on.
    A calibration set with a single observed outcome collapses to the constant
    base rate rather than failing.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.clf_: LogisticRegression | None = None
        self.constant_: float | None = None  # set iff only one class is present

    def fit(self, X: np.ndarray, correct: np.ndarray) -> "ReliabilityModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(correct, dtype=bool)
        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
            raise ValueError("X must be (N, F) and aligned with a non-empty correct array")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # guard constant columns
        if y.all() or (~y).all():
            self.constant_ = float(y.mean())  # 1.0 or 0.0; nothing to discriminate
            return self
        self.constant_ = None
        Xs = (X - self.mean_) / self.std_
        self.clf_ = LogisticRegression(max_iter=1000).fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(correct)`` for each row of ``X``."""
        if self.mean_ is None:
            raise RuntimeError("ReliabilityModel is not fit; call fit() first")
        X = np.asarray(X, dtype=float)
        if self.constant_ is not None:
            return np.full(X.shape[0], self.constant_)
        Xs = (X - self.mean_) / self.std_
        return self.clf_.predict_proba(Xs)[:, 1]

    @property
    def coefficients(self) -> np.ndarray | None:
        """Standardised logistic coefficients, one per feature (``None`` if the
        model degenerated to a constant)."""
        return None if self.clf_ is None else self.clf_.coef_.ravel()

    def reliability_diagram(
        self, X: np.ndarray, correct: np.ndarray, n_bins: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Binned ``(mean_confidence, empirical_accuracy, count)``. A
        well-calibrated map sits on the diagonal."""
        p = self.predict(X)
        y = np.asarray(correct, dtype=float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        conf = np.full(n_bins, np.nan)
        acc = np.full(n_bins, np.nan)
        count = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            m = idx == b
            count[b] = int(m.sum())
            if m.any():
                conf[b] = float(p[m].mean())
                acc[b] = float(y[m].mean())
        return conf, acc, count

    def ece(self, X: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error: support-weighted mean gap between binned
        confidence and accuracy. Lower is better."""
        conf, acc, count = self.reliability_diagram(X, correct, n_bins)
        m = count > 0
        if not m.any():
            return 0.0
        return float(np.sum(count[m] * np.abs(conf[m] - acc[m])) / count.sum())


# --------------------------------------------------------------------------- #
# 2. Risk-targeted thresholds on the reliability score                         #
# --------------------------------------------------------------------------- #


def risk_targeted_threshold(
    confidence: np.ndarray,
    correct: np.ndarray,
    target_risk: float,
    delta: float | None = None,
) -> float:
    """Lowest confidence cut whose accepted set keeps selective risk at target.

    Samples are accepted most-confident-first. Returns ``tau`` such that
    accepting ``confidence >= tau`` holds the selective risk (error rate among
    accepted) at or below ``target_risk``, over the largest such accepted set
    (maximum-coverage operating point). Largest-set rather than first-prefix:
    empirical risk is noisy at low coverage, where a single early error would
    truncate the tier.

    With ``delta`` set, the empirical risk is replaced by a Hoeffding upper
    confidence bound at level ``delta`` (RCPS-style). For the bound to transfer
    out-of-sample, ``confidence`` must come from a scorer fit on data independent
    of ``correct``; :class:`TierCalibrator` arranges this by sample-splitting.
    Too few calibration points and the cut degenerates to ``+inf``.

    Caveat: the bound is pointwise. The threshold is selected over the nested
    family by this same bound and is not Bonferroni/union-corrected for that
    selection, so treat ``delta`` as a per-cut level, not a simultaneous one. A
    union bound (``ln(n/delta)``) would be fully rigorous but much more
    conservative.

    ``+inf`` is returned when no accepted set meets the target (empty tier).
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.size != correct.size or confidence.size == 0:
        raise ValueError("confidence and correct must be non-empty and aligned")
    if not 0 < target_risk < 1:
        raise ValueError("target_risk must be in (0, 1)")

    order = np.argsort(-confidence)  # most confident first
    conf_sorted = confidence[order]
    c = correct[order]
    k = np.arange(1, c.size + 1)
    emp_risk = 1.0 - np.cumsum(c) / k

    if delta is None:
        risk = emp_risk  # empirical max-coverage operating point
    else:
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        # RCPS-style: certify the risk with a Hoeffding upper confidence bound.
        # The sqrt term is large for a tiny accept set and shrinks as it grows,
        # so the largest accept set whose bound holds is the max-coverage cut.
        risk = emp_risk + np.sqrt(np.log(1.0 / delta) / (2.0 * k))

    ok = risk <= target_risk
    if not ok.any():
        return float("inf")
    k_last = int(np.flatnonzero(ok).max()) + 1  # largest prefix under target
    return float(conf_sorted[k_last - 1])


# --------------------------------------------------------------------------- #
# 3. The calibrator: reliability map + risk-targeted HIGH/MED/LOW cuts         #
# --------------------------------------------------------------------------- #


@dataclass
class TierCalibration:
    """Fitted artefacts: the reliability map and the two tier cuts."""

    reliability: ReliabilityModel
    tau_high: float
    tau_med: float
    risk_high: float
    risk_med: float
    delta: float | None
    feature_fn: Callable[[list[OODResult]], np.ndarray] = field(default=ood_features)


class TierCalibrator:
    """Fit calibrated HIGH/MED/LOW cuts from a labelled calibration set.

    ``fit`` takes a feature matrix (or, via :meth:`fit_results`, a list of
    :class:`~pitwaller.ood.OODResult`) plus per-sample correctness. ``tier`` /
    ``tier_results`` then assign tiers to new samples.
    """

    def __init__(
        self,
        risk_high: float = 0.01,
        risk_med: float = 0.05,
        delta: float | None = None,
        feature_fn: Callable[[list[OODResult]], np.ndarray] = ood_features,
    ):
        if not 0 < risk_high <= risk_med < 1:
            raise ValueError("require 0 < risk_high <= risk_med < 1")
        self.risk_high = risk_high
        self.risk_med = risk_med
        self.delta = delta
        self.feature_fn = feature_fn
        self.calibration_: TierCalibration | None = None

    def fit(
        self,
        features: np.ndarray,
        correct: np.ndarray,
        calibration_fraction: float = 0.5,
        seed: int = 0,
    ) -> "TierCalibrator":
        """Fit the reliability map and place the tier cuts.

        The map is fit on one split and the cuts certified on a disjoint one.
        Reusing the same data for both biases the risk estimate optimistically
        (the score is trained to make confident points correct on the exact set
        the bound is computed over), voiding the ``delta`` guarantee
        out-of-sample. Sample-splitting keeps the score independent of the
        certification data, as RCPS / Learn-then-Test require. (A K-fold cross-fit
        would use the data more efficiently.)
        """
        features = np.asarray(features, dtype=float)
        correct = np.asarray(correct, dtype=bool)
        n = correct.size
        if not 0.0 < calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")
        perm = np.random.default_rng(seed).permutation(n)
        n_cal = max(1, min(n - 1, int(round(calibration_fraction * n))))
        cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]

        rel = ReliabilityModel().fit(features[fit_idx], correct[fit_idx])
        cal_score = rel.predict(features[cal_idx])  # independent of the fit fold
        cal_correct = correct[cal_idx]
        tau_high = risk_targeted_threshold(cal_score, cal_correct, self.risk_high, self.delta)
        tau_med = risk_targeted_threshold(cal_score, cal_correct, self.risk_med, self.delta)
        # Enforce HIGH ⊆ MED nesting through finite-sample noise: a looser risk
        # budget can never demand a higher cut than a tighter one.
        tau_med = min(tau_med, tau_high)
        self.calibration_ = TierCalibration(
            reliability=rel,
            tau_high=tau_high,
            tau_med=tau_med,
            risk_high=self.risk_high,
            risk_med=self.risk_med,
            delta=self.delta,
            feature_fn=self.feature_fn,
        )
        return self

    def fit_results(
        self, results: list[OODResult], correct: np.ndarray
    ) -> "TierCalibrator":
        """Convenience: fit straight from OOD readouts using ``feature_fn``."""
        return self.fit(self.feature_fn(results), correct)

    def reliability_score(self, features: np.ndarray) -> np.ndarray:
        return self._cal().reliability.predict(features)

    def tier(self, features: np.ndarray) -> list[Tier]:
        """Assign tiers to a feature matrix."""
        cal = self._cal()
        s = cal.reliability.predict(features)
        return [self._tier_one(si, cal) for si in s]

    def tier_results(self, results: list[OODResult]) -> list[Tier]:
        """Assign tiers to OOD readouts via the stored ``feature_fn``."""
        cal = self._cal()
        return self.tier(cal.feature_fn(results))

    @staticmethod
    def _tier_one(score: float, cal: TierCalibration) -> Tier:
        if score >= cal.tau_high:
            return Tier.HIGH
        if score >= cal.tau_med:
            return Tier.MED
        return Tier.LOW

    def _cal(self) -> TierCalibration:
        if self.calibration_ is None:
            raise RuntimeError("TierCalibrator is not fit; call fit() first")
        return self.calibration_
