"""Threshold selection and confidence evaluation with explicit guarantees.

Alternatives to eyeballed percentile cut-points and the equal-cost Youden's-J
operating point:

* Split / weighted conformal thresholds: a cut on a nonconformity score with a
  finite-sample, distribution-free bound on the rate at which conforming points
  are flagged. The weighted variant preserves that bound under covariate shift
  using density-ratio weights (Tibshirani, Foygel Barber, Candès & Ramdas,
  2019).

* Risk-coverage analysis: evaluate the whole confidence ordering as a selective
  predictor (Geifman & El-Yaniv, 2017) via the risk-coverage curve and AURC.

* Cost- and constraint-based operating points: minimise expected cost, or
  maximise coverage subject to an FPR / precision constraint (Neyman-Pearson).
  Youden's J is the equal-cost, prevalence-independent special case.

* Bootstrap CIs on a threshold: let the auto-QA layer fire a threshold
  adjustment only when the deployed cut is implausible under recent data.

Conventions
-----------
* Nonconformity / OOD scores: higher = more anomalous (e.g. kNN OOD distance).
  Conformal calibration runs on the population to cover (usually in-distribution).
* Confidence scores (risk-coverage): higher = more confident, accepted first.
  Convert an OOD distance with ``confidence = -distance``.
* Binary detection (operating points): the positive class is what the threshold
  flags (reject / outlier / error); ``scores`` are higher for positives, and a
  sample is flagged when ``score >= threshold``.

Pure NumPy; no model or framework dependency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Conformal thresholds


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    r"""Split-conformal threshold on a nonconformity score.

    Given calibration scores from the population to be covered and a target
    error rate ``alpha``, returns the ``ceil((n+1)(1-alpha))``-th smallest
    score. For exchangeable data this guarantees

        P( s(X_new) <= threshold ) >= 1 - alpha,

    i.e. a true conforming point is flagged (``s > threshold``) with probability
    at most ``alpha``. Returns ``+inf`` when there are too few calibration points
    to support the guarantee at this ``alpha`` (then flag nothing).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])  # k-th smallest, 1-indexed


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted ``q``-quantile (inverse weighted CDF, no interpolation)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        raise ValueError("weights must sum to a positive value")
    cutoff = q * cw[-1]
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    return float(v[min(idx, v.size - 1)])


def weighted_conformal_threshold(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    alpha: float,
    test_weight: float | None = None,
) -> float:
    r"""Weighted split-conformal threshold for covariate shift.

    When test inputs follow ``P_test`` but calibration data follow ``P_train``,
    standard conformal loses coverage. Re-weighting each calibration point by the
    density ratio ``w(x) = dP_test/dP_train`` restores it (Tibshirani et al.,
    2019); those weights are what an OOD / density model can estimate.

    With ``test_weight`` given, the test point contributes an atom at ``+inf``
    and the threshold is the weighted ``1-alpha`` quantile against the total
    normaliser. With ``test_weight=None`` it returns the calibration-only
    weighted ``1-alpha`` quantile, used when one shared threshold is applied
    across many test points. Returns ``+inf`` if the guarantee is unattainable.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    s = np.asarray(cal_scores, dtype=float)
    w = np.asarray(cal_weights, dtype=float)
    if s.size == 0 or s.size != w.size:
        raise ValueError("cal_scores and cal_weights must be non-empty and aligned")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    if test_weight is None:
        return weighted_quantile(s, w / w.sum(), 1 - alpha)

    total = w.sum() + float(test_weight)
    p = w / total
    order = np.argsort(s)
    v, pp = s[order], p[order]
    cum = np.cumsum(pp)
    target = 1 - alpha
    if cum[-1] < target:  # even all finite mass can't reach target; test atom is at +inf
        return float("inf")
    idx = int(np.searchsorted(cum, target, side="left"))
    return float(v[min(idx, v.size - 1)])


# Risk-coverage / selective prediction


def risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coverage, selective_risk)`` arrays.

    Samples are accepted most-confident-first; at each coverage level the
    selective risk is the error rate among accepted samples. ``confidence`` is
    higher-is-more-confident; ``correct`` is a boolean array of per-sample
    correctness.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.size != correct.size or confidence.size == 0:
        raise ValueError("confidence and correct must be non-empty and aligned")
    order = np.argsort(-confidence)  # descending confidence
    c = correct[order]
    k = np.arange(1, c.size + 1)
    coverage = k / c.size
    selective_risk = 1.0 - np.cumsum(c) / k
    return coverage, selective_risk


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean selective risk over coverage).
    Lower is better; a perfect confidence ordering pushes errors to the end."""
    _, risk = risk_coverage_curve(confidence, correct)
    return float(np.mean(risk))


def excess_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """AURC minus the AURC of the oracle ordering (all correct accepted first).
    Isolates ranking quality from the model's base error rate."""
    correct = np.asarray(correct, dtype=bool)
    _, risk = risk_coverage_curve(confidence, correct)
    n = correct.size
    n_correct = int(correct.sum())
    k = np.arange(1, n + 1)
    oracle_risk = np.maximum(0.0, (k - n_correct) / k)  # errors only once correct run out
    return float(np.mean(risk) - np.mean(oracle_risk))


def selective_risk_at_coverage(
    confidence: np.ndarray, correct: np.ndarray, coverage: float
) -> float:
    """Selective risk when accepting the top ``coverage`` fraction by confidence."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    cov, risk = risk_coverage_curve(confidence, correct)
    idx = int(np.searchsorted(cov, coverage, side="left"))
    return float(risk[min(idx, risk.size - 1)])


def coverage_at_risk(
    confidence: np.ndarray, correct: np.ndarray, target_risk: float
) -> float:
    """Largest coverage achievable with selective risk <= ``target_risk``
    (0.0 if even accepting the single most-confident sample exceeds it)."""
    cov, risk = risk_coverage_curve(confidence, correct)
    ok = risk <= target_risk
    return float(cov[ok].max()) if ok.any() else 0.0


# Operating-point selection on a binary detection score


def _sweep(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(thresholds, tpr, fpr, precision)`` over candidate thresholds.
    A sample is flagged positive when ``score >= threshold``."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.size != s.size or y.size == 0:
        raise ValueError("y_true and scores must be non-empty and aligned")
    P, N = int(y.sum()), int((~y).sum())
    thr = np.unique(s)
    thr = np.concatenate([thr, [thr[-1] + 1.0]])  # an all-negative operating point
    tpr = np.empty(thr.size)
    fpr = np.empty(thr.size)
    prec = np.empty(thr.size)
    for i, t in enumerate(thr):
        pred = s >= t
        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))
        tpr[i] = tp / P if P else 0.0
        fpr[i] = fp / N if N else 0.0
        prec[i] = tp / (tp + fp) if (tp + fp) else 1.0
    return thr, tpr, fpr, prec


def youden_j_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal-cost, prevalence-independent operating point: argmax of TPR - FPR.
    Provided for comparison; prefer a cost- or constraint-based point when the
    error costs are asymmetric or OOD prevalence is low/shifting."""
    thr, tpr, fpr, _ = _sweep(y_true, scores)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(j[i])


def cost_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    c_fp: float,
    c_fn: float,
    prevalence: float | None = None,
) -> tuple[float, float]:
    r"""Threshold minimising expected per-sample cost

        E[cost] = c_fp * FPR * pi_neg + c_fn * FNR * pi_pos.

    ``prevalence`` (pi_pos) defaults to the empirical positive rate; pass a
    value to reflect an expected production prevalence under shift. Youden's J
    is the special case ``c_fp * pi_neg == c_fn * pi_pos``. Returns
    ``(threshold, expected_cost)``.
    """
    thr, tpr, fpr, _ = _sweep(y_true, scores)
    pi_pos = float(np.mean(np.asarray(y_true, dtype=bool))) if prevalence is None else prevalence
    pi_neg = 1.0 - pi_pos
    cost = c_fp * fpr * pi_neg + c_fn * (1.0 - tpr) * pi_pos
    i = int(np.argmin(cost))
    return float(thr[i]), float(cost[i])


def constraint_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_fpr: float | None = None,
    min_precision: float | None = None,
) -> tuple[float, dict]:
    """Neyman-Pearson style: maximise TPR subject to ``FPR <= max_fpr`` and/or
    ``precision >= min_precision``. Returns ``(threshold, metrics)``; raises if
    no threshold satisfies the constraints."""
    thr, tpr, fpr, prec = _sweep(y_true, scores)
    feasible = np.ones(thr.size, dtype=bool)
    if max_fpr is not None:
        feasible &= fpr <= max_fpr
    if min_precision is not None:
        feasible &= prec >= min_precision
    if not feasible.any():
        raise ValueError("no threshold satisfies the given constraint(s)")
    cand = np.where(feasible)[0]
    i = int(cand[np.argmax(tpr[cand])])
    return float(thr[i]), {"tpr": float(tpr[i]), "fpr": float(fpr[i]), "precision": float(prec[i])}


# Bootstrap confidence intervals on a threshold


def bootstrap_threshold_ci(
    estimator: Callable[[np.ndarray], float],
    data: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a threshold estimator.

    ``estimator`` maps a 1-D sample to a scalar threshold. Returns
    ``(point_estimate, ci_low, ci_high)``. Infinite bootstrap estimates (e.g.
    a conformal threshold that could not be supported on a resample) are
    dropped before taking percentiles.
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    rng = np.random.default_rng(seed)
    ests = np.empty(n_boot)
    for b in range(n_boot):
        ests[b] = estimator(data[rng.integers(0, n, n)])
    finite = ests[np.isfinite(ests)]
    if finite.size == 0:
        return float(estimator(data)), float("nan"), float("nan")
    lo = float(np.percentile(finite, (1 - ci) / 2 * 100))
    hi = float(np.percentile(finite, (1 + ci) / 2 * 100))
    return float(estimator(data)), lo, hi


def conformal_threshold_ci(
    cal_scores: np.ndarray,
    alpha: float,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for :func:`conformal_threshold`: sampling uncertainty of the
    deployed cut, used by the auto-QA layer to gate threshold changes."""
    return bootstrap_threshold_ci(
        lambda d: conformal_threshold(d, alpha), cal_scores, n_boot, ci, seed
    )
