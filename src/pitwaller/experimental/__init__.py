"""Illustrative / standalone components, off the validated core path.

The validated core of pitwaller is embedding-space OOD detection and confidence
tiering (``ood``, ``confidence``, ``tier_calibration``, ``pipeline``, backed by
``embeddings``/``index``/``monitoring``). These modules sit alongside it but are
*not* required by it, and are not validated end-to-end:

* ``decisions``   -- the remediation policy engine. A transparent, heuristic
  if-ladder mapping diagnostics to corrective actions. CNN-oriented (BatchNorm
  recal, backbone retrain), correlational, with no outcome feedback loop. Treat
  its output as a ranked suggestion for a human, not an autopilot.
* ``bn_recal``    -- BatchNorm recalibration (justify with a 2-Wasserstein shift
  test, AdaBN, validate with McNemar). The statistics are tested; the two
  functions that touch a live model are CNN-specific integration points.
* ``calibration`` -- a self-contained toolkit for picking/evaluating a single
  threshold (conformal bounds, risk-coverage/AURC, cost/constraint operating
  points, bootstrap CIs). Useful on its own; the core pipeline does not call it.

Import explicitly, e.g. ``from pitwaller.experimental import recommend``.
"""

from .bn_recal import (
    BNRecalOutcome,
    BNShiftReport,
    bn_shift_report,
    feature_stats,
    gaussian_2wasserstein,
    should_recalibrate,
    symmetric_kl_gaussian,
    validate_recalibration,
)
from .calibration import (
    aurc,
    bootstrap_threshold_ci,
    conformal_threshold,
    conformal_threshold_ci,
    constraint_threshold,
    cost_optimal_threshold,
    coverage_at_risk,
    excess_aurc,
    risk_coverage_curve,
    selective_risk_at_coverage,
    weighted_conformal_threshold,
    weighted_quantile,
    youden_j_threshold,
)
from .decisions import (
    Action,
    EffortProfile,
    EffortTier,
    PolicyThresholds,
    Recommendation,
    Severity,
    ThresholdDriftSignal,
    group_by_effort,
    heaviest_tier,
    recommend,
)

__all__ = [
    "Action",
    "EffortProfile",
    "EffortTier",
    "PolicyThresholds",
    "Recommendation",
    "Severity",
    "ThresholdDriftSignal",
    "group_by_effort",
    "heaviest_tier",
    "recommend",
    "BNRecalOutcome",
    "BNShiftReport",
    "bn_shift_report",
    "feature_stats",
    "gaussian_2wasserstein",
    "should_recalibrate",
    "symmetric_kl_gaussian",
    "validate_recalibration",
    "aurc",
    "bootstrap_threshold_ci",
    "conformal_threshold",
    "conformal_threshold_ci",
    "constraint_threshold",
    "cost_optimal_threshold",
    "coverage_at_risk",
    "excess_aurc",
    "risk_coverage_curve",
    "selective_risk_at_coverage",
    "weighted_conformal_threshold",
    "weighted_quantile",
    "youden_j_threshold",
]
