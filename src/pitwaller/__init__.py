"""pitwaller -- embedding-space OOD confidence tiering and automated model QA.

Public API:

    from pitwaller import ConfidencePipeline, MockEmbedder
    from pitwaller import OODModel, Tier, tier_for
    from pitwaller import aggregate, recommend
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
from .confidence import Tier, tier_all, tier_for
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
from .embeddings import Embedder, MockEmbedder, l2_normalize
from .index import HNSWConfig, VectorIndex
from .monitoring import Diagnostics, PredictionRecord, aggregate
from .ood import OODModel, OODResult
from .pipeline import ConfidencePipeline, ScoredSample
from .tier_calibration import (
    ReliabilityModel,
    TierCalibration,
    TierCalibrator,
    ood_features,
    risk_targeted_threshold,
)

__version__ = "0.1.0"

__all__ = [
    "ConfidencePipeline",
    "ScoredSample",
    "Embedder",
    "MockEmbedder",
    "l2_normalize",
    "VectorIndex",
    "HNSWConfig",
    "OODModel",
    "OODResult",
    "Tier",
    "tier_for",
    "tier_all",
    "TierCalibrator",
    "TierCalibration",
    "ReliabilityModel",
    "ood_features",
    "risk_targeted_threshold",
    "Diagnostics",
    "PredictionRecord",
    "aggregate",
    "Action",
    "Severity",
    "Recommendation",
    "PolicyThresholds",
    "recommend",
    "EffortTier",
    "EffortProfile",
    "ThresholdDriftSignal",
    "group_by_effort",
    "heaviest_tier",
    "conformal_threshold",
    "conformal_threshold_ci",
    "weighted_conformal_threshold",
    "weighted_quantile",
    "risk_coverage_curve",
    "aurc",
    "excess_aurc",
    "selective_risk_at_coverage",
    "coverage_at_risk",
    "youden_j_threshold",
    "cost_optimal_threshold",
    "constraint_threshold",
    "bootstrap_threshold_ci",
    "gaussian_2wasserstein",
    "symmetric_kl_gaussian",
    "feature_stats",
    "bn_shift_report",
    "BNShiftReport",
    "should_recalibrate",
    "validate_recalibration",
    "BNRecalOutcome",
]
