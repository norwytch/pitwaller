"""pitwaller: embedding-space OOD detection and confidence tiering.

Top-level exports cover the core: the pipeline, OOD model, tiers, tier
calibration, retrieval, and monitoring. The heuristic remediation policy,
BatchNorm recalibration, and single-threshold statistics toolkit live under
``pitwaller.experimental`` (e.g. ``from pitwaller.experimental import recommend``).
"""

from .confidence import Tier, tier_all, tier_for
from .embeddings import Embedder, MockEmbedder, l2_normalize
from .index import HNSWConfig, VectorIndex
from .monitoring import Diagnostics, PredictionRecord, aggregate
from .ood import OODModel, OODResult
from .pipeline import ConfidencePipeline, ScoredSample
from .retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    evaluate_retrieval,
    reciprocal_rank_fusion,
)
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
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "evaluate_retrieval",
    "reciprocal_rank_fusion",
]
