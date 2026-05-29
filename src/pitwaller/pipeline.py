"""End-to-end orchestration.

``ConfidencePipeline`` wires the pieces into one object:

    raw inputs --embedder--> features --OODModel--> OODResult --tiering--> Tier

It fits the OOD reference model on the training set's embeddings and then scores
production inputs into ``(OODResult, Tier)`` pairs, which feed straight into
:mod:`pitwaller.monitoring`. The embedder is injected, so the same pipeline runs on
mock features in a test and on real EfficientNet-B4 features in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .confidence import Tier, tier_for
from .embeddings import Embedder
from .ood import HNSWConfig, OODModel, OODResult


@dataclass
class ScoredSample:
    ood: OODResult
    tier: Tier


class ConfidencePipeline:
    def __init__(
        self,
        embedder: Embedder,
        k: int = 10,
        contamination: float = 0.05,
        strict_outlier: bool = True,
        index_config: HNSWConfig | None = None,
        index_backend: str = "auto",
    ):
        self.embedder = embedder
        self.strict_outlier = strict_outlier
        self.ood = OODModel(
            k=k,
            contamination=contamination,
            index_config=index_config,
            index_backend=index_backend,
        )
        self._fitted = False

    def fit(self, train_inputs) -> "ConfidencePipeline":
        """Fit on raw training inputs (embedded internally) or pre-computed
        features. Pass ``np.ndarray`` of shape ``(N, dim)`` to skip embedding."""
        feats = self._to_features(train_inputs)
        self.ood.fit(feats)
        self._fitted = True
        return self

    def score(self, inputs) -> list[ScoredSample]:
        if not self._fitted:
            raise RuntimeError("pipeline not fitted; call fit() first")
        feats = self._to_features(inputs)
        results = self.ood.score(feats)
        return [ScoredSample(r, tier_for(r, self.strict_outlier)) for r in results]

    def _to_features(self, inputs) -> np.ndarray:
        if isinstance(inputs, np.ndarray) and inputs.ndim == 2:
            return inputs.astype(np.float32)
        return self.embedder.embed(inputs)
