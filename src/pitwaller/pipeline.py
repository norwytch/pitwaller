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

from typing import Callable

from .confidence import Tier, tier_for
from .embeddings import Embedder
from .ood import HNSWConfig, OODModel, OODResult
from .tier_calibration import TierCalibrator, ood_features


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
        self._calibrator: TierCalibrator | None = None

    def fit(self, train_inputs) -> "ConfidencePipeline":
        """Fit on raw training inputs (embedded internally) or pre-computed
        features. Pass ``np.ndarray`` of shape ``(N, dim)`` to skip embedding."""
        feats = self._to_features(train_inputs)
        self.ood.fit(feats)
        self._fitted = True
        return self

    def calibrate(
        self,
        cal_inputs,
        cal_correct,
        risk_high: float = 0.01,
        risk_med: float = 0.05,
        delta: float | None = None,
        feature_fn: Callable[[list[OODResult]], np.ndarray] = ood_features,
    ) -> "ConfidencePipeline":
        """Switch from p50/p90 tiering to label-calibrated, risk-targeted tiers.

        ``cal_inputs`` is a labelled calibration set (raw inputs or pre-computed
        features) and ``cal_correct`` the aligned per-sample correctness of the
        downstream classifier. Fits a reliability map and places the HIGH/MED/LOW
        cuts so HIGH keeps selective risk ``<= risk_high`` and MED ``<= risk_med``
        (with a finite-sample guarantee when ``delta`` is set). Until this is
        called, :meth:`score` uses the unsupervised p50/p90 tiering unchanged.
        See :mod:`pitwaller.tier_calibration`.
        """
        if not self._fitted:
            raise RuntimeError("fit the OOD reference before calibrating tiers")
        results = self.ood.score(self._to_features(cal_inputs))
        self._calibrator = TierCalibrator(
            risk_high=risk_high, risk_med=risk_med, delta=delta, feature_fn=feature_fn
        ).fit_results(results, cal_correct)
        return self

    @property
    def calibrated(self) -> bool:
        """True once :meth:`calibrate` has fitted label-based tier cuts."""
        return self._calibrator is not None

    @property
    def calibration(self):
        """The fitted :class:`~pitwaller.tier_calibration.TierCalibration`
        (reliability map + tier cuts), or ``None`` if still on p50/p90."""
        return self._calibrator.calibration_ if self._calibrator else None

    def score(self, inputs) -> list[ScoredSample]:
        if not self._fitted:
            raise RuntimeError("pipeline not fitted; call fit() first")
        feats = self._to_features(inputs)
        results = self.ood.score(feats)
        if self._calibrator is not None:
            tiers = self._calibrator.tier_results(results)
            return [ScoredSample(r, t) for r, t in zip(results, tiers)]
        return [ScoredSample(r, tier_for(r, self.strict_outlier)) for r in results]

    def _to_features(self, inputs) -> np.ndarray:
        if isinstance(inputs, np.ndarray) and inputs.ndim == 2:
            return inputs.astype(np.float32)
        return self.embedder.embed(inputs)
