"""Out-of-distribution scoring in the model's own feature space.

Two *independent* detectors are fit on the training embeddings:

1. **kNN distance.** For a query we take the mean (squared-L2) distance to its
   ``k`` nearest training neighbours -- a non-parametric local-density estimate.
   Calibrating this against the *training set's own* kNN-distance distribution
   gives two thresholds:

   * ``p50`` -- median training density. Inside it is the dense "core".
   * ``p90`` -- the 90th percentile. Beyond it the point is sparser than 90 %
     of training data and is treated as a distance outlier.

2. **Isolation Forest.** A global structural anomaly detector fit on the same
   embeddings. It captures off-manifold points that kNN distance can miss
   (e.g. a point wedged between clusters).

Keeping them independent is deliberate: they fail in different ways, and the
confidence tiering downstream combines their *agreement* into HIGH/MED/LOW.

The design rests on an empirical property: OOD distance and model accuracy tend
to be monotonically related -- nearer the core, more accurate. That monotonicity
is what licenses using distance bands as a confidence proxy in the first place;
``OODModel`` is the component that measures it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from .index import HNSWConfig, VectorIndex


@dataclass
class OODResult:
    """Per-sample OOD readout.

    ``band`` is one of ``"core"`` (<= p50), ``"margin"`` (p50-p90),
    ``"outlier"`` (> p90). ``if_score`` is the Isolation Forest's *continuous*
    anomaly score (``decision_function``: higher = more in-distribution),
    retained alongside the ``if_outlier`` boolean so a calibrated tiering layer
    can fuse the raw signal rather than a thresholded flag (see
    :mod:`pitwaller.tier_calibration`).
    """

    knn_distance: float
    band: str
    if_outlier: bool
    if_score: float = 0.0

    @property
    def dist_concern(self) -> bool:
        """True once the sample leaves the dense core (kNN distance > p50)."""
        return self.band != "core"


class OODModel:
    """Fit on training embeddings; scores new samples.

    Parameters
    ----------
    k:
        Neighbours used for the distance estimate.
    contamination:
        Isolation Forest contamination prior (expected anomaly fraction).
    index_config / index_backend:
        Forwarded to :class:`~pitwaller.index.VectorIndex`.
    """

    def __init__(
        self,
        k: int = 10,
        contamination: float = 0.05,
        index_config: HNSWConfig | None = None,
        index_backend: str = "auto",
        random_state: int = 0,
    ):
        self.k = k
        self.contamination = contamination
        self.index_config = index_config
        self.index_backend = index_backend
        self.random_state = random_state

        self.index: VectorIndex | None = None
        self.iforest: IsolationForest | None = None
        self.p50: float | None = None
        self.p90: float | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray) -> "OODModel":
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        n, dim = X_train.shape

        self.index = VectorIndex(dim, self.index_config, self.index_backend).build(X_train)

        # Self-distance: query k+1 and drop the point itself (the nearest
        # neighbour of a training point in its own index is itself, distance 0).
        dist, _ = self.index.search(X_train, self.k + 1)
        train_scores = dist[:, 1:].mean(axis=1)
        self.p50 = float(np.percentile(train_scores, 50))
        self.p90 = float(np.percentile(train_scores, 90))

        self.iforest = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        ).fit(X_train)
        return self

    # ---------------------------------------------------------------- score
    def _knn_distance(self, X: np.ndarray) -> np.ndarray:
        dist, _ = self.index.search(X, self.k)
        return dist.mean(axis=1)

    def _band(self, score: float) -> str:
        if score <= self.p50:
            return "core"
        if score <= self.p90:
            return "margin"
        return "outlier"

    def score(self, X: np.ndarray) -> list[OODResult]:
        if self.index is None or self.iforest is None:
            raise RuntimeError("OODModel is not fit; call fit() first")
        X = np.ascontiguousarray(X, dtype=np.float32)
        knn = self._knn_distance(X)
        if_pred = self.iforest.predict(X)  # +1 inlier, -1 anomaly
        if_score = self.iforest.decision_function(X)  # higher = more in-distribution
        return [
            OODResult(
                knn_distance=float(knn[i]),
                band=self._band(float(knn[i])),
                if_outlier=bool(if_pred[i] == -1),
                if_score=float(if_score[i]),
            )
            for i in range(X.shape[0])
        ]

    def score_one(self, x: np.ndarray) -> OODResult:
        return self.score(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]
