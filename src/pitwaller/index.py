"""Approximate nearest-neighbour index over the training feature space.

We use a FAISS HNSW graph for kNN. HNSW is the right tool here: the reference
set is the *entire training embedding set* (often millions of vectors), queries
happen online in production, and we only need approximate neighbours to
estimate a local-density / distance score. HNSW gives logarithmic query time
with recall high enough that the OOD percentile thresholds are stable.

A pure-numpy brute-force backend is included so the package runs (and the tests
pass) even where FAISS is unavailable, and so small inputs don't pay graph
build cost. The two backends expose an identical interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import faiss

    _HAVE_FAISS = True
except ImportError:  # pragma: no cover
    _HAVE_FAISS = False


@dataclass
class HNSWConfig:
    """HNSW build/search parameters.

    ``M`` is the graph degree, ``ef_construction`` the build-time beam width,
    ``ef_search`` the query-time beam width (recall/latency knob). The defaults
    are sane for tens-of-thousands to millions of 64-2048-d vectors.
    """

    M: int = 32
    ef_construction: int = 200
    ef_search: int = 64


class VectorIndex:
    """kNN index over an ``(N, D)`` reference matrix.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    config:
        HNSW parameters (ignored by the brute-force backend).
    backend:
        ``"auto"`` (FAISS if available, else brute force), ``"faiss"`` or
        ``"brute"``.
    """

    def __init__(self, dim: int, config: HNSWConfig | None = None, backend: str = "auto"):
        self.dim = dim
        self.config = config or HNSWConfig()
        if backend == "auto":
            backend = "faiss" if _HAVE_FAISS else "brute"
        if backend == "faiss" and not _HAVE_FAISS:
            raise RuntimeError("backend='faiss' requested but faiss is not installed")
        self.backend = backend
        self._index = None
        self._data: np.ndarray | None = None  # retained for brute-force backend

    def build(self, X: np.ndarray) -> "VectorIndex":
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {X.shape[1]}")
        if self.backend == "faiss":
            index = faiss.IndexHNSWFlat(self.dim, self.config.M)
            index.hnsw.efConstruction = self.config.ef_construction
            index.hnsw.efSearch = self.config.ef_search
            index.add(X)
            self._index = index
        else:
            self._data = X
        return self

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances, indices)`` of the ``k`` nearest neighbours of
        each row of ``q``. Distances are squared L2 (FAISS convention)."""
        if self._index is None and self._data is None:
            raise RuntimeError("index not built; call build() first")
        q = np.ascontiguousarray(q, dtype=np.float32)
        if self.backend == "faiss":
            return self._index.search(q, k)
        # Brute force: squared L2 to match FAISS.
        d2 = (
            (q * q).sum(1)[:, None]
            - 2.0 * q @ self._data.T
            + (self._data * self._data).sum(1)[None, :]
        )
        idx = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]
        # argpartition is unordered; sort the k retained per row.
        rows = np.arange(q.shape[0])[:, None]
        order = np.argsort(d2[rows, idx], axis=1)
        idx = idx[rows, order]
        dist = d2[rows, idx]
        return dist.astype(np.float32), idx.astype(np.int64)

    @property
    def size(self) -> int:
        if self.backend == "faiss":
            return 0 if self._index is None else self._index.ntotal
        return 0 if self._data is None else self._data.shape[0]
