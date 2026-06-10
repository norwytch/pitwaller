"""Feature extraction.

The OOD machinery is agnostic to where features come from: it operates on
whatever embedding you feed it. By default that's a model's own penultimate
features, but for detecting genuinely novel *content* a broad foundation
embedding is usually a better substrate (see "Choosing the embedding" in the
README). Either way, we hide the feature extractor behind a small ``Embedder``
protocol. The pipeline, the index, the OOD model and the tiering logic all
depend only on that protocol -- never on PyTorch directly. That keeps the core
importable and testable without heavyweight ML dependencies, and lets you swap
in any feature extractor you like.

Three concrete embedders are provided:

* ``MockEmbedder``  -- deterministic synthetic features, used by the demo and
  the test suite so the whole pipeline runs with zero external weights/data.
* ``EffNetB4Embedder`` -- 1792-dim features from the global-pooled penultimate
  layer of an EfficientNet-B4 (the model's *own* feature space). Lazily imports
  ``torch``/``timm``.
* ``CLIPEmbedder`` -- CLIP vision-encoder features (broad semantic
  representation, stronger for novel-content detection). Lazily imports
  ``open_clip``/``torch``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a batch of inputs into an ``(N, D)`` float array."""

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding space."""
        ...

    def embed(self, batch) -> np.ndarray:
        """Return L2-normalisable float32 features of shape ``(len(batch), dim)``."""
        ...


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation. OOD distances are computed on the unit sphere
    so that cosine geometry (which is what EffNet features behave well under)
    maps cleanly onto the L2 metric FAISS indexes use."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


class MockEmbedder:
    """Deterministic synthetic embedder for demos and tests.

    Produces clustered Gaussian features so that "in-distribution" and
    "out-of-distribution" inputs are actually separable -- enough to exercise
    every branch of the OOD + tiering logic without a trained network.
    """

    def __init__(self, dim: int = 64, n_clusters: int = 8, seed: int = 0):
        self._dim = dim
        rng = np.random.default_rng(seed)
        # Fixed cluster centres define the "training manifold".
        self._centers = rng.normal(scale=3.0, size=(n_clusters, dim)).astype(np.float32)
        self._n_clusters = n_clusters

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, batch) -> np.ndarray:
        """``batch`` is an iterable of ``(cluster_id, jitter, seed)`` tuples.

        ``cluster_id`` outside ``[0, n_clusters)`` yields off-manifold points,
        i.e. synthetic OOD samples.
        """
        out = np.empty((len(batch), self._dim), dtype=np.float32)
        for i, item in enumerate(batch):
            cluster_id, jitter, seed = item
            rng = np.random.default_rng(seed)
            if 0 <= cluster_id < self._n_clusters:
                base = self._centers[cluster_id]
            else:
                # Off-manifold: random direction far from any cluster centre.
                base = rng.normal(scale=10.0, size=self._dim).astype(np.float32)
            out[i] = base + rng.normal(scale=jitter, size=self._dim).astype(np.float32)
        return l2_normalize(out)


class EffNetB4Embedder:
    """Real penultimate-layer features from EfficientNet-B4 (1792-dim).

    Requires ``torch`` and ``timm`` (``pip install 'pitwaller[torch]'``). Kept out
    of the import path of everything else so the core stays dependency-light.
    """

    def __init__(self, device: str = "cpu", pretrained: bool = True):
        try:
            import timm
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only with extras
            raise ImportError(
                "EffNetB4Embedder needs torch and timm. Install with "
                "`pip install 'pitwaller[torch]'`."
            ) from exc

        self._torch = torch
        self.device = device
        # num_classes=0 + global_pool='avg' => model returns the 1792-d
        # global-pooled feature vector, which is exactly the OOD feature space.
        self.model = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=0, global_pool="avg"
        ).eval().to(device)
        self._dim = self.model.num_features  # 1792 for B4

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, batch) -> np.ndarray:
        """``batch`` is a float tensor / array of shape ``(N, 3, H, W)``."""
        torch = self._torch
        x = torch.as_tensor(np.asarray(batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feats = self.model(x).cpu().numpy()
        return l2_normalize(feats)


class CLIPEmbedder:
    """Image features from a CLIP vision encoder (via ``open_clip``).

    Unlike :class:`EffNetB4Embedder`, whose features are tuned to one model's
    training labels and collapse whatever was irrelevant to that task, CLIP's
    encoder is trained on broad image-text data and preserves wide *semantic*
    content. That makes it a far stronger substrate for detecting genuinely
    novel content (near-OOD / open-set), which is what the OOD stack keys on.
    See "Choosing the embedding" in the README for the tradeoff.

    Requires ``open_clip`` and ``torch`` (``pip install 'pitwaller[clip]'``).
    Imported lazily so the core stays dependency-light.

    ``embed`` takes already-preprocessed tensors of shape ``(N, 3, H, W)``. CLIP
    is sensitive to its specific preprocessing, so the canonical transform is
    exposed as ``self.preprocess`` -- build a batch with, e.g.::

        import torch
        batch = torch.stack([emb.preprocess(img) for img in pil_images])
        feats = emb.embed(batch)
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ):
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only with extras
            raise ImportError(
                "CLIPEmbedder needs open_clip and torch. Install with "
                "`pip install 'pitwaller[clip]'`."
            ) from exc

        self._torch = torch
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.eval().to(device)
        self.preprocess = preprocess  # canonical CLIP image transform (PIL -> tensor)

        # Resolve the embedding dim robustly with a single dummy forward.
        size = getattr(self.model.visual, "image_size", 224)
        h, w = (size[-2], size[-1]) if isinstance(size, (tuple, list)) else (size, size)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, h, w, device=device)
            self._dim = int(self.model.encode_image(dummy).shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, batch) -> np.ndarray:
        """``batch`` is a float tensor / array of shape ``(N, 3, H, W)``,
        preprocessed with :attr:`preprocess`."""
        torch = self._torch
        x = torch.as_tensor(np.asarray(batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feats = self.model.encode_image(x).cpu().numpy()
        return l2_normalize(feats)  # CLIP features are used on the unit sphere


class SentenceTransformerEmbedder:
    """Text features from a Sentence-Transformers model (e.g. ``all-MiniLM-L6-v2``).

    Turns raw strings into dense semantic vectors -- the substrate the OOD stack
    scores novelty in for text. Like :class:`CLIPEmbedder` this is a broad
    foundation embedding (strong for detecting novel *content*), not one model's
    task features; see "Choosing the embedding" in the README.

    Requires ``sentence-transformers`` (``pip install 'pitwaller[text]'``), which
    pulls in torch. Imported lazily so the core stays dependency-light.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only with extras
            raise ImportError(
                "SentenceTransformerEmbedder needs sentence-transformers. Install with "
                "`pip install 'pitwaller[text]'`."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        self._dim = int(self.model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, batch) -> np.ndarray:
        """``batch`` is an iterable of strings."""
        feats = self.model.encode(
            list(batch), convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        return l2_normalize(feats)  # used on the unit sphere, like CLIP
