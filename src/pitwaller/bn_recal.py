"""BatchNorm recalibration -- justify it, do it, prove it worked.

The ``BN_RECALIBRATION`` action exists in the policy, but firing it blindly is a
mistake: recalibrating BatchNorm running statistics helps under *covariate
shift* and can quietly hurt otherwise. This module wraps the action in the rigor
it deserves:

1. **Justify** -- measure how far the input statistics have actually moved. For
   each BatchNorm layer we compare the stored running ``(mean, var)`` against the
   statistics of a fresh batch using the closed-form 2-Wasserstein distance
   between the per-channel Gaussians (and symmetric KL as a cross-check).
   Recalibrate only when the shift is real and large.

2. **Recalibrate** -- re-estimate running statistics by forward-passing fresh,
   *unlabelled* inputs with BN in cumulative-moving-average mode (AdaBN; Li et
   al., 2017). No gradients, no labels.

3. **Validate** -- confirm the change is a real improvement, not noise, with
   McNemar's paired test on before/after correctness over a labelled validation
   set.

The statistics (steps 1 and 3) are pure NumPy / stdlib and fully tested. The two
functions that actually touch a network -- :func:`collect_bn_stats` and
:func:`recalibrate_bn` -- are the integration points; they lazily import
``torch`` so the rest of the module works without it. They are real, working
implementations, not mocks: drop in your model and a fresh-input iterable and
they run. They are marked WIRE-IN below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# 1. Justify: distance between stored and fresh BN statistics                  #
# --------------------------------------------------------------------------- #


def gaussian_2wasserstein(
    mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray
) -> float:
    r"""Closed-form squared 2-Wasserstein between two diagonal Gaussians.

    BatchNorm tracks per-channel mean and variance, i.e. a diagonal covariance,
    for which the Bures term collapses and

        W2^2 = ||mu1 - mu2||^2 + ||sqrt(var1) - sqrt(var2)||^2.

    Returns ``W2^2`` summed over channels (0 iff the statistics match).
    """
    mu1, var1, mu2, var2 = (np.asarray(a, dtype=float) for a in (mu1, var1, mu2, var2))
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    std_term = float(np.sum((np.sqrt(np.maximum(var1, 0)) - np.sqrt(np.maximum(var2, 0))) ** 2))
    return mean_term + std_term


def symmetric_kl_gaussian(
    mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray, eps: float = 1e-8
) -> float:
    """Symmetric KL between two diagonal Gaussians, summed over channels.

    A scale-sensitive cross-check on the 2-Wasserstein distance; unlike W2 it
    blows up when a channel's variance collapses, which is often exactly the
    pathology you want flagged.
    """
    mu1, var1, mu2, var2 = (np.asarray(a, dtype=float) for a in (mu1, var1, mu2, var2))
    v1 = np.maximum(var1, eps)
    v2 = np.maximum(var2, eps)
    kl_12 = 0.5 * (np.log(v2 / v1) + (v1 + (mu1 - mu2) ** 2) / v2 - 1.0)
    kl_21 = 0.5 * (np.log(v1 / v2) + (v2 + (mu2 - mu1) ** 2) / v1 - 1.0)
    return float(np.sum(kl_12 + kl_21))


def feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel ``(mean, var)`` of a fresh batch.

    ``x`` is ``(N, C)`` (already pooled) or ``(N, C, ...)``; statistics are taken
    over every axis except the channel axis, matching BatchNorm's convention.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim < 2:
        raise ValueError("expected at least (N, C)")
    axes = (0,) + tuple(range(2, x.ndim))
    return x.mean(axis=axes), x.var(axis=axes)


@dataclass
class LayerShift:
    name: str
    w2: float
    sym_kl: float


@dataclass
class BNShiftReport:
    layers: list[LayerShift]

    @property
    def total_w2(self) -> float:
        return sum(layer.w2 for layer in self.layers)

    @property
    def max_w2(self) -> float:
        return max((layer.w2 for layer in self.layers), default=0.0)

    @property
    def mean_w2(self) -> float:
        return self.total_w2 / len(self.layers) if self.layers else 0.0

    def worst(self, n: int = 3) -> list[LayerShift]:
        return sorted(self.layers, key=lambda layer: layer.w2, reverse=True)[:n]


def bn_shift_report(
    stored: dict[str, tuple[np.ndarray, np.ndarray]],
    fresh: dict[str, tuple[np.ndarray, np.ndarray]],
) -> BNShiftReport:
    """Per-layer shift between stored and fresh BN statistics.

    ``stored`` / ``fresh`` map a layer name to its ``(mean, var)`` arrays; see
    :func:`collect_bn_stats` and :func:`feature_stats` for producing them.
    """
    layers = []
    for name, (mu_s, var_s) in stored.items():
        if name not in fresh:
            continue
        mu_f, var_f = fresh[name]
        layers.append(
            LayerShift(
                name=name,
                w2=gaussian_2wasserstein(mu_s, var_s, mu_f, var_f),
                sym_kl=symmetric_kl_gaussian(mu_s, var_s, mu_f, var_f),
            )
        )
    return BNShiftReport(layers=layers)


def should_recalibrate(report: BNShiftReport, w2_threshold: float) -> bool:
    """Recommend recalibration when any layer's shift exceeds ``w2_threshold``.

    Threshold is data-dependent; calibrate it from the layer-wise W2 you observe
    across known-stable validation windows (e.g. its 99th percentile) so this
    fires on genuine shift, not normal batch-to-batch wobble.
    """
    return report.max_w2 > w2_threshold


# --------------------------------------------------------------------------- #
# 2. Recalibrate: AdaBN over fresh unlabelled inputs   [WIRE-IN: needs torch]  #
# --------------------------------------------------------------------------- #


def collect_bn_stats(model) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """WIRE-IN. Pull stored running ``(mean, var)`` from every BatchNorm layer.

    Pair with :func:`feature_stats` on a fresh batch to build the two arguments
    to :func:`bn_shift_report`.
    """
    try:
        from torch.nn.modules.batchnorm import _BatchNorm
    except ImportError as exc:  # pragma: no cover - exercised only with extras
        raise ImportError("collect_bn_stats requires torch ('pitwaller[torch]').") from exc

    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, module in model.named_modules():
        if isinstance(module, _BatchNorm) and module.running_mean is not None:
            stats[name] = (
                module.running_mean.detach().cpu().numpy(),
                module.running_var.detach().cpu().numpy(),
            )
    return stats


def recalibrate_bn(model, fresh_batches, reset: bool = True):
    """WIRE-IN. Re-estimate BatchNorm running statistics on fresh inputs (AdaBN).

    ``fresh_batches`` is any iterable of model-ready input tensors drawn from the
    *current* production distribution -- no labels required. With ``reset=True``
    the running statistics are rebuilt from scratch using a cumulative moving
    average over the stream (BN ``momentum=None``); set ``reset=False`` to nudge
    the existing stats instead. Affine weights and all other parameters are
    untouched. Returns the (in-place modified) model.

    This is the action behind ``Action.BN_RECALIBRATION``. Validate the result
    with :func:`validate_recalibration` before promoting it to production.
    """
    try:
        import torch
        from torch.nn.modules.batchnorm import _BatchNorm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("recalibrate_bn requires torch ('pitwaller[torch]').") from exc

    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not bns:
        return model
    saved_momentum = [m.momentum for m in bns]
    saved_mode = model.training

    model.eval()  # freeze dropout etc.
    for m in bns:
        if reset:
            m.reset_running_stats()
        m.momentum = None  # cumulative moving average
        m.train()  # but let BN update its running stats
    try:
        with torch.no_grad():
            for batch in fresh_batches:
                model(batch)
    finally:
        for m, mom in zip(bns, saved_momentum):
            m.momentum = mom
            m.eval()
        model.train(saved_mode)
    return model


# --------------------------------------------------------------------------- #
# 3. Validate: McNemar's paired test on before/after correctness               #
# --------------------------------------------------------------------------- #


@dataclass
class BNRecalOutcome:
    """Result of comparing a model before vs after recalibration."""

    acc_before: float
    acc_after: float
    fixed: int           # were wrong, now right (c)
    broken: int          # were right, now wrong (b)
    statistic: float
    p_value: float
    method: str          # "exact" or "chi2"

    @property
    def delta_accuracy(self) -> float:
        return self.acc_after - self.acc_before

    def significant_improvement(self, alpha: float = 0.05) -> bool:
        """Net positive *and* statistically significant. A bare accuracy bump
        that doesn't clear the test is treated as noise -- don't promote it."""
        return self.fixed > self.broken and self.p_value < alpha


def _chi2_1dof_sf(x: float) -> float:
    """Survival function of chi-square with 1 dof: P(X > x) = erfc(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: discordants ~ Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def validate_recalibration(
    correct_before: np.ndarray, correct_after: np.ndarray, exact: bool | None = None
) -> BNRecalOutcome:
    """McNemar's test on paired per-sample correctness over a labelled val set.

    ``correct_before`` / ``correct_after`` are aligned boolean arrays. Only the
    discordant pairs carry information: ``broken`` (was right, now wrong) and
    ``fixed`` (was wrong, now right). Uses the exact binomial test for small
    discordant counts and the continuity-corrected chi-square otherwise; pass
    ``exact`` to force one.
    """
    before = np.asarray(correct_before, dtype=bool)
    after = np.asarray(correct_after, dtype=bool)
    if before.shape != after.shape or before.size == 0:
        raise ValueError("correct_before and correct_after must be non-empty and aligned")

    broken = int(np.sum(before & ~after))   # b
    fixed = int(np.sum(~before & after))    # c
    n_disc = broken + fixed

    use_exact = (n_disc < 25) if exact is None else exact
    if use_exact:
        p = _mcnemar_exact_p(broken, fixed)
        stat = float(min(broken, fixed))
        method = "exact"
    else:
        stat = (abs(broken - fixed) - 1) ** 2 / n_disc  # continuity correction
        p = _chi2_1dof_sf(stat)
        method = "chi2"

    return BNRecalOutcome(
        acc_before=float(before.mean()),
        acc_after=float(after.mean()),
        fixed=fixed,
        broken=broken,
        statistic=float(stat),
        p_value=float(p),
        method=method,
    )
