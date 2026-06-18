"""Confidence tiering.

Folds two independent OOD signals into a single tier:

* ``dist_concern``: the kNN distance left the dense core (band != "core").
* ``if_outlier``: the Isolation Forest flagged the sample.

The rule:

* HIGH: in the core and not an IF outlier (both detectors agree in-distribution).
* MED: an IF outlier or in the 50-90 band, but not both (one detector concerned).
* LOW: an IF outlier and outside the core (both concerned).

The literal rule leaves points beyond p90 without an IF flag in MED, which is
too generous for the sparsest region. ``strict_outlier=True`` (the default)
maps the ``"outlier"`` band to LOW; set it False for the literal rule.

Thresholds live in :class:`~pitwaller.ood.OODModel`, not here.
"""

from __future__ import annotations

from enum import Enum

from .ood import OODResult


class Tier(str, Enum):
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"


def tier_for(result: OODResult, strict_outlier: bool = True) -> Tier:
    """Map a single :class:`~pitwaller.ood.OODResult` to a confidence tier."""
    dist_concern = result.dist_concern
    if_concern = result.if_outlier

    if strict_outlier and result.band == "outlier":
        return Tier.LOW
    if not dist_concern and not if_concern:
        return Tier.HIGH
    if dist_concern and if_concern:
        return Tier.LOW
    return Tier.MED


def tier_all(results: list[OODResult], strict_outlier: bool = True) -> list[Tier]:
    return [tier_for(r, strict_outlier) for r in results]
