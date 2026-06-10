"""Confidence tiering.

Two independent OOD signals are folded into a single operational tier:

* ``dist_concern`` -- the kNN distance left the dense core (band != "core").
* ``if_outlier``   -- the Isolation Forest flagged the sample.

The tiering rule:

* **HIGH** -- in the centre 50th percentile **and** not an IF outlier.
  (both detectors agree the sample is in-distribution)
* **MED**  -- an IF outlier **or** in the 50-90 band, but not both.
  (exactly one detector is concerned)
* **LOW**  -- an IF outlier **and** outside the core.
  (both detectors are concerned)

One honest gap in the literal rule: it never says what happens to points
beyond the 90th percentile when the Isolation Forest *doesn't* fire. Taken
literally those land in MED, which feels too generous for the sparsest region
of feature space. ``strict_outlier=True`` (the recommended default) treats the
``"outlier"`` band as automatically LOW. Set it False to reproduce the literal
rule exactly.

The mapping is pure and table-driven so it is trivially unit-testable and the
policy is auditable -- no hidden thresholds live here, they live in
:class:`~pitwaller.ood.OODModel`.
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
