import pytest

from pitwaller.confidence import Tier, tier_for
from pitwaller.ood import OODResult


def make(band, if_outlier):
    # knn_distance is irrelevant to tiering once band is set.
    return OODResult(knn_distance=0.0, band=band, if_outlier=if_outlier)


@pytest.mark.parametrize(
    "band,if_outlier,strict,expected",
    [
        # core + clean -> HIGH
        ("core", False, True, Tier.HIGH),
        ("core", False, False, Tier.HIGH),
        # exactly one signal -> MED
        ("core", True, False, Tier.MED),       # IF only
        ("margin", False, False, Tier.MED),    # distance only
        ("margin", False, True, Tier.MED),
        # both signals -> LOW
        ("margin", True, False, Tier.LOW),
        ("margin", True, True, Tier.LOW),
        # >p90 outlier band: strict -> LOW, literal -> depends on IF
        ("outlier", False, True, Tier.LOW),    # strict default
        ("outlier", False, False, Tier.MED),   # literal original rule
        ("outlier", True, True, Tier.LOW),
        ("outlier", True, False, Tier.LOW),
    ],
)
def test_tier_truth_table(band, if_outlier, strict, expected):
    assert tier_for(make(band, if_outlier), strict_outlier=strict) is expected


def test_high_requires_both_clean():
    # If either signal trips, it cannot be HIGH.
    assert tier_for(make("core", True)) is not Tier.HIGH
    assert tier_for(make("margin", False)) is not Tier.HIGH
