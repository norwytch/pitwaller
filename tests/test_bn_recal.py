import numpy as np
import pytest

from pitwaller.bn_recal import (
    bn_shift_report,
    feature_stats,
    gaussian_2wasserstein,
    should_recalibrate,
    symmetric_kl_gaussian,
    validate_recalibration,
)


# ----------------------------------------------------------- 2-Wasserstein / KL
def test_w2_zero_for_identical_gaussians():
    mu = np.array([0.0, 1.0, 2.0])
    var = np.array([1.0, 2.0, 0.5])
    assert gaussian_2wasserstein(mu, var, mu, var) == pytest.approx(0.0)


def test_w2_known_value():
    # One channel: means 0 vs 3, vars 1 vs 1 -> (3-0)^2 + (1-1)^2 = 9.
    assert gaussian_2wasserstein([0.0], [1.0], [3.0], [1.0]) == pytest.approx(9.0)
    # means equal, std 1 vs 2 -> 0 + (1-2)^2 = 1.
    assert gaussian_2wasserstein([0.0], [1.0], [0.0], [4.0]) == pytest.approx(1.0)


def test_w2_increases_with_separation():
    base = gaussian_2wasserstein([0.0], [1.0], [1.0], [1.0])
    far = gaussian_2wasserstein([0.0], [1.0], [5.0], [1.0])
    assert far > base


def test_symmetric_kl_zero_for_identical():
    mu = np.array([0.0, 1.0])
    var = np.array([1.0, 2.0])
    assert symmetric_kl_gaussian(mu, var, mu, var) == pytest.approx(0.0, abs=1e-9)


def test_symmetric_kl_is_symmetric():
    a = symmetric_kl_gaussian([0.0], [1.0], [2.0], [3.0])
    b = symmetric_kl_gaussian([2.0], [3.0], [0.0], [1.0])
    assert a == pytest.approx(b)


# ------------------------------------------------------------------ feature_stats
def test_feature_stats_matches_numpy():
    x = np.random.default_rng(0).normal(size=(100, 4))
    mu, var = feature_stats(x)
    assert np.allclose(mu, x.mean(0))
    assert np.allclose(var, x.var(0))


def test_feature_stats_conv_reduces_spatial_axes():
    x = np.random.default_rng(0).normal(size=(8, 3, 5, 5))  # (N, C, H, W)
    mu, var = feature_stats(x)
    assert mu.shape == (3,) and var.shape == (3,)


# -------------------------------------------------------------------- shift report
def test_shift_report_flags_shifted_layer():
    stored = {"bn1": (np.zeros(4), np.ones(4)), "bn2": (np.zeros(4), np.ones(4))}
    fresh = {"bn1": (np.zeros(4), np.ones(4)),          # unchanged
             "bn2": (np.full(4, 3.0), np.ones(4))}      # shifted mean
    report = bn_shift_report(stored, fresh)
    assert report.max_w2 > 0
    assert report.worst(1)[0].name == "bn2"
    assert should_recalibrate(report, w2_threshold=1.0) is True
    assert should_recalibrate(report, w2_threshold=1e6) is False


def test_shift_report_ignores_missing_layers():
    stored = {"a": (np.zeros(2), np.ones(2)), "b": (np.zeros(2), np.ones(2))}
    fresh = {"a": (np.zeros(2), np.ones(2))}  # 'b' absent
    report = bn_shift_report(stored, fresh)
    assert len(report.layers) == 1


# ---------------------------------------------------------------------- McNemar
def test_validate_no_change_is_not_significant():
    correct = np.array([True, True, False, True, False] * 20)
    out = validate_recalibration(correct, correct)
    assert out.fixed == 0 and out.broken == 0
    assert out.significant_improvement() is False
    assert out.delta_accuracy == pytest.approx(0.0)


def test_validate_strong_improvement_is_significant():
    # 30 fixed, 1 broken -> clear, significant net improvement.
    before = np.array([False] * 30 + [True] + [True] * 50)
    after = np.array([True] * 30 + [False] + [True] * 50)
    out = validate_recalibration(before, after)
    assert out.fixed == 30 and out.broken == 1
    assert out.delta_accuracy > 0
    assert out.significant_improvement() is True


def test_validate_symmetric_discordance_not_significant():
    # Equal fixed/broken -> no evidence of improvement even if churny.
    before = np.array([True] * 10 + [False] * 10 + [True] * 30)
    after = np.array([False] * 10 + [True] * 10 + [True] * 30)
    out = validate_recalibration(before, after)
    assert out.fixed == out.broken == 10
    assert out.significant_improvement() is False


def test_validate_picks_exact_for_small_and_chi2_for_large():
    small_b = np.array([True, False, True, False, True])
    small_a = np.array([True, True, True, True, True])  # 2 discordant
    assert validate_recalibration(small_b, small_a).method == "exact"

    rng = np.random.default_rng(1)
    before = rng.random(2000) < 0.5
    after = before.copy()
    flip = rng.random(2000) < 0.1  # many discordant pairs
    after[flip] = ~after[flip]
    assert validate_recalibration(before, after).method == "chi2"


def test_validate_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        validate_recalibration(np.array([True, False]), np.array([True]))
