"""Tests that verify the statistical guarantees, not just structural facts.

These are the claims that could actually be *wrong*: conformal coverage,
risk-targeted selective risk, and the closed-form statistics. Where possible
they check a guarantee by simulation or against a hand-computed reference,
rather than asserting a shape on a rigged input.
"""

import numpy as np
import pytest

from pitwaller.experimental.bn_recal import (
    gaussian_2wasserstein,
    symmetric_kl_gaussian,
    validate_recalibration,
)
from pitwaller.experimental.calibration import (
    conformal_threshold,
    coverage_at_risk,
    weighted_conformal_threshold,
)
from pitwaller.tier_calibration import risk_targeted_threshold


# --------------------------------------------------------------- conformal coverage


def test_conformal_threshold_achieves_marginal_coverage():
    """Split-conformal promises P(s(X_new) <= tau) >= 1 - alpha for exchangeable
    data. Simulate it: the mean coverage should sit right at 1 - alpha."""
    alpha, n_cal, n_test, trials = 0.1, 200, 1000, 200
    rng = np.random.default_rng(0)
    coverages = []
    for _ in range(trials):
        cal = rng.normal(size=n_cal)
        test = rng.normal(size=n_test)  # exchangeable with cal
        tau = conformal_threshold(cal, alpha)
        coverages.append(float(np.mean(test <= tau)))
    mean_cov = float(np.mean(coverages))
    # E[coverage] = ceil((n+1)(1-alpha)) / (n+1) ~ 0.9005 here.
    assert mean_cov >= 1 - alpha - 0.01      # the guarantee direction
    assert abs(mean_cov - (1 - alpha)) < 0.02


def test_conformal_returns_inf_when_too_few_points():
    # 5 points cannot certify a 10% rejection rate: ceil(6*0.9)=6 > 5 -> flag nothing.
    assert conformal_threshold(np.random.default_rng(1).normal(size=5), 0.1) == float("inf")


def test_weighted_conformal_reduces_to_standard_under_uniform_weights():
    rng = np.random.default_rng(2)
    s = rng.normal(size=400)
    std = conformal_threshold(s, 0.1)
    wq = weighted_conformal_threshold(s, np.ones_like(s), 0.1)
    assert abs(std - wq) < 0.2  # both land at the ~90th percentile of s


# ----------------------------------------------------- risk-targeted selective risk


def _confidence_data(n, seed):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.0, 1.0, size=n)
    correct = rng.random(n) < conf  # higher confidence -> more often correct
    return conf, correct


def test_risk_targeted_threshold_respects_target_in_sample():
    conf, correct = _confidence_data(3000, 3)
    target = 0.10
    tau = risk_targeted_threshold(conf, correct, target)
    accepted = conf >= tau
    assert accepted.any()
    realised = 1.0 - correct[accepted].mean()
    assert realised <= target + 1e-9


def test_risk_targeted_finite_sample_cut_is_more_conservative():
    conf, correct = _confidence_data(2000, 4)
    tau_emp = risk_targeted_threshold(conf, correct, 0.10, delta=None)
    tau_delta = risk_targeted_threshold(conf, correct, 0.10, delta=0.05)
    # The guaranteed (delta) cut never accepts more than the empirical one.
    assert tau_delta >= tau_emp


def test_coverage_at_risk_agrees_with_risk_targeted_threshold():
    # Two independently-implemented routines for the same operating point.
    conf, correct = _confidence_data(2000, 6)
    target = 0.10
    cov = coverage_at_risk(conf, correct, target)
    tau = risk_targeted_threshold(conf, correct, target)
    cov_from_tau = float((conf >= tau).mean())
    assert abs(cov - cov_from_tau) < 0.02


# ------------------------------------------------- closed-form stats vs references


def test_mcnemar_exact_matches_hand_computed_p_value():
    # b = 1 (was right, now wrong), c = 8 (was wrong, now right); 9 discordant.
    # two-sided exact p = 2 * sum_{i=0..1} C(9,i) * 0.5^9 = 20/512 = 0.0390625.
    before = np.array([True] + [False] * 8)
    after = np.array([False] + [True] * 8)
    out = validate_recalibration(before, after)
    assert (out.broken, out.fixed, out.method) == (1, 8, "exact")
    assert out.p_value == pytest.approx(0.0390625, abs=1e-9)
    assert out.significant_improvement()  # net positive and p < 0.05


def test_gaussian_2wasserstein_known_value():
    # mean term (0-3)^2 + (0-0)^2 = 9 ; std term (1-1)^2 + (2-1)^2 = 1 ; total 10.
    w2 = gaussian_2wasserstein([0.0, 0.0], [1.0, 4.0], [3.0, 0.0], [1.0, 1.0])
    assert w2 == pytest.approx(10.0, abs=1e-9)


def test_symmetric_kl_zero_for_identical_gaussians():
    kl = symmetric_kl_gaussian([1.0, 2.0], [1.0, 3.0], [1.0, 2.0], [1.0, 3.0])
    assert kl == pytest.approx(0.0, abs=1e-9)
