import numpy as np
import pytest

from pitwaller.calibration import (
    aurc,
    bootstrap_threshold_ci,
    conformal_threshold,
    conformal_threshold_ci,
    constraint_threshold,
    cost_optimal_threshold,
    coverage_at_risk,
    excess_aurc,
    selective_risk_at_coverage,
    weighted_conformal_threshold,
    weighted_quantile,
    youden_j_threshold,
)


# ------------------------------------------------------------------ conformal
def test_conformal_threshold_controls_flag_rate():
    # Averaged over calibration draws, the flag rate on fresh inliers must not
    # exceed alpha (split conformal is marginally valid / slightly conservative).
    alpha = 0.1
    flags = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        cal = rng.normal(size=1000)
        q = conformal_threshold(cal, alpha)
        test = rng.normal(size=2000)
        flags.append(np.mean(test > q))
    mean_flag = float(np.mean(flags))
    assert mean_flag <= alpha + 0.01
    assert mean_flag >= alpha - 0.04  # not absurdly conservative


def test_conformal_threshold_inf_when_too_few_samples():
    # With n=5 you cannot guarantee alpha=0.01: ceil(6*0.99)=6 > 5.
    assert conformal_threshold(np.arange(5.0), alpha=0.01) == float("inf")


def test_conformal_threshold_validates_alpha():
    with pytest.raises(ValueError):
        conformal_threshold(np.arange(10.0), alpha=1.0)


# ------------------------------------------------------------ weighted quantile
def test_weighted_quantile_uniform_matches_order_statistic():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    assert weighted_quantile(v, w, 0.5) == 2.0
    assert weighted_quantile(v, w, 0.75) == 3.0


def test_weighted_quantile_mass_pulls_estimate():
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([10.0, 1.0, 1.0])  # mass on the smallest value
    assert weighted_quantile(v, w, 0.5) == 1.0


def test_weighted_conformal_lower_threshold_when_low_scores_upweighted():
    s = np.linspace(0, 1, 200)
    uniform = np.ones_like(s)
    low_heavy = np.linspace(2.0, 0.1, 200)  # weight decreasing with score
    q_uniform = weighted_conformal_threshold(s, uniform, alpha=0.1)
    q_low = weighted_conformal_threshold(s, low_heavy, alpha=0.1)
    assert q_low < q_uniform


def test_weighted_conformal_with_test_atom_runs():
    s = np.linspace(0, 1, 100)
    w = np.ones_like(s)
    q = weighted_conformal_threshold(s, w, alpha=0.1, test_weight=1.0)
    assert np.isfinite(q)


# ----------------------------------------------------------- risk-coverage/AURC
def _ordered(correct):
    # confidence that accepts the given correctness sequence in order.
    correct = np.asarray(correct, dtype=bool)
    conf = np.linspace(1.0, 0.0, correct.size)
    return conf, correct


def test_selective_risk_full_coverage_equals_overall_error():
    conf, correct = _ordered([True] * 8 + [False] * 2)
    assert selective_risk_at_coverage(conf, correct, 1.0) == pytest.approx(0.2)


def test_excess_aurc_zero_for_oracle_ordering():
    # All correct accepted before any error -> ranking is optimal.
    conf, correct = _ordered([True] * 8 + [False] * 2)
    assert excess_aurc(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_worst_ordering_has_higher_aurc_than_best():
    best_conf, correct = _ordered([True] * 7 + [False] * 3)
    worst_conf, correct_w = _ordered([False] * 3 + [True] * 7)
    assert aurc(worst_conf, correct_w) > aurc(best_conf, correct)


def test_coverage_at_risk_zero_target():
    conf, correct = _ordered([True] * 8 + [False] * 2)
    assert coverage_at_risk(conf, correct, target_risk=0.0) == pytest.approx(0.8)


# --------------------------------------------------------- operating points
def _detection_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3            # positives = the thing we flag
    scores = rng.normal(loc=y * 1.5)   # positives score higher
    return y, scores


def test_cost_asymmetry_moves_threshold():
    y, s = _detection_data()
    # Heavily penalising missed positives (high c_fn) should LOWER the cut
    # (flag more) vs heavily penalising false alarms (high c_fp).
    t_fn, _ = cost_optimal_threshold(y, s, c_fp=1.0, c_fn=10.0)
    t_fp, _ = cost_optimal_threshold(y, s, c_fp=10.0, c_fn=1.0)
    assert t_fn <= t_fp


def test_constraint_threshold_respects_max_fpr():
    y, s = _detection_data()
    _, metrics = constraint_threshold(y, s, max_fpr=0.1)
    assert metrics["fpr"] <= 0.1 + 1e-9


def test_constraint_threshold_infeasible_raises():
    y, s = _detection_data()
    with pytest.raises(ValueError):
        constraint_threshold(y, s, min_precision=2.0)  # impossible (> 1)


def test_youden_returns_valid_threshold():
    y, s = _detection_data()
    thr, j = youden_j_threshold(y, s)
    assert np.isfinite(thr)
    assert -1.0 <= j <= 1.0


# --------------------------------------------------------------- bootstrap CI
def test_bootstrap_ci_brackets_point_estimate():
    data = np.random.default_rng(0).normal(size=500)
    point, lo, hi = bootstrap_threshold_ci(np.median, data, n_boot=300, seed=1)
    assert lo <= point <= hi


def test_conformal_ci_orders_and_finite():
    data = np.random.default_rng(0).uniform(size=800)
    point, lo, hi = conformal_threshold_ci(data, alpha=0.1, n_boot=300, seed=1)
    assert lo <= hi
    assert all(np.isfinite([point, lo, hi]))
