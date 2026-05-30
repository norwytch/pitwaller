import numpy as np
import pytest

from pitwaller.confidence import Tier
from pitwaller.embeddings import MockEmbedder
from pitwaller.ood import OODModel
from pitwaller.pipeline import ConfidencePipeline
from pitwaller.tier_calibration import (
    ReliabilityModel,
    TierCalibrator,
    ood_features,
    risk_targeted_threshold,
)


# --------------------------------------------------------------- reliability map


def _separable_calibration(n=2000, seed=0):
    """One informative feature: larger = more likely correct."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-2.0 * x))  # monotone in x
    correct = rng.random(n) < p
    return x[:, None], correct


def test_reliability_map_is_monotone_and_calibrated():
    X, correct = _separable_calibration()
    rel = ReliabilityModel().fit(X, correct)
    # Higher feature -> higher predicted reliability.
    grid = np.linspace(-2, 2, 20)[:, None]
    p = rel.predict(grid)
    assert np.all(np.diff(p) >= -1e-9)
    # Calibrated: ECE is small on the same distribution.
    assert rel.ece(X, correct) < 0.05
    # Coefficient picks up the positive relationship.
    assert rel.coefficients[0] > 0


def test_reliability_map_handles_single_class():
    X = np.random.default_rng(1).normal(size=(50, 2))
    rel = ReliabilityModel().fit(X, np.ones(50, dtype=bool))
    assert np.allclose(rel.predict(X), 1.0)
    assert rel.coefficients is None  # degenerated to a constant


# ---------------------------------------------------------- risk-targeted cuts


def test_risk_targeted_threshold_holds_empirical_risk():
    X, correct = _separable_calibration()
    score = ReliabilityModel().fit(X, correct).predict(X)
    tau = risk_targeted_threshold(score, correct, target_risk=0.05)
    accepted = score >= tau
    assert accepted.any()
    realised_risk = 1.0 - correct[accepted].mean()
    assert realised_risk <= 0.05 + 1e-9


def test_tighter_risk_gives_higher_or_equal_cut():
    X, correct = _separable_calibration()
    score = ReliabilityModel().fit(X, correct).predict(X)
    tau_strict = risk_targeted_threshold(score, correct, 0.02)
    tau_loose = risk_targeted_threshold(score, correct, 0.10)
    assert tau_strict >= tau_loose


def test_unattainable_risk_returns_inf():
    # Even the most confident sample is wrong -> no certifiable accepted set.
    score = np.array([0.9, 0.8, 0.7])
    correct = np.array([False, True, True])
    assert risk_targeted_threshold(score, correct, 0.01) == float("inf")


def test_finite_sample_guarantee_is_conservative():
    # With a delta bound, a tiny calibration set cannot certify a 1% risk.
    score = np.array([0.99, 0.98, 0.97, 0.96])
    correct = np.array([True, True, True, True])
    assert risk_targeted_threshold(score, correct, 0.01, delta=0.05) == float("inf")


# -------------------------------------------------------------- TierCalibrator


def test_calibrator_orders_tiers_by_reliability():
    X, correct = _separable_calibration()
    cal = TierCalibrator(risk_high=0.02, risk_med=0.10).fit(X, correct)
    # Very confident -> HIGH, middling -> not HIGH, hopeless -> LOW.
    tiers = cal.tier(np.array([[5.0], [0.0], [-5.0]]))
    assert tiers[0] is Tier.HIGH
    assert tiers[2] is Tier.LOW
    # Monotone: reliability of HIGH samples >= MED samples >= LOW samples.
    assert cal.calibration_.tau_high >= cal.calibration_.tau_med


def test_calibrator_rejects_bad_risk_order():
    with pytest.raises(ValueError):
        TierCalibrator(risk_high=0.10, risk_med=0.02)


def test_fit_results_uses_feature_fn():
    emb = MockEmbedder(dim=32, seed=3)
    rng = np.random.default_rng(0)
    train = emb.embed([(int(rng.integers(0, 8)), 0.4, i) for i in range(800)])
    model = OODModel(k=10).fit(train)
    results = model.score(train)
    correct = ood_features(results)[:, 0] < np.median(ood_features(results)[:, 0])
    cal = TierCalibrator(risk_high=0.05, risk_med=0.2).fit_results(results, correct)
    assert len(cal.tier_results(results)) == len(results)


# --------------------------------------------------- pipeline integration (opt-in)


def _pipe_with_data(seed=0):
    emb = MockEmbedder(dim=64, seed=2)
    rng = np.random.default_rng(seed)
    train = emb.embed([(int(rng.integers(0, 8)), 0.4, i) for i in range(2000)])
    pipe = ConfidencePipeline(emb, k=10).fit(train)
    return pipe, emb, rng


def test_pipeline_uncalibrated_by_default():
    pipe, _, _ = _pipe_with_data()
    assert pipe.calibrated is False


def test_calibrate_before_fit_raises():
    emb = MockEmbedder(dim=64)
    with pytest.raises(RuntimeError):
        ConfidencePipeline(emb).calibrate(np.zeros((4, 64), np.float32), [True] * 4)


def test_calibrated_pipeline_swaps_tiering():
    pipe, emb, rng = _pipe_with_data()
    # Build a labelled calibration set where correctness tracks OOD band:
    # in-distribution mostly right, off-manifold mostly wrong.
    cal = emb.embed([(int(rng.integers(0, 8)), 0.4, 50_000 + i) for i in range(400)])
    cal_ood = emb.embed([(99, 0.5, 60_000 + i) for i in range(200)])
    cal_inputs = np.vstack([cal, cal_ood])
    results = pipe.ood.score(cal_inputs)
    correct = np.array([r.band == "core" for r in results])

    pipe.calibrate(cal_inputs, correct, risk_high=0.05, risk_med=0.2)
    assert pipe.calibrated is True

    # Clean in-distribution data should tier HIGH more often than off-manifold.
    indist = emb.embed([(int(rng.integers(0, 8)), 0.4, 70_000 + i) for i in range(200)])
    ood = emb.embed([(99, 0.5, 80_000 + i) for i in range(200)])
    high_in = np.mean([s.tier == Tier.HIGH for s in pipe.score(indist)])
    high_ood = np.mean([s.tier == Tier.HIGH for s in pipe.score(ood)])
    assert high_in > high_ood
