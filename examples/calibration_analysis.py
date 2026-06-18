"""Calibration and threshold selection on kNN OOD scores.

    python examples/calibration_analysis.py

Covers:
  1. a conformal cut that replaces the p90 heuristic with an FPR guarantee,
  2. risk-coverage / AURC over the confidence ordering,
  3. cost vs constraint vs Youden operating points,
  4. a bootstrap CI feeding the CI-gated threshold-adjustment rule.
"""

import numpy as np

from pitwaller.embeddings import MockEmbedder
from pitwaller.experimental import (
    Action,
    ThresholdDriftSignal,
    aurc,
    conformal_threshold,
    conformal_threshold_ci,
    constraint_threshold,
    cost_optimal_threshold,
    coverage_at_risk,
    recommend,
    youden_j_threshold,
)
from pitwaller.monitoring import Diagnostics
from pitwaller.ood import OODModel

rng = np.random.default_rng(0)
emb = MockEmbedder(dim=64, seed=1)

train = emb.embed([(int(rng.integers(0, 8)), 0.4, i) for i in range(2000)])
model = OODModel(k=10).fit(train)

# Calibration set (held-out in-distribution) and a mixed production set.
cal = emb.embed([(int(rng.integers(0, 8)), 0.4, 50_000 + i) for i in range(1000)])
cal_dist = np.array([r.knn_distance for r in model.score(cal)])

prod_in = emb.embed([(int(rng.integers(0, 8)), 0.4, 60_000 + i) for i in range(400)])
# Half "camouflaged" (drawn like inliers) and half clearly drifted; the overlap
# makes operating points differ so the cost asymmetry moves the threshold.
ood_camo = emb.embed([(int(rng.integers(0, 8)), 0.4, 70_000 + i) for i in range(60)])
ood_clear = emb.embed([(int(rng.integers(0, 8)), 0.7, 80_000 + i) for i in range(60)])
prod_ood = np.vstack([ood_camo, ood_clear])
prod = np.vstack([prod_in, prod_ood])
prod_dist = np.array([r.knn_distance for r in model.score(prod)])
is_ood = np.array([False] * len(prod_in) + [True] * len(prod_ood))

# 1. Conformal cut vs the percentile heuristic.
alpha = 0.05
q_conf = conformal_threshold(cal_dist, alpha)
realised_fpr = float(np.mean(prod_dist[~is_ood] > q_conf))
print("1) Conformal outlier cut")
print(f"   p90 (heuristic)        = {model.p90:.4f}")
print(f"   conformal q (alpha={alpha}) = {q_conf:.4f}")
print(f"   guaranteed FPR <= {alpha:.0%}; realised on inliers = {realised_fpr:.1%}\n")

# 2. Risk-coverage of the confidence ordering.
# Correctness falls with OOD distance, so accuracy drops as samples drift out.
p_correct = 1.0 / (1.0 + np.exp((prod_dist - 0.045) / 0.008))
correct = rng.random(prod_dist.size) < p_correct
confidence = -prod_dist  # nearer the core means more confident
print("2) Selective prediction")
print(f"   AURC = {aurc(confidence, correct):.4f}")
print(f"   coverage at <=5% risk = {coverage_at_risk(confidence, correct, 0.05):.0%}\n")

# 3. Operating points: Youden vs cost vs constraint.
t_j, _ = youden_j_threshold(is_ood, prod_dist)
t_cost, _ = cost_optimal_threshold(is_ood, prod_dist, c_fp=1.0, c_fn=5.0)
t_con, m = constraint_threshold(is_ood, prod_dist, max_fpr=0.02)
print("3) Operating points on the OOD score")
print(f"   Youden's J            : thr={t_j:.4f}")
print(f"   cost (miss 5x worse)  : thr={t_cost:.4f}")
print(f"   constraint FPR<=2%    : thr={t_con:.4f} (fpr={m['fpr']:.1%}, tpr={m['tpr']:.0%})\n")

# 4. Bootstrap CI feeding the CI-gated threshold adjustment.
point, lo, hi = conformal_threshold_ci(cal_dist, alpha, n_boot=500, seed=2)
deployed = 0.0335  # stale deployed cut
sig = ThresholdDriftSignal(current_threshold=deployed, new_threshold=point,
                           ci_low=lo, ci_high=hi)
print("4) CI-gated threshold adjustment")
print(f"   re-estimated cut = {point:.4f}  CI=[{lo:.4f}, {hi:.4f}]")
print(f"   deployed cut     = {deployed:.4f}  -> drift significant: {sig.significant}")

healthy = Diagnostics(
    n=len(prod), ood_rate=0.0, margin_rate=0.0, if_outlier_rate=0.0,
    tier_distribution={"HIGH": 0.85, "MED": 0.12, "LOW": 0.03},
    accuracy_overall=0.95, accuracy_by_tier={"HIGH": 0.97, "MED": 0.85, "LOW": 0.6},
    baseline_high_rate=0.85, baseline_accuracy=0.95,
)
recs = recommend(healthy, threshold_drift=sig)
fired = Action.THRESHOLD_ADJUSTMENT in [r.action for r in recs]
print(f"   policy recommends THRESHOLD_ADJUSTMENT: {fired}")
