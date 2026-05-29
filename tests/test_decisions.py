from pitwaller.decisions import Action, PolicyThresholds, recommend
from pitwaller.monitoring import Diagnostics


def diag(**kw):
    """Build Diagnostics with healthy defaults, overriding selected fields."""
    base = dict(
        n=1000,
        ood_rate=0.02,
        margin_rate=0.05,
        if_outlier_rate=0.02,
        tier_distribution={"HIGH": 0.85, "MED": 0.12, "LOW": 0.03},
        accuracy_overall=0.95,
        accuracy_by_tier={"HIGH": 0.97, "MED": 0.85, "LOW": 0.6},
        per_class_recall={i: 0.9 for i in range(8)},
        labelled_fraction=1.0,
        baseline_high_rate=0.85,
        baseline_accuracy=0.95,
    )
    base.update(kw)
    return Diagnostics(**base)


def top_actions(recs):
    return [r.action for r in recs]


def test_healthy_recommends_nothing():
    recs = recommend(diag())
    assert top_actions(recs) == [Action.NONE]


def test_threshold_adjustment_when_tiers_drift_but_accuracy_holds():
    d = diag(
        tier_distribution={"HIGH": 0.65, "MED": 0.25, "LOW": 0.10},
        accuracy_overall=0.94,  # essentially stable
    )
    assert Action.THRESHOLD_ADJUSTMENT in top_actions(recommend(d))


def test_bn_recalibration_on_covariate_shift():
    d = diag(ood_rate=0.18, if_outlier_rate=0.15, accuracy_overall=0.945)
    assert Action.BN_RECALIBRATION in top_actions(recommend(d))


def test_adasyn_on_class_collapse():
    recall = {i: 0.9 for i in range(8)}
    recall[3] = 0.4  # one class collapses
    d = diag(per_class_recall=recall)
    assert Action.ADASYN_REBALANCE in top_actions(recommend(d))


def test_low_recall_class_with_thin_support_does_not_fire_adasyn():
    # One class shows 0% recall but only 3 labelled samples -> too thin to trust;
    # a single mislabel shouldn't trigger a rebalance + retrain.
    recall = {i: 0.9 for i in range(8)}
    recall[3] = 0.0
    support = {i: 200 for i in range(8)}
    support[3] = 3
    d = diag(per_class_recall=recall, per_class_support=support)
    assert Action.ADASYN_REBALANCE not in top_actions(recommend(d))


def test_low_recall_class_with_ample_support_fires_adasyn():
    recall = {i: 0.9 for i in range(8)}
    recall[3] = 0.4
    support = {i: 200 for i in range(8)}
    d = diag(per_class_recall=recall, per_class_support=support)
    assert Action.ADASYN_REBALANCE in top_actions(recommend(d))


def test_escalation_does_not_duplicate_an_already_recommended_action():
    # Tiers drifted (threshold rule) AND covariate shift (BN rule already fires);
    # escalating the threshold fix up to BN must not emit a second BN rec.
    d = diag(
        tier_distribution={"HIGH": 0.65, "MED": 0.25, "LOW": 0.10},
        accuracy_overall=0.94,
        ood_rate=0.18,
        if_outlier_rate=0.15,
    )
    recs = recommend(
        d,
        PolicyThresholds(),
        recent_attempts={Action.THRESHOLD_ADJUSTMENT.value: 3},
    )
    actions = top_actions(recs)
    assert actions.count(Action.BN_RECALIBRATION) == 1


def test_partial_retrain_on_moderate_drop():
    d = diag(accuracy_overall=0.90)  # 5% drop -> moderate
    assert Action.PARTIAL_BACKBONE_RETRAIN in top_actions(recommend(d))


def test_full_retrain_on_severe_drop():
    d = diag(accuracy_overall=0.80)  # 15% drop -> severe
    assert Action.FULL_BACKBONE_RETRAIN in top_actions(recommend(d))


def test_pruning_under_size_pressure_when_healthy():
    recs = recommend(diag(), PolicyThresholds(size_pressure=True))
    assert Action.PRUNING in top_actions(recs)


def test_architecture_rebuild_after_failed_retrains():
    d = diag(ood_rate=0.35, accuracy_overall=0.78)
    recs = recommend(
        d,
        PolicyThresholds(),
        recent_attempts={Action.FULL_BACKBONE_RETRAIN.value: 2},
    )
    assert Action.ARCHITECTURE_REBUILD in top_actions(recs)


def test_recommendations_sorted_cheapest_first():
    d = diag(accuracy_overall=0.80, ood_rate=0.18, if_outlier_rate=0.15)
    recs = recommend(d)
    costs = [r.cost for r in recs]
    assert costs == sorted(costs)


def test_escalation_promotes_repeated_cheap_fix():
    # Tiers drift repeatedly; threshold tweaks haven't stuck -> escalate.
    d = diag(
        tier_distribution={"HIGH": 0.65, "MED": 0.25, "LOW": 0.10},
        accuracy_overall=0.94,
    )
    recs = recommend(
        d,
        PolicyThresholds(),
        recent_attempts={Action.THRESHOLD_ADJUSTMENT.value: 3},
    )
    actions = top_actions(recs)
    assert Action.THRESHOLD_ADJUSTMENT in actions
    assert Action.BN_RECALIBRATION in actions  # promoted one rung up


# --------------------------------------------------------------- effort tiers
from pitwaller.decisions import (  # noqa: E402
    EFFORT,
    EFFORT_ORDER,
    Action as A,
    EffortTier,
    group_by_effort,
    heaviest_tier,
    recommend as _recommend,
)


def test_every_action_has_an_effort_profile():
    assert set(EFFORT) == set(A)


def test_pit_stop_actions_stay_live_and_need_no_training():
    for action in (A.THRESHOLD_ADJUSTMENT, A.BN_RECALIBRATION):
        e = EFFORT[action]
        assert e.tier is EffortTier.PIT_STOP
        assert e.stays_live is True
        assert e.gpu_intensity in {"none", "light"}


def test_heavy_actions_map_to_heavy_tiers():
    assert EFFORT[A.FULL_BACKBONE_RETRAIN].tier is EffortTier.ENGINE_REBUILD
    assert EFFORT[A.ARCHITECTURE_REBUILD].tier is EffortTier.NEW_BUILD
    assert EFFORT[A.ARCHITECTURE_REBUILD].gpu_intensity == "heavy"


def test_effort_tier_increases_with_cost():
    # The pit-lane bucketing must not contradict the cost ladder.
    for a in A:
        for b in A:
            if EFFORT[a].tier is EFFORT[b].tier:
                continue
            if EFFORT_ORDER.index(EFFORT[a].tier) < EFFORT_ORDER.index(EFFORT[b].tier):
                from pitwaller.decisions import _COST
                assert _COST[a] < _COST[b]


def test_healthy_is_green_flag():
    recs = _recommend(diag())
    assert heaviest_tier(recs) is EffortTier.GREEN_FLAG


def test_severe_drop_is_engine_rebuild():
    recs = _recommend(diag(accuracy_overall=0.80))
    assert heaviest_tier(recs) is EffortTier.ENGINE_REBUILD


def test_group_by_effort_orders_light_to_heavy():
    # Pit-stop fixes (threshold/BN) require stable accuracy by design, so they
    # can't co-occur with a severe drop. Two genuinely reachable spreads:

    # (a) GARAGE + ENGINE_REBUILD: a class collapses *and* overall accuracy
    # craters -> targeted rebalance plus a full retrain.
    recall = {i: 0.9 for i in range(8)}
    recall[3] = 0.4
    d = diag(accuracy_overall=0.80, per_class_recall=recall)
    groups = group_by_effort(_recommend(d))
    tiers = list(groups.keys())
    assert tiers == sorted(tiers, key=EFFORT_ORDER.index)
    assert EffortTier.GARAGE in tiers
    assert EffortTier.ENGINE_REBUILD in tiers

    # (b) PIT_STOP + GARAGE: accuracy holds, tiers drift, one class collapses.
    d2 = diag(
        tier_distribution={"HIGH": 0.65, "MED": 0.25, "LOW": 0.10},
        accuracy_overall=0.94,
        per_class_recall=recall,
    )
    tiers2 = list(group_by_effort(_recommend(d2)).keys())
    assert tiers2 == sorted(tiers2, key=EFFORT_ORDER.index)
    assert EffortTier.PIT_STOP in tiers2
    assert EffortTier.GARAGE in tiers2


# ----------------------------------------------------- CI-gated threshold rule
from pitwaller.decisions import ThresholdDriftSignal  # noqa: E402


def test_threshold_drift_significant_when_current_outside_ci():
    sig = ThresholdDriftSignal(current_threshold=5.0, new_threshold=1.0,
                               ci_low=0.5, ci_high=1.5)
    assert sig.significant is True


def test_threshold_drift_not_significant_when_current_inside_ci():
    sig = ThresholdDriftSignal(current_threshold=1.0, new_threshold=1.1,
                               ci_low=0.5, ci_high=1.5)
    assert sig.significant is False


def test_significant_drift_recommends_threshold_adjustment():
    sig = ThresholdDriftSignal(current_threshold=5.0, new_threshold=1.0,
                               ci_low=0.5, ci_high=1.5)
    recs = recommend(diag(), threshold_drift=sig)  # otherwise-healthy diagnostics
    assert Action.THRESHOLD_ADJUSTMENT in top_actions(recs)


def test_nonsignificant_drift_suppresses_heuristic_threshold_rec():
    # HIGH-share dropped (heuristic would fire) but the CI says it's just noise.
    d = diag(tier_distribution={"HIGH": 0.65, "MED": 0.25, "LOW": 0.10},
             accuracy_overall=0.94)
    sig = ThresholdDriftSignal(current_threshold=1.0, new_threshold=1.1,
                               ci_low=0.5, ci_high=1.5)  # not significant
    recs = recommend(d, threshold_drift=sig)
    assert Action.THRESHOLD_ADJUSTMENT not in top_actions(recs)
