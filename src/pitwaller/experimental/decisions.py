"""Heuristic remediation policy mapping diagnostics to corrective actions.

Maps :class:`~pitwaller.monitoring.Diagnostics` onto an ordered ladder of
actions, cheapest and least destructive first:

    THRESHOLD_ADJUSTMENT      recalibrate tier cut-points; no weights touched
    BN_RECALIBRATION          refresh BatchNorm running stats on fresh inputs
    PARTIAL_BACKBONE_RETRAIN  fine-tune later layers / affected classes
    ADASYN_REBALANCE          synthesise minority-class samples, then retrain
    FULL_BACKBONE_RETRAIN     retrain the whole backbone
    PRUNING                   shrink the model (accuracy intact)
    ARCHITECTURE_REBUILD      capacity ceiling; redesign the network

Each rule inspects the diagnostics and, if it fires, emits a
:class:`Recommendation` with the action, severity, rationale, and triggering
signals. ``recommend`` returns the full ranked list, not just the top action.

The policy diagnoses the kind of failure, not just its size: covariate shift
(inputs drift, OOD climbs, accuracy holds) wants BN recalibration, a broad
accuracy drop wants retraining, a single collapsing class wants ADASYN. It also
escalates: ``recent_attempts`` promotes a repeatedly-applied cheap fix up the
ladder so it doesn't loop forever on a problem that needs a retrain.

This is a heuristic policy with no outcome feedback loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..monitoring import Diagnostics


class Action(str, Enum):
    NONE = "NONE"
    THRESHOLD_ADJUSTMENT = "THRESHOLD_ADJUSTMENT"
    BN_RECALIBRATION = "BN_RECALIBRATION"
    PARTIAL_BACKBONE_RETRAIN = "PARTIAL_BACKBONE_RETRAIN"
    ADASYN_REBALANCE = "ADASYN_REBALANCE"
    FULL_BACKBONE_RETRAIN = "FULL_BACKBONE_RETRAIN"
    PRUNING = "PRUNING"
    ARCHITECTURE_REBUILD = "ARCHITECTURE_REBUILD"


# Cost/destructiveness ordering used to rank multiple firing rules.
_COST = {
    Action.NONE: 0,
    Action.THRESHOLD_ADJUSTMENT: 1,
    Action.BN_RECALIBRATION: 2,
    Action.PRUNING: 3,
    Action.PARTIAL_BACKBONE_RETRAIN: 4,
    Action.ADASYN_REBALANCE: 5,
    Action.FULL_BACKBONE_RETRAIN: 6,
    Action.ARCHITECTURE_REBUILD: 7,
}


class EffortTier(str, Enum):
    """How intensive a remediation is: wall-clock, labels, GPU, and whether the
    model can keep serving while you do it.

        GREEN_FLAG     nothing wrong
        PIT_STOP       config only, no training, model stays live
        GARAGE         bounded retrain, redeploy
        ENGINE_REBUILD retrain the whole backbone
        NEW_BUILD      redesign the architecture
    """

    GREEN_FLAG = "GREEN_FLAG"
    PIT_STOP = "PIT_STOP"
    GARAGE = "GARAGE"
    ENGINE_REBUILD = "ENGINE_REBUILD"
    NEW_BUILD = "NEW_BUILD"


@dataclass(frozen=True)
class EffortProfile:
    """What a tier of remediation costs you in practice."""

    tier: EffortTier
    touches_weights: bool      # does it modify model parameters at all?
    needs_labels: bool         # does the fix require labelled data?
    gpu_intensity: str         # "none" | "light" | "moderate" | "heavy"
    stays_live: bool           # can the model keep serving during the fix?
    typical_duration: str      # rough wall-clock
    reversible: bool           # cheap to roll back (prior checkpoint aside)?


# Each action's effort profile. Ordering of tiers mirrors the cost ladder.
EFFORT: dict["Action", EffortProfile] = {
    Action.NONE: EffortProfile(
        EffortTier.GREEN_FLAG, touches_weights=False, needs_labels=False,
        gpu_intensity="none", stays_live=True,
        typical_duration="n/a", reversible=True,
    ),
    # PIT STOP: config / stats only, no training, model stays live
    Action.THRESHOLD_ADJUSTMENT: EffortProfile(
        EffortTier.PIT_STOP, touches_weights=False, needs_labels=False,
        gpu_intensity="none", stays_live=True,
        typical_duration="seconds-minutes", reversible=True,
    ),
    Action.BN_RECALIBRATION: EffortProfile(
        EffortTier.PIT_STOP, touches_weights=False, needs_labels=False,
        gpu_intensity="light", stays_live=True,   # forward passes only; hot-swap
        typical_duration="minutes", reversible=True,
    ),
    # GARAGE: bounded weight surgery + fine-tune, redeploy
    Action.PRUNING: EffortProfile(
        EffortTier.GARAGE, touches_weights=True, needs_labels=True,
        gpu_intensity="light", stays_live=False,
        typical_duration="hours", reversible=False,
    ),
    Action.PARTIAL_BACKBONE_RETRAIN: EffortProfile(
        EffortTier.GARAGE, touches_weights=True, needs_labels=True,
        gpu_intensity="moderate", stays_live=False,
        typical_duration="hours", reversible=False,
    ),
    Action.ADASYN_REBALANCE: EffortProfile(
        EffortTier.GARAGE, touches_weights=True, needs_labels=True,
        gpu_intensity="moderate", stays_live=False,
        typical_duration="hours-1 day", reversible=False,
    ),
    # ENGINE REBUILD: retrain the whole backbone
    Action.FULL_BACKBONE_RETRAIN: EffortProfile(
        EffortTier.ENGINE_REBUILD, touches_weights=True, needs_labels=True,
        gpu_intensity="heavy", stays_live=False,
        typical_duration="days", reversible=False,
    ),
    # NEW BUILD: redesign the network
    Action.ARCHITECTURE_REBUILD: EffortProfile(
        EffortTier.NEW_BUILD, touches_weights=True, needs_labels=True,
        gpu_intensity="heavy", stays_live=False,
        typical_duration="weeks", reversible=False,
    ),
}


def _effort(action: "Action") -> EffortProfile:
    return EFFORT[action]


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass
class Recommendation:
    action: Action
    severity: Severity
    rationale: str
    triggers: dict = field(default_factory=dict)

    @property
    def cost(self) -> int:
        return _COST[self.action]

    @property
    def effort(self) -> EffortProfile:
        """Effort profile of this action."""
        return EFFORT[self.action]

    @property
    def effort_tier(self) -> EffortTier:
        return EFFORT[self.action].tier


@dataclass
class PolicyThresholds:
    """Policy tunables."""

    ood_rate_warn: float = 0.10          # outlier-band fraction worth noticing
    ood_rate_critical: float = 0.30      # persistent heavy OOD -> capacity issue
    high_rate_drop_warn: float = 0.10    # HIGH-confidence fraction slipping
    accuracy_drop_warn: float = 0.03
    accuracy_drop_critical: float = 0.10
    tiered_accuracy_gap: float = 0.05    # HIGH should beat LOW by at least this
    class_recall_floor: float = 0.60     # below this a class is "collapsing"
    class_recall_min_support: int = 20   # min labelled samples before recall is trusted
    class_imbalance_min_classes: int = 1
    size_pressure: bool = False          # latency/footprint flag from serving
    escalate_after_attempts: int = 2     # repeated cheap fix -> promote


@dataclass
class ThresholdDriftSignal:
    """Evidence about whether the deployed threshold is stale.

    Built from :mod:`pitwaller.experimental.calibration`: re-estimate the optimal
    cut on recent data (``new_threshold``) with a bootstrap CI
    (``ci_low``/``ci_high``). The threshold is significant (worth adjusting) when
    ``current_threshold`` falls outside that CI; inside the CI, an apparent drift
    is sampling noise.
    """

    current_threshold: float
    new_threshold: float
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        return not (self.ci_low <= self.current_threshold <= self.ci_high)


def recommend(
    diag: Diagnostics,
    thresholds: PolicyThresholds | None = None,
    recent_attempts: dict[str, int] | None = None,
    threshold_drift: ThresholdDriftSignal | None = None,
) -> list[Recommendation]:
    """Return remediation recommendations, cheapest-first.

    ``recent_attempts`` maps an :class:`Action` value to how many times it has
    already been applied without resolving the issue; this drives escalation.

    ``threshold_drift``, when supplied, replaces the heuristic threshold rule
    with a gated one: a threshold adjustment fires only when the signal is
    significant (see :class:`ThresholdDriftSignal`), regardless of how much the
    HIGH-confidence share moved.
    """
    t = thresholds or PolicyThresholds()
    attempts = recent_attempts or {}
    recs: list[Recommendation] = []

    drop = diag.accuracy_drop
    high_drop = diag.high_rate_drop
    accuracy_stable = drop is None or drop < t.accuracy_drop_warn

    # Rule 1: tier thresholds stale, accuracy intact -> recalibrate cut-points.
    if threshold_drift is not None:
        # Statistically gated: act only on a significant, real drift.
        if threshold_drift.significant and accuracy_stable:
            recs.append(
                Recommendation(
                    Action.THRESHOLD_ADJUSTMENT,
                    Severity.WARN,
                    f"Deployed threshold {threshold_drift.current_threshold:.4g} is "
                    f"outside the bootstrap CI "
                    f"[{threshold_drift.ci_low:.4g}, {threshold_drift.ci_high:.4g}] of the "
                    f"re-estimated optimum {threshold_drift.new_threshold:.4g}; recalibrate.",
                    {
                        "current_threshold": threshold_drift.current_threshold,
                        "new_threshold": threshold_drift.new_threshold,
                        "ci": [threshold_drift.ci_low, threshold_drift.ci_high],
                    },
                )
            )
    elif (
        high_drop is not None
        and high_drop >= t.high_rate_drop_warn
        and accuracy_stable
    ):
        # Heuristic fallback when no calibration signal is available.
        recs.append(
            Recommendation(
                Action.THRESHOLD_ADJUSTMENT,
                Severity.WARN,
                "HIGH-confidence share fell by "
                f"{high_drop:.0%} while accuracy held; tier thresholds are stale.",
                {"high_rate_drop": high_drop, "accuracy_drop": drop},
            )
        )

    # Rule 2: covariate shift (OOD/IF rate up, accuracy stable) -> BN recal,
    # the cheap label-free correction before anything heavier.
    if (
        diag.ood_rate >= t.ood_rate_warn
        and (drop is None or drop < t.accuracy_drop_warn)
        and diag.if_outlier_rate >= t.ood_rate_warn
    ):
        recs.append(
            Recommendation(
                Action.BN_RECALIBRATION,
                Severity.WARN,
                f"Input distribution shifted (OOD {diag.ood_rate:.0%}, "
                f"IF {diag.if_outlier_rate:.0%}) with accuracy stable; "
                "covariate shift; recalibrate BatchNorm on recent inputs.",
                {"ood_rate": diag.ood_rate, "if_outlier_rate": diag.if_outlier_rate},
            )
        )

    # Rule 3: classes collapsing -> targeted rebalance.
    # Only trust low recall once the class has enough labelled support, so one
    # mislabelled rare-class sample can't fire a retrain. No support count
    # recorded (caller built Diagnostics by hand) means don't second-guess it.
    collapsing = [
        c
        for c, r in diag.per_class_recall.items()
        if r < t.class_recall_floor
        and diag.per_class_support.get(c, t.class_recall_min_support) >= t.class_recall_min_support
    ]
    if len(collapsing) >= t.class_imbalance_min_classes and collapsing:
        recs.append(
            Recommendation(
                Action.ADASYN_REBALANCE,
                Severity.WARN,
                f"Recall collapsed on class(es) {sorted(collapsing)} "
                f"(< {t.class_recall_floor:.0%}); synthesise minority samples "
                "(ADASYN) and fine-tune.",
                {"collapsing_classes": sorted(collapsing)},
            )
        )

    # Rule 4: moderate, broad accuracy drop -> partial retrain.
    if drop is not None and t.accuracy_drop_warn <= drop < t.accuracy_drop_critical:
        recs.append(
            Recommendation(
                Action.PARTIAL_BACKBONE_RETRAIN,
                Severity.WARN,
                f"Accuracy down {drop:.1%} (moderate); fine-tune later backbone "
                "layers on recent labelled data.",
                {"accuracy_drop": drop},
            )
        )

    # Rule 5: severe accuracy drop -> full retrain.
    if drop is not None and drop >= t.accuracy_drop_critical:
        recs.append(
            Recommendation(
                Action.FULL_BACKBONE_RETRAIN,
                Severity.CRITICAL,
                f"Accuracy down {drop:.1%} (severe, broad); retrain the full "
                "backbone on a refreshed dataset.",
                {"accuracy_drop": drop},
            )
        )

    # Rule 6: efficiency pressure with healthy accuracy -> prune.
    if t.size_pressure and (drop is None or drop < t.accuracy_drop_warn):
        recs.append(
            Recommendation(
                Action.PRUNING,
                Severity.INFO,
                "Serving under size/latency pressure while accuracy is healthy; "
                "prune to recover headroom.",
                {"size_pressure": True},
            )
        )

    # Rule 7: persistent heavy OOD after retraining -> architecture is the
    # ceiling; the network's inductive biases no longer fit the problem.
    retrain_tries = attempts.get(Action.FULL_BACKBONE_RETRAIN.value, 0)
    if diag.ood_rate >= t.ood_rate_critical and retrain_tries >= t.escalate_after_attempts:
        recs.append(
            Recommendation(
                Action.ARCHITECTURE_REBUILD,
                Severity.CRITICAL,
                f"OOD rate {diag.ood_rate:.0%} persists after {retrain_tries} "
                "full retrains; the architecture has hit its ceiling; rebuild.",
                {"ood_rate": diag.ood_rate, "full_retrain_attempts": retrain_tries},
            )
        )

    # Escalation: promote a repeatedly-applied cheap fix one rung up the ladder.
    if recs:
        cheapest = min(recs, key=lambda r: r.cost)
        tries = attempts.get(cheapest.action.value, 0)
        if tries >= t.escalate_after_attempts:
            promoted = _escalate(cheapest.action)
            already = {r.action for r in recs}
            if promoted is not cheapest.action and promoted not in already:
                recs.append(
                    Recommendation(
                        promoted,
                        Severity.CRITICAL,
                        f"{cheapest.action.value} applied {tries}x without "
                        f"resolution; escalating to {promoted.value}.",
                        {"escalated_from": cheapest.action.value, "attempts": tries},
                    )
                )

    if not recs:
        recs.append(
            Recommendation(
                Action.NONE,
                Severity.INFO,
                "All monitored signals within tolerance; no action required.",
                {},
            )
        )

    # Rank cheapest-first; severity breaks ties (more severe surfaces first).
    sev_rank = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2}
    recs.sort(key=lambda r: (r.cost, sev_rank[r.severity]))
    return recs


def _escalate(action: Action) -> Action:
    """Next rung up the remediation ladder."""
    ladder = [
        Action.THRESHOLD_ADJUSTMENT,
        Action.BN_RECALIBRATION,
        Action.PARTIAL_BACKBONE_RETRAIN,
        Action.ADASYN_REBALANCE,
        Action.FULL_BACKBONE_RETRAIN,
        Action.ARCHITECTURE_REBUILD,
    ]
    if action in ladder:
        i = ladder.index(action)
        return ladder[min(i + 1, len(ladder) - 1)]
    return action


# Effort tiers, lightest -> heaviest, for sorting/reporting.
EFFORT_ORDER = [
    EffortTier.GREEN_FLAG,
    EffortTier.PIT_STOP,
    EffortTier.GARAGE,
    EffortTier.ENGINE_REBUILD,
    EffortTier.NEW_BUILD,
]


def group_by_effort(recs: list[Recommendation]) -> dict[EffortTier, list[Recommendation]]:
    """Bucket recommendations by effort tier, lightest first.

    Empty tiers are omitted. Within a tier, order follows :func:`recommend`.
    """
    buckets: dict[EffortTier, list[Recommendation]] = {}
    for tier in EFFORT_ORDER:
        hits = [r for r in recs if r.effort_tier is tier]
        if hits:
            buckets[tier] = hits
    return buckets


def heaviest_tier(recs: list[Recommendation]) -> EffortTier:
    """The most intensive tier any recommendation calls for."""
    return max(
        (r.effort_tier for r in recs),
        key=EFFORT_ORDER.index,
        default=EffortTier.GREEN_FLAG,
    )
