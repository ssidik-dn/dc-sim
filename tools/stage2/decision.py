"""Stage 2 Gate A: `DecisionValidation` -- pure comparison logic between
a `PlannerPrediction` and one or more `HardwareResult`s. No hardware is
touched here; every function takes already-produced objects (per Gate
A's own scope: "do not execute hardware," "implement ... decision
validation pure logic").
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

from .contracts import (
    DECISION_VALIDATION_VERSION,
    DecisionValidation,
    HardwareResult,
    NoiseFloorSource,
    PlacementComparison,
    PlacementRankAssignment,
    PlannerPrediction,
    Provenance,
    ResolvabilitySpec,
    SloComparison,
    ThroughputComparison,
)
from .validators import is_eligible_for_hardware_best


def compute_hardware_best(
    candidates: Sequence[HardwareResult],
    *,
    metric: str = "mean",
) -> Optional[HardwareResult]:
    """S14: `HardwareBest` is established from a *bounded candidate
    set* under *equivalent measurement conditions* -- same model,
    workload, runtime, regime, topology (the caller is responsible for
    only passing in a set that already satisfies this; this function's
    own job is narrower: exclude anything not eligible at all -- a
    resource-busy refusal or a contended run -- then pick the best of
    what remains).

    Returns `None` if nothing in `candidates` is eligible -- never picks
    a contended or resource-busy result "because nothing else was
    available." A missing `HardwareBest` is itself a fact worth
    reporting, not a reason to lower the bar (S14: "Resource-busy runs
    are NOT planner failures. Contended runs are NOT clean candidates
    for HardwareBest unless explicitly normalized/accepted.")."""
    eligible = [c for c in candidates if is_eligible_for_hardware_best(c)]
    if not eligible:
        return None

    def _metric_value(result: HardwareResult) -> float:
        if result.metrics is None:
            return float("inf")
        value = getattr(result.metrics.tpot, metric)
        return value if value is not None else float("inf")

    return min(eligible, key=_metric_value)


def _tpot_mean(result: HardwareResult) -> Optional[float]:
    if result.metrics is None or result.metrics.tpot is None:
        return None
    return result.metrics.tpot.mean


def compute_decision_validation(
    prediction: PlannerPrediction,
    planner_selected_result: HardwareResult,
    hardware_best: Optional[HardwareResult],
    *,
    noise_floor_source: Optional[NoiseFloorSource],
    k: int = 2,
    requested_placement: Sequence[PlacementRankAssignment] = (),
    slo_floor_ms: Optional[float] = None,
    throughput_floor: Optional[float] = None,
    equivalence_group_hardware_ids: Sequence[str] = (),
) -> DecisionValidation:
    """The one pure comparison this whole contract exists to produce.

    `planner_selected_result` is the real, observed outcome for the
    candidate the planner actually chose -- required even when it is
    not `hardware_best`, since regret is a comparison between the two,
    not a report about only the winner.

    `resolvability` is computed strictly from `noise_floor_source`
    (S13/S15 both apply): pass `None` when no configuration-specific
    measurement exists, and this function marks the result unresolved
    rather than inventing a threshold."""
    planner_candidate_id = prediction.candidate_id
    hardware_best_id = hardware_best.candidate_id if hardware_best is not None else None

    top1_correct: Optional[bool] = None
    regret_absolute: Optional[float] = None
    regret_relative: Optional[float] = None
    planner_margin: Optional[float] = None
    hardware_margin: Optional[float] = None

    selected_tpot = _tpot_mean(planner_selected_result)
    best_tpot = _tpot_mean(hardware_best) if hardware_best is not None else None

    if hardware_best is not None and selected_tpot is not None and best_tpot is not None:
        top1_correct = (planner_candidate_id == hardware_best_id)
        regret_absolute = selected_tpot - best_tpot
        regret_relative = (regret_absolute / best_tpot) if best_tpot else None
        hardware_margin = regret_absolute
        planner_margin = (
            prediction.predicted.objective_value - best_tpot
            if prediction.predicted.objective_value is not None else None
        )

    topk_correct: Optional[bool] = None
    if top1_correct is not None:
        topk_correct = top1_correct or (planner_candidate_id in equivalence_group_hardware_ids)

    # Resolvability -- S13: never inherited, never substituted.
    if noise_floor_source is None or regret_absolute is None:
        resolvability = ResolvabilitySpec(
            resolvable=None,
            reason=("no configuration-specific noise-floor measurement was supplied"
                   if noise_floor_source is None else
                   "no comparable hardware-best result to measure a regret against"),
            noise_floor_source=noise_floor_source,
        )
    else:
        floor = noise_floor_source.measured_ci95_halfwidth
        if floor is None:
            resolvability = ResolvabilitySpec(
                resolvable=None,
                reason="noise_floor_source carries no measured CI half-width",
                noise_floor_source=noise_floor_source,
            )
        elif abs(regret_absolute) < floor:
            resolvability = ResolvabilitySpec(
                resolvable=False,
                reason=(f"regret ({regret_absolute:.4f}) is smaller than the measured "
                       f"noise floor ({floor:.4f}) at this configuration -- hardware "
                       "cannot distinguish the planner's choice from the true optimum"),
                noise_floor_source=noise_floor_source,
            )
        else:
            resolvability = ResolvabilitySpec(
                resolvable=True,
                reason=(f"regret ({regret_absolute:.4f}) exceeds the measured noise "
                       f"floor ({floor:.4f}) at this configuration"),
                noise_floor_source=noise_floor_source,
            )

    # SLO comparison -- planner predicted vs. hardware observed, never
    # conflated (S12: "Do not call a difference meaningful simply
    # because it is non-zero" -- SLO is a pass/fail comparison, not a
    # magnitude one).
    planner_slo_pass = prediction.predicted.slo_pass
    hardware_slo_pass: Optional[bool] = None
    if planner_selected_result.metrics is not None and slo_floor_ms is not None:
        observed_tpot = _tpot_mean(planner_selected_result)
        if observed_tpot is not None:
            hardware_slo_pass = observed_tpot <= slo_floor_ms
    slo = SloComparison(planner_predicted_pass=planner_slo_pass,
                        hardware_observed_pass=hardware_slo_pass)

    # Throughput comparison.
    planner_throughput = prediction.predicted.throughput_rps
    hardware_throughput = (
        planner_selected_result.metrics.request_throughput_rps
        if planner_selected_result.metrics is not None else None
    )
    floor_pass = (
        hardware_throughput >= throughput_floor
        if hardware_throughput is not None and throughput_floor is not None else None
    )
    throughput = ThroughputComparison(
        planner_predicted=planner_throughput, hardware_observed=hardware_throughput,
        floor=throughput_floor, floor_pass=floor_pass,
    )

    observed_placement = planner_selected_result.observed_placement
    exact_match = _placements_match(requested_placement, observed_placement)
    placement = PlacementComparison(
        requested=tuple(requested_placement), observed=tuple(observed_placement),
        exact_match=exact_match,
    )

    tie_handling = (
        f"planner's own winner-equivalence group had "
        f"{prediction.ranking.winner_equivalence_group_size} member(s) "
        f"(indistinguishable_from_winner={prediction.ranking.indistinguishable_from_winner}); "
        "top-k correctness above also credits a hardware winner that matches any id in "
        "equivalence_group_hardware_ids, not only an exact top-1 match"
    )

    return DecisionValidation(
        decision_validation_version=DECISION_VALIDATION_VERSION,
        plan_id=prediction.plan_id,
        planner_selected_candidate=planner_candidate_id,
        hardware_best_candidate=hardware_best_id,
        top1_correct=top1_correct,
        topk_correct=topk_correct,
        k=k,
        regret_absolute=regret_absolute,
        regret_relative=regret_relative,
        planner_margin=planner_margin,
        hardware_margin=hardware_margin,
        tie_handling=tie_handling,
        resolvability=resolvability,
        slo=slo,
        throughput=throughput,
        placement=placement,
        provenance=Provenance(timestamp_utc=datetime.now(timezone.utc).isoformat()),
    )


def _placements_match(
    requested: Sequence[PlacementRankAssignment],
    observed: Sequence[PlacementRankAssignment],
) -> bool:
    if not requested or not observed:
        return False
    key = lambda a: (a.logical_group, a.replica, a.parallel_rank, a.host, a.physical_gpu)  # noqa: E731
    return sorted(requested, key=key) == sorted(observed, key=key)
