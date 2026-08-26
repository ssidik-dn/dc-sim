"""Stage 2 Gate A: hermetic tests for the four planner <-> real-runtime
contract objects (`tools/stage2/`).

Hermetic by construction -- no Frontier subprocess, no GPU, no
`sim_real` import. The three named JSON examples this file also checks
(`contracts/stage2/examples/*.json`) were themselves produced by real
`SimulationEvaluator` runs against real Frontier compute profiles
(`tools/stage2/exporters.py`'s own smoke build, not fabricated here) --
this file only re-validates and re-serializes what is already on disk,
it does not invent new numbers.

`pytest.ini`'s own `pythonpath = src` does not reach `tools/`; this
file adds it itself, the same convention `tests/test_planner_core.py`
already established.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from stage2.contracts import (  # noqa: E402
    ConstraintSpec, DeploymentManifest, HardwareMetrics, HardwareResult,
    HardwareSpecRef, InputIdentity, LatencyStats, ModelSpecRef, NoiseFloorSource,
    OccupancyEvidence, ObjectiveSpec, ParallelismSpec, PlacementRankAssignment,
    PlacementSpec, PlannedFeasibility, PlannerPrediction, PredictedMetrics,
    ProfileProvenance, Provenance, RankingSpec, RuntimeExecutionStatus, RuntimeSpec,
    SearchSpec, SystemInfo, ThroughputFloor, TopologySpecRef, UncertaintySpec,
    WorkloadRealization, WorkloadSpecRef,
    DEPLOYMENT_MANIFEST_VERSION, PLANNER_PREDICTION_VERSION,
)
from stage2.serialization import (  # noqa: E402
    check_major_version, from_dict, from_json, SchemaFieldError, SchemaVersionError, to_json,
)
from stage2.validators import (  # noqa: E402
    is_eligible_for_hardware_best, validate_deployment_manifest,
    validate_hardware_result, validate_manifest_prediction_pair, validate_placement,
    validate_runtime_status_is_not_planner_feasibility, validate_workload_spec,
    ValidationError,
)
from stage2.decision import compute_decision_validation, compute_hardware_best  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "contracts" / "stage2" / "examples"


# --------------------------------------------------------------------------
# helpers -- small, valid building blocks reused across several tests
# --------------------------------------------------------------------------


def _placement(assignments, host_map=None):
    return PlacementSpec(assignments=tuple(assignments),
                         topology_machine_to_host=host_map or {0: "hostA", 1: "hostB"})


def _rank(group="DECODE_ATTN", replica=0, parallel_rank=0, host="hostA", gpu=0):
    return PlacementRankAssignment(logical_group=group, replica=replica,
                                   parallel_rank=parallel_rank, host=host, physical_gpu=gpu)


def _minimal_manifest(*, plan_id="p1", candidate_id="c1", relative=None) -> DeploymentManifest:
    workload = WorkloadSpecRef(regime=WorkloadSpecRef.BURST, num_requests=8,
                               prefill_tokens=128, decode_tokens=32)
    constraints = ConstraintSpec(slo_tpot_ms=100.0, slo_attainment_floor=0.0,
                                 throughput_floor=ThroughputFloor(mode=ThroughputFloor.ABSOLUTE, value=0.0),
                                 memory_margin_fraction=0.2)
    return DeploymentManifest(
        manifest_version=DEPLOYMENT_MANIFEST_VERSION, plan_id=plan_id, candidate_id=candidate_id,
        input_identity=InputIdentity(
            topology=TopologySpecRef(name="t", num_machines=2, gpus_per_machine=8,
                                     scale_up_GBps=400.0, scale_out_GBps=50.0, topology_id="tid"),
            hardware=HardwareSpecRef(device="h800", memory_margin_fraction=0.2),
            model=ModelSpecRef(model_name="m", total_experts=1, router_topk=1, is_moe=False,
                               hidden_size=64, num_attention_heads=4, num_key_value_heads=4, num_layers=2),
            workload=workload, constraints=constraints, objective=ObjectiveSpec(minimize="mean_tpot_ms"),
        ),
        parallelism=ParallelismSpec(attn_tp=2, attn_shape=(2,), ffn_ep=1, ep_shape=(1,),
                                    relative=relative, attn_replicas=1, ffn_replicas=1),
        placement=_placement([_rank(host="hostA", gpu=0), _rank(host="hostA", gpu=1, parallel_rank=1)]),
        runtime=RuntimeSpec(), workload=workload, constraints=constraints,
        profile_provenance=ProfileProvenance(profile_files=("f.csv",), device="h800", model="m",
                                             operator_families_covered=("linear_op",)),
        provenance=Provenance(timestamp_utc="2026-01-01T00:00:00+00:00"),
    )


def _prediction_for(manifest: DeploymentManifest, *, objective_value=5.0,
                    ci95_halfwidth=None, indistinguishable=False, rank=0, total=1,
                    group_size=1) -> PlannerPrediction:
    return PlannerPrediction(
        planner_prediction_version=PLANNER_PREDICTION_VERSION, plan_id=manifest.plan_id,
        selected_manifest_id=manifest.plan_id, candidate_id=manifest.candidate_id,
        predicted=PredictedMetrics(objective_value=objective_value, mean_tpot_ms=objective_value,
                                   throughput_rps=40.0, slo_attainment=1.0, slo_pass=True),
        uncertainty=UncertaintySpec(seed_count=3 if ci95_halfwidth is not None else 1,
                                    method="student_t_95_on_seeded_mean" if ci95_halfwidth is not None
                                    else "none_single_burst_run",
                                    ci95_halfwidth=ci95_halfwidth,
                                    interval_low=(objective_value - ci95_halfwidth) if ci95_halfwidth else None,
                                    interval_high=(objective_value + ci95_halfwidth) if ci95_halfwidth else None),
        ranking=RankingSpec(rank=rank, total_candidates_ranked=total,
                            indistinguishable_from_winner=indistinguishable,
                            winner_equivalence_group_size=group_size),
        search=SearchSpec(regime=WorkloadSpecRef.BURST, method="single_stage",
                          search_space_size=total, candidates_evaluated=total),
        provenance=Provenance(timestamp_utc="2026-01-01T00:00:00+00:00"),
    )


def _hardware_result(*, candidate_id="c1", status=RuntimeExecutionStatus.RUNTIME_CLEAN_SUCCESS,
                     contention=OccupancyEvidence.CLEAN, mean_tpot=5.0,
                     placement=None) -> HardwareResult:
    return HardwareResult(
        hardware_result_version="1.0", manifest_id="p1", candidate_id=candidate_id,
        execution_status=status,
        observed_placement=tuple(placement or [_rank(host="hostA", gpu=0),
                                               _rank(host="hostA", gpu=1, parallel_rank=1)]),
        workload_realization=WorkloadRealization(requested_regime=WorkloadSpecRef.BURST,
                                                  requested_request_count=8, achieved_request_count=8),
        occupancy_evidence=OccupancyEvidence(contention_status=contention),
        system=SystemInfo(hosts=("hostA",)),
        metrics=HardwareMetrics(ttft=LatencyStats(mean=1.0), tpot=LatencyStats(mean=mean_tpot),
                                e2e=LatencyStats(mean=6.0), request_throughput_rps=40.0),
    )


# --------------------------------------------------------------------------
# workload / regime
# --------------------------------------------------------------------------


def test_workload_spec_ref_regime_has_no_default():
    with pytest.raises(TypeError):
        WorkloadSpecRef(num_requests=8, prefill_tokens=128, decode_tokens=32)  # type: ignore[call-arg]


def test_validate_workload_spec_rejects_unknown_regime():
    bad = WorkloadSpecRef(regime="sometimes", num_requests=8, prefill_tokens=128, decode_tokens=32)
    with pytest.raises(ValidationError):
        validate_workload_spec(bad)


def test_validate_workload_spec_rejects_streaming_without_qps_seed_num_seeds():
    bad = WorkloadSpecRef(regime=WorkloadSpecRef.STREAMING, num_requests=8,
                          prefill_tokens=128, decode_tokens=32)
    with pytest.raises(ValidationError):
        validate_workload_spec(bad)


def test_validate_workload_spec_rejects_infinite_qps():
    bad = WorkloadSpecRef(regime=WorkloadSpecRef.STREAMING, num_requests=8, prefill_tokens=128,
                          decode_tokens=32, qps=float("inf"), seed=0, num_seeds=3)
    with pytest.raises(ValidationError):
        validate_workload_spec(bad)


def test_validate_workload_spec_accepts_well_formed_streaming():
    ok = WorkloadSpecRef(regime=WorkloadSpecRef.STREAMING, num_requests=8, prefill_tokens=128,
                         decode_tokens=32, qps=4.0, seed=0, num_seeds=3)
    validate_workload_spec(ok)  # must not raise


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def test_validate_placement_rejects_duplicate_rank_mapping():
    p = _placement([_rank(host="hostA", gpu=0), _rank(host="hostB", gpu=0)])
    with pytest.raises(ValidationError, match="duplicate rank mapping"):
        validate_placement(p)


def test_validate_placement_rejects_duplicate_host_gpu_assignment():
    p = _placement([_rank(host="hostA", gpu=0, parallel_rank=0),
                    _rank(host="hostA", gpu=0, parallel_rank=1)])
    with pytest.raises(ValidationError, match="duplicate host/GPU assignment"):
        validate_placement(p)


def test_validate_placement_rejects_unknown_host():
    p = _placement([_rank(host="hostZ", gpu=0)], host_map={0: "hostA", 1: "hostB"})
    with pytest.raises(ValidationError, match="unknown host"):
        validate_placement(p)


def test_validate_placement_accepts_well_formed_placement():
    p = _placement([_rank(host="hostA", gpu=0), _rank(host="hostB", gpu=0, parallel_rank=1)])
    validate_placement(p)  # must not raise


# --------------------------------------------------------------------------
# manifest / prediction pairing and versioning
# --------------------------------------------------------------------------


def test_validate_manifest_prediction_pair_rejects_mismatched_plan_id():
    manifest = _minimal_manifest(plan_id="p1")
    prediction = _prediction_for(manifest)
    prediction_wrong = _prediction_for(_minimal_manifest(plan_id="p2"))
    with pytest.raises(ValidationError):
        validate_manifest_prediction_pair(manifest, prediction_wrong)
    validate_manifest_prediction_pair(manifest, prediction)  # must not raise


def test_unknown_schema_major_version_rejected():
    with pytest.raises(SchemaVersionError):
        check_major_version("DeploymentManifest", "2.0", DEPLOYMENT_MANIFEST_VERSION)


def test_minor_version_bump_is_accepted():
    check_major_version("DeploymentManifest", "1.7", DEPLOYMENT_MANIFEST_VERSION)  # must not raise


def test_validate_deployment_manifest_end_to_end():
    manifest = _minimal_manifest()
    validate_deployment_manifest(manifest, expected_version=DEPLOYMENT_MANIFEST_VERSION)


def test_missing_required_field_is_a_hard_reject_not_a_silent_default():
    manifest = _minimal_manifest()
    payload = json.loads(to_json(manifest))
    del payload["plan_id"]
    with pytest.raises(SchemaFieldError):
        from_dict(DeploymentManifest, payload)


# --------------------------------------------------------------------------
# runtime status vs. planner feasibility (S5)
# --------------------------------------------------------------------------


def test_runtime_resource_busy_is_a_valid_runtime_status_not_a_planner_verdict():
    validate_runtime_status_is_not_planner_feasibility(RuntimeExecutionStatus.RUNTIME_RESOURCE_BUSY)


def test_planner_feasibility_constant_rejected_as_a_runtime_status():
    with pytest.raises(ValidationError):
        validate_runtime_status_is_not_planner_feasibility(PlannedFeasibility.PLANNED_INFEASIBLE)


def test_hardware_result_with_resource_busy_status_still_validates_structurally():
    result = _hardware_result(status=RuntimeExecutionStatus.RUNTIME_RESOURCE_BUSY,
                              contention=OccupancyEvidence.RESOURCE_BUSY)
    validate_hardware_result(result, expected_version="1.0")  # a fact, not a rejected payload


# --------------------------------------------------------------------------
# occupancy / contention and HardwareBest eligibility (S11/S14)
# --------------------------------------------------------------------------


def test_contended_result_is_not_eligible_for_hardware_best():
    contended = _hardware_result(contention=OccupancyEvidence.CONTENDED)
    assert not is_eligible_for_hardware_best(contended)


def test_resource_busy_result_is_not_eligible_for_hardware_best():
    busy = _hardware_result(status=RuntimeExecutionStatus.RUNTIME_RESOURCE_BUSY,
                            contention=OccupancyEvidence.RESOURCE_BUSY)
    assert not is_eligible_for_hardware_best(busy)


def test_clean_result_is_eligible_for_hardware_best():
    clean = _hardware_result(contention=OccupancyEvidence.CLEAN)
    assert is_eligible_for_hardware_best(clean)


def test_compute_hardware_best_never_falls_back_to_a_contended_result():
    contended = _hardware_result(candidate_id="c1", contention=OccupancyEvidence.CONTENDED, mean_tpot=3.0)
    assert compute_hardware_best([contended]) is None


def test_compute_hardware_best_picks_the_lowest_mean_tpot_among_eligible_results():
    a = _hardware_result(candidate_id="a", mean_tpot=5.0)
    b = _hardware_result(candidate_id="b", mean_tpot=3.0)
    busy = _hardware_result(candidate_id="c", status=RuntimeExecutionStatus.RUNTIME_RESOURCE_BUSY,
                            contention=OccupancyEvidence.RESOURCE_BUSY, mean_tpot=1.0)
    best = compute_hardware_best([a, b, busy])
    assert best.candidate_id == "b"


# --------------------------------------------------------------------------
# decision validation: top1/topk, regret, resolvability, SLO, throughput, placement
# --------------------------------------------------------------------------


def test_top1_correct_when_planner_selection_matches_hardware_best():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.0)
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.0)
    best = _hardware_result(candidate_id="c1", mean_tpot=5.0)
    dv = compute_decision_validation(prediction, selected, best, noise_floor_source=None)
    assert dv.top1_correct is True
    assert dv.regret_absolute == pytest.approx(0.0)


def test_top1_wrong_but_topk_correct_via_equivalence_group():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.2)
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.2)
    best = _hardware_result(candidate_id="c2", mean_tpot=5.0)
    dv = compute_decision_validation(prediction, selected, best, noise_floor_source=None,
                                     equivalence_group_hardware_ids=("c1", "c2"))
    assert dv.top1_correct is False
    assert dv.topk_correct is True


def test_regret_is_the_gap_between_selected_and_hardware_best():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.5)
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.5)
    best = _hardware_result(candidate_id="c2", mean_tpot=5.0)
    dv = compute_decision_validation(prediction, selected, best, noise_floor_source=None)
    assert dv.regret_absolute == pytest.approx(0.5)
    assert dv.regret_relative == pytest.approx(0.1)


def test_resolvability_is_unknown_without_a_per_configuration_noise_source():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.5)
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.5)
    best = _hardware_result(candidate_id="c2", mean_tpot=5.0)
    dv = compute_decision_validation(prediction, selected, best, noise_floor_source=None)
    assert dv.resolvability.resolvable is None


def test_regret_smaller_than_noise_floor_is_unresolvable():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.05)
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.05)
    best = _hardware_result(candidate_id="c2", mean_tpot=5.0)
    noise = NoiseFloorSource(hardware_config_id="hc1", workload_id="w1", regime="burst",
                             repeats=10, measured_ci95_halfwidth=0.2)
    dv = compute_decision_validation(prediction, selected, best, noise_floor_source=noise)
    assert dv.resolvability.resolvable is False


def test_regret_larger_than_noise_floor_is_resolvable():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=6.0)
    selected = _hardware_result(candidate_id="c1", mean_tpot=6.0)
    best = _hardware_result(candidate_id="c2", mean_tpot=5.0)
    noise = NoiseFloorSource(hardware_config_id="hc1", workload_id="w1", regime="burst",
                             repeats=10, measured_ci95_halfwidth=0.2)
    dv = compute_decision_validation(prediction, selected, best, noise_floor_source=noise)
    assert dv.resolvability.resolvable is True


def test_slo_pass_fail_mismatch_is_reported_not_hidden():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.0)  # predicted.slo_pass=True (helper default)
    selected = _hardware_result(candidate_id="c1", mean_tpot=150.0)  # observed misses a 100ms floor
    dv = compute_decision_validation(prediction, selected, None, noise_floor_source=None,
                                     slo_floor_ms=100.0)
    assert dv.slo.planner_predicted_pass is True
    assert dv.slo.hardware_observed_pass is False


def test_throughput_floor_mismatch_is_reported():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.0)
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.0)
    dv = compute_decision_validation(prediction, selected, None, noise_floor_source=None,
                                     throughput_floor=100.0)  # helper's hardware throughput is 40.0
    assert dv.throughput.floor_pass is False


def test_exact_placement_mismatch_is_detected():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.0)
    requested = [_rank(host="hostA", gpu=0), _rank(host="hostA", gpu=1, parallel_rank=1)]
    observed_elsewhere = [_rank(host="hostB", gpu=0), _rank(host="hostB", gpu=1, parallel_rank=1)]
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.0, placement=observed_elsewhere)
    dv = compute_decision_validation(prediction, selected, None, noise_floor_source=None,
                                     requested_placement=requested)
    assert dv.placement.exact_match is False


def test_exact_placement_match_is_detected():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.0)
    same = [_rank(host="hostA", gpu=0), _rank(host="hostA", gpu=1, parallel_rank=1)]
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.0, placement=same)
    dv = compute_decision_validation(prediction, selected, None, noise_floor_source=None,
                                     requested_placement=same)
    assert dv.placement.exact_match is True


def test_missing_hardware_best_is_reported_not_papered_over():
    manifest = _minimal_manifest(candidate_id="c1")
    prediction = _prediction_for(manifest, objective_value=5.0)
    selected = _hardware_result(candidate_id="c1", mean_tpot=5.0)
    dv = compute_decision_validation(prediction, selected, None, noise_floor_source=None)
    assert dv.hardware_best_candidate is None
    assert dv.top1_correct is None
    assert dv.resolvability.resolvable is None


# --------------------------------------------------------------------------
# round-trip: ties, intervals, provenance
# --------------------------------------------------------------------------


def test_tie_group_preserved_through_json_round_trip():
    manifest = _minimal_manifest()
    prediction = _prediction_for(manifest, indistinguishable=True, rank=1, total=4, group_size=3)
    restored = from_json(PlannerPrediction, to_json(prediction))
    assert restored == prediction
    assert restored.ranking.indistinguishable_from_winner is True
    assert restored.ranking.winner_equivalence_group_size == 3


def test_confidence_interval_preserved_through_json_round_trip():
    manifest = _minimal_manifest()
    prediction = _prediction_for(manifest, ci95_halfwidth=0.35)
    restored = from_json(PlannerPrediction, to_json(prediction))
    assert restored == prediction
    assert restored.uncertainty.ci95_halfwidth == pytest.approx(0.35)


def test_burst_prediction_carries_no_fabricated_interval():
    manifest = _minimal_manifest()
    prediction = _prediction_for(manifest, ci95_halfwidth=None)
    assert prediction.uncertainty.ci95_halfwidth is None
    assert prediction.uncertainty.method == "none_single_burst_run"


def test_profile_provenance_files_commit_and_fix_flags_round_trip():
    prov = ProfileProvenance(
        profile_files=("attention.csv", "linear_op.csv"), device="h800", model="m",
        operator_families_covered=("attention", "linear_op"),
        collection_commit="abc123", phase_filter_applied=True, block_table_fix_applied=False,
        known_limitations=("flat-extrapolation gap not closed",),
    )
    manifest = _minimal_manifest()
    manifest = DeploymentManifest(**{**manifest.__dict__, "profile_provenance": prov})
    restored = from_json(DeploymentManifest, to_json(manifest))
    assert restored.profile_provenance == prov
    assert restored.profile_provenance.phase_filter_applied is True
    assert restored.profile_provenance.block_table_fix_applied is False


def test_placement_mapping_with_int_machine_keys_round_trips():
    manifest = _minimal_manifest()
    restored = from_json(DeploymentManifest, to_json(manifest))
    for key in restored.placement.topology_machine_to_host:
        assert isinstance(key, int)
    assert restored.placement.topology_machine_to_host == manifest.placement.topology_machine_to_host


# --------------------------------------------------------------------------
# real, on-disk named examples (task 56/57's own natural-split boundary)
# --------------------------------------------------------------------------


def _load_example_manifest(name: str) -> DeploymentManifest:
    path = EXAMPLES_DIR / f"{name}_manifest.json"
    if not path.exists():
        pytest.skip(f"{path} not present -- run tools/stage2's own example builder first")
    return from_json(DeploymentManifest, path.read_text())


def test_single_host_tp2_example_validates():
    manifest = _load_example_manifest("single_host_tp2")
    validate_deployment_manifest(manifest, expected_version=DEPLOYMENT_MANIFEST_VERSION)
    assert manifest.parallelism.attn_tp == 2
    assert manifest.parallelism.relative is None
    assert len(manifest.placement.domains_used()) == 1


def test_two_host_tp4_example_validates():
    manifest = _load_example_manifest("two_host_tp4")
    validate_deployment_manifest(manifest, expected_version=DEPLOYMENT_MANIFEST_VERSION)
    assert manifest.parallelism.attn_tp == 4
    assert manifest.parallelism.attn_shape == (2, 2)
    assert len(manifest.placement.domains_used()) == 2


def test_interval_example_carries_a_real_measured_ci95_halfwidth():
    """`planner_prediction_with_interval.json` came from a real
    `Regime(seeded=True, num_seeds=3)` search (two placements of the
    same tp=2/ep=1 candidate, three real Frontier seeds each) --
    `ci95_halfwidth` here is a real number, not synthesized, and this
    checks it survived the file as written, not merely that some
    interval object round-trips (already covered above)."""
    path = EXAMPLES_DIR / "planner_prediction_with_interval.json"
    if not path.exists():
        pytest.skip(f"{path} not present -- run tools/stage2's own example builder first")
    prediction = from_json(PlannerPrediction, path.read_text())
    assert prediction.uncertainty.seed_count == 3
    assert prediction.uncertainty.ci95_halfwidth is not None
    assert prediction.uncertainty.ci95_halfwidth > 0


def _load_example(name: str, cls):
    path = EXAMPLES_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path} not present -- run tools/stage2's own example builder first")
    return from_json(cls, path.read_text())


def test_clean_hardware_result_example_is_eligible_for_hardware_best():
    result = _load_example("clean_hardware_result", HardwareResult)
    validate_hardware_result(result, expected_version="1.0")
    assert is_eligible_for_hardware_best(result)


def test_contended_hardware_result_example_is_not_eligible_for_hardware_best():
    result = _load_example("contended_hardware_result", HardwareResult)
    validate_hardware_result(result, expected_version="1.0")
    assert not is_eligible_for_hardware_best(result)
    assert result.occupancy_evidence.in_run_conflicting_process_evidence


def test_decision_validation_example_matches_its_own_clean_hardware_result():
    from stage2.contracts import DecisionValidation
    dv = _load_example("decision_validation_example", DecisionValidation)
    assert dv.top1_correct is True
    assert dv.resolvability.resolvable is False  # regret 0.0 < the example's own 0.30 noise floor
    assert dv.resolvability.noise_floor_source is not None


def test_attn_whole_a_ffn_whole_b_example_remains_distinct_after_serialization():
    """The regression this exact test exists to catch: task 56 found
    that a domain-blind placement signature collapsed the natural split
    into the colocated arrangement's own key. This manifest's own
    `parallelism.relative` must survive a full JSON round trip as
    `"disjoint"`, and its `placement.domains_used()` must show the two
    real groups on two different hosts -- not merely present in the
    file under some key, matching what task 57's own regression test
    already checks at the `planner_core` layer, now checked again at
    the contract layer."""
    manifest = _load_example_manifest("attn_a_ffn_b")
    validate_deployment_manifest(manifest, expected_version=DEPLOYMENT_MANIFEST_VERSION)
    assert manifest.parallelism.relative == "disjoint"
    restored = from_json(DeploymentManifest, to_json(manifest))
    assert restored.parallelism.relative == "disjoint"

    attn_hosts = {a.host for a in manifest.placement.assignments if a.logical_group == "DECODE_ATTN"}
    ffn_hosts = {a.host for a in manifest.placement.assignments if a.logical_group == "DECODE_FFN"}
    assert attn_hosts, "no DECODE_ATTN assignments in this example"
    assert ffn_hosts, "no DECODE_FFN assignments in this example"
    assert not (attn_hosts & ffn_hosts), "attention and FFN groups share a host in the 'disjoint' example"
