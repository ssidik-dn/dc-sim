"""Stage 2 Gate A: build `DeploymentManifest`/`PlannerPrediction` objects
from this project's own real `tools/planner_core.py` objects.

This module is the one place in `tools/stage2/` allowed to import
`planner_core`/`planner` -- it is the producer side of the contract.
`sim_real` never imports this module, or anything it imports; it only
ever reads the JSON a caller writes with `serialization.to_json`.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .contracts import (
    ConstraintSpec,
    DEPLOYMENT_MANIFEST_VERSION,
    DeploymentManifest,
    HardwareSpecRef,
    InputIdentity,
    ModelSpecRef,
    ObjectiveSpec,
    ParallelismSpec,
    PLANNER_PREDICTION_VERSION,
    PlacementRankAssignment,
    PlacementSpec,
    PlannerPrediction,
    PredictedMetrics,
    ProfileProvenance,
    Provenance,
    RankingSpec,
    RuntimeSpec,
    SearchSpec,
    ThroughputFloor,
    TopologySpecRef,
    UncertaintySpec,
    WorkloadSpecRef,
)


# --------------------------------------------------------------------------
# input_identity components
# --------------------------------------------------------------------------


def topology_id_for(name: str, num_machines: int, gpus_per_machine: int,
                    scale_up_GBps: float, scale_out_GBps: float) -> str:
    """A stable identity string for a topology -- not a serialization of
    the `Fabric` graph itself (this project's own `Fabric` does not
    retain its own construction-time bandwidth parameters as inspectable
    attributes; they are baked into individual `Link` objects), just
    enough to tell whether two manifests were planned against the *same*
    topology definition."""
    raw = f"{name}|{num_machines}|{gpus_per_machine}|{scale_up_GBps}|{scale_out_GBps}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def topology_spec_ref(topology, *, num_machines: int, gpus_per_machine: int,
                      scale_up_GBps: float, scale_out_GBps: float) -> TopologySpecRef:
    """`topology` is a real `planner_core.Topology` -- its own `.name`
    is used for the human-readable label; the four numeric parameters
    are supplied explicitly by the caller (whoever built the topology
    already has them -- this is the S1 "no ambient context" instruction
    applied to the exporter itself, not only to `plan()`'s own inputs)."""
    return TopologySpecRef(
        name=topology.name,
        num_machines=num_machines,
        gpus_per_machine=gpus_per_machine,
        scale_up_GBps=scale_up_GBps,
        scale_out_GBps=scale_out_GBps,
        topology_id=topology_id_for(topology.name, num_machines, gpus_per_machine,
                                    scale_up_GBps, scale_out_GBps),
    )


def hardware_spec_ref(hardware) -> HardwareSpecRef:
    return HardwareSpecRef(device=hardware.device,
                           memory_margin_fraction=hardware.memory_margin_fraction)


def model_spec_ref(model) -> ModelSpecRef:
    return ModelSpecRef(
        model_name=model.model_name, total_experts=model.total_experts,
        router_topk=model.router_topk, is_moe=model.is_moe,
        hidden_size=model.hidden_size, num_attention_heads=model.num_attention_heads,
        num_key_value_heads=model.num_key_value_heads, num_layers=model.num_layers,
        head_dim=model.head_dim, kv_factor=model.kv_factor,
        runtime_num_kv_heads=model.runtime_num_kv_heads,
        runtime_head_dim=model.runtime_head_dim,
    )


def workload_spec_ref(workload, regime, *, workload_identity: Optional[str] = None,
                      seed: Optional[int] = None) -> WorkloadSpecRef:
    """`regime.seeded` (`planner_core.Regime`) maps directly to
    `"streaming"`/`"burst"` -- there is no third option in either
    vocabulary, and none is invented here."""
    is_streaming = bool(regime.seeded)
    return WorkloadSpecRef(
        regime=WorkloadSpecRef.STREAMING if is_streaming else WorkloadSpecRef.BURST,
        num_requests=workload.num_requests,
        prefill_tokens=workload.prefill_tokens,
        decode_tokens=workload.decode_tokens,
        qps=workload.qps if is_streaming else None,
        seed=seed if is_streaming else None,
        num_seeds=regime.num_seeds if is_streaming else None,
        workload_identity=workload_identity,
    )


def constraint_spec(objectives, *, memory_margin_fraction: float,
                    throughput_floor: Optional[ThroughputFloor] = None) -> ConstraintSpec:
    """`objectives.min_throughput_rps` is always absolute in this
    project's own current search (S7's own documented gap) -- the
    caller may pass a `relative_to_baseline` `ThroughputFloor` instead
    to *represent* that mode in the contract, but it does not change
    what `planner_core.Objectives` itself enforces.

    `memory_margin_fraction` is required, not defaulted: it belongs to
    `Hardware`, not `Objectives`, so this function cannot supply it on
    its own without reading a second real object -- the caller (which
    already has both) passes it through explicitly rather than this
    function guessing or leaving it `None` even transiently."""
    if throughput_floor is None:
        throughput_floor = ThroughputFloor(mode=ThroughputFloor.ABSOLUTE,
                                           value=objectives.min_throughput_rps)
    return ConstraintSpec(
        slo_tpot_ms=objectives.slo_tpot_ms,
        slo_attainment_floor=objectives.slo_attainment_floor,
        throughput_floor=throughput_floor,
        memory_margin_fraction=memory_margin_fraction,
    )


def objective_spec(objectives) -> ObjectiveSpec:
    return ObjectiveSpec(minimize=objectives.minimize, direction="minimize")


def parallelism_spec(candidate) -> ParallelismSpec:
    return ParallelismSpec(
        attn_tp=candidate.attn_tp, attn_shape=tuple(candidate.attn_shape),
        ffn_ep=candidate.ffn_ep, ep_shape=tuple(candidate.ep_shape),
        relative=candidate.relative,
        attn_replicas=candidate.attn_replicas, ffn_replicas=candidate.ffn_replicas,
    )


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def placement_spec_from_mapping(mapping, *, topology_machine_to_host: Dict[int, str]) -> PlacementSpec:
    """`mapping` is a real `engine.placement.placement.Placement.mapping`
    (`Dict[Rank, GpuId]`) -- the exact structure task 55/56/57 built and
    read directly throughout. `topology_machine_to_host` is required,
    explicit, and supplied by the caller: the planner's own `GpuId.machine`
    is an abstract domain index, and nothing in `tools/planner_core.py`
    or `src/engine/` has ever known a real fleet hostname (S3's own
    reasoning, restated in `contracts.PlacementSpec`'s own docstring)."""
    unknown = set()
    assignments: List[PlacementRankAssignment] = []
    for rank, gpu in mapping.items():
        host = topology_machine_to_host.get(gpu.machine)
        if host is None:
            unknown.add(gpu.machine)
            continue
        assignments.append(PlacementRankAssignment(
            logical_group=rank.pool, replica=rank.replica, parallel_rank=rank.index,
            host=host, physical_gpu=gpu.index,
        ))
    if unknown:
        raise ValueError(
            f"placement uses machine index/indices {sorted(unknown)} with no entry in "
            f"topology_machine_to_host={topology_machine_to_host!r}")
    assignments.sort(key=lambda a: (a.logical_group, a.replica, a.parallel_rank))
    return PlacementSpec(assignments=tuple(assignments),
                         topology_machine_to_host=dict(topology_machine_to_host))


# --------------------------------------------------------------------------
# top-level exporters
# --------------------------------------------------------------------------


def export_deployment_manifest(
    *,
    plan_id: str,
    candidate_id: str,
    topology, model, workload, hardware, objectives, regime,
    candidate, placement_mapping,
    topology_machine_to_host: Dict[int, str],
    num_machines: int, gpus_per_machine: int, scale_up_GBps: float, scale_out_GBps: float,
    profile_provenance: ProfileProvenance,
    runtime: Optional[RuntimeSpec] = None,
    planner_git_sha: Optional[str] = None,
    simulator_git_sha: Optional[str] = None,
    seed: Optional[int] = None,
    throughput_floor: Optional[ThroughputFloor] = None,
) -> DeploymentManifest:
    """Builds one `DeploymentManifest` from real planner objects --
    `topology`/`model`/`workload`/`hardware`/`objectives`/`regime` are
    the exact `planner_core` dataclasses `plan()` itself takes;
    `candidate` is a real `Candidate`; `placement_mapping` is a real
    `Placement.mapping`."""
    topo_ref = topology_spec_ref(topology, num_machines=num_machines,
                                 gpus_per_machine=gpus_per_machine,
                                 scale_up_GBps=scale_up_GBps, scale_out_GBps=scale_out_GBps)
    constraints = constraint_spec(objectives, memory_margin_fraction=hardware.memory_margin_fraction,
                                  throughput_floor=throughput_floor)
    workload_ref = workload_spec_ref(workload, regime, seed=seed)
    return DeploymentManifest(
        manifest_version=DEPLOYMENT_MANIFEST_VERSION,
        plan_id=plan_id, candidate_id=candidate_id,
        input_identity=InputIdentity(
            topology=topo_ref, hardware=hardware_spec_ref(hardware),
            model=model_spec_ref(model), workload=workload_ref,
            constraints=constraints, objective=objective_spec(objectives),
        ),
        parallelism=parallelism_spec(candidate),
        placement=placement_spec_from_mapping(
            placement_mapping, topology_machine_to_host=topology_machine_to_host),
        runtime=runtime if runtime is not None else RuntimeSpec(),
        workload=workload_ref,
        constraints=constraints,
        profile_provenance=profile_provenance,
        provenance=Provenance(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            planner_git_sha=planner_git_sha, simulator_git_sha=simulator_git_sha,
            topology_id=topo_ref.topology_id, seed=seed,
        ),
    )


def export_planner_prediction(
    *,
    plan_id: str,
    manifest: DeploymentManifest,
    evaluated_row: dict,
    ranked_rows: List[dict],
    minimize: str,
    method: str,
    search_space_size: int,
    shortlist_size: Optional[int] = None,
    planner_git_sha: Optional[str] = None,
    simulator_git_sha: Optional[str] = None,
) -> PlannerPrediction:
    """Builds one `PlannerPrediction` from a real `plan()`/`plan_two_stage()`
    result row (`evaluated_row`, one dict from `PlanResult.ranked` /
    `TwoStagePlanResult.ranked`) and the full `ranked_rows` list it came
    from (needed to compute `rank` and `winner_equivalence_group_size`
    -- both properties of the *list*, not of one row in isolation).

    Every field in `PredictedMetrics`/`UncertaintySpec` is read directly
    from `evaluated_row`; none is recomputed or approximated."""
    candidate = evaluated_row["candidate"]
    rank = next(i for i, r in enumerate(ranked_rows) if r is evaluated_row)
    winner_group_size = sum(
        1 for r in ranked_rows
        if r.get("indistinguishable_from_winner") or r is ranked_rows[0]
    ) if ranked_rows else 0

    ci = evaluated_row.get("ci95_halfwidth")
    mean = evaluated_row[minimize]
    slo_attainment = evaluated_row.get("slo_attainment")
    slo_floor = manifest.constraints.slo_attainment_floor
    predicted = PredictedMetrics(
        objective_value=mean,
        mean_tpot_ms=evaluated_row.get("mean_tpot_ms"),
        throughput_rps=evaluated_row.get("throughput_rps"),
        slo_attainment=slo_attainment,
        # Matches plan()'s own constraint check exactly (planner_core.plan:
        # "r['slo_attainment'] < objectives.slo_attainment_floor - 1e-9" is a
        # rejection) -- not a hardcoded ">= 1.0", since slo_attainment_floor
        # is frequently 0.0 ("SLO reported, not constrained"), where every
        # candidate that reached evaluation already passes trivially.
        slo_pass=(slo_attainment >= slo_floor - 1e-9) if slo_attainment is not None else None,
        # Not carried by evaluate()'s own result dict today -- see
        # PredictedMetrics's own docstring; left None rather than guessed.
        memory_bytes=None, communication_ns=None, compute_ns=None,
    )
    uncertainty = UncertaintySpec(
        seed_count=evaluated_row.get("n_seeds", 1),
        method=("student_t_95_on_seeded_mean" if ci is not None else "none_single_burst_run"),
        ci95_halfwidth=ci,
        interval_low=(mean - ci) if ci is not None else None,
        interval_high=(mean + ci) if ci is not None else None,
    )
    ranking = RankingSpec(
        rank=rank, total_candidates_ranked=len(ranked_rows),
        indistinguishable_from_winner=bool(evaluated_row.get("indistinguishable_from_winner", False)),
        winner_equivalence_group_size=winner_group_size,
    )
    search = SearchSpec(
        regime=manifest.workload.regime, method=method,
        search_space_size=search_space_size, candidates_evaluated=len(ranked_rows),
        shortlist_size=shortlist_size,
    )
    return PlannerPrediction(
        planner_prediction_version=PLANNER_PREDICTION_VERSION,
        plan_id=plan_id, selected_manifest_id=manifest.plan_id, candidate_id=manifest.candidate_id,
        predicted=predicted, uncertainty=uncertainty, ranking=ranking, search=search,
        provenance=Provenance(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            planner_git_sha=planner_git_sha, simulator_git_sha=simulator_git_sha,
            topology_id=manifest.provenance.topology_id, seed=manifest.provenance.seed,
        ),
    )
