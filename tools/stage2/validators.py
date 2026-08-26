"""Stage 2 Gate A: structural validation for the four contract objects.

Every function here either returns `None` (valid) or raises
`ValidationError` with a specific reason -- never returns a bare `False`,
since "what exactly is wrong" is what a caller (a test, or `sim_real`'s
own pre-flight check) actually needs to act on.
"""
from __future__ import annotations

from .contracts import (
    DeploymentManifest,
    HardwareResult,
    OccupancyEvidence,
    PlacementSpec,
    PlannerPrediction,
    RuntimeExecutionStatus,
    WorkloadSpecRef,
)
from .serialization import check_major_version, SchemaVersionError  # noqa: F401  (re-exported)


class ValidationError(ValueError):
    pass


def validate_workload_spec(workload: WorkloadSpecRef) -> None:
    """S6: regime is required (the dataclass itself enforces this --
    there is no default -- so a missing regime is already a
    `TypeError`/`SchemaFieldError` at construction; this function checks
    the *value*, not just its presence) and streaming requires QPS, a
    seed, and a seed count -- burst must not smuggle in `qps=inf`
    (S6's own "not qps=infinity")."""
    if workload.regime not in (WorkloadSpecRef.BURST, WorkloadSpecRef.STREAMING):
        raise ValidationError(
            f"WorkloadSpecRef.regime must be 'burst' or 'streaming', got {workload.regime!r}")
    if workload.regime == WorkloadSpecRef.STREAMING:
        if workload.qps is None or workload.seed is None or workload.num_seeds is None:
            raise ValidationError(
                "streaming regime requires qps, seed, and num_seeds to all be set "
                f"(got qps={workload.qps!r}, seed={workload.seed!r}, "
                f"num_seeds={workload.num_seeds!r})")
        if workload.qps == float("inf"):
            raise ValidationError("streaming regime must not use qps=infinity")
    if workload.regime == WorkloadSpecRef.BURST and workload.qps not in (None,):
        # Burst may still carry a qps value for provenance (e.g. "what the
        # streaming-equivalent qps would have been"), but it must not be
        # infinite -- the one thing S6 explicitly forbids.
        if workload.qps == float("inf"):
            raise ValidationError("burst regime must not encode qps=infinity")


def validate_placement(placement: PlacementSpec) -> None:
    """Duplicate rank mapping, duplicate host/GPU assignment, and
    unknown-host checks -- the three structural placement failures this
    task's own S23 test list names by name."""
    seen_ranks = set()
    seen_gpus = set()
    known_hosts = set(placement.topology_machine_to_host.values())
    for a in placement.assignments:
        rank_key = (a.logical_group, a.replica, a.parallel_rank)
        if rank_key in seen_ranks:
            raise ValidationError(f"duplicate rank mapping: {rank_key} assigned more than once")
        seen_ranks.add(rank_key)

        gpu_key = (a.host, a.physical_gpu)
        if gpu_key in seen_gpus:
            raise ValidationError(
                f"duplicate host/GPU assignment: {a.host}:{a.physical_gpu} assigned to "
                "more than one rank -- placement is not injective")
        seen_gpus.add(gpu_key)

        if known_hosts and a.host not in known_hosts:
            raise ValidationError(
                f"unknown host {a.host!r} -- not present in "
                f"topology_machine_to_host's own values {sorted(known_hosts)}")


def validate_workload_and_constraints_consistency(manifest: DeploymentManifest) -> None:
    """Stage 2 Gate A.1: `DeploymentManifest` deliberately carries
    `workload`/`constraints` twice -- once at the top level, once inside
    `input_identity` (`docs/stage-2-gate-a-contract-report.md` §3's own
    "redundancy with a stated purpose"). A manifest is only ever
    produced by one exporter call in this project (`export_deployment_manifest`
    builds both copies from the same real objects in one pass), so the
    two copies cannot diverge *from this project's own producer*. But
    `sim_real` receives JSON, not the in-memory Python objects that
    guaranteed that -- nothing stops a hand-edited or corrupted file
    from carrying two different answers to "what workload is this,"
    and a consumer that silently picked one copy over the other would
    make that divergence invisible instead of a hard failure.

    Equality here is **semantic** (Python's own dataclass `__eq__`,
    comparing every field by value, recursively through `ThroughputFloor`),
    not object identity -- two separately-constructed but field-identical
    `WorkloadSpecRef`/`ConstraintSpec` instances must pass; the two
    copies literally cannot be the same object once a manifest has gone
    through a JSON round trip regardless (`from_dict` builds two fresh
    instances from a `dict`), so an identity check here would always
    fail and would not be checking the thing that matters."""
    if manifest.workload != manifest.input_identity.workload:
        raise ValidationError(
            f"DeploymentManifest.workload != input_identity.workload -- two sources of "
            f"truth disagree: {manifest.workload!r} vs. {manifest.input_identity.workload!r}")
    if manifest.constraints != manifest.input_identity.constraints:
        raise ValidationError(
            f"DeploymentManifest.constraints != input_identity.constraints -- two sources "
            f"of truth disagree: {manifest.constraints!r} vs. {manifest.input_identity.constraints!r}")


def validate_deployment_manifest(manifest: DeploymentManifest, *, expected_version: str) -> None:
    check_major_version("DeploymentManifest", manifest.manifest_version, expected_version)
    validate_workload_spec(manifest.workload)
    validate_placement(manifest.placement)
    validate_workload_and_constraints_consistency(manifest)
    if not manifest.plan_id:
        raise ValidationError("DeploymentManifest.plan_id must not be empty")
    if not manifest.candidate_id:
        raise ValidationError("DeploymentManifest.candidate_id must not be empty")


def validate_manifest_prediction_pair(
    manifest: DeploymentManifest, prediction: PlannerPrediction,
) -> None:
    """S23: "mismatched manifest/prediction plan_id rejected" -- a
    `PlannerPrediction` that claims to describe a different plan than
    the manifest it travels with is a corrupted pair, not a warning."""
    if manifest.plan_id != prediction.plan_id:
        raise ValidationError(
            f"plan_id mismatch: manifest.plan_id={manifest.plan_id!r} != "
            f"prediction.plan_id={prediction.plan_id!r}")
    if manifest.candidate_id != prediction.candidate_id:
        raise ValidationError(
            f"candidate_id mismatch: manifest.candidate_id={manifest.candidate_id!r} != "
            f"prediction.candidate_id={prediction.candidate_id!r}")
    if prediction.selected_manifest_id != manifest.plan_id:
        raise ValidationError(
            f"prediction.selected_manifest_id={prediction.selected_manifest_id!r} does "
            f"not reference manifest.plan_id={manifest.plan_id!r}")


def validate_runtime_status_is_not_planner_feasibility(status: str) -> None:
    """S5's own load-bearing distinction, enforced as code rather than
    only as a naming convention: a runtime status must be one of the
    `RUNTIME_*` constants, never one of the `PLANNED_*` ones -- a
    resource-busy launch must never be recorded using a planner
    feasibility vocabulary word, which would make it look like a
    property of the *request* rather than of *this attempt, right
    now*."""
    from .contracts import ALL_RUNTIME_EXECUTION_STATUSES
    if status not in ALL_RUNTIME_EXECUTION_STATUSES:
        raise ValidationError(
            f"{status!r} is not a valid RuntimeExecutionStatus -- if this is a "
            "planner-side feasibility verdict (PLANNED_FEASIBLE / "
            "PLANNED_INFEASIBLE / ...), it belongs on the planner side, not "
            "on a HardwareResult.")


def validate_hardware_result(result: HardwareResult, *, expected_version: str) -> None:
    check_major_version("HardwareResult", result.hardware_result_version, expected_version)
    validate_runtime_status_is_not_planner_feasibility(result.execution_status)
    if result.occupancy_evidence.contention_status not in (
        OccupancyEvidence.CLEAN, OccupancyEvidence.RESOURCE_BUSY,
        OccupancyEvidence.CONTENDED, OccupancyEvidence.UNKNOWN,
    ):
        raise ValidationError(
            f"unknown contention_status {result.occupancy_evidence.contention_status!r}")


def is_eligible_for_hardware_best(result: HardwareResult) -> bool:
    """S11/S14: a contended run, or one the launcher refused outright
    for resource-busy reasons, is not a clean candidate for
    `HardwareBest` unless explicitly normalized/accepted elsewhere --
    Gate A's own rule (S11: "document the rule") is the strict one:
    excluded, not merely flagged. `decision.py`'s own `compute_hardware_best`
    calls this before considering any result."""
    if result.execution_status not in (
        RuntimeExecutionStatus.RUNTIME_CLEAN_SUCCESS,
        RuntimeExecutionStatus.RUNTIME_SUCCESS,
    ):
        return False
    if result.occupancy_evidence.contention_status != OccupancyEvidence.CLEAN:
        return False
    return True
