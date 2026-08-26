"""Stage 2 Gate A: the four contract objects, as plain dataclasses.

Every object here is pure data -- no Frontier import, no `src/integration/`
import, no filesystem access. `tools/stage2/exporters.py` builds these from
real `tools/planner_core.py` objects; `tools/stage2/serialization.py`
converts them to and from JSON; `tools/stage2/validators.py` checks them;
`tools/stage2/decision.py` compares a `PlannerPrediction` against one or
more `HardwareResult`s.

**Reuse, not reinvention.** `TopologySpecRef`/`HardwareSpecRef`/
`ModelSpecRef`/`WorkloadSpecRef` are lightweight, JSON-serializable
*references* to the real `Topology`/`Hardware`/`ModelSpec`/`Workload`
dataclasses `tools/planner_core.py` already defines and every task since
32 already uses -- not a second, competing definition. They exist because
those originals are not directly JSON-safe (`Topology` wraps a real
`Fabric` graph; the others are already flat, but need to travel inside a
manifest that also carries fields the originals don't have, like an
explicit `regime`).

**Regime has no default anywhere in this file.** `WorkloadSpecRef.regime`
is a required field with no default value; a manifest built without one
is not constructible. This mirrors `tools/planner_core.py`'s own
`Regime` (task 45), which raises rather than lets a caller inherit burst
by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Schema versions -- one constant per top-level object, per this task's own
# S19. A major-version bump means "reject," not "migrate" (S19: "do not
# over-engineer migrations").
# --------------------------------------------------------------------------

DEPLOYMENT_MANIFEST_VERSION = "1.0"
PLANNER_PREDICTION_VERSION = "1.0"
HARDWARE_RESULT_VERSION = "1.0"
DECISION_VALIDATION_VERSION = "1.0"


# --------------------------------------------------------------------------
# Input identity -- TopologySpec / HardwareSpec / ModelSpec / WorkloadSpec /
# ConstraintSpec / ObjectiveSpec (this task's own S1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologySpecRef:
    """A JSON-safe identity for a `planner_core.Topology` -- enough to
    tell whether two manifests were planned against the same fabric,
    not a serialization of the fabric graph itself (which `contracts/`
    has no business reconstructing; `sim_real` does not need the graph,
    only where each rank goes -- see `PlacementSpec`)."""
    name: str
    num_machines: int
    gpus_per_machine: int
    scale_up_GBps: float
    scale_out_GBps: float
    topology_id: str  # stable identity string; see serialization.topology_id_for


@dataclass(frozen=True)
class HardwareSpecRef:
    """Mirrors `planner_core.Hardware` exactly (device, memory margin) --
    already flat and JSON-safe; wrapped here only so `input_identity`
    has one uniform shape across all six spec kinds."""
    device: str
    memory_margin_fraction: float


@dataclass(frozen=True)
class ModelSpecRef:
    """Mirrors `planner_core.ModelSpec`'s own fields that determine a
    model's identity and memory/compute behaviour. Deliberately excludes
    `admissible_tp`/`admissible_ep`/`profiled_tp` -- those are properties
    of *what the search is allowed to try*, not of the model itself, and
    belong in `SearchSpec` (`PlannerPrediction`), not here."""
    model_name: str
    total_experts: int
    router_topk: int
    is_moe: bool
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_layers: int
    head_dim: Optional[int] = None
    kv_factor: Optional[int] = None
    runtime_num_kv_heads: Optional[int] = None
    runtime_head_dim: Optional[int] = None


@dataclass(frozen=True)
class WorkloadSpecRef:
    """`regime` has no default (S6) -- constructing this without one is
    a `TypeError` from the dataclass itself, and `validators.py` checks
    it again explicitly so a JSON payload missing it is also rejected,
    not just a Python call site.

    `qps`/`seed`/`num_seeds` are required whenever `regime == "streaming"`
    (checked in `validators.validate_workload_spec`, not by the type
    system alone) -- S6's own "not qps=infinity" for burst, and "requires
    QPS/seed/request count" for streaming."""
    regime: str  # "burst" | "streaming" -- required, no default
    num_requests: int
    prefill_tokens: int
    decode_tokens: int
    qps: Optional[float] = None
    seed: Optional[int] = None
    num_seeds: Optional[int] = None
    workload_identity: Optional[str] = None

    BURST = "burst"
    STREAMING = "streaming"


@dataclass(frozen=True)
class ThroughputFloor:
    """S7: the floor must support absolute and relative-to-baseline
    modes, because a fixed absolute floor invalidated every candidate
    when the workload changed. **`mode="relative_to_baseline"` is
    representable here but not yet backed by `tools/planner_core.py`'s
    own search** -- `Objectives.min_throughput_rps` (the only throughput
    constraint the real search implements today) is absolute only. This
    is a real gap, reported in the S19 report rather than closed here,
    per this task's own "report the mismatch rather than silently
    changing it.\""""
    mode: str  # "absolute" | "relative_to_baseline"
    value: float
    baseline_candidate_id: Optional[str] = None

    ABSOLUTE = "absolute"
    RELATIVE_TO_BASELINE = "relative_to_baseline"


@dataclass(frozen=True)
class ConstraintSpec:
    """The constraint half of `planner_core.Objectives` -- SLO,
    throughput floor, and memory margin -- split out from the objective
    half (`ObjectiveSpec`) because S7 asks for them as separate concerns,
    even though `Objectives` itself bundles them into one dataclass
    (unchanged; this is a contract-layer decomposition, not a planner
    change)."""
    slo_tpot_ms: float
    slo_attainment_floor: float
    throughput_floor: ThroughputFloor
    memory_margin_fraction: float
    availability_assumptions: Optional[str] = None


@dataclass(frozen=True)
class ObjectiveSpec:
    """The objective half of `planner_core.Objectives`. `minimize` names
    the field `plan()` sorts `evaluated` rows by (today, always
    `"mean_tpot_ms"` -- checked in exporters.py against the live
    `Objectives.minimize` value, not assumed)."""
    minimize: str
    direction: str = "minimize"


@dataclass(frozen=True)
class InputIdentity:
    """The six explicit inputs this task's own S1 requires -- no ambient
    fabric, model, workload, or regime. Every `DeploymentManifest`
    carries its own copy; two manifests can be compared for "was this
    actually the same question" without any side-channel assumption."""
    topology: TopologySpecRef
    hardware: HardwareSpecRef
    model: ModelSpecRef
    workload: WorkloadSpecRef
    constraints: ConstraintSpec
    objective: ObjectiveSpec


# --------------------------------------------------------------------------
# Parallelism and placement (this task's own S3-S4; task 44/56/57)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelismSpec:
    """Mirrors `planner_core.Candidate`'s own parallelism fields exactly,
    including `relative` (task 57) -- the field that distinguishes
    "attention whole on one domain, experts whole on another" from
    colocation. Carrying `relative` here, not just `attn_shape`/`ep_shape`,
    is what keeps this spec able to express what task 56 found the old,
    two-component key could not (see `exporters.py`'s own round-trip
    test and this task's own S4/S11.B)."""
    attn_tp: int
    attn_shape: Tuple[int, ...]
    ffn_ep: int
    ep_shape: Tuple[int, ...]
    relative: Optional[str]  # "same" | "disjoint" | "overlapping" | None
    attn_replicas: int
    ffn_replicas: int


@dataclass(frozen=True)
class PlacementRankAssignment:
    """One logical rank's own physical location -- the leaf of
    `logical group -> replica -> parallel rank -> host -> physical GPU`
    this task's own S3 asks for. `host` is a *real* host identifier
    (e.g. `"xai-3"`), not the planner's own abstract machine index --
    see `PlacementSpec.topology_machine_to_host` for why that binding is
    a separate, explicit field rather than folded in here silently."""
    logical_group: str  # "PREFILL" | "DECODE_ATTN" | "DECODE_FFN"
    replica: int
    parallel_rank: int
    host: str
    physical_gpu: int


@dataclass(frozen=True)
class PlacementSpec:
    """The explicit rank-to-GPU map this manifest commits to, plus the
    binding that produced it. `topology_machine_to_host` is carried
    separately from the assignments themselves because the planner's own
    `Topology`/`Fabric` only knows abstract machine indices (`GpuId.machine`,
    an integer) -- it has no notion of a real fleet hostname at all,
    and cannot supply one. Something *outside* the planner (whoever
    calls the exporter for a specific real run) must supply this binding
    explicitly; leaving it implicit would be exactly the "ambient
    context" this task's own S1 forbids."""
    assignments: Tuple[PlacementRankAssignment, ...]
    topology_machine_to_host: Dict[int, str]

    def domains_used(self) -> Dict[str, Tuple[int, ...]]:
        """Which physical GPUs, grouped by real host, this placement
        actually uses -- the same "which domains does this arrangement
        touch" question task 56/57 asked of the planner's own abstract
        `Placement.domains_spanned()`, now answerable against real host
        identifiers instead of abstract machine indices."""
        by_host: Dict[str, List[int]] = {}
        for a in self.assignments:
            by_host.setdefault(a.host, []).append(a.physical_gpu)
        return {h: tuple(sorted(g)) for h, g in by_host.items()}


@dataclass(frozen=True)
class RuntimeSpec:
    """Engine/runtime facts a real launch needs that a simulation never
    did -- every field optional, since a manifest destined only for
    further simulation (not real hardware) may not have any of them yet."""
    engine: Optional[str] = None            # e.g. "vllm"
    engine_version: Optional[str] = None
    model_revision: Optional[str] = None
    precision: Optional[str] = None
    quantization: Optional[str] = None
    transport: Optional[str] = None         # "tcp" | "rdma" -- tcp is the
                                            # known-good reference (finding 7)
    decode_ffn_scheduler: Optional[str] = None  # e.g. "orca" | "vllm_v1" --
        # tools/planner.py's own _argv/_run_scenario/SimulationEvaluator
        # all thread a decode_ffn_scheduler value through every real
        # evaluation (default "orca"; run_topology_scheduler_study.py's
        # own comparison varies it) -- added here, additively, rather than
        # reported as a gap: this axis is real and already searchable,
        # it was simply missing a place to travel in the contract.


# --------------------------------------------------------------------------
# Profile provenance (this task's own S9; tasks 52-53)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileProvenance:
    """File-level profile provenance -- "profile version" alone cannot
    tell two planner results apart when the actual defect was at file
    granularity (task 52's phase contamination, task 53's block-table
    aliasing and the still-open flat-extrapolation gap). Every boolean
    flag here is `Optional[bool]`: `True`/`False` only when this
    manifest's own exporter could actually confirm which code path ran
    (task 53's own patches are opt-in, source-hash-guarded, and off by
    default -- `tools/stage2/exporters.py` reads whether they were
    actually installed for this run, not assumed)."""
    profile_files: Tuple[str, ...]
    device: str
    model: str
    operator_families_covered: Tuple[str, ...]
    collection_commit: Optional[str] = None
    profile_generation_version: Optional[str] = None
    phase_filter_applied: Optional[bool] = None       # task 53 Fix A
    block_table_fix_applied: Optional[bool] = None    # task 53 Fix B
    grid_bounds: Optional[Dict[str, Any]] = None
    known_limitations: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Provenance:
    """General, non-profile provenance -- present on every one of the
    four top-level objects (this task's own S20: "no unlocated headline
    numbers")."""
    timestamp_utc: str
    planner_git_sha: Optional[str] = None
    simulator_git_sha: Optional[str] = None  # Frontier's own commit
    topology_id: Optional[str] = None
    seed: Optional[int] = None


# --------------------------------------------------------------------------
# 1. DeploymentManifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DeploymentManifest:
    """What to run, exactly, and nothing the runtime needs to interpret.
    Every field here has a stated owner (the planner/exporter) and a
    stated consumer (`sim_real`, reading only this file) -- no
    decorative fields (this task's own S3 instruction), checked by
    `docs/stage-2-gate-a-contract-report.md`'s own field-by-field table."""
    manifest_version: str
    plan_id: str
    candidate_id: str
    input_identity: InputIdentity
    parallelism: ParallelismSpec
    placement: PlacementSpec
    runtime: RuntimeSpec
    workload: WorkloadSpecRef
    constraints: ConstraintSpec
    profile_provenance: ProfileProvenance
    provenance: Provenance


# --------------------------------------------------------------------------
# 2. PlannerPrediction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedMetrics:
    """Mirrors the fields `SimulationEvaluator.evaluate`'s own result
    dict actually returns (`tools/planner.py`) -- `memory_bytes`/
    `communication_ns`/`compute_ns` are `Optional` because the real
    per-candidate result dict does not carry a memory/communication/
    compute breakdown today (checked directly, not assumed): memory
    feasibility is checked once, up front, by `feasible_num_blocks`,
    separately from the priced result, and no component ledger is
    attached to `evaluate()`'s own return value. `"if available"`
    (this task's own S8) is honoured literally -- these three fields
    are `None` for every real prediction this project's search
    currently produces, and the exporter does not fabricate a value to
    fill them."""
    objective_value: float
    mean_tpot_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    slo_attainment: Optional[float] = None
    slo_pass: Optional[bool] = None
    memory_bytes: Optional[int] = None
    communication_ns: Optional[float] = None
    compute_ns: Optional[float] = None


@dataclass(frozen=True)
class UncertaintySpec:
    """Carries exactly what `Regime`/`compute_interval_stats` already
    compute (task 45) -- never invents a confidence figure for a
    `Regime(num_seeds=1)` burst prediction, where `ci95_halfwidth` is
    genuinely `None` in the source data, not merely unreported."""
    seed_count: int
    method: str  # e.g. "student_t_95_on_seeded_mean" | "none_single_burst_run"
    ci95_halfwidth: Optional[float] = None
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None


@dataclass(frozen=True)
class RankingSpec:
    """`rank` is the position in `plan()`'s own sorted-by-objective
    list -- an ordering the search really does compute. What it must
    not be read as is a claim of statistical distinguishability at
    every adjacent pair; `indistinguishable_from_winner` is the one
    tie-relevant fact `plan()` itself computes (task 45's own
    `_mark_indistinguishable_from_winner`), and it is *winner-relative*
    only. This project's own search does not currently compute a full
    pairwise tie/equivalence partition across every candidate (e.g.
    "is rank 5 indistinguishable from rank 6, neither being the
    winner") -- `winner_equivalence_group_size` is a direct count of
    how many ranked rows share `indistinguishable_from_winner=True`
    (including the winner itself), not an invented broader grouping.
    This is the honest boundary this task's own S8 ("Do NOT serialize a
    fake total ordering") is read against: the ordering is real: the
    tie information is exactly as complete as `plan()` itself makes it,
    named precisely rather than padded out."""
    rank: int
    total_candidates_ranked: int
    indistinguishable_from_winner: bool
    winner_equivalence_group_size: int


@dataclass(frozen=True)
class SearchSpec:
    """What kind of search produced this prediction, and how big the
    space was -- `method` is `"single_stage"` (`plan()`) or
    `"two_stage"` (`plan_two_stage()`, task 45 Part B); `shortlist_size`
    is `None` for a single-stage result, since there is no shortlist
    step to size."""
    regime: str
    method: str  # "single_stage" | "two_stage"
    search_space_size: int
    candidates_evaluated: int
    shortlist_size: Optional[int] = None


@dataclass(frozen=True)
class PlannerPrediction:
    """What the planner predicted for the selected candidate, with the
    uncertainty and ranking context `tools/planner_core.py` already
    computes -- never stripped down to a single number before crossing
    the file boundary."""
    planner_prediction_version: str
    plan_id: str
    selected_manifest_id: str
    candidate_id: str
    predicted: PredictedMetrics
    uncertainty: UncertaintySpec
    ranking: RankingSpec
    search: SearchSpec
    provenance: Provenance


# --------------------------------------------------------------------------
# Feasibility vs. runtime precondition statuses (this task's own S5)
# --------------------------------------------------------------------------


class PlannedFeasibility:
    """Planner-side feasibility -- a property of the *request*, computed
    before any hardware is touched. Mirrors `planner_core.py`'s own
    `Rejection`/`Unknown`/`Inadmissible` split, named for this contract."""
    PLANNED_FEASIBLE = "PLANNED_FEASIBLE"
    PLANNED_INFEASIBLE = "PLANNED_INFEASIBLE"       # memory, at this margin
    PLANNED_INADMISSIBLE = "PLANNED_INADMISSIBLE"   # divisibility / lane assignment
    PLANNED_UNKNOWN = "PLANNED_UNKNOWN"             # evaluator.can_evaluate() == False


class RuntimeExecutionStatus:
    """Real-runtime outcomes -- a property of *this specific attempt*,
    on *this specific hardware*, *right now*. A `RUNTIME_RESOURCE_BUSY`
    result says nothing about whether the plan itself was feasible
    (S5's own load-bearing distinction; a busy GPU is a fact about the
    fleet at that moment, not about the candidate)."""
    RUNTIME_CLEAN_SUCCESS = "RUNTIME_CLEAN_SUCCESS"        # ran, and occupancy evidence is clean
    RUNTIME_SUCCESS = "RUNTIME_SUCCESS"                    # ran to completion; contention status separate
    RUNTIME_CONTENDED = "RUNTIME_CONTENDED"                # ran, but occupancy evidence shows contention
    RUNTIME_RESOURCE_BUSY = "RUNTIME_RESOURCE_BUSY"        # launcher refused: requested GPUs occupied
    RUNTIME_UNSUPPORTED_PLACEMENT = "RUNTIME_UNSUPPORTED_PLACEMENT"  # this exact placement cannot be realized
    RUNTIME_RUNTIME_MISMATCH = "RUNTIME_RUNTIME_MISMATCH"  # engine/model/revision did not match the manifest
    RUNTIME_STARTUP_FAILURE = "RUNTIME_STARTUP_FAILURE"    # e.g. port collision, image missing
    RUNTIME_WORKLOAD_FAILURE = "RUNTIME_WORKLOAD_FAILURE"  # launched, but the workload itself failed


ALL_RUNTIME_EXECUTION_STATUSES = (
    RuntimeExecutionStatus.RUNTIME_CLEAN_SUCCESS,
    RuntimeExecutionStatus.RUNTIME_SUCCESS,
    RuntimeExecutionStatus.RUNTIME_CONTENDED,
    RuntimeExecutionStatus.RUNTIME_RESOURCE_BUSY,
    RuntimeExecutionStatus.RUNTIME_UNSUPPORTED_PLACEMENT,
    RuntimeExecutionStatus.RUNTIME_RUNTIME_MISMATCH,
    RuntimeExecutionStatus.RUNTIME_STARTUP_FAILURE,
    RuntimeExecutionStatus.RUNTIME_WORKLOAD_FAILURE,
)

# Statuses S11 says must never be silently treated as a clean benchmark.
CONTENTION_AFFECTED_STATUSES = (
    RuntimeExecutionStatus.RUNTIME_CONTENDED,
    RuntimeExecutionStatus.RUNTIME_RESOURCE_BUSY,
)


# --------------------------------------------------------------------------
# 3. HardwareResult
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadRealization:
    requested_regime: str
    requested_request_count: int
    achieved_request_count: int
    requested_qps: Optional[float] = None
    achieved_qps: Optional[float] = None
    scheduling_lag_ms: Optional[float] = None


@dataclass(frozen=True)
class LatencyStats:
    mean: Optional[float] = None
    median: Optional[float] = None
    p95: Optional[float] = None


@dataclass(frozen=True)
class HardwareMetrics:
    ttft: LatencyStats
    tpot: LatencyStats
    e2e: LatencyStats
    request_throughput_rps: Optional[float] = None
    token_throughput_tps: Optional[float] = None
    slo_attainment: Optional[float] = None


@dataclass(frozen=True)
class MemoryObservation:
    per_host_bytes: Dict[str, int] = field(default_factory=dict)
    per_gpu_bytes: Dict[str, int] = field(default_factory=dict)  # key "{host}:{gpu_index}"
    peak_bytes: Optional[int] = None


@dataclass(frozen=True)
class OccupancyEvidence:
    """S11: occupancy/contention is first-class, not an afterthought on
    `HardwareResult`. `contention_status` is the field
    `DecisionValidation` reads to decide whether this result may ever
    become `HardwareBest` (see `decision.py`)."""
    contention_status: str  # "clean" | "resource_busy" | "contended" | "unknown"
    pre_run_occupancy: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    post_run_occupancy: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    in_run_conflicting_process_evidence: Tuple[str, ...] = field(default_factory=tuple)
    exclusive_run: Optional[bool] = None
    contention_evidence: Tuple[str, ...] = field(default_factory=tuple)

    CLEAN = "clean"
    RESOURCE_BUSY = "resource_busy"
    CONTENDED = "contended"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SystemInfo:
    hosts: Tuple[str, ...]
    gpu_identities: Tuple[str, ...] = field(default_factory=tuple)
    runtime_version: Optional[str] = None
    image_digest: Optional[str] = None


@dataclass(frozen=True)
class HardwareResult:
    """What `sim_real` produced by actually executing a
    `DeploymentManifest`. Every field here is something `sim_real` can
    itself observe -- no simulator-internal component (compute/comm
    breakdowns, contention-model bottleneck attribution) is required,
    since a real launcher has no way to produce one (this task's own
    S10 instruction)."""
    hardware_result_version: str
    manifest_id: str
    candidate_id: str
    execution_status: str
    observed_placement: Tuple[PlacementRankAssignment, ...]
    workload_realization: WorkloadRealization
    occupancy_evidence: OccupancyEvidence
    system: SystemInfo
    observed_runtime_identity: Optional[RuntimeSpec] = None
    observed_model_identity: Optional[str] = None
    metrics: Optional[HardwareMetrics] = None
    memory: Optional[MemoryObservation] = None
    raw_result_paths: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Optional[Provenance] = None


# --------------------------------------------------------------------------
# 4. DecisionValidation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseFloorSource:
    """S13: the noise floor is per-configuration, never inherited.
    Every `DecisionValidation` that claims `resolvable=True` must point
    at one of these, naming exactly which configuration, workload,
    regime, and repeat count it was measured from."""
    hardware_config_id: str
    workload_id: str
    regime: str
    repeats: int
    measured_cv_pct: Optional[float] = None
    measured_ci95_halfwidth: Optional[float] = None
    date: Optional[str] = None
    runtime_version: Optional[str] = None


@dataclass(frozen=True)
class ResolvabilitySpec:
    """`resolvable=None` (unknown) whenever no configuration-specific
    noise source exists -- never silently substituted with an old
    global figure (task 55's own "do not inherit a noise floor" -- this
    task's own S13 restates it for this exact object)."""
    resolvable: Optional[bool]
    reason: str
    noise_floor_source: Optional[NoiseFloorSource] = None


@dataclass(frozen=True)
class SloComparison:
    planner_predicted_pass: Optional[bool]
    hardware_observed_pass: Optional[bool]


@dataclass(frozen=True)
class ThroughputComparison:
    planner_predicted: Optional[float]
    hardware_observed: Optional[float]
    floor: Optional[float]
    floor_pass: Optional[bool]


@dataclass(frozen=True)
class PlacementComparison:
    requested: Tuple[PlacementRankAssignment, ...]
    observed: Tuple[PlacementRankAssignment, ...]
    exact_match: bool


@dataclass(frozen=True)
class DecisionValidation:
    """Whether the planner's selected candidate matches what real
    hardware benchmarking would have chosen -- Task 54's own reframing
    of the whole product question, made concrete and comparable."""
    decision_validation_version: str
    plan_id: str
    planner_selected_candidate: str
    hardware_best_candidate: Optional[str]
    top1_correct: Optional[bool]
    topk_correct: Optional[bool]
    k: Optional[int]
    regret_absolute: Optional[float]
    regret_relative: Optional[float]
    planner_margin: Optional[float]
    hardware_margin: Optional[float]
    tie_handling: str
    resolvability: ResolvabilitySpec
    slo: SloComparison
    throughput: ThroughputComparison
    placement: PlacementComparison
    provenance: Provenance
