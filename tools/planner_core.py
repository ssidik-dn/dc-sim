#!/usr/bin/env python3
"""Task 37: the search over configurations, separated from the oracle
that prices them.

Task 33's `plan()` interleaved three things: representing a candidate
and checking its feasibility, evaluating a candidate by running
Frontier, and ranking the results under constraints. Only the middle
one is simulation. This module holds the other two, plus the
`Evaluator` protocol that names the seam -- nothing here imports
Frontier or `src/integration/`, and nothing here knows how a candidate
actually gets priced.

**Why an `Evaluator` needs two methods, not one.** A simulator answers
a counterfactual: what would this configuration cost, for a
configuration that does not exist. Telemetry answers an observation:
what does the one configuration currently running actually cost. A
telemetry-backed evaluator cannot price an arrangement that is not
deployed, and a search that assumed it could would silently rank
fabrications. `can_evaluate` makes that boundary a first-class,
queryable fact rather than a failure discovered mid-evaluation.

**This module builds no evaluator.** `tools/planner.py` holds the only
one (`SimulationEvaluator`), which is why this module lives in
`tools/` rather than `src/engine/` despite having no Frontier
dependency itself: it exists to be paired with an evaluator that does
invoke Frontier by subprocess, and splitting it into `src/engine/`
would misrepresent what it is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from engine.physical.topology import Fabric
from engine.logical.deployment import Deployment, PoolKind, Replica, ParallelKind
from engine.placement.placement import packed, spread, fragmented, explicit

# ---------------------------------------------------------------- inputs


@dataclass
class Topology:
    """Machines, GPUs, NICs, scale-up domains, per-link bandwidth and
    latency -- an `engine.physical.topology.Fabric`, named for reporting."""
    fabric: Fabric
    name: str


@dataclass
class ModelSpec:
    """Hidden size, layers, attention kind, MoE-ness, and which
    parallelism degrees are admissible -- most of this already lives in
    Frontier's own model config JSON
    (`data/config/models/<model_name>.json`); this names it and states
    which degrees are worth searching, rather than re-deriving it."""
    model_name: str
    total_experts: int
    router_topk: int
    is_moe: bool
    admissible_tp: Tuple[int, ...] = (1, 2, 4, 8)
    admissible_ep: Tuple[int, ...] = (1,)
    # Needed by `feasible_num_blocks` (below) to compute DECODE_ATTN's own
    # parameter and KV-cache memory directly from Frontier's own formula
    # (`param_counter.py`/`memory_planner.py`, verified bit-for-bit against
    # Phi-tiny-MoE-instruct's own calibrated table in task 33/36) rather
    # than a per-model lookup table. `head_dim=None` means "derive as
    # hidden_size // num_attention_heads", matching
    # `ModelConfig.get_head_dim()`'s own fallback -- only override it when
    # the model's own JSON declares an explicit `head_dim` different from
    # that (Phi-tiny-MoE-instruct does; Llama-3.1-405B-Instruct-FP8 does
    # not, both confirmed by reading the JSON directly, not assumed).
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    num_layers: int = 0
    head_dim: Optional[int] = None
    # The tp degrees a *simulation* evaluator actually has profiled data
    # for -- Task 35's own finding: every model in the checkout, on
    # either device with real profiles, covers tp in {1,2,4,8} only,
    # because nobody overrode the profiler's own default sweep. Distinct
    # from `admissible_tp` (the search's own scope, which a caller
    # chooses) -- this is what `SimulationEvaluator.can_evaluate` reads,
    # a fact about the profile data, not a search-scope decision.
    profiled_tp: Tuple[int, ...] = (1, 2, 4, 8)
    # Task 39 Part B: Frontier reads a KV head count and a head size
    # *twice*, for two different purposes, and they are not always the
    # same field. `ParamCounter` (parameter memory, `attn_param_mem_bytes`
    # below) always reads the raw `num_key_value_heads`/`get_head_dim()`
    # pair. `MemoryPlanner`'s own KV-cache sizing reads the
    # *attention-family-resolved* pair instead
    # (`ModelConfig.get_runtime_num_kv_heads()`/`get_runtime_head_size()`,
    # `frontier/attention/families.py`'s own per-family resolvers). For
    # the DENSE_KV family -- every model this project has real h800/
    # rtx_pro_6000 profiles for, confirmed by checking each one's own
    # attention-family binding, not assumed -- the two are identical by
    # construction, so leaving these at `None` (falling back to the raw
    # fields) is correct, not a guess. For a LATENT_MLA model (this
    # checkout's own deepseek-v3/deepseek-r1-0528, mi355x-profiled only,
    # confirmed via `bind_attention_family`: `use_mla` is inferred from
    # `model_type in {deepseek_v2, deepseek_v3, deepseek_mtp, kimi_k2}`
    # plus a declared `kv_lora_rank`) the resolved KV head count is
    # *always* 1 regardless of the declared field, and the resolved head
    # size is `kv_lora_rank + qk_rope_head_dim`, not `get_head_dim()` --
    # a caller with such a model must supply both explicitly. This
    # formula does not implement per-family resolution itself (Part B's
    # own "do not approximate an assertion" trap): a model whose
    # attention family is not DENSE_KV and does not set these two fields
    # will silently get the DENSE_KV answer, which is exactly the failure
    # this project keeps finding. Nothing here auto-detects the family;
    # the caller must know.
    runtime_num_kv_heads: Optional[int] = None
    runtime_head_dim: Optional[int] = None
    # Task 48: `frontier/attention/families.py`'s own per-family
    # `kv_factor` -- 2 for DENSE_KV (separate K and V caches), **1 for
    # LATENT_MLA** (`get_attention_runtime_kv_layout`, `frontier/attention/memory.py`:
    # `elements_per_token_per_worker = kv_factor * runtime_num_kv_heads_per_worker
    # * runtime_head_size`) -- MLA stores one compressed latent, not a
    # separate K and V. `None` means "use this module's own `_KV_FACTOR`
    # constant (2)", correct for every DENSE_KV model this formula has
    # been validated against; a LATENT_MLA caller must pass `kv_factor=1`
    # explicitly, mirroring `runtime_num_kv_heads`/`runtime_head_dim`'s
    # own established idiom -- no auto-detection, same reasoning task 39
    # already gave for those two fields.
    kv_factor: Optional[int] = None


@dataclass
class Workload:
    """Prompt/output length (fixed, this project's own established
    convention -- see task 31 report S1.3 for why that makes a plan
    reproducible by construction), arrival rate, and concurrency."""
    num_requests: int
    qps: float
    prefill_tokens: int
    decode_tokens: int


@dataclass
class Hardware:
    """Device compute profile and the usable-memory knob tasks 24-28
    established as the only one Frontier actually exposes per cluster
    (`memory_margin_fraction`; `gpu_memory_utilization` has no
    per-cluster override, task 24's own finding)."""
    device: str
    memory_margin_fraction: float


@dataclass
class Regime:
    """How requests arrive, and how many independent draws `plan()` asks
    its evaluator to average per candidate before ranking -- task 45's
    own explicit input, alongside `topology`/`model`/`workload`/
    `hardware`/`objectives`, deliberately with no default `plan()`
    itself supplies: every prior task (32-44) evaluated every candidate
    at `Regime(seeded=False, num_seeds=1)` -- every request submitted
    simultaneously at `t=0` -- without ever having chosen that on
    purpose, because nothing before task 31 offered another option
    (task 31's own finding, task 42's own confirmation: every real-
    compute tool through task 28 was "completely deterministic given
    everything except `--seed`," since arrivals were always pinned to
    `t=0`). Task 42/44 both found this regime's own ranking can reverse
    under genuine staggered arrival -- this field exists so a caller
    states which regime a ranking is *for*, rather than inheriting one
    by accident.

    `seeded=True` asks the evaluator to use `seed_stats.seed_argv_fix`
    (Poisson-staggered arrivals at the workload's own qps, task 31's
    own mechanism) instead of the `t=0` burst. `num_seeds` is how many
    independent seeds each candidate is averaged over under that
    regime -- 1 is a single draw, not a measurement (task 45's own
    known trap, echoing task 31 S1.3's "a single seeded run is one
    point on a distribution, not the distribution"); the resulting
    `ci95_halfwidth` this project's `seed_stats.compute_interval_stats`
    reports on that average is this search's own resolution -- two
    candidates whose intervals overlap are indistinguishable at this
    `num_seeds`, not tied and not orderable, and `plan()` marks them as
    such rather than imposing a strict order the measurement does not
    support.

    `num_seeds > 1` with `seeded=False` is rejected outright, not
    silently run: repeating a deterministic `t=0` burst `num_seeds`
    times produces `num_seeds` identical numbers, a zero-width interval
    that looks like confidence but measures nothing -- the same error
    this project's own reports keep finding in a single unseeded run
    presented as if it were a distribution."""
    seeded: bool = False
    num_seeds: int = 1

    def __post_init__(self) -> None:
        if self.num_seeds > 1 and not self.seeded:
            raise ValueError(
                "Regime(num_seeds > 1, seeded=False) repeats a deterministic burst "
                "run num_seeds times -- every draw is identical, so the resulting "
                "interval has zero width and measures nothing. Set seeded=True.")
        if self.num_seeds < 1:
            raise ValueError(f"Regime.num_seeds must be >= 1, got {self.num_seeds}")


@dataclass
class Objectives:
    """Minimise `minimize`, subject to every constraint in `constraints`
    passing. SLO and throughput are constraints, not reported-alongside
    figures -- task 33's own S3: a constraint needs only pass/fail at a
    threshold, which sidesteps exactly the tail-noise problem task 31
    found (near a capacity edge, more seeds revealed more of the tail
    rather than sharpening the estimate)."""
    slo_tpot_ms: float
    min_throughput_rps: float
    minimize: str = "mean_tpot_ms"
    slo_attainment_floor: float = 0.0  # 0.0 = SLO reported, not constrained


@dataclass
class Candidate:
    attn_tp: int
    attn_shape: Tuple[int, ...]
    ffn_ep: int = 1
    # Task 44: the expert-parallel group's own placement, exactly the
    # same kind of value `attn_shape` already is -- `(1,)` at `ffn_ep=1`
    # (no expert-parallel group exists), otherwise a real
    # `Placement.group_shape()` result from `enumerate_joint_arrangements`.
    # Replaces `ep_split: bool`, a field this project carried since task
    # 33 that no placement logic ever actually read (checked directly:
    # neither `_placement_for` nor `_run_scenario` ever inspected it) --
    # exactly the gap this task exists to close, not a second flag kept
    # alongside the real one.
    ep_shape: Tuple[int, ...] = (1,)
    attn_replicas: int = 1
    ffn_replicas: int = 1
    # Task 57: which of the two groups' own occupied-domain sets this
    # candidate uses -- "same" (includes today's colocated arrangement),
    # "disjoint" (attention whole on one machine, experts whole on
    # another -- task 55/56's own unreachable arrangement), or
    # "overlapping"; `None` at `ffn_ep=1`/`attn_tp=1`, where only one
    # real group exists and the question does not apply. Needed because
    # `(attn_shape, ep_shape)` alone is no longer unique once
    # `enumerate_joint_arrangements` can return two placements that
    # share it -- disambiguates exactly the way `attn_shape`/`ep_shape`
    # themselves disambiguate a single group's own split.
    relative: Optional[str] = None

    @property
    def key(self) -> str:
        rel_suffix = f"_rel{self.relative}" if self.relative is not None else ""
        return (f"tp{self.attn_tp}_shape{'-'.join(map(str, self.attn_shape))}"
               f"_ep{self.ffn_ep}_epshape{'-'.join(map(str, self.ep_shape))}"
               f"{rel_suffix}_ar{self.attn_replicas}_fr{self.ffn_replicas}")


# ----------------------------------------------------------------- evaluator


@runtime_checkable
class Evaluator(Protocol):
    """The seam between search and oracle.

    Two methods, not one, because a simulator and a telemetry feed
    answer different questions. A simulator prices a *counterfactual* --
    what this configuration would cost, including ones nothing is
    running right now. Telemetry reports an *observation* -- what the
    one configuration actually deployed costs right now, and nothing
    else. `can_evaluate` lets a search ask "do you have an answer for
    this" before asking "what is it", so a telemetry-backed evaluator
    can truthfully say no to every candidate that isn't the deployment
    it is watching, instead of fabricating a price for one that is not
    running.

    A third shape is worth naming even though nothing here builds it: an
    evaluator that runs a simulation and corrects its own prediction
    using observed error from a configuration that *is* deployed would
    be more useful than either a pure simulator or pure telemetry alone
    -- closer to the counterfactual question, but grounded by whatever
    the running system's own error currently looks like. Nothing about
    telemetry exists in this project to build that against yet; this
    protocol is written so adding it later is a new class, not a
    redesign of `plan()`.
    """

    def can_evaluate(self, candidate: Candidate) -> bool:
        """Whether this evaluator has an answer for `candidate` at all --
        not whether the answer would be good enough. False means
        "unknown", never "rejected"; `plan()` keeps the two separate."""
        ...

    def evaluate(self, candidate: Candidate) -> dict:
        """The candidate's own result dict (`mean_tpot_ms`,
        `throughput_rps`, `slo_attainment`, or `error`). Only meaningful
        when `can_evaluate(candidate)` is true."""
        ...


# ------------------------------------------------------ memory feasibility

# Real device memory, in GB -- `frontier/config/device_sku_config.py`'s own
# SKU table (task 35's own finding), not assumed.
_DEVICE_MEMORY_GB = {
    "a40": 45, "a100": 80, "a800": 80, "h100": 80, "h800": 80,
    "h20": 96, "rtx_pro_6000": 96, "h200": 141, "mi355x": 288,
}

_KV_CACHE_BLOCK_SIZE = 16  # this project's own standing convention (tasks 22-33)
_KV_FACTOR = 2  # DENSE_KV family (standard MHA/GQA, K+V) -- frontier/attention/families.py


class InadmissibleDegree(ValueError):
    """Raised when `attn_tp` does not evenly divide this model's own
    structure -- `frontier/utils/param_counter.py`'s own
    `ParamCounter.__init__` asserts exactly these conditions before
    computing anything, and would raise `AssertionError` rather than
    return a fractional-heads-per-worker answer. A property of the
    (model, degree) *request*, not of available memory (task 39's own
    known trap: reporting this as a memory failure would attribute a
    property of the request to the hardware) -- kept as a distinct
    exception from the plain `ValueError` a malformed `ModelSpec` raises,
    so `plan()` can catch this one specifically without also swallowing
    that one."""


def divisibility_violations(model: ModelSpec, attn_tp: int) -> List[str]:
    """The three conditions `ParamCounter.__init__` asserts, checked
    without raising -- `plan()`'s own loop uses this to decide a
    candidate is inadmissible *before* ever calling
    `feasible_num_blocks`, so an inadmissible degree never gets
    generated into shapes or reported as a memory rejection. Empty list
    means every condition holds.

    Frontier's own third assertion (`embedding_dim % num_q_heads == 0`)
    is a property of the model alone, independent of `attn_tp` -- it is
    still checked here because `attn_param_mem_bytes` computes
    `hidden_size // num_attention_heads` as `head_dim`'s own fallback,
    and that division being inexact would silently truncate rather than
    raise, exactly the failure this project keeps finding."""
    violations = []
    if model.num_attention_heads <= 0:
        return violations  # feasible_num_blocks's own ValueError covers this
    if model.num_attention_heads % attn_tp != 0:
        violations.append(
            f"num_attention_heads ({model.num_attention_heads}) is not divisible "
            f"by attn_tp ({attn_tp})")
    if model.hidden_size % attn_tp != 0:
        violations.append(
            f"hidden_size ({model.hidden_size}) is not divisible by attn_tp ({attn_tp})")
    if model.hidden_size % model.num_attention_heads != 0:
        violations.append(
            f"hidden_size ({model.hidden_size}) is not divisible by num_attention_heads "
            f"({model.num_attention_heads}) -- a model-level property, independent of attn_tp")
    return violations


def _attn_head_dim(model: ModelSpec) -> int:
    return model.head_dim if model.head_dim is not None else model.hidden_size // model.num_attention_heads


def _runtime_kv_heads(model: ModelSpec) -> int:
    """The KV head count `MemoryPlanner`'s own KV-cache sizing reads
    (`Replica.kv_heads_per_tensor_parallel_worker` ->
    `ModelConfig.get_runtime_num_kv_heads()`), as opposed to the raw
    field `ParamCounter` reads for parameter memory. Identical to the
    raw field for the DENSE_KV family (every model this project has
    real h800/rtx_pro_6000 profiles for) -- `runtime_num_kv_heads=None`
    means exactly that, not "unknown." A LATENT_MLA model (this
    checkout's own deepseek-v3/deepseek-r1-0528) resolves this to `1`
    regardless of the declared field; such a caller must set
    `runtime_num_kv_heads` explicitly (task 39's own Part B)."""
    return model.runtime_num_kv_heads if model.runtime_num_kv_heads is not None else model.num_key_value_heads


def _runtime_head_dim(model: ModelSpec) -> int:
    """The head size `MemoryPlanner`'s own KV-cache sizing reads
    (`get_runtime_head_size()`), as opposed to `get_head_dim()` (which
    `attn_param_mem_bytes` uses for parameter memory, matching
    `ParamCounter` exactly). Identical for DENSE_KV; a LATENT_MLA model
    resolves this to `kv_lora_rank + qk_rope_head_dim`, not
    `get_head_dim()` -- such a caller must set `runtime_head_dim`
    explicitly."""
    return model.runtime_head_dim if model.runtime_head_dim is not None else _attn_head_dim(model)


def attn_param_mem_bytes(model: ModelSpec, attn_tp: int) -> int:
    """DECODE_ATTN's own per-device parameter memory at this `attn_tp`,
    computed directly from `frontier/utils/param_counter.py`'s own
    formula (Q/K/V/O only -- DECODE_FFN's MLP/MoE weights are a separate
    cluster's memory, task 35's own S0 scoping point) and
    `frontier/scheduler/utils/memory_planner.py`'s own unconditional
    2-bytes/param assumption. Verified bit-for-bit against
    `ParamCounter.get_num_parameters_per_device()` for
    Phi-tiny-MoE-instruct, Llama-3.1-405B-Instruct-FP8, and
    step-moe-noquant-small (tasks 36/38) before use, not assumed from a
    formula alone.

    Raises `InadmissibleDegree` if `attn_tp` does not evenly divide this
    model -- Frontier's own `ParamCounter.__init__` would assert on
    exactly this rather than compute a fractional-heads-per-worker
    answer (task 39 Part A)."""
    violations = divisibility_violations(model, attn_tp)
    if violations:
        raise InadmissibleDegree(
            f"attn_tp={attn_tp} is inadmissible for {model.model_name!r}: " + "; ".join(violations))
    head_dim = _attn_head_dim(model)
    q_per_worker = model.num_attention_heads / attn_tp
    kv_per_worker = -(-model.num_key_value_heads // attn_tp)  # ceil
    per_layer = (model.hidden_size * head_dim * (q_per_worker + 2 * kv_per_worker)
                + model.hidden_size * head_dim * q_per_worker)
    return int(2 * per_layer * model.num_layers)


def _runtime_kv_factor(model: ModelSpec) -> int:
    """`frontier/attention/families.py`'s own per-family `kv_factor` --
    2 for DENSE_KV (this module's own `_KV_FACTOR`), **1 for LATENT_MLA**
    (task 48's own finding: MLA stores one compressed latent, not a
    separate K and V cache -- confirmed against Frontier's own
    `MemoryPlanner` directly for deepseek-v3, which disagreed with this
    formula by exactly 2x before this field existed). `kv_factor=None`
    means "use `_KV_FACTOR`," correct for every DENSE_KV model this
    formula has been validated against; a LATENT_MLA caller must set
    `kv_factor=1` explicitly."""
    return model.kv_factor if model.kv_factor is not None else _KV_FACTOR


def _kv_cache_page_bytes_per_layer(model: ModelSpec, attn_tp: int, block_size: int) -> int:
    """`frontier/scheduler/utils/memory_planner.py`'s own
    `_get_kv_cache_memory_per_layer_per_block`: 2 bytes/element x
    block_size x kv_factor x kv_heads_per_worker x head_dim -- using the
    *runtime-resolved* kv-head-count, head-dim, and kv_factor (task 39
    Part B; kv_factor added task 48), not the raw/param-counting ones
    `attn_param_mem_bytes` uses. Identical values for every model this
    formula has been validated against (DENSE_KV); see
    `_runtime_kv_heads`/`_runtime_head_dim`/`_runtime_kv_factor`."""
    head_dim = _runtime_head_dim(model)
    kv_per_worker = -(-_runtime_kv_heads(model) // attn_tp)
    return 2 * block_size * _runtime_kv_factor(model) * kv_per_worker * head_dim


def feasible_num_blocks(model: ModelSpec, hardware: Hardware, attn_tp: int) -> Optional[int]:
    """`None` if infeasible (parameter memory alone exceeds the usable
    budget at this margin); otherwise the derived `num_blocks`, from
    `frontier/scheduler/utils/memory_planner.py`'s own `get_num_blocks`
    formula: `available_kv_cache_memory // page_size // num_layers`,
    `available_kv_cache_memory = requested_memory - parameter_memory`.

    A property of `model`/`hardware`/`attn_tp` alone -- no evaluator of
    any kind is consulted, and none would be needed: an oracle backed by
    a running system would compute the identical number from the same
    three inputs. This is why feasibility is filtered here, in the
    search, rather than delegated to whatever `Evaluator` `plan()` was
    given (task 37's own S3).

    Raises `InadmissibleDegree` (via `attn_param_mem_bytes`) if `attn_tp`
    does not evenly divide this model -- `plan()`'s own loop checks
    `divisibility_violations` before ever calling this, so that path is
    for any other caller (task 39 Part A)."""
    if model.num_attention_heads <= 0 or model.hidden_size <= 0 or model.num_layers <= 0:
        raise ValueError(
            f"ModelSpec({model.model_name!r}) is missing hidden_size/"
            "num_attention_heads/num_key_value_heads/num_layers -- "
            "feasible_num_blocks needs them to compute DECODE_ATTN's own "
            "memory directly, not a per-model lookup table.")
    device_bytes = _DEVICE_MEMORY_GB[hardware.device] * 1024 ** 3
    requested_memory = int(device_bytes * (1 - hardware.memory_margin_fraction))
    param_mem = attn_param_mem_bytes(model, attn_tp)
    avail = requested_memory - param_mem
    if avail <= 0:
        return None
    page_size = _kv_cache_page_bytes_per_layer(model, attn_tp, _KV_CACHE_BLOCK_SIZE)
    num_blocks = avail // page_size // model.num_layers
    return int(num_blocks) if num_blocks > 0 else None


def lane_assignment_feasible(attn_replicas: int, ffn_replicas: int, attn_dp_size: int) -> bool:
    """Task 22's own mechanical finding: Frontier's static M2N lane
    assignment needs `attn_replicas * attn_dp_size >= ffn_replicas`, or it
    raises. Not a preference -- a hard constraint on the (attn_replicas,
    ffn_replicas, attn_dp_size) triple."""
    return attn_replicas * attn_dp_size >= ffn_replicas


def default_attn_dp_size_policy(attn_replicas: int, ffn_replicas: int) -> int:
    """Task 32 S7's own open question -- "search [attn_dp_size] jointly or
    fix a policy for setting it per candidate ratio, not carry over this
    task's own fixed value [of 1]" -- resolved the same way
    `tools/planner.py`'s own `_argv` already resolves it (task 33):
    `attn_dp_size := max(ffn_replicas, 1)`, the smallest fixed choice that
    satisfies `lane_assignment_feasible` for every `attn_replicas >= 1`
    unconditionally (`attn_replicas * ffn_replicas >= ffn_replicas` holds
    whenever `attn_replicas >= 1`) -- so under this policy the lane check
    below can never actually be the reason a candidate is inadmissible;
    that outcome is reported plainly in task 41's own report, not hidden.

    Kept as an explicit, overridable policy (not inlined into `plan()`)
    for exactly the reason task 32 S7 named it as an open design
    question: a caller that wants to see the lane check actually bind --
    or a future evaluator with a different dp_size story -- can pass a
    different one without `plan()` itself changing."""
    return max(ffn_replicas, 1)


# --------------------------------------------------------- candidate generation


def enumerate_attn_shapes(topology: Topology, attn_tp: int,
                          n_fragmented_seeds: int = 60) -> Dict[Tuple[int, ...], object]:
    """Every distinct `group_shape()` reachable for a single DECODE_ATTN
    replica's own TP group on `topology.fabric`, via this project's own
    existing placement policies -- task 32's own method, reused
    unchanged, parameterised on the fabric instead of a module constant.

    Pure `engine.placement`/`engine.logical` machinery -- no Frontier,
    no subprocess. This is why it lives here rather than alongside
    `SimulationEvaluator`: it enumerates the *space* placement search
    considers, which is a property of the fabric and the degree, not of
    how any one candidate gets priced.
    """
    fabric = topology.fabric
    d = Deployment("shape-probe")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=attn_tp))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1))
    group = d.replicas[1].groups(ParallelKind.TP)[0] if attn_tp > 1 else None

    candidates = []
    for policy in (packed, spread):
        try:
            candidates.append(policy(d, fabric))
        except Exception:  # noqa: BLE001
            pass
    domain_size = min(len(dom.members) for dom in fabric.domains.values())
    if attn_tp <= domain_size and attn_tp > 1:
        prefill_rank = d.replicas[0].ranks[0]
        ffn_rank = d.replicas[2].ranks[0]
        domain_ids = sorted(fabric.domains)
        attn_domain_members = sorted(fabric.domains[domain_ids[1]].members)
        other_domain_members = sorted(fabric.domains[domain_ids[0]].members)
        mapping = {prefill_rank: other_domain_members[0], ffn_rank: other_domain_members[1]}
        for i, r in enumerate(group.ranks):
            mapping[r] = attn_domain_members[i]
        try:
            candidates.append(explicit(d, fabric, mapping))
        except Exception:  # noqa: BLE001
            pass
    for seed in range(n_fragmented_seeds):
        try:
            candidates.append(fragmented(d, fabric, seed=seed))
        except Exception:  # noqa: BLE001
            pass

    if attn_tp == 1:
        return {(1,): candidates[0]} if candidates else {}

    shapes = {}
    for p in candidates:
        shape = p.group_shape(group)
        if shape not in shapes:
            shapes[shape] = p
    return shapes


def enumerate_replica_arrangements(topology: Topology, attn_tp: int, attn_replicas: int,
                                   ffn_replicas: int = 1,
                                   n_fragmented_seeds: int = 60
                                   ) -> Dict[Tuple[Tuple[int, ...], ...], object]:
    """Task 41's own extension of `enumerate_attn_shapes` to more than one
    replica of a pool: every distinct *arrangement* of `attn_replicas`
    DECODE_ATTN replicas (each its own TP group of degree `attn_tp`)
    reachable on `topology.fabric`, via this project's own existing
    placement policies applied to the FULL multi-replica deployment at
    once -- so a raw candidate reflects whether the fabric has room for
    every replica simultaneously, not `_placement_for`'s own simpler
    single-reference-then-pack-leftovers fallback (`tools/planner.py`,
    task 33).

    Deduplicated as a **multiset**, not a tuple: replicas of the same
    pool are interchangeable (`AGENTS.md`'s own `group_shape()`
    invariant, extended here across replicas rather than only within one
    group's own ranks) -- an arrangement assigning shapes `{A, B}` to two
    attention replicas is the identical arrangement to `{B, A}`, so the
    canonical key sorts the per-replica shapes before using them, the
    same principle `group_shape()` itself applies to one group's own
    per-domain counts.

    `ffn_replicas` (always tp=1 in this project's own convention; task
    41's own known trap explicitly excludes searching FFN's own EP/TP
    placement here) still occupies real ranks and real GPUs, so it
    affects how much room is left for ATTN's own replicas on a shared
    fabric -- but contributes no shape choice of its own, so only the
    ATTN multiset is the key.

    Pure `engine.placement`/`engine.logical`, no Frontier -- this
    enumerates the *space*, exactly `enumerate_attn_shapes`'s own
    purpose, not a priced result."""
    fabric = topology.fabric
    d = Deployment("replica-arrangement-probe")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    for i in range(attn_replicas):
        d.add(Replica(PoolKind.DECODE_ATTN, i, tp=attn_tp))
    for i in range(ffn_replicas):
        d.add(Replica(PoolKind.DECODE_FFN, i, tp=1))

    attn_groups = ([r.groups(ParallelKind.TP)[0] for r in d.pool(PoolKind.DECODE_ATTN)]
                  if attn_tp > 1 else [])

    candidates = []
    for policy in (packed, spread):
        try:
            candidates.append(policy(d, fabric))
        except Exception:  # noqa: BLE001
            pass
    for seed in range(n_fragmented_seeds):
        try:
            candidates.append(fragmented(d, fabric, seed=seed))
        except Exception:  # noqa: BLE001
            pass

    arrangements: Dict[Tuple[Tuple[int, ...], ...], object] = {}
    for p in candidates:
        if attn_tp == 1:
            sig = tuple([(1,)] * attn_replicas)
        else:
            sig = tuple(sorted((p.group_shape(g) for g in attn_groups)))
        if sig not in arrangements:
            arrangements[sig] = p
    return arrangements


def enumerate_joint_arrangements(topology: Topology, attn_tp: int, ffn_ep: int,
                                 n_fragmented_seeds: int = 60
                                 ) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Optional[str]], object]:
    """Task 44's own extension: `enumerate_attn_shapes` (task 32) searches
    DECODE_ATTN's own TP group in isolation; this searches DECODE_ATTN's
    TP group and DECODE_FFN's own expert-parallel group *together*, on
    the same deployment, against the same fabric -- because they may
    share domains, and a search that enumerated each alone and combined
    the results afterward would silently assume the two never interact
    (task 44's own S2 question, answered by construction here: the raw
    candidates below place both groups on one real fabric at once, so
    whatever interaction exists is however `packed`/`spread`/`fragmented`
    actually resolve it, not assumed away).

    **Not task 41's multiset.** An attention TP group and an expert
    group are different kinds of thing -- one shards attention weights,
    the other dispatches tokens to experts -- so giving shape A to
    attention and shape B to the expert group is a *different*
    arrangement from giving A to the expert group and B to attention.
    The canonical key's first two components are an ordered pair,
    `(attn_shape, ep_shape)`, never sorted together: task 41's own
    interchangeable-replica reasoning applies to *enumerating* several
    groups of the *same* kind (task 41's own attention replicas), not to
    two groups of different kinds, which is exactly this task's own
    known trap.

    **The key has a third component, `relative` (task 57).** Task 56
    found that "attention whole on one machine, experts whole on
    another" was unreachable even though `fragmented()` genuinely
    constructs it: `group_shape()`'s own per-group signature records
    only how many ranks land in each domain, never *which* domain, so
    that arrangement's key collided with the colocated arrangement's
    (both `((attn_tp,), (ffn_ep,))`) and whichever was discovered first
    -- always the colocated one, since `packed`/`spread` run before the
    `fragmented` seeds -- silently kept the other one out.
    `_relative_domain_placement` (below) adds exactly the missing
    information, without changing `group_shape()` itself (other callers
    -- `enumerate_attn_shapes`, task 41's replica enumeration -- depend
    on its current, single-group behaviour and must not see this):
    whether the two groups' own occupied-domain sets are identical
    (`"same"`), disjoint (`"disjoint"`), or neither (`"overlapping"`).
    `None` when either group does not exist (`attn_tp=1` or
    `ffn_ep=1`) -- see the docstring below for why that keeps every
    single-group case's own *set* of keys unchanged in substance.

    At `ffn_ep=1` there is no expert-parallel group at all
    (`Replica.groups()` returns `[]` for a degree-1 dimension, same as
    `enumerate_attn_shapes` at `attn_tp=1`) -- `ep_shape` is then always
    `(1,)`, `relative` is always `None`, and the set of `attn_shape`
    values this function reaches is identical to `enumerate_attn_shapes`'s
    own, built from the same deployment shape, the same policies, the
    same seeds. This is what keeps the `ffn_ep=1` case bit-identical to
    every call site that predates task 44 (its own acceptance
    requirement, still honoured after task 57's own key change).
    """
    fabric = topology.fabric
    d = Deployment("joint-arrangement-probe")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=attn_tp))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1, ep=ffn_ep))

    attn_replica = d.pool(PoolKind.DECODE_ATTN)[0]
    ffn_replica = d.pool(PoolKind.DECODE_FFN)[0]
    attn_group = attn_replica.groups(ParallelKind.TP)[0] if attn_tp > 1 else None
    ep_group = ffn_replica.groups(ParallelKind.EP)[0] if ffn_ep > 1 else None

    candidates = []
    for policy in (packed, spread):
        try:
            candidates.append(policy(d, fabric))
        except Exception:  # noqa: BLE001
            pass

    # The same explicit "packed-if-it-fits" fallback enumerate_attn_shapes
    # already needs (task 34's own finding: packed()'s own rank ordering
    # gives a later group a one-slot offset from an earlier one, so it
    # does not reach a clean single-domain shape on its own even when one
    # would fit) -- extended to place the expert group in its own third
    # domain, not just the attention group in its own second one, so the
    # "everything stays whole" arrangement is actually reachable for both
    # groups at once when the fabric has room for it.
    domain_size = min(len(dom.members) for dom in fabric.domains.values())
    domain_ids = sorted(fabric.domains)
    if len(domain_ids) >= 3:
        prefill_rank = d.replicas[0].ranks[0]
        ffn_anchor_rank = ffn_replica.ranks[0]
        mapping = {}
        base_members = sorted(fabric.domains[domain_ids[0]].members)
        mapping[prefill_rank] = base_members[0]
        mapping[ffn_anchor_rank] = base_members[1]
        ok = True
        if attn_group is not None:
            if attn_tp <= domain_size:
                attn_members = sorted(fabric.domains[domain_ids[1]].members)
                for i, r in enumerate(attn_group.ranks):
                    mapping[r] = attn_members[i]
            else:
                ok = False
        if ok and ep_group is not None:
            if ffn_ep <= domain_size:
                ep_members = sorted(fabric.domains[domain_ids[2]].members)
                for i, r in enumerate(ep_group.ranks):
                    mapping[r] = ep_members[i]
            else:
                ok = False
        if ok:
            try:
                candidates.append(explicit(d, fabric, mapping))
            except Exception:  # noqa: BLE001
                pass

    # Task 57: the two-domain analogue of the fallback above. With
    # exactly two real domains (this project's own "two real machines"
    # case, task 54/55/56), the >=3-domain fallback above never fires --
    # it needs a third domain for prefill/the FFN anchor rank, separate
    # from either group's own. Task 56 found that `fragmented()` *can*
    # construct "attention whole on one domain, experts whole on the
    # other" (seeds 3, 9, 49, confirmed live for the two degree pairs
    # this project has tested), but only by luck; a different seed
    # count, or a larger degree pair the random policy is less likely to
    # stumble on, would lose it again silently. This constructs it
    # deterministically instead, whenever both groups exist and each
    # individually fits in one domain -- prefill packed alongside
    # attention (the smaller, or equal, footprint) rather than needing
    # its own domain, since only two are available.
    if len(domain_ids) >= 2 and attn_group is not None and ep_group is not None:
        if attn_tp <= domain_size and ffn_ep <= domain_size:
            prefill_rank = d.replicas[0].ranks[0]
            attn_domain_members = sorted(fabric.domains[domain_ids[0]].members)
            ep_domain_members = sorted(fabric.domains[domain_ids[1]].members)
            if attn_tp + 1 <= len(attn_domain_members):
                mapping = {prefill_rank: attn_domain_members[0]}
                for i, r in enumerate(attn_group.ranks):
                    mapping[r] = attn_domain_members[i + 1]
                for i, r in enumerate(ep_group.ranks):
                    mapping[r] = ep_domain_members[i]
                try:
                    candidates.append(explicit(d, fabric, mapping))
                except Exception:  # noqa: BLE001
                    pass

    for seed in range(n_fragmented_seeds):
        try:
            candidates.append(fragmented(d, fabric, seed=seed))
        except Exception:  # noqa: BLE001
            pass

    arrangements: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Optional[str]], object] = {}
    for p in candidates:
        attn_shape = p.group_shape(attn_group) if attn_group is not None else (1,)
        ep_shape = p.group_shape(ep_group) if ep_group is not None else (1,)
        relative = _relative_domain_placement(p, attn_group, ep_group)
        key = (attn_shape, ep_shape, relative)
        if key not in arrangements:
            arrangements[key] = p
    return arrangements


def _relative_domain_placement(placement, attn_group: Optional[object],
                               ep_group: Optional[object]) -> Optional[str]:
    """Task 57: `group_shape()` records how many ranks of *one* group
    land in each domain, never which domain -- correct for a single
    group (task 32's own original use, where one domain really is
    interchangeable with another), wrong once two *different* groups are
    enumerated together, where whether they share a domain is exactly
    the thing a caller needs to tell apart. This is the third component
    `enumerate_joint_arrangements`'s own key adds, deliberately not
    folded into `group_shape()` itself -- `enumerate_attn_shapes` and
    task 41's replica enumeration both still call `group_shape()`
    directly and must not see this.

    `None` when either group does not exist (`attn_tp=1` or `ffn_ep=1`)
    -- there is nothing for "shared a domain" to mean with only one real
    group, and `None` is this dataclass-free module's own established
    idiom for "not applicable" (`Candidate.ep_shape` reads the same way
    at `ffn_ep=1`). This is what keeps every single-group call's own
    *set* of keys unchanged in substance (task 44's own bit-identical
    requirement) even though each key's own arity grows by one.

    Otherwise one of `"same"` (the two groups' own occupied-domain sets
    are identical -- includes today's colocated arrangement, where both
    are a single, shared domain, but is not limited to that case),
    `"disjoint"` (no domain in common -- the arrangement task 55/56
    found unreachable), or `"overlapping"` (neither -- one group's
    domains are a strict subset of, or partially intersect, the
    other's)."""
    if attn_group is None or ep_group is None:
        return None
    attn_domains = placement.domains_spanned(attn_group.ranks)
    ep_domains = placement.domains_spanned(ep_group.ranks)
    if attn_domains == ep_domains:
        return "same"
    if not (attn_domains & ep_domains):
        return "disjoint"
    return "overlapping"


# ------------------------------------------------------------------- plan()


@dataclass
class Rejection:
    candidate_key: str
    reason: str


@dataclass
class Unknown:
    """A candidate `plan()` never asked its evaluator to price, because
    `can_evaluate` said no -- a gap in the evaluator's own coverage, not
    a property of the candidate. Kept separate from `Rejection` per this
    task's own known trap: conflating the two would report "outside
    what this evaluator can price" as if it meant "worse than the
    alternatives", which it does not claim to know."""
    candidate_key: str
    reason: str


@dataclass
class Inadmissible:
    """A candidate that cannot exist as requested, independent of
    hardware, margin, or which evaluator `plan()` was given -- either a
    degree that does not evenly divide this model's own structure
    (`attn_tp` doesn't divide `num_attention_heads`, or `hidden_size`
    doesn't divide by `attn_tp` or by `num_attention_heads`, task 39
    Part A), or a (attn_replicas, ffn_replicas, attn_dp_size) triple that
    violates Frontier's own static M2N lane-assignment requirement
    (task 22's own finding, task 41's own extension: `attn_replicas *
    attn_dp_size >= ffn_replicas`, checked via `lane_assignment_feasible`
    before any candidate is even constructed). Distinct from both
    `Rejection` (a property of hardware/memory at a given margin) and
    `Unknown` (a gap in one evaluator's own coverage): this project's own
    known trap is attributing a property of the request itself to the
    hardware or the evaluator instead."""
    candidate_key: str
    reason: str


@dataclass
class PlanResult:
    winner: Optional[dict]
    ranked: List[dict]
    rejections: List[Rejection]
    unknown: List[Unknown]
    inadmissible: List[Inadmissible] = field(default_factory=list)
    regime: Optional["Regime"] = None


def _mark_indistinguishable_from_winner(evaluated: List[dict], minimize: str) -> None:
    """Task 45: a candidate's own `ci95_halfwidth` (present whenever its
    result came from `Regime(num_seeds > 1)`; absent, i.e. `None`, from
    a single deterministic burst run) makes some strict orderings
    unsupported by the measurement itself. Sets
    `row["indistinguishable_from_winner"]` on every row in `evaluated`
    (already sorted by `minimize`) -- `True` when that row's own
    `[mean - ci, mean + ci]` interval overlaps the winner's, `False`
    otherwise, including for every row when no row carries a
    `ci95_halfwidth` at all (a burst search's own strict order is
    exactly as supported as it always was -- this function is a no-op
    in effect for `Regime(num_seeds=1)`, per task 45's own acceptance
    requirement that burst results keep reproducing unchanged).

    This does not reorder anything: `evaluated` is still sorted by
    `minimize`, and "indistinguishable" is a property reported
    alongside that order, not a replacement for it (task 45's own known
    trap -- "indistinguishable is not tied" cuts both ways: it is also
    not a silent demotion)."""
    if not evaluated:
        return
    winner = evaluated[0]
    w_ci = winner.get("ci95_halfwidth")
    w_mean = winner[minimize]
    for row in evaluated:
        row["indistinguishable_from_winner"] = False
    if w_ci is None:
        return
    w_lo, w_hi = w_mean - w_ci, w_mean + w_ci
    for row in evaluated[1:]:
        ci = row.get("ci95_halfwidth")
        if ci is None:
            continue
        mean = row[minimize]
        lo, hi = mean - ci, mean + ci
        if not (hi < w_lo or lo > w_hi):
            row["indistinguishable_from_winner"] = True


def plan(topology: Topology, model: ModelSpec, workload: Workload, hardware: Hardware,
        objectives: Objectives, regime: Regime, evaluator: Evaluator, *,
        attn_tp_values: Tuple[int, ...] = None,
        ep_values: Tuple[int, ...] = None,
        replica_ratios: Tuple[Tuple[int, int], ...] = ((1, 1),),
        attn_dp_size_policy: Callable[[int, int], int] = default_attn_dp_size_policy) -> PlanResult:
    """Generate candidates over `attn_tp_values` (default:
    `model.admissible_tp`) x placement shape (per `topology`) x
    `ep_values` (default: `model.admissible_ep`) x `replica_ratios`,
    reject infeasible ones up front, ask `evaluator` whether it can
    price the rest before asking for a price, evaluate what it can, and
    rank the survivors by `objectives.minimize` among those meeting
    every constraint.

    `evaluator` is required here, not defaulted -- this module knows
    nothing about Frontier, so it cannot name a default. `tools/planner.py`'s
    own `plan()` wraps this one with `SimulationEvaluator()` as the
    default, which is what keeps every existing call site unchanged.

    `regime` is required too, with no default -- task 45's own point:
    every candidate's price depends on the arrival process it was
    priced under, and a default here would silently pick one (burst,
    almost certainly, since that is what every evaluator's own
    plumbing predates) instead of making a caller choose. This function
    does not itself run anything under `regime` -- that is
    `evaluator`'s own job, and a `Regime`-blind `Evaluator` (a fake, or
    a future telemetry-backed one with no seed concept at all) is free
    to ignore it -- but every evaluated result's own optional
    `ci95_halfwidth` key (present whenever the evaluator actually
    averaged over `regime.num_seeds > 1` seeds) is used here, after
    sorting, to mark which candidates are indistinguishable from the
    winner rather than merely ranked below it.

    `replica_ratios` defaults to `((1, 1),)` -- every call site from
    tasks 32/33/36 (which never pass it) reproduces bit-identically,
    since a single (1, 1) ratio is exactly what every one of those
    searches already assumed implicitly. Task 41's own extension: a
    caller that passes more ratios gets each checked against
    `lane_assignment_feasible` (via `attn_dp_size_policy`, task 32 S7's
    own open design question) *before* a `Candidate` is even built --
    a violation is `Inadmissible`, not a `Rejection`, since it is a
    property of the (attn_replicas, ffn_replicas, attn_dp_size) triple
    itself, true for every evaluator and every hardware margin, exactly
    the distinction task 39 drew for a non-dividing `attn_tp`.
    """
    attn_tp_values = attn_tp_values or model.admissible_tp
    ep_values = ep_values or model.admissible_ep

    rejections: List[Rejection] = []
    unknown: List[Unknown] = []
    inadmissible: List[Inadmissible] = []
    evaluated: List[dict] = []

    for attn_tp in attn_tp_values:
        violations = divisibility_violations(model, attn_tp)
        if violations:
            inadmissible.append(Inadmissible(f"tp{attn_tp}_*", "; ".join(violations)))
            continue
        num_blocks = feasible_num_blocks(model, hardware, attn_tp)
        if num_blocks is None:
            rejections.append(Rejection(f"tp{attn_tp}_*", "memory: infeasible at this margin"))
            continue
        for ep in ep_values:
            # Task 44: the attention TP group and the expert-parallel
            # group are enumerated *together* (S2's own independence
            # question), not `enumerate_attn_shapes` alone followed by a
            # separately-chosen `ep_shape` -- at `ep=1` this reaches the
            # identical set of `attn_shape` values `enumerate_attn_shapes`
            # itself would (see `enumerate_joint_arrangements`'s own
            # docstring for why), which is what keeps every pre-task-44
            # call site (`ep_values` defaulting to `(1,)`) bit-identical.
            joint = enumerate_joint_arrangements(topology, attn_tp, ep)
            for shape, ep_shape, relative in joint:
                for attn_replicas, ffn_replicas in replica_ratios:
                    attn_dp_size = attn_dp_size_policy(attn_replicas, ffn_replicas)
                    if not lane_assignment_feasible(attn_replicas, ffn_replicas, attn_dp_size):
                        inadmissible.append(Inadmissible(
                            f"tp{attn_tp}_shape{shape}_ep{ep}_epshape{ep_shape}_rel{relative}"
                            f"_ar{attn_replicas}_fr{ffn_replicas}",
                            f"lane assignment: attn_replicas({attn_replicas})*attn_dp_size"
                            f"({attn_dp_size}) < ffn_replicas({ffn_replicas})"))
                        continue
                    candidate = Candidate(attn_tp, shape, ep, ep_shape, attn_replicas, ffn_replicas,
                                          relative=relative)
                    if not evaluator.can_evaluate(candidate):
                        unknown.append(Unknown(candidate.key,
                            "evaluator cannot price this candidate (outside its own coverage)"))
                        continue
                    r = evaluator.evaluate(candidate)
                    if r.get("error"):
                        rejections.append(Rejection(candidate.key, f"evaluation error: {r['error']}"))
                        continue
                    r["candidate"] = candidate
                    if r["throughput_rps"] < objectives.min_throughput_rps:
                        rejections.append(Rejection(candidate.key,
                            f"throughput floor: {r['throughput_rps']:.3f} < {objectives.min_throughput_rps}"))
                        continue
                    if r["slo_attainment"] < objectives.slo_attainment_floor - 1e-9:
                        rejections.append(Rejection(candidate.key,
                            f"SLO: attainment {r['slo_attainment']:.3f} below required "
                            f"{objectives.slo_attainment_floor}"))
                        continue
                    evaluated.append(r)

    evaluated.sort(key=lambda r: r[objectives.minimize])
    _mark_indistinguishable_from_winner(evaluated, objectives.minimize)
    winner = evaluated[0] if evaluated else None
    return PlanResult(winner=winner, ranked=evaluated, rejections=rejections, unknown=unknown,
                     inadmissible=inadmissible, regime=regime)


# ------------------------------------------------------------ two-stage search


def _sizing_key(candidate: Candidate) -> Tuple[int, int, int, int]:
    """The axis task 41/44 already showed reverses under streaming --
    `attn_tp`'s own *degree* is deliberately excluded from this key (it
    is the placement/compute-parallelism axis Part A validated, task
    42's own conclusions 4/5: "tp=2 beats tp=1" held under streaming),
    while `ffn_ep` and the replica counts are included (task 44's own
    ep-degree reversal; task 41's own "more FFN replicas is 34% faster
    [burst] / 2.8% slower [streaming]"). `attn_tp` is grouped alongside
    these because a *placement* shortlist is only comparable within one
    fixed `attn_tp` -- shapes at different `attn_tp` are not
    substitutable choices for the same slot."""
    return (candidate.attn_tp, candidate.ffn_ep, candidate.attn_replicas, candidate.ffn_replicas)


@dataclass
class TwoStagePlanResult:
    """Task 45 Part B. `ranked`/`winner` are always stage 2's own
    measured result -- never stage 1's filtered one -- because a
    filtered ordering and a measured one are not the same kind of
    object (task 45's own known trap) and reporting the cheap one as if
    it were the expensive one is exactly the error this design exists
    to avoid. `shortlisted` is stage 1's own output, kept so a caller
    can see what was filtered out, from what, and under which (cheap)
    regime -- not thrown away once stage 2 runs."""
    winner: Optional[dict]
    ranked: List[dict]
    shortlisted: List[dict]
    shortlist_regime: Regime
    streaming_regime: Regime
    shortlist_size: int
    stage1: PlanResult
    stage2_rejections: List[Rejection]
    stage2_unknown: List[Unknown]


def plan_two_stage(topology: Topology, model: ModelSpec, workload: Workload, hardware: Hardware,
                   objectives: Objectives, shortlist_regime: Regime, shortlist_evaluator: Evaluator,
                   streaming_regime: Regime, streaming_evaluator: Evaluator, *,
                   shortlist_size: int = 1,
                   attn_tp_values: Tuple[int, ...] = None,
                   ep_values: Tuple[int, ...] = None,
                   replica_ratios: Tuple[Tuple[int, int], ...] = ((1, 1),),
                   attn_dp_size_policy: Callable[[int, int], int] = default_attn_dp_size_policy
                   ) -> TwoStagePlanResult:
    """Task 45 Part B's own two-stage design -- scoped exactly to what
    Part A measured, not to "the search" in general (task 45's own
    known trap: a rank correlation on one search is one data point).

    Part A found, on task 33's own 16-candidate table (varying
    `attn_tp` and its own placement `attn_shape` only -- the
    compute-parallelism/placement axis, nothing about capacity), a
    Spearman rank correlation of 1.0 between burst and streaming
    orderings: the streaming winner sat at burst rank 1 of 16. That
    result -- and task 42's own independent finding that this same axis
    "held" under streaming (conclusions 1, 4, 5) -- is why *this*
    function shortlists placements cheaply. It does not, and must not,
    shortlist `ffn_ep` or `(attn_replicas, ffn_replicas)` the same way:
    those are the *sizing* axis task 41 and task 44 already measured
    reversing under streaming (task 41's own replica ratio: burst-better
    becomes streaming-worse; task 44's own EP degree: burst-last becomes
    streaming-first) -- pre-filtering that axis by a burst ranking is
    exactly the failure Part A exists to catch, and this design refuses
    to do it: `stage 2` evaluates every sizing combination `plan()`
    would have generated, none shortlisted, only the placement within
    each narrowed to `shortlist_size`.

    **Stage 1** (`shortlist_regime`, conventionally burst -- cheap,
    deterministic, one run per candidate) runs the *unmodified* `plan()`
    search across the full space, but with `objectives`'s own
    `min_throughput_rps`/`slo_attainment_floor` zeroed out first: those
    two constraints are regime-dependent facts about queueing behaviour
    (task 42's own S1: a burst manufactures contention a real stream
    does not), and rejecting a candidate on them at the cheap stage
    would risk discarding the eventual streaming winner on the
    constraint axis exactly as Part A worried about on the ranking axis.
    Only regime-independent infeasibility (memory, divisibility, lane
    assignment -- `Inadmissible`/the memory `Rejection`) is filtered at
    stage 1; the real objectives are applied only once, at stage 2,
    against real streaming numbers.

    Stage 1's own survivors are grouped by `_sizing_key` (`attn_tp`,
    `ffn_ep`, `attn_replicas`, `ffn_replicas`) and each group's own
    `shortlist_size` best-by-`objectives.minimize` placements
    (`attn_shape`, `ep_shape`) are kept -- everything else is discarded
    here, cheaply, before any seeded evaluation runs.

    **Stage 2** (`streaming_regime`, conventionally seeded with
    `num_seeds > 1` -- expensive, per task 45's own resolution point)
    re-evaluates every shortlisted `Candidate` under `streaming_evaluator`,
    applies the real `objectives` (including its own floors) for the
    first time, ranks the survivors, and marks indistinguishability from
    the winner exactly as `plan()` itself does."""
    relaxed_objectives = _dc_replace(objectives, min_throughput_rps=0.0, slo_attainment_floor=0.0)
    stage1 = plan(topology, model, workload, hardware, relaxed_objectives, shortlist_regime,
                 shortlist_evaluator, attn_tp_values=attn_tp_values, ep_values=ep_values,
                 replica_ratios=replica_ratios, attn_dp_size_policy=attn_dp_size_policy)

    by_sizing: Dict[Tuple[int, int, int, int], List[dict]] = {}
    for row in stage1.ranked:
        by_sizing.setdefault(_sizing_key(row["candidate"]), []).append(row)

    shortlisted: List[dict] = []
    for _, rows in by_sizing.items():
        rows.sort(key=lambda r: r[relaxed_objectives.minimize])
        shortlisted.extend(rows[:shortlist_size])

    rejections: List[Rejection] = []
    unknown: List[Unknown] = []
    evaluated: List[dict] = []
    for row in shortlisted:
        candidate = row["candidate"]
        if not streaming_evaluator.can_evaluate(candidate):
            unknown.append(Unknown(candidate.key,
                "evaluator cannot price this candidate (outside its own coverage)"))
            continue
        r = streaming_evaluator.evaluate(candidate)
        if r.get("error"):
            rejections.append(Rejection(candidate.key, f"evaluation error: {r['error']}"))
            continue
        r["candidate"] = candidate
        if r["throughput_rps"] < objectives.min_throughput_rps:
            rejections.append(Rejection(candidate.key,
                f"throughput floor: {r['throughput_rps']:.3f} < {objectives.min_throughput_rps}"))
            continue
        if r["slo_attainment"] < objectives.slo_attainment_floor - 1e-9:
            rejections.append(Rejection(candidate.key,
                f"SLO: attainment {r['slo_attainment']:.3f} below required "
                f"{objectives.slo_attainment_floor}"))
            continue
        evaluated.append(r)

    evaluated.sort(key=lambda r: r[objectives.minimize])
    _mark_indistinguishable_from_winner(evaluated, objectives.minimize)
    winner = evaluated[0] if evaluated else None
    return TwoStagePlanResult(
        winner=winner, ranked=evaluated, shortlisted=shortlisted,
        shortlist_regime=shortlist_regime, streaming_regime=streaming_regime,
        shortlist_size=shortlist_size, stage1=stage1,
        stage2_rejections=rejections, stage2_unknown=unknown)
