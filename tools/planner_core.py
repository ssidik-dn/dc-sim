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

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

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
    ep_split: bool = False
    attn_replicas: int = 1
    ffn_replicas: int = 1

    @property
    def key(self) -> str:
        return (f"tp{self.attn_tp}_shape{'-'.join(map(str, self.attn_shape))}"
               f"_ep{self.ffn_ep}{'s' if self.ep_split else ''}"
               f"_ar{self.attn_replicas}_fr{self.ffn_replicas}")


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


def _attn_head_dim(model: ModelSpec) -> int:
    return model.head_dim if model.head_dim is not None else model.hidden_size // model.num_attention_heads


def attn_param_mem_bytes(model: ModelSpec, attn_tp: int) -> int:
    """DECODE_ATTN's own per-device parameter memory at this `attn_tp`,
    computed directly from `frontier/utils/param_counter.py`'s own
    formula (Q/K/V/O only -- DECODE_FFN's MLP/MoE weights are a separate
    cluster's memory, task 35's own S0 scoping point) and
    `frontier/scheduler/utils/memory_planner.py`'s own unconditional
    2-bytes/param assumption. Verified bit-for-bit against
    `ParamCounter.get_num_parameters_per_device()` for both
    Phi-tiny-MoE-instruct (671088640/335544320/167772160/100663296 params
    at tp=1/2/4/8) and Llama-3.1-405B-Instruct-FP8
    (71873593344/35936796672/17968398336/8984199168/4756340736/2642411520
    at tp=1/2/4/8/16/32) before use, not assumed from a formula alone."""
    head_dim = _attn_head_dim(model)
    q_per_worker = model.num_attention_heads / attn_tp
    kv_per_worker = -(-model.num_key_value_heads // attn_tp)  # ceil
    per_layer = (model.hidden_size * head_dim * (q_per_worker + 2 * kv_per_worker)
                + model.hidden_size * head_dim * q_per_worker)
    return int(2 * per_layer * model.num_layers)


def _kv_cache_page_bytes_per_layer(model: ModelSpec, attn_tp: int, block_size: int) -> int:
    """`frontier/scheduler/utils/memory_planner.py`'s own
    `_get_kv_cache_memory_per_layer_per_block`: 2 bytes/element x
    block_size x kv_factor x kv_heads_per_worker x head_dim."""
    head_dim = _attn_head_dim(model)
    kv_per_worker = -(-model.num_key_value_heads // attn_tp)
    return 2 * block_size * _KV_FACTOR * kv_per_worker * head_dim


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
    given (task 37's own S3)."""
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
class PlanResult:
    winner: Optional[dict]
    ranked: List[dict]
    rejections: List[Rejection]
    unknown: List[Unknown]


def plan(topology: Topology, model: ModelSpec, workload: Workload, hardware: Hardware,
        objectives: Objectives, evaluator: Evaluator, *, attn_tp_values: Tuple[int, ...] = None,
        ep_values: Tuple[int, ...] = None,
        replica_ratios: Tuple[Tuple[int, int], ...] = ((1, 1),)) -> PlanResult:
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
    """
    attn_tp_values = attn_tp_values or model.admissible_tp
    ep_values = ep_values or model.admissible_ep

    rejections: List[Rejection] = []
    unknown: List[Unknown] = []
    evaluated: List[dict] = []

    for attn_tp in attn_tp_values:
        num_blocks = feasible_num_blocks(model, hardware, attn_tp)
        if num_blocks is None:
            rejections.append(Rejection(f"tp{attn_tp}_*", "memory: infeasible at this margin"))
            continue
        shapes = enumerate_attn_shapes(topology, attn_tp)
        for shape in shapes:
            for ep in ep_values:
                for attn_replicas, ffn_replicas in replica_ratios:
                    if not lane_assignment_feasible(attn_replicas, ffn_replicas, ffn_replicas):
                        rejections.append(Rejection(
                            f"tp{attn_tp}_shape{shape}_ep{ep}_ar{attn_replicas}_fr{ffn_replicas}",
                            "lane assignment: attn_replicas*attn_dp_size < ffn_replicas"))
                        continue
                    candidate = Candidate(attn_tp, shape, ep, False, attn_replicas, ffn_replicas)
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
    winner = evaluated[0] if evaluated else None
    return PlanResult(winner=winner, ranked=evaluated, rejections=rejections, unknown=unknown)
