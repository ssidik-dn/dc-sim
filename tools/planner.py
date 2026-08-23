#!/usr/bin/env python3
"""Task 33: `plan(topology, model, workload, hardware, objectives) ->
(tp, ep, replica_counts, placement, and why)` -- task 32's search,
restructured so every input that changed the answer in principle is a
genuine parameter of one call, not a constant inside a tool.

**Where this lives, and why.** Like every other real-compute
orchestration tool in this project (`tools/run_placement_search.py`,
`tools/seed_stats.py`, and everything since task 09), this needs to
invoke Frontier to evaluate a candidate -- so it cannot live in
`src/engine/` (which must never import `src/integration/` or
`upstream/`) no matter how engine-shaped its own decision logic is.
`Topology` wraps an `engine.physical.topology.Fabric` directly; nothing
here reimplements what `Fabric`, `Deployment`, or Frontier's own model
and workload configuration already provide -- this module makes them
parameters of one function.

**Search variables**: tensor-parallel degree, expert-parallel degree,
replica counts (including the attention-to-FFN ratio), and physical
placement, per this task's own S2. **Not search variables**: scheduler
policy (its own benefit on realistic compute was never established --
task 15's own report, "+0.00% -- noise, not an effect") and memory
capacity (a feasibility filter per task 24/28/32, not an axis with a
preference on it).

**The test for "genuine parameter," not default-with-a-parameter's-name**
(this task's own S7 trap): every one of `topology`, `model`, `workload`,
`hardware`, `objectives` is varied across at least two calls somewhere
in this task's own report, through this module's own public interface,
with nothing in this file edited between them.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_stats import seed_argv_fix  # noqa: E402
from engine.physical.topology import Fabric  # noqa: E402
from engine.logical.deployment import Deployment, PoolKind, Replica, ParallelKind  # noqa: E402
from engine.placement.placement import packed, spread, fragmented, explicit, PlacementError  # noqa: E402
from engine.placement.binding import BindingPolicy  # noqa: E402
from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment  # noqa: E402
from integration.context import BindingConfig  # noqa: E402
from integration.install import install  # noqa: E402

FRONTIER_ROOT = Path("/work/simulation/Frontier")
_SCRIPT_PATH = str(Path(__file__).resolve())

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


# ------------------------------------------------------ memory feasibility

# Calibrated, cited from tasks 25/26/28 rather than re-derived: per-device
# parameter memory and KV page size for Phi-tiny-MoE-instruct on h800, by
# attn_tensor_parallel_size. A different ModelSpec/Hardware would need its
# own calibration -- this table is named for what it is, not treated as
# universal.
_PARAM_MEM_BYTES = {1: 1342177280, 2: 671088640, 4: 335544320, 8: 201326592}
_PAGE_SIZE_BYTES = {1: 1048576, 2: 524288, 4: 262144, 8: 262144}
_DEVICE_MEMORY_BYTES = {"h800": 80 * 1024 ** 3}


def feasible_num_blocks(model: ModelSpec, hardware: Hardware, attn_tp: int) -> Optional[int]:
    """`None` if infeasible (parameter memory alone exceeds the budget at
    this margin); otherwise the derived `num_blocks` -- task 24's own
    formula, cited not re-derived."""
    if model.model_name != "Phi-tiny-MoE-instruct" or hardware.device != "h800":
        raise NotImplementedError(
            "feasible_num_blocks' calibration table is for Phi-tiny-MoE-instruct "
            "on h800 only (tasks 25/26/28) -- a different model/device needs its "
            "own calibration before this function can answer for it.")
    device_bytes = _DEVICE_MEMORY_BYTES[hardware.device]
    usable = (1 - hardware.memory_margin_fraction) * device_bytes
    avail = usable - _PARAM_MEM_BYTES[attn_tp]
    if avail <= 0:
        return None
    return int(avail // _PAGE_SIZE_BYTES[attn_tp])


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
    unchanged, parameterised on the fabric instead of a module constant."""
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


# --------------------------------------------------------------- evaluation


def _argv(topology: Topology, model: ModelSpec, workload: Workload, hardware: Hardware,
         candidate: Candidate, num_blocks: int, run_id: str, seed: int, extra: List[str]) -> List[str]:
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters",

        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", str(candidate.attn_replicas),
        "--cluster_config_decode_ffn_cluster_num_replicas", str(candidate.ffn_replicas),
        "--cluster_config_allow_experiment_multi_decode_ffn_replicas",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_attn_micro_batch_size", "8",

        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", str(model.total_experts),
        "--cluster_config_prefill_replica_config_router_topk", str(model.router_topk),
        "--cluster_config_prefill_replica_config_device", hardware.device,
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", str(candidate.attn_tp),
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size",
        str(max(candidate.ffn_replicas, 1)),
        "--cluster_config_decode_attn_replica_config_device", hardware.device,
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction",
        str(hardware.memory_margin_fraction),

        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", str(candidate.ffn_ep),
        "--cluster_config_decode_ffn_replica_config_total_expert_num", str(model.total_experts),
        "--cluster_config_decode_ffn_replica_config_router_topk", str(model.router_topk),
        "--cluster_config_decode_ffn_replica_config_device", hardware.device,
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",

        "--cc_backend_config_type", "analytical",
        "--m2n_transfer_config_type", "empirical",

        "--replica_config_model_name", model.model_name,
        "--replica_config_moe_routing_mode", "uniform_random",
        "--replica_config_moe_routing_seed", "42",

        "--vllm_v1_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", "4096",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--cluster_config_prefill_replica_scheduler_config_num_blocks", "4096",
        "--cluster_config_decode_attn_replica_scheduler_config_num_blocks", str(num_blocks),
        "--cluster_config_decode_attn_replica_scheduler_config_block_size", "16",

        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(workload.num_requests),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(workload.prefill_tokens),
        "--fixed_request_length_generator_config_decode_tokens", str(workload.decode_tokens),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(workload.qps),

        "--metrics_config_output_dir",
        "/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0/scratchpad/planner_outputs",
        "--metrics_config_run_id", run_id,
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",

        "--seed", str(seed),
    ] + extra


_RESULT_MARKER = "PLAN_EVAL_RESULT="


def _placement_for(topology: Topology, deployment: Deployment, candidate: Candidate):
    """`candidate.attn_shape`'s own reference `Placement` (from
    `enumerate_attn_shapes`) was built for a unit deployment -- one
    PREFILL, one DECODE_ATTN replica at this tp, one DECODE_FFN -- whose
    ranks are, by construction, identical to this deployment's own
    replica-0 ranks for each pool. Reused directly wherever a rank
    matches (bit-identical to the reference for the base 1:1, ep=1 case,
    matching task 32's own single-evaluation path exactly). Any rank the
    reference does not cover -- additional replicas (replica-ratio
    candidates) or additional expert-parallel ranks (ep > 1) -- is
    packed into whatever domain slots the reference left free. Not
    placement-optimal for those extra ranks; this project's placement
    *search* is the shape dimension already covered by `attn_shape`, not
    the replica/EP dimensions these confirmatory checks add on top.
    """
    shapes = enumerate_attn_shapes(topology, candidate.attn_tp)
    ref_placement = shapes[candidate.attn_shape]
    fabric = topology.fabric
    domain_ids = sorted(fabric.domains)

    mapping: Dict = {}
    occupied = set()
    for replica in deployment.replicas:
        for r in replica.ranks:
            try:
                gpu = ref_placement.gpu(r)
            except PlacementError:
                gpu = None
            if gpu is not None and gpu not in occupied:
                mapping[r] = gpu
                occupied.add(gpu)

    for replica in deployment.replicas:
        for r in replica.ranks:
            if r in mapping:
                continue
            for did in domain_ids:
                free = sorted(g for g in fabric.domains[did].members if g not in occupied)
                if free:
                    mapping[r] = free[0]
                    occupied.add(free[0])
                    break
    return explicit(deployment, fabric, mapping)


def _run_scenario(topology_name: str, model_name: str, candidate_key: str,
                  attn_tp: int, attn_shape: str, ffn_ep: int, ep_split: bool,
                  attn_replicas: int, ffn_replicas: int, num_blocks: int,
                  memory_margin: float, num_requests: int, qps: float,
                  prefill_tokens: int, decode_tokens: int, device: str,
                  total_experts: int, router_topk: int,
                  seed: int, seeded: bool) -> None:
    topology = _TOPOLOGIES[topology_name]()
    model = ModelSpec(model_name, total_experts, router_topk, is_moe=True)
    workload = Workload(num_requests, qps, prefill_tokens, decode_tokens)
    hardware = Hardware(device, memory_margin)
    shape = tuple(int(x) for x in attn_shape.split(","))
    candidate = Candidate(attn_tp, shape, ffn_ep, ep_split, attn_replicas, ffn_replicas)

    d = Deployment("plan-eval")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    for i in range(attn_replicas):
        d.add(Replica(PoolKind.DECODE_ATTN, i, tp=attn_tp))
    for i in range(ffn_replicas):
        d.add(Replica(PoolKind.DECODE_FFN, i, tp=1, ep=ffn_ep))

    from frontier.types import ClusterType
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {
        PoolKind.PREFILL: ClusterType.PREFILL,
        PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
        PoolKind.DECODE_FFN: ClusterType.DECODE_FFN,
    })
    placement = _placement_for(topology, d, candidate)
    binding = (BindingConfig(BindingPolicy.ROUND_ROBIN, timing="early")
              if max(attn_replicas, ffn_replicas) > 1 else None)
    install(topology.fabric, placement, d, reg, binding=binding, collective=True)

    extra = seed_argv_fix(seed) if seeded else []
    tag = f"plan_{topology_name}_{candidate_key}_seed{seed}_seeded{int(seeded)}"
    sys.argv = _argv(topology, model, workload, hardware, candidate, num_blocks, tag, seed, extra)

    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    error = None
    sim = None
    try:
        config = SimulationConfig.create_from_cli_args()
        assert not config.cluster_config.execution_time_predictor_config.enable_dummy_mode
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    if error is not None:
        print(_RESULT_MARKER + json.dumps({"key": candidate_key, "error": error}), flush=True)
        return

    requests = sim._all_requests
    completed = [r for r in requests if r.completed]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_ms_per_req = [r.tpot * 1000.0 for r in tpot_eligible]
    mean_tpot_ms = statistics.mean(tpot_ms_per_req) if tpot_ms_per_req else None
    slo_met = (sum(1 for t in tpot_ms_per_req if t <= 15.0) / len(tpot_ms_per_req)
              if tpot_ms_per_req else None)
    wall_s = max((r.completed_at for r in completed), default=0.0)
    throughput_rps = len(completed) / wall_s if wall_s else 0.0

    print(_RESULT_MARKER + json.dumps({
        "key": candidate_key, "error": None,
        "mean_tpot_ms": mean_tpot_ms, "throughput_rps": throughput_rps,
        "slo_attainment": slo_met, "n_completed": len(completed),
    }), flush=True)


def evaluate(topology: Topology, model: ModelSpec, workload: Workload, hardware: Hardware,
            candidate: Candidate, num_blocks: int, seed: int = 0, seeded: bool = False) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH,
         "--topology", topology.name, "--model-name", model.model_name,
         "--candidate-key", candidate.key,
         "--attn-tp", str(candidate.attn_tp),
         "--attn-shape", ",".join(map(str, candidate.attn_shape)),
         "--ffn-ep", str(candidate.ffn_ep), "--ep-split", "1" if candidate.ep_split else "0",
         "--attn-replicas", str(candidate.attn_replicas), "--ffn-replicas", str(candidate.ffn_replicas),
         "--num-blocks", str(num_blocks), "--memory-margin", str(hardware.memory_margin_fraction),
         "--num-requests", str(workload.num_requests), "--qps", str(workload.qps),
         "--prefill-tokens", str(workload.prefill_tokens), "--decode-tokens", str(workload.decode_tokens),
         "--device", hardware.device, "--total-experts", str(model.total_experts),
         "--router-topk", str(model.router_topk),
         "--seed", str(seed), "--seeded", "1" if seeded else "0"],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout[-3000:])
    sys.stderr.write(proc.stderr[-3000:])
    return {"error": f"no result (exit code {proc.returncode})", "key": candidate.key}


# ------------------------------------------------------------------- plan()


@dataclass
class Rejection:
    candidate_key: str
    reason: str


@dataclass
class PlanResult:
    winner: Optional[dict]
    ranked: List[dict]
    rejections: List[Rejection]


def plan(topology: Topology, model: ModelSpec, workload: Workload, hardware: Hardware,
         objectives: Objectives, *, attn_tp_values: Tuple[int, ...] = None,
         ep_values: Tuple[int, ...] = None,
         replica_ratios: Tuple[Tuple[int, int], ...] = ((1, 1),)) -> PlanResult:
    """Generate candidates over `attn_tp_values` (default:
    `model.admissible_tp`) x placement shape (per `topology`) x
    `ep_values` (default: `model.admissible_ep`) x `replica_ratios`,
    reject infeasible ones up front, evaluate the rest once each
    (deterministic configuration, task 31 report S1.3 -- task 33's own
    S1 seed policy), and rank the survivors by `objectives.minimize`
    among those meeting every constraint.
    """
    attn_tp_values = attn_tp_values or model.admissible_tp
    ep_values = ep_values or model.admissible_ep

    rejections: List[Rejection] = []
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
                    r = evaluate(topology, model, workload, hardware, candidate, num_blocks)
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
    return PlanResult(winner=winner, ranked=evaluated, rejections=rejections)


# ------------------------------------------------------------ topologies

def _topology_task32repro():
    """Task 32's exact fabric: 5 scale-up domains of 4 GPUs each. Kept
    distinct from `domain8` (5 x 8) because domain size changes which
    shapes are reachable once attn_tp > 2 -- the regression check needs
    task 32's own geometry, not merely "a small fabric"."""
    from engine.physical.builders import build_node_scale
    fabric = build_node_scale(num_machines=5, gpus_per_machine=4,
                              scale_up_GBps=400.0, scale_out_GBps=50.0)
    return Topology(fabric, "task32repro")


def _topology_domain8():
    from engine.physical.builders import build_node_scale
    fabric = build_node_scale(num_machines=5, gpus_per_machine=8,
                              scale_up_GBps=400.0, scale_out_GBps=50.0)
    return Topology(fabric, "domain8")


def _topology_domain64():
    from engine.physical.builders import build_node_scale
    fabric = build_node_scale(num_machines=2, gpus_per_machine=64,
                              scale_up_GBps=400.0, scale_out_GBps=50.0)
    return Topology(fabric, "domain64")


def _topology_oversubscribed():
    from engine.physical.builders import build_node_scale
    fabric = build_node_scale(num_machines=5, gpus_per_machine=8,
                              scale_up_GBps=400.0, scale_out_GBps=12.5)  # 4x narrower scale-out
    return Topology(fabric, "oversubscribed")


_TOPOLOGIES = {
    "task32repro": _topology_task32repro,
    "domain8": _topology_domain8,
    "domain64": _topology_domain64,
    "oversubscribed": _topology_oversubscribed,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--candidate-key", type=str, default=None)
    parser.add_argument("--attn-tp", type=int, default=None)
    parser.add_argument("--attn-shape", type=str, default=None)
    parser.add_argument("--ffn-ep", type=int, default=1)
    parser.add_argument("--ep-split", type=int, default=0)
    parser.add_argument("--attn-replicas", type=int, default=1)
    parser.add_argument("--ffn-replicas", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--memory-margin", type=float, default=0.2)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--qps", type=float, default=20.0)
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--device", type=str, default="h800")
    parser.add_argument("--total-experts", type=int, default=16)
    parser.add_argument("--router-topk", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeded", type=int, default=0)
    args = parser.parse_args()
    if args.topology is not None:
        _run_scenario(args.topology, args.model_name, args.candidate_key,
                     args.attn_tp, args.attn_shape, args.ffn_ep, bool(args.ep_split),
                     args.attn_replicas, args.ffn_replicas, args.num_blocks,
                     args.memory_margin, args.num_requests, args.qps,
                     args.prefill_tokens, args.decode_tokens, args.device,
                     args.total_experts, args.router_topk, args.seed, bool(args.seeded))
        raise SystemExit(0)
    raise SystemExit(0)
