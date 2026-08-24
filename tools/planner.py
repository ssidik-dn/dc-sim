#!/usr/bin/env python3
"""Task 33's planner, evaluator half. `tools/planner_core.py` holds the
search -- candidate representation, feasibility, shape enumeration,
constraint filtering, ranking -- and knows nothing about Frontier. This
module holds the only thing that does: `SimulationEvaluator`, which
prices a candidate by invoking Frontier in a subprocess, plus the CLI
entry point that subprocess actually runs.

**Where this lives, and why.** Like every other real-compute
orchestration tool in this project (`tools/run_placement_search.py`,
`tools/seed_stats.py`, and everything since task 09), this needs to
invoke Frontier to evaluate a candidate -- so it cannot live in
`src/engine/` (which must never import `src/integration/` or
`upstream/`) no matter how engine-shaped `planner_core`'s own decision
logic is.

`plan()` here is a thin wrapper over `planner_core.plan()` that defaults
its `evaluator` argument to a fresh `SimulationEvaluator` -- the reason
every call site from task 33/36 (which never pass an evaluator at all)
keeps working unchanged (task 37's own acceptance requirement).
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner_core import (  # noqa: E402
    Topology, ModelSpec, Workload, Hardware, Objectives, Candidate,
    Rejection, Unknown, PlanResult, Evaluator,
    feasible_num_blocks, lane_assignment_feasible, enumerate_attn_shapes,
    enumerate_joint_arrangements,
    attn_param_mem_bytes,
    plan as _core_plan,
)
from seed_stats import seed_argv_fix  # noqa: E402
from engine.logical.deployment import Deployment, PoolKind, Replica  # noqa: E402
from engine.placement.placement import explicit, PlacementError  # noqa: E402
from engine.placement.binding import BindingPolicy  # noqa: E402
from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment  # noqa: E402
from integration.context import BindingConfig  # noqa: E402
from integration.install import install  # noqa: E402

FRONTIER_ROOT = Path("/work/simulation/Frontier")
_SCRIPT_PATH = str(Path(__file__).resolve())


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
    """`(candidate.attn_shape, candidate.ep_shape)`'s own reference
    `Placement` (from `enumerate_joint_arrangements`, task 44) was built
    for a unit deployment -- one PREFILL, one DECODE_ATTN replica at
    this tp, one DECODE_FFN replica at this ep -- whose ranks are, by
    construction, identical to this deployment's own replica-0 ranks
    for each pool. Reused directly wherever a rank matches (bit-identical
    to the reference for the base 1:1 case, matching task 32's own
    single-evaluation path exactly, and -- task 44's own extension --
    for replica-0's own expert-parallel ranks too, not just its tensor-
    parallel ones). Any rank the reference does not cover -- additional
    replicas, from a `replica_ratios` candidate beyond replica 0 of
    either pool -- is packed into whatever domain slots the reference
    left free. Not placement-optimal for those extra ranks; task 41's
    own established scope is the replica dimension, not this task's own
    (expert placement within replica 0, which the joint reference now
    covers directly).
    """
    joint = enumerate_joint_arrangements(topology, candidate.attn_tp, candidate.ffn_ep)
    ref_placement = joint[(candidate.attn_shape, candidate.ep_shape)]
    fabric = topology.fabric
    domain_ids = sorted(fabric.domains)

    mapping: dict = {}
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
                  attn_tp: int, attn_shape: str, ffn_ep: int, ep_shape: str,
                  attn_replicas: int, ffn_replicas: int, num_blocks: int,
                  memory_margin: float, num_requests: int, qps: float,
                  prefill_tokens: int, decode_tokens: int, device: str,
                  total_experts: int, router_topk: int, is_moe: bool,
                  hidden_size: int, num_attention_heads: int,
                  num_key_value_heads: int, num_layers: int, head_dim: Optional[int],
                  seed: int, seeded: bool) -> None:
    topology = _TOPOLOGIES[topology_name]()
    model = ModelSpec(model_name, total_experts, router_topk, is_moe=is_moe,
                      hidden_size=hidden_size, num_attention_heads=num_attention_heads,
                      num_key_value_heads=num_key_value_heads, num_layers=num_layers,
                      head_dim=head_dim)
    workload = Workload(num_requests, qps, prefill_tokens, decode_tokens)
    hardware = Hardware(device, memory_margin)
    shape = tuple(int(x) for x in attn_shape.split(","))
    ep_shape_t = tuple(int(x) for x in ep_shape.split(","))
    candidate = Candidate(attn_tp, shape, ffn_ep, ep_shape_t, attn_replicas, ffn_replicas)

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
    """Runs one candidate through Frontier in a subprocess and returns
    its result dict. Kept as a free function, not folded into
    `SimulationEvaluator` alone, because seeded re-runs (tasks 31-36's
    own established method) need `seed`/`seeded`, which the `Evaluator`
    protocol's own `evaluate(candidate)` deliberately does not carry --
    a seed is a property of *how* a simulator is asked to run, not of
    the candidate itself, and a telemetry evaluator has no seed concept
    at all. `SimulationEvaluator.evaluate` calls this with `seed=0,
    seeded=False`, task 31/32's own established deterministic policy."""
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH,
         "--topology", topology.name, "--model-name", model.model_name,
         "--candidate-key", candidate.key,
         "--attn-tp", str(candidate.attn_tp),
         "--attn-shape", ",".join(map(str, candidate.attn_shape)),
         "--ffn-ep", str(candidate.ffn_ep),
         "--ep-shape", ",".join(map(str, candidate.ep_shape)),
         "--attn-replicas", str(candidate.attn_replicas), "--ffn-replicas", str(candidate.ffn_replicas),
         "--num-blocks", str(num_blocks), "--memory-margin", str(hardware.memory_margin_fraction),
         "--num-requests", str(workload.num_requests), "--qps", str(workload.qps),
         "--prefill-tokens", str(workload.prefill_tokens), "--decode-tokens", str(workload.decode_tokens),
         "--device", hardware.device, "--total-experts", str(model.total_experts),
         "--router-topk", str(model.router_topk), "--is-moe", "1" if model.is_moe else "0",
         "--hidden-size", str(model.hidden_size),
         "--num-attention-heads", str(model.num_attention_heads),
         "--num-key-value-heads", str(model.num_key_value_heads),
         "--num-layers", str(model.num_layers),
         "--head-dim", str(model.head_dim) if model.head_dim is not None else "none",
         "--seed", str(seed), "--seeded", "1" if seeded else "0"],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout[-3000:])
    sys.stderr.write(proc.stderr[-3000:])
    return {"error": f"no result (exit code {proc.returncode})", "key": candidate.key}


class SimulationEvaluator:
    """The only `Evaluator` this project has: prices a candidate by
    running Frontier in a subprocess. Bound to one
    topology/model/workload/hardware at construction, matching the scope
    one `plan()` call already fixes them to.

    `can_evaluate` answers from `model.profiled_tp` -- Task 35's own
    finding that every model in this checkout, on every device with
    real profiles, is profiled at tp in {1,2,4,8} only, because nobody
    overrode the profiler's own default sweep
    (`frontier/profiling/linear_op/main.py`'s own
    `--num_tensor_parallel_workers`, default `[1,2,4,8]`). This is a
    real, current limit, not speculative scaffolding: a candidate at
    tp=16 is not rejected by this evaluator, it is *unknown* to it --
    `plan()` keeps that distinct (task 37's own known trap).

    Task 41's own verified finding, added here for the same reason:
    this evaluator cannot price `attn_replicas > 1` at any admissible
    `attn_tp` (every one of them is > 1 -- tp=1 is memory-infeasible for
    every model this project's tasks 32-40 have used) -- confirmed by
    running it, not assumed from reading the code:
    `populate_from_deployment` registers each DECODE_ATTN replica's own
    TP group under the SAME `(cluster_type, comm_domain, num_devices)`
    key (`src/integration/cc_backend/comm_groups.py`'s own docstring:
    "Frontier's cc_backend calls carry a device count and a
    parallelism-domain label -- never a rank identity"), so a second
    replica at the same `attn_tp` collides and
    `CommGroupRegistry.register` raises `CommGroupError`. This is a
    real limit of *this* evaluator's own pipeline, not of the (model,
    degree, ratio) request or of available memory -- exactly what
    `can_evaluate() -> False` (Unknown) exists to report, matching
    task 37's own two-methods design. A different evaluator (one that
    does not route DECODE_ATTN's own TP cost through
    `CommGroupRegistry`, or a telemetry-backed one observing an
    already-running multi-replica deployment) would not necessarily
    share this limit.

    `ffn_replicas > 1` is, despite first appearances, NOT similarly
    restricted -- an earlier version of this finding believed it was,
    from a probe that ran several evaluations in one Python process and
    was fooled by cross-call state leakage into seeing a crash that a
    single, isolated run does not reproduce (task 41's own report,
    S5, explains the mistake and how it was caught). Confirmed clean,
    one subprocess per candidate, at `ffn_replicas` up to 16: this
    evaluator prices it exactly like any other candidate.
    """

    def __init__(self, topology: Topology, model: ModelSpec, workload: Workload, hardware: Hardware):
        self.topology = topology
        self.model = model
        self.workload = workload
        self.hardware = hardware

    def can_evaluate(self, candidate: Candidate) -> bool:
        return candidate.attn_tp in self.model.profiled_tp and candidate.attn_replicas == 1

    def evaluate(self, candidate: Candidate) -> dict:
        num_blocks = feasible_num_blocks(self.model, self.hardware, candidate.attn_tp)
        return evaluate(self.topology, self.model, self.workload, self.hardware,
                        candidate, num_blocks, seed=0, seeded=False)


def plan(topology: Topology, model: ModelSpec, workload: Workload, hardware: Hardware,
        objectives: Objectives, evaluator: Optional[Evaluator] = None, **kwargs) -> PlanResult:
    """`planner_core.plan()`, defaulting `evaluator` to a fresh
    `SimulationEvaluator` bound to this call's own
    topology/model/workload/hardware -- the reason every call site from
    tasks 33/36 (none of which pass an evaluator) reproduces unchanged."""
    if evaluator is None:
        evaluator = SimulationEvaluator(topology, model, workload, hardware)
    return _core_plan(topology, model, workload, hardware, objectives, evaluator, **kwargs)


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


def _topology_domain8_40gpu():
    """Task 36's own Fabric A: 5 domains x 8 GPUs = 40 GPUs total."""
    from engine.physical.builders import build_node_scale
    fabric = build_node_scale(num_machines=5, gpus_per_machine=8,
                              scale_up_GBps=400.0, scale_out_GBps=50.0)
    return Topology(fabric, "domain8_40gpu")


def _topology_domain4_40gpu():
    """Task 36's own Fabric B: 10 domains x 4 GPUs = 40 GPUs total --
    same total capacity as Fabric A, smaller domains only (this task's
    own known trap: equal total GPUs, or the comparison confounds domain
    size with capacity)."""
    from engine.physical.builders import build_node_scale
    fabric = build_node_scale(num_machines=10, gpus_per_machine=4,
                              scale_up_GBps=400.0, scale_out_GBps=50.0)
    return Topology(fabric, "domain4_40gpu")


def _topology_clos_2tier_128():
    """Task 40's own Fabric A: a two-tier leaf-spine Clos, radix 16 --
    spines=8, leaves=16, hosts_per_leaf=8, total hosts (=GPUs, at
    gpus_per_machine=1) = 16^2/2 = 128."""
    from engine.infragraph.blueprints import clos_fat_tree_fabric
    fabric = clos_fat_tree_fabric(switch_radix=16, depth=2, gpus_per_machine=1,
                                  nics_per_machine=1, scale_up_GBps=400.0,
                                  scale_up_latency_ns=936.25, nic_gbps=400.0,
                                  egress_latency_ns=2000.0, scale_out_GBps=50.0,
                                  scale_out_latency_ns=5000.0, name="clos_2tier_128")
    return Topology(fabric, "clos_2tier_128")


def _topology_clos_3tier_128():
    """Task 40's own Fabric B: a three-tier fat tree, radix 8 -- pods=8,
    edges/pod=aggs/pod=4, core=16, hosts_per_edge=4, total hosts (=GPUs,
    at gpus_per_machine=1) = 8^3/4 = 128 -- the same total GPU count as
    Fabric A, so the comparison is tier structure, not capacity (task 36's
    own known trap, reused here)."""
    from engine.infragraph.blueprints import clos_fat_tree_fabric
    fabric = clos_fat_tree_fabric(switch_radix=8, depth=3, gpus_per_machine=1,
                                  nics_per_machine=1, scale_up_GBps=400.0,
                                  scale_up_latency_ns=936.25, nic_gbps=400.0,
                                  egress_latency_ns=2000.0, scale_out_GBps=50.0,
                                  scale_out_latency_ns=5000.0, name="clos_3tier_128")
    return Topology(fabric, "clos_3tier_128")


_TOPOLOGIES = {
    "task32repro": _topology_task32repro,
    "domain8_40gpu": _topology_domain8_40gpu,
    "domain4_40gpu": _topology_domain4_40gpu,
    "domain8": _topology_domain8,
    "domain64": _topology_domain64,
    "oversubscribed": _topology_oversubscribed,
    "clos_2tier_128": _topology_clos_2tier_128,
    "clos_3tier_128": _topology_clos_3tier_128,
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
    parser.add_argument("--ep-shape", type=str, default="1")
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
    parser.add_argument("--is-moe", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--num-attention-heads", type=int, default=0)
    parser.add_argument("--num-key-value-heads", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=0)
    parser.add_argument("--head-dim", type=str, default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeded", type=int, default=0)
    args = parser.parse_args()
    if args.topology is not None:
        head_dim = None if args.head_dim == "none" else int(args.head_dim)
        _run_scenario(args.topology, args.model_name, args.candidate_key,
                     args.attn_tp, args.attn_shape, args.ffn_ep, args.ep_shape,
                     args.attn_replicas, args.ffn_replicas, args.num_blocks,
                     args.memory_margin, args.num_requests, args.qps,
                     args.prefill_tokens, args.decode_tokens, args.device,
                     args.total_experts, args.router_topk, bool(args.is_moe),
                     args.hidden_size, args.num_attention_heads,
                     args.num_key_value_heads, args.num_layers, head_dim,
                     args.seed, bool(args.seeded))
        raise SystemExit(0)
    raise SystemExit(0)
