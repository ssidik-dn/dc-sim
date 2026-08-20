#!/usr/bin/env python3
"""Task 14 spec S3.2: does binding change real serving behaviour?

Unit tests establish that `bind()` is deterministic and that NEAREST beats
ROUND_ROBIN on a hand-verified fabric (tests/test_binding.py). Neither
establishes that any of this matters to a real Frontier run -- that needs a
pd-disaggregation scenario with more than one DECODE replica, which is new
ground for this project: every prior KV/M2N measurement (tasks 09-13) used
exactly one replica per pool.

Scenario: one PREFILL replica, four DECODE replicas, on a `build_node_scale`
fabric where they are not equidistant from the source -- DECODE replica 0
shares PREFILL's scale-up domain ("near"); replicas 1-3 sit on three other
machines, each one scale-out hop away ("far", and symmetric among
themselves). This is the same shape tests/test_binding.py's
test_nearest_beats_round_robin_on_a_split_fabric hand-verifies, just with a
real Frontier run wrapped around it instead of a bare `bind()` call.

For each of the three policies that make sense in an automatic sweep
(ROUND_ROBIN, LEAST_LOADED, NEAREST -- EXPLICIT is excluded, see main()'s
docstring) and both timings (early, late), this script reports:

  - mean inter-token latency (tpot) and mean KV transfer time, both already
    single per-request quantities in Frontier's own units (task 12's
    total-vs-per-token distinction does not apply here: neither number needs
    rescaling to be compared against the other).
  - the distribution of bindings this project's own policy made, from each
    predictor's `.bindings` (empty/all-None under "late" -- see
    binding_support.py's docstring for why: late timing never commits to
    one destination).
  - whether our binding (timing="early" only; "late" never picks one)
    agrees with the replica Frontier's own RoundRobinClusterScheduler
    actually assigned that request to -- read from
    ClusterScheduleEvent's own request_mapping via a read-only monkey-patch,
    the same instrumentation pattern as tasks 11-13 (predictor.calls,
    activation_size_bytes observation).

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally for model-config/profile resolution, same as tasks 12/13:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_binding_study.py

Real compute is not the point here (the binding decision and its fabric
cost are); dummy execution-time mode is used throughout, same as task 09.
Nothing under `upstream/`, `src/engine/`, or the predictors is modified --
all instrumentation below is layered on from outside, at the class object.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

FRONTIER_ROOT = Path("/work/simulation/Frontier")

from engine.logical.deployment import Deployment, PoolKind, Replica  # noqa: E402
from engine.physical.builders import build_node_scale  # noqa: E402
from engine.physical.topology import GpuId  # noqa: E402
from engine.placement.binding import BindingPolicy  # noqa: E402
from engine.placement.placement import explicit  # noqa: E402

from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment  # noqa: E402
from integration.context import BindingConfig  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/binding_study_outputs")

SCALE_UP_GBPS = 400.0
SCALE_OUT_GBPS = 50.0
NUM_DECODE_REPLICAS = 4
NUM_REQUESTS = 12
DECODE_TOKENS = 8

POLICIES = (BindingPolicy.ROUND_ROBIN, BindingPolicy.LEAST_LOADED, BindingPolicy.NEAREST)
TIMINGS = ("early", "late")


def _engine_deployment_and_registry():
    d = Deployment("binding-study")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    for i in range(NUM_DECODE_REPLICAS):
        d.add(Replica(PoolKind.DECODE, i, tp=1))
    reg = CommGroupRegistry()
    from frontier.types import ClusterType
    populate_from_deployment(reg, d, {PoolKind.PREFILL: ClusterType.PREFILL,
                                      PoolKind.DECODE: ClusterType.DECODE})
    return d, reg


def _placement(fabric, deployment):
    """Replica 0 shares the source's scale-up domain; replicas 1-3 are each
    on their own, different machine -- symmetric among themselves, so
    NEAREST always prefers replica 0 and has nothing to say among the rest,
    the same shape as test_nearest_beats_round_robin_on_a_split_fabric."""
    prefill_rank = deployment.replicas[0].ranks[0]
    decode_ranks = [r.ranks[0] for r in deployment.replicas[1:]]
    mapping = {prefill_rank: GpuId(0, 0), decode_ranks[0]: GpuId(0, 1)}
    for i, rank in enumerate(decode_ranks[1:], start=1):
        mapping[rank] = GpuId(i, 0)
    return explicit(deployment, fabric, mapping)


def _argv(run_id: str) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", str(NUM_DECODE_REPLICAS),
        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", "1",
        "--cluster_config_prefill_replica_config_router_topk", "1",
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",
        "--cluster_config_decode_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_decode_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_replica_config_total_expert_num", "1",
        "--cluster_config_decode_replica_config_router_topk", "1",
        "--cluster_config_decode_replica_config_device", "h800",
        "--cluster_config_decode_replica_config_memory_margin_fraction", "0.2",
        "--cc_backend_config_type", "analytical",
        "--kv_cache_transfer_config_type", "empirical",
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--replica_config_moe_routing_mode", "simulation",
        "--replica_config_moe_routing_seed", "42",
        "--replica_scheduler_config_type", "vllm_v1",
        "--decode_cuda_graph_mode", "none",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", "512",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "2048",
        "--fixed_request_length_generator_config_decode_tokens", str(DECODE_TOKENS),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1.0",
        "--metrics_config_output_dir", str(OUTPUT_DIR),
        "--metrics_config_run_id", run_id,
        "--no-metrics_config_write_metrics",
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_utilization_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "1.0",
    ]


_RESULT_MARKER = "BINDING_STUDY_RESULT="


def _run_scenario(policy_name: str, timing: str) -> None:
    from frontier.kv_cache_transfer.base_kv_cache_transfer_predictor import (
        BaseKVCacheTransferPredictor)
    from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
        RoundRobinClusterScheduler)
    from frontier.types import ClusterType

    policy = BindingPolicy(policy_name)
    fabric = build_node_scale(num_machines=NUM_DECODE_REPLICAS, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    placement = _placement(fabric, deployment)
    install(fabric, placement, deployment, registry,
           binding=BindingConfig(policy, timing=timing))

    # Read-only instrumentation, layered on from outside (tasks 11-13's
    # pattern): which request each KV transfer call was for, in call order,
    # to zip against EngineKVCacheTransferPredictor.bindings afterward.
    request_order: list[int] = []
    _original_for_request = BaseKVCacheTransferPredictor.get_transfer_info_for_request

    def _observing_get_transfer_info_for_request(self, source_cluster_type,
                                                  target_cluster_type, request, replica_config):
        request_order.append(request.id)
        return _original_for_request(self, source_cluster_type, target_cluster_type,
                                     request, replica_config)

    BaseKVCacheTransferPredictor.get_transfer_info_for_request = (
        _observing_get_transfer_info_for_request)

    # Frontier's own real destination choice: DECODE's RoundRobinClusterScheduler
    # assigns (replica_id, dp_id, request) tuples in `schedule()` -- see
    # round_robin_cluster_scheduler.py's `_schedule_decode_lane_round_robin`.
    # This is Frontier's actual scheduling decision, recorded read-only.
    frontier_choice: dict[int, int] = {}
    _original_schedule = RoundRobinClusterScheduler.schedule

    def _observing_schedule(self):
        mapping = _original_schedule(self)
        if self._cluster_type == ClusterType.DECODE:
            for replica_id, dp_id, request in mapping:
                if request is not None:
                    frontier_choice[request.id] = replica_id
        return mapping

    RoundRobinClusterScheduler.schedule = _observing_schedule

    try:
        sys.argv = _argv(f"binding_study_{policy_name}_{timing}")
        from frontier.config import SimulationConfig
        from frontier.simulator import Simulator
        from frontier.utils.random import set_seeds

        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
    finally:
        BaseKVCacheTransferPredictor.get_transfer_info_for_request = _original_for_request
        RoundRobinClusterScheduler.schedule = _original_schedule

    requests = sim._all_requests
    kv_time_s = [r.kv_cache_transfer_time for r in requests]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]

    schedulers = getattr(sim._global_scheduler, "_cluster_schedulers", {})
    predictor = None
    for scheduler in schedulers.values():
        p = getattr(scheduler, "_kv_cache_transfer_predictor", None)
        if p is not None:
            predictor = p
            break

    bindings = predictor.bindings if predictor else []
    our_choice = dict(zip(request_order, bindings))

    # Frontier's Replica.id (frontier/entities/base_entity.py's
    # generate_id) is a single counter shared across every cluster type,
    # not reset per cluster -- the PREFILL cluster's one replica is built
    # first and takes id 0, so DECODE's four replicas land on 1-4, not
    # 0-3. Our own replica_id is assigned per-pool (register_pool's
    # `len(existing)`, task 14 S2.2). Raw ids are therefore not
    # comparable; both schemes number DECODE's replicas in the same
    # construction order (Cluster.__init__'s `for _ in range(num_replicas)`
    # matches deployment.add()'s call order), so subtracting the lowest id
    # each scheme actually used recovers a common 0-based index.
    frontier_offset = min(frontier_choice.values()) if frontier_choice else 0

    agreements = 0
    compared = 0
    for rid, ours in our_choice.items():
        if ours is None:
            continue  # late timing never commits to one
        theirs = frontier_choice.get(rid)
        if theirs is None:
            continue
        compared += 1
        if ours == theirs - frontier_offset:
            agreements += 1

    our_distribution: dict[int, int] = {}
    for v in bindings:
        if v is not None:
            our_distribution[v] = our_distribution.get(v, 0) + 1

    frontier_distribution: dict[int, int] = {}
    for v in frontier_choice.values():
        frontier_distribution[v] = frontier_distribution.get(v, 0) + 1

    print(_RESULT_MARKER + json.dumps({
        "mean_kv_time_s": mean(kv_time_s) if kv_time_s else None,
        "mean_tpot_s": mean(tpot_s) if tpot_s else None,
        "n_requests": len(requests),
        "n_tpot": len(tpot_s),
        "our_distribution": our_distribution,
        "frontier_distribution": frontier_distribution,
        "agreements": agreements,
        "compared": compared,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(policy: BindingPolicy, timing: str) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--policy", policy.value, "--timing", timing],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise RuntimeError(f"policy={policy.value!r} timing={timing!r} produced no result "
                       f"(exit code {proc.returncode}); see output above")


def main() -> int:
    """EXPLICIT is left out of the sweep: it requires a mapping registered
    into `BindingState.explicit_map` by something outside `binding.py`
    itself before `bind()` can be called at all (see
    tests/test_binding.py -- no test drives EXPLICIT through `price_transfer`
    for exactly this reason). Sweeping it here would mean this script
    deciding the very mapping the policy is supposed to produce, which is
    circular, not a fourth data point.
    """
    results = {}
    for policy in POLICIES:
        for timing in TIMINGS:
            r = _run_scenario_in_subprocess(policy, timing)
            results[(policy, timing)] = r
            print(f"[{policy.value:<12}/{timing:<5}] mean_kv_time="
                 f"{r['mean_kv_time_s']*1000:9.6f} ms  mean_tpot="
                 f"{r['mean_tpot_s']*1000:9.6f} ms  our_dist={r['our_distribution']}  "
                 f"frontier_dist={r['frontier_distribution']}  "
                 f"agree={r['agreements']}/{r['compared']}")

    print()
    print("early vs late, same policy (the cost of pricing without a committed "
         "destination):")
    for policy in POLICIES:
        early, late = results[(policy, "early")], results[(policy, "late")]
        d_kv = late["mean_kv_time_s"] - early["mean_kv_time_s"]
        d_tpot = late["mean_tpot_s"] - early["mean_tpot_s"]
        print(f"  {policy.value:<12} kv: early={early['mean_kv_time_s']*1000:9.6f} ms  "
             f"late={late['mean_kv_time_s']*1000:9.6f} ms  delta={d_kv*1000:+9.6f} ms  "
             f"({100*d_kv/early['mean_kv_time_s']:+6.2f}%)   "
             f"tpot delta={d_tpot*1000:+9.6f} ms "
             f"({100*d_tpot/early['mean_tpot_s']:+6.2f}%)")

    print()
    print("nearest vs round_robin (same timing):")
    for timing in TIMINGS:
        nearest, rr = results[(BindingPolicy.NEAREST, timing)], results[(BindingPolicy.ROUND_ROBIN, timing)]
        d_kv = rr["mean_kv_time_s"] - nearest["mean_kv_time_s"]
        print(f"  [{timing}] nearest={nearest['mean_kv_time_s']*1000:9.6f} ms  "
             f"round_robin={rr['mean_kv_time_s']*1000:9.6f} ms  "
             f"nearest saves {d_kv*1000:9.6f} ms/transfer "
             f"({100*d_kv/rr['mean_kv_time_s']:6.2f}%)")

    print()
    print("agreement with Frontier's own actual replica choice (early timing only; "
         "late never commits to one):")
    for policy in POLICIES:
        r = results[(policy, "early")]
        rate = r["agreements"] / r["compared"] if r["compared"] else float("nan")
        print(f"  {policy.value:<12} {r['agreements']}/{r['compared']} "
             f"({100*rate:.1f}%)")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=None, help="internal: run one scenario")
    parser.add_argument("--timing", choices=TIMINGS, default=None, help="internal")
    args = parser.parse_args()
    if args.policy:
        _run_scenario(args.policy, args.timing)
        raise SystemExit(0)
    raise SystemExit(main())
