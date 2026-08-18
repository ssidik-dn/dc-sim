#!/usr/bin/env python3
"""Task 09: the end-to-end proof that a real Frontier run's own numbers
change because of where the GPUs are.

Runs the same offline pd-disaggregation workload twice, through
EngineKVCacheTransferPredictor (registered under
KVCacheTransferType.EMPIRICAL, selected purely via
--kv_cache_transfer_config_type empirical -- see tools/probe_kv_selection.py,
task 07), differing only in one thing: whether the PREFILL and DECODE pools'
representative GPUs share a scale-up domain ("packed") or are split across
two ("split"). Reports Frontier's own request.ttft and
request.kv_cache_transfer_time for both, and their ratio.

Read docs/tasks/09-kv-predictor-report.md before trusting the TTFT number:
Frontier's `ttft` is defined as arrival-to-prefill-completion and does not
include KV transfer time (frontier/metrics/constants.py:
`TTFT = "ttft"  # Total time from arrival to prefill completion`). KV
transfer time is tracked as a separate quantity,
`request.kv_cache_transfer_time`. Both are reported below; only the second
is expected to differ.

Environment: same as tasks 07/08 -- Frontier is reached via the ambient
PYTHONPATH, run from the dc-sim root:

    PYTHONPATH=src:/work/Frontier python3 tools/run_kv_integration.py

Device: h800 (AGENTS.md). Execution time uses Frontier's dummy mode (flat
per-operator time) -- that governs compute prediction only, not this
project's KV transfer predictor, which always prices from the real fabric
graph regardless of dummy mode.

Nothing under `upstream/` or `src/engine/` is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

from frontier.types import ClusterType

from engine.logical.deployment import Deployment, PoolKind, Replica
from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId
from engine.placement.placement import explicit

from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment
from integration.install import install

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/kv_integration_outputs")

# Same physical parameters as engine.physical.builders.build_node_scale's
# defaults, named here because the report reasons about them explicitly:
# 400 GB/s scale-up, 50 GB/s scale-out -- an 8:1 bandwidth ratio.
SCALE_UP_GBPS = 400.0
SCALE_OUT_GBPS = 50.0


def _engine_deployment_and_registry():
    """A trivial one-rank-per-pool deployment: task 09 spec S2.2 restricts
    this predictor to exactly one replica per pool, so tp=1 is enough to be
    unambiguous -- the point is the physical placement, not the
    parallelism shape."""
    d = Deployment("kv-integration")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE, 0, tp=1))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.PREFILL: ClusterType.PREFILL,
                                      PoolKind.DECODE: ClusterType.DECODE})
    return d, reg


def _placements(fabric, deployment):
    prefill_rank = deployment.replicas[0].ranks[0]
    decode_rank = deployment.replicas[1].ranks[0]
    packed = explicit(deployment, fabric,
                      {prefill_rank: GpuId(0, 0), decode_rank: GpuId(0, 1)})
    split = explicit(deployment, fabric,
                     {prefill_rank: GpuId(0, 0), decode_rank: GpuId(1, 0)})
    return packed, split


def _argv(run_id: str) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
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
        # the flag under test: our real predictor, not Frontier's analytical one
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
        "--synthetic_request_generator_config_num_requests", "4",
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "2048",
        "--fixed_request_length_generator_config_decode_tokens", "4",
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


def _run(run_id: str) -> tuple[list[float], list[float]]:
    """Returns (ttft_seconds, kv_cache_transfer_time_seconds) per request."""
    sys.argv = _argv(run_id)
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    config = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)
    sim = Simulator(config)
    sim.run()

    requests = sim._all_requests
    return ([r.ttft for r in requests], [r.kv_cache_transfer_time for r in requests])


def main() -> int:
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    packed_placement, split_placement = _placements(fabric, deployment)

    results = {}
    for label, placement in [("packed", packed_placement), ("split", split_placement)]:
        install(fabric, placement, deployment, registry)
        ttft, kv_time = _run(f"kv_integration_{label}")
        results[label] = (mean(ttft), mean(kv_time))
        print(f"{label}: mean ttft={results[label][0]*1000:.6f} ms, "
              f"mean kv_cache_transfer_time={results[label][1]*1000:.6f} ms "
              f"(n={len(ttft)} requests)")

    packed_ttft, packed_kv = results["packed"]
    split_ttft, split_kv = results["split"]

    print()
    print(f"TTFT:                 packed={packed_ttft*1000:.6f} ms  "
          f"split={split_ttft*1000:.6f} ms  "
          f"ratio={split_ttft/packed_ttft if packed_ttft else float('nan'):.4f}")
    print(f"KV cache transfer time: packed={packed_kv*1000:.6f} ms  "
          f"split={split_kv*1000:.6f} ms  "
          f"ratio={split_kv/packed_kv if packed_kv else float('nan'):.4f}")
    print(f"(fabric bandwidth ratio scale_out:scale_up = "
          f"{SCALE_UP_GBPS/SCALE_OUT_GBPS:.1f}:1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
