#!/usr/bin/env python3
"""Task 17 Part B.1: what happens to M2N pricing above one expert group.

Every measurement in this project (tasks 09-16) used
`moe_expert_parallel_size=1` on every cluster. Above that, MoE dispatch is a
genuine many-to-many all-to-all among the expert-parallel ranks within one
replica -- not the point-to-point flow `EngineM2NTransferPredictor` prices.
This script runs the same single-replica-per-pool pd-af-disaggregation
scenario as task 11 (`tools/run_m2n_integration.py`), sweeping
`decode_ffn`'s `moe_expert_parallel_size` across {1, 2, 4}, and observes --
by monkey-patching the real predictor class, the same read-only
instrumentation pattern tasks 11-13 use -- three things per call:
`activation_size_bytes`, `afd_stage_idx`, and the raw call count, plus
whether the run raises at all.

This project's own `Replica`/`Placement` model has no notion of an
expert-parallel *rank* distinct from a replica's single representative rank
-- every EP rank in a replica is given its own GPU (colocated, since intra-
replica EP communication is exactly the kind of tightly-coupled traffic a
real deployment would keep on one scale-up domain), but `price_transfer`
(`integration/binding_support.py`) still only ever prices from/to
`ranks[0]` -- see the report for what this does and does not miss.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as tasks 12-16:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_ep_pricing_probe.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified --
per task 17's own acceptance criteria, this is a measurement script, not a
fix.
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
from engine.placement.placement import explicit  # noqa: E402

from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/ep_pricing_probe_outputs")

SCALE_UP_GBPS = 400.0
SCALE_OUT_GBPS = 50.0
NUM_REQUESTS = 8
DECODE_TOKENS = 8
TOTAL_EXPERTS = 16
ROUTER_TOPK = 2
EP_VALUES = (1, 2, 4)


def _deployment_and_registry(ep: int):
    """One replica per pool, same as task 11 -- except DECODE_FFN's replica
    now has `ep=ep` ranks instead of 1."""
    from frontier.types import ClusterType
    d = Deployment("ep-pricing-probe")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1, ep=ep))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {
        PoolKind.PREFILL: ClusterType.PREFILL,
        PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
        PoolKind.DECODE_FFN: ClusterType.DECODE_FFN,
    })
    return d, reg


def _placement(fabric, deployment, ep: int):
    """PREFILL and DECODE_ATTN colocated with the FFN replica's first (EP
    rank 0) rank; every other EP rank gets its own GPU on the same machine
    -- intra-replica EP traffic is the tightly-coupled case a real
    deployment keeps on one scale-up domain, so this is the natural
    placement to test pricing against, not an adversarial one."""
    prefill_rank = deployment.replicas[0].ranks[0]
    attn_rank = deployment.replicas[1].ranks[0]
    ffn_ranks = deployment.replicas[2].ranks  # `ep` ranks
    mapping = {prefill_rank: GpuId(0, 0), attn_rank: GpuId(0, 1)}
    for i, rank in enumerate(ffn_ranks):
        mapping[rank] = GpuId(0, 2 + i)
    return explicit(deployment, fabric, mapping)


def _argv(run_id: str, ep: int, real: bool = False) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters",

        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", "1",
        "--cluster_config_decode_ffn_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_attn_micro_batch_size", "8",

        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", str(TOTAL_EXPERTS),
        "--cluster_config_prefill_replica_config_router_topk", str(ROUTER_TOPK),
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_device", "h800",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", str(ep),
        "--cluster_config_decode_ffn_replica_config_total_expert_num", str(TOTAL_EXPERTS),
        "--cluster_config_decode_ffn_replica_config_router_topk", str(ROUTER_TOPK),
        "--cluster_config_decode_ffn_replica_config_device", "h800",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",

        "--cc_backend_config_type", "analytical",
        "--m2n_transfer_config_type", "empirical",

        "--replica_config_model_name", "Phi-tiny-MoE-instruct",
        # "simulation" resolves to the "standard_fused_topk" profiling path
        # (frontier/moe_routing_runtime.py), which this model's own
        # profiling data doesn't have -- only "uniform_topk" (confirmed by
        # running it: ValueError listing the available paths). "uniform_random"
        # resolves to that path instead; harmless for this probe, which
        # only cares about M2N call/size/afd_stage_idx behavior, not routing
        # realism.
        "--replica_config_moe_routing_mode", ("uniform_random" if real else "simulation"),
        "--replica_config_moe_routing_seed", "42",

        "--vllm_v1_scheduler_config_max_tokens_in_batch", "1024",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "64",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", "128",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",

        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "32",
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

    ] + ([
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "1.0",
    ] if not real else [])


_RESULT_MARKER = "EP_PRICING_RESULT="


def _run_scenario(ep: int, real: bool = False) -> None:
    import time as _time
    fabric = build_node_scale(num_machines=1, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _deployment_and_registry(ep)
    placement = _placement(fabric, deployment, ep)
    install(fabric, placement, deployment, registry)

    from integration.m2n_transfer.predictor import EngineM2NTransferPredictor
    calls = []
    _original = EngineM2NTransferPredictor.get_transfer_time

    def _observing(self, source_cluster_type, target_cluster_type, batch, activation_size_bytes):
        result = _original(self, source_cluster_type, target_cluster_type, batch, activation_size_bytes)
        calls.append({
            "direction": f"{source_cluster_type}->{target_cluster_type}",
            "activation_size_bytes": activation_size_bytes,
            "afd_stage_idx": self.last_attribution.afd_stage_idx if self.last_attribution else None,
            "layer_id": self.last_attribution.layer_id if self.last_attribution else None,
            "price_ms": result,
        })
        return result

    EngineM2NTransferPredictor.get_transfer_time = _observing

    sys.argv = _argv(f"ep_pricing_ep{ep}{'_real' if real else ''}", ep, real=real)
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    error = None
    wall_start = _time.perf_counter()
    sim = None
    try:
        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
    except Exception as e:  # noqa: BLE001 -- report whatever happens, don't swallow
        error = f"{type(e).__name__}: {e}"
    wall_s = _time.perf_counter() - wall_start

    distinct_sizes = sorted(set(c["activation_size_bytes"] for c in calls))
    distinct_afd = sorted(set(c["afd_stage_idx"] for c in calls if c["afd_stage_idx"] is not None))
    sim_end_s = sim._all_requests[-1].completed_at if (sim is not None and sim._all_requests) else None

    print(_RESULT_MARKER + json.dumps({
        "ep": ep,
        "real": real,
        "error": error,
        "num_calls": len(calls),
        "distinct_activation_sizes": distinct_sizes,
        "distinct_afd_stage_idx": distinct_afd,
        "mean_price_ms": mean(c["price_ms"] for c in calls) if calls else None,
        "wall_clock_s": wall_s,
        "sim_end_s": sim_end_s,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(ep: int, real: bool = False) -> dict:
    argv = [sys.executable, _SCRIPT_PATH, "--ep", str(ep)]
    if real:
        argv.append("--real")
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"ep": ep, "error": f"no result (exit code {proc.returncode}); see stderr above",
           "num_calls": 0, "distinct_activation_sizes": [], "distinct_afd_stage_idx": [],
           "mean_price_ms": None, "wall_clock_s": None, "sim_end_s": None}


def main(real: bool) -> int:
    for ep in EP_VALUES:
        r = _run_scenario_in_subprocess(ep, real=real)
        print(f"[EP={ep} real={real}] error={r['error']!r}  num_calls={r['num_calls']}  "
             f"activation_sizes={r['distinct_activation_sizes']}  "
             f"afd_stage_idx={r['distinct_afd_stage_idx']}  "
             f"wall_clock_s={r['wall_clock_s']}  sim_end_s={r['sim_end_s']}  "
             f"mean_price_ms={r['mean_price_ms']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=None, help="internal: run one EP value")
    parser.add_argument("--real", action="store_true", help="real compute profiles, not dummy mode")
    args = parser.parse_args()
    if args.ep is not None:
        _run_scenario(args.ep, real=args.real)
        raise SystemExit(0)
    raise SystemExit(main(args.real))
