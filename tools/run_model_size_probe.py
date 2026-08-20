#!/usr/bin/env python3
"""Task 17 Part B.2: does the placement penalty shrink with model size, and
what does a "middle" size actually cost to run?

Task 12 measured the colocated-vs-split M2N placement penalty on
`llama2_7b_dense_example` (hidden=4096, 32 layers) and tried
`Llama-3.1-405B-Instruct-FP8` (hidden=16384, 126 layers), abandoning it
after ten minutes of sklearn predictor training. This script reruns the
identical colocated-vs-split comparison (task 12's own methodology, reused
via import from `tools/run_m2n_real_profile.py`/`run_m2n_integration.py`,
not rewritten) on `step-moe-noquant-small` (hidden=7168, 31 layers) --
genuinely between the two in the one dimension that sets activation/KV
transfer size (hidden_size), with real h800 compute profiles already on
disk, unlike a from-scratch model would need.

`step-moe-noquant-small` is MoE (24 experts, top-3), so `total_expert_num`/
`router_topk` are set to match, unlike task 12's dense flags of 1/1; and
its profiling data only has a `uniform_topk` routing runtime path, not the
`standard_fused_topk` `moe_routing_mode=simulation` resolves to (confirmed
by running it: `ValueError` listing the available paths, the same
resolution issue `run_ep_pricing_probe.py` hit for `Phi-tiny-MoE-instruct`)
-- `moe_routing_mode=uniform_random` is used instead.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as every other real-profile tool in this project:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_model_size_probe.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified --
measurement only, per task 17's own acceptance criteria.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

FRONTIER_ROOT = Path("/work/simulation/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m2n_integration import (  # noqa: E402
    SCALE_OUT_GBPS, SCALE_UP_GBPS, _engine_deployment_and_registry,
    _find_m2n_predictor, _placements)
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/model_size_probe_outputs")

MODEL_NAME = "step-moe-noquant-small"
TOTAL_EXPERTS = 24
ROUTER_TOPK = 3
NUM_REQUESTS = 4
DECODE_TOKENS = 16


def _argv(run_id: str, label: str) -> list[str]:
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
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_total_expert_num", str(TOTAL_EXPERTS),
        "--cluster_config_decode_ffn_replica_config_router_topk", str(ROUTER_TOPK),
        "--cluster_config_decode_ffn_replica_config_device", "h800",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",

        "--cc_backend_config_type", "analytical",
        "--m2n_transfer_config_type", "empirical",

        "--replica_config_model_name", MODEL_NAME,
        "--replica_config_moe_routing_mode", "uniform_random",
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
    ]


_RESULT_MARKER = "MODEL_SIZE_RESULT="


def _run_scenario(label: str) -> None:
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    colocated_placement, split_placement = _placements(fabric, deployment)
    placement = colocated_placement if label == "colocated" else split_placement
    install(fabric, placement, deployment, registry)

    sys.argv = _argv(f"model_size_{label}", label)
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.types import ClusterType
    from frontier.utils.random import set_seeds

    wall_start = time.perf_counter()
    error = None
    sim = None
    try:
        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
    except Exception as e:  # noqa: BLE001 -- report whatever happens
        error = f"{type(e).__name__}: {e}"
    wall_s = time.perf_counter() - wall_start

    if error is not None:
        print(_RESULT_MARKER + json.dumps({"label": label, "error": error, "wall_s": wall_s}),
             flush=True)
        return

    requests = sim._all_requests
    m2n_time_s = [r.total_m2n_transfer_time for r in requests]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]

    attn_s, ffn_s, decode_wall_s = [], [], []
    for r in requests:
        attn_s.append(r.get_total_cluster_execution_time(ClusterType.DECODE_ATTN))
        ffn_s.append(r.get_total_cluster_execution_time(ClusterType.DECODE_FFN))
        decode_wall_s.append(r.completed_at - r.prefill_completed_at - r.kv_cache_transfer_time)

    predictor = _find_m2n_predictor(sim)

    print(_RESULT_MARKER + json.dumps({
        "label": label,
        "error": None,
        "wall_s": wall_s,
        "mean_m2n_time_s": mean(m2n_time_s),
        "mean_tpot_s": mean(tpot_s) if tpot_s else None,
        "mean_attn_s": mean(attn_s),
        "mean_ffn_s": mean(ffn_s),
        "mean_decode_wall_s": mean(decode_wall_s),
        "predictor_calls": predictor.calls if predictor else 0,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(label: str) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--scenario", label],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"label": label, "error": f"no result (exit code {proc.returncode})", "wall_s": None}


def main() -> int:
    for label in ("colocated", "split"):
        r = _run_scenario_in_subprocess(label)
        if r.get("error"):
            print(f"[{label}] ERROR: {r['error']}  wall_s={r.get('wall_s')}")
            continue
        wall = r["mean_decode_wall_s"]
        m2n = r["mean_m2n_time_s"]
        pct = 100 * m2n / wall if wall else float("nan")
        print(f"[{label:<10}] wall_clock_s={r['wall_s']:.2f}  mean_m2n={m2n*1000:.6f} ms  "
             f"mean_tpot={r['mean_tpot_s']*1000:.6f} ms  mean_decode_wall={wall*1000:.6f} ms  "
             f"m2n_fraction_of_decode_step={pct:.4f}%  calls={r['predictor_calls']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("colocated", "split"), default=None)
    args = parser.parse_args()
    if args.scenario:
        _run_scenario(args.scenario)
        raise SystemExit(0)
    raise SystemExit(main())
