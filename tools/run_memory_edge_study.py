#!/usr/bin/env python3
"""Task 22 S2.1/S2.3: does KV cache capacity constrain a decode replica
before the network does, and does the network's own share of a decode
step move as that capacity changes?

**The honest knob.** vLLM v1's own admission-control simulation
(`VllmV1SchedulerConfig`, `enable_preemption=True` by default) evicts a
running request's KV blocks under memory pressure -- `request.get_total_preemption_count()`
is the direct, unambiguous signal that capacity bound something, not an
inferred one. `num_blocks`, holding `block_size` fixed, is the capacity
knob: block *size* changes fragmentation granularity, not how much total
capacity a replica has, so sweeping `num_blocks` alone isolates the
variable this task is actually about.  `--cluster_config_decode_attn_replica_scheduler_config_num_blocks`
sets it for DECODE_ATTN specifically (PREFILL/DECODE_FFN stay generously
provisioned throughout, so nothing there confounds the sweep).

**The workload.** `block_size=16`, `prefill_tokens=32`,
`decode_tokens=16` -> 48 tokens/request -> `ceil(48/16)=3` blocks/request.
`num_blocks` therefore has a direct concurrent-request-capacity reading:
`num_blocks // 3`. 32 requests arriving at a high rate (`qps=20`, mean
53ms apart) so real queueing pressure exists at small capacities and
plainly does not at large ones -- both ends of S2.1's own "sweep from
constraining to not" instruction.

**The interaction (S2.3)**: the same sweep, run twice -- `colocated`
(ATTN+FFN share a domain) and `split` (this project's own established
~15%-of-decode-step M2N placement penalty configuration, tasks 11/12) --
so the network's *share* of a decode step at each capacity level can be
compared directly, holding everything else fixed.

Real h800 compute profiles throughout (Phi-tiny-MoE-instruct); dummy mode
would make every ratio here meaningless (task 12's own finding, restated
as a trap in this task's own spec).

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as every real-profile tool in this project:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_memory_edge_study.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified --
measurement only, per this task's own acceptance criteria.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

FRONTIER_ROOT = Path("/work/simulation/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m2n_integration import (  # noqa: E402
    SCALE_OUT_GBPS, SCALE_UP_GBPS, _engine_deployment_and_registry,
    _find_m2n_predictor, _placements)
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/memory_edge_study_outputs")

MODEL_NAME = "Phi-tiny-MoE-instruct"
TOTAL_EXPERTS = 16
ROUTER_TOPK = 2
BLOCK_SIZE = 16
PREFILL_TOKENS = 32
DECODE_TOKENS = 16
BLOCKS_PER_REQUEST = -(-(PREFILL_TOKENS + DECODE_TOKENS) // BLOCK_SIZE)  # ceil = 3
NUM_REQUESTS = 32
QPS = 20.0
GENEROUS_NUM_BLOCKS = 4096  # PREFILL/DECODE_FFN: never the constraint here

# Concurrent-capacity readings: num_blocks // BLOCKS_PER_REQUEST
NUM_BLOCKS_VALUES = (6, 9, 15, 30, 60, 120)
N_REPEATS = 3  # S6's own trap: near a capacity edge, one run may be unrepresentative


def _argv(run_id: str, label: str, num_blocks: int, seed: int) -> list[str]:
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

        "--vllm_v1_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", str(BLOCK_SIZE),
        "--vllm_v1_scheduler_config_num_blocks", str(GENEROUS_NUM_BLOCKS),
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        # The one knob actually swept -- DECODE_ATTN specifically.
        "--cluster_config_decode_attn_replica_scheduler_config_num_blocks", str(num_blocks),
        "--cluster_config_decode_attn_replica_scheduler_config_block_size", str(BLOCK_SIZE),

        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(PREFILL_TOKENS),
        "--fixed_request_length_generator_config_decode_tokens", str(DECODE_TOKENS),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(QPS),

        "--seed", str(seed),
        "--metrics_config_output_dir", str(OUTPUT_DIR),
        "--metrics_config_run_id", run_id,
        # write_metrics and store_utilization_metrics must both stay True
        # (task 18's own finding): they gate the Frontier stage-batch
        # ledger's in-memory capture, not just disk writing -- this study
        # needs that ledger for achieved batch size.
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]


_RESULT_MARKER = "MEMORY_EDGE_RESULT="


def _run_scenario(label: str, num_blocks: int, seed: int) -> None:
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    colocated_placement, split_placement = _placements(fabric, deployment)
    placement = colocated_placement if label == "colocated" else split_placement
    install(fabric, placement, deployment, registry)

    tag = f"mem_{label}_nb{num_blocks}_seed{seed}"
    sys.argv = _argv(tag, label, num_blocks, seed)
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
        print(_RESULT_MARKER + json.dumps({"tag": tag, "error": error}), flush=True)
        return

    requests = sim._all_requests
    completed = [r for r in requests if r.completed]
    preemptions = [r.get_total_preemption_count() for r in requests]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]
    m2n_s = [r.total_m2n_transfer_time for r in requests]

    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    attn_rows = [r for r in rows if r["cluster_type"] == "DECODE_ATTN"]
    batch_sizes = [len(r["request_ids"]) for r in attn_rows]

    wall_s = max((r.completed_at for r in completed), default=0.0)
    throughput_rps = len(completed) / wall_s if wall_s else 0.0

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "label": label, "num_blocks": num_blocks, "seed": seed,
        "n_requests": len(requests), "n_completed": len(completed),
        "total_preemptions": sum(preemptions),
        "requests_preempted": sum(1 for p in preemptions if p > 0),
        "mean_batch_size": mean(batch_sizes) if batch_sizes else None,
        "max_batch_size": max(batch_sizes) if batch_sizes else None,
        "mean_tpot_ms": mean(tpot_s) * 1000.0 if tpot_s else None,
        "mean_m2n_time_ms": mean(m2n_s) * 1000.0 if m2n_s else None,
        "throughput_rps": throughput_rps,
        "wall_s": wall_s,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(label: str, num_blocks: int, seed: int) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--scenario", label,
         "--num-blocks", str(num_blocks), "--seed", str(seed)],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode})",
           "tag": f"mem_{label}_nb{num_blocks}_seed{seed}"}


def _aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return {"error": rows[0].get("error")}
    return {
        "n_runs": len(ok),
        "mean_batch_size": mean(r["mean_batch_size"] for r in ok if r["mean_batch_size"]),
        "max_batch_size": max(r["max_batch_size"] for r in ok if r["max_batch_size"]),
        "total_preemptions_mean": mean(r["total_preemptions"] for r in ok),
        "total_preemptions_stdev": pstdev([r["total_preemptions"] for r in ok]) if len(ok) > 1 else 0.0,
        "throughput_rps_mean": mean(r["throughput_rps"] for r in ok),
        "mean_tpot_ms_mean": mean(r["mean_tpot_ms"] for r in ok if r["mean_tpot_ms"]),
        "mean_m2n_time_ms_mean": mean(r["mean_m2n_time_ms"] for r in ok if r["mean_m2n_time_ms"]),
    }


def main(labels: tuple[str, ...]) -> int:
    results = {}
    for label in labels:
        for nb in NUM_BLOCKS_VALUES:
            runs = [_run_scenario_in_subprocess(label, nb, seed) for seed in range(N_REPEATS)]
            agg = _aggregate(runs)
            results[(label, nb)] = agg
            capacity = nb // BLOCKS_PER_REQUEST
            if agg.get("error"):
                print(f"[{label:<9} nb={nb:>4} cap~{capacity:>3}] ERROR: {agg['error']}")
                continue
            print(f"[{label:<9} nb={nb:>4} cap~{capacity:>3}] n_runs={agg['n_runs']} "
                 f"mean_batch={agg['mean_batch_size']:.2f} max_batch={agg['max_batch_size']:.0f} "
                 f"preemptions={agg['total_preemptions_mean']:.1f}(+/-{agg['total_preemptions_stdev']:.1f}) "
                 f"throughput={agg['throughput_rps_mean']:.3f} req/s "
                 f"tpot={agg['mean_tpot_ms_mean']:.4f}ms "
                 f"m2n={agg['mean_m2n_time_ms_mean']:.4f}ms")

    if len(labels) == 2:
        print()
        print("=== interaction: network share of decode step, colocated vs split, by capacity ===")
        for nb in NUM_BLOCKS_VALUES:
            c, s = results.get(("colocated", nb)), results.get(("split", nb))
            if not c or not s or c.get("error") or s.get("error"):
                continue
            c_tpot, s_tpot = c["mean_tpot_ms_mean"], s["mean_tpot_ms_mean"]
            penalty_pct = 100 * (s_tpot - c_tpot) / c_tpot if c_tpot else float("nan")
            print(f"  nb={nb:>4} cap~{nb//BLOCKS_PER_REQUEST:>3}: colocated_tpot={c_tpot:.4f}ms "
                 f"split_tpot={s_tpot:.4f}ms  network_penalty={penalty_pct:+.2f}%  "
                 f"colocated_m2n={c['mean_m2n_time_ms_mean']:.4f}ms  "
                 f"split_m2n={s['mean_m2n_time_ms_mean']:.4f}ms")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("colocated", "split"), default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", nargs="+", default=["colocated", "split"])
    args = parser.parse_args()
    if args.scenario:
        _run_scenario(args.scenario, args.num_blocks, args.seed)
        raise SystemExit(0)
    raise SystemExit(main(tuple(args.labels)))
