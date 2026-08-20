#!/usr/bin/env python3
"""Task 22 S2.2: does one pool idle waiting on the other at realistic
attention:FFN replica ratios?

Every study in this project before this one used a 1:1 DECODE_ATTN:DECODE_FFN
replica ratio. Task 12 measured attention compute at 34.67 ms against FFN
at 50.47 ms in the same decode step (a ratio near 2:3, ATTN:FFN, for the
two pools' aggregate throughput to balance -- `N_attn/N_ffn ==
attn_time/ffn_time` when each pool's replicas are otherwise identical).
This script sweeps the *replica count* ratio around that point and reads
per-pool busy time directly off Frontier's own stage-batch ledger, the
same source every other real-compute tool in this project already reads.

This is deliberately **not** about the network: no engine-side
`install()`, no `EngineM2NTransferPredictor` -- Frontier's own stock
`analytical` M2N backend prices activation exchange here, which sidesteps
the multi-replica *destination-ambiguity* problem tasks 14-16 built
`ctx.binding` to solve (this script isn't asking that question, and
configuring a binding policy just to avoid a raise would be answering it
by accident). Utilisation, not the network's own cost, is the point.

**Utilisation** is computed as (summed `total_time_ms` across every ledger
row for a cluster type) / (wall-clock ms * that cluster's own replica
count) -- the average fraction of one replica's own wall-clock time it
spent busy, matching the "99% vs under 3%" signature task 12's own report
found and this task's spec asks to look for again.

Real h800 compute profiles throughout (Phi-tiny-MoE-instruct); dummy mode
would make every ratio meaningless.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_compute_balance_study.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

FRONTIER_ROOT = Path("/work/simulation/Frontier")

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/compute_balance_study_outputs")

MODEL_NAME = "Phi-tiny-MoE-instruct"
TOTAL_EXPERTS = 16
ROUTER_TOPK = 2
NUM_REQUESTS = 32
DECODE_TOKENS = 16
PREFILL_TOKENS = 32
QPS = 20.0
N_REPEATS = 3

# (attn_replicas, ffn_replicas) -- 1:1 (every prior study), the two
# extremes, and either side of task 12's own ~2:3 balance point.
RATIOS = ((1, 1), (2, 1), (1, 2), (2, 3), (3, 2))


def _argv(run_id: str, attn_replicas: int, ffn_replicas: int, seed: int) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters",

        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", str(attn_replicas),
        "--cluster_config_decode_ffn_cluster_num_replicas", str(ffn_replicas),
        "--cluster_config_allow_experiment_multi_decode_ffn_replicas",
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
        # Set to ffn_replicas, not 1: Frontier's own static M2N lane
        # assignment (base_cluster_scheduler.py's __init__) requires at
        # least as many DECODE_ATTN dp lanes (attn_replicas * dp_size) as
        # DECODE_FFN replicas, or it raises "must give every target
        # replica at least one decode-attn lane" -- confirmed by running
        # attn=1/ffn=2 with dp_size=1 and hitting exactly that. This is a
        # mechanical requirement of Frontier's own lane coverage, not part
        # of the compute-balance question this script asks; dp lanes are
        # not additional attention *replicas* (which is why the utilisation
        # denominator below still uses `attn_replicas`, not dp-lane count).
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", str(ffn_replicas),
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
        # Frontier's own stock M2N backend -- deliberately not this
        # project's empirical one; see module docstring for why.

        "--replica_config_model_name", MODEL_NAME,
        "--replica_config_moe_routing_mode", "uniform_random",
        "--replica_config_moe_routing_seed", "42",

        "--vllm_v1_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", "4096",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",

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
        # write_metrics/store_utilization_metrics stay True -- task 18's
        # own finding: both gate the stage-batch ledger's in-memory
        # capture, not just disk writing.
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]


_RESULT_MARKER = "COMPUTE_BALANCE_RESULT="


def _run_scenario(attn_replicas: int, ffn_replicas: int, seed: int) -> None:
    tag = f"balance_attn{attn_replicas}_ffn{ffn_replicas}_seed{seed}"
    sys.argv = _argv(tag, attn_replicas, ffn_replicas, seed)
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
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]
    wall_s = max((r.completed_at for r in completed), default=0.0)

    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    attn_busy_ms = sum(r["execution_time"]["total_time_ms"]
                       for r in rows if r["cluster_type"] == "DECODE_ATTN")
    ffn_busy_ms = sum(r["execution_time"]["total_time_ms"]
                      for r in rows if r["cluster_type"] == "DECODE_FFN")
    wall_ms = wall_s * 1000.0

    attn_util = attn_busy_ms / (wall_ms * attn_replicas) if wall_ms else None
    ffn_util = ffn_busy_ms / (wall_ms * ffn_replicas) if wall_ms else None

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "attn_replicas": attn_replicas,
        "ffn_replicas": ffn_replicas, "seed": seed,
        "n_completed": len(completed), "wall_ms": wall_ms,
        "attn_busy_ms": attn_busy_ms, "ffn_busy_ms": ffn_busy_ms,
        "attn_utilization": attn_util, "ffn_utilization": ffn_util,
        "mean_tpot_ms": mean(tpot_s) * 1000.0 if tpot_s else None,
        "throughput_rps": len(completed) / wall_s if wall_s else 0.0,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(attn_replicas: int, ffn_replicas: int, seed: int) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--attn-replicas", str(attn_replicas),
         "--ffn-replicas", str(ffn_replicas), "--seed", str(seed)],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode})",
           "tag": f"balance_attn{attn_replicas}_ffn{ffn_replicas}_seed{seed}"}


def _aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return {"error": rows[0].get("error")}
    return {
        "n_runs": len(ok),
        "attn_utilization": mean(r["attn_utilization"] for r in ok if r["attn_utilization"] is not None),
        "ffn_utilization": mean(r["ffn_utilization"] for r in ok if r["ffn_utilization"] is not None),
        "mean_tpot_ms": mean(r["mean_tpot_ms"] for r in ok if r["mean_tpot_ms"]),
        "throughput_rps": mean(r["throughput_rps"] for r in ok),
    }


def main() -> int:
    for attn_r, ffn_r in RATIOS:
        runs = [_run_scenario_in_subprocess(attn_r, ffn_r, seed) for seed in range(N_REPEATS)]
        agg = _aggregate(runs)
        if agg.get("error"):
            print(f"[attn={attn_r} ffn={ffn_r}] ERROR: {agg['error']}")
            continue
        print(f"[attn={attn_r} ffn={ffn_r} (ratio {attn_r}:{ffn_r})] n_runs={agg['n_runs']} "
             f"attn_util={100*agg['attn_utilization']:.1f}%  ffn_util={100*agg['ffn_utilization']:.1f}%  "
             f"tpot={agg['mean_tpot_ms']:.4f}ms  throughput={agg['throughput_rps']:.3f} req/s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-replicas", type=int, default=None)
    parser.add_argument("--ffn-replicas", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.attn_replicas is not None:
        _run_scenario(args.attn_replicas, args.ffn_replicas, args.seed)
        raise SystemExit(0)
    raise SystemExit(main())
