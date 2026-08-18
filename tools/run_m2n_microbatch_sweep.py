#!/usr/bin/env python3
"""Task 13: does micro-batching pay for itself?

Task 12 measured the M2N placement penalty with
`decode_attn_af_pipeline_num_micro_batch = 1` -- with a single micro-batch
there is nothing for a pool to compute while a transfer is in flight, so the
transfer is serial with compute *by construction*, and the whole of it
reached inter-token latency (task 12 report S2: 100.0%, once the units
error in that comparison was fixed).

Pipelining more micro-batches is supposed to let compute on batch i+1
overlap the transfer of batch i, hiding some of that cost. But task 10
established activation exchange is latency-bound: splitting one transfer
into N smaller ones pays the fixed per-hop latency N times while the
size-dependent part (already small) shrinks by 1/N. Whether pipelining is a
net win, a net loss, or both depending on placement, is genuinely unmeasured
before this script runs it.

Same model, same fabric, same subprocess-per-scenario structure as task 12
(the pd-af-disaggregation replica-id bug it worked around is unrelated and
still present) -- only `decode_attn_af_pipeline_num_micro_batch` and its FFN
counterpart change, swept together (task 13 spec S2: vary independently only
with a reason, and there isn't one here). Real compute profiles throughout,
per the spec -- no dummy-mode row in this sweep.

Environment: same as task 12, run from anywhere; cwd is set internally to
the Frontier root for model-config and compute-profile resolution:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_m2n_microbatch_sweep.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified.
`EngineM2NTransferPredictor.get_transfer_time` is monkey-patched from this
script, at the class object, purely to observe the `activation_size_bytes`
each call actually receives (task 13 spec S4: confirm this rather than
assume it) -- the source file is untouched; this is read-only
instrumentation layered on from outside, the same way `predictor.calls` and
`predictor.total_wall_ns` were already read from outside in tasks 11/12.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

FRONTIER_ROOT = Path("/work/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m2n_integration import (  # noqa: E402
    SCALE_OUT_GBPS, SCALE_UP_GBPS, _engine_deployment_and_registry,
    _find_m2n_predictor, _placements)
from run_m2n_real_profile import DEFAULT_MODEL  # noqa: E402
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/m2n_microbatch_outputs")

NUM_REQUESTS = 8
DECODE_TOKENS = 16
MICRO_BATCH_COUNTS = (1, 2, 4, 8)


def _argv(run_id: str, model_name: str, num_micro_batch: int) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters",

        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", "1",
        "--cluster_config_decode_ffn_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch", str(num_micro_batch),
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch", str(num_micro_batch),
        "--cluster_config_decode_attn_micro_batch_size", "8",

        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", "1",
        "--cluster_config_prefill_replica_config_router_topk", "1",
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
        "--cluster_config_decode_ffn_replica_config_total_expert_num", "1",
        "--cluster_config_decode_ffn_replica_config_router_topk", "1",
        "--cluster_config_decode_ffn_replica_config_device", "h800",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",

        "--cc_backend_config_type", "analytical",
        "--m2n_transfer_config_type", "empirical",

        "--replica_config_model_name", model_name,
        "--replica_config_moe_routing_mode", "simulation",
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
        # real profiles throughout -- no dummy-mode flags at all, so
        # enable_dummy_mode defaults to False; verified below anyway.
    ]


_RESULT_MARKER = "M2N_MICROBATCH_RESULT="


def _run_scenario(label: str, num_micro_batch: int, model_name: str) -> None:
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    colocated_placement, split_placement = _placements(fabric, deployment)
    placement = colocated_placement if label == "colocated" else split_placement
    install(fabric, placement, deployment, registry)

    # Monkey-patch the predictor class (not the source file) to record
    # every activation_size_bytes it actually receives -- task 13 spec S4
    # asks to confirm this rather than assume it shrinks with N.
    from integration.m2n_transfer.predictor import EngineM2NTransferPredictor
    sizes_seen: list[int] = []
    _original = EngineM2NTransferPredictor.get_transfer_time

    def _observing_get_transfer_time(self, source_cluster_type, target_cluster_type,
                                     batch, activation_size_bytes):
        sizes_seen.append(activation_size_bytes)
        return _original(self, source_cluster_type, target_cluster_type,
                         batch, activation_size_bytes)

    EngineM2NTransferPredictor.get_transfer_time = _observing_get_transfer_time

    sys.argv = _argv(f"m2n_microbatch_{model_name}_{num_micro_batch}_{label}",
                     model_name, num_micro_batch)
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.types import ClusterType
    from frontier.utils.random import set_seeds

    config = SimulationConfig.create_from_cli_args()
    actual_dummy_mode = config.cluster_config.execution_time_predictor_config.enable_dummy_mode
    if actual_dummy_mode:
        print(f"WARNING: enable_dummy_mode is True; this sweep expects real profiles",
              file=sys.stderr)

    set_seeds(config.seed)
    sim = Simulator(config)
    sim.run()

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
        "actual_dummy_mode": actual_dummy_mode,
        "mean_m2n_time_s": mean(m2n_time_s),
        "mean_tpot_s": mean(tpot_s) if tpot_s else None,
        "n_m2n": len(m2n_time_s),
        "n_tpot": len(tpot_s),
        "mean_attn_s": mean(attn_s),
        "mean_ffn_s": mean(ffn_s),
        "mean_decode_wall_s": mean(decode_wall_s),
        "predictor_calls": predictor.calls if predictor else 0,
        "n_activation_size_observations": len(sizes_seen),
        "distinct_activation_sizes": sorted(set(sizes_seen)),
        "min_activation_size": min(sizes_seen) if sizes_seen else None,
        "max_activation_size": max(sizes_seen) if sizes_seen else None,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(label: str, num_micro_batch: int, model_name: str) -> dict:
    # Absolute path: the subprocess's cwd is FRONTIER_ROOT, and a relative
    # __file__ would resolve against that instead of where this script
    # actually lives.
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--scenario", label,
         "--micro-batch", str(num_micro_batch), "--model", model_name],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise RuntimeError(f"scenario {label!r} micro_batch={num_micro_batch} produced no "
                       f"result (exit code {proc.returncode}); see output above")


def main(model_name: str) -> int:
    results = {}
    for n in MICRO_BATCH_COUNTS:
        for label in ("colocated", "split"):
            r = _run_scenario_in_subprocess(label, n, model_name)
            results[(n, label)] = r
            print(f"[N={n:2d}] {label:9s}: mean_m2n={r['mean_m2n_time_s']*1000:10.6f} ms  "
                  f"mean_tpot={r['mean_tpot_s']*1000:10.6f} ms  "
                  f"calls={r['predictor_calls']:5d}  "
                  f"activation_sizes(distinct)={r['distinct_activation_sizes']}")

    print()
    print(f"{'N':>3} {'placement':<10}{'mean m2n (ms)':>16}{'mean tpot (ms)':>16}"
         f"{'calls':>8}{'activation size (B)':>22}")
    for n in MICRO_BATCH_COUNTS:
        for label in ("colocated", "split"):
            r = results[(n, label)]
            sizes = r["distinct_activation_sizes"]
            size_str = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
            print(f"{n:>3} {label:<10}{r['mean_m2n_time_s']*1000:16.6f}"
                 f"{r['mean_tpot_s']*1000:16.6f}{r['predictor_calls']:8d}{size_str:>22}")

    print()
    print("total transfer cost vs N (relative to N=1):")
    for label in ("colocated", "split"):
        base = results[(1, label)]["mean_m2n_time_s"]
        for n in MICRO_BATCH_COUNTS:
            r = results[(n, label)]
            ratio_to_base = r["mean_m2n_time_s"] / base
            ratio_to_n = ratio_to_base / n
            print(f"  {label:<10} N={n:2d}: transfer={r['mean_m2n_time_s']*1000:10.6f} ms  "
                 f"({ratio_to_base:6.3f}x of N=1; {ratio_to_n:6.3f}x of linear-in-N)")

    print()
    print("inter-token latency (tpot) vs N:")
    for label in ("colocated", "split"):
        for n in MICRO_BATCH_COUNTS:
            r = results[(n, label)]
            print(f"  {label:<10} N={n:2d}: tpot={r['mean_tpot_s']*1000:10.6f} ms")

    print()
    print("decode-step composition (mean per request, full decode phase):")
    for n in MICRO_BATCH_COUNTS:
        for label in ("colocated", "split"):
            r = results[(n, label)]
            wall = r["mean_decode_wall_s"]
            attn, ffn, m2n = r["mean_attn_s"], r["mean_ffn_s"], r["mean_m2n_time_s"]
            other = wall - (attn + ffn + m2n)
            pct = lambda x: 100 * x / wall if wall else float("nan")
            print(f"  [N={n:2d}/{label:9s}] wall={wall*1000:9.4f} ms  "
                 f"attn={attn*1000:8.4f} ms ({pct(attn):5.1f}%)  "
                 f"ffn={ffn*1000:8.4f} ms ({pct(ffn):5.1f}%)  "
                 f"m2n={m2n*1000:8.4f} ms ({pct(m2n):5.1f}%)  "
                 f"other={other*1000:8.4f} ms ({pct(other):5.1f}%)")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("colocated", "split"), default=None,
                       help="internal: run one scenario in this process and exit")
    parser.add_argument("--micro-batch", type=int, default=None,
                       help="internal: decode_{attn,ffn}_af_pipeline_num_micro_batch")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.scenario:
        _run_scenario(args.scenario, args.micro_batch, args.model)
        raise SystemExit(0)
    raise SystemExit(main(args.model))
