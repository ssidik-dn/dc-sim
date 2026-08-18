#!/usr/bin/env python3
"""Task 12: rerun task 11's colocated-vs-split M2N comparison with dummy
execution-time mode off, so the decode step's compute cost is a real,
profiled number instead of a flat 1 ms per operator.

Task 11 found a transfer-time ratio of 14.6503x but a TPOT (inter-token
latency) ratio of only 1.0020x, and correctly declined to read that as "M2N
placement doesn't matter" -- dummy mode inflates compute to 1 ms/operator
across every layer, which dwarfs a sub-millisecond transfer by construction
and has nothing to do with the network model. This script produces the
number that answers whether the placement penalty is actually visible when
compute is realistic.

No new modelling: same fabric, same predictor, same subprocess-per-scenario
structure from task 11 (the pd-af-disaggregation replica-id bug it worked
around is still present and unrelated to this change). Only the
execution-time-predictor mode, the model name, and the decode-token count
change.

Model: `llama2_7b_dense_example`, not task 11's `meta-llama/Llama-2-7b-hf`.
Both are the same real Llama-2-7b architecture (verified: 32 layers, 4096
hidden, 32 heads, both dense) -- the difference is that `llama2_7b_dense_example`
has real h800 compute profiles under `data/profiling/compute/h800/`, and
`meta-llama/Llama-2-7b-hf` does not (confirmed by directory listing: h800
has profiles for `llama2_7b_dense_example`, `Llama-3.1-405B-Instruct-FP8`,
and a handful of others, not for the generic HF name). Using a name with no
profile would trigger Frontier's model-*architecture* fallback (a different,
harmless fallback already seen in every prior task's logs) while probably
also hitting an unrelated failure trying to load compute profiles that don't
exist -- so this task uses the one name that both resolves to the right
architecture and has the profiles this comparison needs. Both dummy and
real rows use this same name, so the comparison is apples to apples; task
11's own numbers used the generic name and are not expected to match this
script's dummy-mode row exactly for that reason.

Model config AND compute profile resolution are both relative-path lookups
under the Frontier repo root (task 08's finding, for model config; the
`{DEVICE}/{MODEL}` profiling paths in frontier/config/config.py, for compute
profiles) -- this script's subprocess always runs with cwd=Frontier root.

Environment: PYTHONPATH must include Frontier and dc-sim's src (ambient in
this project, see task 07). Run from anywhere; cwd is set internally:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_m2n_real_profile.py

First real-mode run trains and caches sklearn per-operator models from the
profiling CSVs (one-time; several dozen seconds, joblib-parallel) into
`/work/Frontier/cache/*.pkl`, keyed by a content hash of the profiling data
and config. Subsequent runs against the same model/device reuse that cache
and are fast -- this script does not clear it, so repeat runs benefit.

Nothing under `upstream/` or `src/engine/` is modified. No new tests --
this is a measurement task; `python3 -m pytest -q` should stay at 157.
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
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/m2n_real_profile_outputs")

DEFAULT_MODEL = "llama2_7b_dense_example"

# Enough decode tokens that a request's own tpot (already an average over
# num_decode_tokens - 1 intervals) is a stable per-request figure, and
# enough requests that the run's mean isn't one or two samples -- task 11
# used decode_tokens=4/n=2; this task asked explicitly for a stabler mean.
NUM_REQUESTS = 4
DECODE_TOKENS = 16


def _argv(run_id: str, model_name: str, dummy: bool) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    argv = [
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
    ]
    if dummy:
        argv += [
            "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
            "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "1.0",
        ]
    return argv


_RESULT_MARKER = "M2N_REAL_PROFILE_RESULT="


def _run_scenario(label: str, dummy: bool, model_name: str) -> None:
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    colocated_placement, split_placement = _placements(fabric, deployment)
    placement = colocated_placement if label == "colocated" else split_placement
    install(fabric, placement, deployment, registry)

    mode_tag = "dummy" if dummy else "real"
    sys.argv = _argv(f"m2n_real_profile_{model_name}_{mode_tag}_{label}", model_name, dummy)
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.types import ClusterType
    from frontier.utils.random import set_seeds

    config = SimulationConfig.create_from_cli_args()
    # Known trap (task 12 spec S4): confirm the flag that actually reached
    # the parsed config, not a banner -- a prior session found a banner
    # claiming dummy mode was on when the CLI flag had disabled it.
    actual_dummy_mode = config.cluster_config.execution_time_predictor_config.enable_dummy_mode
    if actual_dummy_mode != dummy:
        print(f"WARNING: requested dummy={dummy} but parsed config has "
              f"enable_dummy_mode={actual_dummy_mode}", file=sys.stderr)

    set_seeds(config.seed)
    sim = Simulator(config)
    sim.run()

    requests = sim._all_requests
    m2n_time_s = [r.total_m2n_transfer_time for r in requests]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]

    # Decode-step composition (task 12 spec S3): attention compute, FFN
    # compute, and M2N transfer, all accumulated over the request's full
    # decode phase (prefill-done to fully-done, minus the KV hop), so the
    # three totals and the wall-clock denominator cover the identical
    # window -- avoids the off-by-one-round mismatch between "N rounds of
    # compute" and "N-1 tpot intervals between reported tokens".
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
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(label: str, dummy: bool, model_name: str) -> dict:
    # Absolute path: the subprocess's cwd is FRONTIER_ROOT, and a relative
    # __file__ would resolve against that instead of where this script
    # actually lives (task 13 found this the hard way -- fixed here too).
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--scenario", label,
         "--mode", "dummy" if dummy else "real", "--model", model_name],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise RuntimeError(f"scenario {label!r} mode={dummy=} produced no result "
                       f"(exit code {proc.returncode}); see output above")


def main(model_name: str) -> int:
    results = {}
    for dummy in (True, False):
        for label in ("colocated", "split"):
            r = _run_scenario_in_subprocess(label, dummy, model_name)
            results[(dummy, label)] = r
            mode_tag = "dummy" if dummy else "real "
            print(f"[{mode_tag}] {label:9s}: mean_m2n={r['mean_m2n_time_s']*1000:10.6f} ms  "
                  f"mean_tpot={r['mean_tpot_s']*1000:12.6f} ms  "
                  f"(actual_dummy_mode={r['actual_dummy_mode']}, "
                  f"n_m2n={r['n_m2n']}, n_tpot={r['n_tpot']}, "
                  f"calls={r['predictor_calls']})")

    print()
    print(f"{'mode':<6}{'placement':<11}{'mean m2n (ms)':>16}{'mean tpot (ms)':>17}")
    for dummy in (True, False):
        co = results[(dummy, "colocated")]
        sp = results[(dummy, "split")]
        mode_tag = "dummy" if dummy else "real"
        print(f"{mode_tag:<6}{'colocated':<11}{co['mean_m2n_time_s']*1000:16.6f}"
              f"{co['mean_tpot_s']*1000:17.6f}")
        print(f"{mode_tag:<6}{'split':<11}{sp['mean_m2n_time_s']*1000:16.6f}"
              f"{sp['mean_tpot_s']*1000:17.6f}")
        m2n_ratio = sp['mean_m2n_time_s'] / co['mean_m2n_time_s']
        tpot_ratio = sp['mean_tpot_s'] / co['mean_tpot_s']
        print(f"{mode_tag:<6}{'ratio':<11}{m2n_ratio:16.4f}{tpot_ratio:17.4f}")
        print()

    print("decode-step composition (mean per request, full decode phase):")
    for dummy in (True, False):
        mode_tag = "dummy" if dummy else "real"
        for label in ("colocated", "split"):
            r = results[(dummy, label)]
            wall = r["mean_decode_wall_s"]
            attn, ffn, m2n = r["mean_attn_s"], r["mean_ffn_s"], r["mean_m2n_time_s"]
            other = wall - (attn + ffn + m2n)
            pct = lambda x: 100 * x / wall if wall else float("nan")
            print(f"  [{mode_tag:5s}/{label:9s}] wall={wall*1000:9.4f} ms  "
                  f"attn={attn*1000:8.4f} ms ({pct(attn):5.1f}%)  "
                  f"ffn={ffn*1000:8.4f} ms ({pct(ffn):5.1f}%)  "
                  f"m2n={m2n*1000:8.4f} ms ({pct(m2n):5.1f}%)  "
                  f"other={other*1000:8.4f} ms ({pct(other):5.1f}%)")

    real_co, real_sp = results[(False, "colocated")], results[(False, "split")]
    # transfer delta is a TOTAL (request.total_m2n_transfer_time, summed
    # over every round-trip in the decode phase); tpot delta is a PER-TOKEN
    # AVERAGE (request.tpot, which divides by num_decode_tokens - 1).
    # Comparing them directly divides an extensive quantity by an intensive
    # one -- multiply the tpot delta back up by the same (N-1) denominator
    # tpot itself divides by, so both sides cover the identical decode-phase
    # total before the ratio is taken.
    transfer_delta_s = real_sp["mean_m2n_time_s"] - real_co["mean_m2n_time_s"]
    tpot_delta_per_token_s = real_sp["mean_tpot_s"] - real_co["mean_tpot_s"]
    tpot_delta_total_s = tpot_delta_per_token_s * (DECODE_TOKENS - 1)
    print()
    print(f"real-profile transfer delta (total over decode phase):    "
          f"{transfer_delta_s*1000:.6f} ms")
    print(f"real-profile tpot delta (per-token average):              "
          f"{tpot_delta_per_token_s*1000:.6f} ms")
    print(f"real-profile tpot delta x (decode_tokens - 1) (total):    "
          f"{tpot_delta_total_s*1000:.6f} ms")
    print(f"fraction of transfer delta reaching the decode-phase critical path: "
          f"{100*tpot_delta_total_s/transfer_delta_s if transfer_delta_s else float('nan'):.1f}%")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("colocated", "split"), default=None,
                       help="internal: run one scenario in this process and exit")
    parser.add_argument("--mode", choices=("dummy", "real"), default=None,
                       help="internal: dummy or real execution-time mode")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.scenario:
        _run_scenario(args.scenario, args.mode == "dummy", args.model)
        raise SystemExit(0)
    raise SystemExit(main(args.model))
