#!/usr/bin/env python3
"""Task 24: rerun task 23's grid with KV capacity actually derived from
device memory, so the trade (parallelism buys memory, costs
communication) becomes visible.

**S2.1/S2.3 -- established before running anything, empirically, not
assumed.** Task 23's own coupling check (`run_memory_tp_study.py`'s
`_coupling_check`) left `num_blocks` unset via *omission* -- and Frontier
silently defeats that: `SimulationConfig`'s cluster-config builder gives
every cluster lacking its own per-cluster `..._num_blocks` override a
*shared* `replica_scheduler_config` object (`frontier/config/config.py`'s
`get_cluster_configs_for_disaggregation`), and the memory-planner
derivation (`base_replica_scheduler.py`'s `elif not self._config.num_blocks:`)
runs once, on whichever cluster's replica scheduler happens to construct
first, using *that* cluster's own (tp=1-shaped, at the time) parameters --
then every other cluster sharing the same object inherits the
already-nonzero result and never re-derives. Confirmed directly: with
`num_blocks` merely omitted, `DECODE_ATTN`'s config showed an *identical*
pre-derivation value (matching PREFILL's own unrelated figure) before
`BaseReplicaScheduler.__init__` had even run for it.

**The fix, confirmed to work**: pass `--cluster_config_decode_attn_replica_scheduler_config_num_blocks 0`
*explicitly* (an explicit `0`, not an omitted flag). Frontier's own
per-cluster-override plumbing treats "override present, value 0" as
"give this cluster its own copy," which starts at a genuine `0` and
*does* run the memory-planner branch, independently, for DECODE_ATTN.
Confirmed by direct read-back:
`sim._global_scheduler.get_cluster_scheduler(ClusterType.DECODE_ATTN).get_dp_replica_scheduler(*key)._config.num_blocks`
after this fix tracks `attn_tensor_parallel_size` the way task 23's own
source-reading predicted it should (S2.3): 64,256 (tp=1) -> 129,792
(tp=2, 2.02x) -> 260,864 (tp=4, 4.06x) -> 261,376 (tp=8, barely +0.2%).
The tp=4->8 flattening has a stated mechanism, not a shrug: this model's
`num_kv_heads=4`, and `kv_heads_per_tensor_parallel_worker = ceil(4/attn_tp)`
floors at 1 once `attn_tp>=4` -- so KV-block *geometry* stops shrinking
past tp=4, and only continued weight-memory sharding (which does not
floor) keeps freeing a little more room beyond that point. Task 23's own
conclusion ("coupling is real in source but negligible in magnitude for
this model") is corrected here: the true reason task 23 saw zero movement
was the omitted-flag wiring bug above, not a magnitude argument -- once
wired correctly, the coupling is real, substantial (roughly 2x per TP
doubling up to tp=4), and has a floor mechanism task 23 never got to see.

**S2.1: which mode, and which knob.** `num_blocks_mode` defaults to
`"memory_planner_profiled"` already (no flag needed); `enable_runtime_non_kv_cache_overhead_profiling`
defaults `False`, so no profiling data is required -- the "profiled"
name is misleading here, since without that flag it behaves identically
to plain `"memory_planner"` (`non_kv_cache_overhead_bytes` stays its
0 default either way). `gpu_memory_utilization` (the direct knob the
task's own S1 names) has **no per-cluster CLI override at all** --
confirmed by argparse rejecting
`--cluster_config_decode_attn_replica_scheduler_config_gpu_memory_utilization`,
and confirmed live that the *global* `--vllm_v1_scheduler_config_gpu_memory_utilization`
flag is silently ignored for DECODE_ATTN specifically (reads back as
`None` regardless of what the flag was set to). The only real,
per-cluster-scoped knob left is `memory_margin_fraction`
(`--cluster_config_decode_attn_replica_config_memory_margin_fraction`,
already used by every tool in this project since task 09, always
pinned to `0.2`) -- `gpu_memory_utilization=None` falls back to
`1 - memory_margin_fraction` (`memory_planner.py`'s own
`_get_effective_gpu_memory_utilization`), so sweeping margin *is*
sweeping usable device memory, just through the one door Frontier
actually leaves open per cluster.

**S2.2: does it raise?** Confirmed directly: `margin=0.98438` (util
just below the parameter-memory floor) raises `FrontierMemoryOOMError`
(`reason=parameter_memory_exceeds_requested_budget`), not a silent
clamp. `frontier.main.main()`'s own CLI entrypoint (not used by this
project's tools, which call `Simulator` directly) converts it to
`SystemExit(2)`; this tool's own `except Exception` catches it and
reports the error string per cell, same convention as every prior
real-compute tool here.

**The knob is real but the usable band is razor-thin.** At tp=1, the
formula (calibrated directly, not derived by hand: `page_size_bytes_total
= 1,048,576` exactly, i.e. 1 MiB combined across all layers for this
model) puts the OOM boundary at `margin=0.984375` and a *plateau* past
roughly capacity=8 (this project's own established concurrency ceiling
for a 32-request workload) at `margin<=~0.9840`. The two boundaries are
four decimal places apart. This is reported as a finding (S2.1/S6), not
smoothed over: unlike task 22's `num_blocks` axis (six clean, arbitrary
integers), a device-memory-derived axis for this particular tiny model
on an 80 GB device needs high-precision margin values to land inside
the interesting band at all -- and that band's width itself shifts with
TP degree (S2.3's own finding), so a margin value interesting at tp=1
may already be deep in the unconstrained plateau at tp=8. The grid (S3)
reports derived capacity in every cell for exactly this reason (S7's own
trap).

Real h800 compute profiles throughout (Phi-tiny-MoE-instruct);
`install(..., collective=True)` (task 20) for placement-sensitive
`tensor_parallel_communication_time`, same as task 23.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as every real-profile tool in this project:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_memory_planner_study.py

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
from run_tp_domain_probe import (  # noqa: E402
    MODEL_NAME, ROUTER_TOPK, TOTAL_EXPERTS, TP_KEYS, PP_KEYS,
    SCALE_UP_GBPS, SCALE_OUT_GBPS, _deployment_and_registry, _placement)
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/memory_planner_study_outputs")

BLOCK_SIZE = 16
PREFILL_TOKENS = 32
DECODE_TOKENS = 16
BLOCKS_PER_REQUEST = -(-(PREFILL_TOKENS + DECODE_TOKENS) // BLOCK_SIZE)  # ceil = 3
NUM_REQUESTS = 32
QPS = 20.0
GENEROUS_NUM_BLOCKS = 4096

TP_VALUES = (1, 2, 4, 8)
# S3's own escape valve: 3 device-memory points -- below the knee, at
# it, above -- calibrated at tp=1 (the most memory-hungry degree, so the
# "below the knee" point is genuinely constrained everywhere; higher TP
# degrees may already sit at or above their own knee at the same margin,
# which is the coupling this task exists to see, not a flaw in the
# choice of points).
MARGIN_VALUES = (0.9843, 0.984, 0.9)
N_REPEATS = 3


def _argv(run_id: str, attn_tp: int, margin: float) -> list[str]:
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
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", str(attn_tp),
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_device", "h800",
        # The swept "device memory" axis (S2.1): the only real per-cluster
        # knob into usable memory, since gpu_memory_utilization has no
        # per-cluster override and the global flag is silently ignored
        # for DECODE_ATTN (confirmed live; see module docstring).
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", str(margin),

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
        # PREFILL: pinned generous, explicit override so it never shares
        # DECODE_ATTN's per-cluster copy (each cluster with its own
        # override gets its own object -- see module docstring).
        "--cluster_config_prefill_replica_scheduler_config_num_blocks", str(GENEROUS_NUM_BLOCKS),
        # DECODE_ATTN: explicit 0, NOT omitted -- forces Frontier to give
        # this cluster its own scheduler-config copy and genuinely run
        # the memory-planner derivation using ITS OWN attn_tp/margin,
        # rather than silently inheriting a shared, already-resolved
        # value computed for a different cluster (the bug this task's
        # own S2.3 exists to catch -- see module docstring).
        "--cluster_config_decode_attn_replica_scheduler_config_num_blocks", "0",
        "--cluster_config_decode_attn_replica_scheduler_config_block_size", str(BLOCK_SIZE),

        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(PREFILL_TOKENS),
        "--fixed_request_length_generator_config_decode_tokens", str(DECODE_TOKENS),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(QPS),

        "--metrics_config_output_dir", str(OUTPUT_DIR),
        "--metrics_config_run_id", run_id,
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]


_RESULT_MARKER = "MEMORY_PLANNER_RESULT="
_COUPLING_MARKER = "MEMORY_PLANNER_COUPLING_RESULT="


def _build_and_install(attn_tp: int, split: bool):
    fabric = build_node_scale(num_machines=8, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _deployment_and_registry(attn_tp, 1)
    placement = _placement(fabric, deployment, attn_tp, split)
    install(fabric, placement, deployment, registry, collective=True)
    return placement


def _run_coupling_check(attn_tp: int) -> None:
    """S2.3: does derived capacity actually rise with TP degree, at a
    fixed, comfortably-unconstrained margin (0.2, this project's own
    long-standing default)? One run per degree -- a mechanism check, not
    a headline figure."""
    _build_and_install(attn_tp, split=False)
    tag = f"coupling_tp{attn_tp}"
    sys.argv = _argv(tag, attn_tp, margin=0.2) + ["--seed", "0"]
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds
    from frontier.types import ClusterType

    error = None
    derived_num_blocks = None
    try:
        config = SimulationConfig.create_from_cli_args()
        assert not config.cluster_config.execution_time_predictor_config.enable_dummy_mode
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
        sched = sim._global_scheduler.get_cluster_scheduler(ClusterType.DECODE_ATTN)
        replica_id, dp_id = next(iter(sched._dp_replica_schedulers.keys()))
        replica_sched = sched.get_dp_replica_scheduler(replica_id, dp_id)
        derived_num_blocks = int(replica_sched._config.num_blocks)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    print(_COUPLING_MARKER + json.dumps({
        "tag": tag, "error": error, "attn_tp": attn_tp,
        "derived_num_blocks": derived_num_blocks,
    }), flush=True)


def _run_scenario(attn_tp: int, split: bool, margin: float, seed: int) -> None:
    _build_and_install(attn_tp, split)

    tag = f"tp{attn_tp}_{'split' if split else 'packed'}_m{margin}_seed{seed}"
    sys.argv = _argv(tag, attn_tp, margin)
    sys.argv += ["--seed", str(seed)]
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds
    from frontier.types import ClusterType

    error = None
    sim = None
    derived_num_blocks = None
    try:
        config = SimulationConfig.create_from_cli_args()
        assert not config.cluster_config.execution_time_predictor_config.enable_dummy_mode
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
        sched = sim._global_scheduler.get_cluster_scheduler(ClusterType.DECODE_ATTN)
        replica_id, dp_id = next(iter(sched._dp_replica_schedulers.keys()))
        derived_num_blocks = int(sched.get_dp_replica_scheduler(replica_id, dp_id)._config.num_blocks)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    if error is not None:
        print(_RESULT_MARKER + json.dumps({
            "tag": tag, "error": error, "attn_tp": attn_tp, "split": split,
            "margin": margin, "seed": seed, "derived_num_blocks": derived_num_blocks,
        }), flush=True)
        return

    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    decode_rows = [r for r in rows if r["cluster_type"] in ("DECODE_ATTN", "DECODE_FFN")]
    attn_rows = [r for r in rows if r["cluster_type"] == "DECODE_ATTN"]
    batch_sizes = [len(r["request_ids"]) for r in attn_rows]
    tp_ms = sum(sum(r["execution_time"]["component_ledger_ms"].get(k, 0.0) for k in TP_KEYS)
               for r in decode_rows)
    pp_ms = sum(sum(r["execution_time"]["component_ledger_ms"].get(k, 0.0) for k in PP_KEYS)
               for r in decode_rows)

    requests = sim._all_requests
    completed = [r for r in requests if r.completed]
    preemptions = sum(r.get_total_preemption_count() for r in requests)
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]
    visible_ms = sum(r.total_m2n_transfer_time for r in requests) * 1000.0
    wall_s = max((r.completed_at for r in completed), default=0.0)
    throughput_rps = len(completed) / wall_s if wall_s else 0.0

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "attn_tp": attn_tp, "split": split,
        "margin": margin, "seed": seed, "derived_num_blocks": derived_num_blocks,
        "n_completed": len(completed), "total_preemptions": preemptions,
        "mean_batch_size": mean(batch_sizes) if batch_sizes else None,
        "max_batch_size": max(batch_sizes) if batch_sizes else None,
        "throughput_rps": throughput_rps,
        "mean_tpot_ms": mean(tpot_s) * 1000.0 if tpot_s else None,
        "mean_m2n_time_ms": visible_ms / len(completed) if completed else None,
        "tp_ms": tp_ms, "pp_ms": pp_ms,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(attn_tp: int, split: bool, margin: float, seed: int) -> dict:
    argv = [sys.executable, _SCRIPT_PATH, "--attn-tp", str(attn_tp),
           "--margin", str(margin), "--seed", str(seed)]
    if split:
        argv.append("--split")
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode}); see stderr above",
           "tag": f"tp{attn_tp}_{'split' if split else 'packed'}_m{margin}_seed{seed}"}


def _run_coupling_check_in_subprocess(attn_tp: int) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--coupling-check", "--attn-tp", str(attn_tp)],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_COUPLING_MARKER):
            return json.loads(line[len(_COUPLING_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode}); see stderr above",
           "tag": f"coupling_tp{attn_tp}"}


def _aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return {"error": rows[0].get("error"), "n_runs": 0}
    return {
        "n_runs": len(ok),
        "derived_num_blocks": ok[0]["derived_num_blocks"],
        "mean_batch_size": mean(r["mean_batch_size"] for r in ok if r["mean_batch_size"]),
        "max_batch_size": max(r["max_batch_size"] for r in ok if r["max_batch_size"]),
        "total_preemptions_mean": mean(r["total_preemptions"] for r in ok),
        "throughput_rps_mean": mean(r["throughput_rps"] for r in ok),
        "mean_tpot_ms_mean": mean(r["mean_tpot_ms"] for r in ok if r["mean_tpot_ms"]),
        "mean_tpot_ms_stdev": (pstdev([r["mean_tpot_ms"] for r in ok if r["mean_tpot_ms"]])
                               if len(ok) > 1 else 0.0),
        "tp_ms_mean": mean(r["tp_ms"] for r in ok),
        "mean_m2n_time_ms_mean": mean(r["mean_m2n_time_ms"] for r in ok if r["mean_m2n_time_ms"]),
    }


def main() -> int:
    print("=== S2.2: does insufficient memory raise, or silently clamp? ===")
    oom_probe = _run_scenario_in_subprocess(1, False, 0.98438, 0)
    print(f"  margin=0.98438 (just below the parameter-memory floor): {oom_probe}")
    print()

    print("=== S2.3: does derived capacity rise with TP degree? (margin=0.2, n=1/point) ===")
    coupling = {}
    for tp in TP_VALUES:
        r = _run_coupling_check_in_subprocess(tp)
        coupling[tp] = r
        if r.get("error"):
            print(f"  tp={tp}: ERROR: {r['error']}")
        else:
            print(f"  tp={tp}: derived_num_blocks={r['derived_num_blocks']}")
    print()

    print("=== main grid: device memory (margin) x TP degree x placement ===")
    results = {}
    for tp in TP_VALUES:
        placements = (False,) if tp == 1 else (False, True)
        for split in placements:
            for margin in MARGIN_VALUES:
                runs = [_run_scenario_in_subprocess(tp, split, margin, seed) for seed in range(N_REPEATS)]
                agg = _aggregate(runs)
                results[(tp, split, margin)] = agg
                label = f"tp={tp:<2} {'split ' if split else 'packed'} margin={margin}"
                if agg.get("n_runs", 0) == 0:
                    print(f"[{label}] ERROR: {agg.get('error')}")
                    continue
                print(f"[{label}] n_runs={agg['n_runs']} nb={agg['derived_num_blocks']} "
                     f"cap~{agg['derived_num_blocks']//BLOCKS_PER_REQUEST} "
                     f"batch={agg['mean_batch_size']:.2f} "
                     f"throughput={agg['throughput_rps_mean']:.3f}req/s "
                     f"tpot={agg['mean_tpot_ms_mean']:.4f}ms(+/-{agg['mean_tpot_ms_stdev']:.4f}) "
                     f"tp_comm_sum={agg['tp_ms_mean']:.4f}ms "
                     f"mean_m2n/req={agg['mean_m2n_time_ms_mean']:.4f}ms")

    print()
    print("=== S4: throughput-optimal and latency-optimal TP degree, per device memory ===")
    for margin in MARGIN_VALUES:
        by_tp = {}
        for tp in TP_VALUES:
            placements = (False,) if tp == 1 else (False, True)
            for split in placements:
                r = results.get((tp, split, margin))
                if r and r.get("n_runs", 0) > 0:
                    by_tp[(tp, split)] = r
        if not by_tp:
            continue
        best_throughput = max(by_tp.items(), key=lambda kv: kv[1]["throughput_rps_mean"])
        best_latency = min(by_tp.items(), key=lambda kv: kv[1]["mean_tpot_ms_mean"])
        print(f"margin={margin}: throughput-optimal={best_throughput[0]} "
             f"({best_throughput[1]['throughput_rps_mean']:.3f} req/s)  "
             f"latency-optimal={best_latency[0]} "
             f"({best_latency[1]['mean_tpot_ms_mean']:.4f} ms)  "
             f"same_degree={best_throughput[0][0] == best_latency[0][0]}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-tp", type=int, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coupling-check", action="store_true")
    args = parser.parse_args()
    if args.coupling_check:
        _run_coupling_check(args.attn_tp)
        raise SystemExit(0)
    if args.attn_tp is not None and args.margin is not None:
        _run_scenario(args.attn_tp, args.split, args.margin, args.seed)
        raise SystemExit(0)
    raise SystemExit(main())
