#!/usr/bin/env python3
"""Task 23: memory capacity and tensor-parallel degree, measured together
for the first time.

**S3's own instruction ("check this before running the full grid") --
answered first, mechanically, not assumed:** `frontier/scheduler/replica_scheduler/base_replica_scheduler.py`'s
`elif not self._config.num_blocks:` (around line 234) only invokes
`MemoryPlanner.get_num_blocks()` -- which subtracts per-device parameter
memory, itself divided by `attn_tensor_parallel_size`
(`frontier/utils/param_counter.py`) -- when a scheduler config's own
`num_blocks` was left at its dataclass default of `0`. Any explicit
nonzero value (exactly what task 22's own sweep, and every other real-
compute tool in this project, passes) makes Frontier skip that branch
entirely and use the value verbatim, with **no reference to tensor-
parallel degree anywhere downstream** (checked
`vllm_v1_engine_replica_scheduler.py`, `base_kv_cache_manager.py`,
`kv_cache_block_pool.py`: none divide/multiply an already-set
`num_blocks` by `tp`). So:

- With an **explicit** `num_blocks` (this task's own main grid, S2, for
  the same reason task 22 needed one -- to land on specific, repeatable
  capacity points relative to the knee): memory and tensor-parallel
  degree are two **independent, additive** axes. No crossover found in
  that grid would mean "the trade never happens"; it would mean "the
  trade doesn't happen *when memory is set explicitly*," a materially
  different claim this report keeps separate.
- With `num_blocks` left at its **default `0`** (memory-planner mode):
  the trade is real and directional -- confirmed below (`_coupling_check`)
  by reading the derived value back off the live scheduler after
  `sim.run()` (`sim._global_scheduler.get_cluster_scheduler(...)
  .get_dp_replica_scheduler(0, 0)._config.num_blocks`) at each TP degree,
  since nothing in Frontier exposes this as a metric.

Both are measured. The main grid uses explicit `num_blocks` (S2's own
escape valve: 3 points -- below, at, above task 22's own knee -- rather
than task 22's full 6, to keep the added TP/placement axis affordable);
`_coupling_check` measures the auto-derive path's magnitude separately,
at TP in {1,2,4,8}, one seed each, since it is a mechanism-and-magnitude
check, not a headline figure this task's own repeat-count trap applies to.

**Placement, reused from tasks 19-21, not reinvented.** `attn_tp`'s own
ranks packed onto one scale-up domain vs spread evenly across two
(`run_tp_domain_probe._placement`); PREFILL and DECODE_FFN always packed
together in a third domain, so the ATTN-FFN M2N hop (colocated vs split
in task 22's sense) is held fixed across every cell here -- this task
varies TP placement, not pool placement, per its own S2 table.
`install(..., collective=True)` (task 20) is required for
`tensor_parallel_communication_time` to respond to that placement at
all; task 19 already established Frontier's own profiled table is
placement-blind without it.

Real h800 compute profiles throughout (Phi-tiny-MoE-instruct); dummy
mode would make every ratio here meaningless, same as tasks 12/22.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as every real-profile tool in this project:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_memory_tp_study.py

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

FRONTIER_ROOT = Path("/work/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tp_domain_probe import (  # noqa: E402
    MODEL_NAME, ROUTER_TOPK, TOTAL_EXPERTS, TP_KEYS, PP_KEYS,
    SCALE_UP_GBPS, SCALE_OUT_GBPS, _deployment_and_registry, _placement)
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/memory_tp_study_outputs")

BLOCK_SIZE = 16
PREFILL_TOKENS = 32
DECODE_TOKENS = 16
BLOCKS_PER_REQUEST = -(-(PREFILL_TOKENS + DECODE_TOKENS) // BLOCK_SIZE)  # ceil = 3
NUM_REQUESTS = 32
QPS = 20.0
GENEROUS_NUM_BLOCKS = 4096

TP_VALUES = (1, 2, 4, 8)
# S2's own escape valve: 3 points, not task 22's 6 -- below (cap=2), at
# (cap=10, task 22's own measured knee), above (cap=40) -- to keep a
# 4-degree x 2-placement axis affordable. Cut: nb in {9,15,60} from task
# 22's own sweep; the knee's *shape* was already established there and
# is not this task's question.
NUM_BLOCKS_VALUES = (6, 30, 120)
N_REPEATS = 3


def _argv(run_id: str, attn_tp: int, decode_attn_num_blocks: int) -> list[str]:
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
        # The one memory knob actually swept -- DECODE_ATTN specifically,
        # same as task 22.
        "--cluster_config_decode_attn_replica_scheduler_config_num_blocks", str(decode_attn_num_blocks),
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
        # write_metrics and store_utilization_metrics must both stay True
        # (task 18's finding, hit again in task 22): they gate the
        # Frontier stage-batch ledger's in-memory capture, not just disk
        # writing -- this study needs that ledger for batch size and the
        # tensor-parallel component breakdown.
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]


def _argv_auto_derive(run_id: str, attn_tp: int) -> list[str]:
    """Same as `_argv`, except DECODE_ATTN's own `num_blocks` is left
    unset (dataclass default 0, triggering the memory-planner branch);
    the *global* default is also left unset for the same reason, so
    PREFILL gets its own explicit per-cluster override instead, to keep
    it from also auto-deriving and confounding the reading."""
    base = _argv(run_id, attn_tp, decode_attn_num_blocks=1)  # placeholder, stripped below
    out = []
    skip_next = False
    for i, tok in enumerate(base):
        if skip_next:
            skip_next = False
            continue
        if tok in ("--vllm_v1_scheduler_config_num_blocks",
                  "--cluster_config_decode_attn_replica_scheduler_config_num_blocks"):
            skip_next = True
            continue
        out.append(tok)
    out += ["--cluster_config_prefill_replica_scheduler_config_num_blocks", str(GENEROUS_NUM_BLOCKS)]
    return out


_RESULT_MARKER = "MEMORY_TP_RESULT="
_COUPLING_MARKER = "MEMORY_TP_COUPLING_RESULT="


def _build_and_install(attn_tp: int, split: bool):
    fabric = build_node_scale(num_machines=8, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _deployment_and_registry(attn_tp, 1)
    placement = _placement(fabric, deployment, attn_tp, split)
    install(fabric, placement, deployment, registry, collective=True)
    return placement


def _run_scenario(attn_tp: int, split: bool, num_blocks: int, seed: int) -> None:
    _build_and_install(attn_tp, split)

    tag = f"tp{attn_tp}_{'split' if split else 'packed'}_nb{num_blocks}_seed{seed}"
    sys.argv = _argv(tag, attn_tp, num_blocks)
    sys.argv += ["--seed", str(seed)]
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

    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    decode_rows = [r for r in rows if r["cluster_type"] in ("DECODE_ATTN", "DECODE_FFN")]
    attn_rows = [r for r in rows if r["cluster_type"] == "DECODE_ATTN"]
    batch_sizes = [len(r["request_ids"]) for r in attn_rows]
    denom_ms = sum(r["execution_time"]["total_time_ms"] for r in decode_rows)
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

    denom_total_ms = denom_ms + visible_ms
    network_ms = tp_ms + pp_ms + visible_ms

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "attn_tp": attn_tp, "split": split,
        "num_blocks": num_blocks, "seed": seed,
        "n_completed": len(completed), "total_preemptions": preemptions,
        "mean_batch_size": mean(batch_sizes) if batch_sizes else None,
        "max_batch_size": max(batch_sizes) if batch_sizes else None,
        "throughput_rps": throughput_rps,
        "mean_tpot_ms": mean(tpot_s) * 1000.0 if tpot_s else None,
        "mean_m2n_time_ms": visible_ms / len(completed) if completed else None,
        "tp_ms": tp_ms, "pp_ms": pp_ms, "visible_ms": visible_ms,
        "network_share_of_decode_step_pct": (100 * network_ms / denom_total_ms
                                             if denom_total_ms else None),
    }), flush=True)


def _run_coupling_check(attn_tp: int) -> None:
    _build_and_install(attn_tp, split=False)

    tag = f"coupling_tp{attn_tp}"
    sys.argv = _argv_auto_derive(tag, attn_tp) + ["--seed", "0"]
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
        "derived_capacity": (derived_num_blocks // BLOCKS_PER_REQUEST
                             if derived_num_blocks else None),
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(attn_tp: int, split: bool, num_blocks: int, seed: int) -> dict:
    argv = [sys.executable, _SCRIPT_PATH, "--attn-tp", str(attn_tp),
           "--num-blocks", str(num_blocks), "--seed", str(seed)]
    if split:
        argv.append("--split")
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode}); see stderr above",
           "tag": f"tp{attn_tp}_{'split' if split else 'packed'}_nb{num_blocks}_seed{seed}"}


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
        "mean_batch_size": mean(r["mean_batch_size"] for r in ok if r["mean_batch_size"]),
        "max_batch_size": max(r["max_batch_size"] for r in ok if r["max_batch_size"]),
        "total_preemptions_mean": mean(r["total_preemptions"] for r in ok),
        "throughput_rps_mean": mean(r["throughput_rps"] for r in ok),
        "mean_tpot_ms_mean": mean(r["mean_tpot_ms"] for r in ok if r["mean_tpot_ms"]),
        "mean_tpot_ms_stdev": (pstdev([r["mean_tpot_ms"] for r in ok if r["mean_tpot_ms"]])
                               if len(ok) > 1 else 0.0),
        "tp_ms_mean": mean(r["tp_ms"] for r in ok),
        "visible_ms_mean": mean(r["visible_ms"] for r in ok),
        "mean_m2n_time_ms_mean": mean(r["mean_m2n_time_ms"] for r in ok if r["mean_m2n_time_ms"]),
        "network_share_pct_mean": mean(r["network_share_of_decode_step_pct"] for r in ok
                                       if r["network_share_of_decode_step_pct"] is not None),
    }


def main() -> int:
    print("=== S3 coupling check: does raising TP degree free KV capacity? ===")
    print("(num_blocks left at its Frontier default of 0 -> memory-planner auto-derive; n=1 per point)")
    coupling = {}
    for tp in TP_VALUES:
        r = _run_coupling_check_in_subprocess(tp)
        coupling[tp] = r
        if r.get("error"):
            print(f"  tp={tp}: ERROR: {r['error']}")
        else:
            print(f"  tp={tp}: derived_num_blocks={r['derived_num_blocks']} "
                 f"(~{r['derived_capacity']} concurrent requests)")
    print()

    print("=== main grid: memory x TP degree x placement (explicit num_blocks) ===")
    results = {}
    for tp in TP_VALUES:
        placements = (False,) if tp == 1 else (False, True)
        for split in placements:
            for nb in NUM_BLOCKS_VALUES:
                runs = [_run_scenario_in_subprocess(tp, split, nb, seed) for seed in range(N_REPEATS)]
                agg = _aggregate(runs)
                results[(tp, split, nb)] = agg
                label = f"tp={tp:<2} {'split ' if split else 'packed'} nb={nb:>4}"
                if agg.get("n_runs", 0) == 0:
                    print(f"[{label}] ERROR: {agg.get('error')}")
                    continue
                print(f"[{label}] n_runs={agg['n_runs']} "
                     f"batch={agg['mean_batch_size']:.2f} "
                     f"throughput={agg['throughput_rps_mean']:.3f}req/s "
                     f"tpot={agg['mean_tpot_ms_mean']:.4f}ms(+/-{agg['mean_tpot_ms_stdev']:.4f}) "
                     f"tp_comm_sum={agg['tp_ms_mean']:.4f}ms "
                     f"mean_m2n/req={agg['mean_m2n_time_ms_mean']:.4f}ms "
                     f"network_share={agg['network_share_pct_mean']:.2f}%")

    print()
    print("=== crossover check: network share vs memory-binding, by TP degree ===")
    for tp in TP_VALUES:
        placements = (False,) if tp == 1 else (True,)
        for split in placements:
            below = results.get((tp, split, NUM_BLOCKS_VALUES[0]))
            at = results.get((tp, split, NUM_BLOCKS_VALUES[1]))
            above = results.get((tp, split, NUM_BLOCKS_VALUES[2]))
            if not (below and at and above):
                continue
            if any(r.get("n_runs", 0) == 0 for r in (below, at, above)):
                continue
            print(f"tp={tp:<2} {'split ' if split else 'packed'}: "
                 f"nb={NUM_BLOCKS_VALUES[0]}->network_share={below['network_share_pct_mean']:.2f}% "
                 f"nb={NUM_BLOCKS_VALUES[1]}->{at['network_share_pct_mean']:.2f}% "
                 f"nb={NUM_BLOCKS_VALUES[2]}->{above['network_share_pct_mean']:.2f}%  "
                 f"(throughput {below['throughput_rps_mean']:.2f}->{at['throughput_rps_mean']:.2f}"
                 f"->{above['throughput_rps_mean']:.2f} req/s)")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-tp", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coupling-check", action="store_true")
    args = parser.parse_args()
    if args.coupling_check:
        _run_coupling_check(args.attn_tp)
        raise SystemExit(0)
    if args.attn_tp is not None and args.num_blocks is not None:
        _run_scenario(args.attn_tp, args.split, args.num_blocks, args.seed)
        raise SystemExit(0)
    raise SystemExit(main())
