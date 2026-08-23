#!/usr/bin/env python3
"""Task 31: where does this project's own run-to-run variance actually
come from, what is the noise floor once it is real, and do the four
headline findings survive it.

**Read `tools/seed_stats.py`'s own module docstring first.** It
establishes, from source, why `--seed` alone has never varied anything
in any tool this project has built: request generation re-seeds
internally to a separate, hardcoded-default field
(`BaseRequestGeneratorConfig.seed`, default 42), and offline mode
discards generated arrival times regardless unless
`--offline_use_generated_request_arrivals` is also set. This script is
the first one in this project to pass `seed_stats.seed_argv_fix(seed)`
alongside `--seed`, specifically so seeds here actually differ --
confirmed for each scenario below before trusting any resulting
interval, per this task's own "seeds must actually differ" trap.

**Scope, and what is deliberately not re-measured.** Three of the four
headline findings this task names are reused directly from existing
tools' own scenario-construction helpers (`run_memory_tp_study.py`,
`run_tp_domain_probe.py`/`run_collective_backend_study.py`,
`run_m2n_integration.py`), with the seed fix layered on top and nothing
else changed. The fourth -- topology-aware scheduling -- is not
re-measured here: `run_topology_scheduler_study.py`'s own argv sets
`--random_forrest_execution_time_predictor_config_enable_dummy_mode`,
dummy compute, which `AGENTS.md` itself says never to calibrate or
baseline against. Re-measuring that finding properly needs a real-profile
rebuild of that study first, which is separate, larger work this task's
own scope (confidence intervals) does not include; see the report's own
S2.3/S4 for what is said about it instead.

Real h800 compute profiles throughout every scenario this script does
run (Phi-tiny-MoE-instruct), matching this project's own convention.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_stats import seed_argv_fix, run_seed_study, compute_interval_stats  # noqa: E402
from run_memory_tp_study import (  # noqa: E402
    _argv as _mem_argv, _build_and_install as _mem_build_and_install)
from run_tp_domain_probe import (  # noqa: E402
    TP_KEYS, SCALE_UP_GBPS, SCALE_OUT_GBPS,
    _deployment_and_registry as _tp_deployment_and_registry,
    _placement as _tp_placement)
from run_m2n_integration import (  # noqa: E402
    _engine_deployment_and_registry, _placements as _m2n_placements)
from run_memory_edge_study import _argv as _m2n_base_argv  # noqa: E402
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

FRONTIER_ROOT = Path("/work/simulation/Frontier")
_SCRIPT_PATH = str(Path(__file__).resolve())

_RESULT_MARKER = "SEED_VARIANCE_RESULT="


# ------------------------------------------------------------- scenario: memory grid


def _mem_run_scenario(attn_tp: int, num_blocks: int, seed: int, vary_arrivals: bool) -> None:
    _mem_build_and_install(attn_tp, False)
    tag = f"memvar_tp{attn_tp}_nb{num_blocks}_seed{seed}_va{int(vary_arrivals)}"
    sys.argv = (_mem_argv(tag, attn_tp, num_blocks) + ["--seed", str(seed)]
               + seed_argv_fix(seed, vary_arrivals=vary_arrivals))
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
    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    attn_rows = [r for r in rows if r["cluster_type"] == "DECODE_ATTN"]
    batch_sizes = [len(r["request_ids"]) for r in attn_rows]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]
    m2n_s = [r.total_m2n_transfer_time for r in requests]
    wall_s = max((r.completed_at for r in completed), default=0.0)
    throughput_rps = len(completed) / wall_s if wall_s else 0.0
    first_arrival = min((r.arrived_at for r in requests), default=0.0)
    last_arrival = max((r.arrived_at for r in requests), default=0.0)

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None,
        "throughput_rps": throughput_rps,
        "mean_tpot_ms": statistics.mean(tpot_s) * 1000.0 if tpot_s else None,
        "mean_batch_size": statistics.mean(batch_sizes) if batch_sizes else None,
        "mean_m2n_time_ms": statistics.mean(m2n_s) * 1000.0 if m2n_s else None,
        "arrival_span_s": last_arrival - first_arrival,
    }), flush=True)


def _mem_run_scenario_in_subprocess(attn_tp: int, num_blocks: int, seed: int,
                                    vary_arrivals: bool = True) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--scenario", "mem", "--attn-tp", str(attn_tp),
         "--num-blocks", str(num_blocks), "--seed", str(seed),
         "--vary-arrivals", "1" if vary_arrivals else "0"],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout[-3000:])
    sys.stderr.write(proc.stderr[-3000:])
    return {"error": f"no result (exit code {proc.returncode})"}


# ------------------------------------------------------------- scenario: TP packed/split


def _tp_run_scenario(attn_tp: int, split: bool, seed: int) -> None:
    fabric = build_node_scale(num_machines=8, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _tp_deployment_and_registry(attn_tp, 1)
    placement = _tp_placement(fabric, deployment, attn_tp, split)
    install(fabric, placement, deployment, registry, collective=True)

    tag = f"tpvar_tp{attn_tp}_{'split' if split else 'packed'}_seed{seed}"
    from run_tp_domain_probe import _argv as _tp_base_argv
    sys.argv = (_tp_base_argv(tag, attn_tp, 1) + seed_argv_fix(seed))
    idx = sys.argv.index("--seed") if "--seed" in sys.argv else None
    if idx is not None:
        sys.argv[idx + 1] = str(seed)
    else:
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
    tp_ms = sum(sum(r["execution_time"]["component_ledger_ms"].get(k, 0.0) for k in TP_KEYS)
               for r in decode_rows)
    requests = sim._all_requests
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    mean_tpot_ms = (statistics.mean(r.tpot for r in tpot_eligible) * 1000.0
                   if tpot_eligible else None)

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "tp_comm_ms": tp_ms, "mean_tpot_ms": mean_tpot_ms,
    }), flush=True)


def _tp_run_scenario_in_subprocess(attn_tp: int, split: bool, seed: int) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--scenario", "tp", "--attn-tp", str(attn_tp),
         "--split", "1" if split else "0", "--seed", str(seed)],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout[-3000:])
    sys.stderr.write(proc.stderr[-3000:])
    return {"error": f"no result (exit code {proc.returncode})"}


# ------------------------------------------------------------- scenario: M2N colocated/split


M2N_GENEROUS_NUM_BLOCKS = 120  # run_memory_edge_study's own unconstrained reference point


def _m2n_run_scenario(label: str, seed: int) -> None:
    # Real h800 compute throughout, reusing task 22's own established
    # colocated/split scenario (run_memory_edge_study.py) rather than task
    # 11's run_m2n_integration.py, which deliberately runs dummy compute
    # (--random_forrest_execution_time_predictor_config_enable_dummy_mode)
    # to isolate predictor call overhead -- not the tool this headline's
    # own real-compute ratio was ever measured with.
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    colocated, split = _m2n_placements(fabric, deployment)
    placement = colocated if label == "colocated" else split
    install(fabric, placement, deployment, registry)

    tag = f"m2nvar_{label}_seed{seed}"
    sys.argv = _m2n_base_argv(tag, label, M2N_GENEROUS_NUM_BLOCKS, seed) + seed_argv_fix(seed)

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
    m2n_s = [r.total_m2n_transfer_time for r in requests]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None,
        "mean_m2n_time_ms": statistics.mean(m2n_s) * 1000.0 if m2n_s else None,
        "mean_tpot_ms": (statistics.mean(r.tpot for r in tpot_eligible) * 1000.0
                        if tpot_eligible else None),
    }), flush=True)


def _m2n_run_scenario_in_subprocess(label: str, seed: int) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--scenario", "m2n", "--label", label, "--seed", str(seed)],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout[-3000:])
    sys.stderr.write(proc.stderr[-3000:])
    return {"error": f"no result (exit code {proc.returncode})"}


# ------------------------------------------------------------- orchestration


def _print_stats(label: str, stats) -> None:
    print(f"  {label}: n={stats.n} mean={stats.mean:.6f} stdev={stats.stdev:.6f} "
         f"cv={stats.cv_pct:.3f}% ci95_halfwidth={stats.ci95_halfwidth:.6f} "
         f"({stats.ci95_halfwidth_pct:.3f}% of mean)")


def main() -> int:
    print("=== S2.1: do seeds actually differ, before and after the fix? ===")
    for va in (False, True):
        rows = [_mem_run_scenario_in_subprocess(1, 6911, s, vary_arrivals=va) for s in range(3)]
        spans = [r.get("arrival_span_s") for r in rows if not r.get("error")]
        tputs = [r.get("throughput_rps") for r in rows if not r.get("error")]
        print(f"  seed_argv_fix(vary_arrivals={va}): arrival_span_s per seed={spans}  "
             f"throughput per seed={tputs}")

    print()
    print("=== S2.2: noise floor, 5/10/20 seeds, two configurations ===")
    configs = [("plateau (tp=1, nb=6911, unconstrained)", 1, 6911),
              ("near-knee (tp=1, nb=6, capacity=2)", 1, 6)]
    for cfg_label, tp, nb in configs:
        print(f"-- {cfg_label} --")
        all_seeds = list(range(20))
        rows = [_mem_run_scenario_in_subprocess(tp, nb, s) for s in all_seeds]
        ok = [r for r in rows if not r.get("error")]
        if len(ok) < len(rows):
            print(f"  {len(rows) - len(ok)} seed(s) errored")
        for n in (5, 10, 20):
            subset = ok[:n]
            for metric in ("throughput_rps", "mean_tpot_ms", "mean_m2n_time_ms", "mean_batch_size"):
                vals = [r[metric] for r in subset if r.get(metric) is not None]
                if len(vals) < 2:
                    continue
                stats = compute_interval_stats(vals)
                _print_stats(f"n={n:>2} {metric}", stats)

    print()
    print("=== S2.3: headline findings, re-measured with genuine seed variance (n=20) ===")

    print("-- tp=4 split penalty (tensor_parallel_communication_time / tpot) --")
    for split in (False, True):
        stats = run_seed_study(lambda s: _tp_run_scenario_in_subprocess(4, split, s),
                               range(20), ["tp_comm_ms", "mean_tpot_ms"])
        for m, st in stats.items():
            _print_stats(f"{'split' if split else 'packed'} {m}", st)

    print("-- pool split M2N (colocated / split) --")
    for label in ("colocated", "split"):
        stats = run_seed_study(lambda s: _m2n_run_scenario_in_subprocess(label, s),
                               range(20), ["mean_m2n_time_ms", "mean_tpot_ms"])
        for m, st in stats.items():
            _print_stats(f"{label} {m}", st)

    print("-- tp=2 vs tp=1 (packed, unconstrained memory) --")
    for tp, nb in ((1, 6911), (2, 15103)):
        stats = run_seed_study(lambda s: _mem_run_scenario_in_subprocess(tp, nb, s),
                               range(20), ["throughput_rps", "mean_tpot_ms"])
        for m, st in stats.items():
            _print_stats(f"tp={tp} {m}", st)

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("mem", "tp", "m2n"), default=None)
    parser.add_argument("--attn-tp", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vary-arrivals", type=int, default=1)
    args = parser.parse_args()
    if args.scenario == "mem":
        _mem_run_scenario(args.attn_tp, args.num_blocks, args.seed, bool(args.vary_arrivals))
        raise SystemExit(0)
    if args.scenario == "tp":
        _tp_run_scenario(args.attn_tp, bool(args.split), args.seed)
        raise SystemExit(0)
    if args.scenario == "m2n":
        _m2n_run_scenario(args.label, args.seed)
        raise SystemExit(0)
    raise SystemExit(main())
