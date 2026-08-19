#!/usr/bin/env python3
"""Task 20 spec S4.2: repeat task 19's exact packed-vs-split TP sweep, now
with `EngineCCBackend` actually reachable (`install(..., collective=True)`).

Same scenario, same deployment/placement helpers as
`tools/run_tp_domain_probe.py` (imported, not duplicated): DECODE_ATTN's
`attn_tensor_parallel_size` swept over {2, 4, 8}, packed (one scale-up
domain) vs split (half-and-half across two, matching this project's own
headline placement-penalty shape), real h800 compute throughout.

Task 19 found `tensor_parallel_communication_time` bit-identical between
packed and split at every degree -- Frontier's own profiled table has no
placement input. **This script's whole point is that this number must now
differ.** If it doesn't, the interception isn't taking effect and nothing
else here matters (task 20 spec S4.2's own words).

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as tasks 18/19:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_collective_backend_study.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified by
this script; `install(..., collective=True)` is this project's own
opt-in, not a source edit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FRONTIER_ROOT = Path("/work/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tp_domain_probe import (  # noqa: E402
    MODEL_NAME, ROUTER_TOPK, TOTAL_EXPERTS, TP_KEYS, PP_KEYS,
    SCALE_UP_GBPS, SCALE_OUT_GBPS, NUM_REQUESTS, DECODE_TOKENS,
    _deployment_and_registry, _placement, _argv as _base_argv)
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/collective_backend_study_outputs")

TP_VALUES = (2, 4, 8)


def _argv(run_id: str, attn_tp: int) -> list[str]:
    argv = _base_argv(run_id, attn_tp, 1)
    idx = argv.index("--metrics_config_output_dir")
    argv[idx + 1] = str(OUTPUT_DIR)
    idx = argv.index("--metrics_config_run_id")
    argv[idx + 1] = run_id
    return argv


_RESULT_MARKER = "COLLECTIVE_BACKEND_RESULT="


def _run_scenario(attn_tp: int, split: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fabric = build_node_scale(num_machines=8, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _deployment_and_registry(attn_tp, 1)
    placement = _placement(fabric, deployment, attn_tp, split)
    install(fabric, placement, deployment, registry, collective=True)

    tag = f"tp{attn_tp}_{'split' if split else 'packed'}_collective"
    sys.argv = _argv(tag, attn_tp)
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
    denom_ms = sum(r["execution_time"]["total_time_ms"] for r in decode_rows)
    tp_ms = sum(sum(r["execution_time"]["component_ledger_ms"].get(k, 0.0) for k in TP_KEYS)
               for r in decode_rows)
    pp_ms = sum(sum(r["execution_time"]["component_ledger_ms"].get(k, 0.0) for k in PP_KEYS)
               for r in decode_rows)
    requests = sim._all_requests
    visible_ms = sum(r.total_m2n_transfer_time for r in requests) * 1000.0
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    mean_tpot_ms = (sum(r.tpot for r in tpot_eligible) / len(tpot_eligible) * 1000.0
                   if tpot_eligible else None)
    denom_total_ms = denom_ms + visible_ms

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "attn_tp": attn_tp, "split": split,
        "denom_ms": denom_total_ms, "visible_ms": visible_ms,
        "tp_ms": tp_ms, "pp_ms": pp_ms, "mean_tpot_ms": mean_tpot_ms,
        "num_decode_attn_rows": len([r for r in decode_rows if r["cluster_type"] == "DECODE_ATTN"]),
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(attn_tp: int, split: bool) -> dict:
    argv = [sys.executable, _SCRIPT_PATH, "--attn-tp", str(attn_tp)]
    if split:
        argv.append("--split")
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode}); see stderr above",
           "tag": f"tp{attn_tp}_{'split' if split else 'packed'}_collective"}


def _report_row(r: dict) -> str:
    if r.get("error"):
        return f"[{r['tag']}] ERROR: {r['error']}"
    denom, vis, tp = r["denom_ms"], r["visible_ms"], r["tp_ms"]
    total_comm = vis + tp + r["pp_ms"]
    headline = 100 * vis / total_comm if total_comm else float("nan")
    return (f"[{r['tag']:<24}] denom={denom:9.4f}ms visible={vis:9.4f}ms "
           f"tp_comm={tp:9.6f}ms({100*tp/denom:5.3f}%) "
           f"mean_tpot={r['mean_tpot_ms']:.6f}ms headline={headline:6.2f}%")


def main() -> int:
    results = {}
    for tp in TP_VALUES:
        rp = _run_scenario_in_subprocess(tp, split=False)
        rs = _run_scenario_in_subprocess(tp, split=True)
        results[tp] = (rp, rs)
        print(_report_row(rp))
        print(_report_row(rs))

    print()
    print("=== packed vs split, side by side ===")
    for tp in TP_VALUES:
        rp, rs = results[tp]
        if rp.get("error") or rs.get("error"):
            print(f"tp={tp}: packed_error={rp.get('error')} split_error={rs.get('error')}")
            continue
        same = rp["tp_ms"] == rs["tp_ms"]
        delta_tpot = rs["mean_tpot_ms"] - rp["mean_tpot_ms"]
        print(f"tp={tp:<2} packed_tp_comm={rp['tp_ms']:.6f}ms  split_tp_comm={rs['tp_ms']:.6f}ms  "
             f"identical={same}  ratio={rs['tp_ms']/rp['tp_ms'] if rp['tp_ms'] else float('nan'):.2f}x  "
             f"tpot_delta={delta_tpot:+.6f}ms")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-tp", type=int, default=None)
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    if args.attn_tp is not None:
        _run_scenario(args.attn_tp, args.split)
        raise SystemExit(0)
    raise SystemExit(main())
