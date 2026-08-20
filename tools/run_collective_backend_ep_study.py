#!/usr/bin/env python3
"""Task 21 S4.2/S5.3: what did the all_to_all per-pair-volume fix actually
change for expert dispatch?

Reuses task 18's own EP scenario (`tools/run_blind_spot_probe.py`'s
deployment/placement/argv helpers, imported not duplicated: one PREFILL,
one DECODE_ATTN, one DECODE_FFN replica with `moe_expert_parallel_size`
swept, real h800 compute), with `install(..., collective=True)` added so
`expert_parallel_communication_time` is actually priced by
`EngineCCBackend.predict_all_to_all` instead of Frontier's own profiled
table. Reports EP=2 and EP=4, and the same colocated-vs-domain-split-experts
A/B task 18 ran (S1.1's own point: a blind spot's size and its sensitivity
to placement are different questions).

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as every other tool in this project:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_collective_backend_ep_study.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FRONTIER_ROOT = Path("/work/simulation/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_blind_spot_probe import (  # noqa: E402
    OUTPUT_DIR as _BASE_OUTPUT_DIR, SCALE_UP_GBPS, SCALE_OUT_GBPS,
    _deployment_and_registry, _placement, _argv as _base_argv)
from engine.physical.builders import build_node_scale  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path(str(_BASE_OUTPUT_DIR) + "_collective")

_RESULT_MARKER = "COLLECTIVE_EP_RESULT="


def _argv(run_id: str, attn_tp: int, ffn_tp: int, ffn_ep: int) -> list[str]:
    argv = _base_argv(run_id, attn_tp, ffn_tp, ffn_ep)
    idx = argv.index("--metrics_config_output_dir")
    argv[idx + 1] = str(OUTPUT_DIR)
    return argv


def _run_scenario(label: str, attn_tp: int, ffn_tp: int, ffn_ep: int, split_ep: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fabric = build_node_scale(num_machines=4, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _deployment_and_registry(attn_tp, ffn_tp, ffn_ep, split_ep)
    placement = _placement(fabric, deployment, label, split_ep)
    install(fabric, placement, deployment, registry, collective=True)

    tag = f"attn_tp{attn_tp}_ffn_tp{ffn_tp}_ep{ffn_ep}{'_epsplit' if split_ep else ''}_{label}_collective"
    sys.argv = _argv(tag, attn_tp, ffn_tp, ffn_ep)
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
    ep_ms = sum(r["execution_time"]["component_ledger_ms"].get(
        "expert_parallel_communication_time", 0.0) for r in decode_rows)
    denom_ledger_ms = sum(r["execution_time"]["total_time_ms"] for r in decode_rows)
    requests = sim._all_requests
    visible_ms = sum(r.total_m2n_transfer_time for r in requests) * 1000.0
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    mean_tpot_ms = (sum(r.tpot for r in tpot_eligible) / len(tpot_eligible) * 1000.0
                   if tpot_eligible else None)

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "ffn_ep": ffn_ep, "split_ep": split_ep, "label": label,
        "ep_ms": ep_ms, "denom_ms": denom_ledger_ms + visible_ms, "visible_ms": visible_ms,
        "mean_tpot_ms": mean_tpot_ms,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(label, attn_tp, ffn_tp, ffn_ep, split_ep=False) -> dict:
    argv = [sys.executable, _SCRIPT_PATH, "--scenario", label,
           "--attn-tp", str(attn_tp), "--ffn-tp", str(ffn_tp), "--ffn-ep", str(ffn_ep)]
    if split_ep:
        argv.append("--split-ep")
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode})"}


def main() -> int:
    print("=== EP sweep, collective backend selected (compare against task 18's numbers) ===")
    for ep in (2, 4):
        for label in ("colocated", "split"):
            r = _run_scenario_in_subprocess(label, 1, 1, ep)
            if r.get("error"):
                print(f"[{r.get('tag','?')}] ERROR: {r['error']}")
                continue
            print(f"[ep={ep} {label:<9}] expert_parallel_communication_time={r['ep_ms']:.6f}ms "
                 f"denom={r['denom_ms']:.4f}ms mean_tpot={r['mean_tpot_ms']:.6f}ms")

    print()
    print("=== EP=4, experts colocated vs split across domains (S1.1's own check) ===")
    r_colo = _run_scenario_in_subprocess("colocated", 1, 1, 4, split_ep=False)
    r_split = _run_scenario_in_subprocess("colocated", 1, 1, 4, split_ep=True)
    print(f"experts colocated: ep_ms={r_colo.get('ep_ms')}")
    print(f"experts split:     ep_ms={r_split.get('ep_ms')}")
    if not r_colo.get("error") and not r_split.get("error"):
        print(f"identical: {r_colo['ep_ms'] == r_split['ep_ms']}  "
             f"ratio (split/colocated): "
             f"{r_split['ep_ms']/r_colo['ep_ms'] if r_colo['ep_ms'] else float('nan'):.4f}x")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("colocated", "split"), default=None)
    parser.add_argument("--attn-tp", type=int, default=1)
    parser.add_argument("--ffn-tp", type=int, default=1)
    parser.add_argument("--ffn-ep", type=int, default=1)
    parser.add_argument("--split-ep", action="store_true")
    args = parser.parse_args()
    if args.scenario:
        _run_scenario(args.scenario, args.attn_tp, args.ffn_tp, args.ffn_ep, args.split_ep)
        raise SystemExit(0)
    raise SystemExit(main())
