#!/usr/bin/env python3
"""Task 16 spec S4.2: rerun task 15's exact study, changing exactly one
thing -- this project's own (now source-aware) M2N predictor in place of
Frontier's stock analytical one.

Same scenario, same two scheduler variants, same fabric, same everything
else as `tools/run_topology_scheduler_study.py` (imported from here, not
duplicated): one PREFILL replica, one DECODE_ATTN replica with eight
colocated dp lanes, four DECODE_FFN replicas (one near, three symmetric
far). Task 15 could not run this with `--m2n_transfer_config_type empirical`
at all -- `price_transfer` resolved its *source* pool unconditionally, and
DECODE_FFN is the source on the M2N round trip's return leg, so the run
raised `CommGroupError` immediately with more than one FFN replica. Task 16
fixed that (`src/integration/binding_support.py`): an unambiguous pool is
now priced against its *actual* dp lane when `batch.decode_attn_original_dp_id`
recovers it (exact, not a guess -- both legs), and the genuinely
unrecoverable identity -- which FFN replica is really sending on the return
leg -- falls back to the same `bind()` machinery task 14 built for
destinations, now configured via `BindingConfig(NEAREST, "early")`.

Read the task 16 report S2 before trusting a percentage here: `ctx.binding`
is now consulted on *both* directions of the round trip (destination pick,
forward leg; source guess, return leg) through the *same* `BindingState` --
sharing one round-robin cursor/load counter across two conceptually
different decisions was not disentangled further for this task; the report
says why.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as task 15:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_source_binding_study.py [--sweep-load-margin]

Nothing under `upstream/`, `src/engine/`, or the predictors is modified by
this script; `--sweep-load-margin` patches
`integration.cluster_scheduler.topology_aware.LOAD_MARGIN` at the module
level, from this script, per run -- the same read-and-substitute
instrumentation pattern tasks 11-15 already use, not a source edit.
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
from run_topology_scheduler_study import (  # noqa: E402
    ATTN_DP_SIZE, NUM_FFN_REPLICAS, NUM_REQUESTS, DECODE_TOKENS,
    OUTPUT_DIR as _BASE_OUTPUT_DIR, SCALE_UP_GBPS, SCALE_OUT_GBPS, VARIANTS,
    _engine_deployment_and_registry, _placement)
from engine.physical.builders import build_node_scale  # noqa: E402
from engine.placement.binding import BindingPolicy  # noqa: E402

from integration.context import BindingConfig  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path(str(_BASE_OUTPUT_DIR) + "_source_binding")

DEFAULT_LOAD_MARGIN = 2
SWEEP_LOAD_MARGINS = (0, 1, 2, 4, 8)


def _argv(run_id: str) -> list[str]:
    from run_topology_scheduler_study import _argv as _base_argv
    argv = _base_argv(run_id)
    # The one change from task 15's argv: this project's own M2N predictor,
    # now that it can handle a multi-replica source (task 16).
    argv = [a for a in argv if a != "--m2n_transfer_config_type"]
    idx = argv.index("--metrics_config_output_dir")
    argv[idx + 1] = str(OUTPUT_DIR)
    return argv + ["--m2n_transfer_config_type", "empirical"]


_RESULT_MARKER = "SOURCE_BINDING_RESULT="


def _run_scenario(variant: str, load_margin: int) -> None:
    from frontier.types import ClusterType
    import integration.cluster_scheduler.topology_aware as topology_aware
    from integration.cluster_scheduler.topology_aware import TopologyAwareClusterScheduler

    topology_aware.LOAD_MARGIN = load_margin

    fabric = build_node_scale(num_machines=NUM_FFN_REPLICAS, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    placement = _placement(fabric, deployment)
    # binding= is required now: both legs of the M2N round trip are
    # ambiguous with 4 FFN replicas (forward: destination; return: source,
    # task 16), and price_transfer raises without a policy configured.
    install(fabric, placement, deployment, registry,
           binding=BindingConfig(BindingPolicy.NEAREST, timing="early"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.argv = _argv(f"source_binding_{variant}_lm{load_margin}")
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    config = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)
    sim = Simulator(config)

    ffn_scheduler = sim._global_scheduler._cluster_schedulers[ClusterType.DECODE_FFN]
    if variant == "topology_aware":
        TopologyAwareClusterScheduler._assign_ffn_lanes_by_topology(ffn_scheduler)

    lane_map = dict(ffn_scheduler._ffn_lane_to_target_replica)
    ffn_offset = min(ffn_scheduler._cluster.replicas.keys())
    within_domain = 0
    for lane, ffn_replica_id in lane_map.items():
        attn_replica_id, dp_id = lane
        attn_gpu = placement.gpu(deployment.replicas[1].ranks[dp_id])
        ffn_gpu = placement.gpu(deployment.replicas[2 + (ffn_replica_id - ffn_offset)].ranks[0])
        if fabric.same_domain(attn_gpu, ffn_gpu):
            within_domain += 1
    distribution = {}
    for ffn_replica_id in lane_map.values():
        distribution[ffn_replica_id - ffn_offset] = distribution.get(ffn_replica_id - ffn_offset, 0) + 1

    sim.run()

    requests = sim._all_requests
    m2n_time_s = [r.total_m2n_transfer_time for r in requests]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]

    schedulers = getattr(sim._global_scheduler, "_cluster_schedulers", {})
    predictor = None
    for scheduler in schedulers.values():
        p = getattr(scheduler, "_m2n_transfer_predictor", None)
        if p is not None:
            predictor = p
            break

    print(_RESULT_MARKER + json.dumps({
        "mean_m2n_time_s": mean(m2n_time_s) if m2n_time_s else None,
        "mean_tpot_s": mean(tpot_s) if tpot_s else None,
        "n_requests": len(requests),
        "n_tpot": len(tpot_s),
        "distribution": distribution,
        "within_domain": within_domain,
        "total_lanes": len(lane_map),
        "predictor_calls": predictor.calls if predictor else 0,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(variant: str, load_margin: int) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--variant", variant, "--load-margin", str(load_margin)],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise RuntimeError(f"variant={variant!r} load_margin={load_margin} produced no result "
                       f"(exit code {proc.returncode}); see output above")


def main(sweep: bool) -> int:
    results = {}
    for variant in VARIANTS:
        r = _run_scenario_in_subprocess(variant, DEFAULT_LOAD_MARGIN)
        results[variant] = r
        print(f"[{variant:<14}] mean_m2n={r['mean_m2n_time_s']*1000:9.6f} ms  "
             f"mean_tpot={r['mean_tpot_s']*1000:9.6f} ms  "
             f"distribution={r['distribution']}  "
             f"within_domain={r['within_domain']}/{r['total_lanes']}  "
             f"predictor_calls={r['predictor_calls']}")

    print()
    print("mechanism (within-domain fraction):")
    for variant in VARIANTS:
        r = results[variant]
        frac = r["within_domain"] / r["total_lanes"] if r["total_lanes"] else float("nan")
        print(f"  {variant:<14} {r['within_domain']}/{r['total_lanes']} ({100*frac:.1f}%)")

    print()
    print("inter-token latency (tpot) -- this project's own predictor:")
    rr, ta = results["round_robin"], results["topology_aware"]
    delta_tpot = ta["mean_tpot_s"] - rr["mean_tpot_s"]
    print(f"  round_robin:    {rr['mean_tpot_s']*1000:.6f} ms")
    print(f"  topology_aware: {ta['mean_tpot_s']*1000:.6f} ms")
    print(f"  delta:          {delta_tpot*1000:+.6f} ms ({100*delta_tpot/rr['mean_tpot_s']:+.2f}%)")

    print()
    print("mean M2N transfer time -- this project's own predictor:")
    delta_m2n = ta["mean_m2n_time_s"] - rr["mean_m2n_time_s"]
    print(f"  round_robin:    {rr['mean_m2n_time_s']*1000:.6f} ms")
    print(f"  topology_aware: {ta['mean_m2n_time_s']*1000:.6f} ms")
    print(f"  delta:          {delta_m2n*1000:+.6f} ms ({100*delta_m2n/rr['mean_m2n_time_s']:+.2f}%)")

    if sweep:
        print()
        print(f"LOAD_MARGIN sweep (topology_aware variant only, {SWEEP_LOAD_MARGINS}):")
        for lm in SWEEP_LOAD_MARGINS:
            r = _run_scenario_in_subprocess("topology_aware", lm)
            frac = r["within_domain"] / r["total_lanes"]
            print(f"  LOAD_MARGIN={lm:<3d} distribution={r['distribution']}  "
                 f"within_domain={r['within_domain']}/{r['total_lanes']} ({100*frac:.1f}%)  "
                 f"mean_tpot={r['mean_tpot_s']*1000:9.6f} ms  "
                 f"mean_m2n={r['mean_m2n_time_s']*1000:9.6f} ms")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default=None, help="internal")
    parser.add_argument("--load-margin", type=int, default=DEFAULT_LOAD_MARGIN, help="internal")
    parser.add_argument("--sweep-load-margin", action="store_true",
                       help="also sweep LOAD_MARGIN for the topology_aware variant")
    args = parser.parse_args()
    if args.variant:
        _run_scenario(args.variant, args.load_margin)
        raise SystemExit(0)
    raise SystemExit(main(args.sweep_load_margin))
