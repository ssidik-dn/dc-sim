#!/usr/bin/env python3
"""Task 26 Part B: does task 24's parallelism-versus-memory crossover
survive a plausible non-KV memory overhead?

**Method reused from task 25, not rediscovered.** Neither
`non_kv_cache_overhead_bytes` nor `num_blocks_mode` has a per-cluster CLI
override for DECODE_ATTN (confirmed in task 25: argparse rejects the
per-cluster form, and the global form is silently ignored once
DECODE_ATTN has its own scheduler-config copy). So an assumed overhead
is folded in analytically -- using the exact formula
`memory_planner.py`'s own `MemoryPlanner.get_num_blocks()` uses -- and
the resulting block count is passed as an *explicit*
`--cluster_config_decode_attn_replica_scheduler_config_num_blocks`
override, run through the real `Simulator`, exactly task 25's own
confirmed technique (`tools/run_memory_tp_study.py`'s
`_run_scenario_in_subprocess`, reused directly here, not reimplemented).

**Per-device parameter memory and KV page size, calibrated in task 25,
cited here rather than recomputed** (task 25's report, its own
calibration table):

    tp=1: 1,342,177,280 B param / 1,048,576 B page (all layers)
    tp=2:   671,088,640 B param /   524,288 B page
    tp=4:   335,544,320 B param /   262,144 B page
    tp=8:   201,326,592 B param /   262,144 B page (KV-geometry floor)

Device memory: h800, 80 GiB, same as every task 23-25 run. Feasibility
boundary at a given `(tp, margin, overhead)`: usable memory
`(1-margin)*80GiB` must exceed `param_mem(tp) + overhead`; if it does
not, `MemoryPlanner.get_num_blocks()` raises `FrontierMemoryOOMError`
(confirmed directly in task 24/25) -- there is no valid explicit
`num_blocks` to inject for such a cell, so infeasible cells are
confirmed by calling the real `MemoryPlanner` formula directly (task
25's own diagnostic method) rather than run through a full `Simulator`,
which cannot express "OOM before scheduling starts" as an explicit
override in the first place.

**Overhead=0 is task 24's own grid, cited rather than rerun** (its
report's own table, §2) -- this script only runs the three margins task
24 used (0.9843, 0.984, 0.9) at overhead in {2, 4, 8} GiB, computing
feasibility first (cheap, analytical) so real `Simulator` runs are spent
only on cells that can actually produce a result.

Real h800 compute profiles throughout (Phi-tiny-MoE-instruct);
`install(..., collective=True)` for placement-sensitive
`tensor_parallel_communication_time`, same as tasks 23-25.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_memory_tp_study import _run_scenario_in_subprocess  # noqa: E402

DEVICE_MEMORY_BYTES = 80 * 1024 ** 3
PARAM_MEM_BYTES = {1: 1342177280, 2: 671088640, 4: 335544320, 8: 201326592}
PAGE_SIZE_BYTES = {1: 1048576, 2: 524288, 4: 262144, 8: 262144}
BLOCKS_PER_REQUEST = 3

MARGINS = (0.9843, 0.984, 0.9)          # task 24's own three points
OVERHEADS_GIB = (0, 2, 4, 8)
TP_VALUES = (1, 2, 4, 8)
GIB = 1024 ** 3


def _derived_num_blocks(tp: int, margin: float, overhead_bytes: int):
    """The same formula `MemoryPlanner.get_num_blocks()` uses. Returns
    `None` for an infeasible cell (would raise `FrontierMemoryOOMError`
    if actually run)."""
    requested = (1 - margin) * DEVICE_MEMORY_BYTES
    available = requested - PARAM_MEM_BYTES[tp] - overhead_bytes
    if available <= 0:
        return None
    return int(available // PAGE_SIZE_BYTES[tp])


def _confirm_oom_for_real(tp: int, margin: float, overhead_bytes: int) -> dict:
    """Calls the real Frontier `MemoryPlanner` formula directly (task 25's
    own method) to confirm a predicted-infeasible cell actually raises,
    rather than trusting the arithmetic alone."""
    import subprocess
    script = f"""
import sys
sys.path.insert(0, '/work/simulation/dc-sim/tools')
from run_memory_tp_study import _argv, _build_and_install
sys.argv = _argv('oomconfirm', {tp}, 4096) + ['--seed', '0']
_build_and_install({tp}, False)
from frontier.config import SimulationConfig
from frontier.simulator import Simulator
from frontier.utils.random import set_seeds
from frontier.types import ClusterType
from frontier.scheduler.utils.memory_planner import MemoryPlanner
config = SimulationConfig.create_from_cli_args()
set_seeds(config.seed)
sim = Simulator(config)
sim.run()
sched = sim._global_scheduler.get_cluster_scheduler(ClusterType.DECODE_ATTN)
rid, did = next(iter(sched._dp_replica_schedulers.keys()))
rs = sched.get_dp_replica_scheduler(rid, did)
mp = MemoryPlanner(rs._replica_config, rs._replica, ClusterType.DECODE_ATTN)
try:
    nb = mp.get_num_blocks(block_size=16, gpu_memory_utilization={1 - margin},
                          non_kv_cache_overhead_bytes={overhead_bytes})
    print('CONFIRM_RESULT=' + __import__('json').dumps({{'oom': False, 'num_blocks': nb}}))
except Exception as e:
    print('CONFIRM_RESULT=' + __import__('json').dumps({{'oom': True, 'error': f'{{type(e).__name__}}: {{e}}'}}))
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd="/work/simulation/Frontier")
    for line in proc.stdout.splitlines():
        if line.startswith("CONFIRM_RESULT="):
            return json.loads(line[len("CONFIRM_RESULT="):])
    return {"oom": None, "error": f"no result; stderr={proc.stderr[-2000:]}"}


def main() -> int:
    print("=== feasibility table (analytical, task 25's formula) ===")
    feasible: dict = {}
    for oh_gib in OVERHEADS_GIB:
        oh = oh_gib * GIB
        for m in MARGINS:
            for tp in TP_VALUES:
                nb = _derived_num_blocks(tp, m, oh)
                feasible[(oh_gib, m, tp)] = nb
            row = "  ".join(
                f"tp{tp}={'OOM' if feasible[(oh_gib, m, tp)] is None else 'nb'+str(feasible[(oh_gib, m, tp)])}"
                for tp in TP_VALUES)
            print(f"overhead={oh_gib}GiB margin={m}: {row}")

    print()
    print("=== confirming a sample of predicted-OOM cells for real ===")
    for oh_gib, m, tp in [(2, 0.9843, 1), (4, 0.984, 2), (8, 0.9, 1), (8, 0.9, 8)]:
        r = _confirm_oom_for_real(tp, m, oh_gib * GIB)
        print(f"  overhead={oh_gib}GiB margin={m} tp={tp}: {r}")

    print()
    print("=== real Simulator runs for feasible, non-task-24 cells (overhead in {2,4} GiB) ===")
    print("(overhead=0 is task 24's own grid, cited not rerun; overhead=8 GiB has zero "
         "feasible cells at any of these three margins, confirmed above)")
    results: dict = {}
    for oh_gib in (2, 4):
        oh = oh_gib * GIB
        for m in MARGINS:
            for tp in TP_VALUES:
                nb = feasible[(oh_gib, m, tp)]
                if nb is None:
                    continue
                placements = (False,) if tp == 1 else (False, True)
                for split in placements:
                    r = _run_scenario_in_subprocess(tp, split, nb, seed=0)
                    results[(oh_gib, m, tp, split)] = r
                    label = f"oh={oh_gib}GiB m={m} tp={tp} {'split' if split else 'packed'}"
                    if r.get("error"):
                        print(f"[{label}] ERROR: {r['error']}")
                        continue
                    print(f"[{label}] nb={nb} batch={r['mean_batch_size']:.2f} "
                         f"throughput={r['throughput_rps']:.3f}req/s "
                         f"tpot={r['mean_tpot_ms']:.4f}ms "
                         f"tp_comm={r['tp_ms']:.4f}ms")

    print()
    print("=== crossover: throughput-optimal / latency-optimal degree per (overhead, margin) ===")
    for oh_gib in (2, 4):
        for m in MARGINS:
            cells = {(tp, split): results.get((oh_gib, m, tp, split))
                    for tp in TP_VALUES for split in ((False,) if tp == 1 else (False, True))}
            cells = {k: v for k, v in cells.items() if v and not v.get("error")}
            if not cells:
                print(f"overhead={oh_gib}GiB margin={m}: no feasible cell ran")
                continue
            best_tp_throughput = max(cells.items(), key=lambda kv: kv[1]["throughput_rps"])
            best_tp_latency = min(cells.items(), key=lambda kv: kv[1]["mean_tpot_ms"])
            print(f"overhead={oh_gib}GiB margin={m}: "
                 f"throughput-optimal={best_tp_throughput[0]} "
                 f"({best_tp_throughput[1]['throughput_rps']:.3f} req/s)  "
                 f"latency-optimal={best_tp_latency[0]} "
                 f"({best_tp_latency[1]['mean_tpot_ms']:.4f} ms)  "
                 f"same={best_tp_throughput[0] == best_tp_latency[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
