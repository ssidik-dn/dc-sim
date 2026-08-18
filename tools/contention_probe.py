#!/usr/bin/env python3
"""Measure whether a backend models contention, and by how much.

The discriminator is simple. Put k INDEPENDENT collectives in one Chakra trace
-- no dependency edges between them, so nothing forces an order -- and compare
the completion time against a single collective.

    contention modelled:      T(k) grows with k     (flows share bandwidth)
    contention not modelled:  T(k) stays flat       (flows are free)

This answers the last open question in the blueprint two ways at once. It
confirms empirically that the analytical backend does not model link
contention -- currently asserted from reading its topology format -- and it
produces the target curve the flow model has to reproduce.

Usage from the dc-sim root:

    ASTRA=/work/astra-sim PYTHONPATH=/work/astra-sim python3 tools/contention_probe.py
    ASTRA=/work/astra-sim PYTHONPATH=/work/astra-sim python3 tools/contention_probe.py --backend htsim
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ASTRA = Path(os.environ.get("ASTRA", "/work/astra-sim"))
EX = ASTRA / "examples"
WORK = Path("/tmp/contention_probe")
COMM_TIME = re.compile(r"Comm time:\s*([0-9.eE+-]+)")

BACKENDS = {
    "analytical": ASTRA / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware",
    "congestion_aware": ASTRA / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
    "htsim": ASTRA / "build/astra_htsim/build/bin/AstraSim_HTSim",
}

sys.path.insert(0, str(ASTRA))
try:
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        GlobalMetadata, COMM_COLL_NODE, ALL_REDUCE,
        AttributeProto as ChakraAttr, Node as ChakraNode)
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
        encodeMessage as encode_message)
except ImportError as e:
    sys.exit(f"cannot import Chakra protobuf: {e}\nRun with PYTHONPATH={ASTRA}")


def generate_concurrent(ranks: int, size_bytes: int, n_flows: int,
                        root: Path, chained: bool = False) -> Path:
    """Write a trace with n_flows all-reduce nodes per rank.

    chained=False -> no data_deps, so the nodes are independent and the system
                     layer is free to run them concurrently. This is the case
                     that exposes contention.
    chained=True  -> each node depends on the previous, forcing serial
                     execution. Gives the "fully serialised" reference.
    """
    tag = "chain" if chained else "conc"
    d = root / f"all_reduce_{tag}_{n_flows}x" / f"{ranks}npus_{size_bytes}B"
    d.mkdir(parents=True, exist_ok=True)
    for r in range(ranks):
        with open(d / f"all_reduce.{r}.et", "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            for i in range(n_flows):
                n = ChakraNode()
                n.id = i
                n.name = f"all_reduce_{i}"
                n.type = COMM_COLL_NODE
                n.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
                n.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_REDUCE))
                n.attr.append(ChakraAttr(name="comm_size", int64_val=size_bytes))
                if chained and i > 0:
                    n.data_deps.append(i - 1)
                encode_message(et, n)
    return d / "all_reduce"


def run(binary: Path, workload: Path, network: Path, system: Path,
        topo: Path | None) -> tuple[float, int]:
    cmd = [str(binary),
           f"--workload-configuration={workload}",
           f"--system-configuration={system}",
           f"--network-configuration={network}",
           f"--remote-memory-configuration={EX}/remote_memory/analytical/no_memory_expansion.json"]
    if topo is not None:
        cmd += ["--htsim_opts", "-topo", str(topo)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(binary.parent))
    times = [float(m) for m in COMM_TIME.findall(p.stdout + p.stderr)]
    if not times:
        raise RuntimeError(f"no timing output (exit {p.returncode}): "
                           f"{(p.stderr or p.stdout).strip()[-400:]}")
    return max(times), len(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="analytical", choices=list(BACKENDS))
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--size-kb", type=int, default=1024)
    ap.add_argument("--flows", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--chunks", type=int, default=None,
                    help="active-chunks-per-dimension; raise it to allow "
                         "more collectives in flight")
    ap.add_argument("--policy", default=None, choices=["LIFO", "FIFO"],
                    help="scheduling-policy in the system config")
    args = ap.parse_args()

    binary = BACKENDS[args.backend]
    if not binary.exists():
        return int(print(f"binary not built: {binary}") or 1)

    import json as _json
    network = EX / "network/analytical/HGX-H100-validated.yml"
    system = EX / "system/native_collectives/HGX-H100-validated.json"
    if args.chunks is not None or args.policy is not None:
        cfg = _json.loads(system.read_text())
        if args.chunks is not None:
            cfg["active-chunks-per-dimension"] = args.chunks
        if args.policy is not None:
            cfg["scheduling-policy"] = args.policy
        WORK.mkdir(parents=True, exist_ok=True)
        system = WORK / "system_tuned.json"
        system.write_text(_json.dumps(cfg, indent=2))
    topo = EX.parent / "examples/network/htsim/8nodes.topo" if args.backend == "htsim" else None
    if topo is not None and not topo.exists():
        return int(print(f"htsim topology not found: {topo}") or 1)

    size = args.size_kb * 1024
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"backend   {args.backend}")
    print(f"ranks     {args.ranks}")
    print(f"size      {args.size_kb} KB per collective")
    print(f"network   {network.name}")
    print(f"system    {system.name}"
          f"{f'  chunks={args.chunks}' if args.chunks else ''}"
          f"{f'  policy={args.policy}' if args.policy else ''}")
    if topo:
        print(f"topology  {topo.name}")
    print()
    print(f"{'flows':>6}{'concurrent (ns)':>18}{'chained (ns)':>16}"
          f"{'conc/1flow':>12}{'conc/chained':>14}")
    print("-" * 68)

    base_conc = None
    rows = []
    for k in args.flows:
        wl_c = generate_concurrent(args.ranks, size, k, WORK, chained=False)
        wl_s = generate_concurrent(args.ranks, size, k, WORK, chained=True)
        try:
            t_conc, n1 = run(binary, wl_c, network, system, topo)
            t_chain, n2 = run(binary, wl_s, network, system, topo)
        except RuntimeError as e:
            print(f"{k:>6}  FAILED: {e}")
            continue
        if n1 != args.ranks:
            print(f"  ! reported {n1} ranks, expected {args.ranks}")
        base_conc = base_conc or t_conc
        rows.append((k, t_conc, t_chain))
        print(f"{k:>6}{t_conc:>18.0f}{t_chain:>16.0f}"
              f"{t_conc/base_conc:>11.2f}x{t_conc/t_chain:>13.2f}x")

    print()
    if len(rows) >= 2:
        k_last, t_last, c_last = rows[-1]
        k_first, t_first, _ = rows[0]
        growth = (t_last / t_first) / (k_last / k_first)
        overlap = t_last / c_last          # concurrent vs forced-serial
        print(f"growth per flow  {growth:.2f}   (1.0 = time proportional to k)")
        print(f"conc vs chained  {overlap:.2f}   (1.0 = no overlap at all)")
        print()
        if overlap > 0.95:
            print("  SERIALISED. Dependency-free collectives took exactly as long")
            print("  as explicitly chained ones, so they never ran concurrently.")
            print("  This is not contention -- the network backend never saw two")
            print("  flows at once. No backend beneath this workload layer can")
            print("  exhibit inter-collective contention.")
            print("  Next: try --chunks 4 or --chunks 8, and --policy FIFO.")
        elif overlap < 0.6 and growth > 0.7:
            print("  CONTENTION MODELLED. Concurrent flows overlapped but still")
            print("  cost more as k grew -- they are sharing bandwidth.")
            print("  This curve is the reference the flow model must reproduce.")
        elif overlap < 0.6 and growth < 0.3:
            print("  FREE CONCURRENCY. Flows overlapped with no cost increase,")
            print("  so contention is not modelled at all.")
        else:
            print("  PARTIAL. Inspect per-flow times before concluding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
