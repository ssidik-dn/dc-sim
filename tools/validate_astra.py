#!/usr/bin/env python3
"""Validate the engine's ASTRA-sim backend against hand-run measurements.

The engine should reproduce, from a Placement object, the exact numbers you
got from hand-written config files. If it does, the Phase 2 estimate path is
correct end to end. If it doesn't, the collapse from fabric graph to
per-dimension parameters is wrong somewhere.

Reference values from the hand-run experiment (1 MB all-reduce, 8 ranks,
400 GB/s scale-up, 50 GB/s inter-machine, congestion-unaware backend):

    packed      (8,)          55222 ns
    split 4+4   (4,4)         54955 ns
    split 2x4   (2,2,2,2)    132133 ns

Usage, from the dc-sim root, with astra-sim built:

    ASTRA=/work/astra-sim PYTHONPATH=src:$ASTRA python3 tools/validate_astra.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ASTRA = Path(os.environ.get("ASTRA", "/work/astra-sim"))
BIN = ASTRA / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
REMOTE = ASTRA / "examples/remote_memory/analytical/no_memory_expansion.json"
WORK = Path("/tmp/dcsim_validate")

REFERENCE = {(8,): 55222.0, (4, 4): 54955.0, (2, 2, 2, 2): 132133.0}
SIZE_BYTES = 1024 * 1024
TOLERANCE = 0.02          # 2%; the model should reproduce, not approximate

for p, what in [(BIN, "binary"), (REMOTE, "remote-memory config")]:
    if not p.exists():
        sys.exit(f"missing {what}: {p}\nSet ASTRA=/path/to/astra-sim and build first.")

sys.path.insert(0, str(ASTRA))
try:
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        GlobalMetadata, COMM_COLL_NODE, ALL_REDUCE,
        AttributeProto as ChakraAttr, Node as ChakraNode)
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
        encodeMessage as encode_message)
except ImportError as e:
    sys.exit(f"cannot import Chakra protobuf: {e}\n"
             f"Run with PYTHONPATH=src:{ASTRA}")

from engine.physical.builders import build_node_scale
from engine.cost.astra_backend import AstraSimBackend, MockBackend, FallbackBackend
from engine.cost.astra_config import topology_for_shape, NotExpressible


def generate_trace(collective: str, ranks: int, size_bytes: int, root: Path) -> None:
    """Write Chakra traces in the layout AstraSimBackend expects:
    <root>/<collective>/<ranks>npus_<bytes>B/<collective>.<rank>.et
    """
    d = root / collective / f"{ranks}npus_{size_bytes}B"
    d.mkdir(parents=True, exist_ok=True)
    for r in range(ranks):
        with open(d / f"{collective}.{r}.et", "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            n = ChakraNode()
            n.id = r
            n.name = f"{collective}_{ranks}npus_{size_bytes}B"
            n.type = COMM_COLL_NODE
            n.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            n.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_REDUCE))
            n.attr.append(ChakraAttr(name="comm_size", int64_val=size_bytes))
            encode_message(et, n)


# Fabric parameters chosen to reproduce the hand-run experiment exactly:
#   scale-up      400 GB/s @  936.25 ns
#   inter-machine  50 GB/s @ 5000 ns  (egress latency 0 + scale-out 5000)
# 400G NIC -> 50 GB/s egress, so the emitter's min(egress, scale_out) = 50.
fabric = build_node_scale(
    num_machines=4, gpus_per_machine=8, nics_per_machine=4,
    scale_up_GBps=400.0, scale_up_latency_ns=936.25,
    nic_gbps=400.0, egress_latency_ns=0.0,
    scale_out_GBps=50.0, scale_out_latency_ns=5000.0,
)
print(fabric.summary())
print()

WORK.mkdir(parents=True, exist_ok=True)
generate_trace("all_reduce", 8, SIZE_BYTES, WORK)

backend = AstraSimBackend(binary=BIN, remote_memory_config=REMOTE,
                          workload_root=WORK, work_dir=WORK / "cfg")

print(f"{'shape':<14}{'dims':<6}{'npus_count':<14}{'measured':>12}"
      f"{'reference':>12}{'delta':>9}")
print("-" * 68)

failures = 0
for shape, expected in REFERENCE.items():
    topo = topology_for_shape(shape, fabric)
    res = backend.estimate("all_reduce", SIZE_BYTES, shape, fabric)
    delta = (res.duration_ns - expected) / expected
    ok = abs(delta) <= TOLERANCE
    failures += (not ok)
    print(f"{str(shape):<14}{topo.dims:<6}{str(topo.npus_count):<14}"
          f"{res.duration_ns:>12.0f}{expected:>12.0f}{delta:>8.1%}"
          f"  {'ok' if ok else 'MISMATCH'}")

print()
print("--- memoisation ---")
before = backend.calls
for _ in range(50):
    backend.estimate("all_reduce", SIZE_BYTES, (4, 4), fabric)
print(f"calls={backend.calls}  hits={backend.hits}  hit rate={backend.hit_rate:.1%}")
print(f"50 repeats of an already-seen shape added "
      f"{backend.calls - before} calls and {backend.hits} cache hits")

print()
print("--- inexpressible shape ---")
try:
    topology_for_shape((3, 3, 2), fabric)
    print("  (3,3,2) unexpectedly accepted")
    failures += 1
except NotExpressible:
    print("  (3,3,2) correctly rejected; FallbackBackend would route it")
    fb = FallbackBackend(primary=backend, fallback=MockBackend())
    r = fb.estimate("all_reduce", SIZE_BYTES, (3, 3, 2), fabric)
    print(f"  fallback returned {r.duration_ns:.0f} ns via {r.backend}")

print()
if failures:
    print(f"FAILED: {failures} shape(s) outside {TOLERANCE:.0%} tolerance.")
    print("The fabric-to-dimension collapse does not reproduce the hand-run "
          "configs. Compare the emitted YAML in", WORK / "cfg")
    sys.exit(1)
print("PASS: the engine reproduces the hand-run measurements from a Placement.")
