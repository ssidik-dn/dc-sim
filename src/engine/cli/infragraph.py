#!/usr/bin/env python3
"""Emit a fabric as InfraGraph, read it back, and confirm nothing was lost.

    PYTHONPATH=src python3 -m engine.cli.infragraph --platform node --out /tmp/n.json
    PYTHONPATH=src python3 -m engine.cli.infragraph --platform rack --out /tmp/r.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from ..infragraph.emit import write_infragraph
from ..infragraph.parse import read_infragraph
from ..physical.builders import build_node_scale, build_rack_scale
from ..physical.topology import Fabric, LinkClass


def build_fabric(kind: str) -> Fabric:
    if kind == "node":
        return build_node_scale(num_machines=2)
    if kind == "rack":
        return build_rack_scale(num_racks=1)
    raise SystemExit(f"unknown platform {kind!r}; use 'node' or 'rack'")


def _class_counts(fabric: Fabric) -> Dict[LinkClass, int]:
    counts: Dict[LinkClass, int] = {}
    for lk in fabric.links:
        counts[lk.link_class] = counts.get(lk.link_class, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="node", choices=["node", "rack"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    fab = build_fabric(args.platform)
    write_infragraph(fab, args.out)
    print(f"wrote {args.out}")

    round_tripped = read_infragraph(args.out)
    before, after = _class_counts(fab), _class_counts(round_tripped)

    print(f"{'':<12}{'original':<12}{'round-tripped':<12}")
    print(f"{'GPUs':<12}{len(fab.gpus):<12}{len(round_tripped.gpus):<12}")
    print(f"{'links':<12}{len(fab.links):<12}{len(round_tripped.links):<12}")
    for cls in LinkClass:
        print(f"{cls.value:<12}{before.get(cls, 0):<12}{after.get(cls, 0):<12}")

    ok = (len(fab.gpus) == len(round_tripped.gpus)
          and len(fab.links) == len(round_tripped.links)
          and before == after)
    print()
    print("round-trip OK" if ok else "round-trip MISMATCH")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
