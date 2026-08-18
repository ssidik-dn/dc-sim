#!/usr/bin/env python3
"""Inspect how a deployment lands on a fabric under different policies.

    python3 -m engine.cli.place --platform node --tp 8
    python3 -m engine.cli.place --platform rack --tp 16 --ep 4

Reports, per parallel group, the scale-up domain shape and the link classes
touched. The shape tuple is the canonical signature Phase 2 memoises on.
"""
from __future__ import annotations

import argparse
from collections import Counter

from ..logical.deployment import Deployment, ParallelKind, PoolKind, Replica
from ..physical.builders import build_node_scale, build_rack_scale
from ..physical.topology import LinkClass
from ..placement.placement import PlacementError, fragmented, packed, spread


def build_fabric(kind: str):
    if kind == "node":
        return build_node_scale(num_machines=4, gpus_per_machine=8, nics_per_machine=4)
    if kind == "rack":
        return build_rack_scale(num_racks=1, trays_per_rack=18, gpus_per_tray=4)
    raise SystemExit(f"unknown platform {kind!r}; use 'node' or 'rack'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="node", choices=["node", "rack"])
    ap.add_argument("--pool", default="DECODE_ATTN",
                    choices=[p.value for p in PoolKind])
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fab = build_fabric(args.platform)
    dep = Deployment(f"{args.pool}-tp{args.tp}")
    dep.add(Replica(PoolKind(args.pool), 0, tp=args.tp, pp=args.pp,
                    dp=args.dp, ep=args.ep))

    print(fab.summary())
    print(dep.summary())
    print()

    policies = [
        ("packed", lambda: packed(dep, fab)),
        ("spread", lambda: spread(dep, fab)),
        ("fragmented", lambda: fragmented(dep, fab, seed=args.seed, occupied=0.35)),
    ]

    hdr = f"{'policy':<14}{'group':<8}{'domain shape':<20}{'links':<8}{'classes touched'}"
    print(hdr)
    print("-" * len(hdr))

    for name, make in policies:
        try:
            p = make()
        except PlacementError as e:
            print(f"{name:<14}FAILED: {e}")
            continue
        printed = False
        for kind in (ParallelKind.TP, ParallelKind.EP, ParallelKind.PP, ParallelKind.DP):
            groups = dep.replicas[0].groups(kind)
            if not groups:
                continue
            g = groups[0]                       # first group is representative
            links = p.induced_links(g)
            classes = Counter(lk.link_class.value for lk in links.values())
            cls = ", ".join(f"{k}:{v}" for k, v in sorted(classes.items()))
            label = name if not printed else ""
            printed = True
            crossed = "" if not p.crosses_scale_up_boundary(g.ranks) else "  <-- crosses"
            print(f"{label:<14}{kind.value:<8}{str(p.group_shape(g)):<20}"
                  f"{len(links):<8}{cls}{crossed}")
        print()

    print("A shape of (n,) is one scale-up domain. (4,4) or (2,2,2,2) means the")
    print("group is split, and the classes column shows the egress and scale-out")
    print("links that split introduces.")


if __name__ == "__main__":
    main()
