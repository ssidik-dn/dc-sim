"""Reconstruct a Fabric from an InfraGraph document.

Validates first and raises on anything ambiguous rather than filling in a
default -- see `validate.py`'s docstring for why.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ..physical.topology import (Fabric, GpuId, Link, LinkClass, Machine,
                                 NicId, Node, ScaleUpDomain, SwitchId)
from .validate import validate_infragraph


def from_infragraph(doc: dict) -> Fabric:
    validate_infragraph(doc)

    fab = Fabric(doc["name"])
    nodes: Dict[str, Node] = {}
    machine_gpus: Dict[int, List[GpuId]] = {}
    machine_nics: Dict[int, List[NicId]] = {}

    for dev in doc["devices"]:
        instance, mid = dev["instance"], dev["index"]
        for comp in dev["components"]:
            component, cidx = comp["component"], comp["index"]
            name = f"{instance}.{mid}.{component}.{cidx}"
            if instance == "machine" and component == "gpu":
                node: Node = GpuId(mid, cidx)
                machine_gpus.setdefault(mid, []).append(node)
            elif instance == "machine" and component == "nic":
                node = NicId(mid, cidx)
                machine_nics.setdefault(mid, []).append(node)
            elif component == "asic":
                node = SwitchId(instance, mid)
            else:
                raise ValueError(f"unrecognised component in node {name!r}")
            nodes[name] = node

    for mid in sorted(machine_gpus.keys() | machine_nics.keys()):
        gpus = sorted(machine_gpus.get(mid, []), key=lambda g: g.index)
        nics = sorted(machine_nics.get(mid, []), key=lambda n: n.index)
        fab.add_machine(Machine(mid, gpus, nics))

    for dom in doc["domains"]:
        members = frozenset(nodes[m] for m in dom["members"])
        fab.add_domain(ScaleUpDomain(dom["domain_id"], members))

    # Edges are directed and the emitter writes both directions explicitly
    # (add_link's own bidirectional=True would double them), so every edge
    # here is added as a one-way link.
    for edge in doc["edges"]:
        src, dst = nodes[edge["src"]], nodes[edge["dst"]]
        link = Link(src, dst, LinkClass(edge["link_type"]),
                   edge["attrs"]["bandwidth_GBps"], edge["attrs"]["latency_ns"])
        fab.add_link(link, bidirectional=False)
        # GPU-to-NIC binding is implied by the forward egress edge only --
        # the reverse (nic -> gpu) edge must not overwrite it.
        if link.link_class is LinkClass.EGRESS and isinstance(src, GpuId) and isinstance(dst, NicId):
            fab.bind_nic(src, dst)

    return fab


def read_infragraph(path: Path) -> Fabric:
    doc = json.loads(Path(path).read_text())
    return from_infragraph(doc)
