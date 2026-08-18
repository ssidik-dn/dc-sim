"""Serialise a Fabric to our InfraGraph document.

See `src/engine/infragraph/__init__.py` and `docs/tasks/01-infragraph.md` for
why this is our own format rather than a published one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..physical.topology import Fabric, GpuId, NicId, Node, SwitchId

SCHEMA_VERSION = "0.1-dcsim"


def _gpu_name(g: GpuId) -> str:
    return f"machine.{g.machine}.gpu.{g.index}"


def _nic_name(n: NicId) -> str:
    return f"machine.{n.machine}.nic.{n.index}"


def _switch_name(s: SwitchId) -> str:
    return f"{s.tier}.{s.index}.asic.0"


def _node_name(node: Node) -> str:
    if isinstance(node, GpuId):
        return _gpu_name(node)
    if isinstance(node, NicId):
        return _nic_name(node)
    if isinstance(node, SwitchId):
        return _switch_name(node)
    raise TypeError(f"unrecognised node type: {type(node)!r}")


def _switches(fabric: Fabric) -> List[SwitchId]:
    """Switches aren't tracked as inventory the way machines are -- they only
    appear as link endpoints -- so recover them from the link graph."""
    found: Dict[tuple, SwitchId] = {}
    for lk in fabric.links:
        for node in (lk.src, lk.dst):
            if isinstance(node, SwitchId):
                found[(node.tier, node.index)] = node
    return [found[k] for k in sorted(found)]


def to_infragraph(fabric: Fabric) -> dict:
    """Emit `fabric` as an InfraGraph document (see the schema in the task doc).

    Devices are grouped by machine, not by component: hierarchical naming
    (`<instance>.<index>.<component>.<index>`) only recovers the machine
    boundary if a machine's GPUs and NICs share one device entry. Switches
    get their own device, one `asic` component each, since a switch is not
    part of any machine.

    List order (machines, then switches; components and edges sorted by
    name) is fixed rather than incidental -- it is what makes re-emitting a
    parsed document byte-identical, which is the idempotency test.
    """
    devices: List[Dict[str, Any]] = []
    for mid in sorted(fabric.machines):
        m = fabric.machines[mid]
        components: List[Dict[str, Any]] = []
        for g in sorted(m.gpus, key=lambda g: g.index):
            attrs: Dict[str, Any] = {}
            dom = fabric.domain_of(g)
            if dom is not None:
                attrs["scale_up_domain"] = dom
            components.append({"component": "gpu", "index": g.index, "attrs": attrs})
        for n in sorted(m.nics, key=lambda n: n.index):
            components.append({"component": "nic", "index": n.index, "attrs": {}})
        devices.append({"instance": "machine", "index": mid, "components": components})

    for s in _switches(fabric):
        devices.append({
            "instance": s.tier,
            "index": s.index,
            "components": [{"component": "asic", "index": 0, "attrs": {}}],
        })

    domains = []
    for did in sorted(fabric.domains):
        members = sorted(fabric.domains[did].members)  # GpuId is order=True
        domains.append({"domain_id": did, "members": [_gpu_name(g) for g in members]})

    edges = []
    for lk in sorted(fabric.links, key=lambda l: (_node_name(l.src), _node_name(l.dst))):
        edges.append({
            "src": _node_name(lk.src),
            "dst": _node_name(lk.dst),
            "link_type": lk.link_class.value,
            "attrs": {"bandwidth_GBps": lk.capacity_GBps, "latency_ns": lk.latency_ns},
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "name": fabric.name,
        "devices": devices,
        "domains": domains,
        "edges": edges,
    }


def write_infragraph(fabric: Fabric, path: Path) -> None:
    Path(path).write_text(json.dumps(to_infragraph(fabric), indent=2) + "\n")
