"""Structural validation of an InfraGraph document.

The lesson this applies (see AGENTS.md's ASTRA-sim config-pairing trap): a
format mismatch that gets silently accepted produces a plausible wrong
answer, which is worse than a crash. So this rejects anything ambiguous
rather than guessing, and `parse.py` calls it before touching the document.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Set

NODE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+\.\d+\.[A-Za-z0-9_]+\.\d+$")
KNOWN_SCHEMA_VERSIONS = {"0.1-dcsim"}


class InfraGraphError(ValueError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise InfraGraphError(msg)


def validate_infragraph(doc: dict) -> None:
    for field in ("schema_version", "name", "devices", "domains", "edges"):
        _require(field in doc, f"missing required field: {field!r}")

    version = doc["schema_version"]
    _require(version in KNOWN_SCHEMA_VERSIONS,
             f"unknown schema_version: {version!r}")

    node_names: Set[str] = set()
    gpu_domain_attr: Dict[str, int] = {}

    for dev in doc["devices"]:
        for f in ("instance", "index", "components"):
            _require(f in dev, f"device missing required field: {f!r}")
        instance, index = dev["instance"], dev["index"]
        for comp in dev["components"]:
            for f in ("component", "index"):
                _require(f in comp, f"component missing required field: {f!r}")
            name = f"{instance}.{index}.{comp['component']}.{comp['index']}"
            _require(NODE_NAME_RE.match(name) is not None,
                     f"node name violates naming convention: {name!r}")
            _require(name not in node_names, f"duplicate node name: {name!r}")
            node_names.add(name)
            attrs: Dict[str, Any] = comp.get("attrs", {})
            if comp["component"] == "gpu" and "scale_up_domain" in attrs:
                gpu_domain_attr[name] = attrs["scale_up_domain"]

    domain_members: Dict[int, Set[str]] = {}
    for dom in doc["domains"]:
        for f in ("domain_id", "members"):
            _require(f in dom, f"domain missing required field: {f!r}")
        did = dom["domain_id"]
        for member in dom["members"]:
            _require(member in node_names,
                     f"domain {did} references unknown node {member!r}")
        domain_members[did] = set(dom["members"])

    # Membership is recorded twice on purpose (domains list + per-GPU
    # attribute); the redundancy is what lets us catch a malformed document
    # instead of silently trusting one side of it.
    for name, did in gpu_domain_attr.items():
        _require(did in domain_members and name in domain_members[did],
                 f"domain disagreement: {name} has scale_up_domain={did} "
                 f"but the domains list does not place it there")
    for did, members in domain_members.items():
        for name in members:
            _require(gpu_domain_attr.get(name) == did,
                     f"domain disagreement: {name} is listed in domain {did} "
                     f"but its scale_up_domain attr says "
                     f"{gpu_domain_attr.get(name)!r}")

    for edge in doc["edges"]:
        for f in ("src", "dst", "link_type", "attrs"):
            _require(f in edge, f"edge missing required field: {f!r}")
        _require(edge["src"] in node_names,
                 f"edge references nonexistent node: {edge['src']!r}")
        _require(edge["dst"] in node_names,
                 f"edge references nonexistent node: {edge['dst']!r}")
        attrs = edge["attrs"]
        for f in ("bandwidth_GBps", "latency_ns"):
            _require(f in attrs, f"edge attrs missing required field: {f!r}")
        bw = attrs["bandwidth_GBps"]
        _require(isinstance(bw, (int, float)) and not isinstance(bw, bool) and bw > 0,
                 f"edge {edge['src']}->{edge['dst']} has non-positive "
                 f"bandwidth: {bw!r}")
