"""Physical inventory and the fabric graph.

The central idea: communication inside a machine is two distinct paths with
different physics, and conflating them is how models of this kind go wrong.

    GPU --scale-up--> GPU    NVLink / xGMI / UALink, memory-semantic, fast
    GPU --egress----> NIC    PCIe or integrated; SHARED between GPUs, ~10x slower
    NIC --scale-out-> switch Ethernet / InfiniBand

A scale-up domain is NOT the same as a machine. On rack-scale designs a single
scale-up domain spans many chassis, so the two boundaries are modelled
independently.

Bandwidth is stored in GB/s throughout, matching ASTRA-sim. NIC vendors quote
Gb/s; use gbps_to_GBps() at the boundary rather than dividing by 8 inline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple


def gbps_to_GBps(gbps: float) -> float:
    """400G NIC -> 50 GB/s. Explicit because mixing the two silently is an
    easy way to be wrong by 8x."""
    return gbps / 8.0


class LinkClass(Enum):
    SCALE_UP = "scale_up"      # GPU <-> GPU over the memory-semantic fabric
    EGRESS = "egress"          # GPU -> NIC; shared, the narrow step off-machine
    SCALE_OUT = "scale_out"    # NIC <-> switch <-> switch


# ---------------------------------------------------------------- endpoints

@dataclass(frozen=True, order=True)
class GpuId:
    machine: int
    index: int                 # index within the machine

    def __str__(self) -> str:
        return f"m{self.machine}.g{self.index}"


@dataclass(frozen=True, order=True)
class NicId:
    machine: int
    index: int

    def __str__(self) -> str:
        return f"m{self.machine}.n{self.index}"


@dataclass(frozen=True, order=True)
class SwitchId:
    tier: str                  # "leaf" | "spine"
    index: int

    def __str__(self) -> str:
        return f"{self.tier}{self.index}"


Node = object  # GpuId | NicId | SwitchId


@dataclass(frozen=True)
class Link:
    """A directed capacity between two nodes. Directed so that egress and
    ingress can carry different capacities if a platform requires it."""
    src: Node
    dst: Node
    link_class: LinkClass
    capacity_GBps: float
    latency_ns: float

    @property
    def id(self) -> str:
        return f"{self.src}->{self.dst}"

    def __str__(self) -> str:
        return f"{self.id} [{self.link_class.value} {self.capacity_GBps:g}GB/s]"


# ---------------------------------------------------------------- inventory

@dataclass
class Machine:
    machine_id: int
    gpus: List[GpuId]
    nics: List[NicId]

    @property
    def gpus_per_nic(self) -> float:
        return len(self.gpus) / len(self.nics) if self.nics else float("inf")


@dataclass
class ScaleUpDomain:
    """The set of GPUs reachable over the memory-semantic fabric. May span
    machines: a Helios rack is 72 GPUs across 18 trays."""
    domain_id: int
    members: FrozenSet[GpuId]

    def __contains__(self, gpu: GpuId) -> bool:
        return gpu in self.members

    @property
    def size(self) -> int:
        return len(self.members)


class Fabric:
    """Physical inventory plus the link graph over it."""

    def __init__(self, name: str = "fabric") -> None:
        self.name = name
        self.machines: Dict[int, Machine] = {}
        self.domains: Dict[int, ScaleUpDomain] = {}
        self._links: Dict[Tuple[Node, Node], Link] = {}
        self._adj: Dict[Node, List[Link]] = {}
        self._gpu_domain: Dict[GpuId, int] = {}
        self._gpu_nic: Dict[GpuId, NicId] = {}

    # -- construction ------------------------------------------------------
    def add_machine(self, machine: Machine) -> None:
        self.machines[machine.machine_id] = machine

    def add_domain(self, domain: ScaleUpDomain) -> None:
        self.domains[domain.domain_id] = domain
        for g in domain.members:
            self._gpu_domain[g] = domain.domain_id

    def add_link(self, link: Link, bidirectional: bool = True) -> None:
        self._links[(link.src, link.dst)] = link
        self._adj.setdefault(link.src, []).append(link)
        if bidirectional:
            rev = Link(link.dst, link.src, link.link_class,
                       link.capacity_GBps, link.latency_ns)
            self._links[(rev.src, rev.dst)] = rev
            self._adj.setdefault(rev.src, []).append(rev)

    def bind_nic(self, gpu: GpuId, nic: NicId) -> None:
        """Which NIC a GPU egresses through. This is what makes egress
        capacity a property of a GPU rather than of a machine."""
        self._gpu_nic[gpu] = nic

    # -- queries -----------------------------------------------------------
    @property
    def gpus(self) -> List[GpuId]:
        return sorted(g for m in self.machines.values() for g in m.gpus)

    @property
    def links(self) -> List[Link]:
        return list(self._links.values())

    def domain_of(self, gpu: GpuId) -> Optional[int]:
        return self._gpu_domain.get(gpu)

    def nic_of(self, gpu: GpuId) -> Optional[NicId]:
        return self._gpu_nic.get(gpu)

    def same_domain(self, a: GpuId, b: GpuId) -> bool:
        da, db = self.domain_of(a), self.domain_of(b)
        return da is not None and da == db

    def neighbours(self, node: Node) -> List[Link]:
        return self._adj.get(node, [])

    def link(self, src: Node, dst: Node) -> Optional[Link]:
        return self._links.get((src, dst))

    def path(self, a: GpuId, b: GpuId) -> List[Link]:
        """Links traversed from a to b. Breadth-first over the graph, so it
        stays correct as topologies get less regular. Returns [] for a == b."""
        if a == b:
            return []
        from collections import deque
        prev: Dict[Node, Tuple[Node, Link]] = {}
        seen = {a}
        q = deque([a])
        while q:
            cur = q.popleft()
            for lk in self.neighbours(cur):
                if lk.dst in seen:
                    continue
                seen.add(lk.dst)
                prev[lk.dst] = (cur, lk)
                if lk.dst == b:
                    out: List[Link] = []
                    n: Node = b
                    while n != a:
                        p, l = prev[n]
                        out.append(l)
                        n = p
                    return list(reversed(out))
                q.append(lk.dst)
        raise ValueError(f"no path from {a} to {b} in fabric {self.name!r}")

    def link_set(self, pairs: Iterable[Tuple[GpuId, GpuId]]) -> Dict[str, Link]:
        """Every distinct link touched by a set of GPU-to-GPU flows. This is
        what a contention model queues on."""
        out: Dict[str, Link] = {}
        for a, b in pairs:
            for lk in self.path(a, b):
                out[lk.id] = lk
        return out

    def summary(self) -> str:
        by_class: Dict[LinkClass, int] = {}
        for lk in self._links.values():
            by_class[lk.link_class] = by_class.get(lk.link_class, 0) + 1
        parts = [f"{c.value}={n}" for c, n in sorted(
            by_class.items(), key=lambda x: x[0].value)]
        return (f"{self.name}: {len(self.machines)} machines, "
                f"{len(self.gpus)} GPUs, {len(self.domains)} scale-up domains, "
                f"links({', '.join(parts)})")
