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

import hashlib
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


class FabricMode(Enum):
    """How a flow's bytes are routed when more than one equal-cost path
    exists between its endpoints. See `Fabric.route()` and
    `Fabric.spray_routes()`."""
    SINGLE_PATH = "single_path"        # one path for everyone; today's path()
    PER_FLOW_ECMP = "per_flow_ecmp"    # one path per flow, hash-selected
    SPRAYED = "sprayed"                # every equal-cost path at once


# ---------------------------------------------------------------- endpoints

@dataclass(frozen=True, order=True)
class GpuId:
    machine: int
    index: int                 # index within the machine

    def __post_init__(self) -> None:
        # Cached once per instance -- these are immutable (frozen) and their
        # string form is read millions of times (Link.id below, on every
        # fabric-rebuild call a contention model makes). `object.__setattr__`
        # on an unannotated name is invisible to the dataclass machinery: it
        # does not become a field, so __eq__/__hash__/__lt__ (all generated
        # from `machine`/`index` only) are unaffected -- this is a cache
        # slot, not new state.
        object.__setattr__(self, "_str", f"m{self.machine}.g{self.index}")

    def __str__(self) -> str:
        return self._str


@dataclass(frozen=True, order=True)
class NicId:
    machine: int
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_str", f"m{self.machine}.n{self.index}")

    def __str__(self) -> str:
        return self._str


@dataclass(frozen=True, order=True)
class SwitchId:
    tier: str                  # "leaf" | "spine"
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_str", f"{self.tier}{self.index}")

    def __str__(self) -> str:
        return self._str


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

    def __post_init__(self) -> None:
        # `id` was a `@property` re-formatting an f-string (itself calling
        # `str()` on both endpoints) on every access -- measured at 63.5% of
        # total wall-clock on a 512-GPU fabric (task 26 report, S2.2), since
        # every per-call fabric rebuild touches every link. Endpoints are
        # already cached above; this precomputes `id` itself once, as a
        # plain instance attribute rather than a property, so `lk.id` is a
        # direct attribute lookup with no `__get__` indirection at all.
        object.__setattr__(self, "id", f"{self.src}->{self.dst}")

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
        self._link_index_cache: Optional[Dict[str, Link]] = None
        self._capacity_index_cache: Optional[Dict[str, float]] = None
        self._path_cache: Dict[Tuple[GpuId, GpuId], List[Link]] = {}

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
        # Every current builder finishes adding links before a fabric is
        # ever handed to install() or a predictor (checked directly across
        # src/ and tests/: every add_link call site sits inside a builder
        # function, none after). This invalidates rather than assumes that
        # stays true forever -- if something does call add_link after
        # link_index() was already read once, the next read recomputes
        # instead of serving a stale map.
        self._link_index_cache = None
        self._capacity_index_cache = None
        self._path_cache = {}

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

    def link_index(self) -> Dict[str, Link]:
        """`{link.id: link}` over every link in this fabric, built once and
        cached on this instance rather than rebuilt by every caller.

        Measured need: `network_for()`/`_path_latency_ns()`
        (`engine/network/transfers.py`) each rebuilt this exact mapping on
        every single call regardless of which one or two links a transfer
        actually touched -- 87% of wall-clock on a 512-GPU fabric (task 26
        report, S2.2). Safe to cache because nothing in this project
        mutates a `Fabric` once it has been handed to `install()` or a
        predictor: every `add_link` call site (`builders.py`,
        `infragraph/parse.py`, `infragraph/blueprints.py`, and every test's
        own fabric-construction helper) runs inside a builder function,
        before the fabric is used elsewhere -- checked directly, not
        assumed from `Fabric` not being a frozen dataclass. `add_link`
        still invalidates this cache defensively (see there) in case that
        ever stops being true.
        """
        if self._link_index_cache is None:
            self._link_index_cache = {lk.id: lk for lk in self._links.values()}
        return self._link_index_cache

    def capacity_index(self) -> Dict[str, float]:
        """`{link.id: link.capacity_GBps}` over every link -- the specific
        projection `network_for()`'s flow model needs. Same cache, same
        invalidation, same immutability argument as `link_index()`; kept
        separate because `FlowNetwork` wants capacities, not `Link` objects,
        and building this dict fresh from `link_index()` on every call
        would still pay for a fresh 131,600-entry dict each time -- the
        cost `link_index()` exists to remove, not move.
        """
        if self._capacity_index_cache is None:
            self._capacity_index_cache = {lk.id: lk.capacity_GBps
                                          for lk in self._links.values()}
        return self._capacity_index_cache

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
        stays correct as topologies get less regular. Returns [] for a == b.

        Memoised on the ordered `(a, b)` pair (task 30): for a fixed fabric
        the path between two GPUs never changes, and the BFS itself never
        got cheaper as fabrics grew (task 29 report, S5 -- it stayed the
        dominant cost precisely because nothing before this task touched
        it). Keyed on GPUs, not ranks or placement, so one cache serves
        every placement built on the same fabric object -- confirmed by
        this method's own signature, which takes only `GpuId`s. Same
        immutability argument and same `add_link` invalidation as
        `link_index()`/`capacity_index()`, reused rather than re-derived.
        """
        if a == b:
            return []
        cached = self._path_cache.get((a, b))
        if cached is not None:
            return cached
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
                    result = list(reversed(out))
                    self._path_cache[(a, b)] = result
                    return result
                q.append(lk.dst)
        raise ValueError(f"no path from {a} to {b} in fabric {self.name!r}")

    def equal_cost_paths(self, a: GpuId, b: GpuId) -> List[List[Link]]:
        """Every minimum-hop path from a to b, not just the first BFS finds.

        "Equal-cost" means equal hop count: this model has never costed a
        path by link capacity (`path()` doesn't either), and this doesn't
        start now. Returns `[[]]` for a == b, matching `path()`'s "no links"
        convention for the trivial case. Path order is sorted by link id, so
        it is deterministic for a fixed fabric regardless of insertion or
        traversal order -- routing decisions built on top of this (`route()`,
        `spray_routes()`) depend on that determinism.
        """
        if a == b:
            return [[]]

        from collections import deque
        dist: Dict[Node, int] = {a: 0}
        preds: Dict[Node, List[Link]] = {}
        q = deque([a])
        while q:
            cur = q.popleft()
            nd = dist[cur] + 1
            for lk in self.neighbours(cur):
                if lk.dst not in dist:
                    dist[lk.dst] = nd
                    preds[lk.dst] = [lk]
                    q.append(lk.dst)
                elif dist[lk.dst] == nd:
                    preds[lk.dst].append(lk)

        if b not in dist:
            raise ValueError(f"no path from {a} to {b} in fabric {self.name!r}")

        paths: List[List[Link]] = []

        def backtrack(node: Node, acc: List[Link]) -> None:
            if node == a:
                paths.append(list(reversed(acc)))
                return
            for lk in preds[node]:
                acc.append(lk)
                backtrack(lk.src, acc)
                acc.pop()

        backtrack(b, [])
        paths.sort(key=lambda p: tuple(lk.id for lk in p))
        return paths

    def route(self, mode: FabricMode, flow_key: str, a: GpuId, b: GpuId) -> List[Link]:
        """The path one flow should use to get from a to b under `mode`.

        This chooses a route; it does not execute one. Nothing here submits
        to a flow model or knows about bytes or completion time.
        """
        if mode is FabricMode.SINGLE_PATH:
            return self.path(a, b)

        if mode is FabricMode.PER_FLOW_ECMP:
            paths = self.equal_cost_paths(a, b)
            if len(paths) == 1:
                return paths[0]
            # hashlib, not hash(): builtin str hashing is salted per process
            # (PYTHONHASHSEED), so the same flow could pick a different path
            # on every run. This must be reproducible the way placement is
            # for a given seed.
            digest = hashlib.sha256(f"{flow_key}|{a}|{b}".encode()).digest()
            idx = int.from_bytes(digest, "big") % len(paths)
            return paths[idx]

        if mode is FabricMode.SPRAYED:
            raise NotImplementedError(
                "SPRAYED mode has no single path by definition -- call "
                "spray_routes() for the path+fraction split instead. "
                "Executing a spread flow (one Transfer as several concurrent "
                "partial flows with a combined completion) needs changes to "
                "FlowNetwork's submit/completion semantics that this task "
                "does not make; see the task 04 report.")

        raise ValueError(f"unknown fabric mode: {mode!r}")

    def spray_routes(self, a: GpuId, b: GpuId) -> List[Tuple[List[Link], float]]:
        """Every equal-cost path from a to b, paired with the fraction of a
        flow's bytes it would carry under SPRAYED semantics.

        Even split (1/N across N equal-cost paths): every fabric built by
        this project so far gives same-hop-count paths equal capacity, so
        there is no basis yet for weighting them unevenly. This is a stated
        assumption, not a derived one -- nothing downstream currently
        consumes this to notice if it were wrong. See the task 04 report.

        Routing decision only: this does not execute anything. Turning it
        into an actual spread transfer is out of scope here -- see
        `route()`'s SPRAYED branch and the task 04 report.
        """
        paths = self.equal_cost_paths(a, b)
        fraction = 1.0 / len(paths)
        return [(p, fraction) for p in paths]

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
