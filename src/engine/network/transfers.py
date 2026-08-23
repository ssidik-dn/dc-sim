"""Cost a set of concurrent transfers over a real fabric, with contention.

This is where the two halves meet. The placement determines which links each
transfer traverses; the flow model shares those links among whatever is in
flight at the time.

The contrast with the estimate path is the point. `estimate()` prices one
operation in isolation and is served by ASTRA-sim. This prices a *set* of
operations against each other, which ASTRA-sim structurally cannot do -- its
workload layer serialises independent collectives, so no backend beneath it
ever sees two flows at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..physical.topology import Fabric, FabricMode, GpuId, LinkClass
from .model import Completion, FlowNetwork


@dataclass(frozen=True)
class Transfer:
    """One point-to-point transfer between placed GPUs."""
    key: str
    src: GpuId
    dst: GpuId
    size_bytes: int
    submit_ns: int = 0


def network_for(fabric: Fabric, verify: bool = True) -> FlowNetwork:
    """A flow model over every link in the fabric.

    `fabric.capacity_index()` is cached on the `Fabric` object itself
    (task 29) -- this used to rebuild a fresh `{id: capacity}` dict over
    every link in the fabric on every call, measured as the dominant cost
    at scale (task 26 report, S2.2). `FlowNetwork.__init__` copies whatever
    dict it's given (`self.capacity = dict(capacity)`), so handing it the
    same cached dict on every call is safe -- nothing here can mutate the
    cache through the `FlowNetwork` it feeds."""
    return FlowNetwork(fabric.capacity_index(), verify=verify)


def _path_latency_ns(fabric: Fabric, links: List[str]) -> float:
    """Sum of each hop's propagation latency along a path. `FlowNetwork`
    only ever sees link ids (`links: Sequence[LinkKey]` in `submit()`), not
    `Link` objects -- it has no `latency_ns` to read. This module has the
    real `Link` objects (from `fabric.path()`/`fabric.route()`), so it
    computes the path term here and passes it through explicitly.

    `fabric.link_index()` (task 29) is exactly the `{id: Link}` mapping
    this used to rebuild fresh on every call -- same cache `network_for`
    now uses, cached once on the `Fabric` object rather than here."""
    by_id = fabric.link_index()
    return sum(by_id[lid].latency_ns for lid in links)


def _links_for(fabric: Fabric, t: "Transfer", mode: FabricMode) -> List[str]:
    """Which links this transfer traverses, under the given routing mode.

    The ECMP hash keys on the TRANSFER KEY, not on the endpoint pair. That is
    deliberate and it models real hardware: switch ECMP hashes the flow 5-tuple,
    so two TCP connections between the same pair of hosts carry different source
    ports, hash differently, and take different paths. Keying on (src, dst)
    instead would pin every transfer between a GPU pair to one path and
    reproduce the very concentration this exists to remove.
    """
    if mode is FabricMode.SPRAYED:
        raise NotImplementedError(
            "SPRAYED routing needs multi-leg flows in FlowNetwork: a sprayed "
            "transfer completes when its slowest leg does, so Completion would "
            "carry several bottlenecks rather than one. That is a change to "
            "completion semantics, not to routing.")
    if mode is FabricMode.SINGLE_PATH:
        return [lk.id for lk in fabric.path(t.src, t.dst)]
    return [lk.id for lk in fabric.route(mode, t.key, t.src, t.dst)]


def run_transfers(fabric: Fabric, transfers: Sequence[Transfer],
                  verify: bool = True,
                  mode: FabricMode = FabricMode.SINGLE_PATH) -> List[Completion]:
    """Admit each transfer at its submit time and run until all complete.

    Transfers are admitted in submit-time order, and each admission reallocates
    bandwidth -- which revises the predicted completion of everything already
    in flight that shares a link with the newcomer.

    `mode` selects routing. SINGLE_PATH is the default so that every existing
    result is unchanged; it takes whichever path breadth-first search finds
    first, which on a fabric with equal-cost paths concentrates every flow onto
    the same one. PER_FLOW_ECMP disperses them, which is what a real fabric with
    several spines does and what makes added spine capacity visible at all.
    """
    net = network_for(fabric, verify=verify)
    ordered = sorted(transfers, key=lambda t: (t.submit_ns, t.key))
    done: List[Completion] = []
    for t in ordered:
        # Advancing to the next submit time can complete earlier flows. Those
        # completions must be collected, not discarded -- dropping them loses
        # every transfer that finishes before the last one is admitted.
        done.extend(net.advance_to(t.submit_ns))
        links = _links_for(fabric, t, mode)
        net.submit(t.key, links, t.size_bytes, at_ns=t.submit_ns,
                   path_latency_ns=_path_latency_ns(fabric, links))
    done.extend(net.run_to_idle())
    return sorted(done, key=lambda c: (c.completion_ns, c.key))


@dataclass
class ContentionReport:
    completions: List[Completion]
    makespan_ns: int
    per_transfer_ns: Dict[str, int]
    bottlenecks: Dict[str, Optional[str]]
    bottleneck_classes: Dict[str, int]

    def slowdown_vs(self, isolated_ns: Dict[str, int]) -> Dict[str, float]:
        """How much longer each transfer took than it would have alone. This is
        the quantity a contention-free estimator cannot produce."""
        return {k: self.per_transfer_ns[k] / isolated_ns[k]
                for k in self.per_transfer_ns if isolated_ns.get(k)}


def analyse(fabric: Fabric, transfers: Sequence[Transfer],
            verify: bool = True,
            mode: FabricMode = FabricMode.SINGLE_PATH) -> ContentionReport:
    """Run the transfers and summarise where the time went."""
    done = run_transfers(fabric, transfers, verify=verify, mode=mode)
    per = {c.key: c.duration_ns for c in done}
    bn = {c.key: c.bottleneck for c in done}

    by_class: Dict[str, int] = {}
    for lid in bn.values():
        if lid is None:
            continue
        lk = next((l for l in fabric.links if l.id == lid), None)
        if lk is not None:
            name = lk.link_class.value
            by_class[name] = by_class.get(name, 0) + 1

    makespan = max((c.completion_ns for c in done), default=0)
    return ContentionReport(done, makespan, per, bn, by_class)


def isolated_durations(fabric: Fabric, transfers: Sequence[Transfer],
                       mode: FabricMode = FabricMode.SINGLE_PATH
                       ) -> Dict[str, int]:
    """Each transfer run alone, for comparison. Same model, one flow at a
    time -- so any difference is contention and nothing else.

    Takes the same mode as the contended run, so a slowdown ratio compares
    like with like. Using different modes for the two would attribute a routing
    difference to contention.
    """
    out: Dict[str, int] = {}
    for t in transfers:
        net = network_for(fabric, verify=False)
        links = _links_for(fabric, t, mode)
        net.submit(t.key, links, t.size_bytes, at_ns=0,
                  path_latency_ns=_path_latency_ns(fabric, links))
        done = net.run_to_idle()
        out[t.key] = done[0].duration_ns if done else 0
    return out
