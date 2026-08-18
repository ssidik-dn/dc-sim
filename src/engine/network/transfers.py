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

from ..physical.topology import Fabric, GpuId, LinkClass
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
    """A flow model over every link in the fabric."""
    return FlowNetwork({lk.id: lk.capacity_GBps for lk in fabric.links},
                       verify=verify)


def run_transfers(fabric: Fabric, transfers: Sequence[Transfer],
                  verify: bool = True) -> List[Completion]:
    """Admit each transfer at its submit time and run until all complete.

    Transfers are admitted in submit-time order, and each admission reallocates
    bandwidth -- which revises the predicted completion of everything already
    in flight that shares a link with the newcomer.
    """
    net = network_for(fabric, verify=verify)
    ordered = sorted(transfers, key=lambda t: (t.submit_ns, t.key))
    done: List[Completion] = []
    for t in ordered:
        # Advancing to the next submit time can complete earlier flows. Those
        # completions must be collected, not discarded -- dropping them loses
        # every transfer that finishes before the last one is admitted.
        done.extend(net.advance_to(t.submit_ns))
        links = [lk.id for lk in fabric.path(t.src, t.dst)]
        net.submit(t.key, links, t.size_bytes, at_ns=t.submit_ns)
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
            verify: bool = True) -> ContentionReport:
    """Run the transfers and summarise where the time went."""
    done = run_transfers(fabric, transfers, verify=verify)
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


def isolated_durations(fabric: Fabric,
                       transfers: Sequence[Transfer]) -> Dict[str, int]:
    """Each transfer run alone, for comparison. Same model, one flow at a
    time -- so any difference is contention and nothing else."""
    out: Dict[str, int] = {}
    for t in transfers:
        net = network_for(fabric, verify=False)
        links = [lk.id for lk in fabric.path(t.src, t.dst)]
        net.submit(t.key, links, t.size_bytes, at_ns=0)
        done = net.run_to_idle()
        out[t.key] = done[0].duration_ns if done else 0
    return out
