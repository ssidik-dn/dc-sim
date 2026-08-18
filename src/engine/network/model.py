"""Flow-level network model with revisable completion times.

The causality argument, restated because this module exists to satisfy it: a
function evaluated when an operation starts cannot know about operations that
begin later. Contention means flow A's duration depends on flows admitted after
A. So a completion time cannot be computed once and kept -- it must be an
estimate that is revised whenever the in-flight set changes.

Frontier fixes durations at transfer start and schedules the end event
immediately, which is precisely why that path has to be replaced rather than
merely widened.

Revision strategy: exactly one pending completion event is ever live, at
`next_completion_time_ns()`. On any change to the flow set the model bumps
`revision`; a host holding a stale event compares revisions and drops it. This
needs no support from the host's event queue -- no cancellation, no indexed
heap -- which matters when the host is a dependency rather than a fork.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .allocator import (Allocation, FlowKey, LinkKey, MIN_RATE,
                        max_min_fair_share, verify_conservation)


class FlowState(Enum):
    IN_FLIGHT = "in_flight"    # moving bytes; competes for bandwidth
    DRAINED = "drained"        # bytes fully moved; waiting out its path latency
    COMPLETED = "completed"


@dataclass
class Flow:
    key: FlowKey
    links: Tuple[LinkKey, ...]
    total_bytes: int
    submit_ns: int
    path_latency_ns: float = 0.0
    remaining_bytes: float = 0.0
    start_ns: int = 0
    drained_ns: Optional[float] = None
    bottleneck: Optional[LinkKey] = None
    completion_ns: Optional[int] = None
    state: FlowState = FlowState.IN_FLIGHT

    def __post_init__(self) -> None:
        if self.remaining_bytes == 0.0:
            self.remaining_bytes = float(self.total_bytes)
        self.start_ns = self.submit_ns


@dataclass
class Completion:
    key: FlowKey
    start_ns: int
    completion_ns: int
    total_bytes: int
    bottleneck: Optional[LinkKey]

    @property
    def duration_ns(self) -> int:
        return self.completion_ns - self.start_ns


class FlowNetwork:
    """Admits flows onto shared links and tracks when each finishes.

    Time is integer nanoseconds throughout. Rates are bytes per nanosecond,
    numerically equal to GB/s.
    """

    def __init__(self, capacity: Dict[LinkKey, float],
                 verify: bool = False) -> None:
        self.capacity = dict(capacity)
        self.verify = verify          # assert conservation on every realloc
        self.now_ns: int = 0
        self.revision: int = 0
        self._flows: Dict[FlowKey, Flow] = {}
        self._alloc: Allocation = Allocation()
        self._completed: List[Completion] = []
        self.reallocations: int = 0

    # -- state --------------------------------------------------------------
    @property
    def in_flight(self) -> List[FlowKey]:
        """Not yet reported complete -- includes DRAINED flows, which hold
        no bandwidth but haven't finished waiting out their path latency."""
        return [k for k, f in self._flows.items()
                if f.state is not FlowState.COMPLETED]

    def _active(self) -> List[FlowKey]:
        """Actually competing for bandwidth. Excludes DRAINED: a flow with
        zero bytes left has nothing left to allocate, and the reallocation
        that matters -- freeing its share to survivors -- already happened
        when it drained, not when its latency tail expires."""
        return [k for k, f in self._flows.items()
                if f.state is FlowState.IN_FLIGHT]

    @property
    def completed(self) -> List[Completion]:
        return list(self._completed)

    def rate(self, key: FlowKey) -> float:
        return self._alloc.rate(key)

    def link_utilisation(self) -> Dict[LinkKey, float]:
        live = {k: self._flows[k].links for k in self._active()}
        link_flows: Dict[LinkKey, set] = {}
        for k, links in live.items():
            for lk in links:
                link_flows.setdefault(lk, set()).add(k)
        return self._alloc.utilisation(link_flows, self.capacity)

    # -- admission ----------------------------------------------------------
    def submit(self, key: FlowKey, links: Sequence[LinkKey],
               total_bytes: int, at_ns: int,
               path_latency_ns: float = 0.0) -> None:
        """Admit a flow. Advances internal time to `at_ns` first, so flows
        already in flight are credited for the bytes they moved before this
        one arrived -- then reallocates, which is what revises their
        completions.

        `path_latency_ns` is charged once, when this flow finishes moving
        its bytes (see `advance_to`) -- not here, and not on every
        reallocation in between. The model has no way to derive it from
        `links` (a bare sequence of ids here, not `Link` objects with a
        `latency_ns` field), so the caller -- `engine.network.transfers`,
        which does have the real `Link` objects -- computes and passes it.
        """
        if key in self._flows:
            raise ValueError(f"flow {key!r} already submitted")
        if at_ns < self.now_ns:
            raise ValueError(
                f"cannot submit at {at_ns} ns, model is already at {self.now_ns} ns")
        for lk in links:
            if lk not in self.capacity:
                raise KeyError(f"unknown link {lk!r}")

        self._advance_without_completing(at_ns)
        self._flows[key] = Flow(key, tuple(links), total_bytes, at_ns,
                                path_latency_ns=path_latency_ns)
        self._reallocate()

    # -- allocation ---------------------------------------------------------
    def _reallocate(self) -> None:
        live = self._active()
        flow_links = {k: self._flows[k].links for k in live}
        self._alloc = max_min_fair_share(flow_links, self.capacity)
        if self.verify:
            verify_conservation(self._alloc, flow_links, self.capacity)
        self.revision += 1
        self.reallocations += 1

    def _bandwidth_event_ns(self) -> Optional[float]:
        """When the next active flow's bytes finish moving, ignoring latency.
        This is the moment that matters for reallocation -- freeing a link
        for survivors -- and it is deliberately kept separate from the
        latency-inclusive prediction `next_completion_time_ns()` reports:
        draining every flow up to a latency-inflated time would hold
        survivors at their old, throttled rate for longer than the finishing
        flow actually occupies the link."""
        best: Optional[float] = None
        for k in self._active():
            r = self._alloc.rate(k)
            if r == float("inf"):
                t = float(self.now_ns)
            elif r <= MIN_RATE:
                continue                      # starved; no completion predicted
            else:
                t = self.now_ns + self._flows[k].remaining_bytes / r
            if best is None or t < best:
                best = t
        return best

    def _tail_event_ns(self) -> Optional[float]:
        """When the next DRAINED flow's path latency finishes elapsing."""
        best: Optional[float] = None
        for f in self._flows.values():
            if f.state is FlowState.DRAINED:
                t = f.drained_ns + f.path_latency_ns
                if best is None or t < best:
                    best = t
        return best

    def next_completion_time_ns(self) -> Optional[int]:
        """Earliest predicted completion under the current allocation, or None
        if nothing is in flight. This is what the host schedules its single
        pending event at -- so it must be the latency-inclusive time a
        `Completion` will actually carry, not merely the moment bytes finish
        moving. Path latency is added once per flow (its fixed path-derived
        constant), regardless of how many times that flow's rate has been
        revised, because the addition happens here -- against the flow's
        currently-known bandwidth ETA -- and nowhere else."""
        best: Optional[float] = None
        for k in self._active():
            r = self._alloc.rate(k)
            f = self._flows[k]
            if r == float("inf"):
                bw_finish = float(self.now_ns)
            elif r <= MIN_RATE:
                continue                      # starved; no completion predicted
            else:
                bw_finish = self.now_ns + f.remaining_bytes / r
            t = bw_finish + f.path_latency_ns
            if best is None or t < best:
                best = t
        tail = self._tail_event_ns()
        if tail is not None and (best is None or tail < best):
            best = tail
        if best is None:
            return None
        # Ceil, so a modelled completion never lands before the time the bytes
        # could physically have moved.
        return int(math.ceil(best))

    # -- time ---------------------------------------------------------------
    def _drain(self, until_ns: int) -> None:
        """Credit every actively-transferring flow for the bytes it moved up
        to until_ns. DRAINED flows have nothing left to move -- only their
        latency tail is left, which elapses on its own, not by rate*time."""
        dt = until_ns - self.now_ns
        if dt <= 0:
            return
        for k in self._active():
            r = self._alloc.rate(k)
            if r == float("inf"):
                self._flows[k].remaining_bytes = 0.0
            elif r > MIN_RATE:
                f = self._flows[k]
                f.remaining_bytes = max(f.remaining_bytes - r * dt, 0.0)
        self.now_ns = until_ns

    def _advance_without_completing(self, to_ns: int) -> None:
        self._drain(to_ns)

    def advance_to(self, to_ns: int) -> List[Completion]:
        """Advance to `to_ns`, returning every flow that finished on the way.

        Two kinds of event are interleaved, in time order: a bandwidth event
        (a flow's bytes finish moving -- it leaves the active pool and
        reallocation redistributes its share to survivors, but it is not yet
        reported) and a tail event (a previously-drained flow's path latency
        finishes elapsing -- it is reported now, but nothing reallocates,
        because it stopped competing for bandwidth when it drained).
        """
        out: List[Completion] = []
        if to_ns < self.now_ns:
            raise ValueError(f"cannot rewind from {self.now_ns} to {to_ns}")

        while True:
            candidates = [t for t in (self._bandwidth_event_ns(), self._tail_event_ns())
                         if t is not None]
            if not candidates:
                break
            # Ceil to an integer ns before using it as a time: `now_ns` is
            # integer nanoseconds throughout, and a modelled event must
            # never land before the time it could physically have occurred.
            nxt = int(math.ceil(min(candidates)))
            if nxt > to_ns:
                break
            self._drain(nxt)

            newly_drained = [k for k in self._active()
                            if self._flows[k].remaining_bytes <= 1e-9]
            newly_completed = [
                k for k, f in self._flows.items()
                if f.state is FlowState.DRAINED
                and self.now_ns >= f.drained_ns + f.path_latency_ns - 1e-9]

            if not newly_drained and not newly_completed:
                # Rounding left a sliver; nudge forward to avoid spinning.
                if nxt >= to_ns:
                    break
                self._drain(nxt + 1)
                continue

            for k in sorted(newly_drained):
                f = self._flows[k]
                f.remaining_bytes = 0.0
                f.drained_ns = self.now_ns
                # Captured now, from the allocation that actually governed
                # this flow -- `_reallocate()` below, and any further one
                # before this flow's latency tail expires, replaces
                # `self._alloc` with one that no longer has an entry for it.
                f.bottleneck = self._alloc.bottleneck.get(k)
                if f.path_latency_ns <= 0:
                    f.state = FlowState.COMPLETED
                    f.completion_ns = self.now_ns
                    c = Completion(k, f.start_ns, self.now_ns, f.total_bytes,
                                  f.bottleneck)
                    self._completed.append(c)
                    out.append(c)
                else:
                    f.state = FlowState.DRAINED
            for k in sorted(newly_completed):
                f = self._flows[k]
                f.state = FlowState.COMPLETED
                f.completion_ns = self.now_ns
                c = Completion(k, f.start_ns, self.now_ns, f.total_bytes,
                              f.bottleneck)
                self._completed.append(c)
                out.append(c)

            if newly_drained:
                # Only a bandwidth event changes who competes for what --
                # a tail expiry reports a flow that already stopped
                # competing when it drained, so nothing to reallocate.
                self._reallocate()

        self._drain(to_ns)
        return out

    def run_to_idle(self, limit_ns: int = 10 ** 15) -> List[Completion]:
        """Advance until nothing is in flight. Convenience for tests and for
        costing one operation set in isolation."""
        out: List[Completion] = []
        while self.in_flight:
            nxt = self.next_completion_time_ns()
            if nxt is None or nxt > limit_ns:
                break
            out.extend(self.advance_to(nxt))
        return out
