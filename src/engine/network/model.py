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

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .allocator import (Allocation, FlowKey, LinkKey, MIN_RATE,
                        max_min_fair_share, verify_conservation)


class FlowState(Enum):
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"


@dataclass
class Flow:
    key: FlowKey
    links: Tuple[LinkKey, ...]
    total_bytes: int
    submit_ns: int
    remaining_bytes: float = 0.0
    start_ns: int = 0
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
        return [k for k, f in self._flows.items()
                if f.state is FlowState.IN_FLIGHT]

    @property
    def completed(self) -> List[Completion]:
        return list(self._completed)

    def rate(self, key: FlowKey) -> float:
        return self._alloc.rate(key)

    def link_utilisation(self) -> Dict[LinkKey, float]:
        live = {k: self._flows[k].links for k in self.in_flight}
        link_flows: Dict[LinkKey, set] = {}
        for k, links in live.items():
            for lk in links:
                link_flows.setdefault(lk, set()).add(k)
        return self._alloc.utilisation(link_flows, self.capacity)

    # -- admission ----------------------------------------------------------
    def submit(self, key: FlowKey, links: Sequence[LinkKey],
               total_bytes: int, at_ns: int) -> None:
        """Admit a flow. Advances internal time to `at_ns` first, so flows
        already in flight are credited for the bytes they moved before this
        one arrived -- then reallocates, which is what revises their
        completions."""
        if key in self._flows:
            raise ValueError(f"flow {key!r} already submitted")
        if at_ns < self.now_ns:
            raise ValueError(
                f"cannot submit at {at_ns} ns, model is already at {self.now_ns} ns")
        for lk in links:
            if lk not in self.capacity:
                raise KeyError(f"unknown link {lk!r}")

        self._advance_without_completing(at_ns)
        self._flows[key] = Flow(key, tuple(links), total_bytes, at_ns)
        self._reallocate()

    # -- allocation ---------------------------------------------------------
    def _reallocate(self) -> None:
        live = self.in_flight
        flow_links = {k: self._flows[k].links for k in live}
        self._alloc = max_min_fair_share(flow_links, self.capacity)
        if self.verify:
            verify_conservation(self._alloc, flow_links, self.capacity)
        self.revision += 1
        self.reallocations += 1

    def next_completion_time_ns(self) -> Optional[int]:
        """Earliest predicted completion under the current allocation, or None
        if nothing is in flight. This is what the host schedules its single
        pending event at."""
        best: Optional[float] = None
        for k in self.in_flight:
            r = self._alloc.rate(k)
            if r == float("inf"):
                t = float(self.now_ns)
            elif r <= MIN_RATE:
                continue                      # starved; no completion predicted
            else:
                t = self.now_ns + self._flows[k].remaining_bytes / r
            if best is None or t < best:
                best = t
        if best is None:
            return None
        # Ceil, so a modelled completion never lands before the time the bytes
        # could physically have moved.
        import math
        return int(math.ceil(best))

    # -- time ---------------------------------------------------------------
    def _drain(self, until_ns: int) -> None:
        """Credit every in-flight flow for the bytes it moved up to until_ns."""
        dt = until_ns - self.now_ns
        if dt <= 0:
            return
        for k in self.in_flight:
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

        Completions are processed one at a time in time order, reallocating
        after each -- because a flow finishing frees bandwidth, which speeds up
        everything sharing its links, which changes when they finish.
        """
        out: List[Completion] = []
        if to_ns < self.now_ns:
            raise ValueError(f"cannot rewind from {self.now_ns} to {to_ns}")

        while True:
            nxt = self.next_completion_time_ns()
            if nxt is None or nxt > to_ns:
                break
            self._drain(nxt)
            finished = [k for k in self.in_flight
                        if self._flows[k].remaining_bytes <= 1e-9]
            if not finished:
                # Rounding left a sliver; nudge forward to avoid spinning.
                if nxt >= to_ns:
                    break
                self._drain(nxt + 1)
                continue
            for k in sorted(finished):
                f = self._flows[k]
                f.state = FlowState.COMPLETED
                f.completion_ns = self.now_ns
                f.remaining_bytes = 0.0
                c = Completion(k, f.start_ns, self.now_ns, f.total_bytes,
                               self._alloc.bottleneck.get(k))
                self._completed.append(c)
                out.append(c)
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
