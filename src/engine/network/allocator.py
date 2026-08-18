"""Max-min fair-share bandwidth allocation over a set of shared links.

Progressive filling. Raise every flow's rate together until some link
saturates; fix the flows crossing that link at that rate; remove their demand
from every link they touch; repeat with what remains.

The result is the unique max-min fair allocation, which has exact closed-form
answers for the cases that matter -- two flows sharing one link get half each,
three flows through a 2:1 oversubscribed uplink split determinately. That makes
arithmetic the acceptance criterion rather than another simulator's output,
which is why this is testable without a reference implementation.

Units: capacity in GB/s and rate in bytes per nanosecond are the same number,
since 1 GB/s == 1 byte/ns. No conversion, and no factor-of-1e9 bugs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Set

# A flow is identified by an opaque key; a link by its id.
FlowKey = str
LinkKey = str

# Rates below this are treated as zero: a flow on a saturated link makes no
# progress, and dividing by a denormal rate produces nonsense completions.
MIN_RATE = 1e-12


@dataclass
class Allocation:
    """Rate per flow, plus which link constrained each one."""
    rates: Dict[FlowKey, float] = field(default_factory=dict)
    bottleneck: Dict[FlowKey, LinkKey] = field(default_factory=dict)

    def rate(self, flow: FlowKey) -> float:
        return self.rates.get(flow, 0.0)

    def utilisation(self, link_flows: Mapping[LinkKey, Set[FlowKey]],
                    capacity: Mapping[LinkKey, float]) -> Dict[LinkKey, float]:
        """Fraction of each link's capacity actually allocated. Never above
        1.0 -- that is the conservation invariant."""
        out: Dict[LinkKey, float] = {}
        for lk, flows in link_flows.items():
            cap = capacity[lk]
            used = sum(self.rates.get(f, 0.0) for f in flows)
            out[lk] = used / cap if cap > 0 else 0.0
        return out


def max_min_fair_share(
    flow_links: Mapping[FlowKey, Sequence[LinkKey]],
    capacity: Mapping[LinkKey, float],
) -> Allocation:
    """Allocate `capacity` among flows, each of which traverses `flow_links`.

    A flow with an empty link list is unconstrained -- it never leaves a GPU --
    and receives infinite rate, meaning it completes immediately.
    """
    alloc = Allocation()

    unfixed: Set[FlowKey] = set()
    for f, links in flow_links.items():
        if not links:
            alloc.rates[f] = float("inf")
        else:
            unfixed.add(f)

    if not unfixed:
        return alloc

    # Flows on each link, and capacity still unclaimed on it.
    link_flows: Dict[LinkKey, Set[FlowKey]] = {}
    for f in unfixed:
        for lk in flow_links[f]:
            link_flows.setdefault(lk, set()).add(f)
    remaining = {lk: capacity[lk] for lk in link_flows}

    for lk in link_flows:
        if lk not in capacity:
            raise KeyError(f"no capacity given for link {lk!r}")

    while unfixed:
        # Fair share each link could offer its remaining unfixed flows.
        best_link: LinkKey | None = None
        best_share = float("inf")
        for lk, flows in link_flows.items():
            live = flows & unfixed
            if not live:
                continue
            share = remaining[lk] / len(live)
            if share < best_share:
                best_share, best_link = share, lk

        if best_link is None:
            # No link constrains the rest; they are unconstrained.
            for f in unfixed:
                alloc.rates[f] = float("inf")
            break

        rate = max(best_share, 0.0)
        newly_fixed = link_flows[best_link] & unfixed
        for f in newly_fixed:
            alloc.rates[f] = rate
            alloc.bottleneck[f] = best_link
        unfixed -= newly_fixed

        # Charge the fixed flows against every link they cross.
        for f in newly_fixed:
            for lk in flow_links[f]:
                if lk in remaining:
                    remaining[lk] = max(remaining[lk] - rate, 0.0)

    return alloc


def verify_conservation(alloc: Allocation,
                        flow_links: Mapping[FlowKey, Sequence[LinkKey]],
                        capacity: Mapping[LinkKey, float],
                        tol: float = 1e-6) -> None:
    """No link may carry more than its capacity. Raises if violated.

    Cheap enough to assert on every reallocation during development, and the
    invariant most likely to catch an allocator bug.
    """
    load: Dict[LinkKey, float] = {}
    for f, links in flow_links.items():
        r = alloc.rates.get(f, 0.0)
        if r == float("inf"):
            continue
        for lk in links:
            load[lk] = load.get(lk, 0.0) + r
    for lk, used in load.items():
        cap = capacity[lk]
        if used > cap * (1 + tol) + tol:
            raise AssertionError(
                f"link {lk} carries {used:g} but capacity is {cap:g}")
