"""Task 50: pure observation counters for whether the contention mechanism
has ever actually fired, as opposed to sitting correct-but-unexercised.

Read by nothing in `model.py`/`transfers.py`/`allocator.py` -- every value
here is written once, by `FlowNetwork` itself, at the moment a fact becomes
true, and never read back into any computation. Incrementing a counter
cannot change a completion time, a rate, or a bottleneck attribution, which
is exactly what this task's own acceptance criteria requires ("keep it to
counters that do not alter any computed value"). Confirmed directly, not
merely asserted: Task 33's own sixteen-row table and Task 36's own
two-fabric result both still reproduce bit-identical with this module
imported and wired into `FlowNetwork` (docs/tasks/50-contention-reach-report.md).

A module-level singleton, not a constructor argument threaded through every
call site, for the same reason `AGENTS.md`'s own zoning treats this file's
neighbours as sensitive: adding a required parameter to `FlowNetwork.__init__`
or `_reallocate()` would touch every call site across `src/integration/` and
every test in this project that constructs one -- a far larger, riskier
change than this task's own "keep it to counters" instruction asks for.
`reset()` exists so a caller (a probe script, a test) can zero it before one
measured run and read it after, without any prior activity (imports,
warm-up calls, other tests in the same process) polluting the count.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContentionCounters:
    networks_constructed: int = 0
    max_flows_in_flight: int = 0
    rate_reductions: int = 0
    completion_revisions: int = 0

    def reset(self) -> None:
        self.networks_constructed = 0
        self.max_flows_in_flight = 0
        self.rate_reductions = 0
        self.completion_revisions = 0


# The one instance every FlowNetwork reports into. Not per-instance, because
# the question this task asks ("has this ever fired, across a whole run") is
# about the aggregate across every network any predictor call ever
# constructs, not about any one of them individually.
COUNTERS = ContentionCounters()
