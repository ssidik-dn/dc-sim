"""Binding: choosing which replica, among several candidates in the same
pool, receives a given transfer.

A cost predictor is told a source and a target *pool* -- never a *replica*
(task 06 spec S3, task 09 report S2.2). With one replica per pool that
ambiguity never surfaces. With several, something must decide, and this
module is that something. It is engine-side because binding is a placement
question -- which physical replica is closest, or least busy -- not a
Frontier-integration one; `src/integration/` calls into it but this module
knows nothing about Frontier.

Determinism is the operating requirement, not a nicety: two runs with the
same seed and configuration must produce identical bindings, or every
downstream measurement that depends on binding becomes unreproducible. Every
policy here breaks ties by ascending `replica_id`, with no reliance on
dict/set iteration order or on wall-clock/random state.

Task 14's own report covers the harder question this module does not answer
by itself: Frontier decides its *own* real destination replica later than
we are asked to price one (KV: at transfer arrival; M2N: on a bookkeeping
object built after the predictor returns), so a binding chosen here is a
model of that decision, not a guarantee of matching it. See
docs/tasks/14-binding-report.md S3 for whether the two agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from ..logical.deployment import Rank
from ..physical.topology import Fabric
from .placement import Placement


class BindingPolicy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    NEAREST = "nearest"
    EXPLICIT = "explicit"


class BindingError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    """One bindable replica, reduced to what binding actually needs: an
    identity to break ties and record load on, and the ranks to resolve a
    GPU from. Deliberately not `engine.logical.deployment.Replica` -- that
    type carries `pool`/`tp`/`pp`/`dp`/`ep`, none of which any policy here
    reads, and constructing one just to satisfy a type would manufacture
    fields with no meaning at the call site."""
    replica_id: int
    ranks: Tuple[Rank, ...]


@dataclass
class BindingState:
    """Mutable state a policy may need across calls within one run.

    Must be constructed fresh per run. Task 11 found Frontier's own replica
    ids increment globally across `Simulator` instances built in one
    process; state held here is this project's analogue, and reusing a
    `BindingState` across two scenarios would silently carry the first run's
    load history into the second.

    `LEAST_LOADED`'s load metric is a **cumulative per-replica assignment
    count**, not a true concurrent-in-flight count. `bind()` is a stateless,
    synchronous oracle called once per transfer with no callback for when
    that transfer completes, so there is no signal available here to expire
    an assignment -- only to record that one was made. This is stated as a
    real limitation, not hidden: a replica that was busy once early in the
    run and has long since finished is indistinguishable, to this policy,
    from one that is busy now. See the task 14 report for what this does and
    does not capture.
    """
    round_robin_cursor: int = 0
    assignment_count: Dict[int, int] = field(default_factory=dict)
    explicit_map: Dict[Tuple[Rank, ...], int] = field(default_factory=dict)

    def record(self, replica_id: int) -> None:
        self.assignment_count[replica_id] = self.assignment_count.get(replica_id, 0) + 1


def _by_replica_id(candidates: Sequence[Candidate]) -> list:
    return sorted(candidates, key=lambda r: r.replica_id)


def _round_robin(candidates: Sequence[Candidate], state: BindingState) -> Candidate:
    ordered = _by_replica_id(candidates)
    chosen = ordered[state.round_robin_cursor % len(ordered)]
    state.round_robin_cursor += 1
    return chosen


def _least_loaded(candidates: Sequence[Candidate], state: BindingState) -> Candidate:
    ordered = _by_replica_id(candidates)
    return min(ordered, key=lambda r: (state.assignment_count.get(r.replica_id, 0),
                                       r.replica_id))


def _nearest(source_ranks: Sequence[Rank], candidates: Sequence[Candidate],
            fabric: Fabric, placement: Placement) -> Candidate:
    """Fewest hops from the source's first rank to the candidate's first
    rank, same-domain candidates preferred outright (a same-domain hop is
    fewer physical links even when a raw hop count could tie -- see
    test_nearest_prefers_same_scale_up_domain for the case this exists for).
    Ties broken by ascending replica_id.
    """
    if not source_ranks:
        raise BindingError("nearest needs at least one source rank")
    source_gpu = placement.gpu(source_ranks[0])

    def key(candidate: Candidate) -> Tuple[bool, int, int]:
        candidate_gpu = placement.gpu(candidate.ranks[0])
        same_domain = fabric.same_domain(source_gpu, candidate_gpu)
        hops = len(fabric.path(source_gpu, candidate_gpu))
        return (not same_domain, hops, candidate.replica_id)

    ordered = _by_replica_id(candidates)
    return min(ordered, key=key)


def _explicit(source_ranks: Sequence[Rank], candidates: Sequence[Candidate],
             state: BindingState) -> Candidate:
    key = tuple(source_ranks)
    if key not in state.explicit_map:
        raise BindingError(
            f"no explicit binding registered for source ranks {key!r}; "
            f"call state.explicit_map[...] = replica_id before bind()")
    replica_id = state.explicit_map[key]
    for candidate in candidates:
        if candidate.replica_id == replica_id:
            return candidate
    raise BindingError(
        f"explicit binding points at replica_id={replica_id}, which is not "
        f"among the {len(candidates)} candidates offered")


def bind(policy: BindingPolicy, source_ranks: Sequence[Rank],
        candidate_replicas: Sequence[Candidate], state: BindingState,
        *, fabric: Optional[Fabric] = None,
        placement: Optional[Placement] = None) -> Candidate:
    """Choose one of `candidate_replicas` to receive a transfer from
    `source_ranks`, under `policy`.

    Raises `BindingError` on an empty candidate list, on `NEAREST` without
    `fabric`/`placement`, or on `EXPLICIT` without a matching registration in
    `state.explicit_map` -- refusing beats guessing, the same rule the
    predictors already apply to an unconfigured binding entirely (task 14
    spec S2.3).
    """
    if not candidate_replicas:
        raise BindingError("no candidate replicas to bind to")

    if policy is BindingPolicy.ROUND_ROBIN:
        chosen = _round_robin(candidate_replicas, state)
    elif policy is BindingPolicy.LEAST_LOADED:
        chosen = _least_loaded(candidate_replicas, state)
    elif policy is BindingPolicy.NEAREST:
        if fabric is None or placement is None:
            raise BindingError("NEAREST requires both fabric and placement")
        chosen = _nearest(source_ranks, candidate_replicas, fabric, placement)
    elif policy is BindingPolicy.EXPLICIT:
        chosen = _explicit(source_ranks, candidate_replicas, state)
    else:
        raise BindingError(f"unknown binding policy: {policy!r}")

    state.record(chosen.replica_id)
    return chosen
