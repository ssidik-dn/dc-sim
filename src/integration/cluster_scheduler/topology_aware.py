"""Task 15: a cluster scheduler that actually picks a replica by fabric
distance, instead of pricing a route the real scheduler never takes.

Task 14 built four binding *policies* for the KV/M2N predictors and found
that nothing connects them to Frontier's real replica assignment --
`RoundRobinClusterScheduler` decides independently, so a predictor
configured with `nearest` was pricing a transfer Frontier's own scheduler
agreed with only 3 times in 12 (see docs/tasks/14-binding-report.md S3).
That is not a pricing bug to fix with a better policy; it is proof that
distance-awareness belongs in the component that actually chooses a
replica, not in the one that only prices whatever choice was already made
elsewhere. This module is that component.

Read docs/tasks/15-topology-scheduler-report.md S1 before assuming this is
selectable the way tasks 07/09 found KV/M2N to be: `ClusterSchedulerType`
(frontier/types/cluster_scheduler_type.py) is a closed 5-member IntEnum with
every member already registered to a concrete class
(cluster_scheduler_registry.py), and `BaseRegistry.register()` no-ops on a
key collision (frontier/utils/base_registry.py) rather than overwriting --
so, unlike KV/M2N's unused `EMPIRICAL` slot, there is no free member here to
claim, and Frontier's real `--cluster_scheduler_config_type` flag can never
be made to construct this class. What follows is implemented to the ceiling
that finding leaves: a genuine, directly-constructible, directly-testable
`BaseClusterScheduler` subclass, exercised in `tests/test_topology_scheduler.py`
without going through the closed CLI path, and driven for real in
`tools/run_topology_scheduler_study.py` by patching an *already-constructed*
Frontier scheduler's own selection state in place (not by pretending the
flag path works).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

from frontier.config.config import BaseClusterSchedulerConfig
from frontier.scheduler.cluster_scheduler.cluster_scheduler_registry import (
    ClusterSchedulerRegistry)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler)
from frontier.types import ClusterSchedulerType, ClusterType

from engine.logical.deployment import PoolKind, Rank
from engine.placement.binding import Candidate, distance_key

from ..context import require_context

if TYPE_CHECKING:
    from frontier.entities import Request

# How much more loaded the nearest candidate must be than the least-loaded
# one before load overrides distance. An absolute difference, not a ratio:
# LOR's own signal (`num_pending_requests`) is a small integer in every
# scenario this project measures (single digits per replica), where a ratio
# like "2x" is noise-sensitive near zero (1 vs 0 pending requests is a 100%
# difference that means nothing). A margin of 2 tolerates ordinary arrival-
# timing noise -- one request landing on the near replica just before
# another arrives elsewhere -- while still catching a replica that has
# genuinely accumulated more outstanding work than every alternative. See
# test_load_overrides_distance_when_near_replica_is_saturated for the exact
# numbers this threshold was chosen against.
LOAD_MARGIN = 2


def select_replica(source_rank: Rank, candidates: Sequence[Candidate],
                   load_by_replica: Dict[int, int], fabric, placement) -> Candidate:
    """The combined distance+load rule, as a pure function of engine types --
    no Frontier object required, so this is what tests/test_topology_scheduler.py
    exercises directly rather than constructing a real `BaseClusterScheduler`
    (whose `__init__` pulls in Cluster/ClusterConfig/ReplicaScheduler
    construction well beyond what the selection rule itself needs).

    Prefers the candidate nearest `source_rank` (`engine.placement.binding`'s
    distance metric, task 14, reused via `distance_key` rather than
    re-derived -- see that function's own docstring for why it was
    refactored out of `_nearest` for exactly this reuse). Overridden only
    when the nearest candidate's outstanding load exceeds the least-loaded
    candidate's by more than `LOAD_MARGIN`, in which case the least-loaded
    candidate is chosen instead (ties in either comparison broken by
    `distance_key`, so "least loaded" still prefers the nearer of two
    equally-idle replicas, and ultimately by ascending replica_id --
    deterministic throughout, never wall-clock or dict-order dependent).

    This is Task 14's `nearest` logic, plus one rule Task 14 never needed
    (`bind()` has no live load signal to combine it with -- see
    `BindingState`'s own docstring): a scheduler that always went nearest
    regardless of load would just relocate Task 14's mispricing problem into
    a real overload, which is exactly what spec S3.1 warns against.
    """
    if not candidates:
        raise ValueError("no candidate replicas to schedule onto")

    def load_key(c: Candidate) -> Tuple[int, bool, int, int]:
        return (load_by_replica.get(c.replica_id, 0),) + distance_key(
            fabric, placement, source_rank, c)

    nearest = min(candidates, key=lambda c: distance_key(fabric, placement, source_rank, c))
    least_loaded = min(candidates, key=load_key)

    nearest_load = load_by_replica.get(nearest.replica_id, 0)
    least_load = load_by_replica.get(least_loaded.replica_id, 0)
    if nearest_load > least_load + LOAD_MARGIN:
        return least_loaded
    return nearest


@dataclass
class TopologyAwareClusterSchedulerConfig(BaseClusterSchedulerConfig):
    """A real `BaseClusterSchedulerConfig` subclass -- but see this module's
    docstring and the task 15 report S1: `get_type()`
    must return a `ClusterSchedulerType` member for Frontier's own CLI-
    flattening machinery to type-check at all (frontier/config/flat_dataclass.py's
    `reconstruct_original_dataclass`, which matches purely on
    `str(subclass.get_type())`), and every member is already claimed. It
    returns ROUND_ROBIN not because this class IS one, but because that is
    the only way to construct this config object through Frontier's normal
    polymorphic-field machinery at all -- and doing so does *not* make
    `--cluster_scheduler_config_type round_robin` select this class: the
    scheduler actually built comes from
    `ClusterSchedulerRegistry.get(cluster._config.cluster_scheduler_config.get_type(), ...)`
    (frontier/scheduler/global_scheduler/base_global_scheduler.py), keyed
    purely by that enum value, always resolving to Frontier's own
    `RoundRobinClusterScheduler` (registered first; `BaseRegistry.register()`
    no-ops on the collision). Construct `TopologyAwareClusterScheduler`
    directly instead -- this config class exists to document the ceiling,
    not to be a working selector.
    """

    @staticmethod
    def get_type() -> "ClusterSchedulerType":
        return ClusterSchedulerType.ROUND_ROBIN

    @classmethod
    def get_name(cls) -> str:
        return "topology_aware"


class TopologyAwareClusterScheduler(RoundRobinClusterScheduler):
    """Subclasses `RoundRobinClusterScheduler` rather than `BaseClusterScheduler`
    directly: every batch-mode/AFD-pipeline/barrier code path in the base
    class is inherited unchanged (reimplementing it would dwarf this task's
    actual scope, S3.1's "not a placement optimiser" warning included), and
    only the two places a replica gets *chosen* are overridden:

    - `_schedule_decode_lane_round_robin` -- the per-request, dynamic
      decision pd-disaggregation's unified DECODE cluster makes at KV
      arrival time (the same call this project's KV predictor cannot see,
      per task 14's finding).
    - `__init__`'s post-construction lane assignment for DECODE_FFN -- see
      `_assign_ffn_lanes_by_topology`'s own docstring for why this one is a
      one-time static map, not a per-request decision, and how that changes
      what "load" can mean for it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._cluster_type == ClusterType.DECODE_FFN and hasattr(
                self, "_ffn_lane_to_target_replica"):
            self._assign_ffn_lanes_by_topology()

    def _live_load_by_replica(self) -> Dict[int, int]:
        """Sum of `num_pending_requests` across this replica's dp lanes --
        the exact signal `LORClusterScheduler._schedule_lor` already uses
        (lor_cluster_scheduler.py), not a new one invented for this task.
        Real, current state (unlike task 14's `BindingState.assignment_count`,
        which could only accumulate a count with no way to expire it -- see
        that module's docstring); this scheduler has direct access to the
        replica schedulers themselves; task 14's `bind()` never did.
        """
        replica_ids = list(self._cluster.replicas.keys())
        loads: Dict[int, int] = {}
        for replica_id in replica_ids:
            total = 0
            for dp_id in range(self._replica_dp_size):
                total += self._dp_replica_schedulers[(replica_id, dp_id)].num_pending_requests
            loads[replica_id] = total
        return loads

    def _decode_candidates(self) -> List[Candidate]:
        """Built directly from `self._cluster.replicas.keys()` -- Frontier's
        own, real replica ids for *this* cluster -- not this project's own
        `CommGroupRegistry` (whose ids are per-pool, 0-indexed; Frontier's
        are a single counter shared across every cluster type in
        construction order, task 14 report S3). Using Frontier's own ids
        directly means `select_replica`'s answer can be fed straight back
        into `request_mapping`/`_dp_replica_schedulers` with no conversion,
        and no risk of the two id spaces being silently conflated. Only the
        *rank* (for distance lookups against this project's own
        `Placement`) needs the offset back to a per-pool index; assumes
        tp=1, matching every prior integration in this project."""
        offset = min(self._cluster.replicas.keys())
        return [Candidate(rid, (Rank(PoolKind.DECODE.value, rid - offset, 0),))
               for rid in self._cluster.replicas.keys()]

    def _schedule_decode_lane_round_robin(self) -> List[Tuple[int, int, "Request"]]:
        """Overrides the dynamic per-request assignment
        round_robin_cluster_scheduler.py's own version makes for the unified
        DECODE cluster (pd-disaggregation). Same dp-lane-within-replica
        mechanics (least-pending dp lane inside whichever replica is
        chosen); only *which replica* changes, from a lane-index modulo to
        `select_replica`.
        """
        ctx = require_context()
        replica_ids = list(self._cluster.replicas.keys())
        if not replica_ids:
            return []

        src_ranks = ctx.groups.resolve_pool(ClusterType.PREFILL)
        candidates = self._decode_candidates()
        request_mapping: List[Tuple[int, int, "Request"]] = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            loads = self._live_load_by_replica()
            chosen = select_replica(src_ranks[0], candidates, loads, ctx.fabric, ctx.placement)
            dp_id = min(range(self._replica_dp_size),
                       key=lambda d: self._dp_replica_schedulers[
                           (chosen.replica_id, d)].num_pending_requests)
            request_mapping.append((chosen.replica_id, dp_id, request))

        return request_mapping

    def _assign_ffn_lanes_by_topology(self) -> None:
        """Recomputes `_ffn_lane_to_target_replica` (base_cluster_scheduler.py's
        `__init__`, `lane_ordinal % len(self._ffn_replica_ids)`) by fabric
        distance from each DECODE_ATTN lane's replica instead.

        Unlike the DECODE cluster's per-request assignment above, M2N's FFN
        target is decided exactly once, for every lane, before the run
        starts (`__init__`, not `schedule()`) -- there is no live
        `num_pending_requests` to read yet, so "load" here can only mean
        "how many lanes this replica has already been given in this same
        pass," a running balance rather than a queue depth.

        Two phases, in `self._ffn_expected_lanes` order:

        1. **Coverage.** Frontier itself asserts every DECODE_FFN replica
           gets at least one lane (`__init__`'s
           "must give every target replica at least one decode-attn lane"
           check) -- a real invariant, not a suggestion. Plain nearest-only
           assignment can violate it outright on an asymmetric fabric (one
           near replica, several symmetric far ones): every lane's nearest
           candidate is the *same* near replica, so a single-phase nearest+
           load-margin rule can spend its whole run oscillating between just
           the nearest replica and whichever one is currently
           least-loaded, never reaching a third or fourth replica at all
           (confirmed by hand-tracing a 4-lane/4-replica example before
           writing this, not assumed). Phase 1 sidesteps this by assigning
           the first `len(self._ffn_replica_ids)` lanes to whichever
           still-uncovered replica is nearest, guaranteeing every replica
           has exactly one lane before phase 2 starts -- after which the
           invariant can never be violated by anything phase 2 does.
        2. **Balance.** Remaining lanes (if `len(self._ffn_expected_lanes) >
           len(self._ffn_replica_ids)`) go through the same `select_replica`
           rule the DECODE cluster uses, now safe because coverage is
           already guaranteed.
        """
        ctx = require_context()
        # Frontier's own ids for this (DECODE_FFN) cluster, same reasoning as
        # _decode_candidates: `_ffn_lane_to_target_replica`'s values are read
        # back by base_cluster_scheduler.py as real replica ids, so
        # `Candidate.replica_id` must be one, not this project's per-pool id.
        ffn_offset = min(self._cluster.replicas.keys())
        candidates = [Candidate(rid, (Rank(PoolKind.DECODE_FFN.value, rid - ffn_offset, 0),))
                     for rid in self._cluster.replicas.keys()]
        lane_count: Dict[int, int] = {c.replica_id: 0 for c in candidates}
        new_map: Dict[Tuple[int, int], int] = {}
        new_by_target: Dict[int, List[Tuple[int, int]]] = {c.replica_id: [] for c in candidates}

        # Frontier's own replica ids are a single counter shared across every
        # cluster type in construction order (frontier/entities/base_entity.py's
        # generate_id -- the same fact task 14's report S3 had to correct
        # for): `lane`'s first element is that global id, offset by
        # `decode_attn_replica_id_start_for_ffn` (== prefill_cluster_num_replicas,
        # frontier/config/config.py), not this project's own per-pool
        # replica_id (CommGroupRegistry.register_pool's `len(existing)`).
        # Subtracting the offset recovers the per-pool id our own Rank/
        # Placement objects were built with.
        attn_id_start = int(self._config.decode_attn_replica_id_start_for_ffn)

        def source_rank_for(lane: Tuple[int, int]) -> Rank:
            attn_replica_id, dp_id = lane
            return Rank(PoolKind.DECODE_ATTN.value, attn_replica_id - attn_id_start, dp_id)

        lanes = list(self._ffn_expected_lanes)
        num_replicas = len(candidates)

        uncovered = set(c.replica_id for c in candidates)
        for lane in lanes[:num_replicas]:
            source_rank = source_rank_for(lane)
            remaining = [c for c in candidates if c.replica_id in uncovered]
            chosen = min(remaining, key=lambda c: distance_key(
                ctx.fabric, ctx.placement, source_rank, c))
            new_map[lane] = chosen.replica_id
            new_by_target[chosen.replica_id].append(lane)
            lane_count[chosen.replica_id] += 1
            uncovered.discard(chosen.replica_id)

        for lane in lanes[num_replicas:]:
            source_rank = source_rank_for(lane)
            chosen = select_replica(source_rank, candidates, lane_count, ctx.fabric, ctx.placement)
            new_map[lane] = chosen.replica_id
            new_by_target[chosen.replica_id].append(lane)
            lane_count[chosen.replica_id] += 1

        self._ffn_lane_to_target_replica = new_map
        self._ffn_expected_lanes_by_target = new_by_target
        self._ffn_group_micro_batches = max(len(v) for v in new_by_target.values())


# Mechanical registration only -- documents that the registry API itself
# does not distinguish a real ClusterSchedulerType member from any other
# hashable key (BaseRegistry.register()'s `key` parameter is a type hint,
# not a runtime check), the same finding task 06 made for CCBackendFactory.
# This round-trips through get_class()/get() (see
# tests/test_topology_scheduler.py); it does not make this class reachable
# by CLI flag, which goes through the closed enum instead (see this module's
# docstring and TopologyAwareClusterSchedulerConfig's).
ClusterSchedulerRegistry.register("dc_sim_topology_aware", TopologyAwareClusterScheduler)
