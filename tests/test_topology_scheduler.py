"""Task 15: a cluster scheduler that picks a replica by fabric distance
(and load), instead of a predictor pricing a route Frontier's own scheduler
never takes (task 14's finding).

`select_replica` is a pure function of engine types (Candidate/Fabric/
Placement/a load dict) -- tests 2-5 exercise it directly, the same way
tests/test_binding.py exercises `bind()` directly, rather than constructing
a real `BaseClusterScheduler` (whose `__init__` requires a full
Cluster/ClusterConfig/ReplicaScheduler object graph well beyond what the
selection rule itself needs). Test 1 is the exception -- it needs a real
Frontier run, because it is proving a negative about Frontier's own closed
registry (see topology_aware.py's module docstring and the task 15 report
S1), not testing engine logic.
"""
from __future__ import annotations

import pytest

from frontier.scheduler.cluster_scheduler.cluster_scheduler_registry import (
    ClusterSchedulerRegistry)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler)

from engine.logical.deployment import Deployment, PoolKind, Rank, Replica
from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId
from engine.placement.binding import Candidate
from engine.placement.placement import explicit

from integration.cluster_scheduler.topology_aware import (
    LOAD_MARGIN, TopologyAwareClusterScheduler, TopologyAwareClusterSchedulerConfig,
    select_replica)

SOURCE = Rank("SRC", 0, 0)


def _split_fabric_and_placement():
    """Same shape as test_binding.py's split-fabric test: three domains, the
    source shares a domain with candidate A only; B and C are each on their
    own, symmetric-to-each-other, cross-domain machine."""
    fab = build_node_scale(num_machines=3, gpus_per_machine=8)
    d = Deployment("t")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    src_rank = d.replicas[0].ranks[0]
    ranks = {0: Rank("DECODE", 0, 0), 1: Rank("DECODE", 1, 0), 2: Rank("DECODE", 2, 0)}
    placement = explicit(d, fab, {src_rank: GpuId(0, 0)})
    placement.assign(ranks[0], GpuId(0, 1))  # A: same domain as source
    placement.assign(ranks[1], GpuId(1, 0))  # B: domain 1
    placement.assign(ranks[2], GpuId(2, 0))  # C: domain 2
    candidates = [Candidate(i, (ranks[i],)) for i in range(3)]
    return fab, placement, src_rank, candidates


# ------------------------------------------------------------- registration


def test_scheduler_is_selectable_by_flag():
    """Registration is not selection (task 06's own trap; task 14 S4).

    `ClusterSchedulerType` is a closed 5-member IntEnum with every member
    already registered to a concrete class, and Frontier's own CLI
    flattening (`flat_dataclass.py::reconstruct_original_dataclass`)
    validates a `--cluster_scheduler_config_type` value purely against
    `str(subclass.get_type())` for every discovered `BaseClusterSchedulerConfig`
    subclass -- so there is no name this config class can expose that isn't
    already one of the five. This is proven with an actual run (matching a
    real, reproduced `AssertionError` -- not assumed from reading the
    enum), and it is exactly the failure mode --closed registries raise on
    an unknown name-- that task 06 already found for `CCBackendType`.

    The registry API itself is a separate matter from the CLI flag: it does
    not runtime-check its key at all (`BaseRegistry.register()`'s `key`
    parameter is a type hint only), so `TopologyAwareClusterScheduler` was
    registered under a fabricated string key at import time
    (topology_aware.py's module-level `ClusterSchedulerRegistry.register(...)`
    call) and that round-trips through `get_class()` -- mechanically true,
    same as task 06's `test_registration_is_idempotent`, but not a CLI path.
    """
    assert ClusterSchedulerRegistry.get_class("dc_sim_topology_aware") is (
        TopologyAwareClusterScheduler)

    import sys
    sys.argv = ["frontier.main", "--cluster_scheduler_config_type", "topology_aware"]
    from frontier.config import SimulationConfig
    with pytest.raises(AssertionError, match="Invalid type topology_aware"):
        SimulationConfig.create_from_cli_args()


# -------------------------------------------------------------- select_replica


def test_prefers_same_domain_replica():
    fab, placement, src_rank, candidates = _split_fabric_and_placement()
    loads = {0: 0, 1: 0, 2: 0}
    chosen = select_replica(src_rank, candidates, loads, fab, placement)
    assert chosen.replica_id == 0  # A: same domain, idle


def test_load_overrides_distance_when_near_replica_is_saturated():
    """LOAD_MARGIN is 2 (an absolute difference, not a ratio -- see
    topology_aware.py's own docstring for why). Below the margin, distance
    still wins; only once the near replica's load exceeds the least-loaded
    candidate's by *more than* the margin does load win."""
    fab, placement, src_rank, candidates = _split_fabric_and_placement()

    at_margin = {0: LOAD_MARGIN, 1: 0, 2: 0}  # 2 > 0+2 is False -- still near
    assert select_replica(src_rank, candidates, at_margin, fab, placement).replica_id == 0

    over_margin = {0: LOAD_MARGIN + 1, 1: 0, 2: 0}  # 3 > 0+2 is True -- overridden
    chosen = select_replica(src_rank, candidates, over_margin, fab, placement)
    assert chosen.replica_id != 0
    # B and C are symmetric (equal distance from the source); the least-loaded
    # tie is broken by distance_key's own replica_id tie-break -- B (id 1).
    assert chosen.replica_id == 1


def test_selection_is_deterministic():
    fab, placement, src_rank, candidates = _split_fabric_and_placement()
    loads = {0: 3, 1: 1, 2: 1}

    def run():
        return [select_replica(src_rank, candidates, loads, fab, placement).replica_id
               for _ in range(5)]

    assert run() == run()


def test_falls_back_cleanly_with_one_replica():
    """The guard that matters most (task 14's own S4 lesson, restated for
    this scheduler): every prior end-to-end measurement in this project used
    exactly one replica per pool. With a single candidate, `select_replica`
    must return it regardless of load or distance -- there is no other
    choice to make, and none of this machinery should be able to change
    that."""
    fab, placement, src_rank, _ = _split_fabric_and_placement()
    only = Candidate(0, (Rank("DECODE", 9, 0),))
    placement.assign(only.ranks[0], GpuId(2, 1))  # deliberately far, unused slot
    chosen = select_replica(src_rank, [only], {0: 999}, fab, placement)
    assert chosen.replica_id == 0


def test_select_replica_rejects_empty_candidates():
    fab, placement, src_rank, _ = _split_fabric_and_placement()
    with pytest.raises(ValueError):
        select_replica(src_rank, [], {}, fab, placement)


def test_is_a_real_round_robin_cluster_scheduler_subclass():
    """S3.1: subclasses RoundRobinClusterScheduler rather than
    BaseClusterScheduler directly, so every batch-mode/AFD-pipeline code
    path not touched by this task is inherited unchanged."""
    assert issubclass(TopologyAwareClusterScheduler, RoundRobinClusterScheduler)
    assert TopologyAwareClusterSchedulerConfig.get_name() == "topology_aware"
