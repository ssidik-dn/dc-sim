"""Task 14: binding -- choosing which replica, among several candidates in
one pool, receives a transfer.

Tests 1-5 exercise `engine.placement.binding` directly, with expected
values computed by hand (or verified by a short script and shown in a
comment, per this project's convention for anything involving fabric
arithmetic). Tests 6-7 exercise the integration-level behaviour: a
predictor with no binding policy configured still raises on multiple
replicas, and a predictor with exactly one replica per pool is unaffected
by any of this.
"""
from __future__ import annotations

import pytest

from frontier.types import ClusterType

from engine.logical.deployment import Deployment, PoolKind, Rank, Replica
from engine.network.transfers import Transfer, isolated_durations
from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId
from engine.placement.binding import BindingPolicy, BindingState, Candidate, bind
from engine.placement.placement import explicit

from integration.binding_support import price_transfer
from integration.cc_backend.comm_groups import (CommGroupError, CommGroupRegistry,
                                                 populate_from_deployment)
from integration.context import BindingConfig, EngineContext

SOURCE = Rank("SRC", 0, 0)


# ---------------------------------------------------------------- pure binding


def test_round_robin_is_deterministic():
    candidates = [Candidate(0, ()), Candidate(1, ()), Candidate(2, ())]

    def run():
        state = BindingState()
        return [bind(BindingPolicy.ROUND_ROBIN, [SOURCE], candidates, state).replica_id
                for _ in range(7)]

    first, second = run(), run()
    assert first == second
    assert first == [0, 1, 2, 0, 1, 2, 0]


def test_least_loaded_prefers_idle_replica():
    c0, c1 = Candidate(0, ()), Candidate(1, ())
    state = BindingState()
    state.assignment_count[0] = 3  # replica 0 already busy
    chosen = bind(BindingPolicy.LEAST_LOADED, [SOURCE], [c0, c1], state)
    assert chosen.replica_id == 1


def test_nearest_prefers_same_scale_up_domain():
    fab = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = Deployment("t")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    src_rank = d.replicas[0].ranks[0]
    near_rank, far_rank = Rank("DECODE", 0, 0), Rank("DECODE", 1, 0)
    placement = explicit(d, fab, {src_rank: GpuId(0, 0)})
    placement.assign(near_rank, GpuId(0, 1))   # same domain as source
    placement.assign(far_rank, GpuId(1, 0))    # different domain

    near = Candidate(0, (near_rank,))
    far = Candidate(1, (far_rank,))
    chosen = bind(BindingPolicy.NEAREST, [src_rank], [far, near], BindingState(),
                 fabric=fab, placement=placement)
    assert chosen.replica_id == 0


def test_nearest_beats_round_robin_on_a_split_fabric():
    """The binding test. Three domains; the source shares a domain with
    candidate A only. Same 1 MiB transfer from the source to whichever
    candidate each policy picks, three times.

    Hand-verified fabric costs (build_node_scale defaults: 400 GB/s scale-up
    @ 936.25 ns, 50 GB/s scale-out via egress+scale_out+scale_out+egress @
    14000 ns total):

        A (same domain):   936.25 + 1048576/400        =   3559 ns
        B, C (cross domain): 1048576/50 + 14000         =  34972 ns each

    NEAREST always picks A (distance is the only input, load is not), so
    three transfers cost 3 x 3559 = 10677 ns.
    ROUND_ROBIN cycles A, B, C: 3559 + 34972 + 34972 = 73503 ns.
    10677 < 73503 -- nearest must beat round-robin here, by close to 7x.
    """
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
    size = 1 << 20

    def total_cost(policy: BindingPolicy) -> int:
        state = BindingState()
        total = 0
        for i in range(3):
            chosen = bind(policy, [src_rank], candidates, state,
                         fabric=fab, placement=placement)
            dst_gpu = placement.gpu(chosen.ranks[0])
            total += isolated_durations(
                fab, [Transfer(f"t{i}", GpuId(0, 0), dst_gpu, size)])[f"t{i}"]
        return total

    nearest_total = total_cost(BindingPolicy.NEAREST)
    round_robin_total = total_cost(BindingPolicy.ROUND_ROBIN)
    assert nearest_total == 10677
    assert round_robin_total == 73503
    assert nearest_total < round_robin_total


def test_ties_break_deterministically():
    """Two candidates equidistant from the source (both cross a domain
    boundary the same way) and equally loaded: the tie must resolve the same
    way every time, and specifically to the lower replica_id -- see
    _by_replica_id / the (same_domain, hops, replica_id) sort key in
    binding.py."""
    fab = build_node_scale(num_machines=3, gpus_per_machine=8)
    d = Deployment("t")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    src_rank = d.replicas[0].ranks[0]
    rank_a, rank_b = Rank("DECODE", 1, 0), Rank("DECODE", 2, 0)
    placement = explicit(d, fab, {src_rank: GpuId(0, 0)})
    placement.assign(rank_a, GpuId(1, 0))  # domain 1, 1 hop away either way
    placement.assign(rank_b, GpuId(2, 0))  # domain 2, 1 hop away either way

    candidate_a, candidate_b = Candidate(5, (rank_a,)), Candidate(3, (rank_b,))
    for _ in range(3):
        chosen = bind(BindingPolicy.NEAREST, [src_rank], [candidate_a, candidate_b],
                     BindingState(), fabric=fab, placement=placement)
        assert chosen.replica_id == 3, "ties must resolve to the lower replica_id"


# ---------------------------------------------------------- integration-level


def _two_decode_replicas():
    fab = build_node_scale(num_machines=1, gpus_per_machine=4)
    d = Deployment("t")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE, 0, tp=1))
    d.add(Replica(PoolKind.DECODE, 1, tp=1))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.PREFILL: ClusterType.PREFILL,
                                      PoolKind.DECODE: ClusterType.DECODE})
    placement = explicit(d, fab, {
        d.replicas[0].ranks[0]: GpuId(0, 0),
        d.replicas[1].ranks[0]: GpuId(0, 1),
        d.replicas[2].ranks[0]: GpuId(0, 2),
    })
    return fab, d, reg, placement


def test_no_policy_still_raises():
    """The guard from task 09/11 must survive task 14: with no BindingConfig
    set (EngineContext.binding defaults to None), an ambiguous destination
    pool still raises rather than picking one -- refusing beats guessing,
    unchanged."""
    fab, d, reg, placement = _two_decode_replicas()
    ctx = EngineContext(fab, placement, d, reg)  # binding=None
    with pytest.raises(CommGroupError):
        price_transfer(ctx, ClusterType.PREFILL, ClusterType.DECODE, 1 << 20, key="t")


def test_single_replica_unchanged():
    """The guard that matters most (task 14 spec S4): every measurement in
    this project so far used one replica per pool. Configuring a binding
    policy must not change a single-replica result at all -- resolve_pool
    succeeds directly and price_transfer never reaches bind()."""
    fab = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = Deployment("t")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE, 0, tp=1))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.PREFILL: ClusterType.PREFILL,
                                      PoolKind.DECODE: ClusterType.DECODE})
    placement = explicit(d, fab, {
        d.replicas[0].ranks[0]: GpuId(0, 0),
        d.replicas[1].ranks[0]: GpuId(1, 0),
    })
    size = 1 << 20

    ctx_unconfigured = EngineContext(fab, placement, d, reg)
    price_without, chosen_without = price_transfer(
        ctx_unconfigured, ClusterType.PREFILL, ClusterType.DECODE, size, key="t")

    ctx_with_policy = EngineContext(
        fab, placement, d, reg,
        binding=BindingConfig(BindingPolicy.ROUND_ROBIN, timing="early"))
    price_with, chosen_with = price_transfer(
        ctx_with_policy, ClusterType.PREFILL, ClusterType.DECODE, size, key="t")

    assert price_without == price_with
    assert chosen_without is None and chosen_with is None
