"""Task 16: resolving a *source* pool with more than one replica.

Task 14 scoped binding to "which replica receives a transfer" and left the
source side raising unconditionally. Task 15's own study found this
matters for real: activation exchange is a round trip (DECODE_ATTN sends to
DECODE_FFN, DECODE_FFN sends back), so a multi-replica DECODE_FFN pool is
the *source* on the return leg, and the whole run raised the moment it had
more than one replica.

Tests 1-2 are the guards: unchanged single-replica behaviour, and the
`CommGroupError` from task 15 actually gone. Test 3 exercises the
recovered-identity path (task 16 report S1: `batch.decode_attn_original_dp_id`
lets an unambiguous pool be priced against its *actual* lane instead of a
representative rank -- not a policy, an exact recovery). Tests 4-5 exercise
the `bind()`-based fallback for the genuinely-ambiguous case (*which*
replica sent it, not just which lane) -- the same machinery task 14 already
built for destinations, now reachable for sources too.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.types import ClusterType

from engine.logical.deployment import Deployment, PoolKind, Rank, Replica
from engine.network.transfers import Transfer, isolated_durations
from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId
from engine.placement.binding import BindingPolicy
from engine.placement.placement import explicit

from integration.binding_support import price_transfer
from integration.cc_backend.comm_groups import CommGroupError, CommGroupRegistry, populate_from_deployment
from integration.context import BindingConfig, EngineContext

SIZE = 1 << 20


def _batch(dp_id):
    return SimpleNamespace(decode_attn_original_dp_id=dp_id)


# --------------------------------------------------------- single-replica guard


def _single_replica_scenario():
    fab = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = Deployment("t")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
                                      PoolKind.DECODE_FFN: ClusterType.DECODE_FFN})
    placement = explicit(d, fab, {
        d.replicas[0].ranks[0]: GpuId(0, 0),
        d.replicas[1].ranks[0]: GpuId(1, 0),
    })
    return fab, d, reg, placement


def test_single_replica_source_unchanged():
    """The guard that matters most: every measurement before task 16
    resolved a single-replica source to its representative rank. A `batch`
    with no useful lane information (or none at all) must produce exactly
    the same price as before, on both legs."""
    fab, d, reg, placement = _single_replica_scenario()
    ctx = EngineContext(fab, placement, d, reg)

    forward_no_batch, chosen_a = price_transfer(
        ctx, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, SIZE, key="f")
    forward_with_batch, chosen_b = price_transfer(
        ctx, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, SIZE, key="f2", batch=_batch(0))
    assert forward_no_batch == forward_with_batch
    assert chosen_a is None and chosen_b is None

    return_no_batch, chosen_c = price_transfer(
        ctx, ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN, SIZE, key="r")
    assert chosen_c is None
    assert return_no_batch == forward_no_batch  # tp=1 both ways, symmetric fabric cost


# --------------------------------------------------- multi-replica source guard


def _multi_ffn_source_scenario():
    """DECODE_ATTN: one replica, dp=2 (two lanes, one per machine -- not
    equidistant from either FFN replica). DECODE_FFN: two replicas, also
    one per machine. Every rank placed on its own GPU."""
    fab = build_node_scale(num_machines=3, gpus_per_machine=8)
    d = Deployment("t")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=1, dp=2))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_FFN, 1, tp=1))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
                                      PoolKind.DECODE_FFN: ClusterType.DECODE_FFN})
    attn_ranks = d.replicas[0].ranks  # dp lane 0, dp lane 1
    ffn0, ffn1 = d.replicas[1].ranks[0], d.replicas[2].ranks[0]
    placement = explicit(d, fab, {
        attn_ranks[0]: GpuId(0, 0),  # attn lane 0: machine 0
        attn_ranks[1]: GpuId(1, 0),  # attn lane 1: machine 1
        ffn0: GpuId(0, 1),           # ffn replica 0: machine 0 (near lane 0)
        ffn1: GpuId(2, 0),           # ffn replica 1: machine 2 (far from both)
    })
    return fab, d, reg, placement, attn_ranks, ffn0, ffn1


def test_multi_replica_source_does_not_raise():
    """The task 15 blocker, gone: DECODE_FFN as a *source* (the M2N return
    leg) with more than one replica no longer raises CommGroupError, given
    a binding policy."""
    fab, d, reg, placement, attn_ranks, ffn0, ffn1 = _multi_ffn_source_scenario()

    ctx_unconfigured = EngineContext(fab, placement, d, reg)
    with pytest.raises(CommGroupError):
        price_transfer(ctx_unconfigured, ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN,
                       SIZE, key="r", batch=_batch(0))

    ctx = EngineContext(fab, placement, d, reg,
                        binding=BindingConfig(BindingPolicy.NEAREST, timing="early"))
    price_ms, chosen = price_transfer(ctx, ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN,
                                      SIZE, key="r", batch=_batch(0))
    assert price_ms > 0
    assert chosen in (0, 1)


# --------------------------------------------------------- recovered identity


def test_source_identity_is_recovered():
    """§2's answer for the forward leg: the sending DECODE_ATTN dp lane is
    exactly recoverable from `batch.decode_attn_original_dp_id` (confirmed
    against a real run in the task 16 report S1) -- not a guess. Pricing
    against lane 1 (machine 1) must differ from pricing against lane 0
    (machine 0), against the *same* fixed FFN destination, and match a
    manually-computed isolated duration from that lane's real rank -- not
    from `ranks[0]`, which would give the wrong (lane 0) answer for lane 1."""
    fab, d, reg, placement, attn_ranks, ffn0, ffn1 = _multi_ffn_source_scenario()
    ctx = EngineContext(fab, placement, d, reg)  # FFN destination has 2
    # replicas, so use the single-replica ATTN pool as *destination* instead,
    # to isolate the source-lane recovery from the multi-replica-destination
    # path already covered by test_multi_replica_source_does_not_raise.

    def expected_ms(dp_id):
        src_gpu = placement.gpu(attn_ranks[dp_id])
        t = Transfer(key=f"expected-{dp_id}", src=src_gpu, dst=placement.gpu(ffn0),
                     size_bytes=SIZE)
        return isolated_durations(fab, [t])[t.key] / 1_000_000.0

    # Route to the single unambiguous FFN replica candidate isn't directly
    # expressible via price_transfer with 2 FFN replicas registered, so
    # register a second registry with only ffn0 to isolate the source side.
    reg_single_dst = CommGroupRegistry()
    reg_single_dst.register_pool(ClusterType.DECODE_ATTN, list(attn_ranks))
    reg_single_dst.register_pool(ClusterType.DECODE_FFN, [ffn0])
    ctx_single_dst = EngineContext(fab, placement, d, reg_single_dst)

    price_lane0, chosen0 = price_transfer(ctx_single_dst, ClusterType.DECODE_ATTN,
                                          ClusterType.DECODE_FFN, SIZE, key="l0", batch=_batch(0))
    price_lane1, chosen1 = price_transfer(ctx_single_dst, ClusterType.DECODE_ATTN,
                                          ClusterType.DECODE_FFN, SIZE, key="l1", batch=_batch(1))

    assert chosen0 is None and chosen1 is None  # both pools resolved directly; no bind() involved
    assert price_lane0 == pytest.approx(expected_ms(0))
    assert price_lane1 == pytest.approx(expected_ms(1))
    assert price_lane0 != price_lane1  # lane 0 (machine 0) is nearer ffn0 than lane 1 (machine 1)


# ---------------------------------------------------------- near beats far


def test_near_source_prices_lower_than_far_source():
    fab, d, reg, placement, attn_ranks, ffn0, ffn1 = _multi_ffn_source_scenario()
    ctx = EngineContext(fab, placement, d, reg,
                        binding=BindingConfig(BindingPolicy.NEAREST, timing="early"))

    # Fixed destination: attn lane 0 (machine 0). ffn0 is on machine 0 (near);
    # ffn1 is on machine 2 (far). NEAREST must pick ffn0 and price accordingly.
    price_ms, chosen = price_transfer(ctx, ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN,
                                      SIZE, key="near", batch=_batch(0))
    assert chosen == 0

    far_gpu = placement.gpu(ffn1)
    near_dst_gpu = placement.gpu(attn_ranks[0])
    t_far = Transfer(key="far-reference", src=far_gpu, dst=near_dst_gpu, size_bytes=SIZE)
    far_price_ms = isolated_durations(fab, [t_far])[t_far.key] / 1_000_000.0

    assert price_ms < far_price_ms


# ------------------------------------------------------- both legs of a trip


def test_round_trip_prices_both_legs():
    """Task 15's exact blocker: an attention-to-FFN call, and its return,
    both succeeding. Neither leg raises; both price a positive duration."""
    fab, d, reg, placement, attn_ranks, ffn0, ffn1 = _multi_ffn_source_scenario()
    ctx = EngineContext(fab, placement, d, reg,
                        binding=BindingConfig(BindingPolicy.NEAREST, timing="early"))

    forward_ms, forward_chosen = price_transfer(
        ctx, ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, SIZE, key="fwd", batch=_batch(0))
    return_ms, return_chosen = price_transfer(
        ctx, ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN, SIZE, key="ret", batch=_batch(0))

    assert forward_ms > 0
    assert return_ms > 0
    assert forward_chosen in (0, 1)
    assert return_chosen in (0, 1)
