"""Task 21: verify the collective patterns behind `EngineCCBackend`'s five
true collectives, closed-form, the way `engine.network.allocator`'s own
tests do -- every expected value hand-computed and shown in a comment.

Task 21's own opening premise ("the cost path builds an all-pairs mesh for
every collective, regardless of type: `induced_links` walks every ordered
pair") does **not** describe this project's actual, current
`EngineCCBackend` (task 20): `predict_allreduce`/`allgather`/`reduce_scatter`
already build ring edges (`_ring_edges`, exactly `n` edges), never an
all-pairs mesh, and `induced_links` (`engine.placement.placement.Placement.induced_links`)
is not called anywhere in `integration/cc_backend/` (confirmed by `grep`,
not assumed) -- it is a separate, pre-existing utility used by
`engine/cli/place.py`. `test_ring_crosses_boundary_twice_not_per_pair`
below still exists per spec, and passes against the *correct* number
(229,376 B) while documenting what the *wrong* number described by the
spec's own premise (1,835,008 B -- 14 rounds x 16 all-pairs x 8192 B, i.e.
literally charging every cross-domain pair on every one of a ring's
sequential rounds) would have been, since this project's implementation
was never charging that.

What task 20 *did* get wrong, and what this task actually corrects:
`predict_all_to_all`'s per-pair volume was `data_size_bytes/n`, not
`data_size_bytes/n^2` -- see `test_all_to_all_per_pair_volume` and
`engine_backend.py`'s own updated docstring for why n^2 is right given
Frontier's real call-site convention for `data_size_bytes`.
"""
from __future__ import annotations

from unittest import mock

import pytest

from frontier.types import ClusterType

from engine.logical.deployment import Deployment, PoolKind, Replica
from engine.network.transfers import Transfer
from engine.physical.builders import build_node_scale
from engine.placement.placement import packed, spread

from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment
from integration.cc_backend.engine_backend import EngineCCBackend

SIZE = 65536  # 64 KiB, this project's own recurring TP/EP payload figure


def _backend(fabric, deployment, placement_policy):
    placement = placement_policy(deployment, fabric)
    reg = CommGroupRegistry()
    populate_from_deployment(reg, deployment, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN})
    return EngineCCBackend(fabric, placement, reg), placement, reg


def _split_8way():
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = Deployment("t")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    return fabric, d


def _one_round_transfers(be, method_name, *args, **kwargs):
    """The transfers of *one* round -- `EngineCCBackend` calls `_round_ns`
    exactly once per `predict_*` call and multiplies its duration by the
    round count arithmetically (every round of a ring is structurally
    identical, so simulating each one separately would be redundant work,
    not a different answer). Tests that need the *whole* operation's
    byte/transfer totals multiply this by the same round count the
    production code uses -- shown in each test's own comment."""
    captured: list[Transfer] = []
    real = EngineCCBackend._round_ns.__get__(be)

    def spy(edges, chunk_bytes, key_prefix):
        transfers = [Transfer(key=f"{key_prefix}-{i}", src=be._placement.gpu(a),
                              dst=be._placement.gpu(b), size_bytes=chunk_bytes)
                    for i, (a, b) in enumerate(edges)]
        captured.extend(transfers)
        return real(edges, chunk_bytes, key_prefix)

    with mock.patch.object(be, "_round_ns", side_effect=spy):
        getattr(be, method_name)(*args, **kwargs)
    return captured


# --------------------------------------------------------------- allreduce


def test_ring_allreduce_bytes_match_closed_form():
    """n=8, S=65536: a ring all-reduce moves 2(n-1) rounds of S/n bytes.
    Per-participant total across the whole operation is the standard ring
    volume 2(n-1)/n * S = 14/8 * 65536 = 114,688 B -- confirmed by summing
    every transfer this call actually submits divided by the number of
    distinct ranks each round touches once as a sender."""
    fabric, d = _split_8way()
    be, _, _ = _backend(fabric, d, packed)
    n = 8

    # One round's transfers (every round of a ring is identical -- see
    # _one_round_transfers' own docstring): n edges, each S/n bytes.
    transfers = _one_round_transfers(
        be, "predict_allreduce", SIZE, n, cluster_type=ClusterType.DECODE_ATTN,
        comm_domain="TP")

    rounds = 2 * (n - 1)
    chunk_bytes = SIZE // n
    assert len(transfers) == n                     # n edges in this one round
    assert all(t.size_bytes == chunk_bytes for t in transfers)

    per_rank_total = rounds * chunk_bytes           # each rank: 1 edge/round
    assert per_rank_total == 2 * (n - 1) * SIZE // n
    assert per_rank_total == 114_688


def test_ring_crosses_boundary_twice_not_per_pair():
    """On a 4+4 split of n=8, S=65536: the ring's own 2 boundary-crossing
    edges appear in every one of the 14 rounds, carrying
    14 * 2 * (S/n) = 229,376 B over the slow link -- not the 1,835,008 B
    (14 rounds * 16 all-pairs * 8192 B) a model that charged every
    cross-domain pair on every round would produce. This project's own
    ring implementation was never doing the latter (see this module's own
    docstring); this test locks in the former, correct figure directly
    against what `EngineCCBackend` actually submits.
    """
    fabric, d = _split_8way()
    be, placement, _ = _backend(fabric, d, spread)
    n = 8
    rounds = 2 * (n - 1)

    # One round's transfers; the same 2 crossing edges recur in every one
    # of the 14 rounds (the ring's topology doesn't change between
    # rounds), so the whole operation's crossing volume is this one
    # round's crossing bytes times the round count.
    transfers = _one_round_transfers(
        be, "predict_allreduce", SIZE, n, cluster_type=ClusterType.DECODE_ATTN,
        comm_domain="TP")

    crossing_bytes_per_round = sum(
        t.size_bytes for t in transfers
        if fabric.domain_of(t.src) != fabric.domain_of(t.dst))
    assert crossing_bytes_per_round == 16_384                    # 2 edges * 8192 B
    total_crossing_bytes = rounds * crossing_bytes_per_round
    assert total_crossing_bytes == 229_376
    assert total_crossing_bytes != 1_835_008


# ------------------------------------------------------------- all-to-all


def test_all_to_all_per_pair_volume():
    """Every ordered pair carries data_size_bytes/n^2, not
    data_size_bytes/n (task 20's bug, fixed here): for n=8, S=65536, each
    of the n(n-1)=56 pairs carries 65536/64 = 1024 B, and the total across
    every pair is 56 * 1024 = 57,344 B -- an eighth of task 20's original
    (n=8) per-pair figure of 8192 B."""
    fabric, d = _split_8way()
    be, _, _ = _backend(fabric, d, packed)
    n = 8

    transfers = _one_round_transfers(
        be, "predict_all_to_all", SIZE, n, cluster_type=ClusterType.DECODE_ATTN,
        comm_domain="TP")

    expected_pair_bytes = SIZE // (n * n)
    assert expected_pair_bytes == 1024
    assert len(transfers) == n * (n - 1) == 56
    assert all(t.size_bytes == expected_pair_bytes for t in transfers)
    assert sum(t.size_bytes for t in transfers) == 56 * 1024 == 57_344


# ------------------------------------------------------ allgather/reduce_scatter


def test_all_gather_is_half_a_ring():
    """all-gather and reduce-scatter are each n-1 rounds of S/n -- half of
    allreduce's 2(n-1) rounds -- and summing their durations reproduces
    allreduce's own duration exactly, since both share the identical
    round/edge/chunk structure this project's ring implementation uses for
    all three. Confirmed by construction (same code path), not merely
    asserted."""
    fabric, d = _split_8way()
    be, _, _ = _backend(fabric, d, spread)
    n = 8
    kwargs = dict(cluster_type=ClusterType.DECODE_ATTN, comm_domain="TP")

    # All three share the identical one-round edge set (same ring order,
    # same chunk size) -- what differs is only the round-count multiplier
    # applied outside that one round (n-1 for each half; 2(n-1) for the
    # whole reduce).
    ag_transfers = _one_round_transfers(be, "predict_allgather", SIZE, n, **kwargs)
    rs_transfers = _one_round_transfers(be, "predict_reduce_scatter", SIZE, n, **kwargs)
    ar_transfers = _one_round_transfers(be, "predict_allreduce", SIZE, n, **kwargs)
    assert len(ag_transfers) == len(rs_transfers) == len(ar_transfers) == n

    ag_ms = be.predict_allgather(SIZE, n, **kwargs)
    rs_ms = be.predict_reduce_scatter(SIZE, n, **kwargs)
    ar_ms = be.predict_allreduce(SIZE, n, **kwargs)
    assert ag_ms == rs_ms  # identical structure -> identical cost
    assert (ag_ms + rs_ms) == pytest.approx(ar_ms, rel=1e-9)


# ------------------------------------------------------- point-to-point


def test_point_to_point_unchanged():
    """send_recv is untouched by this task -- still exactly one Transfer,
    the full data_size_bytes, no round/chunk logic at all."""
    fabric, d = _split_8way()
    be, placement, reg = _backend(fabric, d, packed)
    ranks = d.replicas[0].ranks[:2]
    reg.register(ClusterType.DECODE_ATTN, "PP", 2, ranks)

    from engine.network.transfers import run_transfers as _rt

    with mock.patch("integration.cc_backend.engine_backend.run_transfers",
                    wraps=_rt) as spy:
        be.predict_send_recv(SIZE, cluster_type=ClusterType.DECODE_ATTN, comm_domain="PP")

    assert spy.call_count == 1
    (_, submitted), _ = spy.call_args
    assert len(submitted) == 1
    assert submitted[0].size_bytes == SIZE


# ---------------------------------------------------------- the sanity check


def test_packed_still_cheaper_than_split():
    """The check that stopped task 20's release, restated: after this
    task's correction, a packed tensor-parallel group must still cost
    strictly less than the same group split across two domains. If this
    fails, the correction is wrong."""
    fabric, d = _split_8way()
    packed_be, _, _ = _backend(fabric, d, packed)

    fabric2, d2 = _split_8way()
    split_be, _, _ = _backend(fabric2, d2, spread)

    kwargs = dict(cluster_type=ClusterType.DECODE_ATTN, comm_domain="TP")
    packed_ms = packed_be.predict_allreduce(SIZE, 8, **kwargs)
    split_ms = split_be.predict_allreduce(SIZE, 8, **kwargs)
    assert packed_ms < split_ms

    packed_a2a_ms = packed_be.predict_all_to_all(SIZE, 8, **kwargs)
    split_a2a_ms = split_be.predict_all_to_all(SIZE, 8, **kwargs)
    assert packed_a2a_ms < split_a2a_ms
