"""Task 10: path latency in the flow model.

Every expected value here is closed-form and hand computed, shown in a
comment, as with test_contention.py's allocator tests. Units: capacity in
GB/s equals bytes per nanosecond (see test_contention.py's own note); a
100 GB/s link moves 100 bytes every nanosecond.

Model chosen (see docs/tasks/10-latency-report.md S1 for the full
justification and its stated limitation): latency is added to the computed
completion, exactly once, at the moment a flow's bytes finish moving --
not at submission, and not re-added on every reallocation in between. A
flow that has finished moving bytes (DRAINED) holds no bandwidth and cannot
be re-delayed by a later arrival; only its fixed latency tail remains.
"""
from __future__ import annotations

import pytest

from engine.network.model import FlowNetwork
from engine.network.transfers import Transfer, isolated_durations
from engine.physical.builders import build_node_scale
from engine.physical.topology import (Fabric, GpuId, Link, LinkClass, NicId,
                                      SwitchId)


# ---------------------------------------------------------------- FlowNetwork


def test_zero_latency_matches_previous_behaviour():
    """1000 bytes at 100 bytes/ns, no latency: 10 ns, exactly as before
    task 10 (test_contention.py's test_single_flow_completion_time)."""
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f", ["L"], 1000, at_ns=0, path_latency_ns=0.0)
    done = n.run_to_idle()
    assert done[0].completion_ns == 10
    assert done[0].duration_ns == 10


def test_single_hop_latency_is_added_once():
    """1000 bytes at 100 bytes/ns is 10 ns of bandwidth time; +20 ns latency
    is 30 ns total. Exact, no contention to complicate it."""
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f", ["L"], 1000, at_ns=0, path_latency_ns=20.0)
    done = n.run_to_idle()
    assert done[0].completion_ns == 30
    assert done[0].duration_ns == 30


def test_multi_hop_latency_accumulates():
    """Three hops of 100, 200, 300 ns (egress, then two scale_out legs of a
    custom 3-link fabric) sum to 600 ns. 1000 bytes at the bottleneck
    100 bytes/ns is 10 ns of bandwidth time. Total: 600 + 10 = 610 ns."""
    fab = Fabric("mini")
    g0, nic, sw, g1 = GpuId(0, 0), NicId(0, 0), SwitchId("leaf", 0), GpuId(1, 0)
    fab.add_link(Link(g0, nic, LinkClass.EGRESS, 100.0, 100.0))
    fab.add_link(Link(nic, sw, LinkClass.SCALE_OUT, 100.0, 200.0))
    fab.add_link(Link(sw, g1, LinkClass.SCALE_OUT, 100.0, 300.0))

    duration = isolated_durations(fab, [Transfer("t", g0, g1, 1000)])["t"]
    assert duration == 610


def test_latency_not_recharged_on_reallocation():
    """The core test. Link L, capacity 100 bytes/ns, latency 20 ns.

        t=0   f1 submitted, 1000 B, alone at rate 100 -> bandwidth ETA 10,
              predicted completion (with latency) 10 + 20 = 30
        t=5   f1 has moved 500 B (250 B/ns... no, 5 ns * 100 = 500 B),
              500 B remain. f2 submitted, 250 B, latency 0. Both drop to
              rate 50. f1's bandwidth ETA revises to 5 + 500/50 = 15
        t=10  f2 completes: 250 B / 50 B/ns = 5 ns from t=5. f2 has zero
              latency, so it is reported immediately, completion_ns = 10.
              f1 has moved another 5*50 = 250 B; 250 B remain, alone again
              at rate 100 -> new bandwidth ETA 10 + 250/100 = 12.5
        t=13  f1's bytes finish moving (12.5 ceiled to 13, matching
              test_contention.py's test_late_arrival_revises_an_incumbent_completion
              for this identical 1000/250/rate-100 setup). It drains
              (frees its link share -- nothing left to free it to, since
              f2 is already gone) and enters its 20 ns latency tail.
        t=33  f1's tail expires: 13 + 20 = 33. Reported now.

    f1's total duration is 33 ns: its own path latency (20) appears exactly
    once, added at the one moment its bytes finished moving -- not at t=0,
    and not re-added at either of the two reallocations at t=5 and t=10.
    A model that (incorrectly) recharged latency on each reallocation could
    not produce this exact number.
    """
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f1", ["L"], 1000, at_ns=0, path_latency_ns=20.0)
    n.submit("f2", ["L"], 250, at_ns=5, path_latency_ns=0.0)
    done = n.run_to_idle()
    by = {c.key: c for c in done}
    assert by["f2"].completion_ns == 10
    assert by["f1"].completion_ns == 33
    assert by["f1"].duration_ns == 33


def test_small_transfer_is_latency_dominated():
    """1 byte at 400 bytes/ns (bandwidth term 0.0025 ns) against 1000 ns of
    latency: total is 1000.0025 ns, ceiled to 1001 -- essentially all
    latency, none of it serialisation. This is the regime activation
    exchange (task 08: 192 calls/run, small payloads) occupies."""
    n = FlowNetwork({"L": 400.0}, verify=True)
    n.submit("f", ["L"], 1, at_ns=0, path_latency_ns=1000.0)
    done = n.run_to_idle()
    assert done[0].duration_ns == 1001
    assert done[0].duration_ns == pytest.approx(1000.0, abs=2)


def test_large_transfer_is_bandwidth_dominated():
    """400,000,000 bytes at 400 bytes/ns is exactly 1,000,000 ns of
    bandwidth time. +1000 ns latency is 0.1% of the 1,001,000 ns total --
    negligible, and both regimes fall out of the same formula."""
    n = FlowNetwork({"L": 400.0}, verify=True)
    n.submit("f", ["L"], 400_000_000, at_ns=0, path_latency_ns=1000.0)
    done = n.run_to_idle()
    assert done[0].duration_ns == 1_001_000
    latency_fraction = 1000 / done[0].duration_ns
    assert latency_fraction < 0.001


def test_latency_changes_split_versus_packed_ratio():
    """Task 09's scenario, repeated. Before task 10 the ratio was exactly
    the fabric's bandwidth ratio (8.0, task 09 report S3) because the flow
    model never read Link.latency_ns at all. Now it does, and the ratio
    must move -- and it must move UP, not down: packed crosses one scale-up
    hop (low latency), split crosses four hops -- egress, scale_out,
    scale_out, egress (higher combined latency) -- and that additional
    latency penalises split on top of its bandwidth disadvantage, the same
    direction, not against it. (Contrast test_contention.py's
    test_gpus_sharing_a_nic_contend, where two flows share the SAME path and
    latency cancels out of the ratio instead of compounding it.)
    """
    fab = build_node_scale(num_machines=2, gpus_per_machine=8,
                           scale_up_GBps=400.0, scale_out_GBps=50.0)
    size = 1 << 30  # 1 GiB, the same order of magnitude as task 09's KV size
    packed = isolated_durations(
        fab, [Transfer("k", GpuId(0, 0), GpuId(0, 1), size)])["k"]
    split = isolated_durations(
        fab, [Transfer("k", GpuId(0, 0), GpuId(1, 0), size)])["k"]

    ratio = split / packed
    bandwidth_only_ratio = 400.0 / 50.0
    assert ratio != pytest.approx(bandwidth_only_ratio, rel=1e-6)
    assert ratio > bandwidth_only_ratio, (
        "split's extra hops add latency on top of its bandwidth "
        "disadvantage, so the ratio should exceed the pure bandwidth ratio, "
        f"not fall short of it (got {ratio}, bandwidth-only was "
        f"{bandwidth_only_ratio})")
