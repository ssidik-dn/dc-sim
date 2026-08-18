"""Contention model tests.

Every expected value here is closed-form max-min fair share, computed by hand.
That is deliberate: arithmetic cannot itself be wrong, which makes it a better
acceptance criterion than another simulator's output -- and it was the only
option once ASTRA-sim turned out to serialise independent collectives and so
could not supply a reference curve at all.

Units: capacity in GB/s equals bytes per nanosecond, so a 100 GB/s link moves
100 bytes every nanosecond and 1000 bytes takes 10 ns.
"""
import pytest

from engine.network.allocator import (max_min_fair_share, verify_conservation)
from engine.network.model import FlowNetwork


# ---------------------------------------------------------------- allocator

def test_single_flow_gets_full_capacity():
    a = max_min_fair_share({"f": ["L"]}, {"L": 100.0})
    assert a.rate("f") == 100.0
    assert a.bottleneck["f"] == "L"


def test_two_flows_sharing_one_link_split_it():
    """The canonical case: half each."""
    a = max_min_fair_share({"f1": ["L"], "f2": ["L"]}, {"L": 100.0})
    assert a.rate("f1") == 50.0
    assert a.rate("f2") == 50.0


def test_n_flows_sharing_one_link():
    flows = {f"f{i}": ["L"] for i in range(8)}
    a = max_min_fair_share(flows, {"L": 400.0})
    for f in flows:
        assert a.rate(f) == 50.0


def test_disjoint_flows_do_not_interfere():
    a = max_min_fair_share({"f1": ["A"], "f2": ["B"]},
                           {"A": 100.0, "B": 50.0})
    assert a.rate("f1") == 100.0
    assert a.rate("f2") == 50.0


def test_flow_with_no_links_is_unconstrained():
    """Both ranks on one GPU: nothing crosses the fabric."""
    a = max_min_fair_share({"f": []}, {})
    assert a.rate("f") == float("inf")


def test_bottleneck_is_the_narrowest_link_on_the_path():
    a = max_min_fair_share({"f": ["fast", "slow"]},
                           {"fast": 400.0, "slow": 50.0})
    assert a.rate("f") == 50.0
    assert a.bottleneck["f"] == "slow"


def test_max_min_gives_more_to_the_unconstrained_flow():
    """f1 crosses a narrow link and is capped at 10. f2 shares only the wide
    link, so it gets everything f1 left: 100 - 10 = 90. Equal splitting would
    wrongly give both 50."""
    a = max_min_fair_share({"f1": ["wide", "narrow"], "f2": ["wide"]},
                           {"wide": 100.0, "narrow": 10.0})
    assert a.rate("f1") == pytest.approx(10.0)
    assert a.rate("f2") == pytest.approx(90.0)


def test_oversubscribed_uplink():
    """Four flows, each on its own 100 GB/s access link, all crossing one
    200 GB/s uplink -- a 2:1 oversubscription. Each gets 50."""
    flows = {f"f{i}": [f"acc{i}", "up"] for i in range(4)}
    cap = {f"acc{i}": 100.0 for i in range(4)}
    cap["up"] = 200.0
    a = max_min_fair_share(flows, cap)
    for f in flows:
        assert a.rate(f) == pytest.approx(50.0)
    verify_conservation(a, flows, cap)


def test_conservation_holds_on_a_tangled_topology():
    flows = {"a": ["L1", "L2"], "b": ["L2", "L3"], "c": ["L1", "L3"],
             "d": ["L2"], "e": ["L1"]}
    cap = {"L1": 100.0, "L2": 70.0, "L3": 40.0}
    a = max_min_fair_share(flows, cap)
    verify_conservation(a, flows, cap)


def test_unknown_link_raises():
    with pytest.raises(KeyError):
        max_min_fair_share({"f": ["nope"]}, {"L": 10.0})


# ---------------------------------------------------------------- flow model

def test_single_flow_completion_time():
    """1000 bytes at 100 bytes/ns is 10 ns."""
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f", ["L"], 1000, at_ns=0)
    assert n.next_completion_time_ns() == 10
    done = n.advance_to(10)
    assert len(done) == 1
    assert done[0].completion_ns == 10
    assert done[0].duration_ns == 10


def test_two_concurrent_flows_each_take_twice_as_long():
    """Sharing halves the rate, so both finish at 20 ns rather than 10."""
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f1", ["L"], 1000, at_ns=0)
    n.submit("f2", ["L"], 1000, at_ns=0)
    done = n.run_to_idle()
    assert {c.key for c in done} == {"f1", "f2"}
    assert all(c.completion_ns == 20 for c in done)


def test_late_arrival_revises_an_incumbent_completion():
    """THE core test: a completion time changes after it was first predicted.

        t=0    f1 admitted, 1000 B, alone at 100 B/ns -> predicted done at 10
        t=5    f1 has moved 500 B. f2 admitted, 250 B. Both drop to 50 B/ns,
               so f1's prediction is revised from 10 to 15
        t=10   f2 completes (250 / 50). f1 moved another 250, 250 remain, and
               is alone again at 100 B/ns -- revised a second time
        t=12.5 f1 completes, ceiling to 13

    Two revisions in one run, in both directions. A model that fixes duration
    at submit time -- which is what Frontier does -- cannot produce either.
    """
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f1", ["L"], 1000, at_ns=0)
    assert n.next_completion_time_ns() == 10          # before f2 exists

    n.submit("f2", ["L"], 250, at_ns=5)
    assert n.next_completion_time_ns() == 10          # f2 is now the earliest

    done = n.run_to_idle()
    by = {c.key: c for c in done}
    assert by["f2"].completion_ns == 10
    assert by["f1"].completion_ns == 13               # not 10, and not 15
    assert by["f1"].duration_ns == 13


def test_freed_bandwidth_is_returned_to_survivors():
    """The second half of revision: f1 speeds back up when f2 leaves. If it
    stayed throttled it would finish at 15 rather than 13."""
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f1", ["L"], 1000, at_ns=0)
    n.submit("f2", ["L"], 250, at_ns=5)
    n.advance_to(10)
    assert n.rate("f1") == 100.0, "f1 must return to full rate once alone"


def test_revision_counter_bumps_on_every_change():
    n = FlowNetwork({"L": 100.0})
    r0 = n.revision
    n.submit("f1", ["L"], 1000, at_ns=0)
    assert n.revision > r0
    r1 = n.revision
    n.submit("f2", ["L"], 1000, at_ns=0)
    assert n.revision > r1


def test_completion_frees_bandwidth_for_survivors():
    """f2 is small and finishes early; f1 should then speed up. Total bytes are
    1000 + 200 = 1200 at 100 bytes/ns, so the last completion is at 12 ns."""
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f1", ["L"], 1000, at_ns=0)
    n.submit("f2", ["L"], 200, at_ns=0)
    done = n.run_to_idle()
    by = {c.key: c for c in done}
    assert by["f2"].completion_ns == 4                # 200 at 50 bytes/ns
    assert by["f1"].completion_ns == 12               # 200 left at 100 after t=4


def test_disjoint_flows_are_unaffected_by_each_other():
    n = FlowNetwork({"A": 100.0, "B": 100.0}, verify=True)
    n.submit("f1", ["A"], 1000, at_ns=0)
    n.submit("f2", ["B"], 1000, at_ns=0)
    done = n.run_to_idle()
    assert all(c.completion_ns == 10 for c in done)


def test_bytes_are_conserved():
    n = FlowNetwork({"L": 100.0}, verify=True)
    sizes = {"f1": 1000, "f2": 300, "f3": 700}
    for k, s in sizes.items():
        n.submit(k, ["L"], s, at_ns=0)
    done = n.run_to_idle()
    assert sum(c.total_bytes for c in done) == sum(sizes.values())
    assert not n.in_flight


def test_bottleneck_is_attributed():
    n = FlowNetwork({"wide": 400.0, "narrow": 50.0}, verify=True)
    n.submit("f", ["wide", "narrow"], 500, at_ns=0)
    done = n.run_to_idle()
    assert done[0].bottleneck == "narrow"


def test_link_utilisation_never_exceeds_one():
    n = FlowNetwork({"L": 100.0}, verify=True)
    for i in range(5):
        n.submit(f"f{i}", ["L"], 1000, at_ns=0)
    u = n.link_utilisation()
    assert u["L"] == pytest.approx(1.0)
    assert u["L"] <= 1.0 + 1e-9


def test_completion_never_precedes_submission():
    n = FlowNetwork({"L": 100.0}, verify=True)
    n.submit("f1", ["L"], 1000, at_ns=0)
    n.submit("f2", ["L"], 10, at_ns=7)
    for c in n.run_to_idle():
        assert c.completion_ns >= c.start_ns


def test_cannot_rewind_time():
    n = FlowNetwork({"L": 100.0})
    n.submit("f", ["L"], 1000, at_ns=100)
    n.advance_to(105)
    with pytest.raises(ValueError):
        n.advance_to(103)


def test_duplicate_submission_raises():
    n = FlowNetwork({"L": 100.0})
    n.submit("f", ["L"], 100, at_ns=0)
    with pytest.raises(ValueError):
        n.submit("f", ["L"], 100, at_ns=1)


def test_many_flows_staggered_conserve_bytes():
    """Staggered arrivals over a shared link: every flow must still complete
    and the byte total must match."""
    n = FlowNetwork({"L": 100.0}, verify=True)
    total = 0
    for i in range(10):
        n.submit(f"f{i}", ["L"], 500, at_ns=i * 3)
        total += 500
    done = n.run_to_idle()
    assert len(done) == 10
    assert sum(c.total_bytes for c in done) == total
    assert not n.in_flight


# ------------------------------------------------- fabric integration

from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId, LinkClass
from engine.network.transfers import (Transfer, analyse, isolated_durations,
                                      run_transfers)


def _fab():
    return build_node_scale(num_machines=2, gpus_per_machine=8,
                            nics_per_machine=4, scale_up_GBps=400.0,
                            nic_gbps=400.0, scale_out_GBps=50.0)


def test_intra_domain_transfer_uses_scale_up_only():
    fab = _fab()
    done = run_transfers(fab, [Transfer("t", GpuId(0, 0), GpuId(0, 1), 40_000)])
    lk = next(l for l in fab.links if l.id == done[0].bottleneck)
    assert lk.link_class is LinkClass.SCALE_UP


def test_cross_machine_transfer_bottlenecks_on_the_narrow_step():
    """Leaving the machine must be paced by egress or scale-out, never by the
    wide scale-up fabric."""
    fab = _fab()
    done = run_transfers(fab, [Transfer("t", GpuId(0, 0), GpuId(1, 0), 40_000)])
    lk = next(l for l in fab.links if l.id == done[0].bottleneck)
    assert lk.link_class in (LinkClass.EGRESS, LinkClass.SCALE_OUT)


def test_crossing_a_machine_costs_more_than_staying_inside():
    fab = _fab()
    inside = run_transfers(fab, [Transfer("a", GpuId(0, 0), GpuId(0, 1), 400_000)])
    across = run_transfers(fab, [Transfer("b", GpuId(0, 0), GpuId(1, 0), 400_000)])
    assert across[0].duration_ns > inside[0].duration_ns * 4


def test_gpus_sharing_a_nic_contend():
    """The egress finding made concrete: two GPUs behind one NIC interfere
    before their traffic reaches any switch.

    Task 10 note: this threshold changed from 1.5 to a hand-computed exact
    value once the flow model started charging path latency (see
    engine/network/model.py). Before task 10, latency was zero everywhere
    and the slowdown was exactly 2.0 (pure bandwidth: contended rate 25 GB/s
    is half solo's 50 GB/s). 1.5 was headroom under that 2.0, not a
    physical requirement.

    With latency, x and y's path is 4 hops -- egress(2000ns) + scale_out
    (5000ns) + scale_out(5000ns) + egress(2000ns) = 14000ns -- identical for
    both, and unaffected by contention. Only the scale_out hop out of the
    shared NIC (NIC0->leaf0) is shared; every other hop on each path is
    exclusive to that flow.

        solo:      bandwidth 400_000/50 = 8000ns  + 14000ns latency = 22000ns
        contended: bandwidth 400_000/25 = 16000ns + 14000ns latency = 30000ns
        slowdown = 30000/22000 = 15/11 ~= 1.3636

    Latency is identical in both terms, so it dampens the ratio well below
    the bandwidth-only 2.0 -- that is not a defect, it is what a fabric
    whose hops are dominated by fixed per-hop delay actually does. The
    original ">1.5" line is not one of those two numbers; asserting the
    exact closed-form value is more honest than picking a new threshold
    that merely happens to pass.
    """
    fab = _fab()
    a, b = GpuId(0, 0), GpuId(0, 4)          # 8 GPUs over 4 NICs, round-robin
    assert fab.nic_of(a) == fab.nic_of(b), "fixture must put these on one NIC"

    solo = isolated_durations(fab, [Transfer("x", a, GpuId(1, 0), 400_000)])["x"]
    rep = analyse(fab, [Transfer("x", a, GpuId(1, 0), 400_000),
                        Transfer("y", b, GpuId(1, 1), 400_000)])
    assert rep.per_transfer_ns["x"] > solo, "sharing a NIC must slow both"
    assert solo == 22000
    assert rep.per_transfer_ns["x"] == 30000
    for s in rep.slowdown_vs({"x": solo, "y": solo}).values():
        assert s == pytest.approx(15 / 11)


def test_gpus_on_different_nics_interfere_less():
    """Placement matters within one machine: different NICs means less
    contention than the same NIC, all else equal."""
    fab = _fab()
    same_nic = analyse(fab, [
        Transfer("x", GpuId(0, 0), GpuId(1, 0), 400_000),
        Transfer("y", GpuId(0, 4), GpuId(1, 1), 400_000)])
    diff_nic = analyse(fab, [
        Transfer("x", GpuId(0, 0), GpuId(1, 0), 400_000),
        Transfer("y", GpuId(0, 1), GpuId(1, 1), 400_000)])
    assert fab.nic_of(GpuId(0, 0)) != fab.nic_of(GpuId(0, 1))
    assert diff_nic.makespan_ns <= same_nic.makespan_ns


def test_report_attributes_bottlenecks_by_class():
    fab = _fab()
    rep = analyse(fab, [Transfer(f"t{i}", GpuId(0, i), GpuId(1, i), 200_000)
                        for i in range(4)])
    assert sum(rep.bottleneck_classes.values()) == 4
    assert set(rep.bottleneck_classes) <= {"egress", "scale_out", "scale_up"}


def test_staggered_transfers_conserve_and_complete():
    fab = _fab()
    ts = [Transfer(f"t{i}", GpuId(0, i), GpuId(1, i), 100_000, submit_ns=i * 500)
          for i in range(6)]
    rep = analyse(fab, ts)
    assert len(rep.completions) == 6
    assert rep.makespan_ns >= max(t.submit_ns for t in ts)
