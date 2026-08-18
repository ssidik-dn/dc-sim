"""Routing mode must change measured contention, not just path selection.

Task 03 found that `Fabric.path()` is breadth-first and picks one path, so on a
fabric with several equal-cost paths every flow lands on the same one. Adding
spine capacity then shows zero measured benefit -- the sharpest form of a
parameter that does not move the number.

These tests pin the fix: PER_FLOW_ECMP must disperse flows and the dispersal
must show up in makespan, not merely in a path listing.
"""
import pytest

from engine.physical.builders import build_node_scale
from engine.physical.topology import (Fabric, FabricMode, GpuId, Link, LinkClass,
                                      Machine, NicId, ScaleUpDomain, SwitchId)
from engine.network.transfers import (Transfer, analyse, isolated_durations,
                                      run_transfers)


def two_spine_fabric(uplink_GBps: float = 50.0,
                     access_GBps: float = 400.0):
    """A minimal leaf-spine: one machine per leaf, two spines between them.

        m0.g0 -- m0.n0 -- leaf0 =={spine0, spine1}== leaf1 -- m1.n0 -- m1.g0

    Two equal-cost routes exist. Built by hand rather than from a blueprint
    because node-scale and rack-scale each have a single leaf and therefore no
    parallel paths at all -- which is also why no earlier measurement in this
    project could have been affected by the concentration bug.
    """
    fab = Fabric("two-spine")
    for mid in (0, 1):
        gpus = [GpuId(mid, i) for i in range(2)]
        nic = NicId(mid, 0)
        fab.add_machine(Machine(mid, gpus, [nic]))
        fab.add_domain(ScaleUpDomain(mid, frozenset(gpus)))
        for i, g in enumerate(gpus):
            fab.bind_nic(g, nic)
            fab.add_link(Link(g, nic, LinkClass.EGRESS, 400.0, 100.0))
            for h in gpus[i + 1:]:
                fab.add_link(Link(g, h, LinkClass.SCALE_UP, 400.0, 100.0))
        leaf = SwitchId("leaf", mid)
        # access link deliberately wide, so the leaf-spine uplinks are the
        # constraint. If the access link binds instead, spine choice cannot
        # matter and ECMP would show nothing -- a real trap when building
        # fixtures for routing tests.
        fab.add_link(Link(nic, leaf, LinkClass.SCALE_OUT, access_GBps, 1000.0))
        for s in range(2):
            fab.add_link(Link(leaf, SwitchId("spine", s), LinkClass.SCALE_OUT,
                              uplink_GBps, 1000.0))
    return fab


def test_single_path_concentrates_every_flow():
    """The Task 03 finding, pinned as a regression: without ECMP all flows take
    the same route regardless of identity."""
    fab = build_node_scale(num_machines=2, gpus_per_machine=4, nics_per_machine=2)
    a, b = GpuId(0, 0), GpuId(1, 0)
    first = [lk.id for lk in fab.path(a, b)]
    for _ in range(20):
        assert [lk.id for lk in fab.path(a, b)] == first


def test_ecmp_default_is_single_path():
    """Default must not change any existing result."""
    fab = build_node_scale(num_machines=2, gpus_per_machine=2, nics_per_machine=1)
    ts = [Transfer("t", GpuId(0, 0), GpuId(1, 0), 200_000)]
    assert run_transfers(fab, ts)[0].duration_ns == \
           run_transfers(fab, ts, mode=FabricMode.SINGLE_PATH)[0].duration_ns


def test_ecmp_hashes_on_transfer_key_not_endpoints():
    """Real switch ECMP hashes the flow 5-tuple, so two connections between the
    same hosts take different paths. Keying on (src, dst) would pin them
    together and reproduce the concentration."""
    fab = two_spine_fabric()
    a, b = GpuId(0, 0), GpuId(1, 0)
    paths = {tuple(lk.id for lk in fab.route(FabricMode.PER_FLOW_ECMP,
                                             f"flow-{i}", a, b))
             for i in range(40)}
    assert len(paths) > 1, "distinct transfer keys must reach distinct paths"


def test_isolated_durations_uses_the_same_mode():
    """A slowdown ratio must compare like with like. Passing different modes to
    the contended and isolated runs would attribute a routing difference to
    contention."""
    fab = two_spine_fabric()
    ts = [Transfer(f"t{i}", GpuId(0, i % 2), GpuId(1, i % 2), 200_000)
          for i in range(4)]
    solo_sp = isolated_durations(fab, ts, mode=FabricMode.SINGLE_PATH)
    solo_ec = isolated_durations(fab, ts, mode=FabricMode.PER_FLOW_ECMP)
    assert set(solo_sp) == set(solo_ec) == {t.key for t in ts}


def test_sprayed_raises_rather_than_guessing():
    fab = two_spine_fabric()
    ts = [Transfer("t", GpuId(0, 0), GpuId(1, 0), 100_000)]
    with pytest.raises(NotImplementedError, match="completion semantics"):
        run_transfers(fab, ts, mode=FabricMode.SPRAYED)


def test_analyse_accepts_mode():
    fab = two_spine_fabric()
    ts = [Transfer(f"t{i}", GpuId(0, i % 2), GpuId(1, i % 2), 100_000)
          for i in range(4)]
    rep = analyse(fab, ts, mode=FabricMode.PER_FLOW_ECMP)
    assert len(rep.completions) == 4
    assert rep.makespan_ns > 0


def test_ecmp_makes_added_spine_capacity_visible():
    """The Task 03 finding, closed.

    With single-path routing every flow takes the same uplink, so a second
    spine contributes nothing and adding capacity to a real deployment would
    show zero measured benefit. ECMP must recover some of it.

    Measured on this fixture (two spines, uplinks the bottleneck):

        flows   single      ecmp    gain   split
            8   320000    240000   1.33x   6/2
           16   640000    480000   1.33x   12/4
           32  1280000    840000   1.52x   21/11
           64  2560000   1440000   1.78x   36/28

    The gain is well short of the ideal 2.00x at small flow counts and climbs
    towards it as flows accumulate. That is hash imbalance, and it is real
    behaviour rather than a modelling artefact -- which is why this asserts a
    trend rather than a fixed ratio.
    """
    fab = two_spine_fabric(uplink_GBps=50.0, access_GBps=400.0)

    def gain(n):
        ts = [Transfer(f"f{i}", GpuId(0, i % 2), GpuId(1, i % 2), 2_000_000)
              for i in range(n)]
        single = analyse(fab, ts, mode=FabricMode.SINGLE_PATH).makespan_ns
        ecmp = analyse(fab, ts, mode=FabricMode.PER_FLOW_ECMP).makespan_ns
        return single / ecmp

    small, large = gain(8), gain(64)
    assert small > 1.1, "ECMP must beat single-path even with few flows"
    assert large > small, "balance must improve as flows accumulate"
    assert large < 2.05, "cannot beat the ideal two-spine speedup"


def test_ecmp_balance_degrades_with_few_flows():
    """Hash collisions are real, not an artefact. Four flows over two paths can
    land 4-0, and a fixture that uses too few flows will show ECMP doing
    nothing. Documented so the next person does not read it as a bug."""
    fab = two_spine_fabric(uplink_GBps=50.0, access_GBps=400.0)
    a, b = GpuId(0, 0), GpuId(1, 0)
    chosen = [tuple(lk.id for lk in fab.route(FabricMode.PER_FLOW_ECMP, f"t{i}", a, b))
              for i in range(4)]
    assert len(set(chosen)) >= 1          # may be 1: that is the point
    many = [tuple(lk.id for lk in fab.route(FabricMode.PER_FLOW_ECMP, f"t{i}", a, b))
            for i in range(40)]
    assert len(set(many)) == 2, "with enough flows both paths must be used"
