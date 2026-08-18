"""Phase 1 tests.

The invariants that matter: placement is injective, the three link classes
stay distinct, a scale-up domain can span machines, and group_shape produces
the canonical signature that Phase 2 will memoise on.
"""
import pytest

from engine.physical.topology import (Fabric, GpuId, LinkClass, gbps_to_GBps)
from engine.physical.builders import build_node_scale, build_rack_scale
from engine.logical.deployment import (Deployment, ParallelKind, PoolKind, Replica)
from engine.placement.placement import (PlacementError, fragmented, packed, spread)


# ---------------------------------------------------------------- units

def test_gbps_to_GBps():
    assert gbps_to_GBps(400.0) == 50.0
    assert gbps_to_GBps(800.0) == 100.0


# ---------------------------------------------------------------- fabric

def test_node_scale_shape():
    fab = build_node_scale(num_machines=2, gpus_per_machine=8, nics_per_machine=4)
    assert len(fab.gpus) == 16
    assert len(fab.domains) == 2
    assert fab.domains[0].size == 8


def test_rack_scale_domain_spans_machines():
    """The case a machine-granular model cannot express."""
    fab = build_rack_scale(num_racks=1, trays_per_rack=18, gpus_per_tray=4)
    assert len(fab.gpus) == 72
    assert len(fab.domains) == 1
    assert fab.domains[0].size == 72
    machines_in_domain = {g.machine for g in fab.domains[0].members}
    assert len(machines_in_domain) == 18, "domain must span every tray"


def test_three_link_classes_present():
    fab = build_node_scale(num_machines=2)
    classes = {lk.link_class for lk in fab.links}
    assert classes == {LinkClass.SCALE_UP, LinkClass.EGRESS, LinkClass.SCALE_OUT}


def test_egress_is_narrower_than_scale_up():
    fab = build_node_scale()
    su = [lk.capacity_GBps for lk in fab.links if lk.link_class is LinkClass.SCALE_UP]
    eg = [lk.capacity_GBps for lk in fab.links if lk.link_class is LinkClass.EGRESS]
    assert min(su) > max(eg) * 4, "egress should be far narrower than scale-up"


def test_gpus_share_nics():
    """Two GPUs behind one NIC contend before reaching a switch."""
    fab = build_node_scale(num_machines=1, gpus_per_machine=8, nics_per_machine=4)
    nics = [fab.nic_of(g) for g in fab.gpus]
    assert len(set(nics)) == 4
    assert fab.machines[0].gpus_per_nic == 2.0


def test_intra_domain_path_is_one_scale_up_hop():
    fab = build_node_scale(num_machines=2)
    a, b = GpuId(0, 0), GpuId(0, 3)
    path = fab.path(a, b)
    assert len(path) == 1
    assert path[0].link_class is LinkClass.SCALE_UP


def test_cross_machine_path_traverses_egress():
    fab = build_node_scale(num_machines=2)
    path = fab.path(GpuId(0, 0), GpuId(1, 0))
    classes = [lk.link_class for lk in path]
    assert LinkClass.EGRESS in classes, "leaving a machine must cross egress"
    assert LinkClass.SCALE_OUT in classes


def test_same_domain():
    fab = build_node_scale(num_machines=2)
    assert fab.same_domain(GpuId(0, 0), GpuId(0, 7))
    assert not fab.same_domain(GpuId(0, 0), GpuId(1, 0))


# ---------------------------------------------------------------- logical

def test_replica_world_size_and_groups():
    r = Replica(PoolKind.DECODE_ATTN, 0, tp=4, dp=2)
    assert r.world_size == 8
    tp_groups = r.groups(ParallelKind.TP)
    assert len(tp_groups) == 2
    assert all(g.size == 4 for g in tp_groups)
    # TP is innermost, so its members are contiguous
    idx = [rk.index for rk in tp_groups[0].ranks]
    assert idx == [0, 1, 2, 3]


# ---------------------------------------------------------------- placement

def _deployment(tp=4, dp=2):
    d = Deployment("test")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=tp, dp=dp))
    return d


def test_packed_keeps_tp_in_one_domain():
    fab = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = _deployment(tp=4, dp=2)
    p = packed(d, fab)
    for g in d.replicas[0].groups(ParallelKind.TP):
        assert not p.crosses_scale_up_boundary(g.ranks)
        assert p.group_shape(g) == (4,)


def test_spread_splits_tp_across_domains():
    fab = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = _deployment(tp=4, dp=2)
    p = spread(d, fab)
    g = d.replicas[0].groups(ParallelKind.TP)[0]
    assert p.crosses_scale_up_boundary(g.ranks)
    assert p.group_shape(g) == (2, 2)


def test_group_shape_is_canonical():
    """The signature Phase 2 memoises on: order-independent, so isomorphic
    placements collapse to one cache entry."""
    fab = build_node_scale(num_machines=4, gpus_per_machine=8)
    d = _deployment(tp=8, dp=1)
    assert packed(d, fab).group_shape(
        d.replicas[0].groups(ParallelKind.TP)[0]) == (8,)
    assert spread(d, fab).group_shape(
        d.replicas[0].groups(ParallelKind.TP)[0]) == (2, 2, 2, 2)


def test_placement_rejects_double_booking():
    fab = build_node_scale(num_machines=1, gpus_per_machine=8)
    d = _deployment(tp=4, dp=2)
    p = packed(d, fab)
    ranks = d.ranks
    with pytest.raises(PlacementError):
        p.assign(ranks[0], p.gpu(ranks[1]))


def test_placement_rejects_oversubscribed_fabric():
    fab = build_node_scale(num_machines=1, gpus_per_machine=4)
    d = _deployment(tp=4, dp=2)          # needs 8, fabric has 4
    with pytest.raises(PlacementError):
        packed(d, fab)


def test_induced_links_grow_when_split():
    """A split group touches strictly more links than a packed one, and the
    extra links are egress and scale-out."""
    fab = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = _deployment(tp=4, dp=2)
    g = d.replicas[0].groups(ParallelKind.TP)[0]

    packed_links = packed(d, fab).induced_links(g)
    spread_links = spread(d, fab).induced_links(g)
    assert len(spread_links) > len(packed_links)

    packed_classes = {lk.link_class for lk in packed_links.values()}
    spread_classes = {lk.link_class for lk in spread_links.values()}
    assert packed_classes == {LinkClass.SCALE_UP}
    assert LinkClass.EGRESS in spread_classes


def test_fragmented_is_deterministic_for_a_seed():
    fab = build_node_scale(num_machines=4, gpus_per_machine=8)
    d = _deployment(tp=4, dp=2)
    a = fragmented(d, fab, seed=7)
    b = fragmented(d, fab, seed=7)
    assert a.mapping == b.mapping
    c = fragmented(d, fab, seed=8)
    assert a.mapping != c.mapping


def test_rack_scale_packed_keeps_large_group_in_one_domain():
    """72 GPUs in one scale-up domain: a 16-way TP group still packs."""
    fab = build_rack_scale(num_racks=1)
    d = Deployment("big")
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=16, ep=4))
    p = packed(d, fab)
    g = d.replicas[0].groups(ParallelKind.TP)[0]
    assert p.group_shape(g) == (16,)
    assert not p.crosses_scale_up_boundary(g.ranks)
