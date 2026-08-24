"""Tests for the InfraGraph blueprints (task 02 part B; Clos section
rewritten in task 03 for the corrected two-tier leaf-spine construction).

`test_oversubscription_shows_up_in_contention` is the binding test: a
blueprint parameter that doesn't move measured cost is worse than useless.
`test_no_leaf_to_leaf_links` is the regression guard against task 02's
pod workaround reappearing.
"""
from __future__ import annotations

import pytest

from engine.infragraph.blueprints import clos_fat_tree_fabric, single_tier_fabric
from engine.infragraph.emit import to_infragraph
from engine.infragraph.parse import from_infragraph
from engine.network.transfers import Transfer, analyse
from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId, SwitchId


def _class_counts(fabric):
    counts = {}
    for lk in fabric.links:
        counts[lk.link_class] = counts.get(lk.link_class, 0) + 1
    return counts


def _switch_ids(fabric, tier):
    ids = set()
    for lk in fabric.links:
        for node in (lk.src, lk.dst):
            if isinstance(node, SwitchId) and node.tier == tier:
                ids.add(node)
    return ids


_COMMON_KWARGS = dict(
    scale_up_GBps=400.0, scale_up_latency_ns=936.25,
    nic_gbps=400.0, egress_latency_ns=2000.0,
    scale_out_GBps=50.0, scale_out_latency_ns=5000.0,
)


# ---------------------------------------------------------------- single tier

def test_single_tier_matches_build_node_scale():
    builder = build_node_scale(num_machines=3, gpus_per_machine=8,
                               nics_per_machine=4, **_COMMON_KWARGS)
    blueprint = single_tier_fabric(num_machines=3, gpus_per_machine=8,
                                   nics_per_machine=4, **_COMMON_KWARGS)

    assert set(builder.gpus) == set(blueprint.gpus)
    assert _class_counts(builder) == _class_counts(blueprint)
    assert len(builder.links) == len(blueprint.links)

    for g in builder.gpus:
        assert builder.domain_of(g) == blueprint.domain_of(g)
        assert builder.nic_of(g) == blueprint.nic_of(g)

    b_by_id = {lk.id: lk for lk in blueprint.links}
    for lk in builder.links:
        other = b_by_id[lk.id]
        assert other.link_class == lk.link_class
        assert other.capacity_GBps == lk.capacity_GBps
        assert other.latency_ns == lk.latency_ns


# --------------------------------------------------------------- leaf-spine

@pytest.mark.parametrize("k", [4, 8, 16])
def test_leaf_spine_counts_match_formula(k):
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=2,
                               nics_per_machine=1, **_COMMON_KWARGS)
    half = k // 2
    assert len(_switch_ids(fab, "spine")) == half
    assert len(_switch_ids(fab, "leaf")) == k
    assert len(fab.machines) == k * half
    assert len(fab.gpus) == 2 * k * half


@pytest.mark.parametrize("k", [4, 8, 16])
def test_every_leaf_connects_to_every_spine(k):
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    half = k // 2
    leaf_spine_links = [
        lk for lk in fab.links
        if isinstance(lk.src, SwitchId) and lk.src.tier == "leaf"
        and isinstance(lk.dst, SwitchId) and lk.dst.tier == "spine"
    ]
    # add_link() is bidirectional by default, so the full mesh -- k leaves x
    # k/2 spines, one link each -- appears once per direction.
    assert len(leaf_spine_links) == k * half

    leaves = _switch_ids(fab, "leaf")
    spines = _switch_ids(fab, "spine")
    for leaf in leaves:
        for spine in spines:
            assert fab.link(leaf, spine) is not None, f"missing {leaf}->{spine}"


def test_no_leaf_to_leaf_links():
    """The regression guard: task 02's pod workaround added direct
    leaf-to-leaf links to route around a disconnected topology. That
    topology is gone, and so must these links be -- their presence would
    let cross-leaf traffic dodge the leaf-spine uplinks the oversubscription
    parameter is supposed to govern."""
    fab = clos_fat_tree_fabric(switch_radix=8, depth=2, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    for lk in fab.links:
        if isinstance(lk.src, SwitchId) and isinstance(lk.dst, SwitchId):
            assert lk.src.tier != lk.dst.tier, (
                f"same-tier switch link found: {lk.src} -> {lk.dst}")


def test_same_leaf_path_is_one_hop():
    k = 4
    half = k // 2
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    # Machines 0 and 1 both sit behind leaf 0 (hosts_per_leaf = half = 2).
    path = fab.path(GpuId(0, 0), GpuId(1, 0))
    switches = [n for lk in path for n in (lk.src, lk.dst) if isinstance(n, SwitchId)]
    assert set(switches) == {SwitchId("leaf", 0)}
    assert not any(s.tier == "spine" for s in switches)


def test_cross_leaf_path_traverses_a_spine():
    """Under task 02's pod topology this failed for same-pod, different-leaf
    pairs -- there was no aggregation tier to route through. Named
    separately from the same-leaf case for that reason."""
    k = 4
    half = k // 2
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    path = fab.path(GpuId(0, 0), GpuId(half, 0))  # machine `half` is leaf 1's first host
    assert any(isinstance(n, SwitchId) and n.tier == "spine"
              for lk in path for n in (lk.src, lk.dst))


def test_every_host_reaches_every_other():
    k = 8
    half = k // 2
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    mids = sorted(fab.machines)
    last = mids[-1]
    pairs = [
        (mids[0], mids[1]),                      # same leaf
        (mids[0], mids[half]),                   # adjacent leaf
        (mids[0], last),                         # furthest leaf
    ]
    for a, b in pairs:
        path = fab.path(GpuId(a, 0), GpuId(b, 0))
        assert path or a == b, f"no path between machine {a} and machine {b}"


def test_oversubscription_reduces_uplink_capacity():
    k = 4
    ratio = 4.0
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=2, oversubscription=ratio,
                               **_COMMON_KWARGS)
    leaf = SwitchId("leaf", 0)
    uplink_total = sum(lk.capacity_GBps for lk in fab.neighbours(leaf)
                       if isinstance(lk.dst, SwitchId) and lk.dst.tier == "spine")
    downlink_total = sum(lk.capacity_GBps for lk in fab.neighbours(leaf)
                         if not isinstance(lk.dst, SwitchId))
    assert uplink_total == pytest.approx(downlink_total / ratio)


def test_oversubscription_shows_up_in_contention():
    """The binding test. Same fabric shape at 1:1 and 4:1 oversubscription;
    enough concurrent cross-leaf transfers to saturate the leaf-spine
    uplinks. The 4:1 fabric must show a strictly longer makespan -- if it
    didn't, `oversubscription` would be a parameter that does nothing."""
    k = 4
    half = k // 2  # hosts_per_leaf == num_spines == half
    size_bytes = 5_000_000

    transfers = [
        Transfer("a", GpuId(0, 0), GpuId(half, 0), size_bytes),
        Transfer("b", GpuId(1, 0), GpuId(half + 1, 0), size_bytes),
    ]

    fab_1to1 = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                                    nics_per_machine=1, oversubscription=1.0,
                                    **_COMMON_KWARGS)
    fab_4to1 = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                                    nics_per_machine=1, oversubscription=4.0,
                                    **_COMMON_KWARGS)

    rep_1to1 = analyse(fab_1to1, transfers)
    rep_4to1 = analyse(fab_4to1, transfers)

    assert rep_4to1.makespan_ns > rep_1to1.makespan_ns


# ------------------------------------------------------------- round-tripping

def test_blueprint_fabrics_round_trip():
    single = single_tier_fabric(num_machines=2, gpus_per_machine=4,
                                nics_per_machine=2, **_COMMON_KWARGS)
    clos = clos_fat_tree_fabric(switch_radix=4, depth=2, gpus_per_machine=1,
                                nics_per_machine=1, oversubscription=2.0,
                                **_COMMON_KWARGS)

    for fab in (single, clos):
        doc1 = to_infragraph(fab)
        rt = from_infragraph(doc1)
        doc2 = to_infragraph(rt)
        assert doc1 == doc2
        assert set(rt.gpus) == set(fab.gpus)
        assert _class_counts(rt) == _class_counts(fab)


def test_depth_four_raises():
    """depth=3 is now implemented (task 40); depth=4 is where this
    project's own derived counts and wiring stop, per the same reasoning
    that deferred three tiers in task 03 until they had one."""
    with pytest.raises(NotImplementedError):
        clos_fat_tree_fabric(switch_radix=4, depth=4, gpus_per_machine=1,
                             nics_per_machine=1, **_COMMON_KWARGS)


# ----------------------------------------------------------------- three tier
#
# Task 02's own error was building a three-tier fat tree's edge and core
# counts *without* the aggregation tier that bridges them -- arithmetically
# consistent switch counts, and a disconnected topology, because an edge
# switch's own uplinks could then reach only a subset of a much larger core
# set. `test_plane_structure_is_correct` and `test_cross_pod_path_traverses_core`
# are the direct descendants of that finding: they check the property whose
# absence caused it, not just that today's counts happen to be right.


@pytest.mark.parametrize("k", [4, 6, 8])
def test_three_tier_counts_match_formula(k):
    fab = clos_fat_tree_fabric(switch_radix=k, depth=3, gpus_per_machine=2,
                               nics_per_machine=1, **_COMMON_KWARGS)
    half = k // 2
    assert len(_switch_ids(fab, "edge")) == k * half
    assert len(_switch_ids(fab, "aggregation")) == k * half
    assert len(_switch_ids(fab, "core")) == half * half
    assert len(fab.machines) == k ** 3 // 4
    assert len(fab.gpus) == 2 * (k ** 3 // 4)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_every_switch_port_count_closes(k):
    """Every switch at every tier uses exactly `k` ports -- no more, no
    fewer. This is what the plane structure has to get right: an edge
    switch's own `k/2` uplinks land one-per-aggregation-switch-in-its-pod,
    an aggregation switch's own `k/2` uplinks land one-per-core-switch-in-
    its-plane, and every core switch collects exactly one link per pod
    (one from each pod's own same-indexed aggregation switch)."""
    fab = clos_fat_tree_fabric(switch_radix=k, depth=3, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    for tier in ("edge", "aggregation", "core"):
        for switch in _switch_ids(fab, tier):
            n = len(fab.neighbours(switch))
            assert n == k, f"{switch} has {n} ports used, expected {k}"


def test_no_intra_tier_links():
    """The regression guard, extended to three tiers: no link joins two
    switches of the same tier, at edge, aggregation, or core -- task 03's
    own guard against task 02's pod-workaround links, generalised."""
    fab = clos_fat_tree_fabric(switch_radix=6, depth=3, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    for lk in fab.links:
        if isinstance(lk.src, SwitchId) and isinstance(lk.dst, SwitchId):
            assert lk.src.tier != lk.dst.tier, (
                f"same-tier switch link found: {lk.src} -> {lk.dst}")


def test_every_host_reaches_every_other_three_tier():
    k = 6
    half = k // 2
    fab = clos_fat_tree_fabric(switch_radix=k, depth=3, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    mids = sorted(fab.machines)
    last = mids[-1]
    pairs = [
        (mids[0], mids[1]),                       # same edge
        (mids[0], mids[half]),                    # same pod, different edge
        (mids[0], mids[half * half]),              # different pod
        (mids[0], last),                          # furthest pod
    ]
    for a, b in pairs:
        path = fab.path(GpuId(a, 0), GpuId(b, 0))
        assert path or a == b, f"no path between machine {a} and machine {b}"


def test_cross_pod_path_traverses_core():
    """The property whose absence disconnected task 02's own construction,
    checked directly rather than inferred from switch counts:

    - two hosts on different edges of the *same* pod reach each other
      through aggregation alone -- their pod's own aggregation switches
      connect to every edge switch in that pod, so core is never needed;
    - two hosts in *different* pods must cross core -- no aggregation
      switch connects to another pod's edges at all, so the only route
      between pods is up through a core switch in whichever plane the
      two pods' own same-indexed aggregation switches share.
    """
    k = 4
    half = k // 2
    fab = clos_fat_tree_fabric(switch_radix=k, depth=3, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)

    same_pod_path = fab.path(GpuId(0, 0), GpuId(half, 0))  # pod 0, edge 0 vs edge 1
    same_pod_tiers = {n.tier for lk in same_pod_path for n in (lk.src, lk.dst)
                      if isinstance(n, SwitchId)}
    assert "aggregation" in same_pod_tiers
    assert "core" not in same_pod_tiers

    cross_pod_path = fab.path(GpuId(0, 0), GpuId(half * half, 0))  # pod 0 vs pod 1
    cross_pod_tiers = {n.tier for lk in cross_pod_path for n in (lk.src, lk.dst)
                       if isinstance(n, SwitchId)}
    assert "core" in cross_pod_tiers


@pytest.mark.parametrize("k", [4, 6, 8])
def test_plane_structure_is_correct(k):
    """Aggregation switch j (in any pod) reaches exactly the core switches
    of plane j, and no others -- the structure whose omission is task 02's
    own error, one tier up from where it originally happened."""
    half = k // 2
    fab = clos_fat_tree_fabric(switch_radix=k, depth=3, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    all_core = _switch_ids(fab, "core")
    for agg in _switch_ids(fab, "aggregation"):
        j = agg.index % half  # local (within-pod) index -- see the construction
        expected_plane = {SwitchId("core", j * half + c) for c in range(half)}
        actual_neighbours = {n for lk in fab.neighbours(agg)
                             for n in (lk.dst,) if isinstance(n, SwitchId) and n.tier == "core"}
        assert actual_neighbours == expected_plane, (
            f"{agg} (local index {j}) reaches {sorted(actual_neighbours, key=lambda s: s.index)}, "
            f"expected exactly plane {j}: {sorted(expected_plane, key=lambda s: s.index)}")
        assert actual_neighbours <= all_core


def test_per_tier_oversubscription_is_independent():
    """Setting `oversubscription_edge_agg` must not change
    aggregation-to-core capacity, and setting `oversubscription_agg_core`
    must not change edge-to-aggregation capacity -- two independent
    parameters, not one ratio applied twice."""
    k = 4

    def _capacities(edge_agg_ratio, agg_core_ratio):
        fab = clos_fat_tree_fabric(switch_radix=k, depth=3, gpus_per_machine=1,
                                   nics_per_machine=1,
                                   oversubscription_edge_agg=edge_agg_ratio,
                                   oversubscription_agg_core=agg_core_ratio,
                                   **_COMMON_KWARGS)
        edge0 = SwitchId("edge", 0)
        agg0 = SwitchId("aggregation", 0)
        edge_agg_cap = next(lk.capacity_GBps for lk in fab.neighbours(edge0)
                           if isinstance(lk.dst, SwitchId) and lk.dst.tier == "aggregation")
        agg_core_cap = next(lk.capacity_GBps for lk in fab.neighbours(agg0)
                           if isinstance(lk.dst, SwitchId) and lk.dst.tier == "core")
        return edge_agg_cap, agg_core_cap

    baseline_edge_agg, baseline_agg_core = _capacities(1.0, 1.0)
    changed_edge_agg, unchanged_agg_core = _capacities(4.0, 1.0)
    unchanged_edge_agg, changed_agg_core = _capacities(1.0, 4.0)

    assert changed_edge_agg == pytest.approx(baseline_edge_agg / 4.0)
    assert unchanged_agg_core == pytest.approx(baseline_agg_core)  # untouched by the other ratio

    assert unchanged_edge_agg == pytest.approx(baseline_edge_agg)  # untouched by the other ratio
    assert changed_agg_core == pytest.approx(baseline_agg_core / 4.0)


def test_two_tier_unchanged():
    """depth=2 must produce exactly the fabric it did before this task --
    every study since task 03 depends on it. Checked two ways: the
    existing formula-based assertions still hold, and passing the two new
    depth=3-only oversubscription parameters (at non-default values) has
    no effect at all on a depth=2 fabric, proving they are not silently
    threaded into it."""
    k = 8
    half = k // 2
    baseline = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=2,
                                    nics_per_machine=1, oversubscription=2.0,
                                    **_COMMON_KWARGS)
    with_unused_params = clos_fat_tree_fabric(
        switch_radix=k, depth=2, gpus_per_machine=2, nics_per_machine=1,
        oversubscription=2.0, oversubscription_edge_agg=99.0, oversubscription_agg_core=99.0,
        **_COMMON_KWARGS)

    assert len(_switch_ids(baseline, "spine")) == half
    assert len(_switch_ids(baseline, "leaf")) == k
    assert len(baseline.machines) == k * half

    assert set(baseline.gpus) == set(with_unused_params.gpus)
    assert _class_counts(baseline) == _class_counts(with_unused_params)
    assert len(baseline.links) == len(with_unused_params.links)
    b_by_id = {lk.id: lk for lk in with_unused_params.links}
    for lk in baseline.links:
        other = b_by_id[lk.id]
        assert other.capacity_GBps == lk.capacity_GBps
        assert other.latency_ns == lk.latency_ns
