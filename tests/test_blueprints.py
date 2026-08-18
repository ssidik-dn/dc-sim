"""Tests for the InfraGraph blueprints (task 02 part B).

`test_oversubscription_shows_up_in_contention` is the binding test: a
blueprint parameter that doesn't move measured cost is worse than useless.
"""
from __future__ import annotations

import pytest

from engine.infragraph.blueprints import clos_fat_tree_fabric, single_tier_fabric
from engine.infragraph.emit import to_infragraph
from engine.infragraph.parse import from_infragraph
from engine.network.transfers import Transfer, analyse
from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId, LinkClass, SwitchId


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


def _pod_and_leaf_pos(mid, half):
    """Inverse of clos_fat_tree_fabric's machine-id assignment order: pods
    outermost, then leaf position, then host -- see the blueprint."""
    pod_size = half * half
    pod, within_pod = divmod(mid, pod_size)
    leaf_pos = within_pod // half
    return pod, leaf_pos


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


# --------------------------------------------------------------------- clos

@pytest.mark.parametrize("k", [4, 6, 8])
def test_clos_host_count_matches_formula(k):
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=2,
                               nics_per_machine=1, **_COMMON_KWARGS)
    assert len(fab.machines) == k ** 3 // 4
    assert len(fab.gpus) == 2 * (k ** 3 // 4)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_clos_switch_counts_match_formula(k):
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    half = k // 2
    assert len(_switch_ids(fab, "leaf")) == k * half
    assert len(_switch_ids(fab, "spine")) == half * half


def test_clos_every_host_reaches_every_other():
    k = 4
    half = k // 2
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)
    mids = sorted(fab.machines)
    pairs = []
    for a in mids:
        for b in mids:
            if a >= b:
                continue
            pod_a, _ = _pod_and_leaf_pos(a, half)
            pod_b, _ = _pod_and_leaf_pos(b, half)
            if pod_a != pod_b:
                pairs.append((a, b))
    # Include both same-leaf-position and different-leaf-position cross-pod
    # pairs -- the harder case relies on bridging through the intra-pod
    # leaf mesh on both ends (see the blueprint's design note).
    sample = pairs[:3] + pairs[-3:]
    assert sample
    for a, b in sample:
        path = fab.path(GpuId(a, 0), GpuId(b, 0))
        assert path, f"no path between machine {a} and machine {b}"


def test_clos_cross_pod_path_traverses_spine():
    k = 4
    half = k // 2
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=1, **_COMMON_KWARGS)

    # Two hosts in the same pod, different leaves (positions 0 and 1):
    # direct intra-pod leaf-to-leaf link, no spine.
    within_pod_path = fab.path(GpuId(0, 0), GpuId(half, 0))
    assert not any(isinstance(lk.dst, SwitchId) and lk.dst.tier == "spine"
                  for lk in within_pod_path)
    assert not any(isinstance(lk.src, SwitchId) and lk.src.tier == "spine"
                  for lk in within_pod_path)

    # Two hosts in different pods, same leaf position: must cross a spine.
    pod_size = half * half
    cross_pod_path = fab.path(GpuId(0, 0), GpuId(pod_size, 0))
    assert any((isinstance(lk.src, SwitchId) and lk.src.tier == "spine")
              or (isinstance(lk.dst, SwitchId) and lk.dst.tier == "spine")
              for lk in cross_pod_path)


def test_oversubscription_reduces_uplink_capacity():
    k = 4
    half = k // 2
    ratio = 4.0
    fab = clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                               nics_per_machine=2, oversubscription=ratio,
                               **_COMMON_KWARGS)
    leaf = SwitchId("leaf", 0)  # pod 0, leaf position 0
    uplink_total = sum(lk.capacity_GBps for lk in fab.neighbours(leaf)
                       if isinstance(lk.dst, SwitchId) and lk.dst.tier == "spine")
    downlink_total = sum(lk.capacity_GBps for lk in fab.neighbours(leaf)
                         if not isinstance(lk.dst, SwitchId))
    assert uplink_total == pytest.approx(downlink_total / ratio)


def test_oversubscription_shows_up_in_contention():
    """The binding test. Same fabric shape at 1:1 and 4:1 oversubscription;
    enough concurrent cross-pod transfers to saturate the leaf-spine
    uplinks. The 4:1 fabric must show a strictly longer makespan -- if it
    didn't, `oversubscription` would be a parameter that does nothing."""
    k = 4
    half = k // 2
    pod_size = half * half
    size_bytes = 5_000_000

    transfers = [
        Transfer("a", GpuId(0, 0), GpuId(pod_size, 0), size_bytes),
        Transfer("b", GpuId(1, 0), GpuId(pod_size + 1, 0), size_bytes),
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


def test_depth_three_raises():
    with pytest.raises(NotImplementedError):
        clos_fat_tree_fabric(switch_radix=4, depth=3, gpus_per_machine=1,
                             nics_per_machine=1, **_COMMON_KWARGS)
