"""Tests for FabricMode routing decisions (task 04).

This is the direct fix for the Task 03 finding: Fabric.path()'s BFS always
resolves the same pair of leaves to the same intermediate spine, so
oversubscription pressure never reflects the other k/2 - 1 spines that
exist. `test_ecmp_disperses_across_many_flows` is the binding test for that.

Scope note (see docs/tasks/04-fabric-mode.md §2): these tests cover routing
*decisions* only -- which link(s) a flow would use. Nothing here executes a
flow or touches completion time; that boundary is why `route()` raises for
SPRAYED instead of guessing at multi-leg completion semantics.
"""
from __future__ import annotations

import pytest

from engine.infragraph.blueprints import clos_fat_tree_fabric
from engine.physical.topology import FabricMode, GpuId, SwitchId

_COMMON_KWARGS = dict(
    scale_up_GBps=400.0, scale_up_latency_ns=936.25,
    nic_gbps=400.0, egress_latency_ns=2000.0,
    scale_out_GBps=50.0, scale_out_latency_ns=5000.0,
)


def _leaf_spine(k, **kwargs):
    return clos_fat_tree_fabric(switch_radix=k, depth=2, gpus_per_machine=1,
                                nics_per_machine=1, **{**_COMMON_KWARGS, **kwargs})


def _spines_touched(path):
    return {n for lk in path for n in (lk.src, lk.dst)
            if isinstance(n, SwitchId) and n.tier == "spine"}


# ------------------------------------------------------------- single_path

def test_single_path_mode_matches_path():
    fab = _leaf_spine(8)
    half = 4
    pairs = [(GpuId(0, 0), GpuId(1, 0)),      # same leaf
             (GpuId(0, 0), GpuId(half, 0)),   # cross leaf
             (GpuId(0, 0), GpuId(31, 0))]     # furthest leaf
    for a, b in pairs:
        assert fab.route(FabricMode.SINGLE_PATH, "k", a, b) == fab.path(a, b)


# --------------------------------------------------------- equal_cost_paths

def test_equal_cost_paths_count_matches_spine_count():
    k = 8
    half = k // 2
    fab = _leaf_spine(k)
    paths = fab.equal_cost_paths(GpuId(0, 0), GpuId(half, 0))
    assert len(paths) == half

    touched_spines = set()
    for p in paths:
        spines = _spines_touched(p)
        assert len(spines) == 1, "each equal-cost path should cross exactly one spine"
        touched_spines |= spines
    assert len(touched_spines) == half, "every path should go through a different spine"


def test_equal_cost_paths_is_singular_when_unique():
    fab = _leaf_spine(4)
    a, b = GpuId(0, 0), GpuId(1, 0)  # same leaf: only one shortest path
    paths = fab.equal_cost_paths(a, b)
    assert len(paths) == 1
    assert paths[0] == fab.path(a, b)


def test_equal_cost_paths_same_gpu_matches_path_convention():
    fab = _leaf_spine(4)
    g = GpuId(0, 0)
    assert fab.equal_cost_paths(g, g) == [[]]
    assert fab.path(g, g) == []


# ------------------------------------------------------------- per_flow_ecmp

def test_ecmp_route_is_one_valid_equal_cost_path():
    k = 8
    half = k // 2
    fab = _leaf_spine(k)
    a, b = GpuId(0, 0), GpuId(half, 0)
    valid = fab.equal_cost_paths(a, b)
    route = fab.route(FabricMode.PER_FLOW_ECMP, "flow-42", a, b)
    assert route in valid


def test_ecmp_is_deterministic():
    k = 8
    half = k // 2
    fab = _leaf_spine(k)
    a, b = GpuId(0, 0), GpuId(half, 0)
    first = fab.route(FabricMode.PER_FLOW_ECMP, "flow-7", a, b)
    for _ in range(5):
        assert fab.route(FabricMode.PER_FLOW_ECMP, "flow-7", a, b) == first

    # A fresh Fabric built the same way must agree too -- determinism means
    # reproducible across runs, not just within one object's lifetime.
    fab2 = _leaf_spine(k)
    assert fab2.route(FabricMode.PER_FLOW_ECMP, "flow-7", a, b) == first


def test_ecmp_disperses_across_many_flows():
    """The binding test. Task 03 measured every flow between a given pair of
    leaves collapsing onto the same spine under plain BFS. Many distinct
    flow keys between the same host pair must now collectively use more
    than one spine."""
    k = 8
    half = k // 2
    fab = _leaf_spine(k)
    a, b = GpuId(0, 0), GpuId(half, 0)

    touched = set()
    for i in range(50):
        route = fab.route(FabricMode.PER_FLOW_ECMP, f"transfer-{i}", a, b)
        touched |= _spines_touched(route)

    assert len(touched) > 1, (
        f"50 distinct flows all landed on {touched} -- ECMP isn't dispersing")


# ---------------------------------------------------------------- sprayed

def test_ecmp_route_raises_for_sprayed_mode():
    fab = _leaf_spine(4)
    a, b = GpuId(0, 0), GpuId(2, 0)
    with pytest.raises(NotImplementedError):
        fab.route(FabricMode.SPRAYED, "flow-1", a, b)


def test_spray_routes_covers_every_equal_cost_path():
    k = 8
    half = k // 2
    fab = _leaf_spine(k)
    a, b = GpuId(0, 0), GpuId(half, 0)
    spread = fab.spray_routes(a, b)
    assert [p for p, _ in spread] == fab.equal_cost_paths(a, b)
    assert len(spread) == half


def test_spray_routes_fractions_sum_to_one():
    fab = _leaf_spine(8)
    a, b = GpuId(0, 0), GpuId(4, 0)
    spread = fab.spray_routes(a, b)
    assert sum(frac for _, frac in spread) == pytest.approx(1.0)


def test_spray_routes_is_even_split():
    k = 8
    half = k // 2
    fab = _leaf_spine(k)
    a, b = GpuId(0, 0), GpuId(half, 0)
    spread = fab.spray_routes(a, b)
    for _, frac in spread:
        assert frac == pytest.approx(1.0 / half)

    # Same-leaf pair: a single path should get the whole flow.
    same_leaf = fab.spray_routes(GpuId(0, 0), GpuId(1, 0))
    assert len(same_leaf) == 1
    assert same_leaf[0][1] == pytest.approx(1.0)
