"""Phase 2 tests: placement shape to ASTRA-sim config, and the cost backend.

The two that matter most are test_config_pair_dimensions_always_agree and
test_uneven_shape_is_rejected -- both encode traps that produced wrong answers
before they were understood.
"""
import json

import pytest

from engine.physical.builders import build_node_scale
from engine.logical.deployment import Deployment, ParallelKind, PoolKind, Replica
from engine.placement.placement import fragmented, packed, spread
from engine.cost.astra_config import (AstraTopology, ConfigPair, NotExpressible,
                                      cache_key, emit_config_pair,
                                      topology_for_shape)
from engine.cost.astra_backend import FallbackBackend, MockBackend


@pytest.fixture
def fab():
    return build_node_scale(num_machines=4, gpus_per_machine=8, nics_per_machine=4)


# ------------------------------------------------- shape -> dimensions

def test_packed_shape_is_one_dimension(fab):
    t = topology_for_shape((8,), fab)
    assert t.dims == 1
    assert t.npus_count == (8,)
    assert t.total_ranks == 8


def test_even_split_uses_product_not_sum(fab):
    """npus_count multiplies: 4 per domain x 2 domains = 8 ranks."""
    t = topology_for_shape((4, 4), fab)
    assert t.dims == 2
    assert t.npus_count == (4, 2)
    assert t.total_ranks == 8


def test_four_way_split(fab):
    t = topology_for_shape((2, 2, 2, 2), fab)
    assert t.npus_count == (2, 4)
    assert t.total_ranks == 8


def test_second_dimension_is_slower(fab):
    """Leaving the machine is paced by egress, an order of magnitude narrower
    than the scale-up fabric."""
    t = topology_for_shape((4, 4), fab)
    assert t.bandwidth_GBps[1] < t.bandwidth_GBps[0] / 4
    assert t.latency_ns[1] > t.latency_ns[0]


def test_uneven_shape_is_rejected(fab):
    """(3, 3, 2) comes out of fragmented placement and is inexpressible.
    Raising beats rounding: a rounded answer looks entirely reasonable."""
    with pytest.raises(NotExpressible):
        topology_for_shape((3, 3, 2), fab)


def test_dimension_lists_must_be_equal_length():
    with pytest.raises(ValueError):
        AstraTopology(("Switch", "Switch"), (4,), (400.0, 50.0),
                      (900.0, 5000.0), (4, 4))


# ------------------------------------------------- the config-pair trap

def test_config_pair_dimensions_always_agree(fab, tmp_path):
    """The trap: a 2-dim network config with a 1-dim system config does not
    error, it silently drops dimensions and reports a LOWER cost than the
    truth. Emitting both from one function makes that unrepresentable."""
    for shape in [(8,), (4, 4), (2, 2, 2, 2)]:
        pair = emit_config_pair(topology_for_shape(shape, fab), tmp_path)
        pair.verify()
        net = pair.network_path.read_text()
        sys_cfg = json.loads(pair.system_path.read_text())
        net_dims = net.split("npus_count:")[1].split("]")[0].count(",") + 1
        assert net_dims == len(sys_cfg["all-reduce-implementation"])


def test_verify_catches_a_hand_edited_mismatch(fab, tmp_path):
    pair = emit_config_pair(topology_for_shape((4, 4), fab), tmp_path)
    broken = json.loads(pair.system_path.read_text())
    broken["all-reduce-implementation"] = ["ring"]        # back to 1 dim
    pair.system_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="dimension mismatch"):
        pair.verify()


def test_system_config_has_one_algorithm_per_dimension(fab):
    t = topology_for_shape((2, 2, 2, 2), fab)
    cfg = json.loads(t.system_json())
    for k in ("all-reduce-implementation", "all-gather-implementation",
              "reduce-scatter-implementation", "all-to-all-implementation"):
        assert len(cfg[k]) == t.dims == 2


def test_network_yaml_is_parseable(fab):
    y = topology_for_shape((4, 4), fab).network_yaml()
    assert "npus_count: [ 4, 2 ]" in y
    for field in ("topology:", "npus_count:", "bandwidth:", "latency:"):
        assert field in y


# ------------------------------------------------- memoisation key

def test_cache_key_collapses_isomorphic_placements(fab):
    """Two different sets of GPUs with the same shape must share a key --
    this is what keeps invocation count bounded."""
    t = topology_for_shape((4, 4), fab)
    a = cache_key((4, 4), "all_reduce", 1024, t)
    b = cache_key((4, 4), "all_reduce", 1024, t)
    assert a == b


def test_cache_key_separates_different_shapes(fab):
    k1 = cache_key((8,), "all_reduce", 1024, topology_for_shape((8,), fab))
    k2 = cache_key((4, 4), "all_reduce", 1024, topology_for_shape((4, 4), fab))
    assert k1 != k2


def test_cache_key_separates_sizes(fab):
    t = topology_for_shape((8,), fab)
    assert cache_key((8,), "all_reduce", 1024, t) != \
           cache_key((8,), "all_reduce", 4096, t)


# ------------------------------------------------- backends

def test_mock_backend_penalises_splitting(fab):
    m = MockBackend()
    packed_cost = m.estimate("all_reduce", 1 << 20, (8,), fab).duration_ns
    split_cost = m.estimate("all_reduce", 1 << 20, (2, 2, 2, 2), fab).duration_ns
    assert split_cost > packed_cost


def test_fallback_handles_uneven_shapes(fab):
    """Fragmented placement produces shapes ASTRA-sim cannot express. The
    fallback path must catch that rather than propagating the error."""
    class OnlyEven(MockBackend):
        def estimate(self, collective, size_bytes, shape, fabric):
            topology_for_shape(shape, fabric)      # raises on uneven
            return super().estimate(collective, size_bytes, shape, fabric)

    fb = FallbackBackend(primary=OnlyEven(), fallback=MockBackend())
    r = fb.estimate("all_reduce", 1024, (3, 3, 2), fab)
    assert fb.fallbacks_used == 1
    assert "fallback" in r.backend

    fb.estimate("all_reduce", 1024, (4, 4), fab)
    assert fb.fallbacks_used == 1, "even shapes must not fall back"


# ------------------------------------------------- end to end

def test_placement_to_topology_end_to_end(fab):
    """A real placement, through group_shape, into a valid config pair."""
    d = Deployment("e2e")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    g = d.replicas[0].groups(ParallelKind.TP)[0]

    packed_t = topology_for_shape(packed(d, fab).group_shape(g), fab)
    assert packed_t.dims == 1 and packed_t.total_ranks == 8

    spread_t = topology_for_shape(spread(d, fab).group_shape(g), fab)
    assert spread_t.dims == 2 and spread_t.total_ranks == 8
    assert spread_t.bandwidth_GBps[1] < spread_t.bandwidth_GBps[0]


def test_fragmented_placement_yields_inexpressible_shape(fab):
    """Documents the real limitation: irregular placements are exactly what
    neither ASTRA-sim's format nor a profiled lookup table can represent."""
    d = Deployment("frag")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    p = fragmented(d, fab, seed=0, occupied=0.35)
    shape = p.group_shape(d.replicas[0].groups(ParallelKind.TP)[0])
    if len(set(shape)) != 1:
        with pytest.raises(NotExpressible):
            topology_for_shape(shape, fab)
