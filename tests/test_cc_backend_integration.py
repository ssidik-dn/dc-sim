"""Task 06: register this engine as a Frontier cc_backend.

The dominant risk here is not a raise -- it's a plausible wrong number (see
docs/tasks/06-frontier-cc-backend.md S1). So most of what follows is an
equivalence check against Frontier's own `analytical` backend, not a feature
test.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from frontier.cc_backend.backends.analytical_cc_backend import AnalyticalCCBackend
from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.types import ClusterType

from engine.cost.astra_backend import MockBackend
from engine.logical.deployment import (Deployment, ParallelKind, PoolKind,
                                       Rank, Replica)
from engine.network.transfers import Transfer, isolated_durations
from engine.physical.builders import build_node_scale
from engine.placement.placement import packed, spread

from integration.cc_backend.comm_groups import (CommGroupError,
                                                 CommGroupRegistry,
                                                 populate_from_deployment)
from integration.cc_backend.engine_backend import EngineCCBackend, _ns_to_ms
from integration.install.cc_backend import BACKEND_NAME, install

ENGINE_ROOT = Path(__file__).resolve().parents[1] / "src" / "engine"

# ---------------------------------------------------------------- registry


def test_registry_resolves_a_registered_triple():
    reg = CommGroupRegistry()
    ranks = [Rank("DECODE_ATTN", 0, 0), Rank("DECODE_ATTN", 0, 1)]
    reg.register("cluster-a", "TP", 2, ranks)
    assert reg.resolve("cluster-a", "TP", 2) == ranks


def test_registry_register_rejects_mismatched_count():
    reg = CommGroupRegistry()
    with pytest.raises(ValueError):
        reg.register("cluster-a", "TP", 4, [Rank("DECODE_ATTN", 0, 0)])


def test_registry_register_is_idempotent_for_the_same_ranks():
    reg = CommGroupRegistry()
    ranks = [Rank("DECODE_ATTN", 0, 0), Rank("DECODE_ATTN", 0, 1)]
    reg.register("cluster-a", "TP", 2, ranks)
    reg.register("cluster-a", "TP", 2, ranks)  # must not raise
    assert reg.resolve("cluster-a", "TP", 2) == ranks


def test_registry_register_raises_on_conflicting_ranks_for_same_triple():
    reg = CommGroupRegistry()
    reg.register("cluster-a", "TP", 2,
                 [Rank("DECODE_ATTN", 0, 0), Rank("DECODE_ATTN", 0, 1)])
    with pytest.raises(CommGroupError):
        reg.register("cluster-a", "TP", 2,
                     [Rank("DECODE_ATTN", 1, 0), Rank("DECODE_ATTN", 1, 1)])


def test_populate_from_deployment_registers_the_tp_group():
    d = Deployment("t")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=2))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: "decode-attn-cluster"})

    group = d.replicas[0].groups(ParallelKind.TP)[0]
    assert reg.resolve("decode-attn-cluster", "TP", 2) == group.ranks


def test_pool_resolves_the_single_registered_replica():
    reg = CommGroupRegistry()
    ranks = [Rank("PREFILL", 0, 0), Rank("PREFILL", 0, 1)]
    reg.register_pool("prefill-cluster", ranks)
    assert reg.resolve_pool("prefill-cluster") == ranks


def test_pool_raises_when_never_registered():
    reg = CommGroupRegistry()
    with pytest.raises(CommGroupError):
        reg.resolve_pool("nonexistent-cluster")


def test_pool_raises_on_a_second_replica():
    reg = CommGroupRegistry()
    reg.register_pool("decode-cluster", [Rank("DECODE", 0, 0)])
    reg.register_pool("decode-cluster", [Rank("DECODE", 1, 0)])
    with pytest.raises(CommGroupError, match="binding"):
        reg.resolve_pool("decode-cluster")


def test_pool_candidates_lists_every_registered_replica_with_its_id():
    reg = CommGroupRegistry()
    reg.register_pool("decode-cluster", [Rank("DECODE", 0, 0)], replica_id=7)
    reg.register_pool("decode-cluster", [Rank("DECODE", 1, 0)], replica_id=2)
    candidates = reg.resolve_pool_candidates("decode-cluster")
    assert candidates == [(7, [Rank("DECODE", 0, 0)]), (2, [Rank("DECODE", 1, 0)])]


def test_populate_from_deployment_registers_the_pool_too():
    d = Deployment("t")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=2))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: "decode-attn-cluster"})

    assert reg.resolve_pool("decode-attn-cluster") == d.replicas[0].ranks


# ---------------------------------------------------------------- unit conversion


def test_ns_to_ms_is_exact_for_whole_microseconds():
    """The one conversion point (S6: units, twice over). Both sides are
    floats, so there's no rounding direction to pick -- a whole number of
    microseconds should convert exactly."""
    assert _ns_to_ms(1_000_000) == 1.0
    assert _ns_to_ms(936_250) == pytest.approx(0.93625)
    # round trip: ns -> ms -> ns recovers the original within float epsilon.
    for ns in (1.0, 936.25, 5_000_000.0, 1_048_576 * 2.5e-9 * 1e6):
        assert _ns_to_ms(ns) * 1_000_000.0 == pytest.approx(ns, rel=1e-12)


# ---------------------------------------------------------------- EngineCCBackend


def _backend(fabric, deployment, placement_policy, cost_backend=None):
    placement = placement_policy(deployment, fabric)
    reg = CommGroupRegistry()
    populate_from_deployment(reg, deployment,
                             {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN})
    be = EngineCCBackend(fabric, placement, cost_backend or MockBackend(), reg)
    return be, placement, reg


def test_backend_subclasses_frontier_base():
    assert issubclass(EngineCCBackend, BaseCCBackend)
    assert EngineCCBackend.__abstractmethods__ == frozenset()
    for name in ("predict_allreduce", "predict_allgather", "predict_broadcast",
                 "predict_send_recv", "predict_reduce_scatter",
                 "predict_all_to_all"):
        assert not getattr(getattr(EngineCCBackend, name), "__isabstractmethod__", False)


def test_all_six_methods_return_milliseconds():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=8)
    d = Deployment("six")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    mock = MockBackend()
    be, placement, reg = _backend(fabric, d, packed, mock)
    size = 1 << 20
    kwargs = dict(cluster_type=ClusterType.DECODE_ATTN, comm_domain="TP")

    group = d.replicas[0].groups(ParallelKind.TP)[0]
    shape = placement.group_shape(group)

    # send_recv gets no num_devices from Frontier (see the task report) and
    # is always resolved as a 2-rank group; register that pair explicitly.
    p2p_pair = group.ranks[:2]
    reg.register(ClusterType.DECODE_ATTN, "PP", 2, p2p_pair)

    collectives = {
        "predict_allreduce": ("all_reduce", be.predict_allreduce(size, 8, **kwargs)),
        "predict_allgather": ("all_gather", be.predict_allgather(size, 8, **kwargs)),
        "predict_broadcast": ("broadcast", be.predict_broadcast(size, 8, **kwargs)),
        "predict_reduce_scatter": ("reduce_scatter",
                                  be.predict_reduce_scatter(size, 8, **kwargs)),
        "predict_all_to_all": ("all_to_all", be.predict_all_to_all(size, 8, **kwargs)),
    }
    for method, (op, ms) in collectives.items():
        assert isinstance(ms, float) and ms > 0, method
        expected_ns = mock.estimate(op, size, shape, fabric).duration_ns
        assert ms == pytest.approx(expected_ns / 1_000_000.0), method

    gpus = placement.gpus_for(p2p_pair)
    t = Transfer(key="send_recv", src=gpus[0], dst=gpus[1], size_bytes=size)
    expected_p2p_ns = isolated_durations(fabric, [t])[t.key]
    p2p_ms = be.predict_send_recv(size, cluster_type=ClusterType.DECODE_ATTN,
                                  comm_domain="PP")
    assert isinstance(p2p_ms, float) and p2p_ms > 0
    assert p2p_ms == pytest.approx(expected_p2p_ns / 1_000_000.0)


def test_packed_placement_matches_analytical_within_bound():
    """The binding test. For num_devices=2, Frontier's own ring all-reduce
    volume factor 2*(n-1)/n collapses to exactly 1, so a packed placement
    reduces both sides to the same "latency + size/bandwidth" formula --
    provided both are parameterised from the same physical scale-up link.
    That makes the tolerance tight (1e-6 relative): any drift beyond float
    rounding across the Gbps<->GBps and us<->ns<->ms conversions would mean a
    real unit bug, not two different models disagreeing. See the task 06
    report for what happens at num_devices > 2, where the two backends are
    NOT expected to agree -- this engine's default MockBackend cost path
    does not model Frontier's per-device ring-volume scaling, and widening
    this bound to paper over that would hide a real, reportable gap.
    """
    scale_up_GBps = 400.0
    scale_up_latency_ns = 936.25
    size = 1 << 20

    fabric = build_node_scale(num_machines=1, gpus_per_machine=2,
                              scale_up_GBps=scale_up_GBps,
                              scale_up_latency_ns=scale_up_latency_ns)
    d = Deployment("bind")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=2))
    mock = MockBackend(per_hop_ns=scale_up_latency_ns, per_byte_ns=1.0 / scale_up_GBps)
    be, _, _ = _backend(fabric, d, packed, mock)

    engine_ms = be.predict_allreduce(size, 2, cluster_type=ClusterType.DECODE_ATTN,
                                     comm_domain="TP")

    cfg = AnalyticalCCBackendConfig(network_latency_us=scale_up_latency_ns / 1000.0,
                                    intra_node_bandwidth_gbps=scale_up_GBps * 8.0)
    analytical = AnalyticalCCBackend(cfg, ClusterType.DECODE_ATTN, "h100", "nvlink", 2)
    frontier_ms = analytical.predict_allreduce(size, 2)

    TOLERANCE = 1e-6
    rel_diff = abs(engine_ms - frontier_ms) / frontier_ms
    assert rel_diff <= TOLERANCE, (
        f"engine={engine_ms}ms frontier={frontier_ms}ms rel_diff={rel_diff}")


def test_split_placement_costs_more_than_packed():
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = Deployment("split")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    kwargs = dict(cluster_type=ClusterType.DECODE_ATTN, comm_domain="TP")

    packed_be, _, _ = _backend(fabric, d, packed)
    spread_be, _, _ = _backend(fabric, d, spread)

    packed_ms = packed_be.predict_allreduce(1 << 20, 8, **kwargs)
    spread_ms = spread_be.predict_allreduce(1 << 20, 8, **kwargs)
    assert spread_ms > packed_ms


def test_unresolvable_comm_group_raises():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=8)
    d = Deployment("unresolved")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    be, _, _ = _backend(fabric, d, packed)

    with pytest.raises(CommGroupError):
        be.predict_allreduce(1 << 20, 8, cluster_type=ClusterType.PREFILL,
                             comm_domain="TP")  # never registered


def test_registration_is_idempotent():
    install()
    install()
    assert CCBackendFactory.get_class(BACKEND_NAME) is EngineCCBackend


def test_engine_has_no_frontier_import():
    """Programmatic assertion, not just the CI script (tools/check_import_direction.py)."""
    forbidden = ("integration", "frontier", "astra_sim", "upstream")
    violations = []
    for path in sorted(ENGINE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                names = [node.module] if node.module else []
            else:
                continue
            for mod in names:
                if mod.split(".")[0] in forbidden:
                    violations.append((path, mod))
    assert violations == []
