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

from engine.logical.deployment import (Deployment, ParallelKind, PoolKind,
                                       Rank, Replica)
from engine.network.transfers import Transfer, run_transfers
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


def _backend(fabric, deployment, placement_policy):
    placement = placement_policy(deployment, fabric)
    reg = CommGroupRegistry()
    populate_from_deployment(reg, deployment,
                             {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN})
    be = EngineCCBackend(fabric, placement, reg)
    return be, placement, reg


def test_backend_subclasses_frontier_base():
    assert issubclass(EngineCCBackend, BaseCCBackend)
    assert EngineCCBackend.__abstractmethods__ == frozenset()
    for name in ("predict_allreduce", "predict_allgather", "predict_broadcast",
                 "predict_send_recv", "predict_reduce_scatter",
                 "predict_all_to_all"):
        assert not getattr(getattr(EngineCCBackend, name), "__isabstractmethod__", False)


def test_all_six_methods_return_milliseconds():
    """Task 20 rewrote the five true collectives onto `run_transfers`
    (task 06's original `CostBackend.estimate()` path is gone -- see
    `engine_backend.py`'s module docstring for why); this is now a basic
    sanity check (positive, finite milliseconds for every method) rather
    than an exact-match check against a mock model that no longer exists.
    The algorithm-specific claims (ring beats naive pairwise, packed beats
    split) are tested directly in tests/test_collective_backend.py."""
    fabric = build_node_scale(num_machines=1, gpus_per_machine=8)
    d = Deployment("six")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    be, placement, reg = _backend(fabric, d, packed)
    size = 1 << 20
    kwargs = dict(cluster_type=ClusterType.DECODE_ATTN, comm_domain="TP")

    group = d.replicas[0].groups(ParallelKind.TP)[0]

    # send_recv gets no num_devices from Frontier (see the task report) and
    # is always resolved as a 2-rank group; register that pair explicitly.
    p2p_pair = group.ranks[:2]
    reg.register(ClusterType.DECODE_ATTN, "PP", 2, p2p_pair)

    for method in ("predict_allreduce", "predict_allgather", "predict_broadcast",
                   "predict_reduce_scatter", "predict_all_to_all"):
        ms = getattr(be, method)(size, 8, **kwargs)
        assert isinstance(ms, float) and ms > 0, method

    gpus = placement.gpus_for(p2p_pair)
    t = Transfer(key="send_recv", src=gpus[0], dst=gpus[1], size_bytes=size)
    expected_p2p_ns = run_transfers(fabric, [t])[0].completion_ns
    p2p_ms = be.predict_send_recv(size, cluster_type=ClusterType.DECODE_ATTN,
                                  comm_domain="PP")
    assert isinstance(p2p_ms, float) and p2p_ms > 0
    assert p2p_ms == pytest.approx(expected_p2p_ns / 1_000_000.0)


def test_two_device_allreduce_matches_hand_ring_formula():
    """Task 06's original version of this test compared against Frontier's
    own analytical closed-form and required 1e-6 agreement, reasoning that
    at num_devices=2 the ring volume factor 2*(n-1)/n collapses to 1 so
    both sides reduce to "latency + size/bandwidth". Task 20's ring
    implementation is a genuine sequential ring -- 2*(n-1) = 2 discrete
    rounds, each paying the link's fixed latency separately, not one
    amortised latency term for the whole operation -- and that changes the
    n=2 case too: two sequential rounds pay the link latency *twice*, so
    this engine's own number is Frontier's closed-form plus one extra
    latency term, not equal to it. Confirmed by running it (engine
    exceeded frontier's number by ~26%, not float noise) rather than
    assumed, and the discrepancy is explained exactly by that one term
    below -- this is a real difference between a step-by-step ring and a
    single amortised formula, not a bug in either.
    """
    scale_up_GBps = 400.0
    scale_up_latency_ns = 936.25
    size = 1 << 20

    fabric = build_node_scale(num_machines=1, gpus_per_machine=2,
                              scale_up_GBps=scale_up_GBps,
                              scale_up_latency_ns=scale_up_latency_ns)
    d = Deployment("bind")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=2))
    be, _, _ = _backend(fabric, d, packed)

    engine_ms = be.predict_allreduce(size, 2, cluster_type=ClusterType.DECODE_ATTN,
                                     comm_domain="TP")

    # Hand ring formula: 2*(n-1)=2 rounds, each moving size/n=524288 bytes
    # over the one scale-up link, each round paying that link's own
    # latency separately (a real sequential ring's rounds are not
    # pipelined against each other in this model).
    rounds = 2
    chunk_bytes = size // 2
    round_ns = scale_up_latency_ns + chunk_bytes / scale_up_GBps
    expected_ms = (rounds * round_ns) / 1_000_000.0

    # 1e-3, not 1e-6: the fabric model works in integer nanoseconds
    # (`Completion.completion_ns`), so each of the 2 rounds can round up by
    # up to 1ns against this float hand formula -- a few ns against a
    # ~4500ns total is real integer rounding, not a modelling error.
    TOLERANCE = 1e-3
    rel_diff = abs(engine_ms - expected_ms) / expected_ms
    assert rel_diff <= TOLERANCE, (
        f"engine={engine_ms}ms expected={expected_ms}ms rel_diff={rel_diff}")

    # And the documented, expected divergence from Frontier's own
    # single-latency-term closed-form: approximately one extra link
    # latency (a loose bound -- integer-ns rounding inside the fabric
    # model, ~2ns here, is not the point of this assertion).
    cfg = AnalyticalCCBackendConfig(network_latency_us=scale_up_latency_ns / 1000.0,
                                    intra_node_bandwidth_gbps=scale_up_GBps * 8.0)
    analytical = AnalyticalCCBackend(cfg, ClusterType.DECODE_ATTN, "h100", "nvlink", 2)
    frontier_ms = analytical.predict_allreduce(size, 2)
    extra_latency_ms = scale_up_latency_ns / 1_000_000.0
    assert engine_ms == pytest.approx(frontier_ms + extra_latency_ms, rel=1e-3)


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
