"""Task 20: close the blind spot task 18/19 measured -- make tensor-,
pipeline-, and expert-parallel communication topology-aware by reaching
`EngineCCBackend` for real, through a guarded runtime replacement of
`CCBackendFactory.create` rather than a CLI flag (there is no free
`CCBackendType` member to claim, confirmed twice: task 06, and again in
`integration/cc_backend/collective.py`'s own module docstring).

Tests 1-3 exercise the interception itself (selected when configured,
guarded by a source hash, inert otherwise); tests 4-6 exercise
`EngineCCBackend`'s own correctness -- group membership, the central
packed-vs-split claim, and refusing rather than guessing.
"""
from __future__ import annotations

from unittest import mock

import pytest

from frontier.cc_backend.backends.analytical_cc_backend import AnalyticalCCBackend
from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.types import CCBackendType, ClusterType

from engine.logical.deployment import Deployment, ParallelKind, PoolKind, Replica
from engine.physical.builders import build_node_scale
from engine.placement.placement import packed, spread

from integration.cc_backend import collective as collective_module
from integration.cc_backend.collective import (
    CollectiveBackendSourceMismatch, install_collective_backend)
from integration.cc_backend.comm_groups import CommGroupError, CommGroupRegistry, populate_from_deployment
from integration.cc_backend.engine_backend import EngineCCBackend
from integration.context import EngineContext, set_context


def _reset_interception_state():
    """Every test in this file that touches the module-level `_installed`
    flag must leave it as it found it -- otherwise a later test would see
    `CCBackendFactory.create` already patched (or, for the mismatch test,
    permanently un-patchable) regardless of what it itself configures.
    Task 11's own replica-id-leak precedent, restated for a different kind
    of module-global state."""
    collective_module._installed = False
    CCBackendFactory.create = classmethod(collective_module._original_create)


@pytest.fixture(autouse=True)
def _isolate():
    _reset_interception_state()
    set_context(None)
    yield
    _reset_interception_state()
    set_context(None)


def _context_for(fabric, deployment, placement_policy, collective: bool):
    placement = placement_policy(deployment, fabric)
    reg = CommGroupRegistry()
    populate_from_deployment(reg, deployment, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN})
    ctx = EngineContext(fabric, placement, deployment, reg, collective=collective)
    set_context(ctx)
    return ctx, placement


# --------------------------------------------------------------- selection


def test_backend_is_selected_by_flag():
    """Registration is not selection (task 06/14/15's own trap, twice
    already). Proven with a call through the *same* path Frontier's own
    execution-time predictors use -- CCBackendFactory.create() -- not by
    constructing EngineCCBackend directly."""
    fabric = build_node_scale(num_machines=1, gpus_per_machine=8)
    d = Deployment("select")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    _context_for(fabric, d, packed, collective=True)
    install_collective_backend()

    cfg = AnalyticalCCBackendConfig()
    backend = CCBackendFactory.create(
        backend_type=CCBackendType.ANALYTICAL, config=cfg,
        cluster_type=ClusterType.DECODE_ATTN, device_type="h800",
        network_device="h800_nvlink", num_devices=8)
    assert isinstance(backend, EngineCCBackend)

    with mock.patch.object(EngineCCBackend, "predict_allreduce",
                          wraps=backend.predict_allreduce) as spy:
        backend.predict_allreduce(1 << 20, 8, cluster_type=ClusterType.DECODE_ATTN,
                                  comm_domain="TP")
        assert spy.called


def test_source_hash_guard_fires():
    """A changed upstream CCBackendFactory.create halts install rather
    than diverging silently."""
    with mock.patch.object(
            collective_module, "_EXPECTED_SOURCE_HASH", "not-the-real-hash"):
        with pytest.raises(CollectiveBackendSourceMismatch):
            install_collective_backend()
    # And the real hash, unpatched, installs cleanly -- proving the guard
    # itself is the thing that fired above, not something else broken.
    install_collective_backend()


def test_other_backend_values_unaffected():
    """Without `EngineContext.collective` (every run before this task, and
    every run that doesn't opt in), selecting `analytical` behaves exactly
    as it did before the interception -- construction goes through
    unpatched, and the object really is Frontier's own AnalyticalCCBackend."""
    install_collective_backend()  # the interception exists in the process...
    set_context(None)             # ...but nothing configured it.

    cfg = AnalyticalCCBackendConfig()
    backend = CCBackendFactory.create(
        backend_type=CCBackendType.ANALYTICAL, config=cfg,
        cluster_type=ClusterType.DECODE_ATTN, device_type="h800",
        network_device="h800_nvlink", num_devices=8)
    assert isinstance(backend, AnalyticalCCBackend)
    assert not isinstance(backend, EngineCCBackend)


# ------------------------------------------------------- group membership


def test_tp_group_membership_matches_logical_model():
    """S3.2's own warning: wrong group membership gives plausible, wrong
    numbers. Resolve a split TP group's participants through
    EngineCCBackend's own path and check they are exactly the ranks
    `Replica.groups(ParallelKind.TP)` says they are -- not merely "some
    four ranks"."""
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = Deployment("membership")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    ctx, placement = _context_for(fabric, d, spread, collective=False)

    expected_group = d.replicas[0].groups(ParallelKind.TP)[0]
    resolved = ctx.groups.resolve(ClusterType.DECODE_ATTN, "TP", 8)
    assert resolved == expected_group.ranks

    be = EngineCCBackend(ctx.fabric, ctx.placement, ctx.groups)
    ordered = be._ring_order(resolved)
    assert sorted(ordered) == sorted(expected_group.ranks)
    assert set(ordered) == set(expected_group.ranks)
    # Domain-major: consecutive ranks in the ring order share a domain
    # until the boundary, confirming the ordering used for pricing is
    # actually derived from placement, not just a pass-through of
    # whatever order resolve() happened to return.
    domains = [fabric.domain_of(placement.gpu(r)) for r in ordered]
    assert domains == sorted(domains)


# ----------------------------------------------------------- central claim


def test_packed_tp_group_cheaper_than_split():
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8)
    d = Deployment("central-claim")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    kwargs = dict(cluster_type=ClusterType.DECODE_ATTN, comm_domain="TP")

    ctx_packed, _ = _context_for(fabric, d, packed, collective=False)
    packed_be = EngineCCBackend(ctx_packed.fabric, ctx_packed.placement, ctx_packed.groups)
    packed_ms = packed_be.predict_allreduce(1 << 20, 8, **kwargs)

    d2 = Deployment("central-claim-2")
    d2.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    ctx_split, _ = _context_for(fabric, d2, spread, collective=False)
    split_be = EngineCCBackend(ctx_split.fabric, ctx_split.placement, ctx_split.groups)
    split_ms = split_be.predict_allreduce(1 << 20, 8, **kwargs)

    assert packed_ms < split_ms


# ------------------------------------------------------------- no guessing


def test_unresolvable_domain_raises():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=8)
    d = Deployment("unresolved")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=8))
    ctx, _ = _context_for(fabric, d, packed, collective=False)
    be = EngineCCBackend(ctx.fabric, ctx.placement, ctx.groups)

    with pytest.raises(CommGroupError):
        be.predict_allreduce(1 << 20, 8, cluster_type=ClusterType.PREFILL,
                             comm_domain="TP")  # never registered -- no fallback
