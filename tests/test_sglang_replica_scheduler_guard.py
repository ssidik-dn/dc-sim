"""Task 47: `SGLangStyleReplicaScheduler.__init__` refuses to construct for
any cluster type other than `MONOLITHIC`/`PREFILL`. Task 47 established
(`src/integration/replica_scheduler/sglang_guard.py`'s own docstring) that
this guard protects nothing incorrect -- every other cluster-type check the
class touches is either a log-line label or already gated to a no-op
outside `MONOLITHIC` in the parent class -- so relaxing it to also admit
`DECODE_ATTN`/`DECODE_FFN` (the two additional cluster types this project's
own pd-af-disaggregation architecture uses) is safe.

The guard-check itself is tested in isolation, mocking out
`VLLMv1EngineReplicaScheduler.__init__` (already proven correct for every
cluster type by every real-compute study in this project) rather than
constructing a full, real scheduler -- this task's own change is exactly
the refusal check, nothing upstream of it.
"""
from __future__ import annotations

from unittest import mock

import pytest

from frontier.scheduler.replica_scheduler.sglang_style_replica_scheduler import (
    SGLangStyleReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType

from integration.replica_scheduler import sglang_guard
from integration.replica_scheduler.sglang_guard import (
    SGLangGuardSourceMismatch, install_sglang_replica_scheduler_guard,
)

_ORIGINAL_INIT = SGLangStyleReplicaScheduler.__init__


def _reset_guard_state():
    """Every test that installs the patch must leave both the module-level
    `_installed` flag and the class's own `__init__` attribute as it found
    them -- the same discipline `test_collective_backend.py` established
    for `CCBackendFactory.create`."""
    sglang_guard._installed = False
    SGLangStyleReplicaScheduler.__init__ = _ORIGINAL_INIT


@pytest.fixture(autouse=True)
def _isolate():
    _reset_guard_state()
    yield
    _reset_guard_state()


def _construct_with_cluster_type(init_fn, cluster_type: ClusterType) -> object:
    """Calls `init_fn` (the real or patched `__init__`) on a bare instance,
    with `VLLMv1EngineReplicaScheduler.__init__` mocked out to just record
    `cluster_type` -- isolates the guard check from the (already-proven)
    parent construction logic, matching this task's own "change nothing
    else" scope."""
    obj = SGLangStyleReplicaScheduler.__new__(SGLangStyleReplicaScheduler)
    with mock.patch.object(
            VLLMv1EngineReplicaScheduler, "__init__",
            lambda self, *a, **kw: setattr(self, "_cluster_type", cluster_type)):
        init_fn(obj)
    return obj


# ------------------------------------------------------- pre-patch behaviour


@pytest.mark.parametrize("cluster_type", [ClusterType.MONOLITHIC, ClusterType.PREFILL])
def test_unpatched_guard_admits_monolithic_and_prefill(cluster_type):
    obj = _construct_with_cluster_type(_ORIGINAL_INIT, cluster_type)
    assert obj._cluster_type == cluster_type


@pytest.mark.parametrize("cluster_type", [ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN])
def test_unpatched_guard_rejects_decode_attn_and_decode_ffn(cluster_type):
    """The behaviour Task 47 sets out to change -- confirmed present before
    any patch is installed."""
    with pytest.raises(ValueError, match="only supports MONOLITHIC or PREFILL"):
        _construct_with_cluster_type(_ORIGINAL_INIT, cluster_type)


# -------------------------------------------------------- patched behaviour


def test_install_relaxes_the_guard_for_decode_attn_and_decode_ffn():
    install_sglang_replica_scheduler_guard()
    assert SGLangStyleReplicaScheduler.__init__ is sglang_guard._patched_init

    for cluster_type in (ClusterType.MONOLITHIC, ClusterType.PREFILL,
                         ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN):
        obj = _construct_with_cluster_type(SGLangStyleReplicaScheduler.__init__, cluster_type)
        assert obj._cluster_type == cluster_type


@pytest.mark.parametrize("cluster_type", [ClusterType.DECODE, ClusterType.TRANS])
def test_install_does_not_admit_cluster_types_this_project_never_uses(cluster_type):
    """Task 47's own S2 scope: relax the guard "only for the cluster types
    this project uses" -- plain `DECODE` (pd-disaggregation, no AF split)
    and `TRANS` are not among them, and stay rejected even after the patch."""
    install_sglang_replica_scheduler_guard()
    with pytest.raises(ValueError):
        _construct_with_cluster_type(SGLangStyleReplicaScheduler.__init__, cluster_type)


def test_install_is_idempotent():
    install_sglang_replica_scheduler_guard()
    patched_once = SGLangStyleReplicaScheduler.__init__
    install_sglang_replica_scheduler_guard()
    assert SGLangStyleReplicaScheduler.__init__ is patched_once


# -------------------------------------------------------------- source hash


def test_source_hash_guard_fires():
    """A changed upstream SGLangStyleReplicaScheduler.__init__ halts install
    rather than lifting a refusal this module never reviewed -- task 47's
    own required acceptance test."""
    with mock.patch.object(sglang_guard, "_EXPECTED_SOURCE_HASH", "not-the-real-hash"):
        with pytest.raises(SGLangGuardSourceMismatch):
            install_sglang_replica_scheduler_guard()
    # And the real hash, unpatched, installs cleanly -- proving the guard
    # itself is what fired above, not something else broken.
    install_sglang_replica_scheduler_guard()
    assert SGLangStyleReplicaScheduler.__init__ is sglang_guard._patched_init
