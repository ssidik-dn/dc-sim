"""Task 47: `SGLangStyleReplicaScheduler.__init__` refuses to construct for
any cluster type other than `MONOLITHIC`/`PREFILL` -- a guard, not a missing
capability. Read directly before touching anything (Task 47's own S2):

**The guard is not protecting incorrect behaviour.** Every cluster-type
branch inside the class's own added/overridden methods
(`_emit_schedule_decision_event`, `_schedule_two_phase`) reads `_cluster_type`
only to label a log line -- never to change what runs. The class's own new
logic (`_schedule_prefill_stage_first`, `_schedule_decode_fallback_running_requests`,
`_is_prefill_stage_request`) is cluster-type-blind, and calls only into its
parent's (`VLLMv1EngineReplicaScheduler`) own `_schedule_running_requests`/
`_schedule_waiting_requests`/`_create_batch` -- already exercised for
`DECODE_ATTN`/`DECODE_FFN` by every study this project has ever run. The two
inherited "monolithic_pp_terminal_release" helpers `_schedule_two_phase`
calls unconditionally are themselves gated on `_cluster_type ==
ClusterType.MONOLITHIC` internally (`_get_monolithic_pp_extra_terminal_release_iters`,
`vllm_v1_engine_replica_scheduler.py`) and no-op for every other cluster type
-- already true for the parent class's own callers, so nothing changes for
this subclass either.

**But it is only observable on `DECODE_FFN`.** `VLLMv1EngineReplicaScheduler`'s
own top-level dispatcher (`_get_next_batch`) routes `DECODE_ATTN` to
`_schedule_decode_attn_only` and `PREFILL` to `_schedule_prefill_only` --
neither overridden by `SGLangStyleReplicaScheduler` -- and only the `else`
branch (`MONOLITHIC` or anything else, which in this project's own
architecture means `DECODE_FFN`) reaches the overridden `_schedule_two_phase`.
So relaxing the guard for `DECODE_ATTN` costs nothing (matches Task 46's own
"identical is a result" -- but for a *structural* reason, the override never
running, not a contention-free workload) and buys nothing either: whatever
`--cluster_config_decode_attn_replica_scheduler_config_type` is asked for,
`DECODE_ATTN` behaves identically to plain `VLLMv1EngineReplicaScheduler`
once this scheduler is selected there, by construction. `DECODE_FFN` is the
only disaggregated cluster type where this patch can show anything at all.

**Guarded by a source hash**, exactly `..cc_backend.collective`'s own
pattern (task 20) -- the first and, until now, only instance of this idiom
in this project. An upstream change to `SGLangStyleReplicaScheduler.__init__`
this patch doesn't notice would silently stop enforcing whatever the new
guard protects; `install_sglang_replica_scheduler_guard()` refuses rather
than proceeding if the hash doesn't match what this module was written
against.
"""
from __future__ import annotations

import hashlib
import inspect

from frontier.scheduler.replica_scheduler.sglang_style_replica_scheduler import (
    SGLangStyleReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType

# Computed against the checked-out Frontier's own
# SGLangStyleReplicaScheduler.__init__
# (frontier/scheduler/replica_scheduler/sglang_style_replica_scheduler.py) at
# the time this module was written. A changed hash means __init__'s own body
# changed upstream -- install_sglang_replica_scheduler_guard() raises rather
# than patch over an unknown implementation.
_EXPECTED_SOURCE_HASH = "99fe57176a6359e60f55da65f10cd2a434169055ffb09bb4c5e8dbc5ff8e7755"

# The cluster types this project's own pd-af-disaggregation architecture
# actually constructs (task 25's paths, task 32 onward's own tools) --
# `DECODE` (plain pd-disaggregation, no attention/FFN split) and `TRANS` are
# deliberately excluded, since nothing here ever builds them and this task's
# own S2 asks to relax the guard "only for the cluster types this project
# uses," not for every type the enum happens to name.
_ALLOWED_CLUSTER_TYPES = (
    ClusterType.MONOLITHIC, ClusterType.PREFILL,
    ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN,
)

_installed = False


class SGLangGuardSourceMismatch(RuntimeError):
    pass


def _patched_init(self, *args, **kwargs) -> None:
    VLLMv1EngineReplicaScheduler.__init__(self, *args, **kwargs)
    if self._cluster_type not in _ALLOWED_CLUSTER_TYPES:
        raise ValueError(
            "SGLangStyleReplicaScheduler only supports MONOLITHIC, PREFILL, "
            "DECODE_ATTN, or DECODE_FFN cluster_type (task 47's own relaxed "
            f"guard), got {self._cluster_type!r}"
        )


def install_sglang_replica_scheduler_guard() -> None:
    """Relax `SGLangStyleReplicaScheduler.__init__`'s own cluster-type
    refusal to admit `DECODE_ATTN`/`DECODE_FFN` alongside the two it already
    allows (`MONOLITHIC`/`PREFILL`) -- nothing else about the class changes;
    every method beyond `__init__` is untouched. Safe to call more than once
    (idempotent). Not called by `install()` by default -- a caller must ask
    for it explicitly, the same way `collective=True` is never implied.

    Raises `SGLangGuardSourceMismatch` if `SGLangStyleReplicaScheduler.__init__`'s
    source no longer matches what this module was written against.
    """
    global _installed
    if _installed:
        return
    current_hash = hashlib.sha256(
        inspect.getsource(SGLangStyleReplicaScheduler.__init__).encode()).hexdigest()
    if current_hash != _EXPECTED_SOURCE_HASH:
        raise SGLangGuardSourceMismatch(
            f"SGLangStyleReplicaScheduler.__init__'s source has changed (hash "
            f"{current_hash} != expected {_EXPECTED_SOURCE_HASH}). Refusing "
            f"to install the relaxed-guard patch over an implementation this "
            f"project hasn't reviewed -- update _EXPECTED_SOURCE_HASH in "
            f"{__name__} only after confirming the guard being lifted is "
            f"still the same one, and that no new cluster-type branch was "
            f"added elsewhere in the class (task 47's own S2 check).")
    SGLangStyleReplicaScheduler.__init__ = _patched_init
    _installed = True
