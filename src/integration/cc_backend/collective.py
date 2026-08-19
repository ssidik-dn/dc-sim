"""Task 20: reach `EngineCCBackend` for real, by replacing
`CCBackendFactory.create` at runtime rather than registering a backend
under a name nothing can select.

`CCBackendType` has five members, all claimed (task 06); `BaseRegistry.register`
no-ops on a claimed key; CLI selection validates against the enum member's
own name, so there is no sixth name to claim (confirmed twice -- task 06,
and again here: `AiconfiguratorCCBackendConfig`'s `__include_in_cli__ =
False` blocks even the one nominally-unclaimed member,
`CCBackendType.AICONFIGURATOR`, from being constructed through Frontier's
own CLI path at all -- a `KeyError` inside `reconstruct_original_dataclass`,
confirmed by running it, not a hypothetical). Selection here works the
same way task 14's binding and task 15's topology-aware scheduler do:
through this project's own `install()` call, not through any Frontier CLI
flag. `install(..., collective=True)` is the flag; every run that doesn't
pass it is untouched, regardless of which `--cc_backend_config_type` value
its own argv used.

**The narrowest interception found**: `CCBackendFactory.create` is the one
function every one of the six `predict_*` calls' backend construction
passes through, in both the dense and mixture-of-experts execution-time
predictors (confirmed by reading both -- `sklearn_execution_time_predictor.py`
and `sklearn_moe_execution_time_predictor.py` both hold `self._cc_backend`,
obtained once, from exactly this factory method, at cluster construction
time). Replacing it here means every one of those calls -- allreduce,
allgather, broadcast, reduce_scatter, all_to_all, send_recv, across TP, PP,
and EP -- reaches `EngineCCBackend` uniformly, without needing five
separate interceptions (`create_from_str`, `is_registered`, and the two
CLI-layer closures task 06 already found closed, are all left alone).

**Guarded by a source hash.** Frontier's disaggregated paths are
pre-release (task 12/17's own framing), and this is the most invasive
interception in this project -- an upstream change to `create()`'s own
body that this patch doesn't notice would make every collective silently
diverge from whatever the new code actually does, while looking identical
to a passing test suite. `install_collective_backend()` refuses rather
than proceeding if the hash doesn't match what this module was written
against. No prior instance of this exact pattern exists elsewhere in this
project to point to (`install()`'s own docstring describes the general
"runtime replacement, from install()" idiom task 14/15 already used, but
neither of those guarded with a source hash) -- this is the first, because
this is the first interception invasive enough to warrant one.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Optional

from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.cc_backend.cc_backend_config import BaseCCBackendConfig
from frontier.types import ClusterType

from .engine_backend import EngineCCBackend
from ..context import get_context

# Computed against the checked-out Frontier's CCBackendFactory.create
# (frontier/cc_backend/cc_backend_factory.py) at the time this module was
# written. A changed hash means create()'s own body changed upstream --
# install_collective_backend() raises rather than patch over an unknown
# implementation.
_EXPECTED_SOURCE_HASH = "4b030f9a121aa8d07b2064dcdbc9dd6fd7558394c07e688a7412260bc373ee26"

_original_create = CCBackendFactory.create.__func__
_installed = False


class CollectiveBackendSourceMismatch(RuntimeError):
    pass


def _patched_create(
    cls,
    backend_type,
    config: "BaseCCBackendConfig",
    cluster_type: "ClusterType",
    device_type: str,
    network_device: str,
    num_devices: int,
):
    ctx = get_context()
    if ctx is not None and ctx.collective:
        return EngineCCBackend(
            ctx.fabric, ctx.placement, ctx.groups,
            config=config, cluster_type=cluster_type, device_type=device_type,
            network_device=network_device, num_devices=num_devices)
    return _original_create(cls, backend_type, config, cluster_type,
                            device_type, network_device, num_devices)


def install_collective_backend() -> None:
    """Patch `CCBackendFactory.create` so that a run with
    `EngineContext.collective` set (via `install(..., collective=True)`)
    gets `EngineCCBackend` for every cc_backend construction, regardless of
    which `--cc_backend_config_type` value its own argv used. Safe to call
    more than once (idempotent -- re-patching onto an already-patched
    method would re-wrap it, so this checks first).

    Raises `CollectiveBackendSourceMismatch` if `CCBackendFactory.create`'s
    source no longer matches what this module was written against, rather
    than patch over a changed implementation silently.
    """
    global _installed
    if _installed:
        return
    current_hash = hashlib.sha256(
        inspect.getsource(CCBackendFactory.create.__func__).encode()).hexdigest()
    if current_hash != _EXPECTED_SOURCE_HASH:
        raise CollectiveBackendSourceMismatch(
            f"CCBackendFactory.create's source has changed (hash "
            f"{current_hash} != expected {_EXPECTED_SOURCE_HASH}). Refusing "
            f"to install the collective-backend interception over an "
            f"implementation this project hasn't reviewed -- update "
            f"_EXPECTED_SOURCE_HASH in {__name__} only after confirming "
            f"_patched_create still does the right thing against the new "
            f"body.")
    CCBackendFactory.create = classmethod(_patched_create)
    _installed = True
