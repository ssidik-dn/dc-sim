"""The project's one entry point into Frontier's extension registries.

Call `install()` once, with the engine objects a run needs, before calling
into Frontier (`frontier.main.main()` or equivalent). It fans out to the
per-concern install steps:

- `cc_backend.install()` -- registers `EngineCCBackend` with
  `CCBackendFactory` (task 06). Needs no engine context: nothing currently
  reaches it through Frontier's own CLI/config layer (task 06 finding), so it
  is only ever constructed directly, with a `Fabric`/`Placement`/`CostBackend`
  passed straight into its constructor by whoever builds it.
- `kv_transfer.predictor.set_context()` -- makes the `Fabric`, `Placement`,
  `Deployment`, and `CommGroupRegistry` reachable from
  `EngineKVCacheTransferPredictor.get_transfer_time()`, which Frontier *does*
  construct itself (task 07 finding: the KV path is genuinely selectable),
  and which therefore cannot be handed these objects as constructor
  arguments -- see the task 09 report for why a module-level context, set
  here, was chosen over the alternatives (an InfraGraph file path passed as a
  CLI flag; teaching the config class to serialise the fabric).
"""
from __future__ import annotations

from engine.logical.deployment import Deployment
from engine.physical.topology import Fabric
from engine.placement.placement import Placement

from ..cc_backend.comm_groups import CommGroupRegistry
from . import cc_backend as _cc_backend
from ..kv_transfer.predictor import EngineKVContext, set_context as _set_kv_context


def install(fabric: Fabric, placement: Placement, deployment: Deployment,
           groups: CommGroupRegistry) -> None:
    """Register every engine-backed Frontier extension, and make the engine
    state they need reachable. Safe to call more than once."""
    _cc_backend.install()
    _set_kv_context(EngineKVContext(fabric, placement, deployment, groups))
