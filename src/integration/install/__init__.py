"""The project's one entry point into Frontier's extension registries.

Call `install()` once, with the engine objects a run needs, before calling
into Frontier (`frontier.main.main()` or equivalent). It fans out to the
per-concern install steps:

- `cc_backend.install()` -- registers `EngineCCBackend` with
  `CCBackendFactory` (task 06). Needs no engine context: nothing currently
  reaches it through Frontier's own CLI/config layer (task 06 finding), so it
  is only ever constructed directly, with a `Fabric`/`Placement`/`CostBackend`
  passed straight into its constructor by whoever builds it.
- `context.set_context()` -- makes the `Fabric`, `Placement`, `Deployment`,
  and `CommGroupRegistry` reachable from every predictor Frontier *does*
  construct itself (KV cache transfer, task 09; M2N transfer, task 11 --
  both selectable, per tasks 07/08's findings), and which therefore cannot be
  handed these objects as constructor arguments. See the task 09 report for
  why a module-level context, set here, was chosen over the alternatives (an
  InfraGraph file path passed as a CLI flag; teaching a config class to
  serialise the fabric) -- and task 11's report for why that context is
  shared by every such predictor rather than one per predictor type.
"""
from __future__ import annotations

from engine.logical.deployment import Deployment
from engine.physical.topology import Fabric
from engine.placement.placement import Placement

from ..cc_backend.comm_groups import CommGroupRegistry
from ..context import EngineContext
from ..context import set_context as _set_context
from . import cc_backend as _cc_backend


def install(fabric: Fabric, placement: Placement, deployment: Deployment,
           groups: CommGroupRegistry) -> None:
    """Register every engine-backed Frontier extension, and make the engine
    state they need reachable. Safe to call more than once."""
    _cc_backend.install()
    _set_context(EngineContext(fabric, placement, deployment, groups))
