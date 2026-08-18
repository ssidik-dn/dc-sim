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

Importing `kv_transfer.predictor` and `m2n_transfer.predictor` below is not
decorative: each defines its `Engine*Config` class at module level and
registers its predictor at import time (task 07's weak-reference finding --
see those modules' docstrings), but neither happens until the module is
actually imported. Task 09's own report noted `install()` reaching
`kv_transfer.predictor` only as a side effect of importing `EngineKVContext`
from it; task 11's context consolidation (moving that class to `.context`)
quietly removed that side effect and broke discovery of both predictors'
config classes until this was caught by running the real end-to-end tools
rather than only the unit tests, which import the predictor modules
directly and never hit the gap. Both imports are now explicit, for exactly
that reason.
"""
from __future__ import annotations

from typing import Optional

from engine.logical.deployment import Deployment
from engine.physical.topology import Fabric
from engine.placement.placement import Placement

from ..cc_backend.comm_groups import CommGroupRegistry
from ..context import BindingConfig, EngineContext
from ..context import set_context as _set_context
from . import cc_backend as _cc_backend
from ..kv_transfer import predictor as _kv_transfer_predictor  # noqa: F401  (see docstring)
from ..m2n_transfer import predictor as _m2n_transfer_predictor  # noqa: F401  (see docstring)


def install(fabric: Fabric, placement: Placement, deployment: Deployment,
           groups: CommGroupRegistry, binding: Optional[BindingConfig] = None) -> None:
    """Register every engine-backed Frontier extension, and make the engine
    state they need reachable. Safe to call more than once.

    `binding` (task 14) is optional and defaults to `None` -- unconfigured,
    every predictor's behaviour is exactly what it was before task 14: raise
    on a destination pool with more than one replica, rather than guess.
    """
    _cc_backend.install()
    _set_context(EngineContext(fabric, placement, deployment, groups, binding=binding))
