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
from ..cc_backend.collective import install_collective_backend as _install_collective_backend
from ..context import BindingConfig, EngineContext
from ..context import set_context as _set_context
from . import cc_backend as _cc_backend
from ..kv_transfer import predictor as _kv_transfer_predictor  # noqa: F401  (see docstring)
from ..m2n_transfer import predictor as _m2n_transfer_predictor  # noqa: F401  (see docstring)
from ..replica_scheduler.sglang_guard import (
    install_sglang_replica_scheduler_guard as _install_sglang_replica_scheduler_guard,
)
from ..execution_time_predictor.mla_phase_filter import (
    install_mla_phase_filter as _install_mla_phase_filter,
)
from ..profiling.qk_norm_allowlist_fix import (
    install_qk_norm_allowlist_fix as _install_qk_norm_allowlist_fix,
)
from ..execution_time_predictor.dense_model_moe_routing_guard import (
    install_dense_model_moe_routing_guard as _install_dense_model_moe_routing_guard,
)


def install(fabric: Fabric, placement: Placement, deployment: Deployment,
           groups: CommGroupRegistry, binding: Optional[BindingConfig] = None,
           collective: bool = False, sglang_replica_scheduler: bool = False,
           mla_phase_filter: bool = False, qk_norm_allowlist_fix: bool = False,
           dense_model_moe_routing_guard: bool = False) -> None:
    """Register every engine-backed Frontier extension, and make the engine
    state they need reachable. Safe to call more than once.

    `binding` (task 14) is optional and defaults to `None` -- unconfigured,
    every predictor's behaviour is exactly what it was before task 14: raise
    on a destination pool with more than one replica, rather than guess.

    `collective` (task 20) is optional and defaults to `False`. `True`
    patches `CCBackendFactory.create` (`..cc_backend.collective`, guarded
    by a source hash) so every tensor-/pipeline-/expert-parallel collective
    Frontier's own execution-time predictors ask for is priced by
    `EngineCCBackend` from the fabric instead of Frontier's placement-blind
    profiled table -- regardless of which `--cc_backend_config_type` value
    the run's own argv used. `False` (the default) leaves
    `CCBackendFactory.create` untouched, same as every task before this
    one.

    `sglang_replica_scheduler` (task 47) is optional and defaults to
    `False`. `True` relaxes `SGLangStyleReplicaScheduler.__init__`'s own
    cluster-type refusal (`..replica_scheduler.sglang_guard`, guarded by a
    source hash) so `--...replica_scheduler_config_type sglang` can be
    selected for `DECODE_ATTN`/`DECODE_FFN`, not only `MONOLITHIC`/`PREFILL`
    -- nothing else about the class changes. `False` (the default) leaves
    `SGLangStyleReplicaScheduler.__init__` untouched, same as every task
    before this one.

    `mla_phase_filter` (task 53) is optional and defaults to `False`. `True`
    patches `SklearnExecutionTimePredictor._train_mla_attention_layer_models`
    (`..execution_time_predictor.mla_phase_filter`, guarded by a source hash)
    so each MLA attention operator trains only on the phase(s) its own family
    spec declares, instead of on every profiled row regardless of phase.
    `False` (the default) leaves `_train_mla_attention_layer_models`
    untouched, same as every task before this one.

    `qk_norm_allowlist_fix` (Stage 2 Gate C.1) is optional and defaults to
    `False`. `True` adds `"qwen3"` to `frontier.config.model_config.QK_NORM_MODEL_TYPE_ALLOWLIST`
    (`..profiling.qk_norm_allowlist_fix`, guarded by an exact-contents check
    of the allowlist) so a plain dense Qwen3 model's own real `use_qk_norm`
    flag is correctly inferred `True` from its HF config, instead of only
    `qwen3_moe`/`qwen3_next` -- confirmed against HuggingFace `transformers`'
    own `Qwen3Attention` source, which applies `q_norm`/`k_norm`
    unconditionally for every Qwen3 variant. This flag must be correct
    *before* linear_op profiling runs, not only before evaluation --
    `linear_op_impl.py` reads it to decide whether to actually run the
    QK-norm compute during collection. `False` (the default) leaves
    `QK_NORM_MODEL_TYPE_ALLOWLIST` untouched, same as every task before
    this one.

    `dense_model_moe_routing_guard` (Stage 2 Gate C.1) is optional and
    defaults to `False`. `True` patches
    `SklearnDisaggregationExecutionTimePredictor.__init__`
    (`..execution_time_predictor.dense_model_moe_routing_guard`, guarded
    by a source hash over the whole method) so a dense model
    (`model_config.is_moe=False`) skips MoE expert-routing simulation
    entirely instead of crashing on `total_expert_num=0`
    (`ZeroDivisionError` in `_generate_expert_allocations`), while a
    model whose metadata claims `is_moe=True` but declares
    `total_expert_num<=0` fails loudly with an explicit
    `InconsistentMoeModelMetadataError` instead. Every real MoE model
    (`is_moe=True, total_expert_num>0`) is unaffected -- the original
    routing computation runs unchanged. `False` (the default) leaves
    `SklearnDisaggregationExecutionTimePredictor.__init__` untouched,
    same as every task before this one.
    """
    _cc_backend.install()
    if collective:
        _install_collective_backend()
    if sglang_replica_scheduler:
        _install_sglang_replica_scheduler_guard()
    if mla_phase_filter:
        _install_mla_phase_filter()
    if dense_model_moe_routing_guard:
        _install_dense_model_moe_routing_guard()
    if qk_norm_allowlist_fix:
        _install_qk_norm_allowlist_fix()
    _set_context(EngineContext(fabric, placement, deployment, groups, binding=binding,
                               collective=collective))
