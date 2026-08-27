"""Stage 2 Gate C.1 vLLM-config compat: the smoke test's own second real
blocker, one layer past the RoPE fix (`rope_api_adapter.py`).

**Real, live-traced mechanism** (pinned `vllm/vllm-openai-rocm@sha256:bb44b39a...`,
inspected on `xai-3`, CPU-only, no GPU device claimed):
`vllm.model_executor.custom_op.CustomOp.__init__` unconditionally calls
`self.dispatch_forward(...)`, which calls `get_cached_compilation_config()`
-> `get_current_vllm_config().compilation_config` -> raises
`AssertionError` if no `set_current_vllm_config(vllm_config)` context is
currently active. `vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding`
(the class Frontier's own `get_rope()` constructs, via
`rope_api_adapter.py`) is `CustomOp`-derived -- confirmed live via
`issubclass`. `CustomOp.forward()` just calls the already-bound
`self._forward_method` set once in `__init__` -- **the context is only
needed around construction, not around every subsequent forward call**
(confirmed by reading `CustomOp.forward`'s own real source: no
`get_current_vllm_config()` lookup at call time).

**Which Frontier profiling layers actually need this, checked layer by
layer, not assumed** (`issubclass(cls, CustomOp)`, live, against the
real pinned classes):

| Frontier profiling layer | vLLM class Frontier actually reaches | `CustomOp`? | needs this context? |
|---|---|---|---|
| `attn_rope` (`get_rope()`, `rope_api_adapter.py`) | `vllm...rotary_embedding.base.RotaryEmbedding` | **True** | **Yes** -- this is the failure that blocked Probe 1 |
| `input_layernorm`/`post_attention_layernorm` (Frontier's own `RMSNorm` class, `frontier/profiling/common/layers/layernorm.py`) | none, for a standard (non-Gemma) model -- Frontier's own `RMSNorm.forward()` calls raw functions `rms_norm`/`fused_add_rms_norm` it tries to import from `vllm.model_executor.layers.layernorm`, **not** vLLM's own `CustomOp`-derived `RMSNorm` class | n/a for this path | **No, for Qwen3-0.6B** -- but see the adjacent finding below: those two function names do not exist at all in the pinned vLLM (`ImportError`), a *third*, separate compatibility gap, out of this task's own scope, noted for the record, not fixed here. `VllmGemmaRMSNorm` (`CustomOp`-derived, confirmed live) is only constructed for Gemma-style models -- unreached for Qwen3-0.6B |
| `mlp_act` (Frontier's own `SiluAndMul`, `frontier/profiling/common/layers/activation.py`) | none -- calls `torch.ops._C.silu_and_mul` directly, bypassing vLLM's own `CustomOp`-derived `SiluAndMul` class entirely (confirmed: that vLLM class *is* `CustomOp`-derived, but Frontier never constructs it) | n/a for this path | **No** |
| `attn_pre_proj`/`attn_post_proj` (`ColumnParallelLinear`/`RowParallelLinear`) | **Frontier's own** `frontier.profiling.common.parallel_utils.tensor_parallel_layers.{Column,Row}ParallelLinear` -- **not** vLLM's upstream class of the same name (confirmed from Frontier's own imports in `linear_op_impl.py`); a plain `torch.nn.Module` that calls specific vLLM GEMM/quantization *functions* internally, the same "borrow functions, not classes" pattern as `RMSNorm`/`SiluAndMul` below | **False** (confirmed live: not `CustomOp`-derived; `world_size` is an explicit constructor argument that bypasses vLLM's own tensor-parallel-group global state entirely when supplied, which Frontier's real call sites always do) | **No** -- confirmed both by class inspection and by the real Probe 1 traceback itself, which got past this construction before failing at `get_rope()` |
| `mlp_up_proj`/`mlp_down_proj` (same Frontier-own linear classes) | same | **False** | **No** |
| `emb` (`VocabParallelEmbedding`) | same module, same pattern | **False**, by inspection (plain `torch.nn.Module`, same file/pattern as the three classes above) | **No** |
| MoE fused kernels (`frontier/profiling/moe/*.py`, out of Gate C's own current single-host-first scope) | `vllm.model_executor.layers.fused_moe.*` -- function-based (`fused_topk`, grouped-GEMM helpers), no `class ... (CustomOp)` found anywhere in `frontier/profiling/moe/` (confirmed via grep) | not `CustomOp`-class-based in the paths Frontier actually calls | **Likely no**, not exhaustively re-verified per-function; out of scope for Qwen3-0.6B (dense, `is_moe=False`) |
| Attention-family ops (`attn_kv_cache_save`/`attn_prefill`/`attn_decode`, `frontier/profiling/attention/`) | rotary is **not** part of this family at all (`attn_rope` is a `linear_op` operator, confirmed: `get_rope` has no caller anywhere under `frontier/profiling/attention/`) | not checked for other vLLM classes this tool may construct (e.g. FlashInfer wrapper internals) | **not exercised by this task** -- Gate C's own first validation space never invoked attention profiling in this smoke test; a separate audit would be needed before relying on this context for that tool too |

**Conclusion for the failure actually blocking Probe 1**: exactly one
real `CustomOp` construction (`RotaryEmbedding`, via `get_rope()`) needs
this context for Qwen3-0.6B's `linear_op` profiling. This module does
not "fix" that by patching `RotaryEmbedding` or any other individual
`CustomOp` -- it provides one context manager that wraps Frontier's
own profiling construction (and, harmlessly, execution) region, per
this task's own explicit preference for one profiling-context adapter
over one patch per `CustomOp`.

**Minimum config, and why nothing more is needed**: `dispatch_forward`
only ever reads `vllm_config.compilation_config` (confirmed from its
own real source -- no other `VllmConfig` field is touched on this code
path). `VllmConfig()`'s own real `__post_init__` needs a resolvable
`device_config` to get past device-type auto-detection (which fails
with no real accelerator device file visible -- true of this
CPU-only, `--network none`, no `--device` investigation, expected to
differ once real `/dev/kfd`/`/dev/dri` are attached for the real
retry) -- `DeviceConfig(device="cuda")` is not a workaround value: it
is the literal, only accelerator string in `DeviceConfig`'s own real
`Literal['auto','cuda','cpu','tpu','xpu']` schema -- ROCm has no
separate value; vLLM represents it through the same `"cuda"` string
throughout (matching this project's own already-established finding
that ROCm PyTorch keeps the `torch.cuda` namespace). No `model_config`
is constructed or passed -- confirmed unread by the exact code path
this fix targets, and building one would be real, unrelated serving
state this task's own §5 instruction says not to initialize.

**A real, disclosed fidelity finding, not glossed over (this task's
own §6)**: leaving every other field at its real default (rather than
hand-picking `custom_ops=["all"]`) reproduces this project's own real
Gate B serving default exactly -- confirmed live:
`VllmConfig(device_config=DeviceConfig(device="cuda"))` alone resolves
to `compilation_config.custom_ops == ["none"]`,
`compilation_config.mode == CompilationMode.VLLM_COMPILE` (from
`optimization_level`'s own real default, `O2`, unmodified anywhere in
this project's own real `sim-real` vLLM launch commands). Under
`custom_ops=["none"]`, `RotaryEmbedding.enabled()` is `False`, so
`dispatch_forward` returns the *native* (plain PyTorch, not the
hand-written ROCm/HIP kernel) forward path -- **the same dispatch
decision real production serving's own default config makes**. But
real production serving reaches that decision expecting a *compiled*
model (`torch.compile`, via vLLM's own `@support_torch_compile`
decorator on its real model classes) to fuse/optimize that native
path; Frontier's own profiling harness constructs plain, undecorated
`nn.Module`s and never invokes vLLM's model-level compilation at all,
for *any* operator it profiles, not only rotary. Reproducing
`custom_ops=["none"]` literally therefore risks measuring the *worst*
of both real paths (no hand-written kernel, no compiler fusion either)
rather than either one production actually uses. This module does not
silently resolve that tension -- it is reported in the deliverable
report's own §6, with an explicit, disclosed recommendation
(`custom_ops` left at its real per-op default is faithful to the
*flag*, not to what real compiled serving executes; enabling custom
ops via `optimization_level=OptimizationLevel.O0` is closer to "the
best real, hand-tuned kernel available to an uncompiled, isolated
harness" -- consistent with how every *other* operator this project
profiles is already measured, not a new inconsistency introduced
here) rather than picked silently inside this module.
"""
from __future__ import annotations

import dataclasses
import inspect
from contextlib import contextmanager
from typing import Any, Dict, Optional

_installed = False
_observed_vllm_version: Optional[str] = None
_last_built_config_summary: Optional[Dict[str, Any]] = None


class VllmConfigContextUnknownApi(RuntimeError):
    pass


def _verify_vllm_config_api_shape():
    """Checks the real, installed vLLM's own `VllmConfig`/`DeviceConfig`/
    `set_current_vllm_config` shape before relying on any of them --
    never assumes the API this module was written against is still
    accurate. Raises `VllmConfigContextUnknownApi` (loud, explicit) on
    any mismatch, per this task's own "unknown API -> loud failure"
    requirement -- mirrors `rope_api_adapter.py`'s own detection
    philosophy, applied to a public API surface instead of a Frontier
    patch target (no source hash here -- there is no Frontier source
    being patched by this module, only a public vLLM context manager
    being used as documented).
    """
    import vllm.config.device as device_module
    import vllm.config.vllm as vllm_config_module

    if not hasattr(vllm_config_module, "VllmConfig"):
        raise VllmConfigContextUnknownApi(
            "vllm.config.vllm has no VllmConfig -- this module was written "
            "against a real, live-inspected pinned vLLM (0.27.1); the "
            "installed version's API no longer matches."
        )
    if not hasattr(device_module, "DeviceConfig"):
        raise VllmConfigContextUnknownApi(
            "vllm.config.device has no DeviceConfig."
        )
    if not hasattr(vllm_config_module, "set_current_vllm_config"):
        raise VllmConfigContextUnknownApi(
            "vllm.config.vllm has no set_current_vllm_config."
        )

    device_fields = {f.name for f in dataclasses.fields(device_module.DeviceConfig)}
    if "device" not in device_fields:
        raise VllmConfigContextUnknownApi(
            f"DeviceConfig's real fields {sorted(device_fields)} no longer "
            "include 'device' -- refusing to guess a replacement."
        )
    vllm_config_fields = {f.name for f in dataclasses.fields(vllm_config_module.VllmConfig)}
    if "device_config" not in vllm_config_fields:
        raise VllmConfigContextUnknownApi(
            f"VllmConfig's real fields {sorted(vllm_config_fields)} no longer "
            "include 'device_config' -- refusing to guess a replacement."
        )
    sig = inspect.signature(vllm_config_module.set_current_vllm_config)
    if "vllm_config" not in sig.parameters:
        raise VllmConfigContextUnknownApi(
            f"set_current_vllm_config's real signature {sig} no longer takes "
            "a 'vllm_config' parameter -- refusing to guess a replacement."
        )
    return vllm_config_module, device_module


def build_profiling_vllm_config(**overrides: Any):
    """Builds the minimum real `VllmConfig` `CustomOp` construction
    (specifically the `RotaryEmbedding` path Frontier's own `get_rope()`
    reaches) needs. Every field not explicitly passed in `overrides`
    stays at vLLM's own real default -- this function hard-codes
    exactly one literal, `device="cuda"` (the only real accelerator
    value `DeviceConfig`'s own schema defines; ROCm has no separate
    value there), and nothing model-specific. `**overrides` lets a
    real caller pass real `ModelSpec`/profiling-configuration-derived
    fields explicitly (e.g. a real `model_config=` once one is needed
    for a code path this module's own current scope doesn't reach) --
    never a place for a hidden default.
    """
    vllm_config_module, device_module = _verify_vllm_config_api_shape()
    device_config = overrides.pop("device_config", None)
    if device_config is None:
        device_config = device_module.DeviceConfig(device="cuda")
    return vllm_config_module.VllmConfig(device_config=device_config, **overrides)


@contextmanager
def profiling_vllm_config_context(**overrides: Any):
    """Wraps Frontier's own profiling model construction (and,
    harmlessly, its subsequent execution -- `CustomOp.forward()` never
    re-reads the config, confirmed from source, but wrapping both is no
    more expensive and simpler to reason about) in a real
    `set_current_vllm_config` context, covering the whole profiling
    construction/execution region rather than patching each `CustomOp`
    individually, per this task's own explicit preference.

    Delegates nesting/restoration entirely to vLLM's own real
    `set_current_vllm_config` (a `contextmanager` that saves and
    restores the prior `_current_vllm_config`/`_current_prefix` in its
    own `finally` block, confirmed from source) -- this module does not
    reimplement or shadow that behavior, so a pre-existing outer vLLM
    context (however unlikely inside Frontier's own profiling CLI) is
    handled exactly as vLLM's own nesting semantics intend, not by a
    second, possibly-inconsistent mechanism.
    """
    global _installed, _observed_vllm_version, _last_built_config_summary

    vllm_config_module, _ = _verify_vllm_config_api_shape()
    vllm_config = build_profiling_vllm_config(**overrides)

    try:
        import vllm

        _observed_vllm_version = getattr(vllm, "__version__", None)
    except Exception:  # noqa: BLE001 -- provenance-only, never blocks the context
        _observed_vllm_version = None

    _last_built_config_summary = {
        "device_config.device_type": vllm_config.device_config.device_type,
        "compilation_config.custom_ops": list(vllm_config.compilation_config.custom_ops),
        "compilation_config.mode": (
            vllm_config.compilation_config.mode.name
            if vllm_config.compilation_config.mode is not None
            else None
        ),
        "optimization_level": (
            vllm_config.optimization_level.name
            if hasattr(vllm_config.optimization_level, "name")
            else vllm_config.optimization_level
        ),
    }
    _installed = True

    with vllm_config_module.set_current_vllm_config(vllm_config):
        yield vllm_config


def get_vllm_config_context_status() -> Dict[str, Any]:
    """Provenance-facing snapshot. `None`/empty until
    `profiling_vllm_config_context()` has actually been entered at
    least once -- never a guessed default."""
    return {
        "applied": _installed,
        "detected_vllm_version": _observed_vllm_version,
        "last_built_config": dict(_last_built_config_summary)
        if _last_built_config_summary is not None
        else None,
    }
