"""Stage 2 Gate C.1 ROPE compat: the real MI355X smoke test's own first
real failure. `frontier/profiling/common/layers/rotary_embedding.py`'s
own `get_rope()` calls the real, installed vLLM's
`vllm.model_executor.layers.rotary_embedding.get_rope` with
`head_size=, rotary_dim=, max_position=, base=, is_neox_style=,
rope_scaling=, dtype=` -- every real keyword Frontier's own three call
sites (`frontier/profiling/linear_op/linear_op_impl.py` lines 206, 316,
485, all identical in shape) have ever used. The pinned smoke-test
image's own real vLLM (`0.27.1`, confirmed live via
`inspect.signature`) no longer accepts `rotary_dim` or `base` as
top-level parameters at all:

    def get_rope(
        head_size: int,
        max_position: int,
        is_neox_style: bool = True,
        rope_parameters: dict[str, Any] | None = None,
        dtype: torch.dtype | None = None,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> RotaryEmbedding:

Confirmed from the real, live-fetched source (same image/digest): the
new API derives the rotary dimension from
`rope_parameters.get("rope_dim")` if present, else
`head_size * rope_parameters.get("partial_rotary_factor", 1.0)`, and
reads `base`/theta from `rope_parameters.get("rope_theta", 10000)` --
**not** from a top-level `base` argument. Passing `rope_parameters=None`
(the naive fix) would silently fall back to `rope_theta=10000`,
corrupting every model's real RoPE base (Qwen3-0.6B's real
`rope_theta=1000000.0`) rather than raising -- exactly the "hard
failure into silent wrong construction" this task's own §2 forbids.
The correct, general, always-exact translation is
`rope_parameters["rope_dim"] = rotary_dim` (respected verbatim by the
real source, taking precedence over `partial_rotary_factor` -- true
for *any* `rotary_dim`, not only the `rotary_dim == head_size` case
every current Frontier call site happens to use) and
`rope_parameters["rope_theta"] = base`, merged with whatever scaling
dict Frontier's own `rope_scaling` already carries (its own
`rope_type`/`factor`/etc. keys are read identically by the new API
under the same names, confirmed from source).

Guarded by a source hash over the two functions this module depends on
staying shaped the way it reads them (`_load_vllm_get_rope`, the patch
target, and `get_rope`, whose own call-site kwargs shape this adapter
must match) -- matching this project's established pattern (task 20,
47, task 53 Fix A/B, the qk_norm allowlist fix). Detects the *real
vLLM's* own API shape by inspecting `inspect.signature`, never by
trying a call and catching `TypeError` (a caught `TypeError` cannot
distinguish "wrong RoPE API" from any other unrelated bug) --
recognizes exactly two known shapes (old: has `rotary_dim`+`base`, no
`rope_parameters`; pinned/new: has `rope_parameters`, no
`rotary_dim`/`base`) and raises `RopeApiUnknownSignature` on anything
else, per this task's own explicit "detect deliberately, hard fail on
unknown" instruction.

`_detect_and_build_adapter` (the actual detection+translation logic)
needs no `torch` at all -- pure `inspect.signature` and dict work,
testable directly with a plain-Python stand-in callable. Only
`install_rope_api_adapter()` needs `torch` (it imports the real
Frontier module, which imports `torch` at module level) -- not called
at module-import time, same reasoning as `attention_block_table_fix.py`.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable, Dict, Optional

# Verified live against the real, pinned smoke-test image
# (vllm/vllm-openai-rocm@sha256:bb44b39a...) on xai-3, via
# hashlib.sha256(inspect.getsource(...)).hexdigest() executed inside that
# real, torch-present container -- not computed offline. (An earlier
# offline attempt via `ast.get_source_segment` produced a *different* hash
# for the same function: `ast.get_source_segment` omits the trailing
# newline `inspect.getsource` always includes on the last line -- a real,
# caught-before-any-GPU-time mismatch, exactly what this guard exists to
# catch. Recorded here as the reason live verification, not offline
# parsing, is this module's own source of truth.) A changed hash means
# either function's own body changed upstream -- install_rope_api_adapter()
# raises rather than patch over / assume a call shape this project hasn't
# reviewed.
_EXPECTED_LOAD_VLLM_GET_ROPE_HASH = (
    "f021ec8d00338b6872042f82d4318298226307727a2915dd8aee18d9b978fa06"
)
_EXPECTED_GET_ROPE_HASH = (
    "33776131dee30a060c228c63e5abaeb8c70b10fad414cf1e1d02c50a659864bd"
)

_installed = False

# Filled in by install_rope_api_adapter() the first time it actually detects
# the real, installed vLLM's own get_rope shape -- provenance-facing state,
# read by get_rope_api_adapter_status(). None until a real detection happens;
# never defaulted to a guessed value.
_observed_api_kind: Optional[str] = None
_observed_signature: Optional[str] = None
_observed_vllm_version: Optional[str] = None


class RopeApiAdapterSourceMismatch(RuntimeError):
    pass


class RopeApiUnknownSignature(RuntimeError):
    pass


_OldRopeAdapter = Callable[..., Any]


def _detect_and_build_adapter(real_vllm_get_rope: Callable[..., Any]):
    """Inspect `real_vllm_get_rope`'s own real signature and return
    `(api_kind, adapter, signature_str)`, where `adapter` has exactly
    Frontier's own expected old-style keyword shape
    (`head_size, rotary_dim, max_position, base, is_neox_style,
    rope_scaling, dtype`) regardless of which real API it wraps.

    `api_kind` is `"old"` (the real callable still accepts `rotary_dim`
    and `base` directly, no adaptation needed -- passed straight
    through) or `"new"` (the pinned-vLLM shape this module was written
    against: `rope_parameters`, no `rotary_dim`/`base`). Raises
    `RopeApiUnknownSignature` -- never guesses -- if neither shape
    matches.
    """
    sig = inspect.signature(real_vllm_get_rope)
    params = sig.parameters
    has_rotary_dim = "rotary_dim" in params
    has_base = "base" in params
    has_rope_parameters = "rope_parameters" in params
    signature_str = str(sig)

    if has_rotary_dim and has_base and not has_rope_parameters:
        def _old_adapter(
            *, head_size, rotary_dim, max_position, base, is_neox_style,
            rope_scaling, dtype,
        ):
            return real_vllm_get_rope(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position=max_position,
                base=base,
                is_neox_style=is_neox_style,
                rope_scaling=rope_scaling,
                dtype=dtype,
            )

        return "old", _old_adapter, signature_str

    if has_rope_parameters and not has_rotary_dim and not has_base:
        accepts_dual_chunk = "dual_chunk_attention_config" in params

        def _new_adapter(
            *, head_size, rotary_dim, max_position, base, is_neox_style,
            rope_scaling, dtype,
        ):
            rope_parameters: Dict[str, Any] = dict(rope_scaling) if rope_scaling else {}
            # `rope_dim` (not `partial_rotary_factor`) is the exact, general
            # translation of Frontier's own `rotary_dim` -- respected verbatim
            # by the real source ahead of `partial_rotary_factor`, correct
            # whether or not rotary_dim == head_size.
            injected = {"rope_theta": base, "rope_dim": rotary_dim}
            for key, value in injected.items():
                if key in rope_parameters and rope_parameters[key] != value:
                    raise RopeApiUnknownSignature(
                        f"rope_scaling already contains {key!r}="
                        f"{rope_parameters[key]!r}, conflicting with the value "
                        f"({value!r}) this adapter would inject from Frontier's "
                        f"own top-level argument -- refusing to silently "
                        f"override either one."
                    )
                rope_parameters[key] = value

            kwargs: Dict[str, Any] = dict(
                head_size=head_size,
                max_position=max_position,
                is_neox_style=is_neox_style,
                rope_parameters=rope_parameters,
                dtype=dtype,
            )
            if accepts_dual_chunk:
                kwargs["dual_chunk_attention_config"] = None
            return real_vllm_get_rope(**kwargs)

        return "new", _new_adapter, signature_str

    raise RopeApiUnknownSignature(
        "vllm.model_executor.layers.rotary_embedding.get_rope's real "
        f"signature {signature_str} matches neither the old API "
        "(has 'rotary_dim' and 'base', no 'rope_parameters') nor the "
        "pinned/new API this module was written against (has "
        "'rope_parameters', no 'rotary_dim'/'base'). Refusing to guess a "
        "translation -- update this module only after reviewing the real "
        "new signature."
    )


def get_rope_api_adapter_status() -> Dict[str, Any]:
    """Provenance-facing snapshot. `detected_api_kind`/`detected_signature`/
    `detected_vllm_version` are `None` until `install_rope_api_adapter()`
    has actually run a real detection -- never guessed or defaulted."""
    return {
        "applied": _installed,
        "detected_api_kind": _observed_api_kind,
        "detected_signature": _observed_signature,
        "detected_vllm_version": _observed_vllm_version,
        "frontier_load_vllm_get_rope_hash": _EXPECTED_LOAD_VLLM_GET_ROPE_HASH,
        "frontier_get_rope_hash": _EXPECTED_GET_ROPE_HASH,
    }


def install_rope_api_adapter() -> None:
    """Patch `frontier.profiling.common.layers.rotary_embedding`'s own
    module-level vLLM-loader (`_load_vllm_get_rope`) so it caches an
    *adapter* -- built by inspecting the real, installed vLLM's own
    `get_rope` signature -- instead of the raw vLLM function. Frontier's
    own `get_rope()` (unmodified) and every one of its three real call
    sites (`linear_op_impl.py`) keep calling with the exact same old-style
    keyword shape; the adapter is what changes behavior underneath them,
    only when the real installed vLLM's own signature requires it.

    Safe to call more than once (idempotent). Requires `torch`
    (transitively, importing `frontier.profiling.common.layers.rotary_embedding`
    imports it at module level) and requires `vllm` to be importable (the
    same real dependency Frontier's own `_load_vllm_get_rope` already
    needs) -- raises the same way importing those directly would if
    either is missing.

    Raises `RopeApiAdapterSourceMismatch` if either guarded function's
    source no longer matches what this module was written against.
    Raises `RopeApiUnknownSignature` if the real, installed vLLM's own
    `get_rope` signature matches neither known shape.
    """
    global _installed, _observed_api_kind, _observed_signature, _observed_vllm_version
    if _installed:
        return

    import frontier.profiling.common.layers.rotary_embedding as rope_module

    current_load_hash = hashlib.sha256(
        inspect.getsource(rope_module._load_vllm_get_rope).encode()
    ).hexdigest()
    if current_load_hash != _EXPECTED_LOAD_VLLM_GET_ROPE_HASH:
        raise RopeApiAdapterSourceMismatch(
            f"_load_vllm_get_rope's source has changed (hash {current_load_hash} "
            f"!= expected {_EXPECTED_LOAD_VLLM_GET_ROPE_HASH}). Refusing to "
            "patch over an implementation this project hasn't reviewed."
        )
    current_get_rope_hash = hashlib.sha256(
        inspect.getsource(rope_module.get_rope).encode()
    ).hexdigest()
    if current_get_rope_hash != _EXPECTED_GET_ROPE_HASH:
        raise RopeApiAdapterSourceMismatch(
            f"get_rope's source has changed (hash {current_get_rope_hash} != "
            f"expected {_EXPECTED_GET_ROPE_HASH}). Refusing to install an "
            "adapter written against a call-site kwargs shape that may no "
            "longer be accurate -- review this module against the new source "
            "before updating the expected hash."
        )

    def _patched_load_vllm_get_rope():
        if (
            rope_module._VLLM_GET_ROPE is not None
            or rope_module._VLLM_GET_ROPE_IMPORT_ERROR is not None
        ):
            return rope_module._VLLM_GET_ROPE
        try:
            from vllm.model_executor.layers.rotary_embedding import (
                get_rope as real_vllm_get_rope,
            )
        except Exception as exc:  # noqa: BLE001 -- mirrors the original's own bare except
            rope_module._VLLM_GET_ROPE_IMPORT_ERROR = exc
            return None

        api_kind, adapter, signature_str = _detect_and_build_adapter(real_vllm_get_rope)

        global _observed_api_kind, _observed_signature, _observed_vllm_version
        _observed_api_kind = api_kind
        _observed_signature = signature_str
        try:
            import vllm

            _observed_vllm_version = getattr(vllm, "__version__", None)
        except Exception:  # noqa: BLE001 -- provenance-only, never blocks the adapter
            _observed_vllm_version = None

        rope_module._VLLM_GET_ROPE = adapter
        return adapter

    rope_module._load_vllm_get_rope = _patched_load_vllm_get_rope
    _installed = True
