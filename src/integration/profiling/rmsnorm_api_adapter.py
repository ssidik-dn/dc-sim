"""Stage 2 Gate C.1 RMSNorm compat: the real MI355X Probe 1 retry's own
third, distinct failure, one layer past `get_rope()`/`CustomOp`.

`frontier/profiling/common/layers/layernorm.py::RMSNorm.forward()`
calls free functions `rms_norm`/`fused_add_rms_norm` it tries to import
from `vllm.model_executor.layers.layernorm`. Live-checked against the
pinned image (`vllm/vllm-openai-rocm@sha256:bb44b39a...`, `xai-3`):
neither name exists there (`ImportError`, confirmed by the real Probe 1
traceback and by listing the real module's own exports:
`['CustomOp', 'F', 'GemmaRMSNorm', 'LayerNorm', 'RMSNorm', 'RMSNormGated',
'envs', 'init_logger', 'ir', 'logger', 'nn', 'poly_norm',
'rms_norm_batch_invariant', 'torch', 'vllm']`). The functionality now
lives only on the `RMSNorm` *class*'s own `forward()`.

**Semantic equivalence, established from source, not approximated**:
live-inspected `RMSNorm.__init__(self, hidden_size, eps=1e-6,
var_hidden_size=None, has_weight=True, dtype=None)` -- `hidden_size`/
`eps` are the *same names* Frontier's own constructor already uses;
`var_hidden_size`/`has_weight`/`dtype` are new, optional, and their
real defaults (`None`/`True`/`None`) reproduce exactly what Frontier's
own old code implicitly assumed (full-hidden-size normalization, a
real weight, default dtype) -- omitting them changes nothing.
`forward_native(self, x, residual=None) -> Tensor | tuple[Tensor,
Tensor]` -- **the exact same two-shape contract** Frontier's own
`RMSNorm.forward` already declares in its own type hint
(`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`), confirmed
from the real source, not inferred: `residual=None` returns a lone
tensor (the old `rms_norm` free function's own contract); `residual`
given returns `(normed, updated_residual)` (the old `fused_add_rms_norm`
free function's own contract, including its "maybe in-place" mutation
convention -- `ir.ops.fused_add_rms_norm.maybe_inplace(...)`,
confirmed live, the same real behavior the old free function already
had). `RMSNorm` is `CustomOp`-derived (confirmed live,
`MRO=['RMSNorm','CustomOp','Module','object']`) -- construction needs
`profiling_vllm_config_context()` active, exactly the mechanism §7 of
this task requires reusing, not a new one.

**The fix Frontier's own code already contains, one class over**:
`GemmaRMSNorm` (the sibling wrapper in the same file) already
constructs the real vLLM class and delegates
`self._impl(x, residual)` -- it needed no fix at all, because it never
relied on the now-removed free functions. This module makes plain
`RMSNorm` do the same thing, only when the real installed vLLM's own
API requires it (structurally detected, not assumed) -- the old,
free-function path is left completely alone when it is what the real
API actually offers, matching every other adapter in this project's
own "detect deliberately, hard fail on unknown" convention.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Any, Optional, Tuple, Union

_EXPECTED_RMSNORM_INIT_HASH = (
    "e17dfb85a4534cae4246317ec2dd17fcb8cecd34c67ce823f72d750994c4584c"
)
_EXPECTED_RMSNORM_FORWARD_HASH = (
    "a07a894e55ccf1d133f05faf366a75ba2532432892c88fdfc4f2f9bbcdc1502a"
)

_installed = False
_observed_api_kind: Optional[str] = None


class RmsnormApiAdapterSourceMismatch(RuntimeError):
    pass


class RmsnormApiUnknownSignature(RuntimeError):
    pass


def _detect_rmsnorm_api(vllm_layernorm_module) -> str:
    """Returns `"old"` (free functions `rms_norm`/`fused_add_rms_norm`
    exist -- Frontier's own original code already works, untouched) or
    `"new"` (only the `RMSNorm` class exists, with the exact
    `(x, residual=None)` shape this module was written against).
    Raises `RmsnormApiUnknownSignature` -- never guesses -- on anything
    else."""
    has_old_functions = hasattr(vllm_layernorm_module, "rms_norm") and hasattr(
        vllm_layernorm_module, "fused_add_rms_norm"
    )
    if has_old_functions:
        return "old"

    if not hasattr(vllm_layernorm_module, "RMSNorm"):
        raise RmsnormApiUnknownSignature(
            "vllm.model_executor.layers.layernorm has neither the old "
            "free functions (rms_norm/fused_add_rms_norm) nor an RMSNorm "
            "class -- refusing to guess a translation."
        )
    cls = vllm_layernorm_module.RMSNorm
    init_sig = inspect.signature(cls.__init__)
    if "hidden_size" not in init_sig.parameters or "eps" not in init_sig.parameters:
        raise RmsnormApiUnknownSignature(
            f"vllm's RMSNorm.__init__ real signature {init_sig} no longer has "
            "'hidden_size'/'eps' -- refusing to guess a replacement."
        )
    forward_native = getattr(cls, "forward_native", None)
    if forward_native is None:
        raise RmsnormApiUnknownSignature(
            "vllm's RMSNorm class has no forward_native -- this module was "
            "written against that real method's own (x, residual=None) shape."
        )
    forward_sig = inspect.signature(forward_native)
    if "residual" not in forward_sig.parameters:
        raise RmsnormApiUnknownSignature(
            f"vllm's RMSNorm.forward_native real signature {forward_sig} no "
            "longer has 'residual' -- refusing to guess a replacement."
        )
    return "new"


def get_rmsnorm_api_adapter_status() -> dict:
    return {
        "applied": _installed,
        "detected_api_kind": _observed_api_kind,
        "frontier_init_hash": _EXPECTED_RMSNORM_INIT_HASH,
        "frontier_forward_hash": _EXPECTED_RMSNORM_FORWARD_HASH,
    }


def install_rmsnorm_api_adapter() -> None:
    """Patches `frontier.profiling.common.layers.layernorm.RMSNorm`'s
    own `__init__`/`forward`/`weight` **only if** the real, installed
    vLLM no longer offers the old free functions -- verified by
    structural inspection (§`_detect_rmsnorm_api`), not by catching an
    `ImportError`. When the new API is detected, the patched class
    constructs a real `vllm...layernorm.RMSNorm` instance (mirroring
    the exact pattern Frontier's own sibling `GemmaRMSNorm` wrapper
    already uses, unmodified) and delegates to it -- the class is
    `CustomOp`-derived, so construction must happen inside
    `profiling_vllm_config_context()` (reused, not reimplemented; this
    module does not open that context itself -- it must already be
    active at the call site, exactly as it already is for `get_rope()`
    in the real Probe 1 path).

    Safe to call more than once (idempotent). Requires `torch` and
    `vllm` (transitively, importing the target Frontier module and the
    real vLLM layernorm module) -- raises the same way importing those
    directly would if either is missing.

    Raises `RmsnormApiAdapterSourceMismatch` if Frontier's own
    `RMSNorm.__init__`/`.forward` source has changed from what this
    module was written against. Raises `RmsnormApiUnknownSignature` if
    the real, installed vLLM's own API matches neither known shape.
    """
    global _installed, _observed_api_kind
    if _installed:
        return

    import frontier.profiling.common.layers.layernorm as layernorm_module

    current_init_hash = hashlib.sha256(
        inspect.getsource(layernorm_module.RMSNorm.__init__).encode()
    ).hexdigest()
    if current_init_hash != _EXPECTED_RMSNORM_INIT_HASH:
        raise RmsnormApiAdapterSourceMismatch(
            f"RMSNorm.__init__'s source has changed (hash {current_init_hash} "
            f"!= expected {_EXPECTED_RMSNORM_INIT_HASH}). Refusing to patch "
            "over an implementation this project hasn't reviewed."
        )
    current_forward_hash = hashlib.sha256(
        inspect.getsource(layernorm_module.RMSNorm.forward).encode()
    ).hexdigest()
    if current_forward_hash != _EXPECTED_RMSNORM_FORWARD_HASH:
        raise RmsnormApiAdapterSourceMismatch(
            f"RMSNorm.forward's source has changed (hash {current_forward_hash} "
            f"!= expected {_EXPECTED_RMSNORM_FORWARD_HASH}). Refusing to patch "
            "over an implementation this project hasn't reviewed."
        )

    import vllm.model_executor.layers.layernorm as vllm_layernorm_module

    api_kind = _detect_rmsnorm_api(vllm_layernorm_module)
    _observed_api_kind = api_kind

    if api_kind == "old":
        # Frontier's own original code already works against this real
        # API -- nothing to patch.
        _installed = True
        return

    import torch.nn as nn

    def _patched_init(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        nn.Module.__init__(self)
        from frontier.profiling.common.cuda_timer import CudaTimer

        self._impl = vllm_layernorm_module.RMSNorm(hidden_size, eps=eps)
        self.variance_epsilon = eps
        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)

    def _patched_forward(
        self, x, residual=None
    ) -> Union[Any, Tuple[Any, Any]]:
        with self._norm_timer:
            return self._impl(x, residual)

    def _patched_weight(self):
        return self._impl.weight

    layernorm_module.RMSNorm.__init__ = _patched_init
    layernorm_module.RMSNorm.forward = _patched_forward
    layernorm_module.RMSNorm.weight = property(_patched_weight)
    _installed = True
