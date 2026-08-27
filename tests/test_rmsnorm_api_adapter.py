"""Stage 2 Gate C.1 RMSNorm compat: hermetic tests for
`src/integration/profiling/rmsnorm_api_adapter.py`.

`_detect_rmsnorm_api` needs no `torch`/`vllm` at all (pure `hasattr`/
`inspect.signature` on whatever module object is passed in) -- tests
A/B/G exercise it directly with plain `types.ModuleType` stand-ins.
Tests C/D/E/F/H need the real `torch`/`vllm` (the target Frontier
module imports `torch` at module level) and are
`pytest.importorskip`-gated, matching this project's established
convention exactly.
"""
from __future__ import annotations

import sys
import types

import pytest

from integration.profiling import rmsnorm_api_adapter
from integration.profiling.rmsnorm_api_adapter import (
    RmsnormApiUnknownSignature,
    _detect_rmsnorm_api,
    get_rmsnorm_api_adapter_status,
)


def _reset_module_state():
    rmsnorm_api_adapter._installed = False
    rmsnorm_api_adapter._observed_api_kind = None


def _restore_patched_frontier_layernorm_module():
    if "frontier.profiling.common.layers.layernorm" not in sys.modules:
        return
    import importlib

    import frontier.profiling.common.layers.layernorm as layernorm_module

    importlib.reload(layernorm_module)


@pytest.fixture(autouse=True)
def _isolate():
    _reset_module_state()
    yield
    _reset_module_state()
    _restore_patched_frontier_layernorm_module()


# --------------------------------------------------------------- A: old API


def test_old_api_detected_when_free_functions_exist():
    fake_module = types.ModuleType("fake_vllm_layernorm")
    fake_module.rms_norm = lambda *a, **k: None
    fake_module.fused_add_rms_norm = lambda *a, **k: None
    assert _detect_rmsnorm_api(fake_module) == "old"


def test_old_api_takes_precedence_even_if_a_class_also_exists():
    """If a future vLLM release keeps both for a deprecation window,
    the old, already-working path must still be chosen -- no reason to
    touch it just because a class also exists."""
    fake_module = types.ModuleType("fake_vllm_layernorm")
    fake_module.rms_norm = lambda *a, **k: None
    fake_module.fused_add_rms_norm = lambda *a, **k: None

    class FakeRMSNormClass:
        def __init__(self, hidden_size, eps=1e-6):
            pass

        def forward_native(self, x, residual=None):
            pass

    fake_module.RMSNorm = FakeRMSNormClass
    assert _detect_rmsnorm_api(fake_module) == "old"


# ----------------------------------------------------------- B: pinned API


def test_new_api_detected_when_only_class_exists_with_expected_shape():
    fake_module = types.ModuleType("fake_vllm_layernorm")

    class FakeRMSNormClass:
        def __init__(self, hidden_size, eps=1e-6, var_hidden_size=None):
            pass

        def forward_native(self, x, residual=None):
            pass

    fake_module.RMSNorm = FakeRMSNormClass
    assert _detect_rmsnorm_api(fake_module) == "new"


# --------------------------------------------------------- G: unknown API


def test_unknown_api_raises_when_neither_functions_nor_class_exist():
    fake_module = types.ModuleType("fake_vllm_layernorm")
    with pytest.raises(RmsnormApiUnknownSignature):
        _detect_rmsnorm_api(fake_module)


def test_unknown_api_raises_when_class_init_missing_hidden_size_or_eps():
    fake_module = types.ModuleType("fake_vllm_layernorm")

    class FakeRMSNormClassBadInit:
        def __init__(self, size):  # not 'hidden_size'
            pass

        def forward_native(self, x, residual=None):
            pass

    fake_module.RMSNorm = FakeRMSNormClassBadInit
    with pytest.raises(RmsnormApiUnknownSignature, match="hidden_size"):
        _detect_rmsnorm_api(fake_module)


def test_unknown_api_raises_when_forward_native_missing():
    fake_module = types.ModuleType("fake_vllm_layernorm")

    class FakeRMSNormClassNoForwardNative:
        def __init__(self, hidden_size, eps=1e-6):
            pass

    fake_module.RMSNorm = FakeRMSNormClassNoForwardNative
    with pytest.raises(RmsnormApiUnknownSignature, match="forward_native"):
        _detect_rmsnorm_api(fake_module)


def test_unknown_api_raises_when_forward_native_missing_residual_param():
    fake_module = types.ModuleType("fake_vllm_layernorm")

    class FakeRMSNormClassBadForward:
        def __init__(self, hidden_size, eps=1e-6):
            pass

        def forward_native(self, x):  # no 'residual'
            pass

    fake_module.RMSNorm = FakeRMSNormClassBadForward
    with pytest.raises(RmsnormApiUnknownSignature, match="residual"):
        _detect_rmsnorm_api(fake_module)


def test_status_before_any_use_is_all_none_never_guessed():
    status = get_rmsnorm_api_adapter_status()
    assert status["applied"] is False
    assert status["detected_api_kind"] is None


# ----------------------------------------- C/D/E/F/H: real torch + vllm


def test_c_d_e_fused_residual_epsilon_and_weight_semantics_preserved():
    """C+D+E: constructing the patched RMSNorm and calling it both
    without and with a residual reproduces the exact two-shape contract
    Frontier's own original type hint already declared, with the real
    epsilon actually reaching the real vLLM class, and `weight` exposed
    (delegated, not duplicated) the same way the sibling GemmaRMSNorm
    wrapper already does."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import torch

    from integration.profiling.rmsnorm_api_adapter import install_rmsnorm_api_adapter
    from integration.profiling.vllm_config_context import profiling_vllm_config_context
    from vllm.config.vllm import OptimizationLevel

    install_rmsnorm_api_adapter()
    status = get_rmsnorm_api_adapter_status()
    assert status["applied"] is True

    import frontier.profiling.common.layers.layernorm as layernorm_module
    from frontier.profiling.common.timer_stats_store import TimerStatsStore

    # CudaTimer's own TimerStatsStore is a real process-wide singleton the
    # real profiling CLI initializes once at startup, before any RMSNorm
    # is constructed -- reproduce that here rather than construct RMSNorm
    # in an environment the real CLI never actually leaves it in.
    TimerStatsStore("cuda_event")

    with profiling_vllm_config_context(optimization_level=OptimizationLevel.O0):
        norm = layernorm_module.RMSNorm(hidden_size=8, eps=1e-5, norm_name=None)
        x = torch.randn(4, 8)

        out_no_residual = norm(x)
        assert isinstance(out_no_residual, torch.Tensor)
        assert out_no_residual.shape == x.shape

        residual = torch.randn(4, 8)
        out_with_residual = norm(x, residual)
        assert isinstance(out_with_residual, tuple)
        assert len(out_with_residual) == 2
        normed, updated_residual = out_with_residual
        assert normed.shape == x.shape
        assert updated_residual.shape == x.shape

        # D: the real epsilon reached the real vLLM instance, not a default.
        assert norm._impl.variance_epsilon == 1e-5
        # E: weight is delegated (same object), not duplicated.
        assert norm.weight is norm._impl.weight


def test_f_construction_requires_the_vllm_config_context():
    """F: RMSNorm (CustomOp-derived under the new API) fails to
    construct outside profiling_vllm_config_context(), exactly like
    RotaryEmbedding did -- and succeeds inside it, proving composition
    with the existing, reused mechanism rather than a new one."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    from integration.profiling.rmsnorm_api_adapter import install_rmsnorm_api_adapter

    install_rmsnorm_api_adapter()
    if get_rmsnorm_api_adapter_status()["detected_api_kind"] != "new":
        pytest.skip("this pinned vLLM still exposes the old free-function API")

    import frontier.profiling.common.layers.layernorm as layernorm_module

    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        layernorm_module.RMSNorm(hidden_size=8, eps=1e-6)


def test_h_numerical_comparison_to_simple_pytorch_rmsnorm_reference():
    """H: the patched RMSNorm's own real output matches a simple,
    independent PyTorch RMSNorm reference (same formula the class's own
    docstring declares: x -> w * x / sqrt(E[x^2] + eps)) to float
    tolerance -- not merely "runs without crashing"."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import torch

    from integration.profiling.rmsnorm_api_adapter import install_rmsnorm_api_adapter
    from integration.profiling.vllm_config_context import profiling_vllm_config_context
    from vllm.config.vllm import OptimizationLevel

    install_rmsnorm_api_adapter()

    import frontier.profiling.common.layers.layernorm as layernorm_module
    from frontier.profiling.common.timer_stats_store import TimerStatsStore

    TimerStatsStore("cuda_event")

    torch.manual_seed(0)
    hidden_size = 16
    eps = 1e-6
    x = torch.randn(3, hidden_size, dtype=torch.float32)

    with profiling_vllm_config_context(optimization_level=OptimizationLevel.O0):
        norm = layernorm_module.RMSNorm(hidden_size=hidden_size, eps=eps)
        weight = norm.weight.data.clone()
        actual = norm(x.clone())

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected = x * torch.rsqrt(variance + eps) * weight

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
