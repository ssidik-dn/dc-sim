"""Stage 2 Gate C.1 ROPE compat: hermetic tests for
`src/integration/profiling/rope_api_adapter.py`.

`_detect_and_build_adapter` needs no `torch` at all (pure
`inspect.signature` + dict logic) -- tests A/B/C exercise it directly
with plain-Python stand-in callables, no mocking framework, no skip.
Tests D/E (`install_rope_api_adapter()` itself, and a real Qwen3-0.6B
config run through the *actual* Frontier `get_rope()`) need `torch`
(`frontier.profiling.common.layers.rotary_embedding` imports it at
module level) and are skipped, not failed, where absent -- matching
`tests/test_attention_block_table_fix_guard.py`'s own established
convention exactly.
"""
from __future__ import annotations

import sys

import pytest

from integration.profiling import rope_api_adapter
from integration.profiling.rope_api_adapter import (
    RopeApiUnknownSignature,
    _detect_and_build_adapter,
)


def _reset_module_state():
    rope_api_adapter._installed = False
    rope_api_adapter._observed_api_kind = None
    rope_api_adapter._observed_signature = None
    rope_api_adapter._observed_vllm_version = None


def _restore_patched_frontier_rotary_module():
    """`install_rope_api_adapter()` mutates
    `frontier.profiling.common.layers.rotary_embedding._load_vllm_get_rope`
    (a real, process-global, cross-test attribute on Frontier's own
    module) -- resetting only this module's own `_installed` flag
    leaves that mutation in place for every later test in the same
    process, real, live-caught: `install_rope_api_adapter()`'s own
    hash guard hashes *whatever function object is currently bound* to
    `_load_vllm_get_rope`, which after one test's own successful
    install is this module's own wrapper closure, not Frontier's real
    original -- a later test's own install call then sees a "changed"
    hash and fails for a reason having nothing to do with its own
    behavior. Restores Frontier's real original from a fresh
    `importlib.reload`, and only if that module has ever actually been
    imported (skipped entirely, correctly, in this sandbox where
    `torch` -- and therefore this module -- is never importable at
    all)."""
    if "frontier.profiling.common.layers.rotary_embedding" not in sys.modules:
        return
    import importlib

    import frontier.profiling.common.layers.rotary_embedding as rope_module

    importlib.reload(rope_module)


@pytest.fixture(autouse=True)
def _isolate():
    _reset_module_state()
    yield
    _reset_module_state()
    _restore_patched_frontier_rotary_module()


# --------------------------------------------------------------- A: old API


def test_old_api_selected_and_rotary_dim_passed_through_unchanged():
    """Mocks the pre-existing vLLM shape Frontier's own get_rope() was
    written against. No adaptation should occur -- every argument must
    reach the mock exactly as Frontier's own three real call sites
    already send it (this IS the "no regression for the old-API path"
    check, §7.E, folded in here since the assertion is identical)."""
    received = {}

    def fake_old_get_rope(head_size, rotary_dim, max_position, base,
                           is_neox_style, rope_scaling, dtype=None):
        received.update(
            head_size=head_size, rotary_dim=rotary_dim, max_position=max_position,
            base=base, is_neox_style=is_neox_style, rope_scaling=rope_scaling,
            dtype=dtype,
        )
        return "OLD_ROTARY_EMBEDDING_OBJECT"

    api_kind, adapter, sig = _detect_and_build_adapter(fake_old_get_rope)
    assert api_kind == "old"

    result = adapter(
        head_size=128, rotary_dim=128, max_position=40960, base=1000000.0,
        is_neox_style=True, rope_scaling=None, dtype="bf16-sentinel",
    )
    assert result == "OLD_ROTARY_EMBEDDING_OBJECT"
    assert received == {
        "head_size": 128, "rotary_dim": 128, "max_position": 40960,
        "base": 1000000.0, "is_neox_style": True, "rope_scaling": None,
        "dtype": "bf16-sentinel",
    }


def test_old_api_preserves_a_real_rope_scaling_dict_verbatim():
    received = {}

    def fake_old_get_rope(head_size, rotary_dim, max_position, base,
                           is_neox_style, rope_scaling, dtype=None):
        received["rope_scaling"] = rope_scaling
        return None

    _, adapter, _ = _detect_and_build_adapter(fake_old_get_rope)
    scaling = {"rope_type": "linear", "factor": 4.0}
    adapter(head_size=64, rotary_dim=64, max_position=8192, base=10000.0,
            is_neox_style=True, rope_scaling=scaling, dtype=None)
    assert received["rope_scaling"] == scaling


# ----------------------------------------------------------- B: pinned API


def _fake_pinned_get_rope_factory():
    received = {}

    def fake_pinned_get_rope(head_size, max_position, is_neox_style=True,
                              rope_parameters=None, dtype=None,
                              dual_chunk_attention_config=None):
        received.update(
            head_size=head_size, max_position=max_position,
            is_neox_style=is_neox_style, rope_parameters=rope_parameters,
            dtype=dtype, dual_chunk_attention_config=dual_chunk_attention_config,
        )
        return "PINNED_ROTARY_EMBEDDING_OBJECT"

    return fake_pinned_get_rope, received


def test_pinned_api_selected_and_obsolete_args_not_passed():
    fake_pinned_get_rope, received = _fake_pinned_get_rope_factory()
    api_kind, adapter, sig = _detect_and_build_adapter(fake_pinned_get_rope)
    assert api_kind == "new"

    result = adapter(
        head_size=128, rotary_dim=128, max_position=40960, base=1000000.0,
        is_neox_style=True, rope_scaling=None, dtype="bf16-sentinel",
    )
    assert result == "PINNED_ROTARY_EMBEDDING_OBJECT"
    # rotary_dim/base must never reach the pinned callable as top-level args
    # (it doesn't even accept them -- if the adapter tried, this would raise
    # TypeError before we got here at all, but assert the received dict too).
    assert "rotary_dim" not in received
    assert "base" not in received


def test_pinned_api_injects_rope_theta_and_rope_dim_correctly():
    fake_pinned_get_rope, received = _fake_pinned_get_rope_factory()
    _, adapter, _ = _detect_and_build_adapter(fake_pinned_get_rope)

    adapter(head_size=128, rotary_dim=128, max_position=40960, base=1000000.0,
            is_neox_style=True, rope_scaling=None, dtype="bf16-sentinel")

    assert received["rope_parameters"] == {"rope_theta": 1000000.0, "rope_dim": 128}
    assert received["head_size"] == 128
    assert received["max_position"] == 40960
    assert received["is_neox_style"] is True
    assert received["dtype"] == "bf16-sentinel"
    assert received["dual_chunk_attention_config"] is None


def test_pinned_api_merges_real_scaling_dict_and_preserves_its_keys():
    fake_pinned_get_rope, received = _fake_pinned_get_rope_factory()
    _, adapter, _ = _detect_and_build_adapter(fake_pinned_get_rope)

    scaling = {"rope_type": "linear", "factor": 4.0}
    adapter(head_size=64, rotary_dim=64, max_position=8192, base=10000.0,
            is_neox_style=True, rope_scaling=scaling, dtype=None)

    assert received["rope_parameters"] == {
        "rope_type": "linear", "factor": 4.0, "rope_theta": 10000.0, "rope_dim": 64,
    }


def test_pinned_api_general_translation_holds_when_rotary_dim_ne_head_size():
    """No current Frontier call site ever passes rotary_dim != head_size
    (partial rotary), but the translation itself (`rope_dim` is respected
    verbatim by the real source ahead of `partial_rotary_factor`) is
    general -- prove it holds for a hypothetical future caller too,
    rather than silently relying on today's coincidence."""
    fake_pinned_get_rope, received = _fake_pinned_get_rope_factory()
    _, adapter, _ = _detect_and_build_adapter(fake_pinned_get_rope)

    adapter(head_size=128, rotary_dim=64, max_position=4096, base=500000.0,
            is_neox_style=True, rope_scaling=None, dtype=None)
    assert received["rope_parameters"]["rope_dim"] == 64
    assert received["head_size"] == 128


def test_pinned_api_conflicting_rope_scaling_key_raises_rather_than_silently_overriding():
    fake_pinned_get_rope, _ = _fake_pinned_get_rope_factory()
    _, adapter, _ = _detect_and_build_adapter(fake_pinned_get_rope)

    with pytest.raises(RopeApiUnknownSignature):
        adapter(
            head_size=128, rotary_dim=128, max_position=40960, base=1000000.0,
            is_neox_style=True, rope_scaling={"rope_type": "default", "rope_theta": 999.0},
            dtype=None,
        )


def test_pinned_api_without_dual_chunk_param_omits_it():
    """A pinned callable that doesn't even declare
    dual_chunk_attention_config (an older but still rope_parameters-based
    signature) must not have it injected -- checked against the real
    detected signature, not assumed present."""
    received = {}

    def fake_pinned_get_rope_no_dual_chunk(head_size, max_position,
                                            is_neox_style=True,
                                            rope_parameters=None, dtype=None):
        received.update(head_size=head_size, max_position=max_position)
        return None

    _, adapter, _ = _detect_and_build_adapter(fake_pinned_get_rope_no_dual_chunk)
    adapter(head_size=32, rotary_dim=32, max_position=2048, base=10000.0,
            is_neox_style=True, rope_scaling=None, dtype=None)
    # No TypeError raised despite the callable not accepting the kwarg --
    # proves it was correctly omitted, not just coincidentally unused.
    assert received == {"head_size": 32, "max_position": 2048}


# ----------------------------------------------------------- C: unknown API


def test_unknown_api_raises_explicit_error_never_silently_proceeds():
    def fake_unknown_get_rope(positions, embedding_dim):
        return None  # pragma: no cover -- must never be reachable

    with pytest.raises(RopeApiUnknownSignature):
        _detect_and_build_adapter(fake_unknown_get_rope)


def test_unknown_api_error_names_the_real_observed_signature():
    def fake_unknown_get_rope(positions, embedding_dim):
        return None

    with pytest.raises(RopeApiUnknownSignature, match="positions"):
        _detect_and_build_adapter(fake_unknown_get_rope)


def test_a_signature_with_both_old_and_new_markers_is_treated_as_old_not_guessed():
    """A hypothetical transitional signature carrying both `rotary_dim`
    and `rope_parameters` doesn't match either guarded shape's exact
    exclusion condition (old requires *no* rope_parameters; new requires
    *no* rotary_dim/base) -- must hard-fail, not silently pick one."""
    def fake_transitional_get_rope(head_size, rotary_dim, base, max_position,
                                    is_neox_style=True, rope_parameters=None):
        return None

    with pytest.raises(RopeApiUnknownSignature):
        _detect_and_build_adapter(fake_transitional_get_rope)


# --------------------------------------------------- D/E: real torch/vLLM


def _install_with_fake_vllm_module(monkeypatch, fake_get_rope):
    import types
    fake_pkg = types.ModuleType("vllm")
    fake_model_executor = types.ModuleType("vllm.model_executor")
    fake_layers = types.ModuleType("vllm.model_executor.layers")
    fake_rotary = types.ModuleType("vllm.model_executor.layers.rotary_embedding")
    fake_rotary.get_rope = fake_get_rope
    fake_pkg.__version__ = "0.27.1-fake-for-test"
    monkeypatch.setitem(sys.modules, "vllm", fake_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", fake_model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", fake_layers)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.rotary_embedding", fake_rotary
    )


def test_install_guard_fires_on_source_drift(monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    import frontier.profiling.common.layers.rotary_embedding as rope_module

    monkeypatch.setattr(
        rope_api_adapter, "_EXPECTED_LOAD_VLLM_GET_ROPE_HASH", "deliberately-wrong-hash"
    )
    with pytest.raises(rope_api_adapter.RopeApiAdapterSourceMismatch):
        rope_api_adapter.install_rope_api_adapter()
    # Refused to install -- Frontier's own loader must be untouched.
    assert rope_module._load_vllm_get_rope.__module__ != rope_api_adapter.__name__


def test_install_and_real_qwen3_0_6b_config_produce_correct_rope_parameters(monkeypatch):
    """D: the actual Qwen3-0.6B config values (real, HF-verified, pinned
    revision c1899de289a04d12100db370d81485cdf75e47ca) flow through the
    real, unmodified Frontier `get_rope()` and the installed adapter,
    landing on the pinned API's own `rope_parameters` shape with the
    exact real theta/dim -- not a placeholder."""
    pytest.importorskip("torch")
    received = {}

    def fake_pinned_get_rope(head_size, max_position, is_neox_style=True,
                              rope_parameters=None, dtype=None,
                              dual_chunk_attention_config=None):
        received.update(head_size=head_size, max_position=max_position,
                         is_neox_style=is_neox_style, rope_parameters=rope_parameters)
        return "REAL_QWEN3_ROTARY_EMBEDDING"

    _install_with_fake_vllm_module(monkeypatch, fake_pinned_get_rope)

    import frontier.profiling.common.layers.rotary_embedding as rope_module
    monkeypatch.setattr(rope_module, "_VLLM_GET_ROPE", None)
    monkeypatch.setattr(rope_module, "_VLLM_GET_ROPE_IMPORT_ERROR", None)

    rope_api_adapter.install_rope_api_adapter()

    # Real Qwen3-0.6B fields (config.json @ the pinned revision, already
    # verified live against HF transformers earlier in this initiative).
    qwen3_head_dim = 128
    qwen3_max_position_embeddings = 40960
    qwen3_rope_theta = 1000000.0
    qwen3_is_neox_style = True
    qwen3_rope_scaling = None  # no rope_scaling key in the real HF config

    result = rope_module.get_rope(
        qwen3_head_dim,
        rotary_dim=qwen3_head_dim,  # Frontier's own real call-site convention
        max_position=qwen3_max_position_embeddings,
        base=qwen3_rope_theta,
        is_neox_style=qwen3_is_neox_style,
        rope_scaling=qwen3_rope_scaling,
    )

    assert result == "REAL_QWEN3_ROTARY_EMBEDDING"
    assert received["head_size"] == 128
    assert received["max_position"] == 40960
    assert received["is_neox_style"] is True
    assert received["rope_parameters"] == {"rope_theta": 1000000.0, "rope_dim": 128}

    status = rope_api_adapter.get_rope_api_adapter_status()
    assert status["applied"] is True
    assert status["detected_api_kind"] == "new"
    assert status["detected_vllm_version"] == "0.27.1-fake-for-test"
