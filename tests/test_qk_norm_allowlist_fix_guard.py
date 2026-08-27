"""Stage 2 Gate C.1: `QK_NORM_MODEL_TYPE_ALLOWLIST` is missing plain
`"qwen3"` (confirmed against HuggingFace transformers' own
`Qwen3Attention` source: `q_norm`/`k_norm` apply unconditionally for
every Qwen3 variant, dense or MoE). `install_qk_norm_allowlist_fix()`
adds it. This flag feeds the boolean `use_qk_norm` feature column on
`attn_pre_proj`'s own linear_op profile row (`linear_op_impl.py`
applies the extra RMSNorm compute at collection time when this flag is
set; the sklearn predictor then hard-filters training rows on the same
column) -- not a separate operator (that name, `attn_pre_proj_q_norm`,
belongs to `step3_text`'s own unrelated MFA profile). `frontier.config.model_config`
imports no `torch` (confirmed directly), so this is tested fully
in-process, no subprocess probe needed -- unlike the block-table fix,
which requires `torch`.
"""
from __future__ import annotations

import pytest

from frontier.config.model_config import (
    QK_NORM_MODEL_TYPE_ALLOWLIST,
    _infer_use_qk_norm_from_hf_config,
)

from integration.profiling import qk_norm_allowlist_fix
from integration.profiling.qk_norm_allowlist_fix import (
    QkNormAllowlistMismatch,
    install_qk_norm_allowlist_fix,
)


def _reset_guard_state():
    qk_norm_allowlist_fix._installed = False
    QK_NORM_MODEL_TYPE_ALLOWLIST.discard("qwen3")


@pytest.fixture(autouse=True)
def _isolate():
    _reset_guard_state()
    yield
    _reset_guard_state()


# --------------------------------------------------------------- pre-patch


def test_unpatched_plain_qwen3_is_not_detected():
    """The real, verified gap: Qwen3-0.6B's own pinned config
    (model_type="qwen3", architectures=["Qwen3ForCausalLM"], no explicit
    use_qk_norm key) is not recognized before the fix."""
    cfg = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    assert _infer_use_qk_norm_from_hf_config(cfg) is False


def test_unpatched_qwen3_moe_and_qwen3_next_are_already_detected():
    assert _infer_use_qk_norm_from_hf_config({"model_type": "qwen3_moe"}) is True
    assert _infer_use_qk_norm_from_hf_config({"model_type": "qwen3_next"}) is True


def test_explicit_use_qk_norm_field_always_wins_even_unpatched():
    assert _infer_use_qk_norm_from_hf_config(
        {"model_type": "qwen3", "use_qk_norm": True}) is True
    assert _infer_use_qk_norm_from_hf_config(
        {"model_type": "qwen3_moe", "use_qk_norm": False}) is False


# ---------------------------------------------------------------- patched


def test_patched_plain_qwen3_is_detected():
    install_qk_norm_allowlist_fix()
    cfg = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    assert _infer_use_qk_norm_from_hf_config(cfg) is True


def test_patched_does_not_disturb_the_two_existing_entries():
    install_qk_norm_allowlist_fix()
    assert _infer_use_qk_norm_from_hf_config({"model_type": "qwen3_moe"}) is True
    assert _infer_use_qk_norm_from_hf_config({"model_type": "qwen3_next"}) is True


def test_patched_does_not_affect_unrelated_model_types():
    install_qk_norm_allowlist_fix()
    assert _infer_use_qk_norm_from_hf_config({"model_type": "llama"}) is False
    assert _infer_use_qk_norm_from_hf_config({"model_type": "deepseek_v3"}) is False


def test_install_is_idempotent():
    install_qk_norm_allowlist_fix()
    contents_once = frozenset(QK_NORM_MODEL_TYPE_ALLOWLIST)
    install_qk_norm_allowlist_fix()
    assert frozenset(QK_NORM_MODEL_TYPE_ALLOWLIST) == contents_once


# ------------------------------------------------------------- data guard


def test_guard_fires_if_allowlist_already_contains_qwen3():
    """A future upstream change that already added "qwen3" itself (under
    some different assumption) must not be silently overwritten/assumed
    -- refuse rather than double-add or misjudge intent."""
    QK_NORM_MODEL_TYPE_ALLOWLIST.add("qwen3")
    with pytest.raises(QkNormAllowlistMismatch):
        install_qk_norm_allowlist_fix()


def test_guard_fires_if_an_expected_entry_is_missing():
    QK_NORM_MODEL_TYPE_ALLOWLIST.discard("qwen3_next")
    try:
        with pytest.raises(QkNormAllowlistMismatch):
            install_qk_norm_allowlist_fix()
    finally:
        QK_NORM_MODEL_TYPE_ALLOWLIST.add("qwen3_next")
