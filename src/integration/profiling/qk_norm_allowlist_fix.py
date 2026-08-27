"""Stage 2 Gate C.1: `frontier.config.model_config.QK_NORM_MODEL_TYPE_ALLOWLIST`
is missing plain `"qwen3"` -- a real, verified gap found while scoping a
profiling plan for Qwen3-0.6B, before any GPU time was spent.

`_infer_use_qk_norm_from_hf_config` (`frontier/config/model_config.py`)
auto-detects whether a model needs its QK-norm compute included in the
`attn_pre_proj` operator's own timing, via the boolean `use_qk_norm`
feature column already present on every real `linear_op.csv`
(`linear_op_impl.py` applies the extra RMSNorm compute during profiling
when `getattr(config, "use_qk_norm", False)` is true; the sklearn
predictor then hard-filters training rows on this same flag,
`sklearn_execution_time_predictor.py` lines ~1125-1157). There is no
separate `attn_pre_proj_q_norm` operator/column for this path -- that
name belongs to `step3_text`'s own distinct MFA linear-attention profile
(`frontier/model_architectures.py`), confirmed not to apply here; a
prior draft of this docstring conflated the two. `use_qk_norm` is
inferred from a model's own real HF `config.json`:
an explicit `use_qk_norm` field if present, else membership in
`QK_NORM_MODEL_TYPE_ALLOWLIST` (`{"qwen3_moe", "qwen3_next"}`), else an
`"qwen3next"` substring match on `architectures`. Qwen3-0.6B's own real,
pinned config (`model_type="qwen3"`, `architectures=["Qwen3ForCausalLM"]`,
no `use_qk_norm` key) matches none of these -- `_infer_use_qk_norm_from_hf_config`
would return `False` for it.

**Confirmed wrong against the real, authoritative source**: HuggingFace
`transformers`' own `modeling_qwen3.py` (`Qwen3Attention.__init__`/`.forward`,
fetched live from `main` for this check) applies `self.q_norm`/`self.k_norm`
(`Qwen3RMSNorm`) to every Q/K projection *unconditionally* -- the same
attention module dense and MoE Qwen3 variants both use. The allowlist's
own two entries (`qwen3_moe`, `qwen3_next`) are real but incomplete --
plain dense Qwen3 (`model_type="qwen3"`) needs the exact same treatment
and does not get it.

Left unfixed, this bug would corrupt *collection*, not only prediction:
`linear_op_impl.py` would profile `attn_pre_proj` *without* actually
running the QK-norm RMSNorm compute (since the model config it reads
would carry `use_qk_norm=False`), so the real GPU measurement itself
would be too fast -- re-collecting after the fact would not fix already-
collected rows. This must be installed and confirmed active *before*
the profiling CLI runs, not merely before evaluation -- exactly the
kind of structural coverage gap this project's own Task 52/53 already
found elsewhere, caught here before any profiling GPU time was spent
rather than after.

A data mutation, not a function replacement (`QK_NORM_MODEL_TYPE_ALLOWLIST`
is a plain, module-level, mutable `set` that `_infer_use_qk_norm_from_hf_config`
reads by name at call time -- adding to it changes every caller's
behavior without touching the function's own source). Guarded by an
exact-contents check of the allowlist before mutating, matching this
project's own established pattern (task 20/47/53) of refusing to patch
over an implementation this project hasn't reviewed, adapted to a data
guard since there is no function body to hash here.
"""
from __future__ import annotations

from frontier.config.model_config import QK_NORM_MODEL_TYPE_ALLOWLIST

_EXPECTED_CURRENT_CONTENTS = frozenset({"qwen3_moe", "qwen3_next"})
_ADDED_MODEL_TYPES = frozenset({"qwen3"})

_installed = False


class QkNormAllowlistMismatch(RuntimeError):
    pass


def install_qk_norm_allowlist_fix() -> None:
    """Adds `"qwen3"` to `QK_NORM_MODEL_TYPE_ALLOWLIST`. Safe to call more
    than once (idempotent). Not called by `install()` by default -- a
    caller must ask for it explicitly, the same way every other patch in
    this project is opt-in.

    Raises `QkNormAllowlistMismatch` if the allowlist's own current
    contents don't match what this module was written against (either
    already containing `"qwen3"` under a different assumption, or missing
    one of the two entries this module expects to find) -- refusing to
    silently add to an allowlist whose own meaning may have changed
    upstream in the meantime.
    """
    global _installed
    if _installed:
        return
    current = frozenset(QK_NORM_MODEL_TYPE_ALLOWLIST)
    if current != _EXPECTED_CURRENT_CONTENTS:
        raise QkNormAllowlistMismatch(
            f"QK_NORM_MODEL_TYPE_ALLOWLIST's current contents {sorted(current)} "
            f"no longer match what this module was written against "
            f"{sorted(_EXPECTED_CURRENT_CONTENTS)} -- refusing to add "
            f"{sorted(_ADDED_MODEL_TYPES)} to it without review.")
    QK_NORM_MODEL_TYPE_ALLOWLIST.update(_ADDED_MODEL_TYPES)
    _installed = True
