"""Task 53 Fix A: filter MLA attention training rows to each operator's own
declared phase before fitting.

`_train_mla_attention_layer_models`
(`frontier/execution_time_predictor/sklearn_execution_time_predictor.py`)
drops rows with a NaN target but never filters by phase, so every operator
-- other than the one `CACHE_WRITE` operator that genuinely spans both --
trains on its own real measurements *plus* the other phase's rows, whose
value in that same column is measurement noise (Task 52's own finding:
a prefill-phase row records a noise-floor value for `attn_mla_decode.median`,
not a real decode timing, since the profiling wrapper times all six MLA
scopes on every row regardless of which one the row's own phase exercises).
Task 52 measured the cost directly: leave-one-out error for `attn_mla_decode`
falls from 177.97% to 2.43% once those rows are excluded
(`docs/tasks/52-predictor-error-report.md`).

The discriminator this patch applies is not new. `SklearnExecutionTimePredictor`
already has one -- `_mla_operator_phase_kind` -- used at *prediction* time
(`_is_mla_operator_applicable_to_batch`) to decide whether an operator's cost
even applies to a given batch, and it already raises if any MLA operator's
own declared `phases` don't cleanly resolve to exactly one of
`"cache_write"` / `"prefill"` / `"decode"` (Task 53's own S3.1 check: this
patch calls that exact classifier rather than re-deriving one, so any future
MLA operator with an ambiguous phase declaration fails the *same* way it
already would at prediction time, not silently). This patch reuses that
classification at *training* time, the one place it was missing.

Guarded by a source hash over `_train_mla_attention_layer_models`, following
`..cc_backend.collective`'s (task 20) and `..replica_scheduler.sglang_guard`'s
(task 47) established pattern -- runtime-patch a pinned Frontier method,
guarded, rather than edit the checkout.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Dict

from sklearn.base import BaseEstimator

from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_names,
    get_enabled_shared_predictor_feature_columns,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
    _build_exact_feature_lookup,
)

# Computed against the checked-out Frontier's own
# SklearnExecutionTimePredictor._train_mla_attention_layer_models
# (frontier/execution_time_predictor/sklearn_execution_time_predictor.py) at
# the time this module was written. A changed hash means the method's own
# body changed upstream -- install_mla_phase_filter() raises rather than
# patch over an unknown implementation.
_EXPECTED_SOURCE_HASH = "d503aecab292f3ba5111da4edf04b2f674877f9fbc70e04ad600b28cd5d2ad1c"

_installed = False


class MlaPhaseFilterSourceMismatch(RuntimeError):
    pass


def _patched_train_mla_attention_layer_models(self) -> Dict[str, BaseEstimator]:
    attention_df = self._load_attention_df(self._attention_input_file)
    attention_df = self._get_attention_df_with_derived_features(attention_df)

    model_names = list(get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY))
    target_columns = dict(
        zip(
            model_names,
            get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY),
        )
    )
    feature_columns = get_enabled_shared_predictor_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    )

    models: Dict[str, BaseEstimator] = {}
    for model_name in model_names:
        feature_cols = list(feature_columns[model_name])
        target_col = target_columns[model_name]
        required_columns = [*feature_cols, target_col]
        missing_columns = [
            column for column in required_columns if column not in attention_df.columns
        ]
        all_nan_columns = [
            column
            for column in required_columns
            if column in attention_df.columns and attention_df[column].isna().all()
        ]
        if missing_columns or all_nan_columns:
            raise ValueError(
                "MLA attention profiling data cannot train "
                f"{model_name}."
                f"\nMissing columns: {missing_columns}"
                f"\nAll-NaN columns: {all_nan_columns}"
            )
        op_attention_df = attention_df.dropna(subset=[target_col]).copy()

        # Task 53 Fix A -- the filter missing from the original. `is_prefill`
        # is coerced to int (0/1) by `_get_attention_df_with_derived_features`
        # for the MLA family (`coerce_truthy_int`, called above), before this
        # patch ever sees the dataframe.
        phase_kind = SklearnExecutionTimePredictor._mla_operator_phase_kind(model_name)
        if phase_kind == "prefill":
            op_attention_df = op_attention_df[op_attention_df["is_prefill"] == 1]
        elif phase_kind == "decode":
            op_attention_df = op_attention_df[op_attention_df["is_prefill"] == 0]
        elif phase_kind != "cache_write":
            raise ValueError(f"Unsupported MLA operator phase kind: {phase_kind}")

        if op_attention_df.empty:
            raise ValueError(
                "MLA attention profiling data cannot train "
                f"{model_name}: target column {target_col!r} has no "
                "observed timing rows for this operator's own phase "
                f"({phase_kind!r})."
            )
        nan_feature_columns = [
            column
            for column in feature_cols
            if op_attention_df[column].isna().any()
        ]
        if nan_feature_columns:
            raise ValueError(
                "MLA attention profiling data cannot train "
                f"{model_name}: feature columns contain NaN after "
                f"target filtering: {nan_feature_columns}"
            )

        model = self._train_model(
            model_name=model_name,
            df=op_attention_df,
            feature_cols=feature_cols,
            target_col=target_col,
        )
        model._frontier_exact_lookup = _build_exact_feature_lookup(
            op_attention_df,
            feature_cols,
            target_col,
        )
        models[model_name] = model
    return models


def install_mla_phase_filter() -> None:
    """Patch `SklearnExecutionTimePredictor._train_mla_attention_layer_models`
    to restrict each MLA operator's training rows to the phase(s) its own
    family spec declares. Safe to call more than once (idempotent). Not
    called by `install()` by default -- a caller must ask for it explicitly,
    the same way `collective=True` and `sglang_replica_scheduler=True` are
    never implied.

    Raises `MlaPhaseFilterSourceMismatch` if
    `_train_mla_attention_layer_models`'s source no longer matches what this
    module was written against.
    """
    global _installed
    if _installed:
        return
    current_hash = hashlib.sha256(
        inspect.getsource(
            SklearnExecutionTimePredictor._train_mla_attention_layer_models
        ).encode()
    ).hexdigest()
    if current_hash != _EXPECTED_SOURCE_HASH:
        raise MlaPhaseFilterSourceMismatch(
            f"SklearnExecutionTimePredictor._train_mla_attention_layer_models's "
            f"source has changed (hash {current_hash} != expected "
            f"{_EXPECTED_SOURCE_HASH}). Refusing to install the phase-filter "
            f"patch over an implementation this project hasn't reviewed -- "
            f"update _EXPECTED_SOURCE_HASH in {__name__} only after confirming "
            f"the training loop being filtered is still the same one, and "
            f"that no phase-aware filtering was added upstream in the "
            f"meantime.")
    SklearnExecutionTimePredictor._train_mla_attention_layer_models = (
        _patched_train_mla_attention_layer_models
    )
    _installed = True
