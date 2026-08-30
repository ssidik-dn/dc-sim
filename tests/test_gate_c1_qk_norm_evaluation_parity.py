"""Stage 2 Gate C.1 regression: collection and evaluation must agree on
the model semantics that determine which profile rows are compatible.

The real, confirmed failure this guards against (docs/tasks/66-...): every
real profiling *collection* invocation for Qwen3-0.6B applied
`qk_norm_allowlist_fix`, so every installed `linear_op.csv` row carries
`use_qk_norm=True`. `tools/planner.py::_run_scenario`'s own `install()`
call did not apply the same fix, so Frontier's real evaluation path
inferred `use_qk_norm=False` for the same model -- `BaseModelConfig`
resolves this from `frontier.config.model_config.QK_NORM_MODEL_TYPE_ALLOWLIST`,
a single shared, mutable set both paths read, not a planner-local
constant -- and the predictor's exact-match filter
(`SharedPredictionModelManager._load_linear_op_df`,
`frontier/execution_time_predictor/shared_prediction_model_manager.py`
lines ~2279-2306) then rejected every row for every TP identically.

This file does not re-implement that filter or hard-code Qwen3's
`use_qk_norm` value; it proves the two integration points that must stay
in lockstep for *any* model the allowlist fix ever covers:

1. `tools/planner.py::_run_scenario` wires the same `install()` flag
   every real collection invocation used (static wiring guard).
2. Enabling that flag through the real `install()` entry point -- not a
   private helper -- makes `BaseModelConfig`'s own resolution agree with
   the installed profile's own recorded metadata.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import planner  # noqa: E402  (tools/planner.py)

from frontier.config.model_config import BaseModelConfig, QK_NORM_MODEL_TYPE_ALLOWLIST  # noqa: E402
from integration.install import install  # noqa: E402
from integration.profiling import qk_norm_allowlist_fix as _qk_norm_allowlist_fix  # noqa: E402

MODEL_NAME = "Qwen3-0.6B"

INSTALLED_LINEAR_OP_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "Frontier" / "data" / "profiling" / "compute" / "mi355x" / MODEL_NAME / "linear_op.csv"
)


def _reset_allowlist_guard_state():
    _qk_norm_allowlist_fix._installed = False
    QK_NORM_MODEL_TYPE_ALLOWLIST.discard("qwen3")


@pytest.fixture(autouse=True)
def _isolate():
    _reset_allowlist_guard_state()
    yield
    _reset_allowlist_guard_state()


def test_run_scenario_wires_the_same_qk_norm_allowlist_fix_collection_used():
    """Static wiring guard for the exact regression: `_run_scenario`'s own
    `install()` call must pass `qk_norm_allowlist_fix=True`, the same flag
    every real Qwen3-0.6B collection invocation applied. Without this, the
    two paths silently diverge again the next time a QK-norm model is
    evaluated."""
    source = inspect.getsource(planner._run_scenario)
    assert "qk_norm_allowlist_fix=True" in source


def test_evaluation_install_path_matches_collection_use_qk_norm_semantics(monkeypatch):
    """`tools/planner.py`'s own real `install()` entry point (not the
    private fix module directly) must resolve the same `use_qk_norm` value
    collection used for Qwen3-0.6B: `True`. `BaseModelConfig.create_from_name`
    resolves its model JSON relative to cwd -- real evaluation always runs
    with `cwd=FRONTIER_ROOT` (`tools/planner.py::evaluate`'s own subprocess
    call), so this test does the same rather than relying on whatever cwd
    pytest happened to start in."""
    frontier_root = planner.FRONTIER_ROOT
    if not (frontier_root / "data" / "config" / "models" / f"{MODEL_NAME}.json").exists():
        pytest.skip(f"{MODEL_NAME}.json not installed in this checkout's Frontier root")
    monkeypatch.chdir(frontier_root)

    before = BaseModelConfig.create_from_name(MODEL_NAME).use_qk_norm
    assert before is False  # the real, confirmed pre-fix gap (docs/tasks/66-...)

    install(None, None, None, None, qk_norm_allowlist_fix=True)

    after = BaseModelConfig.create_from_name(MODEL_NAME).use_qk_norm
    assert after is True


def test_installed_profile_metadata_matches_the_fixed_evaluation_semantics(monkeypatch):
    """The installed profile's own recorded `use_qk_norm` column (from real
    collection) must equal what the fixed evaluation path now infers."""
    if not INSTALLED_LINEAR_OP_CSV.exists():
        pytest.skip("installed Qwen3-0.6B mi355x linear_op.csv not present in this checkout")

    monkeypatch.chdir(planner.FRONTIER_ROOT)
    install(None, None, None, None, qk_norm_allowlist_fix=True)
    evaluation_use_qk_norm = BaseModelConfig.create_from_name(MODEL_NAME).use_qk_norm

    df = pd.read_csv(INSTALLED_LINEAR_OP_CSV)
    assert "use_qk_norm" in df.columns
    installed_values = set(df["use_qk_norm"].astype(bool).unique())
    assert installed_values == {evaluation_use_qk_norm}
