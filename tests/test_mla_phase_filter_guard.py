"""Task 53 Fix A: `_train_mla_attention_layer_models` trains every MLA
attention operator on rows regardless of phase, though a phase classifier
(`_mla_operator_phase_kind`) already exists and is used at prediction time
(`_is_mla_operator_applicable_to_batch`). This applies the same classifier
at training time.

Two kinds of test, mirroring `test_sglang_replica_scheduler_guard.py`'s own
split (task 47): the source-hash guard is tested in-process (fast, needs
only `sklearn_execution_time_predictor.py`, which imports cleanly without
`torch`); the actual filtering behaviour against real profiled data is
tested via a subprocess probe (`_mla_phase_filter_probe.py`), mirroring
`test_kv_cache_page_size_vs_memory_planner.py`'s own established pattern
(task 48) for anything that needs a real, CSV-backed Frontier predictor.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)

from integration.execution_time_predictor import mla_phase_filter
from integration.execution_time_predictor.mla_phase_filter import (
    MlaPhaseFilterSourceMismatch,
    install_mla_phase_filter,
)

FRONTIER_ROOT = Path("/work/simulation/Frontier")
_PROBE_SCRIPT = str(Path(__file__).resolve().parent / "_mla_phase_filter_probe.py")
_RESULT_MARKER = "MLA_PHASE_FILTER_PROBE_RESULT="

_FRONTIER_AVAILABLE = FRONTIER_ROOT.is_dir()
pytestmark = pytest.mark.skipif(
    not _FRONTIER_AVAILABLE,
    reason="needs Frontier checked out at /work/simulation/Frontier (ambient PYTHONPATH, "
          "not repo-pinned -- see AGENTS.md/memory)")

_ORIGINAL_TRAIN_MLA = (
    SklearnExecutionTimePredictor._train_mla_attention_layer_models
)


def _reset_guard_state():
    """Mirrors test_sglang_replica_scheduler_guard.py's own discipline:
    leave both the module-level `_installed` flag and the patched class
    attribute as they were found."""
    mla_phase_filter._installed = False
    SklearnExecutionTimePredictor._train_mla_attention_layer_models = (
        _ORIGINAL_TRAIN_MLA
    )


@pytest.fixture(autouse=True)
def _isolate():
    _reset_guard_state()
    yield
    _reset_guard_state()


def _run_probe(patched: bool) -> dict:
    args = [sys.executable, _PROBE_SCRIPT]
    if patched:
        args.append("--patched")
    proc = subprocess.run(args, capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    raise RuntimeError(
        f"probe failed (patched={patched}, exit {proc.returncode}):\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")


# ------------------------------------------------------- pre-patch behaviour


def test_unpatched_trains_every_operator_on_all_phase_rows():
    """The behaviour Fix A changes -- confirmed present before any patch is
    installed. `deepseek-v3`/mi355x/attn_tp=8 has 12 unique feature-tuple
    rows after filtering to attn_tp=8 (13 raw rows collapse to 12 once two
    profiled points that share every feature column are deduplicated by
    `_build_exact_feature_lookup`'s own groupby -- unrelated to phase). Every
    operator's exact-lookup is built from the identical, phase-unfiltered
    12 rows, including the two pure-decode operators."""
    result = _run_probe(patched=False)
    assert result == {
        "attn_mla_kv_cache_save": 12,
        "attn_mla_prefill_kv_up_proj": 12,
        "attn_mla_prefill": 12,
        "attn_mla_decode_q_latent_proj": 12,
        "attn_mla_decode": 12,
        "attn_mla_v_up_proj": 12,
    }


# -------------------------------------------------------- patched behaviour


def test_patched_trains_each_operator_on_only_its_own_phase():
    """After the patch: prefill-phase operators train on only the 4 unique
    prefill rows, decode-phase operators on only the 8 decode rows, and the
    one CACHE_WRITE operator (`attn_mla_kv_cache_save`, which genuinely
    spans both phases) is unaffected -- still all 12."""
    result = _run_probe(patched=True)
    assert result == {
        "attn_mla_kv_cache_save": 12,
        "attn_mla_prefill_kv_up_proj": 4,
        "attn_mla_prefill": 4,
        "attn_mla_decode_q_latent_proj": 8,
        "attn_mla_decode": 8,
        "attn_mla_v_up_proj": 8,
    }
    # No operator ends with zero training rows (task 53's own required check).
    assert all(count > 0 for count in result.values())


def test_install_is_idempotent():
    install_mla_phase_filter()
    patched_once = SklearnExecutionTimePredictor._train_mla_attention_layer_models
    install_mla_phase_filter()
    assert (
        SklearnExecutionTimePredictor._train_mla_attention_layer_models
        is patched_once
    )


def test_install_patches_the_expected_function():
    install_mla_phase_filter()
    assert (
        SklearnExecutionTimePredictor._train_mla_attention_layer_models
        is mla_phase_filter._patched_train_mla_attention_layer_models
    )


# -------------------------------------------------------------- source hash


def test_source_hash_guard_fires():
    """A changed upstream `_train_mla_attention_layer_models` halts install
    rather than filtering rows in a training loop this module never
    reviewed -- task 53's own required acceptance test, matching task 47's
    precedent."""
    with mock.patch.object(mla_phase_filter, "_EXPECTED_SOURCE_HASH", "not-the-real-hash"):
        with pytest.raises(MlaPhaseFilterSourceMismatch):
            install_mla_phase_filter()
    # And the real hash, unpatched, installs cleanly -- proving the guard
    # itself is what fired above, not something else broken.
    install_mla_phase_filter()
    assert (
        SklearnExecutionTimePredictor._train_mla_attention_layer_models
        is mla_phase_filter._patched_train_mla_attention_layer_models
    )
