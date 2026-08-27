"""Stage 2 Gate C.1 (follow-up): hermetic tests for the exact
`effective_tokens` key-space derivation and pre-execution coverage check
(`tools/stage2/gate_c1_coverage.py`).

No GPU, no real profile CSV required for most of this file -- the
derivation itself is pure math over `Workload`/scheduler constants.
`test_verify_gate_c_linear_op_coverage_reads_a_real_csv` is the one
exception: it reads the real, already-collected `Llama-2-7b-hf`
`linear_op.csv` on `mi355x` (present in this checkout) purely to prove
`read_profiled_num_tokens` parses a real file's real column, not a
synthetic fixture -- it is not asserting Llama coverage is sufficient
for anything.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from planner_core import Workload  # noqa: E402
from stage2.gate_c1_coverage import (  # noqa: E402
    derive_decode_effective_tokens,
    derive_linear_op_required_points,
    derive_prefill_effective_tokens,
    missing_keys,
    read_profiled_num_tokens,
    unused_keys,
    verify_gate_c_linear_op_coverage,
)

# Gate C's own real, frozen workload (docs/stage-2-gate-c-planner-handoff-report.md):
GATE_C_WORKLOAD = Workload(num_requests=32, qps=4.0, prefill_tokens=5, decode_tokens=32)
GATE_C_ATTN_TP_VALUES = (1, 2, 4)  # Gate C's own real candidates' union of TP degrees
MAX_TOKENS_IN_BATCH = 4096  # tools/planner.py's own real `_argv()` literal


def test_derive_prefill_effective_tokens_is_exact_multiples_of_prefill_tokens():
    keys = derive_prefill_effective_tokens(GATE_C_WORKLOAD, MAX_TOKENS_IN_BATCH)
    assert keys == frozenset(5 * k for k in range(1, 33))
    assert min(keys) == 5
    assert max(keys) == 160
    assert len(keys) == 32


def test_derive_prefill_effective_tokens_bounded_by_num_requests_not_token_budget():
    """4096 // 5 = 819, far more than num_requests=32 -- num_requests is
    the real binding constraint, confirmed by the derived set's own size."""
    keys = derive_prefill_effective_tokens(GATE_C_WORKLOAD, MAX_TOKENS_IN_BATCH)
    assert len(keys) == GATE_C_WORKLOAD.num_requests


def test_derive_prefill_effective_tokens_bounded_by_token_budget_when_tighter():
    tiny_budget_workload = Workload(num_requests=32, qps=4.0, prefill_tokens=5, decode_tokens=32)
    keys = derive_prefill_effective_tokens(tiny_budget_workload, max_tokens_in_batch=17)
    # floor(17/5) = 3, tighter than num_requests=32
    assert keys == frozenset({5, 10, 15})


def test_derive_decode_effective_tokens_is_contiguous_one_to_num_requests():
    keys = derive_decode_effective_tokens(GATE_C_WORKLOAD)
    assert keys == frozenset(range(1, 33))
    assert len(keys) == 32
    # Contiguous, not sparse -- every integer must be present, not merely bracketed.
    assert 7 in keys and 13 in keys and 31 in keys


def test_derive_linear_op_required_points_tp1_is_union_of_both_shapes():
    required = derive_linear_op_required_points(
        GATE_C_WORKLOAD, GATE_C_ATTN_TP_VALUES, MAX_TOKENS_IN_BATCH
    )
    prefill = derive_prefill_effective_tokens(GATE_C_WORKLOAD, MAX_TOKENS_IN_BATCH)
    decode = derive_decode_effective_tokens(GATE_C_WORKLOAD)
    assert required[1] == (prefill | decode)
    assert len(required[1]) == 58  # 32 + 32 - 6 overlapping multiples of 5 that are also <=32


def test_derive_linear_op_required_points_tp_gt_1_is_decode_shaped_only():
    required = derive_linear_op_required_points(
        GATE_C_WORKLOAD, GATE_C_ATTN_TP_VALUES, MAX_TOKENS_IN_BATCH
    )
    decode = derive_decode_effective_tokens(GATE_C_WORKLOAD)
    assert required[2] == decode
    assert required[4] == decode
    assert 35 not in required[2]  # a prefill-shaped key must never appear at tp=2/4


def test_derive_linear_op_required_points_total_row_count_is_122():
    required = derive_linear_op_required_points(
        GATE_C_WORKLOAD, GATE_C_ATTN_TP_VALUES, MAX_TOKENS_IN_BATCH
    )
    assert sum(len(v) for v in required.values()) == 122


def test_original_grid_has_real_missing_and_unused_keys():
    """The grid actually proposed before this check
    (`docs/tasks/61-...md`'s own §11): `{1,2,4,5,8,16,32,64,128}`,
    applied identically at every tp. This must fail coverage -- that is
    the whole point of this task."""
    original_flat_grid = frozenset({1, 2, 4, 5, 8, 16, 32, 64, 128})
    required = derive_linear_op_required_points(
        GATE_C_WORKLOAD, GATE_C_ATTN_TP_VALUES, MAX_TOKENS_IN_BATCH
    )

    missing_tp1 = missing_keys(required[1], original_flat_grid)
    assert len(missing_tp1) > 0
    assert 3 in missing_tp1  # a real, reachable decode-shaped key, absent from the old grid
    assert 40 in missing_tp1  # a real, reachable prefill-shaped key (8 concurrent x 5), absent

    unused = unused_keys(required[1], original_flat_grid)
    assert unused == frozenset({64, 128})  # neither is ever a real Gate C key at any tp


def test_verify_gate_c_linear_op_coverage_flags_the_real_gap():
    original_flat_grid = frozenset({1, 2, 4, 5, 8, 16, 32, 64, 128})
    profiled_by_tp = {1: original_flat_grid, 2: original_flat_grid, 4: original_flat_grid}
    result = verify_gate_c_linear_op_coverage(
        GATE_C_WORKLOAD, GATE_C_ATTN_TP_VALUES, MAX_TOKENS_IN_BATCH, profiled_by_tp
    )
    assert all(len(v) > 0 for v in result.values())  # every tp has real gaps under the old grid


def test_verify_gate_c_linear_op_coverage_passes_once_corrected():
    required = derive_linear_op_required_points(
        GATE_C_WORKLOAD, GATE_C_ATTN_TP_VALUES, MAX_TOKENS_IN_BATCH
    )
    result = verify_gate_c_linear_op_coverage(
        GATE_C_WORKLOAD, GATE_C_ATTN_TP_VALUES, MAX_TOKENS_IN_BATCH, dict(required)
    )
    assert all(len(v) == 0 for v in result.values())


def test_verify_gate_c_linear_op_coverage_raises_if_a_tp_has_no_recorded_data_at_all():
    with pytest.raises(KeyError):
        verify_gate_c_linear_op_coverage(
            GATE_C_WORKLOAD, GATE_C_ATTN_TP_VALUES, MAX_TOKENS_IN_BATCH,
            {1: frozenset(range(1, 200))},  # tp=2 and tp=4 never mentioned
        )


def test_read_profiled_num_tokens_parses_a_synthetic_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["num_tokens", "other_col"])
        writer.writeheader()
        for n in (1, 2, 4, 8):
            writer.writerow({"num_tokens": n, "other_col": "x"})
        path = fh.name
    try:
        assert read_profiled_num_tokens(path) == frozenset({1, 2, 4, 8})
    finally:
        Path(path).unlink()


def test_read_profiled_num_tokens_parses_a_real_mi355x_csv():
    real_csv = (
        Path(__file__).resolve().parent.parent.parent
        / "Frontier" / "data" / "profiling" / "compute" / "mi355x"
        / "meta-llama" / "Llama-2-7b-hf" / "linear_op.csv"
    )
    if not real_csv.exists():
        pytest.skip("real mi355x Llama-2-7b-hf linear_op.csv not present in this checkout")
    values = read_profiled_num_tokens(str(real_csv))
    assert len(values) > 0
    assert all(isinstance(v, int) for v in values)


def test_read_profiled_num_tokens_rejects_a_missing_column():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["some_other_column"])
        writer.writeheader()
        writer.writerow({"some_other_column": "1"})
        path = fh.name
    try:
        with pytest.raises(ValueError):
            read_profiled_num_tokens(path)
    finally:
        Path(path).unlink()
