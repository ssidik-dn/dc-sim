"""Stage 2 Gate C.1 (dense-routing fix): regression tests for the
now-installed dense-model MoE-routing guard.

`SklearnDisaggregationExecutionTimePredictor.__init__` -> `_simulate_and_store_routing`
-> `_generate_expert_allocations` (`frontier/execution_time_predictor/
sklearn_disaggregation_execution_time_predictor.py`) used to compute MoE
expert-routing allocations *unconditionally* for `PREFILL`/`DECODE_FFN`/
`DECODE` predictors, regardless of `model_config.is_moe`. For a dense
model (`total_expert_num=0`), `allocation_ratios = [1.0 /
total_expert_num] * total_expert_num` raised `ZeroDivisionError`.

Fixed (docs/tasks/68-stage2-gate-c1-dense-routing-report.md, approved and
implemented) by `src/integration/execution_time_predictor/dense_model_moe_routing_guard.py`
-- a guarded, source-hash-checked runtime patch of
`SklearnDisaggregationExecutionTimePredictor.__init__`, following this
project's own established `sglang_guard`/`mla_phase_filter` pattern
(patch a whole pinned-Frontier method, guarded, rather than edit the
checkout). Wired into `tools/planner.py::_run_scenario`'s existing
`install(..., dense_model_moe_routing_guard=True)` call.

Three real, distinct states, each with its own test below:

- `is_moe=False` (any `total_expert_num`): dense. Routing is skipped
  entirely -- no `ZeroDivisionError`, no fabricated expert-routing state.
- `is_moe=True` with `total_expert_num<=0`: **inconsistent model
  metadata**. Raises an explicit `InconsistentMoeModelMetadataError`
  (wrapped by Frontier's own `ExecutionTimePredictorRegistry.get` into a
  `ValueError`), not an incidental `ZeroDivisionError` -- the approved
  amendment to the original design.
- `is_moe=True` with `total_expert_num>0`: the original routing
  computation runs character-for-character unchanged -- verified here as
  an exact-value, bit-for-bit non-regression anchor
  (`Phi-tiny-MoE-instruct`, this project's own established MoE
  regression case, Task 33/36), not merely "no exception raised."
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from planner_core import ModelSpec, Workload, Hardware, Candidate, feasible_num_blocks  # noqa: E402
from planner import evaluate, _TOPOLOGIES  # noqa: E402

GATE_C_WORKLOAD = Workload(num_requests=32, qps=4.0, prefill_tokens=5, decode_tokens=32)


def _domain8():
    return _TOPOLOGIES["domain8"]()


# --------------------------------------------------------------- test A


def test_A_dense_model_no_longer_hits_the_zerodivisionerror_bug():
    """VALID DENSE state (`is_moe=False`, `total_experts=0`) -- Qwen3-0.6B,
    the real model this Gate C.1 initiative is evaluating. Must now
    succeed cleanly: no `ZeroDivisionError`, no `random_forrest`
    predictor-construction failure, a finite, profile-backed prediction.
    No fabricated expert-routing state is asserted here because none is
    supposed to exist for a dense model (docs/tasks/68 S6/S9) -- this
    test only proves the predictor no longer crashes and produces a real
    number."""
    if not (Path("/work/simulation/Frontier/data/profiling/compute/mi355x/Qwen3-0.6B/linear_op.csv").exists()):
        pytest.skip("installed Qwen3-0.6B mi355x profile not present in this checkout")

    topology = _domain8()
    model = ModelSpec(
        model_name="Qwen3-0.6B", total_experts=0, router_topk=0, is_moe=False,
        hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
        num_layers=28, head_dim=128, profiled_tp=(1, 2, 4),
    )
    hardware = Hardware("mi355x", memory_margin_fraction=0.2)
    candidate = Candidate(attn_tp=1, attn_shape=(1,))
    num_blocks = feasible_num_blocks(model, hardware, 1)

    result = evaluate(topology, model, GATE_C_WORKLOAD, hardware, candidate, num_blocks)

    assert result["error"] is None, f"dense Qwen3-0.6B evaluation still failing: {result['error']}"
    assert result["n_completed"] == 32
    assert result["mean_tpot_ms"] is not None
    import math
    assert math.isfinite(result["mean_tpot_ms"])
    assert result["mean_tpot_ms"] > 0


# --------------------------------------------------------------- test B


def test_B_moe_baseline_phi_tiny_moe_instruct_unaffected():
    """VALID MOE state (`is_moe=True`, `total_experts=16`) --
    `Phi-tiny-MoE-instruct`, this project's own established MoE regression
    case (Task 33/36). Must produce this exact, pre-fix-captured baseline
    -- proof the dense-routing fix leaves existing MoE behavior
    unchanged, bit-for-bit, not merely 'tests still pass'."""
    topology = _domain8()
    model = ModelSpec(
        model_name="Phi-tiny-MoE-instruct", total_experts=16, router_topk=2, is_moe=True,
        hidden_size=4096, num_attention_heads=16, num_key_value_heads=4,
        num_layers=32, head_dim=128,
    )
    hardware = Hardware("h800", memory_margin_fraction=0.2)
    candidate = Candidate(attn_tp=1, attn_shape=(1,), ffn_ep=1, ep_shape=(1,))
    num_blocks = feasible_num_blocks(model, hardware, 1)

    result = evaluate(topology, model, GATE_C_WORKLOAD, hardware, candidate, num_blocks)

    assert result["error"] is None
    assert result["n_completed"] == 32
    assert result["mean_tpot_ms"] == pytest.approx(12.317824968905404, abs=1e-9)
    assert result["throughput_rps"] == pytest.approx(50.86139603307486, abs=1e-9)
    assert result["slo_attainment"] == pytest.approx(0.75, abs=1e-9)


# --------------------------------------------------------------- test C


def test_C_inconsistent_moe_metadata_raises_explicit_error():
    """INCONSISTENT MODEL state (`is_moe=True` from the model's own real
    config, but `total_experts=0` passed explicitly): must fail loudly
    with the new, explicit `InconsistentMoeModelMetadataError` message
    (approved amendment to the original design), not an incidental
    `ZeroDivisionError` and not a silent repair -- an `is_moe=True` model
    with corrupt/inconsistent expert-count metadata must keep failing.

    Constructed from a real, already-profiled MoE model
    (`Phi-tiny-MoE-instruct`, `model_config.is_moe=True` per its own real
    JSON) with `total_experts=0` passed explicitly (not the dataclass
    default `1`, which Frontier's own `ReplicaConfig.__post_init__` would
    otherwise auto-correct from the model's real expert count) -- Frontier
    itself never re-derives `total_expert_num` from the model config once
    a caller passes an explicit non-default value (confirmed by reading
    `ReplicaConfig.__post_init__`), so this really does reach Frontier
    with `is_moe=True, total_expert_num=0` simultaneously.
    """
    topology = _domain8()
    model = ModelSpec(
        model_name="Phi-tiny-MoE-instruct", total_experts=0, router_topk=2, is_moe=True,
        hidden_size=4096, num_attention_heads=16, num_key_value_heads=4,
        num_layers=32, head_dim=128,
    )
    hardware = Hardware("h800", memory_margin_fraction=0.2)
    candidate = Candidate(attn_tp=1, attn_shape=(1,), ffn_ep=1, ep_shape=(1,))
    num_blocks = feasible_num_blocks(model, hardware, 1)

    result = evaluate(topology, model, GATE_C_WORKLOAD, hardware, candidate, num_blocks)

    assert result["error"] is not None
    assert "inconsistent MoE model metadata" in result["error"]
    assert "is_moe=True" in result["error"]
    assert "total_expert_num=0" in result["error"]
    assert "float division by zero" not in result["error"]
