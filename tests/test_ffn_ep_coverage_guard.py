"""Stage 2 Gate A.1: `SimulationEvaluator.can_evaluate` must gate
`candidate.ffn_ep` against `model.profiled_ep`, exactly mirroring the
`attn_tp`/`profiled_tp` gate it already had -- closing the coverage gap
Stage 2 Gate A's own report found (`can_evaluate` checked `attn_tp`
against a real profiled grid but let any `ffn_ep` through unchecked).

Hermetic: `can_evaluate` is a pure boolean check against `ModelSpec`
fields, no Frontier subprocess involved. `discover_profiled_ep` does
read a real file, but only the file, not a subprocess -- also hermetic
in the sense that matters here (no GPU, no Frontier import).

`pytest.ini`'s own `pythonpath = src` does not reach `tools/`; this
file adds it itself, the same convention every other `tools/`-testing
file in this project already uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from planner_core import Candidate, ModelSpec  # noqa: E402
from planner import SimulationEvaluator, discover_profiled_ep  # noqa: E402


def _model(**overrides) -> ModelSpec:
    base = dict(model_name="test-model", total_experts=16, router_topk=2, is_moe=True,
               hidden_size=4096, num_attention_heads=16, num_key_value_heads=4, num_layers=2)
    base.update(overrides)
    return ModelSpec(**base)


def _evaluator(model: ModelSpec) -> SimulationEvaluator:
    # can_evaluate never touches topology/workload/hardware/regime --
    # None stand-ins keep this test from needing any of those objects.
    return SimulationEvaluator(topology=None, model=model, workload=None, hardware=None)


# --------------------------------------------------------------------------
# profiled_ep default and gating
# --------------------------------------------------------------------------


def test_profiled_ep_defaults_to_the_trivial_degree_only():
    model = _model()
    assert model.profiled_ep == (1,)


def test_ffn_ep_1_is_always_accepted_regardless_of_profiled_ep():
    """The default -- and every existing call site that never sets
    `ffn_ep` above 1 -- must keep working unchanged."""
    model = _model()
    evaluator = _evaluator(model)
    candidate = Candidate(attn_tp=2, attn_shape=(2,), ffn_ep=1, ep_shape=(1,))
    assert evaluator.can_evaluate(candidate)


def test_known_valid_ffn_ep_is_accepted():
    model = _model(profiled_ep=(1, 2, 4, 8))
    evaluator = _evaluator(model)
    candidate = Candidate(attn_tp=2, attn_shape=(2,), ffn_ep=4, ep_shape=(4,))
    assert evaluator.can_evaluate(candidate)


def test_out_of_grid_ffn_ep_is_rejected_as_unknown_not_silently_accepted():
    """The exact regression this task exists to close: before this fix,
    `can_evaluate` had no `ffn_ep` check at all, so this candidate would
    have returned `True` and been sent to a live Frontier subprocess
    that raises `ValueError` before training (a `Rejection`, not an
    `Unknown`) -- or, for a model whose real grid genuinely lacks the
    requested degree, silently proceeded with no coverage signal
    whatsoever. `can_evaluate() -> False` is `plan()`'s own `Unknown`
    path, never a `Rejection`."""
    model = _model(profiled_ep=(1, 2))  # matches qwen2_moe_example's real a100 grid
    evaluator = _evaluator(model)
    candidate = Candidate(attn_tp=2, attn_shape=(2,), ffn_ep=8, ep_shape=(8,))
    assert not evaluator.can_evaluate(candidate)


def test_attn_tp_gating_is_unchanged_by_this_fix():
    model = _model()  # profiled_tp defaults to (1,2,4,8), unaffected by profiled_ep
    evaluator = _evaluator(model)
    in_grid = Candidate(attn_tp=4, attn_shape=(4,))
    out_of_grid = Candidate(attn_tp=16, attn_shape=(16,))
    assert evaluator.can_evaluate(in_grid)
    assert not evaluator.can_evaluate(out_of_grid)


def test_attn_replicas_gating_is_unaffected_by_this_fix():
    model = _model(profiled_ep=(1, 2))
    evaluator = _evaluator(model)
    candidate = Candidate(attn_tp=2, attn_shape=(2,), ffn_ep=2, ep_shape=(2,), attn_replicas=2)
    assert not evaluator.can_evaluate(candidate)  # attn_replicas > 1, task 41's own block


# --------------------------------------------------------------------------
# discover_profiled_ep -- real files, no Frontier subprocess
# --------------------------------------------------------------------------


def test_discover_profiled_ep_matches_the_real_phi_tiny_grid():
    grid = discover_profiled_ep("h800", "Phi-tiny-MoE-instruct", num_experts=16,
                                router_topk=2, hidden_dim=4096)
    assert grid == (1, 2, 4, 8)


def test_discover_profiled_ep_matches_a_narrower_real_grid():
    grid = discover_profiled_ep("a100", "mixtral_8x7b_moe", num_experts=8,
                                router_topk=2, hidden_dim=4096)
    assert grid == (1,)


def test_discover_profiled_ep_raises_on_missing_model():
    with pytest.raises(FileNotFoundError):
        discover_profiled_ep("h800", "no-such-model-exists", num_experts=1,
                             router_topk=1, hidden_dim=1)


def test_discover_profiled_ep_raises_when_no_row_matches_model_dims():
    with pytest.raises(ValueError):
        discover_profiled_ep("h800", "Phi-tiny-MoE-instruct", num_experts=99999,
                             router_topk=2, hidden_dim=4096)
