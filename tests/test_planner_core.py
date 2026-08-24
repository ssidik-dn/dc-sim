"""Task 37: `tools/planner_core.py` holds the search -- candidate
representation, feasibility, shape enumeration, constraint filtering,
ranking -- separated from `tools/planner.py`'s own `SimulationEvaluator`,
the only thing that actually invokes Frontier.

Two things this file proves, not just asserts:

1. `planner_core.py` imports nothing from `frontier`, `integration`, or
   `subprocess` -- checked by parsing its own source, not by inspecting
   `sys.modules` (which would depend on what some other test file
   already imported earlier in the same pytest session, and so would
   not actually prove anything about this module on its own).
2. `plan()` ranks candidates correctly against a `FakeEvaluator` that
   returns fixed results from a dictionary -- if the core still needed
   a simulator to run, this test would need one too, and it does not.

`pytest.ini`'s own `pythonpath = src` does not reach `tools/`; this
file adds it itself, the same way `tools/planner.py` and
`tools/planner_core.py` add each other.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from planner_core import (  # noqa: E402
    Topology, ModelSpec, Workload, Hardware, Objectives, Candidate, plan,
)
from engine.physical.builders import build_node_scale  # noqa: E402

_PLANNER_CORE_PATH = Path(__file__).resolve().parent.parent / "tools" / "planner_core.py"


# --------------------------------------------------- the seam is real, not nominal


def test_planner_core_imports_nothing_frontier_shaped():
    """Parses `planner_core.py`'s own source rather than trusting
    `sys.modules` -- another test file importing `frontier` first would
    make a `sys.modules` check pass for the wrong reason."""
    tree = ast.parse(_PLANNER_CORE_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "frontier" not in imported
    assert "integration" not in imported
    assert "subprocess" not in imported


# ------------------------------------------------------------- FakeEvaluator


class FakeEvaluator:
    """Returns a fixed result per `attn_tp`, from a plain dictionary --
    no Frontier, no subprocess, no simulation of any kind. Ignores
    `attn_shape`/`ep`/replica ratio deliberately: this test is about
    whether `plan()`'s own ranking and rejection logic works against
    *an* evaluator, not about reproducing a specific placement search."""

    def __init__(self, price_by_tp: dict):
        self._price_by_tp = price_by_tp

    def can_evaluate(self, candidate: Candidate) -> bool:
        return candidate.attn_tp in self._price_by_tp

    def evaluate(self, candidate: Candidate) -> dict:
        p = self._price_by_tp[candidate.attn_tp]
        return {
            "mean_tpot_ms": p["mean_tpot_ms"],
            "throughput_rps": p["throughput_rps"],
            "slo_attainment": p["slo_attainment"],
            "error": None,
        }


def _model(admissible_tp):
    # hidden_size/num_attention_heads/num_key_value_heads/num_layers small
    # enough that every tp in admissible_tp is memory-feasible at this
    # margin on an h800 -- the point of this test is FakeEvaluator's own
    # ranking, not re-exercising the memory-feasibility boundary tasks
    # 24-28/35/36 already cover.
    return ModelSpec("fake-model", total_experts=1, router_topk=1, is_moe=False,
                     admissible_tp=admissible_tp, hidden_size=256,
                     num_attention_heads=8, num_key_value_heads=8, num_layers=2)


def _topology():
    fabric = build_node_scale(num_machines=2, gpus_per_machine=4)
    return Topology(fabric, "fake-test-fabric")


def _workload():
    return Workload(num_requests=1, qps=1.0, prefill_tokens=1, decode_tokens=1)


def _hardware():
    return Hardware(device="h800", memory_margin_fraction=0.2)


def test_plan_ranks_correctly_against_a_fake_evaluator_with_no_frontier_present():
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    evaluator = FakeEvaluator({
        1: {"mean_tpot_ms": 20.0, "throughput_rps": 50.0, "slo_attainment": 0.5},
        2: {"mean_tpot_ms": 10.0, "throughput_rps": 100.0, "slo_attainment": 0.9},
    })

    result = plan(_topology(), _model((1, 2)), _workload(), _hardware(), objectives, evaluator)

    assert result.winner is not None
    assert result.winner["candidate"].attn_tp == 2
    assert result.winner["mean_tpot_ms"] == 10.0
    assert not result.rejections
    assert not result.unknown
    for r in result.ranked:
        expected = evaluator._price_by_tp[r["candidate"].attn_tp]
        assert r["mean_tpot_ms"] == expected["mean_tpot_ms"]
    # ranked strictly by mean_tpot_ms ascending
    assert [r["mean_tpot_ms"] for r in result.ranked] == sorted(
        r["mean_tpot_ms"] for r in result.ranked)


def test_plan_reports_a_throughput_floor_rejection_distinctly():
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=75.0, minimize="mean_tpot_ms")
    evaluator = FakeEvaluator({
        1: {"mean_tpot_ms": 20.0, "throughput_rps": 50.0, "slo_attainment": 0.5},  # below floor
        2: {"mean_tpot_ms": 10.0, "throughput_rps": 100.0, "slo_attainment": 0.9},
    })

    result = plan(_topology(), _model((1, 2)), _workload(), _hardware(), objectives, evaluator)

    assert result.winner["candidate"].attn_tp == 2
    tp1_rejections = [r for r in result.rejections if r.candidate_key.startswith("tp1_")]
    assert tp1_rejections, "tp=1 should have been rejected on the throughput floor"
    assert "throughput floor" in tp1_rejections[0].reason


def test_plan_reports_unknown_separately_from_rejected():
    """`can_evaluate() == False` is not the same outcome as failing a
    constraint -- task 37's own known trap. tp=2 is in `admissible_tp`
    (the search's own scope) but not in the fake evaluator's own
    coverage, so it must show up as `unknown`, never as a rejection."""
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    evaluator = FakeEvaluator({
        1: {"mean_tpot_ms": 20.0, "throughput_rps": 50.0, "slo_attainment": 0.5},
    })

    result = plan(_topology(), _model((1, 2)), _workload(), _hardware(), objectives, evaluator)

    assert result.winner["candidate"].attn_tp == 1
    assert any(u.candidate_key.startswith("tp2_") for u in result.unknown)
    assert not any(r.candidate_key.startswith("tp2_") for r in result.rejections)


def test_plan_still_filters_memory_infeasibility_without_asking_the_evaluator():
    """Feasibility belongs in the core (task 37's own S3) -- an
    infeasible `attn_tp` must never even reach `can_evaluate`."""
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    asked_tp = []

    class RecordingEvaluator(FakeEvaluator):
        def can_evaluate(self, candidate):
            asked_tp.append(candidate.attn_tp)
            return super().can_evaluate(candidate)

    evaluator = RecordingEvaluator({
        1: {"mean_tpot_ms": 20.0, "throughput_rps": 50.0, "slo_attainment": 0.5},
    })
    # An unreasonably tight margin makes every tp infeasible before the
    # evaluator is ever consulted.
    starved_hardware = Hardware(device="h800", memory_margin_fraction=0.999999)

    result = plan(_topology(), _model((1, 2)), _workload(), starved_hardware, objectives, evaluator)

    assert result.winner is None
    assert asked_tp == []
    assert all("memory" in r.reason for r in result.rejections)
