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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from planner_core import (  # noqa: E402
    Topology, ModelSpec, Workload, Hardware, Objectives, Candidate, plan,
    divisibility_violations, attn_param_mem_bytes, InadmissibleDegree,
    _kv_cache_page_bytes_per_layer,
    lane_assignment_feasible, default_attn_dp_size_policy,
    enumerate_replica_arrangements, enumerate_attn_shapes,
    enumerate_joint_arrangements,
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


# ------------------------------------------------- Part A: divisibility (task 39)


def test_divisibility_violations_empty_for_a_dividing_degree():
    model = _model((1, 2, 4, 8))  # num_attention_heads=8, hidden_size=256
    assert divisibility_violations(model, 4) == []


def test_divisibility_violations_catches_non_dividing_attn_tp():
    model = ModelSpec("odd-heads-model", total_experts=1, router_topk=1, is_moe=False,
                      hidden_size=256, num_attention_heads=64, num_key_value_heads=64, num_layers=2)
    violations = divisibility_violations(model, 3)
    assert violations
    assert any("num_attention_heads" in v for v in violations)


def test_divisibility_violations_catches_model_level_mismatch_independent_of_tp():
    # hidden_size not divisible by num_attention_heads at all -- Frontier's
    # own third assertion, which does not even mention attn_tp.
    model = ModelSpec("bad-model", total_experts=1, router_topk=1, is_moe=False,
                      hidden_size=100, num_attention_heads=7, num_key_value_heads=7, num_layers=2)
    violations = divisibility_violations(model, 1)
    assert any("independent of attn_tp" in v for v in violations)


def test_attn_param_mem_bytes_raises_inadmissible_degree_for_non_dividing_tp():
    model = ModelSpec("odd-heads-model", total_experts=1, router_topk=1, is_moe=False,
                      hidden_size=256, num_attention_heads=64, num_key_value_heads=64, num_layers=2)
    with pytest.raises(InadmissibleDegree):
        attn_param_mem_bytes(model, 3)


def test_plan_reports_inadmissible_separately_from_rejected_and_unknown():
    """A non-dividing degree must show up in `result.inadmissible`, never
    in `result.rejections` (it is not a memory failure) or
    `result.unknown` (it is not a gap in the evaluator's own coverage) --
    and the evaluator must never be asked about it at all, since
    inadmissibility is decided before `feasible_num_blocks` is even
    called (task 39's own known trap)."""
    asked_tp = []

    class RecordingEvaluator(FakeEvaluator):
        def can_evaluate(self, candidate):
            asked_tp.append(candidate.attn_tp)
            return super().can_evaluate(candidate)

    model = ModelSpec("64-heads-model", total_experts=1, router_topk=1, is_moe=False,
                      admissible_tp=(1, 3), hidden_size=256, num_attention_heads=64,
                      num_key_value_heads=64, num_layers=2)
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    evaluator = RecordingEvaluator({
        1: {"mean_tpot_ms": 20.0, "throughput_rps": 50.0, "slo_attainment": 0.5},
        3: {"mean_tpot_ms": 5.0, "throughput_rps": 500.0, "slo_attainment": 1.0},
    })

    result = plan(_topology(), model, _workload(), _hardware(), objectives, evaluator)

    assert 3 not in asked_tp  # tp=3 never reached the evaluator at all
    assert any(i.candidate_key.startswith("tp3_") for i in result.inadmissible)
    assert not any(r.candidate_key.startswith("tp3_") for r in result.rejections)
    assert not any(u.candidate_key.startswith("tp3_") for u in result.unknown)
    assert result.winner["candidate"].attn_tp == 1  # the only admissible degree


# --------------------------------------- Part B: runtime vs. raw KV heads (task 39)


def test_kv_cache_page_bytes_defaults_match_the_raw_fields_for_dense_models():
    """Pins the agreement for the DENSE_KV family -- every model this
    formula has been validated against (tasks 36/38). Leaving
    `runtime_num_kv_heads`/`runtime_head_dim` at `None` must give exactly
    the same page size as setting them explicitly to the raw fields --
    proving the default *is* the DENSE_KV resolver, not merely something
    that happens not to have been contradicted yet."""
    dense_model = ModelSpec("dense", total_experts=1, router_topk=1, is_moe=False,
                            hidden_size=4096, num_attention_heads=16, num_key_value_heads=4,
                            num_layers=32, head_dim=128)
    explicit_override = ModelSpec("dense", total_experts=1, router_topk=1, is_moe=False,
                                  hidden_size=4096, num_attention_heads=16, num_key_value_heads=4,
                                  num_layers=32, head_dim=128,
                                  runtime_num_kv_heads=4, runtime_head_dim=128)
    for tp in (1, 2, 4, 8):
        assert (_kv_cache_page_bytes_per_layer(dense_model, tp, 16)
               == _kv_cache_page_bytes_per_layer(explicit_override, tp, 16))


def test_kv_cache_page_bytes_uses_the_override_when_the_raw_fields_would_be_wrong():
    """deepseek-v3's own real numbers (confirmed directly against a real
    Frontier `SimulationConfig` in this task's own investigation, not
    assumed): raw `num_key_value_heads=128`, `get_head_dim()=56`, but
    `get_runtime_num_kv_heads()=1` and `get_runtime_head_size()=576` --
    the LATENT_MLA family's own hard-coded resolution
    (`kv_lora_rank=512 + qk_rope_head_dim=64`), unconditionally, not
    derived from the declared head count at all. Without the override,
    this formula would use the raw pair and get a KV-cache page size
    ~12.4x too large at tp=1 -- this test proves the override, once
    supplied, corrects it, without requiring a full deepseek-v3
    simulation (which needs mi355x profiling data this checkout does
    not exercise through any real-compute tool)."""
    raw_only = ModelSpec("deepseek-v3-like", total_experts=256, router_topk=8, is_moe=True,
                        hidden_size=7168, num_attention_heads=128, num_key_value_heads=128,
                        num_layers=61, head_dim=56)
    with_runtime_override = ModelSpec("deepseek-v3-like", total_experts=256, router_topk=8, is_moe=True,
                                     hidden_size=7168, num_attention_heads=128, num_key_value_heads=128,
                                     num_layers=61, head_dim=56,
                                     runtime_num_kv_heads=1, runtime_head_dim=576)

    raw_page_bytes = _kv_cache_page_bytes_per_layer(raw_only, attn_tp=1, block_size=16)
    corrected_page_bytes = _kv_cache_page_bytes_per_layer(with_runtime_override, attn_tp=1, block_size=16)

    # raw: kv_per_worker=ceil(128/1)=128, head_dim=56 -> 128*56=7168
    # runtime: kv_per_worker=ceil(1/1)=1, head_dim=576 -> 1*576=576
    # ratio: 7168/576 ~= 12.44x
    assert raw_page_bytes == 2 * 16 * 2 * 128 * 56
    assert corrected_page_bytes == 2 * 16 * 2 * 1 * 576
    assert raw_page_bytes > corrected_page_bytes
    ratio = raw_page_bytes / corrected_page_bytes
    assert 12.0 < ratio < 12.5


# ------------------------------------------------- Task 41: replica ratio search


def test_default_attn_dp_size_policy_matches_the_argv_convention():
    """`tools/planner.py`'s own `_argv` sets `attn_data_parallel_size =
    max(candidate.ffn_replicas, 1)` -- this is the SAME formula, kept as
    an explicit, named, overridable policy (task 32 S7's own open
    design question) rather than duplicated inline."""
    assert default_attn_dp_size_policy(1, 1) == 1
    assert default_attn_dp_size_policy(1, 5) == 5
    assert default_attn_dp_size_policy(3, 0) == 1


def test_lane_assignment_feasible_matches_frontiers_own_requirement():
    # 1 attn replica, dp_size=1 lane, cannot cover 2 ffn replicas.
    assert lane_assignment_feasible(1, 2, 1) is False
    # 1 attn replica, dp_size=2 lanes, covers 2 ffn replicas exactly.
    assert lane_assignment_feasible(1, 2, 2) is True
    # 2 attn replicas, dp_size=1 (2 lanes total), covers 2 ffn replicas.
    assert lane_assignment_feasible(2, 2, 1) is True


class ReplicaAwareFakeEvaluator(FakeEvaluator):
    """Unlike `FakeEvaluator` (which "ignores... replica ratio
    deliberately"), this one prices only `attn_replicas == 1` --
    matching `SimulationEvaluator.can_evaluate`'s own task 41 finding
    (confirmed by running it: `attn_replicas > 1` collides in
    `CommGroupRegistry`; `ffn_replicas > 1` does not, and is priced
    normally), so a test against it exercises the same Unknown path a
    real run would hit for `attn_replicas > 1`, without needing
    Frontier."""

    def can_evaluate(self, candidate: Candidate) -> bool:
        return super().can_evaluate(candidate) and candidate.attn_replicas == 1


def test_plan_reports_lane_violation_as_inadmissible_not_rejected():
    """Task 41's own acceptance requirement: a lane-assignment violation
    must surface as `Inadmissible` (task 39's sense), never as a
    `Rejection` -- and the evaluator must never be consulted about it,
    exactly like a non-dividing `attn_tp`. Forces the violation with an
    explicit `attn_dp_size_policy` that always returns 1 (rather than
    this project's own default, `max(ffn_replicas, 1)`, under which the
    check can never actually fire -- task 41's own report explains why)."""
    asked = []

    class RecordingEvaluator(ReplicaAwareFakeEvaluator):
        def can_evaluate(self, candidate):
            asked.append(candidate.key)
            return super().can_evaluate(candidate)

    evaluator = RecordingEvaluator({
        2: {"mean_tpot_ms": 10.0, "throughput_rps": 100.0, "slo_attainment": 0.9},
    })
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")

    result = plan(_topology(), _model((2,)), _workload(), _hardware(), objectives, evaluator,
                 replica_ratios=((1, 1), (1, 2)), attn_dp_size_policy=lambda ar, fr: 1)

    violating = [i for i in result.inadmissible if "_fr2" in i.candidate_key]
    assert violating, "the (1, 2) ratio at attn_dp_size=1 must be inadmissible"
    assert "lane assignment" in violating[0].reason
    assert not any("_fr2" in r.candidate_key for r in result.rejections)
    assert not any("_fr2" in u.candidate_key for u in result.unknown)
    assert not any("_fr2" in key for key in asked), \
        "a lane-inadmissible candidate must never reach the evaluator"
    assert result.winner["candidate"].ffn_replicas == 1


def test_plan_restricted_to_1to1_matches_the_unextended_search():
    """Task 41's own cleanest proof that the extension is an extension:
    a search that only ever considers the (1, 1) ratio (explicitly, or
    by leaving `replica_ratios` at its default) must produce exactly the
    same result either way."""
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    evaluator = ReplicaAwareFakeEvaluator({
        1: {"mean_tpot_ms": 20.0, "throughput_rps": 50.0, "slo_attainment": 0.5},
        2: {"mean_tpot_ms": 10.0, "throughput_rps": 100.0, "slo_attainment": 0.9},
    })

    default_result = plan(_topology(), _model((1, 2)), _workload(), _hardware(),
                          objectives, evaluator)
    explicit_result = plan(_topology(), _model((1, 2)), _workload(), _hardware(),
                           objectives, evaluator, replica_ratios=((1, 1),))

    assert [r["candidate"].key for r in default_result.ranked] == \
        [r["candidate"].key for r in explicit_result.ranked]
    assert [r["mean_tpot_ms"] for r in default_result.ranked] == \
        [r["mean_tpot_ms"] for r in explicit_result.ranked]
    assert default_result.winner["candidate"].key == explicit_result.winner["candidate"].key


def test_plan_adding_more_ratios_does_not_perturb_the_1to1_candidates():
    """Adding `replica_ratios` beyond the default must not change what
    the (1, 1) candidates themselves evaluate to -- task 41's own known
    trap, phrased as acceptance: the new dimension extends the search,
    it does not affect the old default."""
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    evaluator = ReplicaAwareFakeEvaluator({
        1: {"mean_tpot_ms": 20.0, "throughput_rps": 50.0, "slo_attainment": 0.5},
        2: {"mean_tpot_ms": 10.0, "throughput_rps": 100.0, "slo_attainment": 0.9},
    })

    narrow = plan(_topology(), _model((1, 2)), _workload(), _hardware(), objectives, evaluator,
                 replica_ratios=((1, 1),))
    wide = plan(_topology(), _model((1, 2)), _workload(), _hardware(), objectives, evaluator,
               replica_ratios=((1, 1), (2, 1), (1, 2), (3, 2)))

    narrow_1to1 = {r["candidate"].key: r["mean_tpot_ms"] for r in narrow.ranked}
    wide_1to1 = {r["candidate"].key: r["mean_tpot_ms"] for r in wide.ranked
                if r["candidate"].attn_replicas == 1 and r["candidate"].ffn_replicas == 1}
    assert narrow_1to1 == wide_1to1
    assert wide.winner["candidate"].key == narrow.winner["candidate"].key
    # every non-(1,1) candidate the wider search considered is Unknown to
    # this evaluator, not silently dropped or miscounted as a rejection.
    assert any(u.candidate_key for u in wide.unknown
              if "_ar2" in u.candidate_key or "_fr2" in u.candidate_key)


# ---------------------------------------- enumerate_replica_arrangements (task 41)


def _wide_topology():
    """8 GPUs, 2 domains of 4 -- enough room for 2 attention replicas at
    attn_tp=2 (4 GPUs) plus one PREFILL and one DECODE_FFN rank."""
    fabric = build_node_scale(num_machines=2, gpus_per_machine=4)
    return Topology(fabric, "wide-test-fabric")


def test_enumerate_replica_arrangements_single_replica_matches_enumerate_attn_shapes():
    """At `attn_replicas=1`, the multiset-of-one signature must carry
    exactly the same information `enumerate_attn_shapes` already
    reports -- the extension must agree with the un-extended case it
    generalises, not merely resemble it."""
    from planner_core import enumerate_attn_shapes
    topology = _wide_topology()
    single = enumerate_attn_shapes(topology, attn_tp=2)
    multi = enumerate_replica_arrangements(topology, attn_tp=2, attn_replicas=1)
    assert set(multi.keys()) == {(shape,) for shape in single.keys()}


def test_enumerate_replica_arrangements_collapses_permutations():
    """Two attention replicas at attn_tp=2 on an 8-GPU, 2-domain fabric:
    each replica's own shape is (2,) or (1,1) (`enumerate_attn_shapes`'s
    own S=2 result at this degree), so raw placements collapse to at
    most 3 distinct multisets -- {(2,),(2,)}, {(2,),(1,1)},
    {(1,1),(1,1)} -- never 4, which is what an *ordered*-pair signature
    (not treating the two replicas as interchangeable) would allow."""
    topology = _wide_topology()
    arrangements = enumerate_replica_arrangements(topology, attn_tp=2, attn_replicas=2,
                                                  n_fragmented_seeds=60)
    assert len(arrangements) <= 3
    for sig in arrangements:
        assert sig == tuple(sorted(sig)), \
            f"signature {sig} is not in canonical (sorted) multiset form"


def test_enumerate_replica_arrangements_treats_swapped_shapes_as_one_arrangement():
    """The literal case task 41's own spec names: a placement giving
    {A, B} to two replicas must key identically to one giving {B, A}."""
    topology = _wide_topology()
    arrangements = enumerate_replica_arrangements(topology, attn_tp=2, attn_replicas=2,
                                                  n_fragmented_seeds=60)
    # tuple-sort orders (1, 1) before (2,) lexicographically -- the mixed
    # arrangement (one replica packed, one split) has exactly one
    # canonical key, however many raw placements produced it either way.
    mixed_keys = [sig for sig in arrangements if set(sig) == {(2,), (1, 1)}]
    assert len(mixed_keys) <= 1
    if mixed_keys:
        assert mixed_keys[0] == ((1, 1), (2,))


# ---------------------------------------- enumerate_joint_arrangements (task 44)


def _three_domain_topology():
    """12 GPUs, 3 domains of 4 -- enough room for attention (up to
    tp=4) and an expert group (up to ep=4) to each get their own whole
    domain simultaneously, which is what the "packed-if-it-fits"
    explicit fallback needs a third domain for."""
    fabric = build_node_scale(num_machines=3, gpus_per_machine=4)
    return Topology(fabric, "three-domain-test-fabric")


def test_enumerate_joint_arrangements_at_ep1_matches_enumerate_attn_shapes():
    """At `ffn_ep=1` there is no expert-parallel group at all -- the set
    of `attn_shape` values this function reaches, paired with the
    trivial `(1,)` ep_shape, must be exactly what `enumerate_attn_shapes`
    alone already reports. This is the literal check behind task 44's
    own acceptance requirement: the single-expert-group case must not
    move."""
    topology = _wide_topology()
    single = enumerate_attn_shapes(topology, attn_tp=2)
    joint = enumerate_joint_arrangements(topology, attn_tp=2, ffn_ep=1)
    assert set(joint.keys()) == {(shape, (1,)) for shape in single.keys()}


def test_enumerate_joint_arrangements_reaches_the_fully_packed_pair_when_it_fits():
    """On a fabric with room for both groups to each keep their own
    whole domain, the arrangement where neither group splits at all
    must be reachable -- the same "packed-if-it-fits" guarantee task 32
    established for a single group, extended here to two groups placed
    at once (task 44's own S2: enumerated together, not assumed
    independent and combined afterward)."""
    topology = _three_domain_topology()
    joint = enumerate_joint_arrangements(topology, attn_tp=4, ffn_ep=4)
    assert ((4,), (4,)) in joint


def test_enumerate_joint_arrangements_keys_are_ordered_pairs_not_multisets():
    """The known trap this task's own S6 names: an attention group and
    an expert group are not interchangeable, so the canonical key must
    be an ordered pair, `(attn_shape, ep_shape)` -- never sorted
    together the way `enumerate_replica_arrangements`'s own multiset
    key sorts two *attention* replicas' shapes against each other."""
    topology = _three_domain_topology()
    joint = enumerate_joint_arrangements(topology, attn_tp=4, ffn_ep=2)
    for attn_shape, ep_shape in joint:
        # every key is a 2-tuple of (attn_shape, ep_shape) in that fixed
        # role order -- unlike a multiset key, swapping the two would be
        # a different, meaningless pairing (an attention shape is never
        # interchangeable with an expert-group shape), so nothing here
        # sorts attn_shape and ep_shape against each other.
        assert isinstance(attn_shape, tuple) and isinstance(ep_shape, tuple)
    # the two roles have different reachable shape sets at (attn_tp=4,
    # ffn_ep=2) -- attn_shape can be (4,)/(3,1)/... (S=5, task 32's own
    # table); ep_shape can only be (2,)/(1,1) (S=2) -- so an ordered-pair
    # key is not merely a stylistic choice here, the two axes are not
    # even the same size.
    attn_shapes_seen = {a for a, _ in joint}
    ep_shapes_seen = {e for _, e in joint}
    assert len(attn_shapes_seen) >= len(ep_shapes_seen)


def test_enumerate_joint_arrangements_collapses_raw_candidates():
    """The same style of figure task 32 (188 -> 16) and task 41 (its own
    replica equivalent) reported: raw placements collapse to a much
    smaller set of distinct (attn_shape, ep_shape) pairs."""
    topology = _three_domain_topology()
    joint = enumerate_joint_arrangements(topology, attn_tp=4, ffn_ep=4, n_fragmented_seeds=60)
    raw_candidates = 2 + 1 + 60  # packed + spread + the explicit fallback + 60 fragmented
    assert len(joint) < raw_candidates


# ------------------------------------------------- plan() with expert placement


class EpAwareFakeEvaluator(FakeEvaluator):
    """Like `FakeEvaluator`, but also records every candidate it was
    asked to price, so a test can inspect `ep_shape` without needing a
    real evaluator."""

    def __init__(self, price_by_tp: dict):
        super().__init__(price_by_tp)
        self.seen: list = []

    def evaluate(self, candidate: Candidate) -> dict:
        self.seen.append(candidate)
        return super().evaluate(candidate)


def test_plan_default_ep_values_gives_only_the_trivial_ep_shape():
    """`ep_values` defaults to `model.admissible_ep`, itself defaulting
    to `(1,)` -- every candidate `plan()` builds without an explicit
    `ep_values` argument must carry the trivial `ep_shape=(1,)`, exactly
    as it did before this task added a real one."""
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    evaluator = EpAwareFakeEvaluator({
        2: {"mean_tpot_ms": 10.0, "throughput_rps": 100.0, "slo_attainment": 0.9},
    })
    result = plan(_topology(), _model((2,)), _workload(), _hardware(), objectives, evaluator)
    assert result.ranked
    for r in result.ranked:
        assert r["candidate"].ep_shape == (1,)


def test_plan_restricted_to_single_expert_group_matches_the_unextended_search():
    """Task 44's own required acceptance test: a search explicitly
    restricted to `ep_values=(1,)` must give exactly what leaving
    `ep_values` at its default gives -- the cleanest proof that adding
    expert placement is an extension, not a change to the existing
    default."""
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    evaluator = EpAwareFakeEvaluator({
        1: {"mean_tpot_ms": 20.0, "throughput_rps": 50.0, "slo_attainment": 0.5},
        2: {"mean_tpot_ms": 10.0, "throughput_rps": 100.0, "slo_attainment": 0.9},
    })

    default_result = plan(_topology(), _model((1, 2)), _workload(), _hardware(),
                          objectives, evaluator)
    explicit_result = plan(_topology(), _model((1, 2)), _workload(), _hardware(),
                           objectives, evaluator, ep_values=(1,))

    assert [r["candidate"].key for r in default_result.ranked] == \
        [r["candidate"].key for r in explicit_result.ranked]
    assert [r["mean_tpot_ms"] for r in default_result.ranked] == \
        [r["mean_tpot_ms"] for r in explicit_result.ranked]
    assert default_result.winner["candidate"].key == explicit_result.winner["candidate"].key


def test_plan_adding_ep_values_does_not_perturb_the_single_group_candidates():
    """Adding a second `ffn_ep` value to search must not change what the
    `ffn_ep=1` candidates themselves evaluate to -- the same acceptance
    shape task 41 required of `replica_ratios`, applied here to expert
    degree."""
    objectives = Objectives(slo_tpot_ms=15.0, min_throughput_rps=0.0, minimize="mean_tpot_ms")
    evaluator = EpAwareFakeEvaluator({
        2: {"mean_tpot_ms": 10.0, "throughput_rps": 100.0, "slo_attainment": 0.9},
    })

    narrow = plan(_topology(), _model((2,)), _workload(), _hardware(), objectives, evaluator,
                 ep_values=(1,))
    wide = plan(_topology(), _model((2,)), _workload(), _hardware(), objectives, evaluator,
               ep_values=(1, 2))

    narrow_ep1 = {r["candidate"].key: r["mean_tpot_ms"] for r in narrow.ranked}
    wide_ep1 = {r["candidate"].key: r["mean_tpot_ms"] for r in wide.ranked
               if r["candidate"].ffn_ep == 1}
    assert narrow_ep1 == wide_ep1
