# Task 37 — Separate the planner from its oracle

Branch: `task-37-evaluator`, branched from `task-36-fabric-dependence`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`. Run after Task 36's own report
landed, per this task's own instruction, not alongside it.

194 tests pass (189 unchanged + 5 new), and
`python3 tools/check_import_direction.py` exits 0.

---

## 1. The protocol, and what `can_evaluate` returns false for today

```python
class Evaluator(Protocol):
    def can_evaluate(self, candidate: Candidate) -> bool: ...
    def evaluate(self, candidate: Candidate) -> dict: ...
```

Defined in `tools/planner_core.py`, which imports nothing from
`frontier`, `integration`, or `subprocess` — checked by parsing its own
source in a test (§3), not by trusting what else happened to be
imported earlier in the same pytest session.

The docstring on `Evaluator` states the reasoning this task's own §2
gives almost verbatim: a simulator prices a **counterfactual**
(a configuration that does not exist); telemetry reports an
**observation** (the one configuration currently deployed, and nothing
else). It also names, without building, the third shape this task's
own §2 asks for — an evaluator that runs a simulation and corrects its
own prediction against a deployed configuration's observed error —
specifically so that adding it later is a new class, not a redesign of
`plan()`.

**`SimulationEvaluator.can_evaluate` returns false today for any
`attn_tp` outside `model.profiled_tp`**, a new `ModelSpec` field
defaulted to `(1, 2, 4, 8)` — Task 35's own finding that every model in
this checkout, on every device with real profiles, is profiled at
exactly those degrees, because nobody overrode the profiler's own
default sweep (`frontier/profiling/linear_op/main.py`'s own
`--num_tensor_parallel_workers`, `default=[1, 2, 4, 8]`). This is
distinct from `model.admissible_tp` (the search's own scope, a
caller's choice) on purpose: a caller could set `admissible_tp=(1, 2,
4, 8, 16)` for a model only profiled to 8, and `can_evaluate` would
correctly say no to the 16 candidates rather than silently pricing
them or crashing partway through evaluation.

---

## 2. Whether Task 33's table reproduces, in full

**Yes, bit-identical — captured before and after the refactor, diffed,
not asserted.** Task 32/33's own regression check
(`Objectives(min_throughput_rps=0.0, slo_attainment_floor=0.0)`,
unconstrained, Task 32's exact fabric) through `plan()`:

```
$ diff before.log after.log
IDENTICAL
```

Every row — the tp=1 rejection, all 16 ranked shapes at their exact
`mean_tpot_ms`/`throughput_rps`/`slo_attainment`, and the winner
(tp=2, `(2,)`, 11.6803 ms) — matches to the last decimal, both times.

---

## 3. Task 36's own result, since it had reported by the time this ran

**Also reproduces bit-identical** — and this caught a real
methodological hazard worth recording. The first "before" capture of
Task 36's own two-fabric experiment was contaminated partway through:
`SimulationEvaluator`'s own subprocess model means each candidate
re-reads `tools/planner.py` fresh from disk, and I had already
overwritten it with the refactor while that ~940-second run was still
in progress, so some of its own later candidates were unknowingly
evaluated by the *new* code. Caught by noticing the running process's
own command line referenced code paths my edit had just changed;
fixed by `git stash`-ing the refactor, re-running the "before" capture
against the clean original, then restoring the refactor for "after."
Both captures now bracket the same code change cleanly:

```
$ diff before.log after.log
1c1
< === domain8_40gpu (elapsed 507.5s, ...) ===
---
> === domain8_40gpu (elapsed 502.5s, ...) ===
...
< {"mean_tpot_ms": 326.2362462083909}
> {"mean_tpot_ms": 326.2362462083909}
...
< {"mean_tpot_ms": 446.51458620842334}
> {"mean_tpot_ms": 446.51458620842334}
```

Only `elapsed_s` differs (wall-clock, naturally not reproducible
between two separate runs) — every `mean_tpot_ms`, shape, ranking, and
rejection reason is identical.

---

## 4. The fake-evaluator test, and confirmation the core runs without Frontier

`tests/test_planner_core.py`, five tests:

1. **`test_planner_core_imports_nothing_frontier_shaped`** — parses
  `planner_core.py`'s own source with `ast` and asserts `frontier`,
  `integration`, and `subprocess` are not among its imports. Parsing
  the source rather than checking `sys.modules` matters: pytest runs
  every test file in one process, so an *earlier* test file importing
  `frontier` would make a `sys.modules`-based check pass regardless of
  what `planner_core.py` itself does.
2. **`test_plan_ranks_correctly_against_a_fake_evaluator_with_no_frontier_present`**
  — a `FakeEvaluator` backed by a plain `{attn_tp: price}` dict, no
  subprocess, no simulation. `plan()` picks the cheaper of two fixed
  prices correctly and ranks the rest in order.
3. **`test_plan_reports_a_throughput_floor_rejection_distinctly`** —
  the existing constraint-rejection path still works against a fake
  evaluator.
4. **`test_plan_reports_unknown_separately_from_rejected`** — a
  candidate in `admissible_tp` but outside the fake evaluator's own
  coverage shows up in `result.unknown`, never in `result.rejections`
  — the distinction this task's own known trap requires.
5. **`test_plan_still_filters_memory_infeasibility_without_asking_the_evaluator`**
  — an evaluator that records every `attn_tp` it was asked about,
  under a starved memory margin: the recorded list is empty, proving
  `feasible_num_blocks` rejects before `can_evaluate` is ever called,
  per this task's own §3 ("feasibility belongs in the core").

All five pass with no Frontier import anywhere in the process — the
proof this task's own §4 asks for, not merely a passing assertion.

---

## 5. What a telemetry-backed evaluator would still need

The protocol is necessary, not sufficient. Concretely, unbuilt and
unaddressed by this refactor:

- **A binding to a live deployment.** `Candidate` describes an
  arrangement; a telemetry evaluator needs to know which *currently
  running* deployment (if any) a candidate corresponds to before it
  can even ask "do I have data for this" — `can_evaluate` as written
  checks a static fact about a model (`profiled_tp`), not "is this the
  configuration I am watching right now." That check is a different
  kind of lookup a telemetry evaluator would need to add for itself.
- **A staleness/window policy.** A simulator's answer doesn't age; an
  observation does. Nothing here says how old telemetry can be before
  `can_evaluate` should start saying no again.
- **A schema for the result dict.** `evaluate()` currently returns
  whatever `SimulationEvaluator` happens to compute
  (`mean_tpot_ms`/`throughput_rps`/`slo_attainment`/`error`) with no
  contract requiring a second implementation to match those exact
  keys, units, or meaning (is `mean_tpot_ms` measured over the same
  window `plan()` assumes? Over what population of requests?). Two
  evaluators could both satisfy the `Protocol`'s own type signature
  and still be incompatible with `plan()`'s own `objectives.minimize`
  lookup if they don't agree on this by convention alone.
- **Nothing about the "corrects a simulation with observed error"
  evaluator named in §1's own docstring exists to build against** —
  stated there deliberately, not filled in, per this task's own trap
  ("do not build the runtime evaluator").

None of this blocks anything today; it is what the *next* evaluator's
own task would need to settle, and this refactor was scoped to leave
room for that without answering it prematurely.

---

## 6. Anywhere this specification is wrong

Nothing required correction. One thing the spec did not anticipate,
worth recording for whoever refactors a subprocess-based tool like
this next: **a subprocess-per-candidate evaluator makes "before" and
"after" capture order-sensitive in a way an in-process refactor would
not be.** Each `SimulationEvaluator.evaluate()` call re-reads
`tools/planner.py` from disk at the moment its subprocess starts, so
editing the file while a long-running "before" baseline is still
issuing new subprocess calls silently contaminates it partway through
(§3). The fix was procedural (capture "before," *then* edit, then
capture "after," never overlapping) rather than a defect in the
refactor itself, but it is exactly the kind of mistake this task's own
§4 trap ("a refactor that changes a number is a bug") would have let
through undetected had the before/after diff not been taken seriously
enough to look at the process table.

## What shipped

- `tools/planner_core.py` (new) — `Topology`/`ModelSpec`/`Workload`/
  `Hardware`/`Objectives`/`Candidate`/`Rejection`/`Unknown`/
  `PlanResult`, the `Evaluator` protocol, `feasible_num_blocks`/
  `lane_assignment_feasible`/`enumerate_attn_shapes`, and `plan()`
  (evaluator required, no Frontier import anywhere in the file).
- `tools/planner.py` (rewritten) — `SimulationEvaluator`
  (`can_evaluate`/`evaluate`), the unchanged subprocess machinery
  (`_argv`/`_placement_for`/`_run_scenario`/the module-level `evaluate`
  free function/`_TOPOLOGIES`/the CLI entry point), and a `plan()`
  wrapper defaulting `evaluator` to a fresh `SimulationEvaluator` so
  every Task 33/36 call site is unchanged.
- `tests/test_planner_core.py` (new) — the no-Frontier-import proof and
  four `FakeEvaluator` tests (§4).
- `docs/tasks/37-evaluator-report.md`, this report.

One commit on `task-37-evaluator`, stacked on `task-36-fabric-dependence`.
