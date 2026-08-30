# Stage 2 — Gate C.1: collection↔evaluation QK-norm fix (CONFIRMED FIXED),
# and a second, independent, newly-discovered real blocker (NOT fixed, per
# `AGENTS.md`'s own human-only zone -- reported, not patched)

**The QK-norm collection↔evaluation mismatch this task was scoped to fix is
fixed and verified.** Real TP=1/2/4 Frontier evaluation now gets
measurably further than docs/tasks/66's own hard stop -- past `use_qk_norm`
filtering, through real per-operator model training (21 family-scoped
models) -- and then hits a **second, different, real root cause**: a
pre-existing Frontier upstream bug (`ZeroDivisionError` in MoE-routing
simulation for a zero-expert dense model), never previously reachable by
this project because Qwen3-0.6B is the first genuinely dense
(`is_moe=False`, `total_experts=0`) model this project has ever pushed
through a real, end-to-end `pd-af-disaggregation` Frontier evaluation.
**Fixing that second bug means either editing the external Frontier
checkout directly or adding a new implementation under `src/integration/`
-- both outside what `AGENTS.md` allows an agent to implement (`src/integration/`
is explicitly human-only for implementations; agents may write tests
there, not fixes). Stopping here, reporting both findings, not patching
the second one.**

---

## 0. Repository reconstruction (verify-first, no assumptions)

```
branch: stage2-gate-a-contract, up to date with origin/stage2-gate-a-contract
git status: clean except untracked .claude/ and log/ (pre-existing local
            tooling artifacts, unrelated to Gate C.1; left untouched)
no unpushed local commits beyond what's already on origin
no CLAUDE.md / RUNLOG.md exist in this checkout -- AGENTS.md is the real
  governing doc; read directly, not assumed
```

Read directly, in order: `AGENTS.md`, `docs/tasks/66-stage2-gate-c1-profile-install-report.md`
(the most recent Gate C.1 report), `65-...-full-sweep-completion-report.md`,
`src/integration/install/__init__.py`, `src/integration/profiling/qk_norm_allowlist_fix.py`,
`tools/planner.py`, `tools/stage2/gate_c1_coverage.py`. Every item in the
task prompt's §1 (expected prior state) matched the repository exactly --
no discrepancy to report.

---

## 1. Collection↔evaluation compatibility stack comparison

| integration | collection | evaluation (before this task) | evaluation (after fix) | required where |
|---|---:|---:|---:|---|
| QK-norm allowlist fix (`qk_norm_allowlist_fix.py`) | applied | **not applied** | **applied** | both -- feeds `ModelConfig.use_qk_norm`, read by both `linear_op_impl.py` at collection and the predictor's exact-match filter at evaluation |
| RoPE API adapter (`rope_api_adapter.py`) | applied | n/a | n/a | collection only -- patches a live vLLM/torch module; evaluation is CPU-only and never imports `torch` (confirmed: `import torch` fails in this sandbox) |
| `profiling_vllm_config_context()` (`vllm_config_context.py`) | applied | n/a | n/a | collection only -- same reason |
| RMSNorm API adapter (`rmsnorm_api_adapter.py`) | applied | n/a | n/a | collection only -- same reason |
| attention block-table fix (`attention_block_table_fix.py`) | applied (free-standing, not wired into `install()` at all -- its own docstring: needs `torch`, only reachable from the profiling CLI) | n/a | n/a | collection only |
| MLA phase filter (`mla_phase_filter.py`) | n/a | n/a | n/a | neither -- Qwen3-0.6B is dense/GQA, not MLA |

Confirmed by reading each module's own docstring plus a direct grep for
every `install()` call site in the repo (22 call sites, `tools/run_*.py` +
`tools/planner.py`; none pass `qk_norm_allowlist_fix=True` before this
task). No adapter was enabled in evaluation beyond the one this comparison
shows is actually needed there -- the RoPE/RMSNorm/vllm-config-context/
block-table fixes exist solely to make real vLLM/torch GPU profiling
possible and have no evaluation-side counterpart to enable, confirmed
directly (`frontier.config`/`frontier.simulator` import and run with no
`torch` present in this sandbox).

---

## 2. QK-norm mismatch: reproduced, confirmed, root-caused

Live, in this exact checkout, `cwd=/work/simulation/Frontier` (real
evaluation's own cwd, `tools/planner.py::evaluate`'s subprocess call):

```
>>> BaseModelConfig.create_from_name('Qwen3-0.6B').use_qk_norm
False                                    # matches docs/tasks/66 exactly
>>> install_qk_norm_allowlist_fix()
>>> BaseModelConfig.create_from_name('Qwen3-0.6B').use_qk_norm
True
```

Installed profile's own recorded metadata, read directly, not assumed:

```
data/profiling/compute/mi355x/Qwen3-0.6B/linear_op.csv
  use_qk_norm value_counts: True  186   (100% of rows)
```

Predictor's own exact-match filter, read directly
(`shared_prediction_model_manager.py:2279-2306`,
`SharedPredictionModelManager._load_linear_op_df`): reads
`expected_use_qk_norm` from `training_context["use_qk_norm"]` (itself
`ModelConfig.use_qk_norm`), then
`filtered_df[filtered_df["use_qk_norm"].astype(bool) == expected_use_qk_norm]`.
With every row `True` and `expected_use_qk_norm=False` (unfixed
evaluation), the filter keeps zero rows for every `tensor_parallel_size`
identically -- proven, not inferred, exactly the docs/tasks/66 diagnosis.

**Answer to task §5: the previous session's diagnosis was correct**, verified independently in this session by re-running the same live checks.

---

## 3. The fix

One line, `tools/planner.py::_run_scenario`'s own existing `install()`
call (the only place named in the prior report as the gap):

```diff
     install(topology.fabric, placement, d, reg, binding=binding, collective=True,
-           sglang_replica_scheduler=True)
+           sglang_replica_scheduler=True, qk_norm_allowlist_fix=True)
```

No duplicate QK-norm logic added to `planner.py`. Uses the existing,
already-reviewed `install(..., qk_norm_allowlist_fix=True)` mechanism
verbatim -- the same flag every real collection invocation applied,
routed through the same shared `QK_NORM_MODEL_TYPE_ALLOWLIST` object both
collection and evaluation read (not a planner-local constant). This edit
is in `tools/planner.py`, not `src/integration/` -- inside AGENTS.md's
agent-safe collaboration zone, not its human-only one.

---

## 4. Regression test (`tests/test_gate_c1_qk_norm_evaluation_parity.py`, new)

Three tests, all passing:

1. **Static wiring guard** -- `inspect.getsource(planner._run_scenario)`
   contains `"qk_norm_allowlist_fix=True"`. Catches this exact class of
   regression the next time a QK-norm-needing model is wired in.
2. **Semantic parity** -- the real `install()` entry point (not the
   private fix module directly), called with `cwd=FRONTIER_ROOT` (matching
   real evaluation's own cwd), flips `BaseModelConfig.create_from_name('Qwen3-0.6B').use_qk_norm`
   from the confirmed pre-fix `False` to `True`.
3. **Installed-profile parity** -- the installed `linear_op.csv`'s own
   `use_qk_norm` column (all `True`) matches what the fixed evaluation
   path now infers (skips cleanly if the file isn't present in a given
   checkout).

Tests the shared compatibility configuration object
(`QK_NORM_MODEL_TYPE_ALLOWLIST`), not a hard-coded Qwen3-specific constant
in planner code, per the task's own preference.

```
$ python3 -m pytest tests/test_gate_c1_qk_norm_evaluation_parity.py -q
...                                                                      [100%]
3 passed in 1.02s
```

Full suite, unaffected:

```
$ python3 -m pytest -q
412 passed, 16 skipped in 123.51s
$ python3 tools/check_import_direction.py
OK: engine imports nothing from integration or upstream
```

(16 skips are the project's own established GPU/`torch`-gated tests --
same skip set as before this change, confirmed by inspection of the skip
reasons, not a new skip introduced by this fix.)

---

## 5. Hard-code audit

Delegated to an independent sub-review of `tools/planner.py`,
`tools/planner_core.py`, `tools/stage2/gate_c1_coverage.py`,
`tools/seed_stats.py`, and every `src/integration/` module touched by
this task. Findings:

| location | value | verdict |
|---|---|---|
| `tools/planner.py` | SLO threshold (`<= 15.0` ms) | explicit configuration, not a measured value |
| `tools/seed_stats.py` | `_Z_95 = 1.960` | standard-normal 95% CI z-score, a statistical constant |
| `mean_tpot_ms`, `throughput_rps` | computed live from `sim._all_requests` each run | not hard-coded |
| `planner_core.py` `winner = evaluated[0]` | top of a freshly-sorted, freshly-evaluated candidate list each call | not a preset winner |
| `feasible_num_blocks` outputs (134624/269441/539075) | appear only in report prose | code computes them fresh from `model`/`hardware`/`tp` at call time |
| `mla_phase_filter.py` docstring's "177.97% → 2.43%" | cited as historical rationale in a comment | not read as a runtime value anywhere |

**Verdict: no hard-coded measured TPOT, hardware-validation result, preset
winner/ranking, Qwen-specific predicted cost, MI355X timing constant,
Gate B noise value, or cached profile-lookup result found standing in for
a real, profile-driven prediction.** Every numeric prediction traces to
live simulation output or a live profile-CSV read.

(Separately noticed, pre-existing, not introduced by this task and not a
hard-coded *answer*: `tools/planner.py::_argv`'s `--metrics_config_output_dir`
literal is an absolute path stamped with a previous session's scratchpad
UUID, already committed on `main` before this task started (`git blame`:
`6e2f6b3`). It is a leftover local-path literal, not a measured/predicted
value, and did not affect this task's runs -- flagged here for visibility,
not fixed, since it is out of this task's scope.)

---

## 6. Final-location TP-aware coverage -- re-run, PASS

Against the real installed file, not a scratch copy:

```
Installed file: /work/simulation/Frontier/data/profiling/compute/mi355x/Qwen3-0.6B/linear_op.csv
Observed tp keys: {1: 58, 2: 32, 4: 32}
tp=1: missing=[]
tp=2: missing=[]
tp=4: missing=[]
COVERAGE CHECK: PASS
```

Identical to docs/tasks/66's own result -- no profile data was touched by
this task, as instructed.

---

## 7. Real Frontier evaluations, TP=1/2/4 -- QK-norm blocker gone, a SECOND, different blocker found

Ran the real `tools/planner.py` evaluation path (same mechanism
`evaluate()` uses: a real subprocess of `frontier.simulator.Simulator`,
`cwd=FRONTIER_ROOT`), `topology=domain8`, `model.profiled_tp=(1,2,4)`
explicit, Gate C's own frozen workload
(`Workload(num_requests=32, qps=4.0, prefill_tokens=5, decode_tokens=32)`),
`feasible_num_blocks` computed fresh (134624/269441/539075, matching
docs/tasks/66 exactly -- model config still read correctly):

| TP | result |
|---|---|
| 1 | **FAILED** -- `ValueError: Failed to create predictor of type 'random_forrest': float division by zero` |
| 2 | **FAILED** -- identical error |
| 4 | **FAILED** -- identical error |

**The `use_qk_norm` `ValueError` from docs/tasks/66 is confirmed gone.**
Traced with a full (non-truncated) traceback, reproduced directly
in-process (bypassing the subprocess's string-only error capture): the
run now gets past `use_qk_norm` filtering, successfully trains 21
family-scoped sklearn models from the real installed profile, and only
then fails, inside Frontier's own
`SklearnDisaggregationExecutionTimePredictor.__init__` →
`_simulate_and_store_routing` → `_generate_expert_allocations`
(`frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py:473`):

```python
allocation_ratios = [1.0 / total_expert_num] * total_expert_num
```

`total_expert_num` is `Qwen3-0.6B`'s own real `total_experts=0` (it is
dense, not MoE) -- `1.0 / 0` raises `ZeroDivisionError`, wrapped by
`ExecutionTimePredictorRegistry.get`'s own bare `except Exception` into
the `ValueError` seen above. This code path
(`_simulate_and_store_routing`, called for `PREFILL`/`DECODE_FFN`/`DECODE`
cluster types, `PREFILL` always applies) computes MoE expert-routing
allocations **unconditionally**, with no `total_expert_num == 0` /
`is_moe` guard anywhere in the function -- confirmed by reading the full
function body, not guessed from the traceback alone.

### Why this was never seen before

`SimulationEvaluator.can_evaluate` (`tools/planner.py`) has never gated on
`is_moe`/`total_experts`, and every prior real Frontier evaluation this
project's own history documents used an MoE model (nonzero
`total_experts`: Phi-tiny-MoE-instruct, Mixtral-8x7B, Llama-3.1-405B-Instruct-FP8
-- itself MoE despite the "Llama" name, confirmed via its own model JSON
in earlier tasks). Dense models with real installed profiles (e.g.
`Llama-2-7b-hf`) appear in this repo only as static CSVs used to rehearse
the coverage-checker's parser (`tests/test_gate_c1_coverage.py`'s own
docstring says so explicitly) -- never pushed through a real, end-to-end
`pd-af-disaggregation` `evaluate()` call. **Qwen3-0.6B is the first
genuinely dense model this project has ever evaluated for real**, which
is exactly why this bug was never reached until now -- not a regression
this task introduced, and not related to QK-norm.

### Per task §5/§10 checklist, for all three TP values:

- Profile lookup succeeds? **Yes** (confirmed: 21 real models trained
  from the real installed profile, past the `use_qk_norm` gate).
- Exact-key `KeyError`? No.
- `UNKNOWN` returned? No -- loud, hard failure, correct/safe.
- Silently-accepted extrapolation? No.
- Finite, traceable prediction? **No prediction was produced** -- the
  pipeline fails during predictor construction, before pricing any
  candidate.

### Why this was not fixed in this task

Fixing it means changing behavior inside either the external Frontier
checkout (`/work/simulation/Frontier`, a separate pinned repo, not part
of `dc-sim`'s own zones at all) or a new guarded patch module under
`src/integration/` (the same established pattern as every prior adapter
in this project -- e.g. a `moe_routing_dense_model_guard.py` skipping
`_simulate_and_store_routing`'s per-expert allocation when
`total_expert_num == 0`). **`AGENTS.md`'s own zone rule is explicit:
"Human-only -- anything touching event semantics, time ownership,
completion revision, or upstream coupling. That means ... all of
`src/integration/`. Agents may write tests here but not
implementations."** This task's own QK-norm fix stayed inside the
agent-safe zone (`tools/planner.py`, wiring an already-existing,
already-reviewed flag). A new patch for this second bug would not.
Consistent with this project's own established culture (every prior real
blocker in this initiative -- docs/tasks/61-66 -- was reported and left
for explicit follow-up rather than patched on the spot), this is reported
here, not patched.

---

## 8. Uncertainty/tie behavior, hardware noise floor

Not reached -- no candidate was successfully priced for TP=1, 2, or 4, so
there is nothing to compare, tie, or rank. Gate B's own noise-floor
artifacts (`single-mi355x-tp1/tp2`, `dual-mi355x-crosshost-tp2/tp4`, n=5
each) were not read, touched, or referenced by this task's evaluation
attempts -- not needed, since no planner prediction exists yet to compare
them against.

---

## 9. Planner handoff

**Not generated.** Per task §13 ("only then generate ... if TP=1/2/4 all
pass") and this project's own established practice
(docs/tasks/66's own identical stance): evaluation did not succeed for
any of the three required TP values, so no `DeploymentManifest`/
`PlannerPrediction`/decision-descriptor package was produced. No hardware
execution, noise pilot, or planner-vs-hardware comparison was run or
started.

---

## 10. Provenance

```
qk_norm_evaluation_mismatch: FIXED and verified (tools/planner.py:287-288,
  tests/test_gate_c1_qk_norm_evaluation_parity.py, new)
compatibility_stack (evaluation): collective=True, sglang_replica_scheduler=True,
  qk_norm_allowlist_fix=True (new) -- mla_phase_filter not applicable (dense model)
coverage_check_final_location: PASS -- zero missing keys, tp in {1,2,4}
full_test_suite: 412 passed, 16 skipped (unchanged skip set)
import_direction_check: PASS
hard_code_audit: no measured/predicted answer found embedded in code
second_blocker: ZeroDivisionError in frontier/execution_time_predictor/
  sklearn_disaggregation_execution_time_predictor.py:473 (_generate_expert_allocations),
  total_expert_num=0 for a dense model, unconditional MoE-routing simulation
  -- pre-existing Frontier bug, first reachable now because Qwen3-0.6B is
  this project's first genuinely dense real-evaluation model
evaluation_status: "FAILED -- all three TP values, SAME NEW root cause (not qk_norm)"
planner_handoff: null -- not generated, evaluation still fails
```

---

## Final answers

**A. Was the previous session's QK-norm diagnosis correct?**
**YES** -- reproduced independently in this session, confirmed against
the live `ModelConfig`, the installed profile's own metadata, and the
predictor's own filter source.

**B. Do profile collection and Frontier evaluation now use compatible
model semantics?**
**YES** -- both now resolve `Qwen3-0.6B.use_qk_norm = True`, confirmed
live and by a new regression test guarding the wiring.

**C. Does final-location TP-aware coverage still pass?**
**YES** -- PASS, zero missing keys, TP=1/2/4, re-run against the real
installed Frontier location.

**D. Does Frontier now evaluate Qwen3-0.6B/MI355X at TP=1?**
**NO** -- the QK-norm blocker is gone, but a second, different, real,
pre-existing Frontier bug (dense-model MoE-routing `ZeroDivisionError`)
now blocks it, identically for every TP.

**E. At TP=2?**
**NO** -- same second blocker, identical error.

**F. At TP=4?**
**NO** -- same second blocker, identical error.

**G. Did any evaluation use `UNKNOWN`, silent extrapolation, or an
exact-key miss?**
**NO** -- both blockers found (QK-norm, now fixed; the new
`ZeroDivisionError`) are loud, hard failures at construction time, before
any prediction is produced. No `UNKNOWN`, no silent extrapolation, no
exact-key `KeyError`.

**H. Did the hard-code audit find any measured/predicted result embedded
in code?**
**NO.**

**I. Was the Stage 2 Gate C planner handoff generated and validated?**
**NO** -- not generated, per instruction, since evaluation still fails
(now for the second, different reason).

**J. Is the project ready for the next step (planner manifest → real
hardware execution → decision validation)?**

**NO.** The QK-norm task this session was scoped to fix is fixed and
verified. A second, independent, real blocker remains, confirmed not
caused by this fix and not fixable within this task's own agent-safe
scope (`src/integration/` implementations are human-only per `AGENTS.md`).
Next step is a human decision on how to fix the dense-model MoE-routing
`ZeroDivisionError` -- e.g. a new guarded `src/integration/` patch
skipping per-expert routing simulation when `total_expert_num == 0` (same
established pattern as every other adapter in this initiative), or a
Frontier-side fix -- before real TP=1/2/4 evaluation, and therefore
before any planner handoff, can be attempted again.
