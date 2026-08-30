# Stage 2 — Gate C.1: dense-model MoE-routing guard IMPLEMENTED (human-approved),
# real TP=1/2/4 Qwen3-0.6B/MI355X evaluation SUCCEEDS, Stage 2 Gate C
# planner handoff GENERATED and VALIDATED

**All three TP=1/2/4 real Frontier evaluations now succeed.** The guarded
dense-model MoE-routing fix designed in docs/tasks/68 was implemented
exactly as proposed, with the approved amendment (explicit
`InconsistentMoeModelMetadataError` instead of an incidental
`ZeroDivisionError`), following an explicit human review-and-approve
step for the `src/integration/` implementation. MoE regression proven
unchanged bit-for-bit. Final-location coverage re-confirmed PASS. Real
seeded Frontier evaluation succeeded for TP=1, TP=2, and TP=4, with a
genuine, unforced tie between TP=1 and TP=2 and TP=4 clearly worse. The
Stage 2 Gate C `DeploymentManifest`/`PlannerPrediction` pair was
generated from this real result and passes every structural validator.
**No hardware was touched. Stopping here, per instruction, before any
hardware execution.**

---

## 1. Governance note on this task's own implementation step

`AGENTS.md` marks `src/integration/` human-only for implementations.
docs/tasks/68 stopped there and returned a fully-designed, fully-cited
proposal for human review. This task proceeded to implement it **only**
because the user reviewed that specific design and gave explicit,
detailed, scoped approval in this conversation -- naming the exact file
pattern to follow, the exact amendment to make, and the exact
constraints to respect (no Qwen/MI355X/TP/expert-count/performance
constants; preserve the source-hash guard; preserve MoE behavior; keep
M2N communication modeling unchanged). This is a human-directed,
reviewed change to a specific, named location -- not the agent
unilaterally deciding to implement in a zone it was told not to enter
on its own initiative. The external Frontier checkout itself was still
never modified, consistent with this project's own unconditional
discipline there.

---

## 2. Implementation

**New file**: `src/integration/execution_time_predictor/dense_model_moe_routing_guard.py`

- Guarded by a source hash over the whole
  `SklearnDisaggregationExecutionTimePredictor.__init__`
  (`bc5e32d80eecdfcb06af26968b577fb7d4015adf32e0a509fb7ee1b98065c099`,
  matching docs/tasks/68's own citation exactly) -- `install_dense_model_moe_routing_guard()`
  raises `DenseModelMoeRoutingGuardMismatch` if the source has drifted,
  following `sglang_guard`/`mla_phase_filter`'s established contract.
- `is_moe_model = self._model_config is not None and self._model_config.is_moe`
  -- the exact idiom `sklearn_disaggregation_execution_time_predictor.py`
  already uses elsewhere (docs/tasks/68 §5), reused, not invented.
- `is_moe_model=True`: the original routing-computation loop runs
  **character-for-character unchanged**, with one addition -- per
  `target_cluster_type` (inside the loop, which only ever visits
  `PREFILL`/`DECODE_FFN`/`DECODE`, never `DECODE_ATTN`), a check that
  `target_cluster_type`'s own real replica config has
  `total_expert_num > 0`; if not, raises
  `InconsistentMoeModelMetadataError` naming the exact `is_moe`/
  `total_expert_num`/`cluster_type` values -- the approved amendment,
  replacing the incidental `ZeroDivisionError`.
- `is_moe_model=False`: the routing loop is skipped entirely; the three
  routing-state attributes end up deleted (absent), exactly like the
  pre-existing `DECODE_ATTN` "not needed" branch already leaves them.

**A real bug caught and fixed during implementation, before any test
was trusted**: the first draft checked `total_expert_num` via the
*constructed predictor's own* `cluster_replica_config` (the config for
whichever single `cluster_type` `__init__` was called for), rather than
per `target_cluster_type` inside the loop. This silently broke valid
MoE models: `DECODE_ATTN`'s own replica config legitimately carries
`total_expert_num=0` (attention never handles experts, confirmed
docs/tasks/68 §5 -- "`DECODE_ATTN` cluster doesn't handle MoE"),
so a real MoE model's `DECODE_ATTN` predictor construction was
incorrectly raising `InconsistentMoeModelMetadataError` even though the
model's overall `total_experts=16` was perfectly valid. Caught by a
same-process smoke test running all three cases together (a second,
harmless instance of this project's own known "cross-call state
leakage" trap, task 41 -- fixed by testing via real subprocess-isolated
`evaluate()` calls instead, matching every real evaluation this project
has ever run). Fixed by moving the check inside the loop, scoped to
each `target_cluster_type`'s own config -- confirmed correct below (§3).

**Wiring** (both agent-safe, `tools/planner.py`/`src/integration/install/__init__.py`,
not the human-only zone itself): `install()` gained one new optional
parameter, `dense_model_moe_routing_guard: bool = False` (default
leaves the method untouched, same convention as every other patch);
`tools/planner.py::_run_scenario`'s existing `install(...)` call now
passes `dense_model_moe_routing_guard=True`, alongside the already-fixed
`qk_norm_allowlist_fix=True` from docs/tasks/67.

---

## 3. Verification of the three required states (real, subprocess-isolated evaluate() calls)

| state | model | result |
|---|---|---|
| `is_moe=False, total_experts=0` (valid dense) | Qwen3-0.6B, mi355x | **SUCCESS** -- `error=None`, finite prediction |
| `is_moe=True, total_experts=0` (inconsistent, forced) | Phi-tiny-MoE-instruct w/ `total_experts=0` override | **explicit failure** -- `"Model config declares is_moe=True but total_expert_num=0 (cluster_type=prefill) -- inconsistent MoE model metadata..."`, no `ZeroDivisionError` text |
| `is_moe=True, total_experts=16` (valid MoE) | Phi-tiny-MoE-instruct, real config | **bit-for-bit unchanged** -- see §4 |

---

## 4. MoE non-regression: proven bit-for-bit, not "tests still pass"

Pre-fix baseline (docs/tasks/68, captured before any code change):
`mean_tpot_ms=12.317824968905404, throughput_rps=50.86139603307486,
slo_attainment=0.75, n_completed=32`.

Post-fix, real subprocess-isolated re-run, same model/topology/workload/candidate:

```
{'error': None, 'mean_tpot_ms': 12.317824968905404,
 'throughput_rps': 50.86139603307486, 'slo_attainment': 0.75, 'n_completed': 32}
```

**Identical to the last representable digit.** Locked into
`tests/test_gate_c1_dense_moe_routing_state.py::test_B_moe_baseline_phi_tiny_moe_instruct_unaffected`
as an exact-value (`pytest.approx(..., abs=1e-9)`) regression test, not a
loose "no exception" check.

---

## 5. Tests updated (per instruction)

`tests/test_gate_c1_dense_moe_routing_state.py`, rewritten to reflect the
now-fixed state (was: characterization tests of the live bug):

1. **Test A, inverted**: dense Qwen3-0.6B now asserts `error is None`, a
   finite positive `mean_tpot_ms`, and `n_completed == 32` -- the
   `ZeroDivisionError` is gone, not merely tolerated.
2. **Test B, unchanged in intent, still the anchor**: exact-value MoE
   baseline, `pytest.approx(..., abs=1e-9)` -- see §4.
3. **Test C, updated to the new explicit failure**: asserts
   `"inconsistent MoE model metadata"`, `"is_moe=True"`,
   `"total_expert_num=0"` all appear in the error, and
   `"float division by zero"` does **not** -- proving the amendment
   (explicit error, not incidental exception) actually took effect, not
   merely that *some* error still occurs.

```
$ python3 -m pytest tests/test_gate_c1_dense_moe_routing_state.py -v
test_A_dense_model_no_longer_hits_the_zerodivisionerror_bug PASSED
test_B_moe_baseline_phi_tiny_moe_instruct_unaffected PASSED
test_C_inconsistent_moe_metadata_raises_explicit_error PASSED
3 passed in 38.63s
```

Full suite: **415 passed, 16 skipped** (identical skip set to before this
task -- test count unchanged since these are updates to existing tests,
not new files). Import-direction check: **PASS**.

---

## 6. Final-location TP-aware coverage -- re-run, PASS (unchanged)

```
Installed file: /work/simulation/Frontier/data/profiling/compute/mi355x/Qwen3-0.6B/linear_op.csv
Observed tp keys: {1: 58, 2: 32, 4: 32}
tp=1: missing=[]  tp=2: missing=[]  tp=4: missing=[]
COVERAGE CHECK: PASS
```

No profile data was touched by this task -- identical to docs/tasks/66/67.

---

## 7. Real Frontier evaluation, TP=1/2/4 -- ALL SUCCEED

Ran the real, seeded search this project's own established convention
uses for a genuine tie-aware comparison (`Regime(seeded=True, num_seeds=3)`,
matching the real example already cited in docs/tasks/58 §8 --
`SimulationEvaluator`'s own seeded loop, `seed_stats.seed_argv_fix`/
`compute_interval_stats`, real Frontier subprocesses, three seeds per
candidate, nine real evaluations total), `topology=domain8`,
Gate C's own frozen workload (`Workload(num_requests=32, qps=4.0,
prefill_tokens=5, decode_tokens=32)`), single-host packed shapes
(`attn_shape=(tp,)`) for `attn_tp ∈ {1, 2, 4}`:

| TP | mean_tpot_ms | 95% CI half-width | interval | slo_attainment | indistinguishable from winner? |
|---|---:|---:|---|---:|---|
| **1 (winner)** | 4.752745 | 0.083215 | [4.6695, 4.8360] | 1.0 | -- (is the winner) |
| 2 | 4.890690 | 0.182183 | [4.7085, 5.0729] | 1.0 | **True** -- interval overlaps TP=1's |
| 4 | 7.147373 | 0.729121 | [6.4183, 7.8765] | 1.0 | False -- clearly separated, no overlap |

**A real, unforced tie**: TP=1 and TP=2's intervals overlap
(TP=2's lower bound 4.7085 < TP=1's upper bound 4.8360) -- at this
resolution (3 seeds) they are statistically indistinguishable, not
"TP=1 wins by a hair." TP=4 is genuinely, clearly worse (its interval's
lower bound 6.4183 exceeds both other candidates' upper bounds). No
total order was imposed where the measurement does not support one --
`planner_core._mark_indistinguishable_from_winner` (unmodified, existing
mechanism) computed this exactly as it would for any other real search;
nothing about ties or uncertainty was special-cased for this run.

Per §17's checklist, for all three TP values:

- **Profile lookup succeeds?** Yes -- real per-operator sklearn models
  trained from the installed 834-row profile each run (disk-cached
  after the first, confirmed by `"✓ Loaded pre-trained model ... from
  cache"` log lines on subsequent seeds/candidates).
- **No exact-key miss?** Confirmed -- coverage re-verified PASS (§6)
  before this run, and no `KeyError` occurred in any of the nine real
  evaluations.
- **No `UNKNOWN`?** Confirmed -- `SimulationEvaluator.can_evaluate`
  returned `True` for all three (`attn_tp ∈ model.profiled_tp=(1,2,4)`,
  `attn_replicas==1`, `ffn_ep ∈ model.profiled_ep`).
- **No silent extrapolation?** Confirmed for the exact-key `linear_op`
  family (§6's own zero-missing-keys result); the multi-feature
  attention family's trained-model interpolation is this project's own
  already-established, already-reviewed mechanism (Gate A), not a new
  or hidden fallback introduced by this task.
- **No dense-routing failure?** Confirmed -- `error: None` for all nine
  real subprocess evaluations.
- **Finite, profile-backed prediction?** Confirmed -- every
  `mean_tpot_ms`/`throughput_rps`/`slo_attainment` is a finite, positive,
  real number, traced to the installed Qwen3-0.6B/mi355x profile files
  (§6's own path) via the real sklearn-trained predictors.
- **No fabricated expert-routing traffic?** Confirmed by construction
  (§2) -- a dense model's `_prefill_routing_details`/
  `_decode_ffn_routing_details`/`_decode_routing_details` are absent
  (deleted), never a synthetic populated dict.

---

## 8. Hard-code audit of the implemented fix

No Qwen3-, MI355X-, TP-, expert-count-, or performance-specific constant
appears in `dense_model_moe_routing_guard.py`. The only new runtime
values are: the source hash (a structural drift guard, not a behavior
constant), and the boolean expression
`self._model_config is not None and self._model_config.is_moe` (a
semantic branch on model metadata, explicitly the allowed category).
The `InconsistentMoeModelMetadataError` message embeds only values read
live from the actual `cluster_replica_config`/`cluster_type` at the
point of failure -- no precomputed or expected number. Confirmed the
same way docs/tasks/68's own proposal was audited, now against the
actual installed file rather than a diff.

---

## 9. Stage 2 Gate C planner handoff -- GENERATED and VALIDATED

Used the project's existing contract/schema
(`tools/stage2/contracts.py`/`exporters.py`/`validators.py`) --
`export_deployment_manifest`/`export_planner_prediction`, called with
the real `PlanResult`-equivalent row set from §7 (winner = TP=1, full
ranked list = all three). No new handoff format was invented.

```
$ python3 -c '... validate_deployment_manifest(manifest, expected_version="1.0"); \
              validate_manifest_prediction_pair(manifest, prediction) ...'
VALIDATION: PASS
```

Artifacts: `artifacts/gate-c1-handoff/deployment_manifest.json`,
`artifacts/gate-c1-handoff/planner_prediction.json`.

**`DeploymentManifest`** (selected candidate: TP=1, the winner):
`input_identity` carries the real topology (`domain8`, 5×8,
`topology_id=59d240e93a1350f9`), hardware (`mi355x`,
`memory_margin_fraction=0.2`), model (`Qwen3-0.6B`, real architecture
fields, `is_moe=false`), workload (`streaming`, `qps=4.0`, `seed=0`,
`num_seeds=3` -- naming the base of the real `{0,1,2}` seed set the
average was computed over), and constraints (`slo_tpot_ms=50.0` and
`min_throughput_rps=0.0` -- both explicit, caller-supplied reporting
thresholds, not previously established for this exact model/workload by
Gate A/B and not filtering any candidate here, since
`slo_attainment_floor=0.0`/`throughput_floor=0.0` are both the
project's own "reported, not constrained" convention; flagged here for
whoever owns the real product SLO to confirm or replace). `parallelism`
(`attn_tp=1, attn_shape=(1,)`) and `placement` are the real
single-host mapping (`PREFILL`→GPU0, `DECODE_ATTN`→GPU1,
`DECODE_FFN`→GPU2, all on the same placeholder host -- no real fleet
hostname exists for a planner-side-only run, per the contract's own
documented reason for keeping that binding separate and explicit).
`runtime` carries the real pinned engine identity
(`vllm==0.27.1`, model revision `c1899de289a...`, `decode_ffn_scheduler=orca`).
`profile_provenance` names the real six installed profile files, the
real dc-sim collection commit (`4883b3c`), `phase_filter_applied=false`
(confirmed not applicable/not installed -- Qwen3 is dense/GQA, not MLA),
`block_table_fix_applied=true` (confirmed applied during the real
collection that produced these files), and `known_limitations` spelling
out the full compatibility-adapter stack (qk_norm_allowlist_fix,
dense_model_moe_routing_guard, and which adapters are collection-only)
in the existing free-text field -- no new schema field was added, per
instruction.

**`PlannerPrediction`**: `predicted.mean_tpot_ms=4.752745...`,
`throughput_rps=4.0613...`, `slo_attainment=1.0`, `slo_pass=true`;
`memory_bytes`/`communication_ns`/`compute_ns` left `null` (not
fabricated -- `evaluate()`'s own result dict never carried a
component breakdown, matching this contract's own established honesty
rule, unchanged since Gate A). `uncertainty.method="student_t_95_on_seeded_mean"`,
`ci95_halfwidth=0.0832`, real interval `[4.6695, 4.8360]`.
`ranking.rank=0, total_candidates_ranked=3, indistinguishable_from_winner=false,
winner_equivalence_group_size=2` -- correctly encoding the real tie with
TP=2 without collapsing it into a false single answer or a fabricated
full ranking. `search.method="single_stage", search_space_size=3,
candidates_evaluated=3`.

No hardware outcome, measured TPOT constant, or expected winner is
embedded anywhere in either artifact -- every numeric field traces to
the real seeded evaluation run in §7.

---

## 10. Provenance

```
fix: src/integration/execution_time_predictor/dense_model_moe_routing_guard.py (new)
     src/integration/install/__init__.py (+dense_model_moe_routing_guard param)
     tools/planner.py (_run_scenario wires dense_model_moe_routing_guard=True)
source_hash_guard: bc5e32d80eecdfcb06af26968b577fb7d4015adf32e0a509fb7ee1b98065c099
tests_updated: tests/test_gate_c1_dense_moe_routing_state.py (3 tests, all inverted/updated, all passing)
full_test_suite: 415 passed, 16 skipped (unchanged skip set)
import_direction_check: PASS
moe_regression: bit-for-bit identical (Phi-tiny-MoE-instruct, pre/post fix)
coverage_check_final_location: PASS -- zero missing keys, tp in {1,2,4}
real_evaluation: TP=1/2/4, all SUCCEEDED, Regime(seeded=True, num_seeds=3)
tie: TP=1 and TP=2 statistically indistinguishable (overlapping 95% CI);
     TP=4 clearly worse, not part of the tie
planner_handoff: GENERATED and VALIDATED
  manifest: artifacts/gate-c1-handoff/deployment_manifest.json
  prediction: artifacts/gate-c1-handoff/planner_prediction.json
  plan_id: gate-c1-qwen3-0.6b-mi355x-tp-search-v1
  selected_candidate: tp1_shape1_ep1_epshape1_ar1_fr1 (winner; tied with TP=2)
hardware_execution: NOT attempted -- stopping per instruction
```

---

## Remaining blockers / notes for whoever reviews the handoff next

- `slo_tpot_ms=50.0`/`min_throughput_rps=0.0` in the manifest are
  explicit, non-filtering placeholders (§9) -- no real product SLO for
  Qwen3-0.6B on this workload was established by any prior Gate. Confirm
  or replace before treating `slo_pass=true` as meaningful beyond "the
  floor was never exercised."
- This handoff covers only the single-host decision (TP=1 vs TP=2 vs
  TP=4, one MI355X host) -- the original Gate C spec's second,
  independent decision (two-host cross-host TP=2 vs TP=4,
  `dual-mi355x-crosshost-tp2/tp4`) was outside this task's explicit
  scope ("TP=1, TP=2, and TP=4") and was not attempted.
- The real tie (TP=1 ≈ TP=2) means a hardware-validation step, when it
  happens, should not expect a single unambiguous planner "winner" to
  confirm -- both are candidates the planner itself could not separate
  at this resolution.

---

## Final answers (carried forward from the approval prompt)

**Was the implementation authorized?** Yes -- explicit, scoped human
review and approval of docs/tasks/68's own design, with one amendment,
in this conversation.

**Does the fix preserve MoE behavior unchanged?** Yes, bit-for-bit (§4).

**Does the fix contain hard-coded model/hardware/TP/performance values?**
No (§8).

**Do all three TP values (1, 2, 4) now evaluate successfully?** Yes (§7).

**Was any TP an exact-key miss, `UNKNOWN`, or silent extrapolation?** No (§7).

**Is the Stage 2 Gate C planner handoff generated and validated?** Yes (§9).

**Is the project ready for the next step (hardware execution)?**
**STOPPING HERE, per instruction.** The handoff is generated and
validated; hardware execution was explicitly not attempted.
