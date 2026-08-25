# Task 53 — Patch the two profiling defects

Branch: `task-53-predictor-patches`, branched from `task-52-predictor-error`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`. No GPU used.

260 tests pass (254 unchanged + 6 net new: 5 from
`test_mla_phase_filter_guard.py`, 1 from
`test_attention_block_table_fix_guard.py`), 5 skipped (that same file's
remaining tests, which need `torch` -- absent from this sandbox, no GPU
involved), and `python3 tools/check_import_direction.py` exits 0.

---

## 1. Fix A — the missing phase filter

Delivered as a runtime patch, following `install()`'s established
convention exactly (task 20, task 47): one function replaced
(`SklearnExecutionTimePredictor._train_mla_attention_layer_models`),
guarded by a source hash over the current implementation, defaulted off
(`install(..., mla_phase_filter=True)`), with a test that installation
refuses when the upstream function has changed.

- `src/integration/execution_time_predictor/mla_phase_filter.py`
- Wired into `install()` (`src/integration/install/__init__.py`) as a new
  `mla_phase_filter: bool = False` parameter.
- `tests/test_mla_phase_filter_guard.py`, `tests/_mla_phase_filter_probe.py`
  (a subprocess probe against real profiled data, mirroring
  `test_kv_cache_page_size_vs_memory_planner.py`'s own established pattern
  — task 48 — for anything that needs a real, CSV-backed predictor).

**The discriminator is not new.** `SklearnExecutionTimePredictor` already
has `_mla_operator_phase_kind`, used at *prediction* time
(`_is_mla_operator_applicable_to_batch`) to decide whether an operator's
cost even applies to a batch — and it already raises if any MLA operator's
own declared `phases` don't resolve cleanly to exactly one of
`"cache_write"` / `"prefill"` / `"decode"`. Fix A calls this classifier
directly rather than re-deriving one, so a future MLA operator with an
ambiguous phase declaration fails training the same way it would already
fail prediction — not silently.

### 1.1 Before-and-after leave-one-out error, every MLA operator (§3.1)

Real leave-one-out cross-validation (RandomForest, Frontier's own
hyperparameter grid, `deepseek-v3`/`mi355x`/`attn_tp=8`) — not the in-sample
score Frontier's own training log reports (that scores the refit model
against its *own* training rows; Task 52's report explains why this is a
different, less informative number):

| operator | phase kind | rows before | rows after | LOO before | LOO after |
|---|---|---|---|---|---|
| `attn_mla_kv_cache_save` | cache_write | 13 | **13** (unfiltered) | 3.09% | **3.09%** (identical) |
| `attn_mla_prefill_kv_up_proj` | prefill | 13 | 5 | 14.86% | **0.77%** |
| `attn_mla_prefill` | prefill | 13 | 5 | 16.84% | **2.49%** |
| `attn_mla_decode_q_latent_proj` | decode | 13 | 8 | 63.78% | **3.89%** |
| `attn_mla_decode` | decode | 13 | 8 | 177.97% | **2.43%** |
| `attn_mla_v_up_proj` | decode | 13 | 8 | 53.55% | **2.81%** |

**Every operator improves, and `attn_mla_kv_cache_save` — the one operator
that genuinely spans both phases — is bit-for-bit unaffected**, as it must
be (§1.2). **No operator ends with zero training rows after filtering**
(§3.1's own required check; the smallest is 5, for the two prefill-phase
operators) — this is also enforced live, not only measured: the patched
function keeps the original's `if op_attention_df.empty: raise ValueError`
check, now evaluated *after* the phase filter, so a future profiling sweep
narrow enough to empty one phase out fails loudly rather than silently
training on nothing.

(Row counts above are the *raw* rows passed to `GridSearchCV.fit` — the
number that determines fit quality. A separate, smaller number — the size
of each operator's `_frontier_exact_lookup`, used for exact-match prediction
— is 12 before, 4 (prefill)/8 (decode)/12 (cache-write) after; the gap from
13 (raw) is one pair of profiled rows that happen to share every one of the
19 feature columns and collapse into one lookup entry via that lookup's own
groupby-and-average, unrelated to phase. §3's own test file checks this
second number directly, since it is what the automated test observes; this
table reports the first, since it is what determines the LOO figures.)

### 1.2 Whether any operator legitimately spans both phases (§3.1)

**Yes — exactly one, and it is handled by construction, not by a special
case this patch had to add.** `attn_mla_kv_cache_save` has role
`CACHE_WRITE` and phases `(PREFILL, DECODE, MIXED)` (`_ALL_PHASES`, per
`frontier/attention/families.py`) — a KV-cache write happens on every row
regardless of phase, so its training set must include all rows. Fix A's
own `_mla_operator_phase_kind` call routes this operator to the
`"cache_write"` branch, which is a no-op (no filter applied) — the same
code path the pre-existing prediction-time classifier already uses for the
same operator, for the same reason. Every other MLA operator resolves
cleanly to `"prefill"`-only or `"decode"`-only phases; `_mla_operator_phase_kind`
itself raises if a future operator's phases don't fit this pattern, so this
patch cannot silently mis-file an ambiguous operator into the wrong bucket
— it would fail loudly at the same point prediction already would.

## 2. Fix B — the block-table aliasing

**Separated cleanly.** `AttentionWrapper._get_input_tensors`
(`frontier/profiling/attention/attention_wrapper.py` — not
`_get_standard_input_tensors`; see §7) is a single, self-contained method;
the branch's own fix to it needs no new import and no helper the rest of
the branch adds (confirmed by re-reading the branch's own diff to this
file directly: the only other changes are `_mla_result_fields` and the
three `profile*` methods' MLA-column additions, none of which
`_get_input_tensors` calls or is called by). The fix cherry-picked here is
exactly: give each sequence a distinct, non-overlapping block range
(`next_block_index`, incremented per sequence), instead of every sequence
reusing `block_table=list(range(num_blocks))`. Nothing else from the
branch — not the new MLA wrapper classes, not the deleted/replaced CSVs
Task 51 found unsafe, not the `AttentionBackend` enum wiring — is included.

- `src/integration/profiling/attention_block_table_fix.py`
- **Not wired into `install()`**, unlike every other patch in this
  project, for two independent reasons, both real: (1) `_get_input_tensors`
  only runs during a profiling CLI invocation
  (`python -m frontier.profiling.attention.main`), never during a
  simulation — `install()` is called before `frontier.main`, and profiling
  calls neither, so there is no shared call site to hook (confirmed by
  Task 51/52: `frontier/profiling/` is never imported by the simulation
  path). (2) `attention_wrapper.py` imports `torch` unconditionally at
  module level, and this sandbox does not have `torch` installed —
  confirmed directly (`import torch` fails, no GPU involved, simply
  absent; grepping shows no other module this project's own test suite
  touches has ever needed it). Wiring this into `install()`'s own
  module-level imports would make importing `integration.install` itself
  fail everywhere `torch` is unavailable, breaking every one of this
  project's 254 tests that transitively import it.
- `tests/test_attention_block_table_fix_guard.py`: one test
  (`test_expected_hash_matches_the_checked_out_file`) runs without
  `torch`, since it recomputes the guard's own hash by parsing the checked
  -out file's source text directly (`ast`, no import); the rest
  (`pytest.importorskip("torch", ...)`, via a non-autouse fixture so only
  they are skipped) exercise the actual patch — pre-patch aliasing,
  post-patch distinct ranges, the bounds check moving with the fix, and
  the standard idempotency/hash-mismatch pair — and would run wherever
  `torch` is present. All five are skipped in this sandbox, not failed.

**A real, named consequence of the fix, found while writing its own
test**: the bounds check moves too (`next_block_index + num_blocks >
self.max_num_blocks`, not just `num_blocks > self.max_num_blocks`). A
batch that used to "fit" by having every sequence alias the same blocks
can legitimately need more distinct blocks than `max_num_blocks` once
sequences stop sharing them — a future profiling run using this fix at a
large `batch_size` may need a larger `max_num_blocks` than the same sweep
needed before, and would now raise where it previously silently succeeded
with contaminated data. Named here, not encountered on real data, since no
re-profiling was run.

## 3. Regression (§3.2)

Both required checks, run with the patched code present in the tree (Fix A
importable, wired into `install()`'s own optional parameter; neither fix
installed by either script below, matching how every existing tool in this
project calls `install()` — with defaults):

**Task 33's sixteen-row table:**
```
WINNER: tp=2 shape=(2,) mean_tpot_ms=11.6803
```
Bit-identical to every prior reproduction this project has recorded.

**Task 36's two-fabric result:**
```
=== domain8_40gpu ===
WINNER: tp=8 shape=(8,) mean_tpot_ms=326.2362
=== domain4_40gpu ===
WINNER: tp=8 shape=(4,3,1) mean_tpot_ms=446.5146
```
Bit-identical to every prior reproduction this project has recorded.

Both use `h800` and dense attention exclusively — neither fix touches
either code path (Fix A only changes `_train_mla_attention_layer_models`,
never called for `DENSE_ATTENTION_FAMILY`; Fix B is not installed by
default and, per §2, has no call site that would reach it here regardless)
— confirmed by running them, not only by this argument, per this task's
own §3.2 instruction and Task 51's own warning that a data or fitting
change can move a number without breaking a run.

## 4. What Fix A is worth to a prediction (§3.3)

`deepseek-v3` cannot complete a run today (Task 48's own `mlp_up_proj`
gap), so this is answered at predictor-construction time, as this task's
own §3.3 anticipates — built the real predictor (`deepseek-v3`/`mi355x`/
`attn_tp=8`, the exact "final working command" configuration) with and
without Fix A installed, and queried the trained models directly at the
same shapes Task 52 used to demonstrate flat extrapolation:

| shape | before (ms) | after (ms) | ratio |
|---|---|---|---|
| decode @ profiled edge (`kv_cache_size=96`, `batch_size=1`) | 0.116432 | 0.116834 | 1.003 |
| decode @ real workload shape (`kv_cache_size=550`, `batch_size=5`) | 0.206323 | 0.208762 | 1.012 |
| prefill @ real workload shape (512 tokens) | 0.136045 | 0.136453 | 1.003 |

**The predicted value barely moves (0.3%–1.2%) even though the fit quality
improves dramatically (§1.1, 54%–178% LOO MAPE down to 2.4%–3.9%).** This
is a real, precise, and somewhat surprising finding worth stating plainly:
Fix A repairs the model's *internal consistency* within the profiled
range, but every shape a real run actually requests lies *outside* that
range (Task 52's own §2), where the prediction is a flat extrapolation —
whatever value happens to sit in the forest's rightmost leaf. Fixing the
training-row contamination changes which value that leaf holds only
slightly; it does not touch the mechanism (Candidate B) that decides
*which* leaf a real request lands on, or that the answer is flat at all.
**Fix A is a real, independent, low-cost correctness fix — measured here
to be worth very little to the one simulated result this checkout can
actually query, because the extrapolation problem it doesn't touch
dominates that result.** This is not a reason to skip Fix A (§3.1's own
fit-quality case stands on its own), only a precise statement of what it
does and does not buy.

## 5. What an extrapolation guard would require (§1, named not built)

- **Per-operator profiled bounds**, stored alongside each trained model at
  fit time (e.g. min/max of each feature actually seen in
  `op_attention_df`) — cheap to compute, not currently stored anywhere.
- **A check at the one call site both prediction paths share**:
  `_get_on_demand_prediction` (multi-feature models, including every MLA
  operator) and the dense single-feature dict-lookup path
  (`self._predictions[op][(effective_tokens,)]`) are two *different*
  mechanisms (Task 52's own correction to this task's premise) — a guard
  would need to cover both, not just the one this task's own investigation
  happened to focus on.
- **A policy decision on what "outside range" means and what happens
  then**, genuinely open rather than obvious: per-feature marginal
  min/max (cheap, and exactly correct for a full cross-product grid like
  this one, but not in general — a grid with holes would let a marginal
  check pass an unprofiled *combination* of otherwise-in-range values) vs.
  a real convex-hull/nearest-neighbor-distance check (more correct,
  costs more to compute per prediction); and whether a violation should
  raise (loud, breaks any run that extrapolates at all — including every
  run this project has ever completed successfully, since Task 52 showed
  this happens routinely) or log-and-continue (quieter, but exactly the
  silent behavior this task is trying to move away from).
- **Not built here**, per this task's own explicit instruction (§1) — the
  remedy for the extrapolation problem itself is a wider profiled grid,
  not a guard; a guard only makes the current gap loud instead of silent.

## 6. Anywhere this specification is wrong

1. **The method's real name is `_get_input_tensors`, not
   `_get_standard_input_tensors`.** Confirmed by reading
   `attention_wrapper.py` directly (`grep -n "def _get_standard_input_tensors"`
   finds nothing; `_get_input_tensors` is the method both this task's own
   §1 and Task 51/52's reports describe). This is Task 51/52's own
   paraphrase, not this checkout's actual name — worth correcting since
   Fix B's own module and hash are written against the real name.
2. **"At two sites inside the per-sequence loop" is exactly right, once
   read as two *lines* within one method's one loop** (the bounds check
   and the `block_table` construction), not two separate methods. An
   initial reading of this task's own wording could suggest
   `_get_mixed_input_tensors` is a second affected method; it is not
   touched by the branch's own diff at all (confirmed directly), and
   `_get_true_mixed_input_tensors` already has the non-overlapping-range
   pattern on `main`, per the branch's own comment.
3. **Otherwise this specification held up precisely.** Fix A's own
   pattern-matching to task 20/47 was exact; the phase-kind classifier
   already existing in the file (rather than needing to be invented) was
   confirmed directly; the "confirm no operator ends with zero rows" check
   named a real, if not-yet-triggered, risk that the patched function now
   enforces live; and Fix B's own separability held up exactly as this
   task's own §2 predicted it might ("if Fix B cannot be cleanly
   separated... a botched extraction is worse than a deferred one") --
   it separated cleanly, so nothing was deferred.

## What shipped

- `src/integration/execution_time_predictor/mla_phase_filter.py` — Fix A,
  wired into `install()` as `mla_phase_filter: bool = False`.
- `src/integration/profiling/attention_block_table_fix.py` — Fix B,
  standalone, not wired into `install()` (§2).
- `tests/test_mla_phase_filter_guard.py`,
  `tests/_mla_phase_filter_probe.py` — Fix A's guard and behavior tests.
- `tests/test_attention_block_table_fix_guard.py` — Fix B's guard test
  (hash-only, runs without `torch`) and behavior tests (need `torch`,
  skipped in this sandbox).
- `docs/tasks/53-predictor-patches-report.md`, this report.

One commit on `task-53-predictor-patches`, stacked on
`task-52-predictor-error`. Task 33's sixteen-row table and Task 36's
two-fabric result both reproduce bit-identical. 260 tests pass (254 + 6
new), 5 skipped (Fix B's behavioral tests, need `torch`);
`check_import_direction.py` exits 0.
