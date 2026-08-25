# Task 52 — Where does the prediction error come from?

Branch: `task-52-predictor-error`, branched from `task-51-mla-merge`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier at
`/work/simulation/Frontier`.

254 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. Diagnosis only — nothing in Frontier or `dc-sim` was changed. No
GPU was used; Part C is not warranted (§4).

All measurements below were taken by constructing the **real**
`SklearnExecutionTimePredictor` for `deepseek-v3`/`mi355x`/`attn_tp=8`
exactly as `profiling_knowledge/DEEPSEEK_V3_MLA_MI355X_JOURNEY.md`'s own
"final working command" configures it (`co-location`,
`--vllm_v1_scheduler_config_block_size 32`, 512 prefill / 128 decode
tokens, real, non-dummy `mi355x` compute), then querying the trained
models directly — the same object a real `Simulator` would build, not a
toy reconstruction.

---

## 0. What "the symptom" actually is in this checkout

This task's own §1 states the symptom generically ("predictions ...
reported as inaccurate against real serving"). Checked directly: **this
checkout contains no real-vs-simulated serving comparison result for
`deepseek-v3` or any `mi355x` model** — Task 46's own A.3 already found
`tools/inference_bench/` (where such captures would live) absent from
this checkout, and re-confirmed here (`ls`, `find`: no such directory).
`profiling_knowledge/REAL_BENCHMARK_DATA_QUALITY.md` references a real
`deepseek (sglang, vLLM)` capture and states plainly that deepseek is
"safe to compare as-is" (unaffected by the decode-length mismatch bug
documented there) — so a real comparison was evidently run *somewhere*,
but its numeric result, sign, and shapes are not preserved in this
checkout. **The one concrete, quantified accuracy figure that does exist
here** is `DEEPSEEK_V3_MLA_MI355X_JOURNEY.md`'s own: *"Decode-phase MAPE
is high (`attn_mla_decode` ~64%, `attn_mla_decode_q_latent_proj` ~22%,
`attn_mla_v_up_proj` ~22%) because only 8 decode rows exist."* This is
**not** a real-vs-simulated comparison — it is the trained model's own
error against its own training rows (§3 below) — a different question
from §1's. This report treats it as the closest available concrete
instance of "predictions reported as inaccurate," verifies it directly,
and traces where the inaccuracy actually comes from; it also states
plainly where the trace runs into checkout's own missing ground truth.

## 1. Which mechanisms are implicated

**Reproduced the doc's own MAPE figures directly**, training the real
model on `main`'s own committed `attention.csv` (52 rows, `attn_tp=8`
slice = 13 rows after the phase-mixing found in §3):

| operator | doc's cited MAPE | reproduced (in-sample, this run) |
|---|---|---|
| `attn_mla_decode` | ~64% | 76.63% |
| `attn_mla_decode_q_latent_proj` | ~22% | 23.29% |
| `attn_mla_v_up_proj` | ~22% | 19.76% |

Close but not identical (plausibly `sklearn`/environment drift since the
doc's own figures were produced) — close enough to treat the doc's own
citation as accurate and reproducible, not stale or fabricated.

**Sign, established two ways, neither of which is a real-serving
ground truth (§0):**

- The two sign-discriminated candidates from this task's own §1 (backend
  ceiling → overstate/slower; block-table aliasing → understate/faster)
  are both about a *comparison to real serving* this checkout cannot
  make (§0). Concretely, though: **candidate A (backend ceiling) does
  not currently apply to this scenario at all**, sign or no sign — real
  serving would need to run a *tuned* backend for the ceiling to bite,
  and Task 49's own `AITER_KERNELS.md` finding is that `AITER`'s
  prebuilt kernels do not run against host torch on `server1`
  **or** `server3` — the two hosts this repository's own MLA work has
  ever touched. There is no tuned backend actually executing anywhere in
  this fleet for `deepseek-v3`/MLA to diverge from; profiling and any
  hypothetical real run would both be stuck on the same
  `TORCH_SDPA_MLA` reference implementation. A fidelity ceiling that
  nothing on either side of the comparison currently escapes is not an
  active cause of *this* symptom, whatever it might become if `AITER`
  ever worked.
- **Candidate D (the block-table aliasing bug, Task 51) does not explain
  the one piece of quantitative evidence offered for it.** Task 51's own
  three overlapping shapes between the pre-fix and post-fix attention
  data were all `batch_size=1` (checked directly against Task 51's own
  report: the shape key tuples are `('1', '0', 'True', '32', '8', '32',
  '32')` etc. — the first element is `batch_size`). The bug's own named
  mechanism (every sequence in a *batch* aliasing the same cache blocks)
  cannot operate when a batch has exactly one sequence — there is nothing
  for it to alias with. **The 10%–62% discrepancy Task 51 measured must
  have some other cause**; it is real evidence of *something*, but not
  of this mechanism, and Task 51's own report already flagged this
  possibility honestly ("differ in wrapper version *and* in date, host
  and conditions... names a plausible mechanism rather than a confirmed
  one") — this task's own direct check now shows the plausible mechanism
  specifically cannot be the confirmed one, for the evidence cited.

**Candidates B (extrapolation) and C (grid coverage) are both real,
directly demonstrated, and dominant** — §2/§3 below.

## 2. Part A — Profiled range vs. requested range (arithmetic on files already present)

Read directly from `data/profiling/compute/mi355x/deepseek-v3/attention.csv`,
`attn_tp=8` slice:

| axis | profiled range | what the "final working command" requests |
|---|---|---|
| decode `batch_size` | **1 – 2** | however many requests are concurrently decoding — unbounded by the profile, and easily >2 once 16 Poisson(qps=1) requests each run 128 decode steps overlap in flight |
| decode `kv_cache_size` (tokens) | **0 – 96** | **~512 – 639** (a 512-token un-chunked prefill, then 128 decode steps, each one token further into the cache) |
| prefill `total_tokens` | **32 – 96** | **512** (`--fixed_request_length_generator_config_prefill_tokens 512`, chunked prefill explicitly disabled) |

**Every shape this exact, real, documented command requests is outside
the profiled range**, on both axes, for both prefill and decode. This
alone, per this task's own §2, "may settle the question" — it does.

**Confirmed live, not inferred**, by querying the real trained model
directly at a sweep of shapes:

```
attn_mla_decode, batch_size=1, varying kv_cache_size:
  kv_cache_size=     0  predicted=0.112704 ms
  kv_cache_size=    96  predicted=0.116432 ms   <- last profiled point
  kv_cache_size=   128  predicted=0.116204 ms
  kv_cache_size=   550  predicted=0.116204 ms   <- realistic mid-decode value
  kv_cache_size=  5000  predicted=0.116204 ms   <- identical, out to 5000

attn_mla_decode, kv_cache_size=550, varying batch_size:
  batch_size=  1  predicted=0.116204 ms
  batch_size=  2  predicted=0.206323 ms   <- last profiled point
  batch_size=  5  predicted=0.206323 ms
  batch_size= 16  predicted=0.206323 ms   <- identical, at 8x the profiled batch

attn_mla_prefill, batch_size=1, varying total prefill tokens:
  total_tokens=    96  predicted=0.135517 ms   <- last profiled point
  total_tokens=   512  predicted=0.136045 ms   <- the actual requested shape
  total_tokens=  4096  predicted=0.136045 ms   <- identical, at 8x the request
```

**A textbook flat extrapolation, exactly as this task's own §1 describes
("returns the mean of the nearest training leaf... emits a flat value
rather than a trend"), for the exact scenario this project's own
engineering record cites as inaccurate.** A real 512-token prefill and a
real ~550-token-deep decode step are predicted at the *same* cost as the
last profiled point, regardless of how far beyond it the real shape
lies. Since real attention cost cannot be flat in either batch size or
sequence length, this mechanism can only ever **understate** cost for a
real request beyond the profiled edge — a concrete, checkable sign,
independent of the real-serving ground truth this checkout lacks (§0).

## 3. Part B — What the fit itself contributes

**A real, previously undocumented contamination, found by reading
`_train_mla_attention_layer_models` directly (`sklearn_execution_time_predictor.py`)
and confirmed against the actual training rows**: the per-operator
training loop only does `attention_df.dropna(subset=[target_col])` — it
never filters rows to the operator's own declared `phases`
(`AttentionOperatorSpec.phases`, e.g. `_DECODE_MIXED` for
`attn_mla_decode`). Since every profiled row records a value for *every*
MLA timing column regardless of whether that row was a prefill or decode
sample (`DEEPSEEK_V3_MLA_MI355X_JOURNEY.md`'s own step 3: "decode scopes
[measured] at timer-overhead noise floor during a prefill row"), **the
8 genuine decode rows the doc's own explanation cites are not what the
model actually trains on** — it trains on those 8 plus **5 prefill-phase
rows whose `attn_mla_decode.median` value is pure ~0.004–0.005ms
noise-floor**, next to real decode measurements of 0.11–0.21ms — a 20–40x
jump in the same target column, for the same operator, with `is_prefill`
as just one of 19 features rather than a pre-filter.

**Isolated the two contributions with a genuine leave-one-out test**
(the doc's own in-sample MAPE, reproduced in §1, scores the refit model
against its *own* training rows — not held-out; this repeats Frontier's
exact grid-search hyperparameters, LOO'd, both on the data as Frontier
actually trains it and on the correctly phase-filtered subset):

| operator | LOO MAPE, as Frontier trains (13 rows, mixed) | LOO MAPE, LinearRegression on the same 13 rows | LOO MAPE, decode-phase rows only (8 rows) |
|---|---|---|---|
| `attn_mla_decode` | **177.97%** | 23.05% | **2.43%** |
| `attn_mla_decode_q_latent_proj` | **63.78%** | 11.34% | **3.89%** |
| `attn_mla_v_up_proj` | **53.55%** | 2.53% | **2.81%** |

Two findings from this table, both real:

1. **On the properly-scoped 8-row decode-only subset, the forest
   generalizes excellently (2.4%–3.9% LOO MAPE)** — comparable to the
   *good* MAPE figures `MI355X_FOUR_MODEL_PROFILING.md` reports for other
   operators (3.8%–6.1% for `attn_decode`/`attn_prefill`). Eight points on
   a smooth two-feature (`batch_size`, `kv_cache_size`) relationship is,
   in fact, enough — contradicting a naive "too few rows, forest can't
   fit" reading of the doc's own explanation. **Grid coverage (candidate
   C), in the narrow sense of "not enough points," is not the actual
   problem here.**
2. **On the data Frontier actually trains on (contaminated with 5
   off-phase rows), the forest's held-out error is dramatically worse
   (54%–178%) than its in-sample error (20%–77%, §1)** — the gap between
   in-sample and held-out is itself the signature of a real
   generalization failure, not measurement noise. **`LinearRegression` on
   the identical contaminated data is markedly better** (2.5%–23%) —
   exactly the diagnostic this task's own Part B names ("if a polynomial
   fit is markedly better on held-out rows, the forest is the problem
   rather than the profiles"). So: on *this* data mixture, the forest is
   a real, independent contributor to the reported inaccuracy — but the
   forest is not inherently bad at this problem (finding 1); it is bad
   at absorbing a same-column phase mismatch a one-line `phases` filter
   would remove.

**The two model classes also differ qualitatively outside the profiled
range, confirming this task's own Part B framing directly**: refit
`LinearRegression` on the clean 8-row decode-only set and query it at the
same out-of-range shapes §2 used —

```
  kv_cache_size=    96  LinearRegression=0.117326 ms   (forest: 0.116432, ~same)
  kv_cache_size=   550  LinearRegression=0.151433 ms   (forest: 0.116204, flat)
  kv_cache_size=  5000  LinearRegression=0.485739 ms   (forest: 0.116204, flat)
  batch_size=16         LinearRegression=1.540733 ms   (forest: 0.206323, flat)
```

The forest returns a flat value; the linear model keeps extrapolating a
trend. Neither is verified accurate this far outside the data (no ground
truth exists here either), and a runaway linear extrapolation is not
obviously *better* — but this is exactly "a forest that returns a
neighbour's value where a trend is obvious... behaving as documented
rather than as wanted."

**Whether the predictor caches its fitted output — checked directly,
not stale.** `_get_model_hash` (`sklearn_execution_time_predictor.py`)
builds its cache key from `str(self.to_dict())` (the predictor's own
config) **plus an MD5 hash of the full training dataframe's JSON
serialization** (`hashlib.md5(df.to_json()...)`), not a path or mtime.
A content change to the CSV produces a different hash and therefore a
cache miss — the disk cache (`cache/`, `*.pkl`, confirmed present but
containing no `deepseek-v3` entries in this checkout — this was the
first real fit) cannot silently serve a stale model after the data
changes. This measured cleanly: nothing in §1–§3 is an artifact of a
stale cache.

## 4. Whether a backend measurement is warranted

**No.** Parts A and B decisively point at extrapolation (B) and a
training-data-contamination form of the fit question (C/B, refined) —
not at the backend. Per §1, candidate A does not currently apply at all
(no tuned backend runs anywhere in this fleet for MLA to diverge from),
so there is nothing for a device measurement to confirm or refute for
this symptom. Not attempted, per this task's own explicit instruction.

## 5. What each mechanism would cost to address

- **A — backend ceiling.** Not currently active (§1), so nothing to fix
  for this symptom specifically. Making it possible at all is a real ROCm
  build-compatibility effort already documented as blocked
  (`AITER_KERNELS.md`: prebuilt kernels don't load against host torch on
  either host that has them; rebuilding fails to compile) — infrastructure
  work, not a code change here.
- **B — extrapolation.** Two independent, non-exclusive costs: (i)
  widen the profiled grid to cover real request shapes (`batch_size`
  well past 2, `kv_cache_size` well past 96 for decode; `total_tokens`
  well past 96 for prefill) — real GPU-hours, using infrastructure this
  project already has (the same `attention.main` sweep tooling Task 49
  scoped for the `kernel_only` gap); (ii) a code-side guard that fails
  loudly (or interpolates a trend) instead of silently flat-lining past
  the training range — a change to `_get_on_demand_prediction`, outside
  this task's own "change nothing" acceptance bar.
- **C — grid coverage, refined.** The "too few rows" framing turns out
  not to be the real problem (§3, finding 1) — widening the grid (same
  cost as B-i) is not actually required to fix the forest's own fit
  quality, only to fix its *reach*. The genuinely cheap, code-only,
  zero-new-profiling fix is the contamination found in §3: filtering
  `_train_mla_attention_layer_models`'s per-operator training rows to
  the operator's own declared `phases` before dropping NaNs, rather than
  only dropping NaNs. §3's own numbers are the evidence for the payoff
  (54%–178% → 2.4%–3.9% LOO MAPE) — a small, targeted, one-`phases`-filter
  change, not attempted here per this task's own "diagnosis only, change
  nothing" instruction, and named because it is obvious rather than
  because it was tried.
- **D — block-table aliasing.** Real (confirmed present in every current
  attention profile in the repository, per the task's own updated §1),
  but not shown to be active for *this* symptom (§1) — it can only bite
  at `batch_size>1`, and `deepseek-v3`'s own profiled decode grid caps at
  2. Its real cost is elsewhere: `Phi-tiny-MoE-instruct`'s own `h800`
  attention profile sweeps `batch_size` up to 8, and
  `Llama-3.1-405B-Instruct-FP8`'s up to 128 (checked directly against
  their own CSVs) — both used in this project's own actually-reported
  studies (Task 33/36). Fixing it requires merging the fix (already
  written, on the branch Task 51 evaluated and did not merge for
  unrelated reasons) into every profiling wrapper path, then
  **re-profiling every existing attention CSV with `batch_size>1` rows**
  — a project-wide re-profiling cost, unrelated to `deepseek-v3`
  specifically, and the more consequential piece of this task's own
  fourth candidate for the studies this project actually cites.

## 6. Anywhere this specification is wrong

1. **The "never a live model query" claim in this task's own §1 is not
   accurate for the MLA family specifically** — read directly from
   `sklearn_execution_time_predictor.py`. Single-feature dense-attention
   operators (`attn_pre_proj` etc.) do use the dense-grid-dict-with-exact-lookup
   pattern the spec describes (`self._predictions[op][(effective_tokens,)]`,
   a `KeyError` on a true miss — this is exactly the mechanism behind
   Task 48's own `mlp_up_proj` crash, confirmed reproduced live in §0's
   own predictor construction, same warning). **Multi-feature MLA
   operators go through a different path** (`_get_mla_attention_operator_times`
   → `_get_on_demand_prediction`): an exact-match lookup built from the
   training rows themselves, falling back — on any miss — to a genuine,
   live `model.predict()` call, cached only after the fact. This is a
   real mechanism difference, not a nitpick: it is *why* §2's flat
   extrapolation happens via a live call rather than a `KeyError`, and
   it is worth the correction since this task's own diagnosis depends on
   knowing which path actually runs.
2. **The premise that "predictions on the target device are reported as
   inaccurate against real serving" has a documented instance in this
   checkout is not established** (§0). The only concrete, quantified
   figure available is an in-sample training-error MAPE, which is a
   different question from a real-vs-simulated comparison; this report
   treated it as the best available stand-in, precisely because no
   actual real-vs-serving result survives in this checkout, and said so
   rather than treating the MAPE citation as equivalent to the symptom
   description.
3. **The specific quantitative support this task's own added §1 offers
   for candidate D (Task 51's 10%–62% finding) does not support candidate
   D** — all three shapes it measured are `batch_size=1`, where the named
   aliasing mechanism cannot operate (§1). This is not a claim that
   candidate D is false in general (it plainly isn't — the bug is real,
   confirmed present, and would matter wherever profiled `batch_size>1`),
   only that its one piece of cited evidence measures something else.
4. **Otherwise this specification's own framing held up precisely**: the
   sign-based elimination it invites (§1) does eliminate candidate A,
   just not by the mechanism it suggests (a direct real-vs-serving sign
   comparison, unavailable here) — by A requiring an executing tuned
   backend that does not exist anywhere in this fleet. The Part B
   diagnostics (held-out vs. in-sample, forest vs. linear, cache
   staleness) each produced a real, decisive answer exactly where the
   spec predicted one might be found.

## What shipped

Nothing — an investigation and diagnosis, per this task's own acceptance
criteria. `docs/tasks/52-predictor-error-report.md`, this report, is the
only artifact. No source changed in Frontier or `dc-sim`.

One commit on `task-52-predictor-error`, stacked on `task-51-mla-merge`.
254 tests pass, unchanged; `check_import_direction.py` exits 0.
