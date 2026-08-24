# Task 39 — Close the two gaps in the memory formula

Branch: `task-39-formula-gaps`, branched from `task-38-formula-validation`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`.

204 tests pass (197 unchanged + 7 new), and
`python3 tools/check_import_direction.py` exits 0.

---

## Part A — Divisibility

### The checks added

`planner_core.divisibility_violations(model, attn_tp) -> List[str]`
checks exactly the three conditions `frontier/utils/param_counter.py`'s
own `ParamCounter.__init__` asserts before computing anything, matched
condition-for-condition rather than approximated:

1. `num_attention_heads % attn_tp == 0`
2. `hidden_size % attn_tp == 0`
3. `hidden_size % num_attention_heads == 0` — a model-level property,
  independent of `attn_tp` (Frontier's own third assertion doesn't
  mention the degree at all).

`attn_param_mem_bytes` now raises a new `InadmissibleDegree` (a
`ValueError` subclass, kept distinct from the plain `ValueError` a
malformed `ModelSpec` already raises, so `plan()` can catch this one
specifically) if any of the three fail, rather than computing
`num_attention_heads / attn_tp` as a float and silently returning a
fractional-heads-per-worker answer. This is the literal fix Part A
asked for — Frontier's own condition is an assertion, not an
approximation, and nothing here weakens it into "close enough."

### Where the rejection surfaces, and why

**In `plan()`'s own loop, as a new `Inadmissible` outcome — checked
before `feasible_num_blocks` is ever called, not caught as an
exception from it.** `PlanResult` gained a fourth list,
`inadmissible: List[Inadmissible]`, alongside `rejections` and
`unknown`. `plan()` calls `divisibility_violations` directly for each
`attn_tp` and records an `Inadmissible` entry (never a `Rejection` and
never an `Unknown`) for anything that fails, before generating any
shapes or asking the evaluator anything at all.

**This task's own hint pointed at `can_evaluate`; I put it in the core
instead, one level up from where `can_evaluate` sits, and the reasoning
is worth stating precisely.** `can_evaluate` answers a question that
is *evaluator-specific* — Task 37's own `SimulationEvaluator` says no
to a `tp` outside its own profiled range, and a different evaluator
(telemetry, a learned approximation) could legitimately have a
different profiled range or none at all. Divisibility is not
evaluator-specific in that sense: `attn_tp`-way tensor parallelism
either evenly splits this model's own head count or it does not, and
that fact is true for *any* evaluator, including one that has never
heard of Frontier. Placing the check in `plan()`'s own loop (calling a
core-level function, `divisibility_violations`) rather than inside
`SimulationEvaluator.can_evaluate` keeps it available to every future
evaluator automatically, the same reasoning Task 37 already applied to
memory feasibility ("an oracle backed by a running system would compute
the identical number from the same... inputs"). `attn_param_mem_bytes`
itself still raises independently, for any caller that reaches it
without going through `plan()`'s own pre-check — confirmed by a test
that calls it directly (§ below).

A dedicated test (`test_plan_reports_inadmissible_separately_from_rejected_and_unknown`)
confirms all three properties this task's own known trap requires:
the inadmissible candidate appears in `result.inadmissible`, appears in
neither `result.rejections` nor `result.unknown`, and the evaluator's
own `can_evaluate` is never even called for it — recorded by a
evaluator subclass that logs every `attn_tp` it was asked about.

---

## Part B — Whether the two KV-head counts can diverge

### The enumeration

`ModelConfig.get_runtime_num_kv_heads()` delegates to
`AttentionFamilySpec.resolve_runtime_num_kv_heads()`, which calls
whichever `runtime_num_kv_heads_resolver` the model's own bound
attention family declares
(`frontier/attention/model_binding.py::bind_attention_family`). Every
resolver in `frontier/attention/families.py`, read directly rather than
tested on one model and generalised:

| family | binds when | resolver | vs. raw `num_kv_heads` |
|---|---|---|---|
| **DSA** (`dsa_attention`) | a DSA marker is present | frozen — `require_enabled_for_execution()` raises `NotImplementedError` before any resolver runs | moot: this family cannot execute at all, so `feasible_num_blocks` filtering it is unreachable either way |
| **LATENT_MLA** (`latent_mla_attention`) | `use_mla=True`, itself either declared directly or *inferred* (`model_type in {deepseek_v2, deepseek_v3, deepseek_mtp, kimi_k2}` and `kv_lora_rank` is present) | `_latent_mla_runtime_num_kv_heads` returns **`1`, unconditionally**, regardless of the declared field | **diverges whenever the declared `num_kv_heads != 1`** — and the paired head-size resolver diverges too: `_latent_mla_runtime_head_size` returns `kv_lora_rank + qk_rope_head_dim`, not `get_head_dim()` |
| **dense_attention** (via `use_mfa=True`, Step3Text's own MFA topology) | `use_mfa=True` | `_dense_runtime_num_kv_heads` returns the raw field — and `bind_attention_family` itself *asserts* `num_kv_heads == 1` before allowing this binding at all | never diverges, by construction, not by luck — the binding function would raise `ValueError` first if it could |
| **dense_attention** (the ordinary path, no exotic marker) | everything else | `_dense_runtime_num_kv_heads` returns the raw field directly | never diverges — this *is* the raw field |

**Direction and magnitude, for the one family that diverges.** Since
`ceil(1 / attn_tp) = 1` for every `attn_tp >= 1`, while the raw-field
computation `ceil(declared_num_kv_heads / attn_tp)` keeps shrinking
with degree, the error is **largest at `attn_tp=1`** (where the raw
path hasn't started shrinking yet) and narrows — but never
closes — as degree increases. Reading the raw field for an MLA model
makes `_kv_cache_page_bytes_per_layer` **too large**, which makes
`num_blocks` **too small** — an overly pessimistic capacity estimate,
not an unsafe one, but a real, silently wrong one, exactly the
plausible-wrong-answer class Task 38 is about.

### Does any checkout model meet each condition?

- **DSA**: not checked exhaustively (moot per the table above — frozen
  families cannot reach `feasible_num_blocks` in any way that matters,
  since nothing can execute them regardless of what memory arithmetic
  says).
- **`use_mfa` (dense, guaranteed no divergence)**: yes —
  `step-moe-noquant-small`, already tested in Task 38. Re-confirms
  (does not newly discover) that its own agreement was structural, not
  coincidental: `bind_attention_family` would have refused the binding
  outright if its declared `num_kv_heads` were anything but 1.
- **LATENT_MLA (the real divergence)**: yes —
  `deepseek-v3` and `deepseek-r1-0528`, both present in
  `data/config/models/`, both with `model_type="deepseek_v3"` and a
  declared `kv_lora_rank`, so `use_mla` is inferred `True` for both.
  **Neither is profiled on h800 or rtx_pro_6000** — both are profiled
  only on `mi355x` (`data/profiling/compute/mi355x/deepseek-v3`,
  `.../deepseek-r1-0528`), the one device this project's own real-
  compute tools have never targeted (Task 35's own "only h800 and
  rtx_pro_6000 carry full-feature profiles" finding). Checked directly,
  not assumed: a real `SimulationConfig` built for `deepseek-v3`
  confirms `use_mla=True`, raw `num_kv_heads=128` against
  `get_runtime_num_kv_heads()=1`, and `get_head_dim()=56` against
  `get_runtime_head_size()=576` (`kv_lora_rank=512 + qk_rope_head_dim=64`) —
  a live, executed check, not an inference from reading the resolver's
  own source alone.
- **The ordinary dense path**: yes — Phi-tiny-MoE-instruct and
  Llama-3.1-405B-Instruct-FP8 (Tasks 36/38), no divergence, as
  established.

**So: the divergence is real, and it is unexercised by anything this
project's own tools actually run, not merely untested by chance.**
No h800/rtx_pro_6000-profiled model meets the LATENT_MLA condition; the
one pair that does is reachable only through a device this project has
never pointed a real-compute tool at.

### The fix

`ModelSpec` gained two new fields, `runtime_num_kv_heads: Optional[int]`
and `runtime_head_dim: Optional[int]`, both defaulting to `None`.
`_kv_cache_page_bytes_per_layer` now reads these (falling back to the
raw fields when `None`) instead of the raw fields directly.
`attn_param_mem_bytes` is unchanged — it must keep reading the raw
fields, since that is what `ParamCounter` itself reads for parameter
memory, confirmed in Task 38.

This is a **default that is correct for dense_attention (every model
this formula has been validated against), not an auto-detection of
every attention family Frontier supports.** Per this task's own "do not
approximate an assertion" trap, nothing here guesses whether a model is
MLA from its declared fields — the docstring on both new `ModelSpec`
fields states plainly that a LATENT_MLA (or any future non-dense)
model's own caller must supply both explicitly, mirroring the
already-established `head_dim` override pattern (Task 36/38) rather
than inventing a new idiom. A future caller who forgets is in exactly
the position Task 36's own `head_dim` bug was in — not solved by this
task, deliberately, because solving it would mean building family
detection this checkout's own usable models never exercise.

**Two tests pin this**, per this task's own explicit acceptance
requirement:

- `test_kv_cache_page_bytes_defaults_match_the_raw_fields_for_dense_models` —
  the agreement pin: leaving the two new fields at `None` gives exactly
  the same page size as setting them explicitly to the raw fields, at
  every tested degree, for a dense model.
- `test_kv_cache_page_bytes_uses_the_override_when_the_raw_fields_would_be_wrong` —
  built from deepseek-v3's own real, confirmed numbers: the raw-field
  computation and the override-corrected one differ by exactly the
  ratio derived above (`128*56 / (1*576) ≈ 12.44x` at `attn_tp=1`), and
  the test asserts that ratio directly rather than merely asserting
  "they differ."

---

## Whether anything moved

**No.** Per this task's own acceptance requirement — "no model
currently tested should move; if one does, stop and report before
proceeding" — both regression captures were re-run after these changes
and diffed against the versions captured for Task 38, not assumed
identical:

```
$ diff task38_task33_check.log task39_task33_check.log
IDENTICAL
$ diff task38_task36_check.log task39_task36_check.log   # (elapsed_s excluded, as in tasks 37/38)
IDENTICAL
```

This is exactly what both fixes were designed to guarantee for every
currently-tested model: Part A's divisibility check never fires for any
`attn_tp` this project has ever swept (all powers of two, dividing
every tested model's own head count evenly), and Part B's KV-cache
override defaults to the raw fields, which is already correct for
every tested model's own dense_attention family. Nothing moved because
nothing *should* have — both gaps were real but latent for the specific
three models this project has actually run.

---

## Anywhere this specification is wrong

Nothing required correction. One precision worth recording:
`bind_attention_family`'s own `use_mfa` branch does not merely happen
to agree with the raw field for `step-moe-noquant-small` — it *asserts*
`num_kv_heads == 1` as a precondition of accepting that binding at all,
raising `ValueError` first if the model's own declared field were
anything else. Task 38's own report described this model's own
agreement as verified "by construction, not by luck," which undersold
it slightly: it is enforced by an assertion in the same function that
resolves the runtime value, not merely a structural fact about the MFA
family's own resolver formula. Worth stating precisely rather than
leaving as "no divergence found."

## What shipped

- `tools/planner_core.py` — `InadmissibleDegree`,
  `divisibility_violations`, `_runtime_kv_heads`, `_runtime_head_dim`;
  `attn_param_mem_bytes` now raises `InadmissibleDegree` for a non-
  dividing `attn_tp`; `_kv_cache_page_bytes_per_layer` now reads the
  new `runtime_num_kv_heads`/`runtime_head_dim` `ModelSpec` fields
  (falling back to the raw ones); `plan()`'s own loop pre-checks
  divisibility and records a new `Inadmissible` outcome, and
  `PlanResult` gained `inadmissible: List[Inadmissible]`.
- `tests/test_planner_core.py` — seven new tests: three for
  `divisibility_violations` directly, one confirming
  `attn_param_mem_bytes` raises, one confirming `plan()`'s own
  three-way outcome separation and that the evaluator is never asked
  about an inadmissible candidate, and two pinning the KV-cache-
  override agreement/correction from Part B.
- `docs/tasks/39-formula-gaps-report.md`, this report.

One commit on `task-39-formula-gaps`, branched from
`task-38-formula-validation`'s tip.
