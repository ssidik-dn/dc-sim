# Task 38 — Validate the memory formula on a third model

Branch: `task-38-formula-validation`, branched from `task-37-evaluator`'s
tip (run after both Task 34 and Task 37 reported, per this task's own
instruction). Paths per Task 25: working tree at
`/work/simulation/dc-sim`, Frontier at `/work/simulation/Frontier`.

197 tests pass (194 unchanged + 3 new), and
`python3 tools/check_import_direction.py` exits 0. No change to
`tools/planner.py` or `tools/planner_core.py` — this task adds tests
and a report only, so Task 33's and Task 36's own results were expected
to reproduce trivially, and were still re-checked directly rather than
assumed (§4).

---

## 1. Which model, and why

**`step-moe-noquant-small`**, h800-profiled, from Task 35's own
inventory. Checked against both alternatives Task 35 catalogued before
choosing:

| model | MoE? | GQA ratio (q:kv) | declared `head_dim` | vs. naive `hidden_size // num_attention_heads` |
|---|---|---|---|---|
| Phi-tiny-MoE-instruct (already tested) | yes | 16:4 = 4 | 128 | 256 — override *smaller* |
| Llama-3.1-405B-Instruct-FP8 (already tested) | no | 128:8 = 16 | none | 128 — no override, correct by construction |
| **step-moe-noquant-small (this task)** | yes | **64:1 = 64** | **256** | **112 — override *larger*** |

Two reasons this is the most differentiated choice available, not just
"a third one":

**The most extreme grouped-query ratio in the checkout.**
`num_key_value_heads=1` means `ceil(num_key_value_heads / attn_tp) = 1`
for *every* `attn_tp` this project has ever swept — the KV term in
`attn_param_mem_bytes` never shrinks with degree at all, a regime
neither Phi-tiny-MoE-instruct (kv=4) nor Llama-3.1-405B-Instruct-FP8
(kv=8) exercises even once.

**A `head_dim` override in the opposite direction.** Phi-tiny-MoE's own
override makes the declared value *smaller* than the naive default
(128 < 256); step-moe-noquant-small's makes it *larger* (256 > 112).
Per this task's own known trap ("a default that happens to be right is
still untested"), a model whose override happened to move the same
direction as the only other tested override would leave open the
possibility that some sign-dependent mistake in `_attn_head_dim` had
simply never been exercised. This closes that gap specifically, not by
accident.

(`Step2Mini-tiny` was ruled out for the reason this trap names
directly: its own declared `head_dim=128` *equals*
`2048 // 16 = 128` — testing it would confirm nothing about the
override code path at all.)

---

## 2. Whether the two computations agree

**Yes, bit-for-bit, at every tested degree.** Computed via
`planner_core.attn_param_mem_bytes` and, separately, via a real
`ParamCounter(replica_config, ClusterType.DECODE_ATTN).get_num_parameters_per_device()`
built from an actual `SimulationConfig.create_from_cli_args()` (not
assumed, not reused from Task 36's own already-passing table):

| `attn_tp` | formula (bytes) | `ParamCounter` (bytes) | match |
|---|---|---|---|
| 1 | 14,790,164,480 | 14,790,164,480 | yes |
| 2 | 7,508,852,736 | 7,508,852,736 | yes |
| 4 | 3,868,196,864 | 3,868,196,864 | yes |
| 8 | 2,047,868,928 | 2,047,868,928 | yes |

All four also match Task 35's own attention-weight figure for this
model exactly (13.7744 GB at tp=1, `14790164480 / 1024**3`) — three
independent computations (Task 35's own script, this task's formula,
and a live `ParamCounter`) now agree, not two.

**The `plan()` sanity check surfaced a real, separate finding —
reported plainly rather than swapped away.** Running `plan()` against
this model (margin 0.85, Task 32's own small fabric) gave exactly the
feasibility split the formula alone already predicted —
`attn_tp=1` rejected on memory, `attn_tp∈{2,4,8}` passed the
feasibility gate — but every one of the eleven shapes that passed then
failed *evaluation* with
`KeyError: "['time_stats.mlp_up_proj.median'] not in index"`.

Traced, not left as a mystery: `step-moe-noquant-small`'s own
`moe_layers_enum` starts at layer 4 (`"4,5,6,...,59"`, clipped to its
own 31 layers) — layers 0-3 are architecturally **dense**, not MoE, and
Frontier's own MoE execution-time predictor
(`sklearn_moe_execution_time_predictor.py`'s own dense-FFN branch,
"Dense FFN branch for mixed-layer MoE models") tries to price those
four layers' own `mlp_up_proj`/`mlp_down_proj`/`mlp_act` operators —
which this model's own shipped `linear_op.csv` never profiled (it has
`share_expert_up_proj`/`share_expert_down_proj` columns for its shared
expert, and MoE-specific columns for its routed experts, but no
generic dense-MLP operator at all). This is a **gap in this model's own
shipped profiling data**, not in `feasible_num_blocks`: the memory-
feasibility filter did exactly what it should, correctly separating
the one infeasible degree from three feasible ones, before an
unrelated execution-time-modeling limitation (out of this task's own
"Small" scope — it would mean either re-profiling the model or patching
the MoE execution-time predictor, neither of which is what this task
is validating) stopped any of them from producing a full result.

**This is not a disqualifying result for this task's own question.**
§2's own required comparison (formula vs. `ParamCounter`) does not
touch the execution-time predictor at all, and it passed exactly as
cleanly as it did for the first two models. What failed is a strictly
later stage, checked mechanically and understood precisely rather than
patched over.

---

## 3. The field inventory

Every field `frontier/utils/param_counter.py` consults, from
`ParamCounter.__init__` and `get_num_attention_params_per_layer`
(what actually feeds DECODE_ATTN's own memory) plus everything else the
class touches (which does not).

**Confirmed** — read by `attn_param_mem_bytes`/`feasible_num_blocks`,
and now exercised across genuinely different values by all three
models:

| field | how it's exercised |
|---|---|
| `embedding_dim` (`hidden_size`) | 4096 / 16384 / 7168 — three different values |
| `num_q_heads` (`num_attention_heads`) | 16 / 128 / 64 |
| `num_kv_heads` (`num_key_value_heads`) — the **raw** field, not `get_runtime_num_kv_heads()` | 4 / 8 / **1** (the untested-until-now extreme) |
| `get_head_dim()` (explicit-or-derived `head_dim`) | override-smaller (Phi) / no-override (Llama) / **override-larger** (step-moe) |
| `num_layers` | 32 / 126 / 31 |
| `attn_tensor_parallel_size` | 1 through 32 across the three models combined |

**Assumed a default, not exercised by any of the three models** — this
is the honest boundary of what this task's own comparison proves:

| field | what `feasible_num_blocks` assumes | risk if wrong |
|---|---|---|
| `replica_config.num_pipeline_stages` | always 1 (`ParamCounter` would divide `num_layers` by it otherwise) | every real-compute tool in this project has only ever run DECODE_ATTN at pp=1; untested at pp>1 for any model |
| `replica_config.speculative_decoding_config` (MTP) | absent/disabled | `get_num_mtp_parameters_per_device()` would add real parameters for a model using speculative decoding; none of the three tested models declares one, so this path has never actually executed with a nonzero result |
| `num_q_heads % attn_tp == 0` / `embedding_dim % attn_tp == 0` / `embedding_dim % num_q_heads == 0` — Frontier's own `ParamCounter.__init__` asserts all three | `feasible_num_blocks` performs no equivalent check and uses float division; a caller passing a non-dividing `attn_tp` would get a silently-wrong fractional result instead of Frontier's own hard assertion error. Not hit by any tested model, since every `admissible_tp` used so far divides every tested model's own head count evenly |
| `model_config.get_runtime_num_kv_heads()` (the attention-family-resolved KV head count `Replica`/`MemoryPlanner`'s own **KV-cache** sizing path reads) vs. the raw `num_kv_heads` field `ParamCounter` reads for **parameter** memory | `_kv_cache_page_bytes_per_layer` (feeding `num_blocks`, not the parameter-memory comparison in §2) reads the raw field, assuming it equals the runtime-resolved one. Checked directly for step-moe-noquant-small specifically (its own "MFA" attention tag was reason enough to check rather than assume): `get_runtime_num_kv_heads()==1==` the raw field, so no divergence here — but this is a coincidence for *this* model, not a proof the two paths always agree. An MLA-style model would be the case to check next |

**Not consulted by `param_counter.py` for DECODE_ATTN at all** — not a
gap, because Frontier's own code never routes these to attention
memory either (`get_num_mlp_params_per_layer` returns `0`
unconditionally for `ClusterType.DECODE_ATTN`); listed for completeness
since they are real fields the module reads, just for DECODE_FFN's own,
separate memory budget:

`mlp_hidden_dim`/`intermediate_size`, `moe_intermediate_size`,
`use_gated_mlp`, `is_moe`, `num_experts`, `moe_expert_parallel_size`,
`moe_tensor_parallel_size`, `share_expert_dim`,
`counts_share_expert_param_memory` (an architecture-profile flag, not a
per-model JSON field).

**Declared by some models, never consulted by `param_counter.py` at
all** — `attention_bias` (Llama-3.1-405B-Instruct-FP8's own JSON sets it
`false` explicitly). No branch in `get_num_attention_params_per_layer`
reads it, for any cluster type — Frontier's own parameter count never
includes an attention bias term regardless of what a model declares.
Matching `ParamCounter` exactly, as `attn_param_mem_bytes` does,
therefore inherits this — correctly, since the goal is agreement with
Frontier's own accounting, not an independent "true" parameter count.

---

## 4. Whether Tasks 33 and 36 still reproduce

**Yes, both bit-identical — captured fresh, not inferred from "nothing
changed."** No line of `tools/planner.py` or `tools/planner_core.py`
was touched in this task, but both were re-run anyway rather than
assumed:

```
$ diff task37_after_task33.log task38_task33_check.log
IDENTICAL
$ diff <(grep -v elapsed task37_after_task36.log) <(grep -v elapsed task38_task36_check.log)
IDENTICAL
```

(`elapsed_s` excluded from the Task 36 diff for the same reason Task 37
excluded it — wall-clock time across two separate runs is never
expected to match, and was not treated as a sign of anything.)

---

## 5. Anywhere this specification is wrong

Nothing required correction. One clarification worth recording: this
task's own §2 asks to "run one `plan()` against it and confirm the
ranking is sane," which implicitly assumes a chosen model will produce
*a* ranking to inspect. `step-moe-noquant-small` did not, for the
reason in §2 — a finding this report treats as informative rather than
as evidence the model was a poor choice, since the actual formula
comparison (§2's own first three steps, and this task's real subject)
succeeded completely. A future task choosing a model for an end-to-end
`plan()` demonstration (rather than a formula-only comparison) should
screen for exactly this — a "mixed-layer MoE model" whose profiling
data omits the dense-layer operators its own architecture needs — before
committing to it.

## What shipped

- `tests/test_feasible_num_blocks_vs_param_counter.py` (new) —
  parametrized over all three models, each at several `attn_tp`
  degrees (14 comparisons total), asserting exact byte-for-byte
  agreement between `attn_param_mem_bytes` and a real
  `ParamCounter.get_num_parameters_per_device()`.
- `tests/_param_counter_probe.py` (new, not collected by pytest) — the
  real-Frontier half of that test, run as a subprocess with `cwd` set
  to Frontier's own root (needed because
  `BaseModelConfig.create_from_name` resolves model JSON via a
  cwd-relative path; an in-process `os.chdir()` would have leaked into
  every other test file sharing the same pytest session).
- `docs/tasks/38-formula-validation-report.md`, this report.

One commit on `task-38-formula-validation`, branched from
`task-37-evaluator`'s tip.
