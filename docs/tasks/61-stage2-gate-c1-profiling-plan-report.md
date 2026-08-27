# Stage 2 — Gate C.1: Qwen3-0.6B → MI355X Frontier profiling plan (PRE-RUN)

**No GPU touched. This is a plan only.** Everything below was derived
from reading Frontier's own profiling/predictor code, the real,
existing `mi355x` profile CSVs, Qwen3-0.6B's own real config, and this
project's own prior Tasks 48–53 directly — nothing was assumed or
invented. Where a number could not be derived with certainty, that
uncertainty is stated explicitly rather than hidden. This revision
also folds in two additive asks made after the base plan was scoped:
**Item A** — whether this profiler runs on ROCm/MI355X at all has
never been confirmed by real execution; §16 Stage -1 proposes a
minimal two-point smoke test as the mandatory first real step,
pending separate explicit approval, and §17 risk 0 / the Final
Answers below reflect that this remains unconfirmed. **Item B** —
whether `kernel_only` measurement data is needed, addressed at §11a: it
turns out to be required for Gate C's own first candidates under this
project's standing `pd-af-disaggregation` convention, not optional
future work, which corrects that ask's own framing.

---

## §1. Why Gate C is blocked (recap)

Gate C (`docs/stage-2-gate-c-planner-handoff-report.md`) stopped because
no Frontier compute profile exists for `Qwen/Qwen3-0.6B` on any device.
This task's own purpose is to determine the *minimum sufficient*
profiling work to close that gap honestly — not to close it.

---

## §2. Qwen3-0.6B architecture, verified against the actual model config

Confirmed from Qwen3-0.6B's own real `config.json` (live fetch,
cross-checked against `sim-real/CLAUDE.md`'s own independently-recorded
real-download facts — both agree exactly):

| field | value | source |
|---|---|---|
| `model_type` | `"qwen3"` | real config |
| `hidden_size` | `1024` | real config |
| `num_attention_heads` | `16` | real config, matches `sim-real/CLAUDE.md` |
| `num_key_value_heads` | `8` | real config, matches `sim-real/CLAUDE.md` |
| `num_hidden_layers` | `28` | real config, matches `sim-real/CLAUDE.md` |
| `head_dim` | `128` | real config (declared explicitly, not derived) |
| `is_moe` | `False` (dense) | `qwen3` has no `moe_layers_enum`/expert fields in its config; standard dense FFN |
| `tie_word_embeddings` | `True` | real config |

Dense, GQA (16 query heads, 8 KV heads — a 2:1 group size, not MHA, not
MQA), 28 layers. `model_type="qwen3"` is **not** in Frontier's own
LATENT_MLA dispatch set (`frontier/attention/families.py`, re-checked
directly this session: only `deepseek_v2`/`deepseek_v3`/`deepseek_mtp`/
`kimi_k2` trigger it) — this model binds **DENSE_KV**, the ordinary,
already-well-supported family every other real profile in this project
uses. Divisibility for `attn_tp ∈ {1,2,4}` against `num_attention_heads=16`/
`hidden_size=1024` was already confirmed clean in Gate C's own report.

---

## §3. Required Frontier operator families — the real mapping, not a generic list

Verified directly against the real column headers of an already-profiled
DENSE_KV model on `mi355x` (`meta-llama/Llama-2-7b-hf` — the only *dense*
model profiled on this device today; `qwen3-a3b-30b-moe`'s own header
was cross-checked too, confirming the attention-side operator set is
identical between the two — only the FFN side differs by MoE-ness). No
operator below was assumed; every name is a real CSV column this
project's own Frontier checkout already produces for a DENSE_KV model.

| operator family | used by Qwen3-0.6B? | profile source today | present on MI355X? | required for Gate C? |
|---|---|---|---|---|
| `attn_prefill` | yes | `attention.csv` (`frontier.profiling.attention.main`) | for other models only | **yes** |
| `attn_decode` | yes | `attention.csv` | for other models only | **yes** |
| `attn_kv_cache_save` | yes | `attention.csv` | for other models only | **yes** |
| `attn_input_reshape` | yes | `attention.csv` | for other models only | **yes** |
| `attn_output_reshape` | yes | `attention.csv` | for other models only | **yes** |
| `attn_pre_proj` (Q/K/V projection, Frontier's own name) | yes | `linear_op.csv` (`frontier.profiling.linear_op.main`) | for other models only | **yes** |
| `attn_post_proj` (output projection) | yes | `linear_op.csv` | for other models only | **yes** |
| `attn_rope` | yes | `linear_op.csv` | for other models only | **yes** |
| `input_layernorm` / `post_attention_layernorm` | yes | `linear_op.csv` | for other models only | **yes** |
| `emb` (embedding) | yes | `linear_op.csv` | for other models only | **yes** |
| `mlp_up_proj` / `mlp_down_proj` / `mlp_act` (dense MLP) | yes — dense model, no `--is_moe` | `linear_op.csv`, generic dense path | for other models only (`Llama-2-7b-hf` has them; MoE models on mi355x correctly don't, per Task 48's own finding) | **yes** |
| `lm_head_linear` | possibly | `linear_op.csv`, gated by `--include_target_embedded_mtp` per Task 49's own report | not confirmed present for any non-MTP mi355x model | **check at profiling time** — see below |
| MoE gating/routing/grouped-GEMM (`moe_gating_linear`, `moe_grouped_gemm`, etc.) | **no** | `moe.csv` | n/a | **not required** — `is_moe=False` |
| Communication-related compute seams (all-reduce/send-recv) | yes, but separate | `data/profiling/network/mi355x_8gpu/*.csv` (already exists, device-level, model-independent) | **yes, already present** | **not required to re-profile** |
| AITER-backed operator variants | no | n/a | AITER is non-functional on this checkout regardless (Task 49's own `AITER_KERNELS.md` finding: prebuilt kernels don't load against host torch on either host that has them) | **not applicable** |

**`lm_head_linear`, flagged honestly rather than assumed.** Task 49's
own report names it only as part of MTP fusion
(`--include_target_embedded_mtp`, "Also profile target-embedded MTP
compute families (`mtp_fusion_proj`, `lm_head_linear`)") — a
DeepSeek-MTP-specific flag. Qwen3-0.6B has no MTP head, and
`tie_word_embeddings=True` means the LM head shares the embedding
matrix rather than being a separate learned projection some models
declare. Whether Frontier's own execution-time predictor calls a
distinct `lm_head_linear` execution-time function for a tied-embedding
dense model, or folds it into `emb`, was **not independently confirmed
in this task** (it did not block anything §1–§16 below need, since
neither existing real profile — `Llama-2-7b-hf` nor `qwen3-a3b-30b-moe` —
lists it as a column, and both complete real runs). Flagged for the
structural validation stage (§13.A) to check for real, once the profiler
is actually invoked, rather than assumed either way here.

**No operator was invented.** MoE families, AITER-specific wrappers, and
MLA-specific columns (`attn_mla_*`) are all correctly excluded — none of
them apply to a dense, GQA, non-MLA model.

---

## §4. Model family / backend path

- **Standard attention, not MLA** — DENSE_KV, confirmed §2.
- **Dense, not MoE.**
- **GQA**, group size 2 (16 query heads / 8 KV heads) — a real,
  ordinary shape; `n_q_head`/`n_kv_head` are recorded as real training
  features on every existing attention profile (confirmed directly on
  `qwen3-a3b-30b-moe`'s own CSV header), so this ratio is exactly what
  the training data would need to represent, not an approximation.
- **Attention backend**: `TORCH_SDPA` — the same backend every one of
  this project's own real `mi355x` profiling commands has used
  (Task 49's own three commands, `--attention_backend TORCH_SDPA`);
  `AITER` is listed in `frontier.profiling.attention.main`'s own
  `--attention_backend` choices but is non-functional on this checkout
  (§3 above).
- **MLP profiling path**: the generic dense linear-op path
  (`frontier.profiling.linear_op.main`, **without** `--is_moe`) — the
  same tool and flag Task 49's own §3 used for `deepseek-v3`'s dense
  layers.
- **Kernel backend on MI355X**: `TORCH_SDPA` is the *only* attention
  backend with any real precedent on this device in this checkout
  (every existing `mi355x` profile used it; `AITER` has never
  successfully profiled anything here, per Task 49).

**Fidelity limitation, stated plainly per this task's own §2
instruction, not silently accepted**: Gate B's own real hardware
serving used real vLLM on ROCm — its own actual attention kernel
backend at serving time was never independently confirmed in this
project's own record (neither Gate B's nor this task's own reading
established which ROCm attention kernel vLLM itself selected for
Qwen3-0.6B on MI355X — vLLM has its own kernel-selection logic,
separate from Frontier's profiler entirely). `TORCH_SDPA` is Frontier's
own **portable reference implementation**, explicitly not
production-tuned (Task 49's own framing: "AITER (real production ROCm
kernels) remains non-functional... TORCH_SDPA (portable, not
peak-tuned) stays the ceiling on fidelity for this device family"). **A
profile collected with `TORCH_SDPA` should not be treated as
production-kernel-equivalent to whatever vLLM's own serving path
actually ran** — this is a real fidelity ceiling on this whole plan,
not specific to Qwen3-0.6B, and not resolvable without the same
ROCm/AITER build-compatibility fix Task 49 already found blocked.

---

## §5. Existing profile reuse — checked, not assumed from the name

Every `mi355x` attention/linear-op profile's own real shape columns
(`n_embd`, `n_q_head`, `n_kv_head`) were compared directly against
Qwen3-0.6B's real values (`1024`/`16`/`8`):

| candidate | real shape (`n_embd`/`n_q_head`/`n_kv_head`) | classification |
|---|---|---|
| exact `Qwen3-0.6B` | — (does not exist) | **NOT_REUSABLE** — no rows exist |
| `qwen3-a3b-30b-moe` (same "Qwen3" family name, largest architectural cousin on this device) | `2048`/`32`/`4` | **NOT_REUSABLE** — every attention/linear-op operator this table lists is trained on rows filtered by the model's own declared architecture (the same exact-match-by-config-parameter pattern Gate A.1 already confirmed for MoE dataset filtering — `num_experts`/`router_topk`/`hidden_dim`/`expert_hidden_dim` — applies by direct extension to dense/attention profiling's own model-config filter, though the exact filter predicate for the attention-training code path specifically was not independently re-read in this task; stated here as a high-confidence inference from an established pattern, not a re-verified fact). `head_dim=128` happens to match Qwen3-0.6B's own `head_dim=128` coincidentally (both declare it explicitly), but `n_embd`/`n_q_head`/`n_kv_head` do not, and per-head attention cost as actually profiled is a function of the full shape tuple, not `head_dim` alone. |
| `Llama-2-7b-hf` (the only *dense* mi355x model) | `4096`/`32`/`32` (pure MHA, no GQA) | **NOT_REUSABLE** — 4× hidden size, no head-sharing at all, architecturally unrelated |
| any `h800`/`rtx_pro_6000`/`a100`/etc. profile of a small dense model | varies | **NOT_REUSABLE regardless of shape match** — a compute-time profile is device-specific by construction (different ROCm/CUDA kernel implementations, different memory bandwidth/compute ratios); a matching shape on a different device says nothing about MI355X's own timing |

**No row anywhere in this checkout is `EXACT_REUSE` or
`SHAPE_EQUIVALENT_REUSE` for Qwen3-0.6B on `mi355x`.** Reuse saves
**zero** new-profiling rows. This is consistent with, and reconfirms,
Gate C's own original finding — not a new blocker, the same one measured
more precisely.

---

## §6. Gate C runtime shape envelope — derived, not assumed

**The one real, load-bearing fact this whole envelope hinges on**:
Gate B's own real request records (`sim-real/artifacts/noise/*/attempts.jsonl`,
already collected, re-read directly) show `prompt_tokens: 5` on every
single real request across all four Gate B configurations. **Gate C's
real prefill shape is 5 tokens** — far below every existing profiled
grid's own minimum starting point (`qwen3-a3b-30b-moe`'s own prefill
`total_tokens` grid starts at `32`; `deepseek-v3`'s own started at `32`
too, per Task 52's own table). This is the *opposite* direction from
Task 52's own lesson (a profile ending too low relative to a *large*
real request) — here the risk is a profile grid that **starts too
high** relative to an unusually *small* real request. Both are the same
underlying mistake (assuming the real workload sits inside whatever
grid was convenient to build) and both must be checked explicitly,
which is exactly what this section does.

| axis | real Gate C value/behavior | how derived | proposed grid coverage |
|---|---|---|---|
| prefill `total_tokens` | **5** (one real value, from every Gate B request record) | direct read of real `attempts.jsonl` | `{1, 2, 4, 5, 8, 16, 32}` — brackets 5 with real points on both sides, not at an edge |
| decode `kv_cache_size` | starts at 5 (prefill length), grows to `5 + (max_tokens-1) = 5 + 31 = 36` at the last of 32 generated tokens | `max_tokens=32` (Gate C's own frozen workload parameter) means up to 31 decode steps; KV cache depth at decode step *i* is `prefill_tokens + i` | `{0, 8, 16, 24, 32, 48, 64, 96, 128}` — real need is 5–36; `128` gives >3.5× margin past the real requirement, landing every real Gate C decode step inside the grid, never at its last point |
| decode `batch_size` (concurrent in-flight decode requests) | **not independently confirmed** — see below | Frontier's own scheduler, not real vLLM's, determines this; not derivable from Gate B's real serving concurrency alone | `{1, 2, 4, 6, 8, 12, 16}` — matches the range every other model in this project's own real `mi355x` profile already covers (`qwen3-a3b-30b-moe`: `batch_size ∈ {1..16}`), reusing established precedent rather than inventing a new number |
| `attn_tp` (TP degree) | `{1, 2}` for single-host Gate C candidates; `{2, 4}` for cross-host | Gate C's own four candidates, read directly from its own report | `{1, 2, 4}` — the union, exactly what Gate C needs, no more (existing models profile `{1,2,4,8}`; `8` is not needed here) |
| `max_seq_len` / `max_model_len` (profiler-side ceiling, not a real request property) | real requests never exceed `5 + 32 = 37` tokens total | derived from the two rows above | set the profiler's own `--max_seq_len`/`--max_model_len` to a round, comfortably-larger value (e.g. `256`) — cheap to set generously since it only bounds what the profiler is *willing* to sweep, not a per-point cost driver by itself |

**On `batch_size`, the honest gap named rather than hidden**: this
task's own §4 instruction is explicit — "do NOT equate request count
with model batch size... use the actual simulator/planner execution
semantics." Frontier's own `vllm_v1` scheduler (the type every real
evaluation in this project uses,
`--cluster_config_decode_attn_replica_scheduler_config_type vllm_v1`)
determines real concurrent decode batch size from its own internal
step loop, `max_tokens_in_batch` (`4096` by convention, per
`tools/planner.py`'s own already-established `_argv`), and the Poisson
arrival timing — not from `--synthetic_request_generator_config_num_requests`
directly. Given QPS=4.0 and Gate B's own real measured single-host TP=1
TPOT (~6.1ms/token, so a full 31-decode-step request completes its
decode phase in roughly 190ms), the real-world *arrival* overlap is
small (on the order of 1–4 requests decoding concurrently on average).
**This task did not run a dry simulation to confirm Frontier's own
internal batch_size distribution for this exact regime** — doing so
would require the very model profile this task is scoping, a
circularity this task's own §0 instruction does not ask it to resolve.
The proposed `{1,2,4,6,8,12,16}` grid is a safety-margined estimate
using established precedent, not a confirmed simulator trace; this is
the one part of the envelope carrying the most genuine residual
uncertainty, named here rather than presented as certain.

---

## §7. TP coverage — checked against the real predictor, not assumed

**§6's own question, answered directly**: TP coverage is **not**
handled analytically after a single TP=1 profile. Confirmed two ways:
(1) `num_tensor_parallel_workers` is a real, recorded *feature column*
in every existing `attention.csv`/`linear_op.csv` (re-checked on
`qwen3-a3b-30b-moe`'s own header and real values `{1,2,4,8}`) — the
trained sklearn model uses it as an input feature, meaning the training
data itself must contain real rows at each TP value profiled, not a
formula applied after the fact. (2) Physically, TP genuinely changes
per-worker attention head-partitioning (`n_q_head`/`n_kv_head` per
worker) and per-worker linear-op matrix dimensions — real, different
compute shapes, not a linear rescaling of a single TP=1 measurement.

- **Does profiling need separate data for TP=1, 2, 4?** Yes — three
  real profiling passes (or one invocation sweeping all three via
  `--num_tensor_parallel_workers 1 2 4`, matching how every existing
  model's own grid was collected).
- **Is TP=4 compute profiling needed even though single-host Gate C
  only uses TP≤2?** **Yes** — the two-host cross-host TP=4 candidate
  (`dual-mi355x-crosshost-tp4`) needs it. Per-GPU compute cost at TP=4
  is a real, separate profiled shape; cross-host placement only changes
  the *communication* cost (already covered by the existing, unrelated,
  model-independent `data/profiling/network/mi355x_8gpu/` collective
  profiles, per §3's own table) — it does not change what compute-shape
  data each individual GPU's own attention/linear-op predictor needs.
- **`profiled_tp=(1,2,4,8)` was not assumed** — `dc-sim`'s own
  `ModelSpec.profiled_tp` default of `(1,2,4,8)` (Task 35's own finding)
  is a *convention* about what every model *happens* to have been
  profiled at, not a Frontier mechanism guaranteeing any of those values
  work without real data; for Qwen3-0.6B, only `{1,2,4}` would actually
  be true after this plan, and a future `ModelSpec` entry for this model
  must declare `profiled_tp=(1,2,4)` explicitly (not the `(1,2,4,8)`
  default), per Gate A.1's own already-established discipline of never
  inheriting a coverage claim that isn't backed by real data.

---

## §8. Task 52/53 fix verification — against the actual code path this job would use

**A. Phase filter (Task 53 Fix A).** `src/integration/execution_time_predictor/mla_phase_filter.py`,
wired into `install()` as `mla_phase_filter: bool = False`
(`src/integration/install/__init__.py`, confirmed by re-reading Task 53's
own report). **Not applicable to this profiling job** — Fix A patches
`_train_mla_attention_layer_models`, called only for LATENT_MLA
operators (`attn_mla_*`); Qwen3-0.6B is DENSE_KV (§2), so this function
is never invoked for it, with or without the patch installed. Recorded
as `phase_filter_applied: null` (inapplicable) for this model's own
future `ProfileProvenance` — **not** `false`, since `false` would
falsely imply the check was made and failed to apply, when in fact the
condition for it to matter doesn't exist for this model at all.

**B. Block-table aliasing (Task 53 Fix B).** `src/integration/profiling/attention_block_table_fix.py`,
**standalone, not wired into `install()`** — confirmed directly: (1) it
patches `AttentionWrapper._get_input_tensors`
(`frontier/profiling/attention/attention_wrapper.py`), a method that
only runs during a profiling CLI invocation, never during simulation,
so there is no shared `install()` call site to hook; (2) that module
imports `torch` unconditionally, absent from this sandbox. **This fix
is real, general-purpose (not MLA-specific — it fixes how every batched
attention profiling run builds its block table, `batch_size>1` for
*any* attention family), and directly relevant to this job's own
`batch_size ∈ {1,2,4,6,8,12,16}` grid (§6)** — most of those points have
`batch_size>1`, exactly where the aliasing bug (every sequence in a
batch reusing the same cache blocks, corrupting the profiled shape) can
fire. **Verification of "is it active on the actual code path this
profiling command uses" is a real, unresolved gap**: because it needs
`torch` (absent in this sandbox), whether Fix B is actually applied to
the profiling run requires either (a) `torch` being available in
whatever real environment eventually runs the profiler (an environment
question, not a code question — this sandbox's absence of `torch` says
nothing about the real GPU host), and (b) an explicit, deliberate step
to apply the patch before invoking `frontier.profiling.attention.main`
(it is not automatic anywhere, unlike Fix A). **This plan's own §16
pre-run commands must include applying Fix B explicitly, and the
structural validation (§13.A) must confirm it took effect**, or this
job would silently re-collect aliased `batch_size>1` attention data —
precisely the mistake this section exists to catch before spending GPU
time. Recorded as `block_table_fix_applied: true` only if actually
confirmed applied at collection time; `null` until then, never
defaulted to `false` or `true` without a positive check.

**C. Flat extrapolation (Task 52's own finding, re-verified as a
*general* mechanism, not MLA-specific).** Re-read
`sklearn_execution_time_predictor.py` directly this task: `attn_decode`/
`attn_prefill` (DENSE_KV's own multi-feature attention operators, the
ones this job's data trains) call `_get_on_demand_prediction` — **the
same function, same mechanism** Task 52 found for MLA's own multi-feature
operators (exact-feature-vector lookup first, live `model.predict()`
fallback on any miss, which flat-lines at the nearest training leaf
outside the profiled range). **Task 52's own correction to its own
premise — "multi-feature operators go through a different path than
single-feature dict-lookup ones" — is a distinction between operator
*shapes* (single- vs. multi-feature), not between attention *families*
(MLA vs. dense).** This means: the flat-extrapolation risk this plan's
own §6 envelope is designed to avoid is real and applies to Qwen3-0.6B's
own `attn_decode`/`attn_prefill` exactly as it applied to `deepseek-v3`'s
`attn_mla_decode` — this is not a hypothetical caution, it is the
confirmed, general behavior of the exact function this new profile's
own predictions would be served by. **No coverage-metadata mechanism
currently records the new grid's own bounds automatically** — this is
a real, currently-missing piece (§9/§10 below define what must be
recorded manually, since Frontier's own predictor does not persist
per-feature training bounds anywhere today, confirmed by Task 53's own
§5: "per-operator profiled bounds... cheap to compute, not currently
stored anywhere").

**Single-feature dense projections** (`attn_pre_proj`, `mlp_up_proj`,
etc.) use a **different, stricter mechanism**: an exact dict lookup on
`effective_tokens` alone, raising `KeyError` on any miss (Task 48's own
`mlp_up_proj` crash is the real, already-observed instance of this).
**This means §6's grid for these operators must include the *exact*
token counts Gate C's real workload will produce, or the run crashes
outright rather than silently extrapolating** — a harder failure mode
than B/C above, and the reason §6's prefill grid explicitly includes
`5` itself (the real value), not only bracketing values.

**D. Operator/phase coverage — the concrete "no `KeyError: mlp_up_proj`
after spending GPU time" check.** §3's own table is exactly this
check, run before any GPU command: every operator DENSE_KV/dense-MLP
needs (`attn_prefill`, `attn_decode`, `attn_kv_cache_save`,
`attn_input_reshape`, `attn_output_reshape`, `attn_pre_proj`,
`attn_post_proj`, `attn_rope`, `input_layernorm`,
`post_attention_layernorm`, `emb`, `mlp_up_proj`, `mlp_down_proj`,
`mlp_act`) is covered by the proposed `attention.main`/`linear_op.main`
invocations in §16, run **without** `--is_moe` (Qwen3-0.6B is dense —
the exact flag Task 48 found governs whether the dense-MLP columns are
produced at all). `lm_head_linear`'s own status (§3) is the one open
item the structural validation (§13.A) must resolve for real, post-run.

**E. QK-norm allowlist gap — found in this task, fixed, and fully
verified in-sandbox (unlike B, this one needs no `torch`).**
`frontier/config/model_config.py::QK_NORM_MODEL_TYPE_ALLOWLIST =
{"qwen3_moe", "qwen3_next"}` is missing plain `"qwen3"`.
`_infer_use_qk_norm_from_hf_config` feeds the boolean `use_qk_norm`
feature column already present on every real `linear_op.csv`
(confirmed on `Llama-2-7b-hf`'s own mi355x header, column 61) — not a
separate operator; `attn_pre_proj_q_norm` (a name that appears
elsewhere in `frontier/model_architectures.py`) belongs to `step3_text`'s
own unrelated MFA linear-attention profile, confirmed not to apply to
Qwen3-0.6B's generic dense/GQA path. Confirmed live against HuggingFace
`transformers`' own `Qwen3Attention` source (fetched from `main` this
task): `q_norm`/`k_norm` (`Qwen3RMSNorm`) apply **unconditionally** to
every Qwen3 variant, dense or MoE — the allowlist's own two entries are
real but incomplete. **This matters at collection time, not only
prediction time**: `linear_op_impl.py` reads
`getattr(config, "use_qk_norm", False)` to decide whether to actually
run the QK-norm RMSNorm compute while profiling `attn_pre_proj` — left
unfixed, the real GPU measurement itself would be collected too fast,
and re-collecting after the fact would be the only remedy, not a
downstream reprocessing step. Fixed via a new guarded runtime patch,
following this project's own established Task 20/47/53 pattern:
`src/integration/profiling/qk_norm_allowlist_fix.py`
(`install_qk_norm_allowlist_fix()`, opt-in via `install()`'s own
`qk_norm_allowlist_fix: bool = False` parameter, guarded by an
exact-contents check of the allowlist so a future upstream change isn't
silently overwritten). 9 new tests, all passing in this sandbox
(`frontier.config.model_config` imports no `torch`, unlike Fix B).
**Must be installed and confirmed applied before §16's profiling
commands run**, added there explicitly.

---

## §9. Profile provenance — exact schema for this profile

Every field below, `null` when genuinely unknown at plan time (never
`false` as a stand-in):

```
model_id: "Qwen/Qwen3-0.6B"
model_revision: "c1899de289a04d12100db370d81485cdf75e47ca"
device: "mi355x"
device_arch: "gfx950"                          # sim-real/config/gpu_machines.yaml, real
profile_files: [
  "data/profiling/compute/mi355x/Qwen3-0.6B/attention.csv",
  "data/profiling/compute/mi355x/Qwen3-0.6B/attention_kernel_only.csv",
  "data/profiling/compute/mi355x/Qwen3-0.6B/linear_op.csv",
  "data/profiling/compute/mi355x/Qwen3-0.6B/linear_op_kernel_only.csv",
]
profile_collection_commit: null                # dc-sim/Frontier git SHA at collection time -- fill in at run time
profiling_tool_commit: null                    # same -- Frontier's own frontier/profiling/ commit
attention_backend: "TORCH_SDPA"
mlp_backend_identity: "generic dense linear_op (frontier.profiling.linear_op.main, no --is_moe)"
grid_axes: ["num_tensor_parallel_workers", "batch_size", "kv_cache_size", "total_tokens (prefill)"]
grid_points: <§10's own table, verbatim>
tp_coverage: [1, 2, 4]
operator_coverage: <§3's own table, "required for Gate C" column, verbatim>
phase_filter_applied: null                     # inapplicable -- DENSE_KV, not MLA (§8.A)
block_table_fix_applied: null                  # must be confirmed true at collection time, or this stays null (§8.B)
qk_norm_allowlist_fix_applied: true             # verified in-sandbox, no torch needed (§8.E) -- the one fix this plan can already confirm, not left null
rocm_smoke_test_passed: null                    # addendum Item A -- not run in this task (§16 Stage -1); must be true before trusting any row this plan proposes
collection_timestamp: null                      # fill in at run time
host: null                                      # fill in at run time -- must be one of the real, currently-free mi355x hosts, checked fresh
runtime_container_identity: null                # fill in at run time
rocm_version: null                              # fill in at run time -- expect "7.2.3" per this checkout's own sim-real precedent, not assumed
torch_version: null                             # fill in at run time -- expect "2.11.0+gitd0c8b1f" per this checkout's own sim-real precedent, not assumed
known_limitations: [
  "TORCH_SDPA is Frontier's own portable reference backend, not confirmed equivalent to whatever ROCm attention kernel real vLLM serving actually selected in Gate B",
  "AITER is non-functional on this checkout (Task 49) -- no path to a production-tuned comparison",
  "batch_size grid is a safety-margined estimate against established precedent, not a confirmed trace of Frontier's own scheduler behavior for this exact regime (§6)",
  "lm_head_linear coverage status not resolved at plan time (§3) -- must be checked structurally after collection",
]
evaluation_status_vocabulary: ["IN_PROFILE", "INTERPOLATED", "EXTRAPOLATED", "UNKNOWN"]  # see note below
```

**Note on the four-way status vocabulary**: Gate C's own report already
flagged that `tools/stage2/contracts.ProfileProvenance` has no explicit
field for this. This plan does not add one (out of scope — no
planner-core/contract code changes were made investigating this task
either); `known_limitations` (free text) and `grid_bounds` (an untyped
dict) are the "closest existing representation," per this task's own
§8 instruction, and are what §13.C's future validation step should
populate.

---

## §10. Storage / installation path

- **Raw CSVs**: `data/profiling/compute/mi355x/Qwen3-0.6B/` — a **new**
  directory, following the exact convention every existing model uses
  (`data/profiling/compute/<device>/<model_name>/*.csv`). No existing
  model's files are touched.
- **Generated predictor artifact**: none persisted separately —
  confirmed by re-reading `sklearn_execution_time_predictor.py`'s own
  caching (`_get_model_hash`, Task 52's own finding): the trained model
  is cached under `cache/*.pkl`, keyed by an MD5 hash of the config plus
  the training dataframe's own JSON serialization — a new model name
  automatically gets a new cache entry the first time it's constructed;
  no manual cache management is needed.
- **How `ModelSpec` becomes addressable**: a new
  `data/config/models/Qwen3-0.6B.json` (Frontier-side model config,
  matching every other model's own file), plus a new
  `tools/planner_core.ModelSpec(model_name="Qwen3-0.6B", ...)`
  construction on the `dc-sim` side (not built in this task — §17
  explicitly permits "model registration plumbing if no profile data is
  required to test it," and this specific piece *does* require the real
  profile to be meaningful, so it is named here as the next real step,
  not attempted).
- **MI355X device identity**: already established
  (`_DEVICE_MEMORY_GB["mi355x"] = 288`, Task 48/49); no change needed.
- **`profiled_tp` population**: the new `ModelSpec` entry must declare
  `profiled_tp=(1, 2, 4)` explicitly (§7) — not the `(1,2,4,8)` default.
- **What must change vs. what must not**: only new files are added
  (`data/config/models/Qwen3-0.6B.json`, the new `data/profiling/compute/mi355x/Qwen3-0.6B/`
  directory). No existing profile, config, or predictor file is
  modified — confirmed this satisfies §9's own preference exactly.

---

## §11. Minimum sufficient grid — concrete, quantified

Row counts derived from the real grid shapes above, using each
operator family's own actual cross-product structure (attention:
TP × batch_size × decode-kv_cache_size, plus a separate prefill sweep
at TP × prefill-tokens; linear_op: TP × a token-count list, per
Task 49's own established "linear_op's own grid is TP-independent in
count per TP value" pattern).

| operator/profile group | axes | grid values | rows (per profile_method) | why needed |
|---|---|---|---|---|
| `attention.csv` — decode | TP × batch_size × kv_cache_size | `{1,2,4}` × `{1,2,4,6,8,12,16}` × `{0,8,16,24,32,48,64,96,128}` | 3 × 7 × 9 = **189** | §6's own decode envelope, real TP set |
| `attention.csv` — prefill | TP × prefill total_tokens | `{1,2,4}` × `{1,2,4,5,8,16,32}` | 3 × 7 = **21** | §6's own prefill envelope, brackets the real 5-token point |
| `linear_op.csv` (dense: `attn_pre_proj`/`attn_post_proj`/`attn_rope`/norms/`emb`/`mlp_up_proj`/`mlp_down_proj`/`mlp_act`) | TP × effective_tokens | `{1,2,4}` × a token list covering `{1,2,4,5,8,16,32,64,128}` (matches the attention-side envelope; single-feature exact-lookup operators, so the real Gate C token counts — `5` prefill, and per-request decode token positions — must literally appear in this list, not merely be bracketed) | 3 × 9 = **27** | §8.D's own "exact miss = `KeyError`" requirement |

Each of the three rows above must be collected **twice** — once at
`--profile_method cuda_event` (default) and once at
`--profile_method record_function` (the `kernel_only` alias) — because
Task 49's own real, re-confirmed finding (`shared_prediction_model_manager.py`'s
own `_get_measurement_types_for_cluster`) is that under
`pd-af-disaggregation` (the `sys_arch` every real evaluation in this
project uses, `tools/planner.py`'s own `_argv`), `DECODE_ATTN` needs
**both** `CUDA_EVENT` and `KERNEL_ONLY` unconditionally, and
`DECODE_FFN` needs `KERNEL_ONLY` **only**. `PREFILL` needs `CUDA_EVENT`
only. Since attention and linear-op data feed all three clusters, the
conservative, structurally-correct choice — matching what every
`kernel_only`-complete model on `h800`/`rtx_pro_6000` already does — is
to collect both measurement types for all three rows above, rather than
trying to split the grid per-cluster and risk missing one.

| | rows/method | × 2 methods |
|---|---|---|
| attention (decode+prefill) | 189 + 21 = 210 | **420** |
| linear_op | 27 | **54** |
| **TOTAL NEW ROWS** | | **474** |
| **ESTIMATED REUSED ROWS** | | **0** (§5) |
| **TOTAL REAL GPU MEASUREMENTS** | | **474** |

No MoE profiling (`moe.csv`) at all — `is_moe=False` (§3).

### §11a. `kernel_only` — addendum Item B, answered as an option with a price

**This is not actually optional for Gate C, which corrects the
addendum's own framing.** The addendum's own text reasoned that Gate
C's first validation space (single-host co-location) "does not require
the dispatch-free timing variant," scoping `kernel_only` to "the
architecture after this one." Reading
`shared_prediction_model_manager.py::_is_kernel_only_measurement_enabled_for_cluster`
directly (lines 283–303) shows this assumption does not hold **for
this project's own tooling**: under `sys_arch="pd-af-disaggregation"` —
which `tools/planner.py`'s own `_argv()` hardcodes for *every* real
evaluation this project runs, single-host candidates included, since
co-location is a placement choice on top of the same
`DECODE_ATTN`/`DECODE_FFN` pool split, not an architectural change —
the function returns `True` unconditionally (regardless of CUDA-graph
mode) for `DECODE`, `DECODE_ATTN`, `DECODE_FFN`, and `MONOLITHIC` —
every cluster type except `PREFILL`. `_get_measurement_types_for_cluster`
(lines 305–329) then requires: `PREFILL` → `CUDA_EVENT` only;
`DECODE_ATTN` → **both** `CUDA_EVENT` and `KERNEL_ONLY`; `DECODE`/
`DECODE_FFN` → `KERNEL_ONLY` **only**, with no `CUDA_EVENT` fallback at
all. Gate C's own single-host TP1/TP2 candidates build exactly these
pools (`tools/planner.py` lines 264–275: `PoolKind.DECODE_ATTN` →
`ClusterType.DECODE_ATTN`, `PoolKind.DECODE_FFN` → `ClusterType.DECODE_FFN`).
**`kernel_only` is therefore required for Gate C's very first
candidates, not deferred work** — §11's own row count already reflects
this (the "×2 profile_methods" row), not as a hedge but as a
requirement this reading found.

This also explains why Gate A's own real `h800`/`Phi-tiny-MoE-instruct`
runs succeeded under this same `pd-af-disaggregation` convention while
`mi355x` cannot yet run *any* model: `data/profiling/compute/h800/Phi-tiny-MoE-instruct/`
already has `attention_kernel_only.csv`/`linear_op_kernel_only.csv`/
`moe_kernel_only.csv` (confirmed present), while an exhaustive search of
`data/profiling/compute/mi355x/` found **zero** `*kernel_only*` files
for *any* model on this device — a pre-existing, device-wide gap, not
specific to Qwen3-0.6B. (`rtx_pro_6000` also has several models'
`kernel_only` files; `mi355x` alone has none.)

**Which files (the addendum's specific ask).** For Qwen3-0.6B (dense,
no experts, `is_moe=False`): `attention_kernel_only.csv` (feeds
`DECODE_ATTN`) and `linear_op_kernel_only.csv` (feeds `DECODE_FFN`'s
dense-MLP ops). **`moe_kernel_only.csv` is not needed** — there is no
expert-routing operator for a dense model to measure, regardless of
cluster type. This is exactly "linear_op, moe and attention variants
may not all be needed" resolved by reading the resolver, per the
addendum's own instruction, rather than discovered by a crash.

**Marginal cost / one invocation or two.** Confirmed directly in
`frontier/profiling/utils/__init__.py`: `--profile_method` accepts
`cuda`/`cuda_event` (→ `MeasurementType.CUDA_EVENT`, writes
`attention.csv`/`linear_op.csv`) or `kernel_only`/`record_function` (→
`MeasurementType.KERNEL_ONLY`, writes
`attention_kernel_only.csv`/`linear_op_kernel_only.csv`) — one CLI flag
value per invocation, one output file per invocation.
`build_profile_method_output_path`'s own docstring states this
explicitly: "CUDA-event data keeps the primary op filename... Kernel-only
data uses the simulator's dedicated `*_kernel_only.csv`... convention."
**The two conventions cannot be collected in one invocation.** The
addendum's own characterization — "the grid, the shapes and the
tooling do not [change]" — is confirmed exactly: same
`--num_tensor_parallel_workers`/`--batch_size_list`/
`--decode_kv_cache_size_list`/etc. flags, same shapes, same CLI tool,
one flag value flipped. The **price** is therefore not a separate
booking or a different grid — it is a second full pass through the
identical shape sweep, i.e. **roughly 2× the wall-clock of one profile
method alone** (§12's own split accounting already prices this in), on
the *same* GPU session, immediately after the first pass. There is no
cheaper option once `kernel_only` is confirmed required (as it is
here, structurally, not by choice) — the only real decision this task
leaves open is §16 Stage -1's own question of whether `record_function`
mode runs on this device at all.

---

## §12. Estimated GPU time — from this project's own real evidence, not blindly applied

Task 49's own two real anchors (`GPTOSS_TRUE_MIXED_BATCH_PROFILING.md`,
re-quoted exactly, not paraphrased): a **well-sized 325-point attention
grid took 15 seconds (~22 points/sec)**; a **badly-sized 115,818-point
grid ran 86.5 hours at 42% before dying (~1.8 points/sec)**. Neither
figure is for `linear_op` specifically, and — Task 49's own honest
caveat, repeated here because it applies just as directly to this
plan's own `record_function`/`kernel_only` half — **`record_function`
tracing has never been run on `mi355x`, for any model, for any
operator**, so its own per-point overhead relative to `cuda_event` is
unmeasured here.

This plan's own grid (**474 rows**, well inside the "well-sized" scale
Task 49's own 325-point anchor represents, not the 115,818-point
disaster) is the right regime to apply the *fast* anchor to, with the
slow one kept only as an explicit worst-case bound, per this task's own
§11 instruction not to blindly apply either number:

| | cuda_event half (~237 rows) | kernel_only half (~237 rows, unmeasured overhead) | total |
|---|---|---|---|
| optimistic (22 pts/sec both) | ~11s | ~11s | **~25s** + startup |
| expected (22 pts/sec cuda_event; assume 2–5× slower for unmeasured `record_function` overhead, i.e. 4–11 pts/sec) | ~11s | ~20–60s | **~1–2 minutes** + startup |
| pessimistic (1.8 pts/sec, Task 49's own documented worst case, applied to the whole grid as the explicit safety bound) | | | **~4.4 minutes**, or, if `record_function` on this device turns out to behave like the 86.5-hour disaster case rather than merely slow, **open-ended** — this is exactly why §16's commands must run under `tmux`/`nohup` regardless of how short the estimate looks (Task 49's own "86-hour lesson," restated) |

**Startup overhead** (model load, Ray/worker initialization, one-time
per `attention.main`/`linear_op.main` invocation, not per-row): not
independently measured in this project's own record for a model this
small; expect low tens of seconds based on every other model's own
observed startup pattern, not a hard number.

**GPUs required concurrently**: **one.** Every existing `mi355x`
profiling command in this project's own record (Task 49's own three
commands) sets `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`/`--num_gpus 8`
as a *convenience default* for parallel Ray workers across shapes, not
a hard requirement — the actual compute unit under profiling
(`attn_tp=1,2,4`) never needs more than 4 GPUs even at its largest
degree, and a single-GPU-visible invocation (`--num_gpus 1`,
`CUDA_VISIBLE_DEVICES=0`) is the more conservative, real-fleet-friendly
choice this task's own §11 instruction asks to prefer.

---

## §13. Minimal vs. reusable profile — compared, one recommended

**A. Minimal Gate-C profile** (this plan's own §11 grid): 474 rows, TP
∈ {1,2,4}, the narrow shape envelope §6 derives, ~1–5 minutes optimistic/
expected GPU time. Covers exactly, and only, what Gate C's four real
candidates need.

**B. Reusable product profile**: the same operator families, but at
every existing model's own established broader convention — TP∈{1,2,4,8}
(not needed by Gate C, but consistent with `profiled_tp`'s own project-wide
default), `batch_size` up to 32 (double this plan's own ceiling, giving
room for a future, less QPS-constrained workload), decode `kv_cache_size`
up to `512` (matching `deepseek-v3`'s own real serving-shape need Task 52
found missing, applied here defensively even though Qwen3-0.6B's own
real Gate C need tops out at 36), prefill tokens up to `512` similarly.
Rough scaling: roughly 2× the TP axis, 2× the batch axis, ~4× the
kv_cache/token-range axis (more grid points to cover a wider span at
similar density) → on the order of **10–16×** this plan's own row count,
i.e. **~5,000–7,500 rows**, tens of minutes to comfortably over an hour
of GPU time even at the optimistic per-point rate, with the same
`record_function`-overhead uncertainty compounding at a larger scale.

**Recommendation: A, the minimal Gate-C profile.** This task's own
closing instruction is explicit ("we care about unblocking Gate C
quickly without creating another profile too narrow to be useful") —
the minimal grid, per §6's own derivation, is not narrow in the sense
Task 52 found harmful (it does not stop at the request's own edge; it
brackets the real envelope with real margin on both sides). A broader
"reusable" profile is real, useful *future* work — but it is not what
unblocks Gate C, it is a hedge against a *different*, not-yet-specified
future workload, and this task's own §1 instruction explicitly warns
against exactly that ("we do NOT want a universal giant profiling
grid").

---

## §14. Post-profile validation procedure — defined now, run later

**A. Structural** (run immediately after collection, before any
`SimulationEvaluator` invocation): every operator in §3's own "required"
column has a real column in the new CSVs; `num_tensor_parallel_workers ∈
{1,2,4}` are all present; `lm_head_linear`'s real status (§3) is
resolved one way or the other; every provenance field in §9 that was
`null` at plan time is now filled with a real, observed value.

**B. Held-out/internal**: with 474 real rows split across ~10 operators,
most individual operators will have on the order of tens of rows each —
enough for a genuine leave-one-out check (Task 53's own
`attn_mla_decode` example generalized well at only 8 rows once
correctly phase-filtered) but this task does not pre-judge the result;
run LOO per operator, report the real MAPE, and do not over-interpret a
tiny per-operator subset if one turns out smaller than expected (this
task's own explicit caution).

**C. Gate C shape coverage** — the concrete, decisive check: instantiate
the real four Gate C candidates' own exact shapes (prefill=5 tokens,
decode kv_cache_size 5–36, the real batch sizes Frontier's own scheduler
actually produces for this workload — observable for the first time
once a real evaluation can run) and query the trained predictors
directly, the same way Task 52 did for `deepseek-v3`. Every one of
these real queries must land `IN_PROFILE` or `INTERPOLATED` (within the
convex hull of real profiled points) — **never `EXTRAPOLATED`, never
`UNKNOWN`**. If any real Gate C shape lands outside the grid this plan
proposed, that is this plan's own failure to predict correctly, to be
fixed by widening the grid at the specific missing point, not by
accepting the extrapolation.

**D. Frontier smoke**: one real candidate (e.g. `single-mi355x-tp1`)
run through the real `SimulationEvaluator`/`tools/planner.py` pipeline
after `install()` — must produce no `KeyError`, no `Unknown`
classification from `can_evaluate`, no subprocess crash, and a real,
non-dummy `mean_tpot_ms`. This is Gate C's own actual unblocking
condition — a future execution step, not run in this task.

---

## §15. Hard-coded-number audit

Every numeric bound in §6/§11 is traced to a stated source; none is a
magic number unconnected to `ModelSpec`/the real workload:

| value | GOOD or BAD | derivation |
|---|---|---|
| prefill tokens `{1,2,4,5,8,16,32}` | GOOD | `5` is the real Gate B `prompt_tokens`; the rest bracket it |
| decode `kv_cache_size` up to `128` | GOOD | real need is 5–36 (from `max_tokens=32`, a frozen Gate C workload parameter); `128` is an explicit, stated 3.5× margin, not a guess |
| `batch_size` up to `16` | GOOD, but see §6's own honest gap | matches every other model's own real, already-collected precedent (`qwen3-a3b-30b-moe`'s own `{1..16}`), not invented; the *choice* to reuse that precedent (rather than deriving Frontier's own real scheduler behavior) is disclosed as an estimate, not hidden as a fact |
| `attn_tp ∈ {1,2,4}` | GOOD | exactly Gate C's own four candidates' real TP requirements, read from Gate C's own report |
| `288 GB` MI355X memory | GOOD, not touched | already established (`_DEVICE_MEMORY_GB`, Task 48/49), not re-derived or re-guessed here |
| `1024`/`16`/`8`/`28`/`128` (hidden/heads/kv-heads/layers/head_dim) | GOOD | Qwen3-0.6B's own real config, fetched live and cross-checked |
| `gfx950` device_arch | GOOD | `sim-real/config/gpu_machines.yaml`'s own real, already-recorded value |
| any hidden device path, baked-in TP list beyond what's stated, or magic grid unrelated to the derivation above | **none present** | every grid axis in §11 traces to a row in §6's own table |

No predictor output was hard-coded anywhere in this plan — §11/§12 are
row counts and time *estimates*, never fabricated measurement values;
§16 proposes commands, none of which are executed.

---

## §16. Exact pre-run plan — proposed, NOT executed

### Stage -1 — ROCm smoke test (addendum Item A): proposed, execution requires separate explicit approval, NOT run in this task

**Nobody has confirmed this profiler runs on ROCm/MI355X in this
checkout, and this task did not confirm it either** — a full
code-level (static) analysis was done in place of real execution, per
an explicit choice made when this genuine tension (the addendum's own
directive to run one point vs. this task's own base "do not touch a
GPU" instruction) was raised back to whoever requested this plan: do
the static analysis now, propose the smoke test as the mandatory first
real step, execute only after separate explicit approval. Nothing
below was run.

**Code-level reasoning, two paths, treated separately because they
exercise different PyTorch subsystems:**

1. **`--profile_method cuda_event` (the real MI355X convention every
   existing profile in this checkout already uses).**
   `CudaTimer.__enter__`/`__exit__`
   (`frontier/profiling/common/cuda_timer.py` lines 54/93-98) does
   nothing but `torch.cuda.Event(enable_timing=True)` +
   `.record()` — no `torch.profiler`, no Kineto, no `handle_trace`, no
   `filter_str`. `--disable_ray --num_gpus 1` (every existing `mi355x`
   profiling command's own convention, re-confirmed via
   `linear_op/main.py`'s own docstring: "Sequential single-GPU
   profiling") means no Ray workers and no `torch.distributed`/NCCL
   process group at all — the TP-sharded shapes this plan's own grid
   needs are computed analytically per TP value and profiled
   sequentially on one GPU, never requiring a live collective. ROCm
   PyTorch keeps the `torch.cuda` namespace as an alias (confirmed:
   this is how every ROCm PyTorch build works, not specific to this
   checkout), so `torch.cuda.Event`/`.record()`/`.synchronize()` should
   work identically. **This path has the fewest moving parts and the
   highest confidence of the two — but "should work" is a code-reading
   conclusion, not a confirmed execution result.**
2. **`--profile_method record_function`/`kernel_only` (required per
   §11a).** `CudaTimer` enters/exits a bare `torch.profiler.record_function`
   context (`cuda_timer.py` lines 91-92) with no `handle_trace` call of
   its own — the actual duration must come from an *outer*,
   separately-active `torch.profiler.profile(...)` capturing these
   named regions via PyTorch's Kineto-based profiler, which on ROCm
   depends on `roctracer`/`rocprofiler` integration, a genuinely
   different (and historically more fragile) code path than plain CUDA
   events. **The specific "nccl trace filter" recalled in the
   addendum — `filter_str="nccl"`, `collectives_wrapper.py` line 39 —
   is real, confirmed present exactly as described, but lives in a
   separate raw-collectives-benchmarking tool that explicitly uses
   `ProfileMethod.KINETO` (not `RECORD_FUNCTION`) to benchmark real
   multi-GPU all-reduce/send-recv operations; it is not on the code
   path this plan's own single-GPU compute sweep would exercise either
   way, since compute profiling here needs no process group and no
   collective at all.** That specific fix is very unlikely to be
   needed for *this* profiling job specifically — but this does not
   resolve the broader question, because `record_function`/Kineto's
   own ROCm behavior is untested here regardless of NCCL/RCCL naming,
   and — Task 49's own honest caveat, repeated because it is the
   controlling fact — **`record_function` tracing has never been run
   on `mi355x`, for any model, for any operator, in this project's own
   record.**

**Proposed smoke test (two minimal probes, not one — because the two
paths above exercise genuinely different subsystems and the addendum's
own concern applies to each independently):**

```
PROBE 1 (cuda_event):
python3 -m frontier.profiling.linear_op.main --disable_ray --yes \
  --models Qwen3-0.6B --num_gpus 1 --device mi355x \
  --num_tensor_parallel_workers 1 \
  --num_tokens_list 1 \
  --profile_method cuda_event --output_dir <scratch_output_dir>

PROBE 2 (record_function / kernel_only):
<same command, --profile_method record_function>
```

Smallest possible invocation of each path, one shape, no sweep — per
the addendum's own instruction. Requires: `data/config/models/Qwen3-0.6B.json`
to exist first (§10/§16 Stage 4, a static file, no GPU) so `--models
Qwen3-0.6B` resolves at all; Fix B (§8.B) is **not** required for a
`batch_size=1`/single-shape smoke test (the aliasing bug it fixes only
manifests at `batch_size>1`), so it is deliberately not a precondition
for this probe, only for the real sweep.

**What must be reported once (and only once) this is actually
executed, per the addendum's own explicit standard**: whether each
probe produced a real output row (not a crash, not a silently-empty
file); the exact command run, byte-for-byte; every code change
required to make it run, if any (folded into §12's cost estimate, not
treated as free); whether the `nccl`/`filter_str` fix was among them
(this analysis says it should not be, but that is not yet a confirmed
result); and the wall-clock for that one point each — "the only real
anchor available for extrapolating the full grid's cost," superseding
Task 49's own `h800`/different-device anchors used provisionally in
§12 above. **If either probe does not run, this task's own instruction
is unambiguous: stop and report that** — do not proceed to the full
sweep, and treat the failing path's own row count in §11 as blocked
rather than merely delayed.

**This changes the final recommendation (§18/Final answer F) from what
it would otherwise be**: this plan is not "ready for real GPU
execution" in the sense of "run the full sweep next." It is ready for
exactly one thing next — the two-probe smoke test above, pending
separate explicit approval — and the full sweep remains contingent on
that smoke test's own real result, not on this plan's own code-level
reasoning alone.

**Stage 0 — safety preconditions (read-only, before anything else):**
fresh occupancy check on whichever of `xai-3/4/5/6` is targeted
(`sim_real/scripts/preflight_hosts.py`, unchanged, reused exactly as
Gate B already established); hard abort on `RESOURCE_BUSY`; confirm
Docker-only execution, pinned image/runtime (the same
`vllm/vllm-openai-rocm@sha256:bb44b39a...` image family this project's
own real containers already use, or Frontier's own equivalent profiling
image if different — **not independently confirmed which image
Frontier's own profiling containers use on this fleet; this is a real,
open item for whoever executes this plan, not resolved here**); no host
package installs; no other container touched.

**Stage 1 — apply Fix B explicitly** (§8.B): before invoking
`attention.main`, apply `src/integration/profiling/attention_block_table_fix.py`'s
patch to the real, running Frontier checkout on the profiling host (the
exact mechanism — a source patch, a monkeypatch import, or an explicit
CLI-time hook — depends on how `torch`'s presence on that specific host
lets the patch actually be loaded; not resolved further here, since it
depends on the real host's own environment, unknown at plan time).
Verify applied via the guard test
(`tests/test_attention_block_table_fix_guard.py`) before proceeding.

**Stage 2 — attention profiling** (per TP value, or one sweep call):

```
HOST: <whichever mi355x host passes Stage 0, checked fresh>
GPU: CUDA_VISIBLE_DEVICES=0 (one GPU; --num_gpus 1)
IMAGE: <Frontier's own pinned profiling image/runtime -- not yet identified, see Stage 0>
MODEL: Qwen3-0.6B (new Frontier model config, data/config/models/Qwen3-0.6B.json, to be added first)
REVISION: c1899de289a04d12100db370d81485cdf75e47ca (recorded in provenance only -- Frontier's own profiler does not itself download or pin an HF revision, it profiles synthetic tensors at the declared architecture)
PROFILE OUTPUT PATH: data/profiling/compute/mi355x/Qwen3-0.6B/

COMMAND (cuda_event pass):
python3 -m frontier.profiling.attention.main --disable_ray --yes \
  --models Qwen3-0.6B --num_gpus 1 --device mi355x \
  --num_tensor_parallel_workers 1 2 4 \
  --max_model_len 256 --max_seq_len 256 \
  --min_batch_size 1 --max_batch_size 16 \
  --batch_size_list 1 2 4 6 8 12 16 \
  --decode_kv_cache_size_list 0 8 16 24 32 48 64 96 128 \
  --attention_backend TORCH_SDPA --block_size 16 \
  --profile_method cuda_event --output_dir data/profiling

COMMAND (kernel_only pass, identical grid, §11's own requirement):
<same command, --profile_method record_function>

EXPECTED ROWS: 210 attention rows × 2 profile methods = 420
EXPECTED DURATION: ~25s optimistic / ~1-2min expected / up to several
  minutes pessimistic per pass (§12) -- run under tmux/nohup regardless
EXPECTED VRAM: Qwen3-0.6B is ~0.6B params (~1.2GB at BF16) -- trivial
  relative to MI355X's 288GB; no meaningful VRAM risk expected, not
  independently profiled/measured here
MUTATIONS: writes only under data/profiling/compute/mi355x/Qwen3-0.6B/
  (new directory); no existing file touched
CLEANUP: none beyond normal profiling-container teardown (no persistent
  service is started; this is a batch CLI job, not a server)
VALIDATION COMMANDS: §14.A's own structural checks (column presence,
  TP coverage, row counts) run against the new CSVs directly
```

**Stage 3 — linear_op profiling** (same TP sweep, dense/non-MoE path):

```
COMMAND (cuda_event pass):
python3 -m frontier.profiling.linear_op.main --disable_ray --yes \
  --models Qwen3-0.6B --num_gpus 1 --device mi355x \
  --num_tensor_parallel_workers 1 2 4 \
  --num_tokens_list 1 2 4 5 8 16 32 64 128 \
  --profile_method cuda_event --output_dir data/profiling
  # NOTE: no --is_moe (§3/§8.D -- Qwen3-0.6B is dense; omitting the
  # flag is what produces the mlp_up_proj/mlp_down_proj/mlp_act columns)

COMMAND (kernel_only pass): <same command, --profile_method record_function>

EXPECTED ROWS: 27 × 2 = 54
EXPECTED DURATION: a small fraction of Stage 2's own estimate (fewer
  rows, simpler single-feature shapes) -- no independent anchor exists
  in this project's own record for linear_op specifically (§12); treat
  as bounded by Stage 2's own pessimistic case as a conservative ceiling
```

**Stage 4 — install and register** (no GPU): add
`data/config/models/Qwen3-0.6B.json`; construct the `dc-sim`-side
`ModelSpec` with `profiled_tp=(1,2,4)` explicitly.

**Stage 5 — validation** (§14, in order: A structural, B held-out, C
Gate C shape coverage, D Frontier smoke).

---

## §17. Risks

0. **Nobody has confirmed this profiler runs on ROCm/MI355X at all,
   for either `--profile_method`** (§16 Stage -1) — a full code-level
   analysis found no CUDA-specific lock-in on the `cuda_event` path
   used for TP-sweep compute profiling (no distributed backend, no
   Kineto, `torch.cuda` namespace preserved on ROCm) and traced the
   specific "nccl trace filter" concern to a separate tool
   (`collectives_wrapper.py`) not on this job's own code path — but
   this is a reasoned static conclusion, not an executed result, and
   the addendum's own rule is explicit: if the two-probe smoke test
   does not run, stop and report that rather than proceeding to the
   full sweep. This is the single highest-priority open item, ranked
   above the items below because nothing else in this plan matters if
   the profiler cannot run on this device at all.
1. **`record_function`/`kernel_only` has no precedent on `mi355x`, at
   any scale** (§12) — could be simply slow, or could fail outright
   (a ROCm-side gap, per Task 49's own framing of `TORCH_SDPA` itself
   once being an analogous "starting state" gap). This is the specific,
   narrower case of risk 0 that Probe 2 (§16 Stage -1) is designed to
   surface before the full sweep is attempted.
2. **Fix B's real applicability on the actual profiling host is
   unresolved** (§8.B/§16 Stage 1) — the single most important
   pre-execution check this plan names but cannot close from this
   sandbox.
3. **`lm_head_linear` coverage status unresolved** (§3) — could
   surface as a `KeyError` during Stage 5.D's smoke test if it turns
   out required and this plan's own commands don't produce it; caught
   there, not silently.
4. **`batch_size` envelope is an estimate, not a confirmed trace**
   (§6) — if Frontier's own real scheduler for this exact regime
   produces a batch size outside `{1,2,4,6,8,12,16}`, §14.C's own Gate C
   shape-coverage check would catch it as `EXTRAPOLATED`/`UNKNOWN` before
   any planner prediction is trusted — not a silent failure, but a real
   possibility that could require a second, small profiling pass.
5. **Fidelity ceiling**: `TORCH_SDPA` is not confirmed equivalent to
   whatever real vLLM serving actually used (§4) — a limitation of the
   whole comparison this project can make today, not specific to this
   plan.
6. **Fleet occupancy**: this is real, shared, multi-tenant hardware
   (Gate B's own extensively-documented experience) — Stage 0's own
   fresh-check-and-hard-abort discipline is the mitigation, not a
   guarantee of a clear run on the first attempt.

---

## §18. Final recommendation

The **next real step is not the full sweep** — it is the two-probe
ROCm smoke test (§16 Stage -1), one point per `--profile_method`, and
it requires separate explicit approval before running (this task's own
governing constraint). Once that smoke test passes for real, proceed
with the **minimal Gate-C profile** (§13.A), on **one GPU**, **474 real
rows**, an **estimated 1–5 minutes** of GPU time under normal
conditions (with an explicit, named pessimistic tail if
`record_function` behaves badly), covering `attn_tp ∈ {1,2,4}` — exactly
what Gate C's four real candidates need, no more. Three items must be
resolved before the full sweep, not glossed over: (1) the smoke test
itself must actually pass on real ROCm hardware for both
`--profile_method` values (§16 Stage -1) — not yet run, code-level
reasoning only; (2) confirm Fix B is actually applicable/applied on
whatever real host runs this (§8.B/§16 Stage 1); and (3) identify the
real pinned Frontier profiling image/runtime for this fleet (§16 Stage
0). None of the three was resolved from this sandbox. (The QK-norm
allowlist gap, §8.E, *is* resolved — fixed and tested in-sandbox, the
one fix this plan does not defer.)

---

## Final answers

**A. What exact profile data is missing?** All of it, for
`Qwen/Qwen3-0.6B` on `mi355x`: `attention.csv`/`attention_kernel_only.csv`/
`linear_op.csv`/`linear_op_kernel_only.csv`, at `attn_tp ∈ {1,2,4}`. No
`moe.csv` is needed (dense model).

**B. How many new profile rows/measurements are required?** **474**
real GPU measurements (§11). Zero reusable rows exist anywhere in this
checkout (§5).

**C. How long should it take on one MI355X?** Optimistic ~25 seconds;
expected ~1–2 minutes; pessimistic several minutes to open-ended if
`record_function` misbehaves on this device for the first time ever
(§12) — run under `tmux`/`nohup` regardless of the estimate, per this
project's own established discipline.

**D. Will the proposed grid put all four Gate C candidates inside
valid profile coverage?** By derivation, yes for prefill (brackets the
real 5-token point) and for decode `kv_cache_size` (5–36 real need
inside a 0–128 grid). **Not independently confirmed for `batch_size`**
(§6/§17.4) — this is the one axis this plan's own grid choice rests on
precedent rather than a derived certainty, and §14.C's own future
validation step is what would actually prove or disprove this answer.

**E. Are the phase-filter, block-table, and QK-norm fixes active on the
actual code path that will collect this data?** Phase filter: **not
applicable** (DENSE_KV model, §8.A) — not a gap. Block-table fix:
**unresolved** (§8.B) — needs `torch`, absent from this sandbox; must
be confirmed on the real profiling host. QK-norm allowlist fix:
**resolved** (§8.E) — fixed and verified with 9 passing tests in this
sandbox (no `torch` dependency for this one), must simply be invoked
(`install(..., qk_norm_allowlist_fix=True)`) before profiling runs.

**F. Is this profiling plan ready for real GPU execution?**

## YES WITH CONSTRAINTS — and the immediate next real step is a two-point smoke test, not the full sweep.

The grid, row count, and time estimate are real and derived, not
invented, and one of the two real code defects found in this task
(the QK-norm allowlist gap) is already fixed and tested. But whether
this profiler runs on ROCm/MI355X *at all* has never been confirmed by
real execution (addendum Item A) — this task did a full code-level
analysis in place of running it, per an explicit decision to defer
real execution to separate approval, and that analysis, however
reasoned, is not a substitute for the addendum's own required check.
**Three concrete constraints must be resolved before the full sweep**,
by whoever executes this on the real fleet: (1) run the two-probe
smoke test (§16 Stage -1) and confirm both `--profile_method` values
produce a real row on real MI355X hardware — if either does not run,
stop and report that, per this task's own governing instruction, rather
than proceeding; (2) confirm Fix B's applicability on the actual
profiling host (§8.B); and (3) identify the real pinned Frontier
profiling image/runtime (§16 Stage 0). None is a reason to redesign
this plan; all three are execution-time checks this plan names
explicitly rather than glosses over or assumes will simply work.

**STOP here, per this task's own instruction. No GPU was touched — not
even for the smoke test, which requires its own separate approval.**
