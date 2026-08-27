# Stage 2 — Gate C: first planner ↔ real-hardware decision validation — PLANNER-SIDE HANDOFF REPORT

**STOP. No manifest, no PlannerPrediction, and no handoff package were
produced.** §2's own critical model-compatibility check — required to
run *before* planning anything — found a hard incompatibility: this
project's simulator has no compute profile for `Qwen/Qwen3-0.6B` on
`mi355x`, or on any device, anywhere. Producing a `DeploymentManifest`/
`PlannerPrediction` pair for this model would require either fabricating
a cost prediction with no supporting data, or silently substituting a
different model's real profile — both explicitly forbidden by this
gate's own instructions. Everything below documents what was checked,
why it stops here, and the smallest honest paths forward, per §21.G's
own explicit instruction: "If YES: STOP rather than proceeding with a
misleading handoff."

Branch/working tree: `/work/simulation/dc-sim`, continuing directly
from the `stage2-gate-a-contract` branch (Gate A commit `408ae91`, Gate
A.1 commit `6e2f6b3`). No new code was written for the planner-core or
contract layers in this task — the finding below is a research/analysis
result, not an implementation.

---

## §0. Read-first — what was actually reviewed

- `docs/stage-2-gate-a-contract-report.md` (this project's own Gate A
  report) — the four contract objects, the axis readiness table, the
  `attn_replicas>1` structural block, the throughput-floor
  absolute-only gap.
- `docs/stage-2-gate-a1-cleanup-report.md` — the `ffn_ep` coverage fix,
  the workload/constraints consistency validator, the provenance
  tri-state re-verification.
- `contracts/stage2/*.schema.json` and `contracts/stage2/examples/*` —
  confirmed the four schemas and the real, previously-built examples
  (`single_host_tp2`, `two_host_tp4`, `attn_a_ffn_b`) are unchanged and
  still validate (re-checked directly, §2 below).
- Tasks 44 (joint EP placement), 45 (regime/two-stage search), 56
  (natural-split unreachability diagnosis), 57 (the fix) — the
  mechanisms this gate's §5/§6/§7 instructions assume are already
  built: `enumerate_joint_arrangements`'s `relative` key,
  `plan()`'s own `_mark_indistinguishable_from_winner`, `Regime`'s
  seeded/unseeded split. Confirmed still present and unchanged in
  `tools/planner_core.py`.
- `tools/planner_core.py`'s `Objectives`, `tools/planner.py`'s
  `SimulationEvaluator` — see §4 below.
- This session's own consolidated state — Stage 2 Gate A/A.1 (dc-sim)
  and Gate B (the separate `sim_real` project, real hardware,
  completed with all four configurations converging at n=5) are both
  done; this is the first task attempting to connect them.

---

## §1. First validation space, as specified

Two independent two-candidate decisions, exactly as given — not
re-derived, not expanded:

- **A. Single-host**: `single-mi355x-tp1` (S1) vs. `single-mi355x-tp2`
  (S2), one MI355X host, streaming.
- **B. Two-host**: `dual-mi355x-crosshost-tp2` (D1) vs.
  `dual-mi355x-crosshost-tp4` (D2), two MI355X hosts, streaming.

No EP, replicas, scheduler, RDMA, attention/FFN split, memory-margin
sweep, or extra TP values were added or considered — moot in any case,
since §2 stops before any candidate is evaluated.

---

## §2. Critical model-compatibility check — the blocking finding

**A. Is there a valid ModelSpec/profile path for Qwen3-0.6B on MI355X?**

**No.** Checked directly, not assumed:

- `data/profiling/compute/mi355x/` (Frontier's own real compute-profile
  tree) contains exactly these model directories: `deepseek-r1-0528`,
  `deepseek-v3`, `qwen3-a3b-30b-moe`, `meta-llama/Llama-2-7b-hf`,
  `openai/gpt-oss-120b`, `openai/gpt-oss-20b`. None is Qwen3-0.6B, and
  none is even close by name.
- `data/config/models/*.json` — every model this project has ever
  configured, 28 files — contains no Qwen3-0.6B entry, on any device.
- Real Qwen3-0.6B architecture, confirmed from two independent sources
  agreeing exactly: a live fetch of the model's own public
  `config.json` (`hidden_size=1024, num_attention_heads=16,
  num_key_value_heads=8, num_hidden_layers=28, head_dim=128,
  model_type="qwen3"`), and `sim-real/CLAUDE.md`'s own independently
  recorded real-download facts ("28 layers, 16 attention heads / 8 KV
  heads"). `model_type="qwen3"` is not in Frontier's own LATENT_MLA
  dispatch set (`deepseek_v2/v3/mtp/kimi_k2` — Task 39's own finding,
  re-confirmed by reading `frontier/attention/families.py` directly:
  no `"qwen3"` string appears there at all), so this model would bind
  DENSE_KV if it had any profile — an architecturally ordinary,
  standard-GQA dense model, nothing exotic.
- The **only** dense (non-MoE) model profiled on `mi355x` at all is
  `Llama-2-7b-hf`: `hidden_size=4096, 32 heads / 32 kv (pure MHA, no
  GQA), 32 layers` — a 7B model, ~4× the hidden size, a different
  attention structure (no GQA head-sharing), and a different layer
  count. Using its profile to stand in for Qwen3-0.6B would not
  approximate the real model's cost, it would substitute a materially
  different one's.

Since `sklearn_execution_time_predictor.py`'s own dense-model predictor
(the module `tools/planner.py`'s `SimulationEvaluator` invokes via a
real Frontier subprocess) requires a real profile CSV at
`data/profiling/compute/<device>/<model_name>/*.csv` to even construct
itself, and no such file exists for `(mi355x, Qwen3-0.6B)`, **the real
`SimulationEvaluator` cannot produce a genuine cost prediction for this
model on this device — not a low-confidence one, none at all.**

**B. Does the profile cover TP=1 and TP=2?** Not applicable — there is
no profile to check coverage against.

**C. Can the planner meaningfully evaluate the two-host TP=2/TP=4 case
for this model?** No, for the same reason — cross-host placement
enumeration (`enumerate_joint_arrangements`, confirmed unaffected and
working, §0) is orthogonal to and does not require a compute profile,
but the *evaluation* step (pricing the candidate) still routes through
the same missing predictor.

**D. Are the profile shapes within the request range, or extrapolating
beyond known bounds?** Neither — there is no profile at all to be
either within or beyond. This is a stronger gap than Task 52's own
"flat extrapolation beyond a profiled grid" finding (a real prediction
that degrades gracefully but wrongly outside its training range); here
there is no trained model to query in the first place.

**E. Are Task 52/53 defects relevant?** No. Task 53's Fix A (MLA phase
filter) is specific to LATENT_MLA models — irrelevant, since Qwen3-0.6B
is DENSE_KV. Task 53's Fix B (block-table aliasing) and Task 52's
extrapolation finding are properties of an *existing* trained predictor
being queried outside its safe range — also not applicable, since no
predictor for this (device, model) pair exists to be queried at all.

**One independent, non-blocking check performed anyway**: memory
feasibility does *not* depend on a compute profile — it is a closed-form
formula (`feasible_num_blocks`/`attn_param_mem_bytes`,
`tools/planner_core.py`, verified bit-for-bit against Frontier's own
`ParamCounter`/`MemoryPlanner` in Tasks 33/36/38/39). Computed directly,
for real, using Qwen3-0.6B's own real architecture and `mi355x`'s real
288GB device memory (`_DEVICE_MEMORY_GB["mi355x"] = 288`, already
established in this project since Task 48/49) at
`memory_margin_fraction=0.2` (dc-sim's own established default margin
— a different convention from vLLM's own `--gpu-memory-utilization`
knob Gate B used; the two are not directly comparable and neither
substitutes for the other): `attn_tp ∈ {1, 2, 4}` are all divisibility-
admissible, and each yields a `feasible_num_blocks` in the hundreds of
thousands (e.g. 134,624 at tp=1) — trivially memory-feasible at any
realistic request length, since a 0.6B-parameter model uses a
vanishingly small fraction of 288GB. **Memory feasibility is not, and
would not be, a blocking constraint here** — only the compute-cost
prediction is missing.

**§21.G, answered here since it follows directly**: **Yes**, there is a
disqualifying fairness concern — proceeding would require the
`PlannerPrediction` to describe a different model's real cost behavior
while claiming to predict Qwen3-0.6B's, which is exactly the kind of
comparison this gate exists to make trustworthy, not undermine.
**Stopping per instruction, rather than producing a misleading
handoff.**

---

## §3. Workload identity (reviewed, not instantiated)

The exact real Gate B workload identity was confirmed available and
unambiguous, for when this gate is unblocked: streaming, 32 requests,
QPS=4.0, seed=42, max_tokens=32, `Qwen/Qwen3-0.6B` @ revision
`c1899de289a04d12100db370d81485cdf75e47ca`. **The raw prompt text
itself is not recoverable on the planner side** — confirmed already in
Gate B's own report: it was never committed to source in `sim_real`
either (only `prompt_tokens` counts were persisted); Gate B's own real
execution used a reconstructed prompt (`"The capital of France is"`,
inferred from a "Paris" completion recorded in `sim_real/RUNLOG.md`,
not a byte-for-byte recovery), explicitly disclosed as such in its own
report. Per this gate's own §3 instruction ("If the raw prompt text is
unavailable on the planner side, represent it through a stable workload
artifact/hash/ID supplied by sim_real rather than inventing it"), any
future `WorkloadSpecRef.workload_identity` for this gate should carry
`sim_real`'s own stable identifier for that reconstructed prompt (not
yet defined on the `sim_real` side either) — not a second, independent
reconstruction attempt from this side.

---

## §4. Objective (reviewed, matches expectation — no deviation found)

`tools/planner_core.Objectives` (unchanged since Task 33, re-read
directly this session): `minimize: str = "mean_tpot_ms"`,
`slo_tpot_ms: float`, `min_throughput_rps: float`,
`slo_attainment_floor: float = 0.0`. This matches the spec's own
expected formulation exactly — minimize mean TPOT, subject to memory
feasibility (checked separately, up front, via `feasible_num_blocks`,
never delegated to the evaluator — Task 37's own S3), SLO
(`slo_attainment_floor`, `0.0` meaning "reported, not constrained," per
Task 33's own established convention), and a throughput floor
(`min_throughput_rps`, absolute-only — Gate A's own already-documented
gap, unchanged, not revisited here). **No deviation to report.**

---

## §5–§10: not produced — blocked by §2

No shortlist optimization was invoked (moot, §5), no manifests or
predictions exist for either decision (§6), no placement mappings were
built (§7 — though the exact rank maps this gate specifies, e.g.
`rank0→hostA GPU0, rank1→hostB GPU0` for cross-host TP=2, are simple,
already-expressible shapes given Task 57's own fix; nothing about them
is blocked independently of §2), no `hostA`/`hostB` logical-vs-runtime
host binding question was exercised in practice (§8 — the design
principle itself, keeping real aliases out of planner-core algorithms
and only in exported artifacts, is already how `tools/stage2/exporters.py`
is built, confirmed unchanged), no `ProfileProvenance`/
`IN_PROFILE`/`INTERPOLATED`/`EXTRAPOLATED`/`UNKNOWN` status was
recorded for any candidate (§9 — see the observation below), and no
prediction interval or tie/equivalence group was computed (§10).

**One real, forward-looking observation for §9**: `tools/stage2/contracts.ProfileProvenance`
(Gate A) does not currently carry an explicit
`IN_PROFILE`/`INTERPOLATED`/`EXTRAPOLATED`/`UNKNOWN` enum field — only
`known_limitations: Tuple[str, ...]` (free text) and an optional
`grid_bounds` dict. This gate's own §9 instruction asks for exactly
that four-way status explicitly. Adding it would be a small, additive
schema change (a new field with a documented default), squarely inside
the "expose the contract" carve-out Gate A.1 already used once for
`RuntimeSpec.decode_ffn_scheduler` — but it was not made in this task,
since there is nothing yet to populate it with (§2's own finding means
every real candidate this gate would produce is `UNKNOWN`, trivially,
for want of any profile at all). Left as a design note for whoever
resolves §2 next, not built speculatively here.

---

## §11–§17: not applicable — no planner artifacts exist to freeze, no hardware exists to compare against

§11's own principle (planner predictions frozen before any fresh
hardware result exists) and §17's own regret formula (frozen
definition, independent of whether a difference turns out resolvable)
are both correct as *specified* — re-read and confirmed consistent with
`tools/stage2/decision.py`'s own already-built `compute_decision_validation`
(Gate A, unchanged: `regret_absolute = observed_tpot(selected) -
observed_tpot(best)`, resolvability computed strictly from a
per-configuration `NoiseFloorSource`, never zeroed when unresolved).
Nothing here needed correcting; there is simply nothing to apply either
formula to yet.

---

## §18–§20: Gate B cleanup — completed (independent of §2's blocker)

These three items do not depend on model compatibility and were
completed for real, on the actual Gate B artifacts (`/work/sim-real/artifacts/noise/`),
with no hardware touched:

1. **`model_revision`/`image_digest` repaired** in all four
   `noise_floor.json` files, from the placeholder `"unknown"` to the
   real, known, unchanged-throughout values
   (`c1899de289a04d12100db370d81485cdf75e47ca` /
   `sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7`).
   The underlying code (`noise_pilot_orchestrator.run_pilot`) was also
   fixed to default to these values, so future checkpoints don't
   reproduce the gap.
2. **3 attempts reclassified `OTHER` → `CONTENDED`**, on real,
   per-attempt evidence: each one's own `in_run_occupancy` had already
   independently recorded `conflicting_process_observed=True` and
   `exclusive_run=False` at classification time
   (`single-mi355x-tp2` #2, `dual-mi355x-crosshost-tp2` #3,
   `dual-mi355x-crosshost-tp4` #1). Verified by a hard assertion before
   writing that `mean_tpot_ms`/`valid_repeats` in every `noise_floor.json`
   are byte-for-byte unchanged. `excluded_attempt_rate.json` rebuilt
   from the corrected attempts — `OTHER` is now `0` everywhere; the
   pooled single-host (44.44%) and cross-host (50.00%) exclusion rates
   are unchanged, since the reclassification moved counts between
   exclusion *reasons*, never between clean and excluded.
3. **Secondary noise statistics computed** (§19) from each clean
   attempt's own already-recorded raw records — `mean_ttft_ms`,
   `mean_e2e_ms`, `request_throughput_rps`, `token_throughput_tps`,
   same Student's-t procedure, added into each config's own
   `noise_floor.json`. Representative figure
   (`single-mi355x-tp1`): TTFT 385.5ms (rel. half-width 2.33%), E2E
   574.5ms (2.33%), request throughput 4.73 rps (0.16%), token
   throughput 151.4 tps (0.16%). TPOT remains the primary
   decision-resolution metric, per instruction; these are
   supplementary and did not block anything.

Full detail: `sim-real/docs/stage-2-gate-b-noise-pilot-report.md`'s own
new addendum section.

---

## §21. Most important analysis — explicit answers

**A. Can the exact real-hardware model/workload be evaluated by the
planner?** **No.** §2.

**B. Are all four candidate predictions inside known profile coverage?**
Not applicable — no predictions were produced.

**C. Single-host planner prediction (TP1 / TP2 / TIE)?** Not computed.

**D. Two-host planner prediction (cross-host TP2 / TP4 / TIE)?** Not
computed.

**E. Predicted margin and interval/equivalence status?** Not computed.

**F. Are any hard constraints predicted to fail?** Memory feasibility
specifically: **no** — confirmed real and comfortably satisfied at
TP∈{1,2,4} (§2's own independent check). The compute-cost-dependent
constraints (SLO, throughput floor) cannot be evaluated at all, for
want of a cost prediction — not "predicted to fail," genuinely
unknown.

**G. Is there any reason the hardware experiment would not constitute a
fair planner-vs-real comparison? If YES: STOP.** **Yes — see §2's own
closing paragraph. Stopped.**

---

## §22. Smallest honest paths forward (proposed, not decided or built)

1. **Profile Qwen3-0.6B on MI355X for real**, using Frontier's own
   profiler (`frontier/profiling/`) against real MI355X hardware — the
   technically correct fix, but a new real-hardware *profiling* task
   (distinct from Gate B's own *serving* benchmark), requiring Frontier
   itself to be installed and runnable wherever the MI355X access is —
   not yet established whether `sim_real`'s own fleet has that (its own
   `CLAUDE.md` describes a vLLM-serving environment only). A real,
   separate undertaking, not something this session can do or estimate
   further without first checking that prerequisite.
2. **Re-scope the first real decision-validation gate to an
   already-profiled model** — pick one of the three models this
   project genuinely has real `mi355x` compute profiles for
   (`qwen3-a3b-30b-moe`, `deepseek-v3`, `deepseek-r1-0528`) and re-run a
   Gate-B-equivalent real noise pilot for *that* model specifically
   (Gate B's own noise floors are explicitly scoped to Qwen3-0.6B only,
   per this gate's own §0 — they would not carry over). More work than
   option 3, but keeps every future step honest about what model is
   actually being validated.
3. **Do nothing to close the gap now; report it and stop** (this
   report's own choice) — the correct action within this task's own
   stated scope ("Do NOT change planner objective/search algorithm...
   Do NOT introduce new axes... STOP rather than proceeding with a
   misleading handoff"), leaving the actual decision about which of
   options 1/2 to pursue — and who has the access/mandate to pursue it
   — to the user, not to this session's own unilateral judgment.

**No option was chosen or implemented.** This report recommends option
2 as the more tractable next step if real hardware validation should
continue soon (it reuses Gate B's own already-proven pilot machinery
end-to-end, just for a different model), but does not decide this.

---

## Final answer

**ARE THE PLANNER PREDICTIONS NOW FROZEN AND READY FOR BLIND
REAL-HARDWARE VALIDATION?**

## NO.

No planner prediction was produced for either decision problem — the
model-compatibility check that this gate's own §2 requires before
planning anything found a hard incompatibility, not a resolvable
caveat. Nothing was frozen because nothing was built; nothing in
`tools/planner_core.py`, `tools/planner.py`, or `tools/stage2/` was
changed. The three items that *were* independent of this blocker
(Gate B's own metadata repair and secondary statistics, §18–§20) are
complete and real. Stopped here, per this gate's own explicit
instruction, rather than handing `sim_real` a manifest whose prediction
would describe a model Frontier cannot actually price.
