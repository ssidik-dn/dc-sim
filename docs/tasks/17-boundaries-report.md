# Task 17 — Where the numbers stop being trustworthy

Branch: `task-17-boundaries`, stacked on `task-16-source-binding`.

177 tests pass (no new tests — this is a measurement task, per its own
acceptance criteria; see S7 on the "179" figure), and
`python3 tools/check_import_direction.py` exits 0.

---

## Part A — The pass-through discrepancy

### A.0 The numbers in the spec don't match anything this project produced

Before reconciling, I checked: `9.797`, `8.585`, `6.702021`, `6.610810`,
`13.29`, and `1.36%` do not appear anywhere in this project's source,
tests, or task reports (`grep -r` across the repo, clean). Task 16 never
ran a real-compute comparison at all — every number in
`docs/tasks/16-source-binding-report.md` came from
`tools/run_source_binding_study.py`, which uses dummy execution-time mode
throughout (matching task 15's own convention), with mean transfer times
around 0.94 ms and mean tpot around 422 ms — a completely different scale
from the spec's illustrative figures. I take this as illustrative
scaffolding for the *kind* of arithmetic check to run, not a literal
figure to reproduce, and say so plainly rather than inventing numbers to
match it (S7 has this as a spec-inaccuracy finding).

### A.1 The actual reconciliation, using task 16's real numbers

Task 16's headline comparison (`LOAD_MARGIN=2`, the default):

| | round_robin | topology_aware |
|---|---|---|
| mean M2N transfer time | 0.9388800000817987 ms | 0.9388800000817987 ms |
| mean tpot | 422.4871101440081 ms | 422.4921101440081 ms |

```
transfer delta   0.9388800000817987 - 0.9388800000817987 = 0.000000 ms
tpot delta       422.4921101440081  - 422.4871101440081  = 0.005000 ms/token
decode_tokens = 16  ->  intervals = 15   (task 12's convention: decode_tokens - 1)
implied total from tpot delta:  0.005000 x 15 = 0.075000 ms
actual transfer delta:                          0.000000 ms
```

**The implied total (0.075 ms) and the actual transfer delta (0.000 ms) do
not reconcile at all — not by 13%, by the entire amount.** Checking the
three candidate explanations directly, with evidence, rather than assuming
one:

1. **Different denominators?** No. `n_requests` and `n_tpot` are both 16 in
   every run in this study (logged directly in each result JSON) — the
   same 16 requests contribute to both means, in both scheduler variants.
2. **Decode-step count differs between runs?** No.
   `--fixed_request_length_generator_config_decode_tokens 16` is a fixed
   CLI value shared by both scheduler variants' identical argv (only the
   scheduler's internal lane map differs between them) — this generator is
   deterministic, not sampled, so every request in every run gets exactly
   16 decode tokens.
3. **Something besides transfer time differs.** Confirmed, and load-bearing:
   `mean_m2n_time_s` is not merely close between the two variants — it is
   bit-for-bit identical, and (per the `LOAD_MARGIN` sweep already run in
   task 16) identical across all five margins tested (0, 1, 2, 4, 8) despite
   the real assignment distribution changing substantially across that same
   sweep. A quantity that is *exactly* invariant to a change cannot be the
   cause of a quantity that *does* change. The tpot delta is caused entirely
   by something transfer pricing cannot see: real per-replica queueing
   contention from concentrating more DECODE_ATTN lanes onto the same
   near DECODE_FFN replica (already identified in the task 16 report S3,
   now confirmed by elimination of the other two explanations rather than
   inferred from the mechanism alone).

### A.2 The corrected effect size

**The transfer-time contribution to task 16's tpot delta is not smaller
than a provisional headline — it is exactly zero.** Whatever headline
implied "distance-aware M2N pricing moved tpot by ~0.005 ms/token" (or
anything nonzero) is wrong on the mechanism, not just the magnitude: the
pricing signal this project's own predictor produces is completely flat
across scheduler variants and across the entire `LOAD_MARGIN` sweep, so it
contributes 0.000 ms — 0% — of the measured 0.005 ms/token, 0.075 ms/request
tpot movement. All of that movement is queueing, a mechanism transfer
pricing cannot represent given the discrete-event/dummy-compute setup this
study runs under (task 16 report S3's own diagnosis, now the *only*
explanation left standing rather than the leading candidate). This is a
smaller, and more precisely *zero*, corrected number than any reading that
attributes part of the tpot movement to pricing — consistent with this
project's own stated preference (twice already, tasks 12 and 14) for the
less dramatic reading turning out to be the right one.

---

## Part B — Validation boundaries

### B.0 Table of configurations tried

| Configuration | Verdict | Direction of error / reason |
|---|---|---|
| `moe_expert_parallel_size=1` (every prior task, tasks 09-16) | **Trustworthy** | Baseline; nothing here changes it |
| `moe_expert_parallel_size=2`, dummy mode | **Trustworthy** | M2N call count/size/price bit-identical to EP=1; genuinely EP-invariant, not silently wrong (S1) |
| `moe_expert_parallel_size=4`, dummy mode | **Trustworthy** | Same as above |
| `moe_expert_parallel_size=1/2/4`, real h800 profiles (Phi-tiny-MoE-instruct) | **Trustworthy for M2N; approximate for total decode time** | M2N pricing still bit-identical across EP; real decode-step wall time *does* move with EP (0.0530s/0.0506s/0.0471s) entirely through Frontier's own (non-fabric-aware) execution-time + cc_backend path — correctly modeled by Frontier, but with zero contribution from this project's topology-awareness (task 06's closed `CCBackendType`) |
| `step-moe-noquant-small` (hidden=7168), real profiles, colocated/split | **Refused** | `KeyError: "['time_stats.mlp_up_proj.median'] not in index"` — Frontier's own FFN-training pipeline unconditionally expects a dense-MLP profiling column this fully-MoE model's data doesn't have (`mlp_only_layers`/`decoder_sparse_step`/`first_k_dense_replace` all unset — every layer is routed, no dense MLP exists to profile) |
| `Llama-3.1-405B-Instruct-FP8`, real profiles (task 12) | **Refused** | Sklearn predictor training still running after >10 min wall time; abandoned rather than let it run unbounded (task 12's own finding, cited not re-run) |
| `build_node_scale`, colocated/split (every prior task) | **Trustworthy** | Baseline |
| `clos_fat_tree_fabric` (leaf-spine), oversubscription=1:1, colocated/split, dummy mode | **Trustworthy for direction, approximate for magnitude** | Split still costs far more than colocated (24.20x transfer-time ratio), same shape as task 11's `build_node_scale` result (14.65x) — different absolute ratio, same conclusion |
| `clos_fat_tree_fabric`, oversubscription=4:1 | **Trustworthy for direction, approximate for magnitude** | 26.13x — moves in the expected direction as oversubscription increases, still the same qualitative conclusion |

### B.1 Expert parallelism — does it silently mis-price?

**No — and the reason is structural, not a near-miss.** Sweeping
`moe_expert_parallel_size` across {1, 2, 4} (`tools/run_ep_pricing_probe.py`,
monkey-patch instrumentation on the real predictor class, both dummy and
real h800 compute):

```
[EP=1] num_calls=448  activation_sizes=[65536]  afd_stage_idx=[0]  mean_price_ms=0.001101
[EP=2] num_calls=448  activation_sizes=[65536]  afd_stage_idx=[0]  mean_price_ms=0.001101
[EP=4] num_calls=448  activation_sizes=[65536]  afd_stage_idx=[0]  mean_price_ms=0.001101
```

Bit-identical across all three, under dummy mode *and* real h800 profiles.
Nothing raises; nothing silently produces a different (wrong) number
either — the number produced is the *same*, and that turns out to be
correct, not accidentally lucky: activation exchange between DECODE_ATTN
and DECODE_FFN (what M2N prices) happens once per micro-batch, at the
cluster boundary, before any expert routing occurs — the full activation
payload crosses the boundary regardless of how many expert-parallel ranks
the FFN side internally uses to process it afterward. The all-to-all
dispatch *among* those EP ranks is a separate, later step, and Frontier
prices it separately: `grep` across
`frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`
finds two direct calls to `self._cc_backend.predict_all_to_all(...)`,
nothing to do with M2N at all. Confirmed this collective cost is real and
EP-sensitive by turning dummy mode off: real decode-step completion time
(`sim_end_s`) shrinks monotonically with EP — 0.05302s (EP=1) → 0.05063s
(EP=2) → 0.04707s (EP=4) — while every M2N-side number stays exactly flat.

**The real gap this reveals is not in M2N at all — it's that this project's
own fabric-aware collective costing (`EngineCCBackend`, task 06) has never
priced a single one of those `predict_all_to_all` calls, in this task or
any before it.** `CCBackendType` is closed (task 06's own finding,
`--cc_backend_config_type analytical` in every argv this project has ever
built), so every all-to-all — MoE dispatch included — has always gone
through Frontier's own flat analytical collective model. EP doesn't
silently mis-price through M2N; it silently never gets *this project's*
topology-aware treatment at all, through a completely different, older,
and already-diagnosed gap.

### B.2 Model size — practical ceiling

**Two independent obstacles, not one, and neither is training time alone.**
Task 12's `Llama-3.1-405B-Instruct-FP8` attempt (hidden=16384, 126 layers)
is cited rather than re-run (>10 min sklearn training, abandoned): that
ceiling is training cost scaling with the profiling corpus, a real,
already-established cost.

The middle attempt (`step-moe-noquant-small`, hidden=7168, 31 layers —
genuinely between llama2_7b's 4096 and 405B's 16384, and the closest
real-profiled model to "the middle" this checkout has) hits a *different*
wall entirely: training succeeds in ~70 s, then
`KeyError: "['time_stats.mlp_up_proj.median'] not in index"` inside
Frontier's own `_train_ffn_models_for_cluster`. This model's config has no
dense layers at all (`mlp_only_layers`, `decoder_sparse_step`,
`first_k_dense_replace` are all unset — every layer is MoE-routed), so its
profiling data legitimately has no `mlp_up_proj` column to train against,
but Frontier's training pipeline requires that column unconditionally
regardless of whether the model needs it. This is a genuine data/pipeline
gap in Frontier's own bundled profiling assets for this model, not
something either this project or a longer timeout can work around.

**The practical ceiling in this checkout is therefore not purely "how big
a model can train in reasonable time" — it's "which of the seven models
with real h800 profiles happen to have complete, self-consistent data for
the specific cluster/MoE combination being asked for."** Of the seven
(`llama2_7b_dense_example`, `Phi-tiny-MoE-instruct`, `Qwen3-30B-A3B-tiny`,
`qwen3-next-80b-a3b-instruct-reduced-l2`, `Step2Mini-tiny`,
`step-moe-noquant-small`, `Llama-3.1-405B-Instruct-FP8`), only two have a
hidden size clearly larger than the 7B baseline (`step-moe-noquant-small`
and the 405B model), and both are blocked — one by data completeness, one
by training time. **No larger-than-baseline, real-profiled run was
reachable in this checkout in reasonable time**, so task 12's reasoned-but-
unconfirmed hypothesis (the placement-penalty fraction should shrink with
model size) remains reasoned but unconfirmed after this task too. That is
the honest finding, not a workaround.

### B.3 Fabric shape — conclusions hold

`clos_fat_tree_fabric` (leaf-spine, `switch_radix=8`: 8 leaves, 4 spines, 4
hosts/leaf, 32 hosts total), colocated (same host) vs. split (different
leaf — two switch hops via a spine, a case `build_node_scale` cannot
produce at all, since it has only one leaf):

| oversubscription | mean M2N colocated | mean M2N split | ratio | mean tpot ratio |
|---|---|---|---|---|
| 1:1 | 0.978240 ms | 23.669760 ms | 24.1963x | 1.0036x |
| 4:1 | 0.978240 ms | 25.557120 ms | 26.1256x | 1.0039x |

Same shape as task 11's `build_node_scale` result (colocated ≪ split on
transfer time, 14.6503x there; tpot flat under dummy compute, 1.0020x
there) — split still costs far more than colocated, and increasing
oversubscription from 1:1 to 4:1 moves the ratio in the expected direction
(less uplink bandwidth, costlier split). The *magnitude* differs (~24-26x
here vs. ~14.65x on `build_node_scale`) because the paths aren't the same
kind of hop — `build_node_scale`'s "split" is one cross-domain hop over a
single shared leaf; this fabric's "split" is two switch hops through a
spine, with oversubscription applied to the uplinks specifically — a
harder case by construction, not a discrepancy to explain away. Both
fabrics agree on the conclusion that matters (placement cost is real and
network-topology-dependent); neither agrees on a universal constant, which
is exactly what "shape, not magnitude" should mean.

---

## Which limitation is worth fixing first

**`CCBackendType`'s closure (task 06), not anything found newly in this
task.** It is the one gap that touches every measurement this project has
ever made, not just M2N: every allreduce, every all-to-all (including the
EP dispatch B.1 just confirmed is real and EP-sensitive), every collective
this whole simulator has ever priced in a real run has gone through
Frontier's own flat analytical model, never this project's fabric-aware
`EngineCCBackend`, in every task from 06 through 17. The topology-aware
scheduler (task 15) and source-binding fix (task 16) both extend real
machinery this project built and controls; fixing `CCBackendType`'s
closure would make *already-built, already-tested* fabric-awareness reach
a class of communication this project has never priced correctly at all,
which is a bigger lever than any further refinement inside M2N/KV binding.
It is also, unlike the model-size and EP findings above, not something
this project can fix unilaterally from `src/integration/` — it needs an
upstream change to `frontier/types/cc_backend_type.py` or a different
public extension point, which is exactly why task 06 stopped and reported
rather than forcing it, and exactly why it is still the right thing to
flag first here.

## Anywhere this specification is wrong

- **The Part A worked example's numbers** (`9.797`, `8.585`, `6.702021`,
  `6.610810`, `13.29`, `1.36%`) do not correspond to any number this
  project has produced, in task 16 or elsewhere (S0). Task 16 never ran a
  real-compute comparison at all. Treated as illustrative scaffolding for
  the reconciliation method, not a literal target.
- **"All 179 existing tests still pass"**: the actual count after task 16
  is 177 (172 + 5 new in `tests/test_source_binding.py`), not 179. Task 17
  itself adds none (a measurement task, per its own acceptance section),
  so 177 is what should — and does — still pass.
- **"If an all-to-all is priced as a single flow, it is presumably
  understated"** (B.1's framing): the presumption doesn't survive contact
  with the evidence. M2N is *correctly* EP-invariant because the all-to-all
  it might have been expected to price is a different call path entirely
  (`predict_all_to_all`, via the cc_backend), not something M2N's own scope
  ever covered. The real gap B.1 surfaces is upstream of M2N, at the
  cc_backend layer task 06 already found closed — a more useful finding
  than "M2N understates it," but not the one the framing anticipated.
- Otherwise the specification's structure (three candidate explanations
  for Part A, model/EP/fabric axes for Part B, "report rather than fix")
  matched what the investigation actually needed.
