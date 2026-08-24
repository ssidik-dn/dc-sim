# Task 36 — Does the planner's answer depend on the fabric?

Branch: `task-36-fabric-dependence`, branched from `task-35-model-sizing`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`. 189 tests pass, unchanged, and
`python3 tools/check_import_direction.py` exits 0.

**Yes — the planner's answer depends on the fabric it is given, for a
real model at an ordinary operating point.** Same model
(`Llama-3.1-405B-Instruct-FP8`), same workload, same objective; only the
fabric's own domain size changed. The winning arrangement's own mean
per-token latency differs by **+36.9%** deterministically and **+41.4%**
under seeded arrival noise, with 95% confidence intervals that do not
overlap at all — this is what Task 33 could not show.

---

## 0. What had to change in `tools/planner.py` to make this runnable

`feasible_num_blocks` was a lookup table calibrated for
Phi-tiny-MoE-instruct on h800 only, raising `NotImplementedError` for
anything else. Generalized to compute DECODE_ATTN's own parameter and
KV-cache memory directly from Frontier's own formula
(`param_counter.py`/`memory_planner.py`), verified bit-for-bit against
`ParamCounter.get_num_parameters_per_device()` run for real, for **both**
Phi-tiny-MoE-instruct and `Llama-3.1-405B-Instruct-FP8`, before trusting
it for anything (§7 has the one mistake this caught). `ModelSpec` gained
`hidden_size`/`num_attention_heads`/`num_key_value_heads`/`num_layers`/
`head_dim` accordingly, threaded through `evaluate()`'s subprocess CLI.
Two new named topologies (`domain8_40gpu`, `domain4_40gpu`, §1) were
added. No other file changed.

**Regression check, per this task's own §5 requirement, done before
anything else**: Task 32/33's own result reproduces bit-identical
through the generalized `feasible_num_blocks` — same 16-row ranked
table, same winner (tp=2, `(2,)`, 11.6803 ms). One real mistake was
caught in the process: my first attempt at this check omitted
Phi-tiny-MoE-instruct's own explicit `head_dim=128` override (its JSON
declares one; most models don't and fall back to
`hidden_size // num_attention_heads`), which silently used 256 instead
and produced a wrong ranking (tp=4 winning, tp=2 wrongly marked
infeasible) until caught by comparing against the known-correct table.

---

## 1. The premise check

**Margin: 0.7**, the middle of Task 35's own 0.6-0.79 band, chosen
because it is the point where `attn_tp` ∈ {1, 2, 4} are all infeasible
and 8 is comfortably feasible with room on both sides (not a boundary
value itself). Confirmed directly, not assumed, via the same
`feasible_num_blocks` used inside `plan()`:

| attn_tp | feasible at margin=0.7 (h800, 80GB)? | num_blocks |
|---|---|---|
| 1 | no | — |
| 2 | no | — |
| 4 | no | — |
| **8** | **yes** | **7,558** |

**Equal total GPUs, per this task's own known trap**: both fabrics are
`build_node_scale`, 40 GPUs total.

| fabric | machines x GPUs | total GPUs | domain size | tp=8 shapes reachable | single-domain `(8,)` reachable? |
|---|---|---|---|---|---|
| A: `domain8_40gpu` | 5 x 8 | 40 | 8 | `(8,)`, `(7,1)`, and 9 split shapes | **yes** |
| B: `domain4_40gpu` | 10 x 4 | 40 | 4 | `(4,3,1)` and 9 other split shapes | **no** — structurally impossible (8 > 4) |

Both premises hold exactly as Task 35 predicted: `attn_tp=8` is memory-
forced on both fabrics (feasibility doesn't depend on domain size at
all — only placement does), and only Fabric A can place it inside one
domain. The experiment tests what it claims to.

---

## 2. The winner on each fabric

**Objectives were not carried over from Task 33** (this task's own §2
instruction) — this model and workload were never calibrated against
Task 33's throughput/SLO floors, so `plan()` ran fully unconstrained
(`min_throughput_rps=0`, `slo_attainment_floor=0`) and the full ranking
is reported.

**Fabric A (`domain8_40gpu`), deterministic:**

| rank | shape | mean tpot (ms) | throughput | SLO |
|---|---|---|---|---|
| **1** | **`(8,)`** | **326.2362** | 3.798 | 0.000 |
| 2 | `(3,2,1,1,1)` | 435.3951 | 2.897 | 0.000 |
| 3-10 | every other split shape | 446.5146 | 2.829 | 0.000 |

**Fabric B (`domain4_40gpu`), deterministic:**

| rank | shape | mean tpot (ms) | throughput | SLO |
|---|---|---|---|---|
| **1** | **`(4,3,1)`** (tied with every other reachable shape) | **446.5146** | 2.829 | 0.000 |

**Every split shape on either fabric lands at the same 446.5146 ms**
(bar Fabric A's own `(3,2,1,1,1)`, marginally better at 435.4) — at this
model's scale, the communication cost of *any* cross-domain arrangement
appears to be dominated by a shared "at least one cross-domain hop"
term, not by exactly how the group is partitioned. Fabric A's own
winner is the single-domain shape; Fabric B cannot reach it, so its
winner is whichever split shape ties for cheapest — the same number
Fabric A's own *non-winning* split candidates get. SLO attainment is
0.000 everywhere at this model's real per-token latency against Task
32/33's own 15ms illustrative target — expected and not itself a
finding; that target was calibrated against a 4B model, not a 405B one,
and is not reused as a constraint here for exactly that reason.

---

## 3. The margin, deterministic and seeded

**Deterministic**: (446.5146 − 326.2362) / 326.2362 = **+36.85%**
(Fabric B slower).

**Seeded, n=20, Task 31's own method** (genuine arrival-timing
randomness, not the deterministic all-arrivals-at-once configuration —
a different workload regime, per Task 31/32's own established
distinction, so the absolute numbers move but the comparison is still
apples-to-apples within each fabric):

| | mean tpot (ms) | 95% CI half-width | interval |
|---|---|---|---|
| Fabric A winner (`(8,)`) | 291.4884 | ±2.1146 (0.73%) | [289.37, 293.60] |
| Fabric B winner (`(4,3,1)`) | 412.2244 | ±3.2545 (0.79%) | [408.97, 415.48] |

**Seeded margin: +41.42%. The intervals do not overlap at all** — a
gap of over 115 ms between the nearest interval edges, against a noise
floor under 1% at this configuration. That puts the margin at roughly
**54x** the noise floor — decisively past the point Task 31 established
this project can no longer tell two arrangements apart, and far
past even Task 32's own headline margin (24-29x). This is exactly what
this task's own §4 anticipated: **a memory-forced difference should be
large, because the group is compelled onto a narrower path, not merely
statistically preferred one** — and it is.

---

## 4. The fabrics separated — stated as the smaller claim

**The planner's answer depends on the fabric, for this real,
already-available model, at an ordinary memory margin.** Stated at
exactly the size this claim is, no larger, per Task 35's own assessment
(quoted in this task's own spec and unchanged by anything found here):

This is **memory-forced**, not evidence that a split arrangement is
*never* competitive with a whole one, nor a claim that "topology
matters" in general. Fabric B's own `attn_tp=8` group is compelled to
split because no smaller degree is feasible at this margin — there is
no available *whole* placement on Fabric B to compare against, only the
forced-split one. Whether a split arrangement could ever *win* against
a whole one on latency (the compute-forced question Task 35 also named)
is not addressed here and needs profiles at TP degrees this project
does not have (Task 35 §1.3, unchanged).

---

## 5. What the 405B model costs to evaluate

**Cold (first invocation, sklearn/RandomForest predictor training from
scratch): 484.4 seconds.** Matches Task 12's own report almost exactly
in mechanism — 6 CPU-bound joblib/loky worker processes at ~95% each,
confirmed by watching the process table directly during the run, not
inferred. Task 12's own attempt was killed at the 10-minute mark without
finishing; this one was given no such limit and completed at 8.1
minutes — so Task 12's own abandonment was roughly 20% away from
completion, not orders of magnitude off.

**Warm (predictor cached to disk, `frontier/execution_time_predictor/shared_prediction_model_manager.py`'s
own `{model_name}_{model_hash}.pkl` cache, confirmed shared across
separate subprocess invocations sharing Frontier's own `cache/`
directory): ~44.7 seconds/candidate average** (21 evaluations across
both fabrics' own full shape sweep, 939.5s total). This is roughly
6-10x the per-candidate cost this project's other real-compute tools pay
for Phi-tiny-MoE-instruct (a few seconds each, cache already warm from
every prior task) — consistent with Task 12's own expectation that a
model with ~4x Phi-tiny-MoE's hidden size and layer count carries a
proportionally larger execution-time cost per call, not just a larger
one-time training cost.

**What this bounds**: a full sweep of this task's own scope (2 fabrics
x ~10 shapes x 20 seeds, the way Task 31/32 checked a margin) would be
~2 x 10 x 20 x 44.7s ≈ 5 hours if every point needed seeding — this
task avoided that by seeding only the two winners, not the whole grid,
which is exactly the scope discipline Task 32's own §1 already
established for a cheaper model and is now known to matter more, not
less, here.

---

## 6. Anywhere this specification is wrong

Nothing in the specification's own claims required correction — Task
35's own figures (133.875 GB at tp=1, the 0.58-0.79 margin band, the
tp=8 boundary) all reproduced exactly. One thing surfaced during
execution that the spec did not anticipate and is worth recording:

**The auto-derive path and the explicit-`num_blocks` formula disagree
by roughly 14x, and this is expected, not a bug.** A direct check
(leaving `num_blocks` at Frontier's own default so it auto-derives, per
Task 23's own `_coupling_check` method) gave `derived_num_blocks=106,596`
at `attn_tp=8`/margin=0.7 — the explicit formula this task's own
`feasible_num_blocks` computes gives 7,558 for the same point. This
matches Task 25's own prior finding exactly: Frontier's auto-derive
path can resolve to `num_blocks_mode="memory_planner_profiled"` (a
runtime-measured weights-memory estimate) rather than the pure
analytical `memory_planner` formula this project's own tools have
always used for their explicit values. Every real-compute tool since
Task 22 sidesteps this by computing and passing `num_blocks` explicitly
rather than relying on auto-derive — exactly what `feasible_num_blocks`
does — so this ambiguity never reaches the actual experiment; it only
surfaced because this task checked the auto-derive path directly as a
sanity check, and it is recorded here so the next task that finds a
similar mismatch doesn't have to rediscover Task 25's own finding from
scratch.

## What shipped

- `tools/planner.py` — `feasible_num_blocks` generalized from a
  Phi-tiny-MoE-instruct-only lookup table to Frontier's own real
  formula; `ModelSpec` gained `hidden_size`/`num_attention_heads`/
  `num_key_value_heads`/`num_layers`/`head_dim`, threaded through the
  subprocess CLI; two new topologies (`domain8_40gpu`, `domain4_40gpu`,
  equal 40-GPU total).
- `docs/tasks/36-fabric-dependence-report.md`, this report.

One commit on `task-36-fabric-dependence`, branched from
`task-35-model-sizing`'s tip. Task 32/33's own result reproduces
bit-identical through the generalized code (§0).
