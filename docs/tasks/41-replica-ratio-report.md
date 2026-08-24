# Task 41 — Replica ratio in the search

Branch: `task-41-replica-ratio`, branched from `task-40-multirack`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`.

226 tests pass (218 unchanged + 8 net new in `tests/test_planner_core.py`),
and `python3 tools/check_import_direction.py` exits 0. Task 33's own
sixteen-row table and Task 36's own two-fabric result both reproduce
bit-identical, checked directly.

Read `AGENTS.md` and Task 32's own report S7 before starting, per this
task's own instruction — S7 is quoted directly below wherever it scoped
something this task then had to do.

---

## 1. How the space was kept affordable, and what that gives up

Task 32 S7 anticipated the risk correctly ("a materially larger search,
not just one more axis") but not its actual shape. The real bottleneck
this task found was not combinatorial size — it was **reachability**:
most of the space this task's own spec assumes is searchable cannot be
priced by this project's own real-compute evaluator at all, for reasons
unrelated to how many candidates there are. Both halves of the design
question — the placement-multiset question (S2's own main ask) and the
reachability question — are reported below, since the second reshaped
what the first could even be tested against.

### 1.1 The placement dimension: multisets, not a full search

`enumerate_attn_shapes` (task 32/37) already gives every distinct shape
one DECODE_ATTN replica's own TP group can reach. With `attn_replicas`
of them, a raw candidate assigns each replica its own shape — but
replicas of one pool are interchangeable (`AGENTS.md`'s own
`group_shape()` invariant), so an assignment `{A, B}` to two replicas is
the same arrangement as `{B, A}`. `enumerate_replica_arrangements`
(`tools/planner_core.py`) extends `enumerate_attn_shapes`'s own method
exactly this way: build the full multi-replica deployment, generate raw
placements from this project's own existing policies (`packed`,
`spread`, `fragmented(seed=0..59)` — the same 62-candidate generation
task 32 used), and dedupe by a **sorted tuple of each replica's own
`group_shape()`** rather than an ordered tuple. Pure `engine.placement`/
`engine.logical`; no Frontier.

This is enumeration only — it does not feed `plan()`'s own evaluated
search (S1.2 explains why not), so its cost is the 62-candidate
generation task 32 already paid, not a multiplied one. **What it gives
up**: heterogeneous cross-replica placement (one replica packed, another
deliberately spread to avoid contention with the first) is represented
and deduplicated correctly when the existing policies happen to produce
it, but nothing here *searches* for it deliberately — the raw candidates
are exactly the same 62 task 32 already generates, now applied to a
larger deployment, not a larger candidate count.

### 1.2 The reachability dimension: what this evaluator can actually price

This is the part Task 32 S7 did not anticipate, and it turned out to be
the dominant constraint. Confirmed by running real candidates — through
this project's own `SimulationEvaluator`, one subprocess per candidate,
never trusting a prediction about Frontier's own internals without
executing it:

- **`attn_replicas > 1`, at any admissible `attn_tp`, is unreachable.**
  `src/integration/cc_backend/comm_groups.py`'s own docstring states the
  reason precisely: "Frontier's cc_backend calls carry a device count
  and a parallelism-domain label — never a rank identity." Two
  DECODE_ATTN replicas at the same `attn_tp` register the identical
  `(cluster_type, comm_domain, num_devices)` key, and
  `CommGroupRegistry.register` correctly raises `CommGroupError` rather
  than guess which replica's ranks a later query means. This is a
  **documented, principled limitation**, not a bug — and every
  admissible `attn_tp` for every model this project's tasks 32-40 have
  used is > 1 (tp=1 is memory-infeasible), so `attn_replicas > 1` is
  unreachable for every model this project has ever searched with.
  `src/integration/` is human-only per `AGENTS.md`; not something this
  task can fix.
- **`ffn_replicas > 1` is reachable, and this task's own first attempt
  to check that was wrong.** An early probe (three sequential
  `Simulator.run()` calls inside one Python process, at
  `ffn_replicas ∈ {1, 2, 4}`) crashed at `ffn_replicas=2` and `4` with
  Frontier's own `"DECODE_FFN target_ffn_replica_id must be an exact
  non-negative int, got None"`. That looked like a genuine upstream bug
  in Frontier's own dp-lane routing — and was reported as one, briefly,
  in this task's own draft. It was wrong. Re-run with this project's own
  established discipline (`tools/planner.py`'s `evaluate()` spawns a
  fresh subprocess *per candidate*, deliberately, exactly to avoid
  cross-call state leakage — a discipline this task's own first probe
  violated by looping in one process) — `ffn_replicas ∈ {1, 2, 3, 4, 6,
  8, 16}`, each its own subprocess, all succeed. The earlier "crash" was
  contamination from an earlier call's own global state (most likely
  `CCBackendFactory`'s own module-level patch from `install(collective=
  True)`, or Frontier's own `global_vars`), not a property of
  `ffn_replicas` at all. Caught only because this task re-ran it
  cleanly rather than trusting the first result — recorded in S5 as
  this task's own version of the "trust but verify" catch every task
  since 34 has found somewhere.

**What this gives up, honestly**: the search this task can actually
run through the real evaluator is a **one-dimensional slice** —
`attn_replicas` fixed at 1, `ffn_replicas` free — not the full 2-D ratio
space S2's own framing describes. This happens to be exactly the axis
Task 22's own recommendation points along (more FFN capacity, not more
attention capacity), so the slice is not an arbitrary restriction; it
is the one direction this task can honestly test. `SimulationEvaluator.
can_evaluate` (`tools/planner.py`) now reflects this precisely:
`attn_tp in profiled_tp and attn_replicas == 1` — `ffn_replicas` is not
gated at all, since it is not actually restricted.

A second, partial-fidelity path for `attn_replicas > 1` exists and was
used only to *corroborate* the above, not as part of the real search:
skip the colliding per-TP-group registration and pass `collective=
False` to `install()` (Frontier's own placement-blind stock TP cost,
losing this project's own topology-aware ATTN-internal cost, while M2N
activation-transfer costing — `context.py`'s own predictor wiring,
unconditional on `collective` — stays fully placement-aware). Reported
in S3.2 as a secondary, explicitly-caveated data point, not folded into
`plan()` or `SimulationEvaluator` — building a full execution path for a
route this project's own zoning (`src/integration/` is human-only)
would never let become the primary one is exactly the kind of
speculative machinery this project's own conventions discourage.

## 2. The collapse ratio

Computed on Task 32's own fabric (`task32repro`: 5 domains × 4 GPUs =
20 GPUs), the same 62 raw candidates (`packed` + `spread` +
`fragmented(seed=0..59)`) task 32 used, now applied to the full
multi-replica deployment:

| `attn_tp` | single-replica shapes `S` | `attn_replicas` | distinct arrangements | collapse ratio |
|---|---|---|---|---|
| 2 | 2 | 1 | 2 | 31.0x |
| 2 | 2 | 2 | 3 | 20.7x |
| 2 | 2 | 3 | 4 | 15.5x |
| 4 | 5 | 1 | **4** | 15.5x |
| 4 | 5 | 2 | 9 | 6.9x |
| 4 | 5 | 3 | 10 | 6.2x |
| 8 | 9 | 1 | 9 | 6.9x |
| 8 | 9 | 2 | 19 | 3.3x |
| 8 | 9 | 3 | — | **stops here: 26 GPUs needed > 20 available** |

**Two different things stop this from scaling, not one.** `attn_tp=8,
attn_replicas=3` is a hard capacity wall (needs 8×3 + 1(prefill) +
1(ffn) = 26 GPUs on a 20-GPU fabric) — the same kind of stop task 32's
own S2 hit for `tp=8` shapes directly. The other rows show a *softer*
stop: the collapse ratio itself shrinks as either `attn_tp` or
`attn_replicas` grows, because the reachable-arrangement space grows
while the fixed 62-candidate sample does not, so a fixed sample covers
proportionally less of it — at `attn_tp=8, attn_replicas=2`, the
theoretical multiset count from 9 single-replica shapes is
C(9+2-1,2) = 45, and only 19 of those were actually reached by 62 raw
placements. More `fragmented` seeds would close this gap at no
Frontier cost (this enumeration needs none) but were not pushed further
here, since — S1.2 — nothing beyond `attn_replicas=1` can be evaluated
for real regardless of how completely its own arrangement space is
enumerated.

**One further, honestly-reported gap**: the `attn_tp=4, attn_replicas=1`
row shows 4 distinct arrangements, not the 5 `enumerate_attn_shapes`
itself finds for a single replica (task 32's own table: `(4,), (3,1),
(2,2), (2,1,1), (1,1,1,1)`). The missing one is `(4,)` — the single-
domain packed shape. `enumerate_replica_arrangements` does not include
task 32's own extra "packed-if-it-fits" reference candidate (added
there for exactly this reason: `packed()`'s own rank ordering gives
DECODE_ATTN's group a one-slot offset from PREFILL, so it does not
reach a clean single-domain shape on its own even when it fits). This
is a real undercount at `attn_replicas=1`, inherited by not
special-casing that reference candidate here, not a new mechanism —
noted rather than silently accepted.

## 3. The best ratio found, with its margin and interval

Same model and workload as Task 32 (`Phi-tiny-MoE-instruct`, 32
requests, qps=20, prefill=32/decode=16 tokens), same fabric
(`task32repro`), `attn_tp=2` shape `(2,)` — Task 32/33's own established
winning degree and shape, held fixed so this is a ratio comparison, not
a re-run of the degree search. `attn_replicas=1` throughout (S1.2); the
free dimension is `ffn_replicas`.

### 3.1 The reachable slice: `ffn_replicas` free, `attn_replicas=1`

Deterministic pass, each point its own subprocess:

| `ffn_replicas` | mean tpot (ms) | throughput (req/s) | Δ vs. previous |
|---|---|---|---|
| 1 | 11.6803 | 107.171 | — |
| **2** | **7.6972** | **185.712** | **−34.1%** |
| 3 | 6.3220 | 200.272 | −17.9% |
| 4 | 5.3649 | 323.478 | −15.1% |
| 6 | 5.0187 | 341.379 | −6.5% |
| 8 | 4.6770 | 361.208 | −6.8% |
| 16 | 4.1067 | 399.825 | −12.2% |

Confirmed reproducing through `plan()`'s own real interface, not only
the standalone probe: `plan(topology, model, workload, hardware,
objectives, replica_ratios=((1,1),(1,2)))`, restricted to `attn_tp=2`,
returns `mean_tpot_ms=11.6803` at `(1,1)` and `7.6972` at `(1,2)` —
bit-identical to the table above, and `result.unknown == []` (`
ffn_replicas=2` is priced, not gated) — the extended dimension works
through the tool's own public path, not only a bespoke script.

**Monotonic throughout the tested range, with no interior optimum —
under this one particular workload regime.** `Objectives`
(`tools/planner_core.py`) minimizes `mean_tpot_ms` alone — no GPU-count
or cost term — so more FFN capacity never *stops* helping in this
range; it only helps less. Under the deterministic pass alone, the best
point in the range tested is `ffn_replicas=16`, beating `1:1` by
**+64.8%**, and the single largest step is `1:1 → 1:2` at **+34.1%**.

**S3.3 shows this margin does not survive real arrival variance, and
reverses.** Task 31's own report draws a sharp line between the
deterministic pass (32 requests submitted as one simultaneous burst)
and a seeded, streaming-Poisson pass (the same 32 requests, staggered
arrivals) — "a genuinely different workload regime, not the same one
with error bars." The deterministic pass's own large margin is a
burst-arrival artifact: when every request lands at once, extra FFN
replicas immediately parallelize the flood, and the win is real *for
that regime*. Under realistic streaming arrivals, FFN is never
simultaneously overloaded enough for that parallelism to pay off, and
S3.3's own seeded numbers show `ffn_replicas` **increasing** mean tpot,
monotonically, not decreasing it.

### 3.2 The unreachable slice, partial-fidelity corroboration only

Via the `collective=False` + skip-colliding-registration path (S1.2),
calibrated against a same-method `(1,1)` baseline rather than the
normal one, since losing ATTN's own TP-cost placement-awareness shifts
the baseline slightly by itself:

| arrangement | method | mean tpot (ms) | Δ vs. same-method baseline |
|---|---|---|---|
| `(1,1)` | normal (`collective=True`) | 11.6803 | — |
| `(1,1)` | fallback (`collective=False`) | 11.6540 | −0.2% (calibration only) |
| `(2,1)` | fallback (`collective=False`) | 10.2620 | **−11.9%** |

Over-provisioning **attention** instead of FFN helps only modestly
(−11.9%, against FFN's −34.1% for the same one-extra-replica move) —
consistent with, not contradictory to, Task 22's own finding that
attention was never the busier pool. Reported as corroborating context
only; not part of the real search (S1.2), and not seeded (the fallback
path is a one-off calibration, not a result this report asks anyone to
trust to the same standard as S3.1's own real, reachable slice).

### 3.3 Margin against the noise floor (Task 31) — and the real winner

20-seed re-run, `ffn_replicas ∈ {1, 2, 4}`, `attn_replicas=1`, same
configuration as S3.1, task 31's own `seed_stats.run_seed_study`, one
subprocess per (ratio, seed) pair (task 31/32's own established
double-check discipline — never call a deterministic margin real
without this step):

| `ffn_replicas` | seeded mean tpot (ms) | 95% CI half-width | vs. `ffn_replicas=1` |
|---|---|---|---|
| **1** | **3.2378** | ±1.56% | — |
| 2 | 3.3271 | ±0.70% | **+2.76% (worse)** |
| 4 | 3.4706 | ±0.28% | **+7.19% (worse)** |

`ffn_replicas=1`'s own seeded figure (3.2378ms) is bit-identical to
Task 32's own seeded winner for `tp=2, (2,)` — confirming this task's
own seeded methodology reproduces the established one exactly, not a
different convention by coincidence.

**The three 95% CIs do not overlap, and the ordering is monotonic in
the opposite direction from the deterministic pass**: `[3.187, 3.288]`
(fr=1), `[3.304, 3.350]` (fr=2), `[3.461, 3.480]` (fr=4). Both margins
clear task 31's own ≈1.3% flat-region noise floor (+2.76% ≈ 2.1x it;
+7.19% ≈ 5.5x it) — this is a real, measured effect, not noise, and it
says the opposite of the deterministic pass: **under this workload's
own realistic arrival process, adding FFN replicas makes mean tpot
*worse*, not better.** `ffn_replicas=1` — the pre-existing default every
task since 32 already used — is the real winner once arrivals are
staggered rather than bursted.

**This is the central finding of this task's own real-compute study,
and it is the outcome this task's own §4 explicitly asks to be reported
plainly rather than reworked**: *"If the winner is 1:1, that contradicts
Task 22 and matters more than a positive result."* It does, and here is
why, established directly rather than guessed at: a streaming arrival
process staggers requests, so DECODE_FFN's own batch scheduler
(`orca`, distinct from DECODE_ATTN's `vllm_v1`) rarely receives enough
simultaneously-queued work for a second replica to have anything to
parallelize — extra replicas mostly sit idle, and the *added* per-hop
M2N activation-routing cost across more targets (this project's own
subject since task 09) shows up in the per-token latency objective
without a compensating throughput gain. The deterministic burst
(all 32 requests submitted at once) is exactly the one workload shape
where that idle-capacity cost doesn't apply, because there genuinely is
a pile of simultaneous work to split — which is also why the
deterministic margin was so large. Neither pass is wrong; they measure
different, real regimes, and only one of them (streaming) is the one
Task 22's own workload assumption (arrivals, not a single burst)
actually matches.

## 4. Whether it agrees with Task 22

**No — and the disagreement is itself the corroborating detail, once
the workload regime is accounted for.** Task 22's own S3 measured
*busy time*, at 32 requests submitted as one Poisson-arrival stream —
the same streaming regime S3.3 uses, not the deterministic burst S3.1
uses. Read that way, this task's own S3.3 and Task 22's own S3 are
comparable, and they disagree: Task 22 found FFN consistently busier
than ATTN at every ratio tested and recommended more FFN capacity
("two FFN replicas per attention replica" brought the pools within a
few percent); this task's own streaming-regime search finds that move
makes per-token latency *worse*, not better, by a margin well clear of
noise.

**What resolves the disagreement, not just restates it: busy time and
latency are different objectives, exactly this task's own known trap
warns about, and here they genuinely diverge rather than agreeing by
default.** A second FFN replica can absorb more *aggregate* work
without either replica running constantly saturated (lower per-replica
utilisation, Task 22's own metric) while simultaneously adding
per-token latency for any individual request, if the extra replica
mostly sits available rather than busy under staggered arrivals (this
task's own S3.3) and the added M2N routing/coordination cost across
more targets is paid by every request regardless. Both measurements are
almost certainly correct for what each one measures; they simply are
not measuring the same thing, and nothing about a placement-aware
latency search obligates it to agree with a utilisation-balance study
answering a different question. Reported exactly as this task's own §4
anticipated for this specific outcome — plainly, not reworked into a
positive result, and with the mechanism, not just the direction, spelled
out.

**The `attn_replicas`-side corroboration (S3.2) still holds and is now
the more informative of the two secondary results**: growing attention
capacity (the pool Task 22 found *not* starved) helped modestly
(−11.9%, deterministic-only, not seeded) — smaller than FFN's own
deterministic-pass number, but in the *same direction* as attention
being the less-loaded pool. It was not re-run seeded (S1.2's own scope
limit on the fallback path), so it cannot be held to the same standard
as S3.3's own result, and is reported as context only, not as a second
confirmed finding.

## 5. Anywhere this specification is wrong

**One inaccuracy in this task's own opening citation to Task 22,
mechanical rather than substantive.** This task's own S1 describes Task
22's finding as "attention busy about a fifth of the time, FFN better
than two thirds, an imbalance over three to one" **at parity** (the 1:1
ratio). Checked directly against `docs/tasks/22-which-binds-report.md`'s
own S3 table: at 1:1, attention is 18.1% busy ("about a fifth" is
accurate) and FFN is 42.8% busy — not "better than two thirds," and the
actual ratio is 42.8/18.1 ≈ 2.4:1, not "over three to one." The figures
this task's own S1 describes (busy "over two-thirds," imbalance "over
three to one") instead match Task 22's own **2:1 (attn:ffn)** row
(17.4%/82.5%, a 4.7:1 imbalance) — a real ratio in Task 22's own table,
just not the 1:1 row this task's own text attaches it to. This does not
change anything this task built (the reachable, tested direction —
`ffn_replicas` free — is unaffected either way), but it is a citation
error worth naming plainly, the same discipline Task 22's own S0 applied
to a bad citation it found in its own spec.

**The central, load-bearing finding is a reachability problem the
specification did not anticipate, not a combinatorial one.** S2's own
framing ("the space grows fast, and this is the main design question")
correctly predicted that placement multiplies with replica count, and
the multiset treatment (S1.1/S2) answers that question as asked. But
the dominant constraint this task actually hit was that most of the
*ratio* dimension itself (`attn_replicas > 1`, at every admissible
degree) cannot be priced by this project's own real evaluator at all —
a `src/integration/`-level limitation (documented, principled, and
human-only to fix per `AGENTS.md`) that no amount of clever candidate
generation works around. `ffn_replicas > 1`, the direction that
actually matters for Task 22's own recommendation, turned out to be
fully reachable — but only after this task's own first attempt (looping
several evaluations in one process) produced a false crash, caught by
re-running cleanly rather than trusting it. Both of these — the real
`attn_replicas` limitation and the false `ffn_replicas` alarm — are
worth reporting exactly this bluntly: the harder problem here was
knowing what could be tested at all, not managing how much of it to
test.

**The lane-assignment "known trap" (S6) does not actually bind under
this project's own existing policy, and that is worth saying plainly
rather than leaving implicit.** `default_attn_dp_size_policy` (this
task's own name for `tools/planner.py`'s pre-existing `_argv` convention,
`attn_dp_size := max(ffn_replicas, 1)`) satisfies
`lane_assignment_feasible` unconditionally for every `attn_replicas ≥
1` — so the `Inadmissible` classification this task added (correcting
a pre-existing `Rejection` misclassification, task 39's own sense) is
now correctly wired, and directly tested (`attn_dp_size_policy=lambda
ar, fr: 1` forces a real violation in `tests/test_planner_core.py`),
but it does not actually fire anywhere in this task's own real search.
Not a defect — the fixed policy was *designed* to satisfy the
constraint always — but the "respect the lane constraint" instruction
reads as though it expects to matter more than it currently can, given
that policy.

## What shipped

- `tools/planner_core.py` — `enumerate_replica_arrangements` (S1.1);
  `default_attn_dp_size_policy`, named and overridable (S1.2, task 32
  S7); `plan()`'s lane-assignment check reclassified from `Rejection`
  to `Inadmissible`, with `attn_dp_size_policy` now an explicit
  parameter rather than an inlined assumption; `Inadmissible`'s own
  docstring extended to cover both causes.
- `tools/planner.py` — `SimulationEvaluator.can_evaluate` now correctly
  restricts `attn_replicas > 1` (confirmed unreachable) and does *not*
  restrict `ffn_replicas` (confirmed reachable, after this task's own
  false-alarm correction).
- `tests/test_planner_core.py` — 8 new tests: `default_attn_dp_size_policy`
  matches `_argv`'s own convention; `lane_assignment_feasible`'s own
  three-case truth table; a lane violation surfaces as `Inadmissible`,
  never reaching the evaluator; a `replica_ratios=((1,1),)` search
  matches the unextended default exactly (this task's own required
  acceptance test); adding more ratios leaves the `(1,1)` candidates'
  own results untouched; `enumerate_replica_arrangements` at
  `attn_replicas=1` agrees with `enumerate_attn_shapes`; it collapses
  permutations to at most the theoretical multiset count; and it treats
  `{A,B}`/`{B,A}` as one arrangement, the literal case this task's own
  spec names.
- `docs/tasks/41-replica-ratio-report.md`, this report.

One commit on `task-41-replica-ratio`, stacked on `task-40-multirack`.
Task 33's sixteen-row table and Task 36's two-fabric result both
reproduce bit-identical.
