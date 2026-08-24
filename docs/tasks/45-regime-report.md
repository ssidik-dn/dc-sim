# Task 45 — Make the arrival regime part of the search

Branch: `task-45-regime`, branched from `task-44-ep-placement`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`.

240 tests pass (233 unchanged + 7 net new), and `python3
tools/check_import_direction.py` exits 0. Task 33's own sixteen-row
table and Task 36's own two-fabric result both reproduce bit-identical
**when the burst regime is requested explicitly** — the point of this
task, not an exception to it.

---

## 1. Part A: can the cheap regime be trusted to rank?

Task 33's own sixteen-row table (`attn_tp ∈ {2,4,8}` × placement
`attn_shape`, `task32repro` fabric, Phi-tiny-MoE-instruct, 32 requests
at qps=20, `memory_margin_fraction=0.992`) was re-evaluated candidate
by candidate under a realistic streaming-Poisson arrival process
(`seed_stats.seed_argv_fix`), N=6 seeds each, via the same
`evaluate()` path `plan()` itself uses.

**Result: Spearman rank correlation = 1.0 (exact, ties handled by
average rank), and the streaming winner sat at burst rank 1 of 16.**
Every burst tie-group (the three `attn_tp=4` shapes at 27.2465ms, the
two `attn_tp=8` shapes at 42.9118ms, the seven at 45.1854ms) reproduces
as the identical tie-group under streaming, in the identical order:

| burst rank | key | burst mean (ms) | streaming mean (ms), N=6 | 95% CI half-width |
|---|---|---|---|---|
| 1 | `tp2_shape2` | 11.6803 | **3.2235** | ±0.1437 |
| 2 | `tp4_shape4` | 14.4305 | 4.4340 | ±0.1590 |
| 3 | `tp2_shape1-1` | 18.3178 | 6.1644 | ±0.1894 |
| 4 | `tp4_shape2-1-1` | 24.9729 | 9.2281 | ±0.1797 |
| 5–7 | `tp4_shape{3-1,1111,2-2}` | 27.2465 | 10.2090 (×3) | ±0.1813 |
| 8–9 | `tp8_shape{3-2-1-1-1,3-2-2-1}` | 42.9118 | 17.2976 (×2) | ±0.2644 |
| 10–16 | seven `tp8` shapes | 45.1854 | 18.3337 (×7) | ±0.3912 |

Rank 1's own streaming interval `[3.080, 3.367]` does not overlap rank
2's `[4.275, 4.593]` — the winner is not just first, it is clearly,
measurably first. **On this table, retaining the single top burst
candidate would already have kept the streaming winner.**

### This is one data point, on one kind of axis — and Part A itself found no reversal

Task 33's table varies only `attn_tp` (compute-parallelism degree) and
its own placement `attn_shape` — nothing about capacity, replica
count, or expert-parallel degree. Task 42's own report already found
this exact kind of conclusion — "tp=2 beats tp=1," "TP-split costs
~88%" — **holds** under streaming (and grows, in the split-cost case:
88.3% → 131.6%). Part A's own rho=1.0 is a second, direct confirmation
of that same pattern, not a new one: **placement/compute-parallelism
conclusions correlate perfectly across regimes here; this task did not
find a case where they do not.**

This matters for how the required acceptance test below is framed.
The spec asks for "a test that the regime input changes the ranking on
a case where Part A showed it does" — but taken literally, **Part A
(Task 33's table) showed no such case**: it found perfect correlation,
not a reversal. The real, already-established reversal this task
relies on is Task 44's own EP-degree finding (burst: ep=1 last of
three; streaming: ep=1 first) and Task 41's own replica-ratio finding
(burst: more FFN replicas wins by 34%; streaming: more FFN replicas
loses by up to 7%) — both on the **sizing** axis, neither part of
Task 33's table. The acceptance test (§4 below) mirrors that
already-verified sizing-axis reversal, not anything Part A's own
restricted measurement produced. This is worth stating plainly rather
than letting the test's own docstring imply Part A found what it did
not.

**Scope limit, stated once rather than implied**: this is a rank
correlation on one search, over one axis (TP degree + its own
placement shape), on one fabric, one model, one workload. It says
placement/degree search is burst-safe *on this axis*; it says nothing
about search in general, and the rest of this report treats it that
way.

## 2. Part B: the design, and why Part A justifies exactly this much of it

Part A validates a shortlist on the **placement** axis (`attn_shape`,
and by the identical construction `ep_shape` — Task 44's own
placement-within-a-degree question). It says nothing about the
**sizing** axis (`ffn_ep`'s own degree, `attn_replicas`/`ffn_replicas`)
— and Task 41/44 already measured that axis reversing outright. A
single two-stage design that shortlists everything, on the strength of
one placement-axis measurement, would be exactly the mistake the
spec's own known trap warns against ("do not assume the shortlist
works").

**The design actually shipped is a two-stage search scoped to that
distinction**:

- **Stage 1** (cheap, conventionally burst) runs the *unmodified*
  `plan()` search across the full space, but with `objectives`'s own
  `min_throughput_rps`/`slo_attainment_floor` relaxed to zero first.
  Those two floors are exactly as regime-dependent as the ranking
  itself (Task 42's own S1: a burst manufactures a queueing backlog a
  real stream does not) — enforcing them at the cheap stage risks
  discarding the eventual streaming winner on the *constraint* axis
  exactly as Part A worried about on the *ranking* axis. Only
  regime-independent infeasibility (memory, divisibility, lane
  assignment) is filtered at stage 1.
- Stage 1's survivors are grouped by **sizing identity**
  (`attn_tp, ffn_ep, attn_replicas, ffn_replicas`) and each group's own
  best-`shortlist_size` **placements** (`attn_shape`, `ep_shape`) are
  kept. Nothing about `ffn_ep` or the replica counts is shortlisted —
  every sizing combination `plan()` would have generated survives into
  stage 2, unfiltered.
- **Stage 2** (expensive, streaming with `num_seeds > 1`) re-evaluates
  every shortlisted candidate for real, applies the real `objectives`
  for the first time, ranks, and marks indistinguishability from the
  winner by 95% CI overlap.

`plan_two_stage()` (`tools/planner_core.py`), with `tools/planner.py`'s
own convenience wrapper defaulting both evaluators to
`SimulationEvaluator`. `TwoStagePlanResult.ranked`/`.winner` are always
stage 2's measured result, never stage 1's filtered guess —
`.shortlisted` and `.stage1` are kept alongside so a caller can see
what was filtered, from what, and under which regime, per the spec's
own "state clearly which stage produced the ranking."

**Verified against the real EP-degree study** (§3.2 below):
`shortlist_size=1` reproduces the *exact* winner and the *exact*
streaming numbers a full, unshortlisted 48-evaluation seeded search
gives (`ep=1`, `3.4959ms`, CI `±0.1469`) — because Part A's own
placement-axis guarantee held: the burst-best placement for each `ep`
(the fully-packed shape) was already the streaming-best placement too,
so shortlisting it away lost nothing.

## 3. Regime and seed count as explicit inputs

`Regime(seeded: bool, num_seeds: int)` (`tools/planner_core.py`).
`plan()` (both `planner_core.plan()` and `tools/planner.py`'s wrapper)
takes `regime` as a required positional parameter, with **no default**
— every pre-Task-45 call site meant burst and had no way to say so;
reproducing any of them now means passing
`Regime(seeded=False, num_seeds=1)` explicitly. `Regime.__post_init__`
rejects `num_seeds > 1` with `seeded=False` outright: repeating a
deterministic `t=0` burst `N` times gives `N` identical numbers, a
zero-width interval that measures nothing.

`SimulationEvaluator` binds a `Regime` at construction (the same way
it already binds topology/model/workload/hardware) and, when
`num_seeds > 1`, averages `mean_tpot_ms`/`throughput_rps`/
`slo_attainment` over `num_seeds` independent seeded runs, reporting
the 95% CI half-width (`seed_stats.compute_interval_stats`) on the mean
as `ci95_halfwidth`. `plan()`'s own `_mark_indistinguishable_from_winner`
then flags every candidate whose interval overlaps the winner's as
`indistinguishable_from_winner`, rather than reporting a strict order
the measurement does not support — and is a no-op (nobody flagged) for
`num_seeds=1`, since there is no interval to overlap, which is what
keeps every burst result exactly as strictly ordered as it always was.

### The resolution N=6 buys, concretely

Two real studies, both re-run through the new `plan()` API directly
(not a bespoke script), show exactly what N=6 can and cannot separate:

**Replica ratio** (`task32repro`, `attn_tp=2` fixed, `margin=0.992`,
`replica_ratios=((1,1),(1,2),(1,4))`, streaming N=6):

| ratio | mean tpot (ms) | 95% CI half-width | indistinguishable from winner |
|---|---|---|---|
| **(1,1)** | **3.2235** | ±0.1437 | — (winner) |
| (1,2) | 3.3247 | ±0.0620 | **yes** |
| (1,4) | 3.4799 | ±0.0318 | no |

Task 41's own N=20 seeded study found `(1,2)` **2.76% worse** than
`(1,1)` — real, but small enough that N=6's own resolution here (±4.5%
combined half-width against a 3.1% gap) cannot separate it from the
winner. `(1,4)`'s larger ~7.9% gap clears N=6 easily. **One of three
candidates is left indistinguishable at this seed count.**

**Expert-parallel degree** (`domain8`, `attn_tp=1`, `margin=0.2`,
`ep_values=(1,2,4)`, streaming N=6, full joint search — every
placement, not just the packed ones):

| ep (packed placement) | mean tpot (ms) | 95% CI half-width | indistinguishable from winner |
|---|---|---|---|
| **ep=1** | **3.4959** | ±0.1469 | — (winner) |
| ep=2 | 3.6790 | ±0.1783 | **yes** |
| ep=4 | 3.7654 | ±0.0950 | no |

Again, **one of three candidates (`ep=2`) is left indistinguishable**
at N=6; `ep=4`'s own gap (7.71% — see §5 below) clears it.

Neither study needed more seeds to answer this task's own central
question (which degree wins, and by how much); both needed exactly
this much seeding to know *which adjacent comparison* it cannot yet
resolve, and reporting that plainly (rather than picking an arbitrary
strict order) is what §Acceptance's new test and the
`indistinguishable_from_winner` field both exist to enforce.

## 4. Acceptance

```
python3 -m pytest -q                      # 240 passed (233 + 7 net new)
python3 tools/check_import_direction.py   # exit 0
```

- Task 33's sixteen-row table reproduces bit-identical, called with
  `regime=Regime(seeded=False, num_seeds=1)` explicit.
- Task 36's two-fabric result reproduces bit-identical, same explicit
  regime — `domain8_40gpu` winner `(8,)` at `326.2362ms`;
  `domain4_40gpu` winner `(4,3,1)` at `446.5146ms`.
- `test_regime_input_changes_the_ranking`: a `RegimeAwareFakeEvaluator`
  whose own prices depend on `Regime.seeded` shows `plan()`'s own
  ranking flips between the two regimes — mirroring Task 44's own real
  EP-degree reversal (§1's own honesty note: not something Part A's
  own restricted measurement produced, but a real, already-verified
  case the test is built to match).
- `test_plan_marks_overlapping_intervals_as_indistinguishable_from_winner`,
  `test_plan_burst_never_marks_anyone_indistinguishable`,
  `test_regime_rejects_multiple_seeds_of_a_deterministic_burst`,
  `test_regime_rejects_zero_seeds`: the `Regime`/indistinguishability
  machinery in isolation.
- `test_plan_two_stage_shortlists_placement_but_not_sizing`,
  `test_plan_two_stage_relaxes_objectives_floors_at_stage_1_only`:
  `plan_two_stage()`'s own two central properties, against synthetic
  evaluators whose burst/streaming prices are deliberately reversed —
  no Frontier needed to prove the wiring is correct, matching this
  project's own established `FakeEvaluator` convention.

## 5. Part C: what the corrected planner now recommends

### 5.1 Preferred parallelism degree (Tasks 32/33) — **unchanged**

`attn_tp=2`, `attn_shape=(2,)` remains the winner under streaming
(§1's own table): `3.2235ms ± 0.1437`, clearly separated from the
second-best candidate (`4.4340ms ± 0.1590`, non-overlapping). This is
the placement/compute-parallelism axis Part A validated — no change is
exactly the expected, confirmed result, and (per Task 43A's own
precedent, cited by Task 44) worth reporting as plainly as a reversal:
a dimension that doesn't move the answer is still an answer.

### 5.2 Replica ratio (Task 41) — **already reversed by Task 41 itself; now reproduced through the shipped tool**

Task 41's own S3.3 already found, via a bespoke `seed_stats.run_seed_study`
probe (N=20), that `ffn_replicas=1` beats `ffn_replicas=2` (+2.76%) and
`ffn_replicas=4` (+7.19%) under streaming — reversing the deterministic
pass's own "more FFN replicas always wins" finding. §3's own table
above reproduces this **through `plan()`'s own new `Regime` input
directly**, not a bespoke script: `ratio=(1,1)` wins at `3.2235ms`,
matching Task 32's own seeded winner for `tp=2, (2,)` (Task 41's own
cross-check) to within N=6-vs-N=20 sampling noise. **The corrected
recommendation is `ffn_replicas=1`** — unchanged from what every task
before Task 22 already assumed by default, and the one Task 45 confirms
is the recommendation `plan()` gives when asked under the regime that
matters, not merely the direction a hand-run study already found.

### 5.3 Expert-parallel degree (Task 44) — **reverses**

This is new: Task 44 itself never ran its own full joint search under
streaming (it built a hand-picked 3-point packed-only seeded check in
its own final section, not through `plan()`). §3's own table above is
the first *complete*, regime-aware confirmation, through the shipped
`plan_two_stage()`/`plan()` API, of every placement at every degree:

**The corrected recommendation is `ep=1`** — no expert-parallel
splitting at all — reversing Task 44's own burst-regime finding
(`ep=4`, fully packed, `10.2228ms`, beating `ep=1` by 17.1%). Under
streaming, `ep=1` wins at `3.4959ms`, beating `ep=4` (`3.7654ms`) by
**7.71%**, CIs disjoint. `ep=2` (`3.6790ms`) is indistinguishable from
the `ep=1` winner at N=6 (§3's own table).

**Why**: expert-parallel degree is a *capacity* knob — more EP ranks
means more independent dispatch/compute capacity to spread token
routing across — and Task 42's own mechanism (§0 of that report)
applies exactly here: a burst manufactures simultaneous contention for
that capacity that a real qps=20 stream does not generate nearly as
often, so the capacity mostly sits idle under streaming while the
*added* per-hop all-to-all dispatch cost (Task 20's own finding: expert
exchange is priced per pair, unconditionally) is paid on every request
regardless of load. This is the same mechanism Task 41 found for FFN
replica count, applied to a different capacity knob.

## 6. Wall-clock cost

All timings are real (`time`, wall-clock), same EP-degree study
throughout (`domain8`, `attn_tp=1`, `ep_values=(1,2,4)`, full joint
search — 8 distinct `(ep, ep_shape)` placements):

| search | evaluations | wall-clock |
|---|---|---|
| burst (old regime, full search) | 8 | **2m30s** |
| streaming, full search, no shortlist (N=6) | 48 (8 × 6 seeds) | **≈47min** |
| two-stage (`shortlist_size=1`) | 8 burst + 18 seeded (3 sizing groups × 6 seeds) | **19m45s** |

The two-stage design cuts the expensive (seeded) evaluation count from
48 to 18 — a 2.7x reduction — and cuts wall-clock from ≈47min to
≈19m45s (2.4x), while reproducing the **exact same winner and the
exact same streaming numbers** the full 48-evaluation search finds
(`ep=1`, `3.4959ms ± 0.1469`, bit-identical). This is not a
theoretical saving: it is Part A's own placement-axis guarantee
(§1) cashing out directly, on real Frontier evaluations, on the one
axis this task actually validated it for.

The remaining cost (19m45s vs the old regime's 2m30s, still 7.9x more
expensive than burst alone) is the honest price of getting a
regime-correct sizing answer: the sizing axis itself (§2) is never
shortlisted, by design, so every one of its 3 values still needs its
own full `num_seeds=6` streaming evaluation.

## 7. Anywhere this specification is wrong

**Every citation in §1's own comparison table checks out exactly,**
verified against each source report directly rather than trusted:

- "FFN replicas... more is 34% faster [burst] / 2.8% slower
  [streaming]": Task 41 §3.1 states `+34.1%` for `1:1→1:2`; §3.3 states
  `+2.76%` for `ffn_replicas=2` vs `1`. Both match, to one decimal
  place beyond what the spec rounds to.
- "Memory against network... memory dominates [burst] / memory not
  significant, network is [streaming]": Task 42's own conclusion 3
  (§1 of that report) states exactly this reversal, with memory's own
  streaming effect explicitly reported as CI-overlapping (not
  significant) and network's as CI-disjoint (significant). Matches.
- "Expert-parallel degree... four groups win by 17% [burst] / one
  group wins by 7.7% [streaming], intervals disjoint": Task 33's own
  §5 gives `ep=1: 12.3316ms`, `ep=4: 10.2228ms` (this task's own
  reconstruction, Task 44's report) — `(12.3316-10.2228)/12.3316 =
  17.1%`. This task's own §3.1 gives `ep=1: 3.4959ms`,
  `ep=4: 3.7654ms` — `(3.7654-3.4959)/3.4959 = 7.71%`, CIs disjoint.
  Both match exactly.

**One place the spec's own phrasing doesn't match what its own
instructed measurement produces**, already flagged in §1: "add a test
… on a case where Part A showed it does [change the ranking]" reads as
though Part A itself would find a reversal. Run exactly as specified
— Task 33's own sixteen-row table, streaming vs burst — Part A finds
the opposite: perfect rank correlation, no reversal anywhere in that
table. The reversal the acceptance test needs, and gets, comes from
Task 44's and Task 41's own already-established sizing-axis findings,
not from Part A's own placement-axis measurement. This is not a
citation error (nothing is misquoted), but a mismatch between what the
spec's own narrative implies Part A will show and what it actually
shows when run — worth naming exactly because Part A's own honest
result (rho=1.0) is itself informative, and blurring it with the
sizing-axis reversal would understate how cleanly the placement axis
held up.

**Otherwise, nothing else checked in this specification was wrong.**
The mechanism distinction it opens with — placement holds and grows
under streaming, sizing reverses because added capacity only pays off
against a backlog — is exactly what this task's own Part C
measurements confirm, on both the already-established axes (replica
ratio, EP degree) and the one Part A newly measured directly (TP
degree/shape).

## What shipped

- `tools/planner_core.py` — `Regime` (task 45's own explicit
  arrival-process input, with `num_seeds`); `plan()`'s signature
  extended with `regime` as a required parameter (no default);
  `_mark_indistinguishable_from_winner`; `PlanResult.regime`;
  `plan_two_stage()`/`TwoStagePlanResult`/`_sizing_key` (Part B).
- `tools/planner.py` — `SimulationEvaluator` now binds a `Regime` and
  averages over `num_seeds` seeded runs via `seed_stats.compute_interval_stats`
  when `num_seeds > 1`; `plan()`'s wrapper takes `regime` required, no
  default; `plan_two_stage()`'s own convenience wrapper, defaulting
  both stage evaluators to `SimulationEvaluator`.
- `tests/test_planner_core.py` — every pre-existing `plan()` call site
  updated to pass `BURST = Regime(seeded=False, num_seeds=1)`
  explicitly; 7 new tests: the required regime-changes-ranking
  acceptance test, indistinguishability marking (overlapping and
  disjoint), burst's own null case, `Regime`'s own two validation
  rules, and `plan_two_stage()`'s two central properties (placement
  shortlisted, sizing never pre-filtered; stage-1 objectives floors
  relaxed).
- `docs/tasks/45-regime-report.md`, this report.

One commit on `task-45-regime`, stacked on `task-44-ep-placement`.
Task 33's sixteen-row table and Task 36's two-fabric result both
reproduce bit-identical under the explicit burst regime.
