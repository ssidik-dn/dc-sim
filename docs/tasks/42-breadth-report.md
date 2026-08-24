# Task 42 — Which conclusions survive a second setting?

Branch: `task-42-breadth`, branched from `task-41-replica-ratio`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`.

226 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. Measurement only — nothing under `src/`, `tools/`, or `tests/`
changed. Task 33's own sixteen-row table and Task 36's own two-fabric
result both reproduce bit-identical, checked directly. All new figures
below use each conclusion's own original tool (`run_tp_domain_probe.py`,
`run_m2n_real_profile.py`, `run_memory_edge_study.py`,
`run_memory_tp_study.py`, `run_compute_balance_study.py`), imported for
their own `_argv`/deployment/placement helpers, not reimplemented — no
new machinery, per this task's own §1.

---

## 0. The mechanism behind nearly every reversal, found once and confirmed everywhere

Before any of the six conclusions: **every one of tasks 12, 18–24, 26,
and 28's own real-compute tools submitted every request at `t=0`,
regardless of its configured `qps`.** This is Task 31's own finding
(§1.3: *"every configuration any tool in this project has used before
[task 31] is completely deterministic given everything except
`--seed`... there is no experiment to run that would show variance in
that configuration, because there is no seed-dependent input left once
arrivals are fixed to `t=0`"*), confirmed here by reading each of the
six conclusions' own `_argv` builders directly: none of them call
`seed_stats.seed_argv_fix()` or pass
`--offline_use_generated_request_arrivals`, because that helper did not
exist until task 31 — every one of the original six measurements
predates it. The `qps` value quoted in each of those reports (`1.0` for
tasks 18–21, `20.0` for tasks 22/24/26/28) was never actually staggering
anything.

This is the single axis this task varies for all six conclusions, per
its own §2 priority ("vary the workload first... since that is what
Task 41 showed matters and it costs least") — reusing each original
tool's own exact configuration, plus `seed_argv_fix()` and N seeds
(task 31's own convention). Every reversal found below traces back to
this same mechanism: a `t=0` burst puts every request in contention for
capacity/batching at the same instant; genuine Poisson staggering
(mean 50ms apart at `qps=20`) usually does not.

---

## 1. The conclusions-against-settings table

| # | Conclusion | Original (burst) | Streaming (seeded) | Other settings | Verdict |
|---|---|---|---|---|---|
| 1 | TP-split across domains costs ~88% (Task 20/21, tp=4) | packed 5.8033ms, split 10.9297ms, **+88.3%** | packed 3.8726ms(±0.62%), split 8.9693ms(±1.15%), **+131.6%**, CIs disjoint | llama2_7b model: packed 6.169ms, split 11.295ms, **+83.1%** | **Held**, size changed (larger under streaming) |
| 2 | Separating ATTN/FFN pools costs ~15% (Task 12) | colocated 5.8375ms, split 6.7103ms, **+14.9%** (M2N share 14.2%) | colocated 5.8362ms(±0.50%), split 6.6772ms(±0.35%), **+14.4%**, CIs disjoint | decode=64: +15.1%; 32 req: +15.5%; Phi-tiny-MoE model: +20.9% | **Held**, remarkably robustly |
| 3 | Memory binds harder than network (Task 22) | memory effect **+143.7%** (30.06 vs 12.33ms); network effect **+18.4%** (14.61 vs 12.33ms) — memory ≫ network | memory effect **+11.4%** (3.920 vs 3.518ms, CIs overlap — not significant); network effect **+28.5%** (4.522 vs 3.518ms, CIs disjoint — significant) — **network > memory** | — | **Reversed** |
| 4 | Two-way parallelism beats one-way on latency (Task 24/26/28) | tp=1 vs tp=2 at nb=6: 37.32 vs 35.23ms (+5.9%); at nb=120: 14.61 vs 13.95ms (+4.7%) — tp=2 always wins | at nb=6: 5.537 vs 5.004ms (+10.6%, CIs overlap — not significant); at nb=120: 4.522 vs 4.231ms (+6.9%, CIs disjoint — significant) — tp=2 still wins both points | — | **Held**, significance now marginal at the tight-memory point |
| 5 | The optimum degree does not shift with memory (Task 24/26/28) | tp=2 wins at every one of 10 margins tested, 0.001–0.9843 | tp=2 wins at both memory points tested (nb=6, nb=120) — no shift observed | — | **Held**, on a smaller sweep than the original |
| 6 | Pool utilisation unbalanced at parity (Task 22 S3 / Task 41) | 1:1 — attn 18.1%, ffn 42.8% (2.37x); 1:2 — attn 34.8%, ffn 41.2% (ffn still busier) | 1:1 — attn 20.06%(±5.70%), ffn 29.01%(±6.59%), **1.45x**, disjoint; 1:2 — attn **33.31%**(±8.93%), ffn **20.34%**(±9.91%), disjoint — **attn now busier** | — | **Reversed** at 1:2; **held direction, smaller size** at 1:1 |

CIs quoted are 95%, task 31's own `seed_stats.compute_interval_stats`;
"CIs disjoint" means the two intervals do not overlap at all (a
clean separation, not a borderline call); "overlap" is stated exactly
where it happens rather than rounded away.

## 2. Every reversal, with its mechanism

### 2.1 Memory binds harder than network (Task 22) — reversed

**Mechanism**: at `qps=20` (mean 50ms between arrivals) and
`num_blocks=6` (2 concurrent-request capacity), the burst configuration
puts all 32 requests in simultaneous contention for those 2 slots at
`t=0` — a genuine admission queue, the mechanism Task 22's own S2
already identified (zero preemptions, pure queueing delay). Under
real Poisson staggering, a request typically completes its short decode
phase (this model, this workload: a few ms) well within the ~50ms
average gap before the next one arrives — 2 concurrent slots are then
rarely all occupied at once, so the same capacity that was a hard wall
under the burst is barely a constraint at all under streaming. The
network effect (colocated vs split, same `num_blocks=30`) has no such
dependency on how many *other* requests are simultaneously queued — it
is a fixed, roughly per-request routing cost paid regardless of
concurrent load — so it survives streaming with its own effect size
essentially intact (18.4% → 28.5%, the same order of magnitude,
plausibly larger for the same reason task 11/13's own overlap findings
would predict: smaller, more numerous streaming batches see the M2N hop
serialize with a *smaller* compute denominator per batch, the same
mechanism Task 12 §2 already established for a single-batch case).

**What survives**: the *network* effect from Task 22's own S4 is the
part of "memory binds harder than network" that generalises; the
*memory* part was the burst artifact.

### 2.2 Pool utilisation ranking at 1:2 (Task 22 S3 / Task 41) — reversed

**Mechanism**: utilisation here is busy-time divided by
`(wall_time × replica_count)` — adding a second FFN replica divides the
same aggregate FFN work over twice the available replica-time,
mechanically lowering FFN's own per-replica utilisation, while ATTN
(still one replica) absorbs whatever additional *throughput* the extra
FFN capacity now allows through the pipeline. Under the burst, this
project's own Task 22 already found both utilisations rise together
from the 1:1 baseline (18.1%/42.8% → 34.8%/41.2%) without attention ever
overtaking FFN. Under streaming, the *same* mechanism pushes ATTN's
utilisation *past* FFN's (33.3% vs 20.3%) — plausibly because streaming
arrivals let the now-larger FFN capacity actually keep pace with the
staggered stream (never building the batching backlog a burst would),
so FFN's own busy fraction drops further than it did under the burst,
while ATTN — still the single, unexpanded pool — keeps absorbing the
full arrival stream's own attention work. This is the same *kind* of
regime-dependence Task 41 found for the *latency* effect of this exact
ratio move; Conclusion 6 shows it extends to the *utilisation-balance*
question Task 22 asked first, not only to the throughput/latency
question Task 41 asked afterward.

**What survives**: neither pool idles "substantially" in either regime
— Task 22's own broader finding (S3's own "no pool drops below ~17% or
rises above ~86%") is not contradicted by either measurement here.
*Which* pool is busier at a non-1:1 ratio is the part that reverses.

## 3. Which findings are now cross-setting-supported, and which remain single-setting

**Cross-setting-supported after this task** (burst + streaming, at
least one held in both, several also across model or workload length):

- Conclusion 1 (TP-split ~88%): burst, streaming, and a second model —
  the most broadly checked conclusion in this project after this task.
- Conclusion 2 (pool-separation ~15%): burst, streaming, decode length,
  request count, *and* a second model — now the single most
  thoroughly cross-checked figure in the whole project's own history.
- Conclusion 4/5 (tp=2 beats tp=1; no shift with memory): burst (task
  24/26/28's own wide margin sweep) and streaming (this task, two
  points) — the *direction* is now checked in both regimes, though
  streaming was only checked at two memory points, not task 28's own
  ten.

**Still single-setting, and now known to be fragile rather than merely
untested**:

- Conclusion 3 (memory > network): the memory half of this claim is
  now known *not* to generalise past the burst regime it was measured
  in — not "untested," but actively contradicted by a second setting.
- Conclusion 6, the *ranking* at non-1:1 ratios: known to flip between
  regimes, at exactly the ratio (1:2) both Task 22 and Task 41 treated
  as informative. The *existence* of imbalance at 1:1 is
  cross-setting-supported (both regimes show a real, CI-separated gap);
  the *ranking away from 1:1* is not.

**This task's own known trap, applied to itself**: none of the above is
general from two points (§6's own warning — "two points make a line
only if you already know the shape"). Every "held" verdict in §1 means
exactly "held at the specific second setting tested here," not "holds
everywhere." Conclusions 4/5 in particular were checked at only two
memory points under streaming, against task 24/28's own ten under the
burst — the sweep is narrower here, stated as a scope limit, not
papered over as equivalent coverage.

## 4. What the six original documents should say differently

Every one of the six reports below states its headline conclusion
without a workload-regime qualifier, because none of them had reason to
know one was needed — `seed_argv_fix()` did not exist yet for tasks
12–24/26/28, and Task 22 predates it by nine tasks. Specific rewrites:

- **Task 22, S1** ("Memory, when it is scarce... dwarfing every other
  effect this study or any prior one in this project has measured")
  needs the qualifier this task adds: *at requests submitted
  simultaneously, not at the same nominal qps under genuine staggered
  arrival* — where the same capacity point produces an effect an order
  of magnitude smaller and not clearly distinguishable from noise.
- **Task 22, S3 / Task 41's own citation of it** ("FFN is the
  consistently busier pool at every ratio tested") needs: *under
  simultaneous-arrival load*; this task finds the ranking itself can
  flip once arrivals are genuinely staggered.
- **Task 12's headline ~15%** and **Task 20/21's headline ~88%** need
  no correction in direction or rough size — but both should still gain
  a sentence noting they were measured under simultaneous arrival, since
  this task found both figures move by a similar or larger amount
  (not smaller) once arrivals are staggered, and a reader should not
  assume "measured once, holds everywhere" was ever established.
- **Task 24/28's "the optimum does not shift with memory... at any
  device memory this model and device can reach"** is accurate as
  stated (it is about the *memory* axis specifically) but should note
  it was never checked against a *workload* axis at all before this
  task — the claim about memory holds, but was previously silent on
  whether workload could independently move it (it does not, on the
  two points checked here, but "does not shift with memory" and "does
  not shift, period" are different claims and the original text reads
  as the broader one).

## 5. Anywhere this specification is wrong

**The Task 41 quotation is accurate, checked verbatim.** `docs/tasks/41-replica-ratio-report.md`
§3.3 does contain, word for word: *"under this workload's own realistic
arrival process, adding FFN replicas makes mean tpot *worse*, not
better."* No correction needed here — unlike several of this project's
own prior specifications, this one's central citation matches its
source exactly.

**The "~88%" row's own citation to both Task 21 *and* Task 36 is
questionable, and should probably name only Task 21.** Task 21 §2
reproduces Task 20's own tp=4 figure bit-identically (+88.3%, quoted
correctly). Task 36's own report, checked directly, measures a
*different* thing at a *different* magnitude: `attn_tp=8` (not 4),
forced onto a split fabric for `Llama-3.1-405B-Instruct-FP8` (not the
model any TP-split conclusion in this project's history was ever
measured against) — margins of **+36.85%** (deterministic) and
**+41.42%** (seeded), roughly 40–47pp below the "~88%" this table's own
row states. Both are real, both are placement-penalty findings, and
citing Task 36 as *supporting evidence that topology-splitting has a
real, large cost* is defensible — but citing it as a source for the
specific figure "~88%" is not; Task 36 never measured anything close to
88% and was never trying to. This is exactly the kind of citation this
task's own §5 asks to be checked, and — unlike the running count this
project has kept since task 22 — this specific one is genuinely a
partial mismatch rather than either a clean match or a total
fabrication: right conclusion family, wrong number attached to the
wrong report.

**Otherwise, nothing else checked in this specification's own account
of prior tasks was wrong.** The three general framings this task
opens with — that Task 41's reversal implicates "everything else," that
workload is the cheapest axis to vary, and that Task 38's own screening
requirement should be checked before picking a second model — all held
up under direct verification. (This task did not end up needing a
second *model*'s screening decision beyond the one already reused from
Tasks 12/35: `llama2_7b_dense_example` and `Phi-tiny-MoE-instruct` were
both already known-working, real-scale, real-profiled models per Task
35's own inventory; `step-moe-noquant-small` was correctly avoided per
Task 38 §5's own finding, exactly as this task's own §2 instructed —
not attempted, and not needed, since both models actually used already
gave every conclusion tested a genuine architecture change: MoE vs
dense, GQA vs MHA.)

## What shipped

- `docs/tasks/42-breadth-report.md`, this report. No source, tool, or
  test file changed — measurement only, reusing six existing tools'
  own `_argv`/deployment/placement helpers (`run_tp_domain_probe.py`,
  `run_m2n_real_profile.py`/`run_m2n_integration.py`,
  `run_memory_edge_study.py`, `run_memory_tp_study.py`,
  `run_compute_balance_study.py`) plus `tools/seed_stats.py`'s own
  `seed_argv_fix`/`run_seed_study`, per this task's own "no new
  machinery" instruction.

One commit on `task-42-breadth`, stacked on `task-41-replica-ratio`.
Task 33's sixteen-row table and Task 36's two-fabric result both
reproduce bit-identical.
