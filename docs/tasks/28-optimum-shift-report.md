# Task 28 — Does the optimal degree actually shift with memory?

Branch: `task-28-optimum-shift`, stacked on `task-27-penalty-reconcile`.
Paths confirmed per task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`.

189 tests pass (investigation task, no new tests), and
`python3 tools/check_import_direction.py` exits 0.

**The short answer: the optimum does not shift, at any device memory
this model and device can reach. Task 24's own report already says so,
in its own words, quoted below. The document's claim is not a
misreading of an edge case — it asserts the opposite of what task 24's
own §3 states as its headline finding.**

---

## 1. What Task 24 actually reports (§2.1)

Quoted directly, `docs/tasks/24-memory-planner-report.md`, §3
("Whether the optimal degree shifts with device memory") — its very
first sentence:

> **It does not, across every margin tested (0.9843, 0.984, 0.9, and
> the supplementary 0.992) — tp=2, packed, is throughput- and
> latency-optimal at all four**, and both optima are always the same
> degree (no throughput/latency tradeoff to report between them here:
> tp=2 packed wins both simultaneously at every margin).

And, on the specific claim that a roomier device favours tp=1:

> Looser margins (0.984, 0.9): tp=1 runs, unconstrained, but at 8x the
> weight-sharding cost of tp=2/4/8 — **no memory penalty, tp=1 just has
> no reason to be preferred.**

**The device-memory mechanism, quoted from the same report, §2.1:**
margin fraction, not an absolute figure or any other knob —
`memory_margin_fraction`, `--cluster_config_decode_attn_replica_config_memory_margin_fraction`,
with `gpu_memory_utilization = 1 - margin`. Task 24's own §2 grid, converted
to absolute usable memory (`80 GiB × (1-margin)`):

| margin | usable memory | tp=1 throughput / tpot | tp=2 packed throughput / tpot | winner |
|---|---|---|---|---|
| 0.9843 | 1.256 GB | 29.857 req/s / 37.3199 ms (memory-bound) | 90.612 req/s / 13.9539 ms | tp=2, by a wide margin |
| 0.984 | 1.280 GB | 86.806 req/s / 14.6052 ms | 90.612 req/s / 13.9539 ms | tp=2 |
| 0.9 | 8.000 GB | 86.806 req/s / 14.6052 ms | 90.612 req/s / 13.9539 ms | tp=2 |
| 0.992 | 0.640 GB | **infeasible (OOM)** | 90.612 req/s / 13.9539 ms | tp=2 (tp=1 doesn't run) |

**tp=1 does not beat tp=2 at any point in task 24's own grid — not on
throughput, not on latency, not at the tightest margin, not at the
roomiest.** §2.3's question ("is there a point in its grid where tp=1
beats tp=2") is answered directly by the table above: no.

## 2. The device-memory range each study covered, in absolute terms

**Task 24**: 0.640 GB (margin=0.992, tp=1 infeasible) up to 8.000 GB
(margin=0.9, the roomiest margin task 24 tested).

**Task 26 Part B**: the *same three* margins task 24 used (0.9843,
0.984, 0.9 — quoted directly from task 26's own report, §B.1: *"Task
24's own three margins (0.9843, 0.984, 0.9)"*), i.e. 1.256 GB to
8.000 GB — a strict subset of task 24's own range, not a wider one.
Task 26 added an *overhead* axis (0-8 GiB folded in on top), not a
wider margin sweep.

**Neither study tested anything roomier than 8.000 GB usable memory
before this task.** Task 26's own quoted finding — *"The optimum never
shifts. It is either tp=2 packed, or nothing"* (§B.3) — is stated
about the same 8 GB ceiling task 24 used, not about a wider range. The
two reports are not measuring different regions and both being right;
they measured the *same* region and agree with each other. The
document's claim is not supported by either.

## 3. Whether tp=1 is ever optimal — measured beyond both studies' own range

Since §1-2 already settle the question inside the range either report
tested, and both already say no, the open question this task's own
§3 asks is narrower: **is there a roomier device, beyond what either
report checked, where tp=1 would win?** h800's own total memory (80 GiB)
sets the ceiling — margin can approach but not reach 0.

Swept from task 24's own most generous margin (0.9, 8 GB) up to
margin=0.001 (79.92 GB — effectively the entire device), using task
25's own confirmed analytical fold-in
(`run_memory_tp_study.py`'s explicit `num_blocks` injection, the exact
method task 26 already reused, not reimplemented) for `tp=1` and
`tp=2` packed, one seed each (every value in this sweep sits far above
the batch=8 plateau task 22 already established runs on admission-rate
grounds, not memory — a repeat-count question that matters near a
knee does not apply to six points that all land in the same flat
plateau):

| margin | usable memory | tp=1 throughput | tp=1 tpot | tp=2 throughput | tp=2 tpot |
|---|---|---|---|---|---|
| 0.9 | 8.00 GB | 86.806 req/s | 14.6052 ms | 90.612 req/s | 13.9539 ms |
| 0.7 | 24.00 GB | 86.806 req/s | 14.6052 ms | 90.612 req/s | 13.9539 ms |
| 0.5 | 40.00 GB | 86.806 req/s | 14.6052 ms | 90.612 req/s | 13.9539 ms |
| 0.2 | 64.00 GB | 86.806 req/s | 14.6052 ms | 90.612 req/s | 13.9539 ms |
| 0.05 | 76.00 GB | 86.806 req/s | 14.6052 ms | 90.612 req/s | 13.9539 ms |
| 0.001 | 79.92 GB | 86.806 req/s | 14.6052 ms | 90.612 req/s | 13.9539 ms |

**Bit-identical at every point, from 8 GB to 79.92 GB usable — batch
stays at 8 for both degrees throughout.** tp=2 packed wins throughput
(90.612 vs 86.806 req/s) and inter-token latency (13.9539 ms vs
14.6052 ms) simultaneously at every margin tested, spanning
essentially the entire memory range this device can offer. **tp=1 is
never optimal, at any device memory this model and device can reach.**
The mechanism is exactly what task 24's own §3 already stated: once
both degrees are past their own admission-rate ceiling (task 22's own
finding — the batch=8 plateau is set by arrival rate, not memory,
once capacity clears roughly 10 blocks' worth), more memory does
nothing further for either degree, while tp=2 keeps a separate,
memory-independent advantage from dividing the same per-step compute
across two GPUs (13.9539 ms vs 14.6052 ms is a compute-sharding
effect, not a memory one — visible already at task 24's own
margin=0.9, unchanged by every margin since).

**This is the negative result this task's own §3 names as a real
outcome.** Parallelism pays for itself at every memory size this
device and model can reach; there is no roomier-device regime where
undivided execution wins.

## 4. What §3.10 should say

The current text should be replaced. Suggested wording:

> On a constrained device, single-device execution is not merely
> slower but infeasible: weights and fixed overhead exhaust the memory
> before any cache can be allocated, and two-way parallelism is the
> smallest degree that fits. **On a roomier device, single-device
> execution becomes merely worse, not competitive: it can run, but it
> never catches up to two-way parallelism on either throughput or
> inter-token latency, at any memory size this device can offer.
> Two-way parallelism dividing the model's own per-step compute across
> two GPUs is a memory-independent advantage on top of whatever memory
> relief it also provides, and it persists all the way to the device's
> full 80 GB.** The optimal degree does not shift with memory for this
> model; what shifts is only whether the losing degree runs at all.

This removes the false claim (§3's own header sentence: "the optimal
degree therefore falls as memory grows") and states instead what tasks
24, 26, and this task's own wider sweep all agree on. If §3.10 wants
to keep a sentence about *why* this is the interesting first
cross-cutting result, it can keep that framing — the interesting part
survives; only the direction of the claimed effect does not.

## 5. Anywhere this specification is wrong

**Nothing in this specification's own quotations from task 24 or task
26 is inaccurate** — both are quoted correctly and the discrepancy
this task asks about is real. The error is in the *document* this
specification is checking, not in this specification's own account of
the two prior reports. Worth stating plainly since this project has
repeatedly found the reverse (a spec's own citation not matching its
source): here, the spec's citations are the accurate ones, and the
separate document being checked against them is the one that doesn't
hold up.

One small refinement: §1's framing — "Task 26 §B.3 says the opposite:
'The optimum never shifts'... and its own §B.2 grid, at the roomiest
margin tested, has tp=2 packed at 90.612 req/s against tp=1's
86.806" — is accurate but could be read as implying task 26 discovered
something task 24 had not already stated. It had; task 26 reused task
24's own already-established finding rather than independently
arriving at a different one (§2 above). This doesn't change the
conclusion, only the credit: the "no-shift" finding is task 24's, and
task 26 (and now this task, at a much wider range) each reconfirmed
it, rather than each separately discovering it.

## What shipped

No new tool. This task reused `tools/run_memory_tp_study.py`'s own
`_run_scenario_in_subprocess` directly (task 25's confirmed
analytical-fold-in method) for the six-point wide sweep in §3, and
otherwise read the two primary report files directly per this task's
own acceptance criteria.

One commit on `task-28-optimum-shift`, stacked on
`task-27-penalty-reconcile`; no `upstream/`, `src/engine/`, or
predictor changes — investigation plus one confirmatory sweep, nothing
implemented.
