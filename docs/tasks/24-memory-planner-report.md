# Task 24 — The real trade: parallelism buys memory and costs communication

Branch: `task-24-memory-planner`, stacked on `task-23-memory-tp`.

189 tests pass (measurement task, no new tests), and
`python3 tools/check_import_direction.py` exits 0.

Every grid figure rests on **3 seeded runs** per cell; the S2.3 coupling
check and the supplementary tight-margin check (S3) are mechanism
checks, not headline figures, and rest on 1 run per point, per this
project's own convention for that category of check (tasks 22/23). The
S2.2 OOM confirmation and the zero-preemption re-check (3 seeds at the
tightest main-grid point) are stated with their own run counts inline.

---

## 1. What §2 found, before anything else

### 2.1 Which mode, and which knob is actually usable

`num_blocks_mode` defaults to `"memory_planner_profiled"` already — no
flag needed. That name is misleading absent one more flag:
`enable_runtime_non_kv_cache_overhead_profiling` defaults `False`, so no
profiling data is required and none was used; without it,
`"memory_planner_profiled"` behaves identically to plain
`"memory_planner"` (`non_kv_cache_overhead_bytes` stays at its `0`
default either way). Confirmed by reading `base_replica_scheduler.py`'s
own gating, not assumed.

**The knob this task's own S1 names — `gpu_memory_utilization` —
turned out not to be usable for DECODE_ATTN specifically, and this was
checked, not assumed.** It has no per-cluster CLI override at all
(`--cluster_config_decode_attn_replica_scheduler_config_gpu_memory_utilization`
is rejected by argparse as unrecognized), and the *global*
`--vllm_v1_scheduler_config_gpu_memory_utilization` flag is silently
ignored for DECODE_ATTN — confirmed live: setting it to `0.0002` and
reading the value back off the running replica scheduler
(`rs._config.gpu_memory_utilization`) showed `None`, unaffected by the
flag. The one real, per-cluster-scoped door left open is
`memory_margin_fraction`
(`--cluster_config_decode_attn_replica_config_memory_margin_fraction`) —
already used by every real-compute tool in this project since task 09,
always pinned to `0.2`. `gpu_memory_utilization=None` falls back to
`1 - memory_margin_fraction` (`memory_planner.py`'s own
`_get_effective_gpu_memory_utilization`), so sweeping margin *is*
sweeping usable device memory — just through the one door Frontier
actually leaves open per cluster, not the one the task's own S1 names.

### 2.2 Does insufficient memory raise, or silently clamp?

**Raises. Confirmed directly, not inferred from reading `_raise_memory_oom`'s
name.** `margin=0.98438` (just below the boundary where parameter
memory alone exceeds the requested budget) produced:

```
FrontierMemoryOOMError: [FRONTIER_MEMORY_OOM][reason=parameter_memory_exceeds_requested_budget]
Model parameter shard does not fit inside the requested GPU memory budget.
(parameter_memory_per_device_bytes=1342177280, requested_memory_bytes=1341747783)
```

No silent clamp anywhere in the path this task exercised. This project's
own tools call `Simulator` directly rather than through
`frontier.main.main()`, so the CLI-level `except FrontierMemoryOOMError:
SystemExit(2)` wrapper never applies here; this tool's own
`except Exception` catches it and reports the error string per cell,
the same convention as every prior real-compute tool.

### 2.3 Does derived capacity actually rise with degree? — and a correction to Task 23

**Yes, and this required finding and fixing a wiring bug Task 23 never
saw, because Task 23's own coupling check used a method that silently
never exercised the mechanism it was trying to test.**

Task 23 left `num_blocks` unset by *omitting* the flag, relying on the
field's dataclass default of `0`. That does not do what it looks like it
does: Frontier's cluster-config builder gives every cluster that lacks
its **own** per-cluster `..._num_blocks` override flag a *shared*
`replica_scheduler_config` object (`frontier/config/config.py`'s
`get_cluster_configs_for_disaggregation`). The memory-planner derivation
(`base_replica_scheduler.py`'s `elif not self._config.num_blocks:`) then
runs exactly once, on whichever cluster's replica scheduler constructs
first — and every other cluster sharing that object inherits the
already-nonzero result and never re-derives. Confirmed directly: with
`num_blocks` merely omitted, DECODE_ATTN's scheduler config showed the
identical value as DECODE_FFN's (an unrelated `orca`-scheduler cluster)
*before* `BaseReplicaScheduler.__init__` had even run for DECODE_ATTN.

**The fix**: pass `--cluster_config_decode_attn_replica_scheduler_config_num_blocks 0`
*explicitly*. An explicit `0` (as opposed to an omitted flag) is treated
by Frontier's per-cluster-override plumbing as "give this cluster its
own copy" — which then genuinely starts at `0` and runs the
memory-planner branch using DECODE_ATTN's *own* `attn_tensor_parallel_size`.
Confirmed by re-running the exact check Task 23 ran, margin fixed at
`0.2` (this project's own long-standing default), one run per degree:

| tp | derived `num_blocks` | ratio vs tp=1 |
|---|---|---|
| 1 | 64,256 | 1.00x |
| 2 | 129,792 | 2.02x |
| 4 | 260,864 | 4.06x |
| 8 | 261,376 | **4.07x (+0.2% over tp=4)** |

**Rises roughly in proportion to weight memory freed, up to tp=4 — and
the tp=4→8 flattening has a checked mechanism, not a shrug.** This
model's `num_kv_heads=4`; `kv_heads_per_tensor_parallel_worker =
ceil(num_kv_heads / attn_tp)` floors at `1` once `attn_tp >= 4`. Past
that point, KV-block *geometry* stops shrinking with further TP — only
continued attention-weight sharding (which has no such floor) keeps
freeing a little more room, which is why tp=8 adds only +0.2% over tp=4
instead of another ~2x.

**This corrects Task 23's own S1 conclusion**, which read the identical
`num_blocks` at every TP degree as "the mechanism is real in source but
its magnitude is negligible for this model." That explanation is wrong;
the true cause was the omitted-flag wiring bug above, not a magnitude
argument. Once wired correctly, the coupling is real, substantial
(roughly 2x per TP doubling up to tp=4), and has a genuine floor
mechanism Task 23 never got to observe.

**A further, unplanned finding: the usable device-memory band is
razor-thin, and narrows further as TP rises.** At tp=1, the
parameter-memory-exceeds-budget boundary sits at `margin=0.984375`;
this project's own established "unconstrained" plateau (capacity ≥ 8,
tasks 22/23) is reached by `margin≈0.9840` — four decimal places away.
At tp=2, the same arithmetic (using the measured `param_mem(tp=2)≈671MB`,
half of tp=1's, consistent with the weight-halving just confirmed) puts
the OOM boundary at `margin≈0.9922` — even tighter, and in a completely
different place than tp=1's. **A single shared margin axis cannot show
a graded knee at more than one TP degree at once** for this model: any
margin that meaningfully varies tp=1's own binding constraint already
lands every higher TP degree far past its own, much tighter, boundary.
This is reported as a finding about what this knob can show, not
smoothed over (§7's own trap, generalised from "near the knee" to "the
knob itself").

## 2. The grid

Placement, deployment, and `install(..., collective=True)` are unchanged
from Task 23 (`run_tp_domain_probe`'s own packed/split helpers). The
memory axis (S2.1's escape valve, 3 points) is calibrated at tp=1 — the
most memory-hungry degree, so its own "below the knee" point is a
genuine constraint everywhere the axis is shared:

| tp | placement | margin | derived `num_blocks` | capacity | batch | throughput (req/s) | tpot (ms) | tp_comm (Σms) |
|---|---|---|---|---|---|---|---|---|
| 1 | packed | 0.9843 | 6 | 2 | 2.00 | 29.857 | 37.3199 | 0.0000 |
| 1 | packed | 0.984 | 30 | 10 | 8.00 | 86.806 | 14.6052 | 0.0000 |
| 1 | packed | 0.9 | 6,911 | 2,303 | 8.00 | 86.806 | 14.6052 | 0.0000 |
| 2 | packed | 0.9843 | 1,292 | 430 | 8.00 | 90.612 | 13.9539 | 7.8259 |
| 2 | packed | 0.984 | 1,341 | 447 | 8.00 | 90.612 | 13.9539 | 7.8259 |
| 2 | packed | 0.9 | 15,103 | 5,034 | 8.00 | 90.612 | 13.9539 | 7.8259 |
| 2 | split | 0.9843/0.984/0.9 | (same as packed) | (same) | 8.00 | 69.886 | 18.3178 | 112.5581 |
| 4 | packed | 0.9843/0.984/0.9 | 3,864 / 3,962 / 31,487 | 1,288 / 1,320 / 10,495 | 8.00 | 87.747 | 14.4305 | 22.5331 |
| 4 | split | (same three) | (same) | (same) | 8.00 | 47.600 | 27.2465 | 330.1171 |
| 8 | packed | 0.9843/0.984/0.9 | 4,376 / 4,474 / 31,999 | 1,458 / 1,491 / 10,666 | 8.00 | 81.473 | 15.6040 | 51.5021 |
| 8 | split | (same three) | (same) | (same) | 8.00 | 29.019 | 45.1854 | 761.4566 |

**Both regimes visible, but concentrated at tp=1.** `tp=1, margin=0.9843`
is the only memory-bound cell in the whole grid (batch=2, well below
this workload's own 8-request ceiling). Every other cell — every TP
degree ≥2, at every margin tested — sits at the batch=8 plateau,
*identical* across all three margins for that degree. This is not an
artefact of the choice of margins: it is the direct, measured
consequence of §2.3's own finding — raising TP from 1 to 2 alone already
frees far more capacity (2.02x) than the width of the "interesting"
band (roughly 2 to 30 blocks) at tp=1, so the same margin values that
usefully span tp=1's knee land every higher degree deep in the
unconstrained plateau. Zero preemptions were re-confirmed at the single
memory-bound cell (`tp=1, margin=0.9843`) across all 3 seeds — the same
admission-queueing mechanism tasks 22/23 established, not eviction.

**A supplementary check, at a margin tight enough to matter for tp≥2**
(`margin=0.992`, motivated by the tp=2 OOM-boundary estimate above; one
run each, packed only — a mechanism check, not a grid point):

| tp | derived `num_blocks` | capacity | batch | throughput | tpot |
|---|---|---|---|---|---|
| 1 | — | — | — | — | **OOM** (`parameter_memory_exceeds_requested_budget`) |
| 2 | 30 | 10 | 8.00 | 90.612 | 13.9539 |
| 4 | 1,341 | 447 | 8.00 | 87.747 | 14.4305 |
| 8 | 1,853 | 618 | 8.00 | 81.473 | 15.6040 |

At this margin, tp=1 is not merely memory-bound — it is **infeasible**:
the model's own attention-weight shard does not fit at all. tp=2 is
*already* at the unconstrained plateau. This is the coupling in its
most dramatic form: the same device memory that cannot host a tp=1
replica at all comfortably hosts a tp=2, 4, or 8 one.

## 3. Whether the optimal degree shifts with device memory

**It does not, across every margin tested (0.9843, 0.984, 0.9, and the
supplementary 0.992) — tp=2, packed, is throughput- and latency-optimal
at all four**, and both optima are always the same degree (no
throughput/latency tradeoff to report between them here: tp=2 packed
wins both simultaneously at every margin). What *does* change with
device memory is not which degree wins, but whether the losing degree
(tp=1) is merely worse or entirely unreachable:

- Looser margins (0.984, 0.9): tp=1 runs, unconstrained, but at 8x the
  weight-sharding cost of tp=2/4/8 — no memory penalty, tp=1 just has
  no reason to be preferred.
- `margin=0.9843`: tp=1 runs but is memory-bound (batch=2, tpot
  2.68x worse than tp=2 packed).
- `margin=0.992`: tp=1 cannot run at all.

**This is a real, checked, capacity-planning-relevant answer, even
though it is not the shifting-optimum result the task's own S7 warns to
be most sceptical of finding.** The mechanism for why it doesn't shift,
stated rather than left as a coincidence: tp=2's own communication cost
(7.83 ms summed) is already small enough, and its memory relief already
large enough (2.02x), that no margin in the tested range makes tp=4's
or tp=8's *additional* communication cost (22.53 ms, 51.50 ms) worth
paying for memory relief tp=2 doesn't already provide. A model with a
larger per-device weight footprint relative to device memory, or one
whose `num_kv_heads` didn't floor out until a higher TP degree, could
plausibly show the optimum move to tp=4 — this was not tested, since
this project's own model doesn't create that condition, and asserting
it without measuring it would be exactly the "reasoned about instead of
simulated" gap this project exists to avoid.

**Does splitting move the optimum? No — checked at every degree.**
Comparing split-only options (tp=2/4/8 split): tp=2 split (tpot=18.32 ms,
69.89 req/s) beats tp=4 split (27.25 ms, 47.60 req/s) beats tp=8 split
(45.19 ms, 29.02 req/s) — the same ranking as packed. Placement changes
the *absolute* cost of every degree (worse when split, growing with
degree — task 23's own finding, reconfirmed here) but never changes
*which* degree is best. Sizing and placement decisions are independent
for this model, not coupled the way the task's own S4 raised as a
possibility.

## 4. Does Task 22's "memory binds" survive with capacity properly derived?

**Yes at tp=1, and more dramatically than task 22 itself found — but it
does not generalise past tp=1, and that qualification is the real
result.** At tp=1, deriving capacity from device memory rather than
pinning it doesn't soften task 22's finding, it sharpens it: task 22's
own worst measured case was a 2.4-2.9x tpot/throughput swing; this
task's own tp=1 sweep reaches a case (`margin=0.992`) where the
configuration is not merely slow but **cannot run at all**. Memory
binding, properly modelled, is not bounded above by anything task 22
measured — an OOM is a harder constraint than any queueing delay.

But at every TP degree ≥2 tested, across every margin from clearly
constrained (at tp=1) to clearly not, memory never bound anything: batch
stayed at the 8-request plateau throughout. **"Memory binds" was
already a tp=1-specific finding, and deriving capacity honestly is what
makes that scope visible rather than assumed** — task 22's own grid
never varied TP degree at all, so it could not have shown this either
way. Read together with task 23's own parallel finding (network
dominance is placement- and degree-specific, not universal), this
project's three largest-effect claims — network, memory, compute — are
now each qualified by the same kind of boundary: real, large, and
narrower in scope than a first measurement at a single fixed
configuration could show.

## 5. Anywhere this specification is wrong

- **"Task 21 established [a split four-way group] costs about a sixth
  of a decode step"** (S4) — grepped both
  `docs/tasks/21-collective-patterns-report.md` and
  `docs/tasks/20-collective-backend-report.md` directly for "sixth",
  "16.7", and "four-way group": no match in either. Task 21's own real
  figures for a split four-way group are
  `tensor_parallel_communication_time` = 2.628864 ms (packed) /
  38.513664 ms (split) (a whole-run sum, task 21's own convention, not
  a decode-step share), and task 20's own inter-token-latency headline,
  "+5.126 ms at tp=4, ~88% over packed's 5.803 ms tpot" (an *increase*
  ratio, not a share-of-step figure). Neither is "about a sixth" under
  any framing checked. This is the same recurring pattern tasks 17, 19,
  20, 21, 22, and 23 each already found once in their own opening
  citations — here it is task 24's turn.
- Otherwise this specification's structure — establish mode, OOM
  behaviour, and coupling before running anything; keep the memory axis
  cuttable to 3 points rather than dropping TP or placement; report
  derived capacity in every cell; ask for the optimal degree per memory
  point rather than "which is bigger"; ask explicitly whether placement
  moves that optimum — matched exactly what the investigation needed,
  and correctly anticipated that the most likely, least convenient
  outcome (no shift) was worth checking for as rigorously as a shift
  would have been.

## What shipped

- `tools/run_memory_planner_study.py` — the S2 mode/OOM/coupling checks,
  the memory-derived x TP-degree x placement grid, and the tight-margin
  supplementary check, all real-compute, subprocess-per-scenario,
  `N_REPEATS=3` for the main grid.

One commit on `task-24-memory-planner`, stacked on
`task-23-memory-tp`; no `upstream/`, `src/engine/`, or predictor
changes, per this task's own acceptance criteria.
