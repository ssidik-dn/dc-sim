# Task 25 — Two loose ends

Branch: `task-25-loose-ends`, stacked on `task-24-memory-planner`.

189 tests pass (measurement task, no new tests), and
`python3 tools/check_import_direction.py` exits 0.

**An environment note, unrelated to this task's own content but worth
recording:** the session's working directory (`/work/dc-sim`) and
Frontier checkout (`/work/Frontier`) had been replaced by an empty,
freshly-created path; the real repository, with full history through
task 24, is at `/work/simulation/dc-sim` (Frontier at
`/work/simulation/Frontier`). All of this task's work was done there.
Sixteen existing tools hardcoded the old `/work/Frontier` path
(`FRONTIER_ROOT = Path("/work/Frontier")`); this was a mechanical,
one-line-per-file fix (not a scope change) needed before any tool —
old or new — could run at all, so it is included in this branch's
commit.

---

## Part A — Why did placement barely matter?

### What Task 24's split arrangement actually did

Checked directly on a representative cell (`tp=4`), not inferred from
the placement helper's own docstring:

```
--- tp=4 split=False (packed) ---
  TP group GPUs: [GpuId(1,0), GpuId(1,1), GpuId(1,2), GpuId(1,3)]
  TP group_shape: (4,)                    # one domain
  crosses_scale_up_boundary: False
  ATTN pool domains: {1}  FFN pool domains: {0}

--- tp=4 split=True ---
  TP group GPUs: [GpuId(1,0), GpuId(1,1), GpuId(2,0), GpuId(2,1)]
  TP group_shape: (2, 2)                  # two domains, evenly
  crosses_scale_up_boundary: True
  ATTN pool domains: {1, 2}  FFN pool domains: {0}
```

Of the 6 pairs in the TP group's full pairwise exchange, 4 cross a
domain boundary (`fabric.domain_of` disagreeing on the two endpoints) —
a measured **66.7%** cross-domain fraction, using exactly the
`Placement.group_shape`/`crosses_scale_up_boundary`/`induced_links`
primitives task 15 built and this task's own spec points to.

**The parallel group was divided, not just the pools.** DECODE_FFN and
PREFILL stayed packed in domain 0 throughout (matching every task
23/24 cell); DECODE_ATTN's own TP ranks went from one domain (packed)
to two, evenly (split). This rules out the specification's own
"likeliest explanation" (pools separated, groups left whole) — that
branch does not apply here, checked rather than assumed.

### Which figure the 0.03–0.30% number should be compared to, and whether it holds up

**The split arrangement divides the same kind of thing Task 21 measured
(a tensor-parallel group across domains), not the kind Task 22 measured
(pools in different domains) — so if Task 24's placement axis is
comparable to either earlier figure, it is the ~48%/88% one, not the
~15% one.** And task 24's *own real numbers*, re-read directly from its
own report (§2's grid) rather than from this task's summary of them,
say exactly that:

| tp | packed tpot | split tpot | increase |
|---|---|---|---|
| 2 | 13.9539 ms | 18.3178 ms | +31.3% |
| 4 | 14.4305 ms | 27.2465 ms | **+88.8%** |
| 8 | 15.6040 ms | 45.1854 ms | +189.6% |

tp=4's own +88.8% increase is a near-exact match to task 20's own tp=4
figure (+88.3%, `docs/tasks/20-collective-backend-report.md`,
reconfirmed unchanged in task 21) — this project's placement-penalty
measurements for a divided TP group agree closely across tasks 20, 21,
23, **and** 24, at the same degree.

**The "0.03% to 0.30%" figure this task attributes to Task 24 does not
appear anywhere in Task 24's own report, and does not match any
quantity computable from its raw results.** Checked directly
(`grep -n "%" docs/tasks/24-memory-planner-report.md`): the smallest
percentage anywhere in that report is `+0.2%` (§2.3's tp=4→8
KV-geometry-floor finding — a *different* comparison, TP degree at
fixed placement, not packed-vs-split at fixed degree). Every
packed-vs-split comparison actually computed in task 24 — at every one
of the three margins its own grid swept, at every TP degree — lands in
the 31–190% range shown above, identical across margins because task 24
found the memory axis and the placement axis independent for TP≥2 (its
own §2 finding). There is no reading of task 24's real numbers that
produces a figure between 0.03% and 0.30%.

### Are the three figures now consistent?

**Two of them were never in tension: ~15% (pools split) and ~48-88%
(a TP group split) measure different things and both stand, unchanged,
confirmed again here by direct placement inspection.** The apparent
three-way contradiction this task opens with is not between those two —
it is between them and a citation to task 24 that task 24's own report
does not support. Once task 24's *real* numbers are used instead of the
"0.03–0.30%" figure, there is no contradiction left to resolve: task
24's split arrangement measured the same divided-TP-group quantity
task 20/21 measured, at the same degree, and got a matching answer.

**This task's own known trap ("a null result here is a good outcome...
do not look for a defect that is not there") does not quite fit what
was found, and that is worth being precise about.** The null result
this task anticipated — pools separated, groups whole, task 24 measuring
something genuinely smaller — did not happen; the group was divided.
But no defect in Frontier's collective backend or this project's own
`EngineCCBackend` was found either: task 24's actual measurements are
large and consistent with tasks 20/21/23. The loose end here was a
mis-cited summary of task 24, not a measurement or implementation bug —
worth stating plainly rather than either forcing a reconciliation or
manufacturing a defect that would make the task's own framing pay off.

---

## Part B — Calibrate the memory overhead

### 1. What `memory_planner_profiled` needs, and whether it's available

**It needs a real model loaded onto real GPU hardware — categorically
unavailable in this project's environment.** Read directly, not
inferred from the mode's name:
`enable_runtime_non_kv_cache_overhead_profiling=True` (required to
actually *measure* anything; it defaults `False`) triggers
`estimate_non_kv_cache_profile()`
(`frontier/profiling/non_kv_cache_overhead/runtime_estimator.py`),
which calls `initialize_model_parallel()`, builds a real
`torch.nn.Module` via `_build_runtime_profile_model`, and takes CUDA
memory snapshots (`MemorySnapshot`) before and after loading it. This
project is a pure analytical/profiled-timing simulator with no GPU
execution anywhere else in its own real-compute path — collecting this
would mean provisioning real accelerator hardware and this model's
actual checkpoint, running a real forward pass, and measuring real
CUDA memory — infrastructure outside anything this project has used
in 24 prior tasks.

**Absent that flag, `"memory_planner_profiled"` is not doing anything
different from plain `"memory_planner"`, and this project has, in
effect, only ever used the un-profiled one.** `non_kv_cache_overhead_bytes`
stays at its static `0` default in both modes unless the runtime path
above runs. Task 24's own report already stated this
(`enable_runtime_non_kv_cache_overhead_profiling` defaults `False`);
this task confirms *why* — the alternative needs real hardware this
project doesn't have.

### 2. Is 21 GB plausible? — the premise does not match Task 24 as run

**Task 24 never used a 21 GB overhead, never used plain
`"memory_planner"`, and never tested a device smaller than h800 (80 GB)
— all three checked directly against task 24's own report and tool.**
`non_kv_cache_overhead_bytes=0` is quoted verbatim inside task 24's own
S2.2 OOM error message (`non_kv_cache_overhead_bytes=0`); its S2.1
states plainly that `num_blocks_mode` defaults to
`"memory_planner_profiled"` and that this project relied on that
default throughout; only `h800` was ever passed as a device. This
task's own Part B premise — 21 GB, from `memory_planner`, "more than
half of the smaller device considered" — describes a configuration
task 24 did not run.

**The real, checked number is the opposite kind of problem: the
overhead used throughout task 24 was zero, not large.** Every capacity
figure in task 24's grid assumes a decode replica's only non-KV memory
cost is the attention-weight shard itself — no activation buffers, no
CUDA workspace, no framework overhead. That is not "a conservative
default rather than a measurement" in the direction Part B's own §2
anticipates (a suspiciously *large* number); it is the reverse — a
default of exactly zero, which is optimistic relative to any real
serving stack, including the "few gigabytes" this task's own §2 names
as typical.

### 3. How the crossover moves, bounded rather than assumed

Since no real overhead value exists anywhere in this project to halve
or double, this section bounds the effect of *introducing* one, over an
illustrative range matching this task's own "a few gigabytes" framing
(0, 2, 4, 8 GiB) — not a calibrated figure, stated as such.

**The mechanism, established once and reused for every degree:** the
point past which a device is infeasible for a given TP degree is
*usable memory = parameter memory(tp) + overhead*, exactly (checked
against `memory_planner.py`'s own formula, and against task 24's own
confirmed-OOM behavior at overhead=0). Parameter memory per device,
calibrated directly (not estimated) for this model at each degree:

| tp | parameter memory | KV page size (all layers) |
|---|---|---|
| 1 | 1.2500 GB | 1.000 MiB |
| 2 | 0.6250 GB | 0.500 MiB |
| 4 | 0.3125 GB | 0.250 MiB |
| 8 | 0.1875 GB | 0.250 MiB (floor, task 24's own tp=4→8 finding) |

Infeasibility boundary (usable device memory, GB) by assumed overhead:

| overhead | tp=1 | tp=2 | tp=4 | tp=8 | tp=1↔tp=2 gap |
|---|---|---|---|---|---|
| 0 GiB (task 24's actual assumption) | 1.25 | 0.625 | 0.3125 | 0.1875 | 0.625 GB |
| 2 GiB | 3.25 | 2.625 | 2.3125 | 2.1875 | 0.625 GB |
| 4 GiB | 5.25 | 4.625 | 4.3125 | 4.1875 | 0.625 GB |
| 8 GiB | 9.25 | 8.625 | 8.3125 | 8.1875 | 0.625 GB |

**The qualitative finding is robust; the specific crossover point is
not.** The tp=1-to-tp=2 gap (0.625 GB — exactly half of tp=1's own
weight shard, since tp=2 halves it) stays fixed regardless of overhead:
raising TP degree relieves memory pressure by the same fixed amount no
matter what else is competing for that memory. But **task 24's own
"clearly unconstrained" reference point (`margin=0.9`, 8 GB usable)
sits *below* every one of the four degrees' own boundary at 8 GiB
overhead** — meaning a plausible, non-fabricated overhead assumption
(not even the largest device task 24 tested) would make that entire
configuration infeasible for tp=1, 2, 4, *and* 8 alike, not just shift
which degree wins.

**Confirmed at one extreme with a real run, not only the formula.**
At `margin=0.9` (8 GB usable) with a 2 GiB overhead folded in
analytically (`--cluster_config_decode_attn_replica_scheduler_config_num_blocks`
set to the resulting derived count directly, since neither
`non_kv_cache_overhead_bytes` nor `num_blocks_mode` has a per-cluster
CLI override at all — confirmed by argparse rejecting both, the same
gap Part A's own gpu_memory_utilization finding already established):

| tp | num_blocks (2 GiB overhead folded in) | batch | tpot |
|---|---|---|---|
| 1 | 4,865 | 8.00 (unconstrained) | 14.6052 ms |
| 2 | 11,006 | 8.00 (unconstrained) | 13.9539 ms |

Both land exactly at task 24's own zero-overhead plateau values — the
formula's prediction (still comfortably unconstrained at 2 GiB) matches
real system behaviour. The 8 GiB extreme was not run directly (it
predicts `FrontierMemoryOOMError`, and that exact exception path was
already confirmed live in task 24's own S2.2 and reconfirmed by this
task's Part A investigation using the identical formula) — running it
again would reconfirm a mechanism already established twice, not learn
anything new, which is the reason task 25 permits bounding rather than
full profiling.

## Anywhere this specification is wrong

- **Part A's own "0.03% to 0.30%" figure, attributed to Task 24**: not
  found anywhere in `docs/tasks/24-memory-planner-report.md`, and not
  reproducible from any packed-vs-split comparison in task 24's own
  raw grid (every real one is 31–190%). This is the load-bearing
  citation error this task's own investigation turned up.
- **Part B's own premise — "around 21 GB... more than half of the
  smaller device considered... came from `memory_planner`"**: task 24
  used `non_kv_cache_overhead_bytes=0` throughout (quoted verbatim in
  its own S2.2 OOM message), the default `num_blocks_mode` (
  `"memory_planner_profiled"`, not `"memory_planner"`), and only ever
  tested `h800` — no second, smaller device was ever considered. None
  of the three specifics in this premise matches task 24 as actually
  run.
- Otherwise the specification's own structure for both parts —
  establish the mechanism on a representative cell before trusting a
  summary table; ask which earlier figure a new one is comparable to
  rather than just whether numbers match; treat a null result as a
  legitimate outcome; bound an uncalibratable parameter rather than
  profile it; explicitly warn against forcing reconciliation — matched
  exactly what this investigation needed, and both loose ends resolved
  the way the task's own known traps said was the more likely, correct
  outcome: no defect, and a bound rather than a false precision.

## What shipped

- No new tool. Part A's placement inspection and Part B's calibration
  reused `tools/run_memory_tp_study.py`'s and
  `tools/run_memory_planner_study.py`'s own helpers
  (`_build_and_install`, `_run_scenario_in_subprocess`) directly, plus
  `engine.placement.placement.Placement`'s own task-15-era primitives —
  no new measurement code needed writing.
- `tools/*.py` (16 files): `FRONTIER_ROOT` repointed from the
  now-nonexistent `/work/Frontier` to `/work/simulation/Frontier` — an
  environment fix, not a task-25 deliverable, needed before any tool
  could run at all in this session.

One commit on `task-25-loose-ends`, stacked on
`task-24-memory-planner`; no `upstream/`, `src/engine/`, or predictor
changes, per this task's own acceptance criteria.
