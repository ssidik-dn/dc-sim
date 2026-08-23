# Task 30 — Cache paths, and measure growth honestly

Branch: `task-30-path-cache`, stacked on `task-29-affordability`.
Paths confirmed per task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`.

189 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0.

Every figure quoted from task 29's own report in this task's own §1
matches `docs/tasks/29-affordability-report.md` exactly, including the
per-doubling correction — checked by direct computation
(`log2(ratio)` on that report's own before/after wall-clock columns)
before touching anything, not assumed.

---

## 1. The cache design, and whether `route()` needed it

**Memoised on the ordered `(a, b)` GPU pair, on the `Fabric` instance
itself** — `Fabric._path_cache: Dict[Tuple[GpuId, GpuId], List[Link]]`,
populated on first computation inside `path()`, invalidated by
`add_link` (`src/engine/physical/topology.py`), the exact pattern task
29 established for `link_index()`/`capacity_index()` and reused rather
than re-derived, per this task's own trap about not relying on the
earlier finding alone: `add_link`'s invalidation was re-added
explicitly for this cache too, not left to task 29's own comment to
cover it by implication.

**Confirmed, not assumed, that placement is not part of the key**:
`path(self, a: GpuId, b: GpuId)`'s own signature takes only GPUs — no
`Rank`, no `Deployment`, no `Placement` object anywhere in it. One
cache, keyed purely on the fabric's own graph, serves every placement
built on the same `Fabric` object.

**`route()` does not need the same treatment — confirmed by grep, not
assumed.** `_links_for()` (`engine/network/transfers.py`) only calls
`fabric.route()` when `mode is not FabricMode.SINGLE_PATH`; every real
caller — `run_transfers()`, `isolated_durations()`, and every place in
`src/integration/` that reaches either — defaults to, and never
overrides, `FabricMode.SINGLE_PATH`
(`grep -rn "FabricMode\.\|mode=FabricMode" src/integration/` returns
nothing). `route()`'s own `PER_FLOW_ECMP` branch (which calls
`equal_cost_paths()`, the function that actually re-walks the graph
similarly to `path()`) is simply never reached by anything this task's
sweep or acceptance checks exercise, confirmed by its total absence
from every profile taken (before or after). Left untouched.

## 2. Cache size in a real run

**Small — a few hundred entries, confirmed by the same profile that
shows the cache working.** At 512 GPUs, `fabric.path()` no longer
appears anywhere in the top 60 functions of a fresh `cProfile` trace of
the exact same 16-request run task 29 profiled (it did, at 26.19s,
80% of that run's total, before this task). The deployment this sweep
uses places exactly three ranks (`PREFILL`, `DECODE_ATTN`,
`DECODE_FFN`) on three fixed GPUs across two domains, and every M2N
transfer in a 16-request run resolves to one of a small, fixed number
of (source, destination) GPU pairs — the same reasoning task 11's own
report already applied to the *predictor call* count, now applying
identically to the *distinct path* count underneath it. A fabric of
512 GPUs has on the order of a quarter of a million ordered pairs
available; this run touches a handful of them. The cache is trivial in
size regardless of fabric size, because touching most of a fabric's
ordered pairs was never something any placement in this project
exercises — only a bounded number of source/destination pairs are ever
priced, no matter how large the fabric grows around them.

## 3. The new sweep, with per-doubling exponents beside task 29's

Same fabric shape, same fixed 16-request workload, same five sizes.

| n_gpus | task 29 (after both) | this task (+ path cache) |
|---|---|---|
| 32 | 1.89 s | 1.82 s |
| 64 | 2.06 s | 1.84 s |
| 128 | 2.72 s | 2.02 s |
| 256 | 5.81 s | 2.57 s |
| 512 | 32.72 s | **5.87 s** |

(Both columns re-run fresh in this task, side by side with the
acceptance checks below, rather than only quoted from task 29's report
— task 29's own figures reproduced within run-to-run noise: 1.92s,
2.05s, 2.76s, 5.67s, 33.34s measured again here before this task's
change.)

**Per-doubling exponent, before and after this task's own change**
(the number this task exists to make the tooling report correctly,
per its own trap about a single fit hiding a convex curve):

| doubling | before (task 29 state) | after (path cache) |
|---|---|---|
| 32 → 64 | +0.12 | **+0.02** |
| 64 → 128 | +0.40 | **+0.13** |
| 128 → 256 | +1.09 | **+0.35** |
| 256 → 512 | +2.49 | **+1.19** |

Global fit, for reference only (and reported alongside the per-doubling
figures in the tool's own output now, not instead of them): `n_gpus^0.39`.
**The per-doubling figures are what describes behaviour at scale, and
they still rise toward the top of the range — the curve is still
convex, just far shallower than before.** The largest doubling (256→512)
dropped from +2.49 to +1.19: a real, large improvement, not to "flat,"
which would misstate what actually happened the same way task 26's own
single global fit once did.

## 4. The new dominant cost — profiled fresh, not inferred

**Neither `path()` nor the dict-rebuild task 29 fixed. A fresh
`cProfile` trace of the same 512-GPU, 16-request run:**

```
5.817s total (cProfile overhead included; 5.872s timed wall-clock)

isolated_durations                          1.565s  (26.7%)
  network_for                               1.355s  (23.1%)
    FlowNetwork.__init__ (engine/network/model.py:76)   1.321s  (22.5%)
  [fabric.path() -- no longer appears in the top 60 at all]
Frontier's own scheduling/logging/execution-time prediction  ~4.3s  (73.3%)
  (Python's own `logging` module alone: ~0.92s cumulative, 38,378 calls)
```

**`FlowNetwork.__init__`'s own `self.capacity = dict(capacity)` is now
the largest cost tied to fabric size in this project's own code** —
confirmed directly, not assumed from the exponent still creeping up at
the top of the range. `network_for()` hands it the *cached* capacity
map (task 29's own fix), but `FlowNetwork`'s constructor still makes
its *own* defensive copy of whatever dict it receives, and that copy
is `O(n_links)` regardless of whether the source dict was itself
freshly built or reused. This is exactly why the largest doubling
(256→512) still shows `+1.19` rather than something closer to zero:
copying a 131,600-entry dict, 896 times, is real, present cost that
neither this task nor task 29 touched — `FlowNetwork` is squarely the
stateful, event/completion-owning object this project's own zone rules
treat with more care, and its own `dict(capacity)` copy is a
correctness-motivated line (so that no caller can mutate the shared
cache through a `FlowNetwork` instance), not a rebuild-for-no-reason
bug the way `network_for`'s old dict comprehension was. Left as a
finding, not a fix — precisely the kind of change task 29's own zone
question and this task's own scope (path caching, not structural or
contention-model changes) both point away from doing here.

**Frontier's own overhead, including plain Python `logging`, is now
the *majority* of total wall-clock (73.3%) — larger than every
category this project's own code contributes, combined.** This was
never true at any earlier measurement in this line of work (task 26:
~1.4%; task 29: still a small minority) — it only became true once
this project's own contribution shrank enough to expose it. Not
addressed here (out of scope, and not this project's own code to
optimise), but worth naming as the shape of what's left once §2.1's
and §2.2's targets, and now this task's, are gone.

## 5. Whether search is affordable, in arrangements-per-minute

**Yes, at every fabric size this sweep reaches, assuming a search
evaluation looks like this same fixed 16-request workload — measured,
not estimated:**

| n_gpus | wall_s per evaluation | evaluations / minute |
|---|---|---|
| 32 | 1.82 s | ~33 |
| 64 | 1.84 s | ~33 |
| 128 | 2.02 s | ~30 |
| 256 | 2.57 s | ~23 |
| **512** | **5.87 s** | **~10** |

**At 512 GPUs: roughly 10 arrangements per minute, ~600 per hour.**
This is a real, concrete answer to what this whole line of work (tasks
26, 29, this one) was for — a search over thousands of arrangements at
512 GPUs is now a matter of hours, not the ~17 hours task 26's own
original 245.27s/evaluation figure would have implied for the same
count, and nowhere near the "single run takes an hour" scenario task
26's own §1 warned building a three-tier fabric into.

**This measurement is itself conservative for a real search loop.**
Every number above rebuilds a fresh `Fabric` (and therefore a cold
path/link/capacity cache) per evaluation, because that is what
comparing distinct fabric sizes requires. A real placement search holds
the *fabric* fixed and varies *placement* across many evaluations —
exactly the case task 30's own §1 (`path()` "keyed on GPUs, not ranks
or placement") makes the cache serve for free: after the first
evaluation warms the cache, every subsequent evaluation against the
same `Fabric` object pays only for whichever *new* GPU pairs that
specific placement introduces, typically a small fraction of the
pairs already cached from prior evaluations on the same fabric. The
figures above are a fair lower bound on throughput, not an optimistic
one.

## 6. Anywhere this specification is wrong

**Nothing.** Every figure quoted from task 29's report in this task's
own §1 — the wall-clock table, the per-doubling correction (+0.12,
+0.40, +1.09, +2.49), the "before the change the final doubling was
2.79" figure — matches `docs/tasks/29-affordability-report.md` exactly
and recomputes exactly from its own before/after columns
(`log2(245.27/35.34) = 2.795`). This is the second task in a row (after
task 29) whose own opening citations held up completely against their
stated source.

## What shipped

- `src/engine/physical/topology.py` — `Fabric.path()` memoises on the
  ordered `(a, b)` GPU pair (`_path_cache`), invalidated by `add_link`
  alongside `link_index()`/`capacity_index()`'s own invalidation.
- `tools/run_scaling_study.py` — `_per_doubling_exponents()`, reported
  as the primary growth figure in `_fabric_size_sweep()`'s own output,
  with the single global fit kept only as an explicitly-labelled
  secondary reference — the tooling fix this task's own §3 asked for,
  so the convex-curve mistake does not have to be caught by hand again.

One commit on `task-30-path-cache`, stacked on
`task-29-affordability`; nothing under `upstream/` or
`src/integration/` touched. Every acceptance-table figure — collective
backend tp=2/4/8, the memory grid at margin 0.9, the M2N
colocated/split comparison — reproduces bit-identical before and after,
checked side by side.
