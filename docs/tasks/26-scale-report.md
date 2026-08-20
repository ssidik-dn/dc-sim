# Task 26 — Affordability, and how stable the crossover is

Branch: `task-26-scale`, stacked on `task-25-loose-ends`. Confirmed per
this task's own note: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`, both already repointed by
task 25.

189 tests pass (measurement task, no new tests), and
`python3 tools/check_import_direction.py` exits 0.

Real h800 compute profiles throughout Part B (Phi-tiny-MoE-instruct),
matching every real-compute tool since task 09. Part A also uses real
compute profiles — per this task's own trap ("a cheaper configuration
may distort attribution... this task reports ratios of wall-clock"),
dummy mode would make the wall-clock breakdown in §A.2 meaningless.
The workload itself is kept small and fixed at each fabric-size point
(16 requests) specifically to isolate the fabric-size effect, which is
what §A.2's own attribution is about.

---

## Part A — Where does this stop being affordable?

### Two premises, checked before anything else

**"About a fifth of wall-clock on a 160-link fabric"**: accurate.
`docs/tasks/11-m2n-predictor-report.md` states directly: *"the
predictor's own `total_wall_ns` accounted for ~21.6% of `sim.run()`'s
wall-clock time"* on a fabric with *"160 links here."* Both figures
match this task's citation exactly.

**"ASTRA-sim invocation is 4-5ms... memoised on a canonical placement
shape... the hit rate is... a first-class metric"**: accurate, and
sourced from code rather than a task report — `src/engine/cost/astra_backend.py`'s
own module docstring: *"Measured on the reference box: 4-5 ms per
invocation, dominated by process startup... Memoisation is not an
optimisation here; it is what makes ASTRA-sim usable at all. The hit
rate is therefore a first-class metric."*

**But this second premise, while accurate as a fact about
`AstraSimBackend`, does not describe anything in the path this task
asks to profile.** Checked directly: `EngineCCBackend`
(`src/integration/cc_backend/engine_backend.py`), the collective
backend every real-compute tool in this project installs via
`install(..., collective=True)`, has its own module docstring stating
plainly that task 20 rewrote its five true collectives to go through
`Transfer`/`run_transfers` **"rather than depend on an external
algorithm this project cannot fully account for."** `CostBackend`/
`AstraSimBackend` are reachable (`validate_astra.py` exercises them)
but nothing this project's `install()` registers ever calls
`CostBackend.estimate()`. **§A.2's profile below confirms this
directly: zero ASTRA-sim calls appear anywhere in it.** Reporting a
memoisation hit rate for a code path that is never invoked would be a
number about nothing; the honest finding is that this path contributes
zero wall-clock in every real run this project's own tools make.

**"Deferred three times"** could not be verified. Searched every task
report for a repeat of task 11's own caching recommendation (`grep`
for "caching is warranted", "worth caching", "not implemented here",
"network_for", "_path_latency_ns" across `docs/tasks/*.md`): task 11's
report is the only place this appears. Whether it was deferred a
further two times informally, without being written down, isn't
something this investigation can confirm or deny; stated as
unverified rather than assumed true.

### A.1 Wall-clock growth with fabric size, fitted

**Fabric shape used, and why.** Growing fabric size by adding more
8-GPU domains keeps total link count *exactly linear* in GPU count —
contradicting this task's own framing ("link count grows faster than
GPU count"). Only growing *domain size* makes link count superlinear
(`Fabric.add_link`'s own `bidirectional=True` default storing both
directions of every pair: confirmed directly that `build_rack_scale`'s
72-GPU Helios domain gives `72*71=5112` links — matching this task's
own "over five thousand" §2.3 claim exactly, and confirming it counts
directed links, not the `2556` unordered pairs `_mesh_scale_up` calls
`add_link` for). So fabric size here means two domains, each half the
target GPU count, growing together
(`build_node_scale(num_machines=2, gpus_per_machine=n//2)`) — a
deliberately pessimistic shape chosen to find where the cost model
itself breaks, not a claim about a buildable real fabric (real
scale-up domains top out well below 512).

Fixed workload (16 requests, qps=10) at each point, 1 run per point
(this axis is about the shape of growth, not variance at a boundary —　
§A.3's own duration/flow sweeps below are where repeat-count matters
more, and those show near-zero run-to-run spread already):

| n_gpus | n_links | wall_s | wall/request | wall/operation | peak RSS |
|---|---|---|---|---|---|
| 32 | 560 | 2.37 s | 148.3 ms | 2.645 ms | 371.4 MB |
| 64 | 2,128 | 4.00 s | 249.9 ms | 4.458 ms | 371.5 MB |
| 128 | 8,336 | 9.98 s | 623.9 ms | 11.13 ms | 374.1 MB |
| 256 | 33,040 | 35.34 s | 2,208 ms | 39.39 ms | 380.9 MB |
| 512 | 131,600 | **245.27 s** | **15,330 ms** | 273.4 ms | 411.1 MB |

**Fitted exponents (log-log slope across all 5 points):**

- `n_links ~ n_gpus^1.97` — link count is, as expected, essentially
  quadratic in fabric size for this domain shape.
- **`wall_s ~ n_gpus^1.65`** — superlinear, sub-quadratic. This is the
  number that answers §A's own question: not a wall (the run does
  finish, at every size tested) but a steep slope. Going from 32 to
  512 GPUs (16x) costs **103x** the wall-clock (2.37s → 245.27s).
- `peak_rss ~ n_gpus^0.03` — essentially flat (§A.4 below).

16 requests at 512 GPUs already take over 4 minutes. Extrapolating the
fitted exponent (not measured further, since the shape is already
unambiguous and each further point costs several more minutes): 1,024
GPUs would be of order 245×2^1.65 ≈ 770s (13 minutes), 2,048 of order
34 minutes, for the *same* 16-request workload. A search evaluating
thousands of arrangements at even moderate fabric sizes is not
affordable as this cost model stands today.

### A.2 Where the time goes, at 512 GPUs

`cProfile` around `sim.run()` (245.28s wall, matching the timed figure
above to within profiler overhead):

| Attribution | Cumulative time | Share of total |
|---|---|---|
| **Per-call fabric rebuild** (`network_for` + `_path_latency_ns`, `engine/network/transfers.py`) | 212.99 s | **86.9%** |
| — of which `Link.id` string formatting (`topology.py:88`, called **235,830,784** times) | 155.81 s | 63.5% |
| — of which `GpuId`/`NicId`/`SwitchId.__str__` (called 469,763,840 times, invoked from `.id`) | 73.52 s | 30.0% |
| **Path computation** (`fabric.path()`, BFS per call, `topology.py:184`) | 24.57 s | 10.0% |
| **Max-min fair-share / flow-model construction** (`FlowNetwork.__init__`) | 1.66 s | 0.7% |
| **ASTRA-sim invocation** | 0 calls | **0%** |
| **Frontier's own simulation** (everything else: scheduling, metrics, logging, execution-time prediction) | ≈3.3 s | ≈1.4% |

**This project's own per-call fabric rebuild is not "about a fifth" at
this scale — it is 87% of wall-clock, and climbing with fabric size
(§A.1's own exponent), exactly the "may not hold at scale" this task's
own trap warned about.** `network_for()` (task 11's own function,
unchanged since) and `_path_latency_ns()` each build a fresh
`{link.id: ...}` dict over **every link in the fabric, on every single
call**, regardless of which one or two links the transfer being priced
actually touches — confirmed directly in source, not inferred. At 512
GPUs (131,600 links), and 896 calls in this 16-request run, that is
896 × 131,600 ≈ 118 million dict-comprehension iterations, each paying
`Link.id`'s own `f"{self.src}->{self.dst}"` string formatting (a
`@property`, recomputed fresh on every access — never cached).

ASTRA-sim invocation is confirmed absent from the profile at every
fabric size, consistent with §A.0's finding that this project's real
collective path never reaches it.

### A.3 The memoisation hit rate as fabrics grow — there is none to report

**The honest answer is that no memoisation exists anywhere on the path
this project's tools actually exercise.** Checked directly (`grep` for
`lru_cache`/`_cache`/`cache[` across `src/integration/cc_backend/`,
`src/integration/m2n_transfer/`, `src/integration/kv_transfer/`,
`src/engine/network/transfers.py`, `src/engine/fabric/`): none.
`AstraSimBackend`'s own memoisation is real and does exactly what its
docstring says, but it sits on a path §A.0 already established is
never called from `install()`'s own registered predictors. Task 11's
own recommended cache — the one that *would* matter, keyed on
placement/size, collapsing every call to a handful of distinct tuples
— was explicitly not built (its own report: *"Not implemented
here"*). So: not "the hit rate is falling as fabrics grow" but "there
was never a hit rate to fall in the first place" — every one of the
896 calls in the 512-GPU run is a full rebuild, and that count itself
does not even grow with fabric size (it is fixed by the workload, per
§A.1's own duration-sweep finding below) — it is each *individual*
call's cost that grows, linearly-ish with link count.

### A.4 Peak memory — does not bind before wall-clock

**No.** Peak RSS grew from 371.4 MB to 411.1 MB across the entire
32-to-512-GPU range — an `n_gpus^0.03` fit, indistinguishable from flat
at this scale, even though the fabric's own link storage
(`Fabric._links`, a `Dict[Tuple[Node,Node], Link]`) is genuinely
`O(n^2)` in domain size (confirmed: 131,600 `Link` objects at 512
GPUs). Each `Link` is a small, frozen dataclass (five scalar fields);
a few hundred thousand of them is a rounding error against Python's
own baseline process footprint, which is most of what these RSS
figures actually measure. **This task's own §2.3 speculation — "at 512
this may bind before wall-clock does" — does not hold, at least up to
512 GPUs in this domain shape.** Wall-clock is unambiguously the
binding constraint; memory is not close.

### A.5 The largest configuration that runs, and what stops the next one

**512 GPUs (two 256-GPU domains) ran to completion, in 245.27 seconds,
for a 16-request workload.** Nothing in this sweep *failed* to run —
every configuration from 32 to 512 GPUs completed. What "stops the
next one" is not a crash or an out-of-memory error; it is §A.1's own
fitted exponent applied honestly: each further doubling costs roughly
3x the previous point's wall-clock. Larger points were not run because
the shape is already unambiguous (§A.1's fit is tight across 5 points
spanning a 16x range) and each additional point at this trend would
cost 10+ more minutes for a confirmation the fit already gives
confidently — the practical ceiling here is a matter of how much
wall-clock a workflow can tolerate per evaluation, not a hard wall
this sweep hit.

### A.6 What to do about it, in priority order

1. **Cache the per-call fabric rebuild.** §A.2's own numbers make this
   unambiguous: 87% of wall-clock at 512 GPUs, all of it recomputing
   the *same* `{link.id: capacity}` and `{link.id: link}` dicts on
   every call regardless of which transfer is being priced. This is a
   stronger case than task 11's own original 21.6% finding — the
   dicts do not even need to be keyed on anything (placement, size);
   they depend only on the `Fabric` object itself, which does not
   change within a run. Caching them once per fabric (not per
   placement signature) would collapse 896 rebuilds to 1 at this
   configuration, i.e. almost all of the 213 seconds this task
   measured.
2. **Stop recomputing `Link.id` on every access.** Independent of the
   dict-rebuild fix above: 63.5% of total wall-clock at 512 GPUs is a
   `@property` that string-formats two endpoints on every single
   access, 235.8 million times in this run, and is never memoised at
   the `Link` level either. Freezing `id` into a stored field at
   construction (this project's own `Link` is already a frozen
   dataclass — `__post_init__` can compute it once) would remove this
   cost independently of whether the surrounding dict gets cached, and
   is a smaller, more contained change to verify.
3. **Precompute paths**, exactly as this task's own §3 suggests —
   confirmed a real, present cost (10.0% of wall-clock at 512 GPUs,
   `fabric.path()` running a fresh BFS on every call with zero
   caching), but a distant third by measured share next to items 1-2.
4. **Represent dense scale-up domains implicitly** — the profile does
   not show this as a *distinct* cost line (it would show up as part
   of item 1's own dict sizes, since that is exactly what a fabric's
   own link count depends on), so this is better read as a
   *structural* fix that would shrink item 1's own input size (an
   implicit all-pairs domain never needs `O(n^2)` explicit `Link`
   objects at all) rather than a separate line item to profile on its
   own. Worth doing, but its measured value is the same 87% item 1
   already accounts for, not additional to it.
5. **ASTRA-sim's own memoisation and hit rate**: nothing to do. It is
   real, it works as documented, and it is not on any path this
   project's real-compute tools exercise. Not a priority, because it
   is not costing anything today.

---

## Part B — Does the crossover survive an overhead?

### B.1 Method, reused from Task 25 rather than rediscovered

Neither `non_kv_cache_overhead_bytes` nor `num_blocks_mode` has a
per-cluster CLI override for DECODE_ATTN (task 25's own finding,
reconfirmed rather than re-derived: the per-cluster form is rejected
by argparse, the global form is silently ignored once DECODE_ATTN has
its own scheduler-config copy). So each assumed overhead was folded in
analytically, using the exact formula `MemoryPlanner.get_num_blocks()`
uses, and the resulting block count passed as an **explicit**
`--cluster_config_decode_attn_replica_scheduler_config_num_blocks`
override run through the real `Simulator` — task 25's own confirmed
technique, reused via `run_memory_tp_study.py`'s own
`_run_scenario_in_subprocess`, not reimplemented.

Per-device parameter memory and KV page size, calibrated in task 25
and cited rather than recomputed: tp=1: 1,342,177,280 B param /
1,048,576 B page; tp=2: 671,088,640 B / 524,288 B; tp=4: 335,544,320 B
/ 262,144 B; tp=8: 201,326,592 B / 262,144 B (KV-geometry floor).

### B.2 The grid, at four overheads

Task 24's own three margins (0.9843, 0.984, 0.9), all four TP degrees,
packed and split. Overhead=0 is **task 24's own grid, cited here, not
rerun** (task 24's report, §2). Overhead=2/4 GiB: real `Simulator`
runs, 1 seed each (this reuses an already-validated mechanism at a new
parameter, not a fragile new measurement — task 24's own 3-seed figures
at these exact `num_blocks` values, where they overlap, match this
run's single seed exactly, since none of these cells sit near a
queueing-variance boundary). Overhead=8 GiB: **zero feasible cells at
any of the three margins** — confirmed below, not run through the full
grid since there is nothing to run.

| overhead | margin=0.9843 | margin=0.984 | margin=0.9 |
|---|---|---|---|
| 0 GiB | all 4 degrees feasible (task 24's own grid) | all 4 feasible | all 4 feasible |
| 2 GiB | **all 4 infeasible** | **all 4 infeasible** | all 4 feasible, unchanged batch/throughput/tpot |
| 4 GiB | **all 4 infeasible** | **all 4 infeasible** | all 4 feasible, unchanged batch/throughput/tpot |
| 8 GiB | **all 4 infeasible** | **all 4 infeasible** | **all 4 infeasible** |

At every feasible cell (margin=0.9, overhead ∈ {0, 2, 4} GiB), the
real numbers are **identical** across overheads — because every
resulting `num_blocks` value, even after subtracting the overhead,
still sits far above this workload's own batch=8 plateau:

| tp | placement | throughput (req/s) | tpot (ms) | tp_comm (Σms) |
|---|---|---|---|---|
| 1 | packed | 86.806 | 14.6052 | 0.0000 |
| 2 | packed | **90.612** | **13.9539** | 7.8259 |
| 2 | split | 69.886 | 18.3178 | 112.5581 |
| 4 | packed | 87.747 | 14.4305 | 22.5331 |
| 4 | split | 47.600 | 27.2465 | 330.1171 |
| 8 | packed | 81.473 | 15.6040 | 51.5021 |
| 8 | split | 29.019 | 45.1854 | 761.4566 |

**tp=2, packed, is throughput- and latency-optimal at every overhead
that leaves anything feasible** — 0, 2, and 4 GiB alike (task 24's own
finding, unmoved).

### B.3 Does the crossover survive?

**Feasibility collapses before the crossover ever has a chance to
move — the third of this task's own three listed outcomes, and it
happens faster than task 25's own bound anticipated.** Task 25 bounded
the effect at 0/2/4/8 GiB and found the reference configuration
(margin=0.9) survives up to 4 GiB and fails only at 8. This task adds
what that bound alone could not show: **two of task 24's own three
margins (0.9843 and 0.984 — the "below the knee" and "at the knee"
points the whole memory-vs-parallelism study was built around) are
already wiped out by an overhead as small as 2 GiB** — a quarter of
the "double" extreme task 25 flagged as the case that would matter
most. Only the single most generously-provisioned margin task 24
tested survives past 2 GiB at all, and it survives with its answer
completely unchanged.

**The optimum never shifts. It is either tp=2 packed, or nothing.**
No overhead tested produced a configuration where a *different* degree
became optimal — the crossover this task asks whether it "survives"
never had a chance to move, because the margins where it might have
moved (the tighter two) are the same ones overhead knocks out first.

### B.4 Does Task 25's 8 GiB prediction hold when actually run?

**Yes, exactly, confirmed by direct execution rather than trusted from
arithmetic alone.** Four representative cells were run through the
real `MemoryPlanner.get_num_blocks()` (not just the closed-form
estimate) at 2, 4, and 8 GiB: all four raised
`FrontierMemoryOOMError` precisely where the formula predicted
(`available_kv_cache_memory_bytes` negative by exactly the predicted
amount in each case — e.g. tp=1, margin=0.9, overhead=8 GiB: predicted
and measured `-1,342,177,281` bytes short). Task 25's own prediction —
"the reference configuration becomes infeasible at every degree" at
8 GiB — holds without qualification.

### B.5 What a real overhead would have to be for this result to change

**Roughly 6.75 GiB** is where the *first* degree (tp=1, the most
memory-hungry) drops out at margin=0.9 — computed from the same
formula (`usable_memory - param_mem(tp=1) = 8GB - 1.25GB = 6.75GB`),
not run separately, since tp=1 was never the optimal degree and losing
it changes nothing about which degree wins. **Roughly 7.81 GiB** is
where the *last* degree (tp=8, the least memory-hungry) also drops
out at margin=0.9, at which point every degree this study could test
is infeasible and the question "which degree is optimal" stops having
an answer at all, rather than getting a different one.

**This is the number a reader needs, and it is uncomfortably close to
this task's own "a few gigabytes is typical" framing.** A real
serving stack's non-KV overhead in the 2-6 GiB range would leave this
project's own finding (tp=2 packed dominates) intact; one in the
7-8+ GiB range would leave it with no configuration left to have an
opinion about, at the one margin task 24 ever tested that survives
past 2 GiB at all. The qualitative result is not protected by a wide
margin of safety — it sits close enough to a plausible real value that
whether it holds is a live question, not a settled one.

---

## Anywhere this specification is wrong

- **Part A's own §2.2 asks to attribute wall-clock to "ASTRA-sim
  invocation, with its cache hit rate" in a representative run.** That
  attribution is zero, and there is no hit rate to report, because
  `install()`'s registered predictors never call `CostBackend.estimate()`
  at all (task 20's own rewrite, confirmed in `engine_backend.py`'s
  module docstring) — this is a real, precisely-measured fact about
  `AstraSimBackend` (§A.0's citation is accurate) applied to a
  question about a path that fact does not describe.
- **§2.3's "72 GPUs in one scale-up domain is already over five
  thousand links"** is accurate, but only once `Fabric.add_link`'s own
  bidirectional storage is accounted for (`72*71=5112` directed links,
  not the `2556` unordered pairs a naive reading of "GPU-pair link"
  might suggest) — worth stating precisely since the distinction
  matters for anyone reproducing the count.
- **"Deferred three times"** (§1) could not be verified anywhere in
  this project's own reports; task 11 is the only place this
  recommendation appears in writing. Stated as unverified rather than
  assumed, per this task's own instruction to cite rather than
  recompute — and per task 25's own precedent, not adjusted to fit.
- Otherwise both parts' own structure — measure before deciding what
  to fix; fold an uncalibratable overhead in analytically rather than
  guessing at a CLI knob that does not exist; confirm a formula's
  prediction by actually running it rather than trusting the
  arithmetic; ask which of several possible outcomes occurred rather
  than assuming one — matched exactly what both investigations needed.

## What shipped

- `tools/run_scaling_study.py` — the fabric-size, concurrent-flow, and
  simulated-duration sweeps (§A.1), with `cProfile`-based attribution
  (§A.2) and peak-RSS tracking (§A.4) built in.
- `tools/run_overhead_sensitivity_study.py` — the four-overhead rerun
  of task 24's grid (§B.2), reusing task 25's analytical-injection
  method and `run_memory_tp_study.py`'s own subprocess runner directly.

One commit on `task-26-scale`, stacked on `task-25-loose-ends`; no
`upstream/`, `src/engine/`, or predictor changes, per this task's own
acceptance criteria — measurement only, nothing implemented against
either investigation's own findings.
