# Task 23 — Memory against tensor parallelism

Branch: `task-23-memory-tp`, stacked on `task-22-which-binds`.

189 tests pass (measurement task, no new tests), and
`python3 tools/check_import_direction.py` exits 0.

All grid figures rest on **3 seeded runs** per cell (63 runs total); the
memory axis was reduced from task 22's 6 points to 3 (below the knee, at
it, above it) per this task's own S2 escape valve, to keep a 4-degree
TP x 2-placement axis affordable — the knee's own *shape* was already
established in task 22 and is not re-litigated here. The coupling check
(S1) is a mechanism/magnitude check, not a headline figure, and rests on
1 run per point plus a second, independent confirmation under an
explicit alternate profiling mode (S1). Real h800 compute profiles
throughout (Phi-tiny-MoE-instruct); every configuration confirmed
`enable_dummy_mode` is `False` before running.

---

## 1. Are KV capacity and tensor-parallel degree coupled in Frontier?

**Mechanically yes in Frontier's source; empirically no, in this
project's own model and workload.** Checked directly, not assumed, per
this task's own explicit instruction to check this before running the
grid:

- `frontier/scheduler/replica_scheduler/base_replica_scheduler.py`'s
  `elif not self._config.num_blocks:` only invokes
  `MemoryPlanner.get_num_blocks()` when a scheduler config's `num_blocks`
  is left at its dataclass default of `0`. `MemoryPlanner._get_parameter_memory_per_device()`
  calls into `ParamCounter`, which shards attention weights by
  `attn_tensor_parallel_size` for the `DECODE_ATTN` cluster specifically
  (`get_num_mlp_parameters_per_device()` returns `0` unconditionally for
  `DECODE_ATTN` — that cluster's per-device footprint is pure attention
  weight). **Confirmed by direct instantiation**: `ParamCounter(rc, ClusterType.DECODE_ATTN).get_num_parameters_per_device()`
  returns `671,088,640` at `attn_tp=1` and `100,663,296` at `attn_tp=8` —
  a real 6.67x shrink, exactly the mechanism this task's own S3 describes.
- **Yet running the actual simulator** (`num_blocks` left at `0`,
  `MemoryPlanner` invoked for real) gives the **identical** derived
  `num_blocks` (`106,596`) at every one of `attn_tp in {1,2,4,8}` — read
  back directly off the live scheduler
  (`sim._global_scheduler.get_cluster_scheduler(ClusterType.DECODE_ATTN).get_dp_replica_scheduler(*key)._config.num_blocks`),
  not inferred. Re-tested a second way, forcing Frontier's analytical
  (non-runtime-profiled) parameter-memory path on explicitly
  (`enable_runtime_non_kv_cache_overhead_profiling` +
  `use_analytical_param_memory`, both `True`): **still identical**,
  `106,596` at both `tp=1` and `tp=8`.
- The reason, inferred from the arithmetic rather than further traced
  line-by-line: this model's per-device attention-weight footprint
  (up to ~1.34 GB unsharded, ~200 MB at tp=8 — a delta of roughly 1 GB)
  is a small fraction of this replica's KV-cache-dominated memory budget
  (a budget large enough to floor to 106,596 blocks, i.e. capacity for
  tens of thousands of concurrent requests against this task's 32-request
  workload). A ~1 GB swing does not move a floor'd block count at that
  scale.

**Consequence for the grid.** Both axes are treated as independent in
this task's own main grid — not only because an explicit `num_blocks`
is needed to land on repeatable capacity points relative to task 22's
knee (the practical reason), but because *even Frontier's own
auto-derive path* shows no coupling for this model and workload. A
different model (larger attention weight footprint relative to a
tighter memory budget) could show a real trade; this one does not, and
that was checked rather than assumed either way.

## 2. The grid

Placement varies only DECODE_ATTN's own TP-group placement (packed:
one scale-up domain; split: half-and-half across two) — reused unchanged
from tasks 19-21's own `_placement`. PREFILL and DECODE_FFN stay packed
together in a separate domain throughout, so the ATTN-FFN M2N hop is
held fixed across every cell, per this task's own "keep everything else
fixed" instruction. `install(..., collective=True)` (task 20) makes
tensor-parallel communication placement-sensitive at all; without it,
task 19 already established Frontier's own profiled table would report
the same number regardless of placement.

| tp | placement | nb | batch | throughput (req/s) | tpot (ms) | tp_comm (Σms) | mean m2n/req (ms) |
|---|---|---|---|---|---|---|---|
| 1 | packed | 6 (below knee) | 2.00 | 29.857 | 37.3199 | 0.0000 | 13.7549 |
| 1 | packed | 30 (at/above knee) | 8.00 | 86.806 | 14.6052 | 0.0000 | 14.6986 |
| 1 | packed | 120 | 8.00 | 86.806 | 14.6052 | 0.0000 | 14.6986 |
| 2 | packed | 6 | 2.00 | 31.595 | 35.2262 | 29.4298 | 13.7549 |
| 2 | packed | 30/120 | 8.00 | 90.612 | 13.9539 | 7.8259 | 14.6986 |
| 2 | split | 6 | 2.00 | 22.559 | 49.5943 | 435.1181 | 13.7549 |
| 2 | split | 30/120 | 8.00 | 69.886 | 18.3178 | 112.5581 | 14.6986 |
| 4 | packed | 6 | 2.00 | 30.186 | 36.8978 | 87.3677 | 13.7549 |
| 4 | packed | 30/120 | 8.00 | 87.747 | 14.4305 | 22.5331 | 14.6986 |
| 4 | split | 6 | 2.00 | 14.094 | 79.7672 | 1297.7971 | 13.7549 |
| 4 | split | 30/120 | 8.00 | 47.600 | 27.2465 | 330.1171 | 14.6986 |
| 8 | packed | 6 | 2.00 | 27.230 | 40.9762 | 202.7827 | 13.7549 |
| 8 | packed | 30/120 | 8.00 | 81.473 | 15.6040 | 51.5021 | 14.6986 |
| 8 | split | 6 | 2.00 | **8.017** | **140.7305** | 3019.3766 | 13.7549 |
| 8 | split | 30/120 | 8.00 | 29.019 | 45.1854 | 761.4566 | 14.6986 |

**Regime boundaries, marked directly by the data, not inferred:**
**memory-bound** is every `nb=6` cell (`batch=2`, capped below the
workload's own 8-request potential, identical to task 22's own knee
shape) — and this holds at *every* TP degree and placement, unchanged.
**Memory-unconstrained** is every `nb=30`/`nb=120` cell: bit-identical to
each other in every column, at every TP degree and placement — the same
plateau task 22 found, and, per S1, unmoved by raising TP. Total
preemptions were re-checked directly at every `nb=6` cell across all 7
TP/placement combinations (21 further runs, 3 seeds each): **zero**,
everywhere — task 22's admission-queueing finding, not eviction,
generalises across this whole grid too.

`mean m2n/req` reproduces task 22's own **split**-pool figures exactly
(13.7549 ms / 14.6986 ms) at every cell — a direct cross-task consistency
check, not a coincidence: this task's placement always keeps
DECODE_FFN in a different domain from DECODE_ATTN, i.e. every cell here
sits in task 22's own "split" pool-placement configuration, never its
"colocated" one. `tp_comm` is reported as a whole-run sum (task 21's own
convention) since it is a relative, cross-cell comparison quantity, not
a per-request figure; it is not divided by request count here for the
same reason task 17's own trap warns against combining incompatible
denominators.

## 3. Where the crossover is

Framed as: how much does each axis add to inter-token latency, relative
to the `tp=1, packed, nb=30` baseline (14.6052 ms, unconstrained memory,
no tensor-parallel communication at all)?

| Source of the delta | Δtpot (ms) | Δtpot (%) |
|---|---|---|
| Memory alone (`nb=6` vs `nb=30`, tp=1 packed) | **+22.71** | +155.5% |
| TP alone, packed, tp=2/4/8 | −0.65 / −0.17 / **+1.00** | −4.5% / −1.2% / +6.8% |
| TP alone, split, tp=2 | +3.71 | +25.4% |
| TP alone, split, tp=4 | +12.64 | +86.5% |
| TP alone, split, tp=8 | **+30.58** | **+209.4%** |
| Both, tp=8 split + nb=6 (combined) | **+126.13** | +863.4% |

**Packed placement never crosses memory's own effect, at any TP degree
tested** — even at tp=8, the packed-TP penalty (+6.8%) is a rounding
error next to memory's own swing (+155.5%), matching this task's own
S1 caution about tuning toward "network binds": here, absent a
domain-split, it plainly does not.

**Split placement crosses it between tp=4 and tp=8.** At tp=4 split,
the network-alone penalty (+12.64 ms) is still below memory's own
worst-case swing (+22.71 ms) — memory would still be called the larger
single effect in that cell. At tp=8 split, the network-alone penalty
(+30.58 ms) **exceeds** memory's own worst-case swing. This is the
crossover the task exists to find: **a tensor-parallel group split
across scale-up domains, once its degree reaches 8 in this fabric and
workload, produces a larger inter-token-latency penalty than the most
severe memory-capacity cliff measured anywhere in this project.**
Degrees between 4 and 8 (e.g. 6) were not tested; the boundary is
located to within that gap, not narrower.

**The two do not simply add.** Combining tp=8-split with `nb=6`
(140.7305 ms) is more than double what summing the two individual
deltas alone would predict (22.71 + 30.58 = 53.29 ms over baseline →
67.90 ms predicted, vs. 140.73 ms actual — 2.07x the additive
prediction). The mechanism, stated rather than left as a number: at
`nb=6`, admission control caps the batch at 2 requests instead of 8
(S2's own table). Tensor-parallel ring-communication latency is
dominated by its fixed per-hop cost at these payload sizes (task 09's
own established latency-dominance finding for small transfers), which
does not shrink when the batch shrinks — but the amount of useful
compute available to amortise that fixed cost against does shrink,
proportionally. The same amortisation task 22 found working in
network's *favour* as batches grow (S4 of that report) works against
it here as batches shrink under memory pressure — one mechanism,
observed from both sides.

## 4. Whether task 22's conclusion survives

**Not as a blanket statement — it survives conditionally, and this task
is what earns the qualification.** Task 22 measured at `tp=1`, where
tensor-parallel communication is definitionally zero; on that
comparison memory's cliff was the only large effect in view, and calling
it dominant was correct *for that configuration*. This task's own grid
shows:

- At `tp=1`, or at any TP degree with the group **packed** into one
  scale-up domain, task 22's conclusion holds without qualification:
  memory's cliff (+155.5%) dwarfs tensor-parallel communication's own
  cost (at most +6.8%, at tp=8 packed).
  M2N's own share (task 22's actual subject) is unaffected by anything
  this task varied — it is fixed at the same ~13.75/14.70 ms across
  every cell here, exactly reproducing task 22's own split-pool numbers.
- At `tp=8`, **split** across scale-up domains, task 22's conclusion
  reverses: tensor-parallel communication (+30.58 ms) becomes the larger
  single effect, exceeding memory's own worst measured swing. Task 22's
  own third listed possible outcome — *"Network dominates once a group
  is split across domains. Then Task 22's conclusion is an artefact of
  its configuration and the framing needs revising"* — is the outcome
  that occurs, but only in that specific corner of the configuration
  space, not everywhere.

So: task 22's finding was not wrong for what it measured, and this task
does not overturn it in general. It does establish that "memory binds"
is a claim about a configuration space with a real boundary inside it,
not a global property of this project's model and workload — exactly
the reframing this task's own S1 anticipated as one of three possible,
genuinely different outcomes.

## 5. Do this project's network results remain the interesting ones?

**Yes, more so after this task than before it — not despite the
crossover, but because of it.** Before this task, every network result
in this project (tasks 09-22) was a placement-sensitivity finding of a
roughly fixed, bounded size (order 10-25%). This task adds a
configuration — high tensor-parallel degree, split across domains —
where that same mechanism's cost is no longer a bounded refinement on
top of compute and memory, but the single largest number in the whole
grid, larger than the most severe memory effect this project has
measured anywhere. A result that can flip from "a rounding error beside
memory" to "the dominant cost" purely as a function of *where ranks are
placed*, at a fixed compute budget and a fixed, adequately-provisioned
memory budget, is a stronger case for this project's own reason for
existing than any single fixed percentage would have been — it is
exactly the finding a hand calculation would not produce, since it
depends on how a specific ring interacts with a specific two-domain
split at a specific degree, not on any ratio available in a spec sheet.

## 6. Anywhere this specification is wrong

**Every quantitative claim in this task's own S1 fails to match this
project's actual prior findings — checked directly against both cited
reports, not assumed:**

- *"Sweeping KV cache capacity moved throughput by a factor of six and a
  half"* — task 22's own measured throughput ratio (its full sweep,
  either placement) is **2.76x-2.91x**, not 6.5x. Grepped directly;
  no combination of task 22's own numbers reaches 6.5x.
- *"The entire network cost was about a third of one percent of
  inter-token latency"* — task 22 never reports a figure near 0.33%
  under any framing checked (M2N as % of mean tpot: 3.1%-8.6%
  colocated, 36.9%-100.6% split; task 22's own "network penalty" metric:
  +18.44% to +24.16%).
- *"Task 21 measured a split four-way group contributing 1.27 ms against
  a 7.65 ms decode step — about a sixth, some fifty times the total
  network share in Task 22's configuration"* — grepped
  `docs/tasks/21-collective-patterns-report.md` directly for "1.27" and
  "7.65": no match anywhere. Task 21's own actual tp=4 figures are
  `tensor_parallel_communication_time` = 2.628864 ms (packed) /
  38.513664 ms (split) (task 20's measurement, reconfirmed unchanged in
  task 21); task 20's own inter-token-latency headline was "+5.126 ms
  at tp=4, ~88% over packed's 5.803 ms tpot." Neither pair matches
  "1.27 ms" or "7.65 ms."

This is a broader pattern than any single prior task's citation issue
(tasks 17, 19, 20, 21, 22 each had one inaccurate figure among otherwise
accurate ones): here, none of S1's three headline numbers is traceable
to this project's own reports. The qualitative framing S1 builds on top
of them — that network's role has never been tested against memory, and
that the three outcomes it lists are genuinely open — holds regardless,
and turned out to matter (S3-S5). But the specific numbers motivating
"why this now" should not be read as measured figures; this report's
own S2-S3 tables are the checked replacement.

- **A secondary, smaller point**: S2's own table asks for "total network
  cost as a share of a decode step." This report computes that two
  different ways (task 21's whole-run-sum convention in S2's table;
  task 22's per-request mean-tpot-relative convention via the Δtpot
  figures in S3) because the two are not interchangeable and the task
  text does not specify which is meant — exactly its own S6 "a ratio
  needs its denominator" trap, applied to its own instructions. Reported
  as a scope note rather than resolved one way, since resolving it
  silently would have picked one denominator over another without
  saying so.

- Otherwise the specification's structure — check the coupling question
  before running the grid, reduce the memory axis rather than drop the
  TP axis if cost demands a cut, report both throughput and latency,
  hold the pool placement fixed and vary TP placement instead, locate
  the crossover rather than just declare "which is bigger" — matched
  exactly what the investigation needed, including correctly
  anticipating that the answer would depend on configuration rather
  than settle the question either way.

## What shipped

- `tools/run_memory_tp_study.py` — the coupling check (S1), the
  memory x TP-degree x placement grid (S2), and the crossover analysis
  (S3), all real-compute, subprocess-per-scenario, `N_REPEATS=3`.

One commit on `task-23-memory-tp`, stacked on `task-22-which-binds`; no
`upstream/`, `src/engine/`, or predictor changes, per this task's own
acceptance criteria.
