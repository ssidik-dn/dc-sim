# Task 32 — Search, over degree and placement

Branch: `task-32-search`, stacked on `task-31-intervals`. Paths
confirmed per task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`.

189 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. This task adds one new file under `tools/` only — nothing
under `src/engine/` or `src/integration/` changed. All three citations
in this task's own §1 (the 5.87s/512-GPU figure, the ~1.3%/6.4%/0.2%
noise-floor figures, the determinism finding) were checked directly
against `docs/tasks/30-path-cache-report.md` and
`docs/tasks/31-intervals-report.md` before anything else — all three
match exactly.

---

## 1. The seed policy

**Each arrangement is evaluated once, in the deterministic
configuration.** Task 31's own §1.3 establishes that this project's
configuration (fixed request lengths, arrivals submitted at once) has
no seed-dependent input at all — a second evaluation of the same
arrangement would reproduce the first to the last digit, not add
information. Evaluating once is therefore not a shortcut here; it is
the whole of the information available, exactly as this task's own §1
frames it.

**The winner's margin was re-run with genuine seed variance
regardless** — task 31's own `seed_stats` module (staggered arrivals,
matching request-generator seed), 20 seeds, for the top 3 candidates —
so the ranking search produced is checked against realistic noise
before anything is called a winner (§4).

## 2. The size of the space

**Fabric**: 5 scale-up domains of 4 GPUs each (20 GPUs total) — small
enough that every evaluation runs in a few seconds (task 29/30's own
fitted growth), large enough that `tp=8` (needs more GPUs than one
domain holds) has a genuinely varied set of reachable shapes rather
than a single forced answer.

**Feasibility filter, not a dimension** (task 24/28's own finding, per
this task's own §2): `--cluster_config_decode_attn_replica_config_memory_margin_fraction 0.992`
throughout — task 28's own established point. Computed from the same
calibrated formula tasks 25/26/28 validated against real
`MemoryPlanner` behaviour, cited rather than re-derived:

| tp | feasible? | derived `num_blocks` |
|---|---|---|
| 1 | **no** — parameter memory alone exceeds the budget | — (rejected before any placement generated or evaluated) |
| 2 | yes | 30 |
| 4 | yes | 1,341 |
| 8 | yes | 1,853 |

**Candidate placements, generated from this project's own existing
policies** (`packed`, `spread`, `fragmented(seed=0..59)`, plus one
explicit "packed-if-it-fits" reference — `packed()`'s own rank
ordering gives DECODE_ATTN's group a one-slot offset from PREFILL, so
it never reaches a clean single-domain shape on its own even when
`attn_tp <= 4`; added explicitly for the same reason tasks 19-31 built
PREFILL/FFN's own placement by hand rather than relying on a
deployment-wide policy), deduplicated by `Placement.group_shape()` of
DECODE_ATTN's own TP group before evaluating any of them:

| tp | candidates | distinct shapes | shapes |
|---|---|---|---|
| 2 | 63 | 2 | `(2,)`, `(1,1)` |
| 4 | 63 | 5 | `(4,)`, `(3,1)`, `(2,2)`, `(2,1,1)`, `(1,1,1,1)` |
| 8 | 62 | 9 | `(4,3,1)`, `(4,2,2)`, `(4,2,1,1)`, `(3,3,2)`, `(3,3,1,1)`, `(3,2,2,1)`, `(3,2,1,1,1)`, `(2,2,2,2)`, `(2,2,2,1,1)` |
| **total** | **188** | **16** | — |

**188 candidate placements collapsed to 16 distinct shapes — an 11.8x
reduction.** This is exactly what makes exhaustive search of the whole
space affordable rather than merely small: evaluating every *shape*
once, at a few seconds each, is a two-minute search; evaluating every
*candidate placement* the generators actually produced would have been
almost 12x that for no additional information, since `group_shape()`
guarantees isomorphic placements cost identically (task 15's own
invariant). No `tp=8` shape reaches a single domain at all — `4 > 4`
is false for none of the partitions of 8 into parts of at most 4 that
also respect the 5-domain limit, so the fabric's own geometry rules
out "packed" as a reachable shape for this degree, structurally, not
by search failing to find it.

## 3. The best arrangement found, and its margin

**Exhaustive, all 16 feasible shapes, each evaluated once:**

| rank | tp | shape | mean tpot (ms) | throughput (req/s) | SLO attainment |
|---|---|---|---|---|---|
| **1** | **2** | **`(2,)`** | **11.6803** | **107.171** | **0.750** |
| 2 | 4 | `(4,)` | 14.4305 | 87.747 | 0.500 |
| 3 | 2 | `(1,1)` | 18.3178 | 69.886 | 0.500 |
| 4 | 4 | `(2,1,1)` | 24.9729 | 51.805 | 0.250 |
| 5-7 | 4 | `(3,1)`/`(2,2)`/`(1,1,1,1)` | 27.2465 | 47.600 | 0.250 |
| 8-9 | 8 | `(3,2,1,1,1)`/`(3,2,2,1)` | 42.9118 | 30.530 | 0.000 |
| 10-16 | 8 | every other tp=8 shape | 45.1854 | 29.019 | 0.000 |

**Winner: `tp=2`, packed (`(2,)`).** Runner-up: `tp=4`, packed (`(4,)`),
+23.6% in the single deterministic evaluation
((14.4305−11.6803)/11.6803).

**Seeded re-run of the top 3, n=20, genuine arrival variance
(task 31's own `seed_stats`):**

| rank | tp, shape | mean tpot (ms) | 95% CI half-width |
|---|---|---|---|
| 1 | tp=2, `(2,)` | 3.2378 | ±1.56% |
| 2 | tp=4, `(4,)` | 4.4546 | ±1.31% |
| 3 | tp=2, `(1,1)` | 6.2087 | ±1.19% |

**The ranking survives, and the margin is larger under real noise, not
smaller: (4.4546−3.2378)/3.2378 = +37.6%.** (Absolute values differ
from the deterministic pass because streaming arrivals are a
genuinely different workload regime, not the same one with error bars
— task 31's own report makes the same distinction.) None of the three
95% confidence intervals overlap.

## 4. Whether the margin exceeds the noise floor

**Yes, by roughly 24-29x.** Task 31's own noise floor at 20 seeds, flat
region: about 1.3% on per-token latency. The margin between the winner
and the runner-up is 23.6% (deterministic pass) to 37.6% (seeded
re-run) — both a full order of magnitude past the point where this
project could no longer tell two arrangements apart. This search's own
ranking is not a case of preferring a difference the noise floor could
have produced by chance; it is checked, not merely large-looking.

## 5. SLO attainment for the top arrangements

**SLO stated explicitly, since none exists anywhere in this project's
prior record** (checked: `grep` for "SLO"/"service level"/"latency
target" across every report returns nothing) — **15 ms per token**,
chosen because it sits inside the range this project's own real h800
measurements have actually produced across tasks 22-31 (roughly 3-45
ms/token depending on configuration), not because it is a target this
project or any external spec has committed to. Reported as
illustrative, not authoritative.

**The objective and the SLO constraint agree on the winner here — no
divergence to report.** `tp=2` packed has both the lowest mean tpot
*and* the highest SLO attainment (0.750) of every evaluated
arrangement; `tp=4` packed is second on both (0.500, tied with `tp=2`
split). Optimising mean latency directly and optimising SLO
attainment directly would have chosen the same arrangement in this
search. This is reported plainly rather than manufactured into a
disagreement — the two objectives not conflicting is itself the
finding, per this task's own trap about not tuning the objective to
make the search look interesting.

## 6. Anywhere this specification is wrong

**Nothing.** Every figure this task's own §1 quotes from tasks 30 and
31 matches those reports exactly, checked by direct `grep` before
anything else in this task proceeded. This is the first task in this
sequence (25-32) whose own opening citations required no correction at
all.

One structural note, not a factual error: §2's own phrase "through the
split arrangements the placement policies already produce" undersells
what was needed — `packed()`'s own deployment-wide rank ordering does
not, on its own, produce the clean single-domain reference shape a
"packed" comparison needs (§2's own table above explains why, and what
was added). Not a wrong instruction, but one that took an extra,
explicit step to satisfy rather than following automatically from the
named policies.

## 7. What adding replica ratio would require

**A materially larger search, not just one more axis on this one.**
Task 22's own compute-balance study already found FFN structurally
busier than ATTN at every replica ratio it tested, including the ratio
task 12's own real per-step compute times would predict as balanced —
meaning a replica-ratio search cannot assume a good starting point and
would need to explore genuinely, not just confirm an estimate.
Concretely, this would need:

- **A combined feasibility check**, not two independent ones. Task
  22's own lane-assignment constraint
  (`attn_replicas * attn_dp_size >= ffn_replicas`, or Frontier raises)
  couples replica count to `attn_data_parallel_size`, which this
  task's own search held fixed at 1 — a replica-ratio search would
  need to search that jointly or fix a policy for setting it per
  candidate ratio, not carry over this task's own fixed value.
- **A placement sub-search per pool, not per group.** This task
  searched DECODE_ATTN's own shape at a fixed replica count (1). With
  multiple ATTN and FFN replicas, each replica has its own TP group
  needing its own shape, and the replicas themselves need placement
  relative to each other and to the other pool — the shape space this
  task enumerated (16 for one group) would need to be enumerated *per
  replica*, then combined across replicas, a multiplicative rather
  than additive growth.
- **A different feasibility model for memory.** This task's own memory
  filter (§2) is per-replica, at a fixed TP degree; with more than one
  replica of a pool, whether they can coexist on the same device or
  need separate ones changes what "feasible" means, and task 24/28's
  own model was never exercised at replica counts above 1.

None of this is a small extension of this task's own machinery; it is
the next task's own scope, as this task's own spec says.

## What shipped

- `tools/run_placement_search.py` — feasibility filtering, candidate
  generation from `packed`/`spread`/`fragmented` plus one explicit
  packed-if-it-fits reference, `group_shape()` deduplication,
  exhaustive evaluation of every distinct shape, and a seeded re-run
  of the top 3 using task 31's own `seed_stats` module.

One commit on `task-32-search`, stacked on `task-31-intervals`;
nothing under `upstream/`, `src/engine/`, or `src/integration/`
touched. The acceptance table from tasks 29/30 (collective backend
tp=4, the memory grid at margin 0.9) reproduces bit-identical, checked
again here.
