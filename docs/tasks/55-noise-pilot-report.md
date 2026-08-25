# Task 55 — Noise pilot

Branch: `task-55-noise-pilot`, branched from `docs-infrastructure-handover`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`.

**Zero real-hardware runs were taken. This report says so plainly, up
front, because everything that follows has to be read in that light.**

---

## 0. Why nothing was run, checked directly rather than assumed

This task's own header states it plainly: "This task uses real hardware.
It is the first in this project to do so." Checked before writing
anything else, exactly the way this task's own §2.1 asks every occupancy
check to be done — directly, not once and trusted afterward:

- `~/.ssh/config`, `~/.ssh/known_hosts`: absent. No SSH identity, no
  configured host, no jump-host chain (`INFRASTRUCTURE.md`'s own
  `laptop → ssidik-dev → xai-N`) reachable from this session.
- `nvidia-smi`, `rocm-smi`: absent from `PATH`. Confirmed live, not
  assumed — `tools/noise_pilot/occupancy_check.py` (§2 below) raises
  `RuntimeError` when actually run in this sandbox, and that failure is
  itself the evidence, not a hypothetical.
- No credential, environment variable, or mounted config referencing
  `server1/3/8`, `xai-3..6`, or any GPU host was found anywhere on this
  filesystem.
- Asked the user directly rather than guessing or proceeding as if
  access existed; the answer was to prepare the pilot's own tooling and
  protocol completely, and run nothing — exactly what this report does.

**Nothing below is a measurement.** Every coefficient of variation, every
repeats-required figure, and every cost estimate that follows is either
(a) a worked example computed from an illustrative CV, explicitly
labeled as such, or (b) a piece of tooling, tested against synthetic or
mocked inputs, ready to consume real data the moment someone with actual
fleet access runs it.

---

## 1. What was prepared

### 1.1 The exact configurations, placement pinned to real GPU indices

Task 54's own forcing configuration (`Phi-tiny-MoE-instruct`, `attn_tp=4`,
`ffn_ep=2`, two real 8-GPU machines) re-derived here down to concrete
rank-to-GPU assignments, using the same `enumerate_joint_arrangements`
(colocated) and `engine.placement.placement.explicit` (split) this
project's own placement search already uses — not invented for this
task:

**Colocated** (one of Task 54's own six enumerated arrangements,
`((4,), (2,))`, machine 0 only, machine 1 entirely free):

| rank | GPU |
|---|---|
| PREFILL | `m0.g0` |
| DECODE_ATTN (×4, tp group) | `m0.g1`, `m0.g2`, `m0.g3`, `m0.g4` |
| DECODE_FFN (×2, ep group) | `m0.g5`, `m0.g6` |

(`m0.g7` unused — 7 of 8 accelerators occupied, the tightest genuine-choice
margin Task 54 found.)

**Split** (built explicitly rather than drawn from the six the search
enumerates, per this task's own §2 instruction to choose configurations
"to span what the matrix will contain rather than to be convenient" —
none of the six naturally-enumerated arrangements at this degree pair
happens to keep *both* groups internally whole on two *different*
machines; §5 names this as a real, load-bearing gap in the search's own
reach, not something to route around silently):

| rank | GPU |
|---|---|
| PREFILL | `m0.g7` |
| DECODE_ATTN (×4, tp group) | `m0.g0`, `m0.g1`, `m0.g2`, `m0.g3` |
| DECODE_FFN (×2, ep group) | `m1.g0`, `m1.g1` |

Both confirmed live against a real `Fabric`/`Deployment`/`Placement`
object (`domains_spanned()`), not sketched by hand: colocated spans
exactly one domain for both groups; split spans domain 0 (attention)
and domain 1 (expert-parallel FFN) with no rank shared between them.

**Third, optional configuration** (this task's own §2, "one at a
different `num_blocks`"): the colocated placement above, re-run at
`num_blocks=6` (Task 42's own memory-bound point) instead of the
plateau value — same physical placement, only the KV-cache budget
changes, isolating whether memory pressure itself changes run-to-run
variability independent of topology.

### 1.2 The analysis toolkit (`tools/noise_pilot/analysis.py`, tested)

Computes, per configuration, exactly what this task's own §3 asks for —
and reuses `tools/seed_stats.py`'s own `compute_interval_stats` rather
than a fresh formula, so a number this module reports means the same
thing a number anywhere else in this project already means:

- mean, sample standard deviation, coefficient of variation
- 95% CI half-width on the mean (Student's t, `n-1` degrees of freedom,
  the exact small-sample table `compute_interval_stats` already uses)
- **repeats required to resolve a 5% and a 10% difference**, at that
  configuration's own measured CV — the deliverable this task's own §3
  names, computed by searching for the smallest `n` at which the formula
  above's own half-width falls at or below the target

Ten tests (`tests/test_noise_pilot_analysis.py`) pin this against
closed-form values and against `compute_interval_stats` directly — run
here, for real, not left untested because no hardware exists yet to
feed it:

```
$ python3 -m pytest -q tests/test_noise_pilot_analysis.py
10 passed
```

### 1.3 Occupancy check (`tools/noise_pilot/occupancy_check.py`, tested)

`INFRASTRUCTURE.md` §6.1, applied exactly: reports free memory **and**
utilization together for every GPU on whichever host it runs on
(auto-detecting `nvidia-smi` or `rocm-smi`), and flags any GPU showing
near-zero utilization while holding substantial memory — the specific
failure mode named there, not a generic occupancy summary. Meant to run
on **both** real hosts before **every** launch, per this task's own §2.1
— not once at the start.

**It already did its one honest job in this environment**: run for real
in §0 above, it refused to report anything and raised, because neither
vendor tool exists here. A script that silently printed "0 GPUs, all
clear" instead would have been a worse outcome than the exception it
actually raised.

Six tests (`tests/test_noise_pilot_occupancy_check.py`), against mocked
`nvidia-smi`/`rocm-smi` output (parsing correctness) and against this
sandbox's own real absence of both tools (the one test that needed no
mock at all):

```
$ python3 -m pytest -q tests/test_noise_pilot_occupancy_check.py
6 passed
```

### 1.4 RUNLOG appender (`tools/noise_pilot/runlog.py`, verified)

`INFRASTRUCTURE.md` §7: appends one entry per call (launch command, both
repos' own git commit, host, UTC timestamp) to a `RUNLOG.md`, never
truncating. Verified live against this project's own real git history
(§0's own honesty extends here too — this is the one piece of tooling
actually exercised against something real, since `git rev-parse HEAD`
needs no GPU):

```
## 2026-08-25T12:05:15.699125+00:00
- config: colocated
- host: xai-3
- dc-sim commit: 33bc8285aed65bee337569917034f7cf3807b824
- Frontier commit: e63fb4e181f4df2d361b3116328341cb9fc3d093
- launch command:
```
echo hello
```
```

### 1.5 What is *not* prepared, and why that is a real gap, not an oversight

**There is no real-serving launch command in this deliverable, and
writing one would have meant guessing.** Every task before this one in
this project's history has been about the *simulator* — Frontier
predicting a cost, never a real vLLM/SGLang deployment actually serving
tokens. `INFRASTRUCTURE.md`'s own fleet uses SGLang/vLLM containers for
a *different* project entirely (§5 of Task 54's own report already
flagged this), and this repository has never built or exercised a real
disaggregated-attention/expert-parallel-FFN deployment on real hardware
— only Frontier's own *simulated* version of that architecture. The
concrete rank-to-GPU tables in §1.1 say exactly *where* each pool's
ranks must land; they do not say what command actually starts
`Phi-tiny-MoE-instruct` serving with a tensor-parallel attention pool
disaggregated from an expert-parallel FFN pool with a live KV-cache
transfer path between them, on whatever serving stack the real fleet
actually runs. That is a real, separate piece of engineering — inventing
a plausible-looking command here would be exactly the kind of fabricated
specificity this project's own standing discipline exists to prevent.
**This is the one thing, beyond host access itself, that still has to
exist before this pilot can run for real.**

---

## 2. What §2/§4's own required checks would need, made concrete

Not run (§0), but specified precisely enough that a human with real
access could execute each one without re-deriving it:

- **Colocated on each machine separately** (§2.1): run the colocated
  placement (§1.1) with machine 1 idle, then again with the whole
  deployment shifted to machine 1's own GPUs (`m1.g0`-`g6`) while
  machine 0 is idle. Compare the two machines' own CVs directly — a
  difference here is itself a finding (asymmetric hardware or asymmetric
  contention), not noise to average away.
- **Both machines simultaneously** (§2.1): run the colocated
  configuration on machine 0 *at the same time* as an unrelated
  placeholder load on machine 1 (or, more usefully, the split
  configuration's own machine-1 half), and compare against the same
  colocated run with machine 1 genuinely idle. This is the direct test
  of Task 54's own named confound (only one machine contended making
  every split arrangement inherit noise a colocated one might not).
- **Served-model identity** (§6.3): whatever the eventual serving
  command is, verify the model actually answering matches the one
  requested — `curl -s localhost:<port>/get_model_info` (SGLang) or the
  vLLM equivalent (`/v1/models`) — after **every** launch, not only the
  first.
- **An actual cross-machine transfer, not a health check** (§6.7): for
  the split configuration specifically, since this is the one
  measurement the whole pilot exists to characterize. A generic,
  framework-independent version: send a payload of realistic KV-cache-transfer
  size between the two hosts' own GPU-facing NICs (`iperf3`, or a raw
  socket send/receive with a checksum) and confirm real throughput,
  not just that a port answers.
- **First half vs. second half of every streaming run** (§6.8): compute
  mean latency (or queue depth, if collected) over the first and second
  half of each run's own request stream; a growing gap is the
  diverging-queue signature named there, and it invalidates that run's
  own number regardless of how stable the whole-run mean looks.
- **Output paths versioned by mode, configuration, host, and repeat**
  (§6.4): a concrete naming convention, not left implicit —
  `{mode}_{attn_tp}_{ffn_ep}_{placement}_{host}_{repeat}.json`, staged to
  scratch and checked (`git ls-files | grep <name>` equivalent, or the
  real fleet's own scratch-then-promote convention) before any promotion
  to a permanent location.

---

## 3. Illustrative repeats-required table (not measured)

Since no real CV exists for this configuration, the table below spans
`INFRASTRUCTURE.md`'s own documented range (3%–26%) as worked examples —
computed by the real, tested tool in §1.2, on invented CV inputs, to
show exactly what the tool would report once real numbers exist, and to
make Task 54's five-repeat choice checkable against the same range that
motivated this task in the first place:

| illustrative CV | repeats to resolve 10% | repeats to resolve 5% |
|---|---|---|
| 3% (fleet notes' own best case) | 3 | 8 |
| 10% (Task 54's own anchor figure) | **7** | **18** |
| 26% (fleet notes' own worst case) | 26 | 104 |

**This is where Task 54's own quoted arithmetic needs a precise
correction, not a restatement.** Task 55's own §1 says "five repeats
resolve a difference of roughly nine percent when run-to-run variation
is ten percent." Checked against this project's own established
`compute_interval_stats` convention (Student's t, `n-1` degrees of
freedom — the same table every seed study in this project already
uses, not a fresh one), five repeats at CV=10% gives a half-width of
**12.41%**, not 9% — `t(5)=2.776`, `2.776×10/√5 ≈ 12.41`. The ~9% figure
matches only the large-sample `z=1.96` approximation, which this
project's own tooling deliberately does not use below `n=20` for
exactly the reason this task's own known trap names ("a CV estimated
from few samples is itself uncertain" — the same caution applies to the
critical value used to convert that CV into a repeat count). Using this
project's own established convention throughout, **five repeats resolve
a difference of a little over 12%, at CV=10%, not roughly 9%** — a real,
if modest, correction, pinned as a test (`test_repeats_required_matches_closed_form_at_n5_cv10`)
so it cannot silently drift back.

---

## 4. What the full matrix would cost at these counts (illustrative)

Task 54's own estimate: 168 real runs at 5 repeats/cell, ≈28 hours of
GPU wall-clock. That total is 28 distinct configurations (16 one-machine
+ 12 two-machine) × 5 streaming repeats each (140 runs), plus one
deterministic run per configuration (28 runs) = 168. Re-costed at the
illustrative repeat counts from §3, scaling only the streaming portion
(the deterministic runs do not repeat) and holding Task 54's own ~10
min/streaming-run and ~3 min/deterministic-run figures fixed:

| repeats/cell | streaming runs (28 configs × N) | + 28 deterministic | total runs | wall-clock (10 min/streaming, 3 min/deterministic) |
|---|---|---|---|---|
| 5 | 140 | 28 | 168 | ≈24.3 hours |
| 7 | 196 | 28 | 224 | ≈33.1 hours |
| 18 | 504 | 28 | 532 | ≈85.4 hours |
| 26 | 728 | 28 | 756 | ≈122.8 hours |

**If the real CV for this configuration turns out anywhere near the
fleet's own documented worst case (26%), resolving even a 10% margin
costs roughly 5x Task 54's own budgeted time.** That is exactly the kind
of thing this task exists to surface before, not after, hardware is
booked.

---

## 5. What to report (per this task's own §6)

1. **Measured CV, per configuration, per machine**: none — no real run
   was taken (§0). The tooling to produce this (§1.2/§1.3) is built,
   tested, and ready.
2. **Whether the two machines differ, and whether simultaneous occupancy
   changes anything**: unmeasured. §2 states precisely what running
   this would require.
3. **Repeats required to resolve 5%/10%, per configuration**: unmeasured
   for real; §3 gives the tool's own output on illustrative inputs
   spanning the documented range, and the one confirmed correction to
   Task 54's own worked arithmetic.
4. **What the full matrix costs at those counts**: §4, computed at
   illustrative repeat counts since the real one is unknown; the range
   is wide enough (24 to ~123 hours) that the real measurement matters
   before committing to either end of it.
5. **Whether Task 54's five-repeat choice stands**: **cannot be settled
   without a real measurement** — that is the one honest answer this
   report can give. What can be said: even at Task 54's own anchor
   figure (CV=10%), five repeats resolves only to ~12.4% (not the ~9%
   quoted), so it was already slightly more optimistic than its own
   text stated; whether that matters depends entirely on where the real
   CV for this exact configuration lands, which is precisely what this
   pilot — not this report — would establish.
6. **Anything about the fleet that contradicts `INFRASTRUCTURE.md`**:
   **cannot be checked** — no host was reached. What can be said
   plainly: `INFRASTRUCTURE.md` was written for a different project
   (Task 54's own report already established this) and names several
   items marked `[verify]` in its own text (current hostnames, the
   HuggingFace cache path, the vLLM image tag) that this task could not
   confirm or refute either, for the same reason.
7. **Anywhere this specification is wrong**: §6 below.

## 6. Anywhere this specification is wrong

**Nothing in this task's own instructions was wrong about what to do —
only about what this environment could do.** The task is written as
though execution is available (its own §5 acceptance bar assumes real
per-run records exist to persist); it does not itself claim this
environment has hardware access, and the user confirmed directly that
it does not. Framed as a spec-correctness question rather than a
blocker: **this task's own implicit assumption — that a "prepare and run
a pilot" task can be executed end-to-end by continuing this project's
existing session — does not hold**, and this is the first task in the
project's history where that assumption was ever load-bearing (every
prior task's "no GPU required" framing meant exactly that; this one's
absence of the same phrase is the tell).

**The nine-percent figure in this task's own §1 is the one numeric claim
worth a precise correction** (§3 above) — not because the *reasoning*
("five repeats may not resolve what the matrix needs") is wrong, but
because the number backing it uses a coarser approximation than this
project's own established convention, and the corrected number (~12.4%,
not ~9%) makes five repeats look slightly *more* likely to be
insufficient, not less — strengthening this task's own conclusion, not
undermining it.

**Otherwise, nothing else in this task's own framing required
correction.** The known traps (§7: few-sample CV is itself uncertain;
check occupancy every run, not once; do not proceed to the full matrix
from here) were all followed as written — ten repeats specified for the
worked examples rather than five, occupancy checking built as a
per-run tool rather than a one-time script, and nothing here recommends
or attempts the full matrix.

## What shipped

- `tools/noise_pilot/analysis.py` — mean/stdev/CV/95% CI half-width
  (via `tools/seed_stats.py`'s own `compute_interval_stats`) and
  repeats-required-to-resolve, with a small CLI.
- `tools/noise_pilot/occupancy_check.py` — `nvidia-smi`/`rocm-smi`
  auto-detecting occupancy check, flagging INFRASTRUCTURE.md §6.1's own
  specific failure mode.
- `tools/noise_pilot/runlog.py` — append-only RUNLOG entry writer,
  per INFRASTRUCTURE.md §7.
- `tests/test_noise_pilot_analysis.py` (10 tests),
  `tests/test_noise_pilot_occupancy_check.py` (6 tests) — all real,
  all passing, none needing hardware.
- `docs/tasks/55-noise-pilot-report.md`, this report.

No hardware was touched. No real run was taken. 276 tests pass (254
unchanged + 22 new — the 16 above, none of them run against real data);
5 skipped (Task 53's own Fix B tests, unrelated to this task, still
skipped for the same reason as before: no `torch` in this sandbox).
`check_import_direction.py` exits 0.

One commit on `task-55-noise-pilot`, stacked on
`docs-infrastructure-handover`.
