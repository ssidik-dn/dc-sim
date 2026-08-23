# Task 31 — Confidence intervals

Branch: `task-31-intervals`, stacked on `task-30-path-cache`. Paths
confirmed per task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`.

189 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. This task adds two new files under `tools/` only — nothing
under `src/engine/` or `src/integration/` changed, so the acceptance
table's bit-identical requirement holds by construction; it was still
re-run and confirmed (§4).

**The headline finding changes the shape of everything else this task
asks for.** `--seed`, in every tool this project has ever built, has
never varied a single reported result. Two independent, compounding
wiring gaps explain it completely, established from source and then
confirmed empirically before anything else in this task proceeded.

---

## 1. Where the variance comes from

### 1.1 Two compounding gaps, neither of them Poisson

**Gap one: request generation re-seeds internally, to a fixed default,
overriding whatever `--seed` set.** `set_seeds(config.seed)`
(`frontier/utils/random.py`) correctly reseeds the global
`random`/`np.random` state — confirmed directly by printing
`random.random()` immediately after the call for `--seed 0` vs
`--seed 1` and getting different values, as expected. But request
generation happens later, inside `Simulator.__init__()`'s own
construction, reaching `SyntheticRequestGenerator.generate_requests()`
(`frontier/request_generator/synthetic_request_generator.py:143`):

```python
set_seeds(self.config.seed)
```

`self.config` here is the *request generator's own* config object, and
its `seed` field (`BaseRequestGeneratorConfig.seed`,
`frontier/config/config.py:320-323`) defaults to `42` — a field
entirely separate from `SimulationConfig.seed`, the one `--seed` sets.
No tool in this project has ever passed the matching
`--synthetic_request_generator_config_seed` flag to override it
(`grep -rln "request_generator_config_seed" tools/` returns nothing).
So every request-generation random draw — arrival intervals, and any
randomized request length — runs on a fixed, un-varying seed regardless
of `--seed`.

**Gap two: offline mode discards generated arrival times anyway.**
Even with gap one fixed, `Simulator.run()`'s own offline-mode branch
(`frontier/simulator.py:409-424`) forces every request's `arrived_at`
to `0` and submits it immediately, unless
`--offline_use_generated_request_arrivals` is also set
(`frontier/config/config.py:5765-5773`, default `False`; never set by
any tool here either). `--simulation_mode offline` is the universal
convention across every real-compute tool in this project.

**Neither gap involves the Poisson interval generator being wrong or
unseeded.** `PoissonRequestIntervalGenerator.get_next_inter_request_time()`
(`frontier/request_generator/poisson_request_interval_generator.py`)
correctly draws from Python's global `random.random()` — the same
module `set_seeds()` seeds. The problem is entirely about *which*
seed value reaches it, and whether its output is used at all once
computed.

**Everything else checked and ruled out.** No other seeded source of
variation was found: request lengths use `length_generator_config_type=fixed`
in every tool (no randomness to speak of); MoE routing uses its own
separate, hardcoded `--replica_config_moe_routing_seed 42` in every
tool (also never tied to `--seed`); `Fabric.route()`'s `PER_FLOW_ECMP`
branch (which would need `hashlib`-based, not `random`-based,
tie-breaking) is never exercised (task 30's own finding, reconfirmed
here). `os.environ["PYTHONHASHSEED"] = str(seed)` inside `set_seeds()`
itself is worth naming as a latent, separate bug even though it never
mattered here: setting that environment variable mid-process has no
effect on the *current* process's hash randomization (it is read only
at interpreter startup) — harmless only because nothing in the actual
measurement path depends on hash-randomized iteration order.

### 1.2 Confirmed empirically, not just from source

Three seeds, identical configuration, before and after passing
`tools/seed_stats.seed_argv_fix(seed)` (this task's own fix — the
matching generator-seed flag, plus `--offline_use_generated_request_arrivals`):

| | seed=0 | seed=1 | seed=2 |
|---|---|---|---|
| **Without the fix** — throughput (req/s) | 86.80643609045715 | 86.80643609045715 | 86.80643609045715 |
| **Without the fix** — arrival span (s) | 0 | 0 | 0 |
| **With the fix** — throughput (req/s) | 15.514 | 22.964 | 20.105 |
| **With the fix** — arrival span (s) | 1.893 | 1.289 | 1.341 |

Bit-identical to every decimal place without the fix; genuinely
different with it. This is not a small-effect question — it is exactly
the "seeds must actually differ" trap this task's own §6 warns about,
and it would have looked "reassuringly tight" (task's own words) had
this check not been done first.

### 1.3 Which configurations are deterministic

**Every configuration any tool in this project has used before this
task — `simulation_mode=offline`, the default (unset)
`offline_use_generated_request_arrivals`, fixed-length requests, a
hardcoded MoE routing seed — is completely deterministic given
everything *except* `--seed`.** This is stronger than "a fixed-arrival
generator eliminates variance" (this task's own §2.1 framing, which
supposes a Poisson process exists and asks whether fixing arrivals
removes its effect): the Poisson process was never actually driving
anything to begin with, in the default configuration every tool here
uses. There is no experiment to run that would show variance in that
configuration, because there is no seed-dependent input left once
arrivals are fixed to `t=0` and lengths are fixed by config.

This is exactly the distinction this task's own §2.1 anticipated as a
possible outcome and told me to make if it occurred: *"the interval
question becomes a question about which workloads to average over
rather than about simulator noise, and those are different things
worth separating."* They are separated below: §2 measures genuine
variance in a *different*, deliberately-constructed configuration
(streaming, staggered arrivals) — a workload-averaging question, not a
statement about the noise floor of this project's actual existing
measurements, which have none.

## 2. The noise floor, once it is real

Every figure below uses `--offline_use_generated_request_arrivals` plus
the matching request-generator seed (`seed_stats.seed_argv_fix`) — a
new configuration this project has not measured before, chosen
specifically to have genuine seed-to-seed variance to report on. Real
h800 compute throughout (Phi-tiny-MoE-instruct).

**Two configurations, per this task's own trap about variance
depending on where you are:** a plateau point (`tp=1`, `num_blocks=6911`,
memory unconstrained, task 22/24's own established reference) and a
near-knee point (`tp=1`, `num_blocks=6`, capacity 2 — task 22's own
sharp-knee configuration).

| metric | n | plateau mean | plateau CI95 half-width (% of mean) | near-knee mean | near-knee CI95 half-width (% of mean) |
|---|---|---|---|---|---|
| throughput (req/s) | 5 | 19.880 | 17.84% | 19.261 | 15.33% |
| | 10 | 21.165 | 11.12% | 20.661 | 10.65% |
| | 20 | 20.695 | 6.43% | 20.372 | 6.13% |
| tpot (ms) | 5 | 4.512 | 4.53% | 5.325 | 17.72% |
| | 10 | 4.539 | 2.30% | 5.488 | 11.81% |
| | 20 | 4.525 | 1.30% | 5.711 | **11.28%** |
| M2N transfer (ms) | 5 | 13.784 | 0.64% | 13.702 | 0.16% |
| | 10 | 13.794 | 0.31% | 13.709 | 0.10% |
| | 20 | 13.789 | 0.18% | 13.709 | 0.06% |
| batch size | 5 | 1.676 | 17.54% | 1.508 | 10.01% |
| | 10 | 1.743 | 9.64% | 1.562 | 7.04% |
| | 20 | 1.709 | 5.28% | 1.559 | 4.03% |

**The answer to "how small a difference can this project detect,"
stated directly: at 20 seeds, roughly 6% on throughput, 1.3% on tpot
in a flat region (11-13% near a knee), and under 0.2% on M2N transfer
time.** Any measured effect smaller than the relevant one of these, at
the relevant seed count, is not distinguishable from this noise floor.
M2N transfer time is the tightest metric by a wide margin — it is
close to a fixed physical quantity (payload size over link bandwidth
plus a latency term) that arrival timing barely perturbs; throughput
and batch size are the loosest, because they are direct functions of
how many requests happen to be in flight at once, which is exactly
what a staggered arrival pattern randomises.

**The known trap materialised exactly as warned, and is reported
rather than smoothed over.** tpot's own coefficient of variation at
the near-knee point *rose* with seed count (14.3% at n=5, to 24.1% at
n=20) rather than settling — a real, small-sample effect: near a sharp
capacity edge, occasional seeds produce heavily queued outliers, and 5
seeds under-sampled that tail by chance. The 95% CI half-width still
narrowed in percentage terms (17.7% → 11.8% → 11.3%) because the
`t`-critical value shrinks with `n` fast enough to outweigh the rising
standard deviation over this specific range, but it stopped narrowing
much past `n=10` — a different, and worse, shape than the plateau
point's own clean, monotonic narrowing. **Near a knee, more seeds
mostly reveal more of the tail, not a tighter estimate; the plateau
point is where seed count actually buys precision.**

## 3. Do the headline findings survive?

All four re-measured with genuine seed variance (`n=20`) wherever the
underlying tool uses real compute profiles; reported with 95% CI
half-widths, and whether the packed/unsplit and split/other intervals
actually overlap.

| Finding | Original (deterministic) figure | Re-measured mean (n=20) | 95% CI half-width | Survives? |
|---|---|---|---|---|
| TP=4 group split | +88.3% tpot | packed 3.880 ms (±0.38%), split 8.975 ms (±0.67%) | non-overlapping by a wide margin | **Yes** |
| Pool split M2N | ratio 14.65x | colocated 0.938 ms (±0.26%), split 13.789 ms (±0.18%) | non-overlapping by a wide margin | **Yes** |
| Topology-aware scheduling | (cited: −1.4%; real: +0.00%) | not re-measured — see §4 | — | **Already reported as noise, in task 15's own words** |
| tp=2 packed vs tp=1 — **throughput** | 90.612 vs 86.806 req/s (+4.4%) | tp=1: 20.695 (±6.43%), tp=2: 20.747 (±6.46%) | **overlapping almost completely** | **No** |
| tp=2 packed vs tp=1 — **tpot** | (not separately claimed) | tp=1: 4.525 ms (±1.30%), tp=2: 4.231 ms (±1.42%) | non-overlapping | **Yes** |

**Two survive outright. One was already known not to. The fourth is
genuinely split by which objective is named — this task's own trap
about throughput and latency disagreeing, materialising in exactly the
headline finding the task's own §2 flagged to watch.**

The TP-split penalty and the pool-split M2N ratio are both large enough,
and physically direct enough (communication cost over a fixed link
topology), that streaming-arrival noise does not come close to
threatening them — their intervals do not merely fail to overlap, they
are separated by many multiples of either interval's own width.

**tp=2-vs-tp=1 throughput does not survive.** The reported +4.4%
difference (90.612 vs 86.806 req/s, task 24's own deterministic figure)
is smaller than either configuration's own 95% CI half-width (±6.4%)
under genuine arrival variance — the two intervals overlap almost
entirely (tp=1: 19.36-22.03; tp=2: 19.41-22.09 req/s). **This project
cannot currently distinguish "tp=2 has higher throughput than tp=1" from
noise, once request arrivals are allowed to vary the way a live
workload's would.** But tpot tells a different story: tp=2's own
4.231 ms mean sits clearly below tp=1's 4.525 ms, with no interval
overlap at all — **tp=2 is measurably faster per token, even though it
is not measurably higher-throughput**, under this specific streaming
configuration. Task 24's own original claim was about throughput
specifically; restated with an interval, it does not hold as stated,
though a nearby, real claim about latency does.

## 4. Topology-aware scheduling, not re-measured, and why

Two separate reasons neither point to a re-run being straightforward:

1. **The task's own citation ("about −1.4%") does not match task 15's
   report.** Quoted directly, `docs/tasks/15-topology-scheduler-report.md`,
   §5: *"Mean tpot moved by +0.005 ms (+0.00%) — noise, not an
   effect."* Task 15 already answered exactly the question this task
   re-asks, in the same words this task's own §2 anticipates finding
   ("this project has reported an effect it cannot distinguish from
   nothing") — five tasks before this one, about the same finding.
   Checked by direct `grep` for "1.4%"/"−1.4"/"1.4 %" across every
   report in this project: it appears nowhere associated with
   topology-aware scheduling.
2. **Task 15's own measurement used dummy compute, which `AGENTS.md`
   says never to calibrate or baseline against.** `run_topology_scheduler_study.py`'s
   own argv sets
   `--random_forrest_execution_time_predictor_config_enable_dummy_mode`.
   Confirmed directly (`grep`), not inferred. A confidence interval
   around a dummy-mode measurement would not be a meaningful answer to
   this task's own question, and rebuilding that study on real compute
   profiles first is separate, larger work this task's own scope
   (confidence intervals on existing findings) does not include.

**Both facts point the same way: this headline was already known not
to survive, and re-running it properly is out of this task's scope,
not merely skipped.**

## 5. Where the helper lives, and how a tool uses it

`tools/seed_stats.py` — alongside every existing "run N seeds,
aggregate" function in this project (`_aggregate()` in
`run_memory_edge_study.py`, `run_memory_tp_study.py`, and others), not
under `src/engine/`. Nothing in it would violate the import-direction
rule either way (it imports neither `src/integration/` nor
`upstream/`), so that check did not decide the placement — consistency
with the tools that already do the same kind of aggregation did.
`src/engine/` is Phases 1-6's own standalone placement/cost modeling;
orchestrating N subprocess runs of an existing scenario and computing
statistics over the results answers neither a placement nor a
communication-cost question, so it does not belong there even though
it legally could.

Two pieces:

- `seed_argv_fix(seed, vary_arrivals=True) -> List[str]` — the two
  extra flags (§1.1's fix) a tool's own argv needs alongside `--seed`
  for the seed to do anything.
- `compute_interval_stats(values)` / `run_seed_study(runner, seeds, metrics)` —
  mean, sample stdev, CV%, and 95% CI half-width (both in absolute
  terms and as % of the mean), given either a list of values directly
  or a per-seed scenario-runner callable (matching the
  `_run_scenario_in_subprocess(...)` convention every real-compute tool
  here already follows) plus a seed range and metric names to collect.

`tools/run_seed_variance_study.py` (this task's own, not modifying any
existing tool) shows the pattern: import a base `_argv`/deployment
helper from an existing tool, append `seed_argv_fix(seed)` to its argv,
and hand a small per-scenario wrapper to `run_seed_study()`. No
existing tool's own `_argv`/`_run_scenario` was changed, so no
historical figure moved — confirmed in §6.

## 6. Acceptance — the numbers, before and after, side by side

Only new files were added under `tools/`; nothing under `src/engine/`
or `src/integration/` changed, so no existing figure could have moved.
Re-run anyway, per this task's own standard:

| Measurement | Expected | This task |
|---|---|---|
| `run_collective_backend_study.py`, tp=4 | packed 2.628864 ms, split 38.513664 ms, tpot 5.803319 / 10.929719 ms | packed 2.628864 ms, split 38.513664 ms, tpot 5.803319 / 10.929719 ms |
| Memory grid, margin 0.9, tp=2 packed | 90.61152851669767 req/s, 13.953915179821097 ms | 90.61152851669767 req/s, 13.953915179821097 ms |
| `run_m2n_integration.py` | colocated 0.187776 ms, split 2.750976 ms, ratio 14.6503 | colocated 0.187776 ms, split 2.750976 ms, ratio 14.6503 |

Bit-identical, every digit.

## 7. Anywhere this specification is wrong

- **§2.3's own "Topology-aware scheduling: about −1.4%" citation does
  not exist anywhere in this project's records.** The real figure,
  task 15's own report, is +0.00% ("+0.005 ms... noise, not an
  effect") — already, explicitly, self-reported as indistinguishable
  from noise. This is not a case of a small real effect this task
  needed to check; it is a case of the effect having already been
  checked and found absent, misquoted here as a specific, sizeable
  number.
- **§1's framing that "a seed changes the arrival pattern" understates
  what this task actually found.** It is not merely that a fixed-arrival
  generator (deliberately chosen) eliminates variance — every
  configuration this project has used already had no seed-dependent
  arrival pattern to begin with, for two compounding, unrelated-to-Poisson
  reasons (§1.1). The task's own §2.1 anticipated this could happen and
  said how to handle it if so; that anticipation was correct, and is
  what this report follows.
- Otherwise this specification's own structure — establish the
  variance source before measuring anything; check whether seeds
  actually differ before trusting an interval; separate the mean's own
  confidence interval from the data's raw spread; measure at more than
  one point given the known knee-variance effect; treat "does not
  survive" as a real, reportable outcome rather than a failure to avoid —
  matched exactly what this investigation needed, and correctly
  anticipated (§2 and §6's own known traps) two of this report's three
  central findings before they were measured.

## What shipped

- `tools/seed_stats.py` — `seed_argv_fix()`, `compute_interval_stats()`,
  `run_seed_study()`; the reusable helper §5 describes.
- `tools/run_seed_variance_study.py` — the S2.1 seed-differs check, the
  two-configuration noise-floor sweep (S2.2), and the three
  real-compute headline re-measurements (S2.3), all built on the new
  helper and on existing tools' own deployment/argv helpers, none of
  which were modified.

One commit on `task-31-intervals`, stacked on `task-30-path-cache`;
nothing under `upstream/`, `src/engine/`, or `src/integration/`
touched. Every acceptance-table figure reproduces bit-identical,
confirmed side by side, per this task's own standard.
