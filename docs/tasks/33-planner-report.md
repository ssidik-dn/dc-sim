# Task 33 — From search to a planner

Branch: `task-33-planner`, stacked on `task-32-search`. Paths per Task
25: working tree at `/work/simulation/dc-sim`, Frontier at
`/work/simulation/Frontier`.

189 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. One new file under `tools/` (`tools/planner.py`); nothing under
`src/engine/` or `src/integration/` changed.

A parallel investigation, Task 34, ran during this task and audited
`packed()` — the placement policy this task's own `enumerate_attn_shapes`
also calls. It found no defect that affects this task: `packed()`'s own
single-domain reach depends on `1 + degree <= domain_size`, which this
task's own `enumerate_attn_shapes` already works around with the same
explicit fallback Task 32 built, copied here unchanged. See
`docs/tasks/34-packed-audit-report.md` §5 for the direct confirmation.

---

## 1. The interface, and which inputs are genuine parameters

```python
plan(topology: Topology, model: ModelSpec, workload: Workload,
    hardware: Hardware, objectives: Objectives,
    *, attn_tp_values=None, ep_values=None, replica_ratios=((1, 1),)
    ) -> PlanResult
```

`tools/planner.py`, not `src/engine/` — like every other real-compute
orchestration tool in this project, it must invoke Frontier via
subprocess to evaluate a candidate, and `src/engine/` must never import
`src/integration/`. `Topology`, `Deployment`, and Frontier's own model/
workload configuration are reused, not rebuilt; this module makes them
parameters of one call.

| Input | Dataclass | Fields | Genuinely varied in this report? |
|---|---|---|---|
| Topology | `Topology` | `fabric: Fabric`, `name: str` | **Yes** — `domain8`, `domain64`, `oversubscribed`, `task32repro`, four distinct fabrics, all through the same call (§3, §4) |
| Model | `ModelSpec` | `model_name`, `total_experts`, `router_topk`, `is_moe`, `admissible_tp`, `admissible_ep` | held fixed (Phi-tiny-MoE-instruct throughout) — the only model this project has real h800 profiles for; see §6.1's own trap discussion below |
| Workload | `Workload` | `num_requests`, `qps`, `prefill_tokens`, `decode_tokens` | **Yes** — `decode_tokens` varied 8 / 16 / 64 (§4.2), changing the winner |
| Hardware | `Hardware` | `device`, `memory_margin_fraction` | **Yes** — `memory_margin_fraction` varied 0.992 / 0.5 (§4.1c), changing which degrees are feasible without changing the winner |
| Objectives | `Objectives` | `slo_tpot_ms`, `min_throughput_rps`, `minimize`, `slo_attainment_floor` | **Yes** — `min_throughput_rps` varied to demonstrate the fixed-floor trap (§4.2, §7) |

**The test this task's own §7 trap poses** — "an input that is always
the same value is not an input; the test is whether a study can vary it
without editing the tool" — is met for four of the five: every
`Topology`/`Workload`/`Hardware`/`Objectives` variation in this report
was a call-site change, `tools/planner.py` itself untouched between
calls. `ModelSpec` was not varied in this report (no second real
h800-profiled model exists in this project to vary it to), but it is
still a genuine parameter of the call, not a constant folded into the
tool — `feasible_num_blocks` explicitly raises `NotImplementedError`
for any `(model_name, device)` pair outside its own calibration table,
rather than silently reusing Phi-tiny-MoE-instruct's numbers for a
different model, which is the honest way to represent "this parameter
exists, and this report did not exercise it."

**Search variables**: tensor-parallel degree (`attn_tp_values`),
expert-parallel degree (`ep_values`), replica counts
(`replica_ratios`, the attention:FFN ratio), and physical placement
(`attn_shape`, from `enumerate_attn_shapes`, generalizing Task 32's
`enumerate_shapes` to accept an arbitrary `Topology` instead of a
hardcoded fabric). **Not search variables**: scheduler policy (excluded
per this task's own §2 — its benefit is not established on realistic
compute, Task 15's own "+0.00%, noise, not an effect") and memory
capacity (`feasible_num_blocks` is a feasibility filter only, per Task
24/28/32's own finding, reused not re-derived).

---

## 2. Whether Task 32's result reproduces

**Yes, bit-identical — the entire 16-row ranked table, not just the
winner.** `plan()` called with Task 32's exact fabric
(`build_node_scale(5, 4)`, wrapped as `_topology_task32repro`), the same
model/workload/hardware, and no constraints:

| rank | tp | shape | mean tpot (ms) — Task 32 | mean tpot (ms) — through `plan()` |
|---|---|---|---|---|
| 1 | 2 | `(2,)` | 11.6803 | 11.6803 |
| 2 | 4 | `(4,)` | 14.4305 | 14.4305 |
| 3 | 2 | `(1,1)` | 18.3178 | 18.3178 |
| 4 | 4 | `(2,1,1)` | 24.9729 | 24.9729 |
| 5-7 | 4 | `(3,1)`/`(2,2)`/`(1,1,1,1)` | 27.2465 | 27.2465 |
| 8-9 | 8 | `(3,2,1,1,1)`/`(3,2,2,1)` | 42.9118 | 42.9118 |
| 10-16 | 8 | every other tp=8 shape | 45.1854 | 45.1854 |

Every figure matches to four decimal places; the rejection at tp=1
("memory: infeasible at this margin") reproduces Task 32's own §2
feasibility table exactly. Confirmed **before** anything in §4, per this
task's own §5 ordering requirement.

---

## 3. The rejection-breakdown mechanics, and how close accepted arrangements sit to their constraints

Every `plan()` call in this report used `Objectives(slo_tpot_ms=15.0,
min_throughput_rps=40.0, slo_attainment_floor=0.5)` unless stated
otherwise — decided once, before running anything, not re-tuned per
call. Every rejected candidate carries which single constraint rejected
it first (memory, then SLO, then throughput) — never a bare "no."

Representative breakdown (`domain8`, the topology used throughout §4.1):

| accepted | mean tpot | throughput | SLO attainment | margin to its own constraint |
|---|---|---|---|---|
| tp=2 `(2,)` | 11.6803 ms | 107.171 | 0.750 | throughput 2.68x the floor; SLO 1.5x the floor — comfortable |
| tp=4 `(4,)` | 12.1569 ms | 103.186 | 0.750 | same, comfortable |
| tp=8 `(8,)` | 15.6040 ms | 81.473 | **0.500** | **sitting exactly on the SLO floor** |
| tp=2 `(1,1)` | 18.3178 ms | 69.886 | **0.500** | **sitting exactly on the SLO floor** |

**The known trap this task's own §7 names — "a constraint evaluated on
a noisy quantity is still noisy near its threshold"** — is directly
visible here: two of the four accepted arrangements pass the SLO floor
by exactly zero margin (0.500 against a 0.5 floor), computed from a
single deterministic pass. Neither is the winner, so this does not
change which arrangement `plan()` recommends, but it means a reader
should not treat either's *acceptance* as robust — a different seed
could plausibly push either below 0.5 and flip it to rejected. This is
reported here explicitly rather than left implicit, per this task's own
instruction.

Rejected, by constraint, on `domain8` (fourteen of eighteen candidates):

| constraint | count | examples |
|---|---|---|
| memory (feasibility) | 1 | tp=1, every shape (parameter memory alone exceeds budget at margin 0.992) |
| SLO floor (0.25 < 0.5) | 4 | every non-`(4,)` tp=4 shape |
| throughput floor (29.0-30.5 < 40.0) | 9 | every tp=8 shape but `(8,)` |

---

## 4. The two-topology demonstration

### 4.1 Fixed pairs, tried in the order the spec names them

**(a) Domain size — 8-GPU domains vs. 64-GPU domains.** The pair the
spec itself calls "most likely to separate."

`_topology_domain8` (`build_node_scale(5, 8)`) vs. `_topology_domain64`
(`build_node_scale(2, 64)`), same model/workload/hardware/objectives as
§2:

| topology | winner | mean tpot | throughput | SLO |
|---|---|---|---|---|
| domain8 | tp=2 `(2,)` | 11.6803 ms | 107.171 | 0.750 |
| domain64 | tp=2 `(2,)` | 11.6803 ms | 107.171 | 0.750 |

**Did not separate.** The winning shape never needs more than 2 GPUs in
one domain, and both fabrics can hold 2 GPUs in one domain — the
domain-size difference this pair was built to expose only matters for
tp=8, which is not competitive on either fabric (tp=8's own figure
*does* differ between the two — 15.6040 ms on domain8, 13.3304 ms on
domain64, since domain64 can fully colocate PREFILL+ATTN(8)+FFN in one
domain while domain8 cannot fit all ten ranks in an 8-slot domain — but
this narrows the gap to the winner without closing it).

**(b) Well-provisioned vs. oversubscribed** (`scale_out_GBps` 50 vs.
12.5 — a 4x narrower cross-domain link, same domain8 fabric otherwise):

| topology | winner | mean tpot | throughput | SLO |
|---|---|---|---|---|
| domain8 (well-provisioned) | tp=2 `(2,)` | 11.6803 ms | 107.171 | 0.750 |
| oversubscribed | tp=2 `(2,)` | 11.6803 ms | 107.171 | 0.750 |

**Did not separate, for a clean, checkable reason**: the winning
arrangement fits entirely inside one domain and never touches the
scale-out link at all, so its own cost is bit-identical between the two
fabrics. Oversubscription *does* bite every candidate that crosses a
domain — tp=8's own figure worsens (15.6040 -> 16.2331 ms) and the
`tp=2 (1,1)` split arrangement newly fails the SLO floor under
oversubscription — but neither of those was competitive for the win in
the first place, so the winner is untouched.

**(c) Generous vs. constrained device memory** (`memory_margin_fraction`
0.5 vs. 0.992, same domain8 fabric, same objectives):

| hardware margin | winner | mean tpot | newly feasible |
|---|---|---|---|
| 0.992 (constrained) | tp=2 `(2,)` | 11.6803 ms | — |
| 0.5 (generous) | tp=2 `(2,)` | 11.6803 ms | tp=1 (12.3316 ms) |

**Did not separate — confirms Task 28's own finding directly, through
the new interface rather than by citation.** Generous memory makes tp=1
feasible where it previously was not (a feasibility change), but tp=1's
own figure (12.3316 ms) still loses to tp=2 (11.6803 ms) — feasibility
moved, preference did not.

**None of the spec's three named pairs separated.** Per this task's own
§4 instruction, that is reported plainly, not treated as a failed
search: on this model and workload, tp=2 packed is not a fabric- or
memory-margin-specific answer; it is closer to a genuine invariant of
this configuration across every topology axis this task varied. Testing
stopped at three pairs because the spec names exactly three, not
because a fourth was tried and discarded for being inconvenient.

### 4.2 The converse: one topology, two workloads — this is where the answer changes

Same `domain8` topology, same model/hardware, `decode_tokens` varied
8 / 16 (the workload used throughout §2-4.1) / 64:

| decode_tokens | winner (unconstrained ranking) | mean tpot | runner-up | margin |
|---|---|---|---|---|
| 8 | tp=2 `(2,)` | 11.7007 ms | tp=4 `(4,)` at 12.1773 ms | tp=2 by 4.1% |
| 16 | tp=2 `(2,)` | 11.6803 ms | tp=4 `(4,)` at 12.1569 ms | tp=2 by 4.1% |
| 64 | **tp=4 `(4,)`** | **12.1463 ms** | tp=2 `(2,)` at 13.2814 ms | **tp=4 by 9.3%** |

**The answer changes.** A four-times-longer decode phase flips the
winner from tp=2 to tp=4. This is the demonstration the spec calls "the
heart of the task" — the same tool, same fabric, same model, only the
workload's own decode length changed, and the recommended arrangement
changed with it. Qualitatively: a short decode phase is dominated by
per-request fixed costs (prefill, scheduling, the M2N hop) that tp=2
pays less of; a long decode phase is dominated by *per-step* compute
that tp=4 shares across more devices, and enough decode steps
accumulate that this outweighs tp=4's own larger communication share. This
particular margin (+9.3%) is exactly reproducible in this deterministic
configuration (no seed-dependent input exists in it at all, per Task
31's own §1.3), but does **not** clear the noise floor once genuine
arrival randomness is introduced — see the seeded-interval discussion
below, which reports this honestly rather than only the more convincing
deterministic number.

**A trap this run hit directly, and reports rather than hides**: under
the *same fixed* `Objectives(min_throughput_rps=40.0,
slo_attainment_floor=0.5)` used everywhere else in this report,
`decode_tokens=64` accepts **nothing** — every candidate's own
throughput (6-26 req/s, since a four-times-longer decode phase holds
each request's resources four times as long) sits below a floor that
was calibrated against the sixteen-token workload. This is not a bug in
`plan()`; it is exactly the failure mode this task's own §7 first trap
warns about in spirit — a constraint that silently rejects everything
when the workload it is applied to was never the workload it was
calibrated against. The table above uses the *unconstrained* ranking
(`min_throughput_rps=0.0`, `slo_attainment_floor=0.0`) specifically to
answer the qualitative question ("does the preferred degree change") on
its own terms, and reports the all-rejected result under the original
floor as a separate, real finding rather than silently swapping in a
more convenient floor to make the demonstration land. See §6.6 and §7
for what this implies about the objectives interface.

**Margin, with a seeded interval** (Task 31's own `seed_stats`, n=20,
genuine arrival variance, matching Task 32's own re-run method exactly):

| comparison | deterministic margin | seeded mean (tp A) | seeded mean (tp B) | seeded margin | CIs overlap? |
|---|---|---|---|---|---|
| domain8, decode=16: tp=2 vs tp=4 | +4.1% (tp=2 ahead) | 3.2378 ms ±1.56% | 3.4729 ms ±1.46% | +7.26% (tp=2 ahead) | **No** — survives, but only ~4.8x Task 31's own ~1.3-1.56% noise floor at this configuration, far short of Task 32's own 24-29x |
| domain8, decode=64: tp=4 vs tp=2 | +9.3% (tp=4 ahead) | 4.7908 ms ±4.48% | 4.9544 ms ±6.58% | +3.41% (tp=4 still ahead) | **Yes** — `[4.5763, 5.0053]` vs. `[4.6286, 5.2802]` overlap |

**Both margins are far narrower than Task 32's own headline (+23.6% to
+37.6%, 24-29x the noise floor) — and the workload-reversal margin does
not survive at all.** The decode=16 margin (+7.26% seeded) clears the
noise floor, barely (§ above). **The decode=64 reversal — the
demonstration's own central result — does not**: the seeded mean still
favours tp=4 (4.7908 ms vs. 4.9544 ms, same direction as the
deterministic pass), but the two 95% intervals overlap, so twenty seeds
cannot distinguish the two arrangements with confidence. This needs the
same distinction Task 31/32's own reports already make: the
**deterministic** comparison (fixed lengths, all arrivals submitted at
once) has no seed-dependent input at all, so its own +9.3% figure is
not "noise that happened to look real" — it is a real, exactly
reproducible fact about that specific, non-streaming configuration.
Once genuine arrival randomness is introduced (Task 31's own seeded
regime — a *different*, streaming workload, not the same one with error
bars), that same preference is no longer distinguishable at n=20. Both
things are true at once, and this report states both rather than
picking the more convincing one: **the tool's answer changes with
workload in the deterministic configuration it evaluates by default**,
and **that particular reversal is not (yet) shown to survive if the
workload also has streaming arrival noise**. This is exactly the kind
of gap between a single-pass finding and a noise-checked one this
task's own §7 trap asks to be made visible, not the demonstration's own
weakness papered over.

---

## 5. Whether expert-parallel degree responds at all

**Yes — confirmed directly through `plan()`'s own interface, not only
by citing Task 21.** `ffn_ep` swept at 1/2/4, `attn_tp=1`, `domain8`,
loose memory margin (0.2, to isolate the EP axis from the TP/memory
axis):

| ep | mean tpot | throughput | SLO |
|---|---|---|---|
| 1 | 12.3316 ms | 101.888 | 0.750 |
| 2 | **11.3736 ms** | 109.936 | 0.750 |
| 4 | 14.4434 ms | 87.731 | 0.500 |

EP degree moves the objective, confirming this task's own §6.5
requirement — **but non-monotonically**, unlike Task 21's own isolated
EP sweep (which found a monotonic 4.9638 -> 4.5806 -> 4.1202 ms
improvement, ep=1 to ep=4, on a fully hand-placed, EP-colocated
deployment). The difference is placement, not a contradiction: this
report's own `_placement_for` reuses Task 32's own reference placement
for every rank the reference covers, and packs any rank the reference
does not cover — including every EP rank beyond the first — into
whatever domain slots are free, by a simple first-fit walk. At `ep=4`,
three of the four FFN EP ranks are not covered by the reference and
land split across two domains, so `predict_all_to_all`'s own domain-
split penalty (Task 21's own measured 14.93x for domain-split experts)
partially offsets EP's compute-parallelism benefit. **EP degree is kept
as a search variable** — it demonstrably moves the objective — but its
own placement is not yet optimized by this interface the way TP's
`attn_shape` search is; that is named here as a real, stated limitation
(§6.6), not hidden behind an average.

This also corrects a citation this task's own spec makes precisely
(§7): "expert dispatch never reaches the transfer paths" (Task 17) is
accurate for *M2N pricing specifically*, and describes the state
*before* Task 20/21's `predict_all_to_all` fix. After that fix, with
`install(..., collective=True)` (used throughout this report, matching
every real-compute tool since Task 20), EP dispatch **is** priced
through this project's own collective backend and **is**
placement-sensitive — which is exactly why its magnitude and direction
here depend on where the extra EP ranks land.

---

## 6. What it would take to add cost or power as an objective

**The interface does not preclude it, and the addition is
small — a scoring function, not a redesign.** `Objectives.minimize` is
already a string key into whatever a candidate's own evaluation
dictionary contains (`r[objectives.minimize]`); `evaluate()`'s own
returned dict already carries everything a cost or power model would
need as inputs (`n_completed`, `throughput_rps`, and — via the
`Candidate` it attaches — `attn_tp`, `ffn_ep`, `attn_replicas`,
`ffn_replicas`). Adding a `cost_usd_per_hour` or `power_w` objective
requires:

1. A per-GPU-hour cost or power figure on `Hardware` (a new field,
  e.g. `cost_per_gpu_hour: float` — the same kind of hardware-side
  constant `memory_margin_fraction` already is).
2. A scoring function `cost(candidate, result, hardware) -> float` that
  multiplies GPU-count (`attn_tp * attn_replicas + ffn_tp * ep *
  ffn_replicas`, all already on `Candidate`) by that per-hour figure
  and by wall-clock time, added into the result dict `evaluate()`
  already returns, under a new key.
3. Passing that key as `objectives.minimize` — no change to `plan()`'s
  own filtering/ranking logic, which is already generic over whatever
  key `minimize` names.

The one real design question it raises, not yet answered here: cost
and latency are opposed (more GPUs is usually faster and always more
expensive), so a cost objective needs its own constraint — analogous to
the SLO/throughput floors this task already built — rather than a
second `minimize` key, since `plan()`'s own ranking is single-objective
by construction. That is the natural next task's own scope, not
something this task's own interface blocks.

---

## 7. Anywhere this specification is wrong

1. **§6.5's own citation of Task 17 needs the same nuance Task 33's own
  investigation into it required** (see §5 above): "expert dispatch
  never reaches the transfer paths" was true of Task 17's own,
  pre-Task-20 finding about M2N *pricing* specifically, and remains
  true of that narrow claim — but is not the whole story once
  `install(..., collective=True)` and Task 20/21's `predict_all_to_all`
  fix are both in play, which is the configuration every real-compute
  tool in this project (including this one) actually uses. EP dispatch
  *is* placement-sensitive under that configuration, which is precisely
  why this task's own EP sweep (§5) is non-monotonic.

2. **A fixed throughput floor is not portable across workloads of
  different length**, discovered directly while building §4.2's own
  demonstration, not anticipated by the spec. The spec's own §3
  requires a throughput floor as a hard constraint but does not warn
  that a single fixed floor, reused across a workload sweep, can
  silently reject every candidate for a workload it was not calibrated
  against — worth stating explicitly as a trap of the same shape as
  the SLO-near-threshold one the spec's own §7 already names.

No other citation in the spec (the ~88% split-penalty figure, Task 28's
own feasibility finding, Task 31's own noise-floor figures) required
correction — each was checked directly against the cited report before
use, per this project's own standing practice, and matched.

---

## What shipped

- `tools/planner.py` — `Topology`/`ModelSpec`/`Workload`/`Hardware`/
  `Objectives`/`Candidate` dataclasses, `feasible_num_blocks`/
  `lane_assignment_feasible` feasibility checks, `enumerate_attn_shapes`
  (Task 32's own method, generalized to an arbitrary `Topology`),
  `evaluate`/`plan` with per-candidate constraint rejection reporting,
  and three named topology builders
  (`domain8`/`domain64`/`oversubscribed`) plus Task 32's exact fabric
  for the regression check.
- `docs/tasks/33-planner-report.md`, this report.

One commit on `task-33-planner`, stacked on `task-32-search`; nothing
under `upstream/`, `src/engine/`, or `src/integration/` touched. The
acceptance table from Tasks 29/30 (collective backend tp=4, the memory
grid at margin 0.9) reproduces bit-identical, re-checked here.
