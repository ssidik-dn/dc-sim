# Task 43A — Consolidation: fabric as a real axis

Branch: `task-43-consolidation`, branched from `task-42-breadth`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`. `h800` throughout; no device axis, per
this task's own §2. 226 tests pass, unchanged, and
`python3 tools/check_import_direction.py` exits 0. Task 33's own
sixteen-row table and Task 36's own two-fabric result both reproduce
bit-identical, checked directly. Measurement only — no source, tool, or
test file changed; every fabric built directly from
`engine.infragraph.blueprints.clos_fat_tree_fabric` and
`engine.physical.builders.build_rack_scale`, both pre-existing.

---

## 0. The primary metric, and one definitional note it needs on this fabric axis

Per this task's own §1.5: **the network-induced change in per-token
latency, relative to a colocated, uncongested baseline, on the same
fabric**, is the one number reported for every comparison below;
component shares and link-level detail are supporting evidence only,
named explicitly wherever quoted.

**One thing the fabric axis changes about that baseline's own meaning,
stated before any numbers**: every prior task's "colocated" baseline
(Task 12/22/42, `build_node_scale`) means *same machine* — a real
scale-up (NVLink-class) link. The Clos fabrics built for this task
(`clos_fat_tree_fabric`) use `gpus_per_machine=1` throughout (Task 40's
own convention, needed so `attn_tp` directly controls how many switch
ports a group spans) — **there is no same-machine placement available
on them at all.** "Colocated" here means the best the fabric allows:
same leaf (two-tier) or same edge (three-tier), which still crosses
egress + scale-out. This is not a weaker baseline by mistake; it is
what "colocated" *can* mean once a switch fabric, not a multi-GPU
chassis, is the thing being varied. Every comparison below is still
apples-to-apples (colocated vs. split, on the *same* fabric), but the
absolute colocated figure on a Clos fabric is not directly comparable
to `build_node_scale`'s own colocated figure, and the report does not
treat it as if it were.

## 1. The cost gate (§2.1) — one evaluation, every fabric, before anything else

Phi-tiny-MoE-instruct, Task 42's own streaming configuration (32
requests, `qps=20`, `seed_argv_fix` applied), `attn_tp=2`, packed:

| fabric | GPUs | links | domains | wall-clock, one evaluation |
|---|---|---|---|---|
| F1 — leaf-spine (2-tier, os=1:1) | 128 | 768 | 128 | 9.20 s |
| F2 — leaf-spine (2-tier, os=4:1) | 128 | 768 | 128 | 8.85 s |
| F3 — three-tier (Task 40's own, os=1:1) | 128 | 1,024 | 128 | 8.81 s |
| F5 — rack-scale (`build_rack_scale`, all-pairs) | 72 | 5,328 | **1** | 9.51 s |
| F6 — leaf-spine, 72-host (matched to F5) | 72 | 432 | 72 | 8.98 s |

**Every fabric costs the same, within run-to-run noise — this task's
own worry about the rack-scale builder does not hold, at this workload
size.** F5's own all-pairs domain has 12.3x the link count of F6 (5,328
vs. 432) and more than either 128-host Clos fabric, yet its own
evaluation is not meaningfully slower (9.51 s vs. 8.81–9.20 s for the
others). This directly contradicts what this task's own §2.1 predicted
by citing Task 30 ("link count, not GPU count, drove cost there... this
fabric may be far more expensive"): Task 30's own §4 already explains
why the prediction does not transfer — at a 512-GPU fabric with a
*larger* workload, `FlowNetwork.__init__`'s own `O(n_links)` dict copy
was a real, measured cost, but Task 30's own profile also found
Frontier's general overhead (logging, scheduling) at 73.3% of total
wall-clock, *larger than every category this project's own code
contributes combined* — and at this task's own 32-request workload,
that same general overhead dominates so completely that a 12x
difference in link count disappears into noise. **The full grid is
affordable everywhere; no fabric needed reduction to representative
points**, and this is reported as a real finding about where Task 30's
own scaling result does and does not apply, not a contradiction of it.

## 2. Conclusion 1 — dividing a parallel group (Task 20/21's ~88%, Task 42's ~132% streaming)

`attn_tp=4`, streaming (`N=4` seeds), explicit placement: "packed" stays
within the smallest switch unit (one leaf, one edge); "split" is forced
across the next tier up (one spine hop for the two-tier fabric, one
core hop for the three-tier).

| fabric | packed | split | margin | 95% CIs |
|---|---|---|---|---|
| F1 (2-tier) | 10.930 ms (±2.41%) | 15.276 ms (±3.15%) | **+39.8%** | disjoint |
| F3 (3-tier) | 11.677 ms (±2.93%) | 20.570 ms (±4.39%) | **+76.2%** | disjoint |

**Held, and grew on the deeper fabric, exactly as this task's own §3
expected.** F3's own split cost is itself significantly larger than
F1's ([19.667, 21.474] ms vs. [14.795, 15.756] ms — disjoint) — a core
hop costs more than a spine hop, mechanically, and the margin tracks it.

**Oversubscription (4:1), layered on the split configuration only**
(packed traffic never leaves the smallest unit, so it cannot cross an
oversubscribed uplink at all):

| fabric | split, 1:1 | split, 4:1 | Δ | significant? |
|---|---|---|---|---|
| F1 → F2 | 15.276 ms (±3.15%) | 15.774 ms (±4.20%) | +3.3% | No — CIs overlap |
| F3 → F4 | 20.570 ms (±4.39%) | 21.164 ms (±5.26%) | +2.9% | No — CIs overlap |

This null result is exactly the case this task's own §3.1 warns about —
and it is resolved directly, not left as "inert vs. unloaded" (§3
below).

## 3. Whether network cost stays small when the fabric is contended — the headline question

**Yes, for this model and workload — but for a reason narrower than
"oversubscription doesn't do anything," proven rather than assumed.**
A direct check with `engine.network.transfers` (no Frontier, Task 40's
own precedent), on the exact crossing link the split placement above
uses:

**(a) Isolated, one flow, no sharing** — does the oversubscribed link's
own narrower capacity change a single hop's duration at all?

| payload | os=1:1 | os=4:1 | ratio |
|---|---|---|---|
| 2,048 B | 24,041 ns | 24,164 ns | 1.005x |
| 8,192 B | 24,164 ns | 24,656 ns | 1.020x |
| 65,536 B | 25,311 ns | 29,243 ns | 1.155x |
| 1,048,576 B | 44,972 ns | 107,887 ns | 2.399x |

The ratio only approaches the 4:1 oversubscription factor at payload
sizes far larger than a ring-allreduce chunk this tiny model (hidden
size 4,096) ever produces (a few KB) — Task 10's own latency-dominance
finding, generalised here to the oversubscription axis specifically,
not previously checked against it.

**(b) Loaded, eight simultaneous flows sharing the same uplink** — is
the mechanism itself real when actually given something to contend
over?

| payload | os=1:1 slowdown vs. isolated | os=4:1 slowdown vs. isolated |
|---|---|---|
| 2,048 B | 1.012x | 1.047x |
| 65,536 B | 1.362x | 2.255x |
| 1,048,576 B | **4.264x** | **6.443x** |

Every one of the eight flows shows `bottleneck_classes={'scale_out': 8}`
(`ContentionReport`'s own attribution) and `link_utilisation()` confirms
saturated links (utilisation 1.0) at peak load. **The mechanism is not
inert — it produces large, real slowdowns once the fabric is actually
loaded with enough concurrent, large-enough traffic.**

**The conclusion this pair of checks supports, precisely**: this task's
own §3.1 frames a null result as ambiguous between two causes ("the
parameter is inert, or the fabric was never loaded enough"). Neither is
quite what happened here. The parameter is real (b), and the fabric
*can* be loaded (b) — what actually explains the real Frontier runs'
own near-null result is that **this project's own real decode-step
traffic is a single TP-group's own ring, one micro-batch at a time, at
payload sizes several orders of magnitude below where either
bandwidth-limitation or contention shows up** (a). Network cost stays
small under this specific fabric-contention axis not because
oversubscription doesn't matter in general, but because nothing in this
project's own current real-compute pipeline generates traffic shaped to
exercise it — a third, more precise finding than the two the
specification's own §3.1 anticipated, established by measuring both
halves directly rather than picking one.

## 4. What the rack-scale fabric shows

**The comparison actually available, stated exactly as this task's own
§3 requires**: `attn_tp=8` is admissible and profiled on every fabric
here (Task 32's own established feasible-degree set), so this is a
*same-degree, same-model, same-workload* comparison of containment vs.
spread — not a compute-forced claim, which would need a degree no
profile in this checkout covers (Task 35's own finding, still true).

| fabric | GPUs | attn_tp=8 placement | mean tpot |
|---|---|---|---|
| F5 — rack-scale | 72, 1 domain | contained (whole domain) | 3.707 ms (±7.15%) |
| F6 — leaf-spine, 72-host (matched) | 72, hosts/leaf=6 | spread (6+2, forced across 2 leaves) | 32.288 ms (±11.84%) |

**A roughly 8.7x difference, and the mechanism is exactly what §0
predicts: F6's own `gpus_per_machine=1` convention means `attn_tp=8`
can never stay in one unit at all (`hosts_per_leaf=6 < 8`), while F5's
real scale-up domain holds all 72 GPUs at genuine NVLink-class
bandwidth.** This is a real, large, and cleanly-attributable
containment effect — but it is not the compute-forced claim Task 35
could never test (no degree here needed new profiling), and it is
partly a *bandwidth* effect too, not containment alone: `build_rack_scale`'s
own defaults (900 GB/s scale-up) are real Helios-shaped hardware
parameters, faster per-link than the Clos fabrics' own 400 GB/s
node-scale convention, and both properties — staying in one domain, and
that domain's own higher bandwidth — are genuinely what a rack-scale
design offers together in real hardware. Reported as one combined,
real number, not decomposed further; decomposing it would need a
rack-scale fabric built at the *Clos* fabrics' own 400 GB/s to isolate
containment from bandwidth, which was not built here (a fabric this
report did not construct, named as a real limit, not silently assumed
away).

**Rack-scale's own degree curve is nearly flat** — 3.142 / 3.299 / 3.707
ms at `attn_tp` 2/4/8 — because nothing ever leaves the one domain
regardless of degree, the clearest direct illustration in this project's
own history of what "the group stays whole" actually buys.

## 5. Pool separation (Task 12/42's ~14–15%)

`attn_tp=1`, colocated (same leaf/edge) vs. split (different pod/leaf),
single seed per point (this effect is 17–33%, well above the "under a
few percent" threshold this task's own §2.1 reserves seeding for):

| fabric | colocated | split | margin |
|---|---|---|---|
| F1 (2-tier) | 4.321 ms | 5.042 ms | +16.7% |
| F3 (3-tier) | 4.321 ms | 5.764 ms | +33.4% |

Same pattern as Conclusion 1: held, and grew on the deeper fabric.
F1's and F3's own colocated figures match exactly, expected rather than
a coincidence — at `attn_tp=1` there is no ring, only one single-hop
M2N transfer, and F1's leaf and F3's edge use identical link parameters
(this task's own fabric construction), so the smallest-unit hop costs
the same on both by construction.

## 6. Whether the memory-vanishes-under-streaming reversal (Task 42) holds on other fabrics

`attn_tp=1`, `N=8` seeds, F1 and F3, reusing Task 42's own two-point
memory sweep (`num_blocks=6`, memory-bound; `num_blocks=30`, plateau)
and its own colocated/split placement:

| fabric | quantity | value | 95% CI |
|---|---|---|---|
| F1/F3 (identical) | colocated, nb=6 | 5.537 ms (±15.42%) | [4.685, 6.389] |
| F1/F3 (identical) | colocated, nb=30 | 4.522 ms (±2.96%) | [4.388, 4.656] |
| F1 | split, nb=30 | 5.266 ms (±2.61%) | [5.130, 5.403] |
| F3 | split, nb=30 | 6.018 ms (±2.43%) | [5.871, 6.163] |

**Memory effect** (nb=6 vs. nb=30, colocated, both fabrics identically):
**+22.4%**. The two CIs are disjoint, but only just ([4.685, 6.389] vs.
[4.388, 4.656] — the gap is 0.03 ms) — **marginally significant**,
unlike Task 42's own `build_node_scale` finding of +11.4% with fully
overlapping CIs (clearly not significant there).

**Network effect**: F1 **+16.5%** (disjoint CIs), F3 **+33.1%**
(disjoint CIs) — both clearly significant, and F3's is roughly double
F1's, the same "grows on the deeper fabric" pattern as every other
conclusion here.

**Task 42's own ranking (network ≥ memory under streaming) holds on
F3, where network (+33.1%) clearly exceeds memory (+22.4%) with both
comparisons resting on clean, disjoint CIs — but it does not repeat as
cleanly on F1**, where memory (+22.4%) and network (+16.5%) are close
enough, and memory's own significance thin enough, that this task
reports it as **fabric-dependent agreement, not a clean generalisation**
of Task 42's own finding. This is reported exactly as found — the
reversal's *direction* (streaming narrows the historical
memory-dominates-network gap sharply) generalises; whether network
*always* ends up strictly ahead of memory once genuinely staggered does
not, on the evidence collected here.

## 7. Degree preference across the fabric axis

`attn_tp ∈ {2, 4, 8}`, each fabric's own best (most-packed) placement,
streaming, `N=4` seeds:

| fabric | tp=2 | tp=4 | tp=8 | winner |
|---|---|---|---|---|
| F1 (2-tier, 128) | 6.907 ms (±5.03%) | 10.930 ms (±2.41%) | 19.209 ms (±4.65%) | **tp=2** |
| F3 (3-tier, 128) | 7.624 ms (±4.58%) | 11.677 ms (±2.93%) | 33.641 ms (±12.73%) | **tp=2** |
| F5 (rack-scale, 72) | 3.142 ms (±8.74%) | 3.299 ms (±7.89%) | 3.707 ms (±7.15%) | **tp=2** |
| F6 (leaf-spine, 72) | 6.907 ms (±5.03%) | 10.930 ms (±2.41%) | 32.288 ms (±11.84%) | **tp=2** |

**`attn_tp=2` wins on every fabric tested, including rack-scale, where
`attn_tp=8` stays entirely within one real domain.** Per this task's
own §6 (point 1): "if the preferred configuration does not move, that
is still a result worth stating plainly: topology materially changes
cost but not the optimum, for this model and workload." That is exactly
what happened — every fabric here moves *how much* each degree costs
(by up to 8.7x, §4), but never *which* degree the search would actually
recommend. F6's own tp=2/tp=4 figures matching F1's exactly is the same
mechanism as §5's colocated coincidence: `attn_tp≤4` fits within F6's
own `hosts_per_leaf=6` exactly as it fits within F1's `hosts_per_leaf=8`,
so both stay single-leaf, identical link parameters, identical cost —
only `attn_tp=8` (which spills past F6's own leaf) diverges.

## 8. Anywhere this specification is wrong

**Its own citations, checked directly, per its own §6 instruction —
mostly accurate, one figure genuinely differs from this report's own
build:**

- **The "5,112 directed links" rack-scale estimate is close but not
  exact.** This task's own `build_rack_scale(num_racks=1)` produces
  **5,328** links, not 5,112 — the difference is every egress
  (GPU→NIC) and scale-out (NIC→leaf) link the all-pairs GPU mesh sits
  on top of, which the spec's own back-of-envelope figure (`72×71`,
  the mesh alone) did not include. Not a citation error in the sense
  this project has flagged before (a number attributed to a report that
  doesn't contain it) — a reasoned estimate that undercounted by about
  4%, close enough that it did not change §2.1's own conclusion (the
  gate showed cost is dominated by something else entirely).
- **Task 30's own 5.87 s/512-GPU figure is quoted correctly** — checked
  directly against `docs/tasks/30-path-cache-report.md`'s own §3 table
  (`n_gpus=512: 5.87 s`) — and correctly cited as the comparison point
  this task's own §2.1 uses to raise its worry about the rack-scale
  builder. The worry itself did not hold (§1 above), but the citation
  that motivated it was accurate.
- **Task 36 S7's own equal-GPU-count requirement, checked and applied**:
  quoted correctly, and this task built F6 (72 hosts) specifically to
  match F5's own 72 GPUs exactly, rather than reusing the already-built
  128-host F1/F3 pair for the rack-scale comparison, which would have
  repeated exactly the confound Task 36 §7 named.
- **Task 40's own oversubscription-reaches-cost figures (3.68x, 3.57x
  on synthetic traffic) are cited accurately** as the precedent for
  this task's own §3.1 method, and this task's own synthetic check
  (§3 above) reproduces the same *qualitative* finding (oversubscription
  is a real, loadable mechanism) even though the specific ratios differ
  (this task's own synthetic payloads and link topology are not Task
  40's), which is exactly the right thing to be citing it for — the
  mechanism, not the number.
- **Otherwise, nothing checked in this specification's own account of
  prior tasks was wrong** — the seventh figure this project's own
  history has now checked (per its own §6 framing, following six
  in earlier tasks) and the first *since Task 28* whose own citations
  held up without a correction needed, save the minor rack-scale
  link-count estimate above.

## What shipped

- `docs/tasks/43-consolidation-report.md`, this report. No source,
  tool, or test file changed — every fabric built directly from
  `engine.infragraph.blueprints.clos_fat_tree_fabric` and
  `engine.physical.builders.build_rack_scale`, both pre-existing;
  every measurement reused `engine.placement.placement.explicit`,
  `integration.install.install`, and `tools/seed_stats.py`'s own
  `seed_argv_fix`/`run_seed_study`, per this task's own "no new
  machinery" instruction.

One commit on `task-43-consolidation`, stacked on `task-42-breadth`.
Task 33's sixteen-row table and Task 36's two-fabric result both
reproduce bit-identical.
