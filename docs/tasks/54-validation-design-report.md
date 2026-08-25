# Task 54 — Hardware product validation design

Branch: `task-54-validation-design`, branched from `docs-infrastructure-handover`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`. No GPU used; every number below comes from
`tools/planner_core.py`'s own analytical/enumeration machinery
(`feasible_num_blocks`, `enumerate_joint_arrangements`, `Placement.domains_spanned`)
run directly, no Frontier subprocess, no simulation, no hardware. Fleet
notes read from both `profiling_knowledge/INFRASTRUCTURE_MAP.md` (Frontier's
own, as this task names) and `profiling_knowledge/INFRASTRUCTURE.md` (this
repository's own, added just before this task — §5 explains why the second
one turns out to be the one this task's own citations actually match).

**254 tests pass, unchanged; `check_import_direction.py` exits 0 — nothing
in Frontier or `dc-sim` was changed.** Design and analytical investigation
only, per this task's own acceptance bar.

---

## 1. Whether a forcing configuration exists (§2)

### 1.1 The single-axis view degenerates exactly as feared

Checked first, because this task's own §2 and Task 43A's own §7 both name
it as the default outcome: **every parallelism degree this project has
ever profiled, for any model, on any device, tops out at 8** —
`ModelSpec.profiled_tp`'s own comment states this as an already-established
fact ("every model in the checkout... covers tp in {1,2,4,8} only"), and
the same is true of every `ffn_ep` degree this project has actually
profiled (Task 44/45's own EP-placement studies never went past 8 either).
With machines of exactly 8 accelerators, **no single group — attention
alone, or the expert-parallel group alone — ever needs more than one
machine**, for any model this checkout can evaluate. Task 43A's own §7
result (`attn_tp=2` wins on every fabric tested, including rack-scale)
is the direct, already-measured consequence: the winning attention group
never touches more than one domain, so varying the fabric around it
changes nothing it depends on.

### 1.2 The joint (attention + expert) footprint does not

This task's own §2 does not ask only about a single group — it asks for
"the smallest combination of model, parallelism degree, **expert degree**
and workload that forces a genuine cross-machine decision," and
`pd-af-disaggregation` (the architecture every study since Task 32 uses)
keeps the attention pool and the expert-parallel FFN pool as **separate
replicas**, each with its own degree. Checked directly, not assumed:
Frontier's own `validate_frontier_shared_parallel_domains` (`frontier/config/parallel_semantics.py`,
enforced from `config.py` only `if cluster_name in {"prefill", "decode",
"monolithic"}`) is the constraint that forces `attn_tp*attn_dp ==
moe_tp*moe_ep` — and it is **never applied** when `cluster_name` is
`"decode_attn"` or `"decode_ffn"`, i.e. never for the disaggregated pools
this project actually places. **`attn_tp` and `ffn_ep` are independent
degrees in the architecture this project's own placement search uses.**

That independence is what makes a genuine forcing configuration
reachable, using only real, profiled degrees. A DECODE_ATTN replica
(`attn_tp`), a DECODE_FFN replica (`ffn_ep`, `moe_tp=1`), and the PREFILL
replica every deployment this project builds also includes (`tp=1`,
one more real GPU) together need `1 + attn_tp + ffn_ep` accelerators.
Built the real topology (`engine.physical.builders.build_node_scale(num_machines=2,
gpus_per_machine=8)` — the literal, unmodified "two conventional 8-GPU
nodes" builder, already the default in this project's own code, not a
new one written for this task) and ran `enumerate_joint_arrangements` —
the actual function `plan()` itself calls — for every `(attn_tp, ffn_ep)`
pair in `{1,2,4,8}²`, checking whether any reachable arrangement puts
both groups' every rank in the *same* one of the two domains (colocated
on one real machine) and whether any puts them in different domains
(spread across both):

| `attn_tp` | `ffn_ep` | total incl. prefill | arrangements | colocated reachable? | split reachable? |
|---|---|---|---|---|---|
| 1 | 1 | 3 | 1 | yes | no |
| 1 | 2 | 4 | 2 | yes | yes |
| 1 | 4 | 6 | 3 | yes | yes |
| 2 | 1 | 4 | 2 | yes | yes |
| 2 | 2 | 5 | 4 | yes | yes |
| **2** | **4** | **7** | **5** | **yes** | **yes** |
| 4 | 1 | 6 | 3 | yes | yes |
| **4** | **2** | **7** | **6** | **yes** | **yes** |
| 4 | 4 | 9 | 9 | **no** | yes |
| 8 | 1 | 10 | 4 | no | yes |
| 8 | 2 | 11 | 7 | no | yes |
| 1 | 8 | 10 | 3 | no | yes |
| 2 | 8 | 11 | 7 | no | yes |
| 4 | 8 | 13 | 9 | no | yes |
| 8 | 4 | 13 | 10 | no | yes |
| 8 | 8 | 17 | 0 | — (exceeds both machines combined: 17 > 16) | |

**The forcing configuration exists, and it does not need a memory-forced
degree or a contrived machine size.** Every row with total ≤ 8 admits
*both* a colocated-on-one-real-machine arrangement and a genuinely
split-across-two-real-machines arrangement — the exact "could go either
way" comparison §1 of this task's own spec asks for, not a structural
non-choice like Task 36's (where Fabric B could not reach the packed
shape at all). Rows with total in (8, 16] are real too, but every
arrangement in them is forced apart — useful as a second, contrasting
data point, not as the primary test.

**Chosen: `attn_tp=4, ffn_ep=2`** (and, symmetrically, `attn_tp=2,
ffn_ep=4` — kept as a second point, not the primary one). This is the
*tightest* margin among the genuine-choice rows (7 of 8 accelerators used
when colocated, only 1 idle) and the richest space among them (6 distinct
joint arrangements), while still resting on entirely ordinary, already-profiled
degrees.

**Model: `Phi-tiny-MoE-instruct`.** Not a new choice — it is the model
behind every one of "the reversals this project found in sizing
decisions" this task's own §1 refers to (Task 27/28's affordability and
optimum-shift findings, Task 42's memory-vanishes-under-streaming
reversal, Task 43A's own fabric-invariance result), and it is `h800`-profiled
with **no missing operator**: its own `data/config/models/Phi-tiny-MoE-instruct.json`
declares no `moe_layers_enum` at all, which per Task 38's own §5 screening
(the exact gap that made `step-moe-noquant-small`'s and `deepseek-v3`'s
own `plan()` demonstrations fail — a "mixed-layer" MoE model whose
profiling omits the dense-layer operators its architecture still needs)
means **every layer is MoE, so there is no dense-layer gap to hit.**
Confirmed further by the fact that this exact model has completed
dozens of real, non-dummy end-to-end evaluations across this project's
entire history without ever once hitting a missing-column error.

**Not memory-forced, checked directly rather than assumed.**
`feasible_num_blocks(Phi-tiny-MoE-instruct, h800, attn_tp)` is non-`None`
(feasible) at **every** tested degree (1, 2, 4, 8) up to a memory margin
of 0.98 — this model never produces the low-degree-infeasible pattern
`Llama-3.1-405B-Instruct-FP8` shows at margin 0.7 (Task 35/36's own
finding). **`attn_tp=4`/`ffn_ep=2` is a free placement choice, not a
capacity requirement** — exactly what this task's own §1 wants ("does
the planner choose the configuration that exhaustive benchmarking on
real hardware would have chosen"), as opposed to Task 36's own explicit
caveat about its own result ("memory-forced... not evidence a split
arrangement is ever competitive with a whole one").

### 1.3 What this corrects in how the question was framed

This task's own §2 anticipates two outcomes: a single group that cannot
fit eight accelerators, or (failing that) a smaller, possibly-contrived
machine boundary. **Neither was needed.** The forcing mechanism found
here is a third path, implied by this task's own wording ("parallelism
degree... **and** expert degree") but not spelled out: two *independent*
groups, each individually fitting on one real machine, whose **combined**
footprint — together with the prefill replica every real deployment
also needs — does not. This is not a contrivance in the sense this
task's own §2 worries about (no machine smaller than any real SKU is
invented); it uses exactly the hardware this project's own device
roster has always described (8-accelerator boxes) and exactly the
architecture this project's own placement search has always targeted
(`pd-af-disaggregation`, attn and FFN pools as separate, independently-sized
replicas). §5 discusses the one place this task's own citations still
need a correction.

---

## 2. The two enumerated spaces (§3)

Both spaces use `Phi-tiny-MoE-instruct`, `h800`, `pd-af-disaggregation`.
Both are built from the *same* enumeration function the planner's own
`plan()` calls (`enumerate_joint_arrangements` for two machines;
its single-group predecessor, `enumerate_attn_shapes`, degenerates to
one cell per degree pair inside one domain — see §2.2) — satisfying this
task's own "both sides must see the same space" requirement by
construction, not by cross-checking two independently-built lists
afterward.

### 2.1 One machine — sizing

**Axes:**

| axis | values | why included |
|---|---|---|
| `attn_tp` × `ffn_ep` | the 8 pairs from §1.2 with total ≤ 8 (colocated always reachable trivially — one domain, nowhere else to put anything) | the degree/expert-degree space this task's own §1 names; restricted to what fits one real machine, since that is what "validates sizing, not topology" means concretely |
| `num_blocks` | `{6, 30}` | Task 42's own established pair (memory-bound vs. plateau) — reused rather than re-derived, since it is exactly the axis that produced this project's own most-cited sizing reversal |
| workload regime | `{deterministic, streaming (N seeds)}` | Task 31/32's own established distinction; the memory-vanishes-under-streaming reversal (Task 42) is specifically a regime-dependent effect, invisible at `deterministic` alone |

**Cell count**: 8 × 2 × 2 = **32 distinct configurations.** Streaming
cells need repeated seeds for a noise floor (§3); at 5 seeds/streaming
cell (a defensible, if modest, choice — smaller than Task 32's own
`N=20`, chosen because real hardware time costs real money and this is
a first validation pass, not this project's own internal-simulator
seed study) that is `8×2×1` (deterministic, no seed) `+ 8×2×5`
(streaming) = 16 + 80 = **96 real runs.**

**Hardware time, reasoned from the fleet notes rather than measured**
(no benchmark was run for this task): a per-degree-pair-and-`num_blocks`
serving reconfiguration (~5 min: this project's own established practice
is to restart the serving process with a new TP/EP/`num_blocks` config,
not to hot-swap it) × 16 distinct configurations = ~80 min setup; 16
deterministic runs × ~3 min (fixed batch, no queueing dynamics to reach
steady state on) = ~48 min; 80 streaming runs × ~10 min (long enough to
clear `INFRASTRUCTURE.md`'s own §6.8 "100-request pilot looked stable;
the full 1,000-request run... queue growing without bound" trap — a
streaming run here needs to run past that horizon, not stop at a
comfortable-looking pilot length) = ~800 min. **Total ≈ 928 minutes
(~15.5 hours) of GPU wall-clock**, on one 8-GPU machine, serial.

**What the planner would reject as infeasible**: nothing in this space —
every cell is feasible for this model at every margin tested (§1.2).
This is itself worth confirming on real hardware, not skipping: a
"the planner never rejects anything here" result is a claim about the
*model*, and this experiment is exactly the vehicle to confirm it rather
than assume it, per this task's own instruction to say which cells the
planner rejects and whether they are "worth benchmarking anyway to
confirm the rejection is right" — here, there is nothing to confirm a
rejection *of*, which is itself the finding to report if it holds.

### 2.2 Two machines — topology

**Axes:**

| axis | values | why included |
|---|---|---|
| `(attn_tp, ffn_ep)` | `{(4,2), (2,4)}` | the two tightest genuine-choice pairs from §1.2 (7 of 8 accelerators used when colocated) — the two smallest-margin cases, kept both rather than just one so the space is not built around a single arbitrary pick |
| placement (joint arrangement) | every `(attn_shape, ep_shape)` `enumerate_joint_arrangements` reaches for that pair — **6 for each of `(4,2)` and `(2,4)`** | this *is* the planner's own search space; using it directly (rather than a hand-picked subset) is what makes "the real optimum is certainly inside it" true by construction, not by argument |
| `num_blocks` | fixed at the `Task 42` plateau value (30) | held constant deliberately, so any margin measured here is attributable to placement/network, not conflated with a memory effect this mode is not meant to validate (§1's own "one machine validates sizing, two machines validate topology" distinction, taken literally in the design) |
| workload regime | streaming only, `N` seeds | the deterministic point is cheap enough to also collect per arrangement (folded into the same runs, not a separate axis) |

**Cell count**: 6 + 6 = **12 distinct joint arrangements** (two degree
pairs × 6 arrangements each — not 6×6, since the two degree pairs are
not crossed with each other, only with their own reachable placements).
At 5 seeds/arrangement, streaming: **60 real runs**, plus one
deterministic run per arrangement folded in cheaply: **72 real runs
total.**

**Hardware time**: setup per arrangement is higher than the one-machine
case — a genuine cross-machine placement needs the physical rank-to-GPU
mapping re-pinned across two real hosts and, per `INFRASTRUCTURE.md`'s
own §6.7 (RDMA transfer can be dead while health checks pass), an actual
cross-machine data transfer verified, not merely a liveness check — budget
~10 min/arrangement × 12 = ~120 min; 12 deterministic runs × ~3 min = 36
min; 60 streaming runs × ~10 min = 600 min. **Total ≈ 756 minutes
(~12.6 hours)**, split across two real machines (so wall-clock, not
GPU-hours, halves relative to running the same total work serially on
one machine — the two machines are occupied *simultaneously* by
construction, since that is what the experiment is testing).

**What the planner would reject as infeasible**: nothing — every one of
these 12 arrangements is memory-feasible (§1.2's own confirmation
extends unchanged, since `num_blocks` is held at the plateau value here).
Every arrangement is worth benchmarking precisely because none is
rejected; the entire point is ranking arrangements the planner considers
live candidates, not confirming a rejection.

**Combined, both modes: 32 + 12 = 44 distinct configurations, 96 + 72 =
168 real hardware runs, ≈ 15.5 + 12.6 ≈ 28 hours of GPU wall-clock**
(compressible below that in elapsed time with parallel access to
multiple real machines, per the fleet notes — up to 4, per either
fleet document, §5).

---

## 3. The metrics, and the threshold decided in advance (§4)

**Definitions**, all computed over the *one* enumerated set each mode
uses (§2), never over two independently-explored sets:

- **Top-1 accuracy**: 1 if the planner's own top-ranked cell is the
  hardware-measured optimum (lowest real mean TPOT, or whatever this
  project's own established objective is for the workload — matching
  Task 32/33's own `minimize="mean_tpot_ms"` convention), else 0.
- **Top-k accuracy** (`k=2`, given the two-machine space has only 6-12
  cells per degree pair — a `k` larger than that would not be a
  meaningful bar): 1 if the planner's top choice is among the hardware's
  own best 2, else 0.
- **Regret**: `(hardware_latency(planner's choice) − hardware_latency(true optimum)) / hardware_latency(true optimum)`,
  a fraction — reported per configuration, not only pooled, since one
  configuration's regret is a different question from the space's own
  average.
- **The real margin between the top candidates**, and the noise floor
  (below), reported side by side, not folded into one number.

**Noise floor: measured, not inherited** (this task's own §4, and
`INFRASTRUCTURE.md`'s own §6.10, both insist on this specifically). For
every configuration actually benchmarked, the noise floor is the
run-to-run coefficient of variation across that configuration's own `N`
streaming seeds — computed fresh, per cell, exactly the way this
project's own `tools/seed_stats.py` already reports a 95% CI half-width
for a simulated seed sweep, applied here to real hardware repeats
instead. `INFRASTRUCTURE.md`'s own §6.10 citation ("measured between 3%
and 26% across setups on this fleet") is the *reason* to re-measure
rather than the number to use — it is direct evidence the floor moves
enough, across configurations on this exact kind of fleet, that
inheriting one from elsewhere would be a guess dressed as a
measurement.

**Success/failure, decided now, before any hardware is booked** (per
this task's own explicit instruction and known trap):

- **Success**: top-1 correct: **or**, if not top-1, the planner's chosen
  cell's regret is smaller than that configuration's own measured noise
  floor — the hardware itself cannot reliably tell the planner's choice
  apart from the true optimum, which is the strongest claim "correct"
  can mean once measurement has a floor at all.
- **Failure**: not top-1, **and** regret exceeds **twice** the measured
  noise floor — a deliberate buffer above the floor itself, so a result
  right at the boundary is not misclassified by measurement noise in
  either direction.
- **Inconclusive, reported as such, not rounded into either box**:
  regret between one and two times the noise floor. This third bucket
  exists specifically so a borderline result cannot be silently
  absorbed into "success" after the fact — this task's own named trap.

---

## 4. What could invalidate the experiment (§5, item 4)

Drawn from the fleet notes — and here is where this task's own citation
needs the correction §5 below states precisely, since the *specific*
hazards this task's own §5 item 4 names (shared occupancy, port
collisions producing plausible wrong numbers, filename collisions
overwriting results) are not in `INFRASTRUCTURE_MAP.md` at all (checked
directly: no port or filename-collision content anywhere in that file).
They are in `profiling_knowledge/INFRASTRUCTURE.md`, verbatim:

- **Shared occupancy** (`INFRASTRUCTURE.md` §6.1): a machine can show
  idle compute while another job holds its memory, and the benchmark
  either fails on allocation or "succeeds at a tiny batch size and
  silently produces a slow, unrepresentative measurement." For the
  two-machine mode specifically, this can corrupt a *comparison* rather
  than just one run: if only one of the two real machines is
  contended, every split arrangement (which uses both) inherits that
  machine's own noise while every colocated arrangement (which might
  land entirely on the *other*, clean machine, depending on which
  arrangement is drawn) does not — a confound that looks exactly like a
  real topology effect. Check free VRAM **and** utilization on **both**
  machines before every configuration, not once at the start.
- **Port collisions** (§6.3): a stale process holding a port makes a new
  server bind elsewhere while the launch log still reports success; a
  request router can then measure a different job's model entirely,
  producing numbers that are not errors, just meaningless. Verify the
  served model's identity after every launch, not only that the process
  started.
- **Filename collisions** (§6.4): with 44 distinct configurations and
  168 real runs across two modes, un-versioned output paths are a
  near-certainty to collide. Version every path by mode, degree pair,
  arrangement, and seed; stage to scratch and check before copying into
  any permanent location.
- **RDMA/cross-machine transfer dead while health checks pass** (§6.7):
  the exact failure mode that would make the *two-machine mode's own
  distinguishing measurement* — the added cost of a genuine cross-machine
  hop — silently wrong rather than absent. A health check passing is not
  evidence the transfer this mode is specifically trying to measure
  actually happened; test an actual transfer for every split arrangement
  before trusting its own number.
- **Short pilots hide diverging queues** (§6.8): already priced into
  §2's own per-run time budget (streaming runs sized to clear this,
  not to look stable early) — named again here because it is exactly
  the kind of failure that would make a regret number look smaller than
  it is, biasing the whole experiment toward a false "success."
- **Sequential-block data files** (§6.9): if this project's own workload
  generator (or Frontier's) writes records in per-category blocks the
  way the fleet notes describe, a naive `records[:n]` slice for a
  smaller pilot run would silently sample one category — shuffle with a
  fixed seed before slicing, checked directly against this project's
  own generator before assuming it is not exposed to this, rather than
  assumed safe by analogy.

---

## 5. Anywhere this specification is wrong

**The fleet-notes citation names the wrong file, precisely, and in a way
worth stating exactly rather than working around silently.** This task's
own header says "Fleet notes in `profiling_knowledge/INFRASTRUCTURE_MAP.md`"
— but §4's own specific hazards (port collisions, filename collisions,
the exact "3% to 26%" noise-floor figure) exist only in
`profiling_knowledge/INFRASTRUCTURE.md`, added to this repository just
before this task began, and this task's own §4 phrases the citation
itself as *"`INFRASTRUCTURE_MAP.md`'s predecessor recorded run-to-run
variation between 3% and 26%"* — which, read precisely, is not
attributing that figure to `INFRASTRUCTURE_MAP.md` at all; it is naming
a *predecessor* document, and `INFRASTRUCTURE.md`'s own §6.10 has that
exact figure, word for word. This task's own text is therefore
internally consistent and correct about where the number actually lives
— the header's own blanket "Fleet notes in ... `INFRASTRUCTURE_MAP.md`"
is the one place worth flagging as imprecise, since read alone it would
send a reader to a file that does not contain what this task's own body
asks for. Both documents were read for this report; §4 draws on
`INFRASTRUCTURE.md` specifically, for exactly the reason its own content
matches this task's own citations and `INFRASTRUCTURE_MAP.md`'s does
not.

**"With eight accelerators per machine, most models will fit a parallel
group inside one" is true and does not, by itself, imply the two-machine
case degenerates — that inference only goes through if the two pools
(attention, expert-parallel FFN) are considered one at a time.**
Considered jointly, with the real prefill replica every deployment also
needs, a genuine forcing configuration exists using only real, profiled
degrees (§1). This is not a correction to anything asserted outright —
this task's own §2 already lists "expert degree" as an axis to combine,
which is exactly the resolution — only to how easily the single-axis
framing (echoing Task 43A's own §7, which never combined the two pools)
could be read as the whole answer.

**Otherwise, this specification's own framing held up throughout.** The
gate (§2) is exactly as load-bearing as advertised — a naive
single-axis check would have stopped here and (per this task's own
explicit instruction) correctly reported no configuration and no cost;
the joint check is what changes the answer. The "one machine validates
sizing, not topology" framing (§1) held precisely — §2.1's own space has
no placement axis at all, by construction, because there is nowhere
else on one real domain to place anything. The noise-floor caution (§4)
is not overstated: this task's own worked threshold (§3) treats a
regret between 1x and 2x the floor as genuinely unresolved rather than
forcing a verdict, which is the discipline the spec's own "1% regret
against a 15% floor is not a success" example asks for.

## What shipped

Nothing in Frontier or `dc-sim` — a design task, per its own acceptance
criteria. `docs/tasks/54-validation-design-report.md`, this report, is
the only artifact. No hardware was booked; no simulation was run.

One commit on `task-54-validation-design`, stacked on
`docs-infrastructure-handover`. 254 tests pass, unchanged;
`check_import_direction.py` exits 0.
