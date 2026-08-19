# Task 19 — The blind spot at tensor-parallel degrees above one

Branch: `task-19-tp-sweep`, stacked on `task-18-blind-spot`.

177 tests pass (measurement task, no new tests), and
`python3 tools/check_import_direction.py` exits 0.

---

## 0. Correcting the record on what Task 18 actually concluded

Task 19's own S1 states Task 18 "recommended deferring the collective
path." **That is the opposite of Task 18's actual recommendation.**
`docs/tasks/18-blind-spot-report.md` S5 opens: *"Pursuing the collective
path upstream is worth doing, and expert parallelism specifically is the
reason."* Task 18 recommended pursuing it, precisely because the EP A/B
test showed a real, placement-blind bias, not a calibration matter. Task
19's cited headline figures (*"100% for a dense model and 41.5% for a
mixture-of-experts one"*) also don't match anything in that report — the
actual measured range was 100% (EP=1/TP=1, nothing invisible exists yet)
down to 76.84% (EP=4), never 41.5% at any configuration. I flag this here
rather than silently substituting the real numbers, and proceed with what
Task 19 actually needs measured — the two real, correctly-identified gaps
(TP never raised above 1; no placement ever split a replica) — since
those are genuine and worth closing regardless of what the prior report's
headline said. As it turns out, closing them *reinforces* Task 18's actual
recommendation rather than reversing it (S4).

---

## 1. The size of tensor-parallel communication as degree rises

DECODE_ATTN's `attn_tensor_parallel_size` swept 1/2/4/8, packed (all TP
ranks on one machine), real h800 compute, same ledger/units discipline as
task 18 (sums over 224 decode-phase rows; ratios, not absolute times, are
the point):

| tp | decode-step total (ms) | visible (M2N) | tp_comm | headline (visible/(visible+tp_comm)) |
|---|---|---|---|---|
| 1 | 88.860 | 61.75% | 0.0000ms (0.00%) | 100.00% |
| 2 | 86.963 | 63.10% | 0.8395ms (0.97%) | 98.49% |
| 4 | 86.777 | 63.24% | 1.0352ms (1.19%) | 98.15% |
| 8 | 86.781 | 63.23% | 1.1331ms (1.31%) | 97.98% |

**Never zero once tp>1** (the trap this task exists to avoid: task 18's
`0.000000` at tp=1 was arithmetic, not evidence — this table's tp=2/4/8
rows are the actual measurements task 18 never took). It grows
monotonically but sub-linearly (0.84 → 1.04 → 1.13 ms from tp=2 to tp=8,
diminishing increments, consistent with allreduce cost scaling roughly
with `(n-1)/n` of a fixed payload rather than with `n` directly) and stays
small in absolute terms at every degree reached here: **1.31% of a decode
step at tp=8, the largest degree measured.** Not material *in this
configuration* (a small model, tp=8 is already the practical ceiling for
its 4 KV heads under GQA). Whether it stays immaterial at production TP
degrees on a larger model is not established by this study and is flagged
as a real limit on what can be concluded from it, not glossed over.

## 2. Does it vary with placement? No — and the reason is that Frontier cannot represent the change, not that the change is genuinely absent

Same TP sweep, now with each degree's TP group split evenly across two
machines instead of packed onto one (four-and-four at tp=8 — this
project's own headline placement-penalty shape, task 11's `spread`):

```
tp=2  packed_tp_comm=0.839468ms  split_tp_comm=0.839468ms  identical=True
tp=4  packed_tp_comm=1.035203ms  split_tp_comm=1.035203ms  identical=True
tp=8  packed_tp_comm=1.133070ms  split_tp_comm=1.133070ms  identical=True
```

**Bit-for-bit identical at every degree tested**, to the same float
representation the ledger stores. Per this task's own trap (S5: *"identical
numbers... may mean the cost is unchanged, or that the model cannot
represent a change. Establish which by reading how the figure is
produced"*) — read, not inferred:
`frontier/execution_time_predictor/sklearn_execution_time_predictor.py`
and `sklearn_moe_execution_time_predictor.py` predict
`attn_tensor_parallel_allreduce_time`/`mlp_tensor_parallel_allreduce_time`/etc.
from a `RandomForestRegressor` (or the dummy-mode flat figure) fit against
profiled CSV rows keyed by **device type and tensor-parallel worker
count** (confirmed in task 18's own investigation of
`frontier/operators/families.py`'s `CommOperatorSpec.num_devices_builder`
callables — `_attn_tp_devices`/`_moe_tp_devices` return a plain device
*count*, never a rank or GPU identity, and no `Fabric`/`Placement` object
is threaded into this prediction path at all, anywhere). **The identical
figures are the second case, not the first: Frontier's own execution-time
predictor has no input through which "packed" and "split" could ever
produce different numbers, regardless of what the real cost would be.**
This is a stronger, more precise statement of task 18's blind spot than
task 18 itself could make, because task 18 never split a replica to check.

## 3. What the split-replica cost would actually be, priced by this project's fabric model

An estimate, with its assumption stated plainly: the dominant *extra* cost
a split TP group pays over a packed one is that a real ring-allreduce
spanning two domains must cross the domain boundary at least twice (once
each direction around the ring); the ranks within each domain still talk
to each other cheaply. Pricing one point-to-point hop of the same payload
size across `build_node_scale`'s own already-established scale-up vs.
cross-domain links (`engine.network.transfers.isolated_durations` — the
same primitive every topology-aware predictor in this project already
calls, not a new formula):

```
payload=65536B (64 KiB):    within_domain=1101.0ns   cross_domain=15311.0ns   ratio=13.91x
payload=1048576B (1 MiB):   within_domain=3559.0ns   cross_domain=34972.0ns   ratio=9.83x
```

This is a **lower bound**, not a full collective simulation: a real ring
allreduce pays this cross-domain penalty on some fraction of its hops
(roughly `2/n` of them for an n-way ring with one boundary crossing each
direction), not on the whole transfer, and a smarter (e.g. hierarchical,
domain-aware) allreduce algorithm could amortize the crossing further. But
it establishes the right order of magnitude and the right qualitative
fact: **a real fabric would charge roughly an order of magnitude more for
crossing the domain boundary than staying inside it, for this exact
payload size, on this project's own already-validated link model** — and
Frontier's own reported `tensor_parallel_communication_time` charges
nothing extra at all, ever, for making that same crossing.

## 4. Does Task 18's recommendation stand?

**Yes — more firmly than before, not less, because this task closes both
gaps Task 18 flagged and both closings point the same direction.**

- Task 18's recommendation rested on expert parallelism's confirmed,
  placement-blind bias (growing, and demonstrably invisible to placement).
  Tensor parallelism turns out to be **the same kind of bias**, not a
  different, safer case: identical numbers between packed and split, for
  the same structural reason (a profiled table with no fabric input) —
  just smaller in magnitude at the degrees reachable here (1.31% at tp=8
  vs. 23.16% invisible share at EP=4 from task 18). Two independently
  confirmed instances of the same blind spot make the upstream case
  stronger, not weaker.
- Task 18's own mitigating reasoning for TP — "an eight-GPU NVLink node
  is what TP normally stays inside" — turns out to be exactly the
  assumption a model requiring more parallelism than one node offers
  breaks, and this project's own headline measurement (the ~14.65x
  placement penalty from tasks 11/12, and the `spread` policy that
  produces it) is evidence that this project already studies deployments
  where a replica's ranks *do* span domains. The "TP normally stays
  inside one domain" premise is a property of *some* deployments, not a
  structural guarantee — and the moment it doesn't hold, Frontier's own
  accounting has no way to notice.

**The size of the blind spot at tp=8 (1.31%) does not, by itself, demand
urgent action — task 18's own framing is right that magnitude alone isn't
the test.** What changes the recommendation's footing is that Task 19
converts task 18's "EP is placement-blind" finding from a single data
point into a *pattern*: every within-replica communication component this
project has now actually tested (TP here; EP in task 18) is blind to
placement in the identical structural way, for the identical reason.
Pursuing the collective path upstream remains the right call, and it now
rests on two independently confirmed instances rather than one.

## 5. Anywhere this specification is wrong

- **S1's characterization of Task 18's recommendation is backwards** (S0):
  task 18 recommended *pursuing* the collective path, not deferring it.
- **S1's cited headline figures (100%/41.5%) don't match Task 18's actual
  numbers** (S0) — the real range was 100% down to 76.84%. Treated, like
  task 17's similarly-mismatched worked example, as not a literal figure
  to reconcile.
- **Pipeline parallelism cannot be tested in the architecture this
  project's other measurements use at all**, which the spec's S2.1 doesn't
  anticipate: `frontier/config/config.py` unconditionally asserts
  `num_pipeline_stages == 1` for both DECODE_ATTN and DECODE_FFN in
  pd-af-disaggregation — not a flag choice, a hard `assert`. Confirmed by
  running it (`ValueError: decode_attn_replica_config_num_pipeline_stages
  must be 1 for decode_attn, got 2`). A separate check in plain
  pd-disaggregation's unified DECODE cluster (no M2N to compare against,
  so no headline ratio, only the raw question "can this component ever be
  non-zero") confirms `pipeline_parallel_communication_time` *is* real
  and reachable elsewhere in this checkout (0.030 ms, 0.0217% of that
  cluster's own decode-step total, at pp=2) — so the component itself is
  not inert, only unreachable in the specific architecture every other
  measurement in this project has used. This is a genuine, reportable gap
  the specification's S2.1 table implicitly assumed away by listing PP
  alongside TP as if both were freely variable in the same run.
- Otherwise the specification's structure — separate "is it ever material"
  from "does it vary with placement" (repeating task 18's own S1.1
  distinction), demand the identical-packed-vs-split numbers be explained
  by *reading* the mechanism rather than inferred from the value, ask for
  an estimate of the true cost with stated assumptions — matched exactly
  what the investigation needed.

## What shipped

- `tools/run_tp_domain_probe.py` — TP sweep (1/2/4/8) × placement
  (packed/split), a pd-disaggregation PP=2 reachability check, and the
  fabric-model cost estimate. Reuses task 18's argv/ledger-reading
  machinery via import rather than duplicating it.

One commit on `task-19-tp-sweep`, stacked on `task-18-blind-spot`; no
`upstream/`, `src/engine/`, or predictor changes.
