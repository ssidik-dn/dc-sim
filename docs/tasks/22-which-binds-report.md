# Task 22 — Which constraint binds?

Branch: `task-22-which-binds`, stacked on `task-21-collective-patterns`.

189 tests pass (measurement task, no new tests), and
`python3 tools/check_import_direction.py` exits 0.

All figures below are the mean of **3 seeded runs** per configuration
(different Poisson arrival draws), per this task's own trap about
saturation-edge variance; standard deviations are given where they matter
(the memory sweep, S6.2) and were small relative to the effect sizes
reported. Real h800 compute profiles throughout, `Phi-tiny-MoE-instruct`.

---

## 0. A citation that doesn't match the record

Before the measurement: this task's S2.2 attributes to task 12 *"attention
at 99% utilisation against FFN at under 3%."* `docs/tasks/12-real-profile-report.md`
contains no such figure — checked directly (`grep`), not assumed. What it
does contain is the pair this task's own S1 *also* cites correctly (34.67 ms
attention vs. 50.47 ms FFN, per decode step), and that pair says the
opposite of "attention at 99%": FFN takes the *larger* share of the step
(58.6%/50.9% vs. attention's 40.3%/34.9%, task 12's own table). A
configuration where attention is at 99% and FFN under 3% would require
FFN to be nearly idle while doing the *more* expensive work per call —
not impossible in some other lopsided replica ratio, but not anything task
12 measured or reported. Noted here rather than silently reconciled; the
actual measurement below (S3) still looks for the same *kind* of signal
this citation was gesturing at (a pool running near-saturated while the
other idles), it just doesn't take the specific numbers as ground truth.

## 1. Which constraint binds

**Memory, when it is scarce; the network's own effect, once memory is
not.** Compute-ratio mismatches are real but did not dominate either, in
every configuration measured. Restated precisely:

- Below a KV-capacity threshold, batch size is capped by blocks, not by
  arrival rate or anything else — inter-token latency is up to **2.4x**
  worse (30.06 ms vs. 12.33 ms) and throughput up to **2.8x** worse
  (36.9 vs. 101.9 req/s) purely from that cap, dwarfing every other effect
  this study or any prior one in this project has measured.
- Past that threshold, memory stops mattering (batch size, throughput,
  and tpot all plateau — S2), and the network's own placement penalty —
  this project's long-standing subject — resumes being the largest lever
  actually available: 18–24% of a decode step (S4), not swamped by
  anything else once memory is adequately provisioned.
- Compute-ratio mismatches move utilisation substantially (FFN busy
  41–86% of wall time depending on ratio, attention 17–50%) and move
  tpot/throughput mostly through *total replica count* rather than a hard
  wall — real, worth getting right, but it showed up as a gradient in
  these measurements, not a cliff (S3).

**The answer changes with configuration, which is the point of this
task**: a placement decision measured against a memory-starved replica
would be answering a question memory already dominates the answer to.
Every placement result this project has produced (tasks 09–21) was
measured with `num_blocks` generously provisioned relative to its own
small request counts — never near this cliff — so those results stand,
but they were never at risk of being memory-dominated in the first place;
this task establishes *why* that was safe, not that it was accidental.

## 2. The shape of the memory edge — a cliff, with a plateau on both sides

`block_size=16`, 48 tokens/request (32 prefill + 16 decode) → 3 blocks/request,
so `num_blocks` has a direct concurrent-capacity reading
(`num_blocks // 3`). 32 requests, Poisson arrivals at qps=20 (colocated
placement; split repeats the same sweep, S4):

| `num_blocks` | capacity | mean batch | max batch | throughput (req/s) | tpot (ms) |
|---|---|---|---|---|---|
| 6 | 2 | 2.00 | 2 | 36.921 | 30.0575 |
| 9 | 3 | 2.91 | 3 | 48.245 | 23.0448 |
| 15 | 5 | 4.57 | 5 | 67.728 | 16.5263 |
| 30 | 10 | **8.00** | **8** | **101.888** | **12.3316** |
| 60 | 20 | 8.00 | 8 | 101.888 | 12.3316 |
| 120 | 40 | 8.00 | 8 | 101.888 | 12.3316 |

**A cliff, but not the mechanism this task's own S2.1 anticipated.**
Achieved batch size tracks capacity exactly while capacity binds (2, 3,
5 — matching `num_blocks // 3` precisely), then hits a hard plateau at 8
the moment capacity stops being the limiting factor (`num_blocks=30`
onward — three further doublings of capacity produce *zero* further
change in batch size, throughput, or tpot). That plateau is itself a
second, different constraint becoming visible once memory stops binding
— most likely the arrival process itself (mean 50 ms between arrivals
against a per-request decode phase on the order of 150–200 ms at this
tpot) rather than anything this study varied; not chased further since
it is outside this task's own scope, but worth naming rather than leaving
as an unexplained flat line.

**The mechanism observed was admission queueing, not eviction.**
`request.get_total_preemption_count()` was **zero in every one of the 18
memory-sweep runs**, including the two most capacity-starved
configurations. vLLM v1's own admission control (`enable_preemption=True`
by default) simply delays a new request's admission until a slot frees,
rather than evicting one already decoding, in this workload. This task's
own S2.1 framed the edge as "forces eviction or rejection" — the actual
mechanism is a real cliff (batch size hard-caps, tpot/throughput both
degrade sharply) but through queueing delay before decode admission, not
through interrupting a request already in flight. Both are legitimate
"cliff" shapes; they are not the same mechanism, and a report that
assumed eviction without checking would have described the wrong one.

## 3. Whether either pool idles

At no replica ratio tested did either pool reach the extreme S2.2 asks to
look for (99%/3%) — see S0 for why that citation itself doesn't hold up.
What was actually measured, at 32 requests / qps=20, 3 seeds per ratio:

| attn : ffn | attn utilisation | ffn utilisation | tpot (ms) | throughput (req/s) |
|---|---|---|---|---|
| 1:1 (every prior task's own ratio) | 18.1% | 42.8% | 19.1716 | 66.913 |
| 2:1 | 17.4% | 82.5% | 11.5137 | 128.777 |
| 1:2 | 34.8% | 41.2% | 11.5101 | 128.832 |
| 2:3 (task 12's own estimated balance point) | 50.0% | 70.2% | 7.3154 | 247.862 |
| 3:2 | 27.3% | 86.2% | 9.0977 | 202.921 |

**FFN is the consistently busier pool at every ratio tested, including
the ratio (2:3) meant to balance it — the closest to balanced is actually
1:2, not 2:3.** Task 12's own ~2:3 estimate came from a *single-replica*
per-step compute-time ratio (34.67 ms : 50.47 ms); it does not carry over
cleanly to a *replica-count* ratio once queueing, arrival timing, and the
FFN cluster's own `orca` scheduler (batch-mode, distinct from
`DECODE_ATTN`'s `vllm_v1`) are in the loop — a real, useful correction
this measurement makes to a reasoned-but-unconfirmed estimate, the same
kind of correction task 12 itself made to a naive expectation. **No ratio
here idles one pool "substantially" in the sense S2.2 asks about** (no
pool ever drops below ~17% or rises above ~86%) — this project's compute
balance is a real, gradient inefficiency at these replica counts and this
workload, not the dominant, "network is irrelevant beside it" case S2.2
raises as the most important possible finding. It did not happen, at
these ratios; this is reported plainly rather than reached for.

**A scope limit worth stating**: this study's own workload (32 requests,
one Poisson burst) may not sustain load long enough to reach whatever
idling this project's larger evaluation harnesses might show under
continuous, higher-QPS traffic. The gradient found here is real for this
workload; a harder idling cliff at a different, more sustained load
profile is not ruled out by this measurement, only not observed in it.

## 4. The interaction: does the network penalty grow or shrink with memory capacity?

Same memory sweep, run at both `colocated` and `split` placement (the
established ~15%-of-decode-step configuration, tasks 11/12):

| `num_blocks` | capacity | colocated tpot (ms) | split tpot (ms) | network penalty | colocated M2N (ms) | split M2N (ms) |
|---|---|---|---|---|---|---|
| 6 | 2 | 30.0575 | 37.3199 | **+24.16%** | 0.9389 | 13.7549 |
| 9 | 3 | 23.0448 | 28.0906 | **+21.90%** | 0.9578 | 13.9025 |
| 15 | 5 | 16.5263 | 19.8042 | **+19.83%** | 0.9947 | 14.1977 |
| 30 | 10 | 12.3316 | 14.6052 | **+18.44%** | 1.0570 | 14.6986 |
| 60 | 20 | 12.3316 | 14.6052 | +18.44% | 1.0570 | 14.6986 |
| 120 | 40 | 12.3316 | 14.6052 | +18.44% | 1.0570 | 14.6986 |

**The network penalty shrinks as memory capacity rises — from +24.16% at
the most capacity-starved point to +18.44% once memory stops binding —
even though the absolute M2N transfer time grows slightly at the same
time** (0.9389 ms → 1.0570 ms colocated; 13.7549 ms → 14.6986 ms split, a
~13% and ~7% rise respectively, from larger batches carrying larger
activation payloads). **The mechanism, not just the direction**: a bigger
batch means a bigger transfer (more bytes), but it also means more
attention/FFN compute happening in the same decode step to amortise that
transfer's largely-fixed latency component against — task 13's own
established finding (micro-batching pays for itself by overlapping
transfer with compute) generalises here to batch size directly. Compute
scales with batch size roughly linearly; the transfer's fixed latency
term does not scale with batch size at all (only its bandwidth-bound
component does, and weakly at that, task 10's own latency-dominance
finding for small payloads) — so the denominator (tpot, dominated by
growing compute) grows faster than the numerator (the fixed-latency-heavy
transfer cost), and the network's *share* falls even while its *absolute*
cost rises. Both directions are real; conflating them would be exactly
the units error this task's own S6 (citing task 17) warns against.

## 5. Do the network results in this project remain the interesting ones?

**Yes, but conditionally, and this task is what earns that "yes" rather
than assuming it.** Every prior placement measurement in this project
(tasks 09–21) was taken with KV capacity far past this task's own
plateau point (`num_blocks=128` against 4–32 requests, in almost every
prior tool) — comfortably in the region where S2 shows memory does not
bind. Those results were never at risk of being a memory artefact, but
this task is the first time that was actually checked rather than
assumed. Within that safe region, the network's 15–24% share of a decode
step is real, substantial, and — per S4 — the interaction with memory
capacity that this task's own §2.3 anticipated cuts in the network's
*favour*, not against it: as deployments provision adequate memory (the
realistic, intended operating regime), the network's *share* of the
decode step, and therefore the value of getting placement right, does not
shrink to nothing; it settles at roughly 18%, not far below its
capacity-starved value of 24%. Compute-ratio mismatches (S3) are real and
worth correcting but did not, at any ratio tested, produce an effect size
larger than the network's own placement penalty. **The one qualification
this task adds, plainly**: none of that holds if a real deployment is
running closer to its own memory cliff than every study in this project
has (deliberately or not) stayed away from — in that regime, this task's
own S2 says memory dominates everything, network included.

## 6. Anywhere this specification is wrong

- **The "99% attention utilisation against FFN at under 3%" citation to
  task 12** (S2.2) does not appear anywhere in that report and
  contradicts the actual figures task 12 measured (FFN takes the *larger*
  share of a decode step, not the smaller) — S0.
- **"Frontier exposes [memory] through block count and block size"**
  (S2.1) undersells how many *distinct* knobs there actually are: a
  global `num_blocks`/`block_size` pair plus independent per-cluster
  overrides for prefill/decode/decode_attn/decode_ffn — establishing
  "the honest knob" meant confirming the per-cluster override
  (`--cluster_config_decode_attn_replica_scheduler_config_num_blocks`)
  isolates DECODE_ATTN specifically without touching the other clusters'
  own generous provisioning, which the spec's framing doesn't call out as
  a thing to check.
- **§2.2's "sweep the ratio around that point"** (task 12's ~2:3 estimate)
  implicitly assumes a single-replica per-step compute-time ratio
  translates directly into a balanced *replica-count* ratio. It doesn't,
  cleanly, once queueing and two different replica schedulers
  (`vllm_v1` for DECODE_ATTN, `orca` for DECODE_FFN) are both in play —
  S3's own finding (1:2 balances more evenly than 2:3 in this workload)
  is the corrected version of that assumption, not a contradiction of the
  reasoning that produced it.
- **A mechanical requirement the spec doesn't anticipate**: sweeping
  DECODE_FFN's replica count above DECODE_ATTN's dp-lane count hits
  Frontier's own static M2N lane-assignment invariant ("must give every
  target replica at least one decode-attn lane") — not a compute-balance
  finding, a wiring requirement worked around by setting
  `attn_data_parallel_size = ffn_replicas` (documented in
  `tools/run_compute_balance_study.py`'s own argv comment).
- Otherwise the specification's structure — vary memory to find the edge,
  vary the replica ratio to find idling, hold placement fixed and vary
  memory to see the interaction, report both throughput and latency,
  don't tune toward a preferred answer — matched exactly what the
  investigation needed, including correctly anticipating (S5) that the
  most useful thing this task could establish either way was whether the
  project's whole framing needed reconsidering, and being honest that it
  mostly didn't.

## What shipped

- `tools/run_memory_edge_study.py` — the KV-capacity sweep (S2.1) and the
  colocated-vs-split interaction (S2.3).
- `tools/run_compute_balance_study.py` — the DECODE_ATTN:DECODE_FFN
  replica-ratio sweep (S2.2).

One commit on `task-22-which-binds`, stacked on
`task-21-collective-patterns`; no `upstream/`, `src/engine/`, or
predictor changes, per this task's own acceptance criteria.
