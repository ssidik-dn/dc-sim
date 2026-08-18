# Task 11 report — The M2N activation predictor

Branch: `task-11-m2n-predictor` (not merged to main).

`python3 -m pytest -q` (157 passed: 150 existing + 7 new) and
`python3 tools/check_import_direction.py` pass. Four commits: the context
consolidation (task 09's `EngineKVContext` folded into a shared
`EngineContext`, per this task's own instruction to extend rather than
invent a second mechanism), the predictor itself, the end-to-end proof
(which required fixing a real regression the consolidation introduced, and
working around a real Frontier bug), and this report.

---

## 1. The two ratios, and the gap between them

Same workload (Llama-2-7b, dense, pd-af-disaggregation, 2 requests, 32
prefill / 4 decode tokens, dummy execution-time mode), colocated vs split
attn/ffn placement:

```
M2N transfer time:  colocated=0.187776 ms  split=2.750976 ms  ratio=14.6503
TPOT (inter-token):  colocated=422.943318 ms  split=423.797718 ms  ratio=1.0020
```

**The transfer-time ratio (14.65x) is the placement effect in isolation; the
TPOT ratio (1.0020x) is almost all of it disappearing into the decode step's
other costs.** The gap is real and large: split placement costs the M2N
predictor itself nearly 15x more, but a user watching inter-token latency
would see essentially no change (+0.85 ms out of 423 ms, +0.2%).

**This gap is not a finding about the network model — it is an artifact of
dummy execution-time mode**, and I want to be precise about that rather than
let the 1.0020x read as "M2N placement doesn't matter in practice." Dummy
mode charges a flat 1 ms per operator regardless of device or model, which
inflates DECODE_ATTN and DECODE_FFN's per-layer compute cost to a level a
real `h800` profile would not reach for this model size — 32 layers x
(attention + FFN operators) at 1 ms each dwarfs a sub-millisecond transfer
by construction. A real (non-dummy) run, with genuine per-operator compute
times, would show a larger fraction of the transfer-time penalty reaching
TPOT, because the denominator (compute time per decode step) would shrink
while the numerator (transfer time, still governed by this project's fabric
model) stays exactly the same. The direction of the gap -- transfer-time
ratio much larger than TPOT ratio -- would very likely persist; its
*magnitude* would not, and I have not measured a non-dummy run to say by how
much. That measurement is future work, not something to guess at here.

## 2. Per-call predictor cost, and whether caching is warranted

**Measured, not assumed: ~216-251 microseconds per call** (192 calls per
run, averaged; the two placements differ by ~35 us, most plausibly
measurement noise from subprocess/OS scheduling rather than a real
placement effect, since `isolated_durations` does the same amount of work
regardless of which two GPUs are involved).

More precisely, timed around `sim.run()` itself (excluding Python/Frontier
import overhead, which dominates a small run's total wall time and would
otherwise hide the real signal): **the predictor's own `total_wall_ns`
accounted for ~21.6% of `sim.run()`'s wall-clock time** in this small
(2-request) run. `cProfile` traced the cost to exactly where task 10's own
module changes and this project's existing `transfers.py` code already put
it: `network_for()` and `_path_latency_ns()` each rebuild a dict over *every
link in the fabric* (160 links here; real deployments are far larger),
dominated by `Link.id`'s string formatting, on every single call --
regardless of which one or two links the actual transfer touches.

**Caching is warranted, and the obvious key is exactly what the spec names:
placement (source/destination GPU pair) and size.** Every one of the 192
calls in a run resolves to one of only two distinct (source_gpu, dest_gpu,
activation_size_bytes) tuples -- one per direction -- because placement is
fixed for a run and activation size is constant across layers (same hidden
size, same dtype). A cache keyed on that tuple would collapse 192 calls to
2 real computations, eliminating on the order of 99% of the redundant work
measured above. **Not implemented here**, per the task's own instruction not
to add one before establishing the need -- this section is that
establishment, with numbers; building and validating the cache (with its
own invalidation story: does it need to change if the same predictor
instance served two different runs, which task 08/09/10's precedents
suggest it never does) is separate follow-up work.

## 3. Whether the small-payload penalty exceeds the bandwidth ratio

**Yes, and it matches task 10's own sweep exactly: 14.6503x at this
activation size (16 KiB), against an 8:1 scale-up:scale-out bandwidth
ratio.** This is not a coincidence -- `test_small_payload_penalty_exceeds_bandwidth_ratio`
asserts this exact value, and task 10's sweep table independently measured
14.65x at 16384 bytes using the same fabric defaults. Two independent
measurements (a unit test against a synthetic placement, and a real
Frontier run's actual `activation_size_bytes` for this model) landing on the
identical number is a strong confirmation that the predictor is doing what
task 10 designed it to do: pricing a small transfer as latency-dominated,
not bandwidth-dominated. A predictor silently ignoring the latency term
would have returned exactly 8.0x here instead -- the failure mode task 10's
whole model change exists to prevent.

## 4. Whether the derived `layer_id` could diverge from Frontier's own

**Partially verified, and the gap in verification is worth stating rather
than glossing over.** `EngineM2NTransferPredictor._current_layer_id` is a
direct copy of `ClusterBatchEndEvent._get_current_layer_id_from_batch`'s
logic (first non-completed request's `completed_layer_count`, or the first
request's if all are completed) -- confirmed by reading that method, not
guessed. That method is what `frontier/events/cluster_batch_end_event.py`
itself uses to attach `layer_id` to the FFN-to-ATTN (F2A) direction's
`M2NTransferStartEvent`, so for that direction this predictor's derivation
and Frontier's own recorded metadata are computed by the identical formula
applied to the identical `batch` object -- they cannot diverge unless
Frontier's own method changes without this project noticing.

**The ATTN-to-FFN (A2F) direction is less certain.** That path runs through
`round_robin_cluster_scheduler.py`'s `schedule_ffn_with_m2n_immediate` /
`_preflight_decode_ffn_ready_group` (the same code this task's end-to-end
run hit a real bug in -- see S5), and I have not traced whether *that* code
computes `layer_id` via the same helper or a different one before building
its own `M2NTransferInfo`. If it uses a different formula, this predictor's
derived value for A2F calls could diverge from what Frontier's own metrics
would show for the same call, even though both are internally consistent
with "the same batch, the same request state." This is worth closing before
any layer-attributed analysis is trusted for the A2F direction specifically.

## 5. Where the specification is wrong, and what else went wrong

- **Nothing in the spec itself was wrong** -- S2.1's instruction to extend
  the existing context rather than invent a second one, S4's warning that
  "if a result looks like a clean bandwidth ratio, that is a signal
  something is bypassing the latency term," and S5's request to measure
  rather than assume all held up exactly as framed, and each one caught a
  real issue (a self-introduced regression, a genuine risk this task's own
  test guards against, and a real Frontier bug, respectively).
- **A regression I introduced, caught before it shipped.** Consolidating
  `EngineKVContext` into a shared `EngineContext` (S2.1) removed `install()`'s
  only reason to import `kv_transfer.predictor`, and never added a reason to
  import `m2n_transfer.predictor` either -- so neither `Engine*Config`
  subclass was discoverable by Frontier's CLI parsing outside the unit
  tests, which import the predictor modules directly and never exercise
  `install()`'s own import graph. Only running the real end-to-end tool
  caught this (`Invalid type empirical for m2n_transfer_config_type`) --
  unit tests passing was not sufficient evidence the wiring worked, which is
  the same lesson task 07 drew about registration vs selection. Fixed by
  making both imports explicit in `install/__init__.py`, with the reason
  documented there so it isn't quietly re-broken by a future refactor.
- **A real, pre-existing Frontier limitation, unrelated to this project.**
  pd-af-disaggregation's round-robin cluster scheduler builds its FFN-lane
  bookkeeping from replica ids that increment globally across every
  `Simulator` constructed in a process, not reset per run. Two scenarios run
  back to back in one process (task 09's KV script did exactly this
  successfully, because pd-disaggregation's scheduler carries no such
  state) makes the second run's lane lookup miss, raising `"DECODE_FFN
  target_ffn_replica_id must be an exact non-negative int, got None"`.
  Confirmed by bisection: a lone run in a fresh process always succeeds; a
  second run in a process that already built one `Simulator` always fails
  the same way, independent of which placement or order. Not something to
  route around by reaching into Frontier's scheduler state from outside it
  -- `tools/run_m2n_integration.py` now runs each scenario in its own
  subprocess, which is also what every one of Frontier's own shipped example
  shell scripts already assumes (`python3 -m frontier.main`, once per
  invocation, never twice in one process).
