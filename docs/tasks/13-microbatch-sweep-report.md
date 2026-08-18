# Task 13 report — Does micro-batching pay for itself?

Branch: `task-13-microbatch-sweep` (not merged to main). `python3 -m pytest -q`
stays at 157, `python3 tools/check_import_direction.py` passes. One new
sibling tool, `tools/run_m2n_microbatch_sweep.py`; a one-line robustness fix
to task 12's `tools/run_m2n_real_profile.py` (below); no changes to
`src/engine/`, the predictors, or any test.

**The headline finding is not "micro-batching helps" or "micro-batching
hurts" — it's that the mechanism task 10/13 hypothesized for a possible
loss (N separate transfers, each paying full fixed latency) does not
happen. Frontier's own scheduler never issues more than one M2N predictor
call per layer per round-trip, regardless of how many micro-batches it is
configured for; it shrinks the transferred size instead. That single fact
explains every other result in this report.**

---

## 0. A configuration change from task 12, made necessary by the sweep

Task 12 used 4 requests. Sweeping micro-batch counts up to 8 with only 4
requests is degenerate: `AFDStageMetadata.compute_stage_token_lens`
(`frontier/entities/batch.py`) caps the number of stages at `num_reqs` in
decode phase, because each request contributes exactly one token per decode
step and a single token cannot be subdivided further ("If num_reqs <
num_stages: each request is its own stage"). Requesting 8 micro-batches
with 4 requests silently collapses to the same 4 stages as requesting 4 --
confirmed empirically before trusting any other number in this report.
**Requests raised to 8** so that N=8 is not silently equivalent to N=4;
everything else (model, fabric, decode tokens, scheduler types) is
unchanged from task 12.

## 1. The table

`llama2_7b_dense_example`, real h800 profiles, 8 requests, 16 decode
tokens, N (micro-batch count) swept over {1, 2, 4, 8}, both placements:

| N | placement | mean M2N (ms) | mean TPOT (ms) | predictor calls | activation size (B) |
|---|---|---:|---:|---:|---:|
| 1 | colocated | 1.056960 | 5.916337 | 960 | 65536 |
| 1 | split | 14.698560 | 6.825777 | 960 | 65536 |
| 2 | colocated | 0.978240 | 5.911089 | 960 | 32768 |
| 2 | split | 14.069760 | 6.783857 | 960 | 32768 |
| 4 | colocated | 0.938880 | 5.908465 | 960 | 16384 |
| 4 | split | 13.754880 | 6.762865 | 960 | 16384 |
| 8 | colocated | 0.919680 | 5.907185 | 960 | 8192 |
| 8 | split | 13.597440 | 6.752369 | 960 | 8192 |

**Predictor calls are exactly 960 at every N, in both placements.** This is
not a measurement artifact -- confirmed two independent ways: the
predictor's own `calls` counter (already tracked since task 11) and a
separate monkey-patched wrapper around `EngineM2NTransferPredictor.
get_transfer_time` recording every `activation_size_bytes` it receives
(instrumentation added from this script, at the class object, not in the
source file). **Activation size scales as exactly 1/N** (65536/N, to the
byte, at every N tested) -- confirmed the same way. So splitting into N
micro-batches does not multiply the number of transfer events Frontier
prices; it divides the size of the one event it still prices by N.

## 2. Does total transfer cost rise with N, and by how much relative to N?

**It falls, monotonically, with diminishing returns -- the opposite of the
spec's hypothesized "close to N times" outcome, for the reason in S1.**

```
colocated: N=1 1.056960ms -> N=8 0.919680ms  (0.870x of N=1;  13.0% lower)
split:     N=1 14.698560ms -> N=8 13.597440ms (0.925x of N=1;   7.5% lower)
```

Because call count is fixed at 960 and only size shrinks, each call's cost
is `latency + size/bandwidth`, with `latency` constant and `size`
approaching zero as N grows -- so the total approaches
`960 x latency` from above, never `N x (960 x (latency + size/bandwidth))`
as the "repeated fixed latency" hypothesis would require. Task 10's
latency-dominance finding is still visible here, just in a different
place than expected: it's *why* the reduction is small and saturating
(13% total possible improvement even at N=8, most of it already realized
by N=2) rather than *why* more micro-batching would cost more. A model
where the predictor was queried once per micro-batch (matching the spec's
original mental model) would show the opposite curve; that is not the
model Frontier's scheduler implements here.

## 3. Does inter-token latency fall? Is there an optimum?

**Yes, monotonically in both placements, and there is no point where more
micro-batching makes TPOT worse** -- direct consequence of S1/S2: since
total transfer cost only ever falls (never a repeated-latency penalty),
there's no mechanism by which pipelining more could hurt in this model.

```
colocated: 5.916337 -> 5.907185 ms  (0.009152 ms saved, N=1 to N=8)
split:     6.825777 -> 6.752369 ms  (0.073408 ms saved, N=1 to N=8)
```

Both curves are monotone decreasing with the expected diminishing-returns
shape (most of the gain between N=1 and N=2; N=4 to N=8 barely moves
either). "Do not tune towards a preferred answer" (spec S4): the honest
finding is that micro-batching *does* help here, but only by a fraction of
a percent for TPOT, and the reason it can't hurt is structural (S1), not
because the fabric happens to be forgiving.

## 4. Does the optimum differ between colocated and split?

**No crossover in either placement -- but the absolute size of the benefit
differs by almost exactly the fabric's bandwidth ratio, which is a real,
useful, placement-dependent number even without an optimum to tune.**

```
transfer saved, N=1->N=8:  colocated 0.137280 ms   split 1.101120 ms   ratio = 8.021
tpot saved,     N=1->N=8:  colocated 0.009152 ms   split 0.073408 ms   ratio = 8.021
```

Both ratios land on 8.021, matching `build_node_scale`'s 400:50 GB/s
scale-up:scale-out bandwidth ratio (8.0) to three figures. This is the
mechanism, not a coincidence: shrinking the transferred bytes by the same
fraction saves time proportional to `bytes_saved / bandwidth`, and split's
bandwidth is 8x narrower, so the same byte reduction buys 8x more time back
on the split path. **So there is no "optimal N" that differs by placement
in this model (both keep improving, N=8 was the best tested value for
both), but the split placement has 8x more to gain from micro-batching in
absolute terms** -- a real, placement-dependent tuning consideration even
without a crossover to locate.

Also exactly reproducing task 12's finding at every N tested: the TPOT
delta and the transfer delta agree once put on the same footing (task 12
S2's correction, quoted here because it generalises, not just held once):
`(6.825777 - 6.752369) x 15` (multiplying the per-token TPOT saving back up
by `decode_tokens - 1`, the same denominator `tpot` divides by) equals
`1.101120` ms to the last printed digit -- the full transfer benefit reaches
inter-token latency undiminished at every N, not just at N=1.

## 5. Whether the composition still sums

**Yes, at every N, in both placements.** "Other" (decode-phase wall time
minus attention compute minus FFN compute minus M2N transfer) is
indistinguishable from zero (`-0.0000 ms`, floating-point noise) across all
eight (N, placement) combinations. Micro-batching introduces no measurable
scheduling overhead in this model -- attention and FFN compute totals are
identical across every N (35.5849 ms / 50.6610 ms, unchanged), confirming
compute batching is genuinely unaffected by the AF pipeline stage count
(only the transfer sizing changes, per S1) and that Frontier is not
charging anything extra for the act of splitting.

## 6. Where the specification is wrong

- **Not wrong, but its central hypothesis about the mechanism didn't
  survive measurement, exactly as a task titled "does X pay for itself"
  should allow.** §1 of the spec reasoned "splitting a transfer into N
  micro-batches ... N smaller transfers cost close to N times one
  transfer, because the fixed per-hop latency is paid every time." That
  is a sound prediction *if* N micro-batches produced N predictor calls.
  They don't, in this Frontier version's dispatch path for this workload
  shape -- confirmed by direct instrumentation, not inferred. The spec's
  own framing anticipated this kind of outcome ("Nobody in this project
  knows which. This task finds out") and its known-traps section
  literally named "call count should scale with N... understand why
  before interpreting anything else" as the thing to check first. Doing
  exactly that is what surfaced this finding; the spec asked the right
  question and I'm reporting that the honest answer undercuts its own
  illustrative mechanism, not that the spec was factually incorrect about
  Frontier.
- **The "vary independently only with a reason" clause (S2) never came
  up** -- swept together throughout, no reason found to do otherwise.
- One limitation worth naming for whoever builds on this: this result
  describes *Frontier's cost model* for AF pipelining (which shrinks a
  single representative transfer's size rather than modeling N concurrent
  transfers explicitly), not necessarily how a real system's pipelining
  would behave with a fabric-graph-aware predictor underneath it. If a
  future task wires this project's predictor into a dispatch path that
  *does* issue N separate calls per round, the "repeated fixed latency"
  effect this task went looking for and didn't find would become real,
  and the conclusion here would not carry over unchanged.

## Appendix: a real bug fixed along the way

`_run_scenario_in_subprocess` in this script (and, found by inspection and
fixed the same way, in task 12's `tools/run_m2n_real_profile.py`) built its
subprocess command from `[sys.executable, __file__, ...]` while also
passing `cwd=FRONTIER_ROOT`. Both scripts are normally invoked as
`python3 tools/run_m2n_....py` (a relative path) from `dc-sim`'s root, and a
background/non-interactive shell in this environment was observed to launch
the subprocess with `__file__` still relative, which then resolved against
the *subprocess's* cwd (`/work/Frontier`) instead of `dc-sim`, failing with
`can't open file '/work/Frontier/tools/run_m2n_....py'`. Running the same
command in an interactive foreground shell did not show the bug, which is
exactly the kind of environment-dependent fragility worth not trusting.
Fixed by resolving `__file__` to an absolute path once, at import time, in
both scripts.
