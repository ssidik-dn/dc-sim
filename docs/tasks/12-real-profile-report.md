# Task 12 report — What the placement penalty is worth, with real compute

Branch: `task-12-real-profile` (not merged to main). `python3 -m pytest -q`
stays at 157 (unchanged, as this is a measurement task), and
`python3 tools/check_import_direction.py` passes. One new sibling tool,
`tools/run_m2n_real_profile.py`; no changes to `src/engine/`, the
predictors, or any test.

---

## 1. The table

Llama-2-7b (`llama2_7b_dense_example`, real h800 profiles present under
`data/profiling/compute/h800/`), 4 requests, 16 decode tokens, colocated vs
split attention/FFN pools, dummy mode alongside real profiles:

| mode  | placement | mean M2N (ms) | mean TPOT (ms) |
|-------|-----------|---------------:|----------------:|
| dummy | colocated | 0.978240 | 422.561361 |
| dummy | split     | 14.069760 | 423.434129 |
| dummy | **ratio** | **14.3827** | **1.0021** |
| real  | colocated | 0.978240 | 5.837484 |
| real  | split     | 14.069760 | 6.710252 |
| real  | **ratio** | **14.3827** | **1.1495** |

The M2N transfer numbers are identical between dummy and real rows, as they
should be -- this project's predictor prices the transfer from the fabric
graph regardless of the execution-time predictor's mode. Only TPOT changes,
and it changes by two orders of magnitude in what the ratio means: **1.0021x
under dummy mode, 1.1495x under real profiles.** Same network model, same
transfer numbers, confirming the task's own opening arithmetic almost
exactly (dummy decode step ~423 ms vs a real one here of 5.8-6.7 ms, putting
the transfer delta at roughly 15-22% of the real decode step -- squarely in
the "10-26%" band the spec projected from first principles, before this
task ran anything).

## 2. What fraction of the transfer penalty reaches inter-token latency

**Effectively all of it -- 100.0%, not a third, and not overlapped away.**
This needed a correction to my own first attempt at the arithmetic, which is
worth showing rather than hiding:

```
transfer delta (mean_m2n split - colocated, a TOTAL over the whole decode phase):
    14.069760 - 0.978240 = 13.091520 ms
tpot delta (mean_tpot split - colocated, a PER-TOKEN AVERAGE):
    6.710252 - 5.837484 = 0.872768 ms
```

My first pass divided these directly (0.873 / 13.09 = "6.7%") and that
number is wrong: `total_m2n_transfer_time` is a sum over every round-trip in
the decode phase, while `tpot` is `total_decode_time_after_first /
(num_decode_tokens - 1)` -- an *average per token*. Dividing a total by an
average is a units error, not a finding. Multiplying the tpot delta back up
by the same `(num_decode_tokens - 1) = 15` that `tpot` itself divides by
puts both sides on the same footing:

```
0.872768 ms x 15 = 13.091520 ms  ==  13.091520 ms  (exactly, to the last digit printed)
```

**The two numbers are identical.** The decode-step composition breakdown
(S3) confirms why: attention and FFN compute totals are exactly the same in
both placements (34.6730 ms / 50.4688 ms in both rows); only the M2N total
differs, by exactly the transfer delta; and "other" (wall minus attn minus
ffn minus m2n) is zero in both. There is no overlap and no other serialised
cost absorbing any of it -- **neither of the two candidate explanations in
the spec is what's happening.** With
`decode_attn_af_pipeline_num_micro_batch=1` (task 11's choice, carried
forward here, and worth stating plainly: there is nothing else for the
cluster to compute while this transfer is in flight, since there is only
one micro-batch), the transfer is purely serial with compute by
construction, and it shows up in the measured critical path exactly as
large as it is. Task 11's 1.0020x was not partial absorption of a real
effect -- it was dummy mode's inflated compute denominator diluting a fully
present, fully undiminished cost down to noise. The task's own framing
("say which, with evidence, or say you could not tell") anticipated a
partial answer; the evidence says neither candidate applies here, and I'm
reporting that rather than forcing a fit to one of the two options offered.

## 3. Decode-step composition (real profiles)

Mean per request, summed over the full decode phase (prefill-done to
fully-done, minus the KV hop -- see the tool's docstring for why this
window, not `tpot`'s own N-1-interval window, was used for the breakdown):

| scenario | wall (ms) | attn compute | FFN compute | M2N transfer | other |
|---|---:|---|---|---|---|
| real / colocated | 86.1201 | 34.6730 ms (40.3%) | 50.4688 ms (58.6%) | 0.9782 ms (1.1%) | ~0.0 ms (0.0%) |
| real / split      | 99.2116 | 34.6730 ms (34.9%) | 50.4688 ms (50.9%) | 14.0698 ms (14.2%) | ~0.0 ms (0.0%) |

FFN compute dominates even attention compute here (58.6% vs 40.3% colocated)
-- consistent with this model's intermediate size (11008) being larger than
its hidden size (4096) and this being a dense (non-MoE) FFN, so there's
nothing surprising there. The interesting number is **M2N's share moving
from 1.1% to 14.2%** as placement changes -- and "other" being essentially
exactly zero in both rows is itself informative: at this workload size,
Frontier's own scheduling/barrier overhead is negligible next to compute and
transfer, so the composition really is just these three terms.

## 4. Whether the conclusion changes with model size

**Attempted a second model; it was not cheap, and I'm reporting that rather
than skipping the question.** `Llama-3.1-405B-Instruct-FP8` (126 layers,
16384 hidden, 128 attention heads / 8 KV heads, dense, real h800 profiles
present) is the largest dense model with real profiles in this checkout.
Dummy-mode runs for it completed quickly, as expected (dummy mode never
touches profiling data). The first *real*-mode scenario's sklearn model
training was still running after more than 10 minutes of wall time
(6 CPU-bound joblib workers at >90% each) with zero output, at which point I
killed it rather than let it run unbounded -- training time scales with the
profiling data's size and complexity, and a model with 4x the hidden size
and ~4x the layers of the 7B case has a proportionally larger profiling
corpus to fit RandomForest models against. This is a real cost, not a
tooling problem; a from-scratch run against a new model+device combination
should be expected to take Real minutes, not seconds, the first time.

**Expected direction, reasoned from S3's composition rather than guessed:**
per-layer compute for the linear/attention projections scales roughly with
`hidden_size^2` (matrix-multiply cost), while M2N activation size scales
with `hidden_size` directly (`num_tokens x hidden_size x dtype_size`,
confirmed by `AnalyticalM2NTransferPredictor.get_activation_size`, which
this project's predictor delegates to unchanged). Going from 7B to 405B
roughly quadruples hidden size and layer count together, so total compute
should grow considerably faster than total transfer time. **The qualitative
conclusion -- placement matters, and by more than a bandwidth-only model
would say -- should persist** (task 10's latency-dominance finding is a
property of payload size relative to the fabric, not of model size), **but
the fraction of TPOT it accounts for should shrink**, not grow, for a much
larger model -- compute grows faster than the network cost it's being
measured against. I would not expect a 405B model to fall all the way back
to dummy mode's 0.2%, since M2N transfer size also grows with hidden size
(so the absolute transfer delta grows too, partially offsetting the
compute-side dilution) -- but I would expect something meaningfully below
this task's 15% for the 7B case. This is a reasoned expectation, not a
measurement, and should be labelled as such if it's cited later.

## 5. Where the specification is wrong

- **Nothing in the spec was wrong.** Every trap it named was real and hit
  exactly as described: the model-architecture fallback warning is
  cosmetically similar to, but structurally different from, a *compute
  profile* fallback (which turned out not to exist as a silent path at all
  -- `data/profiling/compute/h800/` simply lacks a directory for
  `meta-llama/Llama-2-7b-hf`, and Frontier's on-demand training path made
  that absence loud rather than silent, which is the better failure mode
  and worth noting even though the spec worried about the opposite); the
  "dummy mode set in more than one place" warning was addressed by reading
  `config.cluster_config.execution_time_predictor_config.enable_dummy_mode`
  from the parsed config object directly rather than trusting any print
  banner, and every run's JSON result carries that verified value; and "do
  not tune towards a preferred answer" is exactly what made the S2
  correction visible -- the wrong first number (6.7%) was *more*
  interesting-sounding than the corrected one (100%), and reporting the
  arithmetic error rather than keeping the more dramatic wrong number is
  the point of that instruction.
- One implicit assumption worth naming: the spec's own arithmetic in S1
  ("decode step at a realistic 10-25 ms") assumed a decode step an order of
  magnitude longer than what this specific tiny (4-request, TP1) 7B
  configuration actually produced (5.8-6.7 ms). The conclusion the spec
  anticipated held anyway (10-26% projected, 15-22% measured), but that's
  worth flagging as a coincidence of this configuration landing in the same
  band, not confirmation that "realistic" decode steps are 10-25 ms in
  general -- a larger TP degree, a bigger batch, or a bigger model would
  all move that number, in directions S4 discusses.
