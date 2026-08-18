# Task 14 — Binding: which replica receives a transfer

## 1. What was built

- `src/engine/placement/binding.py`: `BindingPolicy` (ROUND_ROBIN, LEAST_LOADED,
  NEAREST, EXPLICIT), `BindingState`, `Candidate`, `bind()`. Every policy
  breaks ties by ascending `replica_id`; NEAREST additionally prefers same-domain
  candidates outright, then fewest hops.
- `integration/cc_backend/comm_groups.py`: `register_pool` now takes an optional
  `replica_id` (defaults to registration order); `resolve_pool_candidates`
  returns every registered `(replica_id, ranks)` for a cluster type instead of
  raising when there is more than one.
- `integration/context.py`: `BindingConfig(policy, timing)` — `timing` is
  `"early"` or `"late"` — threaded into `EngineContext` as an optional trailing
  field (`None` by default).
- `integration/binding_support.py`: `price_transfer()`, the shared pricing path
  both predictors now call once a destination pool has more than one candidate.
- Both `EngineKVCacheTransferPredictor` and `EngineM2NTransferPredictor` delegate
  to `price_transfer` and record every binding they make in `.bindings`; the
  raise is unchanged when `ctx.binding is None`.
- `tools/run_binding_study.py`: the real multi-replica Frontier sweep.
- 7/7 required tests in `tests/test_binding.py`. Full suite: **165 passed**,
  `check_import_direction.py` clean.

## 2. The study

One PREFILL replica, four DECODE replicas, on a `build_node_scale` fabric.
DECODE replica 0 shares the PREFILL replica's scale-up domain; replicas 1-3 are
each on their own separate machine (symmetric among themselves, one scale-out
hop away). 12 requests, `decode_tokens=8`, dummy execution-time mode (this study
is about the transfer/binding decision, not compute). Swept: `{round_robin,
least_loaded, nearest} x {early, late}`, 6 real Frontier runs via
`tools/run_binding_study.py`. EXPLICIT was excluded — it requires a mapping
registered into `state.explicit_map` by something outside `binding.py` before
`bind()` can even be called, so a script sweeping it would be deciding the very
answer it's supposed to measure.

Measured (`PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_binding_study.py`):

| policy | timing | mean KV transfer time | mean tpot | our binding distribution |
|---|---|---|---|---|
| round_robin | early | 16.787951 ms | 358.000000 ms | {0:3, 1:3, 2:3, 3:3} |
| round_robin | late | 16.787951 ms | 358.000000 ms | (none committed) |
| least_loaded | early | 16.787951 ms | 358.000000 ms | {0:3, 1:3, 2:3, 3:3} |
| least_loaded | late | 16.787951 ms | 358.000000 ms | (none committed) |
| nearest | early | **2.685292 ms** | 358.000000 ms | {0:12} |
| nearest | late | 16.787951 ms | 358.000000 ms | (none committed) |

Frontier's own actual DECODE assignment (from `RoundRobinClusterScheduler`,
offset-corrected — see S3): `{0:3, 1:3, 2:3, 3:3}` in every run, regardless of
which policy or timing our predictor was configured with.

## 3. Does our binding agree with Frontier's own choice?

**This is the most important finding in the task, and the answer is: mostly
not, and the reason is structural, not a bug.** Our binding policy only decides
what number the predictor *returns as a price*. It has no wiring into
Frontier's actual scheduling at all — `RoundRobinClusterScheduler` (the default
`ClusterSchedulerConfig`, unrelated to `KVCacheTransferType`) independently and
obliviously round-robins DECODE's request queue in `_schedule_decode_lane_round_robin`,
with zero knowledge of what our predictor guessed when it priced the transfer
earlier. There is no feedback path from `bind()` to Frontier's cluster
scheduler in either direction.

Measured agreement, after correcting for an id-numbering mismatch (below):

| policy (early) | agreement with Frontier's actual assignment |
|---|---|
| round_robin | 10/12 (83.3%) |
| least_loaded | 10/12 (83.3%) |
| nearest | 3/12 (25.0%) |

`round_robin`/`least_loaded` agree most of the time only because both cycle
through the same 4 replicas in roughly the same request order — two
independent round-robin cursors that happen to line up, not because either
reads from the other. It is not 12/12: our predictor is called at KV-transfer
time (prefill completion), Frontier's DECODE scheduler assigns at arrival, and
those two moments don't always fall in identical request order, so even two
"matching" round-robins can drift by a step. `nearest` agrees only 3/12 — close
to the 1-in-4 a policy that always names the same replica would get by chance
against an independent 4-way round-robin, which is exactly what's happening:
nearest's answer is not wrong, it is simply irrelevant to what Frontier
actually does.

**The id-numbering mismatch**: our own replica ids are per-pool (`register_pool`'s
`len(existing)`, task 14 §2.2), so DECODE's four replicas are 0-3. Frontier's
`Replica.id` (`frontier/entities/base_entity.py`'s `generate_id`) is a single
counter shared across *every* cluster type in construction order — the PREFILL
cluster's one replica is built first and takes id 0, so DECODE's four replicas
land on 1-4. Raw ids are not comparable without subtracting that offset first;
`run_binding_study.py` does this (`frontier_offset = min(frontier_choice.values())`)
before computing agreement. This offset is itself worth flagging: anyone
comparing "our replica_id" against "Frontier's replica_id" without knowing this
would silently compute 0% agreement for a scenario that's actually running the
same round-robin, purely from an id space mismatch — not from behavior.

## 4. Early vs. late

| policy | early | late | delta |
|---|---|---|---|
| round_robin | 16.787951 ms | 16.787951 ms | +0.000000 ms (+0.00%) |
| least_loaded | 16.787951 ms | 16.787951 ms | +0.000000 ms (+0.00%) |
| nearest | 2.685292 ms | 16.787951 ms | **+14.102659 ms (+525.18%)** |

`round_robin`/`least_loaded` show no early/late difference here because, on
this symmetric fabric (replicas 1-3 equidistant), the mean-over-all-4-candidates
that late timing always returns happens to be close to what round-robin's
long-run average would be anyway ((3559 + 3×34972)/4-shaped, i.e. mean over all
candidates including the near one) — and round-robin/least-loaded's *early*
answer, for any single call, is exactly one candidate's real cost, which
averages out to the same figure at n=12. The real cost of late binding only
shows up when a policy's whole point is to deviate from the mean — `nearest`'s
early answer prices the one candidate it always actually means to use (2.69 ms);
its late answer prices the mean over all 4 candidates including the three
expensive cross-domain ones (16.79 ms), a >6x overstatement of what nearest
binding would actually cost if committed early. **The cost of pricing without a
committed destination is not a fixed overhead — it is proportional to how much
a policy's real answer would have differed from the population mean, so it is
largest exactly where late binding is least representative.**

A second, non-obvious result: **late timing makes the policy irrelevant to the
price.** `binding_support.py`'s "late" branch computes `mean(isolated_durations(...)
for all candidates)` without ever calling `bind()` — so all three policies
report the identical 16.787951 ms under late timing. This is a direct
consequence of the design choice below, not an oversight: late timing models
"we don't know which candidate Frontier will end up choosing," and once
that's the premise, *which* policy we'd have consulted if we had known is moot.

## 5. How a late-bound transfer was priced, and what was rejected

`price_transfer` (`binding_support.py`) prices a late-bound transfer as the mean
of the isolated transfer duration to every candidate destination, and returns
`chosen_replica_id=None` (no destination is ever committed to). Two alternatives
the spec offered were considered and rejected:

- **Price against the expected destination under policy.** Rejected: this
  collapses back to what early binding already does — if the pricing model is
  going to consult the same policy either way, "late" stops meaning anything
  different from "early." The whole point of late timing is that Frontier
  itself hasn't decided yet at the moment we're asked to price the transfer
  (S4 confirms this for KV: the real destination is chosen inside
  `on_kv_cache_arrival`'s later `schedule()` call, not at
  `KVCacheTransferStartEvent`).
- **Price against nearest, and record the error.** Rejected: this needs a
  reconciliation step *after* Frontier's own choice becomes known, to compare
  against and correct the earlier guess. Neither predictor's call pattern has
  a hook for that — `get_transfer_time` is called once, synchronously, at
  transfer start, and nothing calls back into it when the destination is
  later confirmed (the same one-shot-completion-time constraint both
  predictors' module docstrings already describe for contention).

Mean-over-candidates was chosen because it is the only one of the three that
doesn't need information the predictor doesn't have at call time, and it
degrades gracefully: with one candidate it *is* the early/single-replica
answer (§S4's `test_single_replica_unchanged` guard), and with several it is
an honest statement of "not committed," rather than a masked commitment.

## 6. Does nearest beat round-robin, and by how much?

Yes, decisively, but only under early timing, and only because this fabric
was built for that asymmetry (one near replica, three symmetric far ones —
the same shape `test_nearest_beats_round_robin_on_a_split_fabric` hand-verifies):

- **Early**: nearest = 2.685292 ms, round_robin = 16.787951 ms — nearest is
  **84.00% cheaper per transfer** (round_robin pays the far replicas' cost
  3 times out of 4).
- **Late**: nearest = round_robin = 16.787951 ms — identical, because late
  timing ignores policy entirely (S4).

`mean_tpot` is unchanged (358.000000 ms) across every single one of the six
runs, including nearest/early vs round_robin/early. This is not a measurement
error: KV transfer happens once, before decode starts (task 09's own finding —
Frontier's `ttft` doesn't include it either), so its cost cannot reach
inter-token spacing during decode at all in this scenario. Where the M2N
predictor's transfer sits *inside* the decode loop and can reach tpot directly
(task 12), the KV predictor's transfer sits *before* it and structurally
cannot. Anyone expecting `run_binding_study.py`'s KV-path numbers to show up in
tpot the way task 12's M2N numbers did would be wrong to expect that — the two
transfer types differ in exactly this way, and it's a difference this project
established once already (task 08 S1) but is worth restating here since it's
easy to assume "faster transfer" always means "faster tokens."

## 7. Where the spec's own framing needed correcting

- The spec says "Frontier picks its own destination too — establish whether
  our binding disagrees with Frontier's actual choice." True, but the deeper
  finding is that this isn't really a disagreement question at all: our
  binding **cannot** disagree or agree in any causal sense, because nothing
  connects the two. Frontier schedules DECODE requests via its own
  `RoundRobinClusterScheduler`, entirely independent of whichever
  `BindingPolicy` our KV/M2N predictor was configured with. "Agreement" as
  measured here is coincidence between two independent round-robin cursors,
  not validation of a shared decision — a materially different claim from
  what "disagree" suggests, and worth stating plainly rather than reporting a
  percentage without the caveat.
- The spec's "expected destination under policy" pricing option for late
  timing, on inspection, isn't actually a third option distinct from early
  binding at all (§5) — it's early binding under another name. That
  narrowed the real choice down to two, not three.
- Comparing our replica ids against Frontier's own required discovering and
  correcting for an id-numbering offset the spec doesn't mention
  (`Replica.id` is global across cluster types, not per-pool, §3). Skipping
  this would have silently produced a near-zero, wrong "agreement" number for
  a scenario that was in fact closely matched.
