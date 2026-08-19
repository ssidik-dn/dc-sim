# Task 16 — Source-side replica resolution

Branch: `task-16-source-binding`, stacked on `task-15-topology-scheduler`.

All 177 tests pass (172 existing + 5 new in `tests/test_source_binding.py`),
and `python3 tools/check_import_direction.py` exits 0.

---

## 1. Is the sending replica recoverable? Yes for one leg, exactly; no for the other

Confirmed empirically, by instrumenting a real multi-lane (`attn_data_parallel_size=8`),
multi-replica (4 `DECODE_FFN`) run and printing what every M2N call actually
receives — not assumed from reading source alone:

```
('decode_attn', 'decode_ffn', 1, 1, 2)   # forward: batch.replica_id=1 (the ATTN
('decode_attn', 'decode_ffn', 1, 1, 5)   #  replica's real id), dp lane varies 0-7
...
('decode_ffn', 'decode_attn', 1, 1, 2)   # return: batch.replica_id is STILL 1
('decode_ffn', 'decode_attn', 1, 1, 5)   #  -- the ATTN replica, not the FFN one
...                                       #  actually sending this leg
```

(Columns: `source_cluster_type, target_cluster_type, batch.replica_id,
batch.decode_attn_original_replica_id, batch.decode_attn_original_dp_id`.)

- **Forward leg (DECODE_ATTN → DECODE_FFN): exactly recoverable.**
  `batch.replica_id` is the real, sending ATTN replica's own Frontier id
  (set once, at true batch creation, in
  `BaseReplicaScheduler._create_batch`: `Batch(self._replica_id, ...)`);
  `batch.decode_attn_original_dp_id` is the specific dp lane. Together they
  identify the sending rank exactly. This is not a guess.
- **Return leg (DECODE_FFN → DECODE_ATTN): not recoverable from `batch` at
  all.** The object passed to `get_transfer_info` on this leg
  (`cluster_batch_end_event.py`'s `batch_for_transfer`, pulled from
  `raw_batch_waiting`) is the *original* batch object created on the ATTN
  side — its `.replica_id` was fixed at creation and never updated to
  reflect which FFN replica is now sending it back. Frontier itself does
  know which FFN replica is sending (the `ClusterBatchEndEvent` that
  triggers this call carries its own `self._replica_id`/`self._dp_id` for
  exactly that replica), but that identity is never passed into
  `get_transfer_time`'s signature — `(source_cluster_type,
  target_cluster_type, batch, activation_size_bytes)`, nothing else. There
  is no channel to it.

This matches the spec's own framing precisely: a source is never
*ambiguous* in principle (exactly one replica really sent it), but it can
still be *unrecoverable in practice* if nothing the predictor receives
carries the identity forward. That is exactly the return leg's situation,
and it is genuinely different in kind from task 08's `layer_id` recovery
(the caller re-derives `layer_id` from data every call site already has,
task 08's whole point) — here, the one piece of data that would answer the
question (which FFN replica) simply never reaches `get_transfer_time`, on
either call site, by construction.

**Consequence for design**: the forward leg gets an exact fix
(`_rank_within_pool`, no `bind()` involved, no `chosen_replica_id`
recorded); the return leg gets the same `bind()`-based approximation task
14 already built for destinations, applied to a source for the first time,
honestly labeled as a guess.

## 2. Side-by-side: task 15's numbers vs. task 16's (this project's own predictor)

Same scenario as task 15 (1 PREFILL, 1 DECODE_ATTN replica with 8 colocated
dp lanes, 4 DECODE_FFN replicas — one near, three symmetric far), same two
scheduler variants, same 16 requests. Only the M2N predictor differs.

| | round_robin (task 15, analytical) | round_robin (task 16, empirical) | topology_aware (task 15, analytical) | topology_aware (task 16, empirical) |
|---|---|---|---|---|
| within-domain fraction | 25.0% | 25.0% | 50.0% | 50.0% |
| mean M2N transfer time | 40.524288 ms | 0.938880 ms | 40.524288 ms | 0.938880 ms |
| mean tpot | 425.126137 ms | 422.487110 ms | 425.131137 ms | 422.492110 ms |

Two things worth separating here, because they answer different questions:

- **The predictor swap alone moved the *absolute* numbers a lot**
  (40.52 ms → 0.94 ms mean transfer time; ~2.6 ms off mean tpot) — this
  project's own fabric-routed predictor prices these small, mostly
  same-domain activation transfers far cheaper than Frontier's flat
  analytical formula does for this workload. That is a real, useful
  finding about the two predictors' absolute disagreement, but it is not
  what this task is actually asking.
- **The predictor swap did *not* create any new difference *between the
  two scheduler variants*.** Within each predictor, `round_robin` and
  `topology_aware` price identically to six significant figures
  (40.524288 ms both, under analytical; 0.938880 ms both, under empirical).
  The within-domain fraction still doubles (25% → 50%), exactly as task 15
  found — the scheduler mechanism itself is unaffected by which predictor
  is running, as it should be. But **the value of a distance-aware
  predictor, isolated exactly as the spec asks, is measured at zero** in
  this study, and S3 explains why.

## 3. Does inter-token latency move this time? Barely, and not for the reason that would matter

**No, not meaningfully — mean tpot moves by +0.005 ms (+0.00%) between
scheduler variants, same as task 15's own (different) null result.** Task
15's diagnosis was "this project's own predictor can't run in this
scenario at all, so pricing is necessarily flat (Frontier's own analytical
model)." That diagnosis was correct as far as it went, but **incomplete**
— fixing the crash (S1/S2) does not fix the actual coupling problem, and
this task's own study proves it directly rather than reasoning about it in
the abstract:

**`mean_m2n_time` is identical to six figures across every one of the
`LOAD_MARGIN` sweep's five runs** (0, 1, 2, 4, 8 — 0.938880 ms in all
five), even though the real, active scheduler's assignment distribution
changes substantially across that same sweep (`{0:2,1:2,2:2,3:2}` at
margin 0 up to `{0:5,1:1,2:1,3:1}` at margin 4+). If the predictor were
actually pricing *whichever replica the scheduler really chose*, this
number could not stay flat while the real distribution changes that much.
It stays flat because `price_transfer` is configured with
`BindingConfig(BindingPolicy.NEAREST, timing="early")` — a policy that
answers "which replica is nearest" independently, every single call,
regardless of which replica the real scheduler actually assigned that
lane to. This is task 14's original finding (`nearest` prices a route
Frontier's own scheduler may not be taking) and task 15's S3.2 finding
(pricing "against whatever the scheduler actually chose" is not reachable
through the predictor's call surface) **restating themselves inside a
scenario where they can now be measured directly, not just diagnosed**.
Every `BindingPolicy` this project has (`ROUND_ROBIN`, `LEAST_LOADED`,
`NEAREST`, `EXPLICIT`) answers its own question from its own state or the
fabric graph — none of them can be made to answer "what did the *real*
scheduler pick for *this* call," because nothing routes that fact to the
predictor. Task 16 fixed *identity recovery* (S1) — genuinely, for the
forward leg — but identity recovery and *decision-agreement* are different
problems, and this task only ever scoped the first one.

The small tpot movement that does exist (422.487 ms → 422.497 ms as
`LOAD_MARGIN` rises from 0 to 4, then flat) is real and monotonic, not
obviously noise, but it does **not** come from transfer pricing (which, as
just shown, cannot see the scheduler's real choice at all here) — it comes
from queueing: concentrating more of the 8 lanes onto the near FFN replica
changes real batch-scheduling contention at that replica, and it makes
tpot slightly *worse*, not better, as concentration increases. This is the
opposite of what the locality mechanism is supposed to buy, and it is a
genuinely useful, if small, finding: in this workload, favoring distance
at the cost of balance is a net negative on the one metric that matters,
exactly the tension task 15's report already flagged as a property of
`LOAD_MARGIN=2` and this sweep now quantifies directly.

## 4. Is lane identity reachable? Yes, on both sides of the pool that carries it, previously unused on both

`batch.decode_attn_original_dp_id` is present and correctly varying on
*both* legs of the round trip (confirmed in the same instrumented run,
S1) — it identifies the DECODE_ATTN dp lane a transfer belongs to whether
that pool is acting as source (forward leg) or destination (return leg).
Task 15's report noted this only for the source case ("all eight ATTN dp
lanes resolve to the same source rank tuple") and called it harmless
"only because all eight ATTN dp lanes are colocated" in that study.
`_rank_within_pool` now uses this identity on *whichever* side of
`price_transfer` resolves to a single, unambiguous pool — meaning the
return leg's *destination* (routing back to the correct ATTN lane) is
fixed by the same change, not just the forward leg's source. Both are
exercised directly in `tests/test_source_binding.py::test_source_identity_is_recovered`,
which prices two different dp lanes against the same fixed destination
and asserts the two prices differ and match a manually-computed reference
distance for each lane specifically (not `ranks[0]`).

**This is now the fix that would matter if lanes were ever split across
domains** — the "interesting case" §3.2 asks about, and the one this
project's studies have avoided so far (every scenario keeps DECODE_ATTN's
dp lanes colocated). With this fix in place, a future study that placed
ATTN's dp lanes across different scale-up domains would price each lane's
forward and return transfers correctly per-lane; this task's own study
did not change that scenario (kept lanes colocated, matching task 15), so
it does not itself demonstrate a case where this refinement changes a
result — only that it is now correct rather than silently wrong when one
is built.

## 5. Anywhere this specification was wrong

- **Implicit framing that fixing source resolution would let inter-token
  latency move** (§4.2's "the difference between them is the value of a
  distance-aware predictor, isolated cleanly, because everything else is
  held fixed"): not quite — everything else was *not* held fixed in the
  way that framing implies. The binding *policy* used to price the
  ambiguous side (`NEAREST`, task 14's own machinery) is itself a
  confound that has nothing to do with source-vs-destination resolution;
  it would produce the identical flat result whether the ambiguity were on
  the source or the destination side, in either task 15 or task 16. The
  spec's own §6.3 anticipated this outcome might occur and asked for the
  diagnosis regardless ("If it still does not, the diagnosis was
  incomplete and that matters more than the feature") — which is the one
  place the spec correctly declined to assume its hypothesis, same as
  task 15's spec did for its own S5.
- **"A source is not ambiguous at all... the transfer is being sent by one
  specific replica, which Frontier already knows"** (§2): true for the
  forward leg, but the return leg shows this needs a footnote — Frontier
  *does* know (the event carries `self._replica_id`), but "Frontier knows"
  and "the predictor can find out" are different claims, and the gap
  between them, for the return leg specifically, is not bridgeable through
  `get_transfer_time`'s existing signature. The spec's own escape hatch
  ("If it is genuinely not recoverable, then a policy is the fallback")
  is what actually applies here, not the general claim.
- Otherwise the spec's structure (recover exact identity first, fall back
  to policy only where genuinely unrecoverable; test the single-replica
  guard; both legs, not one; check units) matched what the investigation
  actually required, and none of its other framing needed correcting.

## What shipped

- `src/integration/binding_support.py` — rewritten: `_rank_within_pool`
  (exact per-lane recovery via `batch.decode_attn_original_dp_id`),
  `_try_resolve_pool`/`_resolve_ambiguous_side` (source and destination
  ambiguity handled symmetrically through the same `bind()` machinery),
  `price_transfer` gains an optional `batch` parameter (defaults to `None`
  — KV's call site, which never has one, is unaffected).
- `src/integration/m2n_transfer/predictor.py` — threads `batch` through to
  `price_transfer`; docstring updated to describe both resolution paths.
- `tests/test_source_binding.py` — 5 tests (the 5 required; none deleted).
- `tools/run_source_binding_study.py` — task 15's study, rerun with this
  project's own M2N predictor, plus a `LOAD_MARGIN` sweep.

Three commits on `task-16-source-binding`, stacked on
`task-15-topology-scheduler`; none touch `upstream/`.
