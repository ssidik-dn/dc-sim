# Task 21 — Collective patterns, not all-pairs meshes

Branch: `task-21-collective-patterns`, stacked on `task-20-collective-backend`.

189 tests pass (183 existing + 6 new in `tests/test_collective_patterns.py`),
and `python3 tools/check_import_direction.py` exits 0.

---

## 0. Correcting the record on what needed fixing

Task 21's own opening premise — *"the cost path builds an all-pairs mesh
for every collective, regardless of type: `induced_links` walks every
ordered pair of participants and charges the links between them"* — does
**not** describe this project's actual `EngineCCBackend` as task 20 left
it. Checked directly: `grep -rn "induced_links" src/integration/` returns
nothing. `predict_allreduce`/`predict_allgather`/`predict_reduce_scatter`
already build exactly `n` ring edges via `_ring_edges`, never an
all-pairs mesh — confirmed in this task's own
`test_ring_crosses_boundary_twice_not_per_pair`, which passes against the
*correct* figure (229,376 B) while showing that the *wrong* figure the
spec's own premise implies (1,835,008 B — 14 rounds × 16 all-pairs × 8192
B, i.e. literally charging every cross-domain pair on every one of a
ring's sequential rounds) was never what this project's ring
implementation computed. `induced_links` (`engine.placement.placement.Placement`)
is a separate, pre-existing utility used only by `engine/cli/place.py`, a
CLI tool unrelated to `EngineCCBackend`.

**What task 20 actually got wrong, and what this task fixes**:
`predict_all_to_all`'s per-pair volume was `data_size_bytes/n`, not
`data_size_bytes/n^2`. Task 20's own report called the pair *set* correct
without checking the per-pair volume — exactly the gap this task's spec
asked to close, and the one real bug this investigation found.

---

## 1. Option A or B

**B — own it, reaffirming task 20's decision rather than revisiting it.**
The reasoning task 20 already worked through with a real ASTRA-sim
binary still holds, and this task's own findings support it further: a
genuine, built ASTRA-sim binary's analytical collective model does not
reliably price a domain-split group as more expensive than a packed one
(task 20 report S2 — measured directly, not assumed), which is the
opposite of what this project needs a collective's placement-sensitivity
to show. Delegating (Option A) would mean extracting a *traffic pattern*
from a tool whose public interface returns a *duration*
(`estimate()`) — the spec's own framing of Option A's difficulty — and
even having done that extraction, the underlying algorithm selection
would still be a black box this project can't independently verify
matches a hand-computable ring.

Owning it (B) has exactly the cost the spec names — "being wrong about it
independently of a validated implementation" — and this task is direct
evidence of that cost materializing: task 20's own owned implementation
had a real bug (all_to_all's per-pair volume). But it was found and fixed
*because* it was owned and closed-form testable — six tests, every
expected value hand-computed, the same standard `engine.network.allocator`'s
own tests already met. A delegated pattern extracted from ASTRA-sim would
not have been checkable this way at all; it would have needed to be
verified against ASTRA-sim's own output, and task 20 already found that
comparison unreliable in the direction that matters most.

## 2. The corrected measurement, against Task 20's

**Tensor-parallel (allreduce) is unchanged — confirmed by re-running it,
not assumed unaffected.** This task's fix is specific to `predict_all_to_all`;
`predict_allreduce` was never touched. Re-running task 20's exact packed-vs-split
TP sweep (`tools/run_collective_backend_study.py`, tp=4 shown; tp=2/8
equally unchanged):

| | packed | split | ratio |
|---|---|---|---|
| Task 20 (`tensor_parallel_communication_time`) | 2.628864 ms | 38.513664 ms | 14.65x |
| Task 21 (same run, re-measured) | 2.628864 ms | 38.513664 ms | 14.65x |

Bit-identical. Task 20's own inter-token-latency correction — the one the
spec's S1 cites (+5.126 ms at tp=4, ~88% over packed's 5.803 ms tpot;
task 20's own headline was measured at tp=4/8, not the single "89%"
figure the spec's S1 states, which this report notes as an
approximation of task 20's own tp=4 ratio rather than a distinct number)
— stands exactly as reported.

**Expert dispatch (all_to_all) does change, but far less than an `n`-fold
naive reading would suggest — and remains correctly placement-sensitive
either way.** Re-running task 18's EP scenario with
`install(..., collective=True)` (`tools/run_collective_backend_ep_study.py`):

| EP | placement | `expert_parallel_communication_time` | mean tpot |
|---|---|---|---|
| 2 | colocated | 0.438144 ms | 4.580558 ms |
| 2 | split (pool-level) | 0.438144 ms | 5.489998 ms |
| 4 | colocated | 0.844928 ms | 4.120234 ms |
| 4 | split (pool-level) | 0.844928 ms | 5.029674 ms |

And the domain-split-experts A/B (S1.1's own check, task 18's exact test):

```
EP=4, experts colocated: ep_ms = 0.844928 ms
EP=4, experts split across two domains: ep_ms = 12.617472 ms  (14.93x)
```

**Placement-sensitivity survives the fix intact** — colocated vs
domain-split experts still differ by ~14.9x, not because the fix
preserved this by construction, but because both the wrong (`S/n`) and
right (`S/n²`) volumes are proportional to the *same* per-pair chunk size,
scaled by the same constant across every edge; scaling every edge's
volume by the same factor doesn't change *which* edges are the bottleneck.

## 3. Was all-to-all also over-charged, and by how much?

**Yes, in the sense the spec means — the per-pair volume was `n` times too
large — but the practical effect on this project's own measurements is
small, not dramatic, because these payloads are latency-bound.** Measured
directly (same group, same placement, only the volume formula changed,
each measured in its own fresh process to avoid a monkey-patch
measurement artifact caught while checking this):

| payload | corrected (S/n²) | original (S/n) | ratio |
|---|---|---|---|
| 65,536 B (this project's own typical EP/M2N payload scale) | 0.014144 ms | 0.015147 ms | **1.07x** |
| 1,048,576 B | 0.016294 ms | 0.032351 ms | **1.99x** |
| 16,777,216 B | 0.050701 ms | 0.307602 ms | **6.07x** |

The over-charge grows toward the full theoretical `n`-fold difference only
at payload sizes far larger than anything this project's own real MoE
model (`Phi-tiny-MoE-instruct`, `embedding_dim=4096`) produces for a
per-layer dispatch — task 10's own established finding (small payloads
are latency-dominated, not bandwidth-dominated) applies here too, not
just to the M2N/KV transfers it was originally found for. **The
correction is real and worth having exactly because it fixes a wrong
volume convention, but anyone expecting this task to move the EP number
by 8x at realistic sizes would be wrong to expect that.**

## 4. The ring ordering assumed

**Domain-major**: every rank in one scale-up domain placed contiguously
before the next domain's, in `_ring_order` (`engine_backend.py`, unchanged
from task 20). A ring built this way crosses a domain boundary exactly
once per domain-to-domain transition around the cycle — two crossings for
a two-domain split, the minimum any ring over a split group can achieve.

**This is stated, per the spec's own instruction, as an assumption about
the runtime, not a property of the ring algorithm itself.** A ring
algorithm says nothing about which order its participants are visited in
— that is a scheduling/implementation choice a real collective library
makes (typically by rank ID or a topology-aware reordering), and this
project models the *best case* a topology-aware implementation would
achieve, not a guarantee that every real NCCL ring actually reorders this
way. A naive ring that visited ranks in raw registration order instead
could cross the boundary up to `n` times if that order happened to
interleave domains — this project's own `spread()` placement policy
produces exactly that kind of interleaving at the *placement* level, and
`_ring_order`'s domain-major sort is what recovers a well-formed ring from
it before pricing, not something `spread()` itself guarantees.

## 5. Do `estimate()` and the flow model agree now?

**No — the same residual difference task 20 already found and accepted,
restated rather than newly discovered.** `EngineCCBackend` does not call
`CostBackend.estimate()` at all (task 20's rewrite removed that path
entirely); a real, built ASTRA-sim binary's own `estimate()` for the same
8-way, 4-and-4-split, 64 KB group prices a domain-split all-reduce as
*cheaper* than packed (51,593 ns vs 52,856 ns) and a domain-split
all-to-all as dramatically cheaper (32,896 ns vs 105,692 ns packed) —
both directions this project needs to be false for its own placement
work to mean anything. This is acceptable, and was already accepted in
task 20, for the same reason: `estimate()`'s own multi-dimensional
collective decomposition is a real, validated behaviour of a real tool,
but it is not one this project can independently verify or reconcile with
a hand-computable ring, and forcing agreement with it would mean
adopting its answer (split is fine, sometimes better) over this project's
own, closed-form-testable one. The two paths disagree by construction,
and only one of them — this project's own — is exercised anywhere in this
codebase's real measurements; `estimate()`/`CostBackend` remain reachable
(`MockBackend`, `AstraSimBackend`) for anyone who wants to compare, but
nothing in the collective backend calls either.

## 6. Anywhere this specification is wrong

- **The opening premise's own description of "the cost path"** — an
  unconditional all-pairs mesh via `induced_links`, for every collective —
  does not describe this project's actual `EngineCCBackend`
  (S0). `predict_allreduce`/`allgather`/`reduce_scatter` already used ring
  edges before this task began; only `predict_all_to_all`'s per-pair
  *volume* (not its pair *set*) needed the fix this task actually made.
- **The cited "89%" inter-token-latency figure** doesn't appear verbatim
  in task 20's own report; the closest real number is task 20's tp=4
  ratio (+5.126 ms over a 5.803 ms packed baseline, ≈88.3%), which this
  report treats as what was meant rather than a distinct, unverifiable
  claim.
- **"Dividing the split figure by eight puts it below the packed figure,
  which is impossible"** (S1) — true in general as a warning against
  scalar correction factors, but not a description of anything this
  project's ring implementation would have produced: task 20's ring
  already priced split strictly above packed (14.38x-14.78x, task 20
  report S4), for the correct algorithmic reason, before this task
  touched anything.
- Otherwise the specification's structure — decide the delegation
  question explicitly, correct the pattern rather than apply a scalar,
  confirm allgather/reduce-scatter's relationship rather than assume it,
  check all-to-all's volume rather than trust the pair set alone, keep
  send_recv untested-because-unchanged, re-measure and compare against
  task 20's own numbers — matched exactly what the investigation needed.

## What shipped

- `src/integration/cc_backend/engine_backend.py` — `predict_all_to_all`'s
  per-pair volume corrected to `data_size_bytes // (num_devices ** 2)`.
- `tests/test_collective_patterns.py` — 6/6 required tests, every expected
  value hand-computed in a comment.
- `tools/run_collective_backend_ep_study.py` — the all_to_all fix's
  measured effect on task 18's EP scenario, collective backend selected.

Two commits on `task-21-collective-patterns`, stacked on
`task-20-collective-backend`; nothing under `upstream/` modified.
