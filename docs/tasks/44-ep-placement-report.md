# Task 44 — Expert-parallel placement in the search

## 1. What was missing, and what changed

Task 33 added expert-parallel degree (`ffn_ep`) as a search variable and found the
objective moved non-monotonically with it. Task 33's own report attributed this to
placement, not degree: `_placement_for` reused Task 32's reference placement (from
`enumerate_attn_shapes`, attention-TP only) for every rank it covered, and packed
every other rank — including every FFN expert-parallel rank beyond the first —
into whatever domain slots were left, by simple first-fit. Degree was searched;
placement was not.

`enumerate_joint_arrangements(topology, attn_tp, ffn_ep)` (`tools/planner_core.py`)
now enumerates the attention TP group and the FFN expert-parallel group *together*,
against one real deployment on one real fabric, the same way Task 41 enumerated
replica arrangements: build several raw candidates (`packed`, `spread`, an explicit
"packed-if-it-fits" fallback, and 60 `fragmented` seeds), then collapse them under a
canonical signature. `plan()`'s main loop now calls this once per `ep` value and
iterates over `(attn_shape, ep_shape)` pairs instead of iterating `enumerate_attn_shapes`
alone and choosing `ep_shape` implicitly via first-fit. `_placement_for` in
`tools/planner.py` now takes its reference placement from this joint enumeration,
so every rank of replica 0 of *both* pools — not just the attention TP group — is
placed by search rather than by fallback.

The dead `ep_split: bool` field on `Candidate` (present since Task 33, never read by
any placement logic, zero test coverage — confirmed by `grep -rn "ep_split" tests/
tools/` before removal) was deleted outright and replaced by the real
`ep_shape: Tuple[int, ...] = (1,)` field, per this project's own convention of not
carrying unused scaffolding forward.

## 2. How expert placement is enumerated, and how it differs from replica placement

Task 41's reasoning — several groups, each with a placement, combinations
multiplying, isomorphic arrangements collapsing under a canonical signature —
carries over structurally: `Replica.groups(ParallelKind.EP)` uses the exact same
`Rank`/`ParallelGroup` machinery as `ParallelKind.TP`, so `group_shape()` works on
an expert group exactly as it does on a TP group.

The semantics do not carry over, and this is the task's own stated trap. Task 41's
canonical key was a **sorted multiset** — `{shape_A, shape_B}` — because two
replicas of the *same* pool are interchangeable. An attention TP group and an
expert-parallel group are not interchangeable: they shard different things and are
never swapped for each other. `enumerate_joint_arrangements`'s key is an **ordered
pair**, `(attn_shape, ep_shape)`, never sorted together. `attn_shape` and `ep_shape`
are not even drawn from comparable sets — on Task 32's own fabric at `attn_tp=4,
ffn_ep=2`, the attention axis reaches 5 distinct shapes and the expert axis reaches
2 — so treating the pair as a multiset would be a category error, not just a missed
optimization (`test_enumerate_joint_arrangements_keys_are_ordered_pairs_not_multisets`
checks exactly this).

At `ffn_ep=1` there is no expert group (`Replica.groups()` returns `[]`, the same
convention `enumerate_attn_shapes` already uses at `attn_tp=1`), so `ep_shape` is
always `(1,)` and the reachable `attn_shape` set is identical to
`enumerate_attn_shapes`'s own — verified directly (smoke test and
`test_enumerate_joint_arrangements_at_ep1_matches_enumerate_attn_shapes`) and
end-to-end through `plan()` (`test_plan_default_ep_values_gives_only_the_trivial_ep_shape`,
`test_plan_restricted_to_single_expert_group_matches_the_unextended_search`,
`test_plan_adding_ep_values_does_not_perturb_the_single_group_candidates`). This is
what keeps every pre-Task-44 call site — Task 33's 16-row table, Task 36's
two-fabric result — bit-identical (§4 below).

### §2(a): does the canonical signature collapse expert arrangements?

Yes, and substantially. On `domain8` (Task 33's own §5 fabric) at `attn_tp=1`:

| `ffn_ep` | raw candidates generated | collapsed `(attn_shape, ep_shape)` pairs |
|---|---|---|
| 1 | 2 (`packed`, `spread`) | 1 |
| 2 | 63 (2 + explicit fallback + 60 fragmented) | 2 |
| 4 | 63 | 5 |

On Task 32's own 5-domain fabric (`_topology_task32repro`) at `attn_tp=2, ffn_ep=4`,
the same 63 raw candidates collapse to 9 pairs. These ratios (63→9, 63→5, 63→2) are
comparable in kind to Task 32's own 188→16 and Task 41's replica-side collapse:
exhaustive enumeration stays affordable. (Task 44's own acceptance study needed
only 8 real Frontier evaluations in total across `ep ∈ {1, 2, 4}` — 1 + 2 + 5 — and
ran in 2m30s wall-clock; see §5.)

### §2(b): is expert placement independent of attention placement?

No — not fully, at least not in what the current raw-candidate generator reaches.
On Task 32's 5-domain fabric at `attn_tp=2, ffn_ep=4`, the reachable `ep_shape` set
differs by which `attn_shape` it is paired with:

```
attn_shape=(2,):   ep_shape ∈ {(4,), (3,1), (2,2), (2,1,1), (1,1,1,1)}   — 5 shapes
attn_shape=(1,1):  ep_shape ∈ {      (3,1), (2,2), (2,1,1), (1,1,1,1)}   — 4 shapes
```

`(1,1)` paired with the fully-packed `(4,)` never appears — confirmed stable at
`n_fragmented_seeds=500` (25x this task's own default), so this is not sampling
noise from the default 60 seeds. It is, however, a gap in this task's own explicit
"packed-if-it-fits" fallback candidate rather than a proven fabric impossibility:
that fallback only ever tries keeping the attention group *whole* in one domain
while concentrating the expert group in another; it never tries "attention
deliberately split, expert group still concentrated elsewhere," and neither
`packed()` nor `spread()`'s own generic policies happen to produce that specific
combination either. Practically this doesn't change what `plan()` ever selects,
since `(1,1)` is a worse attention placement than `(2,)` in every case this project
has measured (Task 32/33's own finding that packed beats split for TP) and would
never be a winning candidate on its own merits — but the honest answer to §2(b) is
that the two groups' placements are **not demonstrably independent under this
task's own raw-candidate generator**, and the search answers this by construction
(building one joint deployment) rather than by assuming a product space, exactly as
the spec asked.

## 3. The study: EP degree, before and after placement is searched

Same fabric, model, and workload as Task 33's own §5 (`domain8`, `attn_tp=1`,
`memory_margin_fraction=0.2`, Phi-tiny-MoE-instruct, 32 requests at qps=20,
deterministic/offline evaluation — the same regime every prior task's cited numbers
use). `plan(topology, model, workload, hardware, objectives, attn_tp_values=(1,),
ep_values=(1,2,4))`:

```
ep=4 ep_shape=(4,)          mean_tpot_ms=10.2228  throughput=121.461
ep=2 ep_shape=(2,)          mean_tpot_ms=11.3736  throughput=109.936
ep=1 ep_shape=(1,)          mean_tpot_ms=12.3316  throughput=101.888
ep=2 ep_shape=(1,1)         mean_tpot_ms=15.8062  throughput=80.511
ep=4 ep_shape=(2,2)         mean_tpot_ms=16.7170  throughput=76.314
ep=4 ep_shape=(1,1,1,1)     mean_tpot_ms=16.7202  throughput=76.300
ep=4 ep_shape=(2,1,1)       mean_tpot_ms=16.7202  throughput=76.300
ep=4 ep_shape=(3,1)         mean_tpot_ms=16.7202  throughput=76.300
WINNER: ep=4 ep_shape=(4,)  mean_tpot_ms=10.2228
```

Eight real Frontier evaluations, zero rejections, zero unknowns, 2m30s wall-clock.
The best arrangement found for every degree is the fully-packed one (all EP ranks
in one domain); every split arrangement is worse, by 39–63%. **Once each degree
uses its own best placement, EP degree is monotonic**: 12.33 > 11.37 > 10.22ms —
increasing EP degree, correctly placed, is a straight win in this offline regime.
This is the answer §5 asks for first, and it holds.

### The complication: Task 33's own `ep=4` citation does not reproduce

Task 33's report cites `ep=4` (first-fit) at **14.4434ms**, with an explicit
mechanism claim: "three of the four FFN EP ranks are not covered by the reference
and land split across two domains." Verifying this directly (per this project's own
standing rule — check citations against source and measurement, don't trust them)
rather than assuming it: `tools/planner.py`'s pre-Task-44 `_placement_for` was
reconstructed byte-for-byte and run today with identical parameters
(`domain8`, `attn_tp=1`, `ep=4`, `margin=0.2`, same model/workload). Result:
**`ep_shape=(4,)`, `mean_tpot_ms=10.222803755821966`** — all four EP ranks land
fully packed in domain 0, not split, and the number matches this task's own new
joint-search result exactly, not Task 33's citation. The same reconstruction at
`ep=2` gives `ep_shape=(2,)` and `mean_tpot_ms=11.373614792393711`, which **does**
match Task 33's own citation exactly.

Four candidate explanations were checked directly, not assumed:

| checked | result |
|---|---|
| `src/engine/logical/deployment.py` changed since Task 33? | No — `git log` shows only "Initial commit" for this file, ever. |
| `_topology_domain8()` changed since Task 33's own commit (`dc3ee82`)? | No — `git show dc3ee82:tools/planner.py` gives the byte-identical 5-line body. |
| `enumerate_attn_shapes` changed since `dc3ee82` (Task 37's split)? | No — the function body is unchanged; only docstring/comments were added when it moved into `planner_core.py`. |
| `feasible_num_blocks` changed since `dc3ee82` (Task 36's generalization from a lookup table to Frontier's real formula)? | Changed in *implementation*, but produces the identical value at `attn_tp=1`: `64256` blocks both ways, checked by direct computation of the old lookup table's own numbers and a call to today's function. |
| The old `_placement_for` algorithm itself, byte-for-byte, at `dc3ee82`? | Identical to what was reconstructed and run. |

Every mechanism this project's own history could plausibly point to — fabric,
deployment/rank code, the shape enumerator, the memory-feasibility formula, and the
placement algorithm's own text — is confirmed unchanged or numerically identical
between Task 33's original commit and today. No candidate explanation for the
`ep=4` discrepancy survived direct checking. Nor does any placement this task's own
search finds for `ep=4` — packed at 10.2228ms, or any of the four ~16.72ms split
shapes — land anywhere near 14.4434ms. **This citation does not reproduce, and this
report cannot identify why**: the most likely account, given everything else checks
out byte-for-byte, is a transcription or one-off run error in Task 33's own report
for that specific row, not a code change. This is reported plainly rather than
smoothed over, per this project's standing rule on citation gaps — this is the 8th
such figure found not to match its own cited report (Task 33's own §7 already
flagged two citation issues; this is a new, ninth-task instance of the same
pattern, on Task 33's own numbers this time rather than an earlier task's).

One consequence of this: the framing of "the search fixes a placement mistake" is
not correct for *this specific config*, because today's first-fit **already**
happens to find the fully-packed (optimal) arrangement for both `ep=2` and `ep=4`
on `domain8` at `attn_tp=1` — matching what the new joint search also finds as
best. The new search's demonstrated value here is not correcting today's baseline;
it's *confirming* today's baseline is already optimal, by also finding and pricing
the worse alternatives (the `(1,1)`, `(2,2)`, `(1,1,1,1)`, `(2,1,1)`, `(3,1)` splits)
and showing none of them beats packing. Task 33's own attribution — that
non-monotonicity in its 2024-vintage `ep=4` run was a placement-confounding
artifact — remains a *reasonable* explanation for a number that, itself, no longer
reproduces; it cannot be confirmed as the correct explanation for something this
report cannot reconstruct.

### Monotonicity is regime-specific: it does not hold under streaming arrivals

The above is all in the offline/deterministic regime `plan()`'s own search
actually optimizes over — the same regime every number in this report and in every
prior task's citations uses. A seeded check (`--seeded 1`,
`--offline_use_generated_request_arrivals`, N=6 seeds per point) of the three
*packed* candidates, run to ask whether the offline monotonicity claim survives
real streaming arrival timing rather than an offline batch arrival, gives:

| candidate | offline (deterministic) mean_tpot_ms | streaming (seeded, N=6) mean_tpot_ms | 95% CI half-width |
|---|---|---|---|
| ep=1 | 12.3316 | 3.4959 | ±0.1469 |
| ep=2 packed | 11.3736 | 3.6789 | ±0.1783 |
| ep=4 packed | 10.2228 | 3.7654 | ±0.0950 |

Under streaming arrivals the absolute numbers are far lower (queueing pressure
from an offline batch-arrival workload is not representative of a paced qps=20
stream — expected, and consistent with why the two regimes give different
absolute latencies), but more importantly **the ranking reverses**: `ep=1` is
fastest, not slowest. The `ep=1` vs `ep=4`-packed gap (3.4959 vs 3.7654, both 95%
CIs disjoint) clears the noise floor; the middle step (`ep=2`) is not statistically
distinguishable from either neighbor at N=6. This is the same pattern Task 36's own
report already surfaced once (a workload-length change flipped its own winner from
`tp=2` to `tp=4`, a reversal that didn't clear the noise floor there) — this
project has now seen this shape of result twice. **"EP degree is monotonic once
placed" is a true statement about the offline evaluation regime this search
targets; it is not a claim about deployed behavior under realistic arrival
patterns**, and should not be read as one.

## 4. Acceptance

- `python3 -m pytest -q` → **233 passed** (226 pre-Task-44 + 7 new).
- `python3 tools/check_import_direction.py` → exit 0, "checked 25 files under
  src/engine/, OK."
- Task 33's 16-row table reproduces bit-identical (winner `tp=2`, `shape=(2,)`,
  `11.6803ms`, full table match) via `plan()` unchanged.
- Task 36's two-fabric result reproduces bit-identical: `domain8_40gpu` winner
  `(8,)` at `326.2362ms`; `domain4_40gpu` winner `(4,3,1)` at `446.5146ms`.
- `test_plan_restricted_to_single_expert_group_matches_the_unextended_search`:
  `plan(..., ep_values=(1,))` gives identical `ranked`/`winner` keys and
  `mean_tpot_ms` values to `plan()` with `ep_values` left at its default — the
  single-expert-group case does not move.

## 5. What shipped

- `tools/planner_core.py`: `enumerate_joint_arrangements`; `Candidate.ep_shape`
  replacing the dead `ep_split`; `plan()`'s main loop restructured to enumerate
  `(attn_shape, ep_shape)` jointly per `ep` value.
- `tools/planner.py`: `_placement_for` rebuilt on the joint reference placement;
  `_run_scenario`/`evaluate`/CLI plumbing carry `ep_shape` instead of `ep_split`.
- `tests/test_planner_core.py`: 7 new tests — ep=1 degeneracy, the fully-packed
  pair being reachable when it fits, ordered-pair (not multiset) keys, raw-candidate
  collapse, default-search triviality, the required single-group-search-equivalence
  acceptance test, and non-perturbation of `ffn_ep=1` candidates when `ep_values`
  is widened.
- `docs/tasks/44-ep-placement-report.md` (this file).

## 6. Where the spec was right, and where a citation didn't hold up

The spec's own framing of the mechanism (Task 33's §6.5 attribution) is a
*reasonable* hypothesis for the non-monotonicity it observed, and the scope/known-traps
guidance (ordered pair not multiset; the single-group case must not move; attention-replica
placement stays out of scope, per Task 41's own "a collective call carries a device
count, never a rank identity") all held up exactly as stated, with no correction
needed. The one figure that does not hold up is Task 33's own `ep=4` citation
(14.4434ms, ranks split) — reconstructed byte-for-byte and re-run today, it gives
10.2228ms, fully packed, matching this task's own new search rather than the old
citation, and four candidate causes for the mismatch (fabric, deployment code,
shape enumerator, memory-feasibility formula) were each checked directly and ruled
out. This is reported as an open discrepancy rather than resolved, since no
candidate explanation survived checking; per the project's own running count, this
is the 8th figure found not to match its own cited report.
