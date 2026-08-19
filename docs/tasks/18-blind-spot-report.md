# Task 18 — How much of a deployment do we actually see?

Branch: `task-18-blind-spot`, stacked on `task-17-boundaries`.

177 tests pass (measurement task, no new tests, matching task 17's own
acceptance criteria), and `python3 tools/check_import_direction.py` exits
0.

---

## 0. What the ledger actually is, and two flags that silently starve it

Frontier's per-batch-stage "component ledger"
(`_build_frontier_stage_batch_component_ledger` in
`frontier/metrics/metrics_store.py`) decomposes every DECODE_ATTN/DECODE_FFN
batch stage into named components in milliseconds, and *enforces* — a
`raise ValueError` if it's ever false — that they sum exactly to
`total_time_ms`. That guarantee is why this task's ratios are trustworthy
rather than estimated: nothing is left over to misattribute.

Getting the rows at all took finding two undocumented couplings, confirmed
by running it rather than assumed:

- The ledger's in-memory capture (not just disk writing) is gated behind
  `metrics_config.write_metrics` itself. Every other tool in this project
  passes `--no-metrics_config_write_metrics` (it never needed the ledger);
  this one can't.
- Pending rows only get *completed* into the readable list if
  `store_utilization_metrics` also stays on — an early return in the
  replica-stage-end hook covers both concerns at once, so disabling
  utilization metrics (as every prior tool in this project also does)
  silently leaves every row permanently pending (confirmed directly: 216
  pending, 0 completed, until this flag went back on).

`tensor_parallel_communication_time`, the spec's own name for one of the
four fields to extract, **does not exist as a ledger key** — confirmed
by reading `frontier/entities/execution_time.py`: it is a constructor
parameter for a legacy, unsplit accounting path used only when the split
fields aren't supplied. The modern, split equivalents this script actually
reads are `attention_all_reduce_time` + `mlp_all_reduce_time` +
`moe_tensor_parallel_allgather_time` + `share_expert_tensor_parallel_allreduce_time`
— confirmed as communication (`trace_kind=TraceKind.COMM`,
`resource_class=ResourceClass.COMM`) via `COMM_FAMILY` in
`frontier/operators/families.py`, not assumed from the name.

`moe_shuffling_time` is classified **compute**, per Frontier's own operator
metadata (`MOE_FAMILY`'s `moe_shuffling` entry:
`trace_kind=TraceKind.COMPUTE`, `role=OperatorRole.RESHAPE`,
`resource_class=ResourceClass.MEMORY`) — a local, single-device token
reorder/permute kernel, not a cross-device transfer. It genuinely has no
multi-GPU "participants" to place in one domain or several — worth
stating plainly since it's the one invisible-adjacent component that
structurally *cannot* cross a domain boundary, not merely one this
project's placements happened to keep together. One further, honestly
reported finding: the *metadata* also tags this operator `ep_agnostic=True`,
which I initially read as "doesn't scale with EP" — the measured totals
below disprove that reading directly (1.07 → 3.75 → 7.50 ms as EP goes
1 → 2 → 4). `ep_agnostic` evidently describes the *prediction model*
(EP isn't an input feature to its regression), not the *aggregate* total
across a batch's routing work, which does grow with how many
expert-parallel ranks' worth of local shuffling gets summed into one
ledger row. Corrected here rather than left as an assumption.

**Units, stated once**: every figure below is a *sum over the run's
decode-phase activity* (8 requests × 8 decode tokens, 224 DECODE_ATTN
ledger rows) — a total, not a per-step or per-token average — for both the
ledger side and `request.total_m2n_transfer_time` (itself a per-request
total across the whole decode phase, task 17's own units trap). Since
numerator and denominator are summed over the *same* window, their ratio
is the average per-decode-step share regardless of the absolute total;
the task itself says the point is the ratio, and that is what's reported.

---

## 1. The headline ratio

*Of all communication in a decode step, what fraction is priced with
topology* (visible M2N ÷ (visible M2N + invisible TP/PP/EP), `moe_shuffling`
excluded per its compute classification):

| Configuration | Visible (M2N) | Invisible (TP+PP+EP) | Headline |
|---|---|---|---|
| Simplest (EP=1, TP=1, colocated) | 3.946 ms | 0.000 ms | **100.00%** |
| Simplest, split placement | 54.875 ms | 0.000 ms | **100.00%** |
| Most expert-parallel measured (EP=4, TP=1, colocated) | 3.946 ms | 1.190 ms | **76.84%** |
| Most expert-parallel measured, split placement | 54.875 ms | 1.190 ms | **97.88%** |

At the simplest configuration there is nothing invisible to miss (TP=1,
EP=1 — no within-replica parallelism at all, so the ratio is trivially
100%). At EP=4 — the most expert-parallel configuration this task's own
§3 table asks for — **23.16% of all communication in a decode step is
invisible to this project**, colocated. That is the number this task
exists to produce.

## 2. How it moves with expert groups and tensor-parallel degree

**Expert groups** (attn_tp=1, ffn_tp=1, colocated placement):

| EP | denom (ms) | visible (%) | invisible (%) | expert_parallel_communication_time (ms) | headline |
|---|---|---|---|---|---|
| 1 | 37.931 | 10.40% | 0.00% | 0.0000 | 100.00% |
| 2 | 57.501 | 6.86% | 1.12% | 0.6437 | 85.97% |
| 4 | 86.358 | 4.57% | 1.38% | 1.1896 | 76.84% |

**Tensor-parallel degree** (attn_tp varies, ffn_tp=1, ep=1, colocated
placement):

| attn_tp | denom (ms) | visible (%) | invisible (%) | tp comm (ms) | headline |
|---|---|---|---|---|---|
| 1 | 37.931 | 10.40% | 0.00% | 0.0000 | 100.00% |
| 2 | 36.034 | 10.95% | 2.33% | 0.8395 | 82.46% |
| 4 | 35.849 | 11.01% | 2.89% | 1.0352 | 79.22% |

**The trend matters more than any single figure, as the spec says, and the
trend is monotonic in both directions this project could realistically
grow along**: invisible communication grows with both EP and TP degree,
and the headline ratio falls in both cases — from 100% down to 76.84%
(EP) and 79.22% (TP) at the highest degree measured here (4), which is
still small next to production MoE deployments (EP=8/16/32+ is common).
Nothing in either trend suggests it plateaus by EP=4 or TP=4 — if
anything, `expert_parallel_communication_time` very roughly doubles from
EP=2 to EP=4 (0.644 → 1.190 ms), consistent with more cross-rank dispatch
work as the expert group grows, not less.

## 3. Does the visible share rise when pools are split? Yes, every time

| Configuration | colocated | split |
|---|---|---|
| EP=1 | 10.40% | 61.75% |
| EP=2 | 6.86% | 50.61% |
| EP=4 | 4.57% | 39.97% |
| attn_tp=2 | 10.95% | 63.10% |
| attn_tp=4 | 11.01% | 63.24% |

Confirmed in every configuration tried, no exceptions — the visible share
of a decode step rises sharply when DECODE_ATTN and DECODE_FFN are split
across machines (this project's own fabric model correctly makes M2N
transfer time explode under that placement, as it should), while the
invisible (within-replica) components stay essentially fixed regardless
of where the *pools* sit (they depend on TP/EP degree, not on
attn-vs-ffn placement at all). The accounting behaves exactly as the
spec predicts it should; nothing here suggests an error.

## 4. Domain-crossing per invisible component — the question that matters more than size

Per §1.1: a fixed figure for a component whose real participants always
sit in one scale-up domain is harmless; the same fixed figure for a
component whose participants *could* be spread across domains is a bias
that grows with exactly the placement decision this simulator exists to
inform. Checked directly, not just reasoned about, for expert parallelism
(the one component this project can actually move):

**Controlled test**: EP=4, everything else held fixed (attn_tp=1, ffn_tp=1,
attn/ffn colocated for the M2N side) — only the *FFN replica's own EP
ranks'* placement changes, from all four colocated on one machine to two
machines apart (a genuinely different scale-up domain assignment for the
EP dispatch's own participants):

```
experts colocated: denom=86.3578ms  ep_ms=1.1896  (everything else identical)
experts split:      denom=86.3578ms  ep_ms=1.1896  (everything else identical)
```

**Bit-identical.** `expert_parallel_communication_time` cannot see the
difference at all — moving the experts across domains changes nothing
Frontier reports, because nothing Frontier's execution-time predictor
receives carries fabric placement information (task 17's own finding,
reconfirmed here with a direct A/B rather than inferred from the
predictor's call signature alone). **Expert parallelism is exactly the
§1.1 "small but biasing" case, empirically, not hypothetically**: at 23%
of communication (EP=4) it is not the dominant component, but its price
is wrong in a way that moves with the placement decision this project's
whole purpose is to inform, while every visible number (M2N transfer time,
the topology-aware one) looks perfectly healthy throughout.

For the other two invisible components, by contrast:

- **Tensor-parallel** (`attention_all_reduce_time`, `mlp_all_reduce_time`,
  `moe_tensor_parallel_allgather_time`, `share_expert_tensor_parallel_allreduce_time`):
  every placement in this study (and every realistic deployment) keeps TP
  ranks inside one scale-up domain — that is the entire reason an
  eight-GPU NVLink node exists. No split-TP scenario was run (there is no
  realistic one to run: splitting TP across domains is a
  misconfiguration, not a design point a real deployment would consider).
  A fixed figure here is, per §1.1, the *right* answer — large in
  principle at high TP degree, but not biased by the variable this
  project studies.
- **Pipeline-parallel** (`pipeline_parallel_communication_time`): **not
  exercised at all in this study** — every configuration used
  `num_pipeline_stages=1`, so this component was 0 throughout, and I have
  no direct measurement of it. Unlike TP, PP send/recv is *not*
  necessarily latency-sensitive enough to require staying in one domain —
  real large-model deployments do split pipeline stages across nodes.
  This makes PP a plausible second instance of the same bias EP has, not
  a component I can rule safe. Reported as a gap, not glossed over.
- **`moe_shuffling_time`**: no multi-GPU participants to place at all — a
  local, single-device reorder kernel (§0). Structurally cannot cross a
  domain boundary, not merely kept together by this study's choices.

**`moe_shuffling_time`'s effect on the headline, both ways** (per the
spec's own trap — decide, then check if the choice matters enough to
report both):

| EP | headline (shuffle = compute) | headline (shuffle = comm) |
|---|---|---|
| 1 | 100.00% | 78.61% |
| 2 | 85.97% | 47.32% |
| 4 | 76.84% | 31.22% |

**This swing is large enough to flip the qualitative conclusion** (a
headline in the high 70s-to-100% range reads as "the collective path is a
footnote"; one in the 31-79% range reads as "the simulator sees less than
half the communication in a decode step" at EP=4). The classification
decision genuinely matters here, which is exactly why §0 grounds it in
Frontier's own operator metadata rather than a guess: `moe_shuffling` has
no cross-device participants, so counting it as *communication* would be
wrong regardless of its size, not just a matter of taste. The 76.84%
figure (shuffle excluded) is the one to trust; the 31.22% figure is
reported because the spec asks for it and because it demonstrates how
sensitive "how bad is the blind spot" is to a classification call someone
else might make differently.

## 5. Recommendation

**Pursuing the collective path upstream is worth doing, and expert
parallelism specifically is the reason — not tensor-parallel, not
pipeline-parallel (unmeasured), and not because the invisible share is
large in absolute terms (it isn't yet, at EP=4).** The argument is the
consistency one the spec itself anticipates, and this project's own recent
history makes it concrete rather than hypothetical: **spreading FFN
replicas' experts across scale-up domains is the exact same deployment
decision this project already prices correctly on one side of an internal
Frontier boundary — activation exchange between DECODE_ATTN and
DECODE_FFN — and prices with a placement-blind constant on the other side
of that same boundary — expert dispatch among the FFN replica's own EP
ranks.** Task 15/16 built and measured real value in getting the first
side right (locality doubled, a real and reproducible effect). Task 18's
controlled A/B (§4) shows the second side cannot register that same
locality decision at all, even in principle, with the mechanism Frontier
currently exposes.

The mechanism to fix it consistently already exists in this codebase
three times over, which is what makes this a consistency argument about
Frontier's own design rather than a feature request for new capability:

1. `EngineCCBackend` (task 06) — a real, tested, fabric-aware collective
   backend, built and passing all-reduce agreement tests against
   Frontier's own analytical model, unreachable only because
   `CCBackendType` has no free enum slot.
2. `engine.placement.binding`'s distance+load logic (task 14/15) —
   already generalized once, from KV/M2N destination selection to a real
   cluster scheduler's replica choice; reusable a third time for "which
   EP ranks are actually far apart" if a topology-aware EP dispatch cost
   were ever wired up.
3. `Transfer`/`isolated_durations` (task 09 onward) — the one fabric-cost
   primitive every other topology-aware measurement in this project
   already goes through.

None of this is buildable unilaterally from `src/integration/` — task 06
already found `CCBackendType` closed and `BaseRegistry.register()`
idempotent-first-wins, so reaching this requires an upstream change (a
free enum member, or a different public extension point), the same
conclusion task 06 and task 17 both already reached for the identical
reason. What task 18 adds is that the case for making that upstream
change is no longer speculative: expert parallelism's invisible share
grows with exactly the axis a real MoE deployment would push furthest
(production EP degrees well past 4), and its blindness to domain
placement is now a measured fact, not a structural inference.

## 6. Anywhere this specification is wrong

- **`tensor_parallel_communication_time` is not a ledger key** (§0) — the
  spec names it as if it were one of four directly-readable fields; the
  modern accounting splits it into (at least) four *different* named
  fields, none sharing that name. Read from the split fields instead, with
  the mapping shown.
- **`moe_shuffling_time`'s `ep_agnostic` tag does not mean its total is
  EP-invariant** (§0) — an assumption worth flagging precisely because it
  looked, from the operator metadata alone, like it would settle the
  "does it vary with EP" question the same way the classification
  question was settled. It didn't; the measured totals correct it.
- Otherwise the specification's structure — separate the "how big" and
  "does it move with placement" questions (§1.1), sweep EP and TP with
  placement crossed (§3), decide `moe_shuffling`'s classification and
  check whether it matters (§6's own trap) — matched exactly what the
  investigation needed, including correctly anticipating that the
  interesting finding might be a small-but-biasing one rather than a
  large-but-harmless one. It was.
