# Task 15 — A topology-aware cluster scheduler

Branch: `task-15-topology-scheduler`, stacked on `task-14-binding`.

All 172 tests pass (165 existing + 7 new in `tests/test_topology_scheduler.py`),
and `python3 tools/check_import_direction.py` exits 0.

---

## 1. Is `ClusterSchedulerType` closed? Yes — and more thoroughly than `CCBackendType` was

`frontier/types/cluster_scheduler_type.py` is a 5-member `BaseIntEnum`
(`ROUND_ROBIN, RANDOM, LOR, STICKY_ROUND_ROBIN, STICKY_LOR`), and
`cluster_scheduler_registry.py` registers all five, at module level, to
concrete classes. There is no unused member the way `KVCacheTransferType`
had `EMPIRICAL` (task 07) — every slot is claimed.

Three separate, independently-confirmed reasons this is airtight, not just
"probably closed":

1. **The registry no-ops on collision.** `BaseRegistry.register()`
   (`frontier/utils/base_registry.py`): `if key in cls._registry: return` —
   silently discards a second registration for an already-used key, no
   exception. Re-registering `ClusterSchedulerType.ROUND_ROBIN` to point at
   a new class (task 06's "arbitrary string key" trick, reused here for
   `TopologyAwareClusterSchedulerConfig`) is therefore not available even as
   a workaround for an *existing* member.
2. **CLI validation is enum-only, not name-only.** Frontier's polymorphic-
   config discovery (`frontier/config/flat_dataclass.py::reconstruct_original_dataclass`)
   validates `--cluster_scheduler_config_type <value>` by comparing
   `<value>` against `str(subclass.get_type())` for every discovered
   `BaseClusterSchedulerConfig` subclass — **not** against a separate
   `get_name()` the way I initially assumed from tasks 07/08's `EMPIRICAL`
   pattern. I tested this directly: defining a probe config subclass with
   `get_name() -> "topology_aware"` but `get_type() -> ClusterSchedulerType.ROUND_ROBIN`
   did not add `"topology_aware"` to the CLI's accepted values — it only
   added a second `"round_robin"` entry to the list. There is no
   independent name-space to claim; the enum member's own lowercase name
   *is* the only selectable string, for every polymorphic field of this
   kind.
3. **Reproduced with a real run.** `--cluster_scheduler_config_type topology_aware`
   raises, verbatim:
   ```
   AssertionError: Invalid type topology_aware for cluster_scheduler_config_type.
   Valid types: ['random', 'round_robin', 'lor', 'sticky_round_robin', 'sticky_lor']
   ```
   `tests/test_topology_scheduler.py::test_scheduler_is_selectable_by_flag`
   locks this in as a regression test, not an assumption.

Even the dispatch layer confirms it independently: `base_global_scheduler.py`
constructs a cluster's scheduler via
`ClusterSchedulerRegistry.get(cluster._config.cluster_scheduler_config.get_type(), ...)`
— keyed purely by the enum value the *config* class reports, decoupled from
which config subclass actually supplied it. So even in the counterfactual
where CLI selection of a new config class somehow worked, the object that
gets constructed is governed entirely by `get_type()`'s enum member, and
with no free member and no way to steal a claimed one, the constructed
scheduler is — deterministically, always — one of Frontier's own five.

**This module is implemented to the ceiling this leaves**, exactly the
pattern task 06 set for `CCBackendType`: a genuine, directly-constructible
`BaseClusterScheduler` subclass (`TopologyAwareClusterScheduler`), a real
config class documenting why it can't select the class it names
(`TopologyAwareClusterSchedulerConfig`), a mechanical (non-CLI) registry
round-trip under a fabricated string key, and a full test suite — none of
it reachable by `--cluster_scheduler_config_type`, and the report says so
rather than pretending otherwise.

## 2. What `BaseClusterScheduler` does with its transfer-predictor references

`BaseClusterScheduler.__init__` stores both
`self._kv_cache_transfer_predictor` and `self._m2n_transfer_predictor`
(`base_cluster_scheduler.py:276-277`). Grepping the whole
`cluster_scheduler/` package for reads of either attribute (not just the
assignment):

- `_kv_cache_transfer_predictor` is **never read again anywhere** in the
  cluster-scheduler layer or any of its five subclasses. Dead weight at
  this layer — the KV predictor is called from elsewhere entirely
  (`cluster_batch_end_event.py`'s `kv_pred.get_transfer_info_for_request`,
  confirmed in task 14).
- `_m2n_transfer_predictor` **is** read, but only to *price* an M2N
  transfer once the destination is already fixed
  (`self._m2n_transfer_predictor.get_transfer_info(...)` at four call
  sites, all inside batch/scheduling bookkeeping) — never to *choose*
  among candidate destinations. No scheduler implementation in this
  checkout calls either predictor as part of a selection decision.

So the spec's suggestion — "a topology-aware scheduler can ask our
predictor directly rather than reimplementing distance" — is not available
even in principle: the predictor reference a scheduler holds has no
selection-relevant API (`get_transfer_time`/`get_transfer_info` require a
destination already chosen), and nothing in Frontier ever calls it that
way. `TopologyAwareClusterScheduler` computes distance itself, by importing
`engine.placement.binding.distance_key` (task 14's own machinery, extracted
from `_nearest` for exactly this reuse) — which is what "this is task 14's
`nearest` logic, reused rather than rewritten" already pointed at.

## 3. The scheduler and its rule

`TopologyAwareClusterScheduler` subclasses `RoundRobinClusterScheduler`
(not `BaseClusterScheduler` directly) — every batch-mode, AFD-pipeline, and
barrier code path this task doesn't touch is inherited unchanged, matching
S3.1's "not a placement optimiser" warning: reimplementing all of that
would dwarf the actual scope. Two places a replica gets chosen are
overridden:

- `_schedule_decode_lane_round_robin` — the unified DECODE cluster's
  per-request, dynamic destination decision (pd-disaggregation's KV path).
- `__init__`'s post-construction lane assignment for DECODE_FFN — see
  below for why this one needed a two-phase design.

**The combined rule (`select_replica`)**: prefer the candidate nearest the
source (`distance_key`: same scale-up domain first, then fewest hops, then
ascending `replica_id`). Overridden only when the nearest candidate's
outstanding load exceeds the least-loaded candidate's by more than
`LOAD_MARGIN = 2` — an **absolute difference, not a ratio**. A ratio is
noise-sensitive near zero (1 pending request vs. 0 is a "100% higher load"
that means nothing with these request counts); an absolute margin of 2
tolerates ordinary arrival-timing noise while still catching a replica
that has genuinely accumulated more outstanding work than every
alternative. For the DECODE cluster, "load" is `num_pending_requests`
summed across a replica's dp lanes — the *exact* signal
`LORClusterScheduler._schedule_lor` already uses, not a new one invented
for this task, and a real, current signal (unlike task 14's
`BindingState.assignment_count`, a cumulative count with no way to expire,
since `bind()` had no completion callback — this scheduler has direct
access to the real replica-scheduler objects, so it doesn't have that
limitation).

**What the rule gets wrong.** For the DECODE_FFN static lane map — decided
once, in `__init__`, before any request has arrived — there is no live
`num_pending_requests` yet, so "load" there can only mean "how many lanes
this replica has already been given in this same construction pass," a
weaker, order-dependent proxy. Worse: a **single-phase** nearest+margin
rule can violate Frontier's own invariant that every DECODE_FFN replica
must receive at least one lane (`__init__`'s own assertion) — on an
asymmetric fabric (one near replica, several symmetric far ones), every
lane's nearest candidate is the *same* replica, and the margin-override
only ever redirects to *one* other (the least-loaded, tie-broken to the
lowest id among the untouched ones), never reaching a third or fourth
replica at all. I hand-traced this before writing the fix, not after
finding it as a crash: with 4 lanes and 4 replicas (one near, three
symmetric far), a single-phase rule oscillates between the near replica
and replica B only, leaving replicas C and D with zero lanes —
`TopologyAwareClusterScheduler` handles this with an explicit two-phase
construction (assign one lane to each still-uncovered replica first, by
nearest-among-remaining, before ever running the load-margin rule on the
surplus) precisely to avoid it, and this is what the study script
exercises for real (S4).

The load-margin threshold itself has a real, measured cost, not just a
theoretical one — S4's study measured it directly: with `LOAD_MARGIN=2`
and only a modest lane surplus, the override never triggers at all, and
the near replica ends up with roughly half of all lanes rather than a
fair share. This is not a bug in the rule so much as evidence that 2 is
generous for small-N regimes; a smaller margin (or a ratio term for larger
N) would balance sooner. Reported here rather than silently retuned to
produce a nicer-looking distribution.

## 4. The study — real numbers

`tools/run_topology_scheduler_study.py`: one PREFILL replica, one
DECODE_ATTN replica with `attn_data_parallel_size=8` (eight dp lanes, all
colocated on one machine — the source side is deliberately kept
unambiguous, since binding is scoped to destinations, task 14), four
DECODE_FFN replicas (replica 0 shares the ATTN lanes' domain; replicas 1-3
are each alone on a separate machine). Model: `Phi-tiny-MoE-instruct` (16
experts) — a dense model's `attn_data_parallel_size` in `decode_attn` is
restricted to 1 by Frontier's own config validation, so multiple dp lanes
require a MoE model; this is Frontier's own example model for exactly this
combination (`examples/architecture/pd-af-disagg/offline/moe_model_basic.sh`).
16 requests, `decode_tokens=16`, dummy compute mode.

Two variants, same real `Simulator`, same real
`RoundRobinClusterScheduler` instance for every cluster:

- **round_robin**: entirely unmodified. DECODE_FFN's `__init__` assigns
  `lane_ordinal % 4`, distance-blind.
- **topology_aware**: the *same already-constructed* scheduler object, with
  `_ffn_lane_to_target_replica` recomputed in place by calling
  `TopologyAwareClusterScheduler._assign_ffn_lanes_by_topology` as an
  unbound method against it, after `Simulator(config)` but before
  `.run()`. Swapping in a freshly-constructed scheduler object instead was
  considered and rejected: every per-dp-lane `ReplicaScheduler` Frontier
  already built holds a `cluster_scheduler=self` back-reference to the
  *original* object, and a second, parallel object graph would leave those
  back-references stale. Mutating the lane-map attributes in place changes
  nothing else already wired up.

Measured:

| | round_robin | topology_aware |
|---|---|---|
| distribution across FFN replicas (id: lane count) | {0:2, 1:2, 2:2, 3:2} | {0:4, 1:2, 2:1, 3:1} |
| within-domain fraction | 2/8 (25.0%) | 4/8 (50.0%) |
| mean M2N transfer time | 40.524288 ms | 40.524288 ms |
| mean inter-token latency (tpot) | 425.126137 ms | 425.131137 ms |

Both numbers are already single per-request quantities in Frontier's own
units — no total-vs-per-token rescaling needed here (task 12's own
correction doesn't apply: neither figure is a decode-phase total being
compared against a per-token average).

## 5. Does inter-token latency actually fall? No — and here is why, honestly

The within-domain fraction doubled (25% → 50%), exactly the mechanism the
scheduler is supposed to produce. **Mean tpot moved by +0.005 ms (+0.00%)
— noise, not an effect.** Mean M2N transfer time is identical to six
significant figures between the two variants.

This is not a disappointing measurement bug; it is a direct, structural
consequence of S3.2's finding, which is worth stating plainly: this
project's own empirical M2N predictor **cannot be used in this scenario at
all**. M2N is a round trip — DECODE_ATTN sends to DECODE_FFN, and
DECODE_FFN sends back (`cluster_batch_end_event.py` has two
`get_transfer_info` call sites, one per direction). `price_transfer`
(`integration/binding_support.py`) resolves its *source* pool
unconditionally, with no `try`/`except` the way the destination side has —
task 14 scoped binding to "which replica *receives*," deliberately leaving
source-side ambiguity alone. On the return leg, DECODE_FFN — now with four
replicas — *is* the source, and the run raises `CommGroupError`
immediately. Confirmed by running it, not assumed: tasks 09-14 never hit
this because every prior scenario kept every pool that ever acts as a
*source* at exactly one replica. This is the first task to put several
replicas on a pool that both sends and receives.

The study therefore runs on Frontier's own stock **analytical** M2N
predictor, which is distance-blind by design — its transfer-time formula
does not depend on which replica was chosen, at all. Given that, `mean_m2n_time`
being identical between variants is not a null result to explain away; it
is exactly what this pricing model guarantees regardless of scheduler. The
only channel left for tpot to move is queueing/distribution — and here,
concentrating lanes onto the near replica (S3's finding that
`LOAD_MARGIN=2` doesn't rebalance within this lane count) works *against*
even that: four lanes on one replica versus two evenly spread is worse
load balance, not better, for a benefit (locality) that this predictor
cannot price into existence. **The honest finding is that this specific
combination — a distance-blind transfer predictor plus a scheduler tuned
to prefer locality over balance — has no verified path to a faster token,
and the study confirms it does not produce one.** A predictor that priced
distance (this project's own empirical one) might change that, but is
exactly the one this scenario cannot use, for the source-ambiguity reason
above — a real, load-bearing limitation, not a choice made for convenience.

## 6. Task 15 S3.2 — reachability of "price against whatever the scheduler chose"

Not reachable, for a reason distinct from and deeper than S3's original
"nothing connects `bind()` to Frontier's real choice" (task 14): here, the
predictor cannot even be used in the multi-FFN-replica scenario at all
(S5), so there is no pricing call left to make agree with anything. Two
narrower findings surfaced while investigating this, both worth recording:

- `price_transfer` resolves a *source* pool's representative rank
  (`src_ranks[0]`), not the specific dp lane a given M2N call is actually
  for. Harmless in this study only because all eight ATTN dp lanes are
  colocated — every lane really is equidistant from every FFN replica from
  the source side — but not a general solution.
- `BindingPolicy.EXPLICIT`, the one existing policy that could in
  principle pin a specific destination per call, keys
  `state.explicit_map` by `tuple(source_ranks)` — one entry per *pool*,
  not per lane. All eight of DECODE_ATTN's dp lanes resolve to the same
  source-rank tuple, so `EXPLICIT` cannot represent "a different
  destination per lane" without a change to task 14's own tested binding
  module, which is out of this task's scope.

## 7. Anywhere this specification was wrong

- **"Compare the new scheduler against `round_robin` and `LOR`"**:
  `LORClusterScheduler.schedule()` unconditionally raises
  `DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR` — *"Disaggregated architecture
  support is currently being optimized and is not included in this
  release... Please use the co-located architecture"* — for any cluster
  type other than `MONOLITHIC`, before it ever reaches its own
  (otherwise fully implemented) LOR logic. Reproduced directly. LOR is not
  a usable baseline for a disaggregated M2N-path study at all; `round_robin`
  is the only one of Frontier's five schedulers that is.
- **"A topology-aware scheduler can ask our predictor directly rather than
  reimplementing distance"**: not available (S2) — the predictor
  references a scheduler holds have no selection-relevant call, and
  nothing in this checkout calls either predictor that way. Distance had
  to be computed independently, which is exactly why `distance_key` was
  worth extracting from task 14's `binding.py` rather than re-derived.
- **"Add a policy or mode meaning 'price against whatever the scheduler
  actually chose'"**: on inspection, this is unreachable for a reason well
  beyond what the spec anticipated (S6) — it isn't just that no channel
  exists to ask the scheduler, it's that this project's own M2N predictor
  cannot run in a multi-FFN-replica scenario at all, for a source-side
  ambiguity task 14 explicitly scoped out. The spec's framing treats this
  as a design choice about pricing; it is actually a hard blocker
  discovered only by trying to run the study.
- **Implicit assumption that a scheduler favoring locality straightforwardly
  improves a serving metric**: S4/S5 show this is not automatic — with a
  distance-blind transfer predictor (the only one usable here) and a load
  margin that doesn't rebalance at this lane count, the scheduler can
  concentrate load without any corresponding latency benefit. The spec's
  own S5 anticipated this exact possibility ("If it does not, that is the
  finding"), which is the one place the spec correctly declined to assume
  its own hypothesis.

## What shipped

- `src/engine/placement/binding.py` — `distance_key` extracted from
  `_nearest` (task 14's logic, reused not rewritten, per S3.1).
- `src/integration/cluster_scheduler/topology_aware.py` —
  `select_replica`, `TopologyAwareClusterSchedulerConfig`,
  `TopologyAwareClusterScheduler`.
- `tests/test_topology_scheduler.py` — 7 tests (the 5 required plus 2
  supporting), covering `select_replica` directly (no full
  `BaseClusterScheduler` object graph needed, matching
  `tests/test_binding.py`'s own approach) and the closure finding with a
  real run.
- `tools/run_topology_scheduler_study.py` — the real M2N-path sweep.

Two commits on `task-15-topology-scheduler`, stacked on `task-14-binding`;
neither touches `upstream/`.
