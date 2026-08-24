# Task 40 — Multi-rack fabrics

Branch: `task-40-multirack`, branched from `task-39-formula-gaps`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`.

218 tests pass (204 unchanged + 14 net new in `tests/test_blueprints.py`),
and `python3 tools/check_import_direction.py` exits 0.

---

## 1. The construction, and port counts

**Extended `clos_fat_tree_fabric` in place** (not a sibling function) —
`depth` was already the dispatcher this function's own callers use to
pick a tier count, and `depth=3` reuses every non-topology parameter
(`switch_radix`, `gpus_per_machine`, `nics_per_machine`, every bandwidth/
latency parameter) unchanged. Splitting into a separate function would
have meant two call sites choosing between them by depth anyway; adding
the branch inside keeps `depth` a genuine dial instead of a discriminator
between two similar-but-separate APIs. The actual depth=3 wiring lives
in a new private helper, `_three_tier_fat_tree`, for the reason
`_wire_machine` already is one: the depth=2 body above it had to stay
byte-for-byte unchanged, and interleaving a third tier's own wiring into
it by editing in place risked moving a line the depth=2 path also runs.

**The construction**, `k = switch_radix`, `half = k/2`:

```
pods = k, edge/pod = half, aggregation/pod = half, core = half^2
hosts/edge = half, total hosts = k * half * half = k^3/4
```

Global (not per-pod) switch indices throughout, so pod A's own
aggregation switch `j` and pod B's own aggregation switch `j` — which
share a *plane*, not an identity — never collide as `SwitchId`s. The
plane structure Task 03's own report calls out as the missing piece:
core switches are grouped into `half` planes of `half` each; aggregation
switch `j` (its *local*, within-pod index) connects to every core switch
in plane `j`, and none outside it.

**Port counts, verified by test at k=4/6/8, not asserted from the
formula alone** (`test_every_switch_port_count_closes`): every edge
switch uses `half` (hosts) + `half` (aggregation, one per pod-mate) =
`k` ports; every aggregation switch uses `half` (edges, one per
pod-mate) + `half` (core, one per plane-mate) = `k`; every core switch
collects exactly one link per pod (that pod's own same-indexed
aggregation switch) = `k`. Every switch, every tier, exactly `k`.

## 2. Whether `depth=2` is bit-identical

**Yes.** All 14 of the depth=1/depth=2 tests that existed before this
task pass completely unmodified, run both before and after this
task's own edits, not merely "should still be true":
`test_single_tier_matches_build_node_scale`,
`test_leaf_spine_counts_match_formula`,
`test_every_leaf_connects_to_every_spine`, `test_no_leaf_to_leaf_links`,
`test_same_leaf_path_is_one_hop`, `test_cross_leaf_path_traverses_a_spine`,
`test_every_host_reaches_every_other`,
`test_oversubscription_reduces_uplink_capacity`,
`test_oversubscription_shows_up_in_contention`,
`test_blueprint_fabrics_round_trip`. A new test,
`test_two_tier_unchanged`, additionally confirms that passing the two
new depth=3-only oversubscription parameters at non-default values has
*no effect at all* on a depth=2 fabric — proving they are not silently
threaded into the unchanged path, not merely absent from its signature
by omission.

`test_depth_three_raises` no longer applies (depth=3 now succeeds) and
was replaced by `test_depth_four_raises`, matching this task's own scope
boundary exactly.

---

## 3. The one comparison

**Model and workload**: Phi-tiny-MoE-instruct, the same
`num_requests=32, qps=20.0, prefill_tokens=32, decode_tokens=16`
convention every planner-search task since Task 32 has used, at
margin=0.992 (Task 28's own calibrated point: `attn_tp=1` infeasible,
`attn_tp∈{2,4,8}` feasible).

**Fabrics, equal total GPU count** (this task's own known trap, reused
from Task 36): a two-tier Clos at `switch_radix=16` (spines=8, leaves=16,
hosts/leaf=8, 128 hosts) against a three-tier Clos at `switch_radix=8`
(pods=8, edges/pod=aggs/pod=4, core=16, hosts/edge=4, 128 hosts) — both
`gpus_per_machine=1`, so 128 GPUs each.

### The premise check

With `gpus_per_machine=1`, every scale-up domain has exactly one member,
so `enumerate_attn_shapes`'s own shape-based deduplication collapses
every candidate at a given `attn_tp` to a single shape — `(1,1,...,1)`
on *both* fabrics, since domain membership (one GPU per machine) is
identical either way. This is not a defect in this comparison; it means
the shape abstraction (built for `build_node_scale`-style multi-GPU
domains) cannot distinguish a "near" cross-machine split from a "far"
one — only the fabric's own real network paths can, and that is exactly
what this comparison is checking. Confirmed directly, before running
anything through Frontier, by checking which switch tiers `packed()`'s
own (only) placement actually traverses at each degree:

| `attn_tp` | machines needed | 2-tier tiers touched | 3-tier tiers touched |
|---|---|---|---|
| 2 | 0-3 | `leaf` only | `edge` only |
| 4 | 0-5 | `leaf` only (fits within `hosts_per_leaf=8`) | `aggregation`, `edge` (spills past `hosts_per_edge=4`) |
| 8 | 0-9 | `leaf`, `spine` (spills past `hosts_per_leaf=8`) | `aggregation`, `edge` (still fits within one pod's own 4 edges) |

This is the premise: `attn_tp=4` is the case where the two-tier fabric
stays on one switch and the three-tier one does not; `attn_tp=8` is the
reverse asymmetry, where the two-tier fabric needs an extra tier and the
three-tier one does not. A comparison that only ever touched the same
tiers on both fabrics would not be testing what it appears to (this
task's own §5 warning) — this one touches different tiers at two
different degrees, in opposite directions, confirmed before any
Frontier run.

### The result

| `attn_tp` | 2-tier mean tpot (ms) | 3-tier mean tpot (ms) | margin |
|---|---|---|---|
| 2 | 18.3178 | 18.3178 | 0% (both single-tier, no crossing) |
| **4** | **27.2465** | **38.4465** | **+41.11% (3-tier slower)** |
| 8 | 69.1854 | 69.1854 | 0% (see §4) |

**Winner on both fabrics: `attn_tp=2`, unchanged** — a null result for
the winner, reported plainly rather than reworked into a positive one,
the same discipline Task 33's own domain8-vs-domain64 comparison
applied. But `attn_tp=4` is not the winner and shows a large, real,
mechanistically understood difference: the three-tier fabric's own
`hosts_per_edge=4` is smaller than the six ranks (`1` prefill + `4`
attention + `1` FFN) `packed()` needs, forcing an aggregation hop the
two-tier fabric's own `hosts_per_leaf=8` avoids entirely. This is
exactly the "planner's answer can differ" property this task exists to
make askable — even though, for *this* particular model, workload, and
margin, it does not flip which degree wins.

---

## 4. Whether oversubscription reaches the measured cost

**Yes, at the tier the traffic actually crosses, and not otherwise** —
checked with `engine.network.transfers.analyse()` directly (the same
mechanism `test_oversubscription_shows_up_in_contention` already uses
for depth=2), not inferred from capacity alone:

| traffic | `oversubscription_edge_agg` | `oversubscription_agg_core` | makespan (ns) |
|---|---|---|---|
| cross-edge, same pod | 1:1 | 1:1 | 224,000 |
| cross-edge, same pod | **4:1** | 1:1 | **824,000** (3.68x) |
| cross-edge, same pod | 1:1 | **4:1** | 224,000 (unchanged) |
| cross-pod | 1:1 | 1:1 | 234,000 |
| cross-pod | 1:1 | **4:1** | **834,000** (3.57x) |

Each ratio moves the makespan substantially when the traffic it governs
is actually on the path, and leaves it untouched when it is not — the
cross-edge traffic never touches the core tier, and
`oversubscription_agg_core` correctly does not move it at all. This is
the same independence `test_per_tier_oversubscription_is_independent`
already established at the capacity level, now confirmed at the
measured-contention level too, on real (if small) concurrent traffic
rather than only on link capacities in isolation.

**Why the baseline comparison in §3 shows no oversubscription effect at
all — a genuine, correctly-modelled result, not a gap**: §3 used
`oversubscription_edge_agg=oversubscription_agg_core=1.0` throughout
(default provisioning), and a single decode step's own communication
(one M2N hop, one within-degree allreduce) never approaches saturating
even a fully-provisioned uplink at this workload's scale. The `attn_tp=4`
margin in §3 is entirely a **hop-count/latency** effect (an extra
switch traversal costs a fixed latency regardless of how much capacity
it has spare), not an oversubscription effect — and `attn_tp=8`'s own
*equal* cost on both fabrics (§3's own table) is explained by the same
mechanism from the other direction: both fabrics need exactly 6 hops at
that degree (`leaf→spine→leaf` on one, `edge→aggregation→edge` on the
other), and every scale-out link in both constructions shares the same
bandwidth/latency parameters, so equal hop count gives equal cost
regardless of which tier names are involved. Oversubscription is a
capacity effect and shows up under contention (§4's own table); the §3
comparison was never loaded enough to contend for anything, correctly
reported as such rather than mistaken for oversubscription doing
nothing.

---

## 5. Anywhere this specification is wrong

Nothing in the construction's own formulas or wiring instructions
required correction — every count and every wiring rule in this task's
own §2 matched what a real, tested construction needed, unlike the
history this task's own §1 explicitly warns about repeating. One
implementation detail is worth recording precisely because it *did*
go wrong once during this task's own development, caught by the very
test this task's own acceptance criteria required:

**The first implementation of the two oversubscription ratios was not
independent, and a required test caught it.** `oversubscription_agg_core`
was first computed relative to `edge_to_agg_GBps` — the *already-
reduced* result of applying `oversubscription_edge_agg` — rather than
relative to the same fully-provisioned base rate `oversubscription_edge_agg`
itself is measured against. This chained the two ratios: setting
`oversubscription_edge_agg=4.0` alone silently changed
aggregation-to-core capacity too, exactly the failure
`test_per_tier_oversubscription_is_independent` (required by this
task's own §4.1) exists to catch, and did catch, on the first run.
Fixed by measuring both ratios against the same base rate
(`nics_per_machine * scale_out_GBps`) independently. This is exactly
the class of error this task's own §1 opens with — a plausible,
arithmetically-motivated formula that was wrong in a way port counts
alone would never reveal, caught only because independence was tested
directly rather than assumed from the formula's own resemblance to the
two-tier case.

One naming collision, mechanical rather than substantive: this task's
own required test table names `test_every_host_reaches_every_other`
for the three-tier section, which already exists (unmodified) for
depth=2 in the same file. Used
`test_every_host_reaches_every_other_three_tier` instead, to avoid
silently shadowing the existing depth=2 test.

## What shipped

- `src/engine/infragraph/blueprints.py` — `clos_fat_tree_fabric` extended
  with `depth=3` (delegating to a new private `_three_tier_fat_tree`)
  and two new parameters, `oversubscription_edge_agg`/
  `oversubscription_agg_core`; `depth>3` now raises instead of `depth>2`.
  `depth=1`/`depth=2` bodies unchanged.
- `tests/test_blueprints.py` — 14 net new tests: the 8 named in this
  task's own §4.1, `test_depth_four_raises` replacing
  `test_depth_three_raises`, and the three parametrized tests
  (`test_three_tier_counts_match_formula`,
  `test_every_switch_port_count_closes`, `test_plane_structure_is_correct`)
  each contributing 3 cases (k=4,6,8).
- `tools/planner.py` — two new named topologies,
  `clos_2tier_128`/`clos_3tier_128` (§3's own pair), registered in
  `_TOPOLOGIES` so `SimulationEvaluator`'s own subprocess reconstruction
  can find them by name.
- `docs/tasks/40-multirack-report.md`, this report.

One commit on `task-40-multirack`, stacked on `task-39-formula-gaps`.
Task 33's own sixteen-row table and Task 36's own two-fabric result both
reproduce bit-identical, checked directly.
