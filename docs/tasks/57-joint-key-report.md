# Task 57 — Let the enumeration reach the natural split

Branch: `task-57-joint-key`, branched from `task-56-enumeration-reach`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`. No
GPU, no fleet access — every number below comes from real, non-dummy
Frontier simulation (CPU-only) or from `tools/planner_core.py`'s own
enumeration/placement machinery directly.

282 tests pass (276 unchanged + 6 new), 5 skipped (Task 53's own Fix B
tests, unrelated, still skipped for lack of `torch`); `check_import_direction.py`
exits 0. Task 33's sixteen-row table and Task 36's two-fabric result
both reproduce bit-identically.

---

## 1. Scope taken: the key fix, and the fallback too

Took Task 56's own first recommendation as scoped: added a *relative*
signature — whether the two groups' own occupied-domain sets are
identical, disjoint, or overlapping — to the key
`enumerate_joint_arrangements` builds, leaving `group_shape()` itself
untouched.

**Also extended the packed-if-it-fits fallback to two domains — the
second option, taken as well, not merely considered.** §3.5 explains
why the key fix alone was not enough on its own terms.

## 2. The enumerated lists, before and after

`attn_tp=4, ffn_ep=2`:

| before (Task 56) | after |
|---|---|
| `((2,2),(1,1))` | `((2,2),(1,1),'same')` |
| `((2,2),(2,))` | `((2,2),(2,),'overlapping')` |
| `((3,1),(1,1))` | `((3,1),(1,1),'same')` |
| `((3,1),(2,))` | `((3,1),(2,),'overlapping')` |
| `((4,),(1,1))` | `((4,),(1,1),'overlapping')` |
| `((4,),(2,))` | `((4,),(2,),'same')` |
| — | **`((4,),(2,),'disjoint')`** |

6 arrangements → **7**.

`attn_tp=2, ffn_ep=4`:

| before (Task 56) | after |
|---|---|
| `((1,1),(2,2))` | `((1,1),(2,2),'same')` |
| `((1,1),(3,1))` | `((1,1),(3,1),'same')` |
| `((2,),(2,2))` | `((2,),(2,2),'overlapping')` |
| `((2,),(3,1))` | `((2,),(3,1),'overlapping')` |
| `((2,),(4,))` | `((2,),(4,),'same')` |
| — | **`((2,),(4,),'disjoint')`** |

5 arrangements → **6**.

**The natural split — attention whole in one domain, experts whole in
the other — is present in both, distinct from the colocated arrangement,
confirmed by inspecting the resolved `Placement` object directly** (not
only the key string): `domains_spanned(attn_group.ranks) = {0}`,
`domains_spanned(ep_group.ranks) = {1}` for both degree pairs.

## 3. Cardinality — the space did not explode

| case | domains | before | after | Δ |
|---|---|---|---|---|
| `attn_tp=4, ffn_ep=2`, 2×8-GPU | 2 | 6 | 7 | +1 |
| `attn_tp=2, ffn_ep=4`, 2×8-GPU | 2 | 5 | 6 | +1 |
| `attn_tp=4, ffn_ep=4`, 3×4-GPU | 3 | 10 | 11 | +1 |
| `attn_tp=8, ffn_ep=8`, 3×8-GPU (larger case) | 3 | 23 | 23 | **0** |

**Growth is modest, not the 3x a naive "triple every key" estimate would
predict, because most `(attn_shape, ep_shape)` pairs cannot realize all
three relative values.** Two singleton domain sets (`(4,)`/`(2,)`, both
spanning exactly one domain each) can only ever be `"same"` or
`"disjoint"` — `"overlapping"` is impossible between two singletons, so
that pair grows by exactly one entry, not two. A pair where one side
already spans every domain (e.g. `(1,1)` on a 2-domain fabric) can only
ever be `"same"` (its domain set already equals the whole fabric) or
`"overlapping"` — never `"disjoint"`, since a set that is everything
cannot be disjoint from anything nonempty. The largest case tested
(`attn_tp=8, ffn_ep=8`) shows **zero** growth: the one place a
`"disjoint"` pair for `((8,),(8,))` exists, the pre-existing ≥3-domain
fallback (attention in its own domain, experts in a separate third one —
genuinely different domains, by construction) was already producing
it; task 57's own key change only made that already-present
arrangement's relative value *nameable*, not new.

**Collapse ratios, computed directly, stay in the same range as Task
32's own 11.8x and Task 41's own 3.3x–31x:**

| case | raw candidates | distinct arrangements | collapse ratio |
|---|---|---|---|
| `(4,2)`, 2×8-GPU | ~63 | 7 | 9.0x |
| `(2,4)`, 2×8-GPU | ~63 | 6 | 10.5x |
| `(4,4)`, 3×4-GPU | ~64 | 11 | 5.8x |
| `(8,8)`, 3×8-GPU | ~64 | 23 | 2.8x |

No case tested shows the affordability property degrading — exhaustive
enumeration stays cheap relative to what it dedupes.

## 4. Whether the packed-if-it-fits fallback should also be extended

**Yes — added, and it is what actually makes the split reachable
without relying on `fragmented()`'s own luck at all**, confirmed by
disabling the random policy outright: with `n_fragmented_seeds=0`, the
`'disjoint'` key is still present for both `(4,2)` and `(2,4)`. Before
this addition, the key fix alone would only have let `fragmented()`'s
*existing, already-correct* discoveries (seeds 3, 9, 49 — Task 56's own
finding) survive deduplication; it would not have made the arrangement
reachable on a run with fewer seeds, or on a degree pair large enough
that chance stops finding it (Task 57's own §2 and §6 both name exactly
this fragility). The new fallback places attention (plus the prefill
rank, since only two domains exist here — there is no third, separate
bookkeeping domain the way the ≥3-domain fallback uses) whole in one
domain and the expert group whole in the other, whenever both fit —
deterministically, every time, independent of any seed.

**Named limit, not silently glossed over**: the new fallback requires
room for `attn_tp + 1` (attention plus the prefill rank) in one domain,
so it does not fire when `attn_tp` exactly equals the domain size (e.g.
`attn_tp=8` on an 8-GPU domain — no room left for prefill alongside a
full domain's worth of attention). Checked directly: this boundary case
does not touch either of the two configurations Task 54 actually
designed (`attn_tp∈{4,2}`, domain size 8 — room to spare, `4+1=5≤8`),
and `attn_tp=8`'s own joint footprint with `ffn_ep=8` (`1+8+8=17`) already
exceeds two real machines' combined 16 accelerators regardless (Task 56's
own finding), so this limit does not bite anywhere this project's own
two-machine validation actually needs it to. Recorded precisely so a
future task choosing a different degree pair does not rediscover it the
hard way.

## 5. The regression comparison — run, not argued

**Task 33's sixteen-row table:**
```
WINNER: tp=2 shape=(2,) mean_tpot_ms=11.6803
```
Bit-identical to every prior reproduction.

**Task 36's two-fabric result:**
```
domain8_40gpu  WINNER: tp=8 shape=(8,) mean_tpot_ms=326.2362
domain4_40gpu  WINNER: tp=8 shape=(4,3,1) mean_tpot_ms=446.5146
```
Bit-identical to every prior reproduction. Neither uses joint
enumeration above a single expert group (`ffn_ep=1` throughout both),
so `relative` is `None` for every candidate either result ever builds —
confirmed by running both fresh on this branch, not inferred from the
fact that `relative` defaults to `None` and `Candidate.key` omits it
when unset.

## 6. Whether the split ranks where Task 56 predicted

**Yes, exactly — priced through the real, fixed path
(`Candidate` → `_placement_for` → real Frontier `Simulator`, no
hand-built bypass), not re-derived by argument:**

| degree pair | colocated (ms) | split (ms) | margin | Task 56's own prediction |
|---|---|---|---|---|
| `(4,2)` | 11.198924926271072 | 13.47252492627015 | **+20.29%** | +20.3% |
| `(2,4)` | 9.571513921663295 | 11.845113921662099 | **+23.76%** | +23.8% |

**Bit-identical to Task 56's own hand-built explicit-placement numbers**,
confirming the fix resolves to the exact same physical arrangement Task
55/56 constructed by hand — this is not merely "an arrangement got
added," it is "the *same* arrangement Task 55 built with `explicit()`
is now reachable through the planner's own real, unmodified evaluation
path." Colocated still wins both times, matching Task 56's own §2.3
finding (unreachable-and-would-have-lost, a completeness gap) — fixing
reachability did not change which arrangement is actually best for
either tested point, as expected; it changed only whether the planner's
own search can *see* the comparison at all.

Pinned as a test (`test_natural_split_prices_where_task_56_predicted`,
`tests/test_planner_core.py`), with the margins checked to within 0.2
percentage points of Task 56's own reported figures, so a future change
that silently resolves to a different physical placement (not merely a
different price) fails loudly rather than passing on a coincidence.

---

## 7. Anywhere this specification is wrong

**Nothing in this specification's own account of Task 56 needed
correction.** Its own quotation, its own framing of the two mechanisms,
and its own predicted margins (20.3%/23.8%) all held up exactly as
stated — confirmed by running the fix, not by re-reading Task 56's
report a second time.

**One precision worth adding, not a correction**: §2's own framing
("Task 56 named two options and recommended the first... consider
whether [the second] is worth doing as well") reads as though the two
options are independent alternatives, either of which might suffice on
its own. §4 above shows they are not quite independent in practice: the
key fix alone is *necessary* (without it, even a deterministically-
constructed split candidate would still collide with the colocated
one's key and be discarded) but not *sufficient* for robustness (without
the fallback, reachability still depends on `fragmented()`'s own luck).
Both were needed to close the gap the way this task's own §2 already
worried about — this is confirmation of the task's own instinct to ask
the question, not a disagreement with it.

**Otherwise, nothing else required correction.** The "Task 56's margins
are a prediction, not a guideline" trap held exactly as framed — §6's
own bit-identical match is the direct confirmation the spec asked for,
not an approximate one.

## What shipped

- `tools/planner_core.py` — `enumerate_joint_arrangements`'s own key
  gains a third component, `relative` (`"same"`/`"disjoint"`/
  `"overlapping"`/`None`), computed by the new
  `_relative_domain_placement`; a new deterministic fallback constructs
  the two-domain split whenever both groups fit, without relying on
  `fragmented()`; `Candidate` gains a matching `relative: Optional[str]
  = None` field (backward-compatible — every existing construction call
  and `.key` string is unchanged when it is not passed).
- `tools/planner.py` — `_placement_for`'s own lookup, `_run_scenario`,
  and `evaluate()`'s subprocess argv all thread `relative` through, so
  the fix reaches the real evaluation path, not only the enumeration
  function in isolation.
- `tests/test_planner_core.py` — three existing tests updated for the
  new key arity (`test_enumerate_joint_arrangements_at_ep1_matches_enumerate_attn_shapes`,
  `test_enumerate_joint_arrangements_reaches_the_fully_packed_pair_when_it_fits`,
  `test_enumerate_joint_arrangements_keys_are_ordered_pairs_not_multisets`,
  plus one comment fix in `test_enumerate_joint_arrangements_collapses_raw_candidates`);
  six new tests: the natural split is reachable and genuinely on two
  different domains, at both degree pairs; the colocated arrangement is
  unaffected; `relative` is `None` when either group is absent;
  `_relative_domain_placement`'s own three-way classification, checked
  directly; and the required-by-this-task's-own-§4 pin against Task
  56's real, priced margins.
- `tests/_natural_split_pricing_probe.py` — subprocess probe (mirrors
  `_memory_planner_probe.py`'s own established pattern), one real
  Frontier evaluation per invocation (Task 41's own discipline).
- `docs/tasks/57-joint-key-report.md`, this report.

One commit on `task-57-joint-key`, stacked on `task-56-enumeration-reach`.
Task 33's sixteen-row table and Task 36's two-fabric result both
reproduce bit-identical.
