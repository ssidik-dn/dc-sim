# Task 56 — Can the planner reach the arrangement a deployment would use?

Branch: `task-56-enumeration-reach`, branched from `task-55-noise-pilot`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`. No GPU,
no fleet access — every number below comes from real, non-dummy Frontier
simulation runs (CPU-only, `tools/planner_core.py` and
`tools/planner.py`'s own real functions) or from reading
`src/engine/placement/placement.py` and `tools/planner_core.py` directly.

254 tests pass, unchanged (see §0 for the discrepancy with this task's
own acceptance bar, which quotes 276); `check_import_direction.py` exits
0. Investigation only — nothing in `src/` or `tools/` was changed.

**Short answer: Task 55's observation is confirmed, the mechanism is
found and read directly (not inferred), and it turns out not to matter
for either degree pair actually tested — the unreachable arrangement
would have placed second, not first, in both cases. A real completeness
gap, not a correctness one, for the two cases checked.**

---

## 0. One correction to this task's own acceptance bar, checked first

This task's own §4 says "all 276 must pass, unchanged." The actual count
on this branch's own parent (`task-55-noise-pilot`) is **254 pass + 22
new = 276 total, with 5 skipped** (Task 53's own Fix B behavioral tests,
which need `torch` — absent from this sandbox, unrelated to this task).
Confirmed by running it before touching anything else:

```
$ python3 -m pytest -q
.sssss.................................................................. [ 25%]
........................................................................ [ 51%]
........................................................................ [ 76%]
.................................................................       [100%]
276 passed, 5 skipped in 39.9s
```

Matches exactly. Stated here because §6 of this report returns to it —
worth confirming rather than assuming the number carried over correctly
from the prior task's own final state.

---

## 1. The full enumerated list, both degree pairs

Real `enumerate_joint_arrangements(topology, attn_tp, ffn_ep)` calls
against `build_node_scale(num_machines=2, gpus_per_machine=8)` — the
same real 2-machine topology Task 54/55 used, not a new one built for
this task.

### `attn_tp=4, ffn_ep=2`

| `(attn_shape, ep_shape)` | attn domains | ep domains | both whole, different machines? |
|---|---|---|---|
| `((2,2),(1,1))` | `{0,1}` | `{0,1}` | no |
| `((2,2),(2,))` | `{0,1}` | `{0}` | no |
| `((3,1),(1,1))` | `{0,1}` | `{0,1}` | no |
| `((3,1),(2,))` | `{0,1}` | `{0}` | no |
| `((4,),(1,1))` | `{1}` | `{0,1}` | no |
| `((4,),(2,))` | `{0}` | `{0}` | **no — this is the colocated arrangement** |

**Six arrangements, none of them the natural split.** The one entry
with both groups whole (`(4,),(2,)`) has both on the *same* domain — it
is the colocated arrangement, not the split one.

### `attn_tp=2, ffn_ep=4` (Task 54's second candidate)

| `(attn_shape, ep_shape)` | attn domains | ep domains | both whole, different machines? |
|---|---|---|---|
| `((1,1),(2,2))` | `{0,1}` | `{0,1}` | no |
| `((1,1),(3,1))` | `{0,1}` | `{0,1}` | no |
| `((2,),(2,2))` | `{1}` | `{0,1}` | no |
| `((2,),(3,1))` | `{1}` | `{0,1}` | no |
| `((2,),(4,))` | `{0}` | `{0}` | **no — colocated** |

**Five arrangements this time, not six** (`ffn_ep=4`'s own richer ep
shape space at `attn_tp=2` collapses one fewer distinct key than
`attn_tp=4`'s does — confirmed directly, not assumed, by the count
itself). **Same result: no natural split, in either degree pair.** Task
55's own observation holds for both, not only the one it happened to be
built around.

---

## 2. Whether the natural split is reachable, and why not

**Not reachable, for either degree pair — confirmed by reading
`enumerate_joint_arrangements` and the placement policies it calls
directly, per this task's own known trap, not inferred from the output
above.** Two mechanisms compound, not one:

### 2.1 `packed()` never constructs it as a raw candidate at all

`packed()` (`src/engine/placement/placement.py`) fills GPUs by walking
one flat pool — every domain's members, sorted, concatenated in domain
order — and assigns `deployment.ranks` to that pool **in rank order**,
with no group-boundary awareness whatsoever. `enumerate_joint_arrangements`
adds ranks to the deployment in the fixed order PREFILL, then
DECODE_ATTN, then DECODE_FFN. Whenever the joint footprint (prefill + 
attn_tp + ffn_ep) fits inside one domain's own size — true for both
`(4,2)` (`1+4+2=7≤8`) and `(2,4)` (`1+2+4=7≤8`) — `packed()`
mechanically never reaches the second domain at all. It is not that
`packed()` "chooses" colocation over the split; the split is not a
possible output of this policy at this size, structurally, for the
same reason Task 32/34 already found ("`packed()`'s own rank ordering
gives a later group a one-slot offset from an earlier one" — the same
lack of group-boundary awareness, one level up).

`spread()` does not reach it either, for a related but distinct reason:
it round-robins **by rank**, not by group — every single rank, in
global order, advances the domain cursor, so a run of consecutive ranks
belonging to one group gets interleaved across domains rather than kept
together. It reliably *fragments* both groups; it does not reliably
keep either one whole.

The "packed-if-it-fits" explicit fallback (the block reading
`if len(domain_ids) >= 3:`) is written to place attention in its own
domain and the expert-parallel group in a *third*, separate one — but
it requires **three** real domains (one for prefill/the FFN anchor
rank, one for attention, one for experts) to fire at all. With exactly
two real machines, `len(domain_ids) == 2`, and this fallback never
executes — confirmed by reading the guard directly, not by its absence
from the output.

**Only `fragmented()` — the random policy — can produce it, and only
by chance.** Confirmed live: inspecting the *raw* candidates
`enumerate_joint_arrangements` generates, before deduplication,
`fragmented(seed=3)` produces exactly the split arrangement for
`(4,2)` (`attn_domains={1}, ep_domains={0}`), and `fragmented(seed=9)`
and `fragmented(seed=49)` both produce it for `(2,4)`. It is not
unreachable by the random policy — it is unreachable by every
*deterministic* one, and present in the *raw* candidate pool the
random policy contributes.

### 2.2 The canonical shape signature then discards it anyway

`Placement.group_shape()`'s own docstring states its purpose plainly:
"how the group distributes across scale-up domains, as a sorted tuple
... most placements are isomorphic, so caching on it keeps memoisation
cardinality bounded." **It records only how many ranks land in each
domain, never which domain** — by design, and correctly so for a single
group considered alone (Task 32's own original use), where "domain 3"
and "domain 7" really are interchangeable.

Once two *different* groups are enumerated jointly, this same blindness
becomes the defect: the split arrangement's own key,
`(attn_shape=(4,), ep_shape=(2,))`, is **identical** to the colocated
arrangement's key — `group_shape()` cannot express "these two domains
are the same one" or "these two are different ones" at all, only "each
group is whole." `enumerate_joint_arrangements`'s own
`if key not in arrangements: arrangements[key] = p` then keeps
whichever placement is discovered **first** for a given key and
silently drops every later one that collides — and `packed`/`spread`
are tried before the 60 `fragmented` seeds, so the colocated
placement (from `packed()`, always available whenever the footprint
fits one domain) claims the key before `fragmented(seed=3)` (or `9`,
or `49`) ever gets a chance to.

**Neither mechanism is sufficient alone to explain the absence; both
apply, and the second is what makes the first one's failure
permanent.** If `group_shape()` recorded relative domain identity for a
*pair* of jointly-enumerated groups (this task's own §2.2 candidate 2),
the random policy's own correct discoveries would survive into the
returned dict even without fixing `packed()`/`spread()` at all. This is
not one of the task's own three named candidates in isolation — it is
the first candidate ("the enumeration builds shapes per group... a
combination... may not be expressible in the shape vocabulary") and the
second ("the canonical signature... may be collapsing this one into
another it considers equivalent") describing the *same* root property
of `group_shape()` from two angles, compounded by the third
(`packed()`'s own fixed rank ordering, confirmed, not the vocabulary
issue but a second, independent contributor) determining that nothing
deterministic ever offers the random policy's correct answer a chance
to be found *without* relying on luck in the first place.

---

## 3. How it would rank against the reachable six

Priced directly — the explicit split placement Task 55 built, against
every reachable arrangement, same model (`Phi-tiny-MoE-instruct`), same
workload (32 requests, `qps=20`, 32 prefill / 16 decode tokens — this
project's own established default), same `num_blocks`
(`feasible_num_blocks` at margin 0.7), same regime (deterministic,
`seed=0`) the planner's own `evaluate()` uses for every one of these
seven — not a shortcut through `_placement_for` (which, per §2, cannot
resolve to the split arrangement at all; the split candidate was run
through a hand-built `explicit()` placement, everything else identical):

**`attn_tp=4, ffn_ep=2`:**

| rank | mean TPOT (ms) | arrangement | attn domains | ep domains |
|---|---|---|---|---|
| 1 | **11.199** | `(4,),(2,)` — reachable, colocated | `{0}` | `{0}` |
| **2** | **13.473** | **explicit split, unreachable** | `{0}` | `{1}` |
| 3 | 15.631 | `(4,),(1,1)` — reachable | `{1}` | `{0,1}` |
| 4 | 24.015 | `(3,1),(2,)` — reachable | `{0,1}` | `{0}` |
| 5 | 26.174 | `(2,2),(1,1)` — reachable | `{0,1}` | `{0,1}` |
| 5 | 26.174 | `(3,1),(1,1)` — reachable (exact tie) | `{0,1}` | `{0,1}` |
| 7 | 26.289 | `(2,2),(2,)` — reachable | `{0,1}` | `{0}` |

**`attn_tp=2, ffn_ep=4`:**

| rank | mean TPOT (ms) | arrangement | attn domains | ep domains |
|---|---|---|---|---|
| 1 | **9.572** | `(2,),(4,)` — reachable, colocated | `{0}` | `{0}` |
| **2** | **11.845** | **explicit split, unreachable** | `{0}` | `{1}` |
| 3 | 13.792 | `(2,),(2,2)` — reachable | `{1}` | `{0,1}` |
| 4 | 13.795 | `(2,),(3,1)` — reachable | `{1}` | `{0,1}` |
| 5 | 18.156 | `(1,1),(2,2)` — reachable | `{0,1}` | `{0,1}` |
| 6 | 20.433 | `(1,1),(3,1)` — reachable | `{0,1}` | `{0,1}` |

**The unreachable arrangement is the true second place in both cases —
not a trivial loser, and not the winner either.** It beats every other
reachable candidate (5 of 6, and 4 of 5, respectively) by a wide
margin, and loses only to the one arrangement the search already finds
and already recommends. The margin against the winner is real but
modest: +20.3% (`(4,2)`) and +23.8% (`(2,4)`) — nowhere near the ~8.7x
containment effect Task 43A found for a whole rack-scale domain, and
in the same rough range as this project's own established
pool-separation effect (Task 12/42/43A's own "~14-15%"/"+16.7%/+33.4%"
figures) — consistent with what a genuine cross-machine hop, and
nothing structurally worse than that, should cost here.

(The exact tie at rank 5 — `(2,2),(1,1)` and `(3,1),(1,1)` both landing
at precisely 26.17388492626897 ms — reproduces Task 36's own already-
established finding exactly: "the communication cost of *any*
cross-domain arrangement appears to be dominated by a shared 'at least
one cross-domain hop' term, not by exactly how the group is
partitioned." Not a new anomaly; the same mechanism recurring on a
different degree pair.)

---

## 4. Which of §2.3's two situations applies

**Unreachable and would have lost anyway — a completeness gap, not a
correctness one — for both degree pairs actually tested.** The planner's
own top choice (colocated) is genuinely the real optimum in both cases;
the blind spot costs nothing to the *decision* Task 54's whole
validation design rests on, for this exact model/workload/degree
combination.

**This does not generalize to "the blind spot never matters," and this
report does not claim it does.** Two things keep it open rather than
closed:

- The margin between colocated and the (unreachable) split is real but
  not overwhelming (+20–24%) — a different model, workload, or degree
  pair where cross-machine cost is proportionally smaller relative to
  compute (a larger model, say, where the fixed hop cost is a smaller
  fraction of a larger per-token compute time) could plausibly close
  or reverse that gap. Nothing checked here rules that out; only these
  two specific points were priced.
- The blind spot is structural (§2), not specific to
  `Phi-tiny-MoE-instruct` or to these two degree pairs — any joint
  `(attn_tp, ffn_ep)` pair whose combined footprint fits one domain
  inherits it, for the same reason. Whether it ever *decides* an
  outcome is a property of the model/workload, checked here for two
  points and not proven safe in general.

For the two-machine validation Task 54 designed specifically, this is
good news stated precisely rather than overclaimed: the one forcing
configuration it chose (`attn_tp=4, ffn_ep=2`) happens to sit on the
safe side of this gap — the search's own blind spot does not change
what the pilot would find, for that exact configuration.

---

## 5. What fixing it would require (described, not built, per §4's own instruction)

Two independent options, either sufficient on its own, not attempted
here:

1. **Make the canonical key domain-identity-aware across the two
   groups, not just within each one.** `group_shape()`'s own per-group
   signature is correct and should stay as-is for a *single* group;
   what is missing is a *relative* signature for the *pair* —
   concretely, whether attention's own set of occupied domains and the
   expert group's own set are the same, disjoint, or overlapping,
   folded into the key `enumerate_joint_arrangements` builds (not
   `group_shape()` itself, which other callers — `enumerate_attn_shapes`,
   Task 41's replica enumeration — still need unchanged). This directly
   targets §2.2's own mechanism and would let the existing `fragmented()`
   seeds' own already-correct discoveries survive deduplication, with
   no change to `packed()`/`spread()` at all.
2. **Extend the "packed-if-it-fits" explicit fallback to the two-domain
   case.** Currently gated on `len(domain_ids) >= 3`; a variant that
   checks whether attention and the expert group could each get a
   *whole*, *separate* domain when there are only two real domains
   (dropping the separate prefill/FFN-anchor domain requirement, or
   co-locating that bookkeeping rank with whichever group has room)
   would deterministically construct the split candidate every time,
   rather than leaving its discovery to `fragmented()`'s own random
   luck. This targets §2.1's mechanism directly and is a smaller,
   more localized change than option 1.

Either fix touches `enumerate_joint_arrangements`, which Task 41
already established the reasoning behind — per this task's own §4,
changing it "deserves its own task with its own regression bar," and
that regression bar is already named precisely: Task 33's sixteen-row
table and Task 36's two-fabric result, both of which must keep
reproducing bit-identically (neither uses `enumerate_joint_arrangements`
at `ffn_ep>1`, so a targeted fix to the two-group case should not touch
either, but that is exactly the kind of claim this project's own
discipline requires checking by running them, not by this argument).
Not attempted here.

---

## 6. Anywhere this specification is wrong

**The test-count discrepancy in §4** (this task's own acceptance bar
says "276 must pass, unchanged"; the correct, checked figure is 254
pass + 22 new = 276, with 5 skipped) — a small imprecision (the 276
figure is internally consistent with Task 55's own final state, just
under-specifies that 5 of them are skipped, not passed) worth
recording exactly rather than silently reading past, per §0.

**Its own quotation of Task 55 is accurate**, checked directly against
`docs/tasks/55-noise-pilot-report.md`'s own text: "none of the six
naturally-enumerated arrangements at this degree pair happens to keep
*both* groups internally whole on two *different* machines" is verbatim,
not paraphrased or taken out of context — Task 55's own surrounding
sentence already called this "a real, load-bearing gap in the search's
own reach, not something to route around silently," which is exactly
the framing this task continues.

**"This may be a false alarm" (§6's own trap) did not hold, but was the
right thing to check first** — Task 55 was, per this task's own §6,
"preparing tooling rather than auditing the enumerator," and flagged
its own finding as an observation, not a conclusion. §3 of this task
confirms it was correct, not a false alarm, but confirming that cost
nothing and was worth doing before treating the observation as
established.

**Otherwise, nothing else in this specification's own framing required
correction.** The two-situation distinction in §2.3 (unreachable-and-
would-have-won vs. unreachable-and-would-have-lost) is exactly the
right cut, and this report's own §4 answers it precisely for the two
cases checked, without extending the answer further than the evidence
supports.

## What shipped

Nothing in `src/` or `tools/` — investigation and enumeration only, per
this task's own acceptance criteria. `docs/tasks/56-enumeration-reach-report.md`,
this report, is the only artifact. `enumerate_joint_arrangements` was
read and exercised directly, never modified.

One commit on `task-56-enumeration-reach`, stacked on
`task-55-noise-pilot`. 276 tests pass (254 unchanged + 22 from Task 55),
5 skipped (Task 53's own Fix B tests, unrelated); `check_import_direction.py`
exits 0.
