# Task 34 — Was "packed" ever packed?

Branch: `task-34-packed-audit-verify`, branched from `task-45-regime`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`.

240 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. No code changed — measurement only, per this task's own §4;
`engine.placement.placement.packed()` was not touched.

---

## 0. A pre-existing branch of the same name, and how this report relates to it

Before doing anything else: a branch literally named `task-34-packed-audit`
already existed in this repository (`git checkout -b` collided on it),
authored by a different email than this session's own, dated two days
before this session, and branched from `task-32-search`'s own tip —
a sibling of `task-33-planner`, not an ancestor of anything from Task 35
onward. Its own commit (`96b0f22`) is report-only (364 lines, no source
changes) and answers this exact spec in detail.

Per instruction, that report was **verified, not blindly trusted or
silently redone**. Every load-bearing claim in it was independently
re-derived from source or re-measured with fresh Frontier runs, not
re-read and taken on faith — the results are below, and they confirm
its central claims to the same or better precision than it reports them
(§2, §3 give the exact re-measured figures). Where this report adds to
it rather than merely confirming it: **§1's own required answer to
whether Task 44's `ep=4` discrepancy shares this root cause** (Task 44
did not exist when the prior branch was written, so it could not have
addressed this), and **an updated blast-radius table covering Tasks
35–45** (also written after it).

---

## 1. Does Task 44's open `ep=4` discrepancy share this root cause? No.

Task 44's report (`docs/tasks/44-ep-placement-report.md`) left open why
Task 33's own cited `ep=4` figure (14.4434ms, "three of four EP ranks
land split across two domains") does not reproduce today (byte-for-byte
reconstruction gives 10.2228ms, fully packed) — four candidates
(topology, deployment code, shape enumerator, memory formula) were
checked and ruled out, with no explanation found.

**This audit's own mechanism does not explain it either, and can be
ruled out precisely rather than by assumption.** `_placement_for`'s
reference at `attn_tp=1` is `enumerate_attn_shapes`'s own single
candidate — `packed()` applied to a 3-rank probe deployment (PREFILL,
DECODE_ATTN tp=1, DECODE_FFN tp=1, ep=1 always, regardless of the real
candidate's own `ffn_ep`). Checked directly against `domain8`
(5 machines × 8 GPUs/machine, confirmed byte-identical to Task 33's own
original commit since Task 44):

```
PREFILL/r0/0     -> m0.g0  (domain 0)
DECODE_ATTN/r0/0 -> m0.g1  (domain 0)
DECODE_FFN/r0/0  -> m0.g2  (domain 0)
```

Three ranks occupy domain 0's first three of **eight** slots, leaving
**five free**. At `ep=4`, the real deployment's own three *extra* FFN
ranks (beyond the one the reference already covers) are more than
covered by those five free slots — the old first-fit fallback's own
domain-order walk fills them into domain 0 before ever touching domain
1. **All four EP ranks land in domain 0 — the offset mechanism this
task documents (§2) never comes close to triggering**, because its own
trigger condition (`ranks-already-placed + degree > domain_size`) needs
`domain_size` this small relative to the degree, and `domain8`'s own
8-slot domains are not. This is the same mechanism, precisely, that
produces Task 44's own reconstructed `10.2228ms`/fully-packed
result — it is not a coincidence, it is the same computation, now
explained rather than merely re-run.

**Conclusion: no shared root cause.** `packed()`'s own rank-offset
defect is real (§2 below) but conditional, and the condition is not met
by `domain8` at `ep=4` — under either today's code or the code at Task
33's own original commit (both confirmed identical), first-fit keeps
`ep=4` packed, never split. Task 33's own `14.4434ms`/"split" citation
remains unexplained by every mechanism checked so far, across both
tasks; Task 44's own "likely a stale or incorrect citation, not a
code regression" reading is *strengthened*, not resolved, by ruling out
this additional, independently plausible candidate. (Note for
completeness, not relevant to the citation in question: the same
condition *would* trip at `ep=8` on this fabric — three base ranks plus
seven extra EP ranks need 10 of domain 0's 8 slots, forcing two into
domain 1 — but Task 33's own sweep never tested `ep=8`.)

## 2. What shape does `packed()` actually produce?

### 2.1 Intended vs. produced — re-measured directly, not re-read

**`packed()` itself**, on Task 32's own fabric (`build_node_scale(5, 4)`,
domain size 4):

| tp | intended | produced (measured) | single-domain? |
|---|---|---|---|
| 2 | single domain | `(2,)` | **yes** |
| 4 | single domain | `(3, 1)` | **no** |
| 8 | single domain | `(4, 3, 1)` | **no** |

**Same policy, larger domains** (`build_node_scale(8, 16)`, domain size
16 — the fabric Tasks 19/20/23/24/28/31 actually use):

| tp | intended | produced (measured) | single-domain? |
|---|---|---|---|
| 2 | single domain | `(2,)` | yes |
| 4 | single domain | `(4,)` | **yes** |
| 8 | single domain | `(8,)` | **yes** |

`packed()` reaches single-domain at every degree tested once the domain
is large enough — confirming the defect is conditional on domain size
relative to degree, not universal.

**But Tasks 19/20/23/24/28/31 never call `packed()` at all.** Grepped
directly: `packed(` appears only in `tools/run_placement_search.py`
(Task 32), `tools/planner_core.py` (Task 33's generalization, and every
task built on it since — §4), `src/engine/cli/place.py` (a standalone
inspection CLI, not used by any study), and `src/engine/placement/placement.py`
(the definition). Tasks 19–31 instead call `run_tp_domain_probe._placement`,
a hand-built `explicit()` mapping (read directly, `tools/run_tp_domain_probe.py:103-128`):
PREFILL → `(0,0)`, FFN → `(0,1)`, every DECODE_ATTN rank of a given
pipeline stage → `(1+stage, i)` — genuinely single-domain for ATTN's
own TP group at every tested degree, by construction, confirmed by
direct read of the function body (not the policy this audit is about).

### 2.2 The mechanism, confirmed by reading `packed()` and by testing the read

```python
def packed(deployment, fabric):
    p = Placement(fabric, policy="packed")
    pool = []
    for d in sorted(fabric.domains):
        pool.extend(sorted(fabric.domains[d].members))
    ...
    it = iter(pool)
    for rank in deployment.ranks:
        p.assign(rank, next(it))
```

One global, domain-ordered GPU pool; `deployment.ranks` (confirmed from
`Deployment.ranks`, `src/engine/logical/deployment.py:109-110`: a flat
concatenation of each replica's own ranks, in `.add()` order) is
assigned into it sequentially, with no notion of "start this group at a
domain boundary." Every affected tool adds `PREFILL` first, so its one
rank always consumes global slot 0, offsetting `DECODE_ATTN`'s own TP
group by one.

**Tested directly, not just read**: rebuilding the same deployment with
`DECODE_ATTN` added *before* `PREFILL` removes the offset —
`packed()` then gives `(4,)` at `tp=4` on the 4-GPU-domain fabric,
not `(3,1)`. The trigger condition is `(ranks already in the pool) +
degree > domain_size`; this project's own universal convention of
adding PREFILL first makes that offset exactly 1, always. **Inherent
to the policy** (any preceding non-empty pool offsets the next group by
however many ranks came first) **applied to deployments whose domain
size happens not to absorb that offset** — not a policy that "never
packs," one that packs correctly whenever `1 + degree <= domain_size`
and does not otherwise.

## 3. Quantifying the real difference: M2N colocation, not TP-splitting

Task 24/28's own tp=2/4/8 scenario (`run_memory_planner_study.py`'s own
tool, its own fabric and config, `margin=0.9843`), re-run three ways —
its own "dedicated" reference, a hand-built fully-colocated placement
(PREFILL+ATTN+FFN all in domain 0), and its own "split" — **one
subprocess per scenario** (Task 41's own established discipline: several
`Simulator.run()` calls in one process leak global state and produce a
false crash; the first attempt at this re-run hit exactly that false
crash, caught by re-running cleanly rather than trusted):

| tp | dedicated (Task 24/28's own figure) | fully colocated | gap | `tp_comm` equal? |
|---|---|---|---|---|
| 2 | **13.953915** ms | **11.680315** ms | **2.273600** ms | **yes**, bit-identical (7.825920 ms both) |
| 4 | **14.430515** ms | **12.156915** ms | **2.273600** ms | **yes** (22.533120 ms both) |
| 8 | **15.603975** ms | **13.330375** ms | **2.273600** ms | **yes** (51.502080 ms both) |

Every one of these six figures matches the pre-existing report's own
table to the sixth decimal place — an independent re-run (fresh
subprocess calls, this session), not a re-read. **The gap is a constant
2.2736ms at every degree — the fixed cost of one cross-domain M2N hop
for this fabric and payload — and `tensor_parallel_communication_time`
does not move at all between "dedicated" and "fully colocated."** This
is not a TP-communication effect; it is exactly what `run_memory_tp_study.py`'s
own docstring already says it holds fixed by design, to isolate the TP
axis specifically.

**Split figures, for completeness** (also reproduced exactly): tp=2
`18.317755ms`, tp=4 `27.246515ms`, tp=8 `45.185415ms` — matching Task
24's own cited split figures and (at tp=4/8) Task 32's own explicit-
fallback figures to four decimal places.

**No committed figure needs correcting.** Task 23/24/28/31 measure
exactly what their own docstrings say they measure (TP-split penalty,
M2N colocation deliberately held fixed); Task 32's own winner at tp=2
(11.6803ms) already *is* the fully-colocated figure, because tp=2's
four total ranks exactly fill one 4-GPU domain on its own fabric — no
better placement exists for its own search to have missed.

## 4. Blast radius, updated through Task 45

Every tool the pre-existing report checked (Tasks 19–31: hand-built
`explicit()`, never `packed()` — unaffected) is unchanged by this
review. Extended to what exists now that did not when that report was
written:

| Study | Calls `packed()`? | Genuinely single-domain where it matters? | Conclusion affected? |
|---|---|---|---|
| 19–31 | No (hand-built `explicit()`) | Yes, at every tested degree | No |
| **32** | Yes, via `run_placement_search.py` | tp=2 natural; tp=4 only via its own explicit "packed-if-it-fits" fallback; tp=8 unreachable single-domain on this fabric regardless | No — self-corrected |
| **33/36/40/41** (`tools/planner_core.py`'s `enumerate_attn_shapes`) | Yes, same fallback, copied from 32 | Same as 32 | No |
| **41** (`enumerate_replica_arrangements`) | Yes, **no** explicit fallback (Task 41's own S1.1: "a real undercount... noted rather than silently accepted") | Only at `attn_tp<=domain_size` naturally; the `(4,)`/`(8,)` single-domain arrangement can be missing from *enumeration* at higher degrees | No — Task 41's own real study fixed `attn_tp=2`, where `packed()` reaches single-domain without the fallback (`1+2<=4`); the gap is in a *different* row's own enumeration completeness, not in any priced figure |
| **44** (`enumerate_joint_arrangements`) | Yes, own explicit fallback extended to both TP and EP groups (Task 44's own construction) | Same protection as 32/33 | No — and §1 above rules out this task's own mechanism as the cause of Task 44's separate, still-open `ep=4` citation gap |
| **45** (`plan_two_stage`) | Indirectly, via the same `enumerate_*` functions | Same | No — no new placement enumeration of its own |

**Every study is unchanged.** None depended on `packed()`'s own defect;
none mislabelled a split arrangement as packed. The one real gap
(`enumerate_replica_arrangements` lacking the explicit fallback) was
already known and reported by Task 41 itself, does not touch any figure
Task 41 actually measured, and remains exactly as it was — this audit
does not find a new instance of it.

## 5. Is Task 24-vs-32's `tp=4` agreement independent corroboration, or a shared error?

**Neither.** Confirmed directly (§3): Task 24's own "dedicated" tp=4
figure (14.430515ms, hand-built via `explicit()`, never touching
`packed()`) and Task 32's own tp=4 `(4,)` entry (built via its own
explicit "packed-if-it-fits" fallback, since raw `packed()` gives
`(3,1)` here, not `(4,)`) report the same number because **both are the
identical hand-built geometry** — ATTN alone in one domain,
PREFILL+FFN colocated in a different one — arrived at independently
(Task 24 by original design; Task 32 by a workaround built for an
unrelated reason, that `packed()` alone didn't reach single-domain).
They do not share `packed()`'s own bug (Task 24 never calls it), and
they are not corroboration of a fact neither tool could get wrong
either — each was checked separately, here, against a third,
genuinely-different construction (full colocation, §3) that shares no
mechanism with either. The agreement is real; it should not be read as
carrying more evidential weight than either figure does on its own.

## 6. What fixing `packed()` would require

Its own docstring's contract ("fill each scale-up domain completely
before moving to the next; keeps TP groups inside one domain wherever
they fit") does not hold for any group after the first rank in
`deployment.ranks`. Two honest, distinct fixes:

- **Narrow the contract**: document that single-domain placement is
  only guaranteed for the first group in rank order, and that every
  caller must check `group_shape()` before trusting "packed" to mean
  single-domain — exactly what Task 32/33/44's own explicit fallback
  already does in practice. No code change.
- **Widen the implementation**: reserve domain-aligned blocks per
  parallel group (e.g., largest-first) instead of filling one blind
  sequential pool. This would change `packed()`'s own output for every
  deployment with more than one multi-rank group where domain size does
  not evenly divide every group's degree — meaning **every study that
  calls it (32, 33, 36, 40, 41's TP axis, 44, 45) would need
  re-running**, since the shapes it produces (and anything keyed to
  them) would change. Given §3's own finding that no committed figure
  is currently wrong, this would be a robustness improvement for future
  callers, not a correction to the existing record.

## 7. Anywhere this specification is wrong

Re-verified, not merely re-read from the pre-existing branch — every
one of the following was checked again directly against source/git this
session:

1. **The central hypothesis does not hold.** Tasks 24/28 never call
  `packed()`; their own reference is genuinely single-domain for ATTN's
  TP group at every tested degree (§2.1, §3). The real 13.9539-vs-
  11.6803ms gap is M2N colocation distance, confirmed to the sixth
  decimal, not TP-group splitting.
2. **"Task 21 measured the tensor-parallel split penalty as +88.3%" is
  imprecise.** Checked directly: `docs/tasks/20-collective-backend-report.md`
  line 177 is where `5.803319ms`/`10.929719ms`/`+5.126400ms` first
  appear as computed figures; Task 21's own report (lines 78, 189)
  explicitly cites and re-confirms them after an unrelated fix. Task
  21's own tool (`run_collective_backend_ep_study.py`) defaults
  `attn_tp=1` and is never called with any other value in its own
  report — confirmed by grep, not merely by the earlier branch's
  citation of it.
3. **"A reference arrangement used across several studies" overstates
  `packed()`'s own usage** at the time it was written (one committed
  tool) — and even now, extended through Task 45 (§4), it is still only
  Tasks 32/33/36/40/41/44/45, all descended from one another, not
  several independent studies.
4. **The "about sixteen percent" estimate** is close in magnitude to
  the real 17.1–19.5% (depending on degree) but for a different
  mechanism — deliberate domain separation by design, not accidental
  TP splitting.

Nothing else in this specification (branching, not fixing `packed()`,
checking Task 33's exposure at the time, the acceptance commands)
required correction.

## What shipped

No source changes — an investigation and measurement task, per its own
acceptance criteria. `docs/tasks/34-packed-audit-report.md` only,
independently re-deriving every load-bearing claim in the pre-existing
`task-34-packed-audit` branch (a different, un-ancestral branch of the
same name, verified rather than trusted or blindly redone) via fresh
source reads and fresh Frontier runs, and additionally answering
whether Task 44's own open `ep=4` citation discrepancy shares this root
cause (no — ruled out precisely, by domain-slot arithmetic, not
assumption) and extending the blast-radius table through Task 45.
`packed()` was not modified.

One commit on `task-34-packed-audit-verify`, branched from
`task-45-regime`'s tip. 240 tests pass, unchanged;
`check_import_direction.py` exits 0.
