# Task 04 — fabric_mode: single_path, per_flow_ecmp, sprayed

Give `Fabric` a real, named path-selection policy instead of an accidental one.

Branch: `git checkout -b task-04-fabric-mode`. Do not merge to main.

Read `AGENTS.md`, then Task 03's report §4 ("Observations on path selection").
This task exists because that report found that `Fabric.path()`'s routing is
not just simple, it is silently *wrong* for any topology with more than one
equal-cost path.

---

## 1. Why this exists

Task 03 built a proper leaf-spine fabric with `k/2` parallel spine links
between every pair of leaves, specifically so a placement experiment could
observe oversubscription pressure changing measured contention. It measured
exactly 4.00x makespan degradation at 4:1 oversubscription in every
configuration tried, and traced why: `Fabric.path()` is plain BFS with a
traversal order fixed by link-insertion order and independent of which hosts
are asking. Every flow between the same pair of leaves — regardless of which
specific hosts originate it — resolves to the *identical* intermediate spine.
Adding more spines added capacity that measured traffic never touched.

This is not a Task 03 bug. It is the base `Fabric` model never having a
concept of "more than one reasonable path" in the first place — every prior
task inherited it. It matters now because Task 03 is the first topology where
multiple equal-cost paths are common rather than incidental.

---

## 2. Scope — read this before writing any code

Per `AGENTS.md`'s zones: `physical/` is agent-safe; the boundary is
`fabric/`'s *contention* code and anything touching **event semantics, time
ownership, or completion revision** — that is `network/model.py`'s
`FlowNetwork`, which owns submit/advance/completion and is human-only.

Those are two different questions, and this task answers only the first:

- **Which link(s) does a flow use, and with what weight?** A pure graph
  question — inputs are the fabric graph and a flow's endpoints, output is a
  path (or a set of paths with fractions). No event, no time, no completion.
  This is what this task implements, in `physical/topology.py`.
- **How does the contention model simulate one flow spread across several
  links concurrently, with one combined completion time?** That changes
  `FlowNetwork`'s submit/completion semantics — genuinely the human-only
  territory `AGENTS.md` describes. **Out of scope for this task.**

Concretely: `single_path` and `per_flow_ecmp` both resolve a flow to exactly
*one* path, so nothing about how `network/transfers.py` drives `FlowNetwork`
needs to change to use them — a caller just needs a different path for the
same flow. `sprayed` is different in kind, not degree: simulating it correctly
means one `Transfer` becomes several concurrent partial flows with a combined
completion, which is `FlowNetwork` territory.

So this task implements the *routing decision* for all three modes —
including computing `sprayed`'s path-and-fraction split, which is itself pure
graph computation — but does **not** wire `sprayed` into `network/transfers.py`
or change `FlowNetwork` to execute a multi-leg flow. That wiring is real,
valuable follow-on work (call it Task 05 if picked up), and is human-only per
`AGENTS.md`: agents may write the tests that specify what a spread transfer's
combined completion should look like, but not the implementation. Say so in
the report rather than guessing at it here.

**Do not modify `src/engine/network/` or `src/engine/cost/`.** Consistent
with every prior task.

---

## 3. What already exists

- `Fabric.path(a, b)` — single-path BFS, in `src/engine/physical/topology.py`.
  **Leave its implementation untouched.** Every existing test (100+, across
  tasks 01-03) depends on its exact behavior, and it is also what
  `single_path` mode must reproduce exactly.
- `Fabric.link_set(pairs)` — links touched by a set of GPU-to-GPU flows, built
  on top of `path()`.
- `network/transfers.py` — `Transfer`, `run_transfers()`, `analyse()`. Calls
  `fabric.path(t.src, t.dst)` today to get one transfer's link list.

---

## 4. What to build

All in `src/engine/physical/topology.py`, as additions — do not change
`path()`'s behavior or signature.

```python
class FabricMode(Enum):
    SINGLE_PATH = "single_path"
    PER_FLOW_ECMP = "per_flow_ecmp"
    SPRAYED = "sprayed"
```

```python
def equal_cost_paths(self, a: GpuId, b: GpuId) -> List[List[Link]]:
    """Every minimum-hop path from a to b (not just the first BFS finds).
    "Equal-cost" means equal hop count -- this model has never costed a path
    by link capacity, and this doesn't start now. Returns [[]] behaviour for
    a == b should match path()'s (empty list, no error). Order is
    deterministic for a fixed fabric so mode selection is reproducible."""

def route(self, mode: FabricMode, flow_key: str, a: GpuId, b: GpuId) -> List[Link]:
    """The path one flow should use under `mode`.

    SINGLE_PATH:     path(a, b), unchanged.
    PER_FLOW_ECMP:   one of equal_cost_paths(a, b), chosen by a stable hash
                     of flow_key (and the endpoints) -- NOT Python's builtin
                     hash(), which is salted per-process and would make
                     "same flow, same path" fail to hold across runs. Same
                     (mode, flow_key, a, b) must always choose the same path.
    SPRAYED:         raises. A single path is the wrong answer for spray
                     semantics -- see spray_routes() and §2 above. Raise
                     NotImplementedError with a message pointing at
                     spray_routes() and explaining that executing a spread
                     flow needs FlowNetwork changes this task doesn't make.
    """

def spray_routes(self, a: GpuId, b: GpuId) -> List[Tuple[List[Link], float]]:
    """Every equal-cost path paired with the fraction of a flow's bytes it
    would carry under SPRAYED semantics. Even split across N equal-cost
    paths (1/N each) -- this model has no reason yet to weight paths
    unevenly, since same-hop-count links in every fabric built so far carry
    equal capacity. Fractions sum to 1.0. Pure routing decision only: does
    NOT execute anything, and nothing downstream currently consumes this."""
```

---

## 5. Known traps

**Determinism.** `route()`'s per-flow choice must be reproducible across
processes and runs -- use `hashlib`, not `hash()`. This project has an
existing invariant that placement is "deterministic for a given seed"
(`AGENTS.md`); this is the same property for routing.

**`equal_cost_paths` must not silently degrade to one path.** The entire
point is finding *every* minimum-hop path. A leaf-spine fabric with `k/2`
spines between two cross-leaf hosts must yield `k/2` paths, not one --
verify the count explicitly, the same lesson as Task 01's bidirectional-link
count and Task 03's leaf-spine link count.

**Dispersion is a distribution property, not a per-call one.** A single
`per_flow_ecmp` call choosing spine 3 proves nothing; the test that matters
is many distinct flow keys between the same host pair collectively using
more than one path. Task 03's finding was exactly this failing under
`single_path` -- the fix must be shown working, not just asserted.

**`sprayed`'s even split is a stated assumption, not a derived one.** Nothing
in the current model would notice if it were wrong (see `spray_routes()`'s
docstring above) -- flag it in the report rather than treating it as
obviously correct.

---

## 6. Tests

Add `tests/test_fabric_mode.py`.

| Test | Asserts |
|---|---|
| `test_single_path_mode_matches_path` | `route(SINGLE_PATH, key, a, b) == fab.path(a, b)` for several pairs |
| `test_equal_cost_paths_count_matches_spine_count` | On a Task-03 leaf-spine fabric (`switch_radix=k`), a cross-leaf pair has exactly `k/2` equal-cost paths, each through a different spine |
| `test_equal_cost_paths_is_singular_when_unique` | A same-leaf pair has exactly one equal-cost path, matching `fab.path()` |
| `test_ecmp_route_is_one_valid_equal_cost_path` | `route(PER_FLOW_ECMP, ...)`'s result is a member of `equal_cost_paths(a, b)` |
| `test_ecmp_is_deterministic` | Same `(mode, flow_key, a, b)` called repeatedly (including via a fresh hash of the same inputs) always returns the same path |
| `test_ecmp_disperses_across_many_flows` | **The binding test.** Many distinct flow keys between the same cross-leaf host pair collectively touch more than one spine -- the direct fix for Task 03's finding |
| `test_ecmp_route_raises_for_sprayed_mode` | `route(SPRAYED, ...)` raises `NotImplementedError`, not a wrong single path |
| `test_spray_routes_covers_every_equal_cost_path` | `spray_routes(a, b)` has exactly one entry per path from `equal_cost_paths(a, b)` |
| `test_spray_routes_fractions_sum_to_one` | Fractions sum to `1.0` (within floating-point tolerance) |
| `test_spray_routes_is_even_split` | Each fraction is `1/N` for `N` equal-cost paths |

---

## 7. Acceptance criteria

```bash
python3 -m pytest -q                      # all existing tests unchanged and passing
python3 tools/check_import_direction.py   # exits 0
```

All tests in §6 by name. `Fabric.path()` unchanged. No edits to
`src/engine/network/`, `src/engine/cost/`, or `upstream/`.

---

## 8. What to report back

Same format as before. Specifically:

1. **Confirm `path()` is byte-for-byte unchanged** and that every pre-task-04
   test still passes without modification -- this task adds capability, it
   does not change default behavior.
2. **The dispersion evidence**: for a concrete leaf-spine fabric and a
   concrete set of flow keys, which spines got used under `single_path`
   versus `per_flow_ecmp`. Numbers, not just "the test passed."
3. **What `sprayed` would need in `network/`** to actually execute, now that
   you've built its routing decision. You are not implementing it -- but
   having built `spray_routes()`, you are the best-positioned person to say
   precisely what `FlowNetwork` would need to change, for whoever picks up
   that follow-on task.
4. **Any place this specification is wrong**, including if you think the
   agent-safe/human-only line drawn in §2 is in the wrong place.
