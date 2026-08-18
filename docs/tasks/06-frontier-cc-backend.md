# Task 06 — Register the engine as a Frontier cc_backend

The first connection between this engine and a serving simulator.

Branch: `git checkout -b task-06-frontier-cc-backend`. Do not merge to main.

Read `AGENTS.md` first, and note that this is the first task touching
`src/integration/`, which that file names as a human-only zone. That
designation stands, with one revision stated in §2 below.

---

## 1. Why this one is different

Tasks 01 to 05 were pure computation over graphs. Nothing they produced could
be wrong in a way that looked right — a disconnected topology raised, a
mismatched dimension count raised, a doubled link count failed an idempotence
test.

This task attaches to another simulator's execution. A mistake here does not
raise. It produces a plausible number: a time-to-first-token that is confidently
wrong by some factor nobody notices, because there is nothing to compare it
against unless you build the comparison deliberately.

That is why the acceptance criteria below are dominated by equivalence tests
rather than feature tests, and why the scope is deliberately small.

---

## 2. Zone rule, restated

`AGENTS.md` says agents may write tests but not implementations in the
human-only zone. That rule was written when the agent had less context. The
revision:

> The zone still means what it said. What changes is the **unit of work**: one
> invariant plus its test per commit, expected values computed by hand where
> arithmetic is involved, and no change larger than can be reviewed in one
> sitting. Do not batch several behaviours into one commit here.

If you find yourself writing more than about eighty lines of implementation
before a test, stop and split it.

---

## 3. What Frontier provides

Do not take this section on trust. Verify each claim against the pinned
checkout at `upstream/frontier` before building on it, and **report any
discrepancy** — an earlier task in this project was built on a description of an
interface rather than the interface, and the argument order was wrong.

- `frontier/cc_backend/base_cc_backend.py` defines `BaseCCBackend`, an abstract
  base with six operations: `predict_allreduce`, `predict_allgather`,
  `predict_broadcast`, `predict_send_recv`, `predict_reduce_scatter`,
  `predict_all_to_all`. Each returns a predicted time in **milliseconds**.
- Signatures are of the shape
  `predict_allreduce(self, data_size_bytes, num_devices, cluster_type=None, comm_domain=None) -> float`.
  Confirm the exact parameter names and defaults.
- `frontier/cc_backend/cc_backend_factory.py` holds `CCBackendFactory`, which
  selects an implementation by string. Five are registered, including
  `analytical` and `aiconfigurator`. **Read `aiconfigurator_cc_backend.py`** —
  it is the closest existing model for what this task builds.
- Selection is by the `--cc_backend_config_type` command-line flag.

Note what these signatures do **not** carry: any indication of *where* the
participating ranks are. They give a count and a parallelism domain. That
absence is the whole reason this project exists, and §5 is how we work around it
without modifying Frontier.

---

## 4. Scope

### 4.1 The backend — `src/integration/cc_backend/engine_backend.py`

A `BaseCCBackend` subclass that answers Frontier's six prediction calls using
this engine's cost path.

- Take a `Fabric`, a `Placement`, and a `CostBackend` at construction.
- Implement all six methods. Return milliseconds, because that is what Frontier
  expects; the engine works in nanoseconds, so convert in exactly one place and
  document the rounding.
- `predict_send_recv` maps to a point-to-point transfer. The other five map to
  collectives over the resolved participant set.

### 4.2 Resolving participants — `src/integration/cc_backend/comm_groups.py`

Frontier gives a count and a domain; the engine needs a rank set and a placement
shape. Bridge with a registry:

```python
class CommGroupRegistry:
    def register(self, cluster_type, comm_domain, num_devices, ranks): ...
    def resolve(self, cluster_type, comm_domain, num_devices) -> list[Rank]: ...
```

Populated from a `Deployment` before the run starts. When a lookup fails,
**raise** — do not fall back to assuming a packed placement. A silent fallback
would produce exactly the plausible-wrong-number failure this task is shaped to
avoid.

### 4.3 Registration — `src/integration/install/cc_backend.py`

Register the backend with `CCBackendFactory` under the name `dc_sim_engine`.
This should be the only place that touches Frontier's factory, and it must be
callable from the existing `install()` entry point.

Nothing under `upstream/` may be modified. If registration appears to require
it, stop and report — that would be a finding worth more than the feature.

---

## 5. Acceptance criteria

```bash
python3 -m pytest -q                      # all 121 existing tests still pass
python3 tools/check_import_direction.py   # exits 0
```

The import check matters more than usual here: `src/integration/` may import
from `src/engine/` and from `upstream/frontier`, but **`src/engine/` must not
gain any import in either direction**. If the check fails, the dependency has
been inverted and the engine is no longer portable.

### Tests — `tests/test_cc_backend_integration.py`

| Test | Asserts |
|---|---|
| `test_backend_subclasses_frontier_base` | The class is a genuine `BaseCCBackend` subclass and implements all six methods |
| `test_all_six_methods_return_milliseconds` | Each returns a positive float, and a value consistent with the engine's nanosecond figure divided by one million |
| `test_packed_placement_matches_analytical_within_bound` | **The binding test.** For a packed placement on a single-domain fabric, predictions must agree with Frontier's own `analytical` backend within a stated tolerance. State the tolerance you chose and why. This is the only check that the two are measuring the same thing at all. |
| `test_split_placement_costs_more_than_packed` | Same collective, same size, same device count, placement split across domains — must cost strictly more. Frontier's own backends cannot produce this difference, so it is the capability being added |
| `test_unresolvable_comm_group_raises` | An unregistered `(cluster_type, comm_domain, num_devices)` raises rather than guessing |
| `test_registration_is_idempotent` | Calling install twice does not double-register or raise |
| `test_engine_has_no_frontier_import` | Programmatic assertion, not just the CI script: no module under `src/engine/` imports Frontier |

### Do not attempt

- Wiring the KV transfer or M2N paths. Those are the hard integration and they
  need event-semantics changes.
- Running a full Frontier simulation end to end. If it happens to work, say so,
  but it is not the criterion.
- Any change to `src/engine/`. If the engine needs a new method to support this,
  **report it rather than adding it** — that is a design question about where the
  boundary sits.

---

## 6. Known traps

Everything from previous tasks still applies. Specifically for this one:

**Units, twice over.** Frontier's clock is float seconds; its collective
predictions are float milliseconds; the engine is integer nanoseconds. Convert in
one place, document the rounding direction, and add a round-trip test. Mixing
these silently is the single most likely way to be wrong by a factor of a
thousand while still producing believable numbers.

**A count is not a placement.** `num_devices=8` says nothing about where those
eight GPUs are. The registry exists precisely so that the answer comes from a
`Deployment` rather than from an assumption. Resist any temptation to infer.

**Read the interface, do not trust its description.** In an earlier task the
argument order of `route()` was taken from a report rather than from the source,
and every call site was wrong. §3 above is a description. Check it.

**Frontier is pre-release on its disaggregated paths.** If the pinned checkout
differs from §3, that is expected and is a finding, not a blocker.

---

## 7. What to report back

Same format as before. Specifically:

1. **Any discrepancy between §3 and the actual Frontier source**, with the real
   signatures.
2. **The tolerance chosen for the analytical-agreement test, and why** — and
   what the actual disagreement was. If the two backends diverge more than
   expected, that is interesting and should not be tuned away by widening the
   bound until it passes.
3. **Whether anything tempted you to modify `src/engine/`**, and what.
4. **Whether a full Frontier run was attempted**, and what happened.
5. **Any place this specification is wrong.** Every task so far has produced one;
   the leaf-spine error came from exactly this question.
