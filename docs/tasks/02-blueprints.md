# Task 02 — InfraGraph blueprints, and hardening the validator

Two pieces. Do the hardening first: it is a correction to work already merged,
and the blueprints will exercise it.

Branch: `git checkout -b task-02-blueprints`. Do not merge to main.

Read `AGENTS.md` and `docs/tasks/01-infragraph.md` first — this builds directly
on that work.

---

## Part A — Harden the validator

### The principle that was missing from Task 01

Task 01 §3.2 listed the rejections the validator must implement. That list was
incomplete, and following it literally was reasonable, so this is a correction
to the specification rather than to the implementation.

The principle it should have stated:

> **The validator defends against documents this project did not write.**

That is why `validate_infragraph()` is a separate function from
`from_infragraph()` — so a document can be checked *without* being parsed.
Reasoning from "our emitter cannot produce that" is the wrong test, because the
entire purpose of a serialisation format is that documents arrive from
elsewhere: hand-written, produced by other tooling, or emitted by a future
version of this code.

At present `validate_infragraph()` returns clean on documents that then fail to
parse. That inconsistency is the bug.

### A.1 Reject unknown link types

An edge whose `link_type` is not one of `scale_up`, `egress`, `scale_out` must
raise `InfraGraphError` from the validator. Currently it passes validation and
raises a bare `ValueError` from `LinkClass(...)` during parsing.

The parser should also raise `InfraGraphError` rather than a bare `ValueError`,
so callers have one exception type to catch.

### A.2 Reject duplicate edges

Two edges with the same `(src, dst)` pair must raise. `Fabric._links` is keyed
by that pair, so a duplicate would silently overwrite — one link's capacity
lost, no error, and a plausible wrong answer.

### A.3 Handle components with no incident edges

Task 01's report identified this honestly: the emitter derives the switch device
list by scanning link endpoints, so a switch with zero links would silently
vanish. `builders.py` cannot produce one, but a hand-written document can.

Pick one and implement it:

- **reject** — the validator raises on any declared component that appears in no
  edge, or
- **preserve** — the parser keeps it, and a round-trip test proves it survives

Preserving is better if it is cheap. Rejecting is acceptable if preserving would
require restructuring `Fabric`. **State which you chose and why in the report** —
this is a genuine design decision, not a detail.

### A.4 Tests

Add to `tests/test_infragraph.py`:

| Test | Asserts |
|---|---|
| `test_validator_rejects_unknown_link_type` | `link_type: "nvlink"` raises `InfraGraphError` |
| `test_validator_rejects_duplicate_edge` | The same `(src, dst)` twice raises |
| `test_parser_raises_infragraph_error_not_value_error` | A malformed document surfaces `InfraGraphError` from the parser too |
| `test_isolated_component` | Whichever of A.3 you chose, tested |
| `test_validator_catches_everything_the_parser_would` | **The binding test.** For a list of malformed documents, assert that if `from_infragraph()` raises, `validate_infragraph()` also raises. No document may pass validation and then fail to parse. |

That last test is the point of Part A. Write it so that adding a new parser
failure mode without a matching validator rule causes it to fail.

---

## Part B — Blueprints

InfraGraph ships composable blueprints that construct a fabric from a handful of
parameters instead of enumerating every link. Implement the two the ASTRA-sim 3.0
paper names.

New module: `src/engine/infragraph/blueprints.py`.

### B.1 SingleTierFabric

```python
def single_tier_fabric(
    num_machines: int,
    gpus_per_machine: int,
    nics_per_machine: int,
    scale_up_GBps: float, scale_up_latency_ns: float,
    nic_gbps: float, egress_latency_ns: float,
    scale_out_GBps: float, scale_out_latency_ns: float,
    name: str = "single-tier",
) -> Fabric
```

A flat single-switch-layer topology: every NIC attaches to one leaf switch. One
scale-up domain per machine.

This should reproduce `build_node_scale()` exactly. **Assert that in a test** —
if the blueprint and the hand-written builder disagree, one of them is wrong, and
finding out now is cheaper than later.

### B.2 ClosFatTreeFabric

```python
def clos_fat_tree_fabric(
    switch_radix: int,
    depth: int,
    gpus_per_machine: int,
    nics_per_machine: int,
    oversubscription: float = 1.0,
    ...bandwidth and latency parameters as above...,
    name: str = "clos",
) -> Fabric
```

Parameterised by switch port count and network depth, computing switch counts and
wiring every link per the standard Clos construction — that is the paper's
description, and it is the behaviour to implement.

For `depth=2` (leaf and spine), a radix-`k` fat tree gives:

```
pods            = k
leaves per pod  = k/2
spines          = (k/2)^2
hosts per leaf  = k/2
total hosts     = k^3/4
```

Start with `depth=2`. If `depth=1`, delegate to `single_tier_fabric`. **Raise
`NotImplementedError` for `depth > 2`** rather than approximating — a
three-tier Clos wired wrongly would produce plausible numbers, which is the
failure mode this project keeps encountering.

`oversubscription` scales leaf-to-spine capacity relative to host-to-leaf: at
`4.0`, uplinks carry a quarter of the aggregate downlink bandwidth. This is the
knob the placement experiments need, so it must be exercised by a test, not just
accepted as a parameter.

### B.3 Tests

Add `tests/test_blueprints.py`:

| Test | Asserts |
|---|---|
| `test_single_tier_matches_build_node_scale` | Identical GPU set, per-class link counts, and domain membership |
| `test_clos_host_count_matches_formula` | For `k` in 4, 6, 8: host count is `k^3/4` |
| `test_clos_switch_counts_match_formula` | Leaf and spine counts match the construction above |
| `test_clos_every_host_reaches_every_other` | `fabric.path()` succeeds between a sample of GPU pairs across different pods |
| `test_clos_cross_pod_path_traverses_spine` | A path between pods includes a spine switch; a path within one pod does not |
| `test_oversubscription_reduces_uplink_capacity` | At 4:1, leaf-to-spine capacity is a quarter of aggregate host-to-leaf |
| `test_oversubscription_shows_up_in_contention` | **The binding test.** Build the same fabric at 1:1 and 4:1. Run `analyse()` from `engine.network.transfers` with enough concurrent cross-pod transfers to saturate the uplinks. The 4:1 fabric must show a strictly longer makespan. A blueprint that accepts the parameter without it affecting cost is worse than useless. |
| `test_blueprint_fabrics_round_trip` | Both blueprints emit, parse, and compare identically through InfraGraph |
| `test_depth_three_raises` | `NotImplementedError`, not a wrong answer |

---

## 3. Known traps

Everything from Task 01 §5 still applies. Additionally:

**Link direction.** `Fabric.add_link()` adds a reverse link by default. When
wiring a Clos by hand it is easy to add both directions explicitly and end up
with four links where two belong. `test_clos_switch_counts_match_formula` will
not catch that — only an explicit link-count assertion will.

**Oversubscription is a ratio, not a capacity.** At 4:1 the uplink carries a
quarter of what the downlinks could deliver in aggregate, not a quarter of one
downlink's capacity. Getting this backwards produces a fabric that is
oversubscribed in the wrong direction and still looks reasonable.

**A blueprint that ignores a parameter is a silent failure.** `oversubscription`
must reach the link capacities and change measured behaviour. Hence the binding
test.

---

## 4. Acceptance criteria

```bash
python3 -m pytest -q                      # all 82 existing tests still pass
python3 tools/check_import_direction.py   # exits 0
```

All tests named in A.4 and B.3, by name. Do not modify `src/engine/network/`,
`src/engine/cost/`, or anything under `upstream/`; if you believe a change is
needed there, say so in the report rather than making it.

---

## 5. What to report back

Same format as Task 01 §8: full test output, anything surprising, anything you
chose not to do, and any acceptance criterion you could not meet.

Specifically for this task:

1. **Which A.3 option you chose, and why.**
2. **Whether `single_tier_fabric` really does reproduce `build_node_scale`.** If
   it does not, do not adjust the blueprint until you have worked out which one
   is wrong — the builder is not automatically correct.
3. **The actual makespan numbers** from
   `test_oversubscription_shows_up_in_contention`, at 1:1 and 4:1. Not just that
   the assertion passed. If the ratio is far from what 4:1 would suggest, say so;
   that would be interesting rather than a problem to hide.

The most useful thing you can report is a place where this specification is
wrong. Task 01's report identified the isolated-switch gap, which is why Part A
exists.
