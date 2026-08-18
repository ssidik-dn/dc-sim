# Task 03 — Correct the Clos blueprint

Replace the pod-and-leaf-mesh construction from Task 02 with a proper two-tier
leaf-spine fabric.

Branch: `git checkout -b task-03-leaf-spine`. Do not merge to main.

Read `AGENTS.md`, then Task 02's report. This task exists because that report
correctly identified a specification error, and the diagnosis in it was right.

---

## 1. What went wrong, and whose fault it was

Task 02 §B.2 asked for a "leaf and spine" topology at `depth=2`, then supplied
these counts:

```
pods            = k
leaves per pod  = k/2
spines          = (k/2)^2
hosts per leaf  = k/2
total hosts     = k^3/4
```

Those are the **edge and core switch counts of a three-tier Al-Fares fat tree,
with the aggregation tier deleted.** `spines = (k/2)^2` is a core count.
`leaves per pod = k/2` is an edge count. The aggregation tier is what bridges
them, and removing it disconnects the topology: a leaf has `k/2` uplinks but
there are `k^2/4` spines, so it reaches only a `k/2` subset, and leaves at
different pod positions land on disjoint subsets with no route between them.

The Task 02 report found this by construction and repaired it with intra-pod
leaf-to-leaf links. That was a reasonable repair for the spec as written. The
specification was wrong, not the implementation.

**The full leaf-to-spine mesh was rejected in Task 02 as costing `k^4/8` links.
That figure followed from the inflated spine count. With the correct one the
mesh costs `k^2/2` links — it was always the cheap answer.**

---

## 2. The correct construction

A two-tier leaf-spine has **no pods**. With radix-`k` switches:

```
leaf switch:  k ports = k/2 down to hosts + k/2 up to spines
spine switch: k ports = k down to leaves

spines            = k/2
leaves            = k
hosts per leaf    = k/2
total hosts       = k * (k/2) = k^2/2
leaf-spine links  = leaves * spines = k^2/2      (full mesh)
```

Check: each spine has `k` ports and `k` leaves attach, so every spine port is
used. Every leaf reaches every spine directly. Any host reaches any other in at
most two hops (leaf, spine, leaf), and hosts on the same leaf in one hop.

Worked examples:

| k | spines | leaves | hosts | leaf-spine links |
|---|---|---|---|---|
| 4 | 2 | 4 | 8 | 8 |
| 8 | 4 | 8 | 32 | 32 |
| 16 | 8 | 16 | 128 | 128 |

---

## 3. What to change

`src/engine/infragraph/blueprints.py`:

```python
def clos_fat_tree_fabric(
    switch_radix: int,
    depth: int = 2,
    gpus_per_machine: int = 1,
    nics_per_machine: int = 1,
    oversubscription: float = 1.0,
    ...bandwidth and latency parameters unchanged...,
    name: str = "leaf-spine",
) -> Fabric
```

- **Remove pods entirely.** No `pods` variable, no pod indices in switch names.
- **Remove the intra-pod leaf mesh.** It was a workaround for the broken spec
  and it must not survive: it gives same-leaf-to-different-leaf traffic a
  bypass around the uplinks, which under-reports oversubscription pressure for
  exactly the traffic the parameter exists to measure.
- **Wire a full leaf-to-spine mesh**: every leaf to every spine, one link each.
- **Keep `depth=1` delegating to `single_tier_fabric`**, and keep
  `num_machines = switch_radix` for it — with no uplink tier every port serves a
  host, so that mapping is correct. Task 02 flagged this as a guess; it was the
  right one.
- **Keep `depth > 2` raising `NotImplementedError`.** A three-tier fat tree
  needs the aggregation tier and different formulas. If it is ever wanted it
  gets its own task, with the counts derived rather than supplied.
- **Machines attach to leaves**, `k/2` machines per leaf, each with
  `gpus_per_machine` GPUs and `nics_per_machine` NICs. Every NIC of a machine
  attaches to that machine's leaf.
- **Scale-up domains stay per-machine**, as in `single_tier_fabric`.

`oversubscription` still scales leaf-to-spine capacity relative to the aggregate
host-to-leaf capacity of that leaf. At `4.0` the uplinks carry a quarter of what
the downlinks could deliver in aggregate. Unchanged in meaning from Task 02.

---

## 4. Tests

Rewrite `tests/test_blueprints.py`'s Clos section. Delete tests that assumed
pods — do not adapt them, since the concept is gone.

| Test | Asserts |
|---|---|
| `test_leaf_spine_counts_match_formula` | For `k` in 4, 8, 16: spines `k/2`, leaves `k`, hosts `k^2/2` |
| `test_every_leaf_connects_to_every_spine` | Leaf-to-spine link count is exactly `k^2/2` in each direction, and every (leaf, spine) pair has a link |
| `test_no_leaf_to_leaf_links` | **The regression guard.** No link joins two switches of the same tier. This is what stops the Task 02 workaround reappearing. |
| `test_same_leaf_path_is_one_hop` | Two hosts on one leaf: the path traverses that leaf and no spine |
| `test_cross_leaf_path_traverses_a_spine` | Two hosts on different leaves: the path includes a spine. Under the Task 02 topology this failed for same-pod pairs, which is why it is named separately from the same-leaf case |
| `test_every_host_reaches_every_other` | `fabric.path()` succeeds for a sample spanning same-leaf, adjacent-leaf, and furthest-leaf pairs |
| `test_oversubscription_reduces_uplink_capacity` | At 4:1, leaf-to-spine capacity is a quarter of that leaf's aggregate host-to-leaf capacity |
| `test_oversubscription_shows_up_in_contention` | Same fabric at 1:1 and 4:1, enough concurrent cross-leaf transfers to saturate uplinks; the 4:1 makespan is strictly longer. **Report both numbers and the ratio.** |
| `test_single_tier_matches_build_node_scale` | Unchanged from Task 02, must still pass |
| `test_blueprint_fabrics_round_trip` | Both blueprints survive InfraGraph emit and parse |
| `test_depth_three_raises` | `NotImplementedError` |

---

## 5. Expect the oversubscription ratio to change, and say so

Task 02 measured exactly 4.00× at 4:1. That was clean because it was degenerate:
`Fabric.path()` is plain BFS and picks one path, so two flows from the same leaf
both took the *same* uplink even though several existed, and fully shared one
link.

With a full mesh there are `k/2` equal-cost uplinks from every leaf, so BFS
concentrating all flows onto one is now a larger distortion. **The ratio may
well not be 4.00× any more, and that is not a failure.** Report the numbers and
say what you think explains them.

Do **not** attempt to fix path selection in this task. That is Task 04 — making
`fabric_mode` real, with `single_path`, `per_flow_ecmp`, and `sprayed` — and it
touches code shared by everything built on `Fabric`. Note anything you observe
about it; do not act on it.

---

## 6. Known traps

Everything from Task 01 §5 and Task 02 §3 still applies. Additionally:

**`add_link()` adds a reverse link by default.** A full mesh wired with explicit
both-direction calls gives `k^2` links where `k^2/2` belong.
`test_every_leaf_connects_to_every_spine` must assert the count, not just
existence.

**Oversubscription is relative to aggregate downlink capacity**, not to a single
downlink. At 4:1 with `k/2` machines per leaf, the uplink budget is
`(k/2 * nics_per_machine * scale_out_GBps) / 4`, divided across the `k/2`
uplinks. Getting this backwards yields a fabric oversubscribed in the wrong
direction that still looks plausible.

**Do not preserve pod naming.** Switch names should be `leaf.<i>.asic.0` and
`spine.<j>.asic.0`. Leaving pod indices in the names would keep a concept that no
longer exists and confuse the InfraGraph round-trip.

---

## 7. Acceptance criteria

```bash
python3 -m pytest -q                      # everything outside the Clos section unchanged
python3 tools/check_import_direction.py   # exits 0
```

All tests in §4 by name. Do not modify `src/engine/network/`,
`src/engine/cost/`, `src/engine/physical/topology.py`, or `upstream/`.

`src/engine/physical/topology.py` is explicitly off-limits this time, because
the ECMP question lives there and belongs to Task 04.

---

## 8. What to report back

Same format as before. Specifically:

1. **The oversubscription makespan numbers at 1:1 and 4:1, and the ratio.** With
   your explanation of why it is what it is.
2. **Whether removing the intra-pod leaf mesh broke anything** that Task 02's
   tests were relying on, and if so what.
3. **Anything you observe about path selection** while working — you are not
   fixing it, but Task 04 will be written from what you find here.
4. **Any place this specification is wrong.** Task 02's report found the error
   that produced this task. That was the most valuable thing in it.
