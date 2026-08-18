# Task 01 — InfraGraph emitter, parser, and round-trip

Implement serialisation of the engine's fabric graph to and from InfraGraph, the
topology representation introduced by ASTRA-sim 3.0.

Work on a branch: `git checkout -b task-01-infragraph`. Do not merge to main.

---

## 1. Why this exists

This project combines two simulators. Frontier models LLM serving but has no
concept of physical machines or network fabric. ASTRA-sim models fabric and
collective cost but has no concept of requests or serving metrics. This
repository supplies the piece neither has: a placement map from logical ranks to
physical GPUs, over a fabric graph, plus a contention model.

The fabric graph currently exists only as Python objects. InfraGraph gives it a
serialised form that other tools can consume, and it is the migration path to
ASTRA-sim 3.0 when that is released. Adopting it now costs little because some
format is needed regardless.

Read `AGENTS.md` first. The rule that must not break: `src/engine/` never
imports from `src/integration/` or `upstream/`. `tools/check_import_direction.py`
enforces it and runs in CI.

---

## 2. What already exists

- `src/engine/physical/topology.py` — `Fabric`, `GpuId`, `NicId`, `SwitchId`,
  `Machine`, `ScaleUpDomain`, `Link`, `LinkClass`
- `src/engine/physical/builders.py` — `build_node_scale`, `build_rack_scale`
- `src/engine/placement/placement.py` — placement map and policies
- `src/engine/cost/astra_config.py` — collapses a placement into ASTRA-sim's
  per-dimension topology description
- `src/engine/network/` — flow model, max-min fair-share allocator
- 66 tests passing. `python3 -m pytest -q`

Read `topology.py` before starting. In particular note that **three link classes
are distinguished and must survive the round-trip**:

```
SCALE_UP    GPU <-> GPU over the memory-semantic fabric (NVLink/xGMI/UALink)
EGRESS      GPU  -> NIC; shared between GPUs, ~10x narrower than scale-up
SCALE_OUT   NIC <-> switch
```

Conflating egress with scale-out or with scale-up would silently destroy the
model's main capability. A scale-up domain is **not** the same as a machine: on
rack-scale platforms one domain spans 18 machines. Both boundaries must be
independently recoverable from the serialised form.

---

## 3. Scope

Three deliverables, in this order.

### 3.1 Emitter — `src/engine/infragraph/emit.py`

```python
def to_infragraph(fabric: Fabric) -> dict
def write_infragraph(fabric: Fabric, path: Path) -> None
```

### 3.2 Validator — `src/engine/infragraph/validate.py`

```python
class InfraGraphError(ValueError): ...
def validate_infragraph(doc: dict) -> None
```

Must reject: unknown `schema_version`; an edge referencing a node that does not
exist; a node whose name violates the naming convention; a negative or zero
bandwidth; a duplicate node name; a missing required field.

### 3.3 Parser and round-trip — `src/engine/infragraph/parse.py`

```python
def from_infragraph(doc: dict) -> Fabric
def read_infragraph(path: Path) -> Fabric
```

Round-trip must be lossless for everything the fabric model represents:
machines, GPUs, NICs, switches, scale-up domain membership, GPU-to-NIC binding,
every link with its class, capacity, and latency.

Do **not** implement: the ClosFatTreeFabric or SingleTierFabric blueprints, the
translator to ns-3 or HTSim configs, or the visualiser. Those are later tasks.

---

## 4. The schema

**Important caveat.** The ASTRA-sim 3.0 paper (arXiv 2606.10440) describes
InfraGraph as a conceptual model with a Python API — it does not publish a file
format. What follows is *our serialisation* of that model, not a standard. So:

- put `"schema_version": "0.1-dcsim"` in every document
- keep the emitter behind the functions above so the format can change
- when real InfraGraph tooling becomes available, expect to revise this and
  treat divergence as expected rather than as a bug

From the paper, these properties are load-bearing and should be honoured:

- a **directed attributed graph**: vertices are hardware components (GPUs, NICs,
  switches, storage), edges are connections between them, and annotations carry
  properties such as bandwidth
- **hierarchical component naming** of the form
  `<device-instance>.<index>.<component>.<index>`
- **edges as tuples** of source node, destination node, and connecting link type

Target document shape:

```json
{
  "schema_version": "0.1-dcsim",
  "name": "node-scale",
  "devices": [
    {
      "instance": "machine",
      "index": 0,
      "components": [
        {"component": "gpu", "index": 0, "attrs": {"scale_up_domain": 0}},
        {"component": "gpu", "index": 1, "attrs": {"scale_up_domain": 0}},
        {"component": "nic", "index": 0, "attrs": {}}
      ]
    },
    {
      "instance": "leaf",
      "index": 0,
      "components": [{"component": "asic", "index": 0, "attrs": {}}]
    }
  ],
  "domains": [
    {"domain_id": 0, "members": ["machine.0.gpu.0", "machine.0.gpu.1"]}
  ],
  "edges": [
    {
      "src": "machine.0.gpu.0",
      "dst": "machine.0.gpu.1",
      "link_type": "scale_up",
      "attrs": {"bandwidth_GBps": 400.0, "latency_ns": 936.25}
    },
    {
      "src": "machine.0.gpu.0",
      "dst": "machine.0.nic.0",
      "link_type": "egress",
      "attrs": {"bandwidth_GBps": 50.0, "latency_ns": 2000.0}
    }
  ]
}
```

Design notes, decided — do not vary from these without saying so in your report:

- **Switches are devices, not a special case.** A `SwitchId(tier="leaf", index=0)`
  becomes device instance `leaf` index `0` with a single `asic` component, so its
  node name is `leaf.0.asic.0`. Round-tripping must recover the original tier
  string and index.
- **Scale-up domain membership is recorded twice**: as a `domains` list, and as a
  `scale_up_domain` attribute on each GPU component. Redundant on purpose — the
  validator must check they agree, because that redundancy is what catches a
  malformed document.
- **GPU-to-NIC binding is implied by the egress edges**, not stored separately.
  Reconstruct `Fabric.bind_nic()` from them.
- **Edges are directed.** `Fabric.add_link()` adds a reverse link by default, so
  a naive emit will produce both directions. Emit both; on parse, do not
  double-add. Getting this wrong doubles the link count on every round-trip, so
  assert the count explicitly.

---

## 5. Known traps

Each of these already produced a wrong answer in this project. They are not
hypothetical.

**Silent format mismatch is worse than a crash.** The ASTRA-sim system config
declares one collective algorithm per topology dimension and must agree with the
network config on dimension count. A mismatch does not error — it silently drops
dimensions and returns a *lower* cost than the truth, which looks entirely
plausible and points the wrong way. It cost a day. Apply the lesson here: the
validator should reject anything ambiguous rather than guessing, and the parser
should raise rather than filling in defaults.

**Bidirectional link double-counting.** See the design note above. A round-trip
that grows the link count each time will still pass a naive "did it parse" test.

**Units.** Bandwidth is GB/s everywhere, matching ASTRA-sim. NIC vendors quote
Gb/s; `gbps_to_GBps()` exists for the boundary. Do not divide by 8 inline. Mixing
them silently is an easy way to be wrong by a factor of 8.

**Do not reorder or normalise away information.** `_representative_bandwidth` in
`astra_config.py` deliberately takes the *slowest* link of a class, because a
collective is paced by its worst hop. Nothing in the emitter should collapse
per-link detail — that detail is the entire reason this project exists, since
ASTRA-sim's own topology format has no individual links and therefore cannot
attribute contention to one.

---

## 6. Acceptance criteria

All of these must hold. They are the definition of done.

```bash
python3 -m pytest -q                      # all pre-existing 66 tests still pass
python3 tools/check_import_direction.py   # exits 0
```

Write `tests/test_infragraph.py` containing at least these, by name:

| Test | Asserts |
|---|---|
| `test_round_trip_node_scale` | `build_node_scale(num_machines=2)` emits, parses, and yields a fabric with identical GPU set, link count, and per-class link counts |
| `test_round_trip_rack_scale` | Same for `build_rack_scale(num_racks=1)` — 72 GPUs, one domain spanning 18 machines |
| `test_round_trip_preserves_link_classes` | Per-class counts are identical before and after, for all three classes |
| `test_round_trip_preserves_domain_membership` | Every GPU's `domain_of()` is unchanged, and a rack-scale domain still spans 18 machines |
| `test_round_trip_preserves_nic_binding` | Every GPU's `nic_of()` is unchanged |
| `test_round_trip_preserves_capacities` | Every link's `capacity_GBps` and `latency_ns` survive exactly |
| `test_round_trip_is_idempotent` | Emitting, parsing, and emitting again gives a byte-identical document — this is what catches link doubling |
| `test_node_names_follow_convention` | Every node name matches `<instance>.<index>.<component>.<index>` |
| `test_validator_rejects_dangling_edge` | An edge to a nonexistent node raises `InfraGraphError` |
| `test_validator_rejects_unknown_schema_version` | Raises rather than attempting a parse |
| `test_validator_rejects_domain_disagreement` | `domains` list contradicting a GPU's `scale_up_domain` attribute raises |
| `test_validator_rejects_nonpositive_bandwidth` | Zero or negative capacity raises |
| `test_parsed_fabric_costs_identically` | Run `analyse()` from `engine.network.transfers` on a fabric and on its round-tripped copy; the makespan and per-transfer durations must match exactly. This is the test that matters most — it proves the round-trip preserved everything the model actually uses. |

Add a CLI at `src/engine/cli/infragraph.py`:

```bash
PYTHONPATH=src python3 -m engine.cli.infragraph --platform node --out /tmp/n.json
PYTHONPATH=src python3 -m engine.cli.infragraph --platform rack --out /tmp/r.json
```

It should emit, re-read, and print a comparison confirming the round-trip held.

---

## 7. Constraints

- Python 3.10, standard library only. No new dependencies.
- Type hints on public functions. Docstrings explaining *why*, not *what*.
- Follow the existing style: dataclasses, `from __future__ import annotations`,
  no clever metaprogramming.
- `src/engine/infragraph/` needs an `__init__.py`.
- Small commits, each with its test.
- Do not modify anything under `upstream/`, and do not touch
  `src/engine/network/` or `src/engine/cost/` — if you believe a change is
  needed there, say so in the report instead of making it.

---

## 8. What to report back

The diff is not enough on its own. Include:

1. **The actual test output** — `python3 -m pytest -q` in full, not a summary.
2. **Anything surprising.** A test that failed for a reason you did not expect,
   a place where this spec was ambiguous or wrong, a decision you had to make
   that was not specified. This is the most valuable part of the report. In this
   project the most informative event so far was a test failing at 13 when the
   spec author expected 15 — the implementation was right and the spec was
   wrong. Surface those; do not smooth them over.
3. **Anything you chose not to do**, and why.
4. **Whether any acceptance criterion could not be met**, stated plainly rather
   than worked around.

Do not report "implemented, all tests pass" without the above. A clean report
with no surprises on a task this size is itself a signal worth questioning.
