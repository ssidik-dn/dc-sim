# AGENTS.md

Read this before changing anything.

## What this is

A topology-aware LLM serving simulator, built by combining two existing ones.
Frontier supplies the serving layer; ASTRA-sim supplies communication cost on a
given topology. This repository supplies the piece neither has: a placement map
binding logical ranks to physical GPUs, plus a contention model over a real
fabric graph.

## The one rule that must not break

```
src/engine/  must never import from  src/integration/  or  upstream/
```

Enforced by `tools/check_import_direction.py`, which runs in CI. The engine
answers placement and communication-cost questions standalone. The moment it
imports Frontier, the host choice stops being reversible and the engine cannot
be tested or published without it.

If you need engine code to react to something in Frontier, invert the
dependency: define the interface in the engine, implement it in integration.

## Layout

```
src/engine/          Phases 1-6. Standalone. No Frontier, no ASTRA-sim imports.
  physical/          machines, GPUs, NICs, scale-up domains, fabric graph
  logical/           pools, replicas, parallel groups
  placement/         placement map and policies
  fabric/            path and link-set computation
  cli/               inspection tools
src/integration/     Phases 3 and 7 ONLY. Frontier registries and replacements.
upstream/            Pinned dependencies. NEVER modified.
tools/               CI checks
tests/               pytest; run with `python3 -m pytest`
```

## Three link classes, not two

Communication inside a machine is two paths with different physics. Conflating
them is the most common way a model of this kind goes wrong.

```
GPU --scale-up--> GPU    NVLink/xGMI/UALink, memory-semantic, ~400-900 GB/s
GPU --egress----> NIC    PCIe or integrated; SHARED between GPUs, ~50 GB/s
NIC --scale-out-> switch Ethernet / InfiniBand
```

Egress is roughly an order of magnitude narrower than scale-up, every byte
leaving the machine crosses it, and it is shared. So egress capacity is a
property of a **GPU**, not of a machine: which NIC a GPU sits behind determines
its usable outbound bandwidth, and placement changes cost even within one
chassis.

A scale-up domain is **not** a machine. A rack-scale domain spans 18 trays.
Keep the two boundaries independent.

## Units

Bandwidth is GB/s everywhere, matching ASTRA-sim. NIC vendors quote Gb/s. Use
`gbps_to_GBps()` at the boundary; never divide by 8 inline. Mixing them
silently is an easy way to be wrong by 8x.

## Known traps

**ASTRA-sim config pairing.** The system config declares one collective
algorithm per topology dimension and must agree with the network config on
dimension count. A mismatch does not error -- it silently drops dimensions and
returns a plausible wrong answer. Measured: a 2-dim topology with a 1-dim
system config reported a *lower* cost than the packed baseline, which is the
opposite of the truth. Emit both files together, and assert the reported rank
count equals the placement size.

**Congestion-aware is 1-dim only.** ASTRA-sim's congestion-aware analytical
backend rejects multi-dimensional topologies. Multi-dimensional placement runs
congestion-unaware. The two capabilities never compose, and contention is ours
regardless.

**Frontier profiles.** Only `h800` and `rtx_pro_6000` carry full-feature
compute profiles; older device directories use a legacy format the predictor
rejects. The shipped examples default to a device lacking profiles for their
own model.

**Dummy mode.** Frontier examples default to `ENABLE_DUMMY_MODE=true` at a flat
1 ms per operator. Never calibrate or baseline against it.

## Zones

**Agent-safe** -- `physical/`, `logical/`, `placement/`, `cli/`, `tests/`,
`docs/`, `configs/`. Well-specified and heavily testable. Write freely, with
tests.

**Human-only** -- anything touching event semantics, time ownership, completion
revision, or upstream coupling. That means `fabric/` contention code when it
lands, and all of `src/integration/`. Agents may write tests here but not
implementations. The correct unit of work is one invariant plus its test.

## Invariants

- Placement is injective. Two ranks on one GPU is an error, not a warning.
- `group_shape()` is order-independent, so isomorphic placements collapse to
  one memoisation key. Do not make it order-sensitive.
- A split group touches strictly more links than a packed one, and the extra
  links are egress and scale-out.
- Fragmented placement is deterministic for a given seed.

## Before committing

```bash
python3 -m pytest -q
python3 tools/check_import_direction.py
```
