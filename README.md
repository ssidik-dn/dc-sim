# Topology-Aware LLM Serving Simulator

Combines Frontier (serving) and ASTRA-sim (communication cost) with the piece
neither has: a placement map from logical ranks to physical GPUs, over a fabric
graph that distinguishes three link classes.

## Quick start

```bash
python3 -m pytest -q                     # 66 tests
python3 tools/check_import_direction.py  # CI boundary check
PYTHONPATH=src python3 -m engine.cli.place --platform node --tp 8
PYTHONPATH=src python3 -m engine.cli.place --platform rack --tp 16 --ep 4
```

## Status

Implemented:
- physical inventory, three link classes, fabric graph with path and link-set
  computation
- three-pool logical model, placement map with four policies
- canonical group-shape signature for memoisation
- ASTRA-sim config emitter producing matched network/system pairs
- cost backend with memoisation, mock, and fallback for inexpressible shapes
- flow-level contention: max-min fair share, per-link queues, revisable
  completions, bottleneck attribution by link class

Not yet: InfraGraph emitter, Frontier integration.

## Why contention lives here

ASTRA-sim cannot supply it. Measured directly: dependency-free collectives in
one trace take exactly as long as explicitly chained ones, in every scheduling
policy and chunk setting tried. Its workload layer serialises independent
collectives, so no backend beneath it ever sees two flows at once.

The allocator is therefore tested against closed-form max-min fair share --
two flows sharing a link get half each, four flows through a 2:1 uplink get a
quarter each -- rather than against another simulator's output. Arithmetic
cannot be wrong.

## Two traps encoded as tests

ASTRA-sim's system config declares one collective algorithm per topology
dimension and must agree with the network config. A mismatch does not error --
it silently drops dimensions and returns a plausible wrong answer. Configs are
therefore emitted as a pair, by one function, and verified.

`npus_count` multiplies across dimensions, so only even splits are
expressible. A group spread 3+3+2 raises `NotExpressible` rather than being
rounded, because a rounded answer here looks entirely reasonable.

## Layout

See `AGENTS.md`. The rule that matters: `src/engine/` never imports from
`src/integration/` or `upstream/`.
