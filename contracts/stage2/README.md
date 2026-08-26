# Stage 2 Gate A: the planner ↔ real-runtime contract

This directory is the file-only boundary between this project's planner
(`tools/planner_core.py`, `tools/planner.py`, produced/consumed here by
`tools/stage2/`) and the separate `sim_real` project. Neither side
imports the other's Python. `sim_real` reads and writes the JSON files
this directory describes; it never imports `tools/stage2/`.

Full design rationale, the axis readiness table, and the answers to
questions A–G live in `docs/stage-2-gate-a-contract-report.md`. This
file is the short, operational reference: what each schema is, what
each example demonstrates, and how the version policy works.

## The four contract objects

| Object | Producer | Consumer | Schema |
|---|---|---|---|
| `DeploymentManifest` | planner (`tools/stage2/exporters.py`) | `sim_real` | `deployment_manifest.schema.json` |
| `PlannerPrediction` | planner | `DecisionValidation` step | `planner_prediction.schema.json` |
| `HardwareResult` | `sim_real` | `DecisionValidation` step | `hardware_result.schema.json` |
| `DecisionValidation` | comparison step (`tools/stage2/decision.py`) | report / human | `decision_validation.schema.json` |

`DeploymentManifest` is what to run, exactly, with no reinterpretation
by `sim_real` — every parallelism degree, every rank's host and
physical GPU, every workload and constraint value is explicit (see
`docs/stage-2-gate-a-contract-report.md` §1/§3). `PlannerPrediction`
carries what the planner predicted for the selected candidate, with
its own uncertainty and ranking context, not a single stripped-down
number. `HardwareResult` is only what `sim_real` can itself observe by
actually running something — no simulator-internal breakdown is
required, because a real launcher has no way to produce one.
`DecisionValidation` is the one comparison this whole contract exists
to produce: does the planner's choice match what real hardware
benchmarking would have picked, and can the two even be told apart at
the noise floor that configuration was actually measured at.

## Schema files

Generated directly from the real `tools/stage2/contracts.py` dataclasses
(via a one-off reflection script, not committed — the dataclasses are
the source of truth; these files cannot drift from them by
hand-transcription error). JSON Schema draft-07. Each top-level schema
inlines every nested type it needs under its own `$defs`.

## Version policy

Every one of the four objects carries its own top-level `*_version`
field (e.g. `DeploymentManifest.manifest_version`), currently `"1.0"`
for all four. **A major-version mismatch between a payload and the
code reading it is a hard reject — there is no migration path.** A
minor-version bump (new optional field, widened enum) is accepted
without any translation step. This is deliberate: this task's own
scope explicitly excludes over-engineering a migration story before a
single real manifest has ever been executed by `sim_real`. See
`tools/stage2/serialization.check_major_version`.

## Examples

Every `*_manifest.json` / `*_prediction.json` pair under `examples/`
was produced by an actual `SimulationEvaluator` run against real
Frontier compute profiles (h800, Phi-tiny-MoE-instruct) — the
`mean_tpot_ms`/`throughput_rps`/`slo_attainment` values in them are
real simulator output, not invented. The `HardwareResult` and
`DecisionValidation` examples are the one place this project has no
choice but to depart from that rule: **no GPU is reachable from this
sandbox** (the same constraint task 55 operated under), so
`clean_hardware_result.json`, `contended_hardware_result.json`, and
`decision_validation_example.json` are explicitly, visibly synthetic —
every provenance/system field that would otherwise carry a real value
instead carries the literal string `SYNTHETIC-ILLUSTRATIVE-NOT-REAL-HARDWARE`,
so nothing here can be mistaken for a real measurement if it is ever
read out of context.

- `single_host_tp2_{manifest,prediction}.json` — TP=2, one host, no
  expert-parallel group (`relative=None`). The single-node shape §17
  asks this contract to be able to express, mirroring Stage 1B's own
  verified single-node TP=1/2/4 finding.
- `two_host_tp4_{manifest,prediction}.json` — TP=4 split 2+2 across two
  hosts (`attn_shape=(2,2)`). Mirrors Stage 1B's own verified two-host
  TP=4 finding (ranks 0,1→hostA; ranks 2,3→hostB) at the placement-shape
  level; this project's own simulator cannot reproduce Stage 1B's exact
  MI355X numbers (no such profile exists in this checkout), so this
  example demonstrates the manifest's *shape*, not a replay of that run.
- `attn_a_ffn_b_{manifest,prediction}.json` — attention whole on one
  host, the expert-parallel group whole on the other
  (`relative="disjoint"`). This is task 56/57's own natural-split
  arrangement, carried all the way through the contract layer: see
  `test_attn_whole_a_ffn_whole_b_example_remains_distinct_after_serialization`
  in `tests/test_stage2_contracts.py` for the regression this example
  exists to pin.
- `planner_prediction_with_interval_{,manifest}.json` — a real
  `Regime(seeded=True, num_seeds=3)` search (two tp=2/ep=1 placements,
  three real Frontier seeds each): a genuine, measured `ci95_halfwidth`,
  not a fabricated one. The two placements it compares turned out to be
  clearly resolvable at 3 seeds (non-overlapping intervals), so this
  example does *not* show a real tie — the winner-relative tie
  mechanism itself (`indistinguishable_from_winner`,
  `winner_equivalence_group_size`) is instead pinned by a synthetic,
  clearly-labeled hermetic unit test
  (`test_tie_group_preserved_through_json_round_trip`), since this
  project's own real search did not happen to produce one in this run.
- `clean_hardware_result.json` / `contended_hardware_result.json` /
  `decision_validation_example.json` — synthetic (see above), built
  from the real `attn_a_ffn_b` manifest/prediction so the three stay
  internally consistent (same `plan_id`/`candidate_id`/placement). The
  contended example's `in_run_conflicting_process_evidence` references
  Stage 1B's own verified finding 8 (a real external-contender failure
  mode — third-party alltoall/RCCL activity — that can invalidate a run
  even when the candidate's own configuration is valid) as the kind of
  evidence this field exists to carry, not as a claim that this
  specific text was observed on real hardware.

## Regenerating the examples

The three real (manifest/prediction) example pairs, the interval
example, and the four schema files are each produced by one-off
scripts that import real `tools/planner_core.py`/`tools/planner.py`
objects and (for the three manifest/prediction pairs) invoke a real
Frontier subprocess through `SimulationEvaluator`. These scripts are
intentionally not checked into this repository — they are throwaway
generators, not a maintained tool, matching this task's own "reuse
existing types" principle rather than adding new permanent surface
area. Re-derive them from `tools/stage2/exporters.py` and
`tools/stage2/contracts.py` directly if the examples ever need
regenerating.
