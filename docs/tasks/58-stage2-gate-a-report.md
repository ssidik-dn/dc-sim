# Stage 2 — Gate A: the planner ↔ real-runtime contract

Branch: `stage2-gate-a-contract`, branched from `task-57-joint-key`'s tip
(commit `725485a`). Paths per the project's own standing convention:
working tree at `/work/simulation/dc-sim`, Frontier at
`/work/simulation/Frontier`. No GPU, no fleet access in this sandbox —
the same constraint Task 55 operated under and documented in
`docs/tasks/55-noise-pilot-report.md` §0. This task's own scope is
architecture only: define the contract, do not run real hardware, do
not modify `sim_real`, do not re-derive what Tasks 56/57 already
answered.

327 tests pass (282 pre-existing + 45 new, `tests/test_stage2_contracts.py`),
5 skipped (Task 53's own Fix B tests, unrelated, still skipped for lack
of `torch`); `check_import_direction.py` exits 0 (this task adds no
file under `src/engine/`, so that check is unaffected by construction).
Every `mean_tpot_ms`/`throughput_rps`/`slo_attainment` figure quoted
below came from a real `SimulationEvaluator` run against real h800
Phi-tiny-MoE-instruct profiles — see §21 and §22.

---

## §0. Read first

**The planner can produce an executable, unambiguous manifest today for
the axes Tasks 32–57 already established as real: TP degree, TP
placement, EP degree, EP placement (including the Task 56/57 natural
split), and memory margin, at `attn_replicas=1`.** It cannot yet do so
for `attn_replicas > 1` at all (a structural block, not a missing
export), and two axes — the EP-degree profiling grid and the
throughput floor's relative-to-baseline mode — carry real, specific
caveats that a product decision must not paper over. The final verdict
(§26) is **YES WITH CONSTRAINTS**, and §24/§25 say exactly which
constraints.

Nothing in this document is a new hardware fact. The eight verified
findings this task was handed (Stage 1B, `sim_real`) are treated as
given throughout and cited by number; no ninth fact is invented.

---

## §1. The no-ambient-context principle, applied to this contract itself

Every one of the four contract objects (§2) carries its own complete
`InputIdentity` (or, for `HardwareResult`, its own `SystemInfo`) — a
manifest or a result is self-describing. Concretely, this principle was
applied in three places where it would have been easy to take a
shortcut instead:

1. **Topology identity.** `TopologySpecRef` requires `num_machines`,
   `gpus_per_machine`, `scale_up_GBps`, `scale_out_GBps` as **explicit
   parameters to the exporter**, not introspected from a live `Fabric`
   object. Checked directly (`src/engine/physical/topology.py`,
   `Fabric.__init__`): a `Fabric` stores `self.name`,
   `self.machines`, `self.domains`, `self._links` — it does not cache
   the bandwidth parameters `engine.physical.builders.build_node_scale`
   was originally called with. An exporter that tried to recover them
   by walking `fabric.domains`/`fabric._links` would be reconstructing
   ambient context from a graph that does not actually retain it in one
   place; requiring the caller (who built the fabric and therefore
   already has these numbers) to pass them again is the honest version
   of "no ambient assumptions," not a missing convenience.
2. **Host binding.** `PlacementSpec.topology_machine_to_host: Dict[int, str]`
   is a required, separate field from the rank assignments themselves.
   The planner's own `GpuId(machine, index)` is an abstract domain
   index with zero knowledge of a real fleet hostname; nothing between
   `tools/planner_core.py` and `src/engine/placement/` has ever known
   one. Folding a guessed or default hostname into `PlacementRankAssignment`
   would be exactly the ambient assumption this task's own scope forbids.
3. **Profile provenance is per-manifest, not global.** `ProfileProvenance`'s
   two boolean flags (`phase_filter_applied`, `block_table_fix_applied`)
   are `Optional[bool]`, and every exporter call in this task's own
   examples (§21) sets them to `None` — not `False` — because Task 53's
   Fix A only applies to LATENT_MLA models (Phi-tiny-MoE-instruct is
   DENSE_KV, so the question does not apply) and Fix B is a standalone
   patch never wired into `install()` (confirmed by reading
   `src/integration/profiling/attention_block_table_fix.py` directly:
   it is not imported by `src/integration/install.py`). `None` here
   means "not applicable/not confirmed," never a silent `False` that
   would misrepresent an unchecked fact as a checked one.

---

## §2. The four contract objects

`tools/stage2/contracts.py` (~460 lines, 31 dataclasses, zero imports
outside `dataclasses`/`typing` — no Frontier, no `src/integration/`,
no filesystem access) defines:

| Object | What it is | Real-code analogue it mirrors |
|---|---|---|
| `DeploymentManifest` | What to run, exactly | `Candidate` + a resolved `Placement` + `Topology`/`ModelSpec`/`Workload`/`Hardware`/`Objectives`/`Regime` |
| `PlannerPrediction` | What the planner predicted, with uncertainty | one row of `PlanResult.ranked` / `TwoStagePlanResult.ranked` |
| `HardwareResult` | What `sim_real` actually observed | nothing in this project — a new object, by design, since no evaluator here has ever produced one |
| `DecisionValidation` | Whether the planner's choice matches reality | `tools/stage2/decision.py`'s own pure comparison, nothing in `tools/planner_core.py` computes this today |

`tools/stage2/serialization.py` provides one generic recursive JSON
(de)serializer for all 31 dataclasses (not 31 hand-written `to_dict`s —
see §21 for why). `tools/stage2/validators.py` and
`tools/stage2/decision.py` are the two places behavior lives; `contracts.py`
itself is pure data.

---

## §3. `DeploymentManifest` design

Eleven top-level fields: `manifest_version`, `plan_id`, `candidate_id`,
`input_identity` (the six S1 inputs, bundled), `parallelism`,
`placement`, `runtime`, `workload`, `constraints`, `profile_provenance`,
`provenance`. `workload` and `constraints` appear twice — once nested
inside `input_identity`, once at the top level — deliberately: a
consumer that only cares "what do I run" (top-level `workload`/`constraints`/
`placement`/`parallelism`/`runtime`) should not have to walk into
`input_identity` to find them, while a consumer asking "was this
manifest planned against the same question as that other one" reads
`input_identity` as one unit. This is redundancy with a stated purpose,
not an accident of two people not talking to each other — there is only
one author here, and the duplication is deliberate per this task's own
"every field has a stated owner and a stated consumer" instruction (§3
of the original spec).

Every field's producer is `tools/stage2/exporters.py`; every field's
consumer is `sim_real`, reading only this file — no field exists here
"for completeness."

---

## §4. Placement expressiveness, verified against Tasks 56–57

`ParallelismSpec` carries `relative: Optional[str]` — the exact field
Task 57 added to `planner_core.Candidate` — alongside `attn_shape`/`ep_shape`,
not a two-component key. This is not a re-derivation of Task 56/57's own
finding; it is a check that the fix survives the contract layer:

- `test_attn_whole_a_ffn_whole_b_example_remains_distinct_after_serialization`
  (`tests/test_stage2_contracts.py`) loads a real, on-disk manifest
  (`contracts/stage2/examples/attn_a_ffn_b_manifest.json`, built from an
  actual `SimulationEvaluator` run, §21) whose `parallelism.relative == "disjoint"`,
  serializes it to JSON and back, and re-asserts `relative == "disjoint"`
  survived — then independently re-derives the same fact from
  `placement.assignments` directly: the DECODE_ATTN group's hosts and
  the DECODE_FFN group's hosts do not intersect. Both checks pass.
- The single-host (`single_host_tp2`) and two-host (`two_host_tp4`)
  examples pin the other two points on this axis: `relative=None` when
  no expert-parallel group exists at all (`ffn_ep=1`), and a TP=4 group
  legitimately split `(2, 2)` across two real hosts with `relative`
  not even applicable to a TP-only candidate.

This is the honest scope of what "verified" means here: the contract
can carry what Task 57 built. It cannot independently confirm Task
57's own regression test at the `planner_core` layer, and does not try
to — that regression is `tests/test_planner_core.py::test_natural_split_is_reachable`,
unchanged and still passing (§21's own full-suite run).

---

## §5. Feasibility vs. runtime-precondition distinction

`PlannedFeasibility` (`PLANNED_FEASIBLE`/`PLANNED_INFEASIBLE`/`PLANNED_INADMISSIBLE`/
`PLANNED_UNKNOWN`) mirrors `planner_core.py`'s own `Rejection`/`Inadmissible`/`Unknown`
split exactly — a property of the *request*, computed before any
hardware is touched. `RuntimeExecutionStatus` (eight `RUNTIME_*`
constants) is a property of *this specific attempt, on this specific
hardware, right now*. `tools/stage2/validators.validate_runtime_status_is_not_planner_feasibility`
enforces this as code, not only as a naming convention: passing a
`PLANNED_*` string where a `RuntimeExecutionStatus` is expected raises
`ValidationError` (`test_planner_feasibility_constant_rejected_as_a_runtime_status`).

This distinction is what makes verified finding 5 (resource availability
is a hard runtime precondition; the launcher must refuse when requested
GPUs are occupied) and finding 6 (hardware feasibility may not be
symmetric across ranks — different observed VRAM per local rank in
cross-host TP=4) representable without corrupting `PlannedFeasibility`:
a `RUNTIME_RESOURCE_BUSY` result says nothing about whether the plan
itself was feasible, and this contract has no field that would let a
launcher's refusal silently overwrite a planner's own feasibility
verdict. Finding 6 in particular is carried as an *observation* —
`OccupancyEvidence`/`MemoryObservation.per_gpu_bytes` on a real
`HardwareResult`, never as a planner-side memory model change, per the
finding's own instruction to treat it as "an observed hardware fact,
not an explained mechanism."

---

## §6. Workload / regime contract

`WorkloadSpecRef.regime` has no default — mirrors `planner_core.Regime`'s
own `__post_init__`, which raises rather than let a caller inherit
burst by accident (Task 45). `validators.validate_workload_spec` adds
the two checks the type system alone cannot: streaming requires
`qps`/`seed`/`num_seeds` all set, and neither regime may carry
`qps=inf`. Both are tested directly
(`test_validate_workload_spec_rejects_streaming_without_qps_seed_num_seeds`,
`test_validate_workload_spec_rejects_infinite_qps`).

Verified finding 4 (burst and streaming are materially different
sizing regimes) is exactly why this field exists and why it has no
default — a manifest that did not say which regime it was planned
under would be unusable for `DecisionValidation`, since Task 41/44
already showed the sizing axis (not the placement axis) can reverse
between them.

---

## §7. Objective / constraint contract

`ConstraintSpec` (SLO, throughput floor, memory margin) and `ObjectiveSpec`
(what to minimize) split `planner_core.Objectives` into the two halves
the original spec's own S7 asked for, without changing `Objectives`
itself. `ThroughputFloor` supports `mode="absolute"` and
`mode="relative_to_baseline"` — but this is the one place in the whole
contract where the schema is ahead of the real search: **`Objectives.min_throughput_rps`
is absolute-only in `tools/planner_core.py` today** (checked directly;
`plan()`'s own constraint check is `r["throughput_rps"] < objectives.min_throughput_rps`,
a single float, no baseline candidate concept anywhere in that module).
`mode="relative_to_baseline"` is representable in the contract and
carries a `baseline_candidate_id` field, but nothing in this project's
real search can currently produce or honor one. This is reported here,
not silently closed by changing `Objectives` (out of this task's own
scope: "do not change planner objective/search behavior unless needed
to expose the contract" — adding a *contract* field is in scope;
teaching `plan()` a new constraint mode is not).

---

## §8. `PlannerPrediction` uncertainty and ties

`UncertaintySpec` carries exactly what `Regime`/`seed_stats.compute_interval_stats`
already compute — `ci95_halfwidth=None` for a `Regime(num_seeds=1)`
burst result, a real Student's-t half-width for a seeded, multi-seed
result. §21's `planner_prediction_with_interval.json` example is a real
one: a `Regime(seeded=True, num_seeds=3)` search over two tp=2/ep=1
placements, three real Frontier seeds each. Its own two candidates
turned out to be clearly resolvable at 3 seeds (non-overlapping
intervals: `[2.56, 3.27]` vs. one candidate roughly 1.8ms slower with a
similarly-sized interval) — so this real example does **not** happen to
show a tie. The tie mechanism itself (`RankingSpec.indistinguishable_from_winner`,
`winner_equivalence_group_size`) is instead pinned by a synthetic,
clearly-labeled hermetic test
(`test_tie_group_preserved_through_json_round_trip`), because a real
search did not produce one in this run and this report will not
present a fabricated one as if it were measured.

`RankingSpec` is deliberately **winner-relative only**, matching
`planner_core._mark_indistinguishable_from_winner` exactly: that
function computes one binary flag per candidate (is this row's own CI
interval overlapping the winner's), never a full pairwise/transitive
equivalence partition across every candidate. `winner_equivalence_group_size`
is a direct count of rows sharing that flag (including the winner
itself), not an invented broader grouping. A caller reading `rank=5,
indistinguishable_from_winner=False` learns nothing about whether rank
5 and rank 6 are distinguishable from each other — this project's own
search has never computed that, and this contract does not pretend it
has.

---

## §9. Profile provenance, file-level

`ProfileProvenance.profile_files` is a tuple of real file paths (e.g.
`data/profiling/compute/h800/Phi-tiny-MoE-instruct/attention.csv`), not
a version string — because Task 52's own diagnosis (phase contamination)
and Task 53's own fixes (block-table aliasing) were both file-level
defects, and "profile version 3" cannot distinguish a manifest built
before Fix A existed from one built after it existed but not installed
for this run. `known_limitations` is a free-text tuple carrying gaps
that are known but not closed by any flag — every example in §21 carries
`"flat-extrapolation gap outside profiled tp grid (task 52) not closed
here"`, since Task 52's third candidate mechanism (extrapolation beyond
the profiled tp grid) remains open per that task's own report and this
task does not close it.

---

## §10. `HardwareResult` schema

Every field is something `sim_real` can itself observe by actually
running something — `WorkloadRealization` (requested vs. achieved
request count/QPS — verified finding 4's "burst and streaming are
materially different" needs a place to record that a streaming run's
achieved QPS fell short of requested), `HardwareMetrics` (TTFT/TPOT/E2E
as `LatencyStats`, matching what a real launcher's own metrics output
would contain, not a simulator-internal breakdown), `MemoryObservation`
(`per_host_bytes`/`per_gpu_bytes` — the field verified finding 6's
asymmetric cross-host VRAM footprint lands in), `OccupancyEvidence`
(§11), `SystemInfo` (real hostnames, GPU identities, runtime version,
image digest). No compute/communication/contention-bottleneck breakdown
is required, because a real launcher has no way to produce one — this
project's own simulator-side breakdown (Task 50's contention model)
does not appear anywhere in `HardwareResult`, on purpose.

---

## §11. Occupancy and contention as first-class

`OccupancyEvidence.contention_status` (`"clean"|"resource_busy"|"contended"|"unknown"`)
is not a footnote on `HardwareResult` — it is the field
`tools/stage2/validators.is_eligible_for_hardware_best` reads before
anything else, and it excludes strictly: a result whose
`execution_status` is not `RUNTIME_CLEAN_SUCCESS`/`RUNTIME_SUCCESS`, or
whose `contention_status` is not `"clean"`, is never eligible to become
`HardwareBest`, full stop — not flagged-but-still-eligible, not
downweighted. `contended_hardware_result.json` (§21) demonstrates this
concretely: its `in_run_conflicting_process_evidence` field references
verified finding 8 by name (third-party alltoall/RCCL activity
invalidating a run even when the candidate's own configuration is
valid) as the *kind* of evidence this field exists to carry — the text
itself is synthetic (no GPU was available to observe this), but the
mechanism it illustrates is the real, given finding, not an invention.
`test_contended_hardware_result_example_is_not_eligible_for_hardware_best`
and `test_compute_hardware_best_never_falls_back_to_a_contended_result`
both check the exclusion holds, the latter against a synthetic set
where the contended result has the best (lowest) synthetic latency —
proving the exclusion is not merely "usually wins on latency anyway."

---

## §12. `DecisionValidation`

The one comparison this whole contract exists to produce:
`top1_correct` (planner's selected candidate id == `HardwareBest`'s
candidate id), `topk_correct` (top1, or the selected id appears in a
caller-supplied `equivalence_group_hardware_ids` — never a fabricated
total order, matching §8's own winner-relative honesty), `regret_absolute`/`regret_relative`
(the gap between the selected candidate's real observed TPOT and
`HardwareBest`'s), `resolvability` (§13), `slo`/`throughput`/`placement`
comparisons (pass/fail against the manifest's own stated floors, not a
"how different is different enough" magnitude judgment — matching
`plan()`'s own constraint semantics exactly). `hardware_best_candidate=None`
is a valid, reportable outcome (`test_missing_hardware_best_is_reported_not_papered_over`)
— nothing in `compute_decision_validation` invents a `HardwareBest` when
every candidate's own result was contended or resource-busy.

---

## §13. Per-configuration noise floor

`NoiseFloorSource` names `hardware_config_id`, `workload_id`, `regime`,
and `repeats` alongside the measured `cv_pct`/`ci95_halfwidth` — every
`DecisionValidation` that claims a resolvability verdict must point at
one of these, naming exactly which configuration it came from.
`compute_decision_validation`'s resolvability logic is strict:
`noise_floor_source=None` or a source with no measured half-width
yields `resolvable=None` (unknown), never an inherited global figure —
`test_resolvability_is_unknown_without_a_per_configuration_noise_source`
checks this directly. `test_regret_smaller_than_noise_floor_is_unresolvable`
and `test_regret_larger_than_noise_floor_is_resolvable` check the two
concrete outcomes on either side of the floor. Task 55's own report
already established why this must be strict: no real noise-floor
measurement exists for any configuration in this project yet (zero
real-hardware runs taken), so every real `DecisionValidation` produced
before that changes will correctly report `resolvable=None` — this is
the contract refusing to manufacture false confidence, not a defect.

---

## §14. `HardwareBest` procedure

`compute_hardware_best(candidates, metric="mean")` filters to
`is_eligible_for_hardware_best` (§11) first, then picks the minimum by
the requested `LatencyStats` field (`mean` by default) — returning
`None`, not a downgraded pick, when nothing survives the filter
(`test_compute_hardware_best_never_falls_back_to_a_contended_result`).
The caller is responsible for passing in a set that already shares
equivalent measurement conditions (same model, workload, runtime,
regime, topology) — this function's own job is narrower by design: it
does not attempt to detect an incomparable set, matching the spec's own
instruction that `HardwareBest` is established from "a bounded candidate
set under equivalent measurement conditions," a caller-side contract,
not something this pure function can verify from the objects alone.

---

## §15. First real product search space (proposed, not run)

Scoped to exactly the axes §24 marks READY, at `attn_replicas=1,
ffn_replicas=1` (§16's own blocker excluded outright, and FFN-replica
placement's own non-optimality (§24) kept out of a *first* search even
though it is technically usable):

- **TP degree** ∈ `{1, 2, 4}` (matching verified finding 1's own
  single-node MI355X coverage exactly, and staying inside
  `model.profiled_tp`'s real grid for whatever model the first real run
  uses).
- **TP placement**: single-host (verified finding 1) and the two-host
  split verified finding 2 names explicitly (TP=2 across two hosts one
  GPU each; TP=4 as 2+2 across two hosts).
- **EP degree/placement**: the Task 56/57 natural split
  (`relative="disjoint"`) as one candidate, colocated (`relative="same"`)
  as the comparison point Task 57's own regression test already
  requires stay reachable.
- **Regime**: burst only, for the first pass — verified finding 4 says
  burst and streaming are materially different regimes, and this
  task's own scope (§25 Q&A) is architecture, not a claim about which
  regime a first real validation should use; burst is the cheaper,
  already-established one to start from.

This is deliberately small: three TP degrees × (one single-host + one
two-host placement each, where reachable) × two EP arrangements ×
`attn_replicas=1, ffn_replicas=1` fixed — on the order of a handful of
manifests, not a sweep. Task 54's own validation design (still not run,
per that task's own gating) is the next step this search space feeds.

---

## §16. Attention-replicas blocker, recorded

`ParallelismSpec.attn_replicas` is a real field — the contract can
*express* `attn_replicas > 1` — but no manifest with that value can be
produced from a real `SimulationEvaluator`-backed `plan()` run today.
`SimulationEvaluator.can_evaluate` (`tools/planner.py`) requires
`candidate.attn_replicas == 1`; the docstring right above it (quoted
directly, not paraphrased) explains why: "`populate_from_deployment`
registers each DECODE_ATTN replica's own TP group under the SAME
`(cluster_type, comm_domain, num_devices)` key... so a second replica at
the same `attn_tp` collides and `CommGroupRegistry.register` raises
`CommGroupError`" — because "Frontier's cc_backend calls carry a device
count and a parallelism-domain label — never a rank identity"
(`src/integration/cc_backend/comm_groups.py`'s own docstring). This is a
real limit of *this* evaluator's pipeline (confirmed by running it,
per that same docstring: "This is a real limit of *this* evaluator's
own pipeline, not of the (model, degree, ratio) request or of available
memory"), not a placement or a memory question — recorded here as §24's
one NOT READY axis, not attempted as a fix (out of this task's own
scope).

---

## §17. Single-host and two-host validation examples

Both built from real `SimulationEvaluator` runs (h800, Phi-tiny-MoE-instruct,
`domain64` — a registered 2-domain, 64-GPUs-per-domain topology in
`tools/planner.py`'s own `_TOPOLOGIES`; real-evaluation runs re-invoke
this file by topology *name* in a subprocess, so the topology used must
already be registered there — see §21 for why `domain8`, tried first,
was rejected in favor of `domain64`):

- `single_host_tp2`: `attn_tp=2`, shape `(2,)`, all ranks on one host —
  `mean_tpot_ms=4.693`, real, `slo_attainment=1.0`.
- `two_host_tp4`: `attn_tp=4`, shape `(2, 2)`, split across two hosts
  three GPUs each (PREFILL/FFN's own single ranks packed alongside two
  of the four TP ranks per host, per `_placement_for`'s own leftover-packing
  fallback) — `mean_tpot_ms=10.009`, real.

Both mirror verified findings 1/2's own placement *shape*, not a replay
of the real MI355X numbers themselves — this project's own simulator has
no MI355X profile in this checkout, so it cannot and does not claim to
reproduce Stage 1B's actual measured latencies. What it demonstrates is
that the manifest schema itself can carry both shapes losslessly through
a full JSON round trip (`test_single_host_tp2_example_validates`,
`test_two_host_tp4_example_validates`), which is what this gate is
actually gating.

---

## §18. `sim_real` handoff boundary

File-only, both directions. No Python import crosses it in either
direction: `tools/stage2/` never imports anything from a `sim_real`
package (it does not exist in this checkout, and nothing here assumes
it will be importable), and this task's own instructions forbid this
project from touching `sim_real` at all. The boundary is exactly the
four JSON schema files in `contracts/stage2/` (§22) plus the version
policy (§19) — a `sim_real` implementation reads a `DeploymentManifest`
JSON file, writes a `HardwareResult` JSON file, and neither side needs
to know the other's internal representation, only the shared schema.

---

## §19. Schema versioning

Every one of the four objects carries its own `*_version` field, all
`"1.0"` today. `tools/stage2/serialization.check_major_version` hard-rejects
a major-version mismatch (`SchemaVersionError`) and accepts any minor
difference without translation — no migration logic exists, deliberately
(§19 of the original spec: "do not over-engineer migrations" before a
single real manifest has been executed). `test_unknown_schema_major_version_rejected`
and `test_minor_version_bump_is_accepted` both check this holds in both
directions. A missing required field is a separate, equally hard
failure (`SchemaFieldError`, `test_missing_required_field_is_a_hard_reject_not_a_silent_default`)
— never silently defaulted.

Mid-task, one real, additive change was made under this exact policy:
`RuntimeSpec` gained a `decode_ffn_scheduler: Optional[str] = None`
field (§24 found the axis real and already searchable — `tools/planner.py`'s
own `_argv`/`_run_scenario`/`run_topology_scheduler_study.py` all vary
it — but the contract had no field for it). Every existing JSON example
in `contracts/stage2/examples/` remains valid after this change without
modification, exactly the "minor addition needs no migration" case this
policy exists to allow.

---

## §20. Provenance

`Provenance` (`timestamp_utc`, `planner_git_sha`, `simulator_git_sha`,
`topology_id`, `seed`) is present on all four top-level objects — no
unlocated headline number. Every real example in §21/§22 carries
`planner_git_sha="unknown-sandbox"` honestly (this checkout is not a git
repository — no commit sha is available to attach — rather than a
fabricated hash), and every synthetic `HardwareResult`/`DecisionValidation`
example (§21) carries `runtime_version="SYNTHETIC-ILLUSTRATIVE-NOT-REAL-HARDWARE"`
in `SystemInfo`, so a reader who encounters one of these files outside
this document's own context cannot mistake it for a real measurement.

---

## §21. Implementation scope — what was actually built

- `tools/stage2/contracts.py` — 31 dataclasses, the four top-level
  objects and their nested types (§2).
- `tools/stage2/serialization.py` — one generic recursive JSON
  (de)serializer for all 31, chosen over 31 hand-written `to_dict`/`from_dict`
  pairs specifically to avoid drift risk as fields are added (the same
  reasoning this project already applied when it chose one generic
  `Placement` abstraction over one class per topology shape). Handles
  nested dataclasses, `Optional[X]`, `List[X]`, `Tuple[X, ...]`,
  `Dict[str/int, X]` (including int-key restoration — JSON has no
  integer object keys, `topology_machine_to_host` needs them back).
- `tools/stage2/validators.py` — structural validation (§5/§6 checks,
  duplicate-rank/duplicate-GPU/unknown-host placement checks,
  manifest/prediction pairing, `HardwareBest` eligibility).
- `tools/stage2/decision.py` — `compute_hardware_best`/`compute_decision_validation`,
  pure comparison logic, no hardware touched.
- `tools/stage2/exporters.py` — builds real `DeploymentManifest`/`PlannerPrediction`
  objects from real `planner_core.Topology`/`ModelSpec`/`Workload`/`Hardware`/`Objectives`/`Regime`/`Candidate`
  and a real resolved `Placement.mapping`. This is the one module
  allowed to import `planner_core`/`planner` — the producer side.
- `tests/test_stage2_contracts.py` — 45 hermetic tests (§23).
- `contracts/stage2/` — four generated JSON Schema files, a `README.md`,
  and `examples/` (§22).

**The synthetic end-to-end round trip** (planner object → manifest JSON
→ prediction JSON → fake `HardwareResult` → `DecisionValidation`) was
built and run for real, not merely designed: `attn_a_ffn_b` (§17/§22)
is a real `Candidate(attn_tp=4, attn_shape=(4,), ffn_ep=2, ep_shape=(2,),
relative="disjoint")`, evaluated by a real `SimulationEvaluator` against
`domain64` (`mean_tpot_ms=5.409`, `throughput_rps=42.96`, `slo_attainment=1.0`
— all real), exported to a `DeploymentManifest`/`PlannerPrediction` pair,
compared against a synthetic `HardwareResult` via `compute_decision_validation`,
producing a real `DecisionValidation` (`top1_correct=True`,
`regret_absolute=0.0`, `resolvability.resolvable=False` against a
synthetic 0.30ms noise floor). One real snag surfaced building this:
`evaluate()` re-invokes `tools/planner.py` in a subprocess by topology
*name* only (the live `Fabric` object never crosses that boundary), so
a topology used for a real evaluation must already be one of
`tools/planner.py`'s own registered `_TOPOLOGIES` — an ad hoc
`Topology(fabric, "two-real-machines")` built inline fails with
`KeyError`. `domain64` (already registered, exactly 2 domains) was used
instead of building a new one, since adding a new registered topology
to `tools/planner.py` would be a planner-module change this task's own
scope does not require.

---

## §22. Contract directory layout

```
contracts/stage2/
├── README.md
├── deployment_manifest.schema.json
├── planner_prediction.schema.json
├── hardware_result.schema.json
├── decision_validation.schema.json
└── examples/
    ├── single_host_tp2_manifest.json          (real)
    ├── single_host_tp2_prediction.json         (real)
    ├── two_host_tp4_manifest.json              (real)
    ├── two_host_tp4_prediction.json             (real)
    ├── attn_a_ffn_b_manifest.json               (real)
    ├── attn_a_ffn_b_prediction.json              (real)
    ├── planner_prediction_with_interval.json           (real)
    ├── planner_prediction_with_interval_manifest.json   (real)
    ├── clean_hardware_result.json                (synthetic, labeled)
    ├── contended_hardware_result.json            (synthetic, labeled)
    └── decision_validation_example.json          (synthetic, labeled)
```

The four schema files were generated directly from the real
`tools/stage2/contracts.py` dataclasses by a one-off reflection script
(not committed — see `contracts/stage2/README.md`'s own "Regenerating
the examples" section for why), so they cannot drift from the Python
types by hand-transcription error. All eleven example files validate
against their respective schema under `jsonschema.validate` (checked
directly, not assumed).

---

## §23. Required tests — coverage

45 tests in `tests/test_stage2_contracts.py`, hermetic (no Frontier
subprocess, no GPU). Coverage against the required scenario list:
missing/invalid regime rejected (3 tests), duplicate rank mapping
rejected, duplicate host/GPU assignment rejected, unknown host
rejected, well-formed placement accepted, mismatched manifest/prediction
plan_id rejected, unknown major version rejected, minor version bump
accepted, missing required field hard-rejected, resource-busy status
valid as a *runtime* status, a `PLANNED_*` constant rejected as a
runtime status, resource-busy `HardwareResult` still validates
structurally (a fact, not a rejected payload), contended/resource-busy
results excluded from `HardwareBest` (3 tests), `compute_hardware_best`
never substitutes a contended result even when it has the best raw
latency, top1 correct, top1 wrong but topk-via-equivalence-group
correct, regret calculation, resolvability unknown without a noise
source, resolvability false below the noise floor, resolvability true
above it, SLO pass/fail mismatch, throughput-floor mismatch, exact
placement match/mismatch (2 tests), missing `HardwareBest` reported not
papered over, tie group round-trips through JSON, confidence interval
round-trips through JSON, a burst prediction carries no fabricated
interval, profile provenance (files/commit/fix-flags) round-trips,
int-keyed `topology_machine_to_host` round-trips with real `int` keys
(not strings), the three real named examples each validate structurally,
the real interval example carries a real `ci95_halfwidth > 0`, the two
synthetic `HardwareResult` examples validate and correctly report
eligible/not-eligible for `HardwareBest`, the synthetic `DecisionValidation`
example matches its own inputs, and — the one this section exists
around — the ATTN-whole-A/FFN-whole-B example remains distinct
(`relative="disjoint"`, disjoint host sets) after a full JSON round
trip.

Full-suite run: 327 passed, 5 skipped (pre-existing, unrelated),
113.6s.

---

## §24. Planner axis readiness table

| Axis | Status | Cited mechanism |
|---|---|---|
| TP degree | **READY** | `SimulationEvaluator.can_evaluate` gates on `attn_tp in model.profiled_tp`; every model in this checkout has real profiles at tp∈{1,2,4,8} (Task 35). Contract: `ParallelismSpec.attn_tp`. |
| TP placement | **READY** | `enumerate_attn_shapes`/`enumerate_joint_arrangements` (Tasks 32/44), fully reachable including cross-domain shapes; demonstrated real end-to-end at contract layer (§17: `single_host_tp2`, `two_host_tp4`). |
| EP degree | **READY WITH CONSTRAINTS** | `can_evaluate` gates `attn_tp` against `model.profiled_tp` but has **no equivalent gate for `ffn_ep`** (checked directly — the method reads only `candidate.attn_tp`/`candidate.attn_replicas`). A caller supplying an `ffn_ep` outside whatever grid Frontier's own MoE cost model actually covers gets no `Unknown` warning from this evaluator; it silently proceeds. Not a claim about what Frontier's own moe cost model does at that point — a verified gap in *this* evaluator's own coverage-reporting. |
| EP placement (`relative`) | **READY** | Task 56 diagnosed, Task 57 fixed, this task (§4) re-verified the fix survives a full contract-layer JSON round trip with a real evaluated example. |
| FFN replicas | **READY WITH CONSTRAINTS** | Confirmed clean up to 16 replicas by actually running it (Task 41, `SimulationEvaluator`'s own docstring). Constraint: `_placement_for`'s own leftover-packing fallback is "not placement-optimal for those extra ranks" beyond replica 0 (its own docstring, verbatim) — functional, not principled placement search, for `ffn_replicas > 1`. |
| Attention replicas | **NOT READY** | Structurally blocked: `can_evaluate` requires `attn_replicas == 1`; `CommGroupRegistry.register` raises `CommGroupError` above that, because Frontier's cc_backend collective calls carry a device-count-plus-domain-label key with no rank/replica identity (§16, cited verbatim from source). |
| Workload regime (burst/streaming) | **READY WITH CONSTRAINTS** | Both representable and distinguishable (`WorkloadSpecRef.BURST`/`STREAMING`); constraint is §7's throughput-floor gap — `mode="relative_to_baseline"` is representable in the contract but not backed by `Objectives.min_throughput_rps`, which is absolute-only in the real search. |
| Memory margin | **READY** | `Hardware.memory_margin_fraction` → `HardwareSpecRef`/`ConstraintSpec` directly; `feasible_num_blocks` has used this exact field since Tasks 24–28. |
| Scheduler policy (decode-FFN) | **READY** | Real, already-searchable axis (`tools/planner.py`'s own `_argv`, `run_topology_scheduler_study.py`'s own comparison) that the contract simply had no field for; closed this task, additively (§19), by adding `RuntimeSpec.decode_ffn_scheduler`. |
| Tie/equivalence representation | **READY WITH CONSTRAINTS** | Winner-relative only (§8) — an honest, real limit of what `plan()` itself computes, not an implementation gap in the contract; a full pairwise partition does not exist anywhere in this project to export. |

---

## §25. Seven questions this gate needed answered

**A. Can the planner produce a manifest a real launcher could execute
without reinterpreting it?** Yes, for the READY axes at
`attn_replicas=1`: `DeploymentManifest.placement` gives an exact
rank→host→physical-GPU mapping (§3/§17), not a placement policy name a
launcher would have to re-resolve.

**B. Does the contract preserve the one placement distinction Task
56/57 fixed?** Yes, verified at the contract layer with a real
evaluated example, not merely inherited from the `planner_core` layer
(§4).

**C. What happens when the requested GPUs are busy?** Recorded as
`RUNTIME_RESOURCE_BUSY`, a runtime fact about this attempt — never
downgraded to, or confused with, a planner-side `PLANNED_INFEASIBLE`
verdict (§5).

**D. What happens when a run is contended but completes anyway?** It
is `RUNTIME_CONTENDED`, and `is_eligible_for_hardware_best` excludes it
from ever becoming `HardwareBest`, unconditionally, even if it has the
best raw observed latency in the set (§11/§14, tested directly).

**E. Can the contract tell "the planner was wrong" apart from "the
measurement can't tell the difference"?** Yes — `resolvability` is
computed strictly against a per-configuration `NoiseFloorSource` and is
`None` (unknown), never a false positive or negative, when no such
measurement exists (§13). As of this report, no real noise-floor
measurement exists anywhere in this project (Task 55 took none), so
every real `DecisionValidation` produced today will correctly say
"unknown" here until that changes.

**F. What must not be included in a first real search space?**
`attn_replicas > 1` (§16, structural). What should be flagged, not
excluded: EP degree outside whatever grid Frontier's MoE cost model
actually covers, and `ffn_replicas > 1`'s non-optimal placement (§24).

**G. What is the smallest real search space worth actually running?**
§15's proposal: TP∈{1,2,4} × (single-host, two-host-split where
reachable) × (colocated, natural-split) × `attn_replicas=1,
ffn_replicas=1` × burst regime — on the order of a handful of manifests,
directly answerable by Task 54's own (still not run) validation design.

---

## §26. Final verdict

**IS THE CURRENT PLANNER READY TO PRODUCE AN EXECUTABLE, UNAMBIGUOUS
MANIFEST FOR REAL HARDWARE VALIDATION?**

## YES WITH CONSTRAINTS.

The contract objects exist, are tested (45 hermetic tests, all
passing), and have been exercised end-to-end against real planner
output three times over (§17/§21/§22) — not merely designed on paper.
For TP degree/placement, EP degree/placement (including the Task 56/57
natural split), memory margin, and decode-FFN scheduler policy, at
`attn_replicas=1, ffn_replicas=1`, the planner can produce a manifest
today that says exactly what to run and exactly where, with nothing
left for a launcher to infer.

The constraints are specific, not generic hedging: `attn_replicas > 1`
is structurally blocked and must be excluded from any first real search
space (§16/§24); the throughput floor's relative-to-baseline mode is
representable in the contract but not backed by the real search (§7);
`ffn_replicas > 1` and out-of-grid `ffn_ep` values work but carry
placement- and coverage-quality caveats respectively that a product
decision must see, not just this report (§24). None of these block
Gate A's own purpose — none of them were invented to pad this list, and
each one traces to a specific, cited line of existing code or an
existing task's own established finding, not to this task's own
convenience.
