# Stage 2 — Gate A.1: contract / coverage cleanup

Branch: `stage2-gate-a-contract` (continuing directly on top of Gate A's
own commits `408ae91`/`caaec12`), working tree at
`/work/simulation/dc-sim`, Frontier at `/work/simulation/Frontier`. Same
scope discipline as Gate A itself: no hardware run, no `sim_real`
modification, no planner search/objective behavior change beyond the
one specific validation gap this task names.

358 tests pass (327 pre-existing + 31 new: 10 in
`tests/test_ffn_ep_coverage_guard.py`, 21 in `tests/test_stage2_contracts.py`),
5 skipped (pre-existing, unrelated); `check_import_direction.py` exits 0.

---

## 1. The exact `ffn_ep` coverage gap

Gate A's own report (§24) found `SimulationEvaluator.can_evaluate`
(`tools/planner.py`) gated `candidate.attn_tp` against `model.profiled_tp`
but had no equivalent check for `candidate.ffn_ep`:

```python
def can_evaluate(self, candidate: Candidate) -> bool:
    return candidate.attn_tp in self.model.profiled_tp and candidate.attn_replicas == 1
```

Read against the real cost-model path this gates (`frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`),
the consequence is concrete, not hypothetical: at training time, this
predictor filters its own `moe.csv` to rows matching
`(num_tensor_parallel_workers, expert_parallel_size)` for the requested
degree, and **raises a bare `ValueError`** — "MoE dataset contract
validation failed before training... Available (TP, EP) pairs for
matched model rows: {available_pairs}" — when no row matches. This
propagates through `tools/planner.py`'s own `evaluate()` subprocess
call as a generic `{"error": "no result (exit code 1)"}`, which
`plan()` then records as a **`Rejection`** ("evaluation error: ..."),
never an `Unknown`. This conflates two things Task 37's own `Rejection`/`Unknown`
split exists to keep apart: "this configuration is bad" (a `Rejection`,
a property of the request) vs. "this evaluator does not know" (an
`Unknown`, a property of the evaluator's own coverage). A caller's
search results would show real EP degrees quietly disappearing into
the wrong bucket, with a raw `ValueError` traceback as the only
diagnostic — not a clean, structured signal.

---

## 2. Source of valid EP-degree metadata

**Not identical to `profiled_tp`, checked directly rather than
assumed.** `profiled_tp` is `(1, 2, 4, 8)` for every model on every
device this project has a real compute profile for (Task 35's own
finding, re-confirmed here, still true). `expert_parallel_size`
coverage is **not** uniform — read directly from every real `moe.csv`
in this checkout (`data/profiling/compute/*/*/moe.csv`):

| Model | Device | `num_experts` | real `expert_parallel_size` grid |
|---|---|---|---|
| `mixtral_8x7b_moe` | a100 | 8 | `(1,)` |
| `qwen2_moe_example` | a100 | 60 | `(1, 2)` |
| `qwen3-a3b-30b-moe` | a800 / mi355x | 128 | `(1, 2, 4, 8)` |
| `qwen3-next-80b-a3b-instruct-reduced-l2{,0}` | a800 | 512 | `(1,)` |
| `qwen3-next-80b-a3b-instruct-reduced-l2` | h800 | 512 | `(1, 2, 4, 8)` |
| `Phi-tiny-MoE-instruct` | h800 | 16 | `(1, 2, 4, 8)` |
| `Phi-tiny-MoE-instruct` (rtx_pro_6000, `Qwen3-30B-A3B-tiny`) | rtx_pro_6000 | 16 | `(1,)` |
| `Step2Mini-tiny`, `step-moe-noquant-small` | h800 | 8 / 24 | `(1, 2, 4, 8)` |
| `deepseek-v3`, `deepseek-r1-0528` | mi355x | 256 | `(1, 2, 4, 8)` |

The same model (`qwen3-next-80b-a3b-instruct-reduced-l2`) is `(1,)` on
one device and `(1, 2, 4, 8)` on another. Defaulting a new `profiled_ep`
field to `(1, 2, 4, 8)` (mirroring `profiled_tp`'s own default) would
have been **wrong** for six of these ten real (model, device) pairs —
exactly the "do not assume it is identical to `profiled_tp`" instruction,
now backed by measurement rather than by inspection of one file.

**The metadata's real source, one level down**: Frontier's own
`sklearn_moe_execution_time_predictor.py`, `_get_profiling_metadata`
(around line 676), computes `available_pairs = sorted({(tp, ep) for
tp, ep in base_df[["num_tensor_parallel_workers", "expert_parallel_size"]]
.drop_duplicates()...})`, where `base_df` is `moe.csv` filtered to rows
matching this model's own `num_experts`, `router_topk`, `hidden_dim`,
and `expert_hidden_dim`. This is the authoritative filter Frontier
itself applies before training. `ModelSpec` (`tools/planner_core.py`)
does not carry `expert_hidden_dim` today, so the new
`tools/planner.discover_profiled_ep` helper (§3) approximates this
filter using the three columns `ModelSpec` does carry
(`num_experts`/`router_topk`/`hidden_dim`) — checked directly against
every real `moe.csv` in this checkout that no file mixes two different
`expert_hidden_dim` values under one `(num_experts, router_topk,
hidden_dim)` triple, so this approximation is exact for every model
this project has today, though it is not a structural guarantee for a
hypothetical future profile that did mix them.

---

## 3. The exact fix

Two additive changes, no existing default changed:

1. **`tools/planner_core.py`, `ModelSpec`**: new field
   `profiled_ep: Tuple[int, ...] = (1,)`. Default `(1,)` — the one
   value present in every real `moe.csv` in this checkout — chosen
   specifically so every existing call site (none of which sets
   `ffn_ep` above 1 without also constructing its own `ModelSpec`) is
   unaffected; a caller reaching `ffn_ep > 1` against a real
   `SimulationEvaluator` must now supply the real grid explicitly.
2. **`tools/planner.py`, `SimulationEvaluator.can_evaluate`**:
   ```python
   def can_evaluate(self, candidate: Candidate) -> bool:
       return (candidate.attn_tp in self.model.profiled_tp
              and candidate.attn_replicas == 1
              and candidate.ffn_ep in self.model.profiled_ep)
   ```
   An out-of-grid `ffn_ep` now returns `False` — routed through
   `plan()`'s own existing `Unknown` path, never sent to a Frontier
   subprocess, never silently accepted, never clamped or extrapolated.
3. **`tools/planner.py`, new function `discover_profiled_ep(device,
   model_name, *, num_experts, router_topk, hidden_dim, frontier_root=
   FRONTIER_ROOT)`**: reads the real `moe.csv` directly and returns the
   real grid — a discovery convenience for setting `profiled_ep`
   correctly, not wired into `can_evaluate` itself (which reads only
   the already-set tuple, exactly like `profiled_tp`, so this is not a
   per-candidate file read).

**Real regression check, not merely a unit test**: re-ran Gate A's own
`attn_a_ffn_b` candidate (`Candidate(attn_tp=4, attn_shape=(4,),
ffn_ep=2, ep_shape=(2,), relative="disjoint")`, Phi-tiny-MoE-instruct,
h800, `domain64`) through a real `SimulationEvaluator`, once with
`profiled_ep=(1,2,4,8)` set explicitly and once with the new default
`(1,)`. With the real grid set: `can_evaluate` returns `True`, and the
real Frontier subprocess produces **the identical result Gate A's own
example already recorded** — `mean_tpot_ms=5.409259642841263`,
bit-for-bit. With the default left unset: `can_evaluate` returns
`False` before any subprocess runs. This is the exact "preserve valid
currently-used EP degrees unchanged" / "reject the out-of-grid case as
`Unknown`" pair this task asked for, demonstrated against real
Frontier output, not only against a fake evaluator.

---

## 4. Workload/constraints consistency validator

New function `tools/stage2/validators.validate_workload_and_constraints_consistency(manifest)`,
wired into `validate_deployment_manifest` so every manifest validation
now checks it automatically:

```python
if manifest.workload != manifest.input_identity.workload:
    raise ValidationError(...)
if manifest.constraints != manifest.input_identity.constraints:
    raise ValidationError(...)
```

Equality is **semantic**, not object identity — Python's own
auto-generated dataclass `__eq__` compares every field by value,
recursively (through `ThroughputFloor` for `ConstraintSpec`), which is
exactly what "two sources of truth agree" needs to mean; an identity
check (`is`) would fail even for a validly-round-tripped manifest,
since `from_dict` always builds fresh objects from a `dict`.
`export_deployment_manifest` (`tools/stage2/exporters.py`) already
builds both copies from one shared local variable in a single pass
(confirmed by reading it — this was not changed for Gate A.1, it was
already true), so no real manifest this project's own exporter
produces can diverge; the new validator exists for exactly the case
that guarantee cannot reach: a JSON file `sim_real` reads that this
project's own exporter did not produce, or one hand-edited or
corrupted after the fact.

Tested (`tests/test_stage2_contracts.py`): identical duplicated
workload/constraints accepted; differing regime, request count,
prefill/decode tokens, memory margin, SLO, and throughput-floor
value/mode/baseline all individually rejected; the divergence is still
caught after a full JSON round trip (not only against the in-memory
object graph); a fully consistent manifest still validates end-to-end
after a round trip.

---

## 5. Provenance tri-state verification

Not redesigned, per this task's own instruction — re-verified, more
thoroughly than Gate A's own test did. Gate A's original test checked
one `True` and one `False` value surviving a round trip; this task adds
a parametrized test over all 9 combinations of
`(phase_filter_applied, block_table_fix_applied) ∈ {True, False, None}²`,
confirming each of the three states on each flag survives a full JSON
round trip as exactly itself (`is` comparison, not `==`, so a
serializer bug that turned `None` into `False` — both falsy under `==`
in some careless comparisons — would be caught), plus a second test
checking the raw JSON payload itself carries a literal `null`, not
`false`, for both flags when unset. All pass. `None` continues to mean
unknown/inapplicable/unproven; it is not coercible to `False` anywhere
in `tools/stage2/serialization.py`'s generic (de)serializer, which was
not modified for this task.

---

## 6. Schemas / examples changed

**No `tools/stage2/contracts.py` dataclass changed in this task** — the
`ffn_ep` fix lives entirely in `tools/planner_core.py`/`tools/planner.py`
(upstream of the contract layer), and the consistency validator is new
*behavior*, not a new *field*. Consequently: **no schema regeneration
was required or performed**; all four schema files are byte-identical
to Gate A's own. Schema major version stays `1.0` (no bump — nothing
additive was needed since nothing removed or changed shape).

**One example file was corrected**, and this was necessary work, not
optional polish: re-validating every existing example against the (now
also-checking-consistency) `validate_deployment_manifest` surfaced a
real, pre-existing defect in `planner_prediction_with_interval_manifest.json`
— its `workload.seed` was `None` under a streaming regime, which
`validate_workload_spec` (existing Gate A code, unchanged) correctly
rejects. This was not a new-in-A.1 bug; it existed since Gate A itself
and had gone uncaught only because Gate A's own test suite never ran
that specific file through the full manifest validator (it only checked
`prediction.uncertainty`, not `manifest.workload`). Fixed by setting
`workload.seed=0` — the real, fixed starting point of
`SimulationEvaluator`'s own `range(num_seeds)` seed sequence — in both
the manifest and its paired prediction's `provenance.seed`, alongside
the already-correct `num_seeds=3`; this is not a claim that only one
seed ran, since `num_seeds=3` remains present and correct alongside it.
No other example file needed correction. All eleven example files
(three real manifest/prediction pairs, the interval pair, two
synthetic `HardwareResult`s, one synthetic `DecisionValidation`)
re-validated against both the schema (`jsonschema.validate`) and this
project's own validators after the fix.

---

## 7. Tests

- `tests/test_ffn_ep_coverage_guard.py` (new, 10 tests): `profiled_ep`
  defaults to `(1,)`; `ffn_ep=1` always accepted regardless of
  `profiled_ep`; a known-valid `ffn_ep` accepted; an out-of-grid
  `ffn_ep` rejected via `can_evaluate() -> False` (not sent to
  Frontier); `attn_tp` gating unchanged; `attn_replicas` gating
  unaffected; `discover_profiled_ep` matches the real Phi-tiny-MoE-instruct
  grid and a real narrower grid (`mixtral_8x7b_moe`); raises on a
  missing model file and on no matching row.
- `tests/test_stage2_contracts.py` (+21 tests): the eight named
  workload/constraints consistency scenarios (§4) plus the divergence
  round-trip check and a consistent-manifest round-trip check; the
  9-combination provenance tri-state parametrization plus the raw-JSON
  `null`-not-`false` check (§5); the corrected interval example now
  validating end-to-end (§6).
- Full suite: 358 passed, 5 skipped (pre-existing, unrelated to this
  task), 113.1s. `check_import_direction.py`: clean (this task adds no
  file under `src/engine/`).

---

## 8. Unexpected regressions

**None in the pytest suite.** One **latent, pre-existing defect** was
found and fixed, not introduced: the interval example's invalid
streaming `workload.seed=None` (§6) — a Gate A-era gap this task's own,
more thorough validation pass surfaced, not a consequence of any Gate
A.1 code change. Reported here rather than silently patched without
comment, per this project's own standing convention of reporting
inaccuracies honestly.

Tasks 56–57's natural-split behavior is unchanged: `tools/planner_core.py`'s
own `enumerate_joint_arrangements`/`_relative_domain_placement` were not
touched, and `attn_a_ffn_b`'s real manifest (`relative="disjoint"`,
disjoint host sets) re-validates identically before and after this
task's changes (confirmed directly, §6).

---

## 9. Final readiness for the noise pilot

**Ready to proceed to the noise pilot on the terms Task 55 already
established** (zero real-hardware access in this sandbox; the pilot's
own tooling was prepared, not run, per that task's own explicit
instruction). This cleanup task closes exactly the three items it was
scoped to and finds the contract layer otherwise consistent with
itself: the `ffn_ep` coverage gap no longer lets an out-of-grid degree
reach a live subprocess un-flagged; the workload/constraints duplication
Gate A's own report flagged as a risk can no longer diverge without a
hard rejection, in-memory or after a JSON round trip; the provenance
tri-state contract that Task 53's own Fix A/Fix B status depends on is
re-confirmed intact under a much wider check than Gate A's own test
gave it. Nothing here changes Gate A's own final verdict (YES WITH
CONSTRAINTS) or its axis readiness table — this task closed one of the
report's own open findings (the `ffn_ep` coverage asymmetry, previously
listed as a caveat under "EP degree: READY WITH CONSTRAINTS") without
touching any of the others (`attn_replicas > 1` remains structurally
NOT READY; the throughput floor's relative-to-baseline gap remains
open; `ffn_replicas > 1` placement non-optimality remains open) — those
were out of this task's own stated scope and were not attempted.
