# Stage 2 — Gate C.1: profile installation/registration + Frontier evaluation validation

**HARD STOP per explicit instruction: all three TP=1/2/4 Frontier
evaluations failed.** Installation and coverage-checker steps
succeeded cleanly; the failure has a precise, confirmed, single root
cause, reported below. **No planner handoff was generated. No fix was
applied. No model/profile/TP was substituted. The result was not
patched.**

---

## 1. Pre-install snapshot (no conflict found)

Checked directly, fresh, before touching anything:

```
data/profiling/compute/mi355x/Qwen3-0.6B/     -- did not exist
data/config/models/Qwen3-0.6B.json            -- did not exist
```

No case/name variants found either (`find ... -iname "*qwen3-0.6b*"`
returned nothing in either location). **Zero conflict.** Frontier
checkout confirmed to be a real git repo, currently at commit
`e63fb4e181f4df2d361b3116328341cb9fc3d093` (2026-08-16) — recorded for
provenance (§6).

---

## 2. Install manifest

`artifacts/profile-install/qwen3-0.6b-mi355x-install-manifest.json` —
source → destination, SHA-256, row count, conflict-check status for
every file:

| source (local, combined from the real xai-3 collection) | destination | rows | sha256 (truncated) |
|---|---|---|---|
| `install_staging/linear_op.csv` | `data/profiling/compute/mi355x/Qwen3-0.6B/linear_op.csv` | 186 | `df884105...4761` |
| `install_staging/linear_op_kernel_only.csv` | `.../linear_op_kernel_only.csv` | 186 | `9cb54311...397b` |
| `install_staging/attention.csv` | `.../attention.csv` | 231 | `60dde564...fedfc` |
| `install_staging/attention_kernel_only.csv` | `.../attention_kernel_only.csv` | 231 | `eefb7af2...c231` |
| `install_staging/attention_combined.csv` | `.../attention_combined.csv` | 231 | `4aa73655...23d` |
| `install_staging/attention_combined_kernel_only.csv` | `.../attention_combined_kernel_only.csv` | 231 | `0a838b26...846d` |
| `install_staging/Qwen3-0.6B.json` | `data/config/models/Qwen3-0.6B.json` | — | `5961c241...937b` |

**`linear_op.csv`/`linear_op_kernel_only.csv` were combined from the
three separate real per-TP collection runs (TP=1, 2, 4) by column
*name*, not position** — the real header column *order* differs
between the TP=1 run and the TP=2/4 runs (a benign dict-ordering
artifact of the profiling tool itself, confirmed, not a data problem);
merging by name confirmed zero rows dropped or duplicated
(`58+64+64=186`, exact).

All seven files copied; **post-copy checksums verified identical to
the manifest** for every file (re-hashed at the destination, not
assumed).

---

## 3. Post-install coverage check (final location, not scratch)

```
Installed file: /work/simulation/Frontier/data/profiling/compute/mi355x/Qwen3-0.6B/linear_op.csv
Observed tp keys: {1: 58, 2: 32, 4: 32}
MISSING KEYS per tp (empty = fully covered):
  tp=1: missing=[]
  tp=2: missing=[]
  tp=4: missing=[]

COVERAGE CHECK: PASS -- zero missing keys, TP=1/2/4, against final installed Frontier location
```

**PASS**, run against the real file at its real, final location, not
a scratch copy.

---

## 4. `ModelSpec` construction

```python
model = ModelSpec(
    model_name="Qwen3-0.6B", total_experts=0, router_topk=0, is_moe=False,
    hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
    num_layers=28, head_dim=128,
    profiled_tp=(1, 2, 4),   # explicit -- NOT the (1,2,4,8) default, NOT inferred
)
```

`profiled_tp=(1, 2, 4)` passed explicitly, per your own instruction —
confirmed by reading `ModelSpec`'s own dataclass default (`(1, 2, 4,
8)`) and overriding it directly, not relying on inheritance.

---

## 5. Real Frontier evaluations — all three FAILED, same confirmed root cause

Ran `tools/planner.py::evaluate()` for real (a real subprocess
invocation of `frontier.simulator.Simulator`, `cwd=FRONTIER_ROOT`, the
same established mechanism every prior real Gate A/B/C evaluation in
this project has used), topology `domain8` (5 machines × 8 GPUs,
single-host shapes `attn_shape=(tp,)` for each), workload =
`Workload(num_requests=32, qps=4.0, prefill_tokens=5, decode_tokens=32)`
(Gate C's own frozen real workload), `feasible_num_blocks` computed
fresh per TP (`134624`/`269441`/`539075` — matching Gate C's own
earlier real memory-feasibility numbers exactly, confirming the model
config is being read correctly):

| TP | `attn_shape` | `feasible_num_blocks` | result |
|---|---|---|---|
| 1 | `(1,)` | 134624 | **FAILED** |
| 2 | `(2,)` | 269441 | **FAILED** |
| 4 | `(4,)` | 539075 | **FAILED** |

Every one of the three failed with the **identical** error:

```
ValueError: No data matches the filtering criteria in ./data/profiling/compute/mi355x/Qwen3-0.6B/linear_op.csv
Required tensor_parallel_size: <tp>
Available tensor_parallel_sizes: [1, 2, 4]
Please run profiling with the correct configuration.
```

### Root cause, confirmed precisely, not guessed

`shared_prediction_model_manager.py`'s own real filter (lines
2279-2306): after filtering by `num_tensor_parallel_workers`, it reads
`expected_use_qk_norm` from the evaluation's own `training_context`
(itself derived from `ModelConfig.from_model_name(...).use_qk_norm`)
and filters `filtered_df = filtered_df[filtered_df["use_qk_norm"] ==
expected_use_qk_norm]`. Confirmed live, directly, in this exact
sandbox:

```python
>>> ModelConfig.from_model_name('Qwen3-0.6B').use_qk_norm
False
```

**Every real row this task collected and installed has
`use_qk_norm=True`** (correctly — the QK-norm allowlist fix was
applied throughout the entire real collection, confirmed in every
prior report). The evaluation subprocess
(`tools/planner.py::_run_scenario`) calls
`install(fabric, placement, deployment, groups, binding=binding,
collective=True, sglang_replica_scheduler=True)` — **without**
`qk_norm_allowlist_fix=True`. Without that fix applied, the evaluation
process infers `use_qk_norm=False` for Qwen3-0.6B, and the exact-match
filter above then rejects **every single row** of the real, correctly-
collected profile, for **every** TP value identically (confirmed: the
error is not TP-specific — `tp=1` fails for the exact same reason as
`tp=2`/`tp=4`).

This is a real, narrow, already-diagnosed integration-wiring gap — the
*same*, already-tested, already-approved `qk_norm_allowlist_fix`
module this whole initiative has used since it was first found; it
was wired into every real profiling *collection* invocation this
task's own predecessors ran, but has never been wired into the
*evaluation* path (`tools/planner.py::_run_scenario`'s own `install()`
call) until this exact moment exposed the gap — because Qwen3-0.6B is
the first model in this project's own real evaluation history that
actually needs it (every other real evaluation used a non-Qwen3-family
model, per the memory record).

### Per §4 of your own instruction, checked explicitly for all three:

- **Profile lookup succeeds?** **No** — it raises before producing anything.
- **Exact-key `KeyError`?** **No** — this is a `ValueError` from the
  `use_qk_norm` exact-match filter, a different (but equally real)
  exact-match rejection, not the token-key lookup this project's own
  `gate_c1_coverage.py` module already guards.
- **`UNKNOWN` returned?** **No** — the evaluation errors out loudly,
  it does not silently classify the model as unevaluable and continue.
- **Silently-accepted extrapolation?** **No** — there is no
  extrapolation here at all; this is a hard, loud failure, which is
  the *correct*, safe behavior for a real mismatch, not a defect in
  the filter itself.
- **Finite, traceable prediction?** **No prediction was produced at
  all** — the pipeline stops before reaching one.

---

## 6. Provenance (as far as this task reached)

```
compatibility_stack: qk_norm/rope/rmsnorm/vllm_config_context/block_table -- all applied during COLLECTION
pinned_vllm_version: "0.27.1"
pinned_image_digest: "sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7"
frontier_commit: "e63fb4e181f4df2d361b3116328341cb9fc3d093"  # real, read directly from the checkout's own git HEAD
profiling_kernel_policy: {optimization_level: "O0", custom_ops: ["all"]}
rmsnorm_real_dispatch: "native (not a distinct hand-tuned kernel)"
model_id: "Qwen/Qwen3-0.6B"
model_revision: "c1899de289a04d12100db370d81485cdf75e47ca"
total_real_measurement_rows: 834
installed_at: "data/profiling/compute/mi355x/Qwen3-0.6B/, data/config/models/Qwen3-0.6B.json"
install_manifest: "artifacts/profile-install/qwen3-0.6b-mi355x-install-manifest.json"
coverage_check_final_location: "PASS -- zero missing keys, tp in {1,2,4}"
evaluation_status: "FAILED -- all three TP values, same root cause (evaluation-path use_qk_norm inference mismatch, SS5)"
planner_handoff: null  # not generated -- hard stop per instruction
```

---

## 7. Cleanup

No GPU was touched in this task (installation and evaluation are both
local, CPU-only operations in this sandbox — `frontier.simulator`/
`frontier.config` import and run without `torch`, confirmed). No
remote host state to clean up. No temporary evaluation artifacts
require removal beyond normal subprocess exit.

---

## Final answer

**Installation: SUCCESS.** No conflict, exact manifest, checksums
verified at destination, coverage check PASS at the final location.

**`ModelSpec` construction: SUCCESS**, `profiled_tp=(1,2,4)` explicit.

**All three TP=1/2/4 real Frontier evaluations: FAILED**, identically,
with a precisely confirmed root cause: the evaluation subprocess
(`tools/planner.py::_run_scenario`) does not apply the
`qk_norm_allowlist_fix` that every real collection run in this
initiative already applies, so it infers `use_qk_norm=False` for
Qwen3-0.6B while every real installed row correctly has
`use_qk_norm=True`, and the predictor's own exact-match filter
correctly, loudly rejects the mismatch rather than silently
guessing.

**Per your own explicit instruction: STOPPING here. No planner handoff
was generated. The result was not patched, no other model/profile/TP
was substituted, and no fix was applied without separate direction.**
The narrow, already-identified fix — adding
`qk_norm_allowlist_fix=True` to `_run_scenario`'s own existing
`install()` call in `tools/planner.py` — is named here as the specific
next step, not applied.
