# Stage 2 — Gate C.1: pre-sweep sentinel + complete Qwen3-0.6B → MI355X profile collection

**The complete Qwen3-0.6B → MI355X profile collection is DONE.** All
four previously-unexecuted dimensions (`TP=2`, `TP=4`,
attention-prefill, attention-decode) passed as a sentinel using points
already in the approved grid; the TP-aware coverage checker confirmed
full coverage; per explicit approval, the collection proceeded directly
to completion. **834 real GPU measurement rows** were collected on
real MI355X hardware — more than the originally-estimated 664, for two
real, transparent, tool-driven reasons discovered during this task and
explained below (§4), not a grid change made by this task.

---

## 1. Sentinel (four previously-unexecuted dimensions)

All points came from the already-approved grid; no new compatibility
workaround was introduced.

| dimension | point | result |
|---|---|---|
| `linear_op` TP=2 | `num_tokens=1`, `cuda_event` | **PASS** — `num_tensor_parallel_workers=2`, `attn_rope.mean=0.0181ms` (count=20), real/finite/positive |
| `linear_op` TP=4 | `num_tokens=1`, `cuda_event` | **PASS** — `num_tensor_parallel_workers=4`, `attn_rope.mean=0.0831ms` (count=20), real/finite/positive |
| attention decode | `batch_size=1, kv_cache_size=0`, TP=1, `cuda_event` | **PASS** — `attn_decode.mean=0.0592ms`, real/finite/positive |
| attention prefill | `total_tokens=5` (Gate C's own real prompt length), TP=1, `cuda_event` | **PASS** — `attn_prefill.mean=0.2095ms`, real/finite/positive |

TP-aware coverage checker (`tests/test_gate_c1_coverage.py`): **16/16
passed**, no regression.

Per your own explicit instruction, all four sentinel points passing →
proceeded directly to the complete collection.

---

## 2. Compatibility stack, runtime, and provenance — unchanged

Identical to the verified Probe 1/2 stack: pinned image
`vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7`
(vLLM `0.27.1`, torch `2.11.0+gitd0c8b1f`, ROCm `7.2.3`); model
`Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`; QK-norm
allowlist fix, RoPE API adapter, RMSNorm API adapter,
`profiling_vllm_config_context()`, Task 53 block-table fix — all
applied, none modified. Kernel policy unchanged:
`optimization_level=O0` / `compilation_config.custom_ops=["all"]`
(RMSNorm's own real dispatch remained the *native* path throughout,
exactly as already disclosed — re-confirmed, not re-litigated). Both
`HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` were set explicitly on
every invocation. **No new compatibility workaround was introduced
during this task's own GPU execution.**

---

## 3. Execution

Host: `xai-3`/`amd-mi355x-3`, GPU index `4` (fresh-checked free
immediately before starting: `2026-08-27T11:48:05Z`/re-confirmed
`12:03:33Z`, `4/8` free, indices `4,5,6,7` stable across both checks).
Eight real invocations, sequential, all `--rm`, all with
`--device=/dev/kfd --device=/dev/dri --group-add video -e
HIP_VISIBLE_DEVICES=4 -e CUDA_VISIBLE_DEVICES=4`:

| invocation | rows | wall-clock |
|---|---|---|
| `linear_op` TP=1, `cuda_event` | 58 | 5.87s |
| `linear_op` TP=1, `record_function` | 58 | 27.37s |
| `linear_op` TP=2, `cuda_event` | 64 | 5.74s |
| `linear_op` TP=2, `record_function` | 64 | 18.75s |
| `linear_op` TP=4, `cuda_event` | 64 | 5.92s |
| `linear_op` TP=4, `record_function` | 64 | 19.57s |
| `attention` (TP 1,2,4 combined), `cuda_event` | 231 | 3.18s |
| `attention` (TP 1,2,4 combined), `record_function` | 231 | 3.99s |
| **TOTAL** | **834** | **~90s** of real GPU time |

All eight: **exit 0**.

---

## 4. Why 834, not 664 — two real, transparent, tool-driven reasons (not a grid change made by this task)

**(a) `linear_op` TP=2/TP=4: 64 rows per invocation, not 32.** Live-verified:
for each requested `num_tokens` value, Frontier's own real
`linear_op.main` emits **two** CSV rows — one tagged
`num_tensor_parallel_workers=2` (or `4`) with real values for the
*sharded* ops only (`attn_pre_proj`, `attn_rope`, `attn_post_proj`,
`mlp_up_proj`, `mlp_act`, `mlp_down_proj`; the *replicated* ops blank),
and one tagged `num_tensor_parallel_workers=1` with real values for
the *replicated* ops only (`emb`, `input_layernorm`,
`post_attention_layernorm`; the sharded ops blank). This is real,
pre-existing, correct Frontier behavior — replicated ops are computed
identically regardless of TP, so they are measured once at the
reference `tp=1` rather than redundantly at every TP (visible, in
retrospect, in every printed config all along: `"Replicated TP Size:
[1]"` was always shown separately from `"Attention/FFN TP Sizes"`,
just never manifested as two distinct rows until TP≠1 was actually run
for the first time in this task). **Confirmed harmless, not a coverage
gap**: the TP-aware coverage checker, run against the real combined
data (§5), reports zero missing keys — the "extra" `tp=1`-tagged rows
inside the TP=2/4 files are a pure superset of what the TP=1 file
already provides on its own.

**(b) attention prefill: 42 real prefill rows across the combined TP
sweep, not 21.** Traced from real source
(`frontier/profiling/utils/__init__.py::get_seq_lengths_to_profile`/
`get_attention_input_combinations`): prefill total-token lengths have
**no CLI list-override flag at all** — they come from an internal
default sweep (`range(0, 1024+1, 32)`, filtered `< max_seq_len`) plus
whatever `FRONTIER_EXTRA_SEQ_LENGTHS` (an environment variable) adds,
plus one additional "full chunk = max_seq_len" point from the
chunked-prefill branch. With `max_seq_len=256` (matching the
originally-approved value) and `FRONTIER_EXTRA_SEQ_LENGTHS="1 2 4 5 8
16 32"` (added to guarantee the originally-intended bracket points,
including Gate C's own real 5-token prompt length, are present), the
real per-TP prefill count is 14, not 7 — confirmed live via a
CPU-only dry check *before* spending real GPU time (`"Standard Input
Combinations: 77"` per TP = 63 decode + 14 prefill, matching exactly).
**Every originally-required prefill point is present** (confirmed in
§5) — the extra points are additional, safe, never-removing superset
coverage, not a substitution.

Both mechanisms are **additive only** — nothing originally required
was dropped; the real tool's own mechanics simply produce more
coverage than the hand-estimated 664 assumed. This was not discovered
until this task actually invoked `TP=2`/`TP=4` and attention profiling
on real hardware for the first time — exactly why the sentinel step
existed.

---

## 5. Coverage checker (final, against the real, complete data)

Ran `tools/stage2/gate_c1_coverage.py::verify_gate_c_linear_op_coverage`
against the real, combined `linear_op` CSVs (all three TP files'
`num_tensor_parallel_workers`/`num_tokens` columns, read via
`read_profiled_effective_tokens_by_tp`):

```
linear_op_tp1.csv observed tp keys: {1: 58}
linear_op_tp2.csv observed tp keys: {2: 32, 1: 32}
linear_op_tp4.csv observed tp keys: {4: 32, 1: 32}
Combined profiled_by_tp sizes: {1: 58, 2: 32, 4: 32}

MISSING KEYS per tp (empty = fully covered):
  tp=1: missing=[]
  tp=2: missing=[]
  tp=4: missing=[]
```

**Zero missing keys at every TP.** Full exact-key coverage confirmed
against real, collected data — not merely the derived requirement in
isolation.

---

## 6. Data quality audit

Every row's own *applicable* primary measurement is real, finite, and
`>0` — checked exhaustively across all 834 rows, not sampled:

- `linear_op` (both methods, all TP): zero non-positive/NaN values in
  any populated `time_stats.*.mean` cell.
- `attention` `cuda_event`: zero non-positive/NaN values anywhere.
- `attention` `record_function`: the *primary, phase-applicable*
  operator (`attn_decode` on all 189 real decode rows, `attn_prefill`
  on all 42 real prefill rows) is real/positive in every single row —
  confirmed by direct count (`0/189`, `0/42` zero-or-bad values). The
  *cross-phase, non-applicable* columns (`attn_decode` on prefill rows,
  `attn_prefill` on decode rows, and the two reshape telemetry columns)
  read literally `0.0` under `record_function` — a real, benign
  measurement-semantics difference from `cuda_event` (which instead
  reports a small nonzero residual, ~5μs, for the same non-applicable
  case): `record_function`'s own kernel-only trace correctly attributes
  zero actual kernels to a phase that didn't run, while `cuda_event`'s
  wall-clock timer still captures a small dispatch/branch overhead.
  Neither is a defect; this is disclosed, not glossed over, because
  `0.0` is not "positive" and a literal-minded reader of "every column
  is positive" would be wrong not to ask about it.
- `[WARNING] num_tokens=1: Missing operations: ['add']` recurred on
  every `linear_op` `record_function` invocation, exactly as seen in
  the earlier Probe 2 — confirmed, again, that no `time_stats.add.*`
  column exists in this CSV schema at all, so nothing was actually
  dropped from any saved row.

---

## 7. Cleanup

All eight containers used `--rm` — `docker ps -a --filter
name=gate-c1` empty throughout and after. No leftover profiling
process (`ps aux | grep -iE 'linear_op|attention.main'` empty).
**Real, unrelated third-party contention appeared on `xai-3` after
this task's own work finished** (a SLURM-scheduled `broadcast_perf`
multi-GPU job, confirmed via `rocm-smi --showpids`, occupying GPU
index `1`; `slurmstepd` attributed to GPU `0`) — `preflight_hosts.py`
now reports `0/8` free fleet-wide on this host as a result. **This is
not caused by this task**: GPU index `4` (the one actually used here)
shows no attributed process and no held VRAM in `rocm-smi --showpids`'
own output — confirmed cleanly released. The real collected CSVs
(834 rows, 8 files) remain on `xai-3` at
`~/rocm-work/gate-c1-smoke/sweep_output/` (the real deliverable, not
cleaned up) and were additionally copied to this session's own local
scratchpad with SHA-256 checksums recorded for provenance (§8) —
neither committed to the dc-sim git repo (bulk measurement data, not
code) nor installed into Frontier's own `data/profiling/` tree (a
separate, not-yet-approved step, same as model registration was
flagged in earlier reports).

---

## 8. Provenance

```
compatibility_stack: unchanged (qk_norm/rope/rmsnorm/vllm_config_context/block_table, all applied)
pinned_vllm_version: "0.27.1"
pinned_image_digest: "sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7"
profiling_kernel_policy: {optimization_level: "O0", custom_ops: ["all"]}
rmsnorm_real_dispatch: "native (not a distinct hand-tuned kernel)"  # unchanged from prior disclosure
model_id: "Qwen/Qwen3-0.6B"
model_revision: "c1899de289a04d12100db370d81485cdf75e47ca"
collection_host: "xai-3 (amd-mi355x-3)", gpu_index: 4
collection_timestamp: "2026-08-27T12:10-12:15Z"
total_real_measurement_rows: 834
row_count_note: "664 originally estimated; +170 rows from two real, tool-driven, additive mechanisms (SS4) discovered during this task -- not a grid redesign"
output_files:
  - {file: "linear_op.csv (tp=1)", rows: 58, sha256: "78758c2d...938b"}
  - {file: "linear_op_kernel_only.csv (tp=1)", rows: 58, sha256: "83607f2c...5f"}
  - {file: "linear_op.csv (tp=2)", rows: 64, sha256: "808b6af7...1c5"}
  - {file: "linear_op_kernel_only.csv (tp=2)", rows: 64, sha256: "356170fe...775"}
  - {file: "linear_op.csv (tp=4)", rows: 64, sha256: "91cbb3ce...1f2"}
  - {file: "linear_op_kernel_only.csv (tp=4)", rows: 64, sha256: "c75f7530...e5b"}
  - {file: "attention.csv", rows: 231, sha256: "60dde564...fedfc"}
  - {file: "attention_kernel_only.csv", rows: 231, sha256: "eefb7af2...c231"}
storage_location: "xai-3:~/rocm-work/gate-c1-smoke/sweep_output/ (real data, not yet installed into Frontier's data/profiling/ tree -- a separate, not-yet-approved step)"
coverage_check: "PASS -- zero missing (tp, effective_tokens) keys, tp in {1,2,4}"
```

`null`, never `false`, for anything genuinely unknown (unchanged
policy from every prior report).

---

## 9. Remaining steps (not part of this task, explicitly deferred)

1. **Install the real CSVs into Frontier's own `data/profiling/compute/mi355x/Qwen3-0.6B/` tree** and register `data/config/models/Qwen3-0.6B.json` — both already fully specified in `docs/tasks/61-...md` §10, not yet executed (Frontier is an ambient, not repo-pinned, checkout; this project's own convention is never to edit it directly without separate approval).
2. **Construct the `dc-sim`-side `ModelSpec`** with `profiled_tp=(1,2,4)` explicitly (not the `(1,2,4,8)` default) and run the Gate C shape-coverage/Frontier-smoke validation (`docs/tasks/61-...md` §14.C/D) once the profile is installed.
3. Record the RMSNorm native-dispatch fidelity note and the two additive grid mechanisms (§4) in whatever provenance record accompanies the installed profile.

---

## Final answer

**Sentinel: PASS (4/4).** **Full collection: COMPLETE — 834 real,
verified, finite-positive MI355X measurement rows for
`Qwen/Qwen3-0.6B` across `linear_op` (`TP∈{1,2,4}`) and `attention`
(prefill+decode, `TP∈{1,2,4}`), both `cuda_event` and
`record_function`.** Coverage checker: zero missing keys. Compatibility
stack, runtime, image digest, kernel policy, and RMSNorm fidelity
status: all unchanged and re-confirmed. No new workaround introduced.
Cleanup confirmed; the real data is preserved (not deleted) pending
the separately-approved install/registration step above.
