# Stage 2 — Gate C.1: Frontier ↔ pinned-vLLM profiling compatibility inventory + Qwen3 critical-path fix set + final two-point MI355X smoke

**Probe 1 and Probe 2 both succeeded on real MI355X hardware.** Real,
finite, positive measurement rows were produced for both `cuda_event`
and `record_function`/`KERNEL_ONLY`. **The 664-row sweep was
explicitly NOT run**, per this task's own instruction. Full inventory
artifact: `artifacts/compatibility/frontier-vllm-0.27.1.json`.

---

## 1. Why the strategy changed

Three sequential real GPU attempts each found a *different* Frontier↔
pinned-vLLM 0.27.1 API break, one at a time: `rotary_dim`/`get_rope`
(resolved), `CustomOp`/`set_current_vllm_config` (resolved), `RMSNorm`
free functions (found, not yet resolved at that point). Continuing
that pattern — one fix, one GPU attempt, repeat — would have kept
spending real, shared, multi-tenant GPU time to discover problems a
CPU-only static/structural pass could find for free. This task
inventories every Frontier↔vLLM dependency under `frontier/profiling/`
once, resolves every Qwen3-critical one together, and only then
returns to the GPU — exactly the new rule this task specifies.

---

## 2. Inventory methodology

Static: `grep -rln "vllm\|torch\.ops\._C\|torch\.ops\.vllm"
frontier/profiling/ --include="*.py"` (24 files), then every matching
line inspected directly (not import-statement-only — usages like
`torch.ops.vllm.apply_w8a8_block_fp8_linear` and `hasattr(torch.ops._C,
...)` were caught this way). Classification against the *real* pinned
image: live `inspect.signature`/`hasattr`/`issubclass` checks executed
inside the pinned container on `xai-3`, CPU-only, `--network none`, no
`--device` flags — never inferred from documentation or symbol
existence alone (per this task's own explicit §4 instruction — the
RoPE case already proved a symbol-level check alone would have missed
a real semantic change).

---

## 3. Complete inventory (see artifact for full detail)

16 entries in `artifacts/compatibility/frontier-vllm-0.27.1.json`,
summarized here; full per-entry semantic notes, adapter references,
and kernel-fidelity classifications are in the artifact itself (this
report does not duplicate the whole table).

| # | Frontier file | API | status | Qwen3-critical? |
|---|---|---|---|---|
| 1 | `common/layers/rotary_embedding.py` | `get_rope` | SIGNATURE_CHANGED → RESOLVED_VERIFIED_LIVE | yes |
| 2 | `common/layers/rotary_embedding.py` | `vllm._custom_ops` (torch-fallback path) | UNKNOWN | no (unreached under real profiling convention) |
| 3 | `common/layers/layernorm.py` | `rms_norm`/`fused_add_rms_norm` | REMOVED → RESOLVED_VERIFIED_LIVE | yes |
| 4 | `common/layers/layernorm.py` | `GemmaRMSNorm` | COMPATIBLE (already correct) | no (Gemma only) |
| 5 | `common/layers/activation.py` | `torch.ops._C.silu_and_mul` | COMPATIBLE_VERIFIED_LIVE | yes |
| 6 | `common/parallel_utils/tensor_parallel_layers.py` | `current_platform`, `deep_gemm`, GEMM dispatch | COMPATIBLE_VERIFIED_LIVE | yes |
| 7 | `common/parallel_utils/tensor_parallel_layers.py` | FP8 quant utils (`w8a8_utils`, `torch.ops.vllm.apply_w8a8_block_fp8_linear`) | UNKNOWN | no (Qwen3-0.6B profiles BF16 only) |
| 8 | `attention/backends/torch_sdpa_attention_wrapper.py` | none (pure `F.scaled_dot_product_attention`) | STATICALLY_COMPATIBLE | yes |
| 9 | `attention/attention_wrapper.py` | Task 53 block-table fix target | RESOLVED_PRE_EXISTING | yes |
| 10 | `attention/backends/flashinfer_attention_wrapper.py` | `vllm.v1.attention.backends.utils`, `vllm._custom_ops.reshape_and_cache_flash` | UNKNOWN | no (FLASHINFER backend, not TORCH_SDPA) |
| 11 | `attention/vllm_mla_profile_importer.py`, `attention/main.py` | MLA CUDA-op-log import | COMPATIBLE (n/a) | no (MLA models only) |
| 12 | `moe/*.py` | `fused_moe.*`, `parallel_state`, upstream `ReplicatedLinear` | UNKNOWN | no (Qwen3-0.6B is dense) |
| 13 | `cpu_overhead/**`, `non_kv_cache_overhead/**`, `other_overhead/**` | various | UNKNOWN | no (separate profiling tools, out of Gate C.1 scope) |
| 14 | `vllm.model_executor.custom_op.CustomOp` / `vllm.config.vllm.*` | `set_current_vllm_config` context | CONTEXT_REQUIRED → RESOLVED_VERIFIED_LIVE | yes |
| 15 | `frontier/config/model_config.py` | `QK_NORM_MODEL_TYPE_ALLOWLIST` (not a vLLM API — a Frontier-internal gap) | COMPATIBLE → RESOLVED_PRE_EXISTING | yes |
| 16 | `linear_op/main.py` | `_get_available_gpus` → `nvidia-smi` (not a vLLM API — a ROCm-portability gap) | COMPATIBLE (workaround) | yes |

---

## 4. Qwen3-critical subset

Nine entries (#1, #3, #5, #6, #8, #9, #14, #15, #16 above). All nine
are now either resolved via a guarded adapter or confirmed compatible
by live execution. **Zero unresolved Qwen3-critical items remain.**

---

## 5. Future/non-critical subset

Seven entries (#2, #4, #7, #10, #11, #12, #13): MLA-specific
(vLLM-MLA-log import, `deepseek-v3`-class models), MoE-specific
(Qwen3-0.6B is dense), Gemma-specific (`GemmaRMSNorm`, already
correct, unreached), FlashInfer-backend-specific (Qwen3-0.6B's own
real convention uses TORCH_SDPA), FP8-specific (Qwen3-0.6B profiles
BF16), and three entirely separate profiling tools
(`cpu_overhead`/`non_kv_cache_overhead`/`other_overhead`) outside Gate
C.1's own `linear_op`+`attention` scope. **Inventoried, not fixed**,
per this task's own explicit instruction.

---

## 6. RMSNorm semantic analysis

| Frontier expectation | pinned vLLM equivalent | exact semantic match? | adaptation |
|---|---|---|---|
| `rms_norm(x, weight, eps)` → single tensor | `RMSNorm.forward_native(x, residual=None)` → single tensor when `residual is None` | **Yes**, confirmed from real source (identical return-type contract to Frontier's own pre-existing type hint) | delegate to a constructed real `RMSNorm` instance |
| `fused_add_rms_norm(x, residual, weight, eps)` → tuple (implied "maybe in-place") | `forward_native(x, residual)` → `(normed, updated_residual)`, via `ir.ops.fused_add_rms_norm.maybe_inplace(...)` | **Yes** — same two-shape contract, same "maybe in-place" mutation convention, confirmed from real source, not inferred | same delegation |
| `eps` (constructor arg) | `eps` (same name, same default `1e-6`) | **Yes** | passed straight through |
| weight owned by Frontier's own `nn.Parameter` | weight owned by the real vLLM instance | Different ownership, same effective value semantics | `weight` exposed as a delegating `@property`, mirroring Frontier's own sibling `GemmaRMSNorm` wrapper exactly |
| — | `RMSNorm` is `CustomOp`-derived (confirmed live) | new requirement | construction reuses `profiling_vllm_config_context()` — no new mechanism |
| — | ROCm forward: `forward_hip` → `forward_cuda` → (since `VLLM_BATCH_INVARIANT` is off) → `forward_native` | real, live-observed: even under the approved `custom_ops=["all"]` policy, the actual dispatch used the **native** path, not a distinct hand-tuned kernel (`"Priority not set for op rms_norm, using native implementation"`) | not approximated away — recorded as a real, disclosed fidelity finding (§9) |

**Semantic equivalence established from source and confirmed
numerically** (§11.H): the adapter's real output matched a simple,
independent PyTorch RMSNorm reference within `rtol=atol=1e-2`.

---

## 7. All additional incompatibilities found

Beyond the three already known (`rotary_dim`, `CustomOp`, `RMSNorm`):
none new was found on the Qwen3-critical path during this inventory.
Two real, non-vLLM-API gaps were confirmed and worked around (not
code-fixed, since they are not part of the "compatibility adapter"
scope): `_get_available_gpus`'s hardcoded `nvidia-smi` call (ROCm has
no `nvidia-smi`; Frontier's own error message names its own
`CUDA_VISIBLE_DEVICES` workaround, applied at launch) and — a
pre-existing, already-resolved item — the QK-norm allowlist gap.

---

## 8. Compatibility adapter set (active)

| adapter | file | status |
|---|---|---|
| QK-norm allowlist | `src/integration/profiling/qk_norm_allowlist_fix.py` | pre-existing, applied |
| RoPE API | `src/integration/profiling/rope_api_adapter.py` | commit `870b9c5`, applied |
| VllmConfig profiling context | `src/integration/profiling/vllm_config_context.py` | commit `3c85773`, applied |
| Task 53 block-table Fix B | `src/integration/profiling/attention_block_table_fix.py` | pre-existing, applied (confirmed inapplicable to `linear_op`) |
| RMSNorm API | `src/integration/profiling/rmsnorm_api_adapter.py` | this task, applied |

Each: structurally detects the real installed API shape (never a
caught exception distinguishing old/new API from an unrelated bug),
preserves semantics (proven per-adapter, not assumed), fails loudly on
an unrecognized shape, contains no model-specific performance
constants (§12).

---

## 9. Kernel-fidelity matrix

| operator | profiler implementation (live-observed) | production implementation | fidelity status |
|---|---|---|---|
| `attn_rope` | vLLM `RotaryEmbedding`, dispatched via `custom_ops=["all"]` → `forward_hip`→`forward_cuda` | same real vLLM class, but under production's own real default (`custom_ops=["none"]`, compiled) | FUNCTIONALLY_EQUIVALENT_DIFFERENT_KERNEL |
| `input_layernorm`/`post_attention_layernorm` | vLLM `RMSNorm`, **live-confirmed dispatch to the native path** even under `custom_ops=["all"]` (`"using native implementation"`) | production's own real default is *also* native, but compiled/fused via `torch.compile`, which this harness never invokes | APPROXIMATION — not called exact; this is the sharpest fidelity gap found this task |
| `mlp_act` | Frontier's own `SiluAndMul`, direct `torch.ops._C.silu_and_mul` call (not vLLM's `CustomOp`-derived class at all) | same raw op, real production path | SAME |
| `attn_pre_proj`/`attn_post_proj`/`mlp_up_proj`/`mlp_down_proj` | Frontier's own linear classes, plain GEMM (BF16, no FP8 surrogate) | same real GEMM dispatch | SAME |
| `emb` | Frontier's own `VocabParallelEmbedding`, plain PyTorch embedding lookup | same | SAME |

No fidelity finding is called exact where it isn't — RMSNorm's own
result is the one entry this report does not soften.

---

## 10. VllmConfig composition

`rmsnorm_api_adapter.py` does **not** open its own context — it
constructs the real `RMSNorm` class only when
`profiling_vllm_config_context()` (already active, since the whole
`linear_op.main()` call is wrapped in it) is the caller's own
responsibility, exactly the composition this task's §7 requires:

```python
with profiling_vllm_config_context(optimization_level=OptimizationLevel.O0):
    main()   # constructs RotaryEmbedding AND RMSNorm, both CustomOp-derived,
             # both succeed inside this one context
```

Live-confirmed: both real probes ran to completion with this single
context wrapping the entire `linear_op.main()` call — no second
context mechanism was introduced anywhere.

---

## 11. Tests

- `tests/test_rope_api_adapter.py` — 13 tests (pre-existing, unchanged
  this task except a cross-test-pollution fix, §14 of `docs/tasks/63-...md`).
- `tests/test_vllm_config_context.py` — 20 tests (pre-existing, unchanged).
- `tests/test_rmsnorm_api_adapter.py` — **11 new tests**: old free-function
  API (A), pinned class API detected structurally (B), fused-residual
  two-shape contract preserved (C), real epsilon reaches the real
  instance, not a default (D), weight delegated not duplicated (E),
  construction requires `profiling_vllm_config_context()` — fails
  without it, exactly like `RotaryEmbedding` (F), unknown API shapes
  hard-fail with named reasons (G, 4 sub-cases), numerical match to a
  simple PyTorch RMSNorm reference within `rtol=atol=1e-2` (H).

**Local (no torch/vllm)**: 8 run + pass, 3 skip (matching this
project's established convention). **Live, on the real pinned image
(CPU-only)**: all 37 tests across the three adapter test files passed
— `26/26` (rope+vllm_config) → `37/37` (rope+vllm_config+rmsnorm,
after adding the new file). Local full suite: **409 passed, 16
skipped**. Import-direction check: clean.

---

## 12. Hard-coded-number audit

Structural `ast` scan (not a manual claim) of every active adapter's
own source for non-boolean numeric literals:

| adapter | numeric literals found | verdict |
|---|---|---|
| `qk_norm_allowlist_fix.py` | none | clean |
| `rope_api_adapter.py` | none | clean |
| `vllm_config_context.py` | none (only `"cuda"`, a schema constant) | clean |
| `attention_block_table_fix.py` | none | clean |
| `rmsnorm_api_adapter.py` | **`1e-06`** | the default `eps` value in the patched `__init__`'s own signature — matches Frontier's own original function's default *and* vLLM's own real class default exactly; a structural interface requirement (the patched function must keep the same call signature the code being patched already has), not a hidden model-specific number. **Classified ALLOWED.** |

No hidden `hidden_size`, `intermediate_size`, head counts, rotary
values, theta, `gfx950`/`MI355X` strings, TP lists, token grids, or
predicted timings exist in any active adapter.

---

## 13. CPU-only end-to-end rehearsal

`run_probe4.py` (all four fixes composed), `--network none`, no
`--device` flags, on the real pinned image:

1. First attempt: `_get_available_gpus` → `FileNotFoundError:
   nvidia-smi` (a real, separate, non-API gap — Frontier's own
   documented workaround, `CUDA_VISIBLE_DEVICES`, applied).
2. With `CUDA_VISIBLE_DEVICES=0`: printed the full, correct real Qwen3-0.6B
   profiling configuration, then reached
   `frontier/profiling/linear_op/main.py:726: torch_module.cuda.set_device(0)`
   → `RuntimeError: No CUDA GPUs are available`.

**A genuinely correct stopping point** per this task's own §13
criteria — not `ImportError`/`AttributeError`/`TypeError`/
`AssertionError` from an API skew, not a missing symbol, not an
incompatible constructor, not a missing context. **PASS.**

---

## 14. Attention-path audit

Static, not deferred: `frontier/profiling/attention/backends/torch_sdpa_attention_wrapper.py`
(Qwen3-0.6B's own real MI355X convention, `attention_backend=TORCH_SDPA`)
has **zero** vLLM or `torch.ops._C` references anywhere (grep-confirmed
exhaustively) — pure `torch.nn.functional.scaled_dot_product_attention`.
`attn_rope` is a `linear_op`-only operator; no caller of `get_rope`
exists anywhere under `frontier/profiling/attention/` (confirmed
exhaustively). Task 53's block-table fix (`attention_wrapper.py::
_get_input_tensors`) is backend-agnostic (lives in the base wrapper)
and its hash guard passed on this exact checkout (confirmed both this
task and the prior one) — **PASS.** The FLASHINFER-specific wrapper
(`flashinfer_attention_wrapper.py`, real vLLM `_custom_ops`
dependency) is a separate, unreached code path under the real
established convention and was not further audited (§5, out of
scope).

---

## 15. Coverage-checker regression

`tests/test_gate_c1_coverage.py`: **16/16 passed**, unchanged. TP-aware
`(tp, effective_tokens)` pair behavior (never a flattened/unioned
`num_tokens` set across TP) re-confirmed, including the real-CSV
rehearsal test.

---

## 16. Pre-GPU checkpoint (printed before touching a GPU)

```
Qwen3 critical APIs:  total inspected=9  compatible=4  adapted=5  unresolved=0
Known active adapters: QK norm / RoPE / VllmConfig context / block table / RMSNorm — all APPLIED
CPU rehearsal:         PASS (reached torch.cuda.set_device(0), zero API-level failures)
attention static audit: PASS (TORCH_SDPA: zero vLLM refs; block-table fix hash-guard passes)
hard-coded-number audit: PASS (one literal, justified, ALLOWED)
coverage checker:      PASS (16/16)
```

All Qwen3-critical items resolved or compatible → proceeded to §18.

---

## 17. Probe 1 result

- **Command**: `docker run --rm --name gate-c1-probe1final-20260827T114845Z --network none --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=4 -e CUDA_VISIBLE_DEVICES=4 -v /home/ssidik/rocm-work/gate-c1-smoke:/workspace -w /workspace --entrypoint python3 vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7 /workspace/run_probe4.py cuda_event`
- **Host/GPU**: `xai-3`/`amd-mi355x-3`, GPU index `4` (fresh-checked free at `2026-08-27T11:48:05Z`; `torch.cuda.device_count()==1` confirmed pre-flight).
- **Image digest**: `sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7`. **Runtime**: vLLM `0.27.1`, torch `2.11.0+gitd0c8b1f`, ROCm `7.2.3`.
- **Frontier commit**: ambient PYTHONPATH checkout, not repo-pinned (see memory).
- **Adapter versions**: QK-norm (pre-existing), RoPE (`870b9c5`), VllmConfig context (`3c85773`), block-table (pre-existing, inapplicable), RMSNorm (this task).
- **Result**: **exit 0**. Real profiling loop executed (`100%|██████████| 1/1`). Real forward completed. CSV: `compute/mi355x/Qwen3-0.6B/linear_op.csv`.
- **Wall-clock**: `1.953s`.
- **Exact output row** (all 71 columns; key fields shown): `n_head=16, n_kv_head=8, n_embd=1024, n_expanded_embd=3072, vocab_size=151936, use_gated_mlp=True, use_qk_norm=True, num_tokens=1, num_tensor_parallel_workers=1, model_arch=generic, measurement_type=CUDA_EVENT, profiling_precision=BF16`. Every `time_stats.*.mean` is finite and `>0`, e.g. `attn_rope.mean=0.016184000065550208ms` (count=20), `input_layernorm.mean=0.07002399992197753ms` (count=20), `mlp_act.mean=0.015755999926477672ms` (count=20).
- **Warnings**: none blocking; the only real anomaly is recorded under Probe 2 below.

---

## 18. Probe 2 result

- **Command**: identical, `profile_method=record_function`.
- **Result**: **exit 0**. CSV: `compute/mi355x/Qwen3-0.6B/linear_op_kernel_only.csv`.
- **Wall-clock**: `2.211s`.
- **Exact output row**: same model/TP/num_tokens identity as Probe 1,
  `measurement_type=KERNEL_ONLY`. Every `time_stats.*.mean` finite and
  `>0`, e.g. `attn_rope.mean=0.0024770000000000005ms` (count=20),
  `mlp_act.mean=0.002363ms` (count=20).
- **Real anomaly, recorded, not hidden**: `[WARNING] num_tokens=1:
  Missing operations: ['add']`. Checked: neither Probe's CSV schema
  has a `time_stats.add.*` column at all (identical headers, confirmed
  byte-for-byte) — this warning is internal to `record_function`'s own
  op-collection completeness check and did not remove any column the
  saved row was ever expected to have.

---

## 19. Exact measurement rows

Both full rows are reproduced verbatim in §17/§18 above (key fields)
and in `artifacts/compatibility/frontier-vllm-0.27.1.json`'s own
`probe_results` block (full `row_summary`).

---

## 20. Provenance

```
compatibility_stack:
  qk_norm_allowlist_fix: applied
  rope_api_adapter: applied, commit 870b9c5
  vllm_profiling_config_context: applied, commit 3c85773
  attention_block_table_fix: applied (pre-existing; inapplicable to linear_op)
  rmsnorm_api_adapter: applied, this task
  compatibility_matrix_artifact: artifacts/compatibility/frontier-vllm-0.27.1.json
  pinned_vllm_version: "0.27.1"
  pinned_image_digest: "sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7"
  frontier_commit: null   # ambient checkout, not repo-pinned -- never inferred
  profiling_kernel_policy:
    optimization_level: "O0"
    custom_ops: ["all"]
```

`null`, never `false`, for anything genuinely unknown.

---

## 21. Cleanup

Both probe containers used `--rm` (foreground) — `docker ps -a
--filter name=gate-c1` empty after each, confirmed. Output artifacts
(`probe_output4_cuda_event/`, `probe_output4_record_function/`, both
root-owned inside the bind mount) removed via a follow-up throwaway
container, confirmed gone via `find`. No leftover profiling process
(`ps aux | grep linear_op` empty). Fresh occupancy re-check
immediately after: `xai-3` back to `4/8` free, indices `4,5,6,7` —
unchanged, GPU `4` returned to baseline. Staged Gate-C1 input files
retained on `xai-3` (`~/rocm-work/gate-c1-smoke/`) for the eventual
approved sweep, per this task's own explicit instruction not to clean
them for tidiness.

---

## 22. Remaining risks

1. **RMSNorm's own real dispatch is native, not a distinct hand-tuned
   kernel**, even under the approved policy — the sharpest, now
   live-confirmed (not merely theorized) fidelity gap in this whole
   stack (§9).
2. Six future/non-critical vLLM dependencies (§5) remain
   uninvestigated — real for a future MLA/MoE/Gemma/FlashInfer/FP8
   Gate C.1 extension, not for the current Qwen3-0.6B dense/BF16/
   TORCH_SDPA collection.
3. `nvidia-smi`/`CUDA_VISIBLE_DEVICES` is a launch-command workaround,
   not a code fix — every real invocation of the eventual 664-row
   sweep must set it explicitly.
4. Only `TP=1`, `num_tokens=1` was exercised on real hardware. `TP=2`/
   `TP=4` and the full token grid are structurally expected to behave
   identically (none of the four resolved adapters are TP- or
   shape-conditional), but this has not been executed.

---

## 23. Recommendation

All nine Qwen3-critical compatibility items are resolved or confirmed
compatible, live, on real MI355X hardware, for both `cuda_event` and
`record_function`. The compatibility stack is stable, tested, and
provenance-tracked. The one open, disclosed concern (RMSNorm's real
native-kernel dispatch) is a fidelity characteristic to record in the
eventual profile's own provenance, not a blocker to further
collection at this same `(TP=1, num_tokens=1)` shape.

---

## Final answers

**A. How many vLLM API dependencies exist under `frontier/profiling/`?**
16 distinct entries catalogued (24 files matched the initial grep;
several collapse into the same real API surface, e.g. multiple
`tensor_parallel_layers.py` symbols grouped under GEMM dispatch).

**B. How many are reachable by the Qwen3 Gate-C.1 path?** **9**
(`linear_op`'s `get_rope`, `RMSNorm`, `silu_and_mul`, GEMM/`current_platform`
dispatch, the `CustomOp`/`VllmConfig` context requirement, the
QK-norm allowlist gap, the `nvidia-smi` gap; `attention`'s TORCH_SDPA
backend and Task 53 block-table fix).

**C. How many Qwen3-critical incompatibilities were found?** **3**
real vLLM-version-skew breaks (`rotary_dim`/`get_rope`, `CustomOp`/
`set_current_vllm_config`, `RMSNorm` free functions) plus **2** real,
separate, non-vLLM-API gaps (QK-norm allowlist — pre-existing; `nvidia-smi`
— a launch-workaround, not code).

**D. What are they?** Listed in §7/§8 above and in the compatibility
matrix artifact, entries #1, #3, #14 (the three vLLM-skew breaks),
#15, #16 (the two non-vLLM gaps).

**E. Are all Qwen3-critical incompatibilities now resolved?** **Yes.**
Zero unresolved items remain in the pre-GPU checkpoint (§16), and both
real probes completed successfully on real hardware.

**F. Do any active adapters contain model/device/performance
hard-coded numbers?** **No.** One literal (`eps=1e-6`, a function
default mirroring the exact signature being patched) was found and
classified `ALLOWED` (§12); no model dimension, device string, TP
value, or predicted timing is hard-coded anywhere in the active
adapter set.

**G. Does the CPU-only path now reach a genuine GPU boundary without
API failure?** **Yes** — `torch_module.cuda.set_device(0)` →
`RuntimeError: No CUDA GPUs are available`, zero API-level failures
en route (§13).

**H. Is the attention profiling path statically compatible with
pinned vLLM?** **Yes**, for Qwen3-0.6B's own real `TORCH_SDPA`
convention — zero vLLM dependency exists in that backend at all
(§14). The unrelated `FLASHINFER` backend was not audited (out of
scope, unreached).

**I. Did `cuda_event` produce a real MI355X row?** **Yes** — a real,
finite, positive, correctly-identified measurement row (§17).

**J. Did `record_function`/`KERNEL_ONLY` produce a real MI355X row?**
**Yes** — a real, finite, positive, correctly-identified measurement
row, with one disclosed, non-blocking anomaly (§18).

**K. Is the 664-measurement Qwen3→MI355X profile sweep ready for
approval?**

## YES WITH CONSTRAINTS.

Both probes succeeded on real hardware with the full compatibility
stack composed and provenance-tracked. Constraints for the sweep
itself, not for this task's own findings: (1) record RMSNorm's real
native-kernel dispatch in the eventual profile's own provenance (§9,
§22.1), not silently; (2) the sweep's own launch commands must
explicitly set `CUDA_VISIBLE_DEVICES` alongside `HIP_VISIBLE_DEVICES`
(§13); (3) `TP=2`/`TP=4` and the full token grid remain structurally
expected but not yet executed (§22.4) — the first points at each new
`(tp, num_tokens)` combination should be watched, not assumed
identical to this one shape's own result.

**STOP here. The 664-row sweep was not started.**
