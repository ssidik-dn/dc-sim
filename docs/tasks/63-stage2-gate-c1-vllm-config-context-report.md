# Stage 2 — Gate C.1: vLLM `CustomOp`/`set_current_vllm_config` compatibility investigation

**STOP after this report (original version). No GPU was touched.** All
investigation and validation ran CPU-only, `--network none`, no
`--device` flags, on the real pinned image
(`vllm/vllm-openai-rocm@sha256:bb44b39a...`, `xai-3`). Probe 1 was not
run at that point. The 664-row sweep was not started.

**Addendum — real Probe 1 retry (§10 below): a real MI355X GPU attempt
approved and executed after this report. Result: FAILED, a real,
third, distinct compatibility break found (`RMSNorm`) — further than
ever before (past `get_rope()`/`CustomOp`/`dispatch_forward` cleanly,
into the real forward pass), but still no measurement row. Probe 2 was
correctly not run. The 664-row sweep is still not started.**

---

## 1. Why pinned vLLM requires `set_current_vllm_config()` (traced from real source)

`vllm.model_executor.custom_op.CustomOp.__init__` unconditionally calls
`self.dispatch_forward(...)`:

```python
def __init__(self, *, enforce_enable=False, compile_native=False):
    super().__init__()
    self._enforce_enable = enforce_enable
    self._forward_method = self.dispatch_forward(compile_native=compile_native)
```

`dispatch_forward` reads `get_cached_compilation_config()` ->
`get_current_vllm_config().compilation_config`, and
`get_current_vllm_config()` raises `AssertionError` if the module-global
`_current_vllm_config` is `None`:

```python
def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError("Current vLLM config is not set. ...")
    return _current_vllm_config
```

`set_current_vllm_config(vllm_config)` is a real, public `@contextmanager`
that sets this global for the duration of a `with` block and restores
the prior value (`None`, or a real outer config) in its own `finally`:

```python
@contextmanager
def set_current_vllm_config(vllm_config, check_compile=False, prefix=None):
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    ...
    try:
        get_cached_compilation_config.cache_clear()
        _current_vllm_config = vllm_config
        yield
    finally:
        _current_vllm_config = old_vllm_config
        get_cached_compilation_config.cache_clear()
```

Real vLLM serving enters this context once, at model initialization
time, wrapping the whole model-construction region (its own docstring:
"Used during model initialization... so that all modules can access
it"). **It is not needed around `forward()` calls** — confirmed
directly: `CustomOp.forward` just invokes the already-bound
`self._forward_method`, set once in `__init__`; it never re-reads
`get_current_vllm_config()`. The context only needs to wrap
*construction*.

`VllmConfig` fields actually required before `CustomOp` construction:
only `compilation_config`, confirmed by reading the exact code path
above — nothing else on this path touches `model_config`,
`parallel_config`, `cache_config`, etc.

---

## 2. Which Frontier profiling layers are affected (checked layer by layer, not assumed)

| Frontier profiling layer | vLLM class Frontier actually reaches | `CustomOp`? (live `issubclass`) | requires this context? |
|---|---|---|---|
| `attn_rope` (`get_rope()`, via `rope_api_adapter.py`) | `vllm...rotary_embedding.base.RotaryEmbedding` | **True** | **Yes** — the exact failure that blocked Probe 1 |
| `input_layernorm`/`post_attention_layernorm` (Frontier's own `RMSNorm`, `frontier/profiling/common/layers/layernorm.py`) | none, for Qwen3-0.6B — Frontier's `RMSNorm.forward()` tries to import raw functions `rms_norm`/`fused_add_rms_norm` from `vllm.model_executor.layers.layernorm`, **not** vLLM's own `CustomOp`-derived `RMSNorm` class | n/a on this path | **No, for Qwen3-0.6B.** Adjacent finding, out of this task's scope: those two function names **do not exist at all** in the pinned vLLM (confirmed live: `ImportError`) — a *third*, separate compatibility gap (`HAS_VLLM_RMSNORM` silently becomes `False`), noted here, not fixed. `VllmGemmaRMSNorm` (confirmed `CustomOp`-derived) is only constructed for Gemma-style models — unreached for Qwen3-0.6B |
| `mlp_act` (Frontier's own `SiluAndMul`, `frontier/profiling/common/layers/activation.py`) | none — calls `torch.ops._C.silu_and_mul` directly, bypassing vLLM's own `CustomOp`-derived `SiluAndMul` class entirely (confirmed: that vLLM class *is* `CustomOp`-derived; Frontier never constructs it) | n/a on this path | **No** |
| `attn_pre_proj`/`attn_post_proj`/`mlp_up_proj`/`mlp_down_proj` (`ColumnParallelLinear`/`RowParallelLinear`) | **Frontier's own** `frontier.profiling.common.parallel_utils.tensor_parallel_layers.{Column,Row}ParallelLinear` — corrected mid-investigation: **not** vLLM's upstream class of the same name (confirmed from Frontier's own real imports); a plain `torch.nn.Module` calling specific vLLM GEMM/quantization *functions* internally, same pattern as `RMSNorm`/`SiluAndMul` | **False**, confirmed live | **No** for `set_current_vllm_config` — but see §9/§10: these classes call `torch.cuda.current_device()` directly for weight placement, a real, different, genuine GPU-hardware requirement, confirmed live |
| `emb` (`VocabParallelEmbedding`, same module) | same | **False**, confirmed live | **No** |
| MoE fused kernels (`frontier/profiling/moe/`, out of Gate C's current single-host-first scope) | `vllm.model_executor.layers.fused_moe.*` — function-based, no `CustomOp` subclass found anywhere under `frontier/profiling/moe/` | not `CustomOp`-class-based on the paths Frontier calls | **Likely no**, not exhaustively re-verified; out of scope (Qwen3-0.6B is dense) |
| Attention-family ops (`frontier/profiling/attention/`) | rotary is not part of this family at all — `get_rope` has no caller anywhere under `frontier/profiling/attention/` (confirmed exhaustive grep) | not checked for other vLLM classes this tool constructs | **not exercised by this task** — a separate audit would be needed before relying on this context for that tool |

**Conclusion**: exactly one real `CustomOp` construction
(`RotaryEmbedding`, via `get_rope()`) needs this context for Qwen3-0.6B's
`linear_op` profiling. This is not "profiling-wide" for Qwen3-0.6B
specifically, though the fix itself (a context wrapping the whole
construction region) is written generally enough to cover any other
`CustomOp` a future model/operator combination reaches.

---

## 3. Minimum correct `VllmConfig`, live-derived, not guessed

Preferred shape, matching the task's own suggestion, implemented exactly:

```
profiling entry
     |
construct profiling VllmConfig   (build_profiling_vllm_config())
     |
with set_current_vllm_config(...):   (profiling_vllm_config_context())
     build + execute profiling model/operators
```

Live-verified on the real pinned image (CPU-only): a bare `VllmConfig()`
fails at `__post_init__` with `RuntimeError: Failed to infer device
type` — device auto-detection needs a real accelerator device file,
absent by design in this CPU-only investigation. Passing an explicit
`device_config=DeviceConfig(device="cuda")` resolves this —
**not a workaround value**: `"cuda"` is the *only* real accelerator
literal in `DeviceConfig`'s own schema
(`Literal['auto','cuda','cpu','tpu','xpu']`) — ROCm has no separate
value; vLLM represents it through the same string (matching this
project's own already-established finding that ROCm PyTorch keeps the
`torch.cuda` namespace).

`VllmConfig(device_config=DeviceConfig(device="cuda"))` alone —
nothing else overridden — successfully constructs, live-confirmed:

```
compilation_config.custom_ops = ['none']
compilation_config.mode       = CompilationMode.VLLM_COMPILE  (=3)
```

No `model_config` is constructed — confirmed unread by the exact code
path this fix targets (§1), and building one would be real, unrelated
serving state this task's own §5 instruction says not to initialize.

---

## 4. Does this preserve production-relevant kernel/operator dispatch? A real, disclosed tension

Real production serving's own actual defaults (this project's own
`sim-real` never overrides any compilation-related flag —
`EngineArgs`'s own real dataclass defaults apply: `optimization_level=2`
(`O2`, not `O0`), `compilation_config.backend="inductor"`) resolve,
live-confirmed, to exactly `custom_ops=['none']`,
`mode=VLLM_COMPILE` — **the same values the minimal profiling config
above naturally produces, with zero extra configuration.** Under
`custom_ops=['none']`, `RotaryEmbedding.enabled()` is `False`
(confirmed: Qwen3-0.6B's own `pass_config.enable_qk_norm_rope_fusion`/
`fuse_rope_kvcache`/`fuse_qk_norm_rope_kvcache` all default `None`/unset,
so no `+rotary_embedding` override fires), so `dispatch_forward` binds
`self.maybe_compile(self.forward_native, ...)` — the *native* (plain
PyTorch), not the hand-written ROCm/HIP (`forward_hip`), forward path.

**This matches the real config flag real serving uses by default —
but not necessarily what real serving actually executes.** Real
compiled serving reaches this same "native path" decision *expecting*
`torch.compile` (vLLM's own `@support_torch_compile` decorator on its
real model classes) to fuse/optimize that native path via Inductor.
Frontier's own profiling harness constructs plain, undecorated
`nn.Module`s and **never invokes vLLM's model-level compilation at
all**, for *any* operator it profiles — not only rotary. Reproducing
`custom_ops=['none']` literally therefore risks measuring the *worse*
of both real paths: no hand-written kernel, and no compiler-generated
fusion either.

**Recommendation, disclosed rather than silently picked**: since
Frontier's harness structurally cannot replicate compiled-model fusion
for *any* operator (a pre-existing characteristic of this whole
profiling methodology, not something introduced by this fix),
`optimization_level=OptimizationLevel.O0` (or equivalently
`custom_ops=["all"]`) — enabling the real, hand-tuned ROCm kernel
(`forward_hip`) — is the closer approximation of "the best real kernel
available to an isolated, uncompiled measurement," consistent with how
every *other* operator this project profiles is already measured (none
of them go through real compilation either). This is a genuine,
disclosed compromise, not a resolved non-issue — either choice is a
real, non-default-vs-production mismatch of a different kind, and this
report does not pick silently between them. `build_profiling_vllm_config(**overrides)`
accepts an explicit override for whichever choice is made, rather than
hard-coding either.

---

## 5. Implementation

**Existing mechanisms checked first**: none of this project's
`src/integration/` modules touch `VllmConfig`/`CustomOp`/compilation
state. Not a duplicate of anything existing.

**New file**: `src/integration/profiling/vllm_config_context.py`. One
profiling-context adapter covering the whole construction/execution
region, per the task's own explicit preference over patching each
`CustomOp` individually:

- `build_profiling_vllm_config(**overrides)` — the minimum config
  (§3), with the one real, schema-derived literal (`"cuda"`); any real
  caller-supplied field (`**overrides`) takes precedence over the
  built-in default, never silently dropped.
- `profiling_vllm_config_context(**overrides)` — a `@contextmanager`
  delegating nesting/restoration entirely to vLLM's own real
  `set_current_vllm_config` (confirmed, from source, to already save
  and restore the prior config in `finally` — no second, possibly
  inconsistent nesting mechanism is introduced).
- `_verify_vllm_config_api_shape()` — checks `VllmConfig`/`DeviceConfig`/
  `set_current_vllm_config`'s real shape (fields present, signature
  parameter names) before relying on any of them; raises
  `VllmConfigContextUnknownApi` on any mismatch — detection by
  structural inspection, not a caught exception from a trial call, per
  this project's own established philosophy (`rope_api_adapter.py`).
  No source-hash guard here — this module talks to a *public* vLLM API
  surface, not a Frontier internal being patched, so there is no
  Frontier source to hash.
- `get_vllm_config_context_status()` — provenance snapshot (§11).

No permanent edit to Frontier's own source anywhere.

---

## 6. Tests

`tests/test_vllm_config_context.py`, 20 tests:

- **D (hard-coded audit, 2 tests, run directly, no torch needed)**: an
  `ast`-based structural scan of the module's own source asserting zero
  non-boolean numeric literals exist anywhere in it; a source-text check
  that `"cuda"` is the only device/model-ish literal present.
- **G (unknown API, 5 tests, run directly, no torch needed — fake
  `vllm.config.vllm`/`vllm.config.device` modules injected via
  `sys.modules`, no real vLLM import required at all)**: missing
  `device` field, missing `device_config` field, missing
  `set_current_vllm_config`, a renamed parameter on
  `set_current_vllm_config` — each raises `VllmConfigContextUnknownApi`
  loudly; a provenance-status check before any use is all-`None`.
- **A/B/C/E/F/H/I (7 tests, real `torch`+`vllm`, `pytest.importorskip`-gated,
  matching this project's established convention)**: a real, minimal
  `CustomOp` subclass fails without the context and succeeds with it
  (A/B); an explicit `device_config` override is the one actually used,
  not silently replaced (C); the config is restored to the prior value
  after the context exits, including when nested two deep (E/F); the
  real RoPE adapter (`rope_api_adapter.py`) still produces a real
  `RotaryEmbedding` when both fixes are composed (H); Frontier's own
  `ReplicatedLinear` (not vLLM's upstream class — corrected mid-task,
  see §2) constructs identically with or without this context — and,
  live, hits the real-GPU boundary (`torch.cuda.current_device()`)
  exactly and only when no real device is present, asserted explicitly
  rather than left as an unexplained failure (I).

**A real, cross-test pollution bug found and fixed while validating
this live**: `test_rope_api_adapter.py`'s own `install_rope_api_adapter()`
test left Frontier's real `_load_vllm_get_rope` permanently
monkeypatched in the shared process, causing a *later*, unrelated
test's own hash-guard check to fail for a reason having nothing to do
with its own behavior (a "changed hash" that was actually "someone
else's wrapper still installed"). Fixed by adding an `importlib.reload`-based
restoration to both `test_rope_api_adapter.py`'s and this file's own
`_isolate` fixture teardown, guarded to a no-op wherever `torch` (and
therefore the affected module) was never imported at all.

**Result**: 401 passed, 13 skipped locally (up from 394/7) —
`vllm_config_context.py` contributes 14 run + 6 skipped. Import-direction
check clean. **Live re-run of the full combined suite
(`test_rope_api_adapter.py` + `test_vllm_config_context.py`) on the
real pinned image, CPU-only: 26/26 passed**, zero skips (real
`torch`/`vllm` present there).

---

## 7. CPU-only live validation against the exact pinned image

**Isolated proof (the task's own explicit ask, §9): does model
construction get past `get_rope()`/`CustomOp.__init__()`/
`dispatch_forward()`?** Yes — live, direct call, both fixes composed:

```python
with profiling_vllm_config_context():
    result = rope_module.get_rope(
        128, rotary_dim=128, max_position=40960, base=1000000.0,
        is_neox_style=True, rope_scaling=None,
    )
```

produces a real `RotaryEmbedding` object, no exception, no GPU device
present (`test_h`, passing live).

**Full, real end-to-end CLI attempt** (`frontier.profiling.linear_op.main`,
all three guarded fixes applied — QK-norm allowlist, RoPE adapter,
VllmConfig context — CPU-only, `--network none`, no `--device` flags):
found and worked through two more real, distinct issues before
confirming the natural stopping point:

1. `_get_available_gpus` shells out to `nvidia-smi` unconditionally —
   not present on a ROCm host at all, real, separate, unrelated to
   this task's own scope; Frontier's own error message names its own
   documented workaround (`Set CUDA_VISIBLE_DEVICES explicitly`),
   applied here, not invented.
2. With `CUDA_VISIBLE_DEVICES=0` set: the CLI printed the full, correct
   profiling configuration for real Qwen3-0.6B (`Embedding Dim: 1024`,
   `MLP Hidden Dim: 3072`, `Num Q Heads: 16`, `Num KV Heads: 8`,
   `Is MoE: No`) and reached `profile_model`'s own
   `torch_module.cuda.set_device(0)` — a real, unambiguous GPU-hardware
   operation, failing with `RuntimeError: No CUDA GPUs are available`.

This is an *earlier* real GPU checkpoint than `get_rope()` itself
(`cuda.set_device(0)` runs before any model layer is constructed at
all) — so this specific end-to-end invocation did not, on its own,
walk through model construction far enough to re-exercise `get_rope()`
a second way. The isolated `test_h` call above is the real proof for
that specific layer; this full-CLI attempt is the real proof for where
the *whole pipeline* naturally needs a GPU next. Both are reported
here, not conflated.

---

## 8. Hard-coded-number audit

| value | allowed? | why |
|---|---|---|
| `"cuda"` | **allowed** | the only real accelerator literal in `DeviceConfig`'s own schema — not a model or device-architecture value |
| field/attribute names used for introspection (`"device"`, `"device_config"`, `"vllm_config"`, `"custom_ops"`, `"mode"`, `"optimization_level"`) | **allowed** | API surface names, not values |
| `128` (head_dim), `1000000.0` (rope_theta), `40960` (max_position_embeddings) | **not present in `vllm_config_context.py`** | appear only in `rope_api_adapter.py`'s own test fixtures and this test file's own H/I checks — confirmed via structural `ast` scan (§6.D), zero non-boolean numeric literals exist in the module itself |
| any GPU-architecture value (`gfx950`, `mi355x`), predicted timing, or compilation setting hand-picked for this model | **none present** | confirmed by the same source-text scan (§6.D) |

---

## 9. Provenance update

`vllm_config_context.get_vllm_config_context_status()` returns:

```python
{
    "applied": bool,
    "detected_vllm_version": str | None,   # None until a real detection has happened
    "last_built_config": {
        "device_config.device_type": str,
        "compilation_config.custom_ops": list[str],
        "compilation_config.mode": str | None,
        "optimization_level": str | int,
    } | None,
}
```

For the eventual Qwen3-0.6B profile's own provenance record (extending
`docs/tasks/61-...md`'s §9 and `docs/tasks/62-...md`'s §9), a new field:

```
vllm_profiling_config_context:
    applied: true                   # only once profiling_vllm_config_context() has actually run
    pinned_vllm_version: "0.27.1"    # from a real detection, never guessed
    compilation_custom_ops: ["none"] | ["all"]   # whichever §4's choice lands on, recorded verbatim
    compilation_mode: "VLLM_COMPILE" | "NONE"
    adapter_version: <this file's own git commit, filled at collection time>
```

Runtime API compatibility provenance, not a performance claim — `null`
(not `false`) whenever a value is genuinely not yet known.

---

## Final answers

**A. Why does pinned vLLM require `set_current_vllm_config()`?**
`CustomOp.__init__` unconditionally calls `dispatch_forward()`, which
reads `get_current_vllm_config().compilation_config` to decide which
forward implementation to bind; that global is `None` — and the lookup
raises — unless a `set_current_vllm_config(vllm_config)` context is
currently active. Real serving enters this once, wrapping model
construction; Frontier's own profiling code never does.

**B. Which Frontier profiling layers are affected?** For Qwen3-0.6B's
`linear_op` profiling specifically: only `attn_rope`
(`RotaryEmbedding`, via `get_rope()`). Every other layer Frontier
actually constructs (`RMSNorm`, `SiluAndMul`, and — corrected
mid-investigation — Frontier's *own* `ColumnParallelLinear`/
`RowParallelLinear`/`ReplicatedLinear`/`VocabParallelEmbedding`, not
vLLM's upstream classes of the same name) is confirmed, live, not
`CustomOp`-derived on the path Frontier reaches. MoE and attention-family
profiling were not exhaustively re-audited (out of this task's scope).

**C. What is the minimum correct `VllmConfig` for profiling?**
`VllmConfig(device_config=DeviceConfig(device="cuda"))` — confirmed
live sufficient to pass `CustomOp.__init__`; no `model_config` needed
(confirmed unread by the exact code path this targets).

**D. Does that config preserve production-relevant kernel/operator
dispatch?** **Not fully, and this is disclosed, not glossed over.**
Left at its own real defaults, it reproduces real production's exact
`custom_ops`/`mode` *flags* — but real production's own default
behavior *assumes* `torch.compile` fusion Frontier's profiling harness
never performs for any operator, so the flag match does not guarantee
kernel-fidelity match. §4 recommends `optimization_level=O0` instead
(closer to "best real kernel available to an isolated, uncompiled
measurement," consistent with every other operator this project
already profiles) but states this explicitly as a disclosed choice,
not a resolved non-issue.

**E. Does the implementation contain any hard-coded model/device
numbers?** No — confirmed by a structural `ast` scan: zero non-boolean
numeric literals anywhere in `vllm_config_context.py`; the one hard-coded
string, `"cuda"`, is an API/schema constant, not a model value.

**F. Did CPU-only construction progress past the current `CustomOp`
failure?** **Yes**, proven directly and live: a real `get_rope()` call
with Qwen3-0.6B's exact real values, inside
`profiling_vllm_config_context()`, produces a real `RotaryEmbedding`
with no GPU present. A separate, full end-to-end CLI attempt found the
*pipeline's own* next real GPU-hardware checkpoint
(`torch_module.cuda.set_device(0)`) earlier in its own sequence,
before re-exercising `get_rope()` a second way — both results are
reported, not conflated.

**G. Is Probe 1 now ready for a REAL MI355X retry?**

## YES WITH CONSTRAINTS.

The `CustomOp`/`set_current_vllm_config` blocker itself is resolved and
live-verified for the exact operator that stopped Probe 1. Before a
real GPU retry: (1) decide §4's own disclosed kernel-fidelity choice
(`custom_ops=["none"]` matching the flag, vs. `O0`/`["all"]` matching
the best-available-isolated-kernel reasoning) rather than defaulting
silently; (2) apply the real, already-identified `nvidia-smi`/
`CUDA_VISIBLE_DEVICES` workaround (§7) in the real retry's own launch
command; (3) Task 53's own block-table fix (unresolved from any
sandbox, named again in every report since `docs/tasks/61-...md`)
still needs confirming on the real host at run time. None of these are
reasons to redesign anything — they are the concrete pre-run checklist
items this investigation surfaced.

**STOP here, per this task's own instruction. No GPU was touched.**

---

## 10. Addendum: real Probe 1 GPU retry (approved after §1–§9 above)

Kernel policy adopted per explicit approval:
**`optimization_level=OptimizationLevel.O0`** (resolves to
`custom_ops` containing `"all"`) — recorded, not defaulted silently,
as **an isolated-uncompiled-kernel approximation, not an exact
reproduction of production's compiled execution** (§4's own disclosed
tension; this is the choice that side of it landed on).

### 10.1 Pre-flight integration-fix status (before touching the GPU)

Fresh occupancy (`xai-3`, `2026-08-27T11:20:42Z`): `4/8` free, indices
`4,5,6,7`. Selected GPU `4`. Status check run first, no profiling,
`--device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=4 -e CUDA_VISIBLE_DEVICES=4`:

| fix | status |
|---|---|
| QK-norm allowlist | `applied`; allowlist = `['qwen3', 'qwen3_moe', 'qwen3_next']` |
| RoPE API adapter | `applied=True` (lazy — `detected_api_kind` stays `None` until first real `get_rope()` call, confirmed expected, not a bug) |
| VllmConfig profiling context | module loaded; `applied=False` before entry (correct — only set once the `with` block is entered) |
| Task 53 block-table Fix B | `applied`; **applicability note recorded verbatim**: `linear_op` profiling never constructs `AttentionWrapper` — this fix targets a different profiling tool (`frontier/profiling/attention/`) and is not exercised by Probe 1/2, installed for completeness/future-readiness only |
| GPU/env | `HIP_VISIBLE_DEVICES=4`, `CUDA_VISIBLE_DEVICES=4`, `torch.cuda.is_available()=True`, **`torch.cuda.device_count()==1`** — confirmed |

One real warning surfaced here, worth recording: `Using
CUDA_VISIBLE_DEVICES on ROCm is deprecated and support will be removed
in vLLM v0.26.0. Please use HIP_VISIBLE_DEVICES instead.` — both were
set per this task's own explicit instruction; `HIP_VISIBLE_DEVICES` is
the one that actually matters on this pinned version's own ROCm path.

### 10.2 Probe 1 (`cuda_event`)

- **Command**: `docker run --rm --name gate-c1-probe1-20260827T112149Z --network none --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=4 -e CUDA_VISIBLE_DEVICES=4 -v /home/ssidik/rocm-work/gate-c1-smoke:/workspace -w /workspace --entrypoint python3 vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7 /workspace/run_probe3.py cuda_event`
- **Runtime identity**: pinned digest above, `xai-3`/`amd-mi355x-3`, GPU index `4`, pinned vLLM `0.27.1` (detected live).
- **Live `VllmConfig` actually used**: `{'device_config.device_type': 'cuda', 'compilation_config.custom_ops': ['+sparse_attn_indexer', 'all'], 'compilation_config.mode': 'NONE', 'optimization_level': 'O0'}` — `mode=NONE` and `custom_ops` containing `'all'` confirm the approved O0 policy actually took effect, not merely requested.
- **Wall-clock**: ~3 seconds to the failure (`START 11:21:49Z` / `END 11:22:01Z`, most of which is process/import startup).
- **Result: FAILED. No CSV row produced** (only the same `linear_op_config.yaml` echo file as before, removed during cleanup — not a measurement).

**Real progress, further than any prior attempt**: got past
`get_rope()`/`CustomOp.__init__()`/`dispatch_forward()` cleanly (no
error there at all this time — both prior fixes, §1–§9, hold up under
real GPU execution, not merely the CPU-only checks). Reached the
**real forward pass** — `LinearOpWrapper.profile()` → `self.model(...)`
→ `GPTBlock.forward()` → `_forward_with_post_attn_norm()` →
`self.input_layernorm(...)` — before failing with a **third, distinct,
real** compatibility break:

```
File "frontier/profiling/common/layers/layernorm.py", line 50, in forward
    raise ImportError(
ImportError: vLLM is required for RMSNorm profiling. Install vllm or set PYTHONPATH to the vllm source tree.
```

**Root cause, confirmed live, not guessed**: Frontier's own `RMSNorm`
(`frontier/profiling/common/layers/layernorm.py`) tries to import
`rms_norm`/`fused_add_rms_norm` as free functions from
`vllm.model_executor.layers.layernorm`. Live-checked the real pinned
module's actual exports: `['CustomOp', 'F', 'GemmaRMSNorm', 'LayerNorm',
'RMSNorm', 'RMSNormGated', 'envs', 'init_logger', 'ir', 'logger', 'nn',
'poly_norm', 'rms_norm_batch_invariant', 'torch', 'vllm']` — **neither
name exists**; the functionality now lives only on the `RMSNorm`/
`GemmaRMSNorm` *classes'* own `forward()` methods, not as standalone
functions. This raises loudly (an explicit `ImportError`, not a silent
fallback) — correcting §2's own earlier, untested guess that this gap
would "silently degrade" rather than fail hard; live execution shows
it fails hard, which is the better of the two possible outcomes but
still a real, unresolved blocker.

**This is a new, third, distinct Frontier↔vLLM API-skew gap** (same
family as the `rotary_dim`/`get_rope` break and the
`set_current_vllm_config` gap, but a different symptom in a different
module) — out of the scope explicitly authorized for this retry (which
approved running with the *existing* three fixes, not investigating a
new one). Per this task's own explicit instruction, **stopped here —
Probe 2 was not run**.

### 10.3 Probe 2

Not attempted — Probe 1 failed; this task's own instruction is explicit
("On any failure stop; do not run Probe 2").

### 10.4 Cleanup evidence

- Container: `--rm` (foreground) — `docker ps -a --filter
  name=gate-c1-probe1` empty, confirmed gone.
- Output artifact: `probe_output_cuda_event/` (only the config-echo
  YAML, no measurement CSV) removed via a follow-up throwaway
  container (root-owned inside the bind mount, same pattern as every
  prior cleanup this initiative has needed) — confirmed gone
  (`find` → "No such file or directory").
- No leftover profiling process (`ps aux | grep linear_op` empty).
- Fresh occupancy re-check immediately after: `xai-3` back to `4/8`
  free, indices `4,5,6,7` — unchanged, GPU `4` returned to baseline.

### 10.5 Remaining blocker for the next retry

`frontier/profiling/common/layers/layernorm.py`'s own `RMSNorm` needs
the same kind of investigation §1–§9 above gave `get_rope`/`CustomOp`:
determine whether the pinned vLLM's real `RMSNorm`/`GemmaRMSNorm`
*classes* expose an equivalent free-function-shaped call, or whether
Frontier's own `RMSNorm.forward()` needs to construct and call the
real class instance instead (which would itself be `CustomOp`-derived,
confirmed earlier in §2 — meaning it would need the *same*
`profiling_vllm_config_context()` this task already built, composing
rather than requiring a fourth new mechanism). Not investigated or
fixed in this task, per its own explicit scope.
