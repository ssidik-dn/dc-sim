# Stage 2 — Gate C.1: Frontier ↔ vLLM RoPE compatibility fix + smoke retry

**STOP before §11. No GPU was touched in this task.** The RoPE
compatibility fix itself is implemented, guarded, live-verified (CPU
only), and tested. But that same live, CPU-only verification found a
**second, distinct** real API incompatibility one layer deeper
(`vLLM's CustomOp.__init__` requiring a `set_current_vllm_config()`
context Frontier's own profiling code never establishes) — running
Probe 1 for real right now would predictably fail again, for a new
reason, wasting real GPU/fleet time on an attempt already known to
fail. Per this task's own §10 instruction, that is a reason to stop,
not a reason to expand scope and patch a second thing that was never
in scope.

---

## 1. Original failure (recap)

`frontier/profiling/common/layers/rotary_embedding.py::get_rope()`
called the real, pinned vLLM's own
`vllm.model_executor.layers.rotary_embedding.get_rope` with
`rotary_dim=` among its keywords. The pinned image's real vLLM
(`0.27.1`) raised `TypeError: get_rope() got an unexpected keyword
argument 'rotary_dim'`. No profiling iteration ran; no measurement row
was produced; Probe 2 was correctly not attempted.

---

## 2. Exact Frontier API

Every real call site (`frontier/profiling/linear_op/linear_op_impl.py`,
lines 206, 316, 485 — identical in shape, confirmed exhaustive via
`grep -rn "get_rope("` across the entire `frontier/` package, no other
caller exists anywhere) does:

```python
self.rotary_emb = get_rope(
    self.head_dim,
    rotary_dim=self.head_dim,
    max_position=config.max_position_embeddings,
    base=config.rope_theta,
    is_neox_style=config.is_neox_style,
    rope_scaling=config.rope_scaling,
)
```

Frontier's own `get_rope()` wrapper
(`frontier/profiling/common/layers/rotary_embedding.py:554-587`) then
forwards to the real vLLM function, all-keyword:

```python
return vllm_get_rope(
    head_size=head_size, rotary_dim=rotary_dim, max_position=max_position,
    base=base, is_neox_style=is_neox_style, rope_scaling=rope_scaling,
    dtype=rope_dtype,
)
```

**`rotary_dim == head_size` at every one of the three real call
sites** — no partial rotary is used anywhere in this codebase today.
`rope_scaling` is Frontier's own normalized dict (`_normalize_rope_scaling`,
maps a legacy `"type"` key to `"rope_type"`) or `None`.

---

## 3. Exact pinned-vLLM API

Live-inspected inside the real, pinned smoke image
(`vllm/vllm-openai-rocm@sha256:bb44b39a...`, host `xai-3`,
`inspect.signature`/`inspect.getsource`, no GPU device claimed):

```python
def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    ...
    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000)
    scaling_type = rope_parameters.get("rope_type", "default")
    if rotary_dim := rope_parameters.get("rope_dim", None):
        pass
    else:
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_size * partial_rotary_factor)
    ...
```

Real, live-observed vLLM version: `0.27.1`. `rotary_dim` and `base` no
longer exist as top-level parameters at all.

---

## 4. Semantic difference

| semantic input | Frontier expects | pinned vLLM 0.27.1 expects | status |
|---|---|---|---|
| head/rotary dimension | top-level `rotary_dim` (always `== head_size` at every real call site) | `rope_parameters.get("rope_dim")`, else `head_size * rope_parameters.get("partial_rotary_factor", 1.0)` | **removed as top-level arg, moved into `rope_parameters["rope_dim"]`** — respected verbatim ahead of `partial_rotary_factor`, for *any* value, not only `== head_size` |
| max position | `max_position` | `max_position` | unchanged |
| base/theta | top-level `base` | `rope_parameters.get("rope_theta", 10000)` | **moved into `rope_parameters["rope_theta"]`** — omitting it silently defaults to vLLM's own `10000`, wrong for any real model (Qwen3-0.6B: `1000000.0`) |
| rope scaling | `rope_scaling` dict, `rope_type`/`factor`/etc. keys | `rope_parameters` dict, **same** `rope_type`/`factor`/etc. keys, now merged with `rope_theta`/`rope_dim` | **renamed container, same inner keys** — `rope_scaling`'s own contents merge in unchanged |
| dtype | `dtype` (optional kwarg) | `dtype` (optional kwarg, same name) | unchanged |
| device | not passed explicitly either way (inferred from tensors at construction) | same | not part of either signature |
| partial rotary factor | not exposed (every real call site implies full rotary via `rotary_dim==head_size`) | `rope_parameters.get("partial_rotary_factor", 1.0)`, used only if `rope_dim` absent | new API exposes it; this adapter always supplies `rope_dim` explicitly, so `partial_rotary_factor` is never consulted — irrelevant to the translation, not silently relied on |
| `is_neox_style` | positional/keyword bool, default `True` | same name, same default | unchanged |
| `dual_chunk_attention_config` | doesn't exist in Frontier's call | new optional kwarg, default `None` | new, unused by Frontier/Qwen3 — omitted (defaults `None`) when the real signature accepts it |

---

## 5. Compatibility strategy

**Is removing `rotary_dim` semantically correct for Qwen3-0.6B?**
Yes, and provably so, not just plausibly: the new API's own default
path (`rope_dim` absent → `partial_rotary_factor` defaults to `1.0` →
`rotary_dim = head_size * 1.0 = head_size`) reproduces
`rotary_dim == head_size` exactly — precisely what every real Frontier
call site (Qwen3-0.6B's included) already assumes. But the **general**
and always-exact translation used here is simpler still and needs no
special-casing: pass `rope_parameters["rope_dim"] = rotary_dim`
directly. The real source respects `rope_dim` verbatim, ahead of
`partial_rotary_factor`, for *any* value — so the adapter is correct
whether or not `rotary_dim == head_size`, not only for today's one
observed case. Qwen3's own real architecture (HF `Qwen3Attention`,
live-verified earlier in this initiative) applies rotary to the full
head dimension, no partial rotary — semantic equivalence is exact, not
assumed.

`base`/`rope_theta` is the sharper risk: omitting `rope_parameters`
entirely (the naive fix) would silently default to vLLM's own
`10000`, corrupting Qwen3-0.6B's real `1000000.0` theta with no error
raised at all — exactly the "hard failure into silent wrong
construction" this task's own §2 forbids. The adapter always injects
`rope_parameters["rope_theta"] = base`, so this can never happen.

**Detection, not exception-catching**: `_detect_and_build_adapter`
inspects `inspect.signature(real_vllm_get_rope).parameters` and
recognizes exactly two shapes — old (`rotary_dim` and `base` present,
`rope_parameters` absent) and new/pinned (`rope_parameters` present,
`rotary_dim`/`base` absent) — raising `RopeApiUnknownSignature` on
anything else, including a hypothetical transitional signature
carrying markers of both (tested explicitly, §7). A caught `TypeError`
from a trial call was deliberately not used — it cannot distinguish
"wrong RoPE API" from any other unrelated bug in the call, exactly
this task's own §4 prohibition.

---

## 6. Implementation

**Existing compatibility mechanisms checked first** (§3 of the task):
searched this project and Frontier for `get_rope compatibility`, vLLM
version guards, `rotary_dim`/`head_size`/`partial_rotary_factor`/
`rope_parameters` handling, and any `src/integration/` patch already
touching this — none exists. This is genuinely new, not a duplicate of
an existing mechanism.

**New file**: `src/integration/profiling/rope_api_adapter.py`, following
this project's own established guarded-patch convention exactly (task
20, 47, task 53 Fix A/B, the qk_norm allowlist fix):

- `_detect_and_build_adapter(real_vllm_get_rope)` — pure, no `torch`
  needed, returns `(api_kind, adapter, signature_str)`.
- `install_rope_api_adapter()` — patches
  `frontier.profiling.common.layers.rotary_embedding._load_vllm_get_rope`
  (the lazy-loader Frontier's own unmodified `get_rope()` already
  calls and caches through) so it caches the **adapter** instead of the
  raw vLLM function. Frontier's own `get_rope()` and all three real
  call sites are untouched — they keep calling with the exact same
  old-style keyword shape; only what answers underneath changes.
  Patching `_load_vllm_get_rope` (not the `get_rope` *name* itself)
  matters because `linear_op_impl.py` does
  `from ...rotary_embedding import get_rope` — a `from...import`
  binding taken at import time. Patching the imported name after that
  binding exists would not reach `linear_op_impl.py`'s own already-bound
  reference; patching the module-global `_load_vllm_get_rope`/`_VLLM_GET_ROPE`
  that `get_rope()`'s own unchanged body reads at *call* time does.
- Guarded by a **source hash over both** `_load_vllm_get_rope` (the
  patch target) and `get_rope` (whose call-site kwargs shape this
  adapter must keep matching) — `RopeApiAdapterSourceMismatch` if
  either has drifted.
- `get_rope_api_adapter_status()` — provenance snapshot (§9).

**A real bug caught by this exact guard, before any GPU time**: the
hashes were first computed offline via `ast.get_source_segment` (no
`torch` needed in this sandbox). Live verification against the real,
torch-present pinned container found they **did not match**
`inspect.getsource`'s own real output for the identical function —
`ast.get_source_segment` omits the trailing newline `inspect.getsource`
always includes on a function's last line. Not a real source drift;
a real extraction-method mismatch, caught by the same install-time
guard that would also catch a genuine upstream change. Corrected to
the live-verified hashes (recorded in the module's own source
comment). This is exactly the kind of thing §10's "no GPU until code
review is complete" step exists to catch cheaply.

---

## 7. Tests

`tests/test_rope_api_adapter.py`, 13 tests:

- **A (old API, 2 tests)**: a plain-Python mock accepting `rotary_dim`/`base`
  directly — old path selected, every argument (including a real
  `rope_scaling` dict) reaches the mock unchanged. This *is* §7.E's
  "no regression for the old-API path" check, verified directly rather
  than merely asserted.
- **B (pinned API, 6 tests)**: a mock matching the exact live-observed
  pinned signature — new/adapted path selected; `rotary_dim`/`base`
  never appear as top-level kwargs; `rope_parameters == {"rope_theta":
  base, "rope_dim": rotary_dim}` exactly; a real scaling dict's own
  keys (`rope_type`/`factor`) survive merged in; the general
  translation is proven to hold even when `rotary_dim != head_size`
  (no current caller does this, but the mapping doesn't rely on that
  coincidence); a conflicting pre-existing `rope_theta` in
  `rope_scaling` raises rather than silently overriding; a pinned-style
  mock that doesn't even declare `dual_chunk_attention_config` doesn't
  get it injected (checked against the real detected signature, not
  assumed).
- **C (unknown API, 3 tests)**: a mock matching neither shape → explicit
  `RopeApiUnknownSignature`, error message names the real observed
  signature; a hypothetical signature carrying *both* `rotary_dim` and
  `rope_parameters` also hard-fails rather than being silently treated
  as either known case.
- **D (real Qwen3-0.6B config, torch-gated)**: real HF-verified field
  values (`head_dim=128`, `max_position_embeddings=40960`,
  `rope_theta=1000000.0`, `is_neox_style=True`, `rope_scaling=None`,
  pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`) driven
  through the real, unmodified Frontier `get_rope()` with the adapter
  installed against a fake pinned-shaped `vllm` module — asserts the
  exact resulting `rope_parameters`.
- **Install/guard test (torch-gated)**: corrupting the expected hash
  makes `install_rope_api_adapter()` raise and leaves Frontier's own
  loader untouched.

`pytest.importorskip("torch")` gates the two torch-dependent tests,
matching `test_attention_block_table_fix_guard.py`'s own established
convention exactly — skipped, not failed, in this CPU-only sandbox.

**Result**: 11 passed, 2 skipped in this sandbox. Full suite: **394
passed, 7 skipped** (up from 383/5 before this task). Import-direction
check clean.

**Live verification beyond the mocked tests** (§10's own "before any
GPU retry" requirement): ran the real hash guard, `install_rope_api_adapter()`,
and a real `get_rope()` call with Qwen3-0.6B's exact real values
against the real pinned image on `xai-3` — CPU-only, no `--device`
flags, no GPU claimed. Hash guard passed; adapter correctly detected
`"new"`; argument translation confirmed correct (see §11 below for
what this run actually surfaced).

---

## 8. Hard-coded-number audit

| value | allowed? | why |
|---|---|---|
| `"rotary_dim"`, `"base"`, `"rope_parameters"`, `"rope_theta"`, `"rope_dim"`, `"dual_chunk_attention_config"`, `"partial_rotary_factor"` | **allowed** | API parameter names, not model values |
| `_EXPECTED_LOAD_VLLM_GET_ROPE_HASH`, `_EXPECTED_GET_ROPE_HASH` | **allowed** | schema/version-identifying constants (a source hash), not a model or performance value |
| `128` (head_dim), `1000000.0` (rope_theta), `40960` (max_position_embeddings) | **not present in `rope_api_adapter.py` itself** | these appear only in the test fixture (§7.D) and the real, already-registered `data/config/models/Qwen3-0.6B.json`/`ModelConfig` — the adapter module contains zero model-specific numeric constants of any kind |
| any predicted timing, device property, or head count | **none present** | this module performs no measurement and touches no device |

The adapter itself is model-agnostic by construction — it translates
whatever `head_size`/`rotary_dim`/`base`/`rope_scaling` values
Frontier's own caller passes, for any model, not only Qwen3-0.6B.

---

## 9. Provenance change

`rope_api_adapter.get_rope_api_adapter_status()` returns:

```python
{
    "applied": bool,                          # False until install_rope_api_adapter() has run
    "detected_api_kind": "old" | "new" | None,  # None until a real detection has happened
    "detected_signature": str | None,
    "detected_vllm_version": str | None,
    "frontier_load_vllm_get_rope_hash": str,   # the guarded, expected hash (always present)
    "frontier_get_rope_hash": str,
}
```

For the eventual Qwen3-0.6B profile's own provenance record (§9 of the
prior Gate C.1 plan, `docs/tasks/61-...md`), this maps onto a new field:

```
rope_api_adapter:
    applied: true                 # only once install_rope_api_adapter() has actually run
    frontier_api: "old (rotary_dim/base top-level kwargs)"
    vllm_api: "new (rope_parameters dict)"
    adapter_version: <this file's own git commit, filled at collection time>
    detected_vllm_version: "0.27.1"   # from a real detection, never guessed
```

This is runtime API compatibility provenance, not a performance or
profile-correctness claim — explicitly not conflated with the eventual
measurement rows themselves. `null` (not `false`) whenever a value is
genuinely not yet known.

---

## 10. Probe 1 result

**Not run.** Live, CPU-only pre-GPU verification (no `--device`
flags, no GPU claimed, per §10's own "no GPU until code review is
complete") ran the real adapter against the real pinned image and
found a **second, distinct** failure one layer past the rope fix:

```
File "vllm/model_executor/custom_op.py", line 133, in __init__
    self._forward_method = self.dispatch_forward(...)
File "vllm/model_executor/custom_op.py", line 177, in dispatch_forward
    compilation_config = get_cached_compilation_config()
File "vllm/config/vllm.py", line 2436, in get_current_vllm_config
    raise AssertionError(
AssertionError: Current vLLM config is not set. This typically means
get_current_vllm_config() was called outside of a
set_current_vllm_config() context, or a CustomOp was instantiated at
module import time or model forward time when config is not set.
```

vLLM 0.27.1's `RotaryEmbedding` now inherits from `CustomOp`, whose
own `__init__` unconditionally calls `dispatch_forward()`, which
requires a global "current vLLM config" to have been set via
`set_current_vllm_config()` first. Checked exhaustively
(`grep -rn "set_current_vllm_config\|VllmConfig\|CompilationConfig"
frontier/profiling/`): **Frontier's own profiling code never calls
this, anywhere.** This is not an artifact of my own isolated test
script skipping some setup Frontier's real path performs — Frontier's
real path performs no such setup either, so the real Probe 1 CLI
invocation would hit this same `AssertionError` once the rope
`TypeError` is fixed, not something new my test invented.

**This is out of this task's own scope** (which named the `rotary_dim`
failure specifically) and is not something "smallest guarded fix"
covers — it is a different vLLM subsystem (custom-op compilation
dispatch), not RoPE argument translation. Running Probe 1 for real
right now would predictably fail again, for this new reason, spending
real GPU/fleet time to learn something already known from a CPU-only
check.

---

## 11. Probe 2 result

Not attempted — Probe 1's own gate was not cleared (per this task's
own explicit rule: stop, do not proceed to Probe 2, if Probe 1 fails).

---

## 12. Measured profiling timings

None — no GPU was used this task.

---

## 13. Output rows

None — no real measurement row was produced this task.

---

## 14. Warnings / anomalies

1. **A real infrastructure mistake, caught and contained**: an early
   signature-inspection attempt used `xai-4` (chosen only because it
   was already reachable from the prior smoke test), which does **not**
   have the exact pinned digest cached — Docker began pulling the full
   34.8GB image before a 60-second client-side timeout killed the SSH
   session. Verified afterward: no new image was registered
   (`docker images` still shows only the old digest), no dangling
   layers, no disk usage change, no leftover container — the pull did
   not complete or leave residue. All further work moved to `xai-3`
   (which has the correct pinned digest already cached).
2. **The `ast.get_source_segment` vs. `inspect.getsource` trailing-newline
   mismatch** (§6) — a real, caught-before-GPU-time bug in this task's
   own hash-guard construction, not a Frontier/vLLM issue.
3. **The `set_current_vllm_config` blocker** (§10) — the substantive
   finding of this task's own pre-GPU verification step.
4. Fresh occupancy on `xai-3` dropped from 8/8 free to 4/8 free between
   two checks roughly 11 minutes apart (real, live third-party
   contention, not stale data) — free indices `4,5,6,7` held stable
   through this task's own CPU-only work.

---

## 15. Cleanup

No container was left running (every invocation used `--rm` in the
foreground; `docker ps -a --filter name=gate-c1-smoke` shows nothing on
either host). No profiling process remains (`ps aux | grep linear_op`
empty on `xai-3`). Requested GPU indices were never claimed by a real
device flag during this task's own work (all verification ran with
`--network none` and no `--device`/`HIP_VISIBLE_DEVICES` at all) so
there is no GPU state to return to baseline. Fresh occupancy re-checked
clean (`4/8 free`, stable) at the end of this task.

**Staged Gate-C1 input files intentionally left in place** on `xai-3`
(`~/rocm-work/gate-c1-smoke/`: `frontier/`, `data/config/models/Qwen3-0.6B.json`,
`qk_norm_allowlist_fix.py`, `run_probe.py`, and now `rope_api_adapter.py`)
— per this task's own explicit instruction, not cleaned merely for
tidiness; they are the exact inputs the next approved profiling step
would reuse. The stale copy on `xai-4` (missing `rope_api_adapter.py`,
staged before the pivot to `xai-3`) was not cleaned up either; harmless,
clearly labeled, safe to leave or remove on request.

---

## 16. Remaining blockers

1. **`set_current_vllm_config` / `CustomOp` dispatch** (§10) — the real,
   next blocker. Needs its own investigation (does Frontier's profiling
   CLI need to call `vllm.config.set_current_vllm_config()` somewhere
   before constructing any model layer that touches `CustomOp`-derived
   classes — not only `RotaryEmbedding`, but potentially every
   vLLM-backed layer Frontier's profiling code constructs? Is there a
   minimal, correct `VllmConfig` Frontier could construct and set for
   pure profiling purposes, with no real serving config available?)
   before any real GPU retry of Probe 1.
2. Everything already named in the prior plan
   (`docs/tasks/61-...md` §16/§17): Fix B (Task 53 block-table)
   applicability on the real profiling host, still unresolved from any
   sandbox; the real pinned Frontier profiling image/runtime identity
   — now resolved and recorded here as `vllm/vllm-openai-rocm@sha256:bb44b39a...`
   itself, confirmed the correct runtime (has `vllm`+`torch`, unlike
   the earlier `rocm/pytorch:latest` attempt).

---

## 17. Recommendation

The RoPE compatibility fix itself is **done, correctly scoped, tested,
and live-verified** — ready to be part of the eventual real collection
once the *next* blocker (§16.1) is resolved. Do not attempt Probe 1 on
real GPU hardware yet; it would fail again, predictably, for a
different, already-identified reason. The right next step is a
focused, CPU-only (no GPU) investigation of the `set_current_vllm_config`
requirement, mirroring exactly how this task investigated the
`rotary_dim` failure — before any further real hardware time is spent.

---

## Final answers

**A. What exactly caused the `rotary_dim` failure?** The pinned
vLLM `0.27.1`'s own `get_rope()` removed `rotary_dim` and `base` as
top-level parameters, moving their real values into a single merged
`rope_parameters` dict (`rope_dim`/`rope_theta` keys, alongside the
scaling-type keys Frontier's own `rope_scaling` already used under the
same names). Frontier's own profiling code was written against the
older, top-level-arguments API and was never updated for this real
upstream API change.

**B. Is the adapter semantically correct for Qwen3-0.6B?** **Yes,
proven, not assumed.** `rope_parameters["rope_dim"] = rotary_dim`
(general and exact for any value, verified for `rotary_dim == head_size`
specifically since that's Qwen3-0.6B's real case) and
`rope_parameters["rope_theta"] = base` (preserving Qwen3-0.6B's real
`1000000.0`, never silently falling back to vLLM's own `10000`
default). Live-verified end-to-end with Qwen3-0.6B's exact real field
values reaching the pinned API's own `rope_parameters` correctly.

**C. Does it contain any model-specific hard-coded numbers?** **No.**
`rope_api_adapter.py` contains zero model-specific numeric constants —
every real Qwen3-0.6B value (`128`, `1000000.0`, `40960`, etc.) exists
only in the test fixture and the already-registered model config, not
in the adapter itself. The adapter's own hard-coded values are API
parameter names and two source-hash strings — both explicitly allowed
categories.

**D. Did `cuda_event` produce a real MI355X profile row?** **No.**
Probe 1 was not run on real GPU hardware this task — a CPU-only
pre-check found a second, distinct blocker first (§10), and this
task's own §10 instruction is to stop rather than spend GPU time on an
attempt already known to fail.

**E. Did `record_function`/`kernel_only` produce a real MI355X profile
row?** **No.** Probe 2 was never attempted — Probe 1's own gate was
not cleared.

**F. Is the 664-measurement sweep now ready for approval?**

## NO.

The RoPE compatibility fix is real, correct, and tested — but it is
necessary, not sufficient. A second, distinct, real API incompatibility
(`set_current_vllm_config`/`CustomOp` dispatch, §10/§16) was found
before any GPU time was spent on it, and remains unresolved. Do not
run Probe 1, Probe 2, or the full sweep until that is investigated and
fixed with the same rigor this task applied to the RoPE issue.

**STOP here, per this task's own instruction. No GPU was touched.**
