# Task 48 — Does any of this work on the device we actually have?

Branch: `task-48-mi355x`, branched from `task-47-scheduler-regime`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`.

254 tests pass (249 unchanged + 5 net new), and `python3
tools/check_import_direction.py` exits 0. Task 33's own sixteen-row table
and Task 36's own two-fabric result both still reproduce bit-identical — the
one real fix this task made (§2) touches only a field no model this project
has ever run supplies, so nothing existing moved.

---

## 1. Part A — What ships

**Compute profiles**, `data/profiling/compute/mi355x/`, five models (row
counts = data rows, header excluded):

| model | `attention.csv` | `linear_op.csv` | `moe.csv` | Complete for its own architecture? |
|---|---|---|---|---|
| `deepseek-v3` | 52 | 76 | 528 | **No** — see below |
| `deepseek-r1-0528` | *(symlink to `deepseek-v3`)* | | | Same data, not an independent second model |
| `meta-llama/Llama-2-7b-hf` | 3388 | 1036 | — (dense, no MoE) | Yes (dense, every layer identical) |
| `openai/gpt-oss-120b` | 3388 (+block/mixed variants) | 1036 | 7056 | Yes |
| `openai/gpt-oss-20b` | 3388 (+block/mixed variants) | 1036 | 7056 | Yes |
| `qwen3-a3b-30b-moe` | 3388 | 1032 | 7056 | Yes |

`deepseek-v3`'s own smaller row counts are not incompleteness in the sense
Task 43B found (a coverage gap within an otherwise-adequate grid) — its own
engineering record (`profiling_knowledge/DEEPSEEK_V3_MLA_MI355X_JOURNEY.md`)
states "52/52 configurations" and "76/76 linear_op configurations,"
i.e. MLA's own grid is genuinely smaller (no separate per-KV-head axis to
sweep). **The real gap is a missing operator, not a missing point in an
existing grid**: `deepseek-v3.json` declares `moe_layers_enum` starting at
layer 3 (`first_k_dense_replace=3`) — layers 0-2 are ordinary dense FFN, not
MoE, and Frontier's own execution-time predictor calls
`_get_mlp_layer_up_proj_execution_time` for them, needing a `mlp_up_proj`
column. Checked directly against every model's own `linear_op.csv` header:
only `meta-llama/Llama-2-7b-hf` (dense throughout) has it;
`qwen3-a3b-30b-moe`/`openai/gpt-oss-*` don't need it (`moe_layers_enum=None`
— *every* layer is MoE, per `model_config.py`'s own "MoE without
`moe_layers_enum`: all layers are MoE"); `deepseek-v3` needs it (mixed
dense+MoE) and does not have it. **The same failure signature Task 38 found
for `step-moe-noquant-small`, on a different model and a different
device — the pattern generalizes, the specific model does not.**

**Network profiles**: yes, separate from compute, `data/profiling/network/mi355x_8gpu/`
(`all_reduce.csv`, `send_recv.csv`) — both present, confirmed distinct
directories per this task's own §2.

**Does the checkout differ from public upstream?** Yes, substantially, and
precisely characterized rather than assumed: `git remote -v` gives
`origin = https://github.com/drivenets/Frontier.git` — a **private fork**,
not `https://github.com/NetX-lab/Frontier` (the public repo the README's
own release note links to). `git log` shows the mi355x profiles, the four
new model configs, and small `frontier/config/{config,device_sku_config}.py`
changes all arrived in one merged PR, `git show c51756d --stat`:
*"Merge pull request #1 from drivenets/task/mi355x-support-deepseek — Add
AMD MI355X support; fix DeepSeek config parsing and scheduler deadlock."*
Separately, `profiling_knowledge/DEEPSEEK_V3_MLA_MI355X_JOURNEY.md` narrates
a *completed*, working end-to-end run — but the code it describes building
(`TorchSdpaMlaAttentionWrapper`, `frontier/profiling/attention/backends/torch_sdpa_mla_attention_wrapper.py`)
does not exist anywhere in this checkout (`find`, `grep`, both empty). The
doc's own account was not produced against what is actually committed
here — most plausibly work done on a different local checkout
(the doc's own "Known follow-ups" mentions a `/home/dn/FrontierBase`
distinct from the "remote MI355X box") that was never merged into this
fork. Reported as a real, checkable discrepancy between what a project
document claims and what the pinned checkout contains — not assumed
consistent.

## 2. Part B — Does the memory formula hold?

**Parameter memory: yes, exactly, extending Task 38's three models to a
fourth.** `attn_param_mem_bytes(deepseek-v3, attn_tp)` against a real
`ParamCounter.get_num_parameters_per_device()` probe
(`tests/_param_counter_probe.py`, unmodified — device-independent, so
reused as-is):

| `attn_tp` | formula | `ParamCounter` |
|---|---|---|
| 1 | 25073549312 | 25073549312 |
| 2 | 12536774656 | 12536774656 |
| 4 | 6268387328 | 6268387328 |
| 8 | 3134193664 | 3134193664 |

Exact agreement at every degree — `ParamCounter` does not special-case
MLA's real compressed weight layout (`q_lora_rank`/`kv_lora_rank`) for
parameter counting; it reads the same raw `num_key_value_heads`/`head_dim`
fields this formula already reads, confirmed live rather than assumed from
Task 39's own "this is what `ParamCounter` itself reads" note (which was
established only for DENSE_KV models).

**KV-cache page size: disagreed by exactly 2x, until fixed — this is the
finding that outranks everything else, per this task's own §3.** Built
`tests/_memory_planner_probe.py`, calling Frontier's own
`MemoryPlanner._get_kv_cache_memory_per_layer_per_block` directly. Before
any fix, this formula (using Task 39's own `runtime_num_kv_heads=1`,
`runtime_head_dim=576`) gave **36864 bytes/block at every `attn_tp`**;
`MemoryPlanner` gives **18432** — exactly half, constant across degree on
both sides (MLA's compressed cache is never partitioned by TP, confirmed
both ways). Root cause, read directly:
`frontier/attention/memory.py`'s own `AttentionRuntimeKVLayout.elements_per_token_per_worker
= kv_factor * runtime_num_kv_heads_per_worker * runtime_head_size`, and
`frontier/attention/families.py` declares **`kv_factor=2` for DENSE_KV**
(separate K and V caches) but **`kv_factor=1` for LATENT_MLA** (one
compressed latent — MLA's whole point). This module's own `_KV_FACTOR = 2`
constant was applied unconditionally, correct for every DENSE_KV model
this formula has ever been checked against, silently wrong for the one
family it was never checked against.

**Fixed**, mirroring Task 39's own established idiom exactly: `ModelSpec`
gained `kv_factor: Optional[int] = None` — `None` means "use this
module's own DENSE_KV default," a LATENT_MLA caller must set `kv_factor=1`
explicitly, no auto-detection (same reasoning task 39 already gave for
`runtime_num_kv_heads`/`runtime_head_dim`). Verified: `deepseek-v3` now
gives exactly 18432 at every tested degree; the three original DENSE_KV
models' own figures are computed by the identical code path they always
were (`kv_factor=None` → `_KV_FACTOR`), confirmed unchanged by the full
suite (254/254, no regression). `tests/test_kv_cache_page_size_vs_memory_planner.py`
is the required new test, covering all four models plus a dedicated
"dropping `kv_factor=1` gives exactly double" pin.

**Task 39's own report was right to flag this as unverifiable, and right
about the mechanism it had already fixed** (the KV-head-count/head-size
override) — it simply could not have found the `kv_factor` gap, since
that path is *unconditional* in the pre-Task-48 formula (there was no
family-specific term to override at all until this task added one). Not a
citation error in Task 39's own report; a genuinely new finding this
task's own access to a profiled MLA model made possible for the first
time.

## 3. Part C — Does a run complete?

**`deepseek-v3`: no.** Ran the exact command
`DEEPSEEK_V3_MLA_MI355X_JOURNEY.md` documents as its own "final working
command" (`co-location`, `attn_tp=8`, `moe_ep=8`, real `mi355x` compute,
dummy mode off). Crashes at the very first scheduled batch:

```
KeyError: 'mlp_up_proj'
  File ".../sklearn_execution_time_predictor.py", line 4439, in _get_mlp_layer_up_proj_execution_time
    raw_time = self._predictions[operator.profiling_name()][(effective_tokens,)]
```

Exactly the missing-operator failure identified in Part A — a dense
(non-MoE) layer needs `mlp_up_proj`, which `deepseek-v3`'s own mi355x
`linear_op.csv` never profiled. **A screening-category failure (Task 38's
own established kind), not a device failure**: the predictor trains
successfully, the simulator constructs successfully, real (non-dummy)
kernels are used throughout up to this point — the device and the pipeline
both work; this one model's own profiling data is short three layers'
worth of one operator.

**A second, non-MLA mi355x model completes cleanly**, isolating the gap to
`deepseek-v3` specifically rather than the device: `qwen3-a3b-30b-moe`
(`moe_layers_enum=None` — every layer MoE, no dense-replacement gap),
`co-location`, `attn_tp=1`, real mi355x compute — **all 32 requests
completed**, mean tpot **29.6244ms**, throughput **65.078 req/s**. `mi355x`
genuinely runs a real, non-dummy simulation end-to-end for at least one
profiled model.

**The pool-separation comparison — unreachable, for a third, independent
reason.** The same `qwen3-a3b-30b-moe` config, switched to
`pd-af-disaggregation` (needed for colocated-vs-separated-pools at all),
fails differently:

```
FileNotFoundError: Linear ops input file
  ./data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op_kernel_only.csv not found
```

`DECODE_ATTN`'s own disaggregated micro-batch path needs a **`kernel_only`
profiling family** — `linear_op_kernel_only.csv`/`moe_kernel_only.csv`/
`attention_kernel_only.csv` — that no mi355x model's own profiling
collected (confirmed: this file does not exist under any
`data/profiling/compute/mi355x/*/` directory, not just this one model's).
**Not the same gap as `deepseek-v3`'s** — this is a `co-location`-vs-`pd-af`
architecture requirement, orthogonal to which model or attention family is
in use. So: `mi355x` runs `co-location` for at least one model; nothing
currently profiled on `mi355x` runs `pd-af-disaggregation` at all, and the
pool-separation comparison — which needs separated pools — cannot be run
on this device today, for either candidate model, for two independent
reasons.

## 4. What would be needed to use this device properly

Two separate, concretely scoped gaps, not one:

1. **For `deepseek-v3` (and `deepseek-r1-0528`, same data)**: profile
  `mlp_up_proj`/`mlp_down_proj` (and whatever else the generic dense MLP
  path needs — check against `meta-llama/Llama-2-7b-hf`'s own column set
  for the full list) at `deepseek-v3`'s real shape
  (`hidden_size=7168`), for the 3 layers `moe_layers_enum` excludes. A
  small, targeted addition — not a re-profile of the 76/76 already-complete
  linear_op grid, and not a GPU-hours-scale undertaking on its own (three
  layers' worth of one operator's shape sweep).
2. **For the disaggregated architecture generally**: profile a
  `kernel_only` family variant (linear_op, MoE, and attention) for at least
  one mi355x model. This is a bigger, model-independent gap — it blocks
  *any* mi355x model from running `pd-af-disaggregation`, not just the
  LATENT_MLA one, and is the higher-priority piece if this project's own
  disaggregated-architecture studies (Tasks 32 onward) are ever meant to
  run on this device family.

Per this task's own known trap, neither of these was attempted — this is
the costed inventory of what profiling work would require, not the work
itself.

## 5. Anywhere this specification is wrong

**The two citations to prior work both check out exactly.** "One attempt
at a second NVIDIA device found the only model profiled for both could
not complete a run" and "a profile with three rows where another device
had thirty-two" both match `docs/tasks/43b-device-report.md` precisely:
`Qwen3-30B-A3B-tiny`'s own `linear_op.csv` covers `num_tensor_parallel_workers ∈ {1,2,4,8}`
(32 rows) on `h800`, `{1}` only (3 rows) on `rtx_pro_6000` — quoted
verbatim in that report's own §2.

**Otherwise, nothing else in this specification required correction.**
Its own central premise — that a device profile appearing locally
suggests other local changes worth checking before attributing a result
to the device — held up precisely (§1's own fork/PR finding), and its own
"screen before committing" instruction correctly anticipated exactly the
kind of failure this task found (twice, for two different reasons).

## What shipped

- `tools/planner_core.py` — `ModelSpec.kv_factor: Optional[int] = None`;
  `_runtime_kv_factor`; `_kv_cache_page_bytes_per_layer` now reads it.
- `tests/_memory_planner_probe.py` — real-Frontier `MemoryPlanner` probe,
  mirroring `_param_counter_probe.py`'s own established pattern.
- `tests/test_kv_cache_page_size_vs_memory_planner.py` — the required new
  test: all four models (three DENSE_KV, one LATENT_MLA) against
  `MemoryPlanner` directly, plus a dedicated pin on the 2x magnitude of
  the divergence this task found.
- `docs/tasks/48-mi355x-report.md`, this report.

One commit on `task-48-mi355x`, stacked on `task-47-scheduler-regime`.
Task 33's sixteen-row table and Task 36's two-fabric result both reproduce
bit-identical. 254 tests pass (249 + 5 new); `check_import_direction.py`
exits 0.
