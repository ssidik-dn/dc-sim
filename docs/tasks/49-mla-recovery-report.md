# Task 49 — The minimum viable `kernel_only` profile set for MI355X

Branch: `task-49-mla-recovery`, branched from `task-48-mi355x`'s tip. Paths
per Task 25: working tree at `/work/simulation/dc-sim`, Frontier at
`/work/simulation/Frontier`.

254 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. No code changed — scoping only, per this task's own §5.

---

## 0. `INFRASTRUCTURE.md`, as named, does not exist

Checked before anything else, since this task's own §6 instruction depends
on it. No file named `INFRASTRUCTURE.md` exists anywhere in this checkout
or on the filesystem searched. The evident intended target is
`profiling_knowledge/INFRASTRUCTURE_MAP.md` (read in full) — but that
document has **no numbered sections at all** (its own headers are
descriptive: "Servers," "What's missing where," "SCP patterns," "The
`nvidia-smi` gotcha," "Long-running jobs"), so there is no §6 or §6.1 to
read, and no content anywhere in it (or anywhere else in the checkout,
grepped directly) describing a GPU reporting zero utilization while another
job holds its memory. It also lists **three** GPU hosts (server1/3/8), not
the four this task's own §2 states. Read the whole document anyway, since
its actual content is exactly the kind of drift-tracking this task needs
(§1, §3 below) — the specific hazard this task's own "Known traps" section
warns about is simply not written down anywhere this task could check. Not
something this task can act on beyond reporting it plainly (§6).

## 1. Part A — Where the MLA wrapper is

Found at the first, cheapest step, well inside the 30-minute budget:

```
$ git -C /work/simulation/Frontier log --all --diff-filter=A --oneline -- 'frontier/profiling/attention/backends/*'
8c87017 Add MLA attention backends and DeepSeek block-size x true-mixed sweep
```

`8c87017` is on `remotes/origin/task/deepseek-mla-attention-port-block-sweep`
— **an unmerged branch already present in this exact repository**, not
merged into `main`. It adds `torch_sdpa_mla_attention_wrapper.py` (440
lines) *and* `aiter_mla_attention_wrapper.py` (611 lines), the
`AttentionBackend.{TORCH_SDPA_MLA,AITER}` enum wiring, and per its own
commit message: *"Port `TorchSdpaMlaAttentionWrapper` and
`AiterMlaAttentionWrapper` from server3 into this checkout... so MLA models
(deepseek-r1-0528 / deepseek-v3) can be profiled here."* It also shipped a
real block-size × true-mixed-batch sweep for `deepseek-v3`'s own attention
(`attention_block{1,16,32}.csv`, replacing the smaller pre-existing
`attention.csv`) — dated 2026-08-12, **after** the sweep Task 48 read (52
rows), so the branch's own data is already ahead of what `main` has.

**It is also on real hardware, independently confirmed by the branch's own
commit to `INFRASTRUCTURE_MAP.md`** (diffed directly against `main`'s
version): *"As of 2026-08-11 both [wrappers] are also in
`server1:~/frontier_work/drivenetsfrontier`, ported from server3."* So the
wrapper exists in (at least) three places — `server3`/`server8`
(original), `server1`'s own checkout (ported, per that note), and this
unmerged branch — and is **not lost**. Per this task's own §2, this is a
merge decision, not a copy-and-review or a recovery effort: `main` already
has the commit reachable in its own git history, at zero network cost.
**Not merged in this task**, per its own explicit instruction.

What this branch does **not** contain: no `linear_op.csv`/`moe.csv`
changes for `deepseek-v3` at all (its own diff stat touches only
`attention_config.yaml` and the `attention*.csv` files) — it does not
touch B.2's own dense-MLP gap, and it does not touch `kernel_only` at all.
Solving Part A does not solve Part B's own two gaps; they are independent,
exactly as this task's own priority ordering assumes.

## 2. Part B.1 — The `kernel_only` family, scoped concretely

**What it is, established by reading both sides, not inferred from the
name.** `measurement_type` in the CSV itself distinguishes them:
`CUDA_EVENT` vs `KERNEL_ONLY`. Same shape grid, same row counts, same
columns, for every already-profiled h800/rtx_pro_6000 model checked
(`Phi-tiny-MoE-instruct`: 484/32/224/100 rows, identical both ways) — the
difference is the *value*, not the *shape*: at the same operator and
shape, `attn_pre_proj.median` is `0.0569760` ms (`CUDA_EVENT`) vs
`0.014064` ms (`KERNEL_ONLY`) — roughly 4x smaller, confirming the name:
`KERNEL_ONLY` excludes CPU-side dispatch/launch overhead the standard
`CUDA_EVENT` timing includes. Collection side: the exact same profiler,
same wrapper scripts (`frontier.profiling.{attention,linear_op,moe}.main`),
selected by `--profile_method record_function` (`kernel_only` is an
accepted alias) instead of the default `cuda_event` —
`frontier/profiling/utils/__init__.py`'s own `profile_method_to_measurement_type`
maps them 1:1. **No new tool, no code change** — every one of
`examples/profiling/profile_{attention_chunked_prefill,linear_op,moe}.sh`
already exposes `--profile-method`/`$PROFILE_METHOD`, defaulting to
`cuda_event`. Only `examples/profiling/profile_mi355x.sh` — the convenience
wrapper used to build every existing mi355x profile — hardcodes
`--profile_method cuda_event` in its own three stage functions; that is a
shell-script convenience default, not a capability gap.

**Consuming side**: `shared_prediction_model_manager.py`'s own
`_get_measurement_types_for_cluster`, read directly. For
`sys_arch == "pd-af-disaggregation"` specifically, the requirement is
**unconditional**, not gated on CUDA-graph mode the way `co-location`'s own
requirement is:

| cluster type | measurement types required |
|---|---|
| `PREFILL` | `CUDA_EVENT` only |
| `DECODE_ATTN` | `CUDA_EVENT` **and** `KERNEL_ONLY` |
| `DECODE_FFN` | `KERNEL_ONLY` only |
| `MONOLITHIC` | `CUDA_EVENT` and `KERNEL_ONLY` (only if `decode_cuda_graph_mode != "none"`) |

This is exactly why Task 48's own `co-location` run of `qwen3-a3b-30b-moe`
succeeded (`decode_cuda_graph_mode=none`, so `MONOLITHIC` never needed
`KERNEL_ONLY` at all) while its `pd-af-disaggregation` attempt failed
immediately — `DECODE_ATTN` needs it unconditionally, with no flag to turn
it off.

**Which files, established by reading `_resolve_measurement_input_files_for_config`
directly rather than waiting for the next crash**: all three —
`linear_op_kernel_only_input_file`, `atten_kernel_only_input_file`,
`moe_kernel_only_input_file` — are built unconditionally for the
`KERNEL_ONLY` measurement type; the failure would not "move to the next
missing one" one file at a time, it needs all three (`cpu_overhead_kernel_only_input_file`
is the one exception — it falls back to the standard `cpu_overhead_input_file`
if unset, so is not independently required).

**Which models elsewhere have it, and what grid**: `h800` (7 models) and
`rtx_pro_6000` (2 models); **no `mi355x` model has any of the four files,
for any model** — confirmed by `find`. The grid is identical to each
model's own standard profile (same row counts, checked directly for
`Phi-tiny-MoE-instruct`) — the template is "re-run the exact same sweep,
same shapes, different flag," not a new grid design.

**The minimum viable model and set.** `qwen3-a3b-30b-moe` is the right
candidate for the reason this task's own spec already gives: `moe_layers_enum=None`
(confirmed, `data/config/models/qwen3-a3b-30b-moe.json`) means every layer
is MoE — no dense-replacement-layer gap (B.2's own problem) to compound
this one, and it already runs `co-location` cleanly on `mi355x` (Task 48's
own confirmed 32/32-request run, 29.6244ms mean tpot). The minimum viable
job is **not** the full `TP_SIZES="1 2 4 8" EP_SIZES="1 2 4 8"` grid
`profile_mi355x.sh` used originally — it is the single `(attn_tp=1,
moe_tp=1, moe_ep=1)` point, the same trivial configuration Task 48's own
`pd-af-disaggregation` attempt used. Concretely, three commands (no script
edit — `profile_mi355x.sh`'s own hardcoded default is bypassed by calling
the underlying tools directly, exactly as its own `linear_moe` stage
already does):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # INFRASTRUCTURE_MAP.md's own gotcha
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m frontier.profiling.attention.main --disable_ray --yes \
  --models qwen3-a3b-30b-moe --num_gpus 8 --max_seq_len 2048 \
  --num_tensor_parallel_workers 1 --max_pipeline_parallel_size 1 \
  --attention_backend TORCH_SDPA --block_size 16 \
  --min_batch_size 1 --max_batch_size 16 --fixed_chunked_prefill_size 64 \
  --enable_chunked_prefill_grid_search --device mi355x \
  --profile_method record_function --output_dir data/profiling

python3 -m frontier.profiling.linear_op.main --disable_ray --yes \
  --models qwen3-a3b-30b-moe --num_gpus 8 --num_tensor_parallel_workers 1 \
  --is_moe --profile_method record_function --device mi355x \
  --output_dir data/profiling  # same qwen-specific token grid the original run used

python3 -m frontier.profiling.moe.main --disable_ray --yes \
  --models qwen3-a3b-30b-moe --num_gpus 8 \
  --num_tokens_list <same $MOE_NUM_TOKENS list> \
  --load_distributions uniform skewed extremely_skewed \
  --num_tensor_parallel_workers 1 --expert_parallel_sizes 1 \
  --routing_runtime_path standard_fused_topk --gating_runtime_context standalone_legacy \
  --profile_method record_function --device mi355x --output_dir data/profiling
```

**Size, by proportion of the existing sweep**: attention's own grid halves
per TP value only roughly (not a clean division, chunked-prefill grid
search interacts with TP) but `linear_op`'s own qwen-specific token grid
is fully TP-independent in count (~258-259 points regardless of TP,
confirmed: the existing 1032-row file already sweeps this same count at
4 TP values, `1032/4 = 258`, matching the "258 of 259" figure
`MI355X_FOUR_MODEL_PROFILING.md` documents directly). `moe.csv`'s own grid
is exactly `1/16` at one `(TP,EP)` pair of the full 16 `(TP,EP)` combos
(`7056/16 = 441` for the 3 load distributions × its own token axis).
Total minimum-viable row count: roughly **850 (attention) + 258
(linear_op) + 441 (moe) ≈ 1,550 rows**, against the existing sweep's 11,476
— **about 14% of the size of what was already collected for this model.**

**Cost, with its own uncertainty stated rather than hidden behind one
number.** The only real timing data point in this project's own record for
this device is `GPTOSS_TRUE_MIXED_BATCH_PROFILING.md`'s own: a
well-sized, 325-point attention grid took **15 seconds** (~22 points/sec);
a *badly*-sized 115,818-point grid ran **86.5 hours** at 42% before dying
(~1.8 points/sec) — a >10x spread depending entirely on grid sizing, not
device speed. Neither figure is for `linear_op`/`moe`, and **`record_function`
tracing has never been run on `mi355x` for any model, for any operator** —
only `cuda_event` has a track record here. `record_function` typically
carries *more* per-call Python-side hooking overhead than a bare CUDA
event pair, which could make `KERNEL_ONLY` collection slower per point
than `CUDA_EVENT`, not faster, despite measuring a smaller quantity.
Taking the documented range as the only available anchor: **~1,550
points at 1.8–22 points/sec is roughly 70 seconds to 15 minutes of wall-clock
GPU time**, plausibly longer if `record_function`'s own overhead
dominates at these small, sub-millisecond op durations — call it
**under half an hour, with real risk of running an order of magnitude
longer if any single shape misbehaves (the exact failure mode
`MI355X_FOUR_MODEL_PROFILING.md` already hit once, at `num_tokens=4000`
for this same model)**. This project's own established discipline
(`profiling_knowledge/GPTOSS_TRUE_MIXED_BATCH_PROFILING.md`'s own "86-hour
lesson") applies directly: there is no checkpointing, size the grid
deliberately, and run under `tmux`/`nohup` regardless of how short the
estimate looks.

**What could go wrong, named rather than folded into the number above**:

1. `record_function`/`KERNEL_ONLY` has no precedent on `mi355x` at all —
  the collection could simply fail (a ROCm-side gap analogous to the
  `TORCH_SDPA` backend originally missing here, per
  `MI355X_FOUR_MODEL_PROFILING.md`'s own "starting state") rather than
  merely being slow.
2. The one documented shape-specific failure for this exact model
  (`num_tokens=4000`, `TP=8`, MoE path, kills the whole worker pool) may
  recur — the minimum-viable job's own `TP=1`-only scope avoids the
  reported trigger condition, but that was never confirmed as the *only*
  bad shape, only the one found.
3. `INFRASTRUCTURE_MAP.md`'s own "assume drift, verify before running"
  applies here directly: this task did not check which of `server1/3/8`
  currently has a clean, working environment for this specific
  invocation, and Task 48's own finding (`DEEPSEEK_V3_MLA_MI355X_JOURNEY.md`
  narrating code that isn't in this checkout) is a live demonstration of
  why that check matters.

## 3. Part B.2 — The `deepseek-v3` dense-layer gap, briefly

**Missing columns**, diffed directly against `meta-llama/Llama-2-7b-hf`
(dense throughout, has every generic MLP column): exactly three —
`mlp_up_proj`, `mlp_down_proj`, `mlp_act`. **Root cause, read directly, not
inferred**: `frontier/profiling/linear_op/main.py`'s own docstring —
*"When `--is_moe` is set, MLP-specific profiling operations are
skipped"* — and its own code, line ~818: `if args.is_moe: ... print("Skipping
dense MLP ops (mlp_up_proj, mlp_down_proj, mlp_act).")`. `deepseek-v3`'s
own original profiling run correctly passed `--is_moe` (it is
predominantly MoE) — which is exactly what suppressed the three columns its
own three genuinely-dense layers (`first_k_dense_replace=3`) still need.

**Shape needed**: `hidden_size=7168`, `intermediate_size=18432`
(`data/config/models/deepseek-v3.json`, read directly) — `deepseek-v3`'s
own real dimensions, not shared with any other already-profiled model.

**Does the existing profiler suffice, or does it need Part A's wrapper?**
**The existing profiler suffices — this gap is entirely independent of
Part A.** `mlp_up_proj`/`mlp_down_proj`/`mlp_act` are generic dense linear
operators, handled by `frontier.profiling.linear_op.main` (the same tool
that already profiled `deepseek-v3`'s own `attn_pre_proj`/`attn_post_proj`/
`attn_rope`/norms/`lm_head_linear`/`mtp_fusion_proj`) — nothing about them
is MLA-specific. The fix is a second `linear_op.main` invocation for
`deepseek-v3`, at its own real shape, **without** `--is_moe` — a small,
independent, one-flag-different re-run, not a re-profile of the existing
76-row grid and not gated on merging Part A's branch at all.

**What it would unblock, stated precisely**: `deepseek-v3` under
`co-location` — the first end-to-end run of a latent-attention model
anywhere in this project, and (§4 below) the first *run-level* exercise of
the `kv_factor` correction Task 48 made. Not the disaggregated
architecture (B.1's own, separate, higher-priority gap), and not on the
critical path this task's own §4 already fixes.

## 4. Part C — The plan

**The minimum viable profiling job**: `qwen3-a3b-30b-moe`, `mi355x`, the
three commands in §2 above (`attention.main`/`linear_op.main`/`moe.main`,
`--profile_method record_function`, `TP=1`/`EP=1` only), ~1,550 rows
total across `attention_kernel_only.csv`/`linear_op_kernel_only.csv`/
`moe_kernel_only.csv`, estimated 70 seconds–15 minutes of GPU wall-clock
under normal conditions (§2's own risk list applies), on any of
`server1`/`server3`/`server8` (checking which is actually free first, per
`INFRASTRUCTURE_MAP.md`'s own drift warning and this task's own occupancy
trap — not established which host is currently idle, since this task never
touched a GPU host).

**What it would let this project do that it cannot today**: run
`pd-af-disaggregation` — the architecture every study since Task 32 uses —
on a **third device family**, for the first time. Concretely: reproduce
Task 48's own attempted (and blocked) pool-separation comparison on
`mi355x`, and settle Task 46's own "device breadth is currently unknown"
question with a real data point rather than none.

**What remains blocked afterwards — named, not left implicit**:

1. **Only one `(TP, EP)` point is covered.** This project's own placement
  search (Task 32 onward) sweeps `attn_tp ∈ {1,2,4,8}`; a single-point
  `kernel_only` collection unblocks *a* run, not the search this project
  actually wants to run on this device. Widening to the full grid is the
  same job, proportionally larger (§2's own 14%-of-full-sweep figure
  inverts directly: the full grid is ~7x this job).
2. **`deepseek-v3`/LATENT_MLA remains blocked**, by two gaps this job does
  not touch: B.2's own dense-layer columns, and (for anything beyond
  `co-location`) `kernel_only` collection for `deepseek-v3` specifically,
  not only `qwen3-a3b-30b-moe`.
3. **`AITER` (real production ROCm kernels) remains non-functional** on
  this checkout regardless — Task 48's own `AITER_KERNELS.md` citation
  (torch/ROCm build mismatch, confirmed identical on `server3`) is
  unrelated to anything in this task and unaffected by it. `TORCH_SDPA`
  (portable, not peak-tuned) stays the ceiling on fidelity for this device
  family until that is resolved separately.
4. **What is *not* blocked, worth naming so it isn't mistaken for a
  gap**: the collective/network side already works well —
  `MI355X_FOUR_MODEL_PROFILING.md`'s own figure, the Vidur CC backend
  trains at 0.18–0.27% MAPE on the existing `mi355x_8gpu` collective data.
  Nothing in this task's own findings touches that.

**Can the `kv_factor` correction be validated more cheaply than an MLA
run?** **The cheapest real check already exists and was already done** —
Task 48's own direct comparison against Frontier's live `MemoryPlanner`
(`tests/_memory_planner_probe.py`, no simulation, no GPU) is the cheapest
possible validation of the *formula*, and it is exact. What it does not
give is a *run-level* confirmation — seeing `num_blocks`/OOM behavior play
out correctly inside an actual `Simulator`. That confirmation is available
**without B.1's `kernel_only` fix at all**: `deepseek-v3` under
`co-location` with `--decode_cuda_graph_mode none` never needs
`KERNEL_ONLY` (§2's own table — `MONOLITHIC` only needs it when CUDA-graph
mode is on), so B.2 alone — independent of, and cheaper than, B.1 — is
sufficient to get the first real run-level exercise of the correction.
B.1's own `kernel_only` work is not a prerequisite for this.

## 5. Anything about the fleet that contradicts `INFRASTRUCTURE_MAP.md`

Nothing contradicts it — this task never touched a GPU host, per its own
acceptance criteria, so there is nothing new to compare against it.
§0 above already reports the one real discrepancy found: the document this
task names does not exist under that name, and the one that plausibly is
meant does not have the sections the task cites. `INFRASTRUCTURE_MAP.md`'s
own content (host list, checkout drift, the `nvidia-smi` gotcha, the
no-checkpointing lesson) was read and is reflected in §2/§4's own risk
list; none of it was found to be stale or wrong within what this task
could check without touching a host.

## 6. Anywhere this specification is wrong

1. **`INFRASTRUCTURE.md`, §6, §6.1, and "four GPU hosts" do not match
  anything in this checkout** — see §0. The closest real document
  (`profiling_knowledge/INFRASTRUCTURE_MAP.md`) has no section numbers and
  lists three GPU hosts. The occupancy hazard this task attributes to
  §6.1 is not written down anywhere searched. This does not change
  anything this task concluded (no host was touched either way), but it
  is worth fixing in whichever document this task's own author intended,
  since the next task that *does* touch a GPU host will hit the same
  dead citation.
2. **Otherwise, nothing else required correction.** The task's own
  framing of Task 48's three findings (§1) matches Task 48's own report
  exactly; the priority ordering (`kernel_only` first, model-independent
  and binding; `deepseek-v3`'s dense-layer gap third, model-specific and
  smaller; the MLA wrapper a cheap, time-boxed side check) held up under
  direct investigation — each gap turned out to be exactly as independent
  from the others as the task's own framing assumed, confirmed rather
  than assumed.

## What shipped

Nothing — an investigation and scoping task, per its own acceptance
criteria. `docs/tasks/49-mla-recovery-report.md` only. No profiling was
performed, no host was touched, no merge was made, per this task's own
explicit instructions.

One commit on `task-49-mla-recovery`, stacked on `task-48-mi355x`. 254
tests pass, unchanged; `check_import_direction.py` exits 0.
