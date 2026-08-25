# Task 51 — Should the MLA branch be merged?

Branch: `task-51-mla-merge`, branched from `task-50-contention-reach`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier at
`/work/simulation/Frontier`.

254 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. **Decision: do not merge.** Nothing was merged into Frontier's
`main` (still pinned at `e63fb4e`); the investigation below was done
against a throwaway `git worktree` of the branch tip, removed afterward.

`profiling_knowledge/INFRASTRUCTURE_MAP.md` was read (per this task's own
note that it, not `INFRASTRUCTURE.md`, is the real document) — nothing in
it bears on the merge decision beyond what Task 49 already found (§6).

---

## 1. The blast radius (§3.1)

The branch is `remotes/origin/task/deepseek-mla-attention-port-block-sweep`
at `8c87017`, diverging from `main` at merge-base `47f3fe2`. Diffed with
`git diff main...8c87017` (three-dot — against the merge-base, not the
current tips — see §6 for why the distinction matters). **21 files, every
one of them under `data/profiling/compute/mi355x/`,
`frontier/profiling/attention/`, or `profiling_knowledge/`. Nothing
outside those three trees.**

| path | +/- |
|---|---|
| `data/profiling/compute/mi355x/attention_config.yaml` | +58/-24 |
| `data/profiling/compute/mi355x/deepseek-v3/attention.csv` | -53 (deleted) |
| `.../deepseek-v3/attention_block{1,16,32}.csv` (3 files) | +238 each |
| `.../deepseek-v3/attention_combined_block{1,16,32}.csv` (3 files) | +310 each |
| `.../deepseek-v3/attention_true_mixed_block{1,16,32}.csv` (3 files) | +73 each |
| `frontier/profiling/attention/attention_wrapper.py` | +119/-7 |
| `frontier/profiling/attention/backends/__init__.py` | +28 |
| `frontier/profiling/attention/backends/aiter_mla_attention_wrapper.py` | +611 (new) |
| `frontier/profiling/attention/backends/torch_sdpa_mla_attention_wrapper.py` | +440 (new) |
| `frontier/profiling/attention/main.py` | +26/-1 |
| `profiling_knowledge/AITER_KERNELS.md` | +59 (new) |
| `profiling_knowledge/INFRASTRUCTURE_MAP.md` | +14/-3 |
| `profiling_knowledge/README.md` | +6/-1 |
| `profiling_knowledge/scripts/profile_deepseek_aiter_mla.sh` | +15/-6 |
| `profiling_knowledge/scripts/profile_true_mixed_batch.sh` | +25/-1 |

**Nothing on this list executes during a simulation.** Confirmed two ways,
not one: (a) grepped every importer of `frontier.profiling.attention.*`
across the tree — only files inside `frontier/profiling/attention/` itself
and unit tests under `tests/unit/` import it; `frontier/simulator.py` and
`frontier/execution_time_predictor/sklearn_execution_time_predictor.py`
(the module that actually loads attention data at simulation time) import
neither. (b) §2 below runs an actual simulation against the branch's own
tree and it fails — for a reason that itself proves the profiling-time
code was never consulted, since the failure is a plain `FileNotFoundError`
raised from `pandas.read_csv`, three frames below
`ExecutionTimePredictorRegistry.get`, with no `frontier.profiling.attention`
frame anywhere in the traceback.

**The enum wiring is additive, confirmed by diffing the one file that
declares it** (`frontier/profiling/attention/backends/__init__.py`, which
already existed on `main` with three members —
`FLASHINFER`/`NO_OP`/`TORCH_SDPA`): the branch adds
`TORCH_SDPA_MLA`/`AITER` as two new members, two new `elif` branches in
`get_attention_wrapper()`, and two new lazy-import entries — every hunk is
a pure addition, and the module-level default,
`ATTENTION_BACKEND = AttentionBackend.NO_OP`, is untouched (outside every
diff hunk). Nowhere else in the tree (`config.py`, `simulator.py`, any
execution-time predictor) references this enum at all — it is
profiling-CLI-only, and no existing model's or config's default backend
changes.

## 2. What happens to the replaced data (§3.2)

**`attention.csv` is deleted outright**, not renamed or left alongside the
new files — `git diff --summary` shows `delete mode 100644`, and the new
tree (`git ls-tree` on the branch) has no file by that name in
`deepseek-v3/`'s own directory, only the nine block/combined/true-mixed
variants.

**Which file the simulator actually reads, and whether that depends on
`block_size` — read directly from
`sklearn_execution_time_predictor.py`, not inferred:**

- The path template, `frontier/config/config.py:2190-2193`:
  `atten_input_file` defaults to
  `"./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv"`.
- Resolution, `_get_input_files`/`_initialize_file_paths`
  (`sklearn_execution_time_predictor.py:775-886`): only `{DEVICE}`,
  `{MODEL}`, `{NETWORK_DEVICE}` are substituted. **`block_size` is never
  part of the path.** It resolves, today and on the branch alike (this
  file is not touched by the branch's diff — it isn't in §1's list), to
  exactly `./data/profiling/compute/mi355x/deepseek-v3/attention.csv`,
  regardless of what `block_size` is configured.
- `block_size` *is* used, but only as a **row filter** after the file is
  already loaded: `_filter_mla_attention_df`
  (`sklearn_execution_time_predictor.py:1231-1281`) requires an exact match
  on `block_size` (among other structural columns) and raises
  `"No MLA attention profiling rows remain after structural filtering"`
  if nothing matches.

**So the branch's own new files — `attention_block1.csv`,
`attention_block16.csv`, `attention_block32.csv` — are not selected by
block size. They are not selected at all.** No code path anywhere in this
checkout builds a filename containing `_block`; confirmed by grep
(`attention_dataset_contract.py`, the one other module that inspects
filenames in this directory for a "mixed-batch" contract, matches only
the exact unsuffixed names `attention_mixed.csv` /
`attention_true_mixed.csv` / `attention_combined.csv` — not the
`_block{N}` variants either). The `_block{N}` naming is purely an output
convention of the profiling CLI (driven by whatever `--block_size` was
passed to `frontier.profiling.attention.main` for that invocation); nothing
downstream has ever read it, for any model, on `main`, before this
branch existed.

**Confirmed live, not just read.** Built a throwaway `git worktree` of the
branch tip and ran the exact "final working command"
`profiling_knowledge/DEEPSEEK_V3_MLA_MI355X_JOURNEY.md` documents (the
same one Task 48 ran against `main`: `co-location`, `attn_tp=8`,
`moe_ep=8`, `--vllm_v1_scheduler_config_block_size 32`, real `mi355x`
compute, dummy mode off):

```
FileNotFoundError: [Errno 2] No such file or directory:
  './data/profiling/compute/mi355x/deepseek-v3/attention.csv'
ValueError: Failed to create predictor of type 'random_forrest': [Errno 2]
  No such file or directory: '...attention.csv'
```

raised from `Simulator.__init__` → `ExecutionTimePredictorRegistry.get` →
`_train_attention_layer_models` → `_load_attention_df` → `pd.read_csv`,
**before a single request is scheduled** — a different, and materially
worse, failure than the one Task 48 found on `main`. On `main`, the exact
same command gets past this point (block_size=32 is the only value
`attention.csv` has, so the filter passes), trains successfully, and
fails much later, in batch scheduling, on the unrelated `mlp_up_proj` gap.
**On the branch, no value of `block_size` rescues this** — the file the
unmodified resolver asks for does not exist, full stop. This is exactly
the "additive is not automatically safe" trap named in §7: the new files
are not additive for `deepseek-v3` the way they are for `gpt-oss-120b`
(§6), because `deepseek-v3`'s own old file was deleted, not kept
alongside.

**Do the old and new data agree where they overlap? No — and where they
do overlap, they disagree substantially and in one direction.** Overlap is
narrow: `main`'s `attention.csv` covers `batch_size∈{1,2}`,
`kv_cache_size∈{0,32,64,96}`, `TP∈{1,2,4,8}`, all at `block_size=32`,
`max_model_len=4096`; the branch's `attention_block32.csv` covers
`batch_size` up to 512, `kv_cache_size` up to 9216, `TP=8` only,
`max_model_len=9216`. Joining on every non-timing column
(shape/config columns, 42 of them) finds exactly **3 rows with identical
config in both files** — `batch_size=1`, `kv_cache_size=0`,
`is_prefill=True`, `block_size=32`, `TP=8`, `total_tokens∈{32,64,96}` —
differing only in `max_model_len` (4096 vs 9216, a KV-cache-sizing
declaration, not something these zero-cache-size prefill rows actually
exercise). At those 3 identical shapes, every measured operator's median
in the branch's data is **10%–62% higher** than `main`'s (e.g.
`attn_mla_decode.median` at `total_tokens=64`: `0.00432` ms → `0.00700`
ms, a 1.62x ratio), and the branch's own standard deviation is **2–5x
larger** at the same shapes too (same `count=5` repetitions on both
sides). One direction, consistently, across nearly every operator at
every one of the 3 overlaps — not scatter consistent with ordinary
sub-millisecond measurement noise, and not explained anywhere in the
branch's own commit message or the `profiling_knowledge/` docs it
updates. This is precisely the finding §3.2 warns needs explaining before
merging, and it is not explained.

## 3. Whether anything currently depends on `deepseek-v3`'s profiles (§3.3)

**No.** Grepped `tools/`, `docs/`, `tests/`, `src/` for `deepseek`: every
hit is either a code comment/docstring (`tools/planner_core.py`'s own
notes on `kv_factor`), a test that calls Frontier's `MemoryPlanner`
directly without touching any attention CSV
(`tests/_memory_planner_probe.py`, `tests/test_kv_cache_page_size_vs_memory_planner.py`
— Task 48's own `kv_factor` fix, verified independent of attention
profiling), or a citation in a task report. No tool in `tools/`
enumerates or iterates over profiled models generically (checked: none of
the `run_*_study.py` scripts glob `data/profiling/compute/*/`), so there
is no automation that could pick up `deepseek-v3` incidentally. This
matches Task 48's own finding directly: `deepseek-v3` cannot complete a
run today (`co-location` dies on the `mlp_up_proj` gap;
`pd-af-disaggregation` dies on the missing `kernel_only` family), so no
number this project has ever reported depends on it.

## 4. The decision

**Do not merge.** §2 found two independent, real, blocking problems, not
one merely-unclear one:

1. **Merging as-is breaks `deepseek-v3`'s attention-profile loading
   outright**, for every `block_size`, at predictor-construction time —
   strictly worse than today's state, where `block_size=32` at least gets
   past this step. Confirmed live (§2), not inferred: the branch's own
   diff never touches the resolver
   (`sklearn_execution_time_predictor.py`/`config.py`), so nothing teaches
   it to find the new filenames, and the one file it does look for is
   gone.
2. **Where old and new data overlap, they disagree by 10%–62%,
   consistently in one direction, with 2-5x more variance in the new
   data** — unexplained. §3.2's own instruction is explicit: this needs
   explaining *before* merging, not after. It has not been explained by
   anything in the branch's own commit or by the `profiling_knowledge/`
   documents it updates.

Per §4's own framing, these findings mean §3 did not "come back clean,"
so the branch is not merged. Per §4's own reassurance: the branch is not
going anywhere (it is already fetched, confirmed present via
`git fetch origin` — no new push happened either), and nothing depends on
it (§3), so there is no cost to leaving it exactly where it is.

**What would need establishing before this could be merged, named
concretely rather than left as "more investigation needed":**

- A resolver change — teaching `atten_input_file`'s template (or a new
  companion field) to interpolate `block_size` into the filename, the way
  `atten_kernel_only_input_file` already interpolates measurement type —
  *or* the branch keeping `attention.csv` alongside the new files the way
  `gpt-oss-120b`'s own directory already does (§6), so the existing
  resolver still finds something. Either fix is a change to (or
  alongside) this branch, not something to invent and apply as part of
  evaluating it.
- An explanation for the 10%–62% overlap disagreement: different
  hardware/thermal/contention conditions at collection time, a real
  wrapper-behavior change between the two collection runs (the branch's
  own `attention_wrapper.py` diff includes a genuine bug fix — distinct,
  non-overlapping KV-cache block ranges per sequence, §6 — which could
  plausibly change measured latency even for `TORCH_SDPA_MLA`, not only
  the AITER path it was written for), or something else. Until named,
  neither data set can be trusted over the other for the shapes they
  share.

## 5. The regression comparison

**Not applicable — nothing was merged.** Frontier's `main` is unchanged
(`e63fb4e`, verified via `git status`/`git log` immediately before and
after this investigation). The live checks in §2 were run against a
throwaway `git worktree` of the branch tip, removed afterward
(`git worktree remove --force`), never touching `main`. `dc-sim` itself
has no code changes this task (investigation only), so Task 33's/Task
36's own results were not re-run — nothing in this task could have moved
them, and re-running them would confirm only that nothing changed, which
§0's own `pytest -q` (254 passed) and `check_import_direction.py` (exit
0) already establish more directly for the one thing this task did touch
(nothing, in Frontier or `dc-sim` alike).

## 6. Anywhere this specification is wrong

**The spec's own summary of what the branch does is accurate**, and its
central instinct — that a data replacement deserves more scrutiny than a
code change — is exactly what surfaced both blocking findings above.
Three things worth stating precisely rather than leaving implicit:

1. **The full picture is one file deleted and *nine* added, not three.**
   The spec's own §1 names `attention_block{1,16,32}.csv`; the branch
   also adds the `attention_combined_block{1,16,32}.csv` and
   `attention_true_mixed_block{1,16,32}.csv` siblings (§1's own table).
   None of the nine is read by anything today (§2) — this doesn't change
   the spec's own framing, only its inventory.
2. **A `git diff` between branch and `main` tips (two dots) is
   misleading here, and it is worth naming why.** `main` has advanced
   past this branch's own merge-base by four commits
   (`origin/main` is at `a5177f3`; this checkout's pinned `main` is at
   `e63fb4e`, itself two commits ahead of the merge-base `47f3fe2`).
   A two-dot diff (`main..branch`) surfaces `tools/validation/compare_plots.py`
   shrinking by 279 lines and `tests/unit/test_validation_compare_plots.py`
   being deleted — which looks like the branch reverts unrelated
   validation-tooling work. **It does not.** The correct three-dot diff
   (`main...branch`, against the actual merge-base) touches neither file
   — confirmed directly (`git diff main...branch --stat -- tools/validation/`
   returns nothing). The apparent reversion is `main`'s own later,
   unrelated work (the `validation-tooling2`/`validation-tooling3` merges)
   that this branch simply predates and never touched; a real `git merge`
   would apply both sides' independent changes without conflict, since
   they don't share a file. **§7's "one branch is one branch" trap is
   real to watch for, but the correct diff form shows this particular
   branch does not trip it** — it is exactly the MLA/attention-profiling
   work its own commit message describes, nothing more.
3. **§3.1's question "does the enum wiring change any default" has a
   clean, checkable answer this task gave directly (no) — worth noting
   because it is the one part of §3 that came back unambiguously clean.**
   The two blocking findings are both in §3.2, not §3.1.

One thing the spec does not ask about but is worth recording since it
bears on why the data might disagree: `attention_wrapper.py`'s diff
includes a real correctness fix, independent of MLA —
`_get_standard_input_tensors` previously gave every sequence in a batch
the *same* KV-cache block range (`block_table=list(range(num_blocks))`
for every sequence); the branch gives each sequence a distinct,
non-overlapping range. The branch's own comment says this "reproducibly
crashed with GPU memory access faults at batch_size > 1" for backends
that batch cache writes across the whole batch in one kernel call (named
for `AiterMlaAttentionWrapper`). This is a plausible, named mechanism by
which `TORCH_SDPA_MLA`'s own measured timings could differ from before
(different memory access pattern, not just a different day/host) — one
candidate answer to the "explanation needed" item above, not confirmed
as the actual explanation, since this task's own acceptance criteria bar
changing behavior to test it.

## What shipped

Nothing in Frontier or `dc-sim` — an evaluation, per this task's own
acceptance criteria. `docs/tasks/51-mla-merge-report.md`, this report,
is the only artifact. No merge was made; Frontier's `main` is unchanged.

One commit on `task-51-mla-merge`, stacked on `task-50-contention-reach`.
254 tests pass, unchanged; `check_import_direction.py` exits 0.
