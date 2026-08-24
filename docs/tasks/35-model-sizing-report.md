# Task 35 — What model would make topology matter?

Branch: `task-35-model-sizing`, branched from `task-33-planner`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`. 189 tests pass, unchanged, and
`python3 tools/check_import_direction.py` exits 0 — arithmetic and
inventory only, nothing under `src/` touched, per this task's own
acceptance criteria.

**The inventory was checked before the arithmetic** (this task's own
§7 trap), and it changed the shape of the answer: the model that
matters most is already in this checkout. It just isn't profiled far
enough to prove it the way this project's other tools do.

---

## 0. What "the winning arrangement" actually constrains

Before any numbers: Task 33's own winning arrangement is DECODE_ATTN's
own tensor-parallel group — two GPUs, one domain. In `pd-af-disaggregation`,
DECODE_ATTN, DECODE_FFN, and PREFILL are separate clusters on separate
devices. `frontier/utils/param_counter.py`'s own `ClusterType.DECODE_ATTN`
branch returns **zero** MLP/MoE parameters — DECODE_ATTN's device memory
holds attention weights (Q/K/V/O) only. This is exactly what Tasks
24/25/28's own calibrated `feasible_num_blocks` table already measures
(confirmed by reproducing it from Frontier's own formula, bit for bit —
§1.1 below), and it is the quantity this task's own Part A question is
actually about: **not the model's total size, but its attention weight
specifically.** A model with enormous MoE/FFN weights and modest
attention weights would not force DECODE_ATTN's own placement to split
at all — the two are decoupled by the architecture itself.

---

## 1. Part A — What "large enough" means

### 1.1 The precision Frontier assumes, checked directly

`frontier/scheduler/utils/memory_planner.py:119`:

```python
def _get_parameter_memory_per_device(self) -> int:
    return 2 * self._param_counter.get_num_parameters_per_device()
```

**Two bytes per parameter, unconditionally** — Frontier's own
`MemoryPlanner` ignores any declared `quantization_config` entirely.
Confirmed against `Llama-3.1-405B-Instruct-FP8`'s own HF config (which
declares `fbgemm_fp8`, one byte per parameter): Frontier still budgets
it at 2 bytes/param internally. This is the precision this task's own
§6.1 deliverable asks for — not the model's own declared dtype.

**The attention-only formula reproduces Tasks 24/25/28's own calibrated
table exactly**, checked by hand before trusting anything downstream:
for Phi-tiny-MoE-instruct (`h=4096, head_dim=256, num_q=16, num_kv=4,
layers=32`), `get_num_attention_params_per_layer`'s own formula
(`embedding_dim * head_dim * (q_per_worker + 2*kv_per_worker) +
embedding_dim * head_dim * q_per_worker`, `kv_per_worker = ceil(num_kv
/ tp)`) gives 1,342,177,280 bytes at tp=1 and 201,326,592 at tp=8 — both
bit-identical to `_PARAM_MEM_BYTES` in `tools/planner.py`, itself cited
from Tasks 25/26/28. The GQA ceiling (`kv_per_worker` cannot fall below
1) is why tp=8's figure is *not* exactly 1/8th of tp=1's — the same
plateau Tasks 22-28 already found, now traced to its exact mechanism
rather than cited.

### 1.2 The threshold, memory-forced only (compute-forced needs profiles this project doesn't have — see §1.3)

Real device memories, not assumed: `frontier/config/device_sku_config.py`'s
own SKU table — A40 45GB, A100/A800/H100/H800 80GB, H20/RTX_PRO_6000
96GB, H200 141GB, MI355X 288GB. Real domain sizes: `build_node_scale`'s
own default (`gpus_per_machine=8`) and `build_rack_scale`'s own default
(`trays_per_rack=18 * gpus_per_tray=4 = 72`) — "the two the fabric
builders already support," confirmed from source, not assumed from the
spec's own claim.

Threshold **attention weight at tp=1**, in GB, beyond which even
filling the whole domain (tp=8 or tp=72) still exceeds the budget —
computed for two GQA shapes bracketing the real models in §2 (`head_dim=128`,
`num_layers=80`, chosen to land near Llama-3.1-405B's own real
proportions):

| domain | device | margin | favorable GQA (kv_heads=8) | tight GQA (kv_heads=1) |
|---|---|---|---|---|
| 8 | H800 (80GB) | 0.9 (realistic) | 64.0 GB | 52.8 GB |
| 8 | H800 (80GB) | 0.5 (generous) | 320.0 GB | 264.0 GB |
| 8 | A40 (45GB) | 0.9 | 36.0 GB | 29.7 GB |
| 8 | MI355X (288GB) | 0.9 | 230.4 GB | 190.1 GB |
| 72 | H800 (80GB) | 0.9 | 304.9 GB | 182.8 GB |
| 72 | H800 (80GB) | 0.5 | 1524.7 GB | 913.8 GB |
| 72 | MI355X (288GB) | 0.9 | 1097.8 GB | 658.0 GB |

**Task 28's own margin (0.992) is not used here** — at 0.8% usable
memory on an 80GB device (0.64 GB), it is an artificial stress point
Task 28 chose specifically to make a 4B model's tp=1 infeasible, not a
margin any real deployment would run at. §2 shows what it does to a
100B+-class model: it makes *everything* look infeasible, which is not
informative. 0.5-0.9 (10-50% usable) is the range worth reading these
thresholds at.

**The domain-8 threshold (52-64 GB attention weight at tp=1, at a
realistic margin) is not an exotic number.** §2 finds a real, already-
profiled model within a factor of ~2-3x of it.

### 1.3 Memory-forced vs. compute-forced — only the first is answered here

Per this task's own trap: everything above is **memory-forced**
(`feasible_num_blocks`'s own mechanism — a placement is infeasible
before any compute or communication cost is ever priced). **Compute-
forced** — a high degree winning on latency *despite* its communication
cost, even when a lower degree would also fit — is a different claim
and needs real execution-time profiles at the degrees being compared.
No such claim is made here. Task 33's own finding (tp=2 beats tp=4 at
short decode, tp=4 beats tp=2 at long decode, both real profiled
degrees) is the only *compute*-driven degree preference this project
has actually measured, and it never needed to leave a single domain to
show it.

---

## 2. Part B — Is such a model already available?

### 2.1 The inventory

Every model profiled for h800 or rtx_pro_6000 — the only two devices
with attention/linear_op/MoE profiles beyond a single stub file, per
this task's own citation, confirmed by listing every device directory
directly (`a100`/`a40`/`a800`/`h100`/`mi355x` all either have a single
`mlp.csv` per model or nothing at all in the checked examples;
`a100/meta-llama/Meta-Llama-3-70B` has one `mlp.csv`, `h100/meta-llama/Meta-Llama-3-70B`
has zero files):

| device | model | layers | hidden | attn config | attn weight @ tp=1 (2B/param) |
|---|---|---|---|---|---|
| h800 | Llama-3.1-405B-Instruct-FP8 | 126 | 16384 | 128 heads / 8 kv (GQA) | **133.875 GB** |
| h800 | step-moe-noquant-small | 31 | 7168 | 64 heads / 1 kv (GQA) | 13.774 GB |
| h800 | llama2_7b_dense_example | 32 | 4096 | 32 heads / 32 kv (MHA) | 4.000 GB |
| h800 | Phi-tiny-MoE-instruct | 32 | 4096 | 16 heads / 4 kv (GQA) | 1.250 GB |
| h800 | Qwen3-30B-A3B-tiny | 8 | 2048 | 32 heads / 4 kv (GQA) | 0.281 GB |
| h800 | qwen3-next-80b-a3b-instruct-reduced-l2 | 2 | 2048 | 16 heads / 2 kv (GQA) | 0.070 GB |
| h800 | Step2Mini-tiny | 8 | 2048 | 16 heads / 1 kv (GQA) | 0.133 GB |
| rtx_pro_6000 | Qwen3-30B-A3B-tiny | 8 | 2048 | 32 heads / 4 kv (GQA) | 0.281 GB |
| rtx_pro_6000 | qwen2_dense_test | 24 | 1024 | 16 heads / 16 kv (MHA) | 0.188 GB |

**Every "80B"/"30B"-named model here is a deliberately reduced test
config** — `qwen3-next-80b-a3b-instruct-reduced-l2` has 2 transformer
layers (the real model has dozens), `Qwen3-30B-A3B-tiny` has 8,
`Step2Mini-tiny` has 8 — the name describes the architecture family
being exercised, not the size actually profiled. Only `Llama-3.1-405B-Instruct-FP8`
and `llama2_7b_dense_example` are configured at their real, full public
scale.

### 2.2 Does anything clear the threshold?

**Yes — `Llama-3.1-405B-Instruct-FP8`, at a realistic margin, on the
device it is already profiled for:**

| attn_tp | attention memory (h800, 80GB) | feasible at margin ≤ | usable memory needed |
|---|---|---|---|
| 1 | 133.875 GB | never (exceeds the whole device) | — |
| 2 | 66.938 GB | 0.16 | 83.7% |
| 4 | 33.469 GB | 0.58 | 41.8% |
| **8** | **16.734 GB** | **0.79** | **20.9%** |
| 16 | 8.859 GB | 0.89 | 11.1% |
| 32 | 4.922 GB | 0.94 | 6.2% |

**For any margin from about 0.58 to 0.79 — 21% to 42% of an 80GB H800
reserved for weights, an entirely ordinary operating point, not a
stress test — this model's smallest feasible `attn_tp` is exactly 8.**
That exactly fills a `build_node_scale`-default 8-GPU domain and
**exceeds** a 4-GPU domain. And tp=8 is inside the model's own profiled
range (§2.3) — this is not a number that needs new profiling to use.

`step-moe-noquant-small` (the largest MoE-family profile) does **not**
clear the domain-8 threshold at any realistic margin — its attention
weight is over an order of magnitude smaller than the 405B model's,
despite MoE parameters dwarfing dense ones in *total* size, because
attention weight scales with `hidden_size` and head count, not with how
many experts a model has. This is the direct, checkable consequence of
§0's own scoping point.

### 2.3 The gap that keeps this from ending here

`Llama-3.1-405B-Instruct-FP8`'s own profile — like **every** model
checked, on both devices — covers `num_tensor_parallel_workers` ∈
`{1, 2, 4, 8}` only, confirmed by reading every `attention.csv` and
`linear_op.csv` directly, not inferred:

```
frontier/profiling/moe/moe_input.py:79:  tensor_parallel_size_list = [1, 2, 4, 8]
frontier/profiling/linear_op/main.py:270: --num_tensor_parallel_workers ... default=[1, 2, 4, 8]
```

**This is the profiler's own default, not a hard ceiling** — the CLI
flag is `nargs="+"`, freely overridable — but every profile in this
checkout was produced with the default left unchanged. Two direct
consequences:

1. **Domain size 72 is untestable with any current profile, for any
  model, regardless of size.** Nothing has ever been profiled past
  tp=8, so no arrangement past tp=8 can be evaluated by Frontier's
  execution-time predictor at all — the question "does an arrangement
  that must split past 72 win" cannot be asked of the current
  inventory, independently of §1's own memory arithmetic.
2. **Domain size 8 is, by the same logic, at its ceiling exactly where
  the 405B model's own memory threshold lands** (tp=8, §2.2) — which
  is why §3's own domain-shrinking substitute, not domain-72, is where
  this task's own arithmetic and the current inventory actually meet.

**So: no, strictly — nothing in the inventory demonstrates a split
*past* domain size 8 today, because nothing is profiled past tp=8 at
all.** But this is not the same finding as "no model is big enough" —
one already is (§2.2); the gap is entirely in how far it was profiled,
not in what it weighs.

---

## 3. Part C — What acquiring the missing piece would cost

**Framing correction from what the spec anticipates**: the missing
piece is not a bigger model. It is more `attn_tp` coverage on a model
already in this checkout. Concretely, three separate costs:

**Extending `Llama-3.1-405B-Instruct-FP8`'s own profile past tp=8**
(needed only for a domain-72 demonstration, not for §2.2's domain-4-
vs-8 result, which already works at tp=8): a `--num_tensor_parallel_workers`
flag change on an existing tool, but not a cheap one to run.
`frontier/profiling/linear_op/main.py`'s own docstring: multi-GPU
profiling spawns one real process per tensor-parallel worker (Ray or
`torch.multiprocessing`) — profiling at tp=16 needs 16 real, physical
GPUs present *simultaneously*, tp=72 needs 72. This is the "hardware,
not software" constraint this task's own Part C names, confirmed from
the profiler's own multi-GPU mode documentation rather than assumed.

**Training cost scales with corpus size, already measured once for
this exact model.** Task 12's own report: a *first* real-mode run
against `Llama-3.1-405B-Instruct-FP8` (already at the tp∈{1,2,4,8}
scope that ships today) had its sklearn/RandomForest predictor training
step still running after 10+ minutes (6 CPU-bound workers) before being
killed — "training time scales with the profiling data's size and
complexity." Adding tp=16/32/64/72 would roughly double the swept
degree count, and each additional degree's own GPU-collection step (not
just the training step) also costs real wall-clock time on real
hardware. This is a real, non-trivial cost, not a formality — but it is
a **profiling cost**, distinct from **acquiring** a model, since the
model itself needs no acquisition at all.

**No new model needs to be found or trained.** Frontier's profiler
targets an attention-backend abstraction, not any hand-picked list —
the spec's own hint is correct — but that only matters if the *size*
were the blocker. It isn't; §2.2 already has one at the right size.

### 3.1 Is a smaller domain a legitimate substitute?

**Yes, for domain size 4 specifically — and it works today, with zero
new profiling or acquisition, which shrinking to 4 does and growing the
model does not.**

`build_node_scale(gpus_per_machine=4)` (already expressible — Task 32's
own fabric) plus `Llama-3.1-405B-Instruct-FP8` at margin ≈ 0.6-0.79 on
an 80GB H800 gives smallest-feasible-`attn_tp`=8, which **exceeds** a
4-GPU domain, using a tensor-parallel degree that **is already
profiled**. Task 32's own §2 table shows an 8-way group split across a
5×4 fabric produces genuinely varied shapes (`(4,3,1)`, `(3,3,2)`, and
seven others) — the placement search machinery Task 32/33 already
built needs no change at all to run this.

Weighing legitimacy against contrivance, honestly:

- **In favour of legitimate**: it is not a fabric nobody would build.
  4-GPU scale-up domains are real (this project's own `build_node_scale`
  ships them as a first-class parameter, not a hack), and plenty of
  real hardware generations shipped 4-way NVLink/xGMI domains before
  8-way became the norm. Testing "does the planner's answer change
  between a 4-GPU-domain fabric and an 8-GPU-domain one" is testing a
  transition that has genuinely happened in real hardware history, not
  an invented one.
- **Against**: the *reason* this specific pairing separates is that a
  405B-parameter model's attention weight happens to cross the 4-vs-8
  boundary at an unremarkable margin — a coincidence of scale, not a
  deliberately chosen "hard" case. And it demonstrates the memory-forced
  claim only (§1.3) — it says nothing about whether a split arrangement
  could ever be *faster* than a whole one, which is the compute-forced
  question this project has never been able to answer without profiles
  it doesn't have.
- **Verdict**: legitimate as a demonstration that the planner's answer
  is genuinely fabric-dependent for *some* real, already-available
  configuration — which is exactly what Task 33 could not yet show.
  Not a substitute for a domain-72 result, and not evidence about
  compute-forced preference. It is a smaller, honest claim than "topology
  matters at scale," and should be reported as that smaller claim, not
  inflated to stand in for it.

---

## 4. What to report — anywhere this specification is wrong

1. The spec's own framing ("acquiring" a model) undersold the actual
  gap. A model that clears Part A's own threshold is already in this
  checkout (`Llama-3.1-405B-Instruct-FP8`, §2.2); the missing piece is
  `attn_tp` profiling coverage past 8, not model size or availability.
  This is not a correction of a factual error, but the spec's own
  Part C already half-anticipates it ("its profiler has an attention-
  backend abstraction, so the constraint is likely hardware rather than
  software") — confirmed true, but for the wrong reason: hardware limits
  *profiling coverage*, not model acquisition.
2. Everything else checked — "only h800 and rtx_pro_6000 carry
  full-feature profiles," "the two [domain sizes] the fabric builders
  already support," "an earlier attempt at a 405B model was abandoned
  after ten minutes of predictor training without completing" (Task
  12's own report, verified directly) — matched what was found. No
  fabricated citation, unlike several earlier tasks in this project's
  own history.

## What shipped

- `docs/tasks/35-model-sizing-report.md`, this report. Arithmetic and
  inventory only — no source file, config, or test changed, per this
  task's own acceptance criteria. One commit on `task-35-model-sizing`,
  branched from `task-33-planner`'s tip.
