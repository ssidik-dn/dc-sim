# Task 46 — What is the host actually good for?

Branch: `task-46-host-audit`, branched from `task-34-packed-audit-verify`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier at
`/work/simulation/Frontier`.

240 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0. Investigation and arithmetic only — no source changed, per this
task's own §3. Part B.3's own comparison was run (§6), since the code could
settle it directly; nothing else in this report required a simulation run
beyond that one.

---

## Part A — Accuracy on a single accelerator

### A.1 What the claim is, and whose

**Frontier's own paper** (arXiv:2605.21312, fetched directly — no local copy
ships in this checkout) states, quoted verbatim from its abstract:

> "On 16-H800 GPU testbed, Frontier achieves an average throughput error
> below 4%. Compared with state-of-the-art simulators, it reduces
> end-to-end latency error from 44.9% to 6.4% under co-location and from
> 51.7% to 2.6% under disaggregation."

So: the claimed baseline is **real hardware** (a 16×H800 testbed — the same
device family, `h800`, this project's own studies use throughout), the
comparison is against **prior simulators'** own error on that same ground
truth (44.9%/51.7%), and Frontier's own claimed error is **6.4%** (co-location)
and **2.6%** (disaggregation) for latency, **<4%** for throughput. The
abstract names no specific model and does not mention vLLM, SGLang, or any
serving-engine scheduling claim at all — that claim lives elsewhere (README,
roadmap; see Part B).

**No documented discrepancy in the repository contradicts this claim
directly** — but the repository's own `profiling_knowledge/` notes (internal
engineering records, not the paper) document two *real, already-found*
fidelity gaps, and they are usefully different in kind, confirming this
task's own §5 "three failures wear the same name" framing:

1. **Profiled-data-itself gap** (`AITER_KERNELS.md`): "Frontier's own
   profiling, by contrast, has been using `TORCH_SDPA` — a portable,
   correctness-first reference backend, explicitly documented... as **not
   peak-tuned**. That gap is a real, plausible contributor to any large
   real-vs-simulated mismatch you see in a comparison report." This is
   MI355X/ROCm-specific (AMD's own `aiter` kernel library vs the reference
   `TORCH_SDPA` backend) — **it does not apply to `h800`**, the device this
   project's own studies actually use; h800 profiling goes through the
   standard NVIDIA/FlashInfer path (`HARDWARE_COOKBOOK.md`), not this
   AMD-specific gap. Cited here as evidence that this *class* of gap is real
   and self-documented, not that it applies to this project's own device.
2. **Predictor gap** (`GPTOSS_TRUE_MIXED_BATCH_PROFILING.md`): "The
   execution-time predictor (a RandomForest) is trained on the profiled
   grid, then queried via a separately widened lookup grid... the model was
   silently **extrapolating** far outside its training range... RandomForests
   cannot extrapolate meaningfully; a tree just returns whatever its
   rightmost leaf learned." This is device- and model-independent: it is a
   property of the RandomForest predictor itself, and applies to `h800` (or
   any device) whenever a real workload's request lengths exceed whatever
   grid a model was profiled against.

Neither gap concerns "something in the simulation loop" (the discrete-event
scheduling/timing mechanics) — no such gap is documented anywhere searched
(README, `docs/`, `profiling_knowledge/`). The two concerns found are squarely
**predictor** (RandomForest extrapolation) and **profiled data** (kernel
backend choice) — exactly the first two of the three categories this task's
own §5 warns not to conflate, and the original deferred concern ("frontier is
not accurate even on single gpu") **is itself scoped precisely by its own
wording**: a single GPU has no cross-GPU communication to model at all, so
"not accurate even on single GPU" can only be about the execution-time
predictor / profiled operator data — never about communication modeling
(TP allreduce, M2N transfer), which is priced by a completely separate model
(the analytical/vidur CC backend) that this concern cannot be about.

### A.2 What it would cost this project if true — the sensitivity bound

Reusing Task 20's own already-recorded tp=4 figures (`docs/tasks/20-collective-backend-report.md`,
its own real h800 measurement, packed vs split TP placement — the same
comparison Task 42/43A/45 all cite as this project's own most-repeated
headline): **packed = 5.803319ms, split = 10.929719ms, delta = +5.126400ms,
ratio = +88.33%.**

**The compute component is structurally identical in both arms.** Placement
(packed vs split) changes *which links* the TP all-reduce crosses — priced by
the CC backend, a model completely separate from the RandomForest execution-time
predictor — but **not** how much per-device compute work happens: same TP
degree, same model, same FLOPs per device, regardless of which domain each
device sits in. Writing `total = C + T` (`C` = compute-predictor contribution,
`T` = CC-backend transfer contribution, both real, unknown individually from
the published figures but structurally separable by construction):

```
packed = C + T_packed = 5.803319
split  = C + T_split  = 10.929719
```

**A uniform compute-model error of any size leaves the absolute delta exactly
unchanged**, not merely "very nearly unchanged": scaling `C` by `(1+e)` gives
`packed' = C(1+e) + T_packed`, `split' = C(1+e) + T_split`, and
`split' − packed' = T_split − T_packed = 5.126400ms` — `C` cancels completely,
for any `e`, because it is identical in both arms. This is the strongest, most
useful form of the claim this task's A.2 asks for: it doesn't need to know `C`
at all.

**The ratio does move, and is bounded** even without knowing `C` precisely,
using the trivial bound `0 ≤ C ≤ packed` (transfer cost is non-negative):

| `e` | worst-case `C` (=`packed`, i.e. `T_packed≈0`) | packed′ | split′ | ratio′ |
|---|---|---|---|---|
| +0.20 (compute overstated) | 5.803319 | 6.963983 | 12.090383 | **+73.62%** |
| −0.20 (compute understated) | 5.803319 | 4.642655 | 9.769055 | **+110.44%** |

So the headline "+88.3%" figure could, in the **worst case** (`C` equal to the
*entire* packed baseline — an extreme, almost certainly unrealistic
assumption, since real TP communication is known to be nonzero even packed:
this project's own component-ledger data, e.g. Task 34's own audit, shows
substantial real `tensor_parallel_communication_time` in every packed
configuration checked), range from **+73.6% to +110.4%** under a ±20%
compute-only error. The true range is almost certainly much tighter — this
project has never isolated a clean, per-token compute-only figure to pin `C`
down further, which is itself a real, specific, closable gap (see A.3).

**The absolute figure moves directly and proportionally to `C`'s own share.**
`packed' − packed = C·e`; in the same worst case, that's up to `±20%` of the
entire reported `5.803ms` — meaning any SLO check phrased as a hard threshold
(*"is TPOT ≤ N ms"*) can flip if the true figure is within 20% of the
threshold and compute is a large share of it. **A ratio is far more robust to
a compute-model error than an absolute figure — exactly this task's own
claim, now shown with a closed-form bound rather than asserted.**

### A.3 What would settle it — costed, not attempted

An actual accuracy check needs, concretely: (1) a profiled model on profiled
hardware, (2) a real deployment of it at a known, reproducible workload, (3)
Frontier's own prediction for the identical scenario, (4) a comparison
metric. Checking what this project **has** versus **does not have**, in this
checkout specifically:

| Needed | Present? |
|---|---|
| Profiled model/hardware pairs | **Yes** — `h800`/`rtx_pro_6000` carry full-feature profiles (Task 35's own finding); several models (Phi-tiny-MoE-instruct, Llama-3.1-405B-Instruct-FP8, llama2_7b) already used throughout this project. |
| A pipeline to run a real workload through Frontier and extract comparable metrics | **Yes** — `tools/validation/` (`real_log_parser.py` → `real_log_aggregator.py` → `frontier_cli_translator.py` → `metrics_extractor.py` → `compare_plots.py`, orchestrated by `run_validation.py`), reads both vLLM and SGLang `bench_serving` logs. |
| Real hardware deployment logs (the ground truth) | **No** — `tools/inference_bench/<model>/<engine>/` is where the validation tool's own docs say this data lives; it does not exist in this checkout (`ls tools/inference_bench` → no such directory). The tooling was clearly built and used somewhere; this checkout has the pipeline but not its input. |
| A clean, already-completed comparison | **No** — no `comparison_report.html` or equivalent output exists anywhere under `outputs/` in this checkout, and the tool's own docs report two **unfixed** data-quality bugs that would corrupt several of the possible comparisons if run naively: an open-loop QPS-source bug (`VALIDATION_TOOL.md`) and a decode-length mismatch for gpt-oss (`REAL_BENCHMARK_DATA_QUALITY.md`, e.g. real gpt-oss-120b/vLLM decode length ~60-70 tokens vs. Frontier's own simulated 1024-token target — "a large, uncorrected workload mismatch that will make Frontier look far slower/higher-latency than the real system for entirely the wrong reason"). |

**Cost, concretely**: even with the pipeline already built, a trustworthy
comparison needs (a) real capture data this checkout does not have — a real
deployment run, which is exactly the GPU-hours + wall-clock this simulator
exists to avoid, so validating it costs a slice of what it's meant to save —
and (b) fixing the two already-documented, already-diagnosed data-quality
bugs first, or the comparison would be measuring the bug, not the simulator.
Neither was attempted here, per this task's own instruction; this is the
costed statement of what it would take, not a result.

---

## Part B — Scheduler coverage

### B.1 What is actually implemented

**Cluster level** (`frontier/scheduler/cluster_scheduler/`, `ClusterSchedulerRegistry`,
5 registered, matching `ClusterSchedulerType`'s own 5 members exactly — no
unclaimed slot):

| Scheduler | Modeled on | Works in disaggregated mode? |
|---|---|---|
| `ROUND_ROBIN` | Generic round-robin load balancing (not engine-specific) | **Yes** — `schedule()` has explicit `DECODE`/`DECODE_ATTN`/`DECODE_FFN` branches, genuinely implemented, not merely tolerated (confirmed by reading the method body) |
| `STICKY_ROUND_ROBIN` (subclasses `ROUND_ROBIN`) | Session-affine round-robin | **Architecturally yes** (inherits the same unrestricted `schedule()`), **but requires `request.session_id`** on every request — this project's own `synthetic` request generator never sets one, so it is unreachable in practice, not by an architecture guard |
| `RANDOM` | Generic random load balancing | **No** — `schedule()`: `if self._cluster_type != ClusterType.MONOLITHIC: raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)` |
| `LOR` (least outstanding requests) | Generic load-based balancing | **No** — identical guard, confirmed by direct read (Task 15's own finding, reconfirmed here) |
| `STICKY_LOR` (subclasses `LOR`) | Session-affine LOR | **No** — inherits `LOR`'s own guard |

Task 15's own "only one of Frontier's five... `round_robin` is the only one"
is accurate for what it tested (round_robin vs. LOR) but slightly
under-states the full picture: `STICKY_ROUND_ROBIN` is *also* architecturally
unrestricted, just gated by a workload requirement (`session_id`) rather than
a cluster-type guard — a real, if practically inert, distinction (§7).

**Replica level** (`frontier/scheduler/replica_scheduler/`, `ReplicaSchedulerRegistry`,
**10** registered, matching `ReplicaSchedulerType`'s own 10 members exactly —
again no unclaimed slot):

| Scheduler | Modeled on (evidence of the code, not the name) | Cluster-type restriction? | Exercised by any dc-sim study? |
|---|---|---|---|
| `VLLM_V1` | vLLM's v1 engine (paged, iteration-level batching, chunked prefill, speculative decoding Phase 1) | None (blocks only if speculative decoding is on, which this project never enables) | **Yes** — every dc-sim tool's own DECODE_ATTN scheduler |
| `ORCA` | Orca (continuous batching, the original iteration-level scheduling paper) | None found | **Yes** — every dc-sim tool's own DECODE_FFN scheduler |
| `VLLM` (legacy) | vLLM's original (pre-v1) scheduler | None found (its own `raise ValueError` is a KV-transfer bookkeeping invariant, unrelated to cluster type) | No |
| `SARATHI` | Sarathi-Serve (chunked-prefill + decode-maximal batching) | None found | No |
| `LIGHTLLM` | LightLLM's own token-level scheduling | None found | No |
| `FASTER_TRANSFORMER` | NVIDIA FasterTransformer (static/batch-level) | None found | No |
| `SGLANG` (`SGLangStyleReplicaScheduler`) | **Not SGLang's own distinguishing logic** (no radix-tree prefix-cache-aware scheduling, no SGLang-specific memory manager) — its own docstring: *"intentionally reuses the existing Frontier vLLM v1 allocation and preemption helpers. The only behavioral change is the high-level scheduling order: prefer schedulable prefill-stage work first, fall back to decode."* A vLLM-v1 subclass with one reordering policy, named after SGLang's own known preference for prefill-priority scheduling — this is exactly this task's own known trap (B.1: "'following X' is a claim about behaviour, not a name") turned around: something *named* SGLang that, on the evidence of the code, is a vLLM-v1 policy variant. | **Yes** — `__init__`: `if self._cluster_type not in (MONOLITHIC, PREFILL): raise ValueError(...)` | No (blocked from DECODE_ATTN/DECODE_FFN, which is all any dc-sim study has ever used) |
| `SJ2Q_PENALTY_ONLY`, `SJ2Q_BOUNDED_CARRYOVER`, `SJ2Q_FASTSERVE_LITE` | Research variants (queue-penalty / bounded-carryover / fast-serve-lite scheduling policies), all subclass `VLLMv1EngineReplicaScheduler` | **Yes**, same pattern: each has its own `if self._cluster_type not in {MONOLITHIC, PREFILL}: raise ValueError(...)` in at least one code path | No |

**So at the replica level, unlike the cluster level, most schedulers are not
blocked by cluster type** — only `SGLANG` and the three `SJ2Q_*` variants
carry an explicit disaggregation guard (4 of 10); the rest (`VLLM`, `SARATHI`,
`LIGHTLLM`, `FASTER_TRANSFORMER`) have no such restriction found, though none
of them is actually exercised by any study in this project — only `VLLM_V1`
and `ORCA` are. This is a real difference from the cluster level (1 of 5
usable in disaggregated mode) worth stating precisely rather than assuming
the same ratio carries over — it does not.

**On the README's own "It currently simulates vLLM-logic serving behavior"
claim**: read literally against the registry, this understates what exists.
Frontier implements distinct, named batching policies for several real
systems (Orca, Sarathi, LightLLM, FasterTransformer, vLLM, vLLM-v1), not only
vLLM — and this project's own studies already exercise **two** of them
(`VLLM_V1` for DECODE_ATTN, `ORCA` for DECODE_FFN), not one. "vLLM-logic" most
plausibly refers to the overall scheduler-batch-engine *loop architecture*
(iteration-based, vLLM-inspired even when a differently-named policy like
Orca is plugged into it) rather than a claim that only vLLM's own specific
heuristic exists — but the README's own wording does not make that
distinction explicit, and read plainly, it is easy to come away thinking
"only vLLM," which is not accurate to the registry.

### B.2 Whether a second family is reachable

This task's own predictive heuristic ("count the unclaimed members of the
relevant enumeration") **gives the wrong signal here, and that is itself
worth reporting.** Both `ClusterSchedulerType` and `ReplicaSchedulerType` have
**zero unclaimed members** — every enum value is registered to a real class.
Applied naively, this predicts "no second family reachable without a new
registration" — but `SGLANG` already exists, is already registered, and is
already selectable via `--replica_scheduler_config_type sglang` on the CLI.
The heuristic was built (and validated three times: transfer paths, CC
backend, cluster scheduler) for the question "does someone need to *write and
register* a new implementation" — it does not capture a *different* kind of
unreachability this task turned up: **an already-implemented class,
functionally gated away from the one architecture this project actually
uses**, by a guard clause unrelated to registration at all.

Given that, the honest answer: reaching `SGLangStyleReplicaScheduler` for
DECODE_ATTN/DECODE_FFN is **not** registration (done), **not** a runtime/CLI
replacement (the flag exists and is accepted; it raises immediately once
`Simulator` constructs the scheduler for a disaggregated cluster type) — it
requires an **upstream code change**: relaxing or removing the
`if self._cluster_type not in (MONOLITHIC, PREFILL): raise ValueError(...)`
guard in `SGLangStyleReplicaScheduler.__init__` (and the equivalent guards in
the three `SJ2Q_*` schedulers, if those are wanted too). This looks like a
**small, low-risk** change in principle — the class already reuses its own
parent's (`VLLMv1EngineReplicaScheduler`) allocation/preemption logic, which
is already proven correct for DECODE_ATTN/DECODE_FFN by this project's own
extensive use of it — but it is still a change to Frontier's own scheduler
code, which every task in this project's history has treated as pinned and
unmodified (not literally under `dc-sim/upstream/`, since Frontier is a
separate checkout at `/work/simulation/Frontier`, but consistently
never-touched in practice for the same reason `upstream/` itself is never
touched) — out of this project's own established scope, not a fundamental
research problem.

### B.3 What it would change — settled directly, not left as speculation

Two schedulers (`vllm_v1`, `sglang`) are both reachable in exactly one
architecture (`co-location`/MONOLITHIC — the one place `SGLangStyleReplicaScheduler`'s
own guard permits) — so, per this task's own instruction, one comparison
answers it directly rather than by argument. Real h800 compute, same model
(Phi-tiny-MoE-instruct), three configurations:

| config | `vllm_v1` mean tpot / throughput | `sglang`-style mean tpot / throughput |
|---|---|---|
| generous batch budget (`max_tokens_in_batch=4096`), 32 req @ qps=20 | 16.4984ms / 120.794 rps | 16.4984ms / 120.794 rps — **identical** |
| tighter batch budget (`max_tokens_in_batch=256`), same workload | 16.8413ms / 94.362 rps | 16.4984ms / 120.794 rps — **real, if modest, difference** |
| tight `num_blocks=40`, chunked prefill on, busier arrivals (64 req @ qps=60) | 15.7710ms / **1.941 rps** | 16.6183ms / **60.129 rps** — **31x throughput difference** |

**Scheduler choice absolutely can change conclusions — demonstrated, not
speculated.** Under enough contention (tight memory budget, mixed
prefill/decode batches, busy arrivals), which scheduling policy prioritizes
prefill-vs-decode admission produces a 31x throughput gap in this single
real run. Under a slack budget, the two are bit-identical, because there is
never a genuine admission choice to make differently. This is exactly Task
42's own established mechanism (*"how many requests are in flight is exactly
what reverses sizing conclusions"*) showing up on a *scheduling* axis, not
only a *sizing* one — confirming this task's own framing that a different
scheduler is a plausible third regime, and settling (for the one reachable
architecture) that it is a **real** one, not merely a plausible one.

**The caveat that matters**: this was only measurable in `MONOLITHIC`, the
one architecture where both schedulers are reachable — not in the
`pd-af-disaggregation` architecture this project's own studies (32 onward)
actually use. Whether the *same* 31x-style effect would appear in the
disaggregated architecture this project cares about is **not answerable now**
without the upstream change in B.2 — this task settles that scheduler choice
*can* matter, not that it *does* matter for this project's own studies
specifically.

---

## Anywhere this specification is wrong

1. **"Two unclaimed members for each transfer path, none for the collective
   backend or the cluster scheduler"** is accurate as a historical record,
   but the "count unclaimed members" heuristic itself does not transfer
   cleanly to the replica scheduler: `ReplicaSchedulerType` also has zero
   unclaimed members, yet a second family (`SGLANG`) is still not reachable
   in this project's own architecture — for a reason (a functional guard, not
   a missing registration) the heuristic was never built to detect. Worth
   flagging as a real limit of an otherwise well-validated rule, not a
   citation error.
2. **"Frontier is not accurate even on single GPU"** turns out to be a
   precisely-scoped concern once checked: since a single GPU has no
   cross-GPU communication, this can only be about the execution-time
   predictor / profiled operator data (confirmed: two real, documented gaps
   of exactly that kind exist), never about the separately-modeled CC
   backend. The spec's own framing (A.1) anticipated this distinction
   correctly.
3. **"Frontier simulating only vLLM"** is directionally right for what this
   *project's own studies* exercise (`VLLM_V1` + `ORCA`, both vLLM-adjacent
   or vLLM-predating designs) and right that SGLang is not usable in this
   project's own disaggregated architecture — but is not right about what
   Frontier's own codebase *contains* (Orca, Sarathi, LightLLM,
   FasterTransformer, vLLM, vLLM-v1, and an — architecturally limited —
   SGLang-named variant all exist as real, distinct implementations, not
   just vLLM). The roadmap's own "Long-term: Serving Engines Integration —
   Support for SGLang and TensorRT-LLM" is consistent with this: it most
   plausibly refers to genuine SGLang-behavior modeling (radix caching,
   SGLang's own memory management), which does not exist yet, not to the
   narrower "is anything named SGLang present at all" question, where the
   answer is already yes (just not reachable here).
4. **Otherwise nothing else checked in this specification was wrong** — the
   paper's own accuracy claim (A.1), the "ratio robust / absolute figure not"
   distinction (A.2, now bounded rather than asserted), and the instruction
   not to attempt validation (A.3) all held up exactly as framed.

## What shipped

No source changes — an investigation and arithmetic task, per its own
acceptance criteria, plus one real comparison run (B.3) where the task
explicitly invited running it. `docs/tasks/46-host-audit-report.md` only.

One commit on `task-46-host-audit`, branched from `task-34-packed-audit-verify`'s
tip. 240 tests pass, unchanged; `check_import_direction.py` exits 0.
