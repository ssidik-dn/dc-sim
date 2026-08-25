# Task 47 — Does the scheduling regime matter where it counts?

Branch: `task-47-scheduler-regime`, branched from `task-46-host-audit`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`.

249 tests pass (240 unchanged + 9 net new), and `python3
tools/check_import_direction.py` exits 0. Task 33's own sixteen-row table
and Task 36's own two-fabric result both reproduce bit-identically **with
the guard patch installed and the original policy selected** — this task's
own required proof that the patch changes what is reachable and nothing
else.

---

## 1. Whether the guard is load-bearing

**The guard this task patches is not load-bearing — but a completely
separate, genuinely load-bearing guard exists, was found while checking,
and was not touched.** Both established directly from the code, not
inferred.

### 1.1 `SGLangStyleReplicaScheduler.__init__`'s own guard — not load-bearing

Read in full (345 lines). Beyond the constructor's own refusal, `_cluster_type`
appears in exactly two other places in this class, both purely for a log
line's own label (`_emit_schedule_decision_event`, `_schedule_two_phase`'s
own logger name) — never as a behavioral branch. The class's own new logic
(`_schedule_prefill_stage_first`, `_schedule_decode_fallback_running_requests`,
`_is_prefill_stage_request`, `_get_split_waiting_requests`) inspects only
request state (`is_prefill_complete`, `_preempted`) and calls only into its
parent's (`VLLMv1EngineReplicaScheduler`) own `_schedule_running_requests`/
`_schedule_waiting_requests`/`_create_batch` — already exercised for
`DECODE_ATTN`/`DECODE_FFN` by every real-compute study this project has ever
run. The two inherited "monolithic_pp_terminal_release" helpers
`_schedule_two_phase` calls unconditionally are themselves internally gated
(`_get_monolithic_pp_extra_terminal_release_iters`: `if self._cluster_type
!= ClusterType.MONOLITHIC: return 0`) and no-op for every other cluster
type — true for the parent class's own callers already, so nothing new is
at risk. **The guard is a conservative "not yet validated for this
combination" gate, not a "known to be broken" one.**

### 1.2 But its own added logic is unreachable for `DECODE_ATTN` — a structural finding, not a defect

`VLLMv1EngineReplicaScheduler`'s own top-level dispatcher
(`_get_next_batch`, its own docstring: *"Route to cluster-specific
scheduling"*) sends `PREFILL` to `_schedule_prefill_only()`, `DECODE` to
`_schedule_decode_only()`, **`DECODE_ATTN` to `_schedule_decode_attn_only()`**
— none overridden by `SGLangStyleReplicaScheduler` — and only the `else`
branch (`MONOLITHIC`, or anything else the enum admits) reaches the
overridden `_schedule_two_phase()`. So for `DECODE_ATTN`, the class's own
distinguishing logic **never runs**, regardless of contention — a
structural fact, confirmed both by reading the dispatch table and by a real
run (§2). `DECODE_FFN` also falls into the `else` branch, so its override
*would* be reached there — which is exactly where §1.3 stops it.

### 1.3 A separate, genuinely load-bearing guard — found, and correctly left alone

Attempting `DECODE_FFN` with `--...decode_ffn_replica_scheduler_config_type
vllm_v1` (even with this task's own patch installed) raises, from
`base_cluster_scheduler.py`, a check this task never touched:

```
ValueError: DECODE_FFN cluster requires 'orca' scheduler, got 'vllm_v1'.
Reason: DECODE_FFN uses EP-based workload grouping which is only
implemented in OrcaReplicaScheduler.
```

This is a **different guard, at a different layer** (the cluster
scheduler's own construction, `ClusterType.DECODE_FFN`-keyed, unconditional
— not `SGLangStyleReplicaScheduler`'s own constructor), stating a concrete,
checkable reason (`OrcaReplicaScheduler` is the only replica scheduler that
implements EP-based workload grouping) rather than a bare refusal. Per this
task's own known trap ("a guard may be correct — the first question is
whether it is protecting something, not how to remove it"): this one reads
as protecting something real, was not built by this task, and was **not
relaxed** — exactly the discipline §2's own instruction asks for, applied a
second time to a guard the spec itself never anticipated.

**Net effect**: in the one architecture this project's studies use
(pd-af-disaggregation), `DECODE_ATTN` is reachable but structurally inert
(§1.2), and `DECODE_FFN` — the only place the override would actually run —
is blocked by an unrelated, valid requirement this task correctly declined
to touch.

## 2. The patch: guarded, minimal, reversible

`src/integration/replica_scheduler/sglang_guard.py` — one function
replaced (`SGLangStyleReplicaScheduler.__init__`), guarded by a SHA-256
hash over its own current source, following `..cc_backend.collective`'s
own established pattern (Task 20) exactly: `install_sglang_replica_scheduler_guard()`
raises `SGLangGuardSourceMismatch` rather than patching over an
implementation this module hasn't reviewed. The patched `__init__` calls
`VLLMv1EngineReplicaScheduler.__init__` (identical to what `super().__init__()`
already did) and then checks against `(MONOLITHIC, PREFILL, DECODE_ATTN,
DECODE_FFN)` instead of `(MONOLITHIC, PREFILL)` — "only the cluster types
this project uses" per this task's own §2, deliberately excluding plain
`DECODE` (pd-disaggregation without an attention/FFN split — not this
project's architecture) and `TRANS`.

Wired into `install()` as `sglang_replica_scheduler: bool = False` —
defaults to off, exactly matching `collective`'s own convention; every
pre-Task-47 call site is unaffected. `tools/planner.py`'s own `_run_scenario`
now always passes `sglang_replica_scheduler=True` (harmless unless a caller
also asks for `decode_ffn_scheduler="sglang"`, which nothing before this
task ever did and `decode_ffn_scheduler` defaults to `"orca"`, unchanged).

**Note on this project's own `AGENTS.md`**: `src/integration/` is marked
"human-only — agents may write tests here but not implementations." This
task's own spec explicitly, narrowly instructs building exactly this
(function, mechanism, scope all specified) — read as the explicit
authorization that zoning rule anticipates needing, consistent with this
project's own established precedent (`collective.py`, Task 20;
`topology_aware.py`, Task 15, both agent-built under the same constraint).
Flagged here rather than silently proceeded past, since it is a real
tension between this task's own instruction and a standing project rule.

## 3. The three-configuration comparison, in the disaggregated architecture

`domain8` fabric, `attn_tp=1`, `ffn_ep=1`, streaming regime (Task 45's own
representative one, N=3 seeds — `seed_stats.seed_argv_fix`), `DECODE_ATTN`'s
own scheduler varied (the only axis §1 leaves reachable); `DECODE_FFN`
fixed at `orca` throughout (§1.3):

| config | `vllm_v1` mean tpot / throughput | `sglang`-style mean tpot / throughput | Task 46's own monolithic figure at the same config |
|---|---|---|---|
| generous (`max_tokens_in_batch=4096`) | 3.5258ms / 19.708 rps | 3.5258ms / 19.708 rps — **bit-identical** | 16.498ms / 120.79 rps (identical) |
| tight batch (`max_tokens_in_batch=256`) | 3.5258ms / 19.708 rps | 3.5258ms / 19.708 rps — **bit-identical** | 16.841 vs 16.498ms, 94.4 vs 120.8 rps (modest, real difference) |
| tight `num_blocks=40`, chunked, busy (64 req @ qps=60) | 14.5394ms / 24.300 rps | 14.5394ms / 24.300 rps — **bit-identical** | 1.94 vs 60.13 rps (31x) |

**Every configuration is bit-identical, at every one of 3 seeds each,
including the exact config Task 46 found a 31-fold throughput gap in
monolithic.** This is not "identical because no contention arose" (Task
46's own mechanism at its own generous config) — contention plainly did
arise here too (the tight/busy config's own throughput and latency both
move substantially from the generous one, for *both* labels identically)
— it is identical because, per §1.2, the code that would ever distinguish
the two policies is dispatched around entirely for `DECODE_ATTN`. **"Identical
is a result," per this task's own known trap — reported here for a
different, more specific reason than Task 46's own identical result, and
that distinction is itself the finding.**

## 4. Whether any conclusion reverses

Re-ran Task 44/45's own expert-degree comparison (`domain8`, `attn_tp=1`,
`ep ∈ {1,2,4}`, streaming, N=2 seeds — sufficient for an exact-equality
question, not an effect-size one) with `DECODE_ATTN`'s scheduler set to
each of `vllm_v1`/`sglang`, `DECODE_FFN` fixed at `orca` (required, §1.3):

| `ep` | `vllm_v1` mean tpot | `sglang` mean tpot | `ep_shape` |
|---|---|---|---|
| 1 | 3.4639ms | 3.4639ms | `(1,)` |
| 2 | 3.6502ms | 3.6502ms | `(2,)` |
| 4 | 3.7686ms | 3.7686ms | `(4,)` |

**Bit-identical at every degree.** The ranking (`ep=1` fastest, matching
Task 45's own streaming-regime finding exactly) is unchanged under either
policy — for the same structural reason as §3, not because the comparison
happened to be insensitive to scheduling this time. **No conclusion
reverses**, and given §1's own finding, none *could* — `DECODE_ATTN`'s own
scheduler choice cannot move anything Frontier computes, and `DECODE_FFN`'s
own scheduler choice is not free to vary at all in this project's
architecture.

## 5. Whether the patch leaves existing results untouched

Yes, confirmed directly, not assumed: Task 33's own sixteen-row table and
Task 36's own two-fabric result both reproduce bit-identical with
`install(..., sglang_replica_scheduler=True)` now called unconditionally by
`tools/planner.py` and `decode_ffn_scheduler` left at its own default
(`"orca"`) — the required proof (§4 of the spec) that installing the patch,
by itself, changes nothing for any call site that still selects the
original policy.

## 6. What this implies for the planner

**Scheduling is not, today, a viable third input alongside arrival regime.**
Not because the question is uninteresting (§1.3's own discovery — a real
engine-level scheduling difference, EP-based workload grouping, genuinely
exists between `Orca` and everything else) but because the one axis this
project can actually vary (`DECODE_ATTN`'s own replica scheduler) is proven
inert, and the one axis that might matter (`DECODE_FFN`'s) is not
selectable at all without addressing the separate `orca`-requirement guard
— an upstream question this task correctly left untouched (§1.3, §2's own
"do not extend to the other gated policies," applied to a guard beyond the
three the spec named). If a future task establishes that guard is *also*
liftable (a real EP-workload-grouping implementation added to `VLLMv1EngineReplicaScheduler`/`SGLangStyleReplicaScheduler`,
or ported the other direction), *then* scheduling would need to join
`Regime` as an explicit `plan()` input, at a cost proportional to however
many policies are actually reachable and distinguishable at that point —
not before. Reporting "not viable given current constraints" plainly,
rather than adding a `Regime`-like input for an axis that cannot move
anything yet, is this task's own answer to §5's own ask.

## 7. Anywhere this specification is wrong

**The central premise needs a correction, established directly rather than
assumed**: the spec's own §1 frames the obstacle as a single guard,
implying that lifting it opens the disaggregated architecture for this
comparison generally. It does — for `DECODE_ATTN` — but that axis turns out
to be structurally inert there (§1.2), a fact the spec's own framing
("the class inherits its allocation and preemption logic from the policy
this project already uses throughout, which is proven correct for those
cluster types") states the *safety* argument correctly but doesn't
anticipate the *reachability* one (that the override is dispatched around
entirely, not merely safe-but-untested). And the cluster type where the
override genuinely would run (`DECODE_FFN`) turns out to be blocked by a
wholly separate, valid guard (§1.3) the spec never mentions — not a
citation error, since the spec makes no claim about it, but a real gap in
what "the obstacle is a guard rather than a capability" turned out to
require checking. Both of these were established by reading the dispatcher
and by a real, direct construction attempt, not assumed.

**Otherwise, nothing else in this specification was wrong.** §2's own
"read what the class does with cluster type beyond the guard" instruction
correctly anticipated exactly the kind of check that mattered here (it did
branch elsewhere — just not incorrectly, and not in `SGLangStyleReplicaScheduler`'s
own code); §4's own bit-identical reproduction requirement is exactly the
right acceptance test for a patch of this kind; and §6's own "identical is
a result" / "do not extend to the other gated policies" known traps both
held, precisely.

## What shipped

- `src/integration/replica_scheduler/__init__.py`, `sglang_guard.py` — the
  guarded, one-function, source-hash-checked patch (§2).
- `src/integration/install/__init__.py` — `install()`'s new
  `sglang_replica_scheduler: bool = False` parameter.
- `tools/planner.py` — always installs the patch (harmless unless asked
  for); `decode_ffn_scheduler` threaded through `_argv`/`_run_scenario`/
  `evaluate()`/the CLI, defaulting to `"orca"` (unchanged behavior).
- `tests/test_sglang_replica_scheduler_guard.py` — 9 tests: the pre-patch
  guard's own behavior (both admitted and rejected cluster types), the
  patch admitting exactly the four cluster types this project uses and
  no others, idempotent installation, and the required source-hash-mismatch
  test.
- `docs/tasks/47-scheduler-regime-report.md`, this report.

One commit on `task-47-scheduler-regime`, stacked on `task-46-host-audit`.
Task 33's sixteen-row table and Task 36's two-fabric result both reproduce
bit-identical with the patch installed and the original policy selected.
249 tests pass (240 + 9 new); `check_import_direction.py` exits 0.
