# Stage 2 — Gate C.1: dense-model routing semantics investigation +
# guarded Frontier compatibility fix design (PROPOSED, NOT IMPLEMENTED)

**This is a design/test/patch proposal, stopped before implementation, per
`AGENTS.md`'s own governance rule.** The `ZeroDivisionError` blocking real
TP=1/2/4 Qwen3-0.6B evaluation (docs/tasks/67) is a real, narrow, correctly-
diagnosable Frontier bug: `SklearnDisaggregationExecutionTimePredictor.__init__`
computes MoE expert-routing allocations unconditionally, with no `is_moe`
check, even though every real *consumer* of that state already gates
itself on `is_moe` and a complete, working, `is_moe`-aware dense path
already exists elsewhere in the very same file and in sibling files. The
correct fix is a one-condition gate, matching an idiom Frontier's own
authors already use three other places in this exact file. It requires
changing code inside the external, pinned Frontier checkout (or, properly,
a new guarded runtime-patch module under `src/integration/`, following
this project's own established pattern) -- both squarely in `AGENTS.md`'s
human-only zone for implementations. **Investigated fully, proposed in
full, not implemented. Tests added (3, all passing against current,
unpatched code). Not one line of Frontier or `src/integration/` code was
changed.**

---

## 0. Governance (read first, respected throughout)

`AGENTS.md`: `src/integration/` is human-only for implementations --
"Agents may write tests here but not implementations." The external
Frontier checkout (`/work/simulation/Frontier`) is not `AGENTS.md`'s own
`upstream/` directory, but every real fix this entire Gate C.1 initiative
has produced (QK-norm, RoPE, RMSNorm, `vllm_config_context`,
`mla_phase_filter`, `sglang_guard`, `attention_block_table_fix`,
`collective`) was implemented as a guarded runtime monkeypatch inside
`src/integration/`, never as a direct edit to the Frontier checkout --
that discipline is treated here as binding, not merely a stylistic
default. Both locations are therefore off-limits for direct
implementation in this task. This report investigates fully, designs the
patch completely, and stops before applying it.

---

## 1. Reproduced failure (fresh, this session, not relying on docs/tasks/67 alone)

```
cwd=/work/simulation/Frontier (real evaluation's own cwd)
model.is_moe == False, model.total_experts == 0    (Qwen3-0.6B, confirmed live)
```

Full (non-truncated) traceback, reproduced directly in-process:

```
File ".../execution_time_predictor_registry.py", line 78, in get
    return predictor_class(...)
File ".../random_forrest_execution_time_predictor.py", line 93, in __new__
    return _RandomForrestExecutionTimePredictor(**kwargs)
File ".../random_forrest_execution_time_predictor.py", line 62, in __init__
    super().__init__(*args, **kwargs)
File ".../sklearn_disaggregation_execution_time_predictor.py", line 169, in __init__
    self._simulate_and_store_routing(target_cluster_type)
File ".../sklearn_disaggregation_execution_time_predictor.py", line 433, in _simulate_and_store_routing
    self._generate_expert_allocations(...)
File ".../sklearn_disaggregation_execution_time_predictor.py", line 473, in _generate_expert_allocations
    allocation_ratios = [1.0 / total_expert_num] * total_expert_num
ZeroDivisionError: float division by zero
  (wrapped by execution_time_predictor_registry.py:91 into:)
ValueError: Failed to create predictor of type 'random_forrest': float division by zero
```

Same call chain and same exception confirmed identically for `PREFILL`'s
own predictor construction, independent of which `attn_tp` (1, 2, or 4)
the DECODE_ATTN candidate uses -- `PREFILL` is always `tp=1` and always
constructed first, so all three TPs fail identically, as previously
observed.

---

## 2. Routing call graph (full trace, both functions read in full)

```
ModelConfig (is_moe, total_expert_num, moe_expert_parallel_size)
        │
SklearnDisaggregationExecutionTimePredictor.__init__   (lines 69-194)
   unconditional, for EVERY predictor whose cluster_type ∈ {PREFILL, DECODE_FFN, DECODE}
   -- no is_moe / total_expert_num check anywhere in this block
        ↓
_simulate_and_store_routing(cluster_type)              (line ~249)
   asserts total_expert_num % expert_parallel_size == 0   (0 % 1 == 0 -- passes silently for dense)
        ↓
_generate_expert_allocations(total_expert_num, ep, replica_id=0, layer_id)   (line 450)
   BALANCED: [1.0/total_expert_num]*total_expert_num  <-- ZeroDivisionError when total_expert_num=0
   RANDOM:   np.random.uniform(...)/sum(...)           <-- would also be nonsensical (0-length array) at N=0
        ↓
routing_details = {replica_id: {layer_id: {expert_id: ratio, ...}, ...}, ...}
        ↓
stored as self._prefill_routing_details / _decode_ffn_routing_details / _decode_routing_details
   (deleted via `del`, not left as `None`, for whichever of the three a given predictor doesn't need --
    DECODE_ATTN already deletes all three, unconditionally, "not needed")
        ↓
ONLY real consumer: _calculate_expert_token_allocation (line 559)
   -> per-expert token counts -> predict_moe_layer_time / _get_grouped_gemm_time
   (compute-time prediction for the MoE FFN operator, and all-to-all comm sizing)
```

Every call site of `_calculate_expert_token_allocation` (PREFILL ~line
2200, DECODE_FFN analogue, unified-DECODE ~line 1781) is wrapped in
`is_moe_model = model_config.is_moe [and model_config.is_moe_layer(layer_id)]`
-- three occurrences of the verbatim comment *"Use model_config.is_moe for
MoE detection - NOT parallelism settings"*. Each has a sibling
`else: # Dense model: use MLP operations` branch calling
`predict_mlp_layer_time(...)` instead -- a complete, independent, already-
working dense-FFN prediction path that never touches routing state at
all. Confirmed live in this project's own logs: `mlp_up_proj`,
`mlp_down_proj`, `mlp_act` were trained successfully for Qwen3-0.6B's
`decode_ffn` *before* the crash -- the dense compute-time path already
works; only the eager, unconditional routing pre-computation in `__init__`
does not. `_get_replica_expert_workload_ratio` (line 506) has zero
callers in the file -- dead code, not a real consumer.

---

## 3. What `allocation_ratios` / `_generate_expert_allocations` represents (proven via a real MoE model, not assumed)

`Phi-tiny-MoE-instruct` (`is_moe=True`, `total_experts=16`,
`router_topk=2`) -- this project's own established MoE regression case
(Task 33/36) -- real, current-HEAD evaluation, `topology=domain8`,
`hardware=h800`, `candidate=Candidate(attn_tp=1, attn_shape=(1,))`,
Gate C's own workload:

```
result = {'error': None, 'mean_tpot_ms': 12.317824968905404,
          'throughput_rps': 50.86139603307486, 'slo_attainment': 0.75,
          'n_completed': 32}
```

No exception -- this MoE path runs cleanly, unmodified, today.
`_generate_expert_allocations(total_expert_num=16, expert_parallel_size=1,
replica_id=0, layer_id=0)`, `WorkloadDistributionType.BALANCED` (this
config's resolved default), returns:

```
[0.0625, 0.0625, ..., 0.0625]   # exactly 1/16, all 16 entries, deterministic given the seed
```

Traced through its one real consumer, `_calculate_expert_token_allocation`:
this ratio is multiplied by `total_batch_tokens * router_topk` (the real
number of routed tokens in a batch) to get **an expected per-expert token
count** -- i.e. `allocation_ratios` is a simulated *load-share* per
expert, feeding (a) `predict_moe_layer_time`'s grouped-GEMM compute-time
prediction (how much work each expert's shard does) and (b) all-to-all
communication *sizing* (how many tokens must be shuffled to which
expert). It is not a probability distribution consumed probabilistically
elsewhere and not a placement/device-allocation value -- it is a
deterministic, seeded simulated load-balance figure, real input to two
downstream real predictions, not a decorative placeholder.

Why `[1/N]*N` specifically: `WorkloadDistributionType.BALANCED` is the
simplest of four supported distributions (`BALANCED`, `RANDOM`, `SKEWED`,
`ZIPF`) simulating a perfectly load-balanced MoE router as the default
assumption absent real routing telemetry -- a legitimate MoE modeling
choice, not a bug in itself. The bug is calling this function at all for
a model with zero experts.

---

## 4. Dense-model semantics: evaluating the task's five candidate options

| option | verdict | reasoning |
|---|---|---|
| **A. Skip expert-routing simulation entirely** | **Correct, chosen** | Matches Frontier's own already-existing behavior for dense models in every other place this project found dense/MoE branching (see §5) -- not invented here. |
| B. Represent dense FFN as one synthetic expert `[1.0]` | **Rejected** | Fabricates MoE routing state (`routing_details[replica_id][layer_id][0] = 1.0`) for a model that has none -- exactly the "plausible wrong number" pattern this project's own prior findings (4 defects, 3 silent) warn hardest against. No downstream code needs it (§2/§6) -- inventing it serves no consumer, only "avoids the exception," which is explicitly forbidden (task §8). |
| C. Represent an empty allocation `[]` | **Semantically correct as an arithmetic detail, but not sufficient alone** | Because `_generate_expert_allocations` only receives `total_expert_num` as a bare int -- not `is_moe` -- a guard placed *only* inside this function cannot distinguish "legitimately dense" (`is_moe=False, total_experts=0`, should skip silently) from "inconsistent MoE metadata" (`is_moe=True, total_experts=0`, task §7/§11-C requires a loud failure). The gate has to live one level up, where both `is_moe` and `total_experts` are simultaneously visible. |
| D. Use a separate existing dense-FFN execution path | **Confirmed to already exist, and is what the fix reuses** | `RandomForrestExecutionTimePredictor._get_base_class` already picks a completely different, non-MoE predictor class (`SklearnExecutionTimePredictor`) for dense models -- but only in non-disaggregated mode (§5/§6). The fix does not invent a new dense path; it extends the *scope* of the is_moe-gated behavior Frontier's authors already wrote, into disaggregated mode, at the one place (`__init__`'s eager routing pre-computation) it was never propagated to. |
| E. Some other representation | Not needed | A/D together are sufficient and fully evidenced. |

---

## 5. Existing dense-vs-MoE branching already in Frontier (search results)

Frontier's own code already distinguishes dense from MoE, correctly, in
at least five places -- the crash site is the outlier, not the norm:

- **`random_forrest_execution_time_predictor.py:22-41`,
  `_get_base_class`**: `elif replica_config.model_config is not None and
  replica_config.model_config.is_moe: ... SklearnMoEExecutionTimePredictor
  ... else: ... SklearnExecutionTimePredictor` -- **but only when `not
  global_vars.is_disaggregated_mode()`**. In disaggregated mode this
  `is_moe` check is skipped entirely; `SklearnDisaggregationExecutionTimePredictor`
  (which itself extends `SklearnMoEExecutionTimePredictor`) is selected
  unconditionally. **This is the actual shape of the bug**: not a missing
  guard invented from nothing, but an inconsistency between the
  disaggregated and non-disaggregated branches of the very same selector
  function.
- **`sklearn_execution_time_predictor.py`, `_get_compute_model_names`
  (~3505-3527)**: MoE-only ops (`moe_gating_linear`,
  `moe_gating_routing_topk`, `moe_shuffling`, `moe_grouped_gemm`) gated
  behind `if is_moe_model`; dense models get plain
  `mlp_up_proj`/`mlp_down_proj`/`mlp_act` instead. Raises loudly if
  `model_config.is_moe` is missing -- never guesses.
- **`_requires_dense_mlp_compute_models` (~2587-2596)**: `if not is_moe:
  return True`.
- **`sklearn_disaggregation_execution_time_predictor.py`,
  `_get_dummy_execution_time_for_cluster` (~679-696) and the
  communication-metadata construction that follows (~690-895)**:
  `is_moe_model = model_config is not None and model_config.is_moe`
  (**the exact idiom this proposed fix reuses**, same file); DECODE_ATTN
  forces `is_moe=False` with the comment *"DECODE_ATTN cluster doesn't
  handle MoE"*; non-MoE branches already set
  `expert_parallel_communication_time=0.0` cleanly (lines 814, 881, 891,
  1419) -- **communication-cost prediction in this exact class already
  handles the dense case correctly**; only the eager `__init__`-time
  routing pre-computation does not.
- **Dummy-mode branch (line ~322)**: `max(1, cluster_replica_config.total_expert_num)`
  -- Frontier's own dummy-mode path already special-cases zero experts
  defensively. The real (non-dummy) branch has no equivalent guard --
  an inconsistency between dummy-mode and real-mode handling of the same
  input, not new evidence for any particular fix shape, but confirmation
  that Frontier's authors were aware zero-expert models are a real input
  they needed to handle somewhere.
- **A separate, unrelated MTP-fusion path (~4990-4997)** uses `total_expert_num or 1`
  as a scalar-arithmetic safety idiom. Noted for completeness, but not
  reused here: it guards a single scalar division in an unrelated
  function, not the fabrication of a whole per-expert routing structure --
  applying it to `_generate_expert_allocations` would be exactly Option
  B/§8's forbidden band-aid in different clothing, not this same idiom.

**Frontier's own shipped model corpus already contains genuinely dense
models**, predating this project's Qwen3-0.6B addition:
`llama2_7b_dense_example`, `llama3.1-8b`, `llama3.1-405b`,
`llama3.3-70b`, `Llama-3.1-405B-Instruct-FP8`, `Llama-3.2-1B-Instruct`,
`qwen2_dense_test` -- files literally named `*_dense_example`/
`*_dense_test` -- versus MoE files (`Phi-tiny-MoE-instruct`,
`mixtral_8x7b_moe`, `deepseek-v3`, `qwen3-a3b-30b-moe`) all carrying
explicit nonzero expert counts. **Dense is a state Frontier's own config
schema already intends to support** -- not a state Qwen3-0.6B introduces
for the first time; this project's Qwen3-0.6B evaluation is simply the
first time a dense model has been pushed through *disaggregated-mode*
real evaluation specifically (§6 explains why non-disaggregated dense
evaluation was never affected).

---

## 6. Is `DECODE_FFN` inherently MoE, or a generic FFN stage? (proven from code)

**Generic.** `_is_kernel_only_measurement_enabled_for_cluster`
(`sklearn_execution_time_predictor.py:896-908`): `if self._cluster_type in
(ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN): return
use_cuda_graph` -- no MoE condition whatsoever, confirming this project's
own prior Task 63 claim verbatim.
`_get_default_measurement_type_for_cluster` (~910-918): for
`sys_arch=="pd-af-disaggregation"`, `DECODE`/`DECODE_ATTN`/`DECODE_FFN`
all map to `KERNEL_ONLY` purely by architecture, independent of model
type. No code or docstring anywhere asserts `DECODE_FFN` requires MoE;
every place that needs MoE-specific behavior branches explicitly on
`is_moe`/`is_moe_model` as a condition layered **on top of** the generic
FFN cluster type, never as the cluster type's own defining property.

**Conclusion: the planner architecture (`pd-af-disaggregation`) is not
fundamentally incompatible with dense models.** The bug is a real,
narrow implementation gap (one predictor-selection function's `is_moe`
check not extended to its disaggregated-mode branch), not an
architectural mismatch requiring the planner to stop using
`pd-af-disaggregation` for dense models.

---

## 7. Valid model-state table

| `is_moe` | `total_experts` | meaning | valid under proposed fix? |
|---|---:|---|---|
| `False` | `0` | dense model (Qwen3-0.6B) | **Valid** -- routing skipped cleanly, matching `_get_base_class`'s own existing non-disaggregated-mode precedent for treating `is_moe` as authoritative. |
| `False` | `>0` | dense model with a stray/unused expert-count field | **Valid, treated as dense** -- confirmed from Frontier's own code (`_get_base_class` never inspects `total_experts` when deciding MoE-ness, only `is_moe`), not invented here. Not rejected as "invalid": Frontier does not reject it today either (in non-disaggregated mode this value is simply never read for a dense model). No shipped model in Frontier's corpus was found in this state. |
| `True` | `0` | **inconsistent MoE metadata** | **Correctly still fails loudly** -- the proposed gate only skips routing when `is_moe` is `False`; an `is_moe=True` model keeps running the exact existing (crashing) code path. Verified directly (test C, §9). |
| `True` | `>0` | MoE (Phi-tiny-MoE-instruct, Mixtral, DeepSeek-V3, ...) | **Valid, existing behavior fully unchanged** -- the proposed gate is a no-op on this path; the original code executes byte-for-byte identically. Verified directly (test B, §9). |

---

## 8. Network/communication consequences

```
MoE:     ATTN -> router -> experts / expert-parallel all-to-all -> ...
                            (routing_details drives grouped-GEMM compute
                             time AND all-to-all sizing, via
                             _calculate_expert_token_allocation)

Dense:   ATTN -> dense FFN (mlp_up_proj/mlp_down_proj/mlp_act) -> ...
                            (no routing_details reference anywhere on
                             this path -- confirmed, §2)
```

Skipping expert-routing computation for a dense model removes only fake
MoE routing/all-to-all traffic that would never have been priced anyway
(every real consumer already gates on `is_moe`, §2) -- it does not touch
the mechanism this project actually relies on to model real ATTN<->FFN
communication in PD-disaggregation. That mechanism is a **separate
Frontier subsystem** entirely: `frontier/m2n_transfer/` (its own
registry, `BaseM2NTransferPredictor`/`AnalyticalM2NTransferPredictor`
classes, its own config, its own simulator events
`m2n_transfer_start_event.py`/`m2n_transfer_end_event.py`), selected via
`tools/planner.py`'s own `--m2n_transfer_config_type empirical` /
`--cc_backend_config_type analytical` (`_argv`, lines 149-150) and this
project's own `install(..., collective=True)` (which patches
`CCBackendFactory.create` so `EngineCCBackend` prices every real
collective from the fabric). None of that lives inside
`sklearn_disaggregation_execution_time_predictor.py`, and none of it
reads `_prefill_routing_details`/`_decode_ffn_routing_details`/
`_decode_routing_details` -- confirmed by file-level separation, not
inference. **The legitimate ATTN<->FFN hidden-state transfer this
project actually cares about pricing is unaffected by this fix in either
direction.**

---

## 9. Proposed patch (design only -- NOT applied)

**File**: `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`
**Function**: `SklearnDisaggregationExecutionTimePredictor.__init__`
**Current source hash** (computed live, this checkout, for a future
guarded-patch module to check against):
```
bc5e32d80eecdfcb06af26968b577fb7d4015adf32e0a509fb7ee1b98065c099
```

Minimal diff (one new local variable, wrapping the existing loop in one
`if`, extending the existing cleanup condition by one clause -- no other
line touched):

```diff
         # Pre-calculate routing details only for relevant clusters to avoid unnecessary computation
         # Each predictor only calculates routing for clusters it will actually serve

         self._prefill_routing_details = None
         self._decode_ffn_routing_details = None
         self._decode_routing_details = None  # For unified DECODE cluster in PD-disaggregation mode

         # Define cluster types that require MoE routing details
         # DECODE is included for PD-disaggregation mode where DECODE handles both attention + MoE
         moe_cluster_types = {ClusterType.PREFILL, ClusterType.DECODE_FFN, ClusterType.DECODE}
         current_cluster_types = {cluster_type} if cluster_type else moe_cluster_types

-        # Calculate routing details for each relevant cluster type
-        for target_cluster_type in current_cluster_types.intersection(
-            moe_cluster_types
-        ):
-            routing_details: Dict[int, Dict[int, Dict[int, float]]] = (
-                self._simulate_and_store_routing(target_cluster_type)
-            )
-
-            if target_cluster_type == ClusterType.PREFILL:
-                self._prefill_routing_details = routing_details
-                del self._decode_ffn_routing_details
-                del self._decode_routing_details
-            elif target_cluster_type == ClusterType.DECODE_FFN:
-                self._decode_ffn_routing_details = routing_details
-                del self._prefill_routing_details
-                del self._decode_routing_details
-            elif target_cluster_type == ClusterType.DECODE:
-                self._decode_routing_details = routing_details
-                del self._prefill_routing_details
-                del self._decode_ffn_routing_details
-
-        # Initialize empty routing details for clusters that don't need MoE routing
-        if cluster_type == ClusterType.DECODE_ATTN:
-            logger.debug(
-                "DECODE_ATTN predictor skipping MoE routing calculation (not needed)"
-            )
+        # Stage 2 Gate C.1 (proposed): a dense model (is_moe=False) has no
+        # experts to route. Every real consumer of routing_details
+        # (_calculate_expert_token_allocation, expert-parallel
+        # communication) already gates itself on model_config.is_moe, not
+        # on whether this state happens to be present -- so computing it
+        # for a dense model is dead work whose only effect is to crash on
+        # total_expert_num=0. `is_moe_model` reuses the exact idiom this
+        # same file already applies elsewhere (see
+        # _get_dummy_execution_time_for_cluster / the communication-
+        # metadata construction just below in this file).
+        is_moe_model = self._model_config is not None and self._model_config.is_moe
+
+        # Calculate routing details for each relevant cluster type
+        if is_moe_model:
+            for target_cluster_type in current_cluster_types.intersection(
+                moe_cluster_types
+            ):
+                routing_details: Dict[int, Dict[int, Dict[int, float]]] = (
+                    self._simulate_and_store_routing(target_cluster_type)
+                )
+
+                if target_cluster_type == ClusterType.PREFILL:
+                    self._prefill_routing_details = routing_details
+                    del self._decode_ffn_routing_details
+                    del self._decode_routing_details
+                elif target_cluster_type == ClusterType.DECODE_FFN:
+                    self._decode_ffn_routing_details = routing_details
+                    del self._prefill_routing_details
+                    del self._decode_routing_details
+                elif target_cluster_type == ClusterType.DECODE:
+                    self._decode_routing_details = routing_details
+                    del self._prefill_routing_details
+                    del self._decode_ffn_routing_details
+
+        # Initialize empty routing details for clusters/models that don't need MoE routing
+        if not is_moe_model or cluster_type == ClusterType.DECODE_ATTN:
+            logger.debug(
+                "%s: skipping MoE routing calculation (not needed)",
+                "dense model (is_moe=False)" if not is_moe_model
+                else "DECODE_ATTN predictor",
+            )
             del self._prefill_routing_details
             del self._decode_ffn_routing_details
             del self._decode_routing_details
-            # self._prefill_routing_details = {}
-            # self._decode_ffn_routing_details = {}
```

Note the diff is written against Frontier's own source for clarity of
*what* changes semantically. The actual installation mechanism, following
this project's own established pattern (`src/integration/replica_scheduler/sglang_guard.py`,
`src/integration/execution_time_predictor/mla_phase_filter.py` -- both
already patch a whole pinned-Frontier `__init__`, guarded by a source
hash, rather than edit the checkout), would be a **new module**:

```
src/integration/execution_time_predictor/dense_model_moe_routing_guard.py
```

shaped exactly like `mla_phase_filter.py`: a module-level
`_EXPECTED_SOURCE_HASH = "bc5e32d8...c099"` (computed above), a
`_patched_init` function containing the full patched `__init__` body
above, an `install_dense_model_moe_routing_guard()` function that hashes
`SklearnDisaggregationExecutionTimePredictor.__init__`'s current source,
raises a new `DenseModelMoeRoutingGuardMismatch` if it no longer matches,
and otherwise assigns
`SklearnDisaggregationExecutionTimePredictor.__init__ = _patched_init`
(idempotent, matching every existing patch's own contract) -- and a new
optional `dense_model_moe_routing_guard: bool = False` parameter on
`install()` (`src/integration/install/__init__.py`), wired into
`tools/planner.py::_run_scenario`'s existing `install(...)` call, the
same way `qk_norm_allowlist_fix=True` was wired in docs/tasks/67.
**None of this was created or installed** -- `src/integration/` is
human-only for implementations.

---

## 10. Governance classification

| location | classification |
|---|---|
| `SklearnDisaggregationExecutionTimePredictor.__init__` (the actual bug) | **C -- external Frontier checkout.** Never modified directly, by this project's own established discipline. |
| A new guarded-patch module implementing the fix | **B -- `src/integration/`, human-only for implementations** per `AGENTS.md`. |
| The one-line `install()` wiring in `tools/planner.py` (if the patch above is approved and installed) | A -- agent-safe, same as docs/tasks/67's own qk_norm wiring -- but this step cannot be taken before B exists. |

**STOP. Not implemented, per instruction and per `AGENTS.md`.**

---

## 11. Tests added (agent-safe: `tests/`, no implementation touched)

`tests/test_gate_c1_dense_moe_routing_state.py`, 3 tests, all passing
against **current, unpatched** code:

- **Test A** (`is_moe=False, total_experts=0`, real Qwen3-0.6B):
  currently asserts the **failure** occurs (`"float division by zero"` in
  the error), with an explicit docstring/inline comment stating this
  assertion must be *inverted* the moment the proposed patch is
  installed -- written so an un-inverted stale test fails loudly rather
  than silently passing for the wrong reason.
- **Test B** (`is_moe=True, total_experts=16`, real Phi-tiny-MoE-instruct):
  asserts the exact captured baseline
  (`mean_tpot_ms=12.317824968905404, throughput_rps=50.86139603307486,
  slo_attainment=0.75, n_completed=32, error=None`) -- the non-regression
  anchor for §12.
- **Test C** (`is_moe=True, total_experts=0`, Phi-tiny-MoE-instruct's own
  real config with an explicit, inconsistent `total_experts=0` override):
  asserts the failure still occurs -- proving Frontier's own
  `ReplicaConfig.__post_init__` does not silently repair this
  inconsistency (its auto-derive-from-model-config logic only fires when
  `total_expert_num == 1`, the untouched default, confirmed by reading
  it directly), so this really does reach Frontier as a genuine
  `is_moe=True, total_expert_num=0` state, and really does fail loudly,
  both today and (by design) after the proposed fix.

```
$ python3 -m pytest tests/test_gate_c1_dense_moe_routing_state.py -v
test_A_dense_model_currently_hits_the_known_zerodivisionerror_bug PASSED
test_B_moe_baseline_phi_tiny_moe_instruct_unaffected PASSED
test_C_inconsistent_moe_metadata_raises_loudly PASSED
3 passed in 26.53s
```

Full suite: **415 passed, 16 skipped** (was 412 passed before this task's
3 new tests; identical skip set). Import-direction check: **PASS**.

No hard-coded Qwen3/MI355X/TP/expert-count values drive the *tests'
assertions about the proposed patch's behavior* -- the tests assert
outcomes for two independently-real models (Qwen3-0.6B, Phi-tiny-MoE-instruct)
to prove the fix generalizes across models by construction, not by
one model's coincidental numbers.

---

## 12. MoE non-regression evidence

Baseline captured **before** any change exists, from current HEAD,
unmodified: `Phi-tiny-MoE-instruct`, TP=1, `domain8`/`h800`, Gate C's own
workload -- `mean_tpot_ms=12.317824968905404`,
`throughput_rps=50.86139603307486`, `slo_attainment=0.75`,
`n_completed=32`, `error=None`. Routing-allocation sample for
`total_expert_num=16, layer_id=0`: `[0.0625]*16` (exactly `1/16`,
deterministic). Locked into test B (§11) as an exact-value assertion
(`pytest.approx(..., abs=1e-9)`), not a loose "still passes" check.

**Why the proposed patch provably preserves this exactly, not merely
"probably"**: the patch's only behavioral change is
`if is_moe_model:` wrapping the pre-existing loop verbatim -- for
`is_moe_model=True` (Phi-tiny-MoE-instruct's own real state), the branch
is taken and the ORIGINAL code executes character-for-character
unchanged, with no new arguments, no altered control flow inside the
loop, and no changed cleanup logic for the `PREFILL`/`DECODE_FFN`/
`DECODE` branches. The `if not is_moe_model or cluster_type ==
ClusterType.DECODE_ATTN:` cleanup condition is a strict superset of the
original `if cluster_type == ClusterType.DECODE_ATTN:` -- for any
`is_moe_model=True` run it evaluates identically to before (`not
is_moe_model` is `False`, leaving only the original `DECODE_ATTN` check).
This is why the proposed diff was designed as an additive wrapper rather
than a rewrite: it makes "MoE behavior unchanged" a property of the diff
itself, checkable by inspection, not something that merely happens to
hold under the tests run so far.

---

## 13. Hard-code audit of the proposed patch

No Qwen3-specific, MI355X-specific, TP-specific, expert-count-specific,
measured-TPOT, or hardware-result constant appears anywhere in the
proposed diff. The only new literal is the boolean expression
`self._model_config is not None and self._model_config.is_moe` -- a
semantic branch on model metadata (`is_moe`), explicitly the allowed
category per the task's own §20 ("semantic branch on model metadata such
as is_moe" is allowed; "synthetic expert count inserted only to avoid
division by zero" is forbidden -- and the proposed patch inserts no
synthetic expert count at all, it skips the computation entirely).

---

## 14-17. Not reached

Per §14-17 of the task and this project's own established practice
(docs/tasks/66, 67): since the fix is a design proposal, not an
implementation, TP=1/2/4 real Frontier evaluation was **not re-run**
against a patched Frontier, and the Stage 2 Gate C planner handoff was
**not generated**. Final-location TP-aware coverage remains at its
already-confirmed PASS state from docs/tasks/67 (unchanged -- no profile
data was touched by this task).

---

## Final answers

**A. Why does Frontier divide by zero for Qwen3-0.6B?**
`SklearnDisaggregationExecutionTimePredictor.__init__` calls
`_simulate_and_store_routing` → `_generate_expert_allocations`
unconditionally for `PREFILL`/`DECODE_FFN`/`DECODE` predictors, with no
`is_moe`/`total_expert_num` check at that call site. Qwen3-0.6B has
`total_expert_num=0` (it is dense), so `1.0 / 0` raises.

**B. Is this just an arithmetic bug or a deeper MoE-assumption bug?**
**A missing conditional, not an architectural incompatibility.**
Frontier's own predictor-selection function (`_get_base_class`) already
branches dense-vs-MoE via `is_moe` correctly -- but only in
non-disaggregated mode. In disaggregated mode that same check was never
propagated into `SklearnDisaggregationExecutionTimePredictor.__init__`'s
own eager routing pre-computation. Every real consumer of the routing
state this computes is *already* `is_moe`-gated. The fix is one
condition, reusing an idiom that already exists three other places in
the same file.

**C. What does `_generate_expert_allocations()` represent?**
A deterministic, seeded, simulated per-expert token-load-share (a
BALANCED/RANDOM/SKEWED/ZIPF distribution choice), consumed by
`_calculate_expert_token_allocation` to feed grouped-GEMM compute-time
prediction and expert-parallel all-to-all communication sizing for a
real MoE model. Not a placeholder, not a probability used elsewhere,
not a device-placement value.

**D. What is the correct Frontier semantic for `is_moe=False, total_experts=0`?**
Skip expert-routing simulation entirely (Option A) -- exactly what
Frontier's own `_get_base_class` already does for dense models in
non-disaggregated mode, extended to disaggregated mode's one gap.

**E. Should dense models have any expert-routing state at all?**
**No.** Confirmed by enumerating every real reader of
`_prefill_routing_details`/`_decode_ffn_routing_details`/
`_decode_routing_details`: none is reachable for `is_moe=False` (every
call site is pre-gated on `is_moe`). No fake empty/placeholder dict is
needed or proposed; the proposed patch leaves the three attributes
deleted (absent), matching the existing `DECODE_ATTN` precedent for "not
needed here" exactly.

**F. Does the fix preserve legitimate ATTN<->FFN network communication?**
**Yes.** That communication is priced by a fully separate Frontier
subsystem (`frontier/m2n_transfer/`, plus this project's own
`EngineCCBackend`/`collective=True` collective pricing) that never reads
`_simulate_and_store_routing`'s output. Confirmed by file-level
separation, not inference.

**G. Is existing MoE behavior provably unchanged?**
**Yes, by construction of the diff** (§12): the `is_moe_model=True`
branch executes the original code unmodified, character-for-character.
Locked in by an exact-value regression test (test B) against a freshly
captured, pre-change baseline.

**H. Does the fix contain any model/hardware/performance hard-coded values?**
**No** (§13) -- one semantic boolean branch on `is_moe`, nothing else.

**I. Does `AGENTS.md` allow the agent to implement the required fix?**
**No.** The fix touches either the external Frontier checkout directly
(never modified, by this project's own established discipline) or a new
`src/integration/` module (explicitly human-only for implementations per
`AGENTS.md`). Investigated and designed in full; not implemented.

**J. If implemented/approved, does Frontier evaluate Qwen3/MI355X at TP=1? TP=2? TP=4?**
**Not attempted** -- implementation was not authorized in this task. The
investigation gives high confidence the proposed one-condition gate
would clear this specific blocker (every consumer traced, every
consumer already `is_moe`-gated), but this is a prediction, not a
verified result, until the patch is actually installed and re-run.

**K. Is the Stage 2 Gate C planner handoff now ready?**
**No** -- unchanged from docs/tasks/67; no implementation, no re-run, no
handoff.

**L. Is the project ready for planner manifest → real hardware execution → decision validation?**
**No.** Two things must happen first, in order: (1) a human reviews and
installs the proposed guarded patch (§9/§10) or an equivalent one; (2)
TP=1/2/4 real Frontier evaluation is re-run and passes all of docs/tasks/67
§17's checks, cleanly, before any planner handoff is generated.
