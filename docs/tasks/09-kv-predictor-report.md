# Task 09 report — A real KV transfer predictor

Branch: `task-09-kv-predictor` (not merged to main).

`python3 -m pytest -q` (143 passed: 134 from before task 06 plus 9 added
across tasks 06/09) and `python3 tools/check_import_direction.py` pass
unchanged. Nothing under `upstream/` or `src/engine/` was touched.

---

## 1. The two TTFT numbers, and the finding underneath them

`tools/run_kv_integration.py` runs the same offline `pd-disaggregation`
workload (Llama-2-7b, 2048 prefill tokens, 4 requests, `h800`, dummy
execution-time mode) twice, differing only in placement, with
`EngineKVCacheTransferPredictor` selected via `--kv_cache_transfer_config_type
empirical`:

```
packed: mean ttft=626.500000 ms, mean kv_cache_transfer_time=2.684355 ms
split:  mean ttft=626.500000 ms, mean kv_cache_transfer_time=21.474837 ms

TTFT:                    packed=626.500000 ms  split=626.500000 ms  ratio=1.0000
KV cache transfer time:  packed=2.684355 ms    split=21.474837 ms   ratio=8.0000
(fabric bandwidth ratio scale_out:scale_up = 8.0:1)
```

**Frontier's `request.ttft` does not differ. That is the honest finding, and
by the spec's own instruction (S5.1) I am reporting it plainly rather than
adjusting anything to make it move.** It isn't a bug in the predictor --
`request.ttft` is *structurally* incapable of reflecting KV transfer time in
this Frontier version:

```python
# frontier/metrics/constants.py
TTFT = "ttft"  # Total time from arrival to prefill completion
```

```python
# frontier/entities/request.py
@property
def ttft(self) -> float:
    """Measured from request arrival to prefill completion."""
    return self._prefill_completed_at - self._arrived_at
```

`_prefill_completed_at` is stamped when the PREFILL cluster finishes its own
compute, *before* any KV transfer to DECODE happens. KV transfer time is
tracked as a separate, explicit quantity -- `request.kv_cache_transfer_time`,
which metrics_store.py breaks out as `ttft_kv_transfer`, a documented
component distinct from the headline `ttft`:

```python
# frontier/metrics/constants.py
TTFT_KV_TRANSFER = "ttft_kv_transfer"  # KV cache transfer time component (PD/PD+AF modes)
```

**`request.kv_cache_transfer_time` is the number the task is actually
looking for, and it does the thing S3.2 asked for: it differs, and the split
placement is slower.** 2.684355 ms packed vs 21.474837 ms split, ratio
exactly 8.0000 -- see S3 below for why that's not a coincidence. This is a
serving-relevant number Frontier now produces that changes because of where
the GPUs are; it just isn't the field literally named `ttft`.

## 2. How the engine's objects reach the predictor

**Chosen: a module-level context, set by `install()`.** Frontier constructs
`KVCacheTransferPredictorRegistry.get(predictor_type, config=...)` from a
config object built entirely from CLI flags -- there is no channel for a
`Fabric`, `Placement`, or `Deployment` Python object to travel alongside it.
`src/integration/kv_transfer/predictor.py` defines
`EngineKVContext(fabric, placement, deployment, groups)` and a module-level
`set_context()`/`_require_context()` pair; `EngineKVCacheTransferPredictor.
get_transfer_time` reads it at call time. `src/integration/install/__init__.py`
is now the one entry point: `install(fabric, placement, deployment, groups)`
calls `cc_backend.install()` (unchanged from task 06) and then
`kv_transfer.predictor.set_context(...)`.

**Rejected: an InfraGraph file path named by a CLI flag.** This would work,
but it means inventing a CLI-flag-driven load path, serialising the fabric
and placement to a file format, and keeping that format in sync with
whatever `Fabric`/`Placement` grow into -- real cost for no benefit, when the
whole reason this predictor exists is that we're already inside one Python
process that built these objects moments before calling into Frontier
(exactly like tasks 07/08's probes, and task 06's `install()`). A file
round-trip is the right shape for crossing a process boundary; there isn't
one here.

**Rejected: teaching `EngineKVCacheTransferConfig` to carry the objects as
non-CLI dataclass fields** (`field(metadata={"include_in_cli": False})`).
Mechanically possible, but it fights `create_flat_dataclass`'s subclass
walk for no reason -- that machinery exists to turn dataclass fields into CLI
arguments, and a field holding a `Fabric` has no CLI representation to skip
around. The module-level context does the same job with less surface.

This same decision will apply to the M2N predictor later: whichever install
call wires it up should extend the same context object (or a sibling one)
rather than re-deriving this from scratch.

## 3. Whether the ratio is what the fabric parameters predict

**Yes, almost exactly, and the "almost" is itself informative.** `packed`
routes over one scale-up link (400 GB/s); `split` crosses egress + scale-out
+ scale-out + egress, all four legs capped at 50 GB/s (`build_node_scale`'s
defaults) -- an 8:1 bandwidth ratio. Observed ratio: 7.999998882..., i.e. 8
to 7 significant figures.

Chasing the residual down to `engine/network/model.py` and
`engine/network/allocator.py` (read-only -- see S4) turned up something
worth stating plainly: **`grep -rn "latency" src/engine/network/` returns
nothing.** The contention-aware flow model
(`engine.network.transfers`/`FlowNetwork`/`max_min_fair_share`) that
`isolated_durations` runs on never reads `Link.latency_ns` at all -- that
field is consumed only by the ASTRA-sim config emission path
(`engine/cost/astra_config.py`). So a transfer's duration through this path
is purely `size_bytes / bottleneck_capacity`, with no propagation-latency
term to perturb the ratio away from the raw bandwidth ratio. The tiny
residual (about 1 part in 10 million) is float allocation/rounding noise in
the max-min-fair-share solver, not latency -- confirmed by hand: `1073741824
/ 400 = 2684354.56` vs the observed `2684355`, and `1073741824 / 50 =
21474836.48` vs the observed `21474837`; both are within 1 ns of the pure
bandwidth-only prediction, not within the ~936 ns or ~14000 ns a latency
term of the kind `astra_config.py` computes would add.

This isn't a defect in this task's predictor -- it's an accurate report of
what the engine module it's built on already does. It does mean this
predictor, as wired, is latency-blind: a fabric with much higher per-hop
latency but the same bandwidth would currently show no difference at all
through this path.

## 4. Whether anything tempted a change to `src/engine/`

One real temptation, and I did not act on it: **giving `isolated_durations`
a latency term**, once S3's investigation turned up that
`engine.network.transfers` doesn't use `Link.latency_ns`. I read
`engine/network/model.py` and `engine/network/allocator.py` to understand
why, and stopped there -- per the task's own constraint and per AGENTS.md,
this is exactly a "report it, don't add it" situation. Whether propagation
latency belongs in the contention-aware flow model, and if so how it should
interact with the max-min-fair-share admission logic, is a design decision
about the engine's cost model, not something task 09 should decide as a side
effect of one predictor's report.

Nothing else came close. `get_kv_cache_size`/`get_kv_cache_size_for_request`
being delegated to Frontier's own `AnalyticalKVCacheTransferPredictor`
(rather than re-derived) meant no temptation there either -- the engine
never needed to know anything about layers, KV heads, or quantisation.

## 5. Where the specification is wrong

- **S3.2's premise that Frontier's own TTFT must differ is incorrect for
  this Frontier version**, and it is incorrect for a structural reason, not
  a tuning one: `ttft` is defined as arrival-to-prefill-completion, which
  occurs *before* the KV transfer this predictor prices. The task's own S5.1
  anticipated exactly this outcome ("if they do not differ, that is the
  finding") and told me how to report it, which is what S1 above does. The
  metric that actually carries this predictor's placement-sensitivity is
  `request.kv_cache_transfer_time` (Frontier's own `ttft_kv_transfer`
  breakdown component) -- worth naming explicitly in any future task that
  asks for "TTFT" from a PD run, since the literal field will not move.
- Everything else in S2–S4 held up as specified. The 80/20 split from tasks
  07/08 (probe reuse, module-level classes, single install() entry point)
  carried over cleanly to a real predictor, not just a sentinel.

## What's still missing

Per S1 of the spec (restated in the predictor module's own docstring): this
does not model contention.
`frontier/events/kv_cache_transfer_start_event.py` computes
`transfer_end_time = self.time + duration` and schedules the end event
immediately, so two concurrent KV transfers cannot make each other slower
through this predictor -- the causality constraint `engine/network/transfers.py`
was written to satisfy structurally cannot reach Frontier's event loop until
that loop's completion-scheduling is replaced. The M2N predictor is
deliberately out of scope here (task 08 already measured its call volume:
192 against KV's handful per run, which changes the performance budget for
whatever wires it up) and belongs in its own task.
