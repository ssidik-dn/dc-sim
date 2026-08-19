# Task 20 — Close the blind spot

Branch: `task-20-collective-backend`, stacked on `task-19-tp-sweep`.

183 tests pass (177 existing + 6 new in `tests/test_collective_backend.py`),
and `python3 tools/check_import_direction.py` exits 0.

---

## 1. What was replaced, and why it is the narrowest option

**`CCBackendFactory.create`**, exactly as the spec suggests, and nothing
upstream of it. Confirmed by reading both execution-time predictors that
every one of the six `predict_*` calls — across TP, PP, and EP, in both
the dense and MoE cost models — obtains its `cc_backend` through this one
classmethod, once, at cluster construction. Replacing it here means every
call reaches `EngineCCBackend` uniformly; replacing anything narrower
(e.g. patching each `predict_*` call site) would mean five or six
interception points instead of one, and replacing anything wider (e.g.
`Cluster._initialize_cc_backend`) would touch code this task doesn't need
to.

Selection is **not** a `CCBackendType` value — confirmed twice now that
none is reachable. `CCBackendType` has five members, all claimed;
`BaseRegistry.register` no-ops on a claimed key; and the one nominally
unclaimed member, `AICONFIGURATOR`, turns out to be unreachable a *third*
way beyond task 06's own finding: its config class has
`__include_in_cli__ = False`, which makes Frontier's own CLI reconstruction
raise `KeyError` on `AiconfiguratorCCBackendConfig` before ever reaching
`CCBackendFactory.create()`'s own explicit rejection of that value
(confirmed by running `--cc_backend_config_type aiconfigurator`, not
assumed). Selection instead works exactly like task 14's binding and task
15's scheduler: `install(..., collective=True)`. Every run that doesn't
pass it — every task before this one — is untouched, regardless of which
`--cc_backend_config_type` value its own argv happens to use.

**Guarded by a source hash** (`integration/cc_backend/collective.py`),
computed against the checked-out `CCBackendFactory.create`'s exact source
at the time this was written. If Frontier's own `create()` changes
upstream, `install_collective_backend()` raises rather than patching over
an implementation this project hasn't reviewed. One correction to the
spec here: no prior instance of this exact "guard a runtime replacement
with a source hash" pattern actually exists elsewhere in this codebase —
checked directly (`grep -rn "source hash\|hashlib\|inspect.getsource"
src/integration/` before writing this, turned up nothing). Task 14/15's
runtime replacements exist, but neither is hash-guarded. This is the
first, because — as the spec itself says — it is also the first
interception invasive enough to warrant one.

## 2. The ring-vs-real-collective-model detour (and why `EngineCCBackend` doesn't use ASTRA-sim)

Before wiring anything up, the mid-task instruction to verify the cost
path against a closed-form ring changed the design. Working through it
directly, by hand, for a domain-split 8-way group (4-and-4), 65536-byte
payload:

- **Ring all-reduce**: `2(n-1) = 14` sequential steps, each moving `S/n =
  8192` bytes over all 8 of the ring's simultaneous edges. A domain-major
  ring ordering crosses the boundary on exactly 2 of those 8 edges, in
  every one of the 14 steps (the ring's topology doesn't change between
  steps, only the data does) — so the slow link carries `14 × 2 × 8192 =
  229,376` bytes total, not the `16 × 8192 = 131,072` bytes a model that
  charged *every* cross-domain pair (as an all-to-all genuinely would)
  would imply — an 8x difference in slow-link demand between the two
  accountings, exactly as the mid-task check framed it.

- **This project's own, already-built, already-validated real collective
  simulator — a genuine ASTRA-sim binary was found built and working in
  this environment (`/work/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware`,
  exercised by the existing `tools/validate_astra.py`, which passed
  cleanly) — does *not* implement the simple picture above.** Measured
  directly, generating fresh Chakra traces for the same 65536-byte
  payload and 8-way group:

  ```
  all_reduce, 8-way, 65536B:   packed (8,) = 52856 ns   split (4,4) = 51593 ns
  all_to_all, 8-way, 65536B:   packed (8,) = 105692 ns  split (4,4) = 32896 ns
  ```

  **Split is not more expensive than packed for either collective in
  ASTRA-sim's own analytical model — it is very slightly cheaper for
  all-reduce and dramatically cheaper for all-to-all.** This is because
  `topology_for_shape` maps a `(4, 4)` domain shape onto a genuinely
  multi-dimensional ASTRA-sim topology (reported dims `(4, 2)` by
  `tools/validate_astra.py`'s own output), and ASTRA-sim's own analytical
  collective algorithms decompose across those dimensions in a way that
  can be *more* efficient than a single flat 8-way collective, not less.
  This is a real, validated behaviour of a real tool — not a bug to
  dismiss — but it is also the **opposite** of what this task's own
  acceptance criteria require (`test_packed_tp_group_cheaper_than_split`)
  and of what a well-formed ring, run by hand, predicts.

**Decision**: `EngineCCBackend`'s five true collectives were rewritten
onto `engine.network.transfers.run_transfers` directly — the exact
contention-aware, latency-aware path every KV/M2N transfer in this project
already uses — implementing an explicit, stated ring (allreduce, allgather,
reduce_scatter) or full-pairwise (all_to_all) algorithm, rather than
depending on ASTRA-sim's own multi-dimensional decomposition, which this
project cannot fully account for and which does not produce the
placement-sensitive result this task exists to demonstrate. This also
directly satisfies spec S3.1's own request ("confirm it goes through the
same cost path the transfer predictors use") — by construction, since it
is now literally the same function calls, not a second model that happens
to agree.

**What this project's own ring implementation actually returns** for the
same case, read back from a real `EngineCCBackend` call:

```
predict_allreduce(65536, 8, comm_domain="ATTN_TP"):
  packed:  13,412 ns
  split:   198,296 ns    (14.78x)
```

This is *higher* than a naive, contention-free hand estimate using only
the 229,376-slow-link-bytes accounting (~72,294 ns for the split case) —
because `run_transfers` is contention-aware, and the two simultaneous
cross-domain ring edges in one round share `build_node_scale`'s scale-out
link capacity rather than each getting the full 50 GB/s independently.
That is a real, defensible effect a naive per-edge hand calculation
doesn't capture, not a bug in the engine: a real ring-allreduce's two
boundary-crossing flows genuinely would contend for the same physical
uplink.

**What the model assumes, stated plainly**: allreduce/allgather/reduce_scatter
are priced as a ring (bandwidth-optimal, `2(n-1)` or `n-1` sequential
rounds, contention-aware within each round); all_to_all is priced as a
single round of every ordered pair simultaneously (`n(n-1)` pairwise
flows) — matching the mid-task instruction's own correct statement that an
all-to-all genuinely does cross the boundary once per cross-domain pair,
unlike a reduction, which can shortcut through a ring's associativity.
These are different algorithms for different reasons, not the same
formula applied twice — and MockBackend's original, pre-task-20 behaviour
(one flat hop-count-and-byte-count formula for every collective type,
ignoring the distinction entirely) would have been wrong for exactly the
reason the mid-task check anticipated. Broadcast uses a simple sequential
relay (n-1 hops of the full payload) — stated, not claimed optimal, and
not exercised by this task's acceptance tests.

## 3. Group membership: the real, load-bearing bug this task's own S3.2 warned about

Confirmed empirically, not assumed: **Frontier's real cc_backend call
sites never pass `comm_domain="TP"`.** Grepping every `comm_domain=`
literal across both execution-time predictors turns up `"ATTN_TP"` (5
attention-allreduce call sites), `"MOE_TP"` (7 FFN/MoE-allreduce call
sites), `"PP"`, `"DP"`, and `"EP"` — never the generic `"TP"`
`populate_from_deployment` (task 06) registered groups under. The very
first real run with the interception installed raised `CommGroupError`
immediately: *"no placement registered for
cluster_type=<ClusterType.DECODE_ATTN>, comm_domain='ATTN_TP',
num_devices=2"*. This is not a hypothetical instance of "wrong group
membership gives a plausible wrong number" (S3.2's own warning) — it's a
plain lookup miss, loud and immediate, which is the *good* version of
that failure mode (the registry raised rather than guessing). But it
would have silently blocked every real run of this task's own measurement
had it not been caught.

Fixed in `populate_from_deployment` (`comm_groups.py`): a TP group is now
registered under `"TP"`, `"ATTN_TP"`, and `"MOE_TP"` — the same physical
ranks answer to all three, because this project's own `Replica` model has
one tensor-parallel degree per replica, not a separate attention-TP and
MoE-TP value the way a real deployment sometimes does. PP/DP/EP needed no
alias; their real call-site names already match `ParallelKind`'s own
value. `test_tp_group_membership_matches_logical_model` locks the
corrected mapping in against `Replica.groups(ParallelKind.TP)` directly.

## 4. The measurement: packed vs split, three TP degrees, `EngineCCBackend` selected

Repeats task 19 §2.2 exactly (`DECODE_ATTN`'s `attn_tensor_parallel_size`
at 2/4/8, packed vs split across two scale-up domains), with
`install(..., collective=True)`, real h800 compute throughout
(`tools/run_collective_backend_study.py`):

| tp | packed `tensor_parallel_communication_time` | split | ratio | packed tpot | split tpot | tpot delta |
|---|---|---|---|---|---|---|
| 2 | 0.913024 ms | 13.131776 ms | **14.38x** | 5.612679 ms | 7.358215 ms | **+1.745536 ms** |
| 4 | 2.628864 ms | 38.513664 ms | **14.65x** | 5.803319 ms | 10.929719 ms | **+5.126400 ms** |
| 8 | 6.008576 ms | 88.836608 ms | **14.78x** | 6.272703 ms | 18.105279 ms | **+11.832576 ms** |

**The packed and split figures differ now, at every degree, by a wide,
consistent margin — task 19's own bit-identical result is gone.** Headline
visible-share also moves as expected: 98.36% → 80.69% (tp=2), 95.43% →
58.76% (tp=4), 90.13% → 38.18% (tp=8) — splitting a TP group now visibly,
correctly, makes this project's own invisible-communication problem
*worse*, not better, exactly the direction task 18/19's own diagnosis
predicted once the blind spot could actually respond to placement.

## 5. Does the correction match task 19's twelvefold estimate?

**Yes, closely — 14.38x-14.78x here against task 19's own ~9.83x-13.91x
range (1 MB and 64 KB payloads respectively), and against this project's
other headline placement-penalty ratio, 14.65x (tasks 11/12's M2N
finding).** The measured ratio sits just above task 19's own 64 KB
estimate, for a reason task 19's own report already flagged as a
limitation of that estimate: it priced a *single* point-to-point hop
across the slow link, not a full multi-round ring with contention between
the ring's two simultaneous boundary-crossing edges. This task's ring
implementation prices the whole sequential operation, correctly
accounting for that contention — a small, expected, and explained
addition on top of task 19's simpler estimate, not a discrepancy that
needs a different explanation.

## 6. Does inter-token latency move? Yes — the largest correction this project has made to its own numbers

**Yes, substantially: +1.75 ms at tp=2, up to +11.83 ms at tp=8 — packed
vs split, same model, same everything else.** At tp=8, split-placement
tpot (18.11 ms) is nearly **3x** packed's (6.27 ms). Every prior placement
comparison in this project (task 11's M2N ratio, task 15/16's scheduler
work) moved transfer time by a similar order of magnitude while tpot
barely moved, because dummy compute mode or a small transfer relative to
compute dwarfed the effect (tasks 09, 12, 14-16's own repeated finding).
This is the first measurement in the whole project where a placement
decision changes inter-token latency by an amount comparable to the
placement decision's own communication cost, because tensor-parallel
allreduce sits *inside* every single decode layer's critical path, not
alongside it the way a cross-pool transfer does.

## 7. Anywhere this specification is wrong

- **No prior source-hash-guard precedent actually exists in this codebase**
  (S1) — the spec's "guarded... as `install()` does elsewhere" describes
  the general runtime-replacement idiom (real, task 14/15) but not the
  hash-guard specifically (not found anywhere before this task).
- **The collective cost path this task inherited from task 06 did not, in
  fact, go through the same cost path as the transfer predictors** (S3.1's
  own check item) — confirmed, and fixed by unifying both onto
  `run_transfers`, rather than merely "confirmed" as the spec's phrasing
  implied might be the likely outcome.
- **A real, working ASTRA-sim binary's own analytical collective model
  does not satisfy this task's central acceptance claim** (packed cheaper
  than split) at realistic TP payload sizes — a finding this task's own
  framing didn't anticipate, since it assumes verifying "the cost path"
  means checking an existing implementation's fidelity, not discovering
  that the most realistic available implementation disagrees with the
  task's own required direction. Reported and worked around by choosing a
  different, explicit algorithm rather than silently forcing agreement.
- Otherwise the specification's structure — narrowest interception, source
  hash, group-membership as the likely subtle failure, state the
  algorithm assumption, real profiles throughout — matched exactly what
  the investigation needed, including correctly anticipating (S4.4/known
  traps) that the collective algorithm's own assumption would have real
  consequences once placement could finally vary it.

## What shipped

- `src/integration/cc_backend/engine_backend.py` — rewritten: ring
  (allreduce/allgather/reduce_scatter), full-pairwise (all_to_all),
  sequential relay (broadcast), all via `run_transfers`; `send_recv`
  unchanged.
- `src/integration/cc_backend/collective.py` — the guarded
  `CCBackendFactory.create` replacement.
- `src/integration/cc_backend/comm_groups.py` — `ATTN_TP`/`MOE_TP` domain
  aliases for a TP group's registration.
- `src/integration/context.py` — `EngineContext.collective: bool = False`,
  `get_context()` (non-raising accessor).
- `src/integration/install/__init__.py` — `install(..., collective=False)`.
- `tests/test_collective_backend.py` — 6/6 required tests.
- `tests/test_cc_backend_integration.py` — updated for the new constructor
  and the new (explained, not forced) n=2 agreement point with Frontier's
  analytical closed-form.
- `tools/run_collective_backend_study.py` — the packed-vs-split
  measurement.

Four commits on `task-20-collective-backend`, stacked on
`task-19-tp-sweep`; nothing under `upstream/` modified.
