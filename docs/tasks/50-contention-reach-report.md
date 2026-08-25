# Task 50 — Has anything ever contended?

Branch: `task-50-contention-reach`, branched from `task-49-mla-recovery`'s
tip. Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`.

254 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0 (26 files, +1 for the new counters module). Task 33's own
sixteen-row table and Task 36's own two-fabric result both reproduce
bit-identical with the instrumentation in place — the required proof that
adding counters changed no computed value.

---

## 1. Whether a flow network persists across calls

**No — confirmed from the code, for the collective path, exactly as this
task's own hypothesis states.** `engine.network.transfers.run_transfers`
calls `network_for(fabric, ...)` — a brand-new `FlowNetwork` — at its own
top, every invocation; nothing outside the function keeps a reference to
it. `EngineCCBackend` (`src/integration/cc_backend/engine_backend.py`,
read in full) stores only `fabric`/`placement`/`groups` — no in-flight set,
no clock, no link-occupancy field anywhere in the class — and its own two
call sites each build a fresh `transfers` list scoped to exactly one
operation: `_round_ns` (line 138) batches only the edges of *one round* of
*one* collective call; `predict_send_recv` (line 270) submits a single-item
list. Two different collectives — or two different rounds of the *same*
collective, since `predict_allreduce` computes `rounds * self._round_ns(...)`
rather than accumulating real per-round completions in one persistent
network — never share a network with each other. Confirmed live, not only
read: §2 below shows tens of thousands of networks constructed across one
run.

**A sharper finding this task's own §1 did not anticipate.** KV-cache and
M2N/activation transfers — priced through `src/integration/binding_support.py`'s
`price_transfer`, the function both `EngineKVCacheTransferPredictor` and
`EngineM2NTransferPredictor` call — **never reach `run_transfers` at all**.
Every code path inside `price_transfer` bottoms out at `_price()`
(the unambiguous-both-sides case, and the `bind()`-resolved branch of
`_resolve_ambiguous_side`) or a direct call (the `"late"`-binding branch of
`_resolve_ambiguous_side`) to **`isolated_durations`** — the function
`transfers.py`'s own docstring names explicitly as the contention-free
comparison baseline ("Each transfer run alone... so any difference is
contention and nothing else"). Grepped exhaustively (`grep -rn
"run_transfers(" src/ tests/ tools/`): the only callers are
`engine_backend.py`'s own two collective-facing sites, `transfers.py`'s own
`analyse()` (used only by direct `engine.network.transfers` probes — Task
40/43A's own precedent, and this task's own §2.2), and three test files.
**`binding_support.py` never appears in that list.** So for an entire
category of traffic — every KV-cache transfer and every M2N/activation
transfer this project prices — the contention mechanism is not merely
isolated per call, it is **never invoked**, by explicit function choice,
independent of payload size or concurrency.

## 2. The four counters, for one real run

Instrumentation: `src/engine/network/contention_counters.py`, a plain
`ContentionCounters` dataclass (four integer fields, a `reset()`), wired
into `FlowNetwork.__init__` (`networks_constructed += 1`) and
`FlowNetwork._reallocate` (the other three, computed from state
`_reallocate` already has — the pre-reallocation `Allocation`, each live
flow's own links, and the shared `capacity` dict). Nothing reads these
counters back into any computed value; §0 above and this section's own
bit-identical reproductions are the direct proof.

One complete `pd-af-disaggregation` run under the streaming regime
(`domain8`, `attn_tp=1`, `ffn_ep=4`, Phi-tiny-MoE-instruct, the exact model
and workload Tasks 44/45 use), at **two configurations** and two seeds
each — deliberately not one arbitrary run, for the reason §5 explains:

| config | `ep_shape` | seed | mean tpot (ms) | networks constructed | max flows/network | rate reductions | completion revisions |
|---|---|---|---|---|---|---|---|
| **winner** (Task 45's own real streaming winner) | `(4,)` — packed | 0 | 3.6586 | 58,592 | 12 | **0** | **0** |
| winner | `(4,)` | 1 | 3.8786 | 45,888 | 12 | **0** | **0** |
| **split** (never a winner in any study; a rejected search alternative) | `(2,2)` | 0 | 6.4738 | 47,136 | 12 | **953,856** | **59,616** |
| split | `(2,2)` | 1 | 6.6784 | 35,776 | 12 | **781,824** | **48,864** |

(`networks_constructed` mixes the two populations §1 distinguishes —
single-flow KV/M2N networks from `isolated_durations` and multi-flow
collective networks from `run_transfers` — both increment the same
counter, since the counter answers "how many networks, total," not "how
many of which kind." `max_flows_in_flight=12` matches `ep=4`'s own full
personalised all-to-all: 4 ranks, 3 outbound edges each = 12 simultaneous
edges in one round, present identically in both configurations, since only
the *placement* differs, not the collective's own structure.)

## 3. Whether a completion has ever been revised outside a test

**Yes — decisively, and for the first time outside a synthetic probe or a
closed-form test.** The `split` configuration's own hundreds of thousands
of revisions per run are a real, `Simulator`-driven measurement, not a
hand-built scenario. **But zero for the `winner` configuration — and every
configuration any task in this project has ever reported a real, cited
number from is a winner, never a split.** Checked directly against this
project's own history: Task 32's own search always ranks packed shapes
first; Task 44/45's own EP-degree studies both report `ep_shape=(4,)` (or
whichever fully-packed shape is reachable) as the winner at every degree
tested, with split shapes (`(2,2)`, `(1,1,1,1)`, etc.) appearing only as
*measured-and-rejected* alternatives in a ranking table, never as the
number a report's own headline cites. **The revision mechanism has fired,
demonstrably, in this project's own real simulator — just never inside a
configuration this project has ever chosen to report a result from.**

## 4. What would make it non-zero in a reported result

- **Larger payloads or more groups (Task 43A's own explanation) — confirmed
  true, and directly demonstrated, not merely restated.** §2's own table is
  the demonstration: same model, same workload, same fabric, same
  collective structure (12 edges either way) — the *only* thing that
  changed between the zero row and the non-zero row is placement shape.
  Task 43A's own synthetic probe (eight megabyte flows, a direct
  `engine.network.transfers` construction) already proved the mechanism
  works at large-enough payloads; this task's own real run proves the
  *same* mechanism fires inside an actual `Simulator`, at this project's
  own real (small, KB-scale) payload — the missing ingredient was never
  size, it was whether the *edges* of one round actually share a link, and
  a split placement is exactly what makes them.
- **Persisting the network across calls** — this task's own §2.3 asks
  whether this is *needed*. It is not shown to be, by §3's own result: the
  mechanism already fires for real, inside one round's own already-shared
  network, without any cross-call persistence at all. Persisting the
  network would let *different* collectives (or a collective and a
  KV/M2N transfer) that overlap in simulated time contend with *each
  other* — a materially different, larger question this task's own
  measurement does not answer either way — but it is not required to get
  the number reported here off zero, and this task does not attempt it,
  per its own explicit instruction. The obstacle the spec's own §2.3
  names — the predictor is called synchronously and told nothing about
  when a transfer actually starts relative to others — remains exactly as
  real as stated, and is not resolved by anything measured here.
- **Something else the reading suggests**: the KV/M2N path (§1's own
  second finding) is a distinct, third lever — moving it from
  `isolated_durations` to `run_transfers` would let KV/M2N transfers
  contend with each other (and, if the network were also persisted, with
  collectives) — but this is a design change with its own correctness
  questions (does an M2N transfer's own "late" binding-averaging semantics
  even make sense against a shared, stateful network?), not something
  this task attempts.

## 5. Whether Task 43A's explanation is complete

**No — it is one of two reasons, and this task establishes the second.**
Task 43A's own explanation (payloads too small, groups too small, for the
traffic this project's real decode step generates) is **true and
sufficient to explain why the actual, chosen configurations never
contend** — confirmed directly (§4). But it is not the *only* reason
nothing has ever contended in a *reported* number, and conflating the two
would overstate what Task 43A alone established. The second reason,
found here: **this project's own search machinery (Task 32 onward)
optimizes for the objective the contention model itself prices, and a
split placement is always slower — so the search never selects one as a
winner, and no report has ever cited a number computed from a
configuration where contention fires**, even though such configurations
exist, are reachable by the same search's own enumeration, and — proven in
§2/§3 — do contend, substantially, when actually run. "The traffic shape
this project generates doesn't contend" and "the traffic shape this
project's search chooses to report doesn't contend" are different claims;
both are true, and the second is the sharper, more complete one, because
it holds even in configurations Task 43A's own explanation would have
called safe from contention by degree alone (`ep=4` here is well within
what this project's own studies already exercise — no exotic scale-out
needed, only a different, already-enumerable placement of the same
degree).

Restated against this task's own §3's own framing: **"a contention model
that prices concurrent transfers" is fully supported — collectives
genuinely do, KV/M2N genuinely don't (a new, narrower claim this task adds).
"A contention model that has priced concurrent transfers, in a result this
project has reported" is not supported — every reported result comes from
the isolated-cost regime, not because contention cannot fire, but because
the winning configurations never expose it.**

## 6. Anywhere this specification is wrong

**The central hypothesis (§1's own "fresh network per call") is
accurate against the current file, not stale**, contrary to this task's
own explicit caution in §6 ("a hypothesis from reading an older version of
the file, not an established fact") — `transfers.py` and `engine_backend.py`
were read fresh in this task, and both confirm it exactly as stated.
Nothing in either file suggests a persistence mechanism was ever added and
removed, or that this reading is outdated.

**One place worth a precise correction, not to the spec but to how its
own two "reasons" should be weighted**: §1's own framing treats "payloads
too small" and "predictor calls never share a network" as two
*competing* hypotheses, one of which should turn out to be the explanation.
Both are real mechanisms, but neither, alone, is what actually kept every
*reported* number contention-free — §5 above is the more precise account:
payload/shape (Task 43A's own axis) determines whether contention *could*
fire for a given placement; the search's own optimization objective
determines which placements are ever *reported*; and (this task's own new
finding) KV/M2N transfers are contention-incapable by construction,
independent of either axis. Three mechanisms, not two, once KV/M2N is
counted separately from collectives — worth stating precisely rather than
picking one of the spec's own two candidates as "the" answer.

**One tension with `AGENTS.md`, flagged as it was in Task 47.**
`AGENTS.md`'s own zoning marks "completion revision" — the literal subject
of `network/model.py`'s own module docstring — human-only, listing
`fabric/` contention code "when it lands" as the example; the contention
model in fact landed under the sibling directory `network/`, not literally
under `fabric/`. This task's own instruction ("if instrumentation requires
a code change, keep it to counters that do not alter any computed value")
is explicit, narrow authorization for exactly this kind of change,
matching the precedent Task 47 already established for a similar tension
— proceeded on that basis, flagged here rather than silently assumed or
used to justify skipping the instrumentation the task asks for.

**Otherwise, nothing else in this specification required correction.**
Its own three-way framing of what §2.3 would require (larger
payloads/groups; persisting the network; something else) anticipated
exactly the three mechanisms §4/§5 above found, in the same order of
plausibility the spec itself suggested.

## What shipped

- `src/engine/network/contention_counters.py` — a four-field counter
  dataclass, read by nothing else in `src/engine/`.
- `src/engine/network/model.py` — `FlowNetwork.__init__` increments
  `networks_constructed`; `_reallocate` computes the other three from
  state it already builds (no new computation added to the allocation
  path itself, only observation of its own inputs/outputs).
- `docs/tasks/50-contention-reach-report.md`, this report.

One commit on `task-50-contention-reach`, stacked on `task-49-mla-recovery`.
Task 33's sixteen-row table and Task 36's two-fabric result both reproduce
bit-identical with the instrumentation in place. 254 tests pass, unchanged;
`check_import_direction.py` exits 0.
