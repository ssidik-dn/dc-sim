# Task 29 — Make it affordable

Branch: `task-29-affordability`, stacked on `task-28-optimum-shift`.
Paths confirmed per task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`.

189 tests pass, unchanged, and `python3 tools/check_import_direction.py`
exits 0.

**Zone check before touching anything**: `network_for`/`_path_latency_ns`
(`src/engine/network/transfers.py`) sit in the same file as the
max-min fair-share flow model (`FlowNetwork`, `Completion`) —
event-semantics/completion-revision territory `AGENTS.md`'s Human-only
zone describes in spirit, even though it names an (empty) `fabric/`
placeholder rather than `network/` by path. Confirmed with the user
before writing anything: proceed as an agent-implemented change, since
the two functions being touched are pure, static lookups over immutable
`Fabric` data with no event semantics of their own — the contention
logic elsewhere in the same file is untouched.

---

## 1. The profile after §2.1 alone

Cached `Link.id` and `GpuId`/`NicId`/`SwitchId`'s string forms once per
instance (`__post_init__` + `object.__setattr__` on an unannotated
attribute name — invisible to the dataclass-generated `__eq__`/`__hash__`/
`__lt__`, which are built from the declared fields only, so equality,
hashing, and ordering are byte-for-byte unaffected). Nothing else
changed. Re-ran task 26's own fabric-size sweep:

| n_gpus | wall_s (before) | wall_s (§2.1 only) |
|---|---|---|
| 32 | 2.37 s | 2.00 s |
| 64 | 4.00 s | 2.41 s |
| 128 | 9.98 s | 3.84 s |
| 256 | 35.34 s | 10.98 s |
| 512 | 245.27 s | **71.61 s** |

**Both the constant and the exponent moved** — `wall_s ~ n_gpus^1.65`
(before) → **`n_gpus^1.25`** (§2.1 alone). A 3.4x speedup at 512 GPUs,
and the growth rate itself dropped substantially, but `1.25` is still
clearly superlinear, not the ~1.0 that would make §2.2 optional per
this task's own §2.1 framing ("if the exponent moves as well as the
constant, the second change may be unnecessary"). The post-§2.1
profile confirmed why: `network_for`'s and `_path_latency_ns`'s own
dict comprehensions dropped from a combined ~213s to ~40s (string
formatting gone), but they still rebuilt a fresh 131,600-entry
structure on every one of 896 calls — the per-call rebuild itself, not
just the string cost inside it, was still there. §2.2 proceeded.

## 2. §2.2: what was chosen, and what it assumes

**Chosen: build the capacity map (and the `{id: Link}` index
`_path_latency_ns` needs) once per fabric and keep it — the first of
this task's own three listed options.** Cached as two lazily-computed
methods on `Fabric` itself (`link_index()`, `capacity_index()`,
`src/engine/physical/topology.py`), rather than lazy-per-call
construction (option 2, which still pays for however many *distinct*
link sets a run touches) or memoising on the link set (option 3, which
degenerates to option 1's own win here anyway, since `network_for`
takes no link-set argument at all — it prices the *whole* fabric
regardless of which transfer is being priced).

**What this assumes, established rather than taken from the dataclass
being frozen** (this task's own trap): that nothing mutates a `Fabric`
after handing it to `install()` or a predictor. Checked directly —
`grep -rn "\.add_link(\|\.add_machine(\|\.add_domain(\|\.bind_nic("
src/ tools/ tests/` — every call site sits inside a builder function
(`build_node_scale`/`build_rack_scale` in `builders.py`, the parsers in
`infragraph/parse.py` and `infragraph/blueprints.py`, and each test's
own fabric-construction helper), always before the fabric is used
elsewhere, never after. `Fabric` itself is not a frozen dataclass — a
plain, mutable class — so this is a property of *how every current
caller uses it*, not a compiler-enforced guarantee, and is stated as
such. As a safety net against that assumption ever breaking silently,
`add_link` now invalidates the cache (`self._link_index_cache = None`,
`self._capacity_index_cache = None`) on every call, so a future mutator
gets a recomputed map instead of a stale one, at zero cost to the
common case (the cache is `None` until first read either way).

`FlowNetwork.__init__` copies whatever dict it receives
(`self.capacity = dict(capacity)`) — confirmed by reading it, not
assumed — so handing the *same* cached dict to a fresh `FlowNetwork` on
every call is safe regardless: nothing downstream can mutate the
shared cache through the object it feeds.

## 3. The new fitted exponent, and the new wall-clock at 512 GPUs

| n_gpus | wall_s (before) | §2.1 only | §2.1 + §2.2 |
|---|---|---|---|
| 32 | 2.37 s | 2.00 s | 1.89 s |
| 64 | 4.00 s | 2.41 s | 2.06 s |
| 128 | 9.98 s | 3.84 s | 2.72 s |
| 256 | 35.34 s | 10.98 s | 5.81 s |
| 512 | 245.27 s | 71.61 s | **32.72 s** |

**`wall_s ~ n_gpus^0.97`** — effectively linear. From `n_gpus^1.65`
(before) through `n_gpus^1.25` (§2.1 alone) to `n_gpus^0.97` (both
changes): the ceiling that was a slope steeper than the fabric's own
already-quadratic link count is now tracking GPU count itself.
16x the GPUs (32→512) now costs about 17x the wall-clock, not the
103x it cost before either change. 512 GPUs, 16 requests: **32.72
seconds**, down from 245.27 — a 7.5x total reduction, `n_links` and
`peak_rss`'s own fitted exponents (`^1.97`, `^0.04`) unchanged, exactly
as expected since neither §2.1 nor §2.2 touched what a fabric *is*,
only how often its link structure gets rebuilt.

## 4. The before-and-after comparison, in full

**`tools/run_collective_backend_study.py`, tp=2/4/8** — bit-identical,
every digit, before and after both changes:

| tp | packed `tp_comm` | split `tp_comm` | packed tpot | split tpot |
|---|---|---|---|---|
| 2 | 0.913024 ms | 13.131776 ms | 5.612679 ms | 7.358215 ms |
| 4 | 2.628864 ms | 38.513664 ms | 5.803319 ms | 10.929719 ms |
| 8 | 6.008576 ms | 88.836608 ms | 6.272703 ms | 18.105279 ms |

Identical in both runs (before this task's changes, and after both
§2.1 and §2.2), to every reported digit — matching the task's own
acceptance table exactly.

**The memory-and-degree grid at margin=0.9, tp=2 packed:**

| | throughput | tpot | batch |
|---|---|---|---|
| Before | 90.61152851669767 req/s | 13.953915179821097 ms | 8 |
| After | 90.61152851669767 req/s | 13.953915179821097 ms | 8 |

Bit-identical, full float precision, not rounded for the comparison.

**M2N placement comparison** (`tools/run_m2n_integration.py`):

| | colocated M2N | split M2N | ratio | colocated tpot | split tpot | tpot ratio |
|---|---|---|---|---|---|---|
| Before | 0.187776 ms | 2.750976 ms | 14.6503 | 422.943318 ms | 423.797718 ms | 1.0020 |
| After | 0.187776 ms | 2.750976 ms | 14.6503 | 422.943318 ms | 423.797718 ms | 1.0020 |

Bit-identical. The only figure that changed anywhere in this
comparison is each run's own reported "mean call cost" — a raw
wall-clock measurement of the predictor call itself (210.06 us → 63.16
us colocated; 250.89 us → 101.18 us split), which is expected to move
and is not one of the reported *results* this task's acceptance table
asks to hold still; it is the speedup being measured.

**Nothing moved.** No figure required to be bit-identical differs by
so much as a floating-point last digit, in either comparison, across
either change.

## 5. What is now the dominant cost, and whether it is worth another pass

**`fabric.path()`'s own breadth-first search — confirmed via a fresh
profile at 512 GPUs, not inferred from the exponent alone.** 26.19s of
the new 32.72s total (80%). `network_for` is down to 1.64s cumulative
(896 calls); `_path_latency_ns` no longer appears distinctly in the
top 60 functions at all. This is exactly this task's own §5 "known
trap" — *"Path computation becomes the dominant term once the rebuild
is gone, at about 10% of the old total"* — and it is worth stating
plainly rather than optimised reflexively, per that same trap's own
instruction: **10% of 245.27s (task 26's original total) is 24.57s,
and the measured `path()` cost after both changes is 26.19s — within
profiler noise of that same absolute figure.** `path()`'s own absolute
cost barely moved (24.57s → 25.53s → 26.19s across all three
measurements) because neither change touched it; what changed is that
everything *around* it got smaller, so the same unchanged cost is now
the large majority of a much smaller total.

**Is it worth another pass?** Not in this task — precomputing paths is
listed in this task's own §2.2 options as a candidate, but the profile
after §2.1+§2.2 shows the *exponent* problem is already solved
(`n_gpus^0.97`); this task's own §2.3 explicitly reserves anything that
would change the exponent (representing dense scale-up domains
implicitly) for separate, careful work, and precomputing paths is a
smaller change in the same spirit — a real, present cost, not a wall,
now dominant only because the two changes actually made worked. Left
as a finding for whoever picks up path caching next, not addressed
here.

## 6. Anywhere this specification is wrong

**Nothing.** Every figure quoted in this task's own §1 — 212.99 s
(86.9%), 155.81 s (63.5%, 235,830,784 calls), 73.52 s (30.0%,
469,763,840 calls), 24.57 s (10.0%), 1.66 s (0.7%), ≈3.3 s (≈1.4%), and
the fitted `wall_s ~ n_gpus^1.65` — matches
`docs/tasks/26-scale-report.md`'s own §A.2/§A.1 exactly, checked by
direct `grep`. This is the first task in this recent sequence (25-29)
whose own opening citations all held up against their stated source
without needing a correction.

## What shipped

- `src/engine/physical/topology.py` — `GpuId`/`NicId`/`SwitchId` cache
  their string form once via `__post_init__`; `Link` does the same for
  `id` (now a plain precomputed attribute, not a `@property`);
  `Fabric` gains `link_index()` and `capacity_index()`, both cached on
  the instance and invalidated by `add_link`.
- `src/engine/network/transfers.py` — `network_for()` and
  `_path_latency_ns()` read the new cached indices instead of
  rebuilding them; no other change to either function's behaviour.

One commit on `task-29-affordability`, stacked on
`task-28-optimum-shift`; nothing under `upstream/` or
`src/integration/` touched. Pure optimisation, per this task's own
acceptance criteria — every regression figure checked reproduces to
the last digit.
