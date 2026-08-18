# Task 10 report — Latency in the flow model

Branch: `task-10-latency` (not merged to main).

`python3 -m pytest -q` (150 passed: 143 existing + 7 new in
`tests/test_latency.py`) and `python3 tools/check_import_direction.py` pass.
Three commits: the model/allocator change plus the one existing-test fix it
required, then the new test file.

---

## 1. Which model was chosen, and what it gets wrong

**Model B — latency added to the computed completion — with one refinement
the spec's description doesn't fully cover.**

A naive reading of B ("allocate as today, then add `path_latency` when
reporting completion") suggests a one-line change: take the existing
bandwidth-only completion time and add a constant. Implementing it that way
breaks immediately, in two ways I only found by writing it:

- If `advance_to`'s stepping loop drains every flow up to the
  *latency-inflated* time, survivors sharing a link with the finishing flow
  are held at their old, throttled rate for the finishing flow's entire
  latency tail -- bandwidth that should have been freed to them the moment
  the finishing flow's bytes stopped moving is instead withheld until its
  latency also elapses. That's a real allocation error, not a rounding
  quirk.
- The bottleneck attributed to a flow that completes during a *later*
  reallocation is read from `self._alloc` at report time -- but by then
  `self._alloc` has already been recomputed by whatever reallocation
  happened when the flow drained (and every one since), and no longer has
  an entry for it. `bottleneck` silently comes back `None`. This is not
  hypothetical: it broke 3 of the 143 existing tests the first time I ran
  them (`test_intra_domain_transfer_uses_scale_up_only`,
  `test_cross_machine_transfer_bottlenecks_on_the_narrow_step`,
  `test_report_attributes_bottlenecks_by_class` -- see S2).

The fix is a third `FlowState`, `DRAINED`, sitting between `IN_FLIGHT` and
`COMPLETED`: a flow's bytes finishing (`remaining_bytes` hits zero) frees
its bandwidth share to survivors *immediately*, exactly as before task 10 --
`_reallocate()` runs over `_active()` (IN_FLIGHT only), never DRAINED. Its
bottleneck is captured onto the `Flow` at that instant, before the next
reallocation can erase it. Only later, when its fixed `path_latency_ns` has
also elapsed (tracked from `drained_ns`, set once), does it become
`COMPLETED` and get reported. `next_completion_time_ns()` reports the
latency-inclusive prediction (what a host actually cares about scheduling
on); a separate, private `_bandwidth_event_ns()` drives the internal
reallocation timing so the two purposes -- "when does a link free up" and
"when does the host get told this is done" -- don't get conflated.

This satisfies the three invariants directly:

1. **Charged once.** `path_latency_ns` is added in exactly one place --
   against the flow's currently-known bandwidth ETA in
   `next_completion_time_ns()`, and stamped onto `Completion.completion_ns`
   from `drained_ns + path_latency_ns` -- never accumulated across
   reallocations.
2. **`next_completion_time_ns()` stays correct.** It's recomputed fresh from
   current state on every call, same as before task 10, just with the fixed
   per-flow latency term added on top.
3. **A drained flow can't be re-delayed.** Once DRAINED, a flow holds no
   link claim at all (excluded from `_active()`), so nothing that arrives
   later can touch it -- it's a plain countdown from a timestamp already
   fixed at the moment it drained.

**What it gets wrong, stated plainly:** this model releases a link to
competing flows the instant a finishing flow's bytes stop moving, treating
its propagation tail as if it no longer occupies anything. A real wire's
last bit is still in flight during that tail; this model shows the link as
fully available to a new arrival during exactly that window. It
understates a finishing flow's true occupancy at the far end of its
lifetime -- the opposite failure mode from Model A's stated one (a link
looking idle at the *front* of a flow's life, before its first bit has
propagated). Neither model is exactly right; this one was chosen because
getting it wrong in this direction doesn't corrupt the bandwidth-sharing
invariants that `test_contention.py`'s allocator tests depend on, and the
alternative (Model A) needs the equivalent complexity at the *front* of a
flow's life instead -- a pending-activation event interleaved with
completions -- for no clear advantage.

## 2. Whether any existing test changed

**One did, `test_gpus_sharing_a_nic_contend` in `tests/test_contention.py`.**
Both numbers, shown in the test's own updated docstring:

- **Before task 10** (latency always zero): contended rate was exactly half
  the solo rate (25 GB/s vs 50 GB/s), so the slowdown was *exactly 2.0*.
  `> 1.5` was headroom under that 2.0, not a physical requirement.
- **After task 10** (real latency): the two flows' path is 4 hops --
  egress + scale_out + scale_out + egress = 2000 + 5000 + 5000 + 2000 =
  14000 ns -- identical for both flows and unaffected by contention (they
  share only the middle scale_out hop). `solo = 400_000/50 + 14000 = 22000`,
  `contended = 400_000/25 + 14000 = 30000`, slowdown = `30000/22000 = 15/11
  ≈ 1.3636`.

The other 3 failures on the first run (`test_intra_domain_transfer_uses_scale_up_only`,
`test_cross_machine_transfer_bottlenecks_on_the_narrow_step`,
`test_report_attributes_bottlenecks_by_class`) were the `bottleneck=None`
bug from S1 -- fixed in the model, not in the tests; their expectations
were never wrong.

`15/11` is not a threshold I picked to make the test pass -- it's the exact
closed-form value, asserted with `pytest.approx`, matching this file's own
stated convention ("every expected value here is closed-form... computed by
hand"). The underlying property the test checks (`rep.per_transfer_ns["x"]
> solo`, "sharing a NIC must slow both") was never violated and needed no
change.

## 3. The new packed-versus-split ratio, and why it moved that way

Task 09's exact scenario (1 GiB KV-sized transfer, `build_node_scale`
defaults: 400 GB/s scale-up, 50 GB/s scale-out):

```
packed: 2,685,292 ns   split: 21,488,837 ns   ratio: 8.002420965764617
```

Before task 10 this was `8.000000` (task 09 report, verified again here).
**It moved up, not down, and that's the correct direction.** Packed crosses
one scale-up hop (936.25 ns latency); split crosses four hops -- egress,
scale_out, scale_out, egress (14000 ns combined). Split's extra latency
adds to its total on top of its bandwidth disadvantage, in the *same*
direction as the 8:1 bandwidth ratio, so the combined ratio exceeds 8.0.

This is the opposite direction from S2's NIC-sharing case, where the two
flows share one identical path and latency -- being equal in both the
numerator and denominator -- damps the ratio toward 1 instead of amplifying
it. The direction depends entirely on whether latency differs between the
two things being compared (task 09/10's packed vs split: different paths,
different latency, ratio moves away from 1) or is common to both (S2's
solo vs contended: same path, same latency, ratio moves toward 1). Both
directions are real and both are now correctly produced by one model --
`tests/test_latency.py::test_latency_changes_split_versus_packed_ratio`
checks the first; `test_contention.py::test_gpus_sharing_a_nic_contend`
(S2) checks the second.

## 4. Whether the bandwidth-latency crossover lands where expected

**Yes, verified by sweep, and precisely enough to name the actual crossover
sizes rather than just "roughly a megabyte":**

```
      1024 B  packed=       940ns  split=     14021ns  ratio=14.9160
     16384 B  packed=       978ns  split=     14328ns  ratio=14.6503
     65536 B  packed=      1101ns  split=     15311ns  ratio=13.9064
    262144 B  packed=      1593ns  split=     19243ns  ratio=12.0797
   1048576 B  packed=      3559ns  split=     34972ns  ratio=9.8264
   4194304 B  packed=     11423ns  split=     97887ns  ratio=8.5693
  16777216 B  packed=     42881ns  split=    349545ns  ratio=8.1515
  67108864 B  packed=    168710ns  split=   1356178ns  ratio=8.0385
```

The crossover (bandwidth term = latency term) is `latency_ns * capacity`:
**~374,500 B (366 KiB) for packed, ~700,000 B (684 KiB) for split** --
both sub-megabyte, both the same order of magnitude as task 09's "under
roughly 1 MB, latency-bound" measurement, and both *below* 1 MB rather than
at it. The two paths don't share one crossover point -- each has its own,
set by its own latency-to-bandwidth ratio, which is worth stating exactly
rather than leaving as "around a megabyte" for both.

The more interesting shape, visible in the sweep: at the smallest sizes the
ratio approaches **~15**, not 8 -- the *latency* ratio (14000/936.25 ≈ 14.95),
because at 1 KB bandwidth is negligible on both sides and the comparison is
almost purely two fixed latencies against each other. Only once payload
size dominates does the ratio settle toward the bandwidth ratio (8.04 at 64
MiB, converging further beyond). This is exactly the two-regime behavior
task 08's small M2N payloads and task 09's large KV payload occupy opposite
ends of, and it is why a model with a fixed crossover was blocking the M2N
predictor in particular.

## 5. Anywhere this specification is wrong

- **§2's description of Model B undersells its real implementation cost.**
  "Simpler, but wrong under contention" reads as if B needs no structural
  change beyond an addition at report time. It does, once you require (as
  §2's own three invariants require) that reallocation timing and report
  timing not be conflated -- a third flow state, not a one-line change. In
  complexity terms B and A converge much closer than the spec's phrasing
  suggests; B still came out ahead because its extra state sits at the
  *end* of a flow's life (a countdown with nothing left to interact with)
  rather than the *start* (where a pending-but-not-yet-active flow has to
  be excluded from allocation while somehow still being scheduled on).
- Everything else held up: the "do not change `physical/topology.py`"
  constraint was never in tension with anything (latency is read, not
  written, and the caller-computes-it design in §3 was exactly right --
  `transfers.py` already had the real `Link` objects, `model.py` never
  needed them). The backwards-compatibility instruction in §3.1 was also
  exactly right: the one break found was a fixture whose margin assumed
  zero latency, not an arithmetic error, and the spec's own instruction
  ("investigate rather than adjusting expected values") is what surfaced
  the real bottleneck-attribution bug in S1/S2 -- investigating the
  *other* three failures, which looked similar at first, is what found it.

M2N predictor work is deliberately not started here, per §3's closing
instruction.
