# Task 43B — Device generalisation

Branch: `task-43b-device`, branched from `task-43-consolidation`'s tip.
Paths per Task 25: working tree at `/work/simulation/dc-sim`, Frontier
at `/work/simulation/Frontier`. 226 tests pass, unchanged, and
`python3 tools/check_import_direction.py` exits 0. Measurement only.

**The short answer, before anything else: the literal question this
task asks — the same model, real end-to-end, on both devices — cannot
be answered for any model in this checkout, verified directly rather
than assumed from Task 35's own inventory table.** What follows is what
that verification found, and what could still be reported honestly once
it did.

---

## 0. Screening (§2's own instruction) — and what it actually found

Task 35's own inventory lists `Qwen3-30B-A3B-tiny` as profiled on
*both* `h800` and `rtx_pro_6000` — the only model with that property,
and therefore the only candidate for a genuine same-model,
cross-device comparison. Task 38 §5's own trap ("a model can be listed
for a device and still fail mid-run because its profiling data omits
operators its architecture needs") is exactly what this section found,
in a more severe form than Task 38 itself encountered:

- **`attn_tp` coverage.** `h800`'s own `linear_op.csv` for this model
  covers `num_tensor_parallel_workers ∈ {1, 2, 4, 8}` (32 rows).
  `rtx_pro_6000`'s own file covers **`{1}` only, and only 3 rows total.**
  A real run at `attn_tp=4` (Task 43A's own TP-split degree) fails
  immediately: `"No data matches the filtering criteria... Required
  tensor_parallel_size: 4, Available: [1]"`.
- **Even at `attn_tp=1`, it still fails.** `FileNotFoundError:
  ...rtx_pro_6000/Qwen3-30B-A3B-tiny/linear_op_kernel_only.csv not
  found`. This file does not exist in this model's own `rtx_pro_6000`
  directory at all.

**This second failure is not "the device lacks kernel-only profiles in
general" — checked directly, not assumed.** `qwen2_dense_test` (the
*other* model profiled on `rtx_pro_6000`, with no `h800` companion)
has `linear_op_kernel_only.csv`, `attention_kernel_only.csv`, and
`attention_combined_kernel_only.csv` in its own `rtx_pro_6000`
directory — more kernel-only files than even `Qwen3-30B-A3B-tiny`'s
own `h800` directory has. **The gap is specific to this one model's
own `rtx_pro_6000` profile** — 3 rows of `CUDA_EVENT` data (against
32 for the same model on `h800`) and no kernel-only fallback data —
not a property of the device's own profiling methodology, which
`qwen2_dense_test` shows is capable of producing a complete profile.

**Consequence, stated plainly**: `Qwen3-30B-A3B-tiny` is the only model
listed for both devices, and its own `rtx_pro_6000` profile cannot
complete a real run at any tensor-parallel degree, `1` included.
`qwen2_dense_test` runs cleanly on `rtx_pro_6000` but has no `h800`
profile to compare against. **No model in this checkout supports the
same-model, cross-device comparison this task's own §2 asks for.**
This is a real, verified finding about what this checkout's own
profiling data can currently support — not a configuration mistake,
and not something worked around by relaxing to dummy mode (this
project's own standing prohibition) or by any other substitute that
would misrepresent what was actually measured.

## 1. What was still measured, and why it is not the same thing

Two things, kept distinct rather than presented as if either answered
the original question:

**(a) The same model (`Qwen3-30B-A3B-tiny`), on `h800` only** — a new
baseline this checkout has never measured, at `attn_tp=1` (pool
separation) and `attn_tp=4` (TP-split), on Task 43A's own F1
(two-tier), F3 (three-tier), and F4 (three-tier, oversubscribed 4:1)
fabrics. This checks whether Task 43A's own *qualitative* pattern
(three-tier costs more than two-tier; oversubscription doesn't move a
real per-token run) generalises to a *third model*, still on the one
device Task 43A used. It is a model-generalisation check, not a
device-generalisation one.

**(b) A different, working model (`qwen2_dense_test`) on `rtx_pro_6000`
alone** — the closest thing to a device check this checkout's own data
supports. Not a cross-device comparison (there is no `h800` figure for
this model to compare against) — a check of whether the same
*mechanism* (dividing costs more on the deeper fabric; oversubscription
doesn't move a real, small-payload run) still shows up on the second
device at all, with a model that can actually complete a run there.

Both are reported in full, and both are clearly marked, throughout, for
what they are — neither substitutes for the comparison §0 found
unavailable.

## 2. The selected points

Per this task's own §2 ("three or four points... to span the range Task
43A found"): F1 (smallest network-share fabric in Task 43A's own
table), F3 (largest), and F4 (three-tier, oversubscription 4:1 — the
setting Task 43A's own §3.1 built direct contention evidence for). All
three, at both `attn_tp=1` (pool separation — the conclusion that
survives on every device this checkout can actually run) and, where
the profile allowed it, `attn_tp=4` (TP-split).

## 3. The values, in Task 43A's own §1.5 primary metric

**(a) `Qwen3-30B-A3B-tiny`, `h800` only:**

| conclusion | F1 | F3 | F4 (os 4:1) vs. F3 |
|---|---|---|---|
| TP-split (`attn_tp=4`) | +34.9% | +66.2% | +0.35% |
| Pool separation (`attn_tp=1`) | +11.8% | +23.6% | +0.30% |

**(b) `qwen2_dense_test`, `rtx_pro_6000` only (not a cross-device pair — §1):**

| conclusion | F1 | F3 | F4 (os 4:1) vs. F3 |
|---|---|---|---|
| Pool separation (`attn_tp=1`) | +31.2% | +63.2% | +0.24% |

**Every one of these numbers is a *new* data point for this checkout —
none of these three model/device/fabric combinations had been run
before this task.** All single-seed (no CI); every effect above is
either far above the "a few percent" threshold this line of work
reserves seeding for (Task 42/43A's own convention), or, for the
oversubscription row, small enough that Task 43A's own seeded check on
a *different* model already established the mechanism (its §3, direct
`engine.network.transfers` instrumentation) rather than asking this
task's own single unseeded point to re-prove it.

## 4. Whether anything moved

**Within `h800` (model generalisation, (a) above): no.** The same
qualitative pattern Task 43A found with `Phi-tiny-MoE-instruct` —
three-tier costs more than two-tier to divide a group across, and
4:1 oversubscription does not move a real per-token run — reproduces
with a third, architecturally different model (MoE, but 8 layers,
`hidden_size=2048`, against Phi's 32 layers/4096). **The F3-to-F1
*ratio* is the sharpest confirmation**: Task 43A's own pool-separation
margins (16.7% / 33.4%) gave a ratio of 2.00x; this task's own
`Qwen3-30B-A3B-tiny` pair (11.8% / 23.6%) gives **1.99x**; the
`rtx_pro_6000` pair with a *third* model (b) gives **2.03x**. Three
models, two devices, the same ~2x structural ratio between the two
fabrics' own cost to divide a group — strong evidence that this ratio
is a property of the *fabric's own hop-count structure* (one spine hop
vs. one core hop), not of any one model or device, exactly the
mechanism Task 43A's own §2 (TP-split) and §5 (pool separation) already
attributed it to.

**Across devices (the comparison §0 found unavailable): unknown, not
"no."** This is not the null result this task's own §1 frames as the
expected, informative outcome — that would require the comparison to
have actually run. It did not. Reporting "device-general" would be
overclaiming from (b) alone, since (b) uses a different model with no
`h800` figure to compare against; reporting "unknown" is the honest
answer this checkout's own data supports.

## 5. Whether compute times differ enough to matter (§3's own trap)

**Checked in the one place this task's own data allows: the
oversubscription row (F4 vs. F3), the smallest effect measured, where
a share-vs-delta divergence would show up most easily if it exists at
all.** `qwen2_dense_test` on `rtx_pro_6000` (0.24%) and
`Qwen3-30B-A3B-tiny` on `h800` (0.30% for pool separation, 0.35% for
TP-split) land within a similar small range of each other — no sign
that a different device's own compute time inflated or shrank this
share by pulling the denominator around, in what little evidence this
task's own data provides. **This is weak evidence, stated as such**: it
compares two different models on two different devices, exactly the
confound §1 already flagged, so it cannot cleanly separate "device
changed the denominator" from "model changed the denominator." The
*absolute* network delta (not the share) was not separately isolated
here — Task 43A's own §3 already did that decomposition once (isolated
vs. loaded payload tests), and this task did not repeat it, since doing
so meaningfully needs the same-model pair §0 found unavailable.

## 6. Anywhere this specification is wrong

**Its own premise that a second full-feature-profiled device exists to
compare against is accurate** (Task 35's own inventory, checked, not
just cited) — but the premise this task's own §2 builds on top of it
("Same model, workload and arrival regime as 43A, so the device is the
only thing that changes") assumes that model's own profile is complete
enough to run, which §0 found false, at the file level, for the one
model that makes the premise possible at all. This is not a citation
that doesn't match its source (this project's own recurring pattern
in six, now seven, prior tasks) — it is a real gap in what the
checkout's own data can support, one level deeper than the
directory-existence check Task 35 itself performed and correctly
scoped ("every model checked, on both devices" refers to files
existing, not to what training or inference against them can actually
complete — Task 35 never claimed otherwise).

**One instruction is worth flagging as harder to satisfy than stated**:
"Confirm the model is profiled for this device before running
anything" (§2) reads as a single check; it took two (tp-coverage, then
a second, unrelated failure at `attn_tp=1` once the first was found)
to actually establish the model was unusable, and a third
(`qwen2_dense_test`'s own directory) to confirm the second failure was
model-specific rather than device-wide. "Confirm... before running
anything" is the right instruction; it does not by itself say how many
layers of confirmation a real profiling directory might need, and this
task needed more than one.

**Otherwise nothing else checked was wrong** — the reasoning in this
task's own §1 (the cost model prices bytes, not device identity; a null
result would be expected and useful) is sound and is exactly why (a)'s
own model-generalisation result, though not the device result this
task set out to get, is still worth having: it is corroborating
evidence for the same claim, from the one axis this checkout's own data
could actually support.

## What shipped

- `docs/tasks/43b-device-report.md`, this report. No source, tool, or
  test file changed — every fabric and model reused from Task 43A's own
  construction and Frontier's own existing model configs
  (`Qwen3-30B-A3B-tiny.json`, `qwen2_dense_test.json`); no new machinery.

One commit on `task-43b-device`, stacked on `task-43-consolidation`.
