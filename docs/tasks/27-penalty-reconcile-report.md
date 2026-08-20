# Task 27 — Which is the TP split penalty: 48% or 89%?

Branch: `task-27-penalty-reconcile`, stacked on `task-26-scale`. Paths
confirmed per task 25: working tree at `/work/simulation/dc-sim`,
Frontier at `/work/simulation/Frontier`.

189 tests pass (measurement/investigation task, no new tests), and
`python3 tools/check_import_direction.py` exits 0.

**The short answer: 88.8% is right. The 48% figure — and the entire
table this task's own §1 attributes to task 21 (5.08 ms, 1.27 ms,
12.57 ms, 7.65 ms) — does not appear anywhere in task 20's or task
21's actual reports, or in either commit's own code diff.** Not
mis-derived from a real computation with a subtle unit error; simply
not traceable to anything either report or the code it describes ever
produced.

---

## 1. What Task 21 actually reports (§2.1)

Quoted directly, `docs/tasks/21-collective-patterns-report.md`, §2
("The corrected measurement, against Task 20's"):

> **Tensor-parallel (allreduce) is unchanged — confirmed by re-running
> it, not assumed unaffected.** This task's fix is specific to
> `predict_all_to_all`; `predict_allreduce` was never touched.

with the table immediately following it:

| | packed | split | ratio |
|---|---|---|---|
| Task 20 (`tensor_parallel_communication_time`) | 2.628864 ms | 38.513664 ms | 14.65x |
| Task 21 (same run, re-measured) | 2.628864 ms | 38.513664 ms | 14.65x |

> Bit-identical. Task 20's own inter-token-latency correction — the
> one the spec's S1 cites (+5.126 ms at tp=4, ~88% over packed's 5.803
> ms tpot; task 20's own headline was measured at tp=4/8, not the
> single "89%" figure the spec's S1 states, which this report notes as
> an approximation of task 20's own tp=4 ratio rather than a distinct
> number) — stands exactly as reported.

And, in task 21's own §6 ("Anywhere this specification is wrong"):

> **The cited "89%" inter-token-latency figure** doesn't appear
> verbatim in task 20's own report; the closest real number is task
> 20's tp=4 ratio (+5.126 ms over a 5.803 ms packed baseline, ≈88.3%),
> which this report treats as what was meant rather than a distinct,
> unverifiable claim.

**There is no "before and after" for the TP=4 allreduce/inter-token-latency
figure, because task 21 never changed it.** Packed tpot (5.803319 ms)
and split tpot (10.929719 ms) are the *same* numbers before task 21's
commit and after it — task 21's own table proves this by re-measuring
and getting bit-identical results, not by asserting it. The correction
task 21 actually made — `predict_all_to_all`'s per-pair volume,
`S/n` → `S/n²` — is a different function, pricing a different
collective (expert dispatch / MoE, §3 of that report), never called
by a tensor-parallel allreduce.

**Confirmed a third way, against the code itself, not only the
prose.** `git log --oneline --all -- src/integration/cc_backend/engine_backend.py`
shows exactly three commits touching that file: task 06 (original),
task 20 (rewrite), task 21 (the fix). `git show 48be434` (task 21's
commit) is an 18-line diff entirely inside `predict_all_to_all`'s body
and docstring — `predict_allreduce` does not appear in the diff at
all.

**No figure in this task's own opening table — 5.08 ms, 1.27 ms,
12.57 ms, 7.65 ms, or 48% — appears anywhere in either report.**
Checked directly: `grep -n "5\.08\|1\.27\|12\.57\|7\.65\|48%\|48\."
docs/tasks/20-collective-backend-report.md
docs/tasks/21-collective-patterns-report.md` returns nothing. This
task's own "Task 21" table is not a misreading of a real number; it
does not correspond to any computation either report performed.

## 2. Whether Task 24 ran with corrected collective patterns

**Yes — established two ways, though the question turns out not to
matter for this specific metric.**

1. **Git history.** `task-24-memory-planner` stacks on
   `task-23-memory-tp` on `task-22-which-binds` on `task-21-collective-patterns`
   — task 21's commit (`48be434`) is an ancestor of every commit task
   24's own branch was built from. There is no code state task 24 ran
   against that predates task 21's fix.
2. **Task 24's own tooling.** `tools/run_memory_tp_study.py`'s
   `_build_and_install()` (which `run_memory_planner_study.py`, task
   24's own script, calls directly) reads:
   `install(fabric, placement, deployment, registry, collective=True)`
   — the exact call that registers `EngineCCBackend`, confirmed by
   `grep -n "collective=True" tools/run_memory_tp_study.py`.

**But since task 21 never touched `predict_allreduce`, "ran with the
correction" is not actually a meaningful distinction for the TP=4
allreduce number.** There is no uncorrected version of that function
anywhere in this project's history to have run instead. Task 24's
+88.8% and task 20/21's own +88.3% are two measurements of the *same,
never-modified* code path, at two different points in this project's
history, in two different workload configurations (task 20/21's own
small fixed-request probe vs. task 24's 32-request, admission-controlled
grid) — the small gap between 88.3% and 88.8% is workload variation,
not a correction taking or not taking effect.

## 3. Which figure is right

**88.8% (task 24/25) is right. 48% was never a real measurement of
anything — not a mis-derivation with an identifiable mechanism, a
fabrication with no traceable source.** Confirmed a fourth, independent
way: re-ran task 20/21's own original probe
(`tools/run_collective_backend_study.py`) fresh, today, against the
current codebase (tasks 22-26 accumulated on top, task 21's fix long
in place):

```
tp=4  packed_tp_comm=2.628864ms  split_tp_comm=38.513664ms  ratio=14.65x  tpot_delta=+5.126400ms
[tp4_packed] mean_tpot=5.803319ms   [tp4_split] mean_tpot=10.929719ms
```

Bit-identical to task 20's original 2026 figures, and to task 21's own
re-measurement — `+5.1264 ms` over a `5.803319 ms` packed baseline is
**+88.33%**, matching task 24's own +88.8% (the small residual gap is
task 24's own different, batched workload, not a code difference) and
task 25's own citation of "task 20's own tp=4 ratio."

This lands closest to this task's own second offered outcome (§3:
"88.8% is right and 48% was mis-derived"), with one correction to how
it's stated: the third outcome's own reasoning turns out to be the
correct diagnosis of *why* — **the pattern correction (S/n → S/n²)
never touched allreduce at all, so there was nothing for the TP=4
ratio to move from in the first place.** The document's own §3
reasoning ("the correction scaled packed and split by the same factor
of four... the ratio should not have moved much") is right about the
ratio being stable, but for a stronger reason than it states: not
"scaled by the same factor" but "never touched."

**§2.3's own framing — "Task 20 is the run that produced the +89%
figure Task 21 then corrected" — is itself part of the false premise.**
Task 20 produced +88.3% (not +89%, and not something task 21
"corrected" — task 21's own report explicitly disclaims "89%" as an
approximation of 88.3%, then reaffirms 88.3% stands unchanged). Task
25's citation of task 20's figure as corroborating task 24 is not "two
uncorrected measurements agreeing by coincidence" — it is the same
never-corrected, never-broken number, measured three separate times
(task 20, task 21's re-measurement, this task's fresh re-run today)
and landing within a fraction of a percent of task 24's own figure
every time.

## 4. What the document should say

**There is no "corrected table for §3.8" to substitute in, because the
table this task's own §1 presents as task 21's was never task 21's.**
The document's own §3.8 (if it currently states or implies a 48%
figure for the TP=4 split penalty) should be corrected to read:

| At four-way, split | Value | Source |
|---|---|---|
| Tensor-parallel communication (packed) | 2.628864 ms | task 20 §4, task 21 §2 (re-measured, bit-identical), this task (re-measured again) |
| Tensor-parallel communication (split) | 38.513664 ms | same three sources, bit-identical across all |
| Ratio | 14.65x | same |
| Inter-token latency (packed) | 5.803319 ms | same |
| Inter-token latency (split) | 10.929719 ms | same |
| **Penalty for splitting** | **+88.3%** (task 24's own batched-workload measurement: +88.8%) | task 20, task 21, task 24, task 25, this task |

**§3.10 does not need qualifying in the direction this task's own §3
first bullet anticipates** ("Task 24 ran uncorrected... needs
re-running") — task 24 ran correctly, against the only version of
`predict_allreduce` that has ever existed in this project, and its
number is right. What §3.10 *should* add, if it doesn't already, is
the correction this task makes to the record: the ~48%/89% pair was
never a real before/after of the same measurement, and no rerun of
task 24's grid is needed on this account.

## 5. Anywhere this specification is wrong

- **The entire "Task 21" table in this task's own §1** — packed/split
  tensor-parallel communication of 5.08 ms / 1.27 ms and inter-token
  latency of 12.57 ms / 7.65 ms, with a stated "+48%" — does not
  appear in `docs/tasks/21-collective-patterns-report.md` (or task
  20's) under any search. This is the load-bearing error this task
  exists to catch, and it caught itself: this specification's own
  instruction ("read the primary file, quote it, and say where in it
  the quote appears") is exactly what locates the fabrication, since
  no such location exists.
- **§2.1's own framing — "the closed-form check then found the cost
  path charging every ordered pair of participants for every
  collective, an eightfold over-charge on slow links for a ring"** —
  restates task 21's own §0, itself a correction of an inaccurate
  premise in task 21's *own* spec (task 21's report: this project's
  ring implementation "was never actually committing" that bug; the
  over-charge that was real was `all_to_all`'s per-pair *volume*, an
  8x effect on total EP dispatch bytes at n=8, not a "ring" charging
  ordered pairs). Repeating this description without task 21's own
  correction attached reintroduces the inaccuracy task 21 already
  resolved.
- **§2.3's framing that "Task 20 is the run that produced the +89%
  figure Task 21 then corrected"** is false in both halves: task 20
  produced +88.3%, not +89% (task 21's own §6 disclaims "89%" as an
  approximation), and task 21 did not correct it — it reaffirmed it
  unchanged. Addressed directly in §3 above.
- Otherwise this specification's own structure — read the primary
  files rather than a summary, quote and locate rather than
  paraphrase, treat "both are right about different things" as a
  serious candidate before picking a side, re-run rather than reason
  further once the documentary evidence is ambiguous — is exactly
  right, and following it to the letter is what surfaced the
  fabrication described above.

## What shipped

No new tool. This task reused three already-established measurements
directly — `tools/run_collective_backend_study.py` (tasks 20/21's own
tool, re-run fresh), `tools/run_memory_tp_study.py`'s
`_build_and_install` (confirmed via `grep`, not re-implemented), and
`git log`/`git show` on `src/integration/cc_backend/engine_backend.py`
— plus direct reading of the two primary report files this task's own
acceptance criteria named.

One commit on `task-27-penalty-reconcile`, stacked on
`task-26-scale`; no `upstream/`, `src/engine/`, or predictor changes —
investigation and one confirmatory re-run, nothing implemented.
