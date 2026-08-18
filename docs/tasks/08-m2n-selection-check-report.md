# Task 08 report — Confirm the M2N path is selectable

Branch: `task-08-m2n-selection-check` (not merged to main).

`python3 -m pytest -q` (134 passed, unchanged) and
`python3 tools/check_import_direction.py` pass unchanged. The diff is one
file, `tools/probe_m2n_selection.py`, plus this report; nothing under
`upstream/` or `src/engine/` was touched. This confirmed as expected, in
well under the ~80%-reuse estimate from task 07 — the KV probe's structure
carried over directly, with only the module-level-class change (deliberate,
not incidental) and the substitutions the task named.

## 1. The answer

**Open**, the same way the KV cache transfer path is. A
`BaseM2NTransferPredictor` subclass defined outside `upstream/`, registered
under the previously-unused `M2NTransferType.EMPIRICAL`, is selected and its
`get_transfer_time` called by a real `pd-af-disaggregation` run, purely
through `--m2n_transfer_config_type empirical` — no upstream edit, no
bypass.

## 2. The probe output

```
sentinel calls: 192
sentinel last activation_size_bytes: 1
sentinel layer_ids seen: [0, 1, 2, ..., 31]
sentinel afd_stage_idx seen: [0]
sentinel pipeline_stages seen: ['attn_to_ffn', 'ffn_to_attn']

ANSWER: OPEN -- the sentinel predictor's get_transfer_time was called by a
real Frontier pd-af-disaggregation run selected purely via
--m2n_transfer_config_type empirical.
```

192 calls, exactly 6 per layer across all 32 layers (`meta-llama/Llama-2-7b-hf`,
32 hidden layers), split evenly 96/96 between `attn_to_ffn` and `ffn_to_attn`.
That factors cleanly as **32 layers × 2 directions × 3 round trips**. The run
requested 4 decode tokens per request (2 requests, batched together under
`decode_attn_af_pipeline_num_micro_batch=1`, so they don't multiply the
count); only 3 of those 4 decode steps produced an explicit
`[TOKEN-ROLLOUT]` log line before completion, consistent with the 4th
(final) token not needing an FFN→ATTN return trip — there is no subsequent
attention round to feed it back into. 3 round trips × 2 directions × 32
layers = 192 matches exactly, so the count is explained, not just observed.
Nothing here needed to be taken on faith.

## 3. Whether the per-layer fields arrived populated

Yes, but not the way the spec's phrasing implied. **`M2NTransferInfo` is not
passed to the predictor at all** — it's a `frontier.entities` dataclass built
by the *caller* (`frontier/events/cluster_batch_end_event.py`), strictly
*after* `get_transfer_info()` returns, for metrics/event bookkeeping. It
carries `layer_id`, `afd_stage_idx`, and `pipeline_stage` as fields of that
downstream record, not as arguments to `get_transfer_time`.

What a predictor actually receives is `(source_cluster_type,
target_cluster_type, batch, activation_size_bytes)`. `layer_id` and
`pipeline_stage` are not present as such — but everything needed to derive
them is, on `batch` and on the two cluster-type arguments:

- **`afd_stage_idx`** — a direct attribute on `batch`, always `0` in this
  run (single AF micro-batch, EP=1, dense — no expert-parallel AF staging
  to distinguish).
- **`layer_id`** — not an attribute; derived the same way Frontier's own
  caller derives it (`ClusterBatchEndEvent._get_current_layer_id_from_batch`):
  the first non-completed request's `completed_layer_count`. The sentinel
  replicates that exact logic and saw every value 0–31, confirming it's
  genuinely available at the point a real predictor would need it, not just
  attached afterward for logging.
- **`pipeline_stage`** — not present either; derived trivially from
  `source_cluster_type` (`DECODE_ATTN` → `"attn_to_ffn"`, else
  `"ffn_to_attn"`), matching `M2NTransferInfo.__post_init__`'s own default.
  Both values appeared, evenly split.

So: the fields are not carried *into* the predictor as a struct, but every
value they end up holding is reconstructible from what the predictor does
receive. A real predictor that wants to price a transfer per-layer or
per-stage has what it needs.

## 4. What differed from the KV path, beyond the named substitutions

- **Model-name resolution is cwd-sensitive for Frontier's bundled model
  JSONs.** The shipped `pd-af-disaggregation` example defaults to
  `MODEL_NAME=llama2_7b_dense_example`, a name that isn't a recognized
  preset — Frontier falls through to reading
  `data/config/models/llama2_7b_dense_example.json`, resolved *relative to
  the process's cwd*, not to the Frontier repo root. Running the probe from
  `dc-sim` (as documented) raised
  `ValueError: [BaseModelConfig] Unknown model 'llama2_7b_dense_example' and
  no JSON found at data/config/models/llama2_7b_dense_example.json`. Fixed
  by using `meta-llama/Llama-2-7b-hf`, the same generic HF name task 07's KV
  probe used — it resolves through a built-in architecture fallback
  ("Model architecture profile fallback selected generic") with no file
  lookup, and needs no `cd`. Not a gate, just a portability trap worth
  flagging for whoever writes the real M2N example config later — the
  shipped shell scripts get away with it because they `cd` to
  `$REPO_ROOT` first.
- **Everything else was exactly the named substitution set**: `sys_arch`,
  three pool configs instead of two, `M2NTransferType`/`BaseM2NTransferConfig`/
  `BaseM2NTransferPredictor`/`M2NTransferPredictorRegistry` in place of their
  KV equivalents, and the module-level-class fix already applied
  pre-emptively per the task's instruction (no rediscovery of the
  weak-reference trap needed).
- Call volume is real and structured (32 × 2 × 3), not a token gesture —
  this path fires far more often per run than KV cache transfer (2 calls in
  task 07's probe), which matters for anyone designing a real predictor's
  performance budget.

## 5. Where the specification is wrong

- **§3's `M2NTransferInfo` framing.** The spec asks whether "the sentinel's
  `M2NTransferInfo` carries `layer_id`, `afd_stage_idx`, and
  `pipeline_stage`" — phrased as if the predictor receives an
  `M2NTransferInfo` object. It doesn't; that object is constructed by the
  caller from the predictor's *return value*, not passed to it. The
  underlying question (do these attribution fields actually arrive
  populated at the point a predictor could use them) is answered yes in §3
  above, but the mechanism is different from what the wording suggests.
- Nothing else in §1–§4 needed correcting — the "same shape, low-cost probe"
  premise from the task 07 report held exactly, including the reused
  weak-reference-avoidance instruction, which cost nothing extra to follow.

With both KV cache transfer and M2N confirmed open, the collectives path
(task 06) remains the one closed extension point among the three
investigated so far. The next task can build a real predictor on either
open path without first re-litigating whether the seam exists.
