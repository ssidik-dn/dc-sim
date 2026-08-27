"""Stage 2 Gate C.1: exact `effective_tokens` key-space derivation and
pre-execution coverage checking for Qwen3-0.6B's DENSE_KV single-feature
`linear_op` operators (`attn_pre_proj`, `attn_post_proj`, `attn_rope`,
`input_layernorm`, `post_attention_layernorm`, `emb`, `mlp_up_proj`,
`mlp_down_proj`, `mlp_act`).

These operators are served by an exact dictionary lookup on
`effective_tokens` (`sklearn_execution_time_predictor.py`,
`self._predictions[op][(effective_tokens,)]`) -- a missing key raises
`KeyError`, unlike the multi-feature attention operators
(`attn_prefill`/`attn_decode`/`attn_kv_cache_save`), which fall back to
a trained model's own interpolation/extrapolation. Sparse "bracket the
envelope with margin" grid design (safe for the multi-feature family) is
therefore NOT safe here: every `effective_tokens` value the real
scheduler can actually request must be a literal profiled key.

Real mechanism, traced directly (`frontier/entities/batch.py`,
`Batch.get_effective_total_tokens_for_compute`/`get_effective_total_tokens_rounded`):
despite its name, `get_effective_total_tokens_rounded` no longer applies
fixed multiple-of-8 rounding ("For vLLM V1 eager-path parity, this
helper no longer applies additional fixed multiple-of-8 rounding" --
the method's own docstring). Under this project's own real Gate C
config (`tools/planner.py`'s `_argv()`: `af_pipeline_num_micro_batch=1`
for both `decode_attn`/`decode_ffn`, no `--use_cuda_graph` flag ever
set, defaulting `False`), `AFDStageMetadata`'s own CUDA-Graph/stage
padding (`apply_padding`) is a no-op -- `effective_tokens` is the raw,
unpadded token count for the batch actually scheduled in that step, no
rounding, no stage-splitting.

Two, and only two, real token-count shapes arise for Qwen3-0.6B's Gate C
workload:

- **Prefill-shaped** (`PoolKind.PREFILL`, always `tp=1` in every real
  `Replica(PoolKind.PREFILL, ...)` construction in this project --
  `tools/planner.py`/`tools/planner_core.py`, confirmed by reading, not
  assumed): every real Gate C request has exactly `workload.prefill_tokens`
  prompt tokens (non-chunked), and PD-disaggregation means the PREFILL
  pool never processes a decode token -- so a prefill batch of `k`
  concurrently-admitted requests has `effective_tokens = k * prefill_tokens`
  exactly, for `k` from 1 up to the tighter of two real, scheduler-derived
  bounds: the token budget (`--vllm_v1_scheduler_config_max_tokens_in_batch`,
  `4096` in this project's own `_argv()`) divided by `prefill_tokens`, and
  the workload's own total request count (`num_requests`) -- concurrent
  prefill admission can never exceed either.
- **Decode-shaped** (`PoolKind.DECODE_ATTN` -- the only pool whose own
  `tp` actually varies across Gate C's real candidates;
  `PoolKind.DECODE_FFN` is likewise always `tp=1`, same as PREFILL):
  each concurrently-decoding real request contributes exactly one token
  per step (no speculative decoding in this workload), so
  `effective_tokens` equals the real decode batch size directly. No
  scheduler-level admission cap narrower than the workload's own total
  request count was found in this project's config schema (only the
  same token budget, which admits far more than `num_requests` decode
  tokens at once and is therefore not the binding constraint) -- the
  safe mathematical upper bound is every integer from 1 up to
  `num_requests`, not a sparse sample of it.
"""
from __future__ import annotations

import csv
from typing import Dict, FrozenSet, Iterable, Set

from planner_core import Workload


def derive_prefill_effective_tokens(
    workload: Workload, max_tokens_in_batch: int
) -> FrozenSet[int]:
    """Exact multiples of `workload.prefill_tokens`, from 1 concurrently-
    admitted request up to `min(max_tokens_in_batch // prefill_tokens,
    num_requests)` -- both bounds are real scheduler/workload constraints,
    not guessed margins."""
    if workload.prefill_tokens <= 0:
        raise ValueError(f"prefill_tokens must be positive, got {workload.prefill_tokens}")
    max_concurrent_by_token_budget = max_tokens_in_batch // workload.prefill_tokens
    max_concurrent = min(max_concurrent_by_token_budget, workload.num_requests)
    if max_concurrent < 1:
        raise ValueError(
            f"max_tokens_in_batch={max_tokens_in_batch} cannot admit even one "
            f"request of prefill_tokens={workload.prefill_tokens}"
        )
    return frozenset(workload.prefill_tokens * k for k in range(1, max_concurrent + 1))


def derive_decode_effective_tokens(workload: Workload) -> FrozenSet[int]:
    """Every integer concurrent-decode-request count from 1 to
    `workload.num_requests` -- the total request count is the only real,
    code-derivable upper bound on decode concurrency found in this
    project's scheduler config (no separate `max_num_seqs`-style cap
    exists); every integer in between is a legally reachable batch size
    under Poisson arrivals, not merely powers of two."""
    if workload.num_requests < 1:
        raise ValueError(f"num_requests must be positive, got {workload.num_requests}")
    return frozenset(range(1, workload.num_requests + 1))


def derive_linear_op_required_points(
    workload: Workload,
    attn_tp_values: Iterable[int],
    max_tokens_in_batch: int,
) -> Dict[int, FrozenSet[int]]:
    """Required `{tp: effective_tokens_keys}` for every DENSE_KV
    single-feature `linear_op` operator, given Gate C's own real
    `attn_tp_values` (the union of TP degrees its own candidates use).

    `tp=1` always needs both prefill-shaped and decode-shaped keys
    (PREFILL and DECODE_FFN are always `tp=1`, and `DECODE_ATTN`'s own
    `tp=1` candidate needs decode-shaped keys too). Every other real
    `tp` value needs decode-shaped keys only -- `DECODE_ATTN` is the
    only pool whose own `tp` varies, and it never runs prefill-shaped
    batches under PD-disaggregation.
    """
    prefill_keys = derive_prefill_effective_tokens(workload, max_tokens_in_batch)
    decode_keys = derive_decode_effective_tokens(workload)
    tp_values = sorted(set(attn_tp_values) | {1})
    required: Dict[int, FrozenSet[int]] = {}
    for tp in tp_values:
        if tp == 1:
            required[tp] = frozenset(prefill_keys | decode_keys)
        else:
            required[tp] = frozenset(decode_keys)
    return required


def missing_keys(required: FrozenSet[int], profiled: FrozenSet[int]) -> FrozenSet[int]:
    """Required keys with no matching profiled row -- each one is a real,
    would-be `KeyError` if Gate C ever requests it."""
    return frozenset(required - profiled)


def unused_keys(required: FrozenSet[int], profiled: FrozenSet[int]) -> FrozenSet[int]:
    """Profiled keys no Gate C query can ever land on -- wasted profiling
    rows, not a correctness problem, but worth trimming."""
    return frozenset(profiled - required)


def read_profiled_num_tokens(csv_path: str, column: str = "num_tokens") -> FrozenSet[int]:
    """Read the real, already-collected `linear_op.csv`'s own `num_tokens`
    column values -- the actual profiled key set, from the real file, not
    from whatever grid was intended."""
    values: Set[int] = set()
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path!r} has no column {column!r}")
        for row in reader:
            values.add(int(float(row[column])))
    return frozenset(values)


def verify_gate_c_linear_op_coverage(
    workload: Workload,
    attn_tp_values: Iterable[int],
    max_tokens_in_batch: int,
    profiled_by_tp: Dict[int, FrozenSet[int]],
) -> Dict[int, FrozenSet[int]]:
    """The pre-execution gate: for every `tp` Gate C actually needs,
    return the missing keys (empty per `tp` means fully covered).
    Raises `KeyError` in the same shape a real, uncaught profiling gap
    would, naming the exact `tp`/keys, if any `tp` Gate C needs has no
    profiled data recorded for it at all (a caller mistake -- pass an
    empty `frozenset()` explicitly for a `tp` with real-but-empty
    coverage, not an absent dict entry).
    """
    required = derive_linear_op_required_points(workload, attn_tp_values, max_tokens_in_batch)
    result: Dict[int, FrozenSet[int]] = {}
    for tp, required_keys in required.items():
        if tp not in profiled_by_tp:
            raise KeyError(f"no profiled data recorded for tp={tp} at all")
        result[tp] = missing_keys(required_keys, profiled_by_tp[tp])
    return result
