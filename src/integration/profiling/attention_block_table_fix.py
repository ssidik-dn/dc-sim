"""Task 53 Fix B: give each profiled sequence its own, non-overlapping
KV-cache block range.

`AttentionWrapper._get_input_tensors`
(`frontier/profiling/attention/attention_wrapper.py`) assigns every sequence
in a profiled batch the *same* `block_table`
(`block_table=list(range(num_blocks))`, inside the per-sequence loop, at two
sites: the bounds check and the block-table construction). At `batch_size >
1` every sequence therefore aliases the same physical cache blocks -- cache
locality no real serving run has, which can only make profiled attention
optimistically fast (Task 52's candidate D).

A fix for exactly this exists on the unmerged branch Task 51 evaluated
(`origin/task/deepseek-mla-attention-port-block-sweep`, commit `8c87017`) --
but that branch also deletes an `attention.csv` the simulator's resolver
still looks for, unconditionally, regardless of `block_size` (Task 51's own
finding), so it was not merged. This module cherry-picks only the fix
itself -- a distinct, non-overlapping block range per sequence, tracked by
`next_block_index` -- and nothing else from that branch: not the new MLA
wrapper classes, not the deleted/replaced CSVs, not the `AttentionBackend`
enum wiring. Confirmed separable: the diff to `_get_input_tensors` needs no
new import and no helper the rest of the branch adds; the docstring on the
branch's own equivalent change even says so directly ("same pattern as
`_get_true_mixed_input_tensors` below" -- that function already has this
pattern on `main`, unmodified by the branch).

**Why this is not wired into `install()`, unlike every other patch in this
project.** `_get_input_tensors` runs only during a profiling CLI invocation
(`python -m frontier.profiling.attention.main`), never during a simulation
-- `install()` is called before `frontier.main`, and profiling never calls
either. There is no shared call site to hook. Separately, and just as
decisive: `attention_wrapper.py` imports `torch` unconditionally at module
level, which this checkout's own CPU-only sandbox does not have installed
(confirmed: `import torch` fails here, no GPU involved -- it is simply
absent). Importing this module from `install()`'s own module-level imports
would make importing `integration.install` itself fail everywhere `torch`
is unavailable, breaking every one of this project's 254 tests that
transitively import it. This module is therefore free-standing: nothing in
this project's own tooling calls `install_attention_block_table_fix()`
today, since no re-profiling is scheduled (this task's own S2) -- it exists
for a future profiling run to call explicitly, on a host that has `torch`,
before invoking Frontier's own profiling CLI.

**Also why this module's own test is skipped in this sandbox.** The
guarded function it patches cannot be imported without `torch` either; the
accompanying test in `tests/` calls `pytest.importorskip("torch")` before
touching anything in this module, so it is skipped (not failed, not
errored) wherever `torch` is unavailable, and would actually exercise the
guard on a host where it is. The expected source hash below was computed
without importing the target module -- by parsing
`frontier/profiling/attention/attention_wrapper.py`'s own source text
directly (`ast`, no execution) and hashing the exact substring
`inspect.getsource` would return for the live function -- so the guard
itself does not depend on this sandbox having `torch` to have been written
correctly; only *installing* it does.

Guarded by a source hash over `_get_input_tensors`, matching this project's
established pattern (task 20, task 47, and this task's own Fix A).
"""
from __future__ import annotations

import hashlib
import inspect
from math import ceil
from typing import List

# Deferred: this import requires `torch` (transitively, via
# `attention_wrapper.py`'s own module-level import), which this checkout's
# sandbox does not have. Not imported at module level for exactly the
# reason this module's own docstring gives -- only inside the one function
# that actually needs it, so importing *this* module never fails on a host
# without `torch`, only calling `install_attention_block_table_fix()` does.

# Computed by parsing frontier/profiling/attention/attention_wrapper.py's
# own source text directly (no import, no torch needed) and hashing the
# exact lines AttentionWrapper._get_input_tensors spans at the time this
# module was written. A changed hash means the method's own body changed
# upstream -- install_attention_block_table_fix() raises rather than patch
# over an unknown implementation.
_EXPECTED_SOURCE_HASH = "0b797cc7ebd0aa5f6a6f2023c01f731c737723332a27402c07eed1d4e32a4a51"

_installed = False


class AttentionBlockTableFixSourceMismatch(RuntimeError):
    pass


def _patched_get_input_tensors(self, attention_input):
    from frontier.profiling.attention.sequence_proxy import SequenceMetadataProxy

    num_tokens_per_seq = (
        attention_input.prefill_chunk_size if attention_input.is_prefill else 1
    )
    batch_size = attention_input.batch_size
    total_tokens = batch_size * num_tokens_per_seq
    query, key, value = self._make_qkv_tensors(total_tokens)
    # Each sequence gets a distinct, non-overlapping block range (task 53
    # Fix B), rather than every sequence reusing block_table=range(num_blocks).
    seq_metadata_list: List["SequenceMetadataProxy"] = []
    next_block_index = 0
    for _ in range(attention_input.batch_size):
        num_blocks = ceil(
            (num_tokens_per_seq + attention_input.kv_cache_size) / self._block_size
        )
        if next_block_index + num_blocks > self.max_num_blocks:
            raise ValueError(
                "Requested block_table size exceeds max_num_blocks: "
                f"num_blocks={next_block_index + num_blocks} "
                f"max_num_blocks={self.max_num_blocks}"
            )
        seq_metadata = SequenceMetadataProxy(
            is_prompt=attention_input.is_prefill,
            total_len=num_tokens_per_seq + attention_input.kv_cache_size,
            processed_len=attention_input.kv_cache_size,
            block_table=list(
                range(next_block_index, next_block_index + num_blocks)
            ),
        )
        seq_metadata_list.append(seq_metadata)
        next_block_index += num_blocks
    return seq_metadata_list, query, key, value, self.kv_cache


def install_attention_block_table_fix() -> None:
    """Patch `AttentionWrapper._get_input_tensors` so each profiled sequence
    gets its own, non-overlapping KV-cache block range. Safe to call more
    than once (idempotent). Requires `torch` to be importable (this
    function imports `frontier.profiling.attention.attention_wrapper`,
    which imports `torch` at module level) -- raises `ModuleNotFoundError`
    on a host without it, same as importing that module directly would.

    Raises `AttentionBlockTableFixSourceMismatch` if `_get_input_tensors`'s
    source no longer matches what this module was written against.
    """
    global _installed
    if _installed:
        return
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    current_hash = hashlib.sha256(
        inspect.getsource(AttentionWrapper._get_input_tensors).encode()
    ).hexdigest()
    if current_hash != _EXPECTED_SOURCE_HASH:
        raise AttentionBlockTableFixSourceMismatch(
            f"AttentionWrapper._get_input_tensors's source has changed (hash "
            f"{current_hash} != expected {_EXPECTED_SOURCE_HASH}). Refusing "
            f"to install the block-table patch over an implementation this "
            f"project hasn't reviewed -- update _EXPECTED_SOURCE_HASH in "
            f"{__name__} only after confirming the per-sequence loop being "
            f"replaced is still the same one.")
    AttentionWrapper._get_input_tensors = _patched_get_input_tensors
    _installed = True
