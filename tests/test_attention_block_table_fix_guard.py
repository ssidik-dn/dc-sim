"""Task 53 Fix B: `AttentionWrapper._get_input_tensors`
(`frontier/profiling/attention/attention_wrapper.py`) gives every sequence
in a profiled batch the same KV-cache block range -- optimistically fast
cache locality no real serving run has (Task 52's candidate D). This
module's own docstring explains why it is not wired into `install()`
(profiling never calls it) and why its own guard needs `torch`, which this
sandbox does not have (confirmed: `import torch` fails here, no GPU
involved).

The hash check itself does not need `torch` -- it was computed by parsing
`attention_wrapper.py`'s source text directly. The first test below
confirms that computation is still correct against the checked-out file,
without importing anything Frontier-side, so it runs (and is meaningful)
regardless of whether `torch` is installed. Everything that actually
patches the class needs `torch` importable and is skipped otherwise, via
`pytest.importorskip`.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from unittest import mock

import pytest

FRONTIER_ROOT = Path("/work/simulation/Frontier")
_ATTENTION_WRAPPER_PATH = (
    FRONTIER_ROOT / "frontier/profiling/attention/attention_wrapper.py"
)

_FRONTIER_AVAILABLE = FRONTIER_ROOT.is_dir()
pytestmark = pytest.mark.skipif(
    not _FRONTIER_AVAILABLE,
    reason="needs Frontier checked out at /work/simulation/Frontier (ambient PYTHONPATH, "
          "not repo-pinned -- see AGENTS.md/memory)")


def _hash_get_input_tensors_source() -> str:
    """Recomputes the guard's own expected hash by parsing the checked-out
    file's source text directly -- no import, no torch needed. Mirrors
    exactly what `inspect.getsource` would return for the live method."""
    text = _ATTENTION_WRAPPER_PATH.read_text()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_input_tensors":
            func_src = "".join(lines[node.lineno - 1:node.end_lineno])
            return hashlib.sha256(func_src.encode()).hexdigest()
    raise AssertionError("_get_input_tensors not found in attention_wrapper.py")


def test_expected_hash_matches_the_checked_out_file():
    """Confirms the guard's own hardcoded hash is still correct against
    what is actually checked out -- runs without `torch`, since it never
    imports Frontier, only reads the file's text."""
    from integration.profiling import attention_block_table_fix

    assert (
        _hash_get_input_tensors_source()
        == attention_block_table_fix._EXPECTED_SOURCE_HASH
    )


@pytest.fixture
def _isolate():
    """Requested explicitly (not autouse) by every test below that actually
    patches the class -- `test_expected_hash_matches_the_checked_out_file`
    above deliberately does not request it, since that one test must keep
    running (and being meaningful) without `torch` installed. Everything
    that does request it needs `torch` importable (`attention_wrapper.py`
    imports it at module level), hence the `importorskip` here rather than
    at module scope, which would have skipped the whole file, including the
    one test that does not need it."""
    pytest.importorskip("torch", reason="attention_wrapper.py imports torch at module level")
    from integration.profiling import attention_block_table_fix
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    original = AttentionWrapper._get_input_tensors
    attention_block_table_fix._installed = False
    yield
    AttentionWrapper._get_input_tensors = original
    attention_block_table_fix._installed = False


def _make_attention_input(batch_size: int, kv_cache_size: int, is_prefill: bool = False):
    from frontier.profiling.attention.attention_input import AttentionInput

    return AttentionInput(
        batch_size=batch_size,
        kv_cache_size=kv_cache_size,
        is_prefill=is_prefill,
        prefill_chunk_size=0,
    )


def _make_wrapper_stub(block_size: int, max_num_blocks: int):
    """A bare object exposing only what `_get_input_tensors` reads --
    isolates the guard's own change (block_table construction) from
    `_make_qkv_tensors`'s real tensor allocation, which needs a live device
    and is already proven correct."""
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    obj = AttentionWrapper.__new__(AttentionWrapper)
    obj._block_size = block_size
    obj.max_num_blocks = max_num_blocks
    obj.kv_cache = "stub-kv-cache"
    obj._make_qkv_tensors = lambda total_tokens: ("q", "k", "v")
    return obj


def test_unpatched_gives_every_sequence_the_same_block_range(_isolate):
    """The bug Fix B changes -- confirmed present before any patch is
    installed. Two decode sequences (kv_cache_size=31, block_size=32 ->
    1 block each) both get block_table=[0]."""
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    obj = _make_wrapper_stub(block_size=32, max_num_blocks=8)
    seq_metadata_list, *_ = AttentionWrapper._get_input_tensors(
        obj, _make_attention_input(batch_size=2, kv_cache_size=31)
    )
    assert [sm.block_table for sm in seq_metadata_list] == [[0], [0]]


def test_patched_gives_each_sequence_a_distinct_block_range(_isolate):
    from integration.profiling.attention_block_table_fix import (
        install_attention_block_table_fix,
    )
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    install_attention_block_table_fix()
    obj = _make_wrapper_stub(block_size=32, max_num_blocks=8)
    seq_metadata_list, *_ = AttentionWrapper._get_input_tensors(
        obj, _make_attention_input(batch_size=2, kv_cache_size=31)
    )
    assert [sm.block_table for sm in seq_metadata_list] == [[0], [1]]


def test_patched_still_refuses_when_batch_needs_more_blocks_than_available(_isolate):
    """The bounds check moves with the fix (`next_block_index + num_blocks`,
    not just `num_blocks`) -- a batch that used to fit by aliasing blocks
    can legitimately need more distinct blocks than `max_num_blocks` once
    sequences stop sharing them. This is a real, named consequence of the
    fix (see this module's own docstring / task 53's report), not a defect
    in the patch."""
    from integration.profiling.attention_block_table_fix import (
        install_attention_block_table_fix,
    )
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    install_attention_block_table_fix()
    obj = _make_wrapper_stub(block_size=32, max_num_blocks=1)
    with pytest.raises(ValueError, match="exceeds max_num_blocks"):
        AttentionWrapper._get_input_tensors(
            obj, _make_attention_input(batch_size=2, kv_cache_size=31)
        )


def test_install_is_idempotent(_isolate):
    from integration.profiling.attention_block_table_fix import (
        install_attention_block_table_fix,
    )
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper

    install_attention_block_table_fix()
    patched_once = AttentionWrapper._get_input_tensors
    install_attention_block_table_fix()
    assert AttentionWrapper._get_input_tensors is patched_once


def test_source_hash_guard_fires(_isolate):
    from integration.profiling import attention_block_table_fix

    with mock.patch.object(
        attention_block_table_fix, "_EXPECTED_SOURCE_HASH", "not-the-real-hash"
    ):
        with pytest.raises(attention_block_table_fix.AttentionBlockTableFixSourceMismatch):
            attention_block_table_fix.install_attention_block_table_fix()
    attention_block_table_fix.install_attention_block_table_fix()
    from frontier.profiling.attention.attention_wrapper import AttentionWrapper
    assert (
        AttentionWrapper._get_input_tensors
        is attention_block_table_fix._patched_get_input_tensors
    )
