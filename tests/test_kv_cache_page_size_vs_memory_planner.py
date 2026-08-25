"""Task 48: `_kv_cache_page_bytes_per_layer` generalizes Frontier's own
`MemoryPlanner._get_kv_cache_memory_per_layer_per_block` -- checked here
against four models, extending
`test_feasible_num_blocks_vs_param_counter.py`'s own three (all
DENSE_KV) with the first LATENT_MLA one this project's own tools have
ever pointed a real-compute check at.

`deepseek-v3` (mi355x-profiled; `data/config/models/deepseek-v3.json`
infers `use_mla=True`) is task 39's own "corrected blind" case: its
own report supplied `runtime_num_kv_heads=1`/`runtime_head_dim=576`
from reading `frontier/attention/families.py`'s resolvers directly, but
no model in the checkout could run the comparison this file now runs
for real. Doing so surfaced a second, independent divergence task 39
never needed to touch: `_KV_CACHE_FACTOR`'s own hardcoded `2` (separate
K and V caches) is also wrong for LATENT_MLA, which stores one
compressed latent (`frontier/attention/families.py`'s own
`kv_factor=1` for `LATENT_MLA_ATTENTION_FAMILY`, vs `kv_factor=2` for
`DENSE_KV_ATTENTION_FAMILY`) -- confirmed directly: before `ModelSpec.kv_factor`
existed, this formula gave 36864 bytes/block at every `attn_tp` for
deepseek-v3, while `MemoryPlanner` gives 18432 -- exactly 2x, exactly
the missing factor. Fixed by adding `kv_factor` as a fourth explicit
override field, alongside `runtime_num_kv_heads`/`runtime_head_dim`,
defaulting to `None` (this module's own `_KV_FACTOR=2`) so all three
already-tested DENSE_KV models are unaffected.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from planner_core import ModelSpec, _kv_cache_page_bytes_per_layer  # noqa: E402

FRONTIER_ROOT = Path("/work/simulation/Frontier")
_PROBE_SCRIPT = str(Path(__file__).resolve().parent / "_memory_planner_probe.py")
_RESULT_MARKER = "MEMORY_PLANNER_PROBE_RESULT="

_FRONTIER_AVAILABLE = FRONTIER_ROOT.is_dir()
pytestmark = pytest.mark.skipif(
    not _FRONTIER_AVAILABLE,
    reason="needs Frontier checked out at /work/simulation/Frontier (ambient PYTHONPATH, "
          "not repo-pinned -- see AGENTS.md/memory)")


def _real_page_bytes(model_name: str, attn_tp: int, total_experts: int, router_topk: int,
                     block_size: int = 16) -> int:
    proc = subprocess.run(
        [sys.executable, _PROBE_SCRIPT, "--model-name", model_name, "--attn-tp", str(attn_tp),
         "--total-experts", str(total_experts), "--router-topk", str(router_topk),
         "--block-size", str(block_size)],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return int(line[len(_RESULT_MARKER):])
    raise RuntimeError(
        f"probe failed for {model_name} tp={attn_tp} (exit {proc.returncode}):\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")


# (model_name, ModelSpec kwargs beyond model_name/total_experts/is_moe/router_topk, tp_values)
MODELS = [
    ("Phi-tiny-MoE-instruct",
     dict(hidden_size=4096, num_attention_heads=16, num_key_value_heads=4,
         num_layers=32, head_dim=128),
     16, 2, (1, 2, 4, 8)),
    ("Llama-3.1-405B-Instruct-FP8",
     dict(hidden_size=16384, num_attention_heads=128, num_key_value_heads=8,
         num_layers=126, head_dim=None),
     1, 1, (1, 2, 4, 8, 16, 32)),
    ("step-moe-noquant-small",
     dict(hidden_size=7168, num_attention_heads=64, num_key_value_heads=1,
         num_layers=31, head_dim=256),
     24, 3, (1, 2, 4, 8)),
    ("deepseek-v3",
     dict(hidden_size=7168, num_attention_heads=128, num_key_value_heads=128,
         num_layers=61, head_dim=None, runtime_num_kv_heads=1, runtime_head_dim=576,
         kv_factor=1),
     256, 8, (1, 2, 4, 8)),
]


@pytest.mark.parametrize("model_name,spec_kwargs,total_experts,router_topk,tp_values", MODELS)
def test_kv_cache_page_bytes_matches_memory_planner_exactly(
        model_name, spec_kwargs, total_experts, router_topk, tp_values):
    model = ModelSpec(model_name, total_experts=total_experts, router_topk=router_topk,
                      is_moe=(total_experts > 1), **spec_kwargs)
    for tp in tp_values:
        formula_bytes = _kv_cache_page_bytes_per_layer(model, tp, block_size=16)
        real_bytes = _real_page_bytes(model_name, tp, total_experts, router_topk, block_size=16)
        assert formula_bytes == real_bytes, (
            f"{model_name} at attn_tp={tp}: formula={formula_bytes} "
            f"!= MemoryPlanner={real_bytes}")


def test_deepseek_v3_kv_factor_matters_by_exactly_2x():
    """Pins the magnitude of the divergence this task found, not just its
    existence: dropping `kv_factor=1` (i.e. leaving deepseek-v3 at this
    module's own DENSE_KV default) must give exactly double the real,
    correct figure -- the single missing factor, not some other bug."""
    with_factor = ModelSpec("deepseek-v3", total_experts=256, router_topk=8, is_moe=True,
                            hidden_size=7168, num_attention_heads=128, num_key_value_heads=128,
                            num_layers=61, head_dim=None, runtime_num_kv_heads=1,
                            runtime_head_dim=576, kv_factor=1)
    without_factor = ModelSpec("deepseek-v3", total_experts=256, router_topk=8, is_moe=True,
                               hidden_size=7168, num_attention_heads=128, num_key_value_heads=128,
                               num_layers=61, head_dim=None, runtime_num_kv_heads=1,
                               runtime_head_dim=576)  # kv_factor left at the DENSE_KV default
    correct = _kv_cache_page_bytes_per_layer(with_factor, attn_tp=8, block_size=16)
    wrong = _kv_cache_page_bytes_per_layer(without_factor, attn_tp=8, block_size=16)
    assert correct == 18432
    assert wrong == 2 * correct
