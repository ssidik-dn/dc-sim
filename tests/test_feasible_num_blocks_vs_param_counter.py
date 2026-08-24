"""Task 38: `feasible_num_blocks` (via `attn_param_mem_bytes`) generalizes
Frontier's own `ParamCounter.get_num_parameters_per_device()` formula for
DECODE_ATTN's parameter memory. This is now load-bearing -- every
candidate any `plan()` call evaluates, for any model, is filtered
through it first. Checked here against three structurally different
models, at several tensor-parallel degrees each, so a future change to
either side cannot silently diverge without a test failing.

Three models, chosen for genuinely different `head_dim`/GQA structure
(task 38's own known trap -- "a default that happens to be right is
still untested"):

- Phi-tiny-MoE-instruct: declares an explicit `head_dim=128` that
  *differs* from the naive `hidden_size // num_attention_heads = 256`
  -- the override case task 36's own bug was in.
- Llama-3.1-405B-Instruct-FP8: declares no `head_dim` at all (falls
  back to `hidden_size // num_attention_heads = 128`, which happens to
  be correct for this architecture) -- the no-override case.
- step-moe-noquant-small: declares an explicit `head_dim=256` that
  differs from the naive default (`7168 // 64 = 112`) in the *opposite*
  direction from Phi-tiny-MoE-instruct's own override, and has the most
  extreme grouped-query ratio of any model in this checkout
  (`num_key_value_heads=1` -- the KV term never shrinks with `attn_tp`
  at all, a regime neither of the other two models exercises).

Delegates the real-Frontier half to `tests/_param_counter_probe.py`,
run as a subprocess with `cwd` set to Frontier's own root -- like every
other real-compute tool in this project, because
`BaseModelConfig.create_from_name` resolves
`data/config/models/<name>.json` via a path relative to the process's
own cwd, and an in-process `os.chdir()` would leak into every other test
file sharing this pytest session.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from planner_core import ModelSpec, attn_param_mem_bytes  # noqa: E402

FRONTIER_ROOT = Path("/work/simulation/Frontier")
_PROBE_SCRIPT = str(Path(__file__).resolve().parent / "_param_counter_probe.py")
_RESULT_MARKER = "PARAM_COUNTER_PROBE_RESULT="

_FRONTIER_AVAILABLE = FRONTIER_ROOT.is_dir()
pytestmark = pytest.mark.skipif(
    not _FRONTIER_AVAILABLE,
    reason="needs Frontier checked out at /work/simulation/Frontier (ambient PYTHONPATH, "
          "not repo-pinned -- see AGENTS.md/memory)")


def _real_param_mem_bytes(model_name: str, attn_tp: int, total_experts: int, router_topk: int) -> int:
    proc = subprocess.run(
        [sys.executable, _PROBE_SCRIPT, "--model-name", model_name, "--attn-tp", str(attn_tp),
         "--total-experts", str(total_experts), "--router-topk", str(router_topk)],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return int(line[len(_RESULT_MARKER):])
    raise RuntimeError(
        f"probe failed for {model_name} tp={attn_tp} (exit {proc.returncode}):\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")


MODELS = [
    # (model_name, hidden_size, num_attention_heads, num_key_value_heads,
    #  num_layers, head_dim, total_experts, router_topk, tp_values)
    ("Phi-tiny-MoE-instruct", 4096, 16, 4, 32, 128, 16, 2, (1, 2, 4, 8)),
    ("Llama-3.1-405B-Instruct-FP8", 16384, 128, 8, 126, None, 1, 1, (1, 2, 4, 8, 16, 32)),
    ("step-moe-noquant-small", 7168, 64, 1, 31, 256, 24, 3, (1, 2, 4, 8)),
]


@pytest.mark.parametrize("model_name,hidden_size,num_attention_heads,num_key_value_heads,"
                        "num_layers,head_dim,total_experts,router_topk,tp_values", MODELS)
def test_attn_param_mem_bytes_matches_param_counter_exactly(
        model_name, hidden_size, num_attention_heads, num_key_value_heads,
        num_layers, head_dim, total_experts, router_topk, tp_values):
    model = ModelSpec(model_name, total_experts=total_experts, router_topk=router_topk,
                      is_moe=(total_experts > 1), hidden_size=hidden_size,
                      num_attention_heads=num_attention_heads,
                      num_key_value_heads=num_key_value_heads, num_layers=num_layers,
                      head_dim=head_dim)
    for tp in tp_values:
        formula_bytes = attn_param_mem_bytes(model, tp)
        real_bytes = _real_param_mem_bytes(model_name, tp, total_experts, router_topk)
        assert formula_bytes == real_bytes, (
            f"{model_name} at attn_tp={tp}: formula={formula_bytes} "
            f"!= ParamCounter={real_bytes}")
