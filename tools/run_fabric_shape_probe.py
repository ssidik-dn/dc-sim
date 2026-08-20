#!/usr/bin/env python3
"""Task 17 Part B.3: does the placement-penalty conclusion hold in shape on
a fabric this project has never measured on?

Every prior task (09-16) built its fabric with `engine.physical.builders.build_node_scale`
-- one scale-up domain per machine, all machines hanging off one shared leaf
switch. This script reruns task 11's own colocated-vs-split M2N comparison
(same predictor, same deployment shape, reused via import from
`tools/run_m2n_integration.py`, not rewritten) on
`engine.infragraph.blueprints.clos_fat_tree_fabric` (task 02/03's real
two-tier leaf-spine blueprint) instead, at a non-1.0 oversubscription ratio,
and at two placements:

- **colocated**: PREFILL/DECODE_ATTN/DECODE_FFN on the same host (0 switch
  hops).
- **split**: DECODE_FFN moved to a host under a *different leaf* -- the one
  genuinely new case a leaf-spine fabric has that `build_node_scale` never
  did (that blueprint has only one leaf; every "split" placement in tasks
  09-16 was one switch-hop away, never two).

Dummy execution-time mode throughout: this check is about the *network*
model holding shape on new fabric geometry, the same thing task 11
established before task 12 added real compute to look at TPOT composition
specifically -- not a re-run of that composition question.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as every other tool in this project:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_fabric_shape_probe.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

FRONTIER_ROOT = Path("/work/simulation/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m2n_integration import (  # noqa: E402
    _engine_deployment_and_registry, _find_m2n_predictor)
from engine.infragraph.blueprints import clos_fat_tree_fabric  # noqa: E402
from engine.physical.topology import GpuId  # noqa: E402
from engine.placement.placement import explicit  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/fabric_shape_probe_outputs")

SWITCH_RADIX = 8    # -> 8 leaves, 4 spines, 4 hosts/leaf, 32 hosts total
OVERSUBSCRIPTION = 4.0
NUM_REQUESTS = 4
DECODE_TOKENS = 16


def _placements(fabric, deployment):
    prefill_rank = deployment.replicas[0].ranks[0]
    attn_rank = deployment.replicas[1].ranks[0]
    ffn_rank = deployment.replicas[2].ranks[0]
    # host 0 (leaf 0) for prefill/attn in both scenarios; ffn colocated on
    # host 0 vs moved to host 4 -- leaf 1 (hosts_per_leaf=4), two switch
    # hops away (leaf -> spine -> leaf), not reachable in build_node_scale.
    colocated = explicit(deployment, fabric, {
        prefill_rank: GpuId(0, 0), attn_rank: GpuId(0, 1), ffn_rank: GpuId(0, 2)})
    split = explicit(deployment, fabric, {
        prefill_rank: GpuId(0, 0), attn_rank: GpuId(0, 1), ffn_rank: GpuId(4, 0)})
    return colocated, split


def _argv(run_id: str) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters",

        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", "1",
        "--cluster_config_decode_ffn_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_attn_micro_batch_size", "8",

        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", "1",
        "--cluster_config_prefill_replica_config_router_topk", "1",
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_device", "h800",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_total_expert_num", "1",
        "--cluster_config_decode_ffn_replica_config_router_topk", "1",
        "--cluster_config_decode_ffn_replica_config_device", "h800",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",

        "--cc_backend_config_type", "analytical",
        "--m2n_transfer_config_type", "empirical",

        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--replica_config_moe_routing_mode", "simulation",
        "--replica_config_moe_routing_seed", "42",

        "--vllm_v1_scheduler_config_max_tokens_in_batch", "1024",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "64",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", "128",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",

        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "32",
        "--fixed_request_length_generator_config_decode_tokens", str(DECODE_TOKENS),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1.0",

        "--metrics_config_output_dir", str(OUTPUT_DIR),
        "--metrics_config_run_id", run_id,
        "--no-metrics_config_write_metrics",
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_utilization_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",

        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms", "1.0",
    ]


_RESULT_MARKER = "FABRIC_SHAPE_RESULT="


def _run_scenario(label: str) -> None:
    fabric = clos_fat_tree_fabric(switch_radix=SWITCH_RADIX, gpus_per_machine=4,
                                  oversubscription=OVERSUBSCRIPTION)
    deployment, registry = _engine_deployment_and_registry()
    colocated_placement, split_placement = _placements(fabric, deployment)
    placement = colocated_placement if label == "colocated" else split_placement
    install(fabric, placement, deployment, registry)

    sys.argv = _argv(f"fabric_shape_{label}")
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    config = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)
    sim = Simulator(config)
    sim.run()

    requests = sim._all_requests
    m2n_time_s = [r.total_m2n_transfer_time for r in requests]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]
    predictor = _find_m2n_predictor(sim)

    print(_RESULT_MARKER + json.dumps({
        "label": label,
        "mean_m2n_time_s": mean(m2n_time_s),
        "mean_tpot_s": mean(tpot_s) if tpot_s else None,
        "predictor_calls": predictor.calls if predictor else 0,
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(label: str) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--scenario", label],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise RuntimeError(f"scenario {label!r} produced no result "
                       f"(exit code {proc.returncode}); see output above")


def main() -> int:
    results = {}
    for label in ("colocated", "split"):
        r = _run_scenario_in_subprocess(label)
        results[label] = r
        print(f"[{label:<10}] mean_m2n={r['mean_m2n_time_s']*1000:.6f} ms  "
             f"mean_tpot={r['mean_tpot_s']*1000:.6f} ms  calls={r['predictor_calls']}")

    c, s = results["colocated"], results["split"]
    ratio_m2n = s["mean_m2n_time_s"] / c["mean_m2n_time_s"]
    ratio_tpot = s["mean_tpot_s"] / c["mean_tpot_s"]
    print()
    print(f"switch_radix={SWITCH_RADIX} oversubscription={OVERSUBSCRIPTION}:1")
    print(f"M2N transfer time ratio (split/colocated): {ratio_m2n:.4f}x")
    print(f"TPOT ratio (split/colocated):               {ratio_tpot:.4f}x")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("colocated", "split"), default=None)
    args = parser.parse_args()
    if args.scenario:
        _run_scenario(args.scenario)
        raise SystemExit(0)
    raise SystemExit(main())
