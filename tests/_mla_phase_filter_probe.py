"""Task 53: build the real deepseek-v3/mi355x MLA attention predictor,
with or without the phase-filter patch installed, and report each
operator's training-row count -- run as a subprocess (mirroring
`_memory_planner_probe.py`'s own established pattern) so importing this
probe never entangles the pytest process with Frontier's own import graph.

Usage: python3 _mla_phase_filter_probe.py [--patched]
Prints one MLA_PHASE_FILTER_PROBE_RESULT=<json> line.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

sys.path.insert(0, "/work/simulation/Frontier")
sys.path.insert(0, "/work/simulation/dc-sim/src")

logging.disable(logging.INFO)

_RESULT_MARKER = "MLA_PHASE_FILTER_PROBE_RESULT="


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patched", action="store_true")
    args = parser.parse_args()

    if args.patched:
        from integration.execution_time_predictor.mla_phase_filter import (
            install_mla_phase_filter,
        )
        install_mla_phase_filter()

    sys.argv = [
        "frontier.main", "--simulation_mode", "offline", "--sys_arch", "co-location",
        "--cc_backend_config_type", "analytical", "--cluster_config_num_replicas", "1",
        "--replica_config_device", "mi355x", "--replica_config_model_name", "deepseek-v3",
        "--replica_config_attn_tensor_parallel_size", "8",
        "--replica_config_attn_data_parallel_size", "1",
        "--replica_config_moe_tensor_parallel_size", "1",
        "--replica_config_moe_expert_parallel_size", "8",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_block_size", "32",
        "--decode_cuda_graph_mode", "none",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "16",
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "512",
        "--fixed_request_length_generator_config_decode_tokens", "128",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1.0",
        "--metrics_config_output_dir", "/tmp/mla_phase_filter_probe_outputs",
        "--metrics_config_run_id", "mla_phase_filter_probe",
        "--no-metrics_config_write_metrics", "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_utilization_metrics",
        "--no-metrics_config_store_plots", "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]

    from frontier.config import SimulationConfig
    from frontier.execution_time_predictor.execution_time_predictor_registry import (
        ExecutionTimePredictorRegistry,
    )
    from frontier.types import ClusterType

    config = SimulationConfig.create_from_cli_args()
    cluster_config = config.cluster_config
    predictor = ExecutionTimePredictorRegistry.get(
        cluster_config.execution_time_predictor_config.get_type(),
        predictor_config=cluster_config.execution_time_predictor_config,
        replica_config=cluster_config.replica_config,
        replica_scheduler_config=cluster_config.replica_scheduler_config,
        metrics_config=config.metrics_config,
        cluster_config=config.cluster_config,
        model_manager=None,
        cluster_type=ClusterType.MONOLITHIC,
        cc_backend=None,
    )

    ops = [
        "attn_mla_kv_cache_save",
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_prefill",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_decode",
        "attn_mla_v_up_proj",
    ]
    result = {
        op: len(getattr(predictor._models[op], "_frontier_exact_lookup", {}))
        for op in ops
    }
    print(_RESULT_MARKER + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
