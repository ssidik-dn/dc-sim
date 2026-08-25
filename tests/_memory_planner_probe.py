#!/usr/bin/env python3
"""Task 48: like `_param_counter_probe.py`, but for the KV-cache page-size
side of the memory formula (`MemoryPlanner._get_kv_cache_memory_per_layer_per_block`)
-- Task 39's own fix (runtime-resolved `num_kv_heads`/`head_dim` for
LATENT_MLA) computed but never run against Frontier's own accounting, for
lack of a profiled MLA model. `deepseek-v3` (mi355x-profiled) is the first.
"""
import argparse
import sys

_RESULT_MARKER = "MEMORY_PLANNER_PROBE_RESULT="


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--attn-tp", type=int, required=True)
    parser.add_argument("--total-experts", type=int, default=1)
    parser.add_argument("--router-topk", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    args, _ = parser.parse_known_args()

    sys.argv = [
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
        "--cluster_config_prefill_replica_config_total_expert_num", str(args.total_experts),
        "--cluster_config_prefill_replica_config_router_topk", str(args.router_topk),
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",
        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", str(args.attn_tp),
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_device", "h800",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", "0.2",
        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_total_expert_num", str(args.total_experts),
        "--cluster_config_decode_ffn_replica_config_router_topk", str(args.router_topk),
        "--cluster_config_decode_ffn_replica_config_device", "h800",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",
        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",
        "--cc_backend_config_type", "analytical",
        "--m2n_transfer_config_type", "analytical",
        "--replica_config_model_name", args.model_name,
        "--replica_config_moe_routing_mode", "uniform_random",
        "--replica_config_moe_routing_seed", "42",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", str(args.block_size),
        "--vllm_v1_scheduler_config_num_blocks", "4096",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--cluster_config_prefill_replica_scheduler_config_num_blocks", "4096",
        "--cluster_config_decode_attn_replica_scheduler_config_num_blocks", "4096",
        "--cluster_config_decode_attn_replica_scheduler_config_block_size", str(args.block_size),
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "1",
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "2",
        "--fixed_request_length_generator_config_decode_tokens", "1",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1.0",
        "--metrics_config_output_dir", "/tmp/task48_test_output",
        "--metrics_config_run_id", f"memprobe_{args.model_name}_{args.attn_tp}",
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        "--seed", "0",
    ]
    from frontier.config import SimulationConfig
    from frontier.types import ClusterType
    from frontier.entities.replica import Replica
    from frontier.scheduler.utils.memory_planner import MemoryPlanner

    config = SimulationConfig.create_from_cli_args()
    replica_config = config.cluster_config.decode_attn_replica_config
    replica = Replica(replica_config, config.request_generator_config, ClusterType.DECODE_ATTN)
    planner = MemoryPlanner(replica_config=replica_config, replica=replica,
                            cluster_type=ClusterType.DECODE_ATTN)
    page_size = planner._get_kv_cache_memory_per_layer_per_block(args.block_size)
    print(_RESULT_MARKER + str(page_size), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
