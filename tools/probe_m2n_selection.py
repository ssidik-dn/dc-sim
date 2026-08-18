#!/usr/bin/env python3
"""Task 08: can a M2N (attention<->FFN activation) transfer predictor
registered from outside `upstream/` actually be selected and used by a real
Frontier pd-af-disaggregation run?

Modelled directly on tools/probe_kv_selection.py (task 07). Same question,
same method: a sentinel `BaseM2NTransferPredictor`, registered under the
unused `M2NTransferType.EMPIRICAL`, selected purely via
`--m2n_transfer_config_type empirical`, observed by running rather than by
reading source.

One change from the KV probe, made deliberately: the sentinel config and
predictor classes below are defined at MODULE level, not inside a builder
function. Task 07's probe first failed with an error that read exactly like
a closed gate (`Invalid type empirical ... Valid types: ['analytical']`) and
was actually `type.__subclasses__()` dropping a function-local class once
nothing kept a strong reference to it between CLI parsing and
reconstruction. Defining the classes at module level means the module's own
namespace holds that strong reference for the process's lifetime -- the trap
does not need rediscovering.

Environment: same as task 07 -- Frontier is reached via the ambient
PYTHONPATH (no `upstream/frontier` pin in this repo). Run from the dc-sim
root:

    PYTHONPATH=src:/work/Frontier python3 tools/probe_m2n_selection.py

Device: h800 for all three clusters (prefill, decode_attn, decode_ffn), per
AGENTS.md. Runs with ENABLE_DUMMY_MODE, so compute profiles are not actually
consulted, but h800 is used anyway for consistency with prior tasks.

Nothing under `upstream/` or `src/engine/` is modified.
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from frontier.config.m2n_transfer_config import BaseM2NTransferConfig
from frontier.m2n_transfer.base_m2n_transfer_predictor import BaseM2NTransferPredictor
from frontier.m2n_transfer.m2n_transfer_predictor_registry import (
    M2NTransferPredictorRegistry)
from frontier.types import ClusterType, M2NTransferType

if TYPE_CHECKING:
    from frontier.config import ReplicaConfig
    from frontier.entities import Batch, Request

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/m2n_probe_outputs")

SENTINEL_TRANSFER_TIME_MS = 313131.0


@dataclass
class EmpiricalM2NTransferConfig(BaseM2NTransferConfig):
    """Sentinel config. EMPIRICAL has no implementation anywhere in this
    Frontier checkout (task 07 found the same for KVCacheTransferType;
    M2NTransferType is the identical shape) -- this class is the first."""

    @classmethod
    def get_type(cls) -> "M2NTransferType":
        return M2NTransferType.EMPIRICAL

    @classmethod
    def get_name(cls) -> str:
        return "empirical"


class SentinelM2NTransferPredictor(BaseM2NTransferPredictor):
    """Registered under EMPIRICAL. Every call records itself, including the
    fields the task report asks about -- layer_id and afd_stage_idx are not
    parameters of get_transfer_time (see the task report: they are attached
    to M2NTransferInfo by the *caller*, after this method returns, for
    metrics bookkeeping), so this predictor derives them from `batch` the
    same way frontier/events/cluster_batch_end_event.py does, to confirm
    they are actually available and populated at the point a predictor
    could use them.
    """

    calls: int = 0
    seen_layer_ids: set = set()
    seen_afd_stage_idx: set = set()
    seen_pipeline_stages: set = set()
    last_activation_size_bytes: Optional[int] = None

    @staticmethod
    def _current_layer_id(batch: "Batch") -> Optional[int]:
        """Mirrors ClusterBatchEndEvent._get_current_layer_id_from_batch."""
        requests = getattr(batch, "requests", None) or []
        for request in requests:
            if not request.completed:
                return request.completed_layer_count
        return requests[0].completed_layer_count if requests else None

    def get_transfer_time(self, source_cluster_type: "ClusterType",
                          target_cluster_type: "ClusterType",
                          batch: "Batch", activation_size_bytes: int) -> float:
        cls = SentinelM2NTransferPredictor
        cls.calls += 1
        cls.last_activation_size_bytes = activation_size_bytes

        layer_id = self._current_layer_id(batch)
        afd_stage_idx = getattr(batch, "afd_stage_idx", None)
        pipeline_stage = ("attn_to_ffn" if source_cluster_type == ClusterType.DECODE_ATTN
                          else "ffn_to_attn")
        cls.seen_layer_ids.add(layer_id)
        cls.seen_afd_stage_idx.add(afd_stage_idx)
        cls.seen_pipeline_stages.add(pipeline_stage)

        print(f"SENTINEL_CALLED get_transfer_time "
              f"source={source_cluster_type} target={target_cluster_type} "
              f"activation_size_bytes={activation_size_bytes} "
              f"layer_id={layer_id} afd_stage_idx={afd_stage_idx} "
              f"pipeline_stage={pipeline_stage} -> {SENTINEL_TRANSFER_TIME_MS}",
              flush=True)
        return SENTINEL_TRANSFER_TIME_MS

    def get_activation_size(self, batch: "Batch", replica_config: "ReplicaConfig",
                            source_cluster_type: "ClusterType") -> int:
        return 1

    def get_activation_size_for_request(self, request: "Request",
                                        replica_config: "ReplicaConfig",
                                        source_cluster_type: "ClusterType") -> int:
        return 1


M2NTransferPredictorRegistry.register(
    M2NTransferType.EMPIRICAL, SentinelM2NTransferPredictor)


def _argv() -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters",

        # Cluster replica counts
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", "1",
        "--cluster_config_decode_ffn_cluster_num_replicas", "1",

        # AF pipeline micro-batch
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_attn_micro_batch_size", "8",

        # Prefill cluster replica config (dense: MoE_TP=MoE_EP=1, EP=1)
        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", "1",
        "--cluster_config_prefill_replica_config_router_topk", "1",
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",

        # Decode-Attn cluster replica config
        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_device", "h800",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", "0.2",

        # Decode-FFN cluster replica config
        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_total_expert_num", "1",
        "--cluster_config_decode_ffn_replica_config_router_topk", "1",
        "--cluster_config_decode_ffn_replica_config_device", "h800",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",

        # Per-cluster scheduler types (decode-ffn uses orca in the shipped example)
        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",

        # Backend config
        "--cc_backend_config_type", "analytical",
        # the flag under test: select EMPIRICAL, not the shipped ANALYTICAL
        "--m2n_transfer_config_type", "empirical",

        # Model / MoE routing (dense: total_experts=1, router_topk=1).
        # A generic HF name rather than one of Frontier's bundled model
        # JSONs (e.g. llama2_7b_dense_example) -- those resolve relative to
        # the Frontier repo root as cwd, which this probe does not assume.
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--replica_config_moe_routing_mode", "simulation",
        "--replica_config_moe_routing_seed", "42",

        # Scheduler parameters
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "1024",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "64",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", "128",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",

        # Workload -- small, just enough to see multiple layers/decode steps
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "2",
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "32",
        "--fixed_request_length_generator_config_decode_tokens", "4",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1.0",

        # KV transfer (prefill -> decode_attn) -- unrelated to this probe,
        # left on the shipped analytical backend
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps", "200.0",
        "--analytical_kv_cache_transfer_config_network_latency_ms", "0.5",

        "--metrics_config_output_dir", str(OUTPUT_DIR),
        "--metrics_config_run_id", "probe_m2n_selection",
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


def main() -> int:
    sys.argv = _argv()
    print("argv:", " ".join(sys.argv), "\n")

    try:
        from frontier.main import main as frontier_main
        frontier_main()
    except SystemExit as exc:
        print(f"\nRESULT: Frontier exited with SystemExit({exc.code})")
    except Exception:
        print("\nRESULT: Frontier raised before/without calling the sentinel:")
        traceback.print_exc()

    cls = SentinelM2NTransferPredictor
    print(f"\nsentinel calls: {cls.calls}")
    print(f"sentinel last activation_size_bytes: {cls.last_activation_size_bytes}")
    print(f"sentinel layer_ids seen: {sorted(x for x in cls.seen_layer_ids if x is not None)}")
    print(f"sentinel afd_stage_idx seen: {sorted(x for x in cls.seen_afd_stage_idx if x is not None)}")
    print(f"sentinel pipeline_stages seen: {sorted(cls.seen_pipeline_stages)}")

    if cls.calls > 0:
        print("\nANSWER: OPEN -- the sentinel predictor's get_transfer_time was "
              "called by a real Frontier pd-af-disaggregation run selected "
              "purely via --m2n_transfer_config_type empirical.")
        return 0

    print("\nANSWER: the sentinel was registered but never called -- see the "
          "traceback above for exactly where the run stopped.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
