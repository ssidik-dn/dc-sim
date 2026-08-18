#!/usr/bin/env python3
"""Task 07: can a KV cache transfer predictor registered from outside
`upstream/` actually be selected and used by a real Frontier run?

This is a probe, not a test -- it is not part of `python3 -m pytest -q`,
which stays fast and free of external dependencies. It runs a full Frontier
pd-disaggregation simulation, in-process, and answers one question by
observation rather than by reading source: does a sentinel
`BaseKVCacheTransferPredictor`, registered under the previously-unused
`KVCacheTransferType.EMPIRICAL`, actually get its `get_transfer_time` called
when a run is configured with `--kv_cache_transfer_config_type empirical`?

Environment this needed (see the task 07 report for how this differs from a
pinned `upstream/` checkout):

    PYTHONPATH must include the Frontier checkout (this project reaches it at
    /work/Frontier via the ambient PYTHONPATH -- see task 06's report; there
    is no `upstream/frontier` in this repo). Run from the dc-sim root:

        PYTHONPATH=src:/work/Frontier python3 tools/probe_kv_selection.py

Device: h800, per AGENTS.md -- only h800 and rtx_pro_6000 carry full-feature
compute profiles, and the shipped examples default to a device (a800) that
lacks them. This probe runs with ENABLE_DUMMY_MODE (a flat per-operator time,
see AGENTS.md's "Dummy mode" trap), so the compute profile is not actually
consulted here -- h800 is used anyway, for consistency with prior tasks and
in case the KV/memory-margin path reads device info independent of dummy
mode.

Nothing under `upstream/` or `src/engine/` is modified. The sentinel class
and its registration live entirely in this file.
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from frontier.entities import Batch, Request
    from frontier.config import ReplicaConfig

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/kv_probe_outputs")

SENTINEL_TRANSFER_TIME_MS = 424242.0

# `type.__subclasses__()` (what `frontier.config.utils.get_all_subclasses`
# walks to find BasePolyConfig implementations) holds only weak references.
# The sentinel config class must be kept alive by a strong reference for the
# whole probe run, or it is garbage-collected between CLI parsing and
# reconstruction and silently drops out of the candidate list -- this is
# exactly what happened on the first run of this probe; see the task 07
# report.
_SENTINEL_CONFIG_CLS = None


def _build_sentinel():
    """Import Frontier and define the sentinel config/predictor.

    Deferred into a function so the module can be imported (e.g. for
    `SENTINEL_TRANSFER_TIME_MS`) without requiring Frontier on the path.
    """
    global _SENTINEL_CONFIG_CLS
    from frontier.config.kv_cache_transfer_config import BaseKVCacheTransferConfig
    from frontier.kv_cache_transfer.base_kv_cache_transfer_predictor import (
        BaseKVCacheTransferPredictor)
    from frontier.kv_cache_transfer.kv_cache_transfer_predictor_registry import (
        KVCacheTransferPredictorRegistry)
    from frontier.types import ClusterType, KVCacheTransferType

    @dataclass
    class EmpiricalKVCacheTransferConfig(BaseKVCacheTransferConfig):
        """Sentinel config for the probe. EMPIRICAL has no implementation
        anywhere in this Frontier checkout (task 07 report S3.1) -- this
        class is the first."""

        @classmethod
        def get_type(cls) -> "KVCacheTransferType":
            return KVCacheTransferType.EMPIRICAL

        @classmethod
        def get_name(cls) -> str:
            return "empirical"

    _SENTINEL_CONFIG_CLS = EmpiricalKVCacheTransferConfig

    class SentinelKVCacheTransferPredictor(BaseKVCacheTransferPredictor):
        """Registered under EMPIRICAL. Every call records itself;
        get_transfer_time returns a value no real predictor would produce."""

        calls: int = 0
        last_kv_cache_size_bytes: Optional[int] = None

        def get_transfer_time(self, source_cluster_type: "ClusterType",
                              target_cluster_type: "ClusterType",
                              batch: "Batch", kv_cache_size_bytes: int) -> float:
            SentinelKVCacheTransferPredictor.calls += 1
            SentinelKVCacheTransferPredictor.last_kv_cache_size_bytes = kv_cache_size_bytes
            print(f"SENTINEL_CALLED get_transfer_time "
                  f"source={source_cluster_type} target={target_cluster_type} "
                  f"kv_cache_size_bytes={kv_cache_size_bytes} "
                  f"-> {SENTINEL_TRANSFER_TIME_MS}", flush=True)
            return SENTINEL_TRANSFER_TIME_MS

        def get_kv_cache_size(self, batch: "Batch",
                              replica_config: "ReplicaConfig") -> int:
            return 1

        def get_kv_cache_size_for_request(self, request: "Request",
                                          replica_config: "ReplicaConfig") -> int:
            return 1

        def supports_latency_hiding(self) -> bool:
            return False

    KVCacheTransferPredictorRegistry.register(
        KVCacheTransferType.EMPIRICAL, SentinelKVCacheTransferPredictor)

    return SentinelKVCacheTransferPredictor


def _argv() -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", "1",
        "--cluster_config_prefill_replica_config_router_topk", "1",
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",
        "--cluster_config_decode_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_decode_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_replica_config_total_expert_num", "1",
        "--cluster_config_decode_replica_config_router_topk", "1",
        "--cluster_config_decode_replica_config_device", "h800",
        "--cluster_config_decode_replica_config_memory_margin_fraction", "0.2",
        "--cc_backend_config_type", "analytical",
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--replica_config_moe_routing_mode", "simulation",
        "--replica_config_moe_routing_seed", "42",
        "--replica_scheduler_config_type", "vllm_v1",
        "--decode_cuda_graph_mode", "none",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "1024",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", "128",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "2",
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "64",
        "--fixed_request_length_generator_config_decode_tokens", "8",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1.0",
        # the flag under test: select EMPIRICAL, not the shipped ANALYTICAL
        "--kv_cache_transfer_config_type", "empirical",
        "--metrics_config_output_dir", str(OUTPUT_DIR),
        "--metrics_config_run_id", "probe_kv_selection",
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
    sentinel_cls = _build_sentinel()

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

    print(f"\nsentinel calls: {sentinel_cls.calls}")
    print(f"sentinel last kv_cache_size_bytes: {sentinel_cls.last_kv_cache_size_bytes}")

    if sentinel_cls.calls > 0:
        print("\nANSWER: OPEN -- the sentinel predictor's get_transfer_time was "
              "called by a real Frontier run selected purely via "
              "--kv_cache_transfer_config_type empirical.")
        return 0

    print("\nANSWER: the sentinel was registered but never called -- see the "
          "traceback above for exactly where the run stopped.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
