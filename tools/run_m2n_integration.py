#!/usr/bin/env python3
"""Task 11: the end-to-end proof that a real Frontier run's inter-token
latency changes because of where the attention and FFN pools sit.

Runs the same offline pd-af-disaggregation workload twice, through
EngineM2NTransferPredictor (registered under M2NTransferType.EMPIRICAL,
selected purely via --m2n_transfer_config_type empirical -- see
tools/probe_m2n_selection.py, task 08), differing only in whether the
DECODE_ATTN and DECODE_FFN pools' representative GPUs share a scale-up
domain ("colocated") or are split across two ("split"). KV cache transfer
stays on Frontier's own `analytical` backend throughout -- this run is about
M2N, not KV (task 09 already covers KV).

Read docs/tasks/11-m2n-predictor-report.md before trusting a metric name.
Task 09's lesson (Frontier's `ttft` structurally excludes KV transfer time)
applies here too: `request.tpot` (Time Per Output Token) is the field that
actually includes M2N transfer time as a component -- confirmed by reading
metrics_store.py, where `tpot_computation = request.tpot - tpot_transfer`
only makes sense if `tpot` already contains the transfer term. `ttft` would
be exactly the wrong field again, for the same structural reason as before.

Environment: same as tasks 07/08/09 -- Frontier is reached via the ambient
PYTHONPATH, run from the dc-sim root:

    PYTHONPATH=src:/work/Frontier python3 tools/run_m2n_integration.py

Each scenario runs in its own subprocess (`--scenario colocated|split` is
how this script re-invokes itself) rather than both in one process. That is
not incidental: pd-af-disaggregation's round-robin cluster scheduler keeps
FFN-lane bookkeeping keyed by replica ids that increment globally across
every `Simulator` built in the process, so a second in-process run raises
"DECODE_FFN target_ffn_replica_id must be an exact non-negative int, got
None" -- a real, pre-existing Frontier limitation unrelated to this
project's predictor (confirmed by bisection: a lone run always succeeds; a
second run in an already-used process always fails the same way, regardless
of placement or order). Frontier's own shipped examples never run two
simulations per process either. See the task 11 report.

Device: h800 (AGENTS.md). Execution time uses Frontier's dummy mode; that
governs compute prediction only, not this project's M2N predictor.

Nothing under `upstream/` or `src/engine/` is modified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

from frontier.types import ClusterType

from engine.logical.deployment import Deployment, PoolKind, Replica
from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId
from engine.placement.placement import explicit

from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment
from integration.install import install

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/m2n_integration_outputs")

SCALE_UP_GBPS = 400.0
SCALE_OUT_GBPS = 50.0


def _engine_deployment_and_registry():
    """One replica per pool -- task 11 spec S2.3 restricts this predictor to
    exactly that, same as the KV predictor (task 09 spec S2.2)."""
    d = Deployment("m2n-integration")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {
        PoolKind.PREFILL: ClusterType.PREFILL,
        PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
        PoolKind.DECODE_FFN: ClusterType.DECODE_FFN,
    })
    return d, reg


def _placements(fabric, deployment):
    prefill_rank = deployment.replicas[0].ranks[0]
    attn_rank = deployment.replicas[1].ranks[0]
    ffn_rank = deployment.replicas[2].ranks[0]
    # PREFILL rides along with DECODE_ATTN's domain in both scenarios -- its
    # placement doesn't matter here, KV stays on Frontier's own analytical
    # backend, untouched by this project's engine.
    colocated = explicit(deployment, fabric, {
        prefill_rank: GpuId(0, 0), attn_rank: GpuId(0, 1), ffn_rank: GpuId(0, 2)})
    split = explicit(deployment, fabric, {
        prefill_rank: GpuId(0, 0), attn_rank: GpuId(0, 1), ffn_rank: GpuId(1, 0)})
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

        # KV transfer stays on Frontier's own analytical backend -- this run
        # is about M2N (task 09 already covers KV).
        "--cc_backend_config_type", "analytical",
        # the flag under test: our real predictor, not Frontier's analytical one
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
        "--synthetic_request_generator_config_num_requests", "2",
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "32",
        "--fixed_request_length_generator_config_decode_tokens", "4",
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


def _find_m2n_predictor(sim):
    """The registry constructs exactly one instance per run
    (frontier/simulator.py), handed to the DECODE_ATTN and DECODE_FFN
    cluster schedulers as `_m2n_transfer_predictor`
    (base_cluster_scheduler.py) -- not stored anywhere higher up, so dig it
    out through whichever cluster scheduler has one."""
    schedulers = getattr(sim._global_scheduler, "_cluster_schedulers", {})
    for scheduler in schedulers.values():
        predictor = getattr(scheduler, "_m2n_transfer_predictor", None)
        if predictor is not None:
            return predictor
    return None


_RESULT_MARKER = "M2N_INTEGRATION_RESULT="


def _run_scenario(label: str) -> None:
    """Runs exactly one scenario, in this process, and prints its result as
    one JSON line. Deliberately never called twice in the same process --
    see `main()`'s docstring for why."""
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    colocated_placement, split_placement = _placements(fabric, deployment)
    placement = colocated_placement if label == "colocated" else split_placement
    install(fabric, placement, deployment, registry)

    sys.argv = _argv(f"m2n_integration_{label}")
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
        "mean_m2n_time_s": mean(m2n_time_s),
        "mean_tpot_s": mean(tpot_s),
        "n_m2n": len(m2n_time_s),
        "n_tpot": len(tpot_s),
        "predictor_calls": predictor.calls if predictor else 0,
        "predictor_total_wall_ns": predictor.total_wall_ns if predictor else 0,
    }), flush=True)


def _run_scenario_in_subprocess(label: str) -> dict:
    """Frontier's own shipped examples (examples/architecture/*/offline/*.sh)
    always invoke `python3 -m frontier.main` as a fresh process, once per
    run -- never twice in one interpreter. Task 09's KV integration ran two
    scenarios back to back in a single process and that worked, because
    pd-disaggregation's cluster scheduler carries no cross-run state that
    matters. pd-af-disaggregation's round-robin cluster scheduler does: its
    FFN-lane bookkeeping is built from replica ids that increment globally
    across every Simulator constructed in the process, not reset per run,
    and a second in-process run raised
    "DECODE_FFN target_ffn_replica_id must be an exact non-negative int, got
    None" -- confirmed by bisection (a single run, alone, in a fresh
    process, always succeeds; a second run in the same process that already
    built one Simulator always fails the same way, regardless of which
    placement is used or in which order). This is a real, pre-existing
    Frontier limitation, unrelated to this project's predictor, and not
    something to route around by mutating Frontier's own scheduler state
    from outside it -- so each scenario gets its own subprocess instead,
    matching the shipped examples' own assumption.
    """
    proc = subprocess.run([sys.executable, __file__, "--scenario", label],
                          capture_output=True, text=True)
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
        mean_call_us = (r["predictor_total_wall_ns"] / r["predictor_calls"] / 1000.0
                        if r["predictor_calls"] else 0.0)
        print(f"{label}: mean total_m2n_transfer_time={r['mean_m2n_time_s']*1000:.6f} ms, "
              f"mean tpot={r['mean_tpot_s']*1000:.6f} ms "
              f"(n_m2n={r['n_m2n']}, n_tpot={r['n_tpot']}), "
              f"predictor calls={r['predictor_calls']}, "
              f"mean call cost={mean_call_us:.2f} us")

    colocated, split = results["colocated"], results["split"]
    m2n_ratio = split["mean_m2n_time_s"] / colocated["mean_m2n_time_s"]
    tpot_ratio = split["mean_tpot_s"] / colocated["mean_tpot_s"]

    print()
    print(f"M2N transfer time: colocated={colocated['mean_m2n_time_s']*1000:.6f} ms  "
          f"split={split['mean_m2n_time_s']*1000:.6f} ms  ratio={m2n_ratio:.4f}")
    print(f"TPOT (inter-token): colocated={colocated['mean_tpot_s']*1000:.6f} ms  "
          f"split={split['mean_tpot_s']*1000:.6f} ms  ratio={tpot_ratio:.4f}")
    print(f"(fabric bandwidth ratio scale_up:scale_out = "
          f"{SCALE_UP_GBPS/SCALE_OUT_GBPS:.1f}:1)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("colocated", "split"), default=None,
                       help="internal: run one scenario in this process and exit")
    args = parser.parse_args()
    if args.scenario:
        _run_scenario(args.scenario)
        raise SystemExit(0)
    raise SystemExit(main())
