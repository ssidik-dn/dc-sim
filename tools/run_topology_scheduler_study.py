#!/usr/bin/env python3
"""Task 15 spec S4.2: does a scheduler that actually picks the near replica
change real serving behaviour, on the M2N path (not KV -- task 09/14 both
established KV transfer completes before decode starts and so cannot reach
inter-token latency; task 14's own study measured mean_tpot at exactly
358.000000 ms in all six of its runs for exactly that reason. Repeating that
mistake here would make this study's headline question unanswerable).

Scenario: one PREFILL replica, one DECODE_ATTN replica with
attn_data_parallel_size=4 (four dp lanes, all on the same machine -- the
"source" side is deliberately not split, so this study isolates the
destination decision this task is actually about), four DECODE_FFN
replicas -- replica 0 shares the ATTN lanes' scale-up domain, replicas 1-3
are each on their own separate machine (the same asymmetric shape task 14's
own study and this task's unit tests use).

Two scheduler variants share one CLI config (`--cluster_scheduler_config_type
round_robin` -- the only one of Frontier's five usable for a disaggregated
architecture at all; see S1/S3 below for why LOR can't be the second
baseline the spec asked for):

- **round_robin**: Frontier's own, entirely unmodified
  `RoundRobinClusterScheduler`, whose DECODE_FFN `__init__` assigns each
  DECODE_ATTN dp lane to `lane_ordinal % num_ffn_replicas` -- distance-blind.
- **topology_aware**: the same real `Simulator`, the same real
  `RoundRobinClusterScheduler` instance Frontier just built -- with its
  `_ffn_lane_to_target_replica` map recomputed in place, after construction
  and before `.run()`, by calling
  `TopologyAwareClusterScheduler._assign_ffn_lanes_by_topology` as an
  unbound method against it. This is deliberate, and documented as such: S1
  found `ClusterSchedulerType` closed, so there is no CLI flag that
  constructs `TopologyAwareClusterScheduler` for real, and swapping in a
  freshly-constructed instance is unsafe (every per-dp-lane
  `ReplicaScheduler` Frontier already built holds a `cluster_scheduler=self`
  back-reference to the *original* object -- see topology_aware.py's module
  docstring). Mutating the three lane-map attributes the base `__init__`
  already created is the one substitution that changes nothing else in an
  already-wired object graph.

Task 15 S3.2 ("price against whatever the scheduler actually chose"): tried,
and this project's own empirical M2N predictor cannot be used here at all --
confirmed by running it, not assumed. M2N is a round trip: DECODE_ATTN sends
activations to DECODE_FFN, and DECODE_FFN sends them back
(`cluster_batch_end_event.py`'s two `get_transfer_info` call sites, one per
direction). `price_transfer` (integration/binding_support.py) resolves its
*source* pool unconditionally -- `ctx.groups.resolve_pool(source_cluster_type)`,
with no try/except, unlike the destination side, because task 14 scoped
binding to "which replica *receives*" and left the source side alone. With
four DECODE_FFN replicas, DECODE_FFN is exactly this ambiguous as a *source*
on the return leg, and the run raises `CommGroupError` immediately. Tasks
09-14 never hit this because every prior scenario kept every pool that was
ever a *source* single-replica; this is the first task to put several
replicas on a pool that sends as well as receives.

This study therefore uses Frontier's own stock **analytical** M2N predictor
(the default -- no `--m2n_transfer_config_type` flag at all), which is
distance-blind by construction. One direct consequence and one direct
benefit follow from that: `mean_m2n_time` is not expected to differ, and
does not, between the two scheduler variants (S4) -- and, as a benefit, any
tpot difference this study finds is therefore an unconfounded queueing/
distribution effect, not commingled with either predictor's own pricing
choices. See the task 15 report S2/S4 for the measured numbers.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as tasks 12/13/14:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_topology_scheduler_study.py

Real compute profiles are not the point (the scheduling decision and its
fabric cost are); dummy execution-time mode is used, matching task 09/14.
Nothing under `upstream/`, `src/engine/`, or the predictors is modified --
the scheduler-instance patch happens from this script, at an object already
constructed by a real, unmodified Frontier run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

FRONTIER_ROOT = Path("/work/Frontier")

from engine.logical.deployment import Deployment, PoolKind, Replica  # noqa: E402
from engine.physical.builders import build_node_scale  # noqa: E402
from engine.physical.topology import GpuId  # noqa: E402
from engine.placement.placement import explicit  # noqa: E402

from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/topology_scheduler_outputs")

SCALE_UP_GBPS = 400.0
SCALE_OUT_GBPS = 50.0
NUM_FFN_REPLICAS = 4
ATTN_DP_SIZE = 8
NUM_REQUESTS = 16
DECODE_TOKENS = 16

VARIANTS = ("round_robin", "topology_aware")


def _engine_deployment_and_registry():
    from frontier.types import ClusterType
    d = Deployment("topology-scheduler-study")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=1, dp=ATTN_DP_SIZE))
    for i in range(NUM_FFN_REPLICAS):
        d.add(Replica(PoolKind.DECODE_FFN, i, tp=1))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {
        PoolKind.PREFILL: ClusterType.PREFILL,
        PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
        PoolKind.DECODE_FFN: ClusterType.DECODE_FFN,
    })
    return d, reg


def _placement(fabric, deployment):
    """ATTN's four dp lanes and FFN replica 0 share machine 0's scale-up
    domain; FFN replicas 1-3 are each alone on their own machine."""
    prefill_rank = deployment.replicas[0].ranks[0]
    attn_ranks = deployment.replicas[1].ranks  # 4 ranks, dp lanes 0-3
    ffn_ranks = [r.ranks[0] for r in deployment.replicas[2:]]

    mapping = {prefill_rank: GpuId(0, 0)}
    for i, rank in enumerate(attn_ranks):
        mapping[rank] = GpuId(0, 1 + i)
    mapping[ffn_ranks[0]] = GpuId(0, 1 + len(attn_ranks))  # near: same domain
    for i, rank in enumerate(ffn_ranks[1:], start=1):
        mapping[rank] = GpuId(i, 0)  # far: each its own machine
    return explicit(deployment, fabric, mapping)


def _argv(run_id: str) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        "frontier.main",
        "--simulation_mode", "offline",
        "--sys_arch", "pd-af-disaggregation",
        "--no-enable_parallel_clusters",

        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_attn_cluster_num_replicas", "1",
        "--cluster_config_decode_ffn_cluster_num_replicas", str(NUM_FFN_REPLICAS),
        "--cluster_config_allow_experiment_multi_decode_ffn_replicas",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch", "1",
        "--cluster_config_decode_attn_micro_batch_size", "8",

        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", "16",
        "--cluster_config_prefill_replica_config_router_topk", "2",
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",

        # attn_data_parallel_size>1 in decode_attn requires a MoE model
        # (config.py's dense-model validation: "Dense models require
        # attn_data_parallel_size=1 in decode_attn cluster") -- Phi-tiny-MoE-instruct
        # (16 experts, data/config/models/Phi-tiny-MoE-instruct.json) is
        # Frontier's own example model for exactly this combination
        # (examples/architecture/pd-af-disagg/offline/moe_model_basic.sh).
        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", str(ATTN_DP_SIZE),
        "--cluster_config_decode_attn_replica_config_device", "h800",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_total_expert_num", "16",
        "--cluster_config_decode_ffn_replica_config_router_topk", "2",
        "--cluster_config_decode_ffn_replica_config_device", "h800",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",

        "--cluster_scheduler_config_type", "round_robin",

        "--cc_backend_config_type", "analytical",
        # Frontier's own analytical M2N predictor, not this project's
        # empirical one -- see module docstring S3.2 for why: our own
        # predictor's price_transfer resolves a *source* pool unconditionally
        # (integration/binding_support.py, no try/except around it, unlike
        # the destination side), and DECODE_FFN is the source on the
        # FFN->ATTN return leg of every M2N round trip. With four FFN
        # replicas that raises CommGroupError immediately -- confirmed by
        # running it, not assumed -- so it cannot be used here at all.

        "--replica_config_model_name", "Phi-tiny-MoE-instruct",
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


_RESULT_MARKER = "TOPOLOGY_SCHEDULER_RESULT="


def _run_scenario(variant: str) -> None:
    from frontier.types import ClusterType
    from integration.cluster_scheduler.topology_aware import TopologyAwareClusterScheduler

    fabric = build_node_scale(num_machines=NUM_FFN_REPLICAS, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _engine_deployment_and_registry()
    placement = _placement(fabric, deployment)
    # No `binding=` here: this study doesn't use this project's own M2N
    # predictor at all (see module docstring S3.2), so nothing would ever
    # read `ctx.binding`. `install()` is still needed purely to make
    # `ctx.fabric`/`ctx.placement` reachable to
    # TopologyAwareClusterScheduler's own `require_context()` call.
    install(fabric, placement, deployment, registry)

    sys.argv = _argv(f"topology_scheduler_{variant}")
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    config = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)
    sim = Simulator(config)

    ffn_scheduler = sim._global_scheduler._cluster_schedulers[ClusterType.DECODE_FFN]
    if variant == "topology_aware":
        TopologyAwareClusterScheduler._assign_ffn_lanes_by_topology(ffn_scheduler)

    # Read the real, static lane->replica map directly -- no predictor
    # involved (see module docstring on why predictor pricing can't see it).
    lane_map = dict(ffn_scheduler._ffn_lane_to_target_replica)
    ffn_offset = min(ffn_scheduler._cluster.replicas.keys())
    within_domain = 0
    for lane, ffn_replica_id in lane_map.items():
        attn_replica_id, dp_id = lane
        attn_gpu = placement.gpu(deployment.replicas[1].ranks[dp_id])
        ffn_gpu = placement.gpu(deployment.replicas[2 + (ffn_replica_id - ffn_offset)].ranks[0])
        if fabric.same_domain(attn_gpu, ffn_gpu):
            within_domain += 1
    distribution = {}
    for ffn_replica_id in lane_map.values():
        distribution[ffn_replica_id - ffn_offset] = distribution.get(ffn_replica_id - ffn_offset, 0) + 1

    sim.run()

    requests = sim._all_requests
    m2n_time_s = [r.total_m2n_transfer_time for r in requests]
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_s = [r.tpot for r in tpot_eligible]

    print(_RESULT_MARKER + json.dumps({
        "mean_m2n_time_s": mean(m2n_time_s) if m2n_time_s else None,
        "mean_tpot_s": mean(tpot_s) if tpot_s else None,
        "n_requests": len(requests),
        "n_tpot": len(tpot_s),
        "lane_map": {f"{k[0]}.{k[1]}": v for k, v in lane_map.items()},
        "distribution": distribution,
        "within_domain": within_domain,
        "total_lanes": len(lane_map),
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(variant: str) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--variant", variant],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise RuntimeError(f"variant={variant!r} produced no result "
                       f"(exit code {proc.returncode}); see output above")


def main() -> int:
    """LOR is not a second baseline here, or anywhere in a disaggregated
    architecture: `LORClusterScheduler.schedule()` unconditionally raises
    `DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR` for any cluster type other
    than MONOLITHIC (lor_cluster_scheduler.py), before it ever reaches its
    own (otherwise-applicable) LOR logic. Reproduced directly: constructing
    a LOR-scheduled pd-af-disaggregation run raises immediately. See the
    task 15 report S1/S6 for the exact message and citation. round_robin is
    therefore the only real baseline available.
    """
    results = {}
    for variant in VARIANTS:
        r = _run_scenario_in_subprocess(variant)
        results[variant] = r
        print(f"[{variant:<14}] mean_m2n={r['mean_m2n_time_s']*1000:9.6f} ms  "
             f"mean_tpot={r['mean_tpot_s']*1000:9.6f} ms  "
             f"distribution={r['distribution']}  "
             f"within_domain={r['within_domain']}/{r['total_lanes']}")

    print()
    print("mechanism (fraction of DECODE_ATTN->DECODE_FFN lanes assigned within "
         "one scale-up domain):")
    for variant in VARIANTS:
        r = results[variant]
        frac = r["within_domain"] / r["total_lanes"] if r["total_lanes"] else float("nan")
        print(f"  {variant:<14} {r['within_domain']}/{r['total_lanes']} ({100*frac:.1f}%)")

    print()
    print("inter-token latency (tpot):")
    rr, ta = results["round_robin"], results["topology_aware"]
    delta = ta["mean_tpot_s"] - rr["mean_tpot_s"]
    print(f"  round_robin:    {rr['mean_tpot_s']*1000:.6f} ms")
    print(f"  topology_aware: {ta['mean_tpot_s']*1000:.6f} ms")
    print(f"  delta:          {delta*1000:+.6f} ms ({100*delta/rr['mean_tpot_s']:+.2f}%)")

    print()
    print("mean M2N transfer time (see module docstring: not expected to differ -- "
         "the predictor cannot see which scheduler variant is active):")
    print(f"  round_robin:    {rr['mean_m2n_time_s']*1000:.6f} ms")
    print(f"  topology_aware: {ta['mean_m2n_time_s']*1000:.6f} ms")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default=None, help="internal")
    args = parser.parse_args()
    if args.variant:
        _run_scenario(args.variant)
        raise SystemExit(0)
    raise SystemExit(main())
