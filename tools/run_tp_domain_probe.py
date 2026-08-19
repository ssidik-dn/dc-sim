#!/usr/bin/env python3
"""Task 19: does within-replica communication become placement-varying once
a replica's own ranks are split across scale-up domains?

Task 18 measured tensor-parallel and pipeline-parallel communication only
at tp=1/pp=1 -- where, by construction, both are exactly zero. A zero that
follows from the configuration is not a measurement (this task's own S5
trap). This script raises tensor-parallel degree to 2, 4, and 8 (S2.1),
then repeats the sweep with the same TP group's ranks split across two
scale-up domains instead of packed into one (S2.2) -- the placement task
18 never tried, and the configuration in which the blind-spot question
can actually arise.

**Reusing, not reinventing.** Reads the same Frontier stage-batch ledger
as task 18's own probe (`tools/run_blind_spot_probe.py`, imported for its
argv/deployment helpers rather than duplicated), through the same two
config-flag couplings that script found and documented
(`write_metrics`/`store_utilization_metrics` both gate the ledger's
in-memory capture, not just disk writes). Real h800 compute profiles
throughout -- no dummy-mode flag anywhere in this script's argv.

**The split placement.** DECODE_ATTN's own `attn_tensor_parallel_size`
ranks -- normally packed onto one machine, since that is what an
NVLink-scale-up domain is for -- are instead spread evenly across two
machines (four-and-four at tp=8, matching this project's own headline
15x-placement-penalty configuration and the `spread` placement policy's
own shape). DECODE_FFN stays at tp=1/ep=1 throughout: task 18 already
measured expert-parallel domain-splitting directly; this script isolates
the tensor-parallel axis task 18 never exercised at all. Consequently
"whether expert participants now sit in different domains" (spec S2.2)
does not apply to any configuration this script builds -- FFN's own ranks
never move here, and that is stated as a scope limit, not answered as if
it were "no" by design.

**Estimating the true split cost (S4.3).** `expert_parallel_communication_time`
and `tensor_parallel_communication_time`'s modern split fields come from
Frontier's own profiled, device+worker-count-keyed table -- no fabric
object is ever consulted. To estimate what a real fabric would charge for
the *same* payload, this script prices one point-to-point hop of the same
size across `build_node_scale`'s own already-established cross-domain link
(50 GB/s, 5000 ns latency, plus 2000 ns egress each way -- the exact
figures task 09 onward have used) via `engine.network.transfers.isolated_durations`,
the same primitive every topology-aware predictor in this project already
calls -- not a new formula. This is a lower bound on a real ring-allreduce's
extra cost (a ring spanning two domains crosses the boundary at least
twice, and the intra-domain hops are barely affected), stated as such, not
as a precise substitute for a full collective simulation.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as task 18's probe:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_tp_domain_probe.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified --
measurement only, per this task's own acceptance criteria.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FRONTIER_ROOT = Path("/work/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_blind_spot_probe import (  # noqa: E402
    MODEL_NAME, ROUTER_TOPK, TOTAL_EXPERTS, TP_KEYS, PP_KEYS,
    OUTPUT_DIR as _BASE_OUTPUT_DIR, SCALE_UP_GBPS, SCALE_OUT_GBPS)
from engine.logical.deployment import Deployment, PoolKind, Replica  # noqa: E402
from engine.network.transfers import Transfer, isolated_durations  # noqa: E402
from engine.physical.builders import build_node_scale  # noqa: E402
from engine.physical.topology import GpuId  # noqa: E402
from engine.placement.placement import explicit  # noqa: E402

from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path(str(_BASE_OUTPUT_DIR) + "_tp_domain")

NUM_REQUESTS = 8
DECODE_TOKENS = 8
TP_VALUES = (1, 2, 4, 8)
PP_VALUES = (1, 2)


def _deployment_and_registry(attn_tp: int, attn_pp: int):
    from frontier.types import ClusterType
    d = Deployment("tp-domain-probe")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=attn_tp, pp=attn_pp))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1, ep=1))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {
        PoolKind.PREFILL: ClusterType.PREFILL,
        PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
        PoolKind.DECODE_FFN: ClusterType.DECODE_FFN,
    })
    return d, reg


def _placement(fabric, deployment, attn_tp: int, split: bool):
    """`split=False` (packed): every DECODE_ATTN rank -- across every
    pipeline stage -- shares one machine (one scale-up domain). `split=True`:
    each pipeline stage's own tp-sized TP group is spread evenly across two
    machines (four-and-four at tp=8) -- the placement task 18 never
    exercised. DECODE_FFN and PREFILL are always packed; this script's
    variable is DECODE_ATTN's TP placement only.
    """
    prefill_rank = deployment.replicas[0].ranks[0]
    attn_ranks = deployment.replicas[1].ranks       # pp * tp ranks, TP-innermost
    ffn_rank = deployment.replicas[2].ranks[0]

    mapping = {prefill_rank: GpuId(0, 0), ffn_rank: GpuId(0, 1)}
    num_stages = len(attn_ranks) // attn_tp
    for stage in range(num_stages):
        stage_ranks = attn_ranks[stage * attn_tp:(stage + 1) * attn_tp]
        if not split or attn_tp == 1:
            for i, rank in enumerate(stage_ranks):
                mapping[rank] = GpuId(1 + stage, i)
        else:
            half = attn_tp // 2
            for i, rank in enumerate(stage_ranks[:half]):
                mapping[rank] = GpuId(1 + stage * 2, i)
            for i, rank in enumerate(stage_ranks[half:]):
                mapping[rank] = GpuId(2 + stage * 2, i)
    return explicit(deployment, fabric, mapping)


def _argv(run_id: str, attn_tp: int, attn_pp: int) -> list[str]:
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
        "--cluster_config_prefill_replica_config_total_expert_num", str(TOTAL_EXPERTS),
        "--cluster_config_prefill_replica_config_router_topk", str(ROUTER_TOPK),
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", str(attn_pp),
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", str(attn_tp),
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_device", "h800",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_ffn_replica_config_total_expert_num", str(TOTAL_EXPERTS),
        "--cluster_config_decode_ffn_replica_config_router_topk", str(ROUTER_TOPK),
        "--cluster_config_decode_ffn_replica_config_device", "h800",
        "--cluster_config_decode_ffn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_prefill_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type", "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type", "orca",

        "--cc_backend_config_type", "analytical",
        "--m2n_transfer_config_type", "empirical",

        "--replica_config_model_name", MODEL_NAME,
        "--replica_config_moe_routing_mode", "uniform_random",
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
        # write_metrics and store_utilization_metrics must both stay True
        # (task 18's own finding): they gate the Frontier stage-batch
        # ledger's in-memory capture, not just disk writing.
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]


_RESULT_MARKER = "TP_DOMAIN_RESULT="


def _run_scenario(attn_tp: int, attn_pp: int, split: bool) -> None:
    fabric = build_node_scale(num_machines=8, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _deployment_and_registry(attn_tp, attn_pp)
    placement = _placement(fabric, deployment, attn_tp, split)
    install(fabric, placement, deployment, registry)

    tag = f"tp{attn_tp}_pp{attn_pp}_{'split' if split else 'packed'}"
    sys.argv = _argv(tag, attn_tp, attn_pp)
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    error = None
    sim = None
    try:
        config = SimulationConfig.create_from_cli_args()
        assert not config.cluster_config.execution_time_predictor_config.enable_dummy_mode
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    if error is not None:
        print(_RESULT_MARKER + json.dumps({"tag": tag, "error": error}), flush=True)
        return

    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    decode_rows = [r for r in rows if r["cluster_type"] in ("DECODE_ATTN", "DECODE_FFN")]
    denom_ms = sum(r["execution_time"]["total_time_ms"] for r in decode_rows)
    tp_ms = sum(sum(r["execution_time"]["component_ledger_ms"].get(k, 0.0) for k in TP_KEYS)
               for r in decode_rows)
    pp_ms = sum(sum(r["execution_time"]["component_ledger_ms"].get(k, 0.0) for k in PP_KEYS)
               for r in decode_rows)
    requests = sim._all_requests
    visible_ms = sum(r.total_m2n_transfer_time for r in requests) * 1000.0
    denom_total_ms = denom_ms + visible_ms

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None,
        "attn_tp": attn_tp, "attn_pp": attn_pp, "split": split,
        "denom_ms": denom_total_ms, "visible_ms": visible_ms,
        "tp_ms": tp_ms, "pp_ms": pp_ms,
        "num_decode_attn_rows": len([r for r in decode_rows if r["cluster_type"] == "DECODE_ATTN"]),
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(attn_tp: int, attn_pp: int, split: bool) -> dict:
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--attn-tp", str(attn_tp), "--attn-pp", str(attn_pp)]
        + (["--split"] if split else []),
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode}); see stderr above",
           "tag": f"tp{attn_tp}_pp{attn_pp}_{'split' if split else 'packed'}"}


def _report_row(r: dict) -> str:
    if r.get("error"):
        return f"[{r['tag']}] ERROR: {r['error']}"
    denom, vis, tp, pp = r["denom_ms"], r["visible_ms"], r["tp_ms"], r["pp_ms"]
    total_comm = vis + tp + pp
    headline = 100 * vis / total_comm if total_comm else float("nan")
    return (f"[{r['tag']:<16}] denom={denom:9.4f}ms visible={vis:9.4f}ms({100*vis/denom:5.2f}%) "
           f"tp_comm={tp:9.4f}ms({100*tp/denom:5.2f}%) pp_comm={pp:9.4f}ms({100*pp/denom:5.2f}%) "
           f"headline={headline:6.2f}%  rows={r['num_decode_attn_rows']}")


def _estimate_split_allreduce_cost(size_bytes: int) -> dict:
    """S4.3: what a same-sized payload would cost as one point-to-point hop
    across build_node_scale's own cross-domain link, vs within one domain --
    the same isolated_durations primitive every topology-aware predictor in
    this project already calls. A lower bound on a real ring-allreduce's
    extra cost from splitting (see module docstring)."""
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    within = Transfer(key="within", src=GpuId(0, 0), dst=GpuId(0, 1), size_bytes=size_bytes)
    across = Transfer(key="across", src=GpuId(0, 0), dst=GpuId(1, 0), size_bytes=size_bytes)
    durations = isolated_durations(fabric, [within, across])
    return {"size_bytes": size_bytes, "within_domain_ns": durations["within"],
           "cross_domain_ns": durations["across"],
           "ratio": durations["across"] / durations["within"]}


_PD_PP_RESULT_MARKER = "PD_PP_CHECK_RESULT="


def _run_pd_pp_check() -> None:
    """pd-af-disaggregation forbids pipeline parallelism outright for both
    DECODE_ATTN and DECODE_FFN (`frontier/config/config.py`: unconditional
    asserts, "Decode attention/FFN cluster must have 1 pipeline stage" --
    not this script's choice of flags, confirmed by the ValueError this
    script's own pp=2 attempt raises). Plain pd-disaggregation's unified
    DECODE cluster has no such restriction, so this check runs there
    instead, at pp=2, purely to establish whether
    `pipeline_parallel_communication_time` can ever be non-zero at all --
    it has no M2N transfer to compare against (pd-disaggregation doesn't
    split attention from FFN), so this is not a headline-ratio run, only
    the "is a non-zero PP figure reachable anywhere in this checkout"
    question S2.1 also asks."""
    from frontier.types import ClusterType
    fabric = build_node_scale(num_machines=1, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    d = Deployment("pd-pp-check")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE, 0, tp=1, pp=2))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.PREFILL: ClusterType.PREFILL,
                                      PoolKind.DECODE: ClusterType.DECODE})
    mapping = {d.replicas[0].ranks[0]: GpuId(0, 0)}
    for i, rank in enumerate(d.replicas[1].ranks):
        mapping[rank] = GpuId(0, 1 + i)
    placement = explicit(d, fabric, mapping)
    install(fabric, placement, d, reg)

    sys.argv = [
        "frontier.main", "--simulation_mode", "offline", "--sys_arch", "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_config_prefill_cluster_num_replicas", "1",
        "--cluster_config_decode_cluster_num_replicas", "1",
        "--cluster_config_prefill_replica_config_num_pipeline_stages", "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_prefill_replica_config_total_expert_num", str(TOTAL_EXPERTS),
        "--cluster_config_prefill_replica_config_router_topk", str(ROUTER_TOPK),
        "--cluster_config_prefill_replica_config_device", "h800",
        "--cluster_config_prefill_replica_config_memory_margin_fraction", "0.2",
        "--cluster_config_decode_replica_config_num_pipeline_stages", "2",
        "--cluster_config_decode_replica_config_attn_tensor_parallel_size", "1",
        "--cluster_config_decode_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_replica_config_moe_tensor_parallel_size", "1",
        "--cluster_config_decode_replica_config_moe_expert_parallel_size", "1",
        "--cluster_config_decode_replica_config_total_expert_num", str(TOTAL_EXPERTS),
        "--cluster_config_decode_replica_config_router_topk", str(ROUTER_TOPK),
        "--cluster_config_decode_replica_config_device", "h800",
        "--cluster_config_decode_replica_config_memory_margin_fraction", "0.2",
        "--cc_backend_config_type", "analytical",
        "--kv_cache_transfer_config_type", "empirical",
        "--replica_config_model_name", MODEL_NAME,
        "--replica_config_moe_routing_mode", "uniform_random",
        "--replica_config_moe_routing_seed", "42",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "1024",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", "128",
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "32",
        "--fixed_request_length_generator_config_decode_tokens", str(DECODE_TOKENS),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1.0",
        "--metrics_config_output_dir", str(OUTPUT_DIR),
        "--metrics_config_run_id", "pd_pp_check",
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds
    config = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)
    sim = Simulator(config)
    sim.run()

    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    decode_rows = [r for r in rows if r["cluster_type"] == "DECODE"]
    pp_total = sum(r["execution_time"]["component_ledger_ms"].get(
        "pipeline_parallel_communication_time", 0.0) for r in decode_rows)
    denom_total = sum(r["execution_time"]["total_time_ms"] for r in decode_rows)
    print(_PD_PP_RESULT_MARKER + json.dumps({
        "num_decode_rows": len(decode_rows), "pp_total_ms": pp_total,
        "denom_total_ms": denom_total,
        "share": pp_total / denom_total if denom_total else None,
    }), flush=True)


def _run_pd_pp_check_in_subprocess() -> dict:
    proc = subprocess.run([sys.executable, _SCRIPT_PATH, "--pd-pp-check"],
                          capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_PD_PP_RESULT_MARKER):
            return json.loads(line[len(_PD_PP_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode})"}


def main() -> int:
    print("=== 2.1: TP degree, packed (one domain) ===")
    for tp in TP_VALUES:
        r = _run_scenario_in_subprocess(tp, 1, split=False)
        print(_report_row(r))

    print()
    print("=== 2.1: PP degree, packed (one domain), tp=1 ===")
    for pp in PP_VALUES:
        r = _run_scenario_in_subprocess(1, pp, split=False)
        print(_report_row(r))

    print()
    print("=== 2.2: TP degree, split across two domains ===")
    packed_results, split_results = {}, {}
    for tp in TP_VALUES:
        rp = _run_scenario_in_subprocess(tp, 1, split=False)
        rs = _run_scenario_in_subprocess(tp, 1, split=True) if tp > 1 else rp
        packed_results[tp], split_results[tp] = rp, rs
        print(_report_row(rs))

    print()
    print("=== packed vs split, tp_comm and headline side by side ===")
    for tp in TP_VALUES:
        rp, rs = packed_results[tp], split_results[tp]
        if rp.get("error") or rs.get("error"):
            print(f"tp={tp}: packed_error={rp.get('error')} split_error={rs.get('error')}")
            continue
        same = rp["tp_ms"] == rs["tp_ms"]
        print(f"tp={tp:<2} packed_tp_comm={rp['tp_ms']:.6f}ms  split_tp_comm={rs['tp_ms']:.6f}ms  "
             f"identical={same}  packed_headline={100*rp['visible_ms']/(rp['visible_ms']+rp['tp_ms']+rp['pp_ms']):.2f}%  "
             f"split_headline={100*rs['visible_ms']/(rs['visible_ms']+rs['tp_ms']+rs['pp_ms']):.2f}%")

    print()
    print("=== 4.3: estimated true cost of a split allreduce hop (build_node_scale defaults) ===")
    for size_bytes in (65536, 1 << 20):
        est = _estimate_split_allreduce_cost(size_bytes)
        print(f"payload={size_bytes}B: within_domain={est['within_domain_ns']:.1f}ns  "
             f"cross_domain={est['cross_domain_ns']:.1f}ns  ratio={est['ratio']:.4f}x")

    print()
    print("=== pd-disaggregation PP=2 check (pp forbidden in pd-af-disaggregation) ===")
    r = _run_pd_pp_check_in_subprocess()
    print(r)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-tp", type=int, default=None)
    parser.add_argument("--attn-pp", type=int, default=1)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--pd-pp-check", action="store_true")
    args = parser.parse_args()
    if args.pd_pp_check:
        _run_pd_pp_check()
        raise SystemExit(0)
    if args.attn_tp is not None:
        _run_scenario(args.attn_tp, args.attn_pp, args.split)
        raise SystemExit(0)
    raise SystemExit(main())
