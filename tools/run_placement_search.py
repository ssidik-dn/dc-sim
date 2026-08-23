#!/usr/bin/env python3
"""Task 32: exhaustive search over tensor-parallel degree and physical
placement, objective = mean per-token latency (SLO attainment and
throughput reported alongside, not optimised).

**Seed policy (task 32 spec S1), decided explicitly, not by default.**
Each arrangement is evaluated **once**, in the deterministic
configuration every tool in this project already uses (task 31 report,
S1.3: fixed request lengths, arrivals submitted at once -- no
seed-dependent input exists in this configuration at all, so a second
evaluation of the same arrangement would reproduce the first bit for
bit, not add information). This is what makes exhaustive search of the
whole space affordable in the first place. The winner's margin over the
runner-up is then re-run **with genuine seed variance** (task 31's own
`seed_stats` module -- staggered arrivals, matching request-generator
seed) for the top candidates specifically, so the one number search
cannot see on its own (whether its own ranking survives realistic
noise) gets an interval before anything is called a winner.

**The search space (task 32 S2).** Two dimensions: tensor-parallel
degree (1, 2, 4, 8) and physical placement of the resulting group,
searched as **distinct canonical shapes** (`Placement.group_shape()`),
not distinct concrete placements -- task 32's own point, and task 15's
own `group_shape()` docstring: "isomorphic placements collapse to one
memoisation key." Candidate concrete placements are generated from
this project's own existing policies
(`engine.placement.placement.packed`/`spread`/`fragmented`, not
hand-built partitions) -- `packed` and `spread` give the two extremes
(one domain; maximally spread), and `fragmented(seed=k)` over many
seeds discovers whatever irregular shapes a real, under-load scheduler
could produce, exactly as its own docstring describes. Deduplicated by
`group_shape()` of the DECODE_ATTN replica's own TP group before any
one of them is evaluated -- how many concrete placements collapsed to
how many shapes is itself reported (S4).

**Memory as a feasibility filter, not a dimension (task 32 S2).**
`--cluster_config_decode_attn_replica_config_memory_margin_fraction 0.992`
throughout -- task 28's own established point where tp=1 is
infeasible (parameter memory alone exceeds the budget) while
tp=2/4/8 are not merely feasible but already past task 22's own
plateau. Feasibility, and the resulting `num_blocks` for each feasible
degree, is computed from the same calibrated formula tasks 25/26/28
already validated against real `MemoryPlanner` behaviour (cited, not
re-derived): tp=2 -> num_blocks=30, tp=4 -> 1341, tp=8 -> 1853,
tp=1 -> infeasible, rejected before any placement of it is generated
or evaluated.

Real h800 compute profiles throughout (Phi-tiny-MoE-instruct);
`install(..., collective=True)` for placement-sensitive
`tensor_parallel_communication_time`, matching every real-compute tool
here since task 20.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_stats import seed_argv_fix, compute_interval_stats  # noqa: E402
from run_tp_domain_probe import (  # noqa: E402
    MODEL_NAME, ROUTER_TOPK, TOTAL_EXPERTS, TP_KEYS,
    _deployment_and_registry)
from engine.physical.builders import build_node_scale  # noqa: E402
from engine.placement.placement import packed, spread, fragmented, explicit  # noqa: E402
from engine.logical.deployment import ParallelKind  # noqa: E402
from integration.install import install  # noqa: E402

FRONTIER_ROOT = Path("/work/simulation/Frontier")
_SCRIPT_PATH = str(Path(__file__).resolve())

SCALE_UP_GBPS = 400.0
SCALE_OUT_GBPS = 50.0

# 5 domains x 4 GPUs = 20 GPUs: small enough that every evaluation is a
# couple of seconds (task 29/30's own fitted growth applies), large
# enough that tp=8 (needs >4 GPUs across domains no bigger than 4) has
# a genuinely varied set of reachable shapes, not just "fits or does
# not."
NUM_DOMAINS = 5
DOMAIN_SIZE = 4

TP_VALUES = (1, 2, 4, 8)
MARGIN = 0.992  # task 28's own established tp=1-infeasible / tp>=2-plateau point
# Cited from tasks 25/26/28's own calibrated formula, not re-derived:
# param_mem(tp) + margin's implied overhead vs 80 GiB * (1-margin).
FEASIBLE_NUM_BLOCKS = {2: 30, 4: 1341, 8: 1853}  # tp=1 absent: infeasible

NUM_REQUESTS = 32
QPS = 20.0
BLOCK_SIZE = 16
PREFILL_TOKENS = 32
DECODE_TOKENS = 16
GENEROUS_NUM_BLOCKS = 4096

# SLO: stated explicitly, since nothing in this project has set one
# before. 15 ms/token sits inside the range this project's own real
# h800 measurements have actually produced across tasks 22-31 (roughly
# 3-45 ms/token depending on configuration) rather than a number picked
# to be trivially met or trivially missed by everything -- it is
# illustrative, not derived from an external spec, and is reported as
# such rather than presented as an authoritative target.
SLO_TPOT_MS = 15.0


def _fabric():
    return build_node_scale(num_machines=NUM_DOMAINS, gpus_per_machine=DOMAIN_SIZE,
                            scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)


def _attn_tp_group(deployment):
    attn_replica = deployment.replicas[1]
    return attn_replica.groups(ParallelKind.TP)[0]


def enumerate_shapes(attn_tp: int, n_fragmented_seeds: int = 60):
    """Every distinct `group_shape()` reachable for DECODE_ATTN's own TP
    group at this degree, on this fabric, via this project's own
    existing placement policies -- and one concrete `Placement`
    realising each. Returns `(shapes: Dict[shape, Placement], n_candidates: int)`.
    """
    fabric = _fabric()
    deployment, registry = _deployment_and_registry(attn_tp, 1)
    group = _attn_tp_group(deployment)

    candidates = []
    try:
        candidates.append(packed(deployment, fabric))
    except Exception:  # noqa: BLE001
        pass
    try:
        candidates.append(spread(deployment, fabric))
    except Exception:  # noqa: BLE001
        pass
    # packed()'s own rank order gives PREFILL the first slot in domain 0,
    # so DECODE_ATTN's own group starts offset by one and never reaches a
    # clean single-domain shape even when attn_tp <= DOMAIN_SIZE (e.g.
    # tp=4 packed() gives ATTN (3,1), not (4,)) -- the same reason tasks
    # 19-31 built PREFILL/FFN's placement explicitly, in a domain
    # separate from the group under study, rather than relying on a
    # deployment-wide policy. Added here explicitly for the same reason:
    # a genuine, reachable "fits in one domain" reference point is what
    # this task's own S5 needs to validate against the established
    # result, and it does not exist in this fabric via packed() alone.
    if attn_tp <= DOMAIN_SIZE:
        prefill_rank = deployment.replicas[0].ranks[0]
        ffn_rank = deployment.replicas[2].ranks[0]
        attn_ranks = group.ranks
        from engine.physical.topology import GpuId
        mapping = {prefill_rank: GpuId(0, 0), ffn_rank: GpuId(0, 1)}
        for i, r in enumerate(attn_ranks):
            mapping[r] = GpuId(1, i)
        try:
            candidates.append(explicit(deployment, fabric, mapping))
        except Exception:  # noqa: BLE001
            pass
    for seed in range(n_fragmented_seeds):
        try:
            candidates.append(fragmented(deployment, fabric, seed=seed))
        except Exception:  # noqa: BLE001
            pass

    shapes = {}
    for p in candidates:
        shape = p.group_shape(group)
        if shape not in shapes:
            shapes[shape] = p
    return shapes, len(candidates)


def _argv(run_id: str, attn_tp: int, num_blocks: int, seed: int, extra: list[str]) -> list[str]:
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

        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", str(attn_tp),
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_device", "h800",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", str(MARGIN),

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

        "--vllm_v1_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", str(BLOCK_SIZE),
        "--vllm_v1_scheduler_config_num_blocks", str(GENEROUS_NUM_BLOCKS),
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        "--cluster_config_prefill_replica_scheduler_config_num_blocks", str(GENEROUS_NUM_BLOCKS),
        "--cluster_config_decode_attn_replica_scheduler_config_num_blocks", str(num_blocks),
        "--cluster_config_decode_attn_replica_scheduler_config_block_size", str(BLOCK_SIZE),

        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(PREFILL_TOKENS),
        "--fixed_request_length_generator_config_decode_tokens", str(DECODE_TOKENS),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(QPS),

        "--metrics_config_output_dir", "/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0/scratchpad/placement_search_outputs",
        "--metrics_config_run_id", run_id,
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",

        "--seed", str(seed),
    ] + extra


_RESULT_MARKER = "SEARCH_RESULT="


def _run_scenario(attn_tp: int, shape_key: str, num_blocks: int, seed: int, seeded: bool) -> None:
    from run_tp_domain_probe import _deployment_and_registry as _dep_fn
    fabric = _fabric()
    deployment, registry = _dep_fn(attn_tp, 1)

    # Rebuild the same candidate pool and pick the placement matching
    # shape_key -- avoids serialising a Placement object across the
    # subprocess boundary; regenerating is cheap and deterministic
    # (packed/spread always agree; fragmented(seed=k) is seeded).
    shapes, _ = enumerate_shapes(attn_tp)
    shape = tuple(int(x) for x in shape_key.split(","))
    placement = shapes[shape]
    # enumerate_shapes() built its own fabric/deployment; re-resolve the
    # placement's mapping onto THIS fabric/deployment pair directly.
    from engine.placement.placement import explicit
    mapping = {r: placement.gpu(r) for r in deployment.ranks}
    placement = explicit(deployment, fabric, mapping)

    install(fabric, placement, deployment, registry, collective=True)

    extra = seed_argv_fix(seed) if seeded else []
    tag = f"search_tp{attn_tp}_{shape_key.replace(',', '-')}_seed{seed}_seeded{int(seeded)}"
    sys.argv = _argv(tag, attn_tp, num_blocks, seed, extra)

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

    requests = sim._all_requests
    completed = [r for r in requests if r.completed]
    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    decode_rows = [r for r in rows if r["cluster_type"] in ("DECODE_ATTN", "DECODE_FFN")]
    tp_ms = sum(sum(r["execution_time"]["component_ledger_ms"].get(k, 0.0) for k in TP_KEYS)
               for r in decode_rows)
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    tpot_ms_per_req = [r.tpot * 1000.0 for r in tpot_eligible]
    mean_tpot_ms = statistics.mean(tpot_ms_per_req) if tpot_ms_per_req else None
    slo_met = (sum(1 for t in tpot_ms_per_req if t <= SLO_TPOT_MS) / len(tpot_ms_per_req)
              if tpot_ms_per_req else None)
    wall_s = max((r.completed_at for r in completed), default=0.0)
    throughput_rps = len(completed) / wall_s if wall_s else 0.0

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "attn_tp": attn_tp, "shape": shape_key,
        "mean_tpot_ms": mean_tpot_ms, "throughput_rps": throughput_rps,
        "slo_attainment": slo_met, "tp_comm_ms": tp_ms,
        "n_completed": len(completed),
    }), flush=True)


def _run_scenario_in_subprocess(attn_tp: int, shape: tuple, num_blocks: int,
                                seed: int = 0, seeded: bool = False) -> dict:
    shape_key = ",".join(str(x) for x in shape)
    proc = subprocess.run(
        [sys.executable, _SCRIPT_PATH, "--attn-tp", str(attn_tp), "--shape", shape_key,
         "--num-blocks", str(num_blocks), "--seed", str(seed), "--seeded", "1" if seeded else "0"],
        capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout[-3000:])
    sys.stderr.write(proc.stderr[-3000:])
    return {"error": f"no result (exit code {proc.returncode})", "attn_tp": attn_tp, "shape": shape_key}


def main() -> int:
    print("=== S2: feasibility filter (margin=0.992, task 28's own point) ===")
    for tp in TP_VALUES:
        if tp in FEASIBLE_NUM_BLOCKS:
            print(f"  tp={tp}: feasible, num_blocks={FEASIBLE_NUM_BLOCKS[tp]}")
        else:
            print(f"  tp={tp}: INFEASIBLE (parameter memory alone exceeds the budget "
                 f"at margin={MARGIN}) -- rejected, no placement generated or evaluated")

    print()
    print("=== S4: candidate placements -> distinct shapes, per feasible degree ===")
    all_shapes = {}
    total_candidates = 0
    total_shapes = 0
    for tp in TP_VALUES:
        if tp not in FEASIBLE_NUM_BLOCKS:
            continue
        shapes, n_candidates = enumerate_shapes(tp)
        all_shapes[tp] = shapes
        total_candidates += n_candidates
        total_shapes += len(shapes)
        print(f"  tp={tp}: {n_candidates} candidate placements -> {len(shapes)} distinct shapes: "
             f"{sorted(shapes.keys(), reverse=True)}")
    print(f"  TOTAL: {total_candidates} candidates -> {total_shapes} distinct shapes "
         f"({total_candidates/total_shapes:.1f}x collapse)")

    print()
    print("=== evaluating every distinct shape once (deterministic configuration) ===")
    results = []
    for tp in TP_VALUES:
        if tp not in FEASIBLE_NUM_BLOCKS:
            continue
        for shape in all_shapes[tp]:
            r = _run_scenario_in_subprocess(tp, shape, FEASIBLE_NUM_BLOCKS[tp])
            results.append(r)
            if r.get("error"):
                print(f"  [tp={tp} shape={shape}] ERROR: {r['error']}")
            else:
                print(f"  [tp={tp} shape={shape}] mean_tpot_ms={r['mean_tpot_ms']:.4f} "
                     f"throughput={r['throughput_rps']:.3f}req/s "
                     f"slo_attainment={r['slo_attainment']:.3f} "
                     f"tp_comm_ms={r['tp_comm_ms']:.4f}")

    ok = [r for r in results if not r.get("error")]
    ok.sort(key=lambda r: r["mean_tpot_ms"])

    print()
    print("=== ranked by mean per-token latency (the objective) ===")
    for i, r in enumerate(ok):
        print(f"  #{i+1}: tp={r['attn_tp']} shape={r['shape']} "
             f"mean_tpot_ms={r['mean_tpot_ms']:.4f} throughput={r['throughput_rps']:.3f}req/s "
             f"slo_attainment={r['slo_attainment']:.3f}")

    print()
    print("=== S1/S6.3: seeded re-run of the top 3, with intervals ===")
    top = ok[:3]
    for r in top:
        tp = r["attn_tp"]
        shape = tuple(int(x) for x in r["shape"].split(","))
        seeded_rows = [_run_scenario_in_subprocess(tp, shape, FEASIBLE_NUM_BLOCKS[tp], seed=s, seeded=True)
                      for s in range(20)]
        seeded_ok = [x for x in seeded_rows if not x.get("error")]
        if len(seeded_ok) < 2:
            print(f"  tp={tp} shape={r['shape']}: too few successful seeded runs to compute an interval")
            continue
        stats = compute_interval_stats([x["mean_tpot_ms"] for x in seeded_ok])
        print(f"  tp={tp} shape={r['shape']}: n={stats.n} mean={stats.mean:.4f}ms "
             f"stdev={stats.stdev:.4f} ci95_halfwidth={stats.ci95_halfwidth:.4f} "
             f"({stats.ci95_halfwidth_pct:.2f}% of mean)")

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-tp", type=int, default=None)
    parser.add_argument("--shape", type=str, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeded", type=int, default=0)
    args = parser.parse_args()
    if args.attn_tp is not None and args.shape is not None:
        _run_scenario(args.attn_tp, args.shape, args.num_blocks, args.seed, bool(args.seeded))
        raise SystemExit(0)
    raise SystemExit(main())
