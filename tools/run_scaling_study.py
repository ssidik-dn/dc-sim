#!/usr/bin/env python3
"""Task 26 Part A: where does this project's own simulation stop being
affordable, and where does the wall-clock actually go?

**Fabric shape chosen, and why.** Growing fabric size by adding more
8-GPU domains keeps total link count exactly linear in GPU count (every
domain contributes a fixed `C(8,2)*2=56` links, independent of how many
domains exist) -- which contradicts this task's own framing ("link count
grows faster than GPU count"). Only *domain size* makes link count
superlinear (`C(n,2)*2 = n(n-1)` links per domain, `Fabric.add_link`'s
own `bidirectional=True` default storing both directions -- confirmed
directly: `build_rack_scale`'s 72-GPU Helios domain gives
`72*71=5112` links, matching this task's own "over five thousand" S2.3
claim exactly). So fabric size here means two domains, each half the
target GPU count, growing together --
`build_node_scale(num_machines=2, gpus_per_machine=n//2)` -- which
exercises the same quadratic-per-domain cost `build_rack_scale` does,
while still giving a genuine cross-domain placement to measure M2N
transfer cost over (real hardware does not run 512 GPUs in one
scale-up domain; this is a deliberately pessimistic shape chosen to
find where the *cost model itself* breaks, not a claim about a buildable
fabric).

**Real compute profiles throughout** (h800, Phi-tiny-MoE-instruct) --
per this task's own trap, dummy compute would make the wall-clock
ratios this task reports meaningless. The workload itself is kept
small and FIXED across every fabric-size point (16 requests, qps=10)
specifically to isolate the fabric-size effect from workload size,
which is axis 2.1's own third row's job, not this one's; that separate,
smaller sweep is `_duration_sweep()` below.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as every real-profile tool in this project:

    PYTHONPATH=/work/simulation/astra-sim:/work/simulation/Frontier:/work/simulation/dc-sim/src \\
        python3 tools/run_scaling_study.py

Nothing under `upstream/`, `src/engine/`, or the predictors is modified --
measurement only, per this task's own acceptance criteria.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

FRONTIER_ROOT = Path("/work/simulation/Frontier")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m2n_integration import _engine_deployment_and_registry  # noqa: E402
from run_blind_spot_probe import MODEL_NAME, TOTAL_EXPERTS, ROUTER_TOPK  # noqa: E402
from engine.physical.builders import build_node_scale  # noqa: E402
from engine.placement.placement import explicit  # noqa: E402
from engine.physical.topology import GpuId  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/scaling_study_outputs")

SCALE_UP_GBPS = 400.0
SCALE_OUT_GBPS = 50.0

FABRIC_SIZES = (32, 64, 128, 256, 512)
FIXED_NUM_REQUESTS = 16
FIXED_QPS = 10.0
GENEROUS_NUM_BLOCKS = 4096


def _fabric_for(n_gpus: int):
    return build_node_scale(num_machines=2, gpus_per_machine=n_gpus // 2,
                            scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)


def _placement_for(fabric, deployment):
    prefill_rank = deployment.replicas[0].ranks[0]
    attn_rank = deployment.replicas[1].ranks[0]
    ffn_rank = deployment.replicas[2].ranks[0]
    return explicit(deployment, fabric, {
        prefill_rank: GpuId(0, 0), attn_rank: GpuId(0, 1), ffn_rank: GpuId(1, 0)})


def _argv(run_id: str, num_requests: int, qps: float) -> list[str]:
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

        "--cluster_config_decode_attn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", "1",
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

        "--vllm_v1_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold", "0",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks", str(GENEROUS_NUM_BLOCKS),
        "--no-vllm_v1_scheduler_config_enable_chunked_prefill",

        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(num_requests),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", "16",
        "--fixed_request_length_generator_config_decode_tokens", "8",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(qps),

        "--metrics_config_output_dir", str(OUTPUT_DIR),
        "--metrics_config_run_id", run_id,
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]


_RESULT_MARKER = "SCALING_RESULT="


def _run_scenario(n_gpus: int, num_requests: int, qps: float, seed: int,
                  profile: bool = False) -> None:
    fab_build_start = time.perf_counter()
    fabric = _fabric_for(n_gpus)
    deployment, registry = _engine_deployment_and_registry()
    placement = _placement_for(fabric, deployment)
    install(fabric, placement, deployment, registry)
    fab_build_s = time.perf_counter() - fab_build_start

    tag = f"scaling_n{n_gpus}_req{num_requests}_qps{qps}_seed{seed}"
    sys.argv = _argv(tag, num_requests, qps) + ["--seed", str(seed)]
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    error = None
    sim = None
    wall_s = None
    profiler = None
    try:
        config = SimulationConfig.create_from_cli_args()
        assert not config.cluster_config.execution_time_predictor_config.enable_dummy_mode
        set_seeds(config.seed)
        sim = Simulator(config)
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
        run_start = time.perf_counter()
        sim.run()
        wall_s = time.perf_counter() - run_start
        if profiler is not None:
            profiler.disable()
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if error is not None:
        print(_RESULT_MARKER + json.dumps({
            "tag": tag, "error": error, "n_gpus": n_gpus,
            "fabric_build_s": fab_build_s, "peak_rss_kb": peak_rss_kb,
        }), flush=True)
        return

    requests = sim._all_requests
    completed = [r for r in requests if r.completed]
    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    from integration.context import get_context
    n_links = len(get_context().fabric.links)

    print(_RESULT_MARKER + json.dumps({
        "tag": tag, "error": None, "n_gpus": n_gpus, "n_links": n_links,
        "num_requests": num_requests, "qps": qps, "seed": seed,
        "n_completed": len(completed), "n_operations": len(rows),
        "fabric_build_s": fab_build_s, "wall_s": wall_s,
        "wall_per_request_ms": (wall_s / len(completed) * 1000.0) if completed else None,
        "wall_per_operation_ms": (wall_s / len(rows) * 1000.0) if rows else None,
        "peak_rss_kb": peak_rss_kb,
    }), flush=True)

    if profiler is not None:
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(60)
        (OUTPUT_DIR / f"{tag}.profile.txt").write_text(buf.getvalue())
        print(f"PROFILE_WRITTEN={OUTPUT_DIR / f'{tag}.profile.txt'}", flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(n_gpus: int, num_requests: int = FIXED_NUM_REQUESTS,
                                qps: float = FIXED_QPS, seed: int = 0,
                                profile: bool = False, timeout: float = 900.0) -> dict:
    argv = [sys.executable, _SCRIPT_PATH, "--n-gpus", str(n_gpus),
           "--num-requests", str(num_requests), "--qps", str(qps), "--seed", str(seed)]
    if profile:
        argv.append("--profile")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(FRONTIER_ROOT),
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s", "n_gpus": n_gpus}
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            result = json.loads(line[len(_RESULT_MARKER):])
            for line2 in proc.stdout.splitlines():
                if line2.startswith("PROFILE_WRITTEN="):
                    result["profile_path"] = line2[len("PROFILE_WRITTEN="):]
            return result
    sys.stderr.write(proc.stdout[-4000:])
    sys.stderr.write(proc.stderr[-4000:])
    return {"error": f"no result (exit code {proc.returncode}); see stderr above", "n_gpus": n_gpus}


def _fit_exponent(xs: list[float], ys: list[float]) -> float:
    """Slope of log(y) vs log(x) across the whole sweep -- a single global
    fit. Kept for reference only: task 30's own finding is that this
    number hides a convex curve (a global fit averages a shallow start
    against a steep finish), and is not what should be read as "the"
    exponent -- see `_per_doubling_exponents` below, which is."""
    import math
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float("nan")


def _per_doubling_exponents(xs: list[float], ys: list[float]) -> list[tuple[float, float, float]]:
    """The exponent for each consecutive doubling: `log2(y[i+1]/y[i])`,
    valid when `xs` doubles step to step (this sweep's own `FABRIC_SIZES`
    does). This is the number that describes behaviour at scale for a
    convex curve -- task 30's own correction to task 26's single global
    fit, which averaged a shallow start against a steep finish and
    reported the average as "effectively linear" when the growth at the
    largest doubling was still ~2.5. Returns `(x_from, x_to, exponent)`
    triples."""
    import math
    out = []
    for i in range(len(xs) - 1):
        ratio = ys[i + 1] / ys[i] if ys[i] else float("nan")
        out.append((xs[i], xs[i + 1], math.log2(ratio) if ratio > 0 else float("nan")))
    return out


def _fabric_size_sweep() -> None:
    print("=== 2.1 fabric-size sweep (fixed workload: "
         f"{FIXED_NUM_REQUESTS} requests, qps={FIXED_QPS}) ===")
    rows = []
    for n in FABRIC_SIZES:
        r = _run_scenario_in_subprocess(n, profile=(n == FABRIC_SIZES[-1]))
        rows.append(r)
        if r.get("error"):
            print(f"[n_gpus={n:>4}] ERROR: {r['error']}")
            continue
        print(f"[n_gpus={n:>4}] n_links={r['n_links']:>7} "
             f"fabric_build_s={r['fabric_build_s']:.4f} wall_s={r['wall_s']:.4f} "
             f"wall/req={r['wall_per_request_ms']:.3f}ms "
             f"wall/op={r['wall_per_operation_ms']:.4f}ms "
             f"peak_rss_mb={r['peak_rss_kb']/1024:.1f}")

    ok = [r for r in rows if not r.get("error")]
    if len(ok) >= 2:
        xs = [r["n_gpus"] for r in ok]
        wall = [r["wall_s"] for r in ok]
        links = [r["n_links"] for r in ok]
        rss = [r["peak_rss_kb"] for r in ok]

        print()
        print("=== per-doubling exponent (the one that describes behaviour at scale) ===")
        for (x0, x1, e_wall), (_, _, e_links), (_, _, e_rss) in zip(
                _per_doubling_exponents(xs, wall),
                _per_doubling_exponents(xs, links),
                _per_doubling_exponents(xs, rss)):
            print(f"  {x0:>4} -> {x1:>4}: wall_s x{wall[xs.index(x1)]/wall[xs.index(x0)]:.2f} "
                 f"(exp {e_wall:+.2f})  n_links exp {e_links:+.2f}  peak_rss exp {e_rss:+.2f}")

        exp_wall = _fit_exponent(xs, wall)
        exp_links = _fit_exponent(xs, links)
        exp_rss = _fit_exponent(xs, rss)
        print()
        print(f"global fitted exponent (reference only -- see per-doubling above for "
             f"the number that matters on a convex curve): "
             f"wall_s ~ n_gpus^{exp_wall:.2f}  "
             f"n_links ~ n_gpus^{exp_links:.2f}  peak_rss ~ n_gpus^{exp_rss:.2f}")
    return rows


def _concurrent_flows_sweep() -> None:
    print()
    print("=== 2.1 concurrent-flows sweep (fixed fabric: n_gpus=64) ===")
    for qps in (5.0, 20.0, 50.0, 100.0):
        r = _run_scenario_in_subprocess(64, num_requests=32, qps=qps)
        if r.get("error"):
            print(f"[qps={qps:>6}] ERROR: {r['error']}")
            continue
        print(f"[qps={qps:>6}] wall_s={r['wall_s']:.4f} n_operations={r['n_operations']} "
             f"wall/op={r['wall_per_operation_ms']:.4f}ms n_completed={r['n_completed']}")


def _duration_sweep() -> None:
    print()
    print("=== 2.1 simulated-duration sweep (fixed fabric: n_gpus=64, qps=10) ===")
    for num_requests in (8, 16, 32, 64, 128):
        r = _run_scenario_in_subprocess(64, num_requests=num_requests, qps=10.0)
        if r.get("error"):
            print(f"[requests={num_requests:>4}] ERROR: {r['error']}")
            continue
        print(f"[requests={num_requests:>4}] wall_s={r['wall_s']:.4f} "
             f"wall/req={r['wall_per_request_ms']:.3f}ms fabric_build_s={r['fabric_build_s']:.4f}")


def main() -> int:
    _fabric_size_sweep()
    _concurrent_flows_sweep()
    _duration_sweep()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gpus", type=int, default=None)
    parser.add_argument("--num-requests", type=int, default=FIXED_NUM_REQUESTS)
    parser.add_argument("--qps", type=float, default=FIXED_QPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.n_gpus is not None:
        _run_scenario(args.n_gpus, args.num_requests, args.qps, args.seed, profile=args.profile)
        raise SystemExit(0)
    raise SystemExit(main())
