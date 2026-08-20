#!/usr/bin/env python3
"""Task 18: how much of a decode step's communication does this project
actually see?

Task 17 drew the boundary precisely: activation exchange between
DECODE_ATTN and DECODE_FFN is priced by this project, from the fabric, with
topology. Everything inside a replica -- tensor-parallel allreduce,
pipeline-parallel send/recv, expert-parallel dispatch -- is priced by
Frontier's own execution-time predictor from profiled figures, with no
notion of where the participating GPUs sit (reachable only through
`CCBackendType`, closed since task 06). This script measures how big that
second, invisible category actually is, and whether it varies with
placement -- the two different questions task 18 spec S1.1 insists on
keeping separate.

**The ledger.** Frontier's own per-batch-stage "component ledger"
(`frontier/metrics/metrics_store.py::_build_frontier_stage_batch_component_ledger`)
already decomposes every DECODE_ATTN/DECODE_FFN batch stage into named
components in milliseconds, and *enforces* (an assertion in
`_build_frontier_stage_batch_ledger_row`) that the components sum exactly
to `total_time_ms` -- a clean, closed decomposition, not an approximation
this script has to construct. Rows accumulate in-memory at
`sim._metric_store._frontier_stage_batch_ledger_rows` regardless of the
`--no-metrics_config_*` disk-writing flags this project's other tools use
(a *separate* config flag, `store_frontier_stage_batch_ledger`, defaults to
True and is left alone here).

**`tensor_parallel_communication_time` does not exist as a ledger key.**
Confirmed by reading `frontier/entities/execution_time.py`: that name is a
constructor parameter for a *legacy*, unsplit accounting path
(`legacy_tensor_parallel_allreduce_time`), used only when neither
`attn_tensor_parallel_allreduce_time` nor `moe_tensor_parallel_allreduce_time`
is supplied directly. The modern, split ledger keys this script actually
reads are `attention_all_reduce_time` + `mlp_all_reduce_time` +
`moe_tensor_parallel_allgather_time` + `share_expert_tensor_parallel_allreduce_time`
-- all four are `trace_kind=TraceKind.COMM` / `resource_class=ResourceClass.COMM`
in `frontier/operators/families.py`'s `COMM_FAMILY`, confirming they are
communication, not an assumption from the name alone.

**`moe_shuffling_time` is compute, not communication** -- also confirmed
from `frontier/operators/families.py`'s `MOE_FAMILY` entry for
`moe_shuffling`: `trace_kind=TraceKind.COMPUTE`, `role=OperatorRole.RESHAPE`,
`resource_class=ResourceClass.MEMORY`, and `ep_agnostic=True` (it does not
even vary with expert-parallel degree -- a local token-reordering kernel,
not a cross-device transfer). Classified into the denominator's "neither"
bucket by default; reported both ways per the spec's own instruction, since
the classification is a judgment call worth being able to check.

**Units**, on the same footing throughout (a trap this project has hit
before -- task 17 S0/A.1): every ledger component is a per-batch-stage
total in milliseconds, already aggregated across a decode step's layers by
Frontier itself (see `attention_all_reduce_time`'s own docstring:
"aggregated across all layers"). `request.total_m2n_transfer_time` is a
per-*request* total across the whole decode phase (not per step), so it is
divided by that request's own decode-step count (the number of
DECODE_ATTN ledger rows attributed to it) before being combined with the
per-step ledger components -- not divided by `decode_tokens` directly,
which would assume every generated token corresponds to exactly one
ledger row and this script confirms that count rather than assuming it.

Environment: run from the dc-sim root; cwd is set to FRONTIER_ROOT
internally, same as every real-profile tool in this project:

    PYTHONPATH=/work/dc-sim/src:/work/Frontier python3 tools/run_blind_spot_probe.py

Real compute profiles throughout -- no dummy-mode flag anywhere in this
script's argv, per task 18's own explicit warning that dummy mode inflates
the denominator roughly seventyfold and makes every ratio meaningless
(task 12's own finding, restated as this task's first trap). Nothing under
`upstream/`, `src/engine/`, or the predictors is modified -- measurement
only, per task 18's acceptance criteria.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

FRONTIER_ROOT = Path("/work/simulation/Frontier")

from engine.logical.deployment import Deployment, PoolKind, Replica  # noqa: E402
from engine.physical.builders import build_node_scale  # noqa: E402
from engine.physical.topology import GpuId  # noqa: E402
from engine.placement.placement import explicit  # noqa: E402

from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment  # noqa: E402
from integration.install import install  # noqa: E402

OUTPUT_DIR = Path("/tmp/claude-1001/-work-dc-sim/e8e62237-6408-4b18-a2fe-76ed0916f3d0"
                  "/scratchpad/blind_spot_probe_outputs")

MODEL_NAME = "Phi-tiny-MoE-instruct"   # 16 experts; real h800 profiles present
TOTAL_EXPERTS = 16
ROUTER_TOPK = 2
NUM_REQUESTS = 8
DECODE_TOKENS = 8
SCALE_UP_GBPS = 400.0
SCALE_OUT_GBPS = 50.0

# The invisible (priced-without-topology) communication components -- the
# modern, split equivalents of the spec's four named fields (see module
# docstring for why "tensor_parallel_communication_time" isn't a literal
# ledger key).
TP_KEYS = ("attention_all_reduce_time", "mlp_all_reduce_time",
          "moe_tensor_parallel_allgather_time", "share_expert_tensor_parallel_allreduce_time")
PP_KEYS = ("pipeline_parallel_communication_time",)
EP_KEYS = ("expert_parallel_communication_time",)
MOE_SHUFFLE_KEYS = ("moe_shuffling_time",)
# Data-parallel allreduce: not one of the spec's four, tracked separately
# and expected to be ~0 throughout (every scenario here uses dp=1).
DP_KEYS = ("dp_input_allreduce_time", "dp_output_allreduce_time")

INVISIBLE_KEYS = TP_KEYS + PP_KEYS + EP_KEYS


def _deployment_and_registry(attn_tp: int, ffn_tp: int, ffn_ep: int, split_ep: bool):
    from frontier.types import ClusterType
    d = Deployment("blind-spot-probe")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=attn_tp))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=ffn_tp, ep=ffn_ep))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {
        PoolKind.PREFILL: ClusterType.PREFILL,
        PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
        PoolKind.DECODE_FFN: ClusterType.DECODE_FFN,
    })
    return d, reg


def _placement(fabric, deployment, label: str, split_ep: bool):
    """`label` picks the pool-crossing (M2N) placement this project prices:
    "colocated" (attn+ffn share a machine) or "split" (ffn on a different
    machine). `split_ep`, independent of that, spreads the FFN replica's
    own EP ranks across two machines instead of colocating them -- the
    S1.1 case ("experts spread across FFN replicas that sit in different
    domains") this script uses to test whether `expert_parallel_communication_time`
    notices at all."""
    prefill_rank = deployment.replicas[0].ranks[0]
    attn_ranks = deployment.replicas[1].ranks   # attn_tp ranks
    ffn_ranks = deployment.replicas[2].ranks    # ffn_tp * ffn_ep ranks

    mapping = {prefill_rank: GpuId(0, 0)}
    for i, rank in enumerate(attn_ranks):
        mapping[rank] = GpuId(0, 1 + i)

    ffn_machine = 0 if label == "colocated" else 1
    if split_ep:
        half = len(ffn_ranks) // 2
        for i, rank in enumerate(ffn_ranks[:half]):
            mapping[rank] = GpuId(ffn_machine, 8 + i)
        for i, rank in enumerate(ffn_ranks[half:]):
            mapping[rank] = GpuId(ffn_machine + 2, i)   # a third, separate machine
    else:
        for i, rank in enumerate(ffn_ranks):
            mapping[rank] = GpuId(ffn_machine, 8 + i)

    return explicit(deployment, fabric, mapping)


def _argv(run_id: str, attn_tp: int, ffn_tp: int, ffn_ep: int) -> list[str]:
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
        "--cluster_config_decode_attn_replica_config_attn_tensor_parallel_size", str(attn_tp),
        "--cluster_config_decode_attn_replica_config_attn_data_parallel_size", "1",
        "--cluster_config_decode_attn_replica_config_device", "h800",
        "--cluster_config_decode_attn_replica_config_memory_margin_fraction", "0.2",

        "--cluster_config_decode_ffn_replica_config_num_pipeline_stages", "1",
        "--cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size", str(ffn_tp),
        "--cluster_config_decode_ffn_replica_config_moe_expert_parallel_size", str(ffn_ep),
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
        # "simulation" resolves to a profiling routing path this model's
        # data doesn't have (task 17's own finding, reconfirmed while
        # building this probe); "uniform_random" resolves to the path it
        # does have ("uniform_topk").
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
        # write_metrics must stay True here, unlike every other tool in
        # this project: metrics_store.py gates the Frontier stage-batch
        # ledger's in-memory capture behind `self._config.write_metrics`
        # itself (confirmed by running it: rows list was empty with it
        # off, not merely "not written to disk") -- this is the one probe
        # that actually needs the ledger, so the granular store_* flags
        # below turn off disk writing for everything else instead.
        "--no-metrics_config_store_request_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        # store_utilization_metrics must also stay True: the replica-stage
        # end hook that completes a pending ledger row into
        # `_frontier_stage_batch_ledger_rows` is gated behind it with an
        # early return covering *both* concerns (confirmed by running it:
        # 216 rows created, 0 completed, until this flag went back on) --
        # an undocumented coupling between two config flags that otherwise
        # sound unrelated.
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        # deliberately no dummy-mode flags -- real compute throughout (S6's
        # own trap: "do not let dummy mode near this").
    ]


_RESULT_MARKER = "BLIND_SPOT_RESULT="


def _run_scenario(label: str, attn_tp: int, ffn_tp: int, ffn_ep: int, split_ep: bool) -> None:
    fabric = build_node_scale(num_machines=4, gpus_per_machine=16,
                              scale_up_GBps=SCALE_UP_GBPS, scale_out_GBps=SCALE_OUT_GBPS)
    deployment, registry = _deployment_and_registry(attn_tp, ffn_tp, ffn_ep, split_ep)
    placement = _placement(fabric, deployment, label, split_ep)
    install(fabric, placement, deployment, registry)

    tag = f"attn_tp{attn_tp}_ffn_tp{ffn_tp}_ep{ffn_ep}{'_epsplit' if split_ep else ''}_{label}"
    sys.argv = _argv(tag, attn_tp, ffn_tp, ffn_ep)
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds

    error = None
    sim = None
    try:
        config = SimulationConfig.create_from_cli_args()
        assert not config.cluster_config.execution_time_predictor_config.enable_dummy_mode, (
            "dummy mode must stay off for this probe (S6)")
        set_seeds(config.seed)
        sim = Simulator(config)
        sim.run()
    except Exception as e:  # noqa: BLE001 -- report whatever happens
        error = f"{type(e).__name__}: {e}"

    if error is not None:
        print(_RESULT_MARKER + json.dumps({"tag": tag, "error": error}), flush=True)
        return

    from frontier.types import ClusterType
    rows = sim._metric_store._frontier_stage_batch_ledger_rows
    decode_rows = [r for r in rows if r["cluster_type"] in ("DECODE_ATTN", "DECODE_FFN")]

    bucket_totals = {}
    for key in INVISIBLE_KEYS + MOE_SHUFFLE_KEYS + DP_KEYS:
        bucket_totals[key] = sum(r["execution_time"]["component_ledger_ms"].get(key, 0.0)
                                 for r in decode_rows)
    denom_ledger_ms = sum(r["execution_time"]["total_time_ms"] for r in decode_rows)
    num_decode_rows = len([r for r in decode_rows if r["cluster_type"] == "DECODE_ATTN"])

    requests = sim._all_requests
    visible_ms_total = sum(r.total_m2n_transfer_time for r in requests) * 1000.0

    invisible_ms = sum(bucket_totals[k] for k in INVISIBLE_KEYS)
    invisible_ms_with_shuffle = invisible_ms + sum(bucket_totals[k] for k in MOE_SHUFFLE_KEYS)
    denom_ms = denom_ledger_ms + visible_ms_total

    print(_RESULT_MARKER + json.dumps({
        "tag": tag,
        "error": None,
        "attn_tp": attn_tp, "ffn_tp": ffn_tp, "ffn_ep": ffn_ep,
        "split_ep": split_ep, "label": label,
        "num_decode_rows": num_decode_rows,
        "denom_ms": denom_ms,
        "visible_ms": visible_ms_total,
        "invisible_ms": invisible_ms,
        "invisible_ms_with_shuffle": invisible_ms_with_shuffle,
        "bucket_totals": bucket_totals,
        "tp_ms": sum(bucket_totals[k] for k in TP_KEYS),
        "pp_ms": sum(bucket_totals[k] for k in PP_KEYS),
        "ep_ms": sum(bucket_totals[k] for k in EP_KEYS),
        "moe_shuffle_ms": sum(bucket_totals[k] for k in MOE_SHUFFLE_KEYS),
        "dp_ms": sum(bucket_totals[k] for k in DP_KEYS),
    }), flush=True)


_SCRIPT_PATH = str(Path(__file__).resolve())


def _run_scenario_in_subprocess(label: str, attn_tp: int, ffn_tp: int, ffn_ep: int,
                                split_ep: bool = False) -> dict:
    argv = [sys.executable, _SCRIPT_PATH, "--scenario", label,
           "--attn-tp", str(attn_tp), "--ffn-tp", str(ffn_tp), "--ffn-ep", str(ffn_ep)]
    if split_ep:
        argv.append("--split-ep")
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(FRONTIER_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return {"error": f"no result (exit code {proc.returncode}); see stderr above"}


def _report_row(r: dict) -> str:
    if r.get("error"):
        return f"[{r.get('tag', '?')}] ERROR: {r['error']}"
    denom = r["denom_ms"]
    vis, inv = r["visible_ms"], r["invisible_ms"]
    inv_shuf = r["invisible_ms_with_shuffle"]
    total_comm = vis + inv
    headline = 100 * vis / total_comm if total_comm else float("nan")
    headline_shuf = 100 * vis / (vis + inv_shuf) if (vis + inv_shuf) else float("nan")
    return (f"[{r['tag']}] denom={denom:9.4f}ms visible={vis:9.4f}ms({100*vis/denom:5.2f}%) "
           f"invisible={inv:9.4f}ms({100*inv/denom:5.2f}%) "
           f"[tp={r['tp_ms']:.4f} pp={r['pp_ms']:.4f} ep={r['ep_ms']:.4f} "
           f"shuffle={r['moe_shuffle_ms']:.4f} dp={r['dp_ms']:.4f}] "
           f"headline(visible/total_comm)={headline:6.2f}%  with_shuffle_in_comm={headline_shuf:6.2f}%  "
           f"rows={r['num_decode_rows']}")


def main() -> int:
    print("=== EP sweep (attn_tp=1, ffn_tp=1) ===")
    for ep in (1, 2, 4):
        for label in ("colocated", "split"):
            r = _run_scenario_in_subprocess(label, 1, 1, ep)
            print(_report_row(r))

    print()
    print("=== TP sweep (ffn_ep=1, ffn_tp=1) ===")
    for tp in (1, 2, 4):
        for label in ("colocated", "split"):
            r = _run_scenario_in_subprocess(label, tp, 1, 1)
            print(_report_row(r))

    print()
    print("=== EP=4, experts spread across domains (colocated M2N placement) ===")
    r_ep_colocated_experts = _run_scenario_in_subprocess("colocated", 1, 1, 4, split_ep=False)
    r_ep_split_experts = _run_scenario_in_subprocess("colocated", 1, 1, 4, split_ep=True)
    print("experts colocated: " + _report_row(r_ep_colocated_experts))
    print("experts split:     " + _report_row(r_ep_split_experts))
    if not r_ep_colocated_experts.get("error") and not r_ep_split_experts.get("error"):
        print(f"expert_parallel_communication_time colocated={r_ep_colocated_experts['ep_ms']:.6f}ms "
             f"split={r_ep_split_experts['ep_ms']:.6f}ms "
             f"(identical: {r_ep_colocated_experts['ep_ms'] == r_ep_split_experts['ep_ms']})")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("colocated", "split"), default=None)
    parser.add_argument("--attn-tp", type=int, default=1)
    parser.add_argument("--ffn-tp", type=int, default=1)
    parser.add_argument("--ffn-ep", type=int, default=1)
    parser.add_argument("--split-ep", action="store_true")
    args = parser.parse_args()
    if args.scenario:
        _run_scenario(args.scenario, args.attn_tp, args.ffn_tp, args.ffn_ep, args.split_ep)
        raise SystemExit(0)
    raise SystemExit(main())
