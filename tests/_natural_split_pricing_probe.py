#!/usr/bin/env python3
"""Task 57: prices the colocated and the (now-reachable) natural-split
arrangement for `(attn_tp, ffn_ep)`, real Frontier, no dummy mode,
`Phi-tiny-MoE-instruct` on two real 8-GPU machines -- the exact
configuration task 54 designed and task 56 hand-priced. Run as a
subprocess with `cwd` set to Frontier's own root (mirrors
`_memory_planner_probe.py`'s own established pattern -- model-config
resolution needs a cwd-relative path).
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/work/simulation/dc-sim/tools")
sys.path.insert(0, "/work/simulation/dc-sim/src")

_RESULT_MARKER = "NATURAL_SPLIT_PROBE_RESULT="


def _mean_tpot(topology, model, workload, hardware, candidate, num_blocks):
    from planner_core import Topology  # noqa: F401  (type context only)
    from planner import _placement_for, _argv
    from engine.logical.deployment import Deployment, Replica, PoolKind
    from frontier.types import ClusterType
    from integration.cc_backend.comm_groups import CommGroupRegistry, populate_from_deployment
    from integration.install import install

    d = Deployment("natural-split-probe")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=candidate.attn_tp))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1, ep=candidate.ffn_ep))

    placement = _placement_for(topology, d, candidate)

    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {
        PoolKind.PREFILL: ClusterType.PREFILL,
        PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
        PoolKind.DECODE_FFN: ClusterType.DECODE_FFN,
    })
    install(topology.fabric, placement, d, reg, collective=True, sglang_replica_scheduler=True)

    sys.argv = _argv(topology, model, workload, hardware, candidate, num_blocks,
                     f"natural_split_probe_{candidate.key}", 0, [])

    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.utils.random import set_seeds
    import statistics

    config = SimulationConfig.create_from_cli_args()
    assert not config.cluster_config.execution_time_predictor_config.enable_dummy_mode
    set_seeds(config.seed)
    sim = Simulator(config)
    sim.run()

    requests = sim._all_requests
    tpot_eligible = [r for r in requests
                     if r.num_decode_tokens > 1 and r.first_decode_token_completed_at > 0]
    return statistics.mean(r.tpot * 1000.0 for r in tpot_eligible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-tp", type=int, required=True)
    parser.add_argument("--ffn-ep", type=int, required=True)
    parser.add_argument("--relative", choices=["same", "disjoint"], required=True)
    args = parser.parse_args()

    from planner_core import ModelSpec, Workload, Hardware, Topology, Candidate, feasible_num_blocks
    from engine.physical.builders import build_node_scale

    model = ModelSpec("Phi-tiny-MoE-instruct", total_experts=16, router_topk=2, is_moe=True,
                      hidden_size=4096, num_attention_heads=16, num_key_value_heads=4,
                      num_layers=32, head_dim=128)
    workload = Workload(num_requests=32, qps=20.0, prefill_tokens=32, decode_tokens=16)
    hardware = Hardware(device="h800", memory_margin_fraction=0.7)
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=400.0, scale_out_GBps=50.0)
    topology = Topology(fabric, "two-real-machines-probe")
    num_blocks = feasible_num_blocks(model, hardware, args.attn_tp)

    # One evaluation per process (task 41's own established discipline --
    # looping several `Simulator.run()` calls in one process risks
    # cross-call state leakage, caught the hard way there). The caller
    # invokes this script twice, once per `--relative` value, rather
    # than this script pricing both itself.
    candidate = Candidate(attn_tp=args.attn_tp, attn_shape=(args.attn_tp,), ffn_ep=args.ffn_ep,
                          ep_shape=(args.ffn_ep,), relative=args.relative)
    mean_tpot_ms = _mean_tpot(topology, model, workload, hardware, candidate, num_blocks)

    print(_RESULT_MARKER + json.dumps({"relative": args.relative, "mean_tpot_ms": mean_tpot_ms}),
         flush=True)


if __name__ == "__main__":
    main()
