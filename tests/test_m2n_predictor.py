"""Task 11: a real M2N (attention<->FFN activation) transfer predictor,
priced from the fabric graph and the placement map.

Unit tests only -- the end-to-end proof against a real Frontier
pd-af-disaggregation run, and the per-call performance measurement, live in
tools/run_m2n_integration.py. See docs/tasks/11-m2n-predictor-report.md.
"""
from __future__ import annotations

import pytest

from frontier.types import ClusterType

from engine.logical.deployment import Deployment, PoolKind, Replica
from engine.network.transfers import Transfer, isolated_durations
from engine.physical.builders import build_node_scale
from engine.physical.topology import GpuId
from engine.placement.placement import explicit

from integration.cc_backend.comm_groups import (CommGroupError,
                                                 CommGroupRegistry,
                                                 populate_from_deployment)
from integration.context import EngineContext, set_context
from integration.m2n_transfer import predictor as predictor_module
from integration.m2n_transfer.predictor import (EngineM2NTransferConfig,
                                                EngineM2NTransferPredictor)

ACTIVATION_SIZE_BYTES = 16384  # 2 tokens x 4096 hidden x 2 bytes (fp16) -- small


class _Request:
    def __init__(self, completed: bool, completed_layer_count: int):
        self.completed = completed
        self.completed_layer_count = completed_layer_count


class _Batch:
    def __init__(self, requests, afd_stage_idx=0):
        self.requests = requests
        self.afd_stage_idx = afd_stage_idx


def _one_attn_one_ffn():
    d = Deployment("m2n")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1))
    return d, d.replicas[0].ranks[0], d.replicas[1].ranks[0]


def _predictor(fabric, placement, deployment, groups) -> EngineM2NTransferPredictor:
    set_context(EngineContext(fabric, placement, deployment, groups))
    return EngineM2NTransferPredictor(EngineM2NTransferConfig())


def test_returns_milliseconds():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=2,
                              scale_up_GBps=400.0, scale_up_latency_ns=936.25)
    d, attn_rank, ffn_rank = _one_attn_one_ffn()
    placement = explicit(d, fabric, {attn_rank: GpuId(0, 0), ffn_rank: GpuId(0, 1)})
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
                                      PoolKind.DECODE_FFN: ClusterType.DECODE_FFN})
    predictor = _predictor(fabric, placement, d, reg)

    batch = _Batch([_Request(False, 5)])
    ms = predictor.get_transfer_time(ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN,
                                     batch, ACTIVATION_SIZE_BYTES)

    expected_ns = isolated_durations(
        fabric, [Transfer(key="t", src=GpuId(0, 0), dst=GpuId(0, 1),
                          size_bytes=ACTIVATION_SIZE_BYTES)])["t"]
    assert isinstance(ms, float) and ms > 0
    assert ms == pytest.approx(expected_ns / 1_000_000.0)


def test_split_pools_cost_more_than_colocated():
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8)
    d, attn_rank, ffn_rank = _one_attn_one_ffn()
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
                                      PoolKind.DECODE_FFN: ClusterType.DECODE_FFN})
    colocated = explicit(d, fabric, {attn_rank: GpuId(0, 0), ffn_rank: GpuId(0, 1)})
    split = explicit(d, fabric, {attn_rank: GpuId(0, 0), ffn_rank: GpuId(1, 0)})

    batch = _Batch([_Request(False, 0)])
    colocated_ms = _predictor(fabric, colocated, d, reg).get_transfer_time(
        ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, batch, ACTIVATION_SIZE_BYTES)
    split_ms = _predictor(fabric, split, d, reg).get_transfer_time(
        ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, batch, ACTIVATION_SIZE_BYTES)

    assert split_ms > colocated_ms


def test_small_payload_penalty_exceeds_bandwidth_ratio():
    """The binding test. build_node_scale's defaults give a 400:50 = 8:1
    scale-up:scale-out bandwidth ratio. At this activation-sized payload
    (16 KiB, deep in the latency-bound regime task 10 measured -- crossover
    was ~366 KiB packed / ~684 KiB split), the split path's fixed per-hop
    latency dominates both totals, so the ratio approaches the LATENCY
    ratio instead: task 10's own sweep measured 14.65x at exactly this size.
    A predictor that returned 8.0x here would mean something upstream is
    silently ignoring latency -- see task 10's report for how easy that is
    to do by accident (the bug this whole model change exists to fix).
    """
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8,
                              scale_up_GBps=400.0, scale_out_GBps=50.0)
    d, attn_rank, ffn_rank = _one_attn_one_ffn()
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
                                      PoolKind.DECODE_FFN: ClusterType.DECODE_FFN})
    colocated = explicit(d, fabric, {attn_rank: GpuId(0, 0), ffn_rank: GpuId(0, 1)})
    split = explicit(d, fabric, {attn_rank: GpuId(0, 0), ffn_rank: GpuId(1, 0)})

    batch = _Batch([_Request(False, 0)])
    colocated_ms = _predictor(fabric, colocated, d, reg).get_transfer_time(
        ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, batch, ACTIVATION_SIZE_BYTES)
    split_ms = _predictor(fabric, split, d, reg).get_transfer_time(
        ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN, batch, ACTIVATION_SIZE_BYTES)

    ratio = split_ms / colocated_ms
    bandwidth_only_ratio = 400.0 / 50.0
    assert ratio > bandwidth_only_ratio, (
        f"got {ratio}, expected > {bandwidth_only_ratio} (bandwidth-only) "
        "-- a small payload should be latency-bound, not bandwidth-bound")
    assert ratio == pytest.approx(14.65030674846626)


def test_layer_id_is_derived_and_advances():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=2)
    d, attn_rank, ffn_rank = _one_attn_one_ffn()
    placement = explicit(d, fabric, {attn_rank: GpuId(0, 0), ffn_rank: GpuId(0, 1)})
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
                                      PoolKind.DECODE_FFN: ClusterType.DECODE_FFN})
    predictor = _predictor(fabric, placement, d, reg)

    seen = []
    for layer in range(3):
        predictor.get_transfer_time(ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN,
                                    _Batch([_Request(False, layer)]),
                                    ACTIVATION_SIZE_BYTES)
        seen.append(predictor.last_attribution.layer_id)
    assert seen == [0, 1, 2]


def test_pipeline_stage_reflects_direction():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=2)
    d, attn_rank, ffn_rank = _one_attn_one_ffn()
    placement = explicit(d, fabric, {attn_rank: GpuId(0, 0), ffn_rank: GpuId(0, 1)})
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
                                      PoolKind.DECODE_FFN: ClusterType.DECODE_FFN})
    predictor = _predictor(fabric, placement, d, reg)
    batch = _Batch([_Request(False, 0)])

    predictor.get_transfer_time(ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN,
                                batch, ACTIVATION_SIZE_BYTES)
    assert predictor.last_attribution.pipeline_stage == "attn_to_ffn"

    predictor.get_transfer_time(ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN,
                                batch, ACTIVATION_SIZE_BYTES)
    assert predictor.last_attribution.pipeline_stage == "ffn_to_attn"


def test_multiple_replicas_raises():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=4)
    d = Deployment("m2n")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_FFN, 0, tp=1))
    d.add(Replica(PoolKind.DECODE_FFN, 1, tp=1))  # a second DECODE_FFN replica
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: ClusterType.DECODE_ATTN,
                                      PoolKind.DECODE_FFN: ClusterType.DECODE_FFN})
    placement = explicit(d, fabric, {
        d.replicas[0].ranks[0]: GpuId(0, 0),
        d.replicas[1].ranks[0]: GpuId(0, 1),
        d.replicas[2].ranks[0]: GpuId(0, 2),
    })
    predictor = _predictor(fabric, placement, d, reg)

    with pytest.raises(CommGroupError, match="binding"):
        predictor.get_transfer_time(ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN,
                                    _Batch([_Request(False, 0)]), ACTIVATION_SIZE_BYTES)


def test_config_class_is_module_level():
    """Guards the weak-reference trap (task 07)."""
    assert getattr(predictor_module, "EngineM2NTransferConfig") is EngineM2NTransferConfig
    assert EngineM2NTransferConfig.__qualname__ == "EngineM2NTransferConfig"
    assert EngineM2NTransferPredictor.__qualname__ == "EngineM2NTransferPredictor"
