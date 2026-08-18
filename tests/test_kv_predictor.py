"""Task 09: a real KV cache transfer predictor, priced from the fabric graph
and the placement map.

Unit tests only -- the end-to-end proof that Frontier actually uses this
(the split placement producing a slower TTFT-adjacent number) lives in
tools/run_kv_integration.py, not here. See docs/tasks/09-kv-predictor-report.md
S1 for why: this predictor does not model contention, so nothing here should
be read as validating anything beyond "the isolated cost of one transfer is
placement-sensitive."
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
from integration.kv_transfer import predictor as predictor_module
from integration.kv_transfer.predictor import (EngineKVCacheTransferConfig,
                                               EngineKVCacheTransferPredictor,
                                               EngineKVContext, set_context)

SIZE_BYTES = 1 << 20


def _one_prefill_one_decode(fabric):
    d = Deployment("kv")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE, 0, tp=1))
    return d, d.replicas[0].ranks[0], d.replicas[1].ranks[0]


def _predictor(fabric, placement, deployment, groups) -> EngineKVCacheTransferPredictor:
    set_context(EngineKVContext(fabric, placement, deployment, groups))
    return EngineKVCacheTransferPredictor(EngineKVCacheTransferConfig())


def test_returns_milliseconds():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=2,
                              scale_up_GBps=400.0, scale_up_latency_ns=936.25)
    d, prefill_rank, decode_rank = _one_prefill_one_decode(fabric)
    placement = explicit(d, fabric, {prefill_rank: GpuId(0, 0), decode_rank: GpuId(0, 1)})
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.PREFILL: ClusterType.PREFILL,
                                      PoolKind.DECODE: ClusterType.DECODE})
    predictor = _predictor(fabric, placement, d, reg)

    ms = predictor.get_transfer_time(ClusterType.PREFILL, ClusterType.DECODE,
                                     None, SIZE_BYTES)

    expected_ns = isolated_durations(
        fabric, [Transfer(key="k", src=GpuId(0, 0), dst=GpuId(0, 1),
                          size_bytes=SIZE_BYTES)])["k"]
    assert isinstance(ms, float) and ms > 0
    assert ms == pytest.approx(expected_ns / 1_000_000.0)


def test_packed_placement_cheaper_than_split():
    fabric = build_node_scale(num_machines=2, gpus_per_machine=8)
    d, prefill_rank, decode_rank = _one_prefill_one_decode(fabric)
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.PREFILL: ClusterType.PREFILL,
                                      PoolKind.DECODE: ClusterType.DECODE})

    packed_placement = explicit(d, fabric,
                                {prefill_rank: GpuId(0, 0), decode_rank: GpuId(0, 1)})
    split_placement = explicit(d, fabric,
                               {prefill_rank: GpuId(0, 0), decode_rank: GpuId(1, 0)})

    packed_ms = _predictor(fabric, packed_placement, d, reg).get_transfer_time(
        ClusterType.PREFILL, ClusterType.DECODE, None, SIZE_BYTES)
    split_ms = _predictor(fabric, split_placement, d, reg).get_transfer_time(
        ClusterType.PREFILL, ClusterType.DECODE, None, SIZE_BYTES)

    assert split_ms > packed_ms


def test_multiple_replicas_raises():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=4)
    d = Deployment("kv")
    d.add(Replica(PoolKind.PREFILL, 0, tp=1))
    d.add(Replica(PoolKind.DECODE, 0, tp=1))
    d.add(Replica(PoolKind.DECODE, 1, tp=1))  # a second DECODE replica
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.PREFILL: ClusterType.PREFILL,
                                      PoolKind.DECODE: ClusterType.DECODE})
    placement = explicit(d, fabric, {
        d.replicas[0].ranks[0]: GpuId(0, 0),
        d.replicas[1].ranks[0]: GpuId(0, 1),
        d.replicas[2].ranks[0]: GpuId(0, 2),
    })
    predictor = _predictor(fabric, placement, d, reg)

    with pytest.raises(CommGroupError, match="binding"):
        predictor.get_transfer_time(ClusterType.PREFILL, ClusterType.DECODE,
                                    None, SIZE_BYTES)


def test_unresolvable_pool_raises():
    fabric = build_node_scale(num_machines=1, gpus_per_machine=2)
    d, prefill_rank, decode_rank = _one_prefill_one_decode(fabric)
    placement = explicit(d, fabric, {prefill_rank: GpuId(0, 0), decode_rank: GpuId(0, 1)})
    reg = CommGroupRegistry()
    # Only PREFILL is registered -- DECODE never was.
    populate_from_deployment(reg, Deployment("prefill-only", [d.replicas[0]]),
                             {PoolKind.PREFILL: ClusterType.PREFILL})
    predictor = _predictor(fabric, placement, d, reg)

    with pytest.raises(CommGroupError):
        predictor.get_transfer_time(ClusterType.PREFILL, ClusterType.DECODE,
                                    None, SIZE_BYTES)


def test_config_class_is_module_level():
    """Guards the weak-reference trap (task 07): a class defined inside a
    function has a __qualname__ containing the function's name, and is not
    reachable as a plain module attribute once that function returns."""
    assert getattr(predictor_module, "EngineKVCacheTransferConfig") is EngineKVCacheTransferConfig
    assert EngineKVCacheTransferConfig.__qualname__ == "EngineKVCacheTransferConfig"
