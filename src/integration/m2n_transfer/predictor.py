"""EngineM2NTransferPredictor: price an attention<->FFN activation exchange
from the fabric graph and the placement map, instead of Frontier's flat
bandwidth/latency formula.

Same incompleteness as the KV predictor (task 09), for the same reason:
`M2NTransferStartEvent` (mirroring `KVCacheTransferStartEvent`) fixes a
completion time at transfer start rather than letting a later arrival
revise it, so this predictor makes the number placement-sensitive without
making concurrent transfers contend.

What's different here, and why this predictor exists as its own task rather
than being a copy-paste of the KV one: activation exchange is called 192
times per run against KV's handful (task 08), and its payloads are small
enough to be latency-bound rather than bandwidth-bound (task 10). A model
that ignored path latency would have understated the placement penalty by
roughly a factor of two in exactly this regime -- which is why task 10 had
to land before this predictor could be trusted.

`EngineM2NTransferConfig` and `EngineM2NTransferPredictor` are both
module-level (task 07's weak-reference finding), registered under
`M2NTransferType.EMPIRICAL` at import time, and read the same
`EngineContext` the KV predictor does (`..context`, consolidated there in
this task rather than inventing a second mechanism -- see task 09/11
reports).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from frontier.config.m2n_transfer_config import (AnalyticalM2NTransferConfig,
                                                  BaseM2NTransferConfig)
from frontier.m2n_transfer.analytical_m2n_transfer_predictor import (
    AnalyticalM2NTransferPredictor)
from frontier.m2n_transfer.base_m2n_transfer_predictor import BaseM2NTransferPredictor
from frontier.m2n_transfer.m2n_transfer_predictor_registry import (
    M2NTransferPredictorRegistry)
from frontier.types import ClusterType, M2NTransferType

from engine.network.transfers import Transfer, isolated_durations

from ..context import require_context

if TYPE_CHECKING:
    from frontier.config import ReplicaConfig
    from frontier.entities import Batch, Request

_NS_PER_MS = 1_000_000.0


def _ns_to_ms(duration_ns: float) -> float:
    """The one conversion point between the engine's nanoseconds and
    Frontier's milliseconds. Both sides are floats, so there is no integer
    rounding direction to pick."""
    return duration_ns / _NS_PER_MS


@dataclass(frozen=True)
class LayerAttribution:
    """Per-call attribution recorded alongside a transfer-time answer.

    Not part of `BaseM2NTransferPredictor`'s contract: Frontier never passes
    layer_id/afd_stage_idx/pipeline_stage as arguments to `get_transfer_time`
    (task 08 finding) -- they are attached to an `M2NTransferInfo` the
    *caller* builds afterward, from data available at the call site but not
    handed to the predictor. This predictor derives the same three fields
    from what it does receive (`batch`, `source_cluster_type`), the same way
    the caller does, and records the result here for a later analysis to
    read via `EngineM2NTransferPredictor.last_attribution` -- see the task 11
    report for whether this derivation could diverge from Frontier's own.
    """
    layer_id: Optional[int]
    afd_stage_idx: Optional[int]
    pipeline_stage: str


@dataclass
class EngineM2NTransferConfig(BaseM2NTransferConfig):
    """EMPIRICAL has no other implementation in this Frontier checkout
    (task 08); this is the first."""

    @classmethod
    def get_type(cls) -> "M2NTransferType":
        return M2NTransferType.EMPIRICAL

    @classmethod
    def get_name(cls) -> str:
        return "engine"


class EngineM2NTransferPredictor(BaseM2NTransferPredictor):
    """Prices an M2N transfer as one isolated point-to-point flow between
    the source and destination pools' representative (first) rank, routed
    over the real fabric graph -- the same shape as
    `EngineKVCacheTransferPredictor` (task 09), for the same reason: with
    exactly one replica per pool, "first rank of the pool's one replica" is
    the only unambiguous choice (see `resolve_pool`'s raise otherwise).

    `get_activation_size`/`get_activation_size_for_request` are delegated to
    Frontier's own `AnalyticalM2NTransferPredictor` rather than
    reimplemented -- that computation (hidden size, activation dtype,
    quantisation) is already correct. Only `get_transfer_time` is new.

    `calls` and `total_wall_ns` track this predictor's own overhead (task 11
    S2.4: 192 calls/run means per-call cost matters); `last_attribution`
    holds the most recent `LayerAttribution`.
    """

    def __init__(self, config: "BaseM2NTransferConfig") -> None:
        super().__init__(config)
        self._size_predictor = AnalyticalM2NTransferPredictor(
            AnalyticalM2NTransferConfig())
        self.calls: int = 0
        self.total_wall_ns: int = 0
        self.last_attribution: Optional[LayerAttribution] = None

    @staticmethod
    def _current_layer_id(batch: "Batch") -> Optional[int]:
        """Mirrors ClusterBatchEndEvent._get_current_layer_id_from_batch:
        the first non-completed request's completed_layer_count, or the
        first request's if every request in the batch has completed."""
        requests = getattr(batch, "requests", None) or []
        for request in requests:
            if not request.completed:
                return request.completed_layer_count
        return requests[0].completed_layer_count if requests else None

    def get_transfer_time(
        self,
        source_cluster_type: "ClusterType",
        target_cluster_type: "ClusterType",
        batch: "Batch",
        activation_size_bytes: int,
    ) -> float:
        wall_start = time.perf_counter_ns()

        ctx = require_context()
        src_ranks = ctx.groups.resolve_pool(source_cluster_type)
        dst_ranks = ctx.groups.resolve_pool(target_cluster_type)
        src_gpu = ctx.placement.gpu(src_ranks[0])
        dst_gpu = ctx.placement.gpu(dst_ranks[0])

        self.last_attribution = LayerAttribution(
            layer_id=self._current_layer_id(batch),
            afd_stage_idx=getattr(batch, "afd_stage_idx", None),
            pipeline_stage=("attn_to_ffn" if source_cluster_type == ClusterType.DECODE_ATTN
                            else "ffn_to_attn"),
        )

        t = Transfer(key="m2n_transfer", src=src_gpu, dst=dst_gpu,
                    size_bytes=activation_size_bytes)
        duration_ns = isolated_durations(ctx.fabric, [t])[t.key]
        result = _ns_to_ms(duration_ns)

        self.calls += 1
        self.total_wall_ns += time.perf_counter_ns() - wall_start
        return result

    def get_activation_size(self, batch: "Batch", replica_config: "ReplicaConfig",
                            source_cluster_type: "ClusterType") -> int:
        return self._size_predictor.get_activation_size(
            batch, replica_config, source_cluster_type)

    def get_activation_size_for_request(
        self, request: "Request", replica_config: "ReplicaConfig",
        source_cluster_type: "ClusterType",
    ) -> int:
        return self._size_predictor.get_activation_size_for_request(
            request, replica_config, source_cluster_type)


M2NTransferPredictorRegistry.register(
    M2NTransferType.EMPIRICAL, EngineM2NTransferPredictor)
