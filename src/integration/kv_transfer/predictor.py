"""EngineKVCacheTransferPredictor: price a KV cache transfer from the fabric
graph and the placement map, instead of Frontier's flat bandwidth/latency
formula.

This is a real integration, but not a complete one. `KVCacheTransferStartEvent`
computes `transfer_end_time = self.time + duration` at transfer start and
schedules the end event immediately (see
frontier/events/kv_cache_transfer_start_event.py). Whatever this predictor
returns, a second concurrent transfer cannot make the first one slower,
because the second transfer's existence cannot reach the first one's
already-scheduled completion. That is the causality constraint this project
has run into before (engine/network/transfers.py's module docstring), and
closing it here needs the runtime event replaced with something that can
revise a completion after it is scheduled -- a separate, harder task. This
predictor makes the number placement-sensitive; it does not make concurrent
transfers contend.

`EngineKVCacheTransferConfig` and `EngineKVCacheTransferPredictor` are both
module-level, deliberately -- task 07 found that Frontier's polymorphic-config
subclass discovery walks weak references (`type.__subclasses__()`), so a class
defined inside a function is collected between CLI parsing and reconstruction,
producing an error that reads exactly like a closed extension point. Both
classes are registered at import time, mirroring how Frontier's own backends
self-register (e.g. frontier/cc_backend/backends/analytical_cc_backend.py's
last line).

The fabric/placement/deployment context this predictor needs now lives in
`..context` (task 11 folded it in there so the M2N predictor could share it
rather than inventing a second mechanism). `EngineKVContext` and
`set_context` are re-exported here unchanged, so nothing that already
imports them from this module needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from frontier.config.kv_cache_transfer_config import (AnalyticalKVCacheTransferConfig,
                                                       BaseKVCacheTransferConfig)
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor)
from frontier.kv_cache_transfer.base_kv_cache_transfer_predictor import (
    BaseKVCacheTransferPredictor)
from frontier.kv_cache_transfer.kv_cache_transfer_predictor_registry import (
    KVCacheTransferPredictorRegistry)
from frontier.types import ClusterType, KVCacheTransferType

from ..binding_support import price_transfer
from ..context import EngineContext as EngineKVContext
from ..context import require_context as _require_context
from ..context import set_context

if TYPE_CHECKING:
    from frontier.config import ReplicaConfig
    from frontier.entities import Batch, Request


@dataclass
class EngineKVCacheTransferConfig(BaseKVCacheTransferConfig):
    """Sentinel-free config for the real predictor. EMPIRICAL has no other
    implementation in this Frontier checkout (task 07); this is the first."""

    @classmethod
    def get_type(cls) -> "KVCacheTransferType":
        return KVCacheTransferType.EMPIRICAL

    @classmethod
    def get_name(cls) -> str:
        return "engine"


class EngineKVCacheTransferPredictor(BaseKVCacheTransferPredictor):
    """Prices a KV transfer as one isolated point-to-point flow between the
    source and destination pools' representative (first) rank, routed over
    the real fabric graph. See task 09 report S2.2/S2 for why "first rank of
    the pool's one replica" is the resolvable unit, and S1 for why this is
    isolated rather than contended.

    `get_kv_cache_size` / `get_kv_cache_size_for_request` are delegated to
    Frontier's own AnalyticalKVCacheTransferPredictor rather than
    reimplemented: that computation (layers, KV heads, head dimension, dtype,
    quantisation, MLA/GQA) is already correct, and re-deriving it here is
    exactly the kind of duplication that produces a second, silently
    diverging answer. Only `get_transfer_time` -- what this project actually
    adds -- is new.
    """

    def __init__(self, config: "BaseKVCacheTransferConfig") -> None:
        super().__init__(config)
        self._size_predictor = AnalyticalKVCacheTransferPredictor(
            AnalyticalKVCacheTransferConfig())
        # Every binding this predictor has made (task 14): None where the
        # destination pool was unambiguous, or where timing="late" declined
        # to commit to one -- see price_transfer's own docstring.
        self.bindings: list = []

    def get_transfer_time(
        self,
        source_cluster_type: "ClusterType",
        target_cluster_type: "ClusterType",
        batch: "Batch",
        kv_cache_size_bytes: int,
    ) -> float:
        ctx = _require_context()
        price_ms, chosen_replica_id = price_transfer(
            ctx, source_cluster_type, target_cluster_type,
            kv_cache_size_bytes, key="kv_transfer")
        self.bindings.append(chosen_replica_id)
        return price_ms

    def get_kv_cache_size(self, batch: "Batch", replica_config: "ReplicaConfig") -> int:
        return self._size_predictor.get_kv_cache_size(batch, replica_config)

    def get_kv_cache_size_for_request(
        self, request: "Request", replica_config: "ReplicaConfig"
    ) -> int:
        return self._size_predictor.get_kv_cache_size_for_request(request, replica_config)

    def supports_latency_hiding(self) -> bool:
        # Latency hiding is a scheduling/event-semantics capability, not a
        # cost-model one -- out of scope alongside contention (see the
        # module docstring), so this predictor never claims to support it
        # regardless of what the config says.
        return False


KVCacheTransferPredictorRegistry.register(
    KVCacheTransferType.EMPIRICAL, EngineKVCacheTransferPredictor)
