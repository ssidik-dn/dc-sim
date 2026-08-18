"""EngineCCBackend: answer Frontier's six BaseCCBackend prediction calls with
this project's Fabric/Placement/CostBackend model instead of Frontier's own
closed-form ones.

Unit conversion happens in exactly one place, `_ns_to_ms`, immediately below.
Frontier's six `predict_*` methods return float milliseconds; this engine
computes nanoseconds throughout (int for point-to-point transfers, float for
CostBackend.estimate results). Both representations are floats or exact ints
converted through a single divide-by-1e6, so there is no integer rounding
direction to choose -- see `test_ns_to_ms_is_exact_for_whole_microseconds` in
tests/test_cc_backend_integration.py for the round-trip check.
"""
from __future__ import annotations

from typing import List, Optional

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import BaseCCBackendConfig
from frontier.types import ClusterType

from engine.cost.astra_backend import CostBackend
from engine.logical.deployment import ParallelGroup, ParallelKind
from engine.network.transfers import Transfer, isolated_durations
from engine.physical.topology import Fabric
from engine.placement.placement import Placement

from .comm_groups import CommGroupRegistry

_NS_PER_MS = 1_000_000.0


def _ns_to_ms(duration_ns: float) -> float:
    """The one conversion point between the engine's nanoseconds and
    Frontier's milliseconds."""
    return duration_ns / _NS_PER_MS


class EngineCCBackend(BaseCCBackend):
    """A `BaseCCBackend` backed by this project's fabric-aware cost model.

    Construction needs both what Frontier supplies (config, cluster_type,
    device_type, network_device, num_devices -- required by
    `BaseCCBackend.__init__`) and what only this project has: a `Fabric`, a
    `Placement`, a `CostBackend` to price collectives, and a
    `CommGroupRegistry` to turn Frontier's (cluster_type, comm_domain,
    num_devices) triple into the rank set that triple actually means. See
    task 06 spec S7 for why the two constructor shapes cannot be reconciled
    through Frontier's own CCBackendFactory.create() without an upstream
    change.
    """

    def __init__(
        self,
        fabric: Fabric,
        placement: Placement,
        cost_backend: CostBackend,
        groups: CommGroupRegistry,
        *,
        config: Optional[BaseCCBackendConfig] = None,
        cluster_type: Optional[ClusterType] = None,
        device_type: str = "",
        network_device: str = "",
        num_devices: int = 1,
    ) -> None:
        super().__init__(
            config if config is not None else BaseCCBackendConfig(),
            cluster_type, device_type, network_device, num_devices)
        self._fabric = fabric
        self._placement = placement
        self._cost_backend = cost_backend
        self._groups = groups

    # -- shared path for the five true collectives --------------------------
    def _collective_ms(self, op: str, data_size_bytes: int, num_devices: int,
                       cluster_type: Optional[ClusterType],
                       comm_domain: Optional[str]) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, op)
        ranks = self._groups.resolve(cluster_type, comm_domain, num_devices)
        # `kind` is unused by group_shape()/induced_links() -- ParallelGroup
        # is reused here purely as the (kind, ranks) container Placement
        # already knows how to read a shape out of.
        group = ParallelGroup(kind=ParallelKind.TP, ranks=ranks)
        shape = self._placement.group_shape(group)
        result = self._cost_backend.estimate(op, data_size_bytes, shape, self._fabric)
        return _ns_to_ms(result.duration_ns)

    def predict_allreduce(self, data_size_bytes: int, num_devices: int,
                          cluster_type: Optional[ClusterType] = None,
                          comm_domain: Optional[str] = None) -> float:
        return self._collective_ms("all_reduce", data_size_bytes, num_devices,
                                   cluster_type, comm_domain)

    def predict_allgather(self, data_size_bytes: int, num_devices: int,
                          cluster_type: Optional[ClusterType] = None,
                          comm_domain: Optional[str] = None) -> float:
        return self._collective_ms("all_gather", data_size_bytes, num_devices,
                                   cluster_type, comm_domain)

    def predict_broadcast(self, data_size_bytes: int, num_devices: int,
                          cluster_type: Optional[ClusterType] = None,
                          comm_domain: Optional[str] = None) -> float:
        return self._collective_ms("broadcast", data_size_bytes, num_devices,
                                   cluster_type, comm_domain)

    def predict_reduce_scatter(self, data_size_bytes: int, num_devices: int,
                               cluster_type: Optional[ClusterType] = None,
                               comm_domain: Optional[str] = None) -> float:
        return self._collective_ms("reduce_scatter", data_size_bytes,
                                   num_devices, cluster_type, comm_domain)

    def predict_all_to_all(self, data_size_bytes: int, num_devices: int,
                           cluster_type: Optional[ClusterType] = None,
                           comm_domain: Optional[str] = None) -> float:
        return self._collective_ms("all_to_all", data_size_bytes, num_devices,
                                   cluster_type, comm_domain)

    # -- point-to-point -------------------------------------------------
    def predict_send_recv(self, data_size_bytes: int,
                          cluster_type: Optional[ClusterType] = None,
                          comm_domain: Optional[str] = None) -> float:
        """Frontier's `predict_send_recv` carries no `num_devices` (task 06
        spec S3 describes a uniform signature across all six methods; the
        actual base class does not -- see the task report). Point-to-point is
        always exactly two ranks, so the registry is queried with that fixed
        arity rather than one Frontier never gives us.
        """
        self._validate_data_size(data_size_bytes)
        ranks = self._groups.resolve(cluster_type, comm_domain, 2)
        if len(ranks) != 2:
            raise ValueError(
                f"send_recv needs exactly 2 ranks, registry returned "
                f"{len(ranks)} for cluster_type={cluster_type!r}, "
                f"comm_domain={comm_domain!r}")
        src = self._placement.gpu(ranks[0])
        dst = self._placement.gpu(ranks[1])
        t = Transfer(key="send_recv", src=src, dst=dst, size_bytes=data_size_bytes)
        durations = isolated_durations(self._fabric, [t])
        return _ns_to_ms(durations[t.key])
