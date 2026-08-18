"""Bridge Frontier's (cluster_type, comm_domain, num_devices) triple to the
placement-aware rank set it actually means.

Frontier's cc_backend calls carry a device count and a parallelism-domain
label -- never a rank identity or a physical location (see task 06 spec S3,
last paragraph). `Placement` is the piece that has both. This registry is
populated once, from a `Deployment` and a `Placement`, before a run starts,
and is then queried by `EngineCCBackend` on every prediction call.

Lookup failure raises rather than falling back to an assumed packed
placement: a silent fallback here is exactly the "plausible wrong number"
failure mode the task spec is shaped to avoid.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from engine.logical.deployment import Rank

GroupKey = Tuple[Any, Any, int]


class CommGroupError(KeyError):
    """Raised when a (cluster_type, comm_domain, num_devices) triple cannot
    be resolved to a unique, previously-registered rank set."""


class CommGroupRegistry:
    """Maps a Frontier (cluster_type, comm_domain, num_devices) triple to the
    engine ranks it refers to."""

    def __init__(self) -> None:
        self._groups: Dict[GroupKey, List[Rank]] = {}

    def register(self, cluster_type: Any, comm_domain: Any, num_devices: int,
                 ranks: List[Rank]) -> None:
        ranks = list(ranks)
        if len(ranks) != num_devices:
            raise ValueError(
                f"registering {len(ranks)} ranks but num_devices={num_devices} "
                f"for (cluster_type={cluster_type!r}, comm_domain={comm_domain!r})")
        key: GroupKey = (cluster_type, comm_domain, num_devices)
        existing = self._groups.get(key)
        if existing is not None and existing != ranks:
            raise CommGroupError(
                f"(cluster_type={cluster_type!r}, comm_domain={comm_domain!r}, "
                f"num_devices={num_devices}) is already registered to a "
                f"different rank set. Frontier's triple cannot disambiguate "
                f"which group is meant, so this must be reported rather than "
                f"silently picking one.")
        self._groups[key] = ranks

    def resolve(self, cluster_type: Any, comm_domain: Any,
                num_devices: int) -> List[Rank]:
        key: GroupKey = (cluster_type, comm_domain, num_devices)
        try:
            return list(self._groups[key])
        except KeyError:
            raise CommGroupError(
                f"no placement registered for cluster_type={cluster_type!r}, "
                f"comm_domain={comm_domain!r}, num_devices={num_devices}; "
                f"refusing to guess a packed placement") from None
