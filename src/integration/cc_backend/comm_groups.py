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

from typing import Any, Dict, List, Mapping, Tuple

from engine.logical.deployment import Deployment, ParallelKind, PoolKind, Rank

GroupKey = Tuple[Any, Any, int]


class CommGroupError(KeyError):
    """Raised when a (cluster_type, comm_domain, num_devices) triple cannot
    be resolved to a unique, previously-registered rank set."""


class CommGroupRegistry:
    """Maps a Frontier (cluster_type, comm_domain, num_devices) triple to the
    engine ranks it refers to.

    A second, coarser resolution mode lives alongside it: `register_pool` /
    `resolve_pool` answer "which ranks make up the one replica of this
    pool", for Frontier calls (KV cache transfer, M2N transfer) that carry
    only a cluster_type -- no comm_domain, no num_devices to key a triple
    lookup on. See task 09 spec S2.2: with more than one replica per pool,
    nothing in the call tells us which one is meant, so this raises rather
    than choosing.
    """

    def __init__(self) -> None:
        self._groups: Dict[GroupKey, List[Rank]] = {}
        self._pools: Dict[Any, List[List[Rank]]] = {}

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

    def register_pool(self, cluster_type: Any, ranks: List[Rank]) -> None:
        """Record one replica's full rank set for `cluster_type`'s pool.

        Called once per replica. A second call for the same cluster_type
        records a second replica -- `resolve_pool` is what turns that into
        a raise, not this method, since registering two replicas is not
        itself an error until something asks to resolve one.
        """
        self._pools.setdefault(cluster_type, []).append(list(ranks))

    def resolve_pool(self, cluster_type: Any) -> List[Rank]:
        """The ranks of the single replica registered for this pool.

        Raises if zero replicas are registered (unresolvable) or more than
        one (binding a cross-pool request to a specific replica among
        several is unimplemented -- see task 09 spec S2.2).
        """
        replicas = self._pools.get(cluster_type, [])
        if not replicas:
            raise CommGroupError(
                f"no replica registered for pool cluster_type={cluster_type!r}")
        if len(replicas) > 1:
            raise CommGroupError(
                f"{len(replicas)} replicas registered for pool "
                f"cluster_type={cluster_type!r}; binding a cross-pool "
                f"transfer to one specific replica is unimplemented")
        return list(replicas[0])


def populate_from_deployment(registry: CommGroupRegistry, deployment: Deployment,
                             pool_cluster_type: Mapping[PoolKind, Any]) -> None:
    """Register every TP/PP/DP/EP group of every replica in `deployment`.

    `pool_cluster_type` supplies the frontier.types.ClusterType each PoolKind
    maps to -- this module does not import Frontier, so the caller (which does,
    and lives in src/integration/) provides the mapping rather than this
    function guessing it.
    """
    for replica in deployment.replicas:
        cluster_type = pool_cluster_type[replica.pool]
        registry.register_pool(cluster_type, replica.ranks)
        for kind in ParallelKind:
            for group in replica.groups(kind):
                registry.register(cluster_type, kind.value, group.size, group.ranks)
