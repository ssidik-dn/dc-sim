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

from typing import Any, Dict, List, Mapping, Optional, Tuple

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
        self._pools: Dict[Any, List[Tuple[int, List[Rank]]]] = {}

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

    def register_pool(self, cluster_type: Any, ranks: List[Rank],
                      replica_id: Optional[int] = None) -> None:
        """Record one replica's full rank set for `cluster_type`'s pool.

        Called once per replica. A second call for the same cluster_type
        records a second replica -- `resolve_pool` is what turns that into
        a raise, not this method, since registering two replicas is not
        itself an error until something asks to resolve one unambiguously.

        `replica_id` defaults to registration order (0, 1, 2, ...) when not
        given, so existing callers that only ever registered one replica per
        pool are unaffected.
        """
        existing = self._pools.setdefault(cluster_type, [])
        if replica_id is None:
            replica_id = len(existing)
        existing.append((replica_id, list(ranks)))

    def resolve_pool(self, cluster_type: Any) -> List[Rank]:
        """The ranks of the single replica registered for this pool.

        Raises if zero replicas are registered (unresolvable) or more than
        one -- resolving a transfer to one specific replica among several
        needs a binding policy (task 14), which is a decision for the
        caller to make deliberately via `resolve_pool_candidates` plus
        `engine.placement.binding`, not something this method guesses at.
        """
        replicas = self._pools.get(cluster_type, [])
        if not replicas:
            raise CommGroupError(
                f"no replica registered for pool cluster_type={cluster_type!r}")
        if len(replicas) > 1:
            raise CommGroupError(
                f"{len(replicas)} replicas registered for pool "
                f"cluster_type={cluster_type!r}; resolving to one specific "
                f"replica needs a binding policy (see resolve_pool_candidates "
                f"and engine.placement.binding) rather than a guess")
        return list(replicas[0][1])

    def resolve_pool_candidates(self, cluster_type: Any) -> List[Tuple[int, List[Rank]]]:
        """Every registered replica for this pool, as (replica_id, ranks)
        pairs -- for binding among several, when `resolve_pool` itself
        raises because there is more than one.

        Raises under the same zero-replicas condition as `resolve_pool`;
        callers that want "raise unless exactly one" should use
        `resolve_pool` instead, since that is a different, stricter
        contract than "give me whatever is registered, however many."
        """
        replicas = self._pools.get(cluster_type, [])
        if not replicas:
            raise CommGroupError(
                f"no replica registered for pool cluster_type={cluster_type!r}")
        return [(rid, list(ranks)) for rid, ranks in replicas]


# Task 20's own finding, from running it rather than assuming the generic
# `ParallelKind.value` strings would match: Frontier's real cc_backend call
# sites never pass `comm_domain="TP"` -- they split by which sublayer is
# allreducing, `comm_domain="ATTN_TP"` for attention's own tensor-parallel
# allreduce and `comm_domain="MOE_TP"` for the FFN/MoE one (grep across
# frontier/execution_time_predictor/*.py's `comm_domain=` call sites; "PP",
# "DP", and "EP" do match their `ParallelKind` counterpart directly, no
# alias needed). This project's own `Replica`/`ParallelKind` model has one
# TP degree per replica, not a separate attention-TP and MoE-TP value, so
# the one TP group registered for a replica is physically both -- the same
# ranks answer either name.
_DOMAIN_ALIASES: Mapping[ParallelKind, Tuple[str, ...]] = {
    ParallelKind.TP: ("TP", "ATTN_TP", "MOE_TP"),
}


def populate_from_deployment(registry: CommGroupRegistry, deployment: Deployment,
                             pool_cluster_type: Mapping[PoolKind, Any]) -> None:
    """Register every TP/PP/DP/EP group of every replica in `deployment`,
    under every domain name Frontier's own cc_backend calls actually use
    for it (see `_DOMAIN_ALIASES`).

    `pool_cluster_type` supplies the frontier.types.ClusterType each PoolKind
    maps to -- this module does not import Frontier, so the caller (which does,
    and lives in src/integration/) provides the mapping rather than this
    function guessing it.
    """
    for replica in deployment.replicas:
        cluster_type = pool_cluster_type[replica.pool]
        registry.register_pool(cluster_type, replica.ranks, replica_id=replica.replica_id)
        for kind in ParallelKind:
            for group in replica.groups(kind):
                for domain in _DOMAIN_ALIASES.get(kind, (kind.value,)):
                    registry.register(cluster_type, domain, group.size, group.ranks)
