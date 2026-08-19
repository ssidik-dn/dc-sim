"""EngineCCBackend: answer Frontier's six BaseCCBackend prediction calls with
this project's Fabric/Placement cost model instead of Frontier's own
closed-form or profiled ones.

Task 06 built this against a separate `CostBackend.estimate()` abstraction
(ASTRA-sim or a mock), before task 10 put latency and contention into
`engine.network.transfers`. Task 20 rewrites the five true collectives
(`predict_send_recv` was already correct) to go through that same
`Transfer`/`run_transfers` path every other topology-aware predictor in
this project already uses -- so a collective and a point-to-point transfer
over the same links now agree by construction, per task 20 spec S3.1,
rather than needing a separate agreement test against a second model. See
this module's own investigation (task 20 report S4) for why: a real,
built ASTRA-sim binary was available and was tried first, and its own
analytical collective model turned out *not* to reliably price a
domain-split tensor-parallel group as more expensive than a packed one at
realistic message sizes -- the opposite of what this task's acceptance
criteria require and what a well-formed ring should do. Rather than depend
on an external algorithm this project cannot fully account for, the five
collectives below are priced from an explicitly stated, defensible
algorithm each (ring for allreduce/allgather/reduce_scatter, full pairwise
for all_to_all, sequential relay for broadcast), computed from the exact
same fabric contention model (`run_transfers`) as every KV/M2N transfer.

**Task 16's limitation restated, not worked around**: a collective's
participants are resolved as a whole registered group
(`CommGroupRegistry.resolve`), keyed by `(cluster_type, comm_domain,
num_devices)` -- there is no channel here, any more than there was for a
KV/M2N transfer's sending replica, for *which* rank issued *this specific*
call when more than one group could answer to the same triple (multiple
replicas of the same pool, each with its own same-shaped TP group). Every
call that resolves to the same triple is therefore priced identically,
against whichever one group was registered for it; `CommGroupRegistry`
raises rather than silently disambiguating if a second, differently-placed
group is ever registered under the same triple (see `register`'s own
conflict check) -- refusing beats guessing, the same rule as everywhere
else in this project.

Unit conversion happens in exactly one place, `_ns_to_ms`, immediately
below. Frontier's six `predict_*` methods return float milliseconds; this
engine computes nanoseconds throughout (int for point-to-point transfers),
converted through a single divide-by-1e6.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import BaseCCBackendConfig
from frontier.types import ClusterType

from engine.logical.deployment import Rank
from engine.network.transfers import Transfer, run_transfers
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
    `Placement` to price collectives against, and a `CommGroupRegistry` to
    turn Frontier's (cluster_type, comm_domain, num_devices) triple into
    the rank set that triple actually means. See task 06 spec S7 for why
    the two constructor shapes cannot be reconciled through Frontier's own
    `CCBackendFactory.create()` without an upstream change, and task 20's
    report for how it is reached anyway (a guarded runtime replacement of
    `create()` itself, not a CLI flag).
    """

    def __init__(
        self,
        fabric: Fabric,
        placement: Placement,
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
        self._groups = groups

    # -- shared machinery -----------------------------------------------

    def _ring_order(self, ranks: Sequence[Rank]) -> List[Rank]:
        """Domain-major ordering: every rank in one scale-up domain
        contiguous before the next domain's. A ring visiting ranks in this
        order crosses a domain boundary exactly once per domain-to-domain
        transition around the cycle -- two crossings for a two-domain
        split, the minimum any ring over a split group can achieve, and
        the "well-formed ring" this project's own measurements (task 19)
        and this task's own report reason about. Ties within a domain
        break on the rank's own tuple ordering, for determinism -- never
        dict/set iteration order.
        """
        return sorted(ranks, key=lambda r: (
            self._fabric.domain_of(self._placement.gpu(r)) or 0, r))

    @staticmethod
    def _ring_edges(ordered: Sequence[Rank]) -> List[Tuple[Rank, Rank]]:
        n = len(ordered)
        return [(ordered[i], ordered[(i + 1) % n]) for i in range(n)]

    def _round_ns(self, edges: Sequence[Tuple[Rank, Rank]], chunk_bytes: int,
                  key_prefix: str) -> int:
        """One round of simultaneous point-to-point transfers, contention-
        aware (`run_transfers`, not `isolated_durations` -- multiple edges
        in one round can share a physical link, most importantly the one
        crossing link a domain-split group's ring uses twice). A round's
        duration is gated by its slowest edge, exactly like a real
        collective's synchronous step."""
        if chunk_bytes <= 0 or not edges:
            return 0
        transfers = [
            Transfer(key=f"{key_prefix}-{i}", src=self._placement.gpu(a),
                     dst=self._placement.gpu(b), size_bytes=chunk_bytes)
            for i, (a, b) in enumerate(edges)
        ]
        completions = run_transfers(self._fabric, transfers)
        return max(c.completion_ns for c in completions)

    def _resolve_group(self, cluster_type: Optional[ClusterType],
                       comm_domain: Optional[str], num_devices: int) -> List[Rank]:
        return self._groups.resolve(cluster_type, comm_domain, num_devices)

    # -- the five true collectives, each with a stated algorithm ----------

    def predict_allreduce(self, data_size_bytes: int, num_devices: int,
                          cluster_type: Optional[ClusterType] = None,
                          comm_domain: Optional[str] = None) -> float:
        """Ring all-reduce: reduce-scatter (n-1 rounds) then all-gather
        (n-1 rounds), 2(n-1) rounds total, each moving data_size_bytes/n --
        the standard bandwidth-optimal ring volume (task 06's own report
        already cited Frontier's matching `2*(n-1)/n` closed-form factor).
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allreduce")
        ranks = self._resolve_group(cluster_type, comm_domain, num_devices)
        if num_devices <= 1:
            return 0.0
        edges = self._ring_edges(self._ring_order(ranks))
        chunk_bytes = max(1, data_size_bytes // num_devices)
        rounds = 2 * (num_devices - 1)
        return _ns_to_ms(rounds * self._round_ns(edges, chunk_bytes, "allreduce"))

    def predict_allgather(self, data_size_bytes: int, num_devices: int,
                          cluster_type: Optional[ClusterType] = None,
                          comm_domain: Optional[str] = None) -> float:
        """Ring all-gather: n-1 rounds, each moving data_size_bytes/n --
        half of allreduce's ring (the all-gather phase alone)."""
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allgather")
        ranks = self._resolve_group(cluster_type, comm_domain, num_devices)
        if num_devices <= 1:
            return 0.0
        edges = self._ring_edges(self._ring_order(ranks))
        chunk_bytes = max(1, data_size_bytes // num_devices)
        rounds = num_devices - 1
        return _ns_to_ms(rounds * self._round_ns(edges, chunk_bytes, "allgather"))

    def predict_reduce_scatter(self, data_size_bytes: int, num_devices: int,
                               cluster_type: Optional[ClusterType] = None,
                               comm_domain: Optional[str] = None) -> float:
        """Ring reduce-scatter: n-1 rounds, each moving data_size_bytes/n --
        the mirror of all-gather; same round count and per-round volume,
        so the same cost."""
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "reduce_scatter")
        ranks = self._resolve_group(cluster_type, comm_domain, num_devices)
        if num_devices <= 1:
            return 0.0
        edges = self._ring_edges(self._ring_order(ranks))
        chunk_bytes = max(1, data_size_bytes // num_devices)
        rounds = num_devices - 1
        return _ns_to_ms(rounds * self._round_ns(edges, chunk_bytes, "reduce_scatter"))

    def predict_all_to_all(self, data_size_bytes: int, num_devices: int,
                           cluster_type: Optional[ClusterType] = None,
                           comm_domain: Optional[str] = None) -> float:
        """Full pairwise exchange, one round: every ordered pair of
        participants exchanges data_size_bytes/n^2 simultaneously. Unlike a
        ring, this genuinely does cross a domain boundary once for every
        cross-domain *pair* -- 16 crossing edges for a 4-and-4 split of 8,
        not 2 -- because expert dispatch is not a ring: every rank has
        distinct data for every other rank, so there is no shortcut through
        an intermediate hop the way a reduction's associativity allows.

        Task 21's own correction to task 20: `data_size_bytes` is Frontier's
        same global-buffer convention as `predict_allreduce`'s (confirmed
        against the real call site,
        `sklearn_moe_execution_time_predictor.py`'s
        `data_size_bytes = embedding_dim * 2 * routed_tokens` -- a total
        across the whole dispatch, not one rank's share). Each of the n
        source ranks holds `data_size_bytes/n` of that total, and a
        personalised all-to-all splits *that* evenly across its `n`
        possible destinations -- `data_size_bytes/n^2` per pair, not
        `data_size_bytes/n` (task 20's original, over-charging by a factor
        of `n`; task 20's own report called the pair *set* correct without
        checking the per-pair volume, which is what this task's spec asked
        to verify and found wrong).
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "all_to_all")
        ranks = self._resolve_group(cluster_type, comm_domain, num_devices)
        if num_devices <= 1:
            return 0.0
        chunk_bytes = max(1, data_size_bytes // (num_devices * num_devices))
        edges = [(a, b) for a in ranks for b in ranks if a != b]
        return _ns_to_ms(self._round_ns(edges, chunk_bytes, "all_to_all"))

    def predict_broadcast(self, data_size_bytes: int, num_devices: int,
                         cluster_type: Optional[ClusterType] = None,
                         comm_domain: Optional[str] = None) -> float:
        """Sequential ring relay: the source forwards the full payload to
        its ring neighbour, which forwards it on, for n-1 hops -- simple
        and defensible, not claimed optimal (a real broadcast would more
        likely use a tree); not exercised by this task's acceptance tests,
        which are about allreduce specifically."""
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "broadcast")
        ranks = self._resolve_group(cluster_type, comm_domain, num_devices)
        if num_devices <= 1:
            return 0.0
        ordered = self._ring_order(ranks)
        total_ns = 0
        for i in range(len(ordered) - 1):
            total_ns += self._round_ns([(ordered[i], ordered[i + 1])],
                                       data_size_bytes, f"broadcast-{i}")
        return _ns_to_ms(total_ns)

    # -- point-to-point ---------------------------------------------------
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
        ranks = self._resolve_group(cluster_type, comm_domain, 2)
        if len(ranks) != 2:
            raise ValueError(
                f"send_recv needs exactly 2 ranks, registry returned "
                f"{len(ranks)} for cluster_type={cluster_type!r}, "
                f"comm_domain={comm_domain!r}")
        src = self._placement.gpu(ranks[0])
        dst = self._placement.gpu(ranks[1])
        t = Transfer(key="send_recv", src=src, dst=dst, size_bytes=data_size_bytes)
        completions = run_transfers(self._fabric, [t])
        return _ns_to_ms(completions[0].completion_ns)
