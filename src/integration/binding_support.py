"""Shared transfer pricing for the KV and M2N predictors, once a
destination pool has more than one candidate replica.

Both predictors already resolve source and destination ranks, build one
`engine.network.transfers.Transfer`, and convert nanoseconds to
milliseconds -- task 09's and task 11's own pattern, duplicated between the
two files. Task 14 adds a third path (more than one destination candidate,
consulting `ctx.binding`) that would otherwise be duplicated a third time;
this module is where it lives instead, called from both predictors.

Source-side ambiguity is deliberately out of scope: task 14's own framing is
"which replica *receives* a transfer" (spec S1), not which one sends it, so
`resolve_pool` for the *source* pool still raises unchanged if it is itself
ambiguous. Binding only ever resolves a destination.
"""
from __future__ import annotations

from statistics import mean
from typing import Optional, Tuple

from engine.network.transfers import Transfer, isolated_durations
from engine.placement.binding import Candidate, bind

from .cc_backend.comm_groups import CommGroupError
from .context import EngineContext

_NS_PER_MS = 1_000_000.0


def _ns_to_ms(duration_ns: float) -> float:
    """The one conversion point between the engine's nanoseconds and
    Frontier's milliseconds. Both sides are floats, so there is no integer
    rounding direction to pick."""
    return duration_ns / _NS_PER_MS


def price_transfer(ctx: EngineContext, source_cluster_type, target_cluster_type,
                   size_bytes: int, key: str) -> Tuple[float, Optional[int]]:
    """Resolve source and destination ranks for one transfer and price it in
    milliseconds.

    Returns `(price_ms, chosen_replica_id)`. `chosen_replica_id` is `None`
    when the destination pool was unambiguous (today's single-replica case,
    unaffected by any of this) or when `ctx.binding.timing == "late"` (no
    destination is chosen at all -- see `EngineContext`'s `BindingConfig`
    docstring for why). It carries a real replica id only under
    `timing == "early"`, once `bind()` has actually committed to one --
    which is also what `state` records the assignment against, and what a
    study can compare against Frontier's own eventual choice (task 14 report
    S3).

    Raises `CommGroupError` (unchanged from tasks 09/11) if the destination
    pool has more than one replica and `ctx.binding` is not configured --
    refusing beats guessing, same as ever.
    """
    src_ranks = ctx.groups.resolve_pool(source_cluster_type)
    src_gpu = ctx.placement.gpu(src_ranks[0])

    try:
        dst_ranks = ctx.groups.resolve_pool(target_cluster_type)
    except CommGroupError:
        if ctx.binding is None:
            raise
        candidates = [Candidate(rid, tuple(ranks)) for rid, ranks in
                     ctx.groups.resolve_pool_candidates(target_cluster_type)]

        if ctx.binding.timing == "late":
            durations = []
            for i, candidate in enumerate(candidates):
                dst_gpu = ctx.placement.gpu(candidate.ranks[0])
                t = Transfer(key=f"{key}-late-{i}", src=src_gpu, dst=dst_gpu,
                             size_bytes=size_bytes)
                durations.append(isolated_durations(ctx.fabric, [t])[t.key])
            return _ns_to_ms(mean(durations)), None

        chosen = bind(ctx.binding.policy, src_ranks, candidates, ctx.binding.state,
                     fabric=ctx.fabric, placement=ctx.placement)
        dst_gpu = ctx.placement.gpu(chosen.ranks[0])
        t = Transfer(key=key, src=src_gpu, dst=dst_gpu, size_bytes=size_bytes)
        duration_ns = isolated_durations(ctx.fabric, [t])[t.key]
        return _ns_to_ms(duration_ns), chosen.replica_id

    dst_gpu = ctx.placement.gpu(dst_ranks[0])
    t = Transfer(key=key, src=src_gpu, dst=dst_gpu, size_bytes=size_bytes)
    duration_ns = isolated_durations(ctx.fabric, [t])[t.key]
    return _ns_to_ms(duration_ns), None
