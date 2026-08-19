"""Shared transfer pricing for the KV and M2N predictors, once a source or
destination pool has more than one candidate replica.

Both predictors already resolve source and destination ranks, build one
`engine.network.transfers.Transfer`, and convert nanoseconds to
milliseconds -- task 09's and task 11's own pattern, duplicated between the
two files. Task 14 added a second path (more than one *destination*
candidate, consulting `ctx.binding`) that would otherwise be duplicated a
third time; task 16 adds the symmetric *source* path task 14 deliberately
left alone ("which replica *receives* a transfer", spec S1) -- Task 15's
own study found this was no longer optional: activation exchange is a
round trip (attention sends to FFN, FFN sends back), so a destination-only
fix left the return leg raising the moment DECODE_FFN had more than one
replica, which is exactly the scenario task 15 needed and could not use
its own predictor for.

Task 16 S1's finding, load-bearing for everything below: **the sending
replica is not always ambiguous the way a destination genuinely is.**
Frontier already knows which replica sent an M2N transfer; the predictor
just isn't told, in most of its call signature. But `batch` -- confirmed by
instrumenting a real multi-lane, multi-replica run, not assumed from
reading source -- carries `decode_attn_original_dp_id`, the DECODE_ATTN dp
lane a transfer belongs to, correctly populated on *both* legs of the round
trip (forward and return). Where that identity is present, `_rank_within_pool`
uses it directly -- no guessing, no `bind()`, no `BindingState` entry.
`bind()` remains the fallback for the one identity that genuinely is not
recoverable this way: *which FFN replica* is sending on the return leg (or
receiving on the forward leg, in the have-many-destinations case task 14
already covered) -- Frontier's own bookkeeping does not expose that to
anything `get_transfer_time` receives, at either call site
(cluster_batch_end_event.py). See the task 16 report S1 for the full
investigation, including why this differs from `layer_id` (task 08),
which really was a value the caller *always* derives itself from data the
predictor also has.
"""
from __future__ import annotations

from statistics import mean
from typing import Any, List, Optional, Tuple

from engine.logical.deployment import Rank
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


def _rank_within_pool(ranks: List[Rank], batch: Any) -> Rank:
    """Refines a resolved (unambiguous) pool's representative rank to the
    exact dp lane a specific M2N transfer belongs to, when that identity is
    recoverable (task 16 report S1: `batch.decode_attn_original_dp_id`).
    Falls back to `ranks[0]` -- the behaviour every measurement before task
    16 depended on -- whenever `batch` is `None` (every KV call: task 15
    found `get_transfer_info_for_request` always passes `batch=None`) or
    doesn't carry this attribute (any non-M2N caller).

    This assumes the recovered dp lane only ever needs applying to the pool
    it was recorded for (DECODE_ATTN) -- true throughout this project,
    where every FFN/PREFILL replica is tp=1 (a single rank), so `len(ranks)`
    naturally guards against misapplying an ATTN lane index to a
    differently-shaped pool.
    """
    dp_id = getattr(batch, "decode_attn_original_dp_id", None) if batch is not None else None
    if isinstance(dp_id, int) and 0 <= dp_id < len(ranks):
        return ranks[dp_id]
    return ranks[0]


def _try_resolve_pool(ctx: EngineContext, cluster_type) -> Tuple[Optional[List[Rank]], Optional[CommGroupError]]:
    try:
        return ctx.groups.resolve_pool(cluster_type), None
    except CommGroupError as e:
        return None, e


def _price(ctx: EngineContext, src_gpu, dst_gpu, size_bytes: int, key: str) -> float:
    t = Transfer(key=key, src=src_gpu, dst=dst_gpu, size_bytes=size_bytes)
    return _ns_to_ms(isolated_durations(ctx.fabric, [t])[t.key])


def _resolve_ambiguous_side(ctx: EngineContext, fixed_gpu, fixed_ranks: List[Rank],
                            candidates: List[Candidate], size_bytes: int, key: str,
                            *, fixed_is_source: bool) -> Tuple[float, Optional[int]]:
    """`fixed_*` is whichever side already resolved to one exact rank (a
    single-replica pool, lane-refined); `candidates` is the ambiguous side.
    `fixed_is_source` only decides which way `Transfer(src=, dst=)` points,
    not which policy runs -- `bind()`/late-mean apply identically regardless
    of whether the ambiguous side is a destination (task 14) or a source
    (task 16)."""
    if ctx.binding.timing == "late":
        durations = []
        for i, candidate in enumerate(candidates):
            candidate_gpu = ctx.placement.gpu(candidate.ranks[0])
            src, dst = (fixed_gpu, candidate_gpu) if fixed_is_source else (candidate_gpu, fixed_gpu)
            t = Transfer(key=f"{key}-late-{i}", src=src, dst=dst, size_bytes=size_bytes)
            durations.append(isolated_durations(ctx.fabric, [t])[t.key])
        return _ns_to_ms(mean(durations)), None

    chosen = bind(ctx.binding.policy, fixed_ranks, candidates, ctx.binding.state,
                 fabric=ctx.fabric, placement=ctx.placement)
    chosen_gpu = ctx.placement.gpu(chosen.ranks[0])
    src, dst = (fixed_gpu, chosen_gpu) if fixed_is_source else (chosen_gpu, fixed_gpu)
    duration_ms = _price(ctx, src, dst, size_bytes, key)
    return duration_ms, chosen.replica_id


def price_transfer(ctx: EngineContext, source_cluster_type, target_cluster_type,
                   size_bytes: int, key: str, batch: Any = None) -> Tuple[float, Optional[int]]:
    """Resolve source and destination ranks for one transfer and price it in
    milliseconds.

    `batch` (task 16; optional, defaults to `None` so KV's call site -- which
    never has one, task 15 S1 -- is unaffected) lets a lane-identifiable M2N
    transfer be priced against its *actual* dp lane on whichever side is
    already unambiguous, instead of that pool's representative rank.

    Returns `(price_ms, chosen_replica_id)`. `chosen_replica_id` is `None`
    when both pools were unambiguous (refined by `batch` or not) or when
    `ctx.binding.timing == "late"`. It carries a real replica id only under
    `timing == "early"`, once `bind()` has committed to one -- for
    whichever side was actually ambiguous: a destination (task 14's
    original case) or, as of task 16, a source.

    Raises `CommGroupError` if the ambiguous side has more than one replica
    and `ctx.binding` is not configured -- refusing beats guessing, same as
    ever -- and also if *both* source and destination are simultaneously
    ambiguous, a combination no scenario in this project produces and which
    this function does not attempt to resolve (see the task 16 report for
    why: at that point "reference point to measure the other side from" is
    itself undefined).
    """
    src_ranks, src_error = _try_resolve_pool(ctx, source_cluster_type)
    dst_ranks, dst_error = _try_resolve_pool(ctx, target_cluster_type)

    if src_error is None and dst_error is None:
        src_gpu = ctx.placement.gpu(_rank_within_pool(src_ranks, batch))
        dst_gpu = ctx.placement.gpu(_rank_within_pool(dst_ranks, batch))
        return _price(ctx, src_gpu, dst_gpu, size_bytes, key), None

    if src_error is not None and dst_error is not None:
        raise src_error

    if ctx.binding is None:
        raise (dst_error if dst_error is not None else src_error)

    if dst_error is not None:
        fixed_rank = _rank_within_pool(src_ranks, batch)
        fixed_gpu = ctx.placement.gpu(fixed_rank)
        candidates = [Candidate(rid, tuple(ranks)) for rid, ranks in
                     ctx.groups.resolve_pool_candidates(target_cluster_type)]
        return _resolve_ambiguous_side(ctx, fixed_gpu, [fixed_rank], candidates,
                                       size_bytes, key, fixed_is_source=True)

    fixed_rank = _rank_within_pool(dst_ranks, batch)
    fixed_gpu = ctx.placement.gpu(fixed_rank)
    candidates = [Candidate(rid, tuple(ranks)) for rid, ranks in
                 ctx.groups.resolve_pool_candidates(source_cluster_type)]
    return _resolve_ambiguous_side(ctx, fixed_gpu, [fixed_rank], candidates,
                                   size_bytes, key, fixed_is_source=False)
