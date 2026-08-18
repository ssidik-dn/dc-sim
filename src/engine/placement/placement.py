"""The placement map: an injective function from logical rank to physical GPU.

This is the piece neither Frontier nor ASTRA-sim has. Frontier has no physical
host concept at all; ASTRA-sim has a fabric but no ranks to place on it.

Injectivity is enforced, not assumed -- two ranks on one GPU is a
configuration error, and silently allowing it produces plausible wrong
numbers rather than a crash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..logical.deployment import Deployment, ParallelGroup, ParallelKind, Rank
from ..physical.topology import Fabric, GpuId, Link


class PlacementError(ValueError):
    pass


@dataclass
class Placement:
    fabric: Fabric
    mapping: Dict[Rank, GpuId] = field(default_factory=dict)
    policy: str = "unspecified"

    # -- construction ------------------------------------------------------
    def assign(self, rank: Rank, gpu: GpuId) -> None:
        if gpu in self.mapping.values():
            other = next(r for r, g in self.mapping.items() if g == gpu)
            raise PlacementError(
                f"GPU {gpu} already holds rank {other}; cannot also hold {rank}")
        if rank in self.mapping:
            raise PlacementError(f"rank {rank} already placed at {self.mapping[rank]}")
        self.mapping[rank] = gpu

    def validate(self) -> None:
        gpus = list(self.mapping.values())
        if len(gpus) != len(set(gpus)):
            raise PlacementError("placement is not injective")
        known = set(self.fabric.gpus)
        for r, g in self.mapping.items():
            if g not in known:
                raise PlacementError(f"rank {r} placed on unknown GPU {g}")

    # -- queries -----------------------------------------------------------
    def gpu(self, rank: Rank) -> GpuId:
        try:
            return self.mapping[rank]
        except KeyError:
            raise PlacementError(f"rank {rank} is not placed") from None

    def gpus_for(self, ranks: Iterable[Rank]) -> List[GpuId]:
        return [self.gpu(r) for r in ranks]

    def domains_spanned(self, ranks: Iterable[Rank]) -> set:
        return {self.fabric.domain_of(self.gpu(r)) for r in ranks}

    def crosses_scale_up_boundary(self, ranks: Iterable[Rank]) -> bool:
        return len(self.domains_spanned(ranks)) > 1

    def induced_links(self, group: ParallelGroup) -> Dict[str, Link]:
        """Every link a fully-connected exchange among the group would touch.

        This is what a contention model queues on, and what gets collapsed
        into per-dimension parameters for an ASTRA-sim estimate.
        """
        gpus = self.gpus_for(group.ranks)
        pairs: List[Tuple[GpuId, GpuId]] = [
            (a, b) for i, a in enumerate(gpus) for b in gpus[i + 1:]]
        return self.fabric.link_set(pairs)

    def group_shape(self, group: ParallelGroup) -> Tuple[int, ...]:
        """How the group distributes across scale-up domains, as a sorted
        tuple. (8,) is packed; (4,4) is split evenly across two; (2,2,2,2)
        is the expensive four-way case.

        This tuple is the canonical shape signature -- most placements are
        isomorphic, so caching on it keeps memoisation cardinality bounded.
        """
        counts: Dict[Optional[int], int] = {}
        for r in group.ranks:
            d = self.fabric.domain_of(self.gpu(r))
            counts[d] = counts.get(d, 0) + 1
        return tuple(sorted(counts.values(), reverse=True))

    def summary(self) -> str:
        return (f"placement[{self.policy}]: {len(self.mapping)} ranks on "
                f"{len(set(self.mapping.values()))} GPUs of {self.fabric.name}")


# ---------------------------------------------------------------- policies

def _ordered_gpus(fabric: Fabric) -> List[GpuId]:
    return fabric.gpus


def packed(deployment: Deployment, fabric: Fabric) -> Placement:
    """Fill each scale-up domain completely before moving to the next.
    Keeps TP groups inside one domain wherever they fit."""
    p = Placement(fabric, policy="packed")
    pool: List[GpuId] = []
    for d in sorted(fabric.domains):
        pool.extend(sorted(fabric.domains[d].members))
    for g in _ordered_gpus(fabric):
        if g not in pool:
            pool.append(g)
    it = iter(pool)
    for rank in deployment.ranks:
        try:
            p.assign(rank, next(it))
        except StopIteration:
            raise PlacementError(
                f"fabric has {len(pool)} GPUs, deployment needs "
                f"{deployment.world_size}") from None
    p.validate()
    return p


def spread(deployment: Deployment, fabric: Fabric) -> Placement:
    """Round-robin across scale-up domains. Maximally splits every group --
    the expensive case, and the one that shows what placement costs."""
    p = Placement(fabric, policy="spread")
    doms = [sorted(fabric.domains[d].members) for d in sorted(fabric.domains)]
    if not doms:
        doms = [_ordered_gpus(fabric)]
    cursors = [0] * len(doms)
    di = 0
    for rank in deployment.ranks:
        placed = False
        for _ in range(len(doms)):
            d = di % len(doms)
            di += 1
            if cursors[d] < len(doms[d]):
                p.assign(rank, doms[d][cursors[d]])
                cursors[d] += 1
                placed = True
                break
        if not placed:
            raise PlacementError(
                f"fabric exhausted placing {rank}; needs {deployment.world_size} GPUs")
    p.validate()
    return p


def explicit(deployment: Deployment, fabric: Fabric,
             mapping: Dict[Rank, GpuId]) -> Placement:
    """Caller supplies the map. Used for reproducing a measured deployment."""
    p = Placement(fabric, policy="explicit")
    for rank in deployment.ranks:
        if rank not in mapping:
            raise PlacementError(f"explicit placement missing rank {rank}")
        p.assign(rank, mapping[rank])
    p.validate()
    return p


def fragmented(deployment: Deployment, fabric: Fabric,
               seed: int = 0, occupied: float = 0.0) -> Placement:
    """GPUs taken wherever free, as a real scheduler under load would.

    `occupied` is the fraction of the fabric already busy. This is the case
    only a real topology-aware simulator can cost -- a profiled lookup table
    covers the placements someone measured, and irregular ones are
    combinatorial.
    """
    import random
    rng = random.Random(seed)
    p = Placement(fabric, policy=f"fragmented(seed={seed},occupied={occupied})")
    free = _ordered_gpus(fabric)[:]
    rng.shuffle(free)
    n_busy = int(len(free) * occupied)
    free = free[n_busy:]
    if len(free) < deployment.world_size:
        raise PlacementError(
            f"{len(free)} GPUs free after {occupied:.0%} occupancy, "
            f"deployment needs {deployment.world_size}")
    for rank, gpu in zip(deployment.ranks, free):
        p.assign(rank, gpu)
    p.validate()
    return p


POLICIES: Dict[str, Callable[..., Placement]] = {
    "packed": packed,
    "spread": spread,
    "fragmented": fragmented,
}
