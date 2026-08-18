"""The logical side: what the serving system wants, independent of hardware.

Mirrors Frontier's three-pool pd-af-disaggregation architecture. A replica
belongs to a pool and owns a set of ranks; parallel groups partition those
ranks by parallelism kind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class PoolKind(Enum):
    PREFILL = "PREFILL"
    DECODE_ATTN = "DECODE_ATTN"
    DECODE_FFN = "DECODE_FFN"
    DECODE = "DECODE"            # pd-disaggregation: attn and ffn merged


class ParallelKind(Enum):
    TP = "TP"
    PP = "PP"
    DP = "DP"
    EP = "EP"


@dataclass(frozen=True, order=True)
class Rank:
    pool: str
    replica: int
    index: int

    def __str__(self) -> str:
        return f"{self.pool}/r{self.replica}/{self.index}"


@dataclass
class ParallelGroup:
    kind: ParallelKind
    ranks: List[Rank]

    @property
    def size(self) -> int:
        return len(self.ranks)


@dataclass
class Replica:
    pool: PoolKind
    replica_id: int
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp * self.ep

    @property
    def ranks(self) -> List[Rank]:
        return [Rank(self.pool.value, self.replica_id, i)
                for i in range(self.world_size)]

    def groups(self, kind: ParallelKind) -> List[ParallelGroup]:
        """Partition ranks into groups of the given parallelism kind.

        Rank layout is TP-innermost, then EP, DP, PP -- the conventional
        ordering, and the one that makes TP groups contiguous so a packed
        placement keeps them inside one scale-up domain.
        """
        sizes = {ParallelKind.TP: self.tp, ParallelKind.EP: self.ep,
                 ParallelKind.DP: self.dp, ParallelKind.PP: self.pp}
        order = [ParallelKind.TP, ParallelKind.EP, ParallelKind.DP, ParallelKind.PP]
        stride = 1
        for k in order:
            if k == kind:
                break
            stride *= sizes[k]
        n = sizes[kind]
        if n == 1:
            return []
        ranks = self.ranks
        out: List[ParallelGroup] = []
        total = len(ranks)
        block = stride * n
        for base in range(0, total, block):
            for off in range(stride):
                members = [ranks[base + off + j * stride] for j in range(n)]
                out.append(ParallelGroup(kind, members))
        return out


@dataclass
class Deployment:
    """A complete logical deployment: pools, each with replicas."""
    name: str = "deployment"
    replicas: List[Replica] = field(default_factory=list)

    def add(self, replica: Replica) -> Replica:
        self.replicas.append(replica)
        return replica

    def pool(self, kind: PoolKind) -> List[Replica]:
        return [r for r in self.replicas if r.pool == kind]

    @property
    def ranks(self) -> List[Rank]:
        return [r for rep in self.replicas for r in rep.ranks]

    @property
    def world_size(self) -> int:
        return sum(r.world_size for r in self.replicas)

    def summary(self) -> str:
        by: Dict[str, int] = {}
        for r in self.replicas:
            by[r.pool.value] = by.get(r.pool.value, 0) + r.world_size
        parts = [f"{k}={v}" for k, v in sorted(by.items())]
        return (f"{self.name}: {len(self.replicas)} replicas, "
                f"{self.world_size} ranks ({', '.join(parts)})")
