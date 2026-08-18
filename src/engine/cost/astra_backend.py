"""Invoke ASTRA-sim for an isolated collective cost, and cache the result.

Measured on the reference box: 4-5 ms per invocation, dominated by process
startup rather than simulation. A single Frontier run issued 31,744
collectives -- about two and a half minutes of pure ASTRA-sim if each were a
separate call.

But those calls collapse to a handful of distinct (collective, size, shape,
bandwidths) tuples. Memoisation is not an optimisation here; it is what makes
ASTRA-sim usable at all. The hit rate is therefore a first-class metric: if it
falls, the cache key has become too specific and cost will blow up.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from ..physical.topology import Fabric
from .astra_config import (AstraTopology, NotExpressible, cache_key,
                           emit_config_pair, topology_for_shape)

_COMM_TIME = re.compile(r"Comm time:\s*([0-9.eE+-]+)")


@dataclass
class CostResult:
    duration_ns: float
    ranks_reported: int
    cached: bool = False
    backend: str = ""


@dataclass
class CostBackend(ABC):
    """Isolated cost of one collective on one placement. No contention here
    by construction -- see the causality argument: a function evaluated when
    an operation starts cannot know about operations that begin later."""

    @abstractmethod
    def estimate(self, collective: str, size_bytes: int,
                 shape: Sequence[int], fabric: Fabric) -> CostResult: ...


@dataclass
class MockBackend(CostBackend):
    """Deterministic stand-in. Lets every test above this layer run without
    ASTRA-sim built, and gives the contract suite something to compare."""
    per_hop_ns: float = 5000.0
    per_byte_ns: float = 1e-5

    def estimate(self, collective, size_bytes, shape, fabric) -> CostResult:
        shape = tuple(shape)
        hops = max(len(shape) - 1, 0) * 2 + 1
        d = hops * self.per_hop_ns + size_bytes * self.per_byte_ns
        return CostResult(d, sum(shape), backend="mock")


@dataclass
class AstraSimBackend(CostBackend):
    """Real ASTRA-sim behind the estimate path.

    workload_root points at a directory of Chakra traces named
    <collective>/<n>npus_<k>B/, matching what the sweep tooling generates.
    """
    binary: Path
    remote_memory_config: Path
    workload_root: Path
    work_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp()))
    cache: Dict[str, CostResult] = field(default_factory=dict)
    calls: int = 0
    hits: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.calls if self.calls else 0.0

    def _workload(self, collective: str, ranks: int, size_bytes: int) -> Path:
        d = self.workload_root / collective / f"{ranks}npus_{size_bytes}B"
        if not d.is_dir():
            raise FileNotFoundError(
                f"no Chakra trace at {d}; generate it before estimating")
        return d / collective

    def estimate(self, collective, size_bytes, shape, fabric) -> CostResult:
        self.calls += 1
        topo = topology_for_shape(shape, fabric)
        key = cache_key(shape, collective, size_bytes, topo)
        if key in self.cache:
            self.hits += 1
            hit = self.cache[key]
            return CostResult(hit.duration_ns, hit.ranks_reported,
                              cached=True, backend="astra-sim")

        pair = emit_config_pair(topo, self.work_dir)
        workload = self._workload(collective, topo.total_ranks, size_bytes)

        proc = subprocess.run(
            [str(self.binary),
             f"--workload-configuration={workload}",
             f"--system-configuration={pair.system_path}",
             f"--network-configuration={pair.network_path}",
             f"--remote-memory-configuration={self.remote_memory_config}"],
            capture_output=True, text=True)

        times = [float(m) for m in _COMM_TIME.findall(proc.stdout + proc.stderr)]
        if not times:
            raise RuntimeError(
                f"ASTRA-sim produced no timing output (exit {proc.returncode}). "
                f"stderr tail: {(proc.stderr or '').strip()[-300:]}")

        # Rank count is the guard against the silent dimension-mismatch trap:
        # if ASTRA-sim drops a dimension it simulates fewer ranks and returns
        # a plausible, lower, wrong number.
        if len(times) != topo.total_ranks:
            raise RuntimeError(
                f"ASTRA-sim reported {len(times)} ranks, expected "
                f"{topo.total_ranks}. The configs likely disagree on "
                f"dimension count.")

        # A collective completes when its slowest rank does.
        res = CostResult(max(times), len(times), backend="astra-sim")
        self.cache[key] = res
        return res


@dataclass
class FallbackBackend(CostBackend):
    """Use ASTRA-sim where the shape is expressible, something else where it
    is not. Fragmented placements produce uneven shapes like (3, 3, 2), which
    ASTRA-sim's product-of-dimensions format cannot represent -- and those are
    exactly the placements a profiled lookup table also cannot cover."""
    primary: CostBackend
    fallback: CostBackend
    fallbacks_used: int = 0

    def estimate(self, collective, size_bytes, shape, fabric) -> CostResult:
        try:
            return self.primary.estimate(collective, size_bytes, shape, fabric)
        except NotExpressible:
            self.fallbacks_used += 1
            r = self.fallback.estimate(collective, size_bytes, shape, fabric)
            return CostResult(r.duration_ns, r.ranks_reported, r.cached,
                              backend=r.backend + " (fallback)")
