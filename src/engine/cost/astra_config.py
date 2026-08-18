"""Collapse a placement into ASTRA-sim's per-dimension topology description.

ASTRA-sim describes a topology as parallel lists -- one entry per dimension:

    topology:   [ Switch, Switch ]
    npus_count: [ 4, 2 ]        # PRODUCT: 4 per group x 2 groups = 8 ranks
    bandwidth:  [ 400.0, 50.0 ]
    latency:    [ 936.25, 5000.0 ]

Two traps this module exists to prevent.

1. The system config declares one collective algorithm PER DIMENSION and must
   agree with the network config on dimension count. A mismatch does not
   error -- ASTRA-sim silently drops the extra dimensions and returns a
   plausible wrong answer. Measured: a 2-dim topology run against a 1-dim
   system config reported a LOWER cost than the packed baseline, which is the
   opposite of the truth. So the two files are emitted together, by one
   function, and the dimension count is asserted.

2. npus_count multiplies across dimensions, so only EVEN splits are
   expressible. A group spread 4+4 is [4, 2]. A group spread 3+3+2 is not
   expressible at all -- and fragmented placement produces exactly that.
   NotExpressible is raised rather than silently rounding, because a wrong
   answer here looks entirely reasonable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..physical.topology import Fabric, LinkClass


class NotExpressible(ValueError):
    """The placement shape cannot be written as ASTRA-sim dimensions."""


@dataclass(frozen=True)
class AstraTopology:
    """A dimension-list topology, plus the signature it was derived from."""
    topology: Tuple[str, ...]
    npus_count: Tuple[int, ...]
    bandwidth_GBps: Tuple[float, ...]
    latency_ns: Tuple[float, ...]
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.topology)
        if not (len(self.npus_count) == len(self.bandwidth_GBps)
                == len(self.latency_ns) == n):
            raise ValueError("dimension lists must be the same length")
        if n == 0:
            raise ValueError("topology needs at least one dimension")

    @property
    def dims(self) -> int:
        return len(self.topology)

    @property
    def total_ranks(self) -> int:
        """npus_count is a product across dimensions, not a sum."""
        t = 1
        for n in self.npus_count:
            t *= n
        return t

    def network_yaml(self) -> str:
        j = lambda vals: ", ".join(str(v) for v in vals)
        return (f"topology:   [ {j(self.topology)} ]\n"
                f"npus_count: [ {j(self.npus_count)} ]\n"
                f"bandwidth:  [ {j(self.bandwidth_GBps)} ]\n"
                f"latency:    [ {j(self.latency_ns)} ]\n")

    def system_json(self, collective: str = "ring",
                    chunks_per_dim: int = 2,
                    dataset_splits: int = 4,
                    local_mem_bw: float = 3350.0) -> str:
        """One algorithm entry per dimension. This is the half of the pair
        that silently breaks when it disagrees with the network config."""
        impl = [collective] * self.dims
        return json.dumps({
            "scheduling-policy": "LIFO",
            "endpoint-delay": 10,
            "active-chunks-per-dimension": chunks_per_dim,
            "preferred-dataset-splits": dataset_splits,
            "all-reduce-implementation": impl,
            "all-gather-implementation": impl,
            "reduce-scatter-implementation": impl,
            "all-to-all-implementation": impl,
            "collective-optimization": "localBWAware",
            "local-mem-bw": local_mem_bw,
            "boost-mode": 0,
        }, indent=2) + "\n"


def _representative_bandwidth(fabric: Fabric, cls: LinkClass,
                              default: float) -> Tuple[float, float]:
    """Slowest link of a class, with its latency. Slowest rather than mean,
    because a collective is paced by its worst hop."""
    vals = [(lk.capacity_GBps, lk.latency_ns)
            for lk in fabric.links if lk.link_class is cls]
    if not vals:
        return default, 1000.0
    return min(vals, key=lambda v: v[0])


def topology_for_shape(shape: Sequence[int], fabric: Fabric,
                       topology_kind: str = "Switch") -> AstraTopology:
    """Build a dimension-list topology from a placement shape.

    shape (8,)        -> 1 dim, npus_count [8]          all in one domain
    shape (4, 4)      -> 2 dims, npus_count [4, 2]      even split, two domains
    shape (2,2,2,2)   -> 2 dims, npus_count [2, 4]      even split, four domains
    shape (3, 3, 2)   -> NotExpressible                 uneven
    """
    shape = tuple(shape)
    if not shape:
        raise NotExpressible("empty shape")

    su_bw, su_lat = _representative_bandwidth(fabric, LinkClass.SCALE_UP, 400.0)

    if len(shape) == 1:
        return AstraTopology((topology_kind,), (shape[0],),
                             (su_bw,), (su_lat,), shape)

    if len(set(shape)) != 1:
        raise NotExpressible(
            f"shape {shape} is uneven; npus_count multiplies across dimensions "
            f"so only even splits are expressible. Fragmented placements "
            f"routinely produce this -- fall back to the analytical backend.")

    per_domain = shape[0]
    n_domains = len(shape)

    # Leaving the machine is paced by the narrower of egress and scale-out.
    eg_bw, eg_lat = _representative_bandwidth(fabric, LinkClass.EGRESS, 50.0)
    so_bw, so_lat = _representative_bandwidth(fabric, LinkClass.SCALE_OUT, 50.0)
    inter_bw = min(eg_bw, so_bw)
    inter_lat = eg_lat + so_lat

    return AstraTopology(
        (topology_kind, topology_kind),
        (per_domain, n_domains),
        (su_bw, inter_bw),
        (su_lat, inter_lat),
        shape,
    )


@dataclass(frozen=True)
class ConfigPair:
    """Network and system configs that agree on dimension count, by
    construction. Never write one without the other."""
    network_path: Path
    system_path: Path
    topology: AstraTopology

    def verify(self) -> None:
        """Re-read both files and assert they still agree. Cheap, and it
        catches a hand-edited config before it produces a wrong number."""
        net = self.network_path.read_text()
        sys_cfg = json.loads(self.system_path.read_text())
        net_dims = net.split("npus_count:")[1].split("]")[0].count(",") + 1
        sys_dims = len(sys_cfg["all-reduce-implementation"])
        if net_dims != sys_dims:
            raise ValueError(
                f"dimension mismatch: network config has {net_dims}, system "
                f"config has {sys_dims}. ASTRA-sim would silently drop "
                f"dimensions and return a plausible wrong answer.")


def emit_config_pair(topology: AstraTopology, out_dir: Path,
                     stem: Optional[str] = None,
                     collective: str = "ring") -> ConfigPair:
    """Write both files together. The only supported way to produce them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or ("shape_" + "x".join(str(n) for n in topology.shape))

    net = out_dir / f"{stem}.network.yml"
    sysf = out_dir / f"{stem}.system.json"
    net.write_text(topology.network_yaml())
    sysf.write_text(topology.system_json(collective=collective))

    pair = ConfigPair(net, sysf, topology)
    pair.verify()
    return pair


def cache_key(shape: Sequence[int], collective: str, size_bytes: int,
              topology: AstraTopology) -> str:
    """Memoisation key. Built on the placement SHAPE, not on GPU identities,
    so isomorphic placements collapse to one entry -- which is what keeps
    ASTRA-sim invocation count bounded when a run issues tens of thousands
    of collectives."""
    bw = ",".join(f"{b:g}" for b in topology.bandwidth_GBps)
    lat = ",".join(f"{l:g}" for l in topology.latency_ns)
    sh = "x".join(str(n) for n in tuple(shape))
    return f"{collective}|{size_bytes}|{sh}|{bw}|{lat}"
