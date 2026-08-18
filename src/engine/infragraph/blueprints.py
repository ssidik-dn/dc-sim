"""Blueprints: build a Fabric from a handful of parameters instead of
enumerating every link by hand. The two the ASTRA-sim 3.0 paper names.

Deliberately independent of `physical/builders.py` rather than wrapping it.
`single_tier_fabric` is meant to be cross-checked against `build_node_scale()`
by test (`tests/test_blueprints.py::test_single_tier_matches_build_node_scale`)
-- if it just called the builder under the hood, that comparison would prove
nothing.
"""
from __future__ import annotations

from ..physical.topology import (Fabric, GpuId, Link, LinkClass, Machine,
                                 NicId, ScaleUpDomain, SwitchId,
                                 gbps_to_GBps)


def _wire_machine(fab: Fabric, mid: int, gpus_per_machine: int,
                  nics_per_machine: int, leaf: SwitchId,
                  scale_up_GBps: float, scale_up_latency_ns: float,
                  egress_GBps: float, egress_latency_ns: float,
                  downlink_GBps: float, downlink_latency_ns: float) -> None:
    """One machine: GPUs meshed over scale-up, NICs shared round-robin for
    egress, each NIC uplinked to `leaf`. One scale-up domain per machine.

    Written independently of builders.py's own `_wire_machine` /
    `_mesh_scale_up` on purpose -- see the module docstring.
    """
    gpus = [GpuId(mid, i) for i in range(gpus_per_machine)]
    nics = [NicId(mid, i) for i in range(nics_per_machine)]
    fab.add_machine(Machine(mid, gpus, nics))

    for i, a in enumerate(gpus):
        for b in gpus[i + 1:]:
            fab.add_link(Link(a, b, LinkClass.SCALE_UP,
                              scale_up_GBps, scale_up_latency_ns))

    for i, g in enumerate(gpus):
        nic = nics[i % len(nics)] if nics else None
        if nic is not None:
            fab.bind_nic(g, nic)
            fab.add_link(Link(g, nic, LinkClass.EGRESS,
                              egress_GBps, egress_latency_ns))

    for nic in nics:
        fab.add_link(Link(nic, leaf, LinkClass.SCALE_OUT,
                          downlink_GBps, downlink_latency_ns))

    fab.add_domain(ScaleUpDomain(mid, frozenset(gpus)))


def single_tier_fabric(
    num_machines: int,
    gpus_per_machine: int,
    nics_per_machine: int,
    scale_up_GBps: float,
    scale_up_latency_ns: float,
    nic_gbps: float,
    egress_latency_ns: float,
    scale_out_GBps: float,
    scale_out_latency_ns: float,
    name: str = "single-tier",
) -> Fabric:
    """Flat, single-switch-layer topology: every NIC uplinks to one leaf
    switch. One scale-up domain per machine -- the same shape as
    `build_node_scale()`, which this is cross-checked against by test.
    """
    fab = Fabric(name)
    egress_GBps = gbps_to_GBps(nic_gbps)
    leaf = SwitchId("leaf", 0)
    for mid in range(num_machines):
        _wire_machine(fab, mid, gpus_per_machine, nics_per_machine, leaf,
                     scale_up_GBps, scale_up_latency_ns,
                     egress_GBps, egress_latency_ns,
                     scale_out_GBps, scale_out_latency_ns)
    return fab


def clos_fat_tree_fabric(
    switch_radix: int,
    depth: int = 2,
    gpus_per_machine: int = 1,
    nics_per_machine: int = 1,
    oversubscription: float = 1.0,
    scale_up_GBps: float = 400.0,
    scale_up_latency_ns: float = 936.25,
    nic_gbps: float = 400.0,
    egress_latency_ns: float = 2000.0,
    scale_out_GBps: float = 50.0,
    scale_out_latency_ns: float = 5000.0,
    name: str = "leaf-spine",
) -> Fabric:
    """A two-tier leaf-spine fabric built from radix-`switch_radix` switches.

    Task 02 built this as a folded, pod-structured Clos using counts the
    ASTRA-sim 3.0 paper gives for `depth=2`:

        pods = k, leaves/pod = k/2, spines = (k/2)^2, hosts/leaf = k/2

    Those are the *edge and core* counts of a three-tier Al-Fares fat tree
    with the aggregation tier removed, and removing that tier disconnects
    the topology: a leaf has only k/2 uplinks but there are k^2/4 spines,
    so it reaches a k/2 subset, and leaves at different pod positions land
    on disjoint subsets. Task 02 patched the resulting gap with ad hoc
    intra-pod leaf-to-leaf links -- a workaround for a wrong spec, not a
    real leaf-spine design, and it let same-leaf-position traffic bypass
    the uplinks entirely, which understates exactly the oversubscription
    pressure this blueprint exists to model.

    This is the actual two-tier construction (no pods, no aggregation
    tier, no leaf-to-leaf links -- a full leaf<->spine mesh instead):

        leaf switch:  k ports = k/2 down to hosts + k/2 up to spines
        spine switch: k ports = k down to leaves

        spines            = k/2
        leaves            = k
        hosts per leaf    = k/2
        total hosts       = k * k/2 = k^2/2
        leaf-spine links  = leaves * spines = k^2/2      (full mesh)

    Every spine's k ports are exactly filled (one per leaf); every leaf's
    k/2 uplink ports are exactly filled (one per spine). Any two hosts are
    at most two switch-hops apart: one hop (their shared leaf) if they sit
    behind the same leaf, otherwise leaf -> spine -> leaf.

    depth=1 has no spine layer and delegates to `single_tier_fabric`,
    treating switch_radix as the host count directly (no uplinks needed, so
    every port serves a host). depth>2 raises rather than approximate a
    three-tier fat tree: it needs a real aggregation tier and different
    counts, not a guess at wiring one in by hand.
    """
    if depth > 2:
        raise NotImplementedError(
            f"depth={depth} fat trees are not implemented; a three-tier "
            f"topology needs an aggregation tier and different counts, and "
            f"wiring one in by hand risks a plausible-looking wrong answer, "
            f"so this raises instead of approximating one")

    if depth == 1:
        return single_tier_fabric(
            num_machines=switch_radix,
            gpus_per_machine=gpus_per_machine,
            nics_per_machine=nics_per_machine,
            scale_up_GBps=scale_up_GBps,
            scale_up_latency_ns=scale_up_latency_ns,
            nic_gbps=nic_gbps,
            egress_latency_ns=egress_latency_ns,
            scale_out_GBps=scale_out_GBps,
            scale_out_latency_ns=scale_out_latency_ns,
            name=name,
        )

    k = switch_radix
    if k % 2 != 0:
        raise ValueError(
            f"switch_radix must be even for a leaf-spine fabric (got {k}); "
            f"an odd radix makes k/2 non-integral and the standard "
            f"construction doesn't apply")

    half = k // 2
    num_leaves, num_spines, hosts_per_leaf = k, half, half

    egress_GBps = gbps_to_GBps(nic_gbps)

    # Oversubscription is a ratio of aggregates, not a per-link fraction: at
    # 4:1 the SUM of a leaf's uplinks is a quarter of the SUM of its
    # downlinks, not a quarter of one downlink. Aggregate downlink per leaf
    # is (hosts_per_leaf * nics_per_machine * scale_out_GBps), split evenly
    # over `num_spines` uplinks; since hosts_per_leaf == num_spines (both
    # equal `half` in this construction) that count cancels, leaving this.
    uplink_GBps = (nics_per_machine * scale_out_GBps) / oversubscription

    fab = Fabric(name)
    spines = [SwitchId("spine", j) for j in range(num_spines)]
    mid = 0
    for leaf_idx in range(num_leaves):
        leaf = SwitchId("leaf", leaf_idx)
        for _ in range(hosts_per_leaf):
            _wire_machine(fab, mid, gpus_per_machine, nics_per_machine, leaf,
                         scale_up_GBps, scale_up_latency_ns,
                         egress_GBps, egress_latency_ns,
                         scale_out_GBps, scale_out_latency_ns)
            mid += 1
        for spine in spines:
            fab.add_link(Link(leaf, spine, LinkClass.SCALE_OUT,
                              uplink_GBps, scale_out_latency_ns))

    return fab
