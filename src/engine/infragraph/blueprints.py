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
    depth: int,
    gpus_per_machine: int,
    nics_per_machine: int,
    oversubscription: float = 1.0,
    scale_up_GBps: float = 400.0,
    scale_up_latency_ns: float = 936.25,
    nic_gbps: float = 400.0,
    egress_latency_ns: float = 2000.0,
    scale_out_GBps: float = 50.0,
    scale_out_latency_ns: float = 5000.0,
    name: str = "clos",
) -> Fabric:
    """A depth-2 (leaf/spine) folded Clos built from radix-`switch_radix`
    switches, per the ASTRA-sim 3.0 paper's `ClosFatTreeFabric` blueprint.

    For depth=2 the paper gives:

        pods            = k
        leaves per pod  = k/2
        spines          = (k/2)^2
        hosts per leaf  = k/2
        total hosts     = k^3/4

    Those four counts are exactly the edge/core counts of a classic k-ary
    fat tree (Al-Fares 2008) -- MINUS its middle, aggregation, tier: this
    model has only two switch kinds, leaf and spine, so there is nothing
    between a pod's leaves but the leaves themselves. To keep the fabric
    fully connected (see the design note below) each leaf uplinks into a
    "plane": leaf position i (0..k/2-1) within its pod connects one link
    to every spine in plane i, and every spine's k ports fill up exactly --
    one link from each of the k pods' plane-i leaf. Leaves within one pod
    are additionally linked directly to each other, standing in for the
    missing aggregation tier.

    Design note -- not in the source spec, stated here because it changes
    measured behaviour: without the intra-pod leaf-to-leaf links, two hosts
    behind different leaves of the *same* pod would have no path between
    them at all (their leaves sit in different, non-overlapping planes).
    The paper's depth-2 counts don't by themselves supply that connectivity
    -- a real 3-tier fat tree gets it from the aggregation switches this
    model omits. See the task 02 report for the full reasoning.

    depth=1 has no spine layer and delegates to `single_tier_fabric`,
    treating switch_radix as the host count directly (no uplinks needed, so
    every port serves a host). depth>2 raises rather than approximate a
    three-tier Clos: a mis-wired one would produce a plausible-looking wrong
    answer, which is the failure mode this project keeps running into.
    """
    if depth > 2:
        raise NotImplementedError(
            f"depth={depth} Clos fat trees are not implemented; wiring one "
            f"by hand risks a plausible-looking wrong answer, so this "
            f"raises instead of approximating one")

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
            f"switch_radix must be even for a depth-2 Clos (got {k}); an "
            f"odd radix makes k/2 non-integral and the standard "
            f"construction doesn't apply")

    half = k // 2
    pods, leaves_per_pod, spines_per_plane, hosts_per_leaf = k, half, half, half

    egress_GBps = gbps_to_GBps(nic_gbps)

    # Oversubscription is a ratio of aggregates, not a per-link fraction: at
    # 4:1 the SUM of a leaf's uplinks is a quarter of the SUM of its
    # downlinks, not a quarter of one downlink. Aggregate downlink per leaf
    # is (hosts_per_leaf * nics_per_machine * scale_out_GBps), split evenly
    # over `spines_per_plane` uplinks; since hosts_per_leaf ==
    # spines_per_plane (both equal `half` in this construction) that count
    # cancels, leaving this. See the task 02 report for the full derivation.
    uplink_GBps = (nics_per_machine * scale_out_GBps) / oversubscription

    # Generous on purpose: this stands in for the missing aggregation tier
    # and must not become an accidental bottleneck that the oversubscription
    # tests would misattribute to the leaf-spine uplink instead.
    intra_pod_GBps = scale_out_GBps * nics_per_machine * hosts_per_leaf

    fab = Fabric(name)
    mid = 0
    leaves_in_pod = []
    for pod in range(pods):
        leaves_in_pod.clear()
        for leaf_pos in range(leaves_per_pod):
            leaf = SwitchId("leaf", pod * leaves_per_pod + leaf_pos)
            leaves_in_pod.append(leaf)
            for _ in range(hosts_per_leaf):
                _wire_machine(fab, mid, gpus_per_machine, nics_per_machine, leaf,
                             scale_up_GBps, scale_up_latency_ns,
                             egress_GBps, egress_latency_ns,
                             scale_out_GBps, scale_out_latency_ns)
                mid += 1
            plane = leaf_pos
            for spine_idx in range(spines_per_plane):
                spine = SwitchId("spine", plane * spines_per_plane + spine_idx)
                fab.add_link(Link(leaf, spine, LinkClass.SCALE_OUT,
                                  uplink_GBps, scale_out_latency_ns))

        for i, a in enumerate(leaves_in_pod):
            for b in leaves_in_pod[i + 1:]:
                fab.add_link(Link(a, b, LinkClass.SCALE_OUT,
                                  intra_pod_GBps, scale_out_latency_ns))

    return fab
