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
    oversubscription_edge_agg: float = 1.0,
    oversubscription_agg_core: float = 1.0,
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
    every port serves a host).

    depth=3 is a three-tier Al-Fares fat tree -- pods, each with an edge
    and an aggregation layer, plus a core tier shared across pods:

        pods              = k
        edge per pod      = k/2
        aggregation/pod   = k/2
        core              = (k/2)^2
        hosts per edge    = k/2
        total hosts       = k^3/4

    Task 02's own error (task 03's report) was building exactly these
    edge/core counts *without* the aggregation tier -- arithmetically
    consistent, and disconnected, because an edge switch's k/2 uplinks
    could then reach only a k/2 subset of a much larger spine set. Wiring
    the aggregation tier correctly requires a plane structure: core
    switches are grouped into k/2 planes of k/2 each, and aggregation
    switch j (0-indexed within its own pod) connects to every core switch
    in plane j, and only those. This is what makes the construction
    connected -- an aggregation switch j in pod A and aggregation switch
    j in pod B share every one of plane j's core switches, so any pod can
    reach any other through its own switch-j aggregation and any core
    switch in plane j, while an edge switch reaches every other edge
    switch *in its own pod* one tier lower, through aggregation alone,
    never touching core.

    Two independent oversubscription ratios, not one, because a real
    three-tier fabric's two uplink tiers are provisioned independently:
    `oversubscription_edge_agg` scales edge-to-aggregation capacity
    relative to the fully-provisioned base rate
    (`nics_per_machine * scale_out_GBps`) an edge switch's own aggregate
    host-facing capacity reduces to (exactly `oversubscription`'s own
    meaning at depth=2, since an edge switch's own downlink/uplink
    structure is identical to a depth=2 leaf's); `oversubscription_agg_core`
    scales aggregation-to-core capacity relative to that *same*
    fully-provisioned base rate, not relative to whatever
    `oversubscription_edge_agg` already reduced edge-to-aggregation links
    to. Measuring both against the same fixed base, rather than chaining
    the second through the first's own already-reduced result, is what
    keeps them independent -- the first implementation of this chained
    the two, and `oversubscription_agg_core` moved
    `oversubscription_edge_agg`'s own capacity as a side effect until a
    test caught it.

    depth>3 raises rather than approximate a four-tier fat tree: the same
    reasoning that deferred three tiers until their own counts and wiring
    were derived (task 03's own report) applies to a fourth, and nothing
    in this project needs one.
    """
    if depth > 3:
        raise NotImplementedError(
            f"depth={depth} fat trees are not implemented; depth=3 is as far "
            f"as this project has derived counts and wiring for (task 40), "
            f"and wiring a fourth tier in by hand risks a plausible-looking "
            f"wrong answer, so this raises instead of approximating one")

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
    egress_GBps = gbps_to_GBps(nic_gbps)

    if depth == 3:
        return _three_tier_fat_tree(
            k, half, gpus_per_machine, nics_per_machine,
            oversubscription_edge_agg, oversubscription_agg_core,
            scale_up_GBps, scale_up_latency_ns, egress_GBps, egress_latency_ns,
            scale_out_GBps, scale_out_latency_ns, name)

    num_leaves, num_spines, hosts_per_leaf = k, half, half

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


def _three_tier_fat_tree(
    k: int, half: int, gpus_per_machine: int, nics_per_machine: int,
    oversubscription_edge_agg: float, oversubscription_agg_core: float,
    scale_up_GBps: float, scale_up_latency_ns: float,
    egress_GBps: float, egress_latency_ns: float,
    scale_out_GBps: float, scale_out_latency_ns: float, name: str,
) -> Fabric:
    """The depth=3 construction `clos_fat_tree_fabric`'s own docstring
    describes. Kept as a private helper rather than inlined, the same
    reason `_wire_machine` is: the depth=2 body above it must stay
    byte-for-byte what it was before this task, and interleaving a third
    tier's own wiring into it by editing in place is exactly the kind of
    change that risks moving a line the depth=2 path also runs.

    `half = k // 2` throughout: pods_per_fabric = k, edges_per_pod =
    aggs_per_pod = num_planes = core_per_plane = hosts_per_edge = half.
    Global (not per-pod) indices for every switch id, so a pod's own
    aggregation switch `j` and a different pod's aggregation switch `j`
    -- which must be distinct switches, sharing only their *plane*, not
    their identity -- never collide.
    """
    num_pods = k
    edges_per_pod = half
    aggs_per_pod = half
    num_planes = half
    core_per_plane = half
    hosts_per_edge = half

    # Two *independent* ratios -- each measured against the same
    # fully-provisioned base rate (nics_per_machine * scale_out_GBps), not
    # against each other's already-oversubscribed result. This is what
    # makes them independent: an aggregation switch's own aggregate
    # downlink-facing capacity is defined here as what its edges_per_pod
    # downlinks would carry at the *fully-provisioned* rate (as if
    # oversubscription_edge_agg were 1.0), not at whatever
    # oversubscription_edge_agg actually reduced them to -- chaining
    # through the already-reduced edge_to_agg_GBps value instead would
    # make oversubscription_agg_core's own effect depend on
    # oversubscription_edge_agg, which is exactly the coupling a caller
    # setting "one ratio" should not get (confirmed by a test failure
    # during this task's own development, not assumed safe by
    # construction -- see the task 40 report).
    #
    # Edge -> aggregation: identical reasoning to the depth=2 leaf ->
    # spine formula, applied to one pod's own edge/aggregation pair --
    # an edge switch's aggregate downlink is (hosts_per_edge *
    # nics_per_machine * scale_out_GBps), split over its own
    # aggs_per_pod uplinks; hosts_per_edge == aggs_per_pod == half here
    # too, so that count cancels the same way.
    edge_to_agg_GBps = (nics_per_machine * scale_out_GBps) / oversubscription_edge_agg

    # Aggregation -> core: an aggregation switch's own aggregate
    # downlink-facing capacity, at the fully-provisioned rate, is
    # (edges_per_pod * nics_per_machine * scale_out_GBps), split over its
    # own core_per_plane uplinks; edges_per_pod == core_per_plane == half
    # again, so the same cancellation gives the same base-rate formula,
    # with its own ratio.
    agg_to_core_GBps = (nics_per_machine * scale_out_GBps) / oversubscription_agg_core

    fab = Fabric(name)
    # core[plane][position] -- plane p's own half switches, globally
    # indexed so no two planes' switches collide.
    core = [[SwitchId("core", p * core_per_plane + c) for c in range(core_per_plane)]
           for p in range(num_planes)]

    mid = 0
    for pod in range(num_pods):
        edges = [SwitchId("edge", pod * edges_per_pod + e) for e in range(edges_per_pod)]
        aggs = [SwitchId("aggregation", pod * aggs_per_pod + j) for j in range(aggs_per_pod)]

        for edge in edges:
            for _ in range(hosts_per_edge):
                _wire_machine(fab, mid, gpus_per_machine, nics_per_machine, edge,
                             scale_up_GBps, scale_up_latency_ns,
                             egress_GBps, egress_latency_ns,
                             scale_out_GBps, scale_out_latency_ns)
                mid += 1
            # Full mesh within this pod: every edge switch to every
            # aggregation switch of the *same* pod, one link each --
            # edges_per_pod == aggs_per_pod == half, so each edge's own
            # half uplink ports and each aggregation switch's own half
            # downlink ports are exactly filled by this loop alone.
            for agg in aggs:
                fab.add_link(Link(edge, agg, LinkClass.SCALE_OUT,
                                  edge_to_agg_GBps, scale_out_latency_ns))

        # The plane structure: aggregation switch j (local to this pod)
        # connects to every core switch in plane j, and only plane j --
        # omitting this (connecting to some other, arbitrary subset of
        # core switches instead) is exactly the mistake that disconnected
        # Task 02's own construction, one tier up from where it happened
        # there.
        for j, agg in enumerate(aggs):
            for core_switch in core[j]:
                fab.add_link(Link(agg, core_switch, LinkClass.SCALE_OUT,
                                  agg_to_core_GBps, scale_out_latency_ns))

    return fab
