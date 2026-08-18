"""Builders for the platform shapes we care about.

Two are provided, and the difference between them is the point:

  build_node_scale()  8 GPUs = 1 machine = 1 scale-up domain (DGX / MI300X)
  build_rack_scale()  72 GPUs across 18 trays = 1 scale-up domain (Helios)

In the second, the scale-up domain spans machines. A model that treats the
machine as the unit cannot express it.
"""
from __future__ import annotations

from typing import List

from .topology import (Fabric, GpuId, Link, LinkClass, Machine, NicId,
                       ScaleUpDomain, SwitchId, gbps_to_GBps)


def _wire_machine(fab: Fabric, mid: int, gpus_per_machine: int,
                  nics_per_machine: int, egress_GBps: float,
                  egress_latency_ns: float) -> Machine:
    gpus = [GpuId(mid, i) for i in range(gpus_per_machine)]
    nics = [NicId(mid, i) for i in range(nics_per_machine)]
    m = Machine(mid, gpus, nics)
    fab.add_machine(m)

    # egress: GPUs share NICs round-robin. Two GPUs behind one NIC contend
    # before their traffic ever reaches a switch.
    for i, g in enumerate(gpus):
        nic = nics[i % len(nics)] if nics else None
        if nic is not None:
            fab.bind_nic(g, nic)
            fab.add_link(Link(g, nic, LinkClass.EGRESS,
                              egress_GBps, egress_latency_ns))
    return m


def _mesh_scale_up(fab: Fabric, gpus: List[GpuId], bw: float, lat: float) -> None:
    """All-to-all scale-up connectivity within a domain."""
    for i, a in enumerate(gpus):
        for b in gpus[i + 1:]:
            fab.add_link(Link(a, b, LinkClass.SCALE_UP, bw, lat))


def build_node_scale(num_machines: int = 2,
                     gpus_per_machine: int = 8,
                     nics_per_machine: int = 4,
                     scale_up_GBps: float = 400.0,
                     scale_up_latency_ns: float = 936.25,
                     nic_gbps: float = 400.0,
                     egress_latency_ns: float = 2000.0,
                     scale_out_GBps: float = 50.0,
                     scale_out_latency_ns: float = 5000.0,
                     name: str = "node-scale") -> Fabric:
    """One scale-up domain per machine. The conventional 8-GPU node."""
    fab = Fabric(name)
    egress = gbps_to_GBps(nic_gbps)
    leaf = SwitchId("leaf", 0)

    for mid in range(num_machines):
        m = _wire_machine(fab, mid, gpus_per_machine, nics_per_machine,
                          egress, egress_latency_ns)
        _mesh_scale_up(fab, m.gpus, scale_up_GBps, scale_up_latency_ns)
        fab.add_domain(ScaleUpDomain(mid, frozenset(m.gpus)))
        for nic in m.nics:
            fab.add_link(Link(nic, leaf, LinkClass.SCALE_OUT,
                              scale_out_GBps, scale_out_latency_ns))
    return fab


def build_rack_scale(num_racks: int = 1,
                     trays_per_rack: int = 18,
                     gpus_per_tray: int = 4,
                     nics_per_tray: int = 2,
                     scale_up_GBps: float = 900.0,
                     scale_up_latency_ns: float = 700.0,
                     nic_gbps: float = 800.0,
                     egress_latency_ns: float = 2000.0,
                     scale_out_GBps: float = 100.0,
                     scale_out_latency_ns: float = 5000.0,
                     name: str = "rack-scale") -> Fabric:
    """Helios-shaped: one scale-up domain per rack, spanning every tray in it.

    Defaults give 18 x 4 = 72 GPUs per rack. The scale-up domain crosses
    chassis boundaries, which is the case node-scale models cannot express.
    """
    fab = Fabric(name)
    egress = gbps_to_GBps(nic_gbps)
    spine = SwitchId("spine", 0)
    mid = 0

    for rack in range(num_racks):
        rack_gpus: List[GpuId] = []
        leaf = SwitchId("leaf", rack)
        for _ in range(trays_per_rack):
            m = _wire_machine(fab, mid, gpus_per_tray, nics_per_tray,
                              egress, egress_latency_ns)
            rack_gpus.extend(m.gpus)
            for nic in m.nics:
                fab.add_link(Link(nic, leaf, LinkClass.SCALE_OUT,
                                  scale_out_GBps, scale_out_latency_ns))
            mid += 1
        # one scale-up domain spanning the whole rack
        _mesh_scale_up(fab, rack_gpus, scale_up_GBps, scale_up_latency_ns)
        fab.add_domain(ScaleUpDomain(rack, frozenset(rack_gpus)))
        if num_racks > 1:
            fab.add_link(Link(leaf, spine, LinkClass.SCALE_OUT,
                              scale_out_GBps * 4, scale_out_latency_ns))
    return fab
