"""Round-trip tests for InfraGraph serialisation.

`test_parsed_fabric_costs_identically` matters most: it proves the round
trip preserved everything the contention model actually reads, not just
everything a naive equality check would notice.
"""
from __future__ import annotations

import copy
import json

import pytest

from engine.infragraph.emit import to_infragraph
from engine.infragraph.parse import _reconstruct_fabric, from_infragraph
from engine.infragraph.validate import InfraGraphError, validate_infragraph
from engine.network.transfers import Transfer, analyse
from engine.physical.builders import build_node_scale, build_rack_scale
from engine.physical.topology import GpuId, LinkClass


def _class_counts(fabric):
    counts = {}
    for lk in fabric.links:
        counts[lk.link_class] = counts.get(lk.link_class, 0) + 1
    return counts


# ------------------------------------------------------------- round trips

def test_round_trip_node_scale():
    fab = build_node_scale(num_machines=2)
    rt = from_infragraph(to_infragraph(fab))

    assert set(rt.gpus) == set(fab.gpus)
    assert len(rt.links) == len(fab.links)
    assert _class_counts(rt) == _class_counts(fab)


def test_round_trip_rack_scale():
    fab = build_rack_scale(num_racks=1)
    rt = from_infragraph(to_infragraph(fab))

    assert len(fab.gpus) == 72
    assert set(rt.gpus) == set(fab.gpus)
    assert len(rt.links) == len(fab.links)
    assert len(rt.domains) == 1
    domain = next(iter(rt.domains.values()))
    assert domain.size == 72
    machines_touched = {g.machine for g in domain.members}
    assert len(machines_touched) == 18


def test_round_trip_preserves_link_classes():
    for fab in (build_node_scale(num_machines=2), build_rack_scale(num_racks=1)):
        rt = from_infragraph(to_infragraph(fab))
        assert _class_counts(rt) == _class_counts(fab)
        assert set(_class_counts(fab)) == set(LinkClass)


def test_round_trip_preserves_domain_membership():
    fab = build_node_scale(num_machines=2)
    rt = from_infragraph(to_infragraph(fab))
    for g in fab.gpus:
        assert rt.domain_of(g) == fab.domain_of(g)

    rack = build_rack_scale(num_racks=1)
    rt_rack = from_infragraph(to_infragraph(rack))
    domain = next(iter(rt_rack.domains.values()))
    assert domain.size == 72
    assert len({g.machine for g in domain.members}) == 18


def test_round_trip_preserves_nic_binding():
    fab = build_node_scale(num_machines=2)
    rt = from_infragraph(to_infragraph(fab))
    for g in fab.gpus:
        assert rt.nic_of(g) == fab.nic_of(g)


def test_round_trip_preserves_capacities():
    fab = build_rack_scale(num_racks=1)
    rt = from_infragraph(to_infragraph(fab))
    rt_by_id = {lk.id: lk for lk in rt.links}
    for lk in fab.links:
        other = rt_by_id[lk.id]
        assert other.capacity_GBps == lk.capacity_GBps
        assert other.latency_ns == lk.latency_ns
        assert other.link_class == lk.link_class


def test_round_trip_is_idempotent():
    """Catches link doubling: a naive emit that re-adds the reverse of an
    already-bidirectional link would grow the edge count on the second pass."""
    fab = build_node_scale(num_machines=2)
    doc1 = to_infragraph(fab)
    rt = from_infragraph(doc1)
    doc2 = to_infragraph(rt)
    assert doc1 == doc2
    assert len(doc1["edges"]) == len(doc2["edges"])
    assert json.dumps(doc1, sort_keys=True) == json.dumps(doc2, sort_keys=True)


def test_node_names_follow_convention():
    import re
    pattern = re.compile(r"^[A-Za-z0-9_]+\.\d+\.[A-Za-z0-9_]+\.\d+$")
    doc = to_infragraph(build_rack_scale(num_racks=1))
    names = set()
    for dev in doc["devices"]:
        for comp in dev["components"]:
            name = f"{dev['instance']}.{dev['index']}.{comp['component']}.{comp['index']}"
            assert pattern.match(name), name
            names.add(name)
    for edge in doc["edges"]:
        assert edge["src"] in names
        assert edge["dst"] in names


# --------------------------------------------------------------- validator

def _valid_doc():
    return to_infragraph(build_node_scale(num_machines=1, gpus_per_machine=2,
                                          nics_per_machine=1))


def test_validator_rejects_dangling_edge():
    doc = _valid_doc()
    doc["edges"].append({
        "src": "machine.0.gpu.0", "dst": "machine.99.gpu.0",
        "link_type": "scale_up", "attrs": {"bandwidth_GBps": 1.0, "latency_ns": 1.0},
    })
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)


def test_validator_rejects_unknown_schema_version():
    doc = _valid_doc()
    doc["schema_version"] = "9.9-nonexistent"
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)


def test_validator_rejects_domain_disagreement():
    doc = _valid_doc()
    # Flip one GPU's recorded domain without touching the domains list.
    gpu_comp = doc["devices"][0]["components"][0]
    assert gpu_comp["component"] == "gpu"
    gpu_comp["attrs"]["scale_up_domain"] = 999
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)


def test_validator_rejects_nonpositive_bandwidth():
    doc = _valid_doc()
    doc["edges"][0]["attrs"]["bandwidth_GBps"] = 0.0
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)

    doc2 = _valid_doc()
    doc2["edges"][0]["attrs"]["bandwidth_GBps"] = -5.0
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc2)


def test_validator_rejects_duplicate_node_name():
    doc = _valid_doc()
    dup = copy.deepcopy(doc["devices"][0]["components"][0])
    doc["devices"][0]["components"].append(dup)
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)


def test_validator_rejects_missing_required_field():
    doc = _valid_doc()
    del doc["edges"]
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)


def test_validator_accepts_valid_document():
    validate_infragraph(_valid_doc())


def test_validator_rejects_unknown_link_type():
    doc = _valid_doc()
    doc["edges"][0]["link_type"] = "nvlink"
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)


def test_validator_rejects_duplicate_edge():
    doc = _valid_doc()
    doc["edges"].append(copy.deepcopy(doc["edges"][0]))
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)


def test_isolated_component():
    """A3: reject. Fabric has no standalone inventory for switches -- they
    exist only as link endpoints -- so preserving an isolated one would mean
    restructuring Fabric. Rejecting is cheap and this is very likely a
    document error regardless (Task 01 found the switch-side instance of
    exactly this gap)."""
    doc = _valid_doc()
    doc["devices"].append({
        "instance": "machine", "index": 999,
        "components": [{"component": "gpu", "index": 0, "attrs": {}}],
    })
    with pytest.raises(InfraGraphError):
        validate_infragraph(doc)


def test_parser_raises_infragraph_error_not_value_error():
    """Bypasses validate_infragraph() on purpose (see _reconstruct_fabric's
    docstring) to exercise the parser's own defence-in-depth guards."""
    bad_link_type = _valid_doc()
    bad_link_type["edges"][0]["link_type"] = "nvlink"
    with pytest.raises(InfraGraphError):
        _reconstruct_fabric(bad_link_type)

    bad_component = _valid_doc()
    bad_component["devices"][0]["components"][0]["component"] = "widget"
    with pytest.raises(InfraGraphError):
        _reconstruct_fabric(bad_component)


# --------------------------------------------------- validator/parser parity

def _mut_dangling_edge(d):
    d["edges"].append({
        "src": "machine.0.gpu.0", "dst": "machine.99.gpu.0",
        "link_type": "scale_up", "attrs": {"bandwidth_GBps": 1.0, "latency_ns": 1.0},
    })


def _mut_unknown_link_type(d):
    d["edges"][0]["link_type"] = "nvlink"


def _mut_unrecognised_component(d):
    d["devices"][0]["components"][0]["component"] = "widget"


def _mut_missing_edges_field(d):
    del d["edges"]


def _mut_missing_devices_field(d):
    del d["devices"]


def _mut_edge_missing_attrs(d):
    del d["edges"][0]["attrs"]


# Each of these is confirmed (by the assertion inside the test below) to make
# the raw, unvalidated parser raise. Documents that instead produce a silent
# wrong Fabric -- duplicate node names, duplicate edges, domain disagreement,
# non-positive bandwidth, an unknown schema_version -- are deliberately left
# out: the parser doesn't crash on those at all, which is exactly why the
# validator has to catch them instead. See the task 02 report.
PARSER_BREAKING_MUTATIONS = [
    _mut_dangling_edge,
    _mut_unknown_link_type,
    _mut_unrecognised_component,
    _mut_missing_edges_field,
    _mut_missing_devices_field,
    _mut_edge_missing_attrs,
]


def test_validator_catches_everything_the_parser_would():
    """The binding test. Uses _reconstruct_fabric(), not from_infragraph():
    from_infragraph() calls validate_infragraph() first on every input, so
    testing through it would make this pass by construction regardless of
    whether the validator rules actually cover the parser's failure modes.
    Going around that guard is what makes this a real regression check --
    add a new way for the parser to blow up without a matching validator
    rule, and this test catches it."""
    for mutate in PARSER_BREAKING_MUTATIONS:
        doc = _valid_doc()
        mutate(doc)

        parser_raised = False
        try:
            _reconstruct_fabric(doc)
        except Exception:
            parser_raised = True
        assert parser_raised, (
            f"{mutate.__name__} was expected to break the raw parser; it "
            f"didn't, so it belongs in the 'silent wrong Fabric' category "
            f"instead, not in this list")

        with pytest.raises(InfraGraphError, match=".*"):
            validate_infragraph(doc)


# -------------------------------------------------------------- cost parity

def test_parsed_fabric_costs_identically():
    fab = build_node_scale(num_machines=2, gpus_per_machine=8, nics_per_machine=4)
    rt = from_infragraph(to_infragraph(fab))

    transfers = [
        Transfer("t0", GpuId(0, 0), GpuId(0, 1), 400_000),
        Transfer("t1", GpuId(0, 0), GpuId(1, 0), 400_000),
        Transfer("t2", GpuId(0, 4), GpuId(1, 1), 400_000, submit_ns=100),
    ]
    rep_orig = analyse(fab, transfers)
    rep_rt = analyse(rt, transfers)

    assert rep_rt.makespan_ns == rep_orig.makespan_ns
    assert rep_rt.per_transfer_ns == rep_orig.per_transfer_ns
