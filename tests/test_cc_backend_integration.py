"""Task 06: register this engine as a Frontier cc_backend.

The dominant risk here is not a raise -- it's a plausible wrong number (see
docs/tasks/06-frontier-cc-backend.md S1). So most of what follows is an
equivalence check against Frontier's own `analytical` backend, not a feature
test.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from engine.logical.deployment import (Deployment, ParallelKind, PoolKind,
                                       Rank, Replica)

from integration.cc_backend.comm_groups import (CommGroupError,
                                                 CommGroupRegistry,
                                                 populate_from_deployment)

# ---------------------------------------------------------------- registry


def test_registry_resolves_a_registered_triple():
    reg = CommGroupRegistry()
    ranks = [Rank("DECODE_ATTN", 0, 0), Rank("DECODE_ATTN", 0, 1)]
    reg.register("cluster-a", "TP", 2, ranks)
    assert reg.resolve("cluster-a", "TP", 2) == ranks


def test_registry_register_rejects_mismatched_count():
    reg = CommGroupRegistry()
    with pytest.raises(ValueError):
        reg.register("cluster-a", "TP", 4, [Rank("DECODE_ATTN", 0, 0)])


def test_registry_register_is_idempotent_for_the_same_ranks():
    reg = CommGroupRegistry()
    ranks = [Rank("DECODE_ATTN", 0, 0), Rank("DECODE_ATTN", 0, 1)]
    reg.register("cluster-a", "TP", 2, ranks)
    reg.register("cluster-a", "TP", 2, ranks)  # must not raise
    assert reg.resolve("cluster-a", "TP", 2) == ranks


def test_registry_register_raises_on_conflicting_ranks_for_same_triple():
    reg = CommGroupRegistry()
    reg.register("cluster-a", "TP", 2,
                 [Rank("DECODE_ATTN", 0, 0), Rank("DECODE_ATTN", 0, 1)])
    with pytest.raises(CommGroupError):
        reg.register("cluster-a", "TP", 2,
                     [Rank("DECODE_ATTN", 1, 0), Rank("DECODE_ATTN", 1, 1)])


def test_populate_from_deployment_registers_the_tp_group():
    d = Deployment("t")
    d.add(Replica(PoolKind.DECODE_ATTN, 0, tp=2))
    reg = CommGroupRegistry()
    populate_from_deployment(reg, d, {PoolKind.DECODE_ATTN: "decode-attn-cluster"})

    group = d.replicas[0].groups(ParallelKind.TP)[0]
    assert reg.resolve("decode-attn-cluster", "TP", 2) == group.ranks
