"""The engine state Frontier's CLI-driven config system cannot carry into a
predictor it constructs itself.

Frontier builds `EngineKVCacheTransferConfig`/`EngineM2NTransferConfig` from
CLI flags alone (`KVCacheTransferPredictorRegistry.get(predictor_type,
config=...)` and its M2N sibling take a config object, nothing else), so
there is no channel for a `Fabric`, `Placement`, or `Deployment` Python
object to travel alongside it. Every engine-backed predictor that Frontier
constructs itself -- KV cache transfer (task 09), M2N transfer (task 11) --
reads the same one instance of `EngineContext` instead, set once by
`install()` before a run starts.

Task 09's report weighed this against an InfraGraph-file-path CLI flag and
against non-CLI dataclass fields on the config itself, and rejected both:
this project is already inside one Python process that built these objects
moments before calling into Frontier, so a module-level context does the
job with far less surface than either alternative. Task 11 carries that
decision forward rather than inventing a second mechanism -- one context,
shared by every predictor that needs it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.logical.deployment import Deployment
from engine.physical.topology import Fabric
from engine.placement.binding import BindingPolicy, BindingState
from engine.placement.placement import Placement

from .cc_backend.comm_groups import CommGroupRegistry


@dataclass
class BindingConfig:
    """How to resolve a transfer to one specific replica when its target
    pool has more than one -- task 14. Absent (the `EngineContext.binding`
    default, `None`) means "no policy configured", which keeps every
    predictor's existing behaviour: raise on ambiguity rather than guess.

    `timing` is the other axis task 14 asks to be modelled explicitly:

    - "early" -- decide the destination replica now, with the binding
      policy's current state, and price the real fabric path to it. The
      decision may be stale by the time the transfer would actually land.
    - "late" -- as Frontier itself does (task 14 spec S1: KV decides at
      arrival, in `on_kv_cache_arrival`; M2N's target appears on a
      bookkeeping object built after the predictor returns). No destination
      exists yet to price a real path against, so the price is the mean
      fabric cost over every candidate -- see the task 14 report S2 for why
      that was chosen over pricing a single guessed destination.
    """
    policy: BindingPolicy
    timing: str
    state: BindingState = field(default_factory=BindingState)

    def __post_init__(self) -> None:
        if self.timing not in ("early", "late"):
            raise ValueError(f"timing must be 'early' or 'late', got {self.timing!r}")


@dataclass(frozen=True)
class EngineContext:
    fabric: Fabric
    placement: Placement
    deployment: Deployment
    groups: CommGroupRegistry
    binding: Optional[BindingConfig] = None


_context: Optional[EngineContext] = None


def set_context(context: EngineContext) -> None:
    global _context
    _context = context


def require_context() -> EngineContext:
    if _context is None:
        raise RuntimeError(
            "An engine-backed Frontier predictor was called before install() "
            "set the engine context (fabric/placement/deployment/groups). "
            "Call integration.install.install(...) before running Frontier.")
    return _context
