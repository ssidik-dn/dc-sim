"""Register EngineCCBackend with Frontier's CCBackendFactory.

This is the only module in the project that touches
`frontier.cc_backend.cc_backend_factory.CCBackendFactory` -- see AGENTS.md's
import-direction rule and task 06 spec S4.3.

FINDING (task 06 spec S7.1 -- see the task report for the full writeup):
`CCBackendFactory.register()` is typed to take a `frontier.types.CCBackendType`
member as its key. `CCBackendType` is a plain `IntEnum` with five fixed
members (VIDUR, ANALYTICAL, COLLECTIVE_SIM, AICONFIGURATOR,
ASTRA_SIM_ANALYTICAL); four are already registered to concrete backends and
the fifth (AICONFIGURATOR) is unconditionally rejected by
`CCBackendFactory.create()` regardless of registry state. There is no free
slot, and a plain `IntEnum` cannot grow a sixth member without editing
`frontier/types/cc_backend_type.py`, which is under `upstream/` and may not be
touched.

`register()` does not check its key's type at runtime (Python type hints are
not enforced), so registering under the literal string "dc_sim_engine" below
succeeds mechanically: `CCBackendFactory.get_class("dc_sim_engine")` and
`CCBackendFactory.get("dc_sim_engine", ...)` both round-trip correctly. But
the path Frontier's own config layer uses to reach the factory --
`create_from_str()`, through `get_key_from_str()`'s `CCBackendType[s.upper()]`
lookup, and beneath that `config.py`'s hardcoded elif chain over
{"analytical", "vidur", "collective_sim", "aiconfigurator",
"astra_sim_analytical"} -- is closed over the same five names. So
`--*_cc_backend_config_type dc_sim_engine` cannot select this backend without
editing two files under `upstream/`. That is out of scope here: this
function registers what the factory's public API actually supports, and
stops there.
"""
from __future__ import annotations

from frontier.cc_backend.cc_backend_factory import CCBackendFactory

from ..cc_backend.engine_backend import EngineCCBackend

BACKEND_NAME = "dc_sim_engine"


def install() -> None:
    """Register EngineCCBackend under BACKEND_NAME.

    Idempotent because `CCBackendFactory.register()` is: it no-ops if the key
    is already present, so calling `install()` twice does not double-register
    or raise.
    """
    CCBackendFactory.register(BACKEND_NAME, EngineCCBackend)
