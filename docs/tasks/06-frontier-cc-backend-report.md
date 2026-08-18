# Task 06 report — Register the engine as a Frontier cc_backend

Branch: `task-06-frontier-cc-backend` (not merged to main).

All 134 tests pass (121 existing + 13 new in `tests/test_cc_backend_integration.py`),
and `python3 tools/check_import_direction.py` exits 0.

---

## 1. Discrepancies between spec S3 and the actual Frontier source

- **`predict_send_recv` has no `num_devices`.** Its real signature is
  `predict_send_recv(self, data_size_bytes, cluster_type=None, comm_domain=None)`.
  The spec's "uniform shape across all six" claim is wrong; point-to-point is
  always exactly 2 ranks, so `EngineCCBackend.predict_send_recv` resolves the
  registry with a fixed arity of 2 rather than one Frontier never supplies.
- **No `upstream/frontier` exists**, in this repo or as a convention. Frontier
  is reached purely through the ambient `PYTHONPATH` (`/work/Frontier`),
  mirroring how ASTRA-sim is reached via `$ASTRA` (see
  `tools/validate_astra.py`) — not a repo-pinned checkout. `import frontier`
  fails with `PYTHONPATH` unset, and nothing in dc-sim's own `pytest.ini` or
  dependency files declares this.
- **No existing `install()` entry point.** `src/integration/__init__.py` was
  empty before this task. Spec S4.3's "callable from the existing `install()`
  entry point" describes something that did not exist yet; this task creates
  the first one, at `src/integration/install/cc_backend.py`.
- **`--cc_backend_config_type` is not one global flag.** It's four per-pool
  fields (`prefill_cc_backend_config_type`, `decode_cc_backend_config_type`,
  `decode_attn_cc_backend_config_type`, `decode_ffn_cc_backend_config_type`).
- **`aiconfigurator_cc_backend.py`, cited as "the closest existing model," is
  fully disabled.** Its `__init__` raises `ValueError` unconditionally before
  ever calling `super().__init__()`, and the module isn't even imported by
  `frontier/cc_backend/backends/__init__.py` — so it isn't in the running
  factory's registry at all in this checkout.

## 2. Registration finding (the central one) — spec S4.3

`CCBackendFactory.register()` is keyed by `frontier.types.CCBackendType`, a
closed 5-member `IntEnum` (`VIDUR`, `ANALYTICAL`, `COLLECTIVE_SIM`,
`AICONFIGURATOR`, `ASTRA_SIM_ANALYTICAL`). Four members are already
registered to concrete backends; the fifth (`AICONFIGURATOR`) is
unconditionally rejected by `CCBackendFactory.create()` regardless of
registry state (`AICONFIGURATOR_BACKEND_RELEASE_ERROR`). **There is no free
slot**, and adding one means editing `frontier/types/cc_backend_type.py` —
forbidden, since that file is under `upstream/`.

`register()` does not type-check its key at runtime, so
`src/integration/install/cc_backend.py`'s `install()` registers
`EngineCCBackend` under the literal string `"dc_sim_engine"`. This works
mechanically and round-trips through `get_class()`/`get()` (verified, and
covered by `test_registration_is_idempotent`). But Frontier's real CLI-flag
path is closed over the same 5 names at **two** layers —
`CCBackendType[s.upper()]` inside `get_key_from_str()`, and a second
hardcoded `elif` chain in `frontier/config/config.py`'s per-cluster config
builder — so `--*_cc_backend_config_type dc_sim_engine` cannot reach this
backend without two separate upstream edits.

This is implemented to do what the factory's public API actually supports,
and documents the ceiling rather than forcing past it — the "stop and
report" case spec S4.3 anticipated.

## 3. Tolerance chosen for the analytical-agreement test, and why

Chose **1e-6 relative**, on a 2-device packed all-reduce
(`test_packed_placement_matches_analytical_within_bound`). At
`num_devices=2`, Frontier's own ring all-reduce volume factor `2*(n-1)/n`
collapses to exactly 1, so both backends reduce to the same
`latency + size/bandwidth` formula when parameterized from the same physical
scale-up link (400 GB/s, 936.25 ns).

**Actual measured disagreement: 0.0** (exact, verified numerically before
the test was written). For `num_devices > 2` the two backends are **not**
expected to agree — this engine's default `MockBackend` cost path does not
model per-device ring-volume scaling the way Frontier's closed-form model
does; reproducing that would need a real ASTRA-sim-backed `CostBackend`,
which the existing test suite deliberately avoids (all 121 pre-existing
tests run in well under a second, no external binary). The bound was not
widened to paper over this gap — it is real, and it's the difference between
"the engine has fabric topology" and "the engine also independently
re-derives Frontier's collective-algorithm arithmetic."

## 4. Whether anything tempted a change to `src/engine/`

Nothing required one. The one thing checked carefully: whether
`Placement.group_shape()` needed a variant that skips constructing a full
`ParallelGroup`. It didn't — `group_shape()` only reads `.ranks`, so
`ParallelGroup(kind=ParallelKind.TP, ranks=ranks)` is a harmless reuse (the
`kind` field is inert for that call).

## 5. Whether a full Frontier run was attempted

Not attempted — out of scope per spec S4's "do not attempt" list, and per
finding 2, not currently possible via the documented CLI path without an
upstream change.

## 6. Where the spec was wrong

Covered in S1–S2 above: the `predict_send_recv` signature, the nonexistent
`upstream/frontier`, the nonexistent prior `install()` entry point, the
single-flag claim, and — most importantly — the assumption that
`CCBackendFactory` supports registering an arbitrary new backend name at
all.

## What shipped

- `src/integration/cc_backend/comm_groups.py` — `CommGroupRegistry` and
  `populate_from_deployment`.
- `src/integration/cc_backend/engine_backend.py` — `EngineCCBackend`, a
  genuine `BaseCCBackend` subclass answering all six prediction calls from
  this project's `Fabric`/`Placement`/`CostBackend`.
- `src/integration/install/cc_backend.py` — `install()`, the sole call site
  touching `CCBackendFactory`.
- `tests/test_cc_backend_integration.py` — 13 tests, covering the 7 required
  by spec S5 plus supporting registry/unit-conversion tests.

Four commits on `task-06-frontier-cc-backend`, none touching `src/engine/`.
