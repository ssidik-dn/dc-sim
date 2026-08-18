# Task 07 report — Is the KV transfer path actually selectable?

Branch: `task-07-kv-selection-check` (not merged to main).

`python3 -m pytest -q` (134 passed) and `python3 tools/check_import_direction.py`
pass unchanged. The diff is one file, `tools/probe_kv_selection.py`, plus this
report; nothing under `upstream/` or `src/engine/` was touched.

---

## 1. The answer

**Open.** A `BaseKVCacheTransferPredictor` subclass registered from outside
`upstream/`, under the previously-unused `KVCacheTransferType.EMPIRICAL`, is
selected and used by a real Frontier `pd-disaggregation` run purely through
`--kv_cache_transfer_config_type empirical` — no upstream edit, no bypass.
This is the opposite finding from task 06's collective backend.

## 2. The evidence

`tools/probe_kv_selection.py` runs a full offline `pd-disaggregation`
simulation (2 requests, dummy execution-time mode, device `h800`) in-process,
with a sentinel `EmpiricalKVCacheTransferConfig` / `SentinelKVCacheTransferPredictor`
pair registered under `KVCacheTransferType.EMPIRICAL` before Frontier's CLI
parsing runs. Output:

```
SENTINEL_CALLED get_transfer_time source=prefill target=decode kv_cache_size_bytes=1 -> 424242.0
SENTINEL_CALLED get_transfer_time source=prefill target=decode kv_cache_size_bytes=1 -> 424242.0
sentinel calls: 2
sentinel last kv_cache_size_bytes: 1
ANSWER: OPEN -- the sentinel predictor's get_transfer_time was called by a real Frontier run selected purely via --kv_cache_transfer_config_type empirical.
```

The run completed end to end (`Sequential simulation ended at: 427.464s`,
2 requests fully rolled out), with the sentinel's distinctive
`424242.0` transfer time actually driving the KV-transfer step for both
requests — not just registered, but on the hot path of a real run.

**A real bug surfaced en route, worth recording because it is the kind of
false negative this task exists to guard against.** The first run of the
probe failed with:

```
AssertionError: Invalid type empirical for kv_cache_transfer_config_type. Valid types: ['analytical']
  File "/work/Frontier/frontier/config/flat_dataclass.py", line 137, in reconstruct_original_dataclass
```

This looked exactly like a closed gate. It wasn't: `get_all_subclasses()`
walks `type.__subclasses__()`, which holds only *weak* references. My
sentinel config class was local to a builder function and returned nowhere,
so once that function returned, nothing kept it alive — it was
garbage-collected between CLI argument parsing (which accepted `empirical`
as a plain string, unchecked) and reconstruction (which re-walks
`__subclasses__()` and no longer found it). Fixed by holding a
module-level strong reference. Recorded in the probe's own comments. This is
the same shape of error the task's own preamble warns about — a plausible,
textually correct-looking failure that is not the thing it appears to be —
just on the tooling side instead of Frontier's.

## 3. Where the (non-)gate is

Traced §3.1 through §3.5; none of them closed:

- **§3.1 enum** — `KVCacheTransferType` has 3 members; only `ANALYTICAL` is
  implemented (`AnalyticalKVCacheTransferPredictor`/`AnalyticalKVCacheTransferConfig`).
  `EMPIRICAL` and `HYBRID` have zero implementations anywhere in the tree —
  confirmed by grep, not just absence of an obvious file.
- **§3.2 registry** — `KVCacheTransferPredictorRegistry` overrides `get()`
  directly (there is no separate `create()`), and the override is just
  `if predictor_type not in cls._registry: raise ValueError(...)`. No
  `AICONFIGURATOR`-style special-cased rejection of any member.
- **§3.3 string→enum** — `get_key_from_str` is a direct
  `KVCacheTransferType.from_str(key_str)`, i.e. `cls[s.upper()]`. This is
  the same *mechanism* that closed the collective path in task 06 — but here
  it's irrelevant to the actual selection path (see §3.4): the CLI flag
  never goes through `KVCacheTransferPredictorRegistry.get_key_from_str` at
  all.
- **§3.4 the layer that mattered in task 06 — open here, and structurally
  different.** `kv_cache_transfer_config` is not a `_config_type: Optional[str]`
  field dispatched through a hardcoded `elif` chain in `config.py` (that's
  the collective-backend shape). It's a single `BaseKVCacheTransferConfig`
  field — a `BasePolyConfig` — and Frontier's CLI-flattening machinery
  (`frontier/config/flat_dataclass.py`) handles every `BasePolyConfig` field
  generically: it calls `get_all_subclasses(field_type)` **live, at
  argument-parsing and reconstruction time**, not a fixed list anywhere in
  source. Any subclass that exists in the process — regardless of which
  module defined it — becomes a legal `--kv_cache_transfer_config_type`
  value, with its own CLI-exposed fields (`--<snake_case_class_name>_*`).
  `config.py` contains **zero** hardcoded references to `kv_cache_transfer_config_type`
  or `KVCacheTransferType` beyond the one default-factory import — confirmed
  by grep, not inference.
- **§3.5 call site** — `frontier/simulator.py`, inside `Simulator.__init__`,
  gated on `self._config.is_disaggregated_mode()`:
  ```python
  kv_cache_transfer_predictor = KVCacheTransferPredictorRegistry.get(
      self._config.kv_cache_transfer_config.get_type(),
      config=self._config.kv_cache_transfer_config,
  )
  ```
  This is genuinely registry-driven (`predictor_type` comes from the config
  object's own polymorphic `get_type()`, not a separately-maintained
  string), and the probe confirms `get_transfer_time` is called on exactly
  that object during a real run — not a third, unconsulted registry.

## 4. Whether the M2N path differs

**No — same shape, read-only, not probed end to end (per the task's
constraint).** `frontier/m2n_transfer/` mirrors `kv_cache_transfer/` almost
exactly:

- `M2NTransferType` (`frontier/types/m2n_transfer_type.py`): `ANALYTICAL=1,
  EMPIRICAL=2, HYBRID=3`, only `ANALYTICAL` implemented.
- `M2NTransferPredictorRegistry.get()` has the same bare
  "not registered → raise" guard, no special-cased rejection.
- `frontier/config/m2n_transfer_config.py` defines only
  `AnalyticalM2NTransferConfig`; `m2n_transfer_config` is a single
  `BaseM2NTransferConfig` (`BasePolyConfig`) field on `SimulationConfig`,
  same as KV cache transfer.
- `config.py` has **zero** hardcoded `m2n_transfer_config_type` /
  `M2NTransferType` branches (grep confirms only the default-factory
  import).
- Call site is the sibling of the KV one in `simulator.py`, gated on
  `self._config.sys_arch == "pd-af-disaggregation"` rather than
  `is_disaggregated_mode()`.

Given the identical architecture and identical absence of a hardcoded
gate, M2N should be open the same way KV cache transfer is. This is read
from source only, as instructed — worth a probe of its own before betting
real work on it, but low-risk given how structurally identical it is.

## 5. What I would do next

- **Reorder the integration roadmap.** KV cache transfer (and, provisionally,
  M2N) should move ahead of collectives as the next real integration target
  — task 06 found collectives structurally closed without an upstream edit,
  and this task found KV cache transfer genuinely open.
- **Probe M2N the same way before committing to it**, since "read from
  source" and "confirmed by running it" are not the same claim, and this
  task's whole premise is that the second one is the only one that counts.
  Cheap to do: the probe script here is ~80% reusable (swap the predictor
  base class, config base class, registry, and the `sys_arch` flag).
- **Watch for the weak-reference trap in any future probe or real
  integration code that defines a `BasePolyConfig` subclass dynamically** —
  a config class registered this way must be kept alive by a real reference
  for the life of the process, not just constructed and discarded. This
  applies to `src/integration/` code we eventually write for real, not only
  to throwaway probes.
- Building the real KV predictor is still out of scope for now (task 06's
  finding-first discipline: confirm the seam before building on it) — this
  task only proves the seam exists.

## 6. Where the specification is wrong

- **§3.2 asked whether `create()` has an `AICONFIGURATOR`-style guard.**
  There is no `create()` at all on `KVCacheTransferPredictorRegistry` — it
  overrides `get()` directly. Not a meaningful discrepancy (the question is
  answered: no guard either way), but the method name in the spec doesn't
  exist.
- **Everything else in §3–§4 held up as described** — this is the rare case
  where the spec's suspicion (two free-looking enum slots, no
  `AICONFIGURATOR`-style guard) matched the source exactly. The one real
  surprise was procedural, not architectural: the weak-reference GC trap in
  §4 above, which the spec could not have anticipated since it's about how
  the *probe* holds its own sentinel class alive, not about Frontier.
