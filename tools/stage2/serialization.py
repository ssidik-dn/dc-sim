"""Stage 2 Gate A: JSON round-trip for the four contract objects.

One generic recursive (de)serializer, not one hand-written `to_dict` per
class -- with ~30 nested dataclasses in `contracts.py`, a hand-written
version would drift the moment a field is added and one call site is
missed. This module is the single place that risk lives.

Handles: nested `@dataclass` instances, `Optional[X]`, `List[X]`,
`Tuple[X, ...]`, `Dict[str, X]`, and plain `str`/`int`/`float`/`bool`/`None`.
Tuples serialize as JSON arrays and are restored to tuples on the way back
(so equality against the original dataclass instance holds after a
round-trip -- checked directly by `tests/test_stage2_contracts.py`, not
assumed).
"""
from __future__ import annotations

import dataclasses
import json
import typing
from typing import Any, Dict, Type, TypeVar, get_args, get_origin

T = TypeVar("T")


class SchemaVersionError(ValueError):
    """Raised when a payload's own major version does not match what
    this code was written against -- S19's own "unknown major version
    -> hard reject," never a silent best-effort migration."""


class SchemaFieldError(ValueError):
    """Raised when a required field is missing -- S19's own "missing
    required field -> hard reject.\""""


def _is_optional(tp) -> bool:
    return get_origin(tp) is typing.Union and type(None) in get_args(tp)


def _optional_inner(tp):
    args = [a for a in get_args(tp) if a is not type(None)]
    return args[0]


def to_jsonable(obj: Any) -> Any:
    """Recursively converts a contract object (or any plain value) into
    a JSON-safe structure. Dataclass instances become dicts; tuples and
    lists become lists; everything else passes through unchanged."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def to_json(obj: Any, *, indent: int = 2) -> str:
    return json.dumps(to_jsonable(obj), indent=indent, sort_keys=False)


def _build_value(tp, value):
    if value is None:
        return None
    if _is_optional(tp):
        return _build_value(_optional_inner(tp), value)
    origin = get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    if origin in (list,):
        (inner,) = get_args(tp)
        return [_build_value(inner, v) for v in value]
    if origin in (tuple,):
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
            return tuple(_build_value(inner, v) for v in value)
        return tuple(_build_value(a, v) for a, v in zip(args, value))
    if origin in (dict,):
        key_tp, val_tp = get_args(tp)
        return {_build_value(key_tp, k) if key_tp is not str else k: _build_value(val_tp, v)
               for k, v in value.items()}
    if tp is int and isinstance(value, str) and value.lstrip("-").isdigit():
        # Dict[int, str] keys round-trip through JSON as strings; restore
        # the int form for e.g. PlacementSpec.topology_machine_to_host.
        return int(value)
    return value


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Reconstructs one dataclass instance from a plain dict, using the
    class's own field type hints to know how to rebuild each field
    (nested dataclass, list, tuple, dict, or plain value). Raises
    `SchemaFieldError` for any field the dataclass declares without a
    default that the payload does not supply -- never silently defaults
    a required field (S19)."""
    if not dataclasses.is_dataclass(cls):
        return data
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        tp = hints[f.name]
        if f.name not in data:
            if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                continue
            raise SchemaFieldError(f"{cls.__name__} is missing required field {f.name!r}")
        raw = data[f.name]
        if raw is None:
            kwargs[f.name] = None
            continue
        # dict keys that should be int (e.g. topology_machine_to_host)
        if get_origin(tp) is dict:
            key_tp, val_tp = get_args(tp)
            kwargs[f.name] = {
                (int(k) if key_tp is int else k): _build_value(val_tp, v)
                for k, v in raw.items()
            }
        else:
            kwargs[f.name] = _build_value(tp, raw)
    return cls(**kwargs)


def from_json(cls: Type[T], s: str) -> T:
    return from_dict(cls, json.loads(s))


def check_major_version(kind: str, payload_version: str, expected_version: str) -> None:
    """S19: an unknown *major* version is a hard reject; a matching or
    higher-minor payload within the same major version is accepted
    without migration (no migrations exist yet -- "do not
    over-engineer migrations")."""
    payload_major = payload_version.split(".")[0]
    expected_major = expected_version.split(".")[0]
    if payload_major != expected_major:
        raise SchemaVersionError(
            f"{kind} major version {payload_version!r} is not compatible with "
            f"the major version this code was written against, "
            f"{expected_version!r} -- refusing rather than guessing at a "
            f"migration.")
