"""Result-object conventions: dataclass results that render as dict, JSON or text.

Every public magform function that returns more than one number returns a
frozen dataclass that mixes in :class:`ResultMixin`.  Assessments derive from
:class:`Assessment`, which adds ``status``, ``margin``, ``warnings`` and
``notes``.

Conventions enforced by :meth:`ResultMixin.as_dict`:

* nested dataclasses become dicts, tuples/lists become lists;
* enums become their ``.value``; ``Material`` objects become their ``key``;
* non-finite floats (``inf``/``nan``) become ``None`` so the JSON is valid;
* callables are omitted.

Units are attached per field with ``field(metadata={"unit": "Pa"})`` (see
:func:`unit`).  :meth:`ResultMixin.text` prints ``name [unit] = value``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from enum import Enum
from typing import Any

from .status import Status


def unit(u: str, **kwargs: Any) -> Any:
    """``dataclasses.field`` with a unit in its metadata, e.g. ``x: float = unit('Pa')``."""
    md = dict(kwargs.pop("metadata", {}) or {})
    md["unit"] = u
    return dataclasses.field(metadata=md, **kwargs)


def to_jsonable(value: Any) -> Any:
    """Convert a value (possibly nested) into JSON-serialisable primitives."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Material (and anything else exposing a registry key) collapses to its key
        if type(value).__name__ == "Material" and hasattr(value, "key"):
            return value.key
        out: dict[str, Any] = {}
        for f in dataclasses.fields(value):
            v = getattr(value, f.name)
            if callable(v) and not dataclasses.is_dataclass(v):
                continue
            out[f.name] = to_jsonable(v)
        return out
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [to_jsonable(v) for v in seq]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if callable(value):
        return None
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return "nan"
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            if len(value) > 8:
                return f"[{len(value)} values]"
            return "[" + ", ".join(_fmt(v) for v in value) + "]"
        return "; ".join(_fmt(v) for v in value) if value else "none"
    if dataclasses.is_dataclass(value) and type(value).__name__ == "Material":
        return str(getattr(value, "key", value))
    return str(value)


class ResultMixin:
    """Mixin giving dataclass results ``as_dict()``, ``json()`` and ``text()``."""

    def as_dict(self) -> dict[str, Any]:
        """Recursive JSON-safe dict (see module docstring for the rules)."""
        return to_jsonable(self)

    def json(self, indent: int | None = 2) -> str:
        """Strict JSON (``allow_nan=False``; non-finite values are already ``None``)."""
        return json.dumps(self.as_dict(), indent=indent, allow_nan=False)

    def text(self, indent: int = 0) -> str:
        """Plain-text rendering, one ``name [unit] = value`` line per field."""
        pad = " " * indent
        lines: list[str] = []
        title = getattr(self, "_title", None) or type(self).__name__
        if indent == 0:
            lines.append(title)
            lines.append("-" * len(title))
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            if f.metadata.get("hidden"):
                continue
            v = getattr(self, f.name)
            if callable(v) and not dataclasses.is_dataclass(v):
                continue
            u = f.metadata.get("unit")
            label = f"{f.name} [{u}]" if u else f.name
            if isinstance(v, ResultMixin):
                lines.append(f"{pad}{label}:")
                lines.append(v.text(indent + 2))
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, ResultMixin) for x in v):
                lines.append(f"{pad}{label}:")
                for i, item in enumerate(v):
                    lines.append(f"{pad}  [{i}]")
                    lines.append(item.text(indent + 4))
            elif isinstance(v, dict) and v and all(isinstance(x, ResultMixin) for x in v.values()):
                lines.append(f"{pad}{label}:")
                for k, item in v.items():
                    lines.append(f"{pad}  {k}:")
                    lines.append(item.text(indent + 4))
            elif isinstance(v, dict):
                lines.append(f"{pad}{label}:")
                for k, item in v.items():
                    lines.append(f"{pad}  {k} = {_fmt(item)}")
            elif f.name in ("warnings", "notes") and isinstance(v, (list, tuple)):
                if v:
                    lines.append(f"{pad}{label}:")
                    for item in v:
                        lines.append(f"{pad}  - {item}")
            else:
                lines.append(f"{pad}{label} = {_fmt(v)}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Assessment(ResultMixin):
    """Base for every ``assess_*`` result.

    ``margin`` is (available / required) or (allowable / actual) so that
    ``margin >= 1`` means the requirement is met; ``None`` when undefined.
    """

    status: Status
    margin: float | None = None
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


__all__ = ["ResultMixin", "Assessment", "unit", "to_jsonable"]
