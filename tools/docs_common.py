"""Shared helpers for the documentation generator: formatting, the Site container, chart specs.

Imported by ``tools/build_docs.py`` and the chapter modules next to it (``docs_story.py``,
``docs_reference.py``).  Standard library only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run from a checkout without installing
    sys.path.insert(0, str(ROOT))

from magform import benchmarks  # noqa: E402
from magform.magnetics import Solenoid, field_for_pressure  # noqa: E402

DATA_DIR = ROOT / "docs" / "_data"
INCLUDE_DIR = ROOT / "docs" / "_includes" / "generated"

REF = Solenoid(0.05, 0.25, 0.5)
"""The reference press used throughout the site: 100 mm bore, 200 mm build, 0.5 m long."""

P_LO, P_HI = 1.6e9, 4.0e9
"""The concept note's pressure window."""


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def sig(x: Any, n: int = 4) -> Any:
    """Round a number to ``n`` significant digits for JSON (None for non-finite)."""
    if x is None or isinstance(x, (bool, str)):
        return x
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    return float(f"{x:.{n}g}")


def sigs(values: Iterable[float], n: int = 4) -> list[Any]:
    return [sig(v, n) for v in values]


def fx(x: float, d: int = 2) -> str:
    """Fixed decimals with thousands separators: fx(100.268, 2) -> '100.27'."""
    return f"{x:,.{d}f}"


def sg(x: float, n: int = 3) -> str:
    """``n`` significant digits as prose: plain decimals between 1e-3 and 1e6, else a x 10^b (HTML)."""
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return "infinite" if x > 0 else "minus infinity"
    x = float(f"{x:.{n}g}")  # round first, so 9.9e7 at one digit is 1e8 and 235052 at three is 235000
    a = abs(x)
    if 1e-3 <= a < 1e6:
        d = max(0, n - 1 - math.floor(math.log10(a)))
        return f"{x:,.{d}f}"
    e = math.floor(math.log10(a))
    m = x / 10 ** e
    ms = f"{m:.{max(0, n - 1)}f}"
    if "." in ms:
        ms = ms.rstrip("0").rstrip(".")
    return f"{ms}&nbsp;&times;&nbsp;10<sup>{e}</sup>" if ms != "1" else f"10<sup>{e}</sup>"


def pct(x: float, d: int = 0) -> str:
    return f"{100.0 * x:.{d}f}"


def gpa(x: float, d: int = 2) -> str:
    return fx(x / 1e9, d)


def mpa(x: float, d: int = 0) -> str:
    return fx(x / 1e6, d)


def label(status: Any) -> str:
    return str(getattr(status, "value", status))


# ---------------------------------------------------------------------------
# site container and chart-spec helpers
# ---------------------------------------------------------------------------
class Site:
    """Everything the generator writes: named numbers, chart specs and other data files."""

    def __init__(self) -> None:
        self.n: dict[str, str] = {}
        self.figures: dict[str, dict] = {}
        self.data: dict[str, Any] = {}
        self.includes: dict[str, str] = {}

    def num(self, key: str, value: str) -> None:
        if key in self.n and self.n[key] != value:
            raise ValueError(f"number {key!r} defined twice with different values")
        self.n[key] = value

    def nums(self, **values: str) -> None:
        for key, value in values.items():
            self.num(key, value)

    def figure(self, key: str, spec: dict) -> None:
        if key in self.figures:
            raise ValueError(f"figure {key!r} defined twice")
        self.figures[key] = spec


def line(name: str, xs: Sequence[float], ys: Sequence[float], slot: int = 1, **extra: Any) -> dict:
    return {"name": name, "kind": "line", "slot": slot, "x": sigs(xs), "y": sigs(ys), **extra}


def area(name: str, xs: Sequence[float], ys: Sequence[float], y0: Sequence[float] | None = None,
         slot: int = 1, **extra: Any) -> dict:
    spec = {"name": name, "kind": "area", "slot": slot, "x": sigs(xs), "y": sigs(ys), **extra}
    if y0 is not None:
        spec["y0"] = sigs(y0)
    return spec


def points(name: str, xs: Sequence[float], ys: Sequence[float], slot: int = 1, **extra: Any) -> dict:
    return {"name": name, "kind": "points", "slot": slot, "x": sigs(xs), "y": sigs(ys), **extra}


def axis(label_: str, scale: str = "linear", lo: float | None = None, hi: float | None = None,
         unit: str = "", short: str | None = None, ticks: Sequence[Any] | None = None,
         time: bool = False) -> dict:
    """An axis spec.  ``time=True`` makes ticks and tooltips read as durations (ns to days);
    ``ticks`` may be numbers or ``{"v": value, "label": text}``."""
    spec: dict[str, Any] = {"label": label_, "scale": scale}
    if time:
        spec["format"] = "time"
    if lo is not None:
        spec["min"] = lo
    if hi is not None:
        spec["max"] = hi
    if unit:
        spec["unit"] = unit
    if short:
        spec["short"] = short
    if ticks is not None:
        spec["ticks"] = list(ticks)
    return spec


def table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> dict:
    return {"columns": list(columns), "rows": [[sig(c, 4) if isinstance(c, float) else c for c in r] for r in rows]}


CONCEPT_BAND_T = {"axis": "x", "from": sig(field_for_pressure(P_LO)), "to": sig(field_for_pressure(P_HI)),
                  "label": "concept note"}
CONCEPT_BAND_GPA = {"axis": "x", "from": 1.6, "to": 4.0, "label": "concept note"}




def bench(ladder: str, key: str) -> benchmarks.Benchmark:
    """One cited benchmark by ladder name and key."""
    for item in benchmarks.ALL_LADDERS[ladder]:
        if item.key == key:
            return item
    raise KeyError(key)


def plain(text: str) -> str:
    """Strip the HTML entities ``sg`` uses, for chart labels (charts render plain text)."""
    return (text.replace("&nbsp;", " ").replace("&times;", "\u00d7").replace("<sup>", "^").replace("</sup>", ""))
