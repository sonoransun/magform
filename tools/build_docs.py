#!/usr/bin/env python3
"""Generate every number, table and chart on the documentation site from the magform package.

    python tools/build_docs.py            # rewrite docs/_data/*.json and docs/_includes/generated/*
    python tools/build_docs.py --check    # exit 1 if the committed files are stale (used by the tests)

Results quoted on the site are not typed by hand: prose pulls them from ``docs/_data/n.json``
(``{{ site.data.n.<key> }}``), charts are JSON specs in ``docs/_data/figures.json`` drawn by
``docs/assets/js/charts.js``, and the ledger, tables, command output and reference lists are
rendered by Liquid from the other data files.  (Inputs chosen for an example, such as a 2 mm
wall, and quotations from the concept note are written out in the pages.)  Output is deterministic (no timestamps; floats rounded) so it can be
committed and diffed.  Standard library only, like the package.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:  # the chapter modules live next to this file
    sys.path.insert(0, str(TOOLS))

from docs_common import DATA_DIR, INCLUDE_DIR, ROOT, Site  # noqa: E402

import magform  # noqa: E402


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def chapters() -> list[Callable[[Site], None]]:
    """Every chapter function, in the order they are run."""
    import docs_story

    import docs_guide
    import docs_interactive

    found = list(docs_story.CHAPTERS)
    found.append(docs_interactive.chapter_home)
    found.append(docs_guide.chapter_guide)
    try:
        import docs_reference
    except ImportError:
        docs_reference = None
    if docs_reference is not None:
        found.append(docs_reference.chapter_reference)
    return found


def build() -> Site:
    site = Site()
    for chapter in chapters():
        chapter(site)
    site.data["meta"] = {"version": magform.__version__, "figures": len(site.figures), "numbers": len(site.n),
                         "modules": len(magform._SUBMODULES)}
    return site


_BARE_EXPONENT = re.compile(r'"(?:\\.|[^"\\])*"|(?<![\d.])(-?\d+)([eE][-+]?\d+)')


def yaml_safe_json(text: str) -> str:
    """Rewrite JSON numbers such as ``6e+20`` as ``6.0e+20``; strings are left alone.

    Jekyll loads ``_data/*.json`` with a YAML 1.1 parser, and YAML 1.1 only recognises a float
    that has a decimal point: ``6e+20`` and ``1e-07`` (which is how Python writes them) arrive
    in Liquid as *strings*, and a chart then silently drops the value.  Both spellings are the
    same JSON number, so nothing else changes.
    """
    return _BARE_EXPONENT.sub(lambda m: m.group(0) if m.group(1) is None else f"{m.group(1)}.0{m.group(2)}", text)


def _dumps(value: Any, **kwargs: Any) -> str:
    return yaml_safe_json(json.dumps(value, ensure_ascii=False, allow_nan=False, **kwargs))


def _dump_compact_map(mapping: dict[str, Any]) -> str:
    """One top-level key per line: readable diffs without one number per line."""
    body = ",\n".join(f" {json.dumps(k)}: {_dumps(v, separators=(',', ':'))}" for k, v in mapping.items())
    return "{\n" + body + "\n}\n"


def render(site: Site) -> dict[Path, str]:
    files: dict[Path, str] = {
        DATA_DIR / "n.json": _dumps(dict(sorted(site.n.items())), indent=1) + "\n",
        DATA_DIR / "figures.json": _dump_compact_map(site.figures),
    }
    for name, value in site.data.items():
        files[DATA_DIR / f"{name}.json"] = _dumps(value, indent=1) + "\n"
    for name, text in site.includes.items():
        files[INCLUDE_DIR / name] = text
    return files


def _same(a: Any, b: Any, rel: float = 1e-6) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or abs(a - b) <= rel * max(abs(a), abs(b))
    return a == b


def stale_files(files: dict[Path, str]) -> list[Path]:
    """Committed files that differ from a fresh build (numbers compared to 1e-6 relative)."""
    out = []
    for path, text in files.items():
        if not path.exists():
            out.append(path)
            continue
        old = path.read_text(encoding="utf-8")
        if old == text:
            continue
        if path.suffix == ".json":
            try:
                if _same(json.loads(old), json.loads(text)):
                    continue
            except ValueError:
                pass
        out.append(path)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if the committed files are stale")
    args = parser.parse_args(argv)
    files = render(build())
    if args.check:
        stale = stale_files(files)
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}", file=sys.stderr)
        if stale:
            print("run: python tools/build_docs.py", file=sys.stderr)
        return 1 if stale else 0
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} files to {DATA_DIR.relative_to(ROOT)} and {INCLUDE_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
