"""Guide pages: everything they show is produced by running the package.

* ``site.data["worked"]``: the worked press design, as code snippets and the text they print;
* ``site.data["cli"]``: the command table from the argument parser, and real output of example commands;
* ``site.data["api"]``: every public name of every module with its one-line summary.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import shlex

from docs_common import Site

import magform
from magform import cli

# ---------------------------------------------------------------------------
# worked design: snippets run in one namespace, in order
# ---------------------------------------------------------------------------
WORKED_STEPS: tuple[tuple[str, str], ...] = (
    ("spec", '''from magform.press import PressSpec, design_press

spec = PressSpec(pressure=4e9, bore_diameter=0.10, length=0.50, winding_thickness=0.20)
design = design_press(spec)

print(f"field             {design.field:.2f} T")
print(f"current density   {design.current_density / 1e6:.0f} A/mm2")
print(f"stored energy     {design.stored_energy / 1e6:.1f} MJ")
print(f"peak hoop stress  {design.peak_hoop_stress / 1e9:.2f} GPa")
'''),
    ("workpiece", '''w = design.workpiece
print(f"rise time             {design.rise_time * 1e3:.1f} ms")
print(f"skin depth            {w.skin_depth * 1e3:.1f} mm")
print(f"pressure, mid-face    {w.surface_mid_ratio:.2f} x nominal")
print(f"pressure, corners     {w.surface_peak_ratio:.1f} x nominal")
print(f"skin heating          +{w.skin_temperature_rise:.0f} K, melts: {w.skin_melts}")
'''),
    ("magnet", '''parts = {"conductor": design.conductor, "structure": design.structure,
         "pulsed operation": design.pulsed_operation}
for name, part in parts.items():
    print(f"{name:18s} {part.status.label}")
print(f"{'concept magnet':18s} {design.status.label}")
for line in design.warnings[:3]:
    print(" !", line)
'''),
    ("architectures", '''for a in design.architectures:
    print(f"{a.name:42s} {a.status.label:17s} limited by {a.limiting}")
print("best:", design.best_architecture, "->", design.best_status.label)
'''),
    ("network", '''n = design.network
print(f"{n.turns} turns, {n.charge_voltage / 1e3:.1f} kV, {n.peak_current / 1e6:.2f} MA")
print(f"bank {n.bank_energy / 1e6:.0f} MJ, peak power {n.peak_power / 1e9:.1f} GW")
print(f"coil ends the shot at {n.conductor_final_temperature:.0f} K")
print("status:", n.status.label)
'''),
    ("thin", '''from dataclasses import replace

tube = design_press(replace(spec, workpiece_wall=0.002))
print(f"rise time {tube.rise_time * 1e6:.0f} us")
for a in tube.architectures:
    print(f"{a.name:42s} {a.status.label}")
print("best:", tube.best_architecture, "->", tube.best_status.label)
'''),
    ("report", '''print(design.workpiece.text())
'''),
)


def _run_steps() -> list[dict]:
    namespace: dict = {}
    out = []
    for key, code in WORKED_STEPS:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, f"<worked:{key}>", "exec"), namespace)   # noqa: S102 - our own snippets
        out.append({"key": key, "code": code.rstrip(), "output": buffer.getvalue().rstrip()})
    return out


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
CLI_EXAMPLES: tuple[tuple[str, str, int | None], ...] = (
    # key, command, number of output lines to keep (None = all)
    ("field", "field --pressure 4000", None),
    ("press", "press --pressure 4000", None),
    ("architectures", "architectures --pressure 4000 --wall 0.002", 9),
    ("pulse", "pulse --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2 --rise 0.01", 14),
    ("forge", "forge tungsten --pressure 2000", 12),
    ("weld", "weld al steel --pressure 100", 14),
    ("synth", "synth --pressure 4000", 11),
    ("pores", "pores gamma_tial --pressure 4000 --porosity 0.005", 10),
    ("contain", "contain --pressure 4000 --bore 0.05", None),
    ("magnetar", "magnetar --field 1e10 --radius 0.01", 12),
    ("claims", "claims C15", None),
)


def _run_cli(command: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(shlex.split(command))
    if code != 0:
        raise RuntimeError(f"example command failed ({code}): magform {command}")
    return buffer.getvalue().rstrip()


def _commands() -> list[dict]:
    parser = cli.build_parser()
    rows = []
    for action in parser._subparsers._group_actions:          # noqa: SLF001 - argparse has no public accessor
        for choice in action._choices_actions:                # noqa: SLF001
            sub = action.choices[choice.dest]
            options = []
            for opt in sub._actions:                          # noqa: SLF001
                if opt.dest in ("help", "func"):
                    continue
                flag = opt.option_strings[0] if opt.option_strings else opt.dest
                # no "required" flag here: argparse reports it differently for optional positionals across versions
                options.append({"flag": flag, "help": (opt.help or "").replace("%(default)s", str(opt.default))})
            rows.append({"name": choice.dest, "help": choice.help or "", "options": options})
    return rows


# ---------------------------------------------------------------------------
# API map
# ---------------------------------------------------------------------------
_GROUPS = (
    ("Foundations", ("constants", "errors", "status", "benchmarks", "materials")),
    ("The press", ("magnetics", "fieldmap", "conductor", "pulse", "press")),
    ("The workpiece", ("eddy", "plasticity", "microstructure", "shock")),
    ("The applications", ("welding", "synthesis", "porosity", "containment")),
    ("Beyond the note", ("magnetar",)),
    ("Ledger and command line", ("claims", "cli")),
)


def _first_sentence(doc: str | None) -> str:
    if not doc:
        return ""
    text = " ".join(inspect.cleandoc(doc).split("\n\n")[0].split())
    for stop in (". ", ".  "):
        if stop in text:
            return text.split(stop)[0] + "."
    return text


def _api() -> list[dict]:
    groups = []
    for title, names in _GROUPS:
        modules = []
        for name in names:
            module = importlib.import_module(f"magform.{name}")
            members = []
            for public in getattr(module, "__all__", ()):
                obj = getattr(module, public)
                if inspect.isclass(obj):
                    kind = "class"
                elif callable(obj):
                    kind = "function"
                else:
                    continue                                   # constants are listed on the reference pages
                if getattr(obj, "__module__", module.__name__) != module.__name__:
                    continue                                   # re-exports are listed where they are defined
                members.append({"name": public, "kind": kind, "summary": _first_sentence(obj.__doc__)})
            modules.append({"name": name, "summary": _first_sentence(module.__doc__),
                            "classes": [m for m in members if m["kind"] == "class"],
                            "functions": [m for m in members if m["kind"] == "function"]})
        groups.append({"title": title, "modules": modules})
    return groups


def chapter_guide(site: Site) -> None:
    site.data["worked"] = {step["key"]: step for step in _run_steps()}
    examples = {}
    for key, command, keep in CLI_EXAMPLES:
        text = _run_cli(command)
        lines = text.split("\n")
        cut = keep is not None and len(lines) > keep
        examples[key] = {"command": f"magform {command}", "output": "\n".join(lines[:keep] if cut else lines),
                         "truncated": len(lines) - keep if cut else 0}
    site.data["cli"] = {"commands": _commands(), "examples": examples}
    api = _api()
    site.data["api"] = api
    exports = sorted(getattr(magform, "_EXPORTS", {}))
    site.data["api_top"] = exports
    site.nums(
        g_modules=str(sum(len(g["modules"]) for g in api)),
        g_functions=str(sum(len(m["functions"]) for g in api for m in g["modules"])),
        g_classes=str(sum(len(m["classes"]) for g in api for m in g["modules"])),
        g_commands=str(len(site.data["cli"]["commands"])),
    )
