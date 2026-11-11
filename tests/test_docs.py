"""The documentation site is generated from the package; these tests keep the two in step.

They need no Jekyll: they check the generated data, the pages' references into it, the commands
and code the pages quote, and (when Node is installed) the browser kernel against the package.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import magform
from magform import benchmarks, claims, cli, eddy, magnetar, materials, pulse, status
from magform.magnetics import Solenoid, field_for_pressure, magnetic_pressure

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "_data"
sys.path.insert(0, str(ROOT / "tools"))

SKIP_DIRS = {"vendor", "archive", "_site", "assets", "_data"}
PAGES = sorted(p for p in list(DOCS.rglob("*.md")) + [DOCS / "404.html"]
               if not SKIP_DIRS & set(p.relative_to(DOCS).parts))
TEMPLATES = sorted(list((DOCS / "_includes").rglob("*.html")) + list((DOCS / "_layouts").glob("*.html")))
BARE_EXPONENT = re.compile(r'"(?:\\.|[^"\\])*"|(?<![\d.])(-?\d+)([eE][-+]?\d+)')


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def front_matter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    head = text[4:].split("\n---\n", 1)[0]
    out = {}
    for line in head.split("\n"):
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def listed(value: str) -> list[str]:
    return [v.strip() for v in value.strip("[]").split(",") if v.strip()]


def source_for(url: str) -> Path | None:
    """The source file that Jekyll turns into a site URL such as /story/ or /ledger.html."""
    url = url.split("#")[0]
    if url in ("", "/"):
        return DOCS / "index.md"
    if url.endswith("/"):
        return DOCS / url.strip("/") / "index.md"
    stem = DOCS / url.strip("/")
    for candidate in (stem.with_suffix(".md"), stem):
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def data() -> dict[str, object]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(DATA.glob("*.json"))}


# ---------------------------------------------------------------------------
# generated data
# ---------------------------------------------------------------------------
def test_committed_data_matches_a_fresh_build():
    import build_docs

    stale = build_docs.stale_files(build_docs.render(build_docs.build()))
    assert not stale, f"run `python tools/build_docs.py`; stale: {[rel(p) for p in stale]}"


def test_generated_json_is_strict_and_safe_for_jekylls_yaml_loader():
    import build_docs

    assert build_docs.yaml_safe_json('{"a": 6e+20, "b": 1.25e-05, "c": "4e-89 s", "d": [-1e-07]}') == \
        '{"a": 6.0e+20, "b": 1.25e-05, "c": "4e-89 s", "d": [-1.0e-07]}'

    def no_constants(name: str) -> float:
        raise AssertionError(f"non-finite number {name} in generated JSON")

    for path in sorted(DATA.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        json.loads(text, parse_constant=no_constants)
        # Jekyll reads these files as YAML 1.1, where 6e+20 (no decimal point) is a string, not a number
        bare = [m.group(0) for m in BARE_EXPONENT.finditer(text) if m.group(1) is not None]
        assert not bare, f"{rel(path)}: numbers YAML 1.1 would read as strings: {bare[:5]}"


def test_every_quoted_number_figure_and_data_file_exists(data):
    numbers, figures = data["n"], data["figures"]
    for path in PAGES + TEMPLATES:
        text = path.read_text(encoding="utf-8")
        missing = sorted({k for k in re.findall(r"\bn\.([A-Za-z0-9_]+)", text) if k not in numbers})
        assert not missing, f"{rel(path)} quotes numbers that are not generated: {missing}"
        absent = sorted({f for f in re.findall(r'figure\.html\s+id="([^"]+)"', text) if f not in figures})
        assert not absent, f"{rel(path)} includes figures that are not generated: {absent}"
        files = sorted({d for d in re.findall(r"site\.data\.([a-z_]+)", text) if d not in data and d != "nav"})
        assert not files, f"{rel(path)} reads data files that do not exist: {files}"


def test_every_generated_figure_is_shown_somewhere(data):
    shown = set()
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        shown |= set(re.findall(r'figure\.html\s+id="([^"]+)"', text))
        shown |= set(re.findall(r"site\.data\.figures\.([a-z_]+)", text))
    assert set(data["figures"]) <= shown, f"figures generated but never shown: {sorted(set(data['figures']) - shown)}"


def test_charts_have_titles_tables_and_finite_values(data):
    def numbers(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from numbers(v)
        elif isinstance(node, list):
            for v in node:
                yield from numbers(v)
        elif isinstance(node, float):
            yield node

    for key, spec in data["figures"].items():
        assert spec["type"] in ("xy", "dot", "bars"), key
        assert spec.get("title") and not spec["title"].endswith("."), key
        assert all(v == v and abs(v) != float("inf") for v in numbers(spec)), key
        if spec["type"] == "xy":
            named = [s for s in spec["series"] if s.get("name")]
            assert named and all(len(s["x"]) == len(s["y"]) for s in spec["series"]), key
            lines = [s for s in named if s.get("kind", "line") == "line" and not s.get("thin")]
            assert len({s.get("slot", s.get("tone")) for s in lines}) == len(lines), f"{key}: two lines share a colour"
            assert "table" in spec, f"{key}: an xy chart needs a table twin"
        else:
            assert spec["rows"], key


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def test_navigation_and_pages_agree():
    nav = (DATA / "nav.yml").read_text(encoding="utf-8")
    urls = re.findall(r"url:\s*([^\s,}]+)", nav)
    assert len(urls) == len(set(urls)) and len(urls) >= 20
    sources = {source_for(u) for u in urls}
    assert None not in sources, [u for u in urls if source_for(u) is None]
    in_sections = {p for p in PAGES if p.parent.name in ("story", "guide", "reference")}
    assert in_sections <= sources, f"pages missing from the navigation: {[rel(p) for p in in_sections - sources]}"
    story = [u for u in urls if u.startswith("/story/")]
    numbers = re.findall(r'url:\s*/story/[^\s,}]*,\s*n:\s*"(\d+)"', nav)
    assert [int(x) for x in numbers] == list(range(len(story)))


def test_front_matter_is_complete_and_valid(data):
    verdicts = {v.value for v in status.Verdict} | {s.value for s in status.Status}
    ledger = data["claims"]["by_id"]
    modules = set(magform._SUBMODULES)
    docs = set()
    for path in PAGES:
        fm = front_matter(path)
        assert fm.get("title"), rel(path)
        if path.name in ("index.md", "404.html") and path.parent == DOCS:
            continue
        assert fm.get("description") and fm.get("kicker") and fm.get("doc"), f"{rel(path)}: needs kicker, doc and description"
        assert fm["doc"] not in docs, f"{rel(path)}: document number {fm['doc']} used twice"
        docs.add(fm["doc"])
        if "verdict" in fm:
            assert fm["verdict"] in verdicts, rel(path)
        for cid in listed(fm.get("claims", "")):
            assert cid in ledger, f"{rel(path)}: unknown claim {cid}"
        for module in listed(fm.get("models", "")):
            assert module in modules, f"{rel(path)}: unknown module {module}"
        if path.parent.name == "story" and path.name != "index.md":
            assert fm.get("brief"), f"{rel(path)}: chapters need a one-sentence brief"


def test_every_claim_is_tested_in_the_chapter_it_points_to(data):
    seen: dict[str, str] = {}
    for path in sorted((DOCS / "story").glob("*.md")):
        for cid in listed(front_matter(path).get("claims", "")):
            assert cid not in seen, f"{cid} is under test in two chapters"
            seen[cid] = f"/story/{path.stem}.html"
    ledger = data["claims"]["by_id"]
    assert set(seen) == set(ledger), sorted(set(ledger) ^ set(seen))
    wrong = {cid: (seen[cid], ledger[cid]["chapter"]) for cid in ledger if seen[cid] != ledger[cid]["chapter"]}
    assert not wrong, wrong


def test_internal_links_resolve():
    for path in PAGES + TEMPLATES:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "#", "mailto:", "{{")):
                continue
            file = (path.parent / target.split("#")[0]).resolve()
            assert file.exists(), f"{rel(path)}: broken link {target}"
            assert file.suffix == ".md" or file.is_dir(), f"{rel(path)}: link to {target}; link to the .md source"
        for url in re.findall(r"""['"](/[^'"]*)['"]\s*\|\s*relative_url""", text):
            if url.startswith("/assets/"):
                assert (DOCS / url.lstrip("/")).exists() or url.endswith("/"), f"{rel(path)}: missing asset {url}"
            elif "append" not in url:
                assert source_for(url) is not None, f"{rel(path)}: broken site link {url}"


def test_no_placeholder_text_and_the_note_is_reproduced_verbatim():
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        for word in ("Stub", "TODO", "FIXME", "lorem ipsum", "traceability.md"):
            assert word not in text, f"{rel(path)} contains {word!r}"
    archive = (DOCS / "archive" / "original-README.md").read_text(encoding="utf-8").strip()
    page = (DOCS / "original-concept.md").read_text(encoding="utf-8")
    body = page.split("{% raw %}", 1)[1].split("{% endraw %}", 1)[0].strip()
    assert body == archive
    for claim in claims.CLAIMS:
        assert claim.quote in archive, claim.id


# ---------------------------------------------------------------------------
# what the pages tell the reader to run
# ---------------------------------------------------------------------------
def _blocks(language: str) -> list[tuple[Path, str]]:
    out = []
    for path in PAGES + [ROOT / "README.md"]:
        text = path.read_text(encoding="utf-8")
        for block in re.findall(rf"^```{language}\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL):
            out.append((path, block))
    return out


def test_console_commands_quoted_in_the_docs_run(capsys):
    ran = 0
    for path, block in _blocks("console"):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            if not line.startswith("$ magform ") or "{{" in line:
                continue
            command = line[len("$ magform "):].split("#")[0].split(">")[0].strip()
            expect_error = i + 1 < len(lines) and lines[i + 1].startswith("magform: ")
            try:
                code = cli.main(shlex.split(command))
            except SystemExit as stop:
                code = stop.code
            capsys.readouterr()
            assert (code == 2) if expect_error else (code == 0), f"{rel(path)}: `magform {command}` exited {code}"
            ran += 1
    assert ran >= 25


def test_python_examples_in_the_docs_run():
    ran = 0
    for path, block in _blocks("python"):
        if "{{" in block:          # filled in by Liquid from generated data; covered by the build itself
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(block, rel(path), "exec"), {"__name__": "__docs__"})   # noqa: S102
            except Exception as problem:   # noqa: BLE001 - report which page
                pytest.fail(f"{rel(path)}: example raised {type(problem).__name__}: {problem}\n{block}")
        ran += 1
    assert ran >= 15


def test_generated_examples_are_what_the_package_prints(data):
    for key, example in data["cli"]["examples"].items():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            assert cli.main(shlex.split(example["command"][len("magform "):])) == 0
        assert buffer.getvalue().rstrip().startswith(example["output"]), key
    assert {c["name"] for c in data["cli"]["commands"]} == set(cli.build_parser()._subparsers._group_actions[0].choices)


def test_readme_ledger_table_matches_the_package(data):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- ledger:begin -->", 1)[1].split("<!-- ledger:end -->", 1)[0].strip().split("\n")[2:]
    expected = [f"| {row['verdict'].replace('_', ' ')} | {row['count']} | {row['meaning']} |" for row in data["claims"]["counts"]]
    assert table == expected
    assert f"{field_for_pressure(4e9):.2f} T" in readme and f"{benchmarks.PULSED_NONDESTRUCTIVE_RECORD:g} T" in readme
    assert f"{benchmarks.SC_FIELD_RECORD:g} T" in readme and f"{benchmarks.SC_CONDUCTOR_RECORD:g} T" in readme
    for folder in re.findall(r"^(\w+)/ ", readme.split("## Repository layout", 1)[1], flags=re.MULTILINE):
        assert (ROOT / folder).is_dir(), folder


def test_examples_run(capsys):
    import runpy

    scripts = sorted((ROOT / "examples").glob("*.py"))
    assert len(scripts) >= 4
    for script in scripts:
        runpy.run_path(str(script), run_name="__main__")
        out = capsys.readouterr().out
        assert len(out.splitlines()) >= 5, script.name


# ---------------------------------------------------------------------------
# the browser kernel
# ---------------------------------------------------------------------------
KERNEL = DOCS / "assets" / "js" / "magform.js"
INTERACTIVE = DOCS / "assets" / "js" / "interactive.js"

NODE_SCRIPT = r"""
const M = require(process.argv[process.argv.length - 1]);
const c = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const T = c.thresholds, out = {};
out.constants = [M.MU0, M.C];
out.field = c.pressures.map(M.fieldForPressure);
out.pressure = c.fields.map(M.magneticPressure);
out.coils = c.coils.map(([a1, a2, L]) => { const s = M.solenoid(a1, a2, L); return c.fields.map(b =>
  [s.fieldFactor, s.currentDensityForField(b), s.storedEnergy(b), s.maxHoopStress(b), s.averageHoopStress(b)]); });
out.skin = c.skin.map(([rho, f]) => M.skinDepth(rho, f));
out.rise = c.rise.map(([rho, wall]) => M.matchedRiseTime(rho, wall, c.depth_ratio));
out.transfer = c.transfer.map(([h, d]) => M.pressureTransfer(h, d));
out.heat = c.heat.map(([b, rho, cp]) => M.skinTemperatureRise(b, rho, cp));
out.power = c.power.map(([e, t]) => M.peakPower(e, t));
out.sc = c.fields_status.map(b => M.superconductingStatus(b, T));
out.pulsed = c.fields_status.map(b => c.bores.map(d => M.pulsedStatus(b, d, T)));
out.energy = c.energies.map(e => M.energyStatus(e, T));
out.powers = c.powers.map(p => M.powerStatus(p, T));
out.voltage = c.voltages.map(v => M.voltageStatus(v, T));
out.current = c.currents.map(i => M.currentStatus(i, T));
out.supply = c.supply.map(([a1, a2, L, b, t]) => { const s = M.supply(M.solenoid(a1, a2, L), b, t, T);
  return [s.ampereTurns, s.singleTurnVoltage, s.turns, s.voltage, s.current, s.coilSkinMelts]; });
out.worst = c.worst.map(M.worst);
out.source = c.sources.map(([b, r]) => { const e = M.sourceEnergy(b, r); return [e, M.massEquivalent(e), M.confinementTime(r)]; });
out.standoff = c.standoff.map(([b, r, th]) => M.standoffRadius(b, r, th));
out.tier = c.tier_fields.map(b => M.tierFor(b, c.tier_bounds));
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_kernel_matches_the_package(data):
    explorer = data["explorer"]
    thresholds = explorer["thresholds"]
    copper, titanium = materials.get("copper_ofhc"), materials.get("ti6al4v")
    coils = [(0.05, 0.25, 0.5), (0.005, 0.11, 0.15), (0.1, 0.15, 0.3)]          # thick, record-like, thin (a2 < 2 a1)
    fields = [10.0, 63.41, 100.27]
    cases = {
        "thresholds": thresholds, "depth_ratio": explorer["constants"]["depth_ratio"],
        "tier_bounds": explorer["magnetar"]["tier_bounds"],
        "pressures": [1e8, 1.6e9, 4e9], "fields": fields, "coils": coils,
        "skin": [(copper.resistivity, 1e4), (titanium.resistivity, 50.0)],
        "rise": [(copper.resistivity, 0.002), (copper.resistivity, 0.04), (titanium.resistivity, 0.01)],
        "transfer": [(0.002, 0.0042), (0.04, 0.0133), (1e-4, 1e-2)],
        "heat": [(100.27, copper.density, copper.specific_heat), (30.0, titanium.density, titanium.specific_heat)],
        "power": [(99.5e6, 2.5e-5), (99.5e6, 1e-2)],
        "fields_status": [20.0, 35.6, 35.7, 48.7, 48.8, 60.0, 60.1, 100.75, 100.8, 300.0],
        "bores": [0.010, 0.0105, 0.011, 0.025, 0.026, 0.1],
        "energies": [1e6, 50e6, 50.1e6, 1.2e9, 1.3e9], "powers": [1e9, 1e12, 1.1e12, 8e13, 9e13],
        "voltages": [1e3, 25e3, 25.1e3, 100e3, 101e3], "currents": [1e5, 5e6, 5.1e6, 26e6, 27e6],
        "supply": [(0.05, 0.25, 0.5, 100.2651, 2.565e-5), (0.05, 0.25, 0.5, 100.2651, 1.026e-2),
                   (0.05, 0.25, 0.5, 63.41, 2.565e-5), (0.01, 0.03, 0.1, 40.0, 1e-4)],
        "worst": [["state_of_the_art", "extrapolation"], ["aspirational", "extrapolation", "infeasible"], ["state_of_the_art"]],
        "sources": [(1e10, 0.01), (1e8, 1e-6), (300.0, 0.5)],
        "standoff": [(1e10, 0.01, 100.27), (1e10, 0.01, 5e-4), (50.0, 0.1, 100.0)],
        "tier_fields": [1.0, 100.0, 120.0, 500.0, 700.0, 5e3, 1e4, 1e5, 1e6, 1e8, 2e8, 1e10],
    }
    run = subprocess.run(["node", "-e", NODE_SCRIPT, str(KERNEL)], input=json.dumps(cases), capture_output=True,
                         text=True, check=False, timeout=60)
    assert run.returncode == 0, run.stderr
    js = json.loads(run.stdout)
    exact = pytest.approx

    from magform.constants import C_LIGHT, MU0
    assert js["constants"] == [MU0, C_LIGHT]
    assert js["field"] == exact([field_for_pressure(p) for p in cases["pressures"]], rel=1e-13)
    assert js["pressure"] == exact([magnetic_pressure(b) for b in fields], rel=1e-13)
    for got, (a1, a2, length) in zip(js["coils"], coils):
        sol = Solenoid(a1, a2, length)
        for row, b in zip(got, fields):
            j = sol.current_density_for_field(b)
            assert row == exact([sol.field_factor(), j, sol.stored_energy(b), sol.max_hoop_stress(j, b)[0],
                                 sol.average_hoop_stress(b)], rel=1e-12)
    assert js["skin"] == exact([eddy.skin_depth(r, f) for r, f in cases["skin"]], rel=1e-13)
    assert js["rise"] == exact([eddy.pulse_frequency_recommendation("copper_ofhc", 0.002).quarter_period,
                                eddy.pulse_frequency_recommendation("copper_ofhc", 0.04).quarter_period,
                                eddy.pulse_frequency_recommendation("ti6al4v", 0.01).quarter_period], rel=1e-12)
    assert js["transfer"] == exact([eddy.sheet_shielding_factor(h, d) for h, d in cases["transfer"]], rel=1e-12)
    assert js["heat"] == exact([eddy.skin_temperature_rise("copper_ofhc", 100.27), eddy.skin_temperature_rise("ti6al4v", 30.0)], rel=1e-12)
    assert js["power"] == exact([pulse.discharge_peak_power(e, t) for e, t in cases["power"]], rel=1e-13)

    assert js["sc"] == [benchmarks.superconducting_field_status(b).value for b in cases["fields_status"]]
    assert js["pulsed"] == [[benchmarks.pulsed_field_status(b, d).value for d in cases["bores"]] for b in cases["fields_status"]]
    assert js["energy"] == [benchmarks.energy_status(e).value for e in cases["energies"]]
    assert js["powers"] == [benchmarks.peak_power_status(p).value for p in cases["powers"]]
    assert js["voltage"] == [benchmarks.pulsed_voltage_status(v).value for v in cases["voltages"]]
    assert js["current"] == [benchmarks.pulsed_current_status(i).value for i in cases["currents"]]
    assert js["worst"] == [status.worst(*[status.Status(s) for s in group]).value for group in cases["worst"]]

    for got, (a1, a2, length, b, t) in zip(js["supply"], cases["supply"]):
        sol = Solenoid(a1, a2, length)
        ampere_turns, single_turn_voltage, turns, voltage, current, melts = got
        assert ampere_turns == exact(sol.current_for_field(b, 1), rel=1e-12)
        assert single_turn_voltage * ampere_turns == exact(pulse.voltage_current_product(sol.stored_energy(b), t), rel=1e-12)
        assert voltage * current == exact(pulse.voltage_current_product(sol.stored_energy(b), t), rel=1e-12)
        full = pulse.assess_pulse_network(sol, b, t)
        assert abs(turns - full.turns) <= 1                       # the package also allows for resistance and stray inductance
        if turns == full.turns:
            assert voltage == exact(full.charge_voltage, rel=0.02) and current == exact(full.peak_current, rel=1e-9)
        assert melts == (full.turns == 1 and full.thermal_status is status.Status.INFEASIBLE)

    for got, (b, r) in zip(js["source"], cases["sources"]):
        src = magnetar.MagnetarSource(b, r)
        assert got == exact([src.energy(), src.mass_equivalent(), src.confinement_time()], rel=1e-12)
    assert js["standoff"] == exact([magnetar.standoff_radius(*args) for args in cases["standoff"]], rel=1e-12)
    assert js["tier"] == [magnetar.classify_regime("iron_armco", b).tier for b in cases["tier_fields"]]


def test_scripts_read_only_data_that_is_generated(data):
    explorer = data["explorer"]
    script = INTERACTIVE.read_text(encoding="utf-8") + KERNEL.read_text(encoding="utf-8")
    read = set(re.findall(r"\bT\.([a-z_]+)", INTERACTIVE.read_text(encoding="utf-8")))
    read |= set(re.findall(r"\bt\.([a-z_]+)", KERNEL.read_text(encoding="utf-8")))
    assert len(read) >= 12
    for key in read:
        assert key in explorer["thresholds"], f"threshold {key} is read by the page scripts but not generated"
    for group, key in set(re.findall(r"\bD\.([a-z_]+)\.([a-z_]+)", script)):
        if group in ("pressures", "materials"):
            continue
        assert key in explorer[group] or key in ("filter", "map", "forEach"), f"explorer.json has no {group}.{key}"
    for key in ("resistivity", "density", "specific_heat", "skin_melting_field", "name", "key"):
        assert all(key in m for m in explorer["materials"])
    mu0 = float(re.search(r"var MU0 = ([0-9.e+-]+);", script).group(1))
    c = float(re.search(r"var C = ([0-9.e+-]+);", script).group(1))
    assert (mu0, c) == (explorer["constants"]["mu0"], explorer["constants"]["c"])


def test_field_line_drawing_is_generated_and_self_contained():
    svg = (DOCS / "_includes" / "generated" / "press-section.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg ") and 'role="img"' in svg and "aria-label=" in svg
    assert "<script" not in svg and "http" not in svg and svg.count("<path") >= 8
    coords = [float(x) for x in re.findall(r"[ML](-?\d+\.\d) ", svg)]
    assert coords and min(coords) >= -0.5 and max(coords) <= 260.5        # one quadrant, mirrored by <use>
