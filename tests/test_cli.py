"""Tests for the magform command-line interface."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys

import pytest

from magform import __version__, cli


def run(capsys, command: str) -> tuple[int, str, str]:
    try:
        code = cli.main(shlex.split(command))
    except SystemExit as exc:  # argparse usage errors
        code = exc.code
    out, err = capsys.readouterr()
    return code, out, err


def test_field_command_both_directions(capsys):
    code, out, _ = run(capsys, "field --pressure 4000")
    assert code == 0 and "100.2651 T" in out and "state of the art" in out and "steady field exerts no pressure" in out
    code, out, _ = run(capsys, "field --field 50 --json")
    data = json.loads(out)
    assert code == 0 and data["pressure"] == pytest.approx(0.9947e9, rel=1e-3)
    assert data["superconducting_status"] == "aspirational" and "hpht_cubic" in data["above"]
    for bad in ("field", "field --pressure 4000 --field 100"):
        code, out, err = run(capsys, bad)
        assert code == 2 and out == "" and "exactly one" in err


def test_ci_smoke_commands(capsys):
    # the two commands the CI workflow runs
    code, out, _ = run(capsys, "press --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2")
    assert code == 0 and "100.27 T" in out and "infeasible" in out and "Resistive pulsed magnet" in out
    assert "Best option: Resistive pulsed magnet" in out and "no enthalpy margin" in out
    code, out, _ = run(capsys, "press --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2 --json")
    data = json.loads(out)
    assert code == 0 and data["status"] == "infeasible" and data["best_architecture"] == "pulsed_resistive"
    assert data["field"] == pytest.approx(100.27, rel=1e-4) and len(data["architectures"]) == 6


def test_press_defaults_to_the_reference_geometry(capsys):
    code, out, _ = run(capsys, "press --pressure 4000 --json")
    short = json.loads(out)
    code2, out2, _ = run(capsys, "press --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2 --json")
    assert code == 0 and code2 == 0 and short == json.loads(out2)
    assert short["spec"]["bore_diameter"] == 0.1 and short["stored_energy"] == pytest.approx(99.5e6, rel=1e-2)


def test_magnetar_command_reports_an_assumed_source(capsys):
    code, out, _ = run(capsys, "magnetar --field 1e10 --radius 0.01")
    assert code == 0 and "assumed, not assessed" in out and "status = assumed" in out
    code, out, _ = run(capsys, "magnetar --field 1e10 --radius 0.01 --json")
    data = json.loads(out)
    assert data["status"] == "assumed" and data["field_energy"] == pytest.approx(2.5e20, rel=1e-2)
    assert data["field_energy_megatons"] == pytest.approx(59751, rel=1e-3)
    code, _, err = run(capsys, "magnetar --radius 0")
    assert code == 2 and "radius must be > 0" in err


def test_press_options_reach_the_model(capsys):
    code, out, _ = run(capsys, "architectures --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2 --wall 0.002 --json")
    archs = {a["key"]: a for a in json.loads(out)}
    assert code == 0 and archs["pulsed_resistive"]["status"] == "infeasible"
    assert archs["pulsed_resistive"]["relaxed_pressure_fraction"] == pytest.approx(0.51, abs=0.03)
    code, out, _ = run(capsys, "press --pressure 1600 --bore 0.1 --length 0.5 --winding 0.2 --workpiece tungsten --rise 0.005 --json")
    data = json.loads(out)
    assert data["spec"]["workpiece"] == "tungsten" and data["rise_time"] == 0.005


@pytest.mark.parametrize("command, fragment", [
    ("forge tungsten --pressure 2000", "infeasible"),
    ("forge copper --pressure 2000", "state of the art"),
    ("micro tungsten --field 100", "heat-affected depth"),
    ("weld al steel --pressure 100", "wavy"),
    ("weld al steel --pressure 4000", "infeasible"),
    ("weld al steel", "drive pressure window   94.5-284.3 MPa"),
    ("synth --pressure 4000", "infeasible"),
    ("synth --pressure 4000 --flyer cu", "22.3 GPa"),
    ("pores gamma_tial --pressure 4000 --porosity 0.005", "healing"),
    ("contain --pressure 4000 --bore 0.05", "10.4 GPa"),
    ("contain --pressure 160 --bore 1.0 --material steel_4340 --pulse 0.01", "quasi-static"),
    ("pulse --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2 --rise 0.01", "homopolar"),
    ("claims", "contradicted: 10"),
    ("claims C08", "perfectly uniform"),
    ("claims --verdict aspirational --full", "C27"),
    ("claims --markdown", "| C01 |"),
    ("materials", "tungsten"),
    ("materials --category refractory", "molybdenum"),
    ("materials W", "Lennon & Ramesh"),
    ("benchmarks", "100.75 T"),
    ("benchmarks energy", "MJ" if False else "J"),
])
def test_commands_run_and_emit_json(capsys, command, fragment):
    code, out, err = run(capsys, command)
    assert code == 0 and err == "" and fragment in out, out[:400]
    if "--markdown" not in command:
        code, out, _ = run(capsys, command + " --json")
        assert code == 0
        json.loads(out)  # strict JSON: would raise on NaN/Infinity


@pytest.mark.parametrize("command, message", [
    ("forge unobtainium --pressure 100", "unknown material"),
    ("press --pressure -5 --bore 0.1 --length 0.5 --winding 0.2", "pressure must be positive"),
    ("claims C99", "unknown claim"),
    ("benchmarks nonsense", "unknown ladder"),
    ("weld al diamond --pressure 100", "magform:"),
    ("pulse --bore 0.1 --length 0.5 --winding 0.2 --rise 0.01", "exactly one"),
])
def test_invalid_input_exits_2_with_a_clean_message(capsys, command, message):
    code, out, err = run(capsys, command)
    assert code == 2 and out == "" and message in err and "Traceback" not in err


def test_usage_errors_and_help(capsys):
    code, out, _ = run(capsys, "")
    assert code == 0 and "usage: magform" in out and "claims" in out
    code, _, err = run(capsys, "nonsense")
    assert code == 2 and "invalid choice" in err
    code, out, _ = run(capsys, "--version")
    assert code == 0 and out.strip() == f"magform {__version__}"
    code, _, err = run(capsys, "press")
    assert code == 2 and "required" in err


def test_module_entry_point():
    out = subprocess.run([sys.executable, "-m", "magform", "field", "--pressure", "1600", "--json"],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)["field"] == pytest.approx(63.41, rel=1e-4)


def test_every_command_is_documented():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert set(sub.choices) == {"field", "press", "architectures", "pulse", "forge", "micro", "weld", "synth",
                                "pores", "contain", "magnetar", "claims", "materials", "benchmarks"}
    for name, p in sub.choices.items():
        assert p.description and name in cli.__doc__, name
        assert any(a.dest == "json" for a in p._actions), name
