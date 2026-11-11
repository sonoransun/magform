"""Tests for magform.shock: impedance matching, phase-transition branches, flyer kinematics."""

from __future__ import annotations

import json
import math

import pytest

from magform import materials, shock
from magform.errors import InputError, UnknownMaterialError
from magform.shock import ShockState
from magform.status import Confidence

V_SPEC = math.sqrt(2.0 * 4e9 * 0.01 / (8960.0 * 0.002))  # spec flyer: 4 GPa, 10 mm gap, 2 mm Cu


def _analytic_match(flyer, target, v, flyer_branch=None, target_branch=None):
    """Closed-form impedance match on single linear branches (independent of the bisection)."""
    f, t = materials.get(flyer), materials.get(target)
    hf = flyer_branch or f.hugoniot_for(None)
    ht = target_branch or t.hugoniot_for(None)
    a_, b_ = f.density * hf.s, f.density * hf.c0
    c_, d_ = t.density * ht.s, t.density * ht.c0
    # (A - C) u^2 - (2 A v + B + D) u + (A v^2 + B v) = 0
    qa, qb, qc = a_ - c_, -(2.0 * a_ * v + b_ + d_), a_ * v * v + b_ * v
    if abs(qa) < 1e-12 * abs(qb):
        u = -qc / qb
    else:
        disc = math.sqrt(qb * qb - 4.0 * qa * qc)
        roots = [(-qb - disc) / (2.0 * qa), (-qb + disc) / (2.0 * qa)]
        u = next(r for r in roots if 0.0 <= r <= v)
    return u, t.density * (ht.c0 + ht.s * u) * u


# ------------------------------------------------------------- spec checks
def test_flyer_velocity_spec_check(cu):
    v = shock.flyer_velocity(4e9, 0.01, cu, 0.002)
    assert v == pytest.approx(2113.0, rel=1e-2)
    assert v == pytest.approx(V_SPEC, rel=1e-12)
    assert shock.flyer_velocity(4e9, 0.01, "copper_ofhc", 0.002) == v  # key and object agree


def test_cu_on_cu_spec_check():
    s = shock.impedance_match("copper_ofhc", "copper_ofhc", 2113.0)
    assert s.pressure == pytest.approx(52.2e9, rel=3e-2)
    assert s.particle_velocity == pytest.approx(2113.0 / 2.0, rel=1e-9)
    assert s.flyer_particle_velocity == pytest.approx(2113.0 / 2.0, rel=1e-9)
    assert s.shock_speed_flyer == pytest.approx(s.shock_speed_target, rel=1e-12)
    assert s.confidence is Confidence.HANDBOOK and s.branch_note == ""


@pytest.mark.parametrize("key, v", [("copper_ofhc", 777.0), ("tungsten", 1500.0),
                                    ("graphite_crystal", 3000.0), ("steel_1018", 2500.0),
                                    ("iron_armco", 700.0)])
def test_identical_materials_split_velocity_exactly(key, v):
    s = shock.impedance_match(key, key, v)
    assert s.particle_velocity == pytest.approx(v / 2.0, rel=1e-9)


# ------------------------------------------------------------ invariants
@pytest.mark.parametrize("flyer, target, v", [("copper_ofhc", "graphite_crystal", V_SPEC),
                                              ("copper_ofhc", "tungsten", 1200.0),
                                              ("titanium_cp2", "copper_ofhc", 900.0),
                                              ("aluminum_6061", "steel_1018", 500.0)])
def test_momentum_jump_holds_on_both_sides(flyer, target, v):
    s = shock.impedance_match(flyer, target, v)
    rho_f, rho_t = materials.get(flyer).density, materials.get(target).density
    assert s.pressure == pytest.approx(rho_t * s.shock_speed_target * s.particle_velocity, rel=1e-9)
    assert s.pressure == pytest.approx(rho_f * s.shock_speed_flyer * s.flyer_particle_velocity, rel=1e-9)
    assert s.particle_velocity + s.flyer_particle_velocity == pytest.approx(v, rel=1e-12)
    # matches the closed-form root of the linear-Hugoniot quadratic (no transition involved here)
    u_ref, p_ref = _analytic_match(flyer, target, v)
    assert s.particle_velocity == pytest.approx(u_ref, rel=1e-9)
    assert s.pressure == pytest.approx(p_ref, rel=1e-9)


def test_swapping_flyer_and_target_preserves_pressure():
    a = shock.impedance_match("copper_ofhc", "graphite_crystal", 1800.0)
    b = shock.impedance_match("graphite_crystal", "copper_ofhc", 1800.0)
    assert a.pressure == pytest.approx(b.pressure, rel=1e-9)
    assert a.particle_velocity == pytest.approx(b.flyer_particle_velocity, rel=1e-9)


def test_acoustic_limit_small_velocity():
    cu, w = materials.get("copper_ofhc"), materials.get("tungsten")
    z_f = cu.density * cu.hugoniot_for().c0
    z_t = w.density * w.hugoniot_for().c0
    v = 0.5
    s = shock.impedance_match(cu, w, v)
    assert s.pressure == pytest.approx(v * z_f * z_t / (z_f + z_t), rel=1e-3)
    # soft target moves faster than v/2, hard target slower
    assert shock.impedance_match(cu, "graphite_crystal", 1000.0).particle_velocity > 500.0
    assert s.particle_velocity < v / 2.0


def test_pressure_monotonic_in_velocity_and_impedance_ordering():
    ps = [shock.impedance_match("copper_ofhc", "graphite_crystal", v).pressure for v in (200, 600, 1200, 2113)]
    assert all(b > a for a, b in zip(ps, ps[1:]))
    by_target = {t: shock.impedance_match("copper_ofhc", t, V_SPEC).pressure
                 for t in ("graphite_polycrystalline", "graphite_crystal", "copper_ofhc", "tungsten")}
    order = sorted(by_target, key=by_target.get)
    assert order == ["graphite_polycrystalline", "graphite_crystal", "copper_ofhc", "tungsten"]


# ---------------------------------------------------- the honest headline
def test_cu_flyer_into_graphite_is_about_20_gpa_not_52():
    s = shock.impedance_match("copper_ofhc", "graphite_crystal", V_SPEC)
    assert 12e9 <= s.pressure <= 30e9
    assert s.pressure == pytest.approx(22.34e9, rel=1e-3)   # value of the spec's own equations
    assert s.confidence is Confidence.ESTIMATED            # graphite Hugoniot is estimated
    assert "validity limit" in s.branch_note                # 22 GPa > the fit's 20 GPa p_max
    poly = shock.impedance_match("copper_ofhc", "graphite_polycrystalline", V_SPEC)
    assert poly.pressure == pytest.approx(15.81e9, rel=1e-3)
    cucu = shock.impedance_match("copper_ofhc", "copper_ofhc", V_SPEC)
    assert cucu.pressure / s.pressure > 2.0


# ------------------------------------------------- phase-transition logic
def test_iron_low_branch_below_transition():
    s = shock.impedance_match("copper_ofhc", "iron_armco", 300.0)
    u_ref, p_ref = _analytic_match("copper_ofhc", "iron_armco", 300.0)
    assert s.pressure == pytest.approx(p_ref, rel=1e-9)
    assert s.pressure < 0.8 * 13e9
    assert "two-wave" not in s.branch_note and "high-pressure" not in s.branch_note
    assert s.confidence is Confidence.ESTIMATED  # alpha-Fe branch is estimated


def test_iron_high_branch_matches_two_pass_resolve():
    fe = materials.get("iron_armco")
    high = fe.hugoniot_for(math.inf)
    s = shock.impedance_match("copper_ofhc", "iron_armco", 2000.0)
    u_ref, p_ref = _analytic_match("copper_ofhc", "iron_armco", 2000.0, target_branch=high)
    assert s.pressure == pytest.approx(p_ref, rel=1e-9)
    assert s.shock_speed_target == pytest.approx(high.c0 + high.s * s.particle_velocity, rel=1e-12)
    assert "high-pressure Hugoniot branch" in s.branch_note
    assert s.pressure > 1.5 * 13e9 and "two-wave" not in s.branch_note
    assert s.confidence is Confidence.HANDBOOK  # epsilon-Fe branch (Steinberg)


def test_iron_mixed_phase_plateau_where_two_pass_is_inconsistent():
    fe = materials.get("iron_armco")
    v = 700.0
    _, p_low = _analytic_match("copper_ofhc", "iron_armco", v)
    _, p_high = _analytic_match("copper_ofhc", "iron_armco", v, target_branch=fe.hugoniot_for(math.inf))
    assert p_low > 13e9 > p_high  # the literal two-pass rule has no self-consistent answer here
    s = shock.impedance_match("copper_ofhc", "iron_armco", v)
    assert s.pressure == pytest.approx(13e9, rel=1e-12)
    assert "pinned" in s.branch_note and "two-wave" in s.branch_note
    # continuity: pressures just outside the plateau bracket the transition
    assert shock.impedance_match("copper_ofhc", "iron_armco", 600.0).pressure < 13e9
    assert shock.impedance_match("copper_ofhc", "iron_armco", 800.0).pressure > 13e9


def test_transforming_flyer_uses_its_own_branch():
    s = shock.impedance_match("steel_1018", "copper_ofhc", 2500.0)
    assert "flyer steel_1018: high-pressure Hugoniot branch" in s.branch_note


def test_material_without_measured_hugoniot_is_estimated():
    s = shock.impedance_match("copper_ofhc", "gamma_tial", 1000.0)
    assert s.confidence is Confidence.ESTIMATED
    assert "estimated" in s.branch_note
    tial = materials.get("gamma_tial")
    assert s.shock_speed_target == pytest.approx(tial.bulk_sound_speed() + 1.4 * s.particle_velocity, rel=1e-9)


# --------------------------------------------------- pressure -> state
def test_shock_state_from_pressure_inverts_the_hugoniot():
    s = shock.impedance_match("copper_ofhc", "copper_ofhc", 2113.0)
    u, us = shock.shock_state_from_pressure("copper_ofhc", s.pressure)
    assert u == pytest.approx(s.particle_velocity, rel=1e-9)
    assert us == pytest.approx(s.shock_speed_target, rel=1e-9)
    # iron above 13 GPa sits on the epsilon branch
    u_fe, us_fe = shock.shock_state_from_pressure("iron_armco", 20e9)
    assert us_fe == pytest.approx(3574.0 + 1.92 * u_fe, rel=1e-12)
    assert 7870.0 * us_fe * u_fe == pytest.approx(20e9, rel=1e-9)
    assert shock.shock_state_from_pressure("copper_ofhc", 0.0) == (0.0, 3940.0)


# ------------------------------------------------------ flyer kinematics
def test_kinematic_inverses_and_scaling(cu):
    p, g, h = 4e9, 0.01, 0.002
    v = shock.flyer_velocity(p, g, cu, h)
    assert shock.required_standoff(v, p, cu, h) == pytest.approx(g, rel=1e-12)
    assert shock.required_pressure_for_velocity(v, g, cu, h) == pytest.approx(p, rel=1e-12)
    t = shock.flyer_transit_time(g, p, cu, h)
    assert t == pytest.approx(2.0 * g / v, rel=1e-12)          # uniform acceleration
    assert t == pytest.approx(9.466e-6, rel=1e-3)
    assert shock.flyer_velocity(p, g, cu, h, efficiency=0.25) == pytest.approx(0.5 * v, rel=1e-12)
    assert shock.flyer_velocity(4 * p, g, cu, h) == pytest.approx(2.0 * v, rel=1e-12)


def test_zero_drive_limits(cu):
    assert shock.flyer_velocity(0.0, 0.01, cu, 0.002) == 0.0
    assert math.isinf(shock.flyer_transit_time(0.01, 0.0, cu, 0.002))
    assert math.isinf(shock.required_standoff(100.0, 0.0, cu, 0.002))
    assert shock.required_standoff(0.0, 0.0, cu, 0.002) == 0.0
    s = shock.impedance_match(cu, "tungsten", 0.0)
    assert s.pressure == 0.0 and s.particle_velocity == 0.0
    assert s.shock_speed_target == pytest.approx(4029.0)


def test_shock_duration_and_temperature_estimates(cu):
    s = shock.impedance_match(cu, cu, V_SPEC)
    assert shock.shock_duration(0.002, s.shock_speed_flyer) == pytest.approx(2 * 0.002 / 5513.04, rel=1e-5)
    assert shock.shock_temperature_rise(s.particle_velocity, cu) == pytest.approx(
        s.particle_velocity ** 2 / (2 * 385.0), rel=1e-12)
    g = shock.impedance_match(cu, "graphite_crystal", V_SPEC)
    assert shock.shock_temperature_rise(g.particle_velocity, "graphite_crystal") == pytest.approx(1769.7, rel=1e-3)


# ----------------------------------------------------------- error paths
@pytest.mark.parametrize("call", [
    lambda: shock.impedance_match("cu", "cu", -1.0),
    lambda: shock.impedance_match("cu", "cu", float("nan")),
    lambda: shock.impedance_match("cu", "cu", math.inf),
    lambda: shock.flyer_velocity(-1e9, 0.01, "cu", 0.002),
    lambda: shock.flyer_velocity(4e9, 0.0, "cu", 0.002),
    lambda: shock.flyer_velocity(4e9, 0.01, "cu", 0.0),
    lambda: shock.flyer_velocity(4e9, 0.01, "cu", 0.002, efficiency=0.0),
    lambda: shock.flyer_velocity(4e9, 0.01, "cu", 0.002, efficiency=1.5),
    lambda: shock.flyer_velocity("lots", 0.01, "cu", 0.002),
    lambda: shock.flyer_transit_time(-0.01, 4e9, "cu", 0.002),
    lambda: shock.required_standoff(-5.0, 4e9, "cu", 0.002),
    lambda: shock.required_pressure_for_velocity(100.0, 0.0, "cu", 0.002),
    lambda: shock.shock_state_from_pressure("cu", -1.0),
    lambda: shock.shock_duration(0.002, 0.0),
    lambda: shock.shock_duration(-0.002, 5000.0),
    lambda: shock.shock_temperature_rise(-10.0, "cu"),
])
def test_invalid_inputs_raise_input_error(call):
    with pytest.raises(InputError):
        call()


def test_unknown_material_raises():
    with pytest.raises(UnknownMaterialError):
        shock.impedance_match("unobtainium", "cu", 100.0)


# ------------------------------------------------------- serialisation
def test_shockstate_json_round_trip_and_text():
    s = shock.impedance_match("copper_ofhc", "iron_armco", 700.0)
    assert isinstance(s, ShockState)
    d = json.loads(s.json())
    assert d["flyer"] == "copper_ofhc" and d["target"] == "iron_armco"
    assert d["confidence"] in ("estimated", "handbook")
    assert d["pressure"] == pytest.approx(13e9)
    assert set(d) == {"flyer", "target", "impact_velocity", "pressure", "particle_velocity",
                      "flyer_particle_velocity", "shock_speed_target", "shock_speed_flyer",
                      "confidence", "branch_note"}
    txt = s.text()
    assert txt.startswith("ShockState") and "pressure [Pa] = 1.3e+10" in txt
    assert "shock_speed_flyer [m/s]" in txt
    with pytest.raises(Exception):
        s.pressure = 1.0  # frozen


def test_public_api_is_documented():
    for name in shock.__all__:
        obj = getattr(shock, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 80, name
    assert all(isinstance(k, str) and v for k, v in shock.REFERENCES.items())
