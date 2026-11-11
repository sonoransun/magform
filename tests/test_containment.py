"""Tests for magform.containment (spec section 8: inverted architecture as containment)."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from magform import containment as C
from magform import materials
from magform._report import Assessment, ResultMixin
from magform.benchmarks import PRESSURE_BENCHMARKS
from magform.constants import G_ACCEL, MU0
from magform.errors import InputError, UnknownMaterialError
from magform.status import Status

S3 = math.sqrt(3.0)


def _von_mises(s1: float, s2: float, s3: float) -> float:
    return math.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))


# ------------------------------------------------------------------ Lame wall
def test_lame_boundary_conditions_and_end_force_balance():
    p_i, p_o, a, b = 4e9, 0.5e9, 0.025, 0.08
    sr_a, st_a, sz = C.lame_stresses(p_i, a, b, a, external_pressure=p_o)
    sr_b, _, sz_b = C.lame_stresses(p_i, a, b, b, external_pressure=p_o)
    assert sr_a == pytest.approx(-p_i, rel=1e-12)
    assert sr_b == pytest.approx(-p_o, rel=1e-9)
    # closed ends: axial force balance on the end caps, uniform through the wall
    assert sz == pytest.approx((p_i * a * a - p_o * b * b) / (b * b - a * a), rel=1e-12)
    assert sz_b == sz
    # sigma_r + sigma_theta = 2A is constant across the wall (Lame invariant)
    for r in (a, 0.04, 0.055, 0.07, b):
        sr, st, szr = C.lame_stresses(p_i, a, b, r, external_pressure=p_o)
        assert sr + st == pytest.approx(2.0 * szr, rel=1e-12)
    assert C.lame_stresses(p_i, a, b, a, closed_ends=False)[2] == 0.0
    # with no external pressure the bore hoop stress is the closed-form maximum
    assert C.lame_stresses(p_i, a, b, a)[1] == pytest.approx(C.max_hoop_stress(p_i, a, b), rel=1e-12)


def test_lame_radial_equilibrium_and_hoop_decay():
    p, a, b, h = 2e9, 0.03, 0.09, 1e-7
    for r in (0.04, 0.06, 0.08):
        def r_sigma_r(x: float) -> float:
            return x * C.lame_stresses(p, a, b, x)[0]

        deriv = (r_sigma_r(r + h) - r_sigma_r(r - h)) / (2.0 * h)  # d(r sigma_r)/dr = sigma_theta
        assert deriv == pytest.approx(C.lame_stresses(p, a, b, r)[1], rel=1e-6)
    hoops = [C.lame_stresses(p, a, b, r)[1] for r in (0.03, 0.045, 0.06, 0.075, 0.09)]
    assert all(x > y > 0.0 for x, y in zip(hoops, hoops[1:]))
    assert C.max_hoop_stress(p, a, a * 1e4) == pytest.approx(p, rel=1e-6)


def test_bore_criteria_match_lame_stress_state():
    p, a = 1e9, 0.02
    for k in (1.2, 2.0, 5.0):
        b = a * k
        sr, st, sz = C.lame_stresses(p, a, b, a, closed_ends=True)
        assert C.von_mises_bore(p, a, b) == pytest.approx(_von_mises(sr, st, sz), rel=1e-12)
        assert C.von_mises_bore(p, a, b, closed_ends=False) == pytest.approx(_von_mises(sr, st, 0.0), rel=1e-12)
        assert C.tresca_bore(p, a, b) == pytest.approx(max(sr, st, sz) - min(sr, st, sz), rel=1e-12)
        assert C.von_mises_bore(p, a, b, closed_ends=False) > C.von_mises_bore(p, a, b)
    # infinitely thick wall: sqrt(3) p (von Mises, either end condition), 2 p (Tresca)
    assert C.von_mises_bore(p, a, a * 1e4) == pytest.approx(S3 * p, rel=1e-6)
    assert C.von_mises_bore(p, a, a * 1e4, closed_ends=False) == pytest.approx(S3 * p, rel=1e-6)
    assert C.tresca_bore(p, a, a * 1e4) == pytest.approx(2.0 * p, rel=1e-6)


def test_minimum_monolithic_strength_for_4gpa():
    assert C.required_monolithic_strength(4e9) == pytest.approx(6.928e9, rel=1e-3)
    assert C.required_monolithic_strength(4e9, criterion="tresca") == pytest.approx(8.0e9, rel=1e-12)
    assert C.required_monolithic_strength(4e9, 1.5) == pytest.approx(10.39e9, rel=1e-3)
    assert C.required_monolithic_strength(4e9, 1.5, "tresca") == pytest.approx(12.0e9, rel=1e-12)
    assert C.required_monolithic_strength(4e9, criterion="hoop") == pytest.approx(4e9)


def test_required_radius_ratio_spec_check_and_impossibility():
    assert math.isinf(C.required_radius_ratio(4e9, 6.9e9))
    k = C.required_radius_ratio(4e9, 7e9)
    assert math.isfinite(k) and k == pytest.approx(9.874, rel=1e-3)
    assert math.isinf(C.required_radius_ratio(4e9, S3 * 4e9))  # exactly at the limit
    assert math.isinf(C.required_radius_ratio(4e9, 8e9, "tresca"))
    assert math.isfinite(C.required_radius_ratio(4e9, 8.1e9, "tresca"))
    assert math.isinf(C.required_radius_ratio(4e9, 4e9, "hoop"))
    assert math.isinf(C.required_radius_ratio(1e9, 0.0))  # never raises for impossibility
    assert C.required_radius_ratio(0.0, 1e9) == 1.0
    # monotone: stronger wall -> thinner; higher pressure -> thicker; k -> 1 for huge strength
    ks = [C.required_radius_ratio(1e9, s) for s in (2e9, 3e9, 5e9, 1e10)]
    assert all(x > y for x, y in zip(ks, ks[1:]))
    assert C.required_radius_ratio(1e9, 1e15) == pytest.approx(1.0, abs=1e-5)
    assert C.required_radius_ratio(1.1e9, 3e9) > C.required_radius_ratio(1e9, 3e9)


def test_required_radius_ratio_inverts_bore_stress_and_elastic_limit():
    p, a = 1e9, 0.02
    pairs = (("von_mises", C.von_mises_bore), ("tresca", C.tresca_bore), ("hoop", C.max_hoop_stress))
    for crit, bore_stress in pairs:
        for sigma in (2.5e9, 5e9, 9e9):
            k = C.required_radius_ratio(p, sigma, crit)
            assert bore_stress(p, a, a * k) == pytest.approx(sigma, rel=1e-9)
            assert C.elastic_limit_pressure(sigma, k, crit) == pytest.approx(p, rel=1e-9)


def test_elastic_limit_pressure_bounds_and_monotonicity():
    s = 1e9
    assert C.elastic_limit_pressure(s, 1.0) == 0.0
    assert C.elastic_limit_pressure(s, math.inf) == pytest.approx(s / S3, rel=1e-12)
    assert C.elastic_limit_pressure(s, math.inf, "tresca") == pytest.approx(s / 2.0, rel=1e-12)
    assert C.elastic_limit_pressure(s, math.inf, "hoop") == pytest.approx(s, rel=1e-12)
    vals = [C.elastic_limit_pressure(s, k) for k in (1.5, 2.0, 3.0, 5.0, 10.0)]
    assert all(x < y for x, y in zip(vals, vals[1:])) and vals[-1] < s / S3
    for k in (1.5, 3.0):  # Tresca is the more conservative criterion
        assert C.elastic_limit_pressure(s, k, "tresca") < C.elastic_limit_pressure(s, k)


# ------------------------------------------------- plastic / compound walls
def test_autofrettage_ratio_spec_check_and_inverse():
    k = C.autofrettage_ratio(4e9, 2e9)
    assert k == pytest.approx(5.66, rel=1e-2)
    assert k == pytest.approx(math.exp(S3), rel=1e-12)
    assert C.fully_plastic_pressure(2e9, k) == pytest.approx(4e9, rel=1e-12)
    k15 = C.autofrettage_ratio(4e9, 2e9, safety_factor=1.5)
    assert C.fully_plastic_pressure(2e9, k15) == pytest.approx(6e9, rel=1e-12)
    assert C.autofrettage_ratio(4e9, 2.39e9) == pytest.approx(4.2606, rel=1e-4)
    assert C.autofrettage_ratio(4e9, 2.39e9, 1.5) == pytest.approx(8.7945, rel=1e-4)
    # Tresca collapse p = sigma ln k is more conservative
    assert C.autofrettage_ratio(4e9, 2e9, criterion="tresca") == pytest.approx(math.exp(2.0), rel=1e-12)
    assert C.fully_plastic_pressure(2e9, math.e, "tresca") == pytest.approx(2e9, rel=1e-12)
    assert C.fully_plastic_pressure(2e9, 1.0) == 0.0
    assert math.isinf(C.autofrettage_ratio(4e9, 1e6))  # overflow -> inf, never raises


def test_plastic_wall_never_thicker_than_elastic_wall():
    for p in (0.2e9, 1e9, 2e9):
        for sigma in (2.0e9, 4.0e9, 1e10):
            k_el = C.required_radius_ratio(p, sigma)
            k_pl = C.autofrettage_ratio(p, sigma)
            assert k_pl <= k_el
            assert C.autofrettage_ratio(p, sigma, criterion="tresca") <= C.required_radius_ratio(p, sigma, "tresca")
    for k in (1.1, 2.0, 5.0, 20.0):
        assert C.fully_plastic_pressure(1e9, k) >= C.elastic_limit_pressure(1e9, k)


def test_shakedown_limits_a_single_autofrettaged_tube():
    s = 1e9
    kc = C.SHAKEDOWN_RATIO
    assert 2.0 * C.elastic_limit_pressure(s, kc) == pytest.approx(C.fully_plastic_pressure(s, kc), rel=1e-9)
    for k in (1.5, 2.0):
        assert C.shakedown_pressure(s, k) == pytest.approx(C.fully_plastic_pressure(s, k), rel=1e-12)
    for k in (3.0, 10.0, 1e6):
        sd = C.shakedown_pressure(s, k)
        assert sd == pytest.approx(2.0 * C.elastic_limit_pressure(s, k), rel=1e-12)
        assert sd < C.fully_plastic_pressure(s, k) and sd < 2.0 * s / S3
    # no single autofrettaged maraging-steel tube holds 4 GPa however thick
    assert C.shakedown_pressure(2.39e9, math.inf) == pytest.approx(2.0 * 2.39e9 / S3, rel=1e-12)
    assert C.shakedown_pressure(2.39e9, math.inf) < 4e9


# ---------------------------------------------------------- dynamic regime
def test_dynamic_regime_threshold():
    steel = materials.get("steel_4340")
    a = 0.025
    t = C.hoop_transit_time(a, steel)
    assert t == pytest.approx(2.0 * math.pi * a / steel.longitudinal_sound_speed(), rel=1e-12)
    assert 2e-5 < t < 3.5e-5  # ~27 us around a 25 mm radius steel bore
    assert C.dynamic_regime(a, steel, 3.01 * t) == "quasi_static"
    assert C.dynamic_regime(a, "steel_4340", 2.99 * t) == "inertial"
    assert C.dynamic_regime(a, steel, 1e-3) == "quasi_static"  # millisecond magnet pulse
    assert C.dynamic_regime(a, steel, 5e-6) == "inertial"  # microsecond single-turn-coil pulse
    # larger bore -> longer transit (linear in radius)
    assert C.hoop_transit_time(2 * a, steel) == pytest.approx(2.0 * t, rel=1e-12)


# --------------------------------------------------------------- load cases
def test_rail_launcher_load_spec_check():
    r = C.rail_launcher_load(4e6, 0.05, 0.05)
    assert r.force == pytest.approx(4.0e6, rel=1e-9)
    assert r.bore_pressure == pytest.approx(1.6e9, rel=1e-9)
    assert r.effective_field == pytest.approx(63.4, rel=1e-3)
    assert r.rail_repulsion_per_length == pytest.approx(8.0e7, rel=1e-9)
    assert r.rail_repulsion_per_length == pytest.approx(r.bore_pressure * r.bore_height, rel=1e-12)
    assert r.ideal_sheet_field == pytest.approx(100.53, rel=1e-3)
    assert r.current_status is Status.STATE_OF_THE_ART
    ideal = C.rail_launcher_load(4e6, 0.05, 0.05, inductance_gradient=MU0)
    assert ideal.bore_pressure == pytest.approx(4.02e9, rel=1e-3)
    assert ideal.effective_field == pytest.approx(ideal.ideal_sheet_field, rel=1e-12)
    # for any rectangular bore, L' = mu0 w/h makes the energy and sheet pictures agree
    lp = C.ideal_inductance_gradient(0.08, 0.04)
    assert lp == pytest.approx(MU0 * 0.5, rel=1e-12)
    rect = C.rail_launcher_load(3e6, 0.08, 0.04, inductance_gradient=lp)
    assert rect.effective_field == pytest.approx(rect.ideal_sheet_field, rel=1e-12)
    assert C.rail_launcher_load(8e6, 0.05, 0.05).force == pytest.approx(4.0 * r.force, rel=1e-12)
    assert any("outside the practical" in n for n in ideal.notes)


def test_coil_launcher_bore_pressure():
    assert C.coil_launcher_bore_pressure(100.27) == pytest.approx(4.0e9, rel=1e-3)
    assert C.coil_launcher_bore_pressure(63.41) == pytest.approx(1.6e9, rel=1e-3)
    assert C.coil_launcher_bore_pressure(-30.0) == C.coil_launcher_bore_pressure(30.0)
    assert C.coil_launcher_bore_pressure(0.0) == 0.0


def test_fusion_magnet_load_sparc_class():
    f = C.fusion_magnet_load(20.0, 12.2, plasma_beta=0.05)
    assert f.coil_pressure == pytest.approx(20.0 ** 2 / (2.0 * MU0), rel=1e-12)
    assert f.coil_pressure == pytest.approx(0.159e9, rel=1e-2)
    assert f.plasma_pressure == pytest.approx(0.05 * 12.2 ** 2 / (2.0 * MU0), rel=1e-12)
    assert f.ratio == pytest.approx(f.coil_pressure / f.plasma_pressure, rel=1e-12) and f.ratio > 10.0
    assert f.field_status is Status.STATE_OF_THE_ART
    assert f.coil_pressure < C.TOKAMAK_TF_STRUCTURAL_STRESS[0]
    zero_beta = C.fusion_magnet_load(20.0, 12.2, plasma_beta=0.0)
    assert math.isinf(zero_beta.ratio) and json.loads(zero_beta.json())["ratio"] is None
    assert C.fusion_magnet_load(100.0, 50.0).field_status is Status.ASPIRATIONAL
    with pytest.raises(InputError):
        C.fusion_magnet_load(20.0, 12.2, plasma_beta=1.5)


def test_mass_driver_lunar_aluminium_spec_check():
    v, length, r = 2400.0, 1.0, 0.05
    md = C.mass_driver(v, "aluminum_6061", length, r, bore_pressure=4e9)
    al = materials.get("aluminum_6061")
    assert md.limited_by == "payload_strength"
    assert md.launcher_length == pytest.approx(42.3, rel=0.02)
    assert md.launcher_length == pytest.approx(v ** 2 * al.density * length / (2 * al.yield_strength / 1.5), rel=1e-12)
    assert md.allowed_pressure == pytest.approx(al.yield_strength / 1.5, rel=1e-12)
    assert md.acceleration == pytest.approx(md.allowed_pressure / (al.density * length), rel=1e-12)
    assert md.launch_time == pytest.approx(2.0 * md.launcher_length / v, rel=1e-12)
    assert md.payload_mass == pytest.approx(al.density * math.pi * r * r * length, rel=1e-12)
    assert md.kinetic_energy == pytest.approx(0.5 * md.payload_mass * v * v, rel=1e-12)
    # work done by the base pressure over the launcher equals the kinetic energy
    assert md.allowed_pressure * math.pi * r * r * md.launcher_length == pytest.approx(md.kinetic_energy, rel=1e-12)
    assert md.bore_field == pytest.approx(math.sqrt(2.0 * MU0 * md.allowed_pressure), rel=1e-12)
    assert md.g_load == pytest.approx(md.acceleration / G_ACCEL, rel=1e-12)
    assert md.payload_material == "aluminum_6061" and md.bore_pressure == 4e9
    # the 4 GPa bore capability would crush the payload: > 20x its allowable base pressure
    assert md.bore_pressure / md.allowed_pressure > 20.0


def test_mass_driver_binding_constraints_and_scaling():
    free = C.mass_driver(2400.0, "al", 1.0, 0.05)
    assert free.limited_by == "payload_strength" and free.bore_pressure is None
    low = C.mass_driver(2400.0, "al", 1.0, 0.05, bore_pressure=50e6)
    assert low.limited_by == "bore_pressure" and low.allowed_pressure == 50e6
    assert low.launcher_length > free.launcher_length
    capped = C.mass_driver(2400.0, "al", 1.0, 0.05, acceleration_limit=1e4)
    assert capped.limited_by == "acceleration_limit" and capped.acceleration == 1e4
    assert capped.allowed_pressure == pytest.approx(2700.0 * 1.0 * 1e4, rel=1e-12)
    orbital = C.mass_driver(7800.0, "al", 1.0, 0.05)
    assert orbital.launcher_length / free.launcher_length == pytest.approx((7800.0 / 2400.0) ** 2, rel=1e-12)
    assert C.mass_driver(2400.0, "al", 2.0, 0.05).launcher_length == pytest.approx(2.0 * free.launcher_length)
    assert C.mass_driver(2400.0, "ti6al4v", 1.0, 0.05).launcher_length < free.launcher_length
    # base-loaded payloads use the compressive yield when it is lower (Mg AZ31 twins in compression)
    assert C.mass_driver(2400.0, "magnesium_az31", 1.0, 0.05).payload_strength == pytest.approx(140e6)
    # fibres do not yield: the ultimate strength is used
    assert C.mass_driver(2400.0, "carbon_fibre_t1100g", 1.0, 0.05).payload_strength == pytest.approx(7.0e9)


# -------------------------------------------------------------- assessments
def test_containment_survey_statuses_at_4gpa_25mm():
    survey = C.containment_survey(4e9, 0.025)
    assert tuple(s.material for s in survey) == C.DEFAULT_CANDIDATES
    assert {s.material: s.status for s in survey} == {
        "steel_4340": Status.INFEASIBLE,
        "maraging_c350": Status.EXTRAPOLATION,
        "mp35n": Status.INFEASIBLE,
        "carbon_fibre_t1100g": Status.EXTRAPOLATION,
        "cnt_fibre_record": Status.ASPIRATIONAL,
        "cns_engineering_target": Status.ASPIRATIONAL,
        "graphene_intrinsic": Status.ASPIRATIONAL,
    }
    by = {s.material: s for s in survey}
    for s in survey:
        assert isinstance(s, Assessment)
        assert s.required_monolithic_strength == pytest.approx(10.39e9, rel=1e-3)
        assert s.radial_stress_at_bore == -4e9
        assert s.dynamic_regime is None
        assert (s.margin >= 1.0) == (s.status is not Status.INFEASIBLE)
        if s.material != "graphene_intrinsic":  # even the 10 GPa CNS target fails monolithic SF 1.5
            assert math.isinf(s.elastic_ratio) and math.isinf(s.outer_radius_elastic)
    assert by["maraging_c350"].plastic_ratio == pytest.approx(8.79, rel=1e-3)
    assert by["maraging_c350"].route == "compound"
    assert by["graphene_intrinsic"].elastic_ratio == pytest.approx(1.043, rel=1e-3)
    assert by["steel_4340"].route == "none" and by["steel_4340"].mass_per_length > 1e6
    assert by["carbon_fibre_t1100g"].transverse_compression_ok is False
    assert by["cns_engineering_target"].transverse_compression_ok is False
    assert by["steel_4340"].transverse_compression_ok is None
    assert by["graphene_intrinsic"].transverse_compression_ok is None
    assert any("liner" in w for w in by["carbon_fibre_t1100g"].warnings)
    assert any("shakes down" in n for n in by["maraging_c350"].notes)
    assert any("LANL 100.75 T" in n for n in by["maraging_c350"].notes)  # 2.5 < p <= 4.04 GPa


def test_assess_containment_low_pressure_and_compound_practice():
    fusion = C.fusion_magnet_load(20.0, 12.2)
    tf = C.assess_containment(fusion.coil_pressure, "steel_4340", 0.5)
    assert tf.status is Status.STATE_OF_THE_ART and tf.route == "elastic_monolithic"
    assert tf.elastic_ratio == pytest.approx(1.4465, rel=1e-4)
    assert not any("LANL" in n for n in tf.notes)  # comparison note only above die practice
    rail = C.rail_launcher_load(4e6, 0.05, 0.05)
    mar = C.assess_containment(rail.bore_pressure, "maraging_c350", 0.025)
    assert mar.route == "compound" and mar.status is Status.STATE_OF_THE_ART  # within cold-forging-die practice
    assert mar.plastic_ratio == pytest.approx(2.386, rel=1e-3)
    assert C.assess_containment(rail.bore_pressure, "steel_4340", 0.025).status is Status.INFEASIBLE
    cf = C.assess_containment(0.2e9, "carbon_fibre_t1100g", 0.05)
    assert cf.route == "elastic_monolithic" and cf.status is Status.EXTRAPOLATION  # not a commercial metal
    assert cf.transverse_compression_ok is True
    assert any("not with SF" in n for n in cf.notes)
    tresca = C.assess_containment(0.2e9, "steel_4340", 0.05, criterion="tresca")
    assert tresca.elastic_ratio > C.assess_containment(0.2e9, "steel_4340", 0.05).elastic_ratio
    hoop = C.assess_containment(0.2e9, "steel_4340", 0.05, criterion="hoop")
    assert any("non-conservative" in w for w in hoop.warnings)
    assert C.COMPOUND_DEMONSTRATED_PRESSURE == next(b.value for b in PRESSURE_BENCHMARKS if b.key == "cold_forging_die")


def test_assess_containment_inertial_pulse_downgrades_status():
    steel = materials.get("steel_4340")
    t = C.hoop_transit_time(0.5, steel)
    slow = C.assess_containment(0.16e9, steel, 0.5, pulse_duration=10.0 * t)
    assert slow.status is Status.STATE_OF_THE_ART and slow.dynamic_regime == "quasi_static"
    fast = C.assess_containment(0.16e9, steel, 0.5, pulse_duration=t)
    assert fast.status is Status.EXTRAPOLATION and fast.dynamic_regime == "inertial"
    assert any("inertial" in w for w in fast.warnings)
    assert fast.pulse_duration == t and fast.hoop_transit_time == pytest.approx(t, rel=1e-12)
    # an already infeasible design stays infeasible
    assert C.assess_containment(4e9, steel, 0.5, pulse_duration=t).status is Status.INFEASIBLE


def test_input_validation():
    bad_calls = [
        lambda: C.lame_stresses(-1.0, 0.02, 0.04, 0.03),
        lambda: C.lame_stresses(1e9, 0.04, 0.02, 0.03),
        lambda: C.lame_stresses(1e9, 0.02, 0.04, 0.05),
        lambda: C.lame_stresses(True, 0.02, 0.04, 0.03),
        lambda: C.max_hoop_stress(1e9, 0.02, 0.02),
        lambda: C.von_mises_bore(1e9, 0.0, 0.02),
        lambda: C.required_radius_ratio(-1e9, 1e9),
        lambda: C.required_radius_ratio(1e9, -1e9),
        lambda: C.required_radius_ratio(1e9, 2e9, "rankine"),
        lambda: C.elastic_limit_pressure(1e9, 0.5),
        lambda: C.fully_plastic_pressure(1e9, 2.0, "hoop"),
        lambda: C.autofrettage_ratio(4e9, 0.0),
        lambda: C.autofrettage_ratio(4e9, 2e9, safety_factor=0.5),
        lambda: C.dynamic_regime(0.025, "steel_4340", 0.0),
        lambda: C.rail_launcher_load(0.0, 0.05, 0.05),
        lambda: C.rail_launcher_load(4e6, 0.05, 0.05, inductance_gradient=-1e-7),
        lambda: C.coil_launcher_bore_pressure(math.inf),
        lambda: C.mass_driver(2400.0, "al", -1.0, 0.05),
        lambda: C.mass_driver(2400.0, "al", 1.0, 0.05, bore_pressure=0.0),
        lambda: C.mass_driver(2400.0, "diamond", 1.0, 0.05),
        lambda: C.assess_containment(0.0, "steel_4340", 0.025),
        lambda: C.assess_containment(math.nan, "steel_4340", 0.025),
        lambda: C.assess_containment(4e9, "diamond", 0.025),  # no tensile strength data
        lambda: C.assess_containment(4e9, "steel_4340", 0.025, pulse_duration=-1.0),
        lambda: C.containment_survey(4e9, 0.025, candidates=()),
    ]
    for call in bad_calls:
        with pytest.raises(InputError):
            call()
    with pytest.raises(UnknownMaterialError):
        C.assess_containment(4e9, "unobtainium", 0.025)
    single = C.containment_survey(4e9, 0.025, candidates="maraging_c350")
    assert len(single) == 1 and single[0].material == "maraging_c350"


def test_every_result_type_round_trips_json_and_renders_text():
    results: list[ResultMixin] = [
        C.rail_launcher_load(4e6, 0.05, 0.05),
        C.fusion_magnet_load(20.0, 12.2),
        C.mass_driver(2400.0, "aluminum_6061", 1.0, 0.05, bore_pressure=4e9),
        *C.containment_survey(4e9, 0.025, pulse_duration=1e-3),
    ]
    for res in results:
        assert dataclasses.is_dataclass(res)
        data = json.loads(res.json())
        assert set(data) == {f.name for f in dataclasses.fields(res)}
        assert all(isinstance(n, str) for n in data["notes"])
        txt = res.text()
        assert txt.splitlines()[0] == type(res).__name__
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.notes = ()  # type: ignore[misc]
    survey_json = [json.loads(s.json()) for s in results[3:]]
    steel = survey_json[0]
    assert steel["status"] == "infeasible" and steel["elastic_ratio"] is None  # inf -> null
    assert steel["dynamic_regime"] == "quasi_static" and steel["transverse_compression_ok"] is None
    assert json.loads(results[0].json())["current_status"] == "state_of_the_art"
    assert json.loads(results[2].json())["limited_by"] == "payload_strength"
    assert "elastic_ratio = inf" in results[3].text()
    assert "bore_pressure [Pa] = 1.6e+09" in results[0].text()


def test_module_metadata():
    assert isinstance(C.REFERENCES, dict) and len(C.REFERENCES) >= 10
    assert all(isinstance(k, str) and isinstance(v, str) and len(v) > 20 for k, v in C.REFERENCES.items())
    for name in C.__all__:
        obj = getattr(C, name)
        if callable(obj):
            assert obj.__doc__ and "Reference" in obj.__doc__, name
    assert C.CRITERIA == ("von_mises", "tresca", "hoop")
    assert C.PULSED_MAGNET_CONTAINED_PRESSURE == pytest.approx(4.04e9, rel=1e-3)
    assert C.FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH == 0.25e9
