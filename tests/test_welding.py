"""Tests for magform.welding: the impact-welding window and what 1.6-4 GPa does to it."""

from __future__ import annotations

import json
import math

import pytest

from magform import eddy, materials, shock
from magform import magnetics as mg
from magform import welding as W
from magform.benchmarks import EMF_CONVENTIONAL_FIELD
from magform.constants import MU0, R_GAS, T_ROOM
from magform.errors import InputError, UnknownMaterialError
from magform.status import Status
from magform.welding import DriveWindow, FlyerImpact, StandoffWindow, WeldAssessment, WeldPair, WeldWindow

AL, STEEL, CU, TI = "aluminum_6061", "steel_1018", "copper_ofhc", "titanium_cp2"
H, GAP = 1e-3, 2e-3               # headline joint: 1 mm flyer, 2 mm standoff
BETA = math.radians(10.0)


def _flow_speed(key: str) -> float:
    m = materials.get(key)
    return math.sqrt(m.hardness_estimate() / m.density)


# ------------------------------------------------------------------ constants / metadata
def test_constants_and_module_metadata():
    assert W.DERIBAS_K_CLEAN == 0.6 and W.DERIBAS_K_TECHNICAL == 1.2
    assert W.COWAN_REYNOLDS_TRANSITION == 10.6 and W.WITTMAN_N == 0.11
    assert W.DEFAULT_COLLISION_ANGLE == pytest.approx(BETA)
    assert W.MPW_TYPICAL_ANGLE[0] < W.DEFAULT_COLLISION_ANGLE < W.MPW_TYPICAL_ANGLE[1]
    assert W.README_PRESSURE_RANGE == (1.6e9, 4.0e9)
    assert W.PULSE_SHAPES == ("constant", "half_sine")
    assert 0.0 < W.INTERFACE_ENERGY_FRACTION < 1.0 and 0.0 < W.DRIVER_TRANSFER_THRESHOLD < 1.0
    assert set(W.__all__) <= set(dir(W))
    assert len(W.REFERENCES) >= 15
    assert all(isinstance(k, str) and isinstance(v, str) and len(v) > 20 for k, v in W.REFERENCES.items())
    for name in W.__all__:
        obj = getattr(W, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 80 and "Reference" in obj.__doc__, name


# ------------------------------------------------------------------ demonstrated pairs
def test_demonstrated_pairs_table():
    assert {"Al-Al", "Al-Cu", "Al-Fe", "Cu-Fe", "Al-Ti", "Cu-Ti", "Al-Mg"} <= set(W.DEMONSTRATED_PAIRS)
    for key, pair in W.DEMONSTRATED_PAIRS.items():
        assert isinstance(pair, WeldPair) and pair.key == key
        assert list(pair.elements) == sorted(pair.elements)
        assert pair.processes and set(pair.processes) <= {"magnetic_pulse", "explosive"}
        assert len(pair.citation) > 20
    # the concept note's two examples are real impact-welded pairs, and both form intermetallics
    assert "FeAl" in W.DEMONSTRATED_PAIRS["Al-Fe"].intermetallics
    assert "CuTi" in W.DEMONSTRATED_PAIRS["Cu-Ti"].intermetallics
    assert W.DEMONSTRATED_PAIRS["Cu-Fe"].intermetallics == ()        # immiscible: no compounds
    assert W.DEMONSTRATED_PAIRS["Al-Al"].intermetallics == ()


def test_pair_lookup_is_by_base_element_and_order_independent():
    assert W.pair_key(STEEL, AL) == W.pair_key(AL, STEEL) == "Al-Fe"
    assert W.pair_key("steel_304", "aluminum_1100") == "Al-Fe"
    assert W.pair_key(TI, CU) == "Cu-Ti" and W.pair_key("ti64", "cu") == "Cu-Ti"
    assert W.demonstrated_pair(TI, CU) is W.DEMONSTRATED_PAIRS["Cu-Ti"]
    assert W.demonstrated_pair(materials.get(AL), STEEL).key == "Al-Fe"
    assert W.demonstrated_pair(CU, "tantalum") is None               # not tabulated
    assert W.demonstrated_pair(AL, "gamma_tial") is None             # intermetallic, not a ductile Ti alloy
    with pytest.raises(UnknownMaterialError):
        W.pair_key("unobtainium", AL)


# ------------------------------------------------------------------ collision kinematics
def test_collision_kinematics_and_inverse():
    assert W.collision_point_velocity(400.0, BETA) == pytest.approx(400.0 / math.sin(BETA), rel=1e-14)
    assert W.collision_point_velocity(400.0, BETA) == pytest.approx(2303.5, rel=1e-4)
    for vp, ang in ((250.0, 0.1), (400.0, BETA), (900.0, 0.5)):
        vc = W.collision_point_velocity(vp, ang)
        assert W.impact_velocity_for_collision(vc, ang) == pytest.approx(vp, rel=1e-13)
    assert W.collision_point_velocity(0.0, BETA) == 0.0
    # collision point outruns everything as the impact becomes flat-on
    assert W.collision_point_velocity(400.0, 1e-6) > 1e8
    # the three geometric conventions agree within 7 % up to 20 degrees
    for deg in (5.0, 10.0, 20.0):
        b = math.radians(deg)
        conv = (1.0 / math.sin(b), 1.0 / math.tan(b), 1.0 / (2.0 * math.sin(0.5 * b)))
        assert max(conv) / min(conv) < 1.07


def test_sonic_limit_and_critical_angle():
    al, st, cu, ti = (materials.get(k) for k in (AL, STEEL, CU, TI))
    assert W.jetting_sonic_limit(AL, STEEL) == pytest.approx(min(al.bulk_sound_speed(), st.bulk_sound_speed()))
    assert W.jetting_sonic_limit(AL, STEEL) == pytest.approx(4547.0, rel=1e-3)
    assert W.jetting_sonic_limit(CU, TI) == pytest.approx(cu.bulk_sound_speed())      # copper is the slower one
    assert W.jetting_sonic_limit(CU, TI) == pytest.approx(3688.0, rel=1e-3)
    assert W.jetting_sonic_limit(TI, CU) == W.jetting_sonic_limit(CU, TI)
    # below the critical angle the collision point is supersonic: flat-on impacts never weld
    crit = W.critical_collision_angle(AL, STEEL, 400.0)
    assert math.degrees(crit) == pytest.approx(5.047, rel=1e-3)
    assert W.collision_point_velocity(400.0, crit) == pytest.approx(W.jetting_sonic_limit(AL, STEEL), rel=1e-12)
    assert W.collision_point_velocity(400.0, 0.5 * crit) > W.jetting_sonic_limit(AL, STEEL)
    assert W.critical_collision_angle(AL, STEEL, 0.0) == 0.0
    assert W.critical_collision_angle(AL, STEEL, 5000.0) is None     # faster than sound: no angle works


# ------------------------------------------------------------------ window limits
def test_lower_limit_deribas_closed_form():
    # k sqrt(HV/rho) of the member that is harder to make flow
    assert W.lower_limit_velocity(AL, STEEL) == pytest.approx(0.6 * max(_flow_speed(AL), _flow_speed(STEEL)))
    assert W.lower_limit_velocity(AL, STEEL) == pytest.approx(374.17, rel=1e-4)                # Al 6061-T6 governs
    assert W.lower_limit_velocity("aluminum_1100", STEEL) == pytest.approx(239.12, rel=1e-4)   # steel governs
    assert W.lower_limit_velocity(CU, TI) == pytest.approx(0.6 * _flow_speed(TI))            # titanium governs
    assert W.lower_limit_velocity(CU, TI) == pytest.approx(346.03, rel=1e-4)
    assert W.lower_limit_velocity(STEEL, AL) == W.lower_limit_velocity(AL, STEEL)
    assert W.lower_limit_velocity(AL, STEEL, W.DERIBAS_K_TECHNICAL) == pytest.approx(
        2.0 * W.lower_limit_velocity(AL, STEEL))
    # inside the 200-500 m/s typical of magnetic pulse welding
    lo, hi = W.MPW_TYPICAL_VELOCITY
    for pair in ((AL, STEEL), ("aluminum_1100", STEEL), (CU, TI), (CU, STEEL)):
        assert lo <= W.lower_limit_velocity(*pair) <= hi


def test_cowan_reynolds_number_and_wavy_transition():
    al, st = materials.get(AL), materials.get(STEEL)
    vc_t = W.wavy_transition_velocity(AL, STEEL)
    expected = math.sqrt(2 * 10.6 * (al.hardness_estimate() + st.hardness_estimate()) / (al.density + st.density))
    assert vc_t == pytest.approx(expected, rel=1e-14)
    assert vc_t == pytest.approx(2147.8, rel=1e-4)
    assert W.wavy_transition_velocity(CU, TI) == pytest.approx(1747.4, rel=1e-4)
    assert W.collision_reynolds_number(AL, STEEL, vc_t) == pytest.approx(10.6, rel=1e-12)
    assert W.collision_reynolds_number(AL, STEEL, 2.0 * vc_t) == pytest.approx(4 * 10.6, rel=1e-12)
    assert W.collision_reynolds_number(AL, STEEL, 0.0) == 0.0
    assert W.wavy_transition_velocity(STEEL, AL) == vc_t
    assert W.wavy_transition_velocity(AL, STEEL, reynolds=42.4) == pytest.approx(2.0 * vc_t)


def test_upper_limit_wittman_closed_form_and_scaling():
    al = materials.get(AL)                                   # the lower-melting member of Al/steel
    c0 = al.bulk_sound_speed()
    group = math.sqrt(al.melting_point * c0) * (al.thermal_conductivity * al.specific_heat * c0
                                                 / (al.density * H)) ** 0.25 / 0.11
    assert W.upper_limit_velocity(AL, STEEL, H, 2500.0) == pytest.approx(group / 2500.0, rel=1e-13)
    assert W.upper_limit_velocity(AL, STEEL, H, 2500.0) == pytest.approx(970.3, rel=1e-4)
    # evaluated with the lower-melting member whichever plate flies
    assert W.upper_limit_velocity(STEEL, AL, H, 2500.0) == pytest.approx(group / 2500.0, rel=1e-13)
    # hyperbola in Vc; h^(-1/4); 1/N
    assert W.upper_limit_velocity(AL, STEEL, H, 5000.0) == pytest.approx(0.5 * group / 2500.0)
    assert W.upper_limit_velocity(AL, STEEL, 16 * H, 2500.0) == pytest.approx(0.5 * group / 2500.0)
    assert W.upper_limit_velocity(AL, STEEL, H, 2500.0, n=0.055) == pytest.approx(2.0 * group / 2500.0)


def test_weld_window_al_on_steel():
    w = W.weld_window(AL, STEEL, H)
    assert isinstance(w, WeldWindow)
    assert (w.flyer, w.target, w.flyer_thickness) == (AL, STEEL, H)
    assert w.collision_angle == pytest.approx(BETA)
    assert w.vp_min == pytest.approx(374.17, rel=1e-4) and w.lower_governed_by == AL
    assert w.vp_melt == pytest.approx(649.02, rel=1e-4) and w.melting_member == AL
    assert w.vp_sonic == pytest.approx(w.vc_sonic * math.sin(BETA))
    assert w.vp_sonic == pytest.approx(789.54, rel=1e-4)
    assert w.vp_max == w.vp_melt and w.upper_limited_by == "melting" and w.exists
    assert w.vp_wavy == pytest.approx(w.vc_wavy * math.sin(BETA))
    assert w.width == pytest.approx(w.vp_max / w.vp_min)
    assert w.width == pytest.approx(1.7346, rel=1e-4)
    # vp_melt is Wittman's limit made self-consistent with Vc = Vp / sin(beta)
    vc = W.collision_point_velocity(w.vp_melt, BETA)
    assert W.upper_limit_velocity(AL, STEEL, H, vc) == pytest.approx(w.vp_melt, rel=1e-12)
    assert w.contains(400.0) and not w.contains(300.0) and not w.contains(700.0)
    assert w.contains(w.vp_min) and w.contains(w.vp_max)


def test_weld_window_other_pairs_and_trends():
    cu_ti = W.weld_window(CU, TI, H)
    assert cu_ti.vp_min == pytest.approx(346.03, rel=1e-4) and cu_ti.lower_governed_by == TI
    assert cu_ti.vp_max == pytest.approx(559.66, rel=1e-4) and cu_ti.melting_member == CU
    assert cu_ti.exists and cu_ti.upper_limited_by == "melting"
    # Al on Cu at 10 degrees is closed by the jetting limit (slow sound in copper), not by melting
    al_cu = W.weld_window(AL, CU, H)
    assert al_cu.upper_limited_by == "sonic" and al_cu.vp_max == al_cu.vp_sonic < al_cu.vp_melt
    # small angles: the sonic limit closes the window altogether
    assert W.weld_window(AL, STEEL, H, math.radians(5.0)).upper_limited_by == "sonic"
    assert not W.weld_window(AL, STEEL, H, math.radians(3.0)).exists
    assert not W.weld_window(AL, STEEL, H, math.radians(4.6)).exists
    assert W.weld_window(AL, STEEL, H, math.radians(4.8)).exists
    # the window widens with angle and narrows slowly with flyer thickness (h^-1/8) and with N
    widths = [W.weld_window(AL, STEEL, H, math.radians(d)).vp_max for d in (5, 10, 15, 20)]
    assert all(b > a for a, b in zip(widths, widths[1:]))
    assert W.weld_window(AL, STEEL, 2 * H).vp_melt == pytest.approx(2 ** -0.125 * W.weld_window(AL, STEEL, H).vp_melt)
    assert W.weld_window(AL, STEEL, H, wittman_n=0.055).vp_melt == pytest.approx(
        math.sqrt(2.0) * W.weld_window(AL, STEEL, H).vp_melt)
    technical = W.weld_window(AL, STEEL, H, lower_coefficient=W.DERIBAS_K_TECHNICAL)
    assert technical.vp_min == pytest.approx(748.33, rel=1e-4) and not technical.exists   # dirty surfaces: no window
    # limits that do not depend on which plate flies
    rev = W.weld_window(STEEL, AL, H)
    assert (rev.vp_min, rev.vp_melt, rev.vc_sonic, rev.vc_wavy) == pytest.approx(
        (W.weld_window(AL, STEEL, H).vp_min, W.weld_window(AL, STEEL, H).vp_melt,
         W.weld_window(AL, STEEL, H).vc_sonic, W.weld_window(AL, STEEL, H).vc_wavy))


# ------------------------------------------------------------------ flyer flight
def test_flyer_impact_constant_pressure_matches_shock_module():
    f = W.flyer_impact(0.1e9, GAP, AL, H)
    assert isinstance(f, FlyerImpact) and f.limited_by == "gap" and f.pulse_duration is None
    assert f.impact_velocity == shock.flyer_velocity(0.1e9, GAP, AL, H) == pytest.approx(384.90, rel=1e-5)
    assert f.transit_time == shock.flyer_transit_time(GAP, 0.1e9, AL, H) == pytest.approx(2 * GAP / 384.90, rel=1e-5)
    assert f.pressure_at_impact == f.peak_pressure == 0.1e9
    assert f.constant_pressure_velocity == f.impact_velocity
    eta = W.flyer_impact(0.1e9, GAP, AL, H, efficiency=0.25)
    assert eta.impact_velocity == pytest.approx(0.5 * f.impact_velocity)
    zero = W.flyer_impact(0.0, GAP, AL, H)
    assert zero.impact_velocity == 0.0 and math.isinf(zero.transit_time) and zero.limited_by == "no_drive"
    json.loads(zero.json())                                   # inf -> null keeps the JSON strict


def test_flyer_impact_rectangular_pulse_is_impulse_limited():
    rho_h = materials.get(AL).density * H
    full = W.flyer_impact(0.1e9, GAP, AL, H)
    # a pulse longer than the flight changes nothing
    long = W.flyer_impact(0.1e9, GAP, AL, H, pulse_duration=2.0 * full.transit_time)
    assert long.impact_velocity == full.impact_velocity and long.limited_by == "gap"
    # a shorter one delivers only its impulse: v = P t / (rho h), then the flyer coasts
    tp = 0.5 * full.transit_time
    short = W.flyer_impact(0.1e9, GAP, AL, H, pulse_duration=tp)
    assert short.limited_by == "pulse" and short.pressure_at_impact == 0.0
    assert short.impact_velocity == pytest.approx(0.1e9 * tp / rho_h, rel=1e-13)
    assert short.impact_velocity == pytest.approx(0.5 * full.impact_velocity)
    a = 0.1e9 / rho_h
    assert short.transit_time == pytest.approx(tp + (GAP - 0.5 * a * tp * tp) / (a * tp), rel=1e-13)
    assert short.transit_time > full.transit_time
    # continuity at the hand-over
    edge = W.flyer_impact(0.1e9, GAP, AL, H, pulse_duration=full.transit_time * (1 - 1e-9))
    assert edge.impact_velocity == pytest.approx(full.impact_velocity, rel=1e-8)


def test_flyer_impact_half_sine_closed_forms():
    rho_h = materials.get(AL).density * H
    a = 4e9 / rho_h
    # (1) slow pulse: the pressure is still rising as t^2 when the gap closes -> x = a pi^2 t^4 / (12 T^2)
    T = 5e-3
    slow = W.flyer_impact(4e9, GAP, AL, H, pulse_duration=T, pulse_shape="half_sine")
    t_asym = (12 * GAP * T * T / (a * math.pi ** 2)) ** 0.25
    assert slow.transit_time == pytest.approx(t_asym, rel=2e-3)
    assert slow.transit_time == pytest.approx(80.04e-6, rel=1e-3)
    assert slow.impact_velocity == pytest.approx(a * math.pi ** 2 * t_asym ** 3 / (3 * T * T), rel=2e-3)
    assert slow.impact_velocity == pytest.approx(99.93, rel=1e-3)            # 4 GPa peak, yet only 100 m/s
    assert slow.pressure_at_impact == pytest.approx(4e9 * math.sin(math.pi * slow.transit_time / T) ** 2, rel=1e-9)
    assert slow.pressure_at_impact == pytest.approx(10.1e6, rel=1e-2)        # 0.25 % of the peak
    assert slow.limited_by == "gap" and slow.constant_pressure_velocity == pytest.approx(2434.3, rel=1e-4)
    # (2) exact kinematics at the impact time, for a pulse comparable to the flight
    T = 3e-5
    mid = W.flyer_impact(4e9, GAP, AL, H, pulse_duration=T, pulse_shape="half_sine")
    t = mid.transit_time
    x = a * (t * t / 4 - T * T / (8 * math.pi ** 2) * (1 - math.cos(2 * math.pi * t / T)))
    v = a * (t / 2 - T / (4 * math.pi) * math.sin(2 * math.pi * t / T))
    assert x == pytest.approx(GAP, rel=1e-9) and mid.impact_velocity == pytest.approx(v, rel=1e-9)
    assert mid.impact_velocity < mid.constant_pressure_velocity
    # (3) short pulse: the flyer leaves with the whole impulse, v = a T / 2 (mean of sin^2 is 1/2)
    T = 2e-6
    short = W.flyer_impact(4e9, GAP, AL, H, pulse_duration=T, pulse_shape="half_sine")
    assert short.limited_by == "pulse" and short.impact_velocity == pytest.approx(0.5 * a * T, rel=1e-13)
    assert short.transit_time == pytest.approx(T + (GAP - a * T * T / 4) / (0.5 * a * T), rel=1e-13)
    # (4) the series branch for very slow pulses stays accurate (no cancellation)
    T = 1.0
    glacial = W.flyer_impact(4e9, GAP, AL, H, pulse_duration=T, pulse_shape="half_sine")
    t_asym = (12 * GAP * T * T / (a * math.pi ** 2)) ** 0.25
    assert glacial.transit_time == pytest.approx(t_asym, rel=1e-5)
    assert glacial.impact_velocity == pytest.approx(4 * GAP / t_asym, rel=1e-5)   # x ~ t^4 -> v = 4 x / t
    # impact velocity falls monotonically as the pulse gets slower than the flight
    v_of_t = [W.flyer_impact(4e9, GAP, AL, H, pulse_duration=T, pulse_shape="half_sine").impact_velocity
              for T in (1e-5, 3e-5, 1e-4, 1e-3, 5e-3)]
    assert all(b < a_ for a_, b in zip(v_of_t, v_of_t[1:]))
    with pytest.raises(InputError):
        W.flyer_impact(4e9, GAP, AL, H, pulse_shape="half_sine")              # needs the half-period


# ------------------------------------------------------------------ drive and standoff windows
def test_drive_pressure_window_is_tens_of_tesla_not_the_readme_range():
    d = W.drive_pressure_window(AL, STEEL, H, GAP)
    assert isinstance(d, DriveWindow) and d.exists and isinstance(d.window, WeldWindow)
    rho_h = materials.get(AL).density * H
    assert d.pressure_min == pytest.approx(rho_h * d.velocity_min ** 2 / (2 * GAP), rel=1e-13)
    assert d.pressure_min == pytest.approx(0.0945e9, rel=1e-3) and d.pressure_max == pytest.approx(0.2843e9, rel=1e-3)
    assert d.field_min == pytest.approx(math.sqrt(2 * MU0 * d.pressure_min))
    assert d.field_min == pytest.approx(15.41, rel=1e-3)
    assert d.field_max == pytest.approx(26.73, rel=1e-3)
    assert d.flight_time_max == pytest.approx(10.69e-6, rel=1e-3) and d.flight_time_min == pytest.approx(6.16e-6, rel=1e-3)
    # the constant-pressure flyer model lands exactly on the window edges
    assert shock.flyer_velocity(d.pressure_min, GAP, AL, H) == pytest.approx(d.window.vp_min, rel=1e-12)
    assert shock.flyer_velocity(d.pressure_max, GAP, AL, H) == pytest.approx(d.window.vp_max, rel=1e-12)
    soft = W.drive_pressure_window("aluminum_1100", STEEL, H, GAP)
    assert soft.pressure_min == pytest.approx(0.0387e9, rel=2e-3) and soft.field_min == pytest.approx(9.87, rel=1e-3)
    cu_ti = W.drive_pressure_window(CU, TI, H, GAP)
    assert cu_ti.pressure_min == pytest.approx(0.268e9, rel=2e-3) and cu_ti.pressure_max == pytest.approx(0.702e9, rel=2e-3)
    assert cu_ti.field_min == pytest.approx(25.96, rel=1e-3) and cu_ti.field_max == pytest.approx(41.99, rel=1e-3)
    # every window sits inside industrial electromagnetic forming and below the concept note's 1.6 GPa
    for win in (d, soft, cu_ti):
        assert win.field_max < EMF_CONVENTIONAL_FIELD[1] and win.pressure_max < W.README_PRESSURE_RANGE[0]
    # scalings: P ~ 1/gap, P ~ 1/efficiency
    assert W.drive_pressure_window(AL, STEEL, H, 2 * GAP).pressure_min == pytest.approx(0.5 * d.pressure_min)
    assert W.drive_pressure_window(AL, STEEL, H, GAP, efficiency=0.5).pressure_max == pytest.approx(2 * d.pressure_max)


def test_standoff_window_at_readme_pressures_is_sub_millimetre():
    lo = W.standoff_window(AL, STEEL, H, 1.6e9)
    hi = W.standoff_window(AL, STEEL, H, 4e9)
    assert isinstance(lo, StandoffWindow) and lo.exists
    assert lo.drive_field == pytest.approx(63.41, rel=1e-4) and hi.drive_field == pytest.approx(100.27, rel=1e-4)
    assert (lo.gap_min, lo.gap_max) == pytest.approx((118.1e-6, 355.4e-6), rel=1e-3)
    assert (hi.gap_min, hi.gap_max) == pytest.approx((47.25e-6, 142.2e-6), rel=1e-3)
    assert (hi.flight_time_min, hi.flight_time_max) == pytest.approx((0.2526e-6, 0.4381e-6), rel=1e-3)
    assert lo.flight_time_max == pytest.approx(1.095e-6, rel=1e-3)
    assert hi.flight_time_max < W.FASTEST_HIGH_FIELD_RISE            # shorter than any 100 T coil's rise
    # inverse of the drive-pressure window, and consistent with the flyer model
    d = W.drive_pressure_window(AL, STEEL, H, GAP)
    back = W.standoff_window(AL, STEEL, H, d.pressure_min)
    assert back.gap_min == pytest.approx(GAP, rel=1e-12)
    assert shock.flyer_velocity(4e9, hi.gap_max, AL, H) == pytest.approx(hi.velocity_max, rel=1e-12)
    assert hi.flight_time_min == pytest.approx(shock.flyer_transit_time(hi.gap_min, 4e9, AL, H), rel=1e-12)


# ------------------------------------------------------------------ interface heat and diffusion
def test_interface_energy_and_melt_layer_estimates():
    al, st = materials.get(AL), materials.get(STEEL)
    e = W.interface_energy(AL, H, 400.0)
    assert e == pytest.approx(0.5 * al.density * H * 400.0 ** 2)
    assert e == pytest.approx(2.16e5)
    # a free target of equal areal mass keeps half of it as kinetic energy
    free = W.interface_energy(AL, H, 400.0, AL, H)
    assert free == pytest.approx(0.5 * e)
    m_f, m_t = al.density * H, st.density * 3e-3
    assert W.interface_energy(AL, H, 400.0, STEEL, 3e-3) == pytest.approx(0.5 * m_f * m_t / (m_f + m_t) * 400.0 ** 2)
    assert W.interface_energy(AL, H, 400.0, STEEL, 10.0) == pytest.approx(e, rel=1e-3)   # thick target -> backed
    assert W.interface_energy(AL, H, 0.0) == 0.0
    # melt layer: f E / (rho (cp dT + L))
    delta = W.equivalent_melt_layer(AL, 2.0e5)
    enthalpy = al.density * (al.specific_heat * (al.melting_point - T_ROOM) + al.latent_heat_fusion)
    assert delta == pytest.approx(0.1 * 2.0e5 / enthalpy, rel=1e-13)
    assert delta == pytest.approx(8.29e-6, rel=1e-3)
    assert W.equivalent_melt_layer(AL, 2.0e5, fraction=1.0) == pytest.approx(10.0 * delta)
    assert W.equivalent_melt_layer(AL, 0.0) == 0.0
    assert W.equivalent_melt_layer(AL, 2.0e5, initial_temperature=600.0) > delta            # preheated: melts more
    # hastelloy has no latent heat on record: sensible heat only
    c276 = materials.get("hastelloy_c276")
    assert W.equivalent_melt_layer(c276, 1e5) == pytest.approx(
        0.1 * 1e5 / (c276.density * c276.specific_heat * (c276.melting_point - T_ROOM)))
    # it is micrometres thick and gone in about a microsecond
    t_q = W.thermal_quench_time(AL, delta)
    assert t_q == pytest.approx(delta ** 2 / al.thermal_diffusivity(), rel=1e-13)
    assert t_q == pytest.approx(0.996e-6, rel=1e-3)
    assert W.thermal_quench_time(AL, 2 * delta) == pytest.approx(4 * t_q) and W.thermal_quench_time(AL, 0.0) == 0.0


def test_solid_state_interdiffusion_is_nanometres_fusion_welding_is_tens_of_micrometres():
    al = materials.get(AL)
    d_melt = W.solid_diffusivity(AL, al.melting_point)
    assert d_melt == pytest.approx(1e-4 * math.exp(-al.self_diffusion_activation / (R_GAS * al.melting_point)))
    assert d_melt == pytest.approx(2.11e-13, rel=1e-2)
    assert W.solid_diffusivity(AL, 600.0) < d_melt and W.solid_diffusivity(AL, T_ROOM) < 1e-28
    assert W.solid_diffusivity(AL, 855.0, prefactor=2e-4) == pytest.approx(2.0 * d_melt)
    assert W.diffusion_length(1e-9, 1.0) == pytest.approx(math.sqrt(2e-9))
    assert W.diffusion_length(1e-9, 1.0) == pytest.approx(44.72e-6, rel=1e-4)
    assert W.diffusion_length(4e-9, 1.0) == pytest.approx(2 * W.diffusion_length(1e-9, 1.0))
    assert W.diffusion_length(0.0, 1.0) == 0.0 and W.diffusion_length(1e-9, 0.0) == 0.0
    # 1 us at the melting point of aluminium: below a nanometre
    x_solid = W.interdiffusion_length(AL, al.melting_point, 1e-6)
    assert x_solid == pytest.approx(math.sqrt(2 * d_melt * 1e-6))
    assert x_solid == pytest.approx(0.65e-9, rel=1e-2)
    x_fusion = W.diffusion_length(W.LIQUID_DIFFUSIVITY, W.FUSION_WELD_TIME)
    assert 1e-10 < x_solid < 1e-9 and 1e-5 < x_fusion < 1e-4 and x_fusion / x_solid > 1e4
    with pytest.raises(InputError):                          # no activation energy on record
        W.solid_diffusivity("mp35n", 1000.0)


# ------------------------------------------------------------------ assessment: headline results
def test_al_on_steel_welds_at_16_tesla_with_gigapascals_at_the_interface():
    a = W.assess_weld(AL, STEEL, H, GAP, 0.1e9)
    assert isinstance(a, WeldAssessment) and a.status is Status.STATE_OF_THE_ART and a.warnings == ()
    assert (a.flyer, a.target, a.pair, a.pair_demonstrated) == (AL, STEEL, "Al-Fe", True)
    assert a.intermetallics == W.DEMONSTRATED_PAIRS["Al-Fe"].intermetallics
    assert a.drive_field == mg.field_for_pressure(0.1e9) == pytest.approx(15.85, rel=1e-3)
    assert a.impact_velocity == pytest.approx(384.90, rel=1e-5) and a.velocity_limited_by == "gap"
    assert a.collision_velocity == pytest.approx(a.impact_velocity / math.sin(BETA))
    assert a.collision_velocity == pytest.approx(2216.6, rel=1e-4)
    assert a.transit_time == pytest.approx(10.39e-6, rel=1e-3)
    assert a.in_window and a.vp_min <= a.impact_velocity <= a.vp_max == a.vp_melt
    assert a.margin == pytest.approx(min(a.impact_velocity / a.vp_min, a.vp_max / a.impact_velocity))
    assert a.margin == pytest.approx(1.0287, rel=1e-4)
    assert a.interface_morphology == "wavy" and a.reynolds_number == pytest.approx(11.29, rel=1e-3)
    # the gap, not the field, makes the gigapascals: 0.1 GPa of drive -> 4.1 GPa at the interface
    expected = shock.impedance_match(AL, STEEL, a.impact_velocity * math.cos(BETA)).pressure
    assert a.impact_pressure == pytest.approx(expected, rel=1e-13)
    assert a.impact_pressure == pytest.approx(4.096e9, rel=1e-3)
    assert a.pressure_amplification == pytest.approx(40.96, rel=1e-3)
    assert W.README_PRESSURE_RANGE[0] < a.impact_pressure          # the note's pressure is already there
    # the flyer stays cold and shields well
    assert a.flyer_skin_temperature_rise == eddy.skin_temperature_rise(AL, a.drive_field) == pytest.approx(41.3, rel=1e-2)
    assert a.flyer_skin_melting_field == eddy.skin_melting_field(AL) and not a.flyer_skin_melts
    assert a.pressure_transfer == pytest.approx(0.954, rel=1e-3) and not a.driver_needed
    # interface estimates: micrometre melt layer, microsecond quench, sub-nanometre solid diffusion
    assert a.melting_member == AL and a.interface_energy == pytest.approx(2.0e5, rel=1e-6)
    assert a.melt_layer_estimate == pytest.approx(8.29e-6, rel=1e-3) and a.quench_time == pytest.approx(0.996e-6, rel=1e-3)
    assert a.interdiffusion_solid == pytest.approx(0.649e-9, rel=1e-3)
    assert a.interdiffusion_liquid == pytest.approx(44.6e-9, rel=1e-3)
    assert a.fusion_weld_interdiffusion == pytest.approx(44.72e-6, rel=1e-4)
    assert a.fusion_weld_interdiffusion / a.interdiffusion_liquid == pytest.approx(1002.0, rel=1e-2)
    text = " ".join(a.notes)
    assert "41x the drive pressure" in text and "0.0945-0.284 GPa (15.4-26.7 T)" in text
    assert "5.63-14.1x above its upper end" in text and "flat-on impact" in text
    assert "Fe4Al13, Fe2Al5, FeAl" in text and "not eliminated" in text


@pytest.mark.parametrize("pressure, velocity, vc", [(1.6e9, 1539.6, 8866.0), (4.0e9, 2434.3, 14019.0)])
def test_readme_pressures_overshoot_the_window(pressure, velocity, vc):
    a = W.assess_weld(AL, STEEL, H, GAP, pressure)
    assert a.status is Status.INFEASIBLE and not a.in_window and a.interface_morphology == "none"
    assert a.impact_velocity == pytest.approx(velocity, rel=1e-4) and a.collision_velocity == pytest.approx(vc, rel=1e-4)
    assert a.impact_velocity > a.vp_sonic > a.vp_melt and a.margin < 0.5
    assert a.margin == pytest.approx(a.vp_max / a.impact_velocity)
    assert any("supersonic collision point" in w for w in a.warnings)
    assert any("beyond the upper limit" in w for w in a.warnings)
    # and the aluminium flyer's eddy-current skin melts (the "no heat-affected zone" claim)
    assert a.drive_field > eddy.skin_melting_field(AL) == pytest.approx(58.45, rel=1e-3)
    assert a.flyer_skin_melts and any("flyer skin melts" in w for w in a.warnings)
    assert T_ROOM + a.flyer_skin_temperature_rise > materials.get(AL).melting_point
    assert any("industrial electromagnetic-forming coils" in w for w in a.warnings)
    # copper onto titanium overshoots too
    cu = W.assess_weld(CU, TI, H, GAP, pressure)
    assert cu.status is Status.INFEASIBLE and any("supersonic" in w for w in cu.warnings)


def test_cu_flyer_onto_titanium_welds_inside_industrial_fields():
    a = W.assess_weld(CU, TI, H, GAP, 0.4e9)
    assert a.status is Status.STATE_OF_THE_ART and a.pair == "Cu-Ti" and a.pair_demonstrated
    assert a.drive_field == pytest.approx(31.71, rel=1e-3) and a.drive_field < EMF_CONVENTIONAL_FIELD[1]
    assert a.impact_velocity == pytest.approx(422.58, rel=1e-4) and a.margin == pytest.approx(1.2212, rel=1e-4)
    assert a.impact_pressure == pytest.approx(6.153e9, rel=1e-3) and a.pressure_amplification == pytest.approx(15.38, rel=1e-3)
    assert a.melting_member == CU and "CuTi" in a.intermetallics and a.interface_morphology == "wavy"
    assert any("explosive welding" in n for n in a.notes)
    # titanium as the flyer is too resistive: same joint, but it needs a driver sheet
    rev = W.assess_weld(TI, CU, H, GAP, 0.2e9)
    assert rev.in_window and rev.driver_needed and rev.status is Status.EXTRAPOLATION
    assert rev.pressure_transfer == pytest.approx(0.591, rel=1e-2)
    assert rev.pressure_transfer < W.DRIVER_TRANSFER_THRESHOLD
    assert any("driver sheet" in w for w in rev.warnings)
    assert W.assess_weld("steel_304", AL, H, GAP, 0.4e9).driver_needed
    assert not W.assess_weld(CU, STEEL, H, GAP, 0.3e9).driver_needed


def test_status_ladder_and_cautions():
    # too slow: no jet
    slow = W.assess_weld(AL, STEEL, H, GAP, 0.05e9)
    assert slow.status is Status.INFEASIBLE and slow.margin == pytest.approx(slow.impact_velocity / slow.vp_min)
    assert [w for w in slow.warnings if "below the bonding threshold" in w] and slow.margin < 1.0
    # just past the melting limit, still subsonic: only the upper-limit rule fires
    hot = W.assess_weld(AL, STEEL, H, GAP, 0.3e9)
    assert hot.status is Status.INFEASIBLE and hot.impact_velocity < hot.vp_sonic
    assert len(hot.warnings) == 1 and "beyond the upper limit" in hot.warnings[0]
    # in the window but not a recorded pair -> extrapolation
    ta = W.assess_weld(CU, "tantalum", H, GAP, 0.3e9)
    assert ta.in_window and not ta.pair_demonstrated and ta.status is Status.EXTRAPOLATION
    assert ta.intermetallics == () and any("not a recorded" in w for w in ta.warnings)
    tial = W.assess_weld(AL, "gamma_tial", H, GAP, 0.2e9)
    assert tial.pair == "Al-Ti" and not tial.pair_demonstrated and tial.status is Status.EXTRAPOLATION
    # in the window at 100 T only with a 0.1 mm gap: beyond industrial coils, skin molten, sub-us flight
    tiny = W.assess_weld(AL, STEEL, H, 1e-4, 4e9)
    assert tiny.in_window and tiny.status is Status.EXTRAPOLATION and tiny.flyer_skin_melts
    assert tiny.transit_time == pytest.approx(0.367e-6, rel=1e-2) and tiny.transit_time < W.FASTEST_HIGH_FIELD_RISE
    assert any("single-turn" in w for w in tiny.warnings) and tiny.pressure_amplification < 2.0
    # copper survives 74 T without melting; still beyond industrial coils
    cu = W.assess_weld(CU, STEEL, H, 2e-4, 2.2e9)
    assert cu.in_window and not cu.flyer_skin_melts and cu.status is Status.EXTRAPOLATION
    # no drive at all
    zero = W.assess_weld(AL, STEEL, H, GAP, 0.0)
    assert zero.status is Status.INFEASIBLE and zero.margin == 0.0 and zero.impact_velocity == 0.0
    assert zero.pressure_amplification is None and zero.pressure_transfer is None and zero.velocity_limited_by == "no_drive"
    json.loads(zero.json())
    # dirty surfaces close the window for this joint
    dirty = W.assess_weld(AL, STEEL, H, GAP, 0.15e9, lower_coefficient=W.DERIBAS_K_TECHNICAL)
    assert dirty.status is Status.INFEASIBLE and any("no weldability window" in w for w in dirty.warnings)
    # (margin >= 1) <=> in window <=> not infeasible, across a sweep
    for p in (0.02e9, 0.08e9, 0.1e9, 0.2e9, 0.28e9, 0.29e9, 0.6e9, 2e9):
        a = W.assess_weld(AL, STEEL, H, GAP, p)
        assert (a.margin >= 1.0) == a.in_window == (a.status is not Status.INFEASIBLE)
    # the angle matters: the same drive is supersonic at 4 degrees, flat-interface at 20 degrees
    steep = W.assess_weld(AL, STEEL, H, GAP, 0.15e9, collision_angle=math.radians(20.0))
    assert steep.in_window and steep.interface_morphology == "flat"
    assert any("outside" in n for n in W.assess_weld(AL, STEEL, H, GAP, 0.15e9, math.radians(25.0)).notes)
    shallow = W.assess_weld(AL, STEEL, H, GAP, 0.15e9, collision_angle=math.radians(4.0))
    assert shallow.status is Status.INFEASIBLE and any("supersonic" in w for w in shallow.warnings)


def test_rise_time_not_peak_pressure_sets_the_weld():
    # a millisecond-class 100 T pulse: the flyer leaves early and lands far below the bonding threshold
    slow = W.assess_weld(AL, STEEL, H, GAP, 4e9, pulse_duration=5e-3, pulse_shape="half_sine")
    assert slow.status is Status.INFEASIBLE and slow.velocity_limited_by == "gap"
    assert slow.impact_velocity == pytest.approx(99.93, rel=1e-3) and slow.impact_velocity < slow.vp_min
    assert slow.pressure_amplification < 1.0                    # interface sees less than the peak drive
    assert any("slow pulse" in w and "rise time" in w for w in slow.warnings)
    assert any("below the bonding threshold" in w for w in slow.warnings)
    # ... and at 100 Hz a 1 mm sheet does not even shield the field
    assert slow.driver_needed and slow.pressure_transfer == pytest.approx(0.180, rel=1e-2)
    assert any("shields poorly" in w for w in slow.warnings)
    # a 20 us pulse at 0.3 GPa peak welds (constant 0.3 GPa would overshoot: the rise costs velocity)
    fast = W.assess_weld(AL, STEEL, H, GAP, 0.3e9, pulse_duration=20e-6, pulse_shape="half_sine")
    assert fast.status is Status.STATE_OF_THE_ART and fast.impact_velocity == pytest.approx(621.1, rel=1e-3)
    assert W.assess_weld(AL, STEEL, H, GAP, 0.3e9).status is Status.INFEASIBLE
    # a rectangular pulse shorter than the flight is impulse-limited
    blip = W.assess_weld(AL, STEEL, H, GAP, 4e9, pulse_duration=2e-7)
    assert blip.velocity_limited_by == "pulse" and blip.status is Status.INFEASIBLE
    assert blip.impact_velocity == pytest.approx(4e9 * 2e-7 / (materials.get(AL).density * H), rel=1e-12)
    assert any("impulse-limited" in w for w in blip.warnings)


def test_free_target_and_material_handling():
    backed = W.assess_weld(AL, STEEL, H, GAP, 0.15e9)
    free = W.assess_weld(AL, STEEL, H, GAP, 0.15e9, target_thickness=1e-3)
    assert free.interface_energy < backed.interface_energy and free.status is backed.status
    assert free.impact_velocity == backed.impact_velocity
    by_alias = W.assess_weld("al", "steel", H, GAP, 0.15e9)
    by_object = W.assess_weld(materials.get(AL), materials.get(STEEL), H, GAP, 0.15e9)
    assert by_alias.json() == by_object.json() == backed.json()      # deterministic, input-form independent
    assert W.weld_window("cu", "ti", H).flyer == CU
    # a Material that is not in the registry (e.g. a softer temper) works everywhere
    soft = materials.get(AL).with_changes(key="al6061_o", vickers_hardness=0.3e9)
    w = W.weld_window(soft, STEEL, H)
    assert w.flyer == "al6061_o" and w.lower_governed_by == STEEL and w.vp_min == pytest.approx(239.12, rel=1e-4)
    d = W.drive_pressure_window(soft, STEEL, H, GAP)
    assert d.pressure_min < W.drive_pressure_window(AL, STEEL, H, GAP).pressure_min
    assert W.standoff_window(soft, STEEL, H, 0.1e9).gap_min == pytest.approx(
        soft.density * H * w.vp_min ** 2 / (2 * 0.1e9))
    a = W.assess_weld(soft, STEEL, H, GAP, 0.05e9)
    assert a.flyer == "al6061_o" and a.pair == "Al-Fe" and a.status is Status.STATE_OF_THE_ART


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("call", [
    lambda: W.collision_point_velocity(-1.0, 0.2),
    lambda: W.collision_point_velocity(400.0, 0.0),
    lambda: W.collision_point_velocity(400.0, math.pi / 2),
    lambda: W.collision_point_velocity(400.0, float("nan")),
    lambda: W.impact_velocity_for_collision(2000.0, -0.1),
    lambda: W.critical_collision_angle(AL, STEEL, -1.0),
    lambda: W.lower_limit_velocity(AL, STEEL, 0.0),
    lambda: W.collision_reynolds_number(AL, STEEL, -5.0),
    lambda: W.wavy_transition_velocity(AL, STEEL, reynolds=0.0),
    lambda: W.upper_limit_velocity(AL, STEEL, 0.0, 2500.0),
    lambda: W.upper_limit_velocity(AL, STEEL, H, 0.0),
    lambda: W.upper_limit_velocity(AL, STEEL, H, 2500.0, n=-0.1),
    lambda: W.weld_window(AL, STEEL, -H),
    lambda: W.weld_window(AL, STEEL, H, 2.0),
    lambda: W.weld_window(AL, STEEL, H, wittman_n=0.0),
    lambda: W.weld_window(AL, "diamond", H),                       # not a metal
    lambda: W.weld_window("zylon_fibre", STEEL, H),
    lambda: W.jetting_sonic_limit("graphite_crystal", STEEL),
    lambda: W.flyer_impact(-1.0, GAP, AL, H),
    lambda: W.flyer_impact(1e8, 0.0, AL, H),
    lambda: W.flyer_impact(1e8, GAP, AL, H, efficiency=0.0),
    lambda: W.flyer_impact(1e8, GAP, AL, H, efficiency=1.5),
    lambda: W.flyer_impact(1e8, GAP, AL, H, pulse_duration=0.0),
    lambda: W.flyer_impact(1e8, GAP, AL, H, pulse_duration=1e-5, pulse_shape="square"),
    lambda: W.drive_pressure_window(AL, STEEL, H, 0.0),
    lambda: W.drive_pressure_window(AL, STEEL, H, GAP, efficiency=2.0),
    lambda: W.standoff_window(AL, STEEL, H, 0.0),
    lambda: W.interface_energy(AL, H, -1.0),
    lambda: W.interface_energy(AL, H, 400.0, STEEL),               # target without its thickness
    lambda: W.interface_energy(AL, H, 400.0, target_thickness=1e-3),
    lambda: W.equivalent_melt_layer(AL, -1.0),
    lambda: W.equivalent_melt_layer(AL, 1e5, fraction=1.5),
    lambda: W.equivalent_melt_layer(AL, 1e5, initial_temperature=900.0),   # above the Al 6061 solidus
    lambda: W.thermal_quench_time(AL, -1e-6),
    lambda: W.solid_diffusivity(AL, 0.0),
    lambda: W.diffusion_length(-1e-9, 1.0),
    lambda: W.interdiffusion_length(AL, 800.0, -1.0),
    lambda: W.assess_weld(AL, STEEL, H, GAP, -1.0),
    lambda: W.assess_weld(AL, STEEL, H, 0.0, 1e8),
    lambda: W.assess_weld(AL, STEEL, 0.0, GAP, 1e8),
    lambda: W.assess_weld(AL, STEEL, H, GAP, 1e8, collision_angle=0.0),
    lambda: W.assess_weld(AL, STEEL, H, GAP, 1e8, efficiency=0.0),
    lambda: W.assess_weld(AL, STEEL, H, GAP, 1e8, target_thickness=0.0),
    lambda: W.assess_weld(AL, STEEL, H, GAP, 1e8, initial_temperature=1000.0),
    lambda: W.assess_weld(AL, STEEL, H, GAP, 1e8, pulse_shape="half_sine"),
    lambda: W.assess_weld(AL, STEEL, H, GAP, float("inf")),
    lambda: W.assess_weld(AL, "tungsten_carbide_co", H, GAP, 1e8),
    lambda: W.assess_weld(AL, STEEL, True, GAP, 1e8),
])
def test_invalid_inputs_raise_input_error(call):
    with pytest.raises(InputError):
        call()


def test_unknown_material_raises():
    with pytest.raises(UnknownMaterialError):
        W.assess_weld("unobtainium", STEEL, H, GAP, 1e8)
    with pytest.raises(UnknownMaterialError):
        W.weld_window(AL, "adamantium", H)


# ------------------------------------------------------------------ serialisation
def test_results_round_trip_through_json_and_render_as_text():
    window = W.weld_window(AL, STEEL, H)
    flight = W.flyer_impact(4e9, GAP, AL, H, pulse_duration=5e-3, pulse_shape="half_sine")
    drive = W.drive_pressure_window(AL, STEEL, H, GAP)
    standoff = W.standoff_window(AL, STEEL, H, 4e9)
    weld = W.assess_weld(AL, STEEL, H, GAP, 0.1e9)
    pair = W.DEMONSTRATED_PAIRS["Al-Fe"]
    for result, title in ((window, "Weldability window"), (flight, "Flyer impact"),
                          (drive, "Drive-pressure window"), (standoff, "Standoff window"),
                          (weld, "Impact-weld assessment"), (pair, "WeldPair")):
        data = json.loads(result.json())
        assert isinstance(data, dict)
        assert result.text().startswith(title)
    data = json.loads(weld.json())
    assert data["status"] == "state_of_the_art" and data["flyer"] == AL and data["pair"] == "Al-Fe"
    assert data["impact_velocity"] == pytest.approx(weld.impact_velocity)
    assert data["intermetallics"] == ["Fe4Al13", "Fe2Al5", "FeAl"] and data["in_window"] is True
    assert data["margin"] == pytest.approx(weld.margin) and data["warnings"] == []
    assert json.loads(drive.json())["window"]["vp_min"] == pytest.approx(window.vp_min)
    assert json.loads(flight.json())["pulse_shape"] == "half_sine"
    assert json.loads(pair.json())["elements"] == ["Al", "Fe"]
    txt = weld.text()
    assert "impact_velocity [m/s] = 384.9" in txt and "drive_field [T] = 15.85" in txt
    assert "collision_angle [rad]" in txt and "notes:" in txt
    assert "vp_min [m/s]" in window.text() and "pressure_min [Pa]" in drive.text()
    with pytest.raises(Exception):
        weld.status = Status.INFEASIBLE  # frozen
