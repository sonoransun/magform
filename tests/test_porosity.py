"""Tests for magform.porosity: pore collapse, healing, the HIP comparator and powder compaction."""

from __future__ import annotations

import json
import math

import pytest

from magform import materials, plasticity
from magform import porosity as po
from magform._report import Assessment, ResultMixin
from magform.benchmarks import Benchmark
from magform.constants import MU0, P_ATM, R_GAS, T_ROOM
from magform.eddy import skin_depth_for_material
from magform.errors import InputError, UnknownMaterialError
from magform.status import Confidence, Status

RT = T_ROOM
TIAL_HIP = po.HIP_PRACTICE["gamma_tial"]


# ------------------------------------------------------------------ metadata
def test_module_metadata_and_constants():
    assert set(po.__all__) <= set(dir(po))
    assert isinstance(po.REFERENCES, dict) and len(po.REFERENCES) >= 12
    assert all(isinstance(k, str) and isinstance(v, str) and len(v) > 40 for k, v in po.REFERENCES.items())
    for name in po.__all__:
        obj = getattr(po, name)
        if callable(obj):
            assert obj.__doc__ and "Reference" in obj.__doc__, name
    assert po.QUASI_STATIC_RATE == plasticity.FALLBACK_REFERENCE_RATE == 1e-3
    assert po.CLOSURE_POROSITY == 1e-4 and po.DEFAULT_POROSITY == 0.01 and po.DEFAULT_PORE_RADIUS == 50e-6
    assert po.RAYLEIGH_COEFFICIENT == pytest.approx(0.915, rel=1e-3)
    assert po.DIFFUSION_PREFACTOR == 1e-4 and po.DEFAULT_BOND_LENGTH == 1e-7
    assert po.HEALING_UNCERTAINTY_FACTOR == 10.0 and po.CREEP_HOMOLOGOUS_TEMPERATURE == 0.5
    assert po.PORE_WALL_CONFINEMENT == pytest.approx(2.0 / 3.0)
    assert po.HIP_PRESSURE_RANGE == (100e6, 200e6)
    assert po.RANDOM_DENSE_PACKING == 0.64 and po.STAGE_TRANSITION_DENSITY == 0.9
    assert isinstance(po.DMC_BENCHMARK, Benchmark) and po.DMC_BENCHMARK.citation == po.REFERENCES["chelluri_barber_1999"]
    assert po.DMC_PRESSURE_CLASS == po.DMC_BENCHMARK.value == 2.0e9
    assert po.DMC_BENCHMARK.confidence is Confidence.ESTIMATED          # a class boundary, not a record


# ------------------------------------------------------------------ plastic collapse (closed forms)
def test_plastic_collapse_pressure_closed_form_and_limits():
    sigma = 1.0e9
    for f, ratio in ((0.1, 1.5351), (0.01, 3.0701), (0.005, 3.5322), (1e-4, 6.1402)):
        assert po.plastic_collapse_pressure(sigma, f) == pytest.approx(ratio * sigma, rel=1e-4)
        assert po.plastic_collapse_pressure(sigma, f) == pytest.approx((2.0 / 3.0) * sigma * math.log(1.0 / f))
    # small porosity is HARDER to collapse; linear in the flow stress
    thresholds = [po.plastic_collapse_pressure(sigma, f) for f in (0.2, 0.05, 0.01, 1e-3, 1e-6)]
    assert all(b > a for a, b in zip(thresholds, thresholds[1:]))
    assert po.plastic_collapse_pressure(3 * sigma, 0.01) == pytest.approx(3 * po.plastic_collapse_pressure(sigma, 0.01))
    # limits
    assert po.plastic_collapse_pressure(sigma, 0.0) == math.inf
    assert po.plastic_collapse_pressure(sigma, 1.0) == 0.0
    assert po.plastic_collapse_pressure(0.0, 0.01) == 0.0 and po.plastic_collapse_pressure(0.0, 0.0) == 0.0
    # exactly the hydrostatic limit of the Gurson yield surface: cosh(3P/(2 sigma)) = (1 + f^2)/(2f)
    for f in (0.2, 0.01, 1e-3):
        p = po.plastic_collapse_pressure(sigma, f)
        assert math.cosh(1.5 * p / sigma) == pytest.approx((1.0 + f * f) / (2.0 * f), rel=1e-12)


def test_porosity_under_pressure_inverts_the_collapse_pressure():
    sigma, f0 = 0.55e9, 0.01
    threshold = po.plastic_collapse_pressure(sigma, f0)
    assert po.porosity_under_pressure(sigma, 0.0, f0) == f0
    assert po.porosity_under_pressure(sigma, 0.99 * threshold, f0) == f0     # the pore does not move
    assert po.porosity_under_pressure(sigma, threshold, f0) == pytest.approx(f0, rel=1e-12)
    for p in (1.01 * threshold, 2.5e9, 4e9, 8e9):
        f = po.porosity_under_pressure(sigma, p, f0)
        assert f < f0 and f == pytest.approx(math.exp(-1.5 * p / sigma), rel=1e-12)
        assert po.plastic_collapse_pressure(sigma, f) == pytest.approx(p, rel=1e-12)
    series = [po.porosity_under_pressure(sigma, p, f0) for p in (0.0, 1e9, 2e9, 3e9, 4e9, 6e9)]
    assert all(b <= a for a, b in zip(series, series[1:]))
    # a matrix without strength closes under any pressure (vacuum pore)
    assert po.porosity_under_pressure(0.0, 1.0, f0) == 0.0 and po.porosity_under_pressure(0.0, 0.0, f0) == f0


def test_trapped_gas_limits_the_collapse():
    f0, p = 0.01, 4e9
    f_gas = po.gas_limited_porosity(f0, P_ATM, p)
    x = (P_ATM / p) * f0 / (1.0 - f0)
    assert f_gas == pytest.approx(x / (1.0 + x), rel=1e-12)
    assert f_gas == pytest.approx(f0 * P_ATM / p, rel=2e-2) and f_gas == pytest.approx(2.56e-7, rel=1e-2)
    assert po.gas_limited_porosity(f0, 0.0, p) == 0.0                 # vacuum shrinkage pore: no gas limit
    assert po.gas_limited_porosity(f0, 2e9, 1e9) == f0                # gas already carries the load
    assert po.gas_limited_porosity(f0, P_ATM, 0.0) == f0
    # with gas the residual porosity solves P = p_gas(f) + (2/3) sigma ln(1/f)
    sigma = 0.55e9
    f = po.porosity_under_pressure(sigma, p, f0, P_ATM)
    balance = P_ATM * (f0 / (1.0 - f0)) * (1.0 - f) / f + (2.0 / 3.0) * sigma * math.log(1.0 / f)
    assert balance == pytest.approx(p, rel=1e-9)
    assert f == pytest.approx(2.090e-5, rel=1e-3) and f > po.porosity_under_pressure(sigma, p, f0) > f_gas
    # a high-pressure gas pore dominates: the answer tends to the gas limit as the matrix weakens
    assert po.porosity_under_pressure(0.0, p, f0, 50e6) == po.gas_limited_porosity(f0, 50e6, p)
    assert po.porosity_under_pressure(1e3, p, f0, 50e6) == pytest.approx(po.gas_limited_porosity(f0, 50e6, p), rel=1e-5)
    # the pore does not move until P exceeds the gas pressure plus the plastic threshold
    start = 50e6 + po.plastic_collapse_pressure(sigma, f0)
    assert po.porosity_under_pressure(sigma, start, f0, 50e6) == f0
    assert po.porosity_under_pressure(sigma, 1.001 * start, f0, 50e6) < f0


def test_pore_wall_strain():
    # halving the pore radius (pore volume per matrix volume / 8) costs exactly 2 ln 2
    f0 = 0.01
    x1 = (f0 / (1.0 - f0)) / 8.0
    assert po.pore_wall_strain(f0, x1 / (1.0 + x1)) == pytest.approx(2.0 * math.log(2.0), rel=1e-12)
    assert po.pore_wall_strain(0.01, 1e-4) == pytest.approx(3.0767, rel=1e-4)
    assert po.pore_wall_strain(0.005, 1e-4) == pytest.approx(2.6113, rel=1e-4)
    assert po.pore_wall_strain(0.01, 0.01) == 0.0 and po.pore_wall_strain(0.01, 0.0) == math.inf
    strains = [po.pore_wall_strain(0.01, f) for f in (5e-3, 1e-3, 1e-4, 1e-6)]
    assert all(b > a for a, b in zip(strains, strains[1:]))
    with pytest.raises(InputError):
        po.pore_wall_strain(0.01, 0.02)


def test_rayleigh_collapse_time_shows_time_is_not_the_obstacle(tial):
    t = po.rayleigh_collapse_time(50e-6, tial, 2.3e9)
    assert t == pytest.approx(0.91468 * 50e-6 * math.sqrt(tial.density / 2.3e9), rel=1e-12)
    assert t == pytest.approx(60.1e-9, rel=1e-2) and t < 1e-3 * 1e-4       # ~60 ns against a 0.1 ms pulse
    assert po.rayleigh_collapse_time(100e-6, tial, 2.3e9) == pytest.approx(2.0 * t)
    assert po.rayleigh_collapse_time(50e-6, tial, 4 * 2.3e9) == pytest.approx(0.5 * t)
    assert po.rayleigh_collapse_time(50e-6, "gamma_tial", 0.0) == math.inf
    assert po.rayleigh_collapse_time(50e-6, "gamma_tial", -1e9) == math.inf


# ------------------------------------------------------------------ matrix strength
def test_matrix_strength_brackets_quasi_static_and_dynamic(tial, w):
    s = po.matrix_strength(tial)
    assert isinstance(s, ResultMixin) and s.material == "gamma_tial" and s.strain_rate == 1e4
    assert s.quasi_static == tial.compressive_yield == 550e6
    assert s.quasi_static_basis == "handbook compressive yield"
    assert s.dynamic == plasticity.flow_stress(tial, 0.0, 1e4, RT) == pytest.approx(1.402e9, rel=1e-3)
    assert s.dynamic_model == tial.constitutive[0].source and s.extrapolated is False
    # tungsten: the weakest published fit at 1e-3 1/s undercuts the handbook yield; strongest fit at 1e4 1/s
    sw = po.matrix_strength(w)
    band_qs = plasticity.flow_stress_band(w, 0.0, 1e-3, RT)
    band_dyn = plasticity.flow_stress_band(w, 0.0, 1e4, RT)
    assert sw.quasi_static == band_qs.minimum < w.yield_strength and sw.quasi_static_basis == w.constitutive[0].source
    assert sw.dynamic == band_dyn.maximum and sw.dynamic_model == w.constitutive[1].source
    # a material without a published fit: handbook yield and the (flagged) Hollomon fallback
    ods = po.matrix_strength("ods_fecral")
    assert ods.quasi_static == 517e6 and ods.quasi_static_basis == "handbook yield strength"
    assert ods.dynamic_model == po.FALLBACK_BASIS and ods.extrapolated is True
    assert ods.dynamic == pytest.approx(517e6 * (1.0 + 0.04 * math.log(1e7)), rel=1e-12)
    assert po.matrix_strength("ods_fecral", 1e-3).extrapolated is False
    # compressive yield is preferred when recorded (Mg twins in compression)
    assert po.matrix_strength("magnesium_az31").quasi_static == 140e6
    # slow rates are evaluated at the quasi-static rate; dynamic never falls below quasi-static
    slow = po.matrix_strength(tial, 1e-5)
    assert slow.strain_rate == 1e-5 and slow.dynamic == po.matrix_strength(tial, 1e-3).dynamic
    for key in ("gamma_tial", "ti6al4v", "inconel_718", "ods_fecral", "ods_ni", "copper_ofhc", "tungsten"):
        for rate in (0.0, 1e-3, 1.0, 1e4):
            for temp in (RT, 600.0, 0.8 * materials.get(key).melting_point):
                st = po.matrix_strength(key, rate, temp)
                assert plasticity.MIN_FLOW_STRESS <= st.quasi_static <= st.dynamic, (key, rate, temp)


def test_matrix_strength_softens_with_temperature(tial):
    t = TIAL_HIP.temperature
    hot = po.matrix_strength(tial, 1e4, t)
    t_star = (t - RT) / (tial.melting_point - RT)
    assert hot.quasi_static == pytest.approx(550e6 * (1.0 - t_star), rel=1e-12)   # linear handbook softening
    assert hot.quasi_static < 0.2 * 550e6 and hot.dynamic < po.matrix_strength(tial).dynamic
    assert hot.extrapolated is True and "773" in hot.extrapolation_note          # Johnson-Cook fit ends at 773 K
    assert po.matrix_strength(tial, 1e4, 77.0).quasi_static == 550e6              # no strengthening modelled below RT


def test_gamma_tial_thresholds_headline():
    # quasi-static (handbook compressive yield 0.55 GPa) versus 1e4 1/s (Johnson-Cook, 1.40 GPa)
    assert po.collapse_pressure("gamma_tial", 0.01) == pytest.approx(1.6886e9, rel=1e-4)
    assert po.collapse_pressure("gamma_tial", 0.005) == pytest.approx(1.9427e9, rel=1e-4)
    assert po.collapse_pressure("gamma_tial", 0.01, 1e4) == pytest.approx(4.3047e9, rel=1e-4)
    assert po.collapse_pressure("gamma_tial", 0.005, 1e4) == pytest.approx(4.9527e9, rel=1e-4)
    # at the pulse's strain rate the concept note's 4 GPa does not even start the collapse
    assert po.collapse_pressure("gamma_tial", 0.01, 1e4) > 4e9 > po.collapse_pressure("gamma_tial", 0.01)
    assert po.collapse_pressure("gamma_tial", 0.01) > 1.6e9          # and 1.6 GPa does nothing on any reading
    assert po.residual_porosity("gamma_tial", 4e9, 0.01) == pytest.approx(1.829e-5, rel=1e-3)
    assert po.residual_porosity("gamma_tial", 4e9, 0.005) == pytest.approx(1.829e-5, rel=1e-3)
    assert po.residual_porosity("gamma_tial", 4e9, 0.01, 1e4) == 0.01
    assert po.residual_porosity("gamma_tial", 1.6e9, 0.01) == 0.01
    assert po.residual_porosity("gamma_tial", 4e9, 0.01, gas_pressure=P_ATM) == pytest.approx(2.090e-5, rel=1e-3)
    # consistent with the pure closed forms
    s = po.matrix_strength("gamma_tial")
    assert po.collapse_pressure("gamma_tial", 0.01, 1e4) == po.plastic_collapse_pressure(s.dynamic, 0.01)
    assert po.residual_porosity("gamma_tial", 4e9, 0.01) == po.porosity_under_pressure(s.quasi_static, 4e9, 0.01)


# ------------------------------------------------------------------ healing
def test_diffusion_arrhenius_and_bonding_time(tial):
    t_hip = TIAL_HIP.temperature
    d_hot = po.self_diffusivity(tial, t_hip)
    assert d_hot == pytest.approx(1e-4 * math.exp(-300e3 / (R_GAS * t_hip)), rel=1e-12)
    assert d_hot == pytest.approx(6.01e-15, rel=1e-2)
    d_cold = po.self_diffusivity(tial, RT)
    assert d_cold == pytest.approx(3.51e-58, rel=1e-2)
    assert po.self_diffusivity(tial, RT, prefactor=1e-5) == pytest.approx(0.1 * d_cold)
    # bonding time L^2/D: seconds at the HIP temperature, ~1e43 s at room temperature
    assert po.diffusion_bonding_time(tial, t_hip) == pytest.approx(1e-14 / d_hot)
    assert po.diffusion_bonding_time(tial, t_hip) == pytest.approx(1.663, rel=1e-2)
    assert po.diffusion_bonding_time(tial, RT) == pytest.approx(2.85e43, rel=1e-2)
    assert po.diffusion_bonding_time(tial, RT, 1e-6) == pytest.approx(100.0 * po.diffusion_bonding_time(tial, RT))
    # atoms move ~2e-31 m during a 0.1 ms pulse at room temperature, 9 um during the HIP hold
    assert po.diffusion_length(tial, RT, 1e-4) == pytest.approx(1.87e-31, rel=1e-2)
    assert po.diffusion_length(tial, t_hip, TIAL_HIP.time) == pytest.approx(9.31e-6, rel=1e-2)
    assert po.diffusion_length(tial, RT, 0.0) == 0.0
    # diffusivity rises monotonically with temperature; underflow gives an infinite bonding time
    ds = [po.self_diffusivity(tial, t) for t in (300.0, 600.0, 900.0, 1200.0, 1500.0)]
    assert all(b > a for a, b in zip(ds, ds[1:]))
    assert po.self_diffusivity("tungsten", 4.0) == 0.0 and po.diffusion_bonding_time("tungsten", 4.0) == math.inf
    with pytest.raises(InputError):
        po.self_diffusivity("hastelloy_c276", 300.0)      # no activation energy recorded


def test_hip_practice_table():
    assert {"gamma_tial", "ti6al4v", "inconel_718"} <= set(po.HIP_PRACTICE)
    for key, cycle in po.HIP_PRACTICE.items():
        m = materials.get(key)
        assert isinstance(cycle, po.HipCycle) and cycle.material == key
        assert po.HIP_PRESSURE_RANGE[0] <= cycle.pressure <= po.HIP_PRESSURE_RANGE[1]      # 0.1-0.2 GPa
        lo, hi = cycle.temperature_window
        assert lo <= cycle.temperature <= hi < m.melting_point and cycle.temperature / m.melting_point > 0.6
        assert 3600.0 <= cycle.time <= 6 * 3600.0 and cycle.medium == "argon"
        assert cycle.source in po.REFERENCES.values() and cycle.note
        assert cycle.confidence is Confidence.REPRESENTATIVE
        assert po.hip_cycle(key) is cycle
        # the diffusion estimate is within its order-of-magnitude uncertainty of the industrial hold
        assert cycle.time * po.HEALING_UNCERTAINTY_FACTOR >= po.diffusion_bonding_time(key, cycle.temperature)
        assert po.matches_hip_practice(key, cycle.pressure, cycle.temperature, cycle.time)
        assert po.matches_hip_practice(key, 4e9, cycle.temperature, 2 * cycle.time)      # more pressure still counts
        assert not po.matches_hip_practice(key, cycle.pressure, cycle.temperature, 0.5 * cycle.time)
        assert not po.matches_hip_practice(key, cycle.pressure, RT, cycle.time)
        assert not po.matches_hip_practice(key, 50e6, cycle.temperature, cycle.time)
        assert not po.matches_hip_practice(key, cycle.pressure, hi + 50.0, cycle.time)
    assert TIAL_HIP.pressure == 172e6 and TIAL_HIP.temperature == pytest.approx(1533.15) and TIAL_HIP.time == 14400.0
    assert po.hip_cycle("TiAl") is TIAL_HIP and po.hip_cycle("ti64") is po.HIP_PRACTICE["ti6al4v"]
    assert po.hip_cycle("copper_ofhc") is None
    assert po.matches_hip_practice("copper_ofhc", 172e6, 1000.0, 1e5) is False


# ------------------------------------------------------------------ assessment: headline results
@pytest.mark.parametrize("f0, pc_qs, pc_dyn, margin, t_collapse", [
    (0.01, 1.6886e9, 4.3047e9, 0.9292, 59.94e-9),
    (0.005, 1.9427e9, 4.9527e9, 0.8076, 63.53e-9),
])
def test_gamma_tial_at_4gpa_room_temperature_is_infeasible(f0, pc_qs, pc_dyn, margin, t_collapse):
    a = po.assess_pore_closure("gamma_tial", 4e9, f0)
    assert isinstance(a, Assessment) and a.material == "gamma_tial"
    assert a.pressure == 4e9 and a.field_equivalent == pytest.approx(100.27, rel=1e-4)
    assert a.temperature == RT and a.pulse_duration == 1e-4 and a.strain_rate == 1e4
    assert a.flow_stress_quasi_static == 550e6 and a.flow_stress_dynamic == pytest.approx(1.402e9, rel=1e-3)
    # collapse: only if the matrix behaves quasi-statically
    assert a.collapse_pressure_quasi_static == pytest.approx(pc_qs, rel=1e-4)
    assert a.collapse_pressure_dynamic == pytest.approx(pc_dyn, rel=1e-4)
    assert a.closure_pressure_quasi_static == pytest.approx(3.3771e9, rel=1e-4)       # 6.14 sigma
    assert a.closure_pressure_dynamic == pytest.approx(8.6094e9, rel=1e-4)
    assert a.collapses_quasi_static is True and a.collapses_dynamic is False
    assert a.residual_porosity_quasi_static == pytest.approx(1.829e-5, rel=1e-3) and a.closed_quasi_static is True
    assert a.residual_porosity_dynamic == f0 and a.closed_dynamic is False
    assert a.margin == pytest.approx(margin, rel=1e-3) and a.margin == pytest.approx(4e9 / a.collapse_pressure_dynamic)
    assert a.closure_status is Status.EXTRAPOLATION
    assert any("closure only at the quasi-static strength" in x and "never yield" in x for x in a.warnings)
    # time is not the obstacle: tens of nanoseconds against 0.1 ms
    assert a.collapse_time == pytest.approx(t_collapse, rel=1e-3) and a.collapse_time < 1e-3 * a.pulse_duration
    assert any("time is not the obstacle" in x for x in a.notes)
    # healing: astronomically longer than the pulse -> an unbonded, crack-like flaw
    assert a.diffusion_bond_time == pytest.approx(2.85e43, rel=1e-2) and a.diffusion_bond_time > 1e40 * a.pulse_duration
    assert a.diffusion_length == pytest.approx(1.87e-31, rel=1e-2) and a.bonded is False
    assert a.healing_status is Status.INFEASIBLE
    assert any("closed but not healed" in x and "crack-like" in x for x in a.warnings)
    assert a.status is Status.INFEASIBLE
    # brittleness: wall strain of ~3 against 1.5 % ductility, below the 973 K DBTT
    assert a.below_dbtt is True and a.dbtt == 973.0 and a.ductility == 0.015
    assert a.wall_strain == pytest.approx(po.pore_wall_strain(f0, po.CLOSURE_POROSITY))
    assert a.wall_strain_to_ductility == pytest.approx(a.wall_strain / 0.015) and a.wall_strain_to_ductility > 100.0
    assert any("below the DBTT" in x and "Bridgman" in x for x in a.warnings)
    assert a.confinement_ratio == pytest.approx(4e9 / a.flow_stress_dynamic)
    # the comparator already does it at 0.17 GPa
    assert a.hip is TIAL_HIP and a.matches_hip_practice is False
    assert any("172 MPa" in x and "1260 C" in x and "4 h" in x for x in a.notes)
    assert any("not isostatic" in x for x in a.notes)
    assert a.creep_regime is False and a.extrapolated_model is False and a.gas_limited_porosity == 0.0


def test_lower_end_of_the_window_does_nothing():
    a = po.assess_pore_closure("gamma_tial", 1.6e9)
    assert a.collapses_quasi_static is False and a.collapses_dynamic is False
    assert a.residual_porosity_quasi_static == a.residual_porosity_dynamic == 0.01
    assert a.closure_status is Status.INFEASIBLE and a.healing_status is Status.INFEASIBLE
    assert a.status is Status.INFEASIBLE and a.margin < 0.4
    assert any("do not yield even at the quasi-static strength" in x for x in a.warnings)
    assert any("no healing either" in x for x in a.warnings)
    assert a.collapse_time == math.inf and any("never yield" in x for x in a.notes)
    # larger pores yield earlier but still do not close: 5 % shrinks to 1.3 %
    big = po.assess_pore_closure("gamma_tial", 1.6e9, initial_porosity=0.05)
    assert big.collapses_quasi_static is True and big.closed_quasi_static is False
    assert big.residual_porosity_quasi_static == pytest.approx(math.exp(-1.5 * 1.6e9 / 550e6), rel=1e-12)
    assert big.closure_status is Status.INFEASIBLE
    assert any("shrink but do not close" in x for x in big.warnings)


def test_strong_alloys_do_not_close_at_4gpa_even_quasi_statically():
    ti = po.assess_pore_closure("ti6al4v", 4e9)
    assert ti.flow_stress_quasi_static == 880e6 and ti.closure_pressure_quasi_static == pytest.approx(5.403e9, rel=1e-3)
    assert ti.residual_porosity_quasi_static == pytest.approx(1.094e-3, rel=1e-3) and ti.closed_quasi_static is False
    assert ti.closure_status is Status.INFEASIBLE and ti.status is Status.INFEASIBLE
    assert any("shrink but do not close" in x and "5.4 GPa" in x for x in ti.warnings)
    ni = po.assess_pore_closure("inconel_718", 4e9)
    assert ni.closure_pressure_quasi_static == pytest.approx(6.754e9, rel=1e-3)
    assert ni.residual_porosity_quasi_static == pytest.approx(4.277e-3, rel=1e-3) and ni.collapses_dynamic is False
    assert ni.closure_status is Status.INFEASIBLE
    for a in (ti, ni):
        assert a.below_dbtt is None and a.dbtt is None and a.hip is po.HIP_PRACTICE[a.material]
    # ODS FeCrAl: closes on the optimistic reading only, with an extrapolated (fallback) flow stress
    ods = po.assess_pore_closure("ods_fecral", 4e9)
    assert ods.closed_quasi_static is True and ods.closed_dynamic is False and ods.collapses_dynamic is True
    assert ods.closure_status is Status.EXTRAPOLATION and ods.extrapolated_model is True and ods.hip is None
    assert any("0.0862 % porosity remains" in x for x in ods.warnings)
    assert any("flow-stress model extrapolated" in x for x in ods.warnings)
    assert any("ODS alloys are made by mechanical alloying" in x for x in ods.notes)
    assert any("no cycle is tabulated" in x for x in ods.notes)


@pytest.mark.parametrize("key", sorted(po.HIP_PRACTICE))
def test_hip_conditions_are_state_of_the_art(key):
    cycle = po.HIP_PRACTICE[key]
    a = po.assess_pore_closure(key, cycle.pressure, temperature=cycle.temperature, pulse_duration=cycle.time)
    assert a.matches_hip_practice is True
    assert a.status is a.closure_status is a.healing_status is Status.STATE_OF_THE_ART
    assert a.margin == pytest.approx(cycle.pressure / 100e6)
    assert any("match industrial HIP practice" in x and "creep" in x for x in a.notes)
    assert any("longer than any magnetic pressure pulse" in x for x in a.notes)
    # time-independent yielding alone would NOT close the pores at HIP pressure: HIP works by creep
    assert a.collapses_quasi_static is False and a.creep_regime is True
    assert a.pressure < 0.05 * 4e9                       # 20-40 times below the concept note's 4 GPa


def test_hot_but_not_hip_practice():
    # the HIP temperature and pressure for only one hour: creep closure is plausible but not tabulated
    short = po.assess_pore_closure("gamma_tial", TIAL_HIP.pressure, temperature=TIAL_HIP.temperature,
                                   pulse_duration=3600.0)
    assert short.matches_hip_practice is False and short.creep_regime is True
    assert short.closure_status is Status.EXTRAPOLATION and short.healing_status is Status.STATE_OF_THE_ART
    assert short.status is Status.EXTRAPOLATION
    assert any("creep closure is not modelled" in x for x in short.warnings)
    # a hot 0.1 ms pulse closes the pores (soft matrix) but still cannot heal them: bonding needs ~1.7 s
    pulse = po.assess_pore_closure("gamma_tial", 4e9, temperature=TIAL_HIP.temperature)
    assert pulse.closed_dynamic is True and pulse.below_dbtt is False
    assert pulse.closure_status is Status.EXTRAPOLATION and pulse.extrapolated_model is True   # fit ends at 773 K
    assert pulse.diffusion_bond_time == pytest.approx(1.663, rel=1e-2)
    assert pulse.healing_status is Status.INFEASIBLE and pulse.status is Status.INFEASIBLE
    assert not any("below the DBTT" in x for x in pulse.warnings)


def test_healing_status_ladder_and_uncertainty_band():
    t_hip = TIAL_HIP.temperature
    t_bond = po.diffusion_bonding_time("gamma_tial", t_hip)
    expected = {20.0: (Status.STATE_OF_THE_ART, True), 2.0: (Status.EXTRAPOLATION, True),
                0.5: (Status.EXTRAPOLATION, False), 0.05: (Status.INFEASIBLE, False)}
    for multiple, (status, bonded) in expected.items():
        a = po.assess_pore_closure("gamma_tial", 4e9, temperature=t_hip, pulse_duration=multiple * t_bond,
                                   strain_rate=1e4)
        assert a.healing_status is status and a.bonded is bonded, multiple
        assert a.matches_hip_practice is False
    assert any("inside the uncertainty of the diffusion estimate" in x
               for x in po.assess_pore_closure("gamma_tial", 4e9, temperature=t_hip, pulse_duration=t_bond).warnings)
    # the interface scale matters only when hot: at room temperature nothing changes the verdict
    wide = po.assess_pore_closure("gamma_tial", 4e9, bond_length=1e-9)
    assert wide.bond_length == 1e-9 and wide.diffusion_bond_time > 1e30 and wide.healing_status is Status.INFEASIBLE
    # no activation energy recorded: healing not assessed, status from closure alone
    mar = po.assess_pore_closure("maraging_c350", 20e9)
    assert mar.healing_status is Status.NOT_APPLICABLE and mar.diffusion_bond_time is None and mar.bonded is None
    assert mar.status is mar.closure_status and any("no self-diffusion data" in x for x in mar.warnings)


def test_ductile_metal_closes_cold_but_heals_only_hot():
    cold = po.assess_pore_closure("copper_ofhc", 1e9)
    assert cold.closed_dynamic is True and cold.closure_status is Status.STATE_OF_THE_ART
    assert cold.healing_status is Status.INFEASIBLE and cold.status is Status.INFEASIBLE
    assert cold.margin > 2.0 and cold.below_dbtt is None
    assert [x for x in cold.warnings] == [x for x in cold.warnings if "closed but not healed" in x]
    # hot pressing for an hour both closes and heals: the module can say yes when the physics does
    hot = po.assess_pore_closure("copper_ofhc", 1e9, temperature=1000.0, pulse_duration=3600.0)
    assert hot.status is hot.closure_status is hot.healing_status is Status.STATE_OF_THE_ART
    assert hot.warnings == () and hot.bonded is True and hot.strain_rate == pytest.approx(1.0 / 3600.0)
    assert hot.diffusion_bond_time == pytest.approx(1.95, rel=1e-2)
    assert any("gas or mechanical press" in x for x in hot.notes)        # not a magnetic pulse


def test_gas_pores_and_already_closed_material():
    gas = po.assess_pore_closure("gamma_tial", 4e9, gas_pressure=P_ATM)
    vac = po.assess_pore_closure("gamma_tial", 4e9)
    assert gas.gas_pressure == P_ATM and gas.gas_limited_porosity == pytest.approx(2.56e-7, rel=1e-2)
    assert gas.collapse_pressure_quasi_static == pytest.approx(vac.collapse_pressure_quasi_static + P_ATM)
    assert gas.closure_pressure_quasi_static > vac.closure_pressure_quasi_static
    assert gas.residual_porosity_quasi_static == pytest.approx(2.090e-5, rel=1e-3)
    assert gas.residual_porosity_quasi_static > vac.residual_porosity_quasi_static
    assert any("re-expands" in x for x in gas.notes) and any("Vacuum (shrinkage) pore" in x for x in vac.notes)
    dense = po.assess_pore_closure("gamma_tial", 4e9, initial_porosity=5e-5)
    assert dense.closure_status is Status.STATE_OF_THE_ART and dense.wall_strain == 0.0
    assert dense.closure_pressure_dynamic == 0.0 and dense.closed_dynamic is True
    assert dense.healing_status is Status.INFEASIBLE                       # nothing heals in a cold pulse
    assert any("already at or below" in x for x in dense.notes)


def test_delivery_check_is_reported_but_not_graded(tial):
    plain = po.assess_pore_closure(tial, 4e9)
    assert plain.skin_depth is None and plain.transparent is None and plain.pressure_transfer is None
    thin = po.assess_pore_closure(tial, 4e9, section_thickness=2e-3, drive_frequency=1e4)
    delta = skin_depth_for_material(tial, 1e4)
    assert thin.skin_depth == pytest.approx(delta) and delta == pytest.approx(4.21e-3, rel=1e-2)   # poor conductor
    assert thin.transparent is True and thin.pressure_transfer == pytest.approx(1.0 - math.exp(-4e-3 / delta))
    assert thin.pressure_transfer == pytest.approx(0.613, rel=1e-2)
    assert any("transparent to the pulse" in x and "driver or can" in x for x in thin.warnings)
    assert thin.status is plain.status and thin.closure_status is plain.closure_status      # reported, not graded
    thick = po.assess_pore_closure(tial, 4e9, section_thickness=20e-3, drive_frequency=1e4)
    assert thick.transparent is False and thick.pressure_transfer > 0.999
    assert not any("transparent" in x for x in thick.warnings)
    # default drive frequency 1/(2 pulse) when only a section is given; skin depth alone when only a frequency is
    auto = po.assess_pore_closure(tial, 4e9, section_thickness=2e-3)
    assert auto.drive_frequency == pytest.approx(5e3) and auto.skin_depth == pytest.approx(delta * math.sqrt(2.0))
    freq_only = po.assess_pore_closure(tial, 4e9, drive_frequency=1e4)
    assert freq_only.skin_depth == pytest.approx(delta) and freq_only.transparent is None
    # a good conductor of the same section shields: copper skin depth 0.66 mm at 10 kHz
    cu = po.assess_pore_closure("copper_ofhc", 1e9, section_thickness=2e-3, drive_frequency=1e4)
    assert cu.transparent is False and cu.skin_depth == pytest.approx(0.658e-3, rel=1e-2)


def test_short_pulse_and_extrapolated_model_flags():
    a = po.assess_pore_closure("copper_ofhc", 1e9, pore_radius=5e-3, pulse_duration=1e-7)
    assert a.strain_rate == pytest.approx(1e7) and a.closed_dynamic is True
    assert a.collapse_time == pytest.approx(1.749e-5, rel=1e-3) and a.collapse_time > a.pulse_duration
    assert a.closure_status is Status.EXTRAPOLATION and a.extrapolated_model is True
    assert any("shorter than the inertial collapse time" in x for x in a.warnings)
    assert any("flow-stress model extrapolated" in x and "strain rate" in x for x in a.warnings)
    # an explicit strain rate overrides the 1/pulse default
    b = po.assess_pore_closure("copper_ofhc", 1e9, pulse_duration=1e-5, strain_rate=1e3)
    assert b.strain_rate == 1e3 and b.extrapolated_model is False and b.closure_status is Status.STATE_OF_THE_ART


# ------------------------------------------------------------------ powder densification
def test_powder_yield_pressure_two_stages():
    s, d0 = 1.0e9, 0.64
    for d in (0.64, 0.7, 0.8, 0.85, 0.89):
        assert po.powder_yield_pressure(d, s) == pytest.approx(3.0 * d * d * (d - d0) / (1.0 - d0) * s, rel=1e-12)
    for d in (0.9, 0.95, 0.99, 0.9999):
        assert po.powder_yield_pressure(d, s) == pytest.approx((2.0 / 3.0) * s * math.log(1.0 / (1.0 - d)), rel=1e-12)
        assert po.powder_yield_pressure(d, s) == pytest.approx(po.plastic_collapse_pressure(s, 1.0 - d), rel=1e-12)
    assert po.powder_yield_pressure(0.64, s) == 0.0 and po.powder_yield_pressure(1.0, s) == math.inf
    assert po.powder_yield_pressure(0.8, s) == pytest.approx(0.8533 * s, rel=1e-4)
    assert po.powder_yield_pressure(0.95, s) == pytest.approx(1.9972 * s, rel=1e-4)
    assert po.powder_yield_pressure(0.99, s) == pytest.approx(3.0701 * s, rel=1e-4)
    # the two stages do not join at D = 0.9 (documented): 1.755 vs 1.535 sigma_y
    assert po.powder_yield_pressure(0.9 - 1e-12, s) == pytest.approx(1.755 * s, rel=1e-6)
    assert po.powder_yield_pressure(0.9, s) == pytest.approx(1.5351 * s, rel=1e-4)
    # monotone within each stage, linear in the flow stress, looser packing is easier at a given density
    stage1 = [po.powder_yield_pressure(d, s) for d in (0.64, 0.7, 0.8, 0.89)]
    stage2 = [po.powder_yield_pressure(d, s) for d in (0.9, 0.95, 0.99, 0.999)]
    assert all(b > a for a, b in zip(stage1, stage1[1:])) and all(b > a for a, b in zip(stage2, stage2[1:]))
    assert po.powder_yield_pressure(0.8, 2 * s) == pytest.approx(2 * po.powder_yield_pressure(0.8, s))
    assert po.powder_yield_pressure(0.8, s, 0.5) > po.powder_yield_pressure(0.8, s, 0.7)
    assert po.powder_yield_pressure(0.95, s, 0.5) == po.powder_yield_pressure(0.95, s, 0.7)   # stage 2 forgets D0


def test_powder_density_at_pressure_is_the_inverse_and_monotone():
    s = 1.0e9
    assert po.powder_density_at_pressure(0.0, s) == 0.64
    assert po.powder_density_at_pressure(1.0, 0.0) == 1.0
    for p in (0.1e9, 0.5e9, 1.0e9, 1.5e9, 1.75e9):                        # stage 1 round trip
        d = po.powder_density_at_pressure(p, s)
        assert 0.64 < d < 0.9 and po.powder_yield_pressure(d, s) == pytest.approx(p, rel=1e-9)
    for p in (1.76e9, 2.0e9, 3.07e9, 6.14e9):                             # stage 2 round trip
        d = po.powder_density_at_pressure(p, s)
        assert d == pytest.approx(1.0 - math.exp(-1.5 * p / s), rel=1e-12)
        assert po.powder_yield_pressure(d, s) == pytest.approx(p, rel=1e-9)
    # the documented step where the stages meet: 0.900 -> 0.928 at 1.755 sigma_y
    assert po.powder_density_at_pressure(1.7549e9, s) == pytest.approx(0.89999, abs=2e-5)
    assert po.powder_density_at_pressure(1.7551e9, s) == pytest.approx(0.9281, abs=1e-4)
    assert po.powder_density_at_pressure(50e9, s) == 1.0
    grid = [po.powder_density_at_pressure(p * 1e7, s) for p in range(0, 800)]
    assert all(b >= a for a, b in zip(grid, grid[1:])) and grid[0] == 0.64 and grid[-1] > 0.9999
    # pre-compacted powder (D0 = 0.8): plateau at 0.9 until stage 2 can continue (1.215-1.535 sigma_y)
    assert po.powder_density_at_pressure(1.3e9, s, 0.8) == 0.9 == po.powder_density_at_pressure(1.5e9, s, 0.8)
    assert po.powder_density_at_pressure(1.6e9, s, 0.8) == pytest.approx(1.0 - math.exp(-2.4), rel=1e-12)
    # D0 already beyond the stage transition: isolated pores only
    assert po.powder_density_at_pressure(1.0e9, s, 0.95) == 0.95
    assert po.powder_density_at_pressure(2.5e9, s, 0.95) == pytest.approx(1.0 - math.exp(-3.75), rel=1e-12)


def test_powder_compaction_statuses():
    fe = po.assess_powder_compaction("iron_armco", 1.6e9)
    assert isinstance(fe, Assessment) and fe.status is Status.STATE_OF_THE_ART and fe.warnings == ()
    assert fe.density_dynamic == pytest.approx(0.99985, abs=2e-5) and fe.density_quasi_static > 0.9999
    assert fe.within_dmc_class is True and fe.sintering_required is True and fe.below_dbtt is False
    assert fe.field_equivalent == pytest.approx(math.sqrt(2.0 * MU0 * 1.6e9))
    assert fe.field_equivalent == pytest.approx(63.41, rel=1e-4)
    assert fe.margin == pytest.approx(1.6e9 / fe.pressure_for_target_dynamic) and fe.margin > 2.5
    assert fe.pressure_for_target_dynamic == pytest.approx(po.powder_yield_pressure(0.95, fe.flow_stress_dynamic))
    assert fe.full_density_pressure_dynamic == pytest.approx(6.1402 * fe.flow_stress_dynamic, rel=1e-4)
    assert any("green compact" in x and "sintering or HIP must follow" in x for x in fe.notes)
    assert any("moving solid part" in x for x in fe.notes)
    assert fe.diffusion_bond_time > 1e30
    # gamma-TiAl powder: 98.6-100 % at 4 GPa but beyond the demonstrated class and brittle
    tial = po.assess_powder_compaction("gamma_tial", 4e9)
    assert tial.status is Status.EXTRAPOLATION and tial.within_dmc_class is False and tial.below_dbtt is True
    assert tial.density_dynamic == pytest.approx(0.9862, abs=2e-4) and tial.density_quasi_static > 0.9999
    assert any("above the 2 GPa class" in x for x in tial.warnings)
    assert any("fracture and rearrange" in x for x in tial.warnings)
    # at 1.6 GPa the target is met only at the quasi-static strength
    low = po.assess_powder_compaction("gamma_tial", 1.6e9)
    assert low.density_quasi_static == pytest.approx(0.9873, abs=2e-4) and low.density_dynamic == pytest.approx(0.836, abs=2e-3)
    assert low.status is Status.EXTRAPOLATION and low.margin < 1.0 and low.within_dmc_class is True
    assert any("quasi-static strength only" in x for x in low.warnings)
    # a strong superalloy powder at 1.6 GPa misses the target even quasi-statically
    ni = po.assess_powder_compaction("inconel_718", 1.6e9)
    assert ni.status is Status.INFEASIBLE and ni.density_quasi_static < 0.95
    assert any("pressure insufficient for the target green density" in x for x in ni.warnings)
    # a looser target and an explicit strain rate; a target at the initial density needs no pressure
    easy = po.assess_powder_compaction("inconel_718", 1.6e9, target_density=0.8, strain_rate=1e-3)
    assert easy.status is Status.STATE_OF_THE_ART and easy.flow_stress_dynamic == pytest.approx(1.126e9, rel=1e-3)
    none = po.assess_powder_compaction("copper_ofhc", 0.0, target_density=0.6)
    assert none.pressure_for_target_dynamic == 0.0 and none.margin == math.inf and none.density_dynamic == 0.64
    # hot and long enough the particles do bond (hot pressing), so no separate sintering step
    hot = po.assess_powder_compaction("copper_ofhc", 0.2e9, temperature=1000.0, pulse_duration=3600.0)
    assert hot.sintering_required is False and hot.status is Status.STATE_OF_THE_ART
    assert po.assess_powder_compaction("maraging_c350", 4e9).sintering_required is None


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("call", [
    lambda: po.plastic_collapse_pressure(-1.0, 0.01),
    lambda: po.plastic_collapse_pressure(1e9, -0.1),
    lambda: po.plastic_collapse_pressure(1e9, 1.5),
    lambda: po.plastic_collapse_pressure(float("nan"), 0.01),
    lambda: po.plastic_collapse_pressure("strong", 0.01),
    lambda: po.porosity_under_pressure(1e9, -1.0, 0.01),
    lambda: po.porosity_under_pressure(1e9, 1e9, 0.0),
    lambda: po.porosity_under_pressure(1e9, 1e9, 1.0),
    lambda: po.porosity_under_pressure(1e9, 1e9, 0.01, gas_pressure=-1.0),
    lambda: po.gas_limited_porosity(0.01, -1.0, 1e9),
    lambda: po.gas_limited_porosity(1.0, 1e5, 1e9),
    lambda: po.pore_wall_strain(0.0, 0.0),
    lambda: po.pore_wall_strain(0.01, -1e-3),
    lambda: po.rayleigh_collapse_time(0.0, "gamma_tial", 1e9),
    lambda: po.rayleigh_collapse_time(1e-5, "gamma_tial", float("inf")),
    lambda: po.matrix_strength("gamma_tial", -1.0),
    lambda: po.matrix_strength("gamma_tial", 1e4, 0.0),
    lambda: po.matrix_strength("gamma_tial", 1e4, 2000.0),
    lambda: po.matrix_strength("diamond"),
    lambda: po.collapse_pressure("gamma_tial", 1.5),
    lambda: po.residual_porosity("gamma_tial", -1.0, 0.01),
    lambda: po.self_diffusivity("gamma_tial", 0.0),
    lambda: po.self_diffusivity("gamma_tial", 1733.0),
    lambda: po.self_diffusivity("gamma_tial", 300.0, prefactor=0.0),
    lambda: po.diffusion_length("gamma_tial", 300.0, -1.0),
    lambda: po.diffusion_bonding_time("gamma_tial", 300.0, 0.0),
    lambda: po.matches_hip_practice("gamma_tial", -1.0, 1500.0, 1.0),
    lambda: po.matches_hip_practice("gamma_tial", 1e8, 1500.0, 0.0),
    lambda: po.assess_pore_closure("gamma_tial", -1.0),
    lambda: po.assess_pore_closure("gamma_tial", "high"),
    lambda: po.assess_pore_closure("gamma_tial", True),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, initial_porosity=0.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, initial_porosity=1.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, pore_radius=0.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, temperature=5000.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, pulse_duration=0.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, strain_rate=-1.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, gas_pressure=-1.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, section_thickness=0.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, drive_frequency=-1.0),
    lambda: po.assess_pore_closure("gamma_tial", 4e9, bond_length=0.0),
    lambda: po.assess_pore_closure("diamond", 4e9),
    lambda: po.powder_yield_pressure(0.5, 1e9),
    lambda: po.powder_yield_pressure(1.1, 1e9),
    lambda: po.powder_yield_pressure(0.8, -1.0),
    lambda: po.powder_yield_pressure(0.8, 1e9, initial_density=1.0),
    lambda: po.powder_density_at_pressure(-1.0, 1e9),
    lambda: po.powder_density_at_pressure(1e9, 1e9, initial_density=0.0),
    lambda: po.assess_powder_compaction("iron_armco", -1.0),
    lambda: po.assess_powder_compaction("iron_armco", 1e9, target_density=1.0),
    lambda: po.assess_powder_compaction("iron_armco", 1e9, initial_density=1.2),
    lambda: po.assess_powder_compaction("iron_armco", 1e9, pulse_duration=-1.0),
    lambda: po.assess_powder_compaction("iron_armco", 1e9, temperature=0.0),
    lambda: po.assess_powder_compaction("diamond", 1e9),
])
def test_invalid_inputs_raise_input_error(call):
    with pytest.raises(InputError):
        call()


def test_unknown_material_raises():
    for call in (lambda: po.assess_pore_closure("unobtainium", 4e9),
                 lambda: po.assess_powder_compaction("unobtainium", 1e9),
                 lambda: po.matrix_strength("unobtainium"),
                 lambda: po.hip_cycle("unobtainium"),
                 lambda: po.rayleigh_collapse_time(1e-5, "unobtainium", 1e9),
                 lambda: po.self_diffusivity("unobtainium", 300.0)):
        with pytest.raises(UnknownMaterialError):
            call()


# ------------------------------------------------------------------ serialisation
def test_results_round_trip_json_and_render_text(tial):
    a = po.assess_pore_closure(tial, 4e9, section_thickness=2e-3, drive_frequency=1e4)
    by_alias = po.assess_pore_closure("TiAl", 4e9, section_thickness=2e-3, drive_frequency=1e4)
    assert a.json() == by_alias.json()                       # deterministic, independent of how the material is given
    data = json.loads(a.json())
    assert data["material"] == "gamma_tial" and data["status"] == "infeasible"
    assert data["closure_status"] == "extrapolation" and data["healing_status"] == "infeasible"
    assert data["margin"] == pytest.approx(a.margin) and data["collapses_dynamic"] is False
    assert data["hip"]["pressure"] == 172e6 and data["hip"]["confidence"] == "representative"
    assert data["hip"]["temperature_window"] == pytest.approx([1458.15, 1533.15])
    assert data["diffusion_bond_time"] == pytest.approx(a.diffusion_bond_time) and data["transparent"] is True
    text = a.text()
    assert text.startswith("Pore-closure assessment") and "material = gamma_tial" in text
    assert "collapse_pressure_dynamic [Pa] = " in text and "hip:" in text and "closure_status = extrapolation" in text
    # non-finite values serialise as null so the JSON stays strict
    none = json.loads(po.assess_pore_closure(tial, 1.0e9).json())
    assert none["collapse_time"] is None and none["skin_depth"] is None
    powder = po.assess_powder_compaction("copper_ofhc", 0.0, target_density=0.6)
    assert json.loads(powder.json())["margin"] is None
    for result, title in ((po.matrix_strength(tial), "Matrix strength"),
                          (po.assess_powder_compaction("iron_armco", 1.6e9), "Powder-compaction assessment"),
                          (TIAL_HIP, "HipCycle")):
        d = json.loads(result.json())
        assert isinstance(d, dict) and d["material"] in ("gamma_tial", "iron_armco")
        assert result.text().startswith(title)
    with pytest.raises(Exception):
        a.pressure = 1.0  # frozen
    with pytest.raises(Exception):
        TIAL_HIP.pressure = 1.0
