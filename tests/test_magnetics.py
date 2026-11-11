"""Tests for magform.magnetics: magnetic pressure, solenoid fields, energy and winding stresses."""

from __future__ import annotations

import json
import math

import pytest

from magform import magnetics as mg
from magform._numerics import linspace, trapz
from magform.constants import MU0
from magform.errors import InputError


# --------------------------------------------------------------- pressure
@pytest.mark.parametrize("pressure, field", [(1.6e9, 63.41), (2.0e9, 70.90), (4.0e9, 100.27)])
def test_field_for_pressure_spec_checks(pressure, field):
    assert mg.field_for_pressure(pressure) == pytest.approx(field, rel=1e-3)
    assert mg.magnetic_pressure(mg.field_for_pressure(pressure)) == pytest.approx(pressure, rel=1e-12)
    assert mg.energy_density(field) == mg.magnetic_pressure(field)
    assert mg.magnetic_pressure(-field) == mg.magnetic_pressure(field)


def test_pressure_inputs_are_validated():
    with pytest.raises(InputError):
        mg.field_for_pressure(-1.0)
    with pytest.raises(InputError):
        mg.magnetic_pressure(float("nan"))
    with pytest.raises(InputError):
        mg.magnetic_pressure(math.inf)
    with pytest.raises(InputError):
        mg.pulse_increment_for_pressure(-1e9, 10.0)
    assert mg.field_for_pressure(0.0) == 0.0


def test_static_background_exerts_no_pressure():
    # a DC field alone, however strong, pushes nothing
    for b_dc in (0.0, 10.0, 35.6, 100.0, -20.0):
        assert mg.pressure_with_background(b_dc, 0.0, 0.0) == 0.0
    # equal increments on both sides (field fully diffused) also give zero
    assert mg.pressure_with_background(30.0, 5.0, 5.0) == 0.0
    # no background: reduces to B^2/(2 mu0) for a fully screening shell
    assert mg.pressure_with_background(0.0, 100.27) == pytest.approx(mg.magnetic_pressure(100.27))
    # inside field larger than outside -> net force towards the coil
    assert mg.pressure_with_background(10.0, 1.0, 2.0) < 0.0


def test_background_reduces_pulse_amplitude_not_total_field():
    p = 4e9
    b_full = mg.field_for_pressure(p)
    assert mg.pulse_increment_for_pressure(p, 0.0) == pytest.approx(b_full, rel=1e-12)
    for b_dc in (10.0, 30.0, 45.5):
        db = mg.pulse_increment_for_pressure(p, b_dc)
        assert db < b_full
        assert mg.pressure_with_background(b_dc, db) == pytest.approx(p, rel=1e-12)
        # the total gap field is larger than the background-free field
        assert b_dc + db > b_full
    # monotone: more background, smaller increment
    assert mg.pulse_increment_for_pressure(p, 30.0) < mg.pulse_increment_for_pressure(p, 10.0)


def test_current_sheet_pressure_consistency():
    b_out = 63.41
    k = b_out / MU0              # sheet that screens b_out completely
    mean_field = 0.5 * b_out     # mean of b_out and 0 on the two faces
    assert mg.current_sheet_pressure(k, mean_field) == pytest.approx(mg.magnetic_pressure(b_out), rel=1e-12)
    assert mg.surface_current_for_pressure(2e9, 50.0) == pytest.approx(4e7)
    assert mg.current_sheet_pressure(mg.surface_current_for_pressure(2e9, 50.0), 50.0) == pytest.approx(2e9)
    assert mg.surface_current_for_pressure(1e9, 0.0) == math.inf
    assert mg.surface_current_for_pressure(0.0, 0.0) == 0.0


# --------------------------------------------------------------- stresses
def test_max_hoop_stress_ratio_branches_and_limits():
    assert mg.max_hoop_stress_ratio(1.5) == pytest.approx(4.0)            # 2/(alpha-1)
    assert mg.max_hoop_stress_ratio(2.0) == pytest.approx(2.0)
    assert mg.max_hoop_stress_ratio(2.0 + 1e-9) == pytest.approx(2.0, rel=1e-6)  # continuous at alpha = 2
    assert mg.max_hoop_stress_ratio(5.0) == pytest.approx(25.0 / 32.0)
    assert mg.max_hoop_stress_ratio(1e6) == pytest.approx(0.5, rel=1e-5)  # never below P/2
    a_star = 2.0 + math.sqrt(2.0)                                          # sigma_max = P here
    assert mg.max_hoop_stress_ratio(a_star) == pytest.approx(1.0, rel=1e-12)
    assert mg.max_hoop_stress_ratio(a_star * 0.99) > 1.0 > mg.max_hoop_stress_ratio(a_star * 1.01)
    ratios = [mg.max_hoop_stress_ratio(a) for a in (1.2, 1.8, 2.5, 4.0, 10.0, 100.0)]
    assert all(x > y for x, y in zip(ratios, ratios[1:]))
    with pytest.raises(InputError):
        mg.max_hoop_stress_ratio(1.0)


def test_thin_shell_and_virial_bounds():
    assert mg.thin_shell_hoop_stress(4e9, 0.05, 0.01) == pytest.approx(2e10)
    with pytest.raises(InputError):
        mg.thin_shell_hoop_stress(4e9, 0.05, 0.0)
    with pytest.raises(InputError):
        mg.thin_shell_hoop_stress(-1.0, 0.05, 0.01)
    # 100 MJ in a 2000 kg/m^3 structure at 1 GPa needs >= 200 kg
    assert mg.virial_minimum_mass(1e8, 2000.0, 1e9) == pytest.approx(200.0)
    assert mg.virial_mean_stress(1e8, 0.1) == pytest.approx(1e9)
    with pytest.raises(InputError):
        mg.virial_minimum_mass(1e8, 2000.0, 0.0)
    with pytest.raises(InputError):
        mg.virial_mean_stress(1e8, -1.0)


def test_graded_winding_thickness():
    # constant J: Delta = B/(mu0 J) exactly
    assert mg.graded_winding_thickness(20.0, lambda b: 4e8) == pytest.approx(20.0 / (MU0 * 4e8), rel=1e-12)
    # J = J0/(1 + b/B1): Delta = (B + B^2/(2 B1))/(mu0 J0) (midpoint rule exact for linear 1/J)
    j0, b1, b = 5e8, 25.0, 60.0
    expect = (b + b * b / (2.0 * b1)) / (MU0 * j0)
    assert mg.graded_winding_thickness(b, lambda x: j0 / (1.0 + x / b1)) == pytest.approx(expect, rel=1e-9)
    # J = J0 (1 + b/B1)^-1/2 -> analytic (2 B1/3)((1+B/B1)^1.5 - 1)/(mu0 J0), midpoint error O(n^-2)
    expect2 = (2.0 * b1 / 3.0) * ((1.0 + b / b1) ** 1.5 - 1.0) / (MU0 * j0)
    got2 = mg.graded_winding_thickness(b, lambda x: j0 / math.sqrt(1.0 + x / b1))
    assert got2 == pytest.approx(expect2, rel=1e-6)
    # a current density that reaches zero inside the range -> no finite winding
    assert mg.graded_winding_thickness(100.0, lambda x: 1e9 * max(0.0, 1.0 - x / 80.0)) == math.inf
    assert mg.graded_winding_thickness(0.0, lambda x: 1e9) == 0.0
    with pytest.raises(InputError):
        mg.graded_winding_thickness(10.0, lambda x: 1e9, n=0)


def test_dipole_far_field_matches_solenoid_on_axis():
    s = mg.Solenoid(0.05, 0.15, 0.30)
    j = 1e8
    m = s.dipole_moment(j)
    assert m == pytest.approx(math.pi * j * 0.30 * (0.15 ** 3 - 0.05 ** 3) / 3.0)
    # on the axis the dipole field is twice the equatorial value
    for z, tol in ((5.0, 2e-3), (50.0, 1e-4)):
        assert s.axial_field(z, j) == pytest.approx(2.0 * mg.dipole_field_on_equator(m, z), rel=tol)
    with pytest.raises(InputError):
        mg.dipole_field_on_equator(m, 0.0)


# --------------------------------------------------------------- solenoid
def test_solenoid_validation():
    for args in ((0.0, 0.1, 0.1), (0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.05, 0.1, 0.0), (0.05, 0.1, -1.0),
                 (-0.05, 0.1, 0.1), (0.05, math.nan, 0.1)):
        with pytest.raises(InputError):
            mg.Solenoid(*args)


def test_solenoid_geometry_properties():
    s = mg.Solenoid(0.05, 0.25, 0.5)
    assert s.winding_thickness == pytest.approx(0.20)
    assert s.mean_radius == pytest.approx(0.15)
    assert s.alpha == pytest.approx(5.0)
    assert s.beta == pytest.approx(5.0)
    assert s.winding_volume == pytest.approx(math.pi * (0.25 ** 2 - 0.05 ** 2) * 0.5)
    assert s.bore_volume == pytest.approx(math.pi * 0.05 ** 2 * 0.5)


@pytest.mark.parametrize("beta, f", [(0.5, 0.33), (1.0, 0.56), (2.0, 0.80), (3.0, 0.89), (5.0, 0.96)])
def test_field_factor_table(beta, f):
    a1 = 0.05
    s = mg.Solenoid(a1, 2 * a1, 2 * beta * a1)    # alpha = 2
    assert s.field_factor() == pytest.approx(f, rel=2e-2)


def test_field_factor_long_coil_limit_and_monotone():
    long_coil = mg.Solenoid(0.05, 0.10, 1000.0)
    assert long_coil.field_factor() == pytest.approx(1.0, rel=1e-4)
    j = 3e8
    assert long_coil.central_field(j) == pytest.approx(MU0 * j * 0.05, rel=1e-4)   # mu0 J Delta
    fs = [mg.Solenoid(0.05, 0.10, length).field_factor() for length in (0.05, 0.1, 0.3, 1.0, 3.0)]
    assert all(x < y for x, y in zip(fs, fs[1:]))


def test_current_relations_are_mutual_inverses():
    s = mg.Solenoid(0.05, 0.15, 0.30)
    b0 = 63.41
    j = s.current_density_for_field(b0)
    assert s.central_field(j) == pytest.approx(b0, rel=1e-12)
    # Montgomery J of a short coil exceeds the long-coil J = B0/(mu0 Delta)
    assert j == pytest.approx(b0 / (MU0 * s.winding_thickness * s.field_factor()), rel=1e-12)
    assert j > b0 / (MU0 * s.winding_thickness)
    n_turns = 400
    i = s.current_for_field(b0, n_turns)
    assert i == pytest.approx(j * s.length * s.winding_thickness / n_turns, rel=1e-12)
    assert s.field_from_current(i, n_turns) == pytest.approx(b0, rel=1e-12)
    with pytest.raises(InputError):
        s.current_for_field(b0, 0)
    with pytest.raises(InputError):
        s.central_field(-1.0)


def test_axial_field_profile():
    s = mg.Solenoid(0.05, 0.15, 0.30)
    j = 2e8
    assert s.axial_field(0.0, j) == pytest.approx(s.central_field(j), rel=1e-9)
    zs = [0.0, 0.03, 0.08, 0.15, 0.3, 1.0]
    bz = [s.axial_field(z, j) for z in zs]
    assert all(x > y for x, y in zip(bz, bz[1:]))                       # falls away from the centre
    assert s.axial_field(-0.08, j) == pytest.approx(s.axial_field(0.08, j), rel=1e-12)
    # long thin coil: the field at the end of the winding is half the centre field
    lc = mg.Solenoid(0.05, 0.051, 50.0)
    assert lc.axial_field(25.0, j) / lc.axial_field(0.0, j) == pytest.approx(0.5, rel=1e-3)


def test_pressure_uniformity_spec_check():
    s = mg.Solenoid(0.05, 0.15, 0.30)
    pu = s.pressure_uniformity(1e8, 0.15)
    assert pu.end_ratio == pytest.approx(0.84, rel=0.03)
    assert len(pu.z) == len(pu.ratio) == 41
    assert pu.ratio[20] == pytest.approx(1.0, rel=1e-12)                 # centre sample
    assert pu.end_ratio < pu.mean_ratio < 1.0
    assert pu.max_deviation == pytest.approx(1.0 - pu.end_ratio, rel=1e-12)
    # the ratio is independent of the current density
    assert s.pressure_uniformity(7e8, 0.15).end_ratio == pytest.approx(pu.end_ratio, rel=1e-12)
    # longer coil -> more uniform pressure over the same workpiece
    assert mg.Solenoid(0.05, 0.15, 0.60).pressure_uniformity(1e8, 0.15).end_ratio > pu.end_ratio
    with pytest.raises(InputError):
        s.pressure_uniformity(1e8, 0.31)
    with pytest.raises(InputError):
        s.pressure_uniformity(0.0, 0.15)


def test_winding_field_and_stored_energy():
    s = mg.Solenoid(0.05, 0.25, 0.5)
    b0 = 100.27
    assert s.winding_field(0.0, b0) == b0
    assert s.winding_field(0.05, b0) == b0
    assert s.winding_field(0.15, b0) == pytest.approx(b0 / 2.0)
    assert s.winding_field(0.25, b0) == 0.0
    assert s.winding_field(1.0, b0) == 0.0
    # closed form equals the numerical integral of B^2/(2 mu0) over the long-coil profile
    rs = linspace(0.0, 0.3, 30001)
    u = [mg.magnetic_pressure(s.winding_field(r, b0)) * 2.0 * math.pi * r * s.length for r in rs]
    assert s.stored_energy(b0) == pytest.approx(trapz(u, rs), rel=1e-6)
    # reported key number: ~99.5 MJ at 100.27 T (4 GPa)
    assert s.stored_energy(b0) == pytest.approx(9.9493e7, rel=1e-4)
    assert s.stored_energy(2 * b0) == pytest.approx(4 * s.stored_energy(b0), rel=1e-12)


def test_end_effect_warning():
    assert mg.Solenoid(0.05, 0.25, 0.5).end_effect_warning() is None            # beta = 5
    assert mg.Solenoid(0.05, 0.15, 0.30).end_effect_warning() is None           # beta = 3
    msg = mg.Solenoid(0.05, 0.15, 0.20).end_effect_warning()                   # beta = 2
    assert isinstance(msg, str) and "beta" in msg


def test_inductances():
    s = mg.Solenoid(0.05, 0.10, 0.60)          # beta = 6, alpha = 2: both methods agree
    n = 500
    am = 0.075
    assert s.inductance(n) == pytest.approx(31.50e-6 * am ** 2 * n ** 2 / (6 * am + 9 * 0.6 + 10 * 0.05), rel=2e-4)
    assert s.inductance(n) == pytest.approx(s.inductance_from_energy(n), rel=0.05)
    assert s.inductance(2 * n) == pytest.approx(4 * s.inductance(n), rel=1e-12)
    # energy-consistent inductance is independent of the field and scales as N^2
    assert s.inductance_from_energy(2 * n) == pytest.approx(4 * s.inductance_from_energy(n), rel=1e-12)
    # short coil (beta = 1): the two disagree by more than 10 % (fringing / end effects)
    short = mg.Solenoid(0.05, 0.10, 0.10)
    assert abs(short.inductance(n) / short.inductance_from_energy(n) - 1.0) > 0.10
    # 1/2 L I^2 recovers the stored energy
    b0 = 40.0
    i = s.current_for_field(b0, n)
    assert 0.5 * s.inductance_from_energy(n) * i ** 2 == pytest.approx(s.stored_energy(b0), rel=1e-12)


def test_max_hoop_stress_spec_check():
    a1, delta = 0.05, 0.20
    s = mg.Solenoid(a1, a1 + delta, 0.5)
    b0 = mg.field_for_pressure(4e9)
    j_long = b0 / (MU0 * delta)
    smax, r_at = s.max_hoop_stress(j_long, b0)
    assert smax == pytest.approx(3.125e9, rel=1e-2)
    assert r_at == pytest.approx(0.125)
    assert smax / 4e9 == pytest.approx(mg.max_hoop_stress_ratio(s.alpha), rel=1e-12)
    # same check through current_density_for_field for a (very) long coil
    lc = mg.Solenoid(a1, a1 + delta, 200.0)
    smax_lc, _ = lc.max_hoop_stress(lc.current_density_for_field(b0), b0)
    assert smax_lc == pytest.approx(3.125e9, rel=1e-2)
    # the Montgomery J of the finite coil is larger, hence a larger stress
    smax_m, _ = s.max_hoop_stress(s.current_density_for_field(b0), b0)
    assert smax_m > smax
    # thin coil (alpha <= 2): the maximum sits at the bore
    thin = mg.Solenoid(0.10, 0.15, 1.0)
    assert thin.max_hoop_stress(1e8, 10.0)[1] == pytest.approx(0.10)


def test_hoop_profile_and_force_balance_average():
    s = mg.Solenoid(0.05, 0.25, 0.5)
    b0 = mg.field_for_pressure(4e9)
    j = b0 / (MU0 * s.winding_thickness)
    prof = s.hoop_stress_profile(j, b0, n=401)
    assert len(prof.r) == len(prof.sigma) == 401
    assert prof.r[0] == s.inner_radius and prof.r[-1] == s.outer_radius
    assert max(prof.sigma) <= prof.max * (1.0 + 1e-12)
    assert max(prof.sigma) == pytest.approx(prof.max, rel=1e-4)
    assert prof.sigma[-1] == pytest.approx(0.0, abs=1e-3)
    # thickness average from radial force balance, P (a2 + 2 a1)/(3 Delta)
    avg = s.average_hoop_stress(b0)
    assert avg == pytest.approx(4e9 * (0.25 + 0.10) / (3 * 0.20), rel=1e-3)
    assert trapz(list(prof.sigma), list(prof.r)) / s.winding_thickness == pytest.approx(avg, rel=1e-5)
    # the virial value E/V is a lower bound, about half the force-balance average here
    assert mg.virial_mean_stress(s.stored_energy(b0), s.winding_volume) < avg
    with pytest.raises(InputError):
        s.hoop_stress_at(0.04, j, b0)
    with pytest.raises(InputError):
        s.hoop_stress_at(0.26, j, b0)


def test_results_json_round_trip_and_text():
    s = mg.Solenoid(0.05, 0.15, 0.30)
    pu = s.pressure_uniformity(1e8, 0.15)
    prof = s.hoop_stress_profile(1e8, 10.0)
    for result in (s, pu, prof):
        data = json.loads(result.json())
        assert isinstance(data, dict)
        txt = result.text()
        assert type(result).__name__ in txt
    assert json.loads(s.json()) == {"inner_radius": 0.05, "outer_radius": 0.15, "length": 0.30}
    d = json.loads(pu.json())
    assert d["end_ratio"] == pytest.approx(pu.end_ratio) and len(d["ratio"]) == 41
    d = json.loads(prof.json())
    assert d["max"] == pytest.approx(prof.max) and d["r_at"] == pytest.approx(prof.r_at)
    assert "end_ratio = " in pu.text() and "max [Pa] = " in prof.text()


def test_module_metadata():
    assert set(mg.__all__) <= set(dir(mg))
    assert mg.REFERENCES and all(isinstance(v, str) and v for v in mg.REFERENCES.values())
    assert mg.WHEELER_COEFFICIENT == pytest.approx(31.50e-6, rel=1e-3)
    for name in mg.__all__:
        obj = getattr(mg, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 40, name
