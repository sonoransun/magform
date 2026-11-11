"""Tests for magform.fieldmap: off-axis solenoid field and the pressure on a real workpiece."""

from __future__ import annotations

import json
import math

import pytest

from magform import fieldmap as fm
from magform.constants import MU0
from magform.errors import InputError
from magform.magnetics import Solenoid

REF = Solenoid(0.05, 0.25, 0.5)          # the reference press: 100 mm bore, 200 mm build, 0.5 m long
J100 = REF.current_density_for_field(100.27)


# ------------------------------------------------------------------ elliptic integrals
def test_ellipke_reference_values_and_legendre_relation():
    k0, e0 = fm.ellipke(0.0)
    assert k0 == pytest.approx(math.pi / 2, rel=1e-15) and e0 == pytest.approx(math.pi / 2, rel=1e-15)
    k, e = fm.ellipke(0.5)
    assert k == pytest.approx(1.8540746773013719, rel=1e-13)
    assert e == pytest.approx(1.3506438810476755, rel=1e-13)
    for m in (0.01, 0.3, 0.7, 0.99, 0.999999):
        k1, e1 = fm.ellipke(m)
        k2, e2 = fm.ellipke(1.0 - m)
        assert e1 * k2 + e2 * k1 - k1 * k2 == pytest.approx(math.pi / 2, rel=1e-11)  # Legendre's relation
    for bad in (-0.1, 1.0, 1.5, float("nan")):
        with pytest.raises(InputError):
            fm.ellipke(bad)


# ------------------------------------------------------------------ circular loop
def test_loop_field_axis_and_dipole_limits():
    a, i = 0.2, 1500.0
    for z in (0.0, 0.1, 0.5):
        br, bz = fm.loop_field(a, 0.0, z, i)
        assert br == 0.0
        assert bz == pytest.approx(MU0 * i * a * a / (2.0 * (a * a + z * z) ** 1.5), rel=1e-14)
    # far field: dipole of moment m = I pi a^2, on the equator B_z = -mu0 m / (4 pi r^3)
    m = i * math.pi * a * a
    _, bz = fm.loop_field(a, 20.0, 0.0, i)
    assert bz == pytest.approx(-MU0 * m / (4.0 * math.pi * 20.0 ** 3), rel=1e-3)
    # symmetry: B_r odd, B_z even in z
    br1, bz1 = fm.loop_field(a, 0.13, 0.07, i)
    br2, bz2 = fm.loop_field(a, 0.13, -0.07, i)
    assert br1 == pytest.approx(-br2) and bz1 == pytest.approx(bz2) and br1 > 0.0
    with pytest.raises(InputError):
        fm.loop_field(a, a, 0.0)
    with pytest.raises(InputError):
        fm.loop_field(-1.0, 0.1, 0.0)


@pytest.mark.parametrize("r, z", [(0.05, 0.02), (0.15, 0.1), (0.35, -0.2), (0.001, 0.3), (2.0, 1.0)])
def test_loop_flux_function_is_consistent_with_the_field(r, z):
    a, h = 0.2, 1e-6 * max(r, 0.01)
    br, bz = fm.loop_field(a, r, z)
    dpsi_dr = (fm.loop_flux_function(a, r + h, z) - fm.loop_flux_function(a, r - h, z)) / (2 * h)
    dpsi_dz = (fm.loop_flux_function(a, r, z + h) - fm.loop_flux_function(a, r, z - h)) / (2 * h)
    assert dpsi_dr / r == pytest.approx(bz, rel=2e-6, abs=1e-12)     # B_z = (1/r) d(psi)/dr
    assert -dpsi_dz / r == pytest.approx(br, rel=2e-6, abs=1e-12)    # B_r = -(1/r) d(psi)/dz


def test_loop_flux_function_small_parameter_series_is_continuous():
    # the series below m = 1e-3 meets the closed form above it
    lo = fm.loop_flux_function(1.0, 2.49e-4, 0.0) / 2.49e-4 ** 2
    hi = fm.loop_flux_function(1.0, 2.51e-4, 0.0) / 2.51e-4 ** 2
    assert lo == pytest.approx(hi, rel=2e-5)
    assert fm.loop_flux_function(1.0, 0.0, 0.3) == 0.0
    # mutual inductance of two coaxial loops, M = 2 pi psi / I; far apart -> mu0 pi a^2 b^2 / (2 d^3)
    a, b, d = 0.1, 0.05, 5.0
    assert 2 * math.pi * fm.loop_flux_function(a, b, d) == pytest.approx(MU0 * math.pi * a * a * b * b / (2 * d ** 3), rel=2e-3)


# ------------------------------------------------------------------ thick solenoid
def test_solenoid_field_reduces_to_montgomery_on_the_axis():
    for z in (0.0, 0.1, 0.25, 0.4):
        p = fm.solenoid_field(REF, J100, 0.0, z)
        assert p.br == 0.0 and p.bz == pytest.approx(REF.axial_field(z, J100), rel=1e-14)
    # just off the axis the 2-D integration agrees with the closed form
    near = fm.solenoid_field(REF, J100, 1e-6, 0.1)
    assert near.bz == pytest.approx(REF.axial_field(0.1, J100), rel=2e-5)
    assert fm.solenoid_field(REF, J100, 0.0, 0.0).magnitude == pytest.approx(100.27, rel=1e-12)


@pytest.mark.parametrize("r, z", [(0.03, 0.1), (0.04, 0.2), (0.3, 0.3), (0.02, 0.4)])
def test_solenoid_field_matches_its_flux_function_outside_the_winding(r, z):
    h = 1e-5
    p = fm.solenoid_field(REF, J100, r, z)
    dpsi_dr = (fm.solenoid_flux_function(REF, J100, r + h, z) - fm.solenoid_flux_function(REF, J100, r - h, z)) / (2 * h)
    dpsi_dz = (fm.solenoid_flux_function(REF, J100, r, z + h) - fm.solenoid_flux_function(REF, J100, r, z - h)) / (2 * h)
    assert dpsi_dr / r == pytest.approx(p.bz, rel=1e-4)
    assert -dpsi_dz / r == pytest.approx(p.br, rel=1e-4)


def test_solenoid_field_symmetry_and_reference_values():
    up = fm.solenoid_field(REF, J100, 0.04, 0.2)
    down = fm.solenoid_field(REF, J100, 0.04, -0.2)
    assert up.br == pytest.approx(-down.br, rel=1e-12) and up.bz == pytest.approx(down.bz, rel=1e-12)
    assert up.br == pytest.approx(6.437, rel=1e-3) and up.bz == pytest.approx(77.92, rel=1e-3)
    assert up.angle == pytest.approx(math.atan2(up.br, up.bz))
    # mid-plane: purely axial, rising slightly towards the bore surface, reversed outside the coil
    assert fm.solenoid_radial_field(REF, J100, 0.1, 0.0) == 0.0
    bore = fm.solenoid_field(REF, J100, 0.05, 0.0)
    assert bore.br == 0.0 and bore.bz == pytest.approx(100.77, rel=1e-3) and bore.bz > 100.27
    assert fm.solenoid_field(REF, J100, 0.4, 0.0).bz < 0.0
    # linear in the current density
    assert fm.solenoid_field(REF, 2 * J100, 0.04, 0.2).bz == pytest.approx(2 * up.bz, rel=1e-12)
    assert fm.solenoid_flux_function(REF, J100, 0.0, 0.1) == 0.0


def test_ampere_law_inside_and_outside_the_winding():
    # curl B = mu0 J in the winding, 0 in the bore:  dB_r/dz - dB_z/dr = mu0 J
    h = 2e-4

    def curl(r, z):
        dbr_dz = (fm.solenoid_field(REF, J100, r, z + h).br - fm.solenoid_field(REF, J100, r, z - h).br) / (2 * h)
        dbz_dr = (fm.solenoid_field(REF, J100, r + h, z).bz - fm.solenoid_field(REF, J100, r - h, z).bz) / (2 * h)
        return dbr_dz - dbz_dr

    assert curl(0.15, 0.1) == pytest.approx(MU0 * J100, rel=2e-2)
    assert abs(curl(0.03, 0.1)) < 2e-3 * MU0 * J100
    # flux through the mid-plane bore equals the long-coil estimate to within the end effect
    assert 2 * math.pi * fm.solenoid_flux_function(REF, J100, 0.05, 0.0) == pytest.approx(100.27 * math.pi * 0.05 ** 2, rel=1e-2)


def test_field_input_validation():
    for call in (lambda: fm.solenoid_field(REF, -1.0, 0.1, 0.0), lambda: fm.solenoid_field(REF, J100, -0.1, 0.0),
                 lambda: fm.solenoid_field("coil", J100, 0.1, 0.0), lambda: fm.solenoid_radial_field(REF, J100, float("nan"), 0.0),
                 lambda: fm.solenoid_flux_function(REF, J100, 0.1, float("inf")), lambda: fm.end_field(REF, n_scan=2),
                 lambda: fm.end_field(None)):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ end field (tape-perpendicular) and peak field
def test_end_field_reference_press():
    e = fm.end_field(REF)
    assert e.fraction == pytest.approx(0.2820, rel=2e-3)          # the package's estimated default is 0.4
    assert REF.inner_radius < e.radius < REF.outer_radius and e.radius == pytest.approx(0.1626, rel=2e-2)
    assert e.z == 0.25 and e.peak_field_ratio == pytest.approx(1.0050, rel=1e-3)
    assert e.field_factor == pytest.approx(REF.field_factor()) and e.alpha == 5.0 and e.beta == 5.0
    # independent of the scan density
    assert fm.end_field(REF, n_scan=12).fraction == pytest.approx(e.fraction, rel=1e-4)


@pytest.mark.parametrize("geometry, fraction, peak", [
    ((0.05, 0.10, 0.30), 0.3872, 1.0121),      # alpha 2, beta 3
    ((0.05, 0.15, 0.30), 0.3371, 1.0155),      # alpha 3, beta 3
    ((0.0075, 0.06, 0.15), 0.2475, 1.0010),    # 15 mm bore insert
    ((0.05, 0.25, 0.10), 0.4331, 1.1069),      # short, thick
    ((0.05, 0.07, 1.00), 0.4613, 1.0001),      # long, thin
])
def test_end_field_across_coil_shapes(geometry, fraction, peak):
    e = fm.end_field(Solenoid(*geometry))
    assert e.fraction == pytest.approx(fraction, rel=2e-3)
    assert e.peak_field_ratio == pytest.approx(peak, rel=2e-3) and e.peak_field_ratio >= 1.0
    assert 0.2 < e.fraction < 0.5


def test_end_face_carries_the_largest_radial_field():
    sol = Solenoid(0.05, 0.15, 0.30)
    j = sol.current_density_for_field(1.0)
    e = fm.end_field(sol)
    inside = max(abs(fm.solenoid_radial_field(sol, j, 0.05 + 0.1 * i / 10, 0.15 * k / 10))
                 for i in range(11) for k in range(10))
    assert inside <= e.fraction * (1 + 1e-9)


# ------------------------------------------------------------------ workpiece outlines
def test_profiles_are_closed_symmetric_outlines():
    cyl = fm.cylinder_profile(0.04, 0.30, 0.004)
    assert cyl[0] == (0.0, 0.15) and cyl[-1] == (0.0, -0.15)
    assert max(p[0] for p in cyl) == pytest.approx(0.04) and all(p[0] >= 0.0 for p in cyl)
    for (r, z), (rm, zm) in zip(cyl, reversed(cyl)):                              # mirror symmetric
        assert r == pytest.approx(rm, abs=1e-15) and z == pytest.approx(-zm, abs=1e-15)
    sharp = fm.cylinder_profile(0.04, 0.30)
    assert (0.04, 0.15) in [(pytest.approx(r), pytest.approx(z)) for r, z in sharp]
    capsule = fm.cylinder_profile(0.04, 0.30, 0.04)                               # hemispherical ends
    assert all(math.hypot(r, abs(z) - 0.11) <= 0.04 * (1 + 1e-9) for r, z in capsule if abs(z) > 0.11)
    sph = fm.sphere_profile(0.05, 32)
    assert len(sph) == 33 and all(math.hypot(r, z) == pytest.approx(0.05) for r, z in sph)
    for call in (lambda: fm.cylinder_profile(0.04, 0.30, 0.05), lambda: fm.cylinder_profile(-1.0, 0.3),
                 lambda: fm.cylinder_profile(0.04, 0.30, n_side=2), lambda: fm.sphere_profile(0.05, 2)):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ perfectly conducting workpiece
def test_sphere_in_uniform_field_matches_the_textbook_solution():
    coil = Solenoid(1.0, 1.1, 12.0)      # long, wide coil: a uniform field around a 50 mm sphere
    j = coil.current_density_for_field(1.0)
    res = fm.surface_pressure(coil, j, fm.sphere_profile(0.05, 64))
    assert res.n_panels == 64 and res.excluded_flux_fraction < 1e-10
    for r, z, b in zip(res.r, res.z, res.surface_field):
        assert b == pytest.approx(1.5 * math.sin(math.atan2(r, z)), abs=4e-3)     # (3/2) B0 sin(theta)
    assert res.mid_ratio == pytest.approx(2.25, rel=2e-3)                         # pressure (3/2)^2 at the equator
    coarse = fm.surface_pressure(coil, j, fm.sphere_profile(0.05, 16))
    assert abs(coarse.mid_ratio - 2.25) > abs(res.mid_ratio - 2.25)               # converges with the mesh
    # linear in the current: fields double, ratios do not move
    double = fm.surface_pressure(coil, 2 * j, fm.sphere_profile(0.05, 64))
    assert double.surface_field[20] == pytest.approx(2 * res.surface_field[20], rel=1e-10)
    assert double.ratio[20] == pytest.approx(res.ratio[20], rel=1e-10)


def test_long_cylinder_in_long_coil_sees_the_ideal_gap_field():
    coil = Solenoid(0.05, 0.06, 3.0)
    res = fm.workpiece_pressure(coil, 0.045, 2.8)
    assert res.mid_ratio == pytest.approx(1.0 / coil.field_factor() ** 2, rel=1e-3)   # B_gap = mu0 J Delta
    assert res.uniformity > 0.998 and res.radial_gap == pytest.approx(0.005)


def test_billet_in_the_reference_press_is_neither_uniform_nor_isostatic():
    res = fm.workpiece_pressure(REF, 0.04, 0.30)
    assert res.reference_field == pytest.approx(1.0) and res.reference_pressure == pytest.approx(1.0 / (2 * MU0))
    assert res.mid_ratio == pytest.approx(1.064, rel=3e-3)            # flux squeezed into the gap
    assert res.uniformity == pytest.approx(0.970, abs=5e-3)           # central 80 % within ~3 %
    assert res.central_min_ratio <= res.central_mean_ratio <= res.central_max_ratio
    assert res.peak_ratio > 4.0 and abs(res.peak_z) > 0.14            # the billet's own corners
    assert res.half_length == pytest.approx(0.15) and res.max_radius == pytest.approx(0.04)
    assert res.central_fraction == fm.DEFAULT_CENTRAL_FRACTION
    # a generous corner radius tames the edge peak but does not remove it
    rounded = fm.workpiece_pressure(REF, 0.04, 0.30, 0.01)
    assert 2.0 < rounded.peak_ratio < res.peak_ratio
    # a billet as long as the coil loses more than half the pressure towards the coil ends
    long_billet = fm.workpiece_pressure(REF, 0.04, 0.60)
    assert long_billet.central_min_ratio == pytest.approx(0.430, abs=0.02) and long_billet.uniformity < 0.5
    # mesh independent in the central region
    fine = fm.workpiece_pressure(REF, 0.04, 0.30, n_side=120)
    assert fine.mid_ratio == pytest.approx(res.mid_ratio, rel=2e-3)
    # the free-space on-axis estimate is pessimistic in the middle and blind to the corners
    assert REF.pressure_uniformity(1e8, 0.30).end_ratio < res.uniformity


def test_flux_is_excluded_from_the_workpiece():
    j = REF.current_density_for_field(1.0)
    profile = fm.cylinder_profile(0.04, 0.30, 0.004)
    res = fm.surface_pressure(REF, j, profile)
    free = fm.solenoid_flux_function(REF, j, 0.02, 0.0)
    inside = fm.flux_function_with_workpiece(REF, j, res, 0.02, 0.0)
    assert abs(inside) < 2e-3 * abs(free)                              # no flux inside the metal
    far_free = fm.solenoid_flux_function(REF, j, 0.6, 0.0)
    far = fm.flux_function_with_workpiece(REF, j, res, 0.6, 0.0)
    assert far == pytest.approx(far_free, rel=5e-2)                    # far away the coil dominates
    gap = fm.flux_function_with_workpiece(REF, j, res, 0.045, 0.0)
    assert 0.0 < gap < fm.solenoid_flux_function(REF, j, 0.045, 0.0)
    assert fm.flux_function_with_workpiece(REF, j, res, 0.0, 0.1) == 0.0
    # induced currents oppose the coil
    assert sum(res.panel_current) < 0.0
    with pytest.raises(InputError):
        fm.flux_function_with_workpiece(REF, j, "nope", 0.02, 0.0)


def test_surface_pressure_input_validation():
    j = REF.current_density_for_field(1.0)
    good = fm.cylinder_profile(0.04, 0.30)
    for call in (
        lambda: fm.surface_pressure(REF, j, fm.cylinder_profile(0.06, 0.30)),          # does not fit the bore
        lambda: fm.surface_pressure(REF, 0.0, good),
        lambda: fm.surface_pressure(REF, j, good, central_fraction=0.0),
        lambda: fm.surface_pressure(REF, j, good, central_fraction=1.5),
        lambda: fm.surface_pressure(REF, j, ((0.0, 0.1), (0.02, 0.0))),                 # too few points
        lambda: fm.surface_pressure(REF, j, ((0.01, 0.1), (0.02, 0.05), (0.02, 0.0), (0.02, -0.05), (0.0, -0.1))),
        lambda: fm.surface_pressure(REF, j, ((0.0, 0.1), (0.02, 0.05), (0.02, 0.05), (0.02, -0.05), (0.0, -0.1))),
        lambda: fm.surface_pressure(REF, j, "outline"),
        lambda: fm.workpiece_pressure(REF, 0.04, 0.30, corner_radius=-0.001),
    ):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ serialisation and metadata
def test_results_round_trip_json_and_render_text():
    e = fm.end_field(REF)
    data = json.loads(e.json())
    assert data["fraction"] == pytest.approx(e.fraction) and "radius [m]" in e.text()
    p = fm.solenoid_field(REF, J100, 0.04, 0.2)
    assert json.loads(p.json())["bz"] == pytest.approx(p.bz) and "magnitude [T]" in p.text()
    res = fm.workpiece_pressure(REF, 0.04, 0.30, n_side=20)
    d = json.loads(res.json())
    assert len(d["ratio"]) == res.n_panels == len(d["z"]) and d["mid_ratio"] == pytest.approx(res.mid_ratio)
    assert "reference_pressure [Pa]" in res.text()
    with pytest.raises(Exception):
        res.mid_ratio = 1.0  # frozen


def test_module_metadata():
    assert set(fm.__all__) <= set(dir(fm))
    assert fm.REFERENCES and all(isinstance(v, str) and len(v) > 20 for v in fm.REFERENCES.values())
    for name in fm.__all__:
        obj = getattr(fm, name)
        if callable(obj) and name != "Profile":
            assert obj.__doc__ and "Reference" in obj.__doc__, name
