"""Tests for magform.microstructure: dislocations, grains, recrystallisation and the eddy-current skin."""

from __future__ import annotations

import json
import math

import pytest

from magform import eddy, materials
from magform import microstructure as ms
from magform import plasticity as pl
from magform._numerics import trapz
from magform._report import Assessment, ResultMixin
from magform.constants import R_GAS, T_ROOM
from magform.errors import InputError, UnknownMaterialError
from magform.materials import HallPetch
from magform.status import Status

RT = T_ROOM


def _taylor_slope(m: materials.Material, alpha: float = ms.TAYLOR_ALPHA) -> float:
    """M alpha G b in Pa m."""
    return ms.taylor_factor(m) * alpha * m.shear_modulus * m.burgers_vector


# ------------------------------------------------------------------ constants / API
def test_constants_and_module_metadata():
    assert ms.TAYLOR_FACTOR == {"fcc": 3.06, "bcc": 2.75} and ms.DEFAULT_TAYLOR_FACTOR == 3.06
    assert ms.TAYLOR_ALPHA == 0.3 and ms.TAYLOR_ALPHA_RANGE[0] <= ms.TAYLOR_ALPHA <= ms.TAYLOR_ALPHA_RANGE[1]
    assert ms.CELL_SIZE_CONSTANT == 10.0 and ms.CELL_SIZE_CONSTANT_RANGE == (10.0, 20.0)
    assert ms.ULTRAFINE_GRAIN_SIZE == 1e-6 and ms.HALL_PETCH_FLOOR == 20e-9
    assert ms.RECRYSTALLISATION_HOMOLOGOUS == 0.4 and ms.RECRYSTALLISATION_HOMOLOGOUS_RANGE == (0.3, 0.5)
    assert ms.RECRYSTALLISATION_REFERENCE_TIME == 3600.0 and ms.RECRYSTALLISATION_REFERENCE_FRACTION == 0.95
    assert ms.AVRAMI_EXPONENT == 2.0 and ms.RECRYSTALLISED_THRESHOLD == 0.01
    assert ms.DERBY_K_RANGE == (1.0, 10.0)
    assert ms.SURFACE_KAPPA == eddy.KAPPA_STEP == 2.4
    # exp(1) erfc(1): surface of an exponentially heated layer one conduction time after the pulse
    assert ms.SURFACE_EXCESS_AT_COOLING_TIME == pytest.approx(0.42758, rel=1e-4)
    assert set(ms.__all__) <= set(dir(ms))
    assert len(ms.REFERENCES) >= 15
    assert all(isinstance(k, str) and isinstance(v, str) and len(v) > 40 for k, v in ms.REFERENCES.items())
    for name in ms.__all__:
        obj = getattr(ms, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 80 and "Reference" in obj.__doc__, name


# ------------------------------------------------------------------ Taylor hardening
def test_taylor_factor_by_crystal_structure(cu, w):
    assert ms.taylor_factor(cu) == 3.06 and ms.taylor_factor("steel_304") == 3.06      # fcc
    assert ms.taylor_factor(w) == 2.75 and ms.taylor_factor("tantalum") == 2.75         # bcc pencil glide
    assert ms.taylor_factor("rhenium") == ms.DEFAULT_TAYLOR_FACTOR                      # hcp: no single value
    assert ms.taylor_factor("gamma_tial") == ms.DEFAULT_TAYLOR_FACTOR
    with pytest.raises(UnknownMaterialError):
        ms.taylor_factor("unobtainium")


def test_taylor_hardening_closed_form_and_round_trip(w):
    slope = _taylor_slope(w)
    assert slope == pytest.approx(2.75 * 0.3 * 160.55e9 * 2.74e-10, rel=1e-3)
    assert ms.taylor_hardening(w, 1e14) == pytest.approx(slope * (1e7 - 1e6), rel=1e-12)
    assert ms.taylor_hardening(w, 1e14, initial_density=0.0) == pytest.approx(slope * 1e7, rel=1e-12)
    assert ms.taylor_hardening(w, 1e12) == 0.0 and ms.taylor_hardening(w, 1e11) < 0.0
    # the density inferred from the package's flow stress reproduces that flow-stress increment
    for key, eps, rate in (("tungsten", 0.3, 1e4), ("copper_ofhc", 0.5, 1e3), ("niobium", 0.2, 1e-3)):
        rho = ms.dislocation_density(key, eps, rate)
        hardening = pl.flow_stress(key, eps, rate) - pl.flow_stress(key, 0.0, rate)
        assert ms.taylor_hardening(key, rho) == pytest.approx(hardening, rel=1e-10), key
    # rho ~ 1/alpha^2: the 0.2-0.5 range of alpha alone is a factor 6.25
    lo, hi = ms.TAYLOR_ALPHA_RANGE
    ratio = (ms.dislocation_density(w, 0.3, 1e4, initial_density=0.0, alpha=lo)
             / ms.dislocation_density(w, 0.3, 1e4, initial_density=0.0, alpha=hi))
    assert ratio == pytest.approx((hi / lo) ** 2, rel=1e-12)
    for kw in (dict(density=-1.0), dict(density=1e14, initial_density=-1.0), dict(density=1e14, alpha=0.0)):
        with pytest.raises(InputError):
            ms.taylor_hardening(w, **kw)


def test_dislocation_density_tungsten_headline_and_closed_form(w):
    rho = ms.dislocation_density(w, 0.3, 1e4, RT)
    assert rho == pytest.approx(3.652e13, rel=1e-3)
    # Zerilli-Armstrong BCC: the hardening term c5 eps^n is athermal and rate independent
    za = w.constitutive[0]
    hardening = za.c5 * 0.3 ** za.n
    assert hardening == pytest.approx(183.0e6, rel=1e-3)
    expected = (math.sqrt(ms.INITIAL_DISLOCATION_DENSITY) + hardening / _taylor_slope(w)) ** 2
    assert rho == pytest.approx(expected, rel=1e-9)
    assert ms.dislocation_density(w, 0.3, 1e-3, 600.0) == pytest.approx(expected, rel=1e-9)  # no thermal softening counted
    assert ms.dislocation_density("copper_ofhc", 0.3, 1e4) == pytest.approx(6.30e14, rel=1e-2)


def test_dislocation_density_limits_monotonicity_and_validation(w):
    for key in ("tungsten", "copper_ofhc", "tantalum", "niobium", "molybdenum", "rhenium"):
        assert ms.dislocation_density(key, 0.0, 1e4) == pytest.approx(ms.INITIAL_DISLOCATION_DENSITY, rel=1e-12)
        series = [ms.dislocation_density(key, e, 1e4) for e in (0.0, 0.02, 0.1, 0.3, 1.0, 3.0)]
        assert all(b >= a for a, b in zip(series, series[1:])), key
        assert series[-1] > series[0], key
    assert ms.dislocation_density(w, 0.3, 1e4, initial_density=1e14) > ms.dislocation_density(w, 0.3, 1e4)
    assert ms.dislocation_density(w, 0.0, 1e4, initial_density=5e13) == pytest.approx(5e13, rel=1e-12)
    for kw in (dict(strain=-0.1), dict(strain=0.3, alpha=-1.0), dict(strain=0.3, initial_density=-1.0),
               dict(strain=0.3, temperature=5000.0), dict(strain=0.3, strain_rate=-1.0), dict(strain=float("nan"))):
        with pytest.raises(InputError):
            ms.dislocation_density(w, **kw)
    with pytest.raises(InputError):          # no Burgers vector recorded
        ms.dislocation_density("mp35n", 0.1)
    with pytest.raises(InputError):          # no Burgers vector and no flow-stress data
        ms.dislocation_density("diamond", 0.1)
    with pytest.raises(UnknownMaterialError):
        ms.dislocation_density("unobtainium", 0.1)


def test_dislocation_density_band_spans_the_published_fits(w):
    lo, hi = ms.dislocation_density_band(w, 0.3, 1e4)
    assert lo == pytest.approx(ms.dislocation_density(w, 0.3, 1e4), rel=1e-12)      # Lennon-Ramesh hardens least
    chen_gray = w.constitutive[1]
    expected_hi = (1e6 + chen_gray.c5 * 0.3 ** chen_gray.n / _taylor_slope(w)) ** 2
    assert hi == pytest.approx(expected_hi, rel=1e-9) and hi == pytest.approx(1.37e14, rel=1e-2)
    assert hi / lo > 3.0                                                             # batch/fit uncertainty
    for key in ("copper_ofhc", "niobium"):                                           # one fit / fallback only
        b = ms.dislocation_density_band(key, 0.3, 1e4)
        assert b[0] == b[1] == pytest.approx(ms.dislocation_density(key, 0.3, 1e4), rel=1e-12)
    ta_lo, ta_hi = ms.dislocation_density_band("tantalum", 0.3, 1e4)
    assert ta_lo < ta_hi and ta_lo <= ms.dislocation_density("tantalum", 0.3, 1e4) <= ta_hi
    with pytest.raises(InputError):
        ms.dislocation_density_band(w, -0.1)


def test_cell_size_and_stored_energy(w):
    assert ms.dislocation_cell_size(1e14) == pytest.approx(1.0e-6, rel=1e-12)
    assert ms.dislocation_cell_size(1e15) == pytest.approx(0.316e-6, rel=1e-2)
    assert ms.dislocation_cell_size(1e14, k=20.0) == pytest.approx(2.0e-6, rel=1e-12)
    rho = ms.dislocation_density(w, 0.3, 1e4)
    cell = ms.dislocation_cell_size(rho)
    assert cell == pytest.approx(1.655e-6, rel=1e-3)
    assert cell < w.typical_grain_size                    # cells sit inside the original 20 um grains
    for bad in (dict(density=0.0), dict(density=-1.0), dict(density=1e14, k=0.0)):
        with pytest.raises(InputError):
            ms.dislocation_cell_size(**bad)
    stored = ms.stored_energy_density(w, rho)
    assert stored == pytest.approx(0.5 * w.shear_modulus * w.burgers_vector ** 2 * rho, rel=1e-12)
    assert stored == pytest.approx(2.2e5, rel=2e-2)
    assert ms.stored_energy_density(w, 0.0) == 0.0
    # a tiny fraction of the plastic work: the rest is heat (Taylor-Quinney)
    eps = [0.3 * i / 300 for i in range(301)]
    work = trapz([pl.flow_stress(w, e, 1e4, RT) for e in eps], eps)
    assert 1e-4 < stored / work < 1e-3
    with pytest.raises(InputError):
        ms.stored_energy_density(w, -1.0)
    with pytest.raises(InputError):
        ms.stored_energy_density("mp35n", 1e14)


# ------------------------------------------------------------------ Hall-Petch
def test_hall_petch_calibration_reproduces_every_recorded_yield():
    checked = 0
    for key, m in materials.MATERIALS.items():
        if m.hall_petch is None or m.yield_strength is None:
            continue
        d_ref = m.hall_petch.reference_grain_size
        assert ms.hall_petch_strength(m, d_ref) == pytest.approx(m.yield_strength, rel=1e-12), key
        assert 0.0 < ms.hall_petch_friction_stress(m) < m.yield_strength, key
        checked += 1
    assert checked >= 12


def test_hall_petch_strength_inverse_and_floor(w, cu):
    sigma0 = ms.hall_petch_friction_stress(w)
    assert sigma0 == pytest.approx(1.3e9 - 0.8e6 / math.sqrt(50e-6), rel=1e-12) == pytest.approx(1186.9e6, rel=1e-4)
    assert ms.hall_petch_strength(w, 1e-7) == pytest.approx(sigma0 + 0.8e6 / math.sqrt(1e-7), rel=1e-12)
    assert ms.hall_petch_strength(w, w.spd_saturation_grain_size) == pytest.approx(3.717e9, rel=1e-3)
    series = [ms.hall_petch_strength(w, d) for d in (1e-3, 5e-5, 1e-6, 1e-7, 2e-8)]
    assert all(b > a for a, b in zip(series, series[1:]))
    # below 20 nm the relation is not extended (plateau instead of inverse Hall-Petch softening)
    assert ms.hall_petch_strength(w, 1e-9) == ms.hall_petch_strength(w, ms.HALL_PETCH_FLOOR)
    for d in (1e-7, 1e-6, 5e-5):
        assert ms.grain_size_for_strength(w, ms.hall_petch_strength(w, d)) == pytest.approx(d, rel=1e-9)
    assert ms.grain_size_for_strength(w, 3e9) == pytest.approx(1.947e-7, rel=1e-3)
    assert ms.grain_size_for_strength(w, sigma0) == math.inf and ms.grain_size_for_strength(w, 1e9) == math.inf
    assert ms.grain_size_for_strength(w, 10e9) is None            # would need grains finer than 20 nm
    # an explicit friction stress is used as recorded
    explicit = cu.with_changes(key="test_cu_hp", hall_petch=HallPetch(k=0.14e6, sigma0=25e6))
    assert ms.hall_petch_friction_stress(explicit) == 25e6
    assert ms.hall_petch_strength(explicit, 1e-6) == pytest.approx(25e6 + 0.14e6 / 1e-3, rel=1e-12)
    with pytest.raises(InputError):                               # no Hall-Petch record
        ms.hall_petch_strength("inconel_718", 1e-6)
    with pytest.raises(InputError):                               # nothing to calibrate sigma0 against
        ms.hall_petch_friction_stress(cu.with_changes(key="test_cu_noyield", yield_strength=None))
    for bad in (0.0, -1e-6, float("inf")):
        with pytest.raises(InputError):
            ms.hall_petch_strength(w, bad)
    with pytest.raises(InputError):
        ms.grain_size_for_strength(w, 0.0)


# ------------------------------------------------------------------ grain refinement
def test_grain_size_after_strain_is_geometric_thinning_towards_saturation(w):
    d0, ds = w.typical_grain_size, w.spd_saturation_grain_size
    assert (d0, ds) == (20e-6, 100e-9)
    assert ms.grain_size_after_strain(w, 0.0) == d0
    d = ms.grain_size_after_strain(w, 0.3)
    assert d == pytest.approx(ds + (d0 - ds) * math.exp(-0.3), rel=1e-12) == pytest.approx(14.84e-6, rel=1e-3)
    # headline: one forming pulse leaves the grain size in the same order of magnitude, far from ultrafine
    assert d > 10 * ms.ULTRAFINE_GRAIN_SIZE and 0.7 < d / d0 < 0.75
    series = [ms.grain_size_after_strain(w, e) for e in (0.0, 0.3, 1.0, 3.0, 6.0, 12.0, 50.0)]
    assert all(b < a for a, b in zip(series, series[1:-1])) and series[-1] == pytest.approx(ds, rel=1e-9)
    assert ms.grain_size_after_strain(w, 6.0) == pytest.approx(0.149e-6, rel=1e-2)
    # explicit feedstock and saturation sizes; a feedstock finer than saturation is left alone
    assert ms.grain_size_after_strain(w, 1.0, 5e-6, 2e-7) == pytest.approx(2e-7 + 4.8e-6 * math.exp(-1.0), rel=1e-12)
    assert ms.grain_size_after_strain(w, 2.0, initial_grain_size=50e-9) == 50e-9
    # no recorded saturation size -> 200 nm estimate
    assert materials.get("gamma_tial").spd_saturation_grain_size is None
    assert ms.grain_size_after_strain("gamma_tial", 60.0) == pytest.approx(ms.DEFAULT_SATURATION_GRAIN_SIZE, rel=1e-9)
    with pytest.raises(InputError):                               # no typical grain size recorded
        ms.grain_size_after_strain("mp35n", 0.3)
    assert ms.grain_size_after_strain("mp35n", 0.0, initial_grain_size=1e-5) == 1e-5
    for kw in (dict(strain=-0.1), dict(strain=0.3, initial_grain_size=0.0), dict(strain=0.3, saturation_grain_size=-1.0)):
        with pytest.raises(InputError):
            ms.grain_size_after_strain(w, **kw)


def test_grain_refinement_needs_spd_strains_not_one_pulse(w):
    eps_sat = ms.grain_refinement_strain(w)                       # to twice the saturation size
    assert eps_sat == pytest.approx(math.log(199.0), rel=1e-12) == pytest.approx(5.293, rel=1e-3)
    assert ms.grain_size_after_strain(w, eps_sat) == pytest.approx(2.0 * w.spd_saturation_grain_size, rel=1e-9)
    eps_uf = ms.grain_refinement_strain(w, ms.ULTRAFINE_GRAIN_SIZE)
    assert eps_uf == pytest.approx(math.log(19.9 / 0.9), rel=1e-9) == pytest.approx(3.096, rel=1e-3)
    assert ms.grain_size_after_strain(w, eps_uf) == pytest.approx(1e-6, rel=1e-9)
    assert eps_sat / 0.3 > 10.0 and eps_uf / 0.3 > 10.0           # a 0.3 forming strain is an order of magnitude short
    # refractory metals and copper land in the 4-8 window of ECAP / high-pressure torsion
    for key in ("tungsten", "molybdenum", "rhenium", "tantalum", "niobium", "copper_ofhc"):
        assert 4.0 <= ms.grain_refinement_strain(key) <= 8.0, key
    assert ms.grain_refinement_strain("copper_ofhc") == pytest.approx(5.004, rel=1e-3)
    assert ms.grain_refinement_strain(w, 20e-6) == 0.0 and ms.grain_refinement_strain(w, 1.0) == 0.0
    assert ms.grain_refinement_strain(w, w.spd_saturation_grain_size) == math.inf
    assert ms.grain_refinement_strain(w, 50e-9) == math.inf
    assert ms.grain_refinement_strain(w, initial_grain_size=150e-9) == 0.0
    assert ms.grain_refinement_strain(w, 1e-6, 5e-6, 2e-7) == pytest.approx(math.log(4.8 / 0.8), rel=1e-12)
    with pytest.raises(InputError):
        ms.grain_refinement_strain(w, -1e-6)


def test_hardness_after_strain_is_tabor(w):
    h0 = ms.hardness_after_strain(w)
    assert h0 == pytest.approx(3.0 * pl.flow_stress(w, 0.08, 1e-3, RT), rel=1e-12)
    assert h0 == pytest.approx(w.vickers_hardness, rel=5e-2)      # 4.21 GPa against the recorded 4.3 GPa
    h1 = ms.hardness_after_strain(w, 0.3)
    assert h1 == pytest.approx(3.0 * pl.flow_stress(w, 0.38, 1e-3, RT), rel=1e-12) and h1 > h0
    assert ms.hardness_after_strain("copper_ofhc", 0.3) > 1.3 * ms.hardness_after_strain("copper_ofhc")
    assert ms.hardness_after_strain(w, 0.0, 900.0) < h0
    with pytest.raises(InputError):
        ms.hardness_after_strain(w, -0.1)


# ------------------------------------------------------------------ recrystallisation kinetics
def test_recrystallisation_anchor_and_arrhenius_scaling(w, cu):
    t_anchor = 0.4 * w.melting_point
    assert t_anchor == pytest.approx(1478.0) and 0.4 * cu.melting_point == pytest.approx(542.4)
    assert ms.recrystallisation_time(w, t_anchor, 0.95) == pytest.approx(3600.0, rel=1e-12)
    assert ms.recrystallised_fraction(w, t_anchor, 3600.0) == pytest.approx(0.95, rel=1e-12)
    assert ms.recrystallisation_time(w, t_anchor) == pytest.approx(3600.0 * math.sqrt(math.log(2.0) / math.log(20.0)),
                                                                    rel=1e-12)
    q = w.self_diffusion_activation
    for t1, t2 in ((1600.0, 1478.0), (1836.0, 1200.0), (2500.0, 900.0)):
        ratio = ms.recrystallisation_time(w, t1) / ms.recrystallisation_time(w, t2)
        assert ratio == pytest.approx(math.exp(q / R_GAS * (1.0 / t1 - 1.0 / t2)), rel=1e-9)
    assert ms.recrystallisation_time(w, 1835.65, 0.95) == pytest.approx(0.3375, rel=1e-2)
    for x in (0.01, 0.5, 0.95, 0.999):
        t = ms.recrystallisation_time(w, 1700.0, x)
        assert ms.recrystallised_fraction(w, 1700.0, t) == pytest.approx(x, rel=1e-9)
    times = [ms.recrystallisation_time(w, t) for t in (1000.0, 1478.0, 2000.0, 3000.0)]
    assert all(b < a for a, b in zip(times, times[1:]))
    fractions = [ms.recrystallised_fraction(w, 1700.0, t) for t in (0.0, 0.1, 1.0, 10.0, 1e3)]
    assert fractions[0] == 0.0 and all(b > a for a, b in zip(fractions, fractions[1:])) and fractions[-1] == 1.0
    # the anchor keyword moves the whole curve: 0.3 T_m (pure, heavily worked) is faster than 0.5 T_m
    assert ms.recrystallisation_time(w, 1700.0, anchor=0.3) < ms.recrystallisation_time(w, 1700.0) \
        < ms.recrystallisation_time(w, 1700.0, anchor=0.5)
    assert ms.recrystallised_fraction(w, 0.5 * w.melting_point, 3600.0, anchor=0.5) == pytest.approx(0.95, rel=1e-12)


def test_recrystallisation_limits_and_validation(w):
    assert ms.recrystallisation_time(w, 77.0) == math.inf          # beyond the floating-point range
    assert ms.recrystallised_fraction(w, 77.0, 1e12) == 0.0
    assert ms.recrystallised_fraction(w, 3600.0, 1.0) == 1.0       # near melting it is instantaneous
    assert ms.recrystallised_fraction(w, 1200.0, 0.0) == 0.0
    for kw in (dict(temperature=0.0), dict(temperature=w.melting_point), dict(temperature=1500.0, fraction=0.0),
               dict(temperature=1500.0, fraction=1.0), dict(temperature=1500.0, anchor=0.0),
               dict(temperature=1500.0, anchor=1.0), dict(temperature=float("nan"))):
        with pytest.raises(InputError):
            ms.recrystallisation_time(w, **kw)
    for kw in (dict(temperature=1500.0, time=-1.0), dict(temperature=-10.0, time=1.0),
               dict(temperature=1500.0, time=1.0, anchor=1.5)):
        with pytest.raises(InputError):
            ms.recrystallised_fraction(w, **kw)
    with pytest.raises(InputError):                                # no activation energy recorded
        ms.recrystallisation_time("hastelloy_c276", 800.0)
    with pytest.raises(UnknownMaterialError):
        ms.recrystallised_fraction("unobtainium", 800.0, 1.0)


def test_bulk_grain_structure_is_frozen_after_cold_forming(w):
    d_t = pl.adiabatic_temperature_rise(w, 0.3, 1e4)
    assert d_t == pytest.approx(189.2, rel=1e-2)
    t_final = RT + d_t
    assert t_final / w.melting_point == pytest.approx(0.13, abs=5e-3)
    assert ms.recrystallisation_time(w, t_final, 0.95) > 1e40      # ~1e46 s
    assert ms.recrystallised_fraction(w, t_final, 60.0) < 1e-50
    assert ms.recrystallised_fraction(w, t_final, 3.15e7) < 1e-50  # a year changes nothing


# ------------------------------------------------------------------ Zener-Hollomon, regimes, Derby
def test_zener_hollomon_and_equivalent_temperature(w):
    q = w.self_diffusion_activation
    log_hot = ms.log10_zener_hollomon(w, 1.0, 1673.0)
    assert log_hot == pytest.approx(q / (R_GAS * 1673.0 * math.log(10.0)), rel=1e-12) == pytest.approx(18.26, abs=0.01)
    assert ms.zener_hollomon(w, 1.0, 1673.0) == pytest.approx(10.0 ** log_hot, rel=1e-9)
    log_cold = ms.log10_zener_hollomon(w, 1e4, 482.35)
    assert log_cold == pytest.approx(67.35, abs=0.01) and log_cold > 3.0 * log_hot
    assert ms.zener_hollomon(w, 1e4, 482.35) == pytest.approx(10.0 ** log_cold, rel=1e-6)
    assert ms.zener_hollomon(w, 1e4, 77.0) == math.inf and ms.zener_hollomon(w, 0.0, 500.0) == 0.0
    with pytest.raises(InputError):
        ms.log10_zener_hollomon(w, 0.0, 500.0)
    with pytest.raises(InputError):
        ms.zener_hollomon(w, -1.0, 500.0)
    # equivalent temperature at the 1 1/s reference: same Z, a fast deformation acts colder
    t_eq = ms.equivalent_temperature(w, 1e4, 482.35)
    assert t_eq == pytest.approx(1.0 / (1.0 / 482.35 + R_GAS / q * math.log(1e4)), rel=1e-12)
    assert t_eq == pytest.approx(453.7, abs=0.1) and t_eq < 482.35
    assert ms.log10_zener_hollomon(w, 1.0, t_eq) == pytest.approx(log_cold, rel=1e-12)
    assert ms.equivalent_temperature(w, 1.0, 482.35) == pytest.approx(482.35, rel=1e-12)
    assert ms.equivalent_temperature(w, 1e-3, 1500.0) > 1500.0
    assert ms.equivalent_temperature(w, 0.0, 300.0) == ms.equivalent_temperature(w, pl.RATE_FLOOR, 300.0)
    assert ms.equivalent_temperature(w, 1e4, 482.35, reference_rate=1e4) == pytest.approx(482.35, rel=1e-12)
    slow_hot = w.with_changes(key="test_low_q", self_diffusion_activation=2e4)
    assert ms.equivalent_temperature(slow_hot, 1e-6, 3000.0) == math.inf
    with pytest.raises(InputError):
        ms.equivalent_temperature(w, 1.0, 482.35, reference_rate=0.0)
    with pytest.raises(InputError):
        ms.zener_hollomon("mp35n", 1.0, 500.0)


def test_deformation_regime_boundaries(w):
    tm = w.melting_point
    assert ms.deformation_regime(w, 1.0, 0.29 * tm) == "cold"
    assert ms.deformation_regime(w, 1.0, 0.31 * tm) == "warm"
    assert ms.deformation_regime(w, 1.0, 0.49 * tm) == "warm"
    assert ms.deformation_regime(w, 1.0, 0.51 * tm) == "hot"
    assert ms.deformation_regime(w, 1e4, RT) == "cold"
    # the same temperature is 'colder' when deformed faster and 'hotter' when deformed slower
    assert ms.deformation_regime(w, 1e6, 0.31 * tm) == "cold"
    assert ms.deformation_regime(w, 1e-6, 0.49 * tm) == "hot"
    assert ms.deformation_regime(w, 1.0, 1673.0) == "warm"         # conventional W forging, 1400 C = 0.45 T_m


def test_derby_piezometer(w):
    sigma = pl.flow_stress(w, 0.3, 1e4, RT)
    fine, coarse = (ms.derby_grain_size(w, sigma, k) for k in ms.DERBY_K_RANGE)
    assert fine == pytest.approx(w.burgers_vector * (w.shear_modulus / sigma) ** 1.5, rel=1e-12)
    assert fine == pytest.approx(0.19e-6, rel=2e-2) and coarse == pytest.approx(6.0e-6, rel=2e-2)
    assert coarse / fine == pytest.approx(10.0 ** 1.5, rel=1e-12)
    assert ms.derby_grain_size(w, 2.0 * sigma, 1.0) == pytest.approx(fine / 2.0 ** 1.5, rel=1e-12)
    # the piezometer inverts: (sigma/G)(d/b)^(2/3) = K
    assert (sigma / w.shear_modulus) * (coarse / w.burgers_vector) ** (2.0 / 3.0) == pytest.approx(10.0, rel=1e-12)
    for bad in (dict(stress=0.0, k=1.0), dict(stress=1e9, k=0.0), dict(stress=-1e9, k=1.0)):
        with pytest.raises(InputError):
            ms.derby_grain_size(w, **bad)
    with pytest.raises(TypeError):                                 # K has no default: the range is all that is known
        ms.derby_grain_size(w, sigma)  # type: ignore[call-arg]


# ------------------------------------------------------------------ eddy-current skin
def test_skin_exposure_tungsten_at_100_tesla(w):
    sk = ms.skin_thermal_exposure(w, 100.0)
    assert isinstance(sk, ResultMixin) and sk.material == "tungsten" and sk.frequency == ms.DEFAULT_PULSE_FREQUENCY
    # consistent with the eddy module
    assert sk.skin_depth == pytest.approx(eddy.skin_depth_for_material(w, 1e4), rel=1e-12)
    assert sk.heated_depth == pytest.approx(0.5 * sk.skin_depth, rel=1e-12)
    assert sk.cooling_time == pytest.approx(sk.heated_depth ** 2 / w.thermal_diffusivity(), rel=1e-12)
    assert sk.temperature_rise == pytest.approx(eddy.skin_temperature_rise(w, 100.0), rel=1e-12)
    assert sk.melting_field == pytest.approx(eddy.skin_melting_field(w), rel=1e-12)
    assert sk.surface_kappa == 2.4 and sk.surface_temperature_rise == pytest.approx(2.4 * sk.temperature_rise, rel=1e-12)
    # headline numbers: layer average 0.50 T_m for about 5 ms, essentially not recrystallised ...
    assert sk.skin_depth == pytest.approx(1.156e-3, rel=1e-3)
    assert sk.temperature_rise == pytest.approx(1542.5, rel=1e-3)
    assert sk.peak_temperature == pytest.approx(1835.6, rel=1e-3)
    assert sk.homologous_temperature == pytest.approx(0.497, abs=1e-3)
    assert sk.cooling_time == pytest.approx(4.99e-3, rel=5e-3)
    assert sk.recrystallisation_time == pytest.approx(0.3375, rel=1e-2)
    assert sk.melted is False and sk.melting_field == pytest.approx(148.5, rel=1e-3)
    assert sk.recrystallised_fraction == pytest.approx(6.5e-4, rel=5e-2)
    assert sk.recrystallised_fraction == pytest.approx(
        ms.recrystallised_fraction(w, sk.peak_temperature, sk.cooling_time), rel=1e-12)
    assert sk.heat_affected is False
    # ... but the 0.3-0.5 T_m anchor band runs from nil to complete, the layer value is within 5 % in
    # temperature of the 1 % threshold, and the surface estimate (kappa 2.4) melts
    lo, hi = sk.recrystallised_fraction_band
    assert lo < 1e-9 < sk.recrystallised_fraction < hi == 1.0
    assert lo == pytest.approx(ms.recrystallised_fraction(w, sk.peak_temperature, sk.cooling_time, anchor=0.5), rel=1e-12)
    assert ms.recrystallised_fraction(w, 1.05 * sk.peak_temperature, sk.cooling_time) > ms.RECRYSTALLISED_THRESHOLD
    assert sk.surface_melted is True and sk.surface_peak_temperature == w.melting_point
    assert sk.surface_homologous_temperature == 1.0 and sk.surface_recrystallised_fraction == 1.0
    assert sk.surface_heat_affected is True and sk.haz_depth == sk.heated_depth
    assert ms.skin_thermal_exposure(w, -100.0).peak_temperature == sk.peak_temperature   # the sign is irrelevant


def test_skin_exposure_melting_and_material_trends(cu, w):
    hot = ms.skin_thermal_exposure(cu, 100.0)
    assert hot.melted is True and hot.peak_temperature == cu.melting_point and hot.homologous_temperature == 1.0
    assert hot.temperature_rise == pytest.approx(1153.4, rel=1e-3)                # uncapped estimate
    assert hot.recrystallised_fraction == 1.0 and hot.recrystallised_fraction_band == (1.0, 1.0)
    assert hot.recrystallisation_time is None and hot.heat_affected is True
    assert hot.melting_field == pytest.approx(96.0, abs=0.1) and hot.haz_depth == pytest.approx(0.329e-3, rel=1e-2)
    assert ms.skin_thermal_exposure(cu, 95.0).melted is False and ms.skin_thermal_exposure(cu, 97.0).melted is True
    assert ms.skin_thermal_exposure("aluminum_6061", 63.41).melted is True        # Al melts from 58 T
    # molybdenum at 100 T: 0.64 T_m for about 6 ms recrystallises on the central estimate
    mo = ms.skin_thermal_exposure("molybdenum", 100.0)
    assert mo.melted is False and mo.homologous_temperature == pytest.approx(0.637, abs=2e-3)
    assert mo.cooling_time == pytest.approx(6.29e-3, rel=1e-2)
    assert mo.recrystallised_fraction == 1.0 and mo.heat_affected is True
    assert mo.recrystallised_fraction_band[0] < 1e-3 and mo.recrystallised_fraction_band[1] == 1.0
    # copper: clean at 30 T, heat-affected at the surface estimate by 50 T, on the layer estimate by 63 T
    low = ms.skin_thermal_exposure(cu, 30.0)
    assert (low.heat_affected, low.surface_heat_affected, low.haz_depth) == (False, False, 0.0)
    mid = ms.skin_thermal_exposure(cu, 50.0)
    assert (mid.heat_affected, mid.surface_heat_affected) == (False, True) and mid.haz_depth == mid.heated_depth
    assert mid.surface_melted is False and mid.surface_recrystallised_fraction == pytest.approx(1.0)
    assert ms.skin_thermal_exposure(cu, 63.41).heat_affected is True
    # bare tungsten: clean at 63 T, surface estimate recrystallises at 71 T (2 GPa)
    assert ms.skin_thermal_exposure(w, 63.41).surface_heat_affected is False
    w71 = ms.skin_thermal_exposure(w, 70.9)
    assert (w71.heat_affected, w71.surface_heat_affected, w71.surface_melted) == (False, True, False)
    # every material is hotter at a higher field
    for key in ("tungsten", "molybdenum", "rhenium", "tantalum", "niobium"):
        rises = [ms.skin_thermal_exposure(key, b).temperature_rise for b in (63.41, 70.9, 100.27)]
        assert all(b > a for a, b in zip(rises, rises[1:])), key
        assert rises[2] / rises[0] == pytest.approx((100.27 / 63.41) ** 2, rel=1e-9), key


def test_skin_exposure_parameters_and_validation(w):
    base = ms.skin_thermal_exposure(w, 100.0)
    # kappa scales the temperature rise; given above 2.4 it is also the surface factor
    k2 = ms.skin_thermal_exposure(w, 100.0, kappa=2.4)
    assert k2.temperature_rise == pytest.approx(2.4 * base.temperature_rise, rel=1e-12) and k2.melted is True
    assert k2.melting_field == pytest.approx(base.melting_field / math.sqrt(2.4), rel=1e-12)
    assert ms.skin_thermal_exposure(w, 50.0, kappa=3.0).surface_kappa == 3.0
    # a faster pulse has a thinner skin, which cools sooner (t_cool ~ delta^2 ~ 1/f)
    fast = ms.skin_thermal_exposure(w, 100.0, frequency=1e5)
    assert fast.skin_depth == pytest.approx(base.skin_depth / math.sqrt(10.0), rel=1e-12)
    assert fast.cooling_time == pytest.approx(base.cooling_time / 10.0, rel=1e-12)
    assert fast.temperature_rise == base.temperature_rise                         # independent of the pulse length
    assert fast.recrystallised_fraction < base.recrystallised_fraction
    # a preheated part starts hotter
    warm = ms.skin_thermal_exposure(w, 100.0, initial_temperature=600.0)
    assert warm.peak_temperature == pytest.approx(600.0 + warm.temperature_rise, rel=1e-12)
    assert warm.recrystallised_fraction > base.recrystallised_fraction
    zero = ms.skin_thermal_exposure(w, 0.0)
    assert zero.temperature_rise == 0.0 and zero.peak_temperature == RT and zero.recrystallised_fraction < 1e-100
    assert zero.heat_affected is False and zero.surface_heat_affected is False
    # without an activation energy only melting can be judged
    c276 = ms.skin_thermal_exposure("hastelloy_c276", 60.0)
    assert c276.recrystallised_fraction is None and c276.recrystallised_fraction_band is None
    assert c276.recrystallisation_time is None and c276.surface_recrystallised_fraction is None
    assert c276.melted is False and c276.heat_affected is False
    for kw in (dict(peak_field=float("nan")), dict(peak_field=50.0, frequency=0.0), dict(peak_field=50.0, kappa=0.0),
               dict(peak_field=50.0, initial_temperature=4000.0), dict(peak_field=50.0, initial_temperature=0.0),
               dict(peak_field="strong")):
        with pytest.raises(InputError):
            ms.skin_thermal_exposure(w, **kw)
    with pytest.raises(UnknownMaterialError):
        ms.skin_thermal_exposure("unobtainium", 50.0)


# ------------------------------------------------------------------ assessment: headline results
def test_assess_tungsten_cold_pulse_headline(w):
    a = ms.assess_microstructure(w)                               # strain 0.3, 1e4 1/s, room temperature
    assert isinstance(a, Assessment) and a.material == "tungsten"
    assert (a.strain, a.strain_rate, a.temperature) == (0.3, 1e4, RT)
    # dislocations: a cell structure inside the original grains
    assert a.initial_dislocation_density == 1e12
    assert a.dislocation_density == pytest.approx(3.652e13, rel=1e-3)
    assert a.dislocation_density_band[0] == pytest.approx(3.652e13, rel=1e-3)
    assert a.dislocation_density_band[1] == pytest.approx(1.37e14, rel=1e-2)
    assert a.cell_size == pytest.approx(1.655e-6, rel=1e-3) and a.cell_size < a.grain_size_after
    assert a.stored_energy == pytest.approx(2.2e5, rel=2e-2)
    # grains: flattened, same order of magnitude, not ultrafine; SPD strains would be needed
    assert a.grain_size_before == 20e-6 and a.grain_size_after == pytest.approx(14.84e-6, rel=1e-3)
    assert a.ultrafine is False and a.saturation_grain_size == 100e-9
    assert a.strain_for_ultrafine == pytest.approx(3.096, rel=1e-3) and a.saturation_strain == pytest.approx(5.293, rel=1e-3)
    # strength: modest work hardening; the ultra-fine structure would be far stronger, if it existed
    assert a.yield_strength_before == pytest.approx(1.233e9, rel=1e-3)
    assert a.yield_strength_after == pytest.approx(1.416e9, rel=1e-3)
    assert a.hardening_ratio == pytest.approx(1.148, rel=1e-3)
    assert a.hardness_before == pytest.approx(4.214e9, rel=1e-3) and a.hardness_after == pytest.approx(4.256e9, rel=1e-3)
    assert a.ultrafine_yield_strength == pytest.approx(3.717e9, rel=1e-3)
    # heat: the bulk stays cold and nothing recrystallises - the grain structure is preserved
    assert a.bulk_temperature_rise == pytest.approx(189.2, rel=1e-2)
    assert a.bulk_final_temperature == pytest.approx(RT + a.bulk_temperature_rise, rel=1e-12)
    assert a.bulk_homologous_temperature == pytest.approx(0.1306, rel=1e-2)
    assert a.deformation_regime == "cold" and a.log10_zener_hollomon == pytest.approx(67.34, abs=0.05)
    assert a.bulk_hold_time == ms.BULK_HOLD_TIME and a.bulk_recrystallised_fraction < 1e-50
    assert a.grain_structure_preserved is True
    # localisation and brittleness keep it from being ordinary practice
    assert a.shear_localisation_strain == pytest.approx(0.004, abs=1e-3) and a.localisation_exceeded is True
    assert a.margin == pytest.approx(a.shear_localisation_strain / 0.3, rel=1e-12)
    assert a.shear_band_grain_size[0] == pytest.approx(0.19e-6, rel=2e-2)
    assert a.shear_band_grain_size[1] == pytest.approx(6.0e-6, rel=2e-2)
    assert (a.dbtt, a.below_dbtt, a.failure_strain_limit, a.ductility_exceeded) == (573.0, True, 0.07, True)
    assert a.status is Status.EXTRAPOLATION and a.extrapolated_model is False
    assert any("adiabatic shear localisation" in x for x in a.warnings)
    assert any("DBTT" in x and "confinement" in x for x in a.warnings)
    assert a.skin is None and a.haz_depth == 0.0
    assert any("NOT modelled" in n and "toughness" in n and "fatigue" in n for n in a.notes)
    assert any("grain structure is preserved" in n for n in a.notes)
    assert any("ECAP" in n and "not of one forming pulse" in n for n in a.notes)
    assert any("No field given" in n for n in a.notes)


def test_assess_status_rules():
    # ordinary cold work of a ductile fcc metal is state of the art
    cu = ms.assess_microstructure("copper_ofhc")
    assert cu.status is Status.STATE_OF_THE_ART and cu.warnings == ()
    assert cu.margin == pytest.approx(cu.shear_localisation_strain / 0.3, rel=1e-12) and cu.margin > 1.0
    assert cu.grain_structure_preserved and not cu.localisation_exceeded and not cu.ductility_exceeded
    assert cu.dislocation_density == pytest.approx(6.30e14, rel=1e-2) and cu.cell_size == pytest.approx(0.40e-6, rel=2e-2)
    assert cu.hardening_ratio > 3.0 and cu.below_dbtt is None
    assert ms.assess_microstructure("steel_1018").status is Status.STATE_OF_THE_ART
    # rule 1: hot, slow deformation recrystallises the bulk
    hot = ms.assess_microstructure("copper_ofhc", 0.3, 1e-3, 800.0)
    assert hot.status is Status.EXTRAPOLATION and hot.grain_structure_preserved is False
    assert hot.deformation_regime == "hot" and hot.bulk_recrystallised_fraction == pytest.approx(1.0)
    assert any("not preserved" in x for x in hot.warnings)
    assert not any("grain structure is preserved" in n for n in hot.notes)
    # rule 3: fallback-only niobium at 1e4 1/s is an extrapolated model
    nb = ms.assess_microstructure("niobium")
    assert nb.status is Status.EXTRAPOLATION and nb.extrapolated_model is True
    assert any("extrapolated" in x for x in nb.warnings) and not nb.localisation_exceeded
    # rule 4 alone: gamma-TiAl far below its DBTT with 1.5 % ductility
    tial = ms.assess_microstructure("gamma_tial")
    assert tial.status is Status.EXTRAPOLATION and tial.ductility_exceeded and not tial.localisation_exceeded
    assert [x for x in tial.warnings if "DBTT" in x] and len(tial.warnings) == 1
    assert any("200" in n or "0.2 um" in n for n in tial.notes if "saturation" in n)   # estimated saturation size noted


def test_assess_skin_rules(w):
    # rule 6: bare copper at 100 T melts its skin
    melt = ms.assess_microstructure("copper_ofhc", peak_field=100.0)
    assert melt.status is Status.INFEASIBLE and melt.skin.melted is True
    assert melt.haz_depth == melt.skin.heated_depth == pytest.approx(0.329e-3, rel=1e-2)
    assert any("skin melts" in x and "driver" in x for x in melt.warnings)
    assert melt.grain_structure_preserved is True                 # the bulk underneath is still cold-worked
    assert ms.assess_microstructure("aluminum_6061", peak_field=63.41).status is Status.INFEASIBLE
    # rule 5: heat-affected zone on the layer estimate (copper at 63 T, molybdenum at 100 T)
    haz = ms.assess_microstructure("copper_ofhc", peak_field=63.41)
    assert haz.status is Status.EXTRAPOLATION and haz.skin.heat_affected and not haz.skin.melted
    assert any("heat-affected zone" in x for x in haz.warnings)
    mo = ms.assess_microstructure("molybdenum", peak_field=100.0)
    assert mo.status is Status.EXTRAPOLATION and mo.skin.heat_affected and mo.haz_depth > 0.5e-3
    # rule 5: heat-affected on the surface estimate only (tungsten at 100 T, copper at 50 T)
    w100 = ms.assess_microstructure(w, peak_field=100.0)
    assert w100.status is Status.EXTRAPOLATION and w100.skin.heat_affected is False
    assert w100.skin.surface_heat_affected is True and w100.haz_depth == w100.skin.heated_depth
    assert any("heat-affected at its surface" in x and "melts on the upper estimate" in x for x in w100.warnings)
    cu50 = ms.assess_microstructure("copper_ofhc", peak_field=50.0)
    assert cu50.status is Status.EXTRAPOLATION and len(cu50.warnings) == 1
    # below those fields the skin is only described in the notes and does not change the status
    cu30 = ms.assess_microstructure("copper_ofhc", peak_field=30.0)
    assert cu30.status is Status.STATE_OF_THE_ART and cu30.haz_depth == 0.0 and cu30.warnings == ()
    assert any("Eddy-current skin at 30 T" in n for n in cu30.notes)
    w63 = ms.assess_microstructure(w, peak_field=63.41)
    assert w63.haz_depth == 0.0 and not any("skin" in x for x in w63.warnings)
    assert w63.warnings == ms.assess_microstructure(w).warnings
    # the layer estimate with kappa = 2.4 melts tungsten at 100 T
    assert ms.assess_microstructure(w, peak_field=100.0, kappa=2.4).status is Status.INFEASIBLE
    # frequency is passed through (and ignored without a field)
    assert ms.assess_microstructure(w, peak_field=100.0, frequency=1e5).skin.frequency == 1e5
    assert ms.assess_microstructure(w, frequency=1e5).skin is None


def test_assess_margin_grain_options_and_missing_data(cu):
    # zero strain: nothing happens, no margin
    still = ms.assess_microstructure(cu, 0.0)
    assert still.margin is None and still.status is Status.STATE_OF_THE_ART
    assert still.dislocation_density == pytest.approx(1e12, rel=1e-12) and still.grain_size_after == still.grain_size_before
    assert still.hardening_ratio == 1.0 and still.bulk_temperature_rise == 0.0
    # no adiabatic instability below 5: the margin is the lower bound 5 / strain
    cold = cu.with_changes(key="test_cu_no_heating", specific_heat=1e12)
    steady = ms.assess_microstructure(cold)
    assert steady.shear_localisation_strain is None and steady.localisation_exceeded is False
    assert steady.margin == pytest.approx(ms.CULVER_MAX_STRAIN / 0.3, rel=1e-12) and steady.shear_band_grain_size is None
    # feedstock grain size: explicit, and an already ultrafine feedstock stays ultrafine
    coarse = ms.assess_microstructure(cu, initial_grain_size=100e-6)
    assert coarse.grain_size_before == 100e-6 and coarse.grain_size_after > 70e-6 and coarse.ultrafine is False
    ufg = ms.assess_microstructure(cu, initial_grain_size=0.5e-6)
    assert ufg.ultrafine is True and ufg.strain_for_ultrafine == 0.0
    assert any("feedstock already was" in n for n in ufg.notes)
    # an SPD-scale strain does refine the grains - but as one adiabatic pulse it also cooks them
    spd = ms.assess_microstructure(cu, 6.0)
    assert spd.ultrafine is True and spd.grain_size_after == pytest.approx(0.274e-6, rel=1e-2)
    assert spd.grain_structure_preserved is False and spd.status is Status.EXTRAPOLATION
    # a preheated start is reported at room temperature for the as-formed strength
    warm = ms.assess_microstructure("tungsten", temperature=700.0)
    assert warm.below_dbtt is False and warm.ductility_exceeded is False
    assert warm.yield_strength_before == pytest.approx(ms.assess_microstructure("tungsten").yield_strength_before)
    # initial dislocation density of wrought stock
    wrought = ms.assess_microstructure("tungsten", initial_density=1e14)
    assert wrought.initial_dislocation_density == 1e14 and wrought.dislocation_density > 1e14
    # no activation energy recorded: judged from the homologous temperature
    c276 = ms.assess_microstructure("hastelloy_c276", 0.3, 1e-3)
    assert c276.bulk_recrystallised_fraction is None and c276.deformation_regime is None
    assert c276.log10_zener_hollomon is None and c276.grain_structure_preserved is True
    assert c276.ultrafine_yield_strength is None and c276.saturation_grain_size == ms.DEFAULT_SATURATION_GRAIN_SIZE
    assert any("no activation energy" in n for n in c276.notes)
    # bulk hold time is a parameter
    assert ms.assess_microstructure(cu, bulk_hold_time=3600.0).bulk_hold_time == 3600.0


def test_assess_input_validation_and_material_handling(w):
    for kw in (dict(strain=-0.1), dict(strain_rate=-1.0), dict(temperature=0.0), dict(temperature=4000.0),
               dict(peak_field=float("inf")), dict(peak_field=50.0, frequency=-1.0), dict(peak_field=50.0, kappa=0.0),
               dict(initial_grain_size=0.0), dict(bulk_hold_time=-1.0), dict(initial_density=-1.0),
               dict(strain=float("nan"))):
        with pytest.raises(InputError):
            ms.assess_microstructure(w, **kw)
    for key in ("mp35n", "diamond", "zylon_fibre"):               # no grain size / Burgers vector / flow stress
        with pytest.raises(InputError):
            ms.assess_microstructure(key)
    with pytest.raises(InputError):                               # still no Burgers vector with a grain size
        ms.assess_microstructure("mp35n", initial_grain_size=1e-5)
    with pytest.raises(UnknownMaterialError):
        ms.assess_microstructure("unobtainium")
    by_alias = ms.assess_microstructure("W", peak_field=100.0)
    by_object = ms.assess_microstructure(w, peak_field=100.0)
    assert by_alias.material == by_object.material == "tungsten"
    assert by_alias.json() == by_object.json()                    # deterministic, independent of how it is named


def test_refractory_metals_of_the_concept_note():
    """W, Mo and Re at the concept note's 2 GPa (70.9 T) and 4 GPa (100.27 T): never state of the art."""
    for key in ("tungsten", "molybdenum", "rhenium"):
        for field in (70.9, 100.27):
            a = ms.assess_microstructure(key, peak_field=field)
            assert a.grain_structure_preserved is True, (key, field)       # 'preserves': holds in the bulk
            assert a.ultrafine is False and a.saturation_strain > 4.0, (key, field)   # 'ultra-fine': not created
            assert a.status in (Status.EXTRAPOLATION, Status.INFEASIBLE), (key, field)
            assert a.skin.melted is False, (key, field)                    # refractory skins do not melt (kappa 1)
        assert ms.assess_microstructure(key, peak_field=100.27).skin.surface_heat_affected is True, key


# ------------------------------------------------------------------ serialisation
def test_results_round_trip_through_json_and_render_as_text(w):
    sk = ms.skin_thermal_exposure(w, 100.0)
    data = json.loads(sk.json())
    assert data["material"] == "tungsten" and data["melted"] is False and data["surface_melted"] is True
    assert data["recrystallised_fraction_band"][1] == 1.0
    assert data["peak_temperature"] == pytest.approx(sk.peak_temperature)
    text = sk.text()
    assert text.startswith("Skin thermal exposure") and "peak_temperature [K] = " in text
    assert "cooling_time [s] = " in text and "skin_depth [m] = " in text
    melted = json.loads(ms.skin_thermal_exposure("copper_ofhc", 100.0).json())
    assert melted["recrystallisation_time"] is None and melted["melted"] is True

    a = ms.assess_microstructure(w, peak_field=100.0)
    data = json.loads(a.json())
    assert data["status"] == "extrapolation" and data["material"] == "tungsten"
    assert data["skin"]["material"] == "tungsten" and data["skin"]["surface_heat_affected"] is True
    assert data["dislocation_density_band"] == pytest.approx(list(a.dislocation_density_band))
    assert data["shear_band_grain_size"] == pytest.approx(list(a.shear_band_grain_size))
    assert data["margin"] == pytest.approx(a.margin) and data["ultrafine"] is False
    text = a.text()
    assert text.startswith("Microstructure assessment") and "material = tungsten" in text
    assert "dislocation_density [1/m^2] = " in text and "grain_size_after [m] = " in text and "skin:" in text
    plain = json.loads(ms.assess_microstructure("copper_ofhc", 0.0).json())
    assert plain["skin"] is None and plain["margin"] is None and plain["status"] == "state_of_the_art"
    # an infinite refinement strain serialises as null (valid JSON)
    fine_target = ms.assess_microstructure("copper_ofhc", initial_grain_size=5e-6)
    assert json.loads(fine_target.json())["saturation_strain"] == pytest.approx(fine_target.saturation_strain)
    with pytest.raises(Exception):
        a.strain = 1.0  # frozen
