"""Tests for magform.plasticity (spec section 3): flow stress, forging and forgeability."""

from __future__ import annotations

import json
import math

import pytest

from magform import materials
from magform import plasticity as pl
from magform._numerics import trapz
from magform.constants import T_ROOM
from magform.errors import InputError, UnknownMaterialError
from magform.materials import JohnsonCook
from magform.status import Status

RT = T_ROOM


def _ideal_plastic(A: float = 1.0e9, melting_point: float = 1723.0) -> materials.Material:
    """Heavy-alloy record with sigma = A (1 - T*): no hardening, no rate term, linear softening."""
    return materials.get("tungsten_heavy_alloy_90w").with_changes(
        key="test_ideal_plastic", melting_point=melting_point,
        constitutive=(JohnsonCook(A, 0.0, 1.0, 0.0, 1.0, 1.0, t_melt=melting_point, source="test"),))


def _bisect(f, lo, hi, n=200):
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ constants / API
def test_constants_and_public_api_match_spec():
    assert pl.RATE_FLOOR == 1e-6
    assert pl.FALLBACK_REFERENCE_RATE == 1e-3
    assert pl.FALLBACK_RATE_SENSITIVITY == {"fcc": 0.015, "bcc": 0.04, "hcp": 0.025, "L1_0": 0.03}
    assert pl.DEFAULT_RATE_SENSITIVITY == 0.02
    assert pl.TAYLOR_QUINNEY == 0.9
    assert pl.MIN_FLOW_STRESS == 1e5
    for name in pl.__all__:
        assert hasattr(pl, name), name
    assert pl.REFERENCES and all(isinstance(v, str) and len(v) > 20 for v in pl.REFERENCES.values())
    for name in ("flow_stress", "fallback_flow_stress", "flow_stress_band", "forging_pressure",
                 "adiabatic_fraction", "adiabatic_temperature_rise", "max_strain_at_pressure",
                 "warm_forming_temperature", "adiabatic_shear_critical_strain", "forging_practice_gap",
                 "assess_forgeability", "FlowStressBand", "PracticeGap", "ForgeAssessment"):
        assert getattr(pl, name).__doc__, name
    assert pl.fallback_rate_sensitivity("niobium") == 0.04
    assert pl.fallback_rate_sensitivity("copper_ofhc") == 0.015
    assert pl.fallback_rate_sensitivity("graphite_crystal") == pl.DEFAULT_RATE_SENSITIVITY


# ------------------------------------------------------------------ flow stress
def test_flow_stress_dispatches_to_primary_fit(w):
    primary = w.constitutive[0]
    expected = primary.stress(0.3, 1e4, RT, RT, w.melting_point)
    assert pl.flow_stress(w, 0.3, 1e4, RT) == pytest.approx(expected, rel=1e-12)
    assert pl.flow_stress(w, 0.3, 1e4, RT) == pytest.approx(2.05e9, rel=2e-2)          # Lennon-Ramesh Z-A
    assert pl.flow_stress("tungsten_heavy_alloy_90w", 0.3, 1e4, RT) == pytest.approx(1.904e9, rel=1e-2)
    assert pl.flow_stress("molybdenum", 0.3, 1e4, RT) == pytest.approx(1.57e9, rel=2e-2)
    # eddy.py calls it positionally through functools.partial(flow_stress, material)
    from functools import partial
    assert partial(pl.flow_stress, w)(0.3, 1e4, RT) == pl.flow_stress(w, 0.3, 1e4, RT)


def test_fallback_closed_form_for_material_without_fit():
    nb = materials.get("niobium")
    assert not nb.constitutive
    K, n = nb.hollomon()
    rate_factor = 1.0 + 0.04 * math.log(1e4 / 1e-3)
    expected = max(nb.yield_strength, K * 0.3 ** n) * rate_factor
    assert pl.flow_stress(nb, 0.3, 1e4, RT) == pytest.approx(expected, rel=1e-12)
    assert pl.fallback_flow_stress(nb, 0.3, 1e4, RT) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(592e6, rel=1e-2)
    # linear thermal softening (1 - T*), T* = (T - T_ROOM)/(Tm - T_ROOM)
    t = 1200.0
    t_star = (t - RT) / (nb.melting_point - RT)
    assert pl.flow_stress(nb, 0.3, 1e4, t) == pytest.approx(expected * (1.0 - t_star), rel=1e-12)


def test_fallback_clamps_rate_and_temperature(cu):
    base = pl.fallback_flow_stress(cu, 0.2, 1e-3, RT)
    # rate factor max(1, 1 + C ln(rate/1e-3)): no softening below the reference rate, zero rate allowed
    assert pl.fallback_flow_stress(cu, 0.2, 1e-5, RT) == pytest.approx(base, rel=1e-12)
    assert pl.fallback_flow_stress(cu, 0.2, 0.0, RT) == pytest.approx(base, rel=1e-12)
    assert pl.fallback_flow_stress(cu, 0.2, 1e3, RT) == pytest.approx(base * (1 + 0.015 * math.log(1e6)), rel=1e-12)
    # T* clamped to [0, 0.999]: no change below room temperature, 0.1 % of the strength at T -> Tm
    assert pl.fallback_flow_stress(cu, 0.2, 1e-3, 77.0) == pytest.approx(base, rel=1e-12)
    near_melt = pl.fallback_flow_stress(cu, 0.2, 1e-3, cu.melting_point - 1e-9)
    assert near_melt == pytest.approx(base * 1e-3, rel=1e-6)
    # but never below the 1e5 Pa floor (Al 1100: 34 MPa x 0.001 = 34 kPa)
    al1100 = materials.get("aluminum_1100")
    assert pl.flow_stress(al1100, 0.0, 1e-3, al1100.melting_point - 1e-9) == pl.MIN_FLOW_STRESS


def test_flow_stress_input_validation(w):
    bad = [dict(strain=-0.1), dict(strain_rate=-1.0), dict(temperature=0.0), dict(temperature=-5.0),
           dict(temperature=w.melting_point), dict(temperature=w.melting_point + 100.0),
           dict(strain=float("nan")), dict(strain_rate=float("inf")), dict(temperature="hot")]
    for kw in bad:
        args = dict(strain=0.1, strain_rate=1e3, temperature=RT)
        args.update(kw)
        with pytest.raises(InputError):
            pl.flow_stress(w, **args)
        with pytest.raises(InputError):
            pl.flow_stress_band(w, **args)
    with pytest.raises(InputError):  # no yield/UTS and no fit
        pl.flow_stress("diamond", 0.1)
    with pytest.raises(InputError):
        pl.fallback_flow_stress("zylon_fibre", 0.01)
    with pytest.raises(UnknownMaterialError):
        pl.flow_stress("unobtainium", 0.1)
    # zero rate is the quasi-static limit, not an error
    assert pl.flow_stress(w, 0.1, 0.0) == pl.flow_stress(w, 0.1, pl.RATE_FLOOR)


def test_flow_stress_monotonic_in_strain_rate_and_temperature():
    for key in ("tungsten", "tantalum", "molybdenum", "copper_ofhc", "niobium", "rhenium",
                "tungsten_heavy_alloy_90w"):
        m = materials.get(key)
        strains = [pl.flow_stress(m, e, 1e4, RT) for e in (0.0, 0.05, 0.1, 0.3, 1.0)]
        assert all(b >= a for a, b in zip(strains, strains[1:])), key
        rates = [pl.flow_stress(m, 0.2, r, RT) for r in (1e-3, 1.0, 1e3, 1e4)]
        assert all(b >= a for a, b in zip(rates, rates[1:])), key
        # ascending temperatures (0.5 Tm is below 900 K for copper and the heavy alloy)
        ladder = sorted((RT, 600.0, 900.0, 0.5 * m.melting_point))
        temps = [pl.flow_stress(m, 0.2, 1e4, t) for t in ladder]
        assert all(b <= a for a, b in zip(temps, temps[1:])), key
        assert all(s >= pl.MIN_FLOW_STRESS for s in strains + rates + temps)


# ------------------------------------------------------------------ band
def test_flow_stress_band_tungsten_spec_check(w):
    band = pl.flow_stress_band(w, 0.3, 1e4, RT)
    assert band.minimum == pytest.approx(2.05e9, rel=2e-2)
    assert band.maximum == pytest.approx(2.58e9, rel=2e-2)
    assert band.primary == pl.flow_stress(w, 0.3, 1e4, RT)
    assert band.primary_model == w.constitutive[0].source
    assert len(band.band_models) == 2 and "fallback" in band.values and len(band.values) == 3
    assert band.minimum <= band.primary <= band.maximum
    assert band.spread == pytest.approx(band.maximum / band.minimum)
    # 1e4 1/s is within a decade of the 4e3 1/s fit range; 1e6 1/s and 1500 K are not
    assert band.extrapolated is False and band.extrapolation_note == ""
    fast = pl.flow_stress_band(w, 0.3, 1e6, RT)
    assert fast.extrapolated is True and "strain rate" in fast.extrapolation_note
    hot = pl.flow_stress_band(w, 0.3, 1e4, 1500.0)
    assert hot.extrapolated is True and "1273" in hot.extrapolation_note
    # 900 K: Zerilli-Armstrong fits give 1.0-1.5 GPa
    warm = pl.flow_stress_band(w, 0.3, 1e4, 900.0)
    assert warm.minimum == pytest.approx(1.00e9, rel=2e-2)
    assert warm.maximum == pytest.approx(1.52e9, rel=2e-2)


def test_band_uses_fallback_only_without_published_fits():
    nb = pl.flow_stress_band("niobium", 0.3, 1e4, RT)
    assert nb.band_models == ("fallback",) and nb.primary_model == "fallback"
    assert nb.minimum == nb.maximum == nb.primary == nb.values["fallback"]
    assert nb.extrapolated is True  # quasi-static tensile data used at 1e4 1/s
    assert pl.flow_stress_band("niobium", 0.3, 1e-3, RT).extrapolated is False
    # the crude fallback is reported for comparison but does not widen a published band
    wha = pl.flow_stress_band("tungsten_heavy_alloy_90w", 0.3, 1e4, RT)
    assert wha.values["fallback"] == pytest.approx(2.94e9, rel=2e-2)
    assert wha.maximum == pytest.approx(1.904e9, rel=1e-2)
    assert wha.maximum < wha.values["fallback"]
    ta = pl.flow_stress_band("tantalum", 0.3, 1e4, RT)
    assert ta.minimum == pytest.approx(1.014e9, rel=1e-2) and ta.maximum == pytest.approx(1.058e9, rel=1e-2)
    assert ta.values["fallback"] < ta.minimum


# ------------------------------------------------------------------ forging pressure
def test_forging_pressure_slab_formula(cu):
    sigma = pl.flow_stress(cu, 0.2, 1e3, RT)
    assert pl.forging_pressure(cu, 0.2, 1e3, RT) == pytest.approx(sigma, rel=1e-12)
    assert pl.forging_pressure(cu, 0.2, 1e3, RT, 1.2, 0.1, 3.0) == pytest.approx(1.2 * sigma * 1.1, rel=1e-12)
    assert pl.forging_pressure(cu, 0.2, 1e3, RT, 2 / math.sqrt(3)) == pytest.approx(1.1547 * sigma, rel=1e-4)
    for kw in (dict(constraint_factor=0.0), dict(friction=-0.1), dict(diameter_to_height=-1.0)):
        with pytest.raises(InputError):
            pl.forging_pressure(cu, 0.2, **kw)


# ------------------------------------------------------------------ adiabatic heating
def test_adiabatic_fraction_formula_and_limits(w):
    alpha = w.thermal_conductivity / (w.density * w.specific_heat)
    t_cond = 1e-6 / alpha
    assert pl.adiabatic_fraction(w, 0.3, 1e4) > 0.999
    assert pl.adiabatic_fraction(w, 0.3, 1e-3) == pytest.approx(t_cond / 300.0, rel=1e-4)
    rate = 0.3 / t_cond  # t_def = t_cond
    assert pl.adiabatic_fraction(w, 0.3, rate) == pytest.approx(1.0 - math.exp(-1.0), rel=1e-12)
    fracs = [pl.adiabatic_fraction(w, 0.3, r) for r in (1e-3, 1e-1, 10.0, 1e3)]
    assert all(b > a for a, b in zip(fracs, fracs[1:]))
    assert pl.adiabatic_fraction(w, 0.3, 0.0) == 0.0 and pl.adiabatic_fraction(w, 0.0, 1.0) == 1.0
    with pytest.raises(InputError):
        pl.adiabatic_fraction(w, 0.3, 1.0, length_scale=0.0)


def test_adiabatic_temperature_rise_tungsten_bounds_and_convergence(w):
    d_t = pl.adiabatic_temperature_rise(w, 0.3, 1e4)
    assert d_t == pytest.approx(189.2, rel=1e-2)
    assert pl.adiabatic_temperature_rise(w, 0.3, 1e4, n_steps=2000) == pytest.approx(d_t, rel=1e-3)
    # bracketed by the isothermal work integrals at the initial and the final temperature
    rho_c = w.density * w.specific_heat
    eps = [0.3 * i / 300 for i in range(301)]
    hot = trapz([pl.flow_stress(w, e, 1e4, RT + d_t) for e in eps], eps)
    cold = trapz([pl.flow_stress(w, e, 1e4, RT) for e in eps], eps)
    assert 0.9 * hot / rho_c < d_t < 0.9 * cold / rho_c
    # quasi-static deformation is isothermal; no heat without plastic work
    assert pl.adiabatic_temperature_rise(w, 0.3, 1e-3) < 0.1
    assert pl.adiabatic_temperature_rise(w, 0.0, 1e4) == 0.0
    assert pl.adiabatic_temperature_rise(w, 0.3, 1e4, taylor_quinney=0.0) == 0.0
    for kw in (dict(taylor_quinney=1.5), dict(n_steps=0), dict(n_steps=2.5), dict(length_scale=-1.0)):
        with pytest.raises(InputError):
            pl.adiabatic_temperature_rise(w, 0.3, 1e4, **kw)


def test_adiabatic_temperature_rise_matches_analytic_linear_softening():
    mat = _ideal_plastic()
    # d(T - Tr)/d eps = beta A (1 - (T - Tr)/dTm)/(rho c)  ->  T - Tr = dTm (1 - exp(-beta A eps/(rho c dTm)))
    rho_c = mat.density * mat.specific_heat
    d_tm = mat.melting_point - RT
    f = pl.adiabatic_fraction(mat, 0.5, 1e4)
    expected = d_tm * (1.0 - math.exp(-f * 0.9 * 1.0e9 * 0.5 / (rho_c * d_tm)))
    assert pl.adiabatic_temperature_rise(mat, 0.5, 1e4) == pytest.approx(expected, rel=1e-6)
    # the march stops at 0.95 Tm
    strong = _ideal_plastic(A=2.0e10)
    assert RT + pl.adiabatic_temperature_rise(strong, 5.0, 1e4) == pytest.approx(0.95 * strong.melting_point)


# ------------------------------------------------------------------ pressure / temperature windows
def test_max_strain_at_pressure(w):
    q = 1.2
    sigma0 = pl.flow_stress(w, 0.0, 1e4, RT)
    assert pl.max_strain_at_pressure(w, 0.99 * q * sigma0, 1e4, RT, q) == 0.0
    assert pl.max_strain_at_pressure(w, 10e9, 1e4, RT, q, max_strain=3.0) == 3.0
    p = q * pl.flow_stress(w, 0.1, 1e4, RT)
    assert pl.max_strain_at_pressure(w, p, 1e4, RT, q) == pytest.approx(0.1, rel=1e-6)
    # annealed copper yields at 90 MPa x 1.23 (rate factor at 1e4 1/s) = 111 MPa: 100 MPa cannot start flow
    reach = [pl.max_strain_at_pressure("copper_ofhc", p_, 1e4) for p_ in (1e8, 2e8, 3e8, 4e8, 5e8)]
    assert all(b >= a for a, b in zip(reach, reach[1:])) and reach[0] == 0.0 and reach[-1] > 0.3
    assert 0.0 < reach[1] < 0.05
    with pytest.raises(InputError):
        pl.max_strain_at_pressure(w, -1.0)


def test_warm_forming_temperature_tungsten(w):
    za = w.constitutive[0]
    p, q, eps, rate = 2e9, 1.2, 0.3, 1e4
    # closed form for the Zerilli-Armstrong fit: c1 exp(T (c4 ln rate - c3)) = P/Q - c0 - c5 eps^n
    analytic = math.log((p / q - za.c0 - za.c5 * eps ** za.n) / za.c1) / (za.c4 * math.log(rate) - za.c3)
    t_w = pl.warm_forming_temperature(w, p, eps, rate, q)
    assert t_w == pytest.approx(analytic, rel=1e-5) and t_w == pytest.approx(459.6, rel=1e-3)
    assert q * pl.flow_stress(w, eps, rate, t_w) <= p * (1 + 1e-9)
    assert q * pl.flow_stress(w, eps, rate, t_w - 1.0) > p
    t_cons = pl.warm_forming_temperature(w, p, eps, rate, q, conservative=True)
    assert t_cons > t_w
    assert pl.warm_forming_temperature(w, 4e9, eps, rate, q) == RT
    assert pl.warm_forming_temperature(w, 0.1e9, eps, rate, q) is None  # athermal floor ~0.23 GPa


# ------------------------------------------------------------------ adiabatic shear (Culver)
def test_culver_heavy_alloy_check_and_closed_form():
    wha = materials.get("tungsten_heavy_alloy_90w")
    jc = wha.constitutive[0]
    rate_factor = 1.0 + jc.C * math.log(1e4 / jc.eps0)
    rho_c = wha.density * wha.specific_heat

    def criterion(e: float, t_melt: float, rc: float) -> float:  # JC with m = 1 (T cancels out)
        return jc.B * jc.n * e ** (jc.n - 1) - 0.9 * rate_factor * (jc.A + jc.B * e ** jc.n) ** 2 / (rc * (t_melt - RT))

    analytic = _bisect(lambda e: criterion(e, 1723.0, rho_c), 1e-6, 1.0)
    assert analytic == pytest.approx(0.0165, rel=1e-2)
    eps_c = pl.adiabatic_shear_critical_strain(wha, 1e4, RT)
    # spec check 'about 0.05 (0.02-0.12)' assumed pure-W Tm = 3695 K; with the 1723 K matrix solidus
    # of the heavy-alloy record the spec's own equations give 0.0165 (see deviations)
    assert abs(eps_c - analytic) <= 1e-3
    fine = pl.adiabatic_shear_critical_strain(wha, 1e4, RT, d_eps=1e-4)
    assert abs(fine - analytic) <= 2e-4
    # the 0.05 of the spec is recovered with pure-tungsten melting point and heat capacity
    pure_w_tm = wha.with_changes(key="wha_pure_w_tm", melting_point=3695.0, density=19250.0, specific_heat=134.0,
                                 constitutive=(JohnsonCook(jc.A, jc.B, jc.n, jc.C, jc.m, jc.eps0, t_melt=3695.0),))
    legacy = pl.adiabatic_shear_critical_strain(pure_w_tm, 1e4, RT)
    assert 0.02 <= legacy <= 0.12 and legacy == pytest.approx(0.047, abs=2e-3)


def test_culver_pure_tungsten_and_limits(w):
    eps_c = pl.adiabatic_shear_critical_strain(w, 1e4, RT)
    assert 0.003 <= eps_c <= 0.006
    # no heating -> no thermal softening -> monotone hardening, no instability
    assert pl.adiabatic_shear_critical_strain(w, 1e4, RT, taylor_quinney=0.0, max_strain=1.0) is None
    # an ideally plastic solid is adiabatically unstable as soon as it yields
    assert pl.adiabatic_shear_critical_strain(_ideal_plastic(), 1e4) == pytest.approx(pl.CULVER_START_STRAIN)
    # strongly hardening fcc copper localises far later than tungsten
    assert pl.adiabatic_shear_critical_strain("copper_ofhc", 1e4) > 1.0
    with pytest.raises(InputError):
        pl.adiabatic_shear_critical_strain(w, 1e4, d_eps=5.0, max_strain=5.0)
    with pytest.raises(InputError):
        pl.adiabatic_shear_critical_strain(w, 1e4, taylor_quinney=1.2)
    with pytest.raises(InputError):
        pl.adiabatic_shear_critical_strain(w, -1.0)


# ------------------------------------------------------------------ practice gap
def test_forging_practice_gap():
    gap = pl.forging_practice_gap("tungsten", RT)
    assert gap.material == "tungsten" and gap.conventional_range == (1473.0, 1873.0)
    assert gap.below_conventional is True
    assert gap.ratio_to_conventional_min == pytest.approx(RT / 1473.0, rel=1e-12)
    assert gap.homologous_temperature == pytest.approx(RT / 3695.0, rel=1e-12)
    assert gap.readme_claim_contradicted is True and "1200-1600 C" in gap.verdict and "2,500 C" in gap.verdict
    assert gap.note == materials.get("tungsten").conventional_working_note
    ta = pl.forging_practice_gap("tantalum", RT)
    assert ta.below_conventional is False and "no practice gap" in ta.verdict and ta.readme_claim_contradicted
    hot = pl.forging_practice_gap("tungsten", 1600.0)
    assert hot.below_conventional is False and "within conventional practice" in hot.verdict
    none = pl.forging_practice_gap("mp35n", RT)
    assert none.conventional_range is None and none.ratio_to_conventional_min is None
    assert none.below_conventional is False and none.readme_claim_contradicted is None
    assert pl.forging_practice_gap("copper_ofhc").readme_claim_contradicted is None  # not refractory
    with pytest.raises(InputError):
        pl.forging_practice_gap("tungsten", 4000.0)


# ------------------------------------------------------------------ forgeability (spec checks)
def test_assess_tungsten_at_2gpa_is_infeasible(w):
    a = pl.assess_forgeability(w, 2e9)
    assert a.status is Status.INFEASIBLE
    assert a.margin == a.margin_conservative < a.margin_optimistic < 1.0
    assert a.required_pressure == pytest.approx(1.2 * a.flow_stress_max, rel=1e-12)
    assert a.flow_stress_min == pytest.approx(2.05e9, rel=2e-2) and a.flow_stress_max == pytest.approx(2.58e9, rel=2e-2)
    assert any("pressure insufficient at this temperature" in x for x in a.warnings)
    assert any("fractures unless confined" in x for x in a.warnings)
    assert a.warm_forming_temperature == pytest.approx(459.6, rel=1e-3)
    assert any("Warm forming" in x and "460 K" in x for x in a.notes)
    assert any("hydrostatic" in x and "deviatoric" in x for x in a.notes)
    assert any(a.practice_gap.verdict in x for x in a.notes)
    assert a.max_strain_at_pressure == 0.0  # 2 GPa < 1.2 x yield at 1e4 1/s


def test_assess_tungsten_at_4gpa_is_not_state_of_the_art(w):
    a = pl.assess_forgeability(w, 4e9)
    assert a.status is not Status.STATE_OF_THE_ART
    assert a.status is Status.EXTRAPOLATION
    assert a.below_dbtt is True and a.dbtt == 573.0 and a.confinement_ratio >= 1.0
    assert a.margin >= 1.0
    assert any("pressure-induced ductility" in x for x in a.warnings)
    assert any("adiabatic shear" in x for x in a.warnings)
    assert a.adiabatic_temperature_rise == pytest.approx(189.2, rel=1e-2)
    assert a.final_temperature == pytest.approx(RT + a.adiabatic_temperature_rise)
    assert a.elastic_volumetric_strain == pytest.approx(4e9 / w.bulk_modulus)
    # the same pressure without hydrostatic confinement fractures (Duemmer et al. 1998)
    bare = pl.assess_forgeability(w, 4e9, hydrostatic_pressure=0.0)
    assert bare.status is Status.INFEASIBLE and bare.confinement_ratio == 0.0
    assert any("fractures unless confined" in x for x in bare.warnings)


def test_assess_tantalum_and_molybdenum_checks():
    ta = pl.assess_forgeability("tantalum", 2e9)
    assert ta.status in (Status.STATE_OF_THE_ART, Status.EXTRAPOLATION)
    assert ta.margin >= 1.0 and ta.below_dbtt is None
    # the m = 0.4 Johnson-Cook fit makes an early Culver instability; it is flagged as a likely artifact
    assert any("artifact" in x for x in ta.warnings if "adiabatic shear" in x)
    mo = pl.assess_forgeability("molybdenum", 4e9)
    assert mo.margin >= 2.0
    assert mo.margin == pytest.approx(4e9 / (1.2 * 1.5715e9), rel=1e-2)


def test_assess_status_rules_straddle_extrapolation_and_sota():
    # rule 4: pressure between Q sigma_min and Q sigma_max of the tantalum fits
    band = pl.flow_stress_band("tantalum", 0.3, 1e4, RT)
    p = 1.2 * 0.5 * (band.minimum + band.maximum)
    a = pl.assess_forgeability("tantalum", p)
    assert a.margin_conservative < 1.0 <= a.margin_optimistic
    assert a.status is Status.EXTRAPOLATION and any("straddles" in x for x in a.warnings)
    # rule 6: fallback-only niobium at 1e4 1/s is an extrapolated model
    nb = pl.assess_forgeability("niobium", 2e9)
    assert nb.extrapolated_model is True and nb.status is Status.EXTRAPOLATION
    assert any("extrapolated" in x for x in nb.warnings)
    # rule 7: electromagnetic forming of copper at 2 GPa is industrial practice
    cu = pl.assess_forgeability("copper_ofhc", 2e9)
    assert cu.status is Status.STATE_OF_THE_ART and cu.warnings == () and cu.margin > 3.0
    # rule 1 needs only the weakest fit to fail
    low = pl.assess_forgeability("copper_ofhc", 0.1e9)
    assert low.status is Status.INFEASIBLE


def test_assess_input_validation_and_material_handling(w):
    for kw in (dict(pressure=-1.0), dict(pressure=4e9, constraint_factor=0.0),
               dict(pressure=4e9, hydrostatic_pressure=-1.0), dict(pressure=4e9, temperature=4000.0),
               dict(pressure=4e9, target_strain=-0.1), dict(pressure=float("nan"))):
        with pytest.raises(InputError):
            pl.assess_forgeability(w, **kw)
    by_alias = pl.assess_forgeability("W", 4e9)
    by_object = pl.assess_forgeability(w, 4e9)
    assert by_alias.material == by_object.material == "tungsten"
    assert by_alias.json() == by_object.json()  # deterministic and independent of how the material is given
    assert pl.assess_forgeability(w, 4e9).json() == by_object.json()


# ------------------------------------------------------------------ serialisation
def test_results_round_trip_through_json_and_render_as_text(w):
    band = pl.flow_stress_band(w, 0.3, 1e4, RT)
    gap = pl.forging_practice_gap(w, RT)
    a = pl.assess_forgeability(w, 2e9)
    for result, title in ((band, "Flow-stress band"), (gap, "Forging practice gap"),
                          (a, "Forgeability assessment")):
        data = json.loads(result.json())
        assert isinstance(data, dict) and data["material"] == "tungsten"
        text = result.text()
        assert text.startswith(title) and "material = tungsten" in text
    data = json.loads(band.json())
    assert data["values"]["fallback"] == pytest.approx(band.values["fallback"])
    assert data["minimum"] == pytest.approx(band.minimum)
    gap_data = json.loads(gap.json())
    assert gap_data["conventional_range"] == [1473.0, 1873.0] and gap_data["below_conventional"] is True
    a_data = json.loads(a.json())
    assert a_data["status"] == "infeasible" and a_data["practice_gap"]["material"] == "tungsten"
    assert a_data["margin"] == pytest.approx(a.margin_conservative)
    assert a_data["warm_forming_temperature"] == pytest.approx(a.warm_forming_temperature)
    assert "flow_stress [Pa] = " in a.text() and "practice_gap:" in a.text()
    none_gap = json.loads(pl.forging_practice_gap("mp35n").json())
    assert none_gap["conventional_range"] is None
