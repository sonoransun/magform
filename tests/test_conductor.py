"""Tests for magform.conductor: REBCO Je model, CNS composite, structure ladder and pulsed operation."""

from __future__ import annotations

import json
import math

import pytest

from magform import conductor as cd
from magform import magnetics as mg
from magform import materials
from magform._numerics import linspace, trapz
from magform.constants import MU0
from magform.errors import InputError, UnknownMaterialError
from magform.status import Status

A_MM2 = 1e6  # A/m^2 per A/mm^2


@pytest.fixture
def tape():
    return cd.REBCOTape()


# ------------------------------------------------------------ REBCO Je model
def test_je_validation_ranges_at_4k(tape):
    perp = lambda b: tape.je(b, 4.2, "perpendicular") / A_MM2  # noqa: E731
    assert 1100 <= perp(12.0) <= 2500
    assert 750 <= perp(18.0) <= 1850
    assert 500 <= perp(30.0) <= 1300
    je63 = perp(63.0)
    assert math.isfinite(je63) and je63 >= 200
    assert cd.je_data_status(63.0) is Status.EXTRAPOLATION
    ratio = tape.je(18.0, 4.2, "parallel") / tape.je(18.0, 4.2, "perpendicular")
    assert 1.8 <= ratio <= 3.5


def test_je_77k_self_field(tape):
    for o in cd.ORIENTATIONS:
        assert 300 <= tape.je(0.0, 77.0, o) / A_MM2 <= 500


@pytest.mark.parametrize("field, je_par, je_perp", [
    (18.0, 2286.7, 1106.1), (30.0, 1947.8, 739.5), (63.0, 1309.8, 295.2), (100.0, 862.6, 75.8),
])
def test_je_reference_values(tape, field, je_par, je_perp):
    # regression values of the model with the spec's default parameters (A/mm^2)
    assert tape.je(field, 4.2, "parallel") / A_MM2 == pytest.approx(je_par, rel=1e-3)
    assert tape.je(field, 4.2, "perpendicular") / A_MM2 == pytest.approx(je_perp, rel=1e-3)


def test_je_closed_form(tape):
    b, t = 25.0, 20.0
    h = (1 - t / 92.0) / (1 - 4.2 / 92.0)
    expect = 3.0e9 * h ** 1.2 * (1 + b / 60.0) ** -0.75 * (1 - b / (250.0 * h ** 1.5))
    assert tape.je(b, t, "parallel") == pytest.approx(expect, rel=1e-12)
    assert tape.birr(4.2, "parallel") == pytest.approx(250.0)
    assert tape.birr(4.2, "perpendicular") == pytest.approx(120.0)
    assert tape.je(0.0, 4.2) == pytest.approx(3.0e9)


def test_je_monotone_and_cutoffs(tape):
    for o in cd.ORIENTATIONS:
        by_field = [tape.je(b, 4.2, o) for b in (0.0, 5.0, 20.0, 45.0, 80.0, 110.0)]
        assert all(x > y for x, y in zip(by_field, by_field[1:]))
        by_temp = [tape.je(10.0, t, o) for t in (4.2, 10.0, 20.0, 40.0, 60.0)]
        assert all(x > y for x, y in zip(by_temp, by_temp[1:]))
        assert tape.je(1.0, 92.0, o) == 0.0 and tape.je(1.0, 120.0, o) == 0.0
        assert tape.birr(92.0, o) == 0.0
        birrs = [tape.birr(t, o) for t in (4.2, 20.0, 50.0, 77.0)]
        assert all(x > y for x, y in zip(birrs, birrs[1:]))
    # no current above the irreversibility field
    assert tape.je(120.0, 4.2, "perpendicular") == 0.0
    assert tape.je(130.0, 4.2, "perpendicular") == 0.0
    assert tape.je(130.0, 4.2, "parallel") > 0.0


def test_ic_and_layer_jc(tape):
    je = tape.je(18.0, 4.2, "perpendicular")
    assert tape.ic(18.0, 4.2, "perpendicular") == pytest.approx(je * 4e-3 * 0.1e-3)
    assert 300 <= tape.ic(18.0, 4.2, "perpendicular") <= 740          # 4 mm tapes, 4.2 K, 18 T, B || c
    assert tape.layer_jc(18.0, 4.2, "perpendicular") == pytest.approx(je * 0.1e-3 / 1.6e-6)


def test_current_sharing_temperature(tape):
    b = 30.0
    j_ops = [3e8, 6e8, 1.2e9]
    tcs = [tape.current_sharing_temperature(b, j) for j in j_ops]
    for j, t in zip(j_ops, tcs):
        assert 4.2 < t < 92.0
        assert tape.je(b, t, "parallel") == pytest.approx(j, rel=1e-8)
    assert all(x > y for x, y in zip(tcs, tcs[1:]))      # more current -> less margin
    # already over Je at 4.2 K: documented return value 4.2 K
    assert tape.current_sharing_temperature(b, 1e10) == 4.2
    assert tape.current_sharing_temperature(b, 1e10, lowest_temperature=1.8) == 1.8
    with pytest.raises(InputError):
        tape.current_sharing_temperature(b, 0.0)


def test_tape_input_validation(tape):
    with pytest.raises(InputError):
        cd.REBCOTape(substrate_thickness=90e-6)            # layers thicker than the tape
    with pytest.raises(InputError):
        cd.REBCOTape(tc=4.0)
    with pytest.raises(InputError):
        cd.REBCOTape(width=0.0)
    with pytest.raises(InputError):
        cd.REBCOTape(compressive_strain_limit=0.001)
    with pytest.raises(InputError):
        cd.REBCOTape(copper_layer_thickness=50e-6)
    with pytest.raises(InputError):
        tape.je(10.0, 4.2, "diagonal")
    with pytest.raises(InputError):
        tape.je(-1.0, 4.2)
    with pytest.raises(InputError):
        tape.je(1.0, 0.0)
    assert tape.je(10.0, 4.2, " Perpendicular ") == tape.je(10.0, 4.2, "perpendicular")


def test_je_data_status_ladder():
    assert cd.je_data_status(0.0) is Status.STATE_OF_THE_ART
    assert cd.je_data_status(31.0) is Status.STATE_OF_THE_ART
    assert cd.je_data_status(31.01) is Status.EXTRAPOLATION
    assert cd.je_data_status(45.5) is Status.EXTRAPOLATION
    assert cd.je_data_status(100.27) is Status.EXTRAPOLATION
    with pytest.raises(InputError):
        cd.je_data_status(-1.0)


# ----------------------------------------------------------------- composite
def test_composite_validation_and_material_handling():
    with pytest.raises(InputError):
        cd.CNSComposite(f_cns=0.6)                          # sums to 1.1
    with pytest.raises(InputError):
        cd.CNSComposite(f_cns=1.2, f_hts=-0.2, f_cu=0.0, f_insulation=0.0)
    with pytest.raises(InputError):
        cd.CNSComposite(insulation_modulus=0.0)
    with pytest.raises(UnknownMaterialError):
        cd.CNSComposite(cns_material="unobtainium")
    assert cd.CNSComposite(cns_material="cns").cns_material == "cns_engineering_target"   # alias -> key
    custom = materials.get("carbon_fibre_m60j").with_changes(youngs_modulus=700e9)
    comp = cd.CNSComposite(cns_material=custom)
    assert comp.cns.youngs_modulus == 700e9                  # a custom Material is used as given
    assert json.loads(comp.json())["cns_material"] == "carbon_fibre_m60j"
    assert cd.CNSComposite.default() == cd.CNSComposite()
    # fractions within the 1e-6 tolerance are accepted
    cd.CNSComposite(f_cns=0.5 + 5e-7)


def test_composite_iso_strain_mechanics():
    comp = cd.CNSComposite.default()
    e_c = comp.modulus()
    assert e_c == pytest.approx(0.5 * 300e9 + 0.3 * 160e9 + 0.1 * 117e9 + 0.1 * 5e9, rel=1e-12)
    assert e_c == pytest.approx(210.2e9, rel=1e-12)
    assert comp.matrix_modulus() == pytest.approx(120.4e9, rel=1e-12)
    assert 0.5 * comp.cns.youngs_modulus + 0.5 * comp.matrix_modulus() == pytest.approx(e_c, rel=1e-12)
    assert comp.strain_limit() == pytest.approx(0.004)
    assert comp.allowable_stress() == pytest.approx(210.2e9 * 0.004, rel=1e-12)
    assert comp.allowable_stress(2.0) == pytest.approx(0.5 * comp.allowable_stress(), rel=1e-12)
    sigma = 0.5e9
    eps = comp.strain_at_stress(sigma)
    assert eps == pytest.approx(sigma / e_c)
    assert comp.reinforcement_stress(sigma) == pytest.approx(300e9 * eps, rel=1e-12)
    # iso-strain force balance: the constituent stresses add up to the applied stress
    parts = (0.5 * 300e9 + 0.3 * 160e9 + 0.1 * 117e9 + 0.1 * 5e9) * eps
    assert parts == pytest.approx(sigma, rel=1e-12)
    assert comp.tape.density() == pytest.approx(9012.2, rel=1e-3)
    assert comp.density() == pytest.approx(0.5 * 2000 + 0.3 * comp.tape.density() + 0.1 * 8960 + 0.1 * 1420, rel=1e-12)


def test_required_modulus_and_fraction():
    # the headline: 4 GPa at 0.4 % strain needs graphene's intrinsic 1 TPa
    assert cd.required_composite_modulus(4e9) == pytest.approx(1e12)
    assert cd.required_composite_modulus(4e9) == pytest.approx(materials.get("graphene_intrinsic").youngs_modulus)
    assert cd.required_composite_modulus(2e9, 0.004, 1.5) == pytest.approx(7.5e11)
    f = cd.required_cns_fraction(2e9, 1000e9, 120.4e9, 0.004, 1.5)
    assert f == pytest.approx((750e9 - 120.4e9) / (1000e9 - 120.4e9), rel=1e-12)
    assert f * 1000e9 + (1 - f) * 120.4e9 == pytest.approx(7.5e11, rel=1e-12)
    assert cd.required_cns_fraction(4e9, 300e9, 120.4e9) > 1.0          # not clipped
    assert cd.required_cns_fraction(1e8, 300e9, 120.4e9) < 0.0          # matrix alone suffices, not clipped
    assert cd.required_cns_fraction(1e9, 100e9, 120.4e9) == math.inf    # softer reinforcement cannot help
    assert cd.required_cns_fraction(1e8, 100e9, 120.4e9) == 0.0
    with pytest.raises(InputError):
        cd.required_composite_modulus(-1.0)
    with pytest.raises(InputError):
        cd.required_composite_modulus(1e9, eps_allow=0.0)


# ------------------------------------------------------- conductor assessment
def test_assess_conductor_margins_and_ladder():
    comp = cd.CNSComposite.default()
    tape = comp.tape
    a = cd.assess_conductor(10.0, 4.2, 1e8, safety_factor=1.5)
    assert a.je_parallel == pytest.approx(tape.je(10.0, 4.2, "parallel"))
    assert a.je_perpendicular == pytest.approx(tape.je(4.0, 4.2, "perpendicular"))     # 0.4 x B at the ends
    assert a.j_allow_parallel == pytest.approx(0.3 * a.je_parallel)
    assert a.j_allow_perpendicular == pytest.approx(0.3 * a.je_perpendicular)
    assert a.margin_parallel == pytest.approx(a.j_allow_parallel / 1e8)
    assert a.margin == pytest.approx(min(a.margin_parallel, a.margin_perpendicular))
    assert a.status is Status.STATE_OF_THE_ART and a.je_data_status is Status.STATE_OF_THE_ART
    j_lim = min(a.j_allow_parallel, a.j_allow_perpendicular)
    assert cd.assess_conductor(10.0, 4.2, j_lim / 1.2, safety_factor=1.5).status is Status.EXTRAPOLATION
    low = cd.assess_conductor(10.0, 4.2, j_lim * 2.0, safety_factor=1.5)
    assert low.status is Status.ASPIRATIONAL and low.margin == pytest.approx(0.5, rel=1e-9)
    assert any("does not exist" in w for w in low.warnings)


def test_assess_conductor_high_field_is_extrapolation_even_with_margin():
    a = cd.assess_conductor(63.4, 4.2, 1e7, safety_factor=1.5)
    assert a.margin > 1.5
    assert a.je_data_status is Status.EXTRAPOLATION and a.status is Status.EXTRAPOLATION
    assert any("separate axis" in n for n in a.notes)
    # at 100 T the coil-end perpendicular field limits a typical winding
    b = cd.assess_conductor(100.27, 4.2, 3e8)
    assert b.margin_perpendicular < b.margin_parallel
    with pytest.raises(InputError):
        cd.assess_conductor(10.0, 4.2, 1e8, safety_factor=0.9)
    with pytest.raises(InputError):
        cd.assess_conductor(10.0, 4.2, 1e8, end_field_fraction=1.5)
    with pytest.raises(InputError):
        cd.assess_conductor(10.0, 4.2, 0.0)


# ------------------------------------------------------- structure assessment
@pytest.mark.parametrize("stress, status, met_by", [
    (0.5e9, Status.STATE_OF_THE_ART, "carbon_fibre_m60j"),
    (0.8e9, Status.STATE_OF_THE_ART, "carbon_fibre_m60j"),
    (0.97e9, Status.EXTRAPOLATION, "cnt_fibre_record"),
    (1.2e9, Status.ASPIRATIONAL, "graphene_intrinsic"),
    (2.0e9, Status.INFEASIBLE, None),
    (4.0e9, Status.INFEASIBLE, None),
])
def test_assess_structure_ladder(stress, status, met_by):
    s = cd.assess_structure(stress)
    assert s.status is status
    assert set(s.benchmark) == set(cd.STRUCTURE_BENCHMARKS)
    if met_by is not None:
        assert s.benchmark[met_by]["meets"]
    # the ladder is consistent with the per-material checks
    commercial = s.benchmark["carbon_fibre_m60j"]["meets"] or s.benchmark["carbon_fibre_t1100g"]["meets"]
    assert commercial == (status is Status.STATE_OF_THE_ART)
    if status is Status.INFEASIBLE:
        assert not any(v["meets"] for v in s.benchmark.values())


def test_assess_structure_quantities():
    comp = cd.CNSComposite.default()
    s = cd.assess_structure(4e9, comp, safety_factor=1.5)
    assert s.composite_modulus == pytest.approx(210.2e9)
    assert s.strain == pytest.approx(4e9 / 210.2e9)
    assert s.strain_limit == pytest.approx(0.004)
    assert s.required_composite_modulus == pytest.approx(1.5e12)
    assert s.required_reinforcement_modulus == pytest.approx((1.5e12 - 60.2e9) / 0.5, rel=1e-12)
    assert s.required_cns_fraction == pytest.approx((1.5e12 - 120.4e9) / (300e9 - 120.4e9), rel=1e-12)
    assert s.reinforcement_stress == pytest.approx(300e9 * 4e9 / 210.2e9, rel=1e-12)
    assert s.reinforcement_strength == pytest.approx(10e9)
    assert s.margin == pytest.approx(210.2e9 * 0.004 / (1.5 * 4e9), rel=1e-12)
    g = s.benchmark["graphene_intrinsic"]
    assert g["required_cns_fraction"] > 1.0 and not g["meets"]
    assert any("even with intrinsic graphene" in n for n in s.notes)
    # 2 GPa: intrinsic graphene would work only at a larger fraction
    s2 = cd.assess_structure(2e9)
    assert s2.benchmark["graphene_intrinsic"]["required_cns_fraction"] == pytest.approx(0.7158, rel=1e-3)
    assert any("intrinsic graphene would need f_cns" in n for n in s2.notes)
    # a stiff commercial reinforcement at modest stress: the specified composite itself passes
    ok = cd.assess_structure(0.5e9, cd.CNSComposite(cns_material="carbon_fibre_m60j"))
    assert ok.status is Status.STATE_OF_THE_ART and ok.margin >= 1.0 and not ok.warnings
    with pytest.raises(InputError):
        cd.assess_structure(-1.0)
    with pytest.raises(InputError):
        cd.assess_structure(1e9, safety_factor=0.5)


def test_structure_infeasible_for_any_thick_4gpa_coil():
    # sigma_max >= P/2 for any winding thickness, so a 4 GPa coil always needs >= 0.5 TPa (SF 1)
    for alpha in (3.0, 11.0, 1e4):
        sigma = 4e9 * mg.max_hoop_stress_ratio(alpha)
        assert cd.required_composite_modulus(sigma) >= 0.5e12 * (1 - 1e-9)
        assert cd.assess_structure(sigma).status is Status.INFEASIBLE


# --------------------------------------------------- graded winding thickness
def test_je_limited_graded_thickness(tape):
    def reference(b_top, o, n=20000):
        bs = linspace(0.0, b_top, n + 1)
        return trapz([1.0 / (MU0 * 0.3 * tape.je(b, 4.2, o)) for b in bs], bs)

    for b_top in (63.4, 100.3):
        par = cd.je_limited_graded_thickness(b_top)
        perp = cd.je_limited_graded_thickness(b_top, orientation="perpendicular")
        assert math.isfinite(par) and math.isfinite(perp) and perp > par
        assert par == pytest.approx(reference(b_top, "parallel"), rel=1e-5)
        assert perp == pytest.approx(reference(b_top, "perpendicular"), rel=1e-4)
        # bounded by the uniform-J windings at the extreme current densities
        j0, j_top = 0.3 * tape.je(0.0, 4.2), 0.3 * tape.je(b_top, 4.2)
        assert b_top / (MU0 * j0) < par < b_top / (MU0 * j_top)
    # reported key numbers (m)
    assert cd.je_limited_graded_thickness(63.4) == pytest.approx(0.08966, rel=1e-3)
    assert cd.je_limited_graded_thickness(63.4, orientation="perpendicular") == pytest.approx(0.2637, rel=1e-3)
    assert cd.je_limited_graded_thickness(100.3) == pytest.approx(0.1828, rel=1e-3)
    assert cd.je_limited_graded_thickness(100.3, orientation="perpendicular") == pytest.approx(0.9237, rel=1e-3)
    # beyond the perpendicular irreversibility field no finite winding exists
    assert cd.je_limited_graded_thickness(125.0, orientation="perpendicular") == math.inf
    # warmer and higher field -> thicker
    assert cd.je_limited_graded_thickness(63.4, temperature=20.0) > cd.je_limited_graded_thickness(63.4)
    assert cd.je_limited_graded_thickness(0.0) == 0.0


# ---------------------------------------------------------- pulse AC losses
def test_loss_helper_limits():
    bp = 0.1
    low = cd._bean_slab_loss(bp, bp)
    assert low == pytest.approx(2 * bp ** 2 / (3 * MU0), rel=1e-12)          # both branches meet at Bp
    assert cd._bean_slab_loss(bp * (1 + 1e-9), bp) == pytest.approx(low, rel=1e-6)
    assert cd._bean_slab_loss(100.0, bp) == pytest.approx(2 * 100.0 * bp / MU0 * (1 - 2 * bp / 300.0))
    assert cd._bean_slab_loss(1.0, 0.0) == 0.0
    assert cd._strip_loss_function(1e4) == pytest.approx(1 - 2 * math.log(2) / 1e4, rel=1e-9)
    assert cd._strip_loss_function(1e-4) == pytest.approx(1e-12 / 6, rel=1e-6)
    assert cd._strip_loss_function(1e-3 * (1 - 1e-9)) == pytest.approx(cd._strip_loss_function(1e-3 * (1 + 1e-9)),
                                                                        rel=1e-5)
    g = [cd._strip_loss_function(x) for x in (0.1, 1.0, 3.0, 10.0, 100.0)]
    assert all(0 < a < b < 1 for a, b in zip(g, g[1:]))


def test_tape_pulse_losses_100t_25us(tape):
    loss = cd.tape_pulse_losses(100.0, 25e-6)
    assert loss.ramp_rate == pytest.approx(4e6)
    sc = 1.6e-6 / 0.1e-3
    # perpendicular: high-field (Norris) limit Jc (w/2) dB_perp, dB_perp = 40 T, Jc at 20 T
    jc_perp = tape.layer_jc(20.0, 4.2, "perpendicular")
    assert loss.q_hysteresis_perpendicular == pytest.approx(jc_perp * 2e-3 * 40.0 * sc, rel=1e-2)
    assert loss.q_hysteresis_perpendicular < jc_perp * 2e-3 * 40.0 * sc
    # parallel: Bean slab, Bm = 50 T >> Bp
    jc_par = tape.layer_jc(50.0, 4.2, "parallel")
    bp = MU0 * jc_par * 0.8e-6
    assert loss.q_hysteresis_parallel == pytest.approx(2 * 50.0 * bp / MU0 * (1 - 2 * bp / 150.0) * sc, rel=1e-12)
    # copper eddy loss with magnetoresistive rho = rho(4.2 K) (1 + 50 T/10 T)
    rho = materials.get("copper_ofhc").resistivity_at(4.2) * 6.0
    assert loss.copper_resistivity == pytest.approx(rho)
    assert loss.q_eddy_copper == pytest.approx((4e6) ** 2 * (20e-6) ** 2 * 50e-6 / (12 * rho) * 0.4, rel=1e-12)
    assert loss.q_total == pytest.approx(0.4 * loss.q_hysteresis_perpendicular + 0.6 * loss.q_hysteresis_parallel
                                         + loss.q_eddy_copper, rel=1e-12)
    assert 1e7 < loss.q_total < 1e8


def test_tape_pulse_losses_scaling():
    fast = cd.tape_pulse_losses(10.0, 1e-4)
    slow = cd.tape_pulse_losses(10.0, 2e-4)
    # hysteresis is rate independent, eddy loss per pulse scales as 1/rise time
    assert fast.q_hysteresis_parallel == pytest.approx(slow.q_hysteresis_parallel)
    assert fast.q_hysteresis_perpendicular == pytest.approx(slow.q_hysteresis_perpendicular)
    assert fast.q_eddy_copper == pytest.approx(2.0 * slow.q_eddy_copper, rel=1e-12)
    # small perpendicular swing: below the high-field formula (strip not fully penetrated)
    small = cd.tape_pulse_losses(0.1, 1.0, perpendicular_fraction=1.0)
    tp = cd.REBCOTape()
    high_field = tp.layer_jc(0.05, 4.2, "perpendicular") * 2e-3 * 0.1 * 1.6e-6 / 0.1e-3
    assert 0 < small.q_hysteresis_perpendicular < 0.5 * high_field
    # no perpendicular share -> no perpendicular loss counted
    par_only = cd.tape_pulse_losses(10.0, 1e-4, perpendicular_fraction=0.0)
    assert par_only.q_hysteresis_perpendicular == 0.0
    assert par_only.q_total == pytest.approx(par_only.q_hysteresis_parallel + par_only.q_eddy_copper)
    with pytest.raises(InputError):
        cd.tape_pulse_losses(0.0, 1e-3)
    with pytest.raises(InputError):
        cd.tape_pulse_losses(10.0, 0.0)
    with pytest.raises(InputError):
        cd.tape_pulse_losses(10.0, 1e-3, perpendicular_fraction=1.5)


def test_enthalpy_margin():
    # hand trapezoid of the tables, 4.2 -> 77 K: Cu 5386.5 J/kg x 8960 x 0.4 + Hastelloy 3119.5 J/kg x 8890 x 0.5
    cu = 0.3006 + 0.70 + 1.33 + 8.9 + 25.25 + 57.25 + 106.25 + 430 + 785 + 1175 + 2796.5
    hast = (0.6 + 1.0 * 0.2 / 6 + 1.6) / 2 * 5.8 + 28 + 70 + 150 + 300 + 2565
    expect = cu * 8960 * 0.4 + hast * 8890 * 0.5
    assert cd.enthalpy_margin(4.2, 77.0) == pytest.approx(expect, rel=1e-3)
    assert cd.enthalpy_margin(4.2, 77.0) == pytest.approx(3.318e7, rel=1e-3)
    assert cd.enthalpy_margin(4.2, 20.0) == pytest.approx(2.84e5, rel=1e-2)
    # additive, monotone, zero for an empty interval
    assert cd.enthalpy_margin(4.2, 30.0) == pytest.approx(cd.enthalpy_margin(4.2, 12.5) + cd.enthalpy_margin(12.5, 30.0),
                                                          rel=1e-9)
    assert cd.enthalpy_margin(10.0, 10.0) == 0.0
    vals = [cd.enthalpy_margin(4.2, t) for t in (5.0, 10.0, 40.0, 92.0, 300.0)]
    assert all(x < y for x, y in zip(vals, vals[1:]))
    assert cd.enthalpy_margin(1.8, 4.2) > 0.0
    with pytest.raises(InputError):
        cd.enthalpy_margin(20.0, 10.0)
    with pytest.raises(InputError):
        cd.enthalpy_margin(4.2, 400.0)
    with pytest.raises(InputError):
        cd.enthalpy_margin(0.0, 10.0)


# ------------------------------------------------------------ pulsed operation
def test_pulsed_100t_in_25us_quenches_every_pulse():
    q = cd.assess_pulsed_operation(100.0, 25e-6, 1e8)
    assert q.status is Status.INFEASIBLE
    assert q.loss_to_margin_ratio > 1.0
    assert q.loss_to_margin_ratio == pytest.approx(37.11, rel=1e-2)        # reported key number
    assert q.margin == pytest.approx(1.0 / q.loss_to_margin_ratio, rel=1e-12)
    assert q.ramp_rate == pytest.approx(4e6) and q.ramp_rate_status is Status.ASPIRATIONAL
    assert q.enthalpy_margin == pytest.approx(cd.enthalpy_margin(4.2, q.current_sharing_temperature), rel=1e-12)
    tape = cd.REBCOTape()
    assert tape.je(100.0, q.current_sharing_temperature) == pytest.approx(1e8 / 0.3, rel=1e-8)
    assert any("quenches every pulse" in w for w in q.warnings)
    # infeasible for any plausible operating current density
    for j in (2e7, 5e7, 2e8):
        assert cd.assess_pulsed_operation(100.0, 25e-6, j).status is Status.INFEASIBLE


def test_pulsed_slow_and_intermediate_ramps():
    slow = cd.assess_pulsed_operation(1.0, 10.0, 1e8)
    assert slow.status is not Status.INFEASIBLE
    assert slow.status is Status.STATE_OF_THE_ART                          # 0.1 T/s, ramp benchmark
    assert slow.loss_to_margin_ratio < 0.3
    assert any("information only" in n for n in slow.notes)
    mid = cd.assess_pulsed_operation(20.0, 1.0, 1e8)                       # adiabatic, ratio ~0.74
    assert 0.3 < mid.loss_to_margin_ratio < 1.0
    assert mid.status is Status.EXTRAPOLATION
    small = cd.assess_pulsed_operation(2.0, 1.0, 1e8)                      # 2 T/s, ratio ~0.03
    assert small.loss_to_margin_ratio < 0.3 and small.status is Status.STATE_OF_THE_ART
    # no DC margin at the peak field: at least aspirational even for a slow ramp
    over = cd.assess_pulsed_operation(50.0, 10.0, 1e9)
    assert over.current_sharing_temperature == 4.2 and over.enthalpy_margin == 0.0
    assert over.loss_to_margin_ratio == math.inf and over.margin == 0.0
    assert over.status is Status.ASPIRATIONAL
    with pytest.raises(InputError):
        cd.assess_pulsed_operation(10.0, 1.0, 1e8, temperature=95.0)
    with pytest.raises(InputError):
        cd.assess_pulsed_operation(10.0, 1.0, 1e8, composite=cd.CNSComposite(f_cns=0.8, f_hts=0.0))
    with pytest.raises(InputError):
        cd.assess_pulsed_operation(10.0, 0.0, 1e8)


def test_terminal_voltage():
    assert cd.terminal_voltage(1e8, 1e5, 0.01) == pytest.approx(2e5)
    s = mg.Solenoid(0.05, 0.25, 0.5)
    n = 200
    b0 = 30.0
    i = s.current_for_field(b0, n)
    e = s.stored_energy(b0)
    assert cd.terminal_voltage(e, i, 10.0) == pytest.approx(s.inductance_from_energy(n) * i / 10.0, rel=1e-12)
    with pytest.raises(InputError):
        cd.terminal_voltage(1e8, 0.0, 1.0)
    with pytest.raises(InputError):
        cd.terminal_voltage(1e8, 1e5, 0.0)


def test_cns_eddy_check():
    r = cd.cns_eddy_check(None, 100.0, 25e-6)
    rho = 1e-6
    assert r.material == "cns_engineering_target"
    assert r.frequency == pytest.approx(1e4)
    assert r.skin_depth == pytest.approx(math.sqrt(rho / (math.pi * 1e4 * MU0)), rel=1e-12)
    assert not r.thick
    assert r.loss_per_pulse_density == pytest.approx((4e6) ** 2 * 1e-6 * 50e-6 / (12 * rho), rel=1e-12)
    # thick laminae: d = skin depth, and the loss 2 dB^2/(3 pi mu0) no longer depends on rho or rise time
    for comp, tr in ((cd.CNSComposite(cns_material="cnt_fibre_record"), 25e-6), (None, 1e-6)):
        thick = cd.cns_eddy_check(comp, 100.0, tr, lamina_thickness=0.05)
        assert thick.thick
        assert thick.loss_per_pulse_density == pytest.approx(2 * 100.0 ** 2 / (3 * math.pi * MU0), rel=1e-12)
    with pytest.raises(InputError):
        cd.cns_eddy_check(None, 100.0, 25e-6, lamina_thickness=0.0)


# --------------------------------------------------------------- serialisation
def test_results_json_round_trip_and_text():
    comp = cd.CNSComposite.default()
    results = [
        cd.REBCOTape(),
        comp,
        cd.assess_conductor(63.4, 4.2, 2e8),
        cd.assess_structure(2e9),
        cd.tape_pulse_losses(100.0, 25e-6),
        cd.assess_pulsed_operation(100.0, 25e-6, 1e8),
        cd.assess_pulsed_operation(50.0, 10.0, 1e9),        # contains inf (ratio) -> JSON null
        cd.cns_eddy_check(comp, 100.0, 25e-6),
    ]
    for res in results:
        data = json.loads(res.json())
        assert isinstance(data, dict) and data
        txt = res.text()
        assert type(res).__name__ in txt
    d = json.loads(results[2].json())
    assert d["status"] == "extrapolation" and d["je_data_status"] == "extrapolation"
    d = json.loads(results[3].json())
    assert d["status"] == "infeasible" and d["benchmark"]["graphene_intrinsic"]["tier"] == "aspirational"
    assert d["benchmark"]["carbon_fibre_m60j"]["meets"] is False
    d = json.loads(results[5].json())
    assert d["status"] == "infeasible" and d["loss"]["q_total"] > 0
    assert json.loads(results[6].json())["loss_to_margin_ratio"] is None
    assert json.loads(comp.json())["tape"]["je_sf_4k"] == 3.0e9
    assert "q_total [J/m^3] = " in results[4].text()
    assert "loss:" in results[5].text()


def test_module_metadata():
    assert set(cd.__all__) <= set(dir(cd))
    assert cd.REFERENCES and all(isinstance(v, str) and v for v in cd.REFERENCES.values())
    for name in cd.__all__:
        obj = getattr(cd, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 40, name
