"""Tests for magform.synthesis: phase boundaries, route table, pulse compatibility, shock route."""

from __future__ import annotations

import json
import math

import pytest

from magform import materials, synthesis
from magform.constants import T_ROOM
from magform.errors import InputError, UnknownMaterialError
from magform.status import Status, best
from magform.synthesis import (
    SYNTHESIS_ROUTES,
    AnvilClass,
    FlyerSpec,
    SynthesisAssessment,
    SynthesisRoute,
    VolumeScaling,
)

CU_FLYER = FlyerSpec("copper_ofhc", 0.002, 0.01)  # spec check: 2 mm Cu over a 10 mm gap
STATIC = [r for r in SYNTHESIS_ROUTES.values() if r.kind != "shock"]
SHOCK = [r for r in SYNTHESIS_ROUTES.values() if r.kind == "shock"]


def _ideal_conditions(r: SynthesisRoute) -> tuple[float, float, float]:
    """A (P, T, dwell) inside the route window and inside the product's stability field."""
    p = r.p_max if r.p_max is not None else 1.2 * r.p_min
    t = r.t_min if r.t_min is not None else 300.0
    return p, t, 1.5 * r.min_dwell


# ------------------------------------------------------- phase boundaries
def test_graphite_diamond_boundary_values():
    gd = synthesis.graphite_diamond_boundary
    assert gd(1500.0) == pytest.approx(5.007125e9, rel=1e-9)
    assert gd(2000.0) == pytest.approx(6.257125e9, rel=1e-9)
    assert gd(1500.0, "berman_simon") == pytest.approx(4.76e9, rel=1e-9)
    assert gd(2000.0, "berman_simon") == pytest.approx(6.11e9, rel=1e-9)
    assert gd(273.15) == pytest.approx(1.94e9, rel=1e-12)            # P[kbar] = 19.4 + T[C]/40
    assert gd(1700.0) - gd(1600.0) == pytest.approx(0.25e9, rel=1e-9)
    # the two models agree within 10 % over the HPHT range
    for t in (1400.0, 1600.0, 1800.0, 2000.0):
        assert gd(t, "berman_simon") == pytest.approx(gd(t), rel=0.1)


def test_hbn_cbn_boundary_values_and_anchor_points():
    hc = synthesis.hbn_cbn_boundary
    assert hc(1800.0) == pytest.approx(0.45388e9, rel=1e-9)
    assert hc(1800.0, "corrigan_bundy_1975") == pytest.approx(5.5652e9, rel=1e-9)
    assert hc(1660.0) == 0.0 and hc(300.0) == 0.0                    # cBN stable at ambient below 1660 K
    t3, p3 = synthesis.SOLOZHENKO_TRIPLE_POINT
    assert hc(t3) == pytest.approx(p3, rel=1e-3)
    t3, p3 = synthesis.CORRIGAN_BUNDY_TRIPLE_POINT
    assert hc(t3, "corrigan_bundy_1975") == pytest.approx(p3, rel=1e-3)
    assert hc(1e-6, "corrigan_bundy_1975") == pytest.approx(1.4e9, rel=1e-6)
    # the literature disagreement is large where industry works
    assert hc(1800.0, "corrigan_bundy_1975") - hc(1800.0) > 5e9


def test_stability_predicates():
    assert synthesis.diamond_stable(5.5e9, 1600.0)
    assert synthesis.diamond_stable(5.5e9, 1600.0, "berman_simon")
    assert not synthesis.diamond_stable(4e9, 1500.0)
    assert not synthesis.diamond_stable(4e9, 1500.0, "berman_simon")
    # thermodynamics allows diamond at 4 GPa and room temperature; kinetics forbid conversion
    assert synthesis.diamond_stable(4e9, 293.15)
    assert synthesis.cbn_stable(0.0, 300.0)
    assert not synthesis.cbn_stable(0.0, 300.0, "corrigan_bundy_1975")
    assert synthesis.cbn_stable(6e9, 1800.0) and synthesis.cbn_stable(6e9, 1800.0, "corrigan_bundy_1975")
    assert not synthesis.cbn_stable(5e9, 1800.0, "corrigan_bundy_1975")  # historical line: marginal


def test_catalytic_cbn_window_inside_modern_cbn_field():
    r = SYNTHESIS_ROUTES["cbn_catalytic"]
    for t in (r.t_min, 0.5 * (r.t_min + r.t_max), r.t_max):
        assert synthesis.cbn_stable(r.p_min, t)


# ------------------------------------------------------------ route table
def test_route_table_matches_spec():
    assert list(SYNTHESIS_ROUTES) == [
        "diamond_hpht_catalytic", "hpht_single_crystal", "pcd_sintering", "diamond_direct_npd",
        "cbn_catalytic", "cbn_direct", "lonsdaleite_shock", "diamond_shock_dupont", "c3n4_dac"]
    d = SYNTHESIS_ROUTES["diamond_hpht_catalytic"]
    assert (d.p_min, d.p_max, d.t_min, d.t_max, d.min_dwell) == (5e9, 6.5e9, 1573.0, 1873.0, 600.0)
    assert d.typical_dwell == (3.6e3, 8.6e4)
    c = SYNTHESIS_ROUTES["c3n4_dac"]
    assert (c.p_min, c.p_max, c.kind) == (72e9, 134e9, "laser_heated_dac")
    assert SYNTHESIS_ROUTES["lonsdaleite_shock"].p_min == 15e9
    assert SYNTHESIS_ROUTES["diamond_shock_dupont"].p_min == 20e9
    for key, r in SYNTHESIS_ROUTES.items():
        assert r.key == key and r.kind in synthesis.ROUTE_KINDS
        assert 1e9 <= r.p_min <= 1e12
        assert r.p_max is None or r.p_max >= r.p_min
        assert r.reference and r.product_form and r.current_scale and r.heating
        if r.kind == "shock":
            assert r.t_min is None and r.t_max is None and r.p_max is None and r.min_dwell == 1e-7
        else:
            assert r.t_min is not None and r.t_max is not None and r.min_dwell >= 1.0


def test_magnetic_window_is_below_every_static_route():
    lowest = min(r.p_min for r in STATIC)
    assert lowest == 4.5e9  # catalytic cBN
    for p in (1.6e9, 2.0e9, 4.0e9):
        for r in STATIC:
            a = synthesis.assess_route(r, p, *(_ideal_conditions(r)[1:]))
            assert a.status is Status.INFEASIBLE and a.pressure_margin < 1.0
            assert any("below the route minimum" in w for w in a.warnings)


# -------------------------------------------------- dwell / heating rules
@pytest.mark.parametrize("pulse", [1e-6, 1e-4, 1e-3])
def test_static_routes_cannot_be_run_by_a_pulse(pulse):
    for r in STATIC:
        p, t, _ = _ideal_conditions(r)
        a = synthesis.assess_route(r, p, t, pulse)
        assert a.status is Status.INFEASIBLE and a.dwell_ok is False
        assert any("cannot supply the dwell" in w and "DC field exerts no pressure" in w for w in a.warnings)


def test_conventional_press_conditions_are_state_of_the_art():
    for r in STATIC:
        p, t, dwell = _ideal_conditions(r)
        a = synthesis.assess_route(r, p, t, dwell)
        assert a.status is Status.STATE_OF_THE_ART, (r.key, a.warnings)
        assert a.dwell_ok and a.temperature_ok and a.pressure_margin >= 1.0
        assert any("conventional press already does this" in n for n in a.notes)


def test_magnet_supplies_no_heating():
    r = SYNTHESIS_ROUTES["diamond_hpht_catalytic"]
    a = synthesis.assess_route(r, 6e9, T_ROOM, 3600.0)
    assert a.status is Status.INFEASIBLE and a.temperature_ok is False
    assert any("supplies no heating" in w for w in a.warnings)


def test_product_outside_stability_field_is_infeasible():
    # corner of the catalytic box (5 GPa, 1873 K) lies on the graphite side of both lines
    a = synthesis.assess_route("diamond_hpht_catalytic", 5.0e9, 1873.0, 3600.0)
    assert a.phase_stable is False and a.status is Status.INFEASIBLE
    assert any("not the thermodynamically stable phase" in w for w in a.warnings)
    # on the line the models disagree (1650 K: Kennedy 5.38 GPa, Berman-Simon 5.17 GPa) -> extrapolation
    near = synthesis.assess_route("diamond_hpht_catalytic", 5.25e9, 1650.0, 3600.0)
    assert near.phase_stable is False and near.status is Status.EXTRAPOLATION
    assert any("literature disagreement" in w for w in near.warnings)
    assert not any("conventional press already does this" in n for n in near.notes)


def test_anvil_intensification_and_moving_parts():
    a = synthesis.assess_route("diamond_hpht_catalytic", 4e9, 1673.0, 3600.0, intensification=1.5)
    assert a.pressure_sample == pytest.approx(6e9)
    assert a.status is Status.STATE_OF_THE_ART and a.anvil.technology == "belt_cubic_wc"
    assert any("moving mechanical parts" in n for n in a.notes)
    npd = synthesis.assess_route("diamond_direct_npd", 4e9, 2600.0, 1200.0, intensification=5.0)   # the routine cycle
    assert npd.status is Status.STATE_OF_THE_ART and npd.anvil.technology == "kawai_wc"
    cn = synthesis.assess_route("c3n4_dac", 4e9, 2500.0, 600.0, intensification=25.0)
    assert cn.anvil.technology == "kawai_sintered_diamond" and "0.1 mm^3" in cn.anvil.sample_volume_class


def test_anvil_feasibility_ladder():
    cases = [(4e9, "belt_cubic_wc"), (7e9, "belt_cubic_wc"), (20e9, "kawai_wc"),
             (50e9, "kawai_sintered_diamond"), (200e9, "dac")]
    for p, tech in cases:
        a = synthesis.anvil_feasibility(p)
        assert a.technology == tech and a.status is Status.STATE_OF_THE_ART and a.max_pressure >= p
        assert "moving mechanical parts" in a.note and a.reference
    top = synthesis.anvil_feasibility(500e9)
    assert top.status is Status.ASPIRATIONAL and top.technology == "none" and top.max_pressure == 400e9


# --------------------------------------------------------- the shock route
def test_shock_route_spec_check_cu_flyer_into_graphite():
    a = synthesis.assess_route("lonsdaleite_shock", 4e9, flyer=CU_FLYER)
    assert a.flyer_velocity == pytest.approx(2113.0, rel=1e-2)
    assert 12e9 <= a.pressure_sample <= 30e9
    assert a.pressure_sample == pytest.approx(22.34e9, rel=1e-3)  # value of the spec's own equations
    assert a.status is Status.EXTRAPOLATION and a.dwell_ok and a.temperature_ok is None
    assert a.shock.target == "graphite_crystal" and a.margin == pytest.approx(a.pressure_sample / 15e9)
    assert any("52.2 GPa" in n for n in a.notes)                 # the Cu-on-Cu figure, for contrast
    assert any("not industrial" in n for n in a.notes)
    cucu = synthesis.assess_route("lonsdaleite_shock", 4e9, flyer=FlyerSpec("copper_ofhc", 0.002, 0.01,
                                                                             sample="copper_ofhc"))
    assert cucu.pressure_sample == pytest.approx(52.2e9, rel=3e-2)
    stiff = synthesis.assess_route("lonsdaleite_shock", 4e9, flyer=FlyerSpec("copper_ofhc", 0.002, 0.01,
                                                                              sample="tungsten"))
    assert stiff.pressure_sample > cucu.pressure_sample
    assert any("higher shock impedance" in n for n in stiff.notes)
    assert any("lower shock impedance" in n for n in a.notes)


def test_porous_graphite_feedstock_is_marginal():
    fl = FlyerSpec("copper_ofhc", 0.002, 0.01, sample="graphite_polycrystalline")
    lons = synthesis.assess_route("lonsdaleite_shock", 4e9, flyer=fl)
    dupont = synthesis.assess_route("diamond_shock_dupont", 4e9, flyer=fl)
    assert lons.pressure_sample == pytest.approx(15.81e9, rel=1e-3)
    assert lons.status is Status.EXTRAPOLATION and lons.pressure_margin < 1.1
    assert dupont.status is Status.INFEASIBLE
    assert any("below the route minimum" in w for w in dupont.warnings)


def test_weak_drive_shock_is_infeasible():
    a = synthesis.assess_route("lonsdaleite_shock", 0.5e9, flyer=CU_FLYER)
    assert a.pressure_sample < 15e9 and a.status is Status.INFEASIBLE


def test_shock_route_without_flyer_is_not_applicable():
    a = synthesis.assess_route("diamond_shock_dupont", 4e9)
    assert a.status is Status.NOT_APPLICABLE and a.shock is None and a.flyer_velocity is None
    assert any("FlyerSpec" in n for n in a.notes)


def test_short_drive_pulse_is_flagged():
    a = synthesis.assess_route("lonsdaleite_shock", 4e9, pulse_duration=1e-6, flyer=CU_FLYER)
    assert any("ends before the flyer crosses the gap" in w for w in a.warnings)


def test_assess_all_headline_results():
    for p in (1.6e9, 4e9):
        plain = synthesis.assess_all(p)
        assert len(plain) == len(SYNTHESIS_ROUTES)
        assert all(a.status is Status.INFEASIBLE for a in plain if a.kind != "shock")
        assert all(a.status is Status.NOT_APPLICABLE for a in plain if a.kind == "shock")
    with_flyer = synthesis.assess_all(4e9, flyer=CU_FLYER)
    assert best(a.status for a in with_flyer) is Status.EXTRAPOLATION   # nothing is state of the art
    by_key = {a.route: a for a in with_flyer}
    assert by_key["lonsdaleite_shock"].status is Status.EXTRAPOLATION
    assert by_key["diamond_shock_dupont"].pressure_margin == pytest.approx(22.34 / 20.0, rel=1e-3)


def test_quantum_materials_note():
    note = synthesis.QUANTUM_MATERIALS_NOTE
    assert "CVD" in note and "H3S" in note and "LaH10" in note and "1.6-4 GPa" in note


# --------------------------------------------------- volume and curves
def test_volume_scaling_bookkeeping():
    vs = synthesis.volume_scaling(0.05, 0.1, 4e9)
    v_bore = math.pi * 0.05 ** 2 * 0.1
    rho_g, rho_d = materials.get("graphite_crystal").density, materials.get("diamond").density
    assert vs.sample_volume == pytest.approx(0.5 * v_bore, rel=1e-12)
    assert vs.graphite_mass == pytest.approx(rho_g * 0.5 * v_bore, rel=1e-12)
    assert vs.diamond_mass_if_converted == vs.graphite_mass          # mass is conserved
    assert vs.mechanical_work == pytest.approx(4e9 * 0.5 * v_bore * (1 - rho_g / rho_d), rel=1e-12)
    assert vs.bore_magnetic_energy == pytest.approx(4e9 * v_bore, rel=1e-12)
    assert vs.mechanical_work < vs.bore_magnetic_energy
    full = synthesis.volume_scaling(0.05, 0.1, 4e9, fill_fraction=1.0)
    assert full.graphite_mass == pytest.approx(2.0 * vs.graphite_mass, rel=1e-12)


def test_phase_boundary_curves_for_plotting():
    curves = synthesis.phase_boundary_curves(300.0, 3500.0, 33)
    assert set(curves) == {"graphite_diamond_kennedy_1976", "graphite_diamond_berman_simon",
                           "hbn_cbn_solozhenko_1999", "hbn_cbn_corrigan_bundy_1975"}
    for ts, ps in curves.values():
        assert len(ts) == len(ps) == 33 and ts[0] == 300.0 and ts[-1] == pytest.approx(3500.0)
        assert all(b >= a for a, b in zip(ps, ps[1:]))
    ts, ps = curves["graphite_diamond_kennedy_1976"]
    assert ps[5] == pytest.approx(synthesis.graphite_diamond_boundary(ts[5]))
    json.dumps(curves)  # plain floats only


# ---------------------------------------------------------- error paths
@pytest.mark.parametrize("call", [
    lambda: synthesis.graphite_diamond_boundary(0.0),
    lambda: synthesis.graphite_diamond_boundary(-5.0),
    lambda: synthesis.graphite_diamond_boundary(1500.0, "bundy"),
    lambda: synthesis.hbn_cbn_boundary(1500.0, "solozhenko"),
    lambda: synthesis.diamond_stable(-1.0, 1500.0),
    lambda: synthesis.cbn_stable(1e9, float("nan")),
    lambda: synthesis.assess_route("unobtainium_route", 4e9),
    lambda: synthesis.assess_route("cbn_catalytic", -1.0),
    lambda: synthesis.assess_route("cbn_catalytic", 4e9, temperature=0.0),
    lambda: synthesis.assess_route("cbn_catalytic", 4e9, pulse_duration=0.0),
    lambda: synthesis.assess_route("cbn_catalytic", 4e9, intensification=0.0),
    lambda: synthesis.assess_route("lonsdaleite_shock", 4e9, flyer="copper_ofhc"),
    lambda: FlyerSpec("copper_ofhc", 0.0, 0.01),
    lambda: FlyerSpec("copper_ofhc", 0.002, -0.01),
    lambda: FlyerSpec("copper_ofhc", 0.002, 0.01, efficiency=1.2),
    lambda: synthesis.anvil_feasibility(-1.0),
    lambda: synthesis.volume_scaling(0.05, 0.1, 4e9, fill_fraction=0.0),
    lambda: synthesis.volume_scaling(0.0, 0.1, 4e9),
    lambda: synthesis.phase_boundary_curves(300.0, 200.0),
    lambda: synthesis.phase_boundary_curves(300.0, 3500.0, 1),
    lambda: SynthesisRoute(key="x", product="diamond", kind="magic", p_min=1e9, p_max=None, t_min=None,
                           t_max=None, min_dwell=1.0, typical_dwell=None, heating="", product_form="",
                           current_scale="", reference=""),
])
def test_invalid_inputs_raise_input_error(call):
    with pytest.raises(InputError):
        call()


def test_unknown_flyer_material():
    with pytest.raises(UnknownMaterialError):
        FlyerSpec("unobtainium", 0.002, 0.01)


# --------------------------------------------------------- serialisation
def test_json_round_trips_and_text_rendering():
    static = synthesis.assess_route("diamond_hpht_catalytic", 4e9, intensification=1.5)
    shock_ = synthesis.assess_route("lonsdaleite_shock", 4e9, flyer=CU_FLYER)
    objs = [SYNTHESIS_ROUTES["diamond_hpht_catalytic"], SYNTHESIS_ROUTES["lonsdaleite_shock"],
            synthesis.anvil_feasibility(500e9), CU_FLYER, static, shock_,
            synthesis.volume_scaling(0.05, 0.1, 4e9)]
    for o in objs:
        d = json.loads(o.json())
        assert isinstance(d, dict) and d
        txt = o.text()
        assert txt.startswith(type(o).__name__)
    s = json.loads(static.json())
    assert s["status"] == "infeasible" and s["anvil"]["technology"] == "belt_cubic_wc"
    assert s["shock"] is None and s["flyer_velocity"] is None
    k = json.loads(shock_.json())
    assert k["status"] == "extrapolation" and k["shock"]["target"] == "graphite_crystal"
    assert k["shock"]["confidence"] == "estimated" and k["temperature_ok"] is None
    assert json.loads(SYNTHESIS_ROUTES["lonsdaleite_shock"].json())["p_max"] is None
    assert json.loads(CU_FLYER.json())["material"] == "copper_ofhc"
    assert json.loads(FlyerSpec(materials.get("cu"), 0.002, 0.01).json())["material"] == "copper_ofhc"
    assert "pressure_sample [Pa]" in shock_.text() and "shock:" in shock_.text()
    assert isinstance(static, SynthesisAssessment) and isinstance(static.anvil, AnvilClass)
    assert isinstance(synthesis.volume_scaling(0.05, 0.1, 4e9), VolumeScaling)


def test_public_api_is_documented():
    for name in synthesis.__all__:
        obj = getattr(synthesis, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 80, name
    assert all(isinstance(k, str) and v for k, v in synthesis.REFERENCES.items())
