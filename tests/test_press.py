"""Tests for magform.press: the press as a system and the comparison of architectures."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from magform import conductor, fieldmap, press
from magform.constants import MU0
from magform.errors import InputError, UnknownMaterialError
from magform.magnetics import Solenoid, field_for_pressure, pulse_increment_for_pressure
from magform.press import PressSpec, compare_architectures, design_press, reference_spec
from magform.status import Status


@pytest.fixture(scope="module")
def ref4():
    return design_press(reference_spec(4e9))


@pytest.fixture(scope="module")
def ref16():
    return design_press(reference_spec(1.6e9))


@pytest.fixture(scope="module")
def tube4():
    return design_press(reference_spec(4e9, workpiece_wall=2e-3))


def _by_key(design):
    return {a.key: a for a in design.architectures}


# ------------------------------------------------------------------ specification
def test_spec_derived_quantities_and_defaults():
    spec = reference_spec()
    assert spec.pressure == 4e9 and spec.field == pytest.approx(100.27, rel=1e-4)
    sol = spec.solenoid
    assert (sol.inner_radius, sol.outer_radius, sol.length) == pytest.approx((0.05, 0.25, 0.5))
    assert spec.billet_radius == pytest.approx(0.04) and spec.billet_length == pytest.approx(0.30)
    assert spec.wall == spec.billet_radius and spec.workpiece == "copper_ofhc"
    # solid copper billet: skin depth = radius/3 needs a ~10 ms rise; a 2 mm wall needs ~26 us
    assert spec.matched_rise_time == pytest.approx(1.026e-2, rel=1e-3)
    tube = reference_spec(4e9, workpiece_wall=2e-3)
    assert tube.wall == 2e-3 and tube.matched_rise_time == pytest.approx(2.565e-5, rel=1e-3)
    assert reference_spec(4e9, rise_time=1e-3).matched_rise_time == 1e-3
    assert PressSpec(4e9, 0.1, 0.5, 0.2, workpiece="Cu").workpiece == "copper_ofhc"   # alias resolved
    assert reference_spec(1.6e9).field == pytest.approx(63.41, rel=1e-4)


@pytest.mark.parametrize("kwargs", [
    dict(pressure=0.0), dict(pressure=-1.0), dict(bore_diameter=0.0), dict(length=float("nan")),
    dict(winding_thickness=-0.1), dict(workpiece_radius=0.05), dict(workpiece_radius=0.06),
    dict(workpiece_wall=0.05), dict(rise_time=0.0), dict(temperature=0.0), dict(safety_factor=0.9),
    dict(background_field=-1.0), dict(composite="carbon"), dict(workpiece="diamond"), dict(workpiece_length=0.0),
])
def test_spec_validation(kwargs):
    base = dict(pressure=4e9, bore_diameter=0.1, length=0.5, winding_thickness=0.2)
    base.update(kwargs)
    with pytest.raises(InputError):
        PressSpec(**base)


def test_unknown_workpiece_material():
    with pytest.raises(UnknownMaterialError):
        PressSpec(4e9, 0.1, 0.5, 0.2, workpiece="unobtainium")
    with pytest.raises(InputError):
        design_press("not a spec")
    with pytest.raises(InputError):
        compare_architectures(None)


# ------------------------------------------------------------------ the reference design at 4 GPa
def test_reference_design_headline_numbers(ref4):
    d = ref4
    assert d.field == pytest.approx(100.27, rel=1e-4)
    assert d.field_factor == pytest.approx(0.853, rel=2e-3)
    assert d.stored_energy == pytest.approx(99.48e6, rel=1e-3)
    assert d.energy_density == 4e9 and d.tnt_fraction == pytest.approx(0.58, abs=0.01)
    # Montgomery current density is 1/f above the long-coil value, and so is the peak hoop stress
    assert d.current_density == pytest.approx(d.field / (MU0 * 0.2 * d.field_factor), rel=1e-9)
    assert d.peak_hoop_stress == pytest.approx(3.125e9 / d.field_factor, rel=1e-6) and d.hoop_radius == 0.125
    assert d.hoop_ratio == pytest.approx(0.916, rel=2e-3) and d.hoop_ratio > 0.5
    assert d.average_hoop_stress == pytest.approx(4e9 * (0.25 + 0.1) / 0.6, rel=1e-6)
    assert d.end_field_fraction == pytest.approx(fieldmap.end_field(Solenoid(0.05, 0.25, 0.5)).fraction)
    assert d.end_field_fraction == pytest.approx(0.282, rel=5e-3)
    assert d.end_field_fraction < conductor.DEFAULT_END_FIELD_FRACTION
    assert d.peak_conductor_field == pytest.approx(100.77, rel=1e-3) and d.peak_conductor_field > d.field
    assert d.rise_time == pytest.approx(1.026e-2, rel=1e-3)
    assert d.virial_mass > 0 and d.placement.nearest == "lanl_100t_pressure"


def test_concept_architecture_is_infeasible_at_4gpa(ref4):
    d = ref4
    assert d.status is Status.INFEASIBLE and d.margin == 0.0
    assert d.conductor.status is Status.ASPIRATIONAL and d.conductor.margin < 1.0
    assert d.conductor.end_field_fraction == pytest.approx(d.end_field_fraction)
    assert d.structure.status is Status.INFEASIBLE
    assert d.structure.required_composite_modulus == pytest.approx(1.5 * d.peak_hoop_stress / 0.004, rel=1e-9)
    assert d.pulsed_operation.status is Status.INFEASIBLE
    assert any("conductor" in w for w in d.warnings) and any("structure" in w for w in d.warnings)
    assert any("melts at the surface" in w for w in d.warnings)            # copper skin at 100 T
    assert any("steady field exerts no pressure" in n for n in d.notes)
    assert d.best_architecture == "pulsed_resistive" and d.best_status is Status.EXTRAPOLATION


def test_concept_architecture_at_1p6gpa_fails_on_pulsing(ref16):
    d = ref16
    assert d.status is Status.INFEASIBLE
    assert d.conductor.status is Status.EXTRAPOLATION and d.conductor.margin > 1.0   # the tape could carry it in DC
    assert d.structure.status is Status.ASPIRATIONAL                                  # needs a material that does not exist
    assert d.pulsed_operation.status is Status.INFEASIBLE
    assert d.pulsed_operation.loss_to_margin_ratio == pytest.approx(99.3, rel=5e-2)
    assert d.margin == pytest.approx(1.0 / d.pulsed_operation.loss_to_margin_ratio, rel=1e-6)
    assert d.best_architecture == "pulsed_resistive" and d.best_status is Status.EXTRAPOLATION
    assert d.network.status is Status.STATE_OF_THE_ART and d.network.bank_energy == pytest.approx(40.8e6, rel=1e-2)
    assert not d.workpiece.skin_melts


# ------------------------------------------------------------------ workpiece
def test_workpiece_load_solid_billet(ref4):
    w = ref4.workpiece
    assert w.material == "copper_ofhc" and w.solid and w.regime == "thick"
    assert w.pressure_transfer > 0.99
    assert w.surface_mid_ratio == pytest.approx(1.064, rel=5e-3) and w.surface_uniformity == pytest.approx(0.970, abs=5e-3)
    assert w.surface_peak_ratio > 4.0                                       # corners
    assert w.delivered_pressure == pytest.approx(4e9 * w.pressure_transfer * w.surface_mid_ratio, rel=1e-12)
    assert w.skin_temperature_rise == pytest.approx(1160, rel=1e-2) and w.skin_melts
    assert w.skin_melting_field == pytest.approx(96.0, rel=1e-2)
    assert w.elastic_volumetric_strain == pytest.approx(0.033, abs=2e-3)


def test_workpiece_load_thin_wall_leaks_slow_pulses():
    spec = reference_spec(4e9, workpiece_wall=2e-3)
    fast = press.workpiece_load(spec)
    assert fast.regime == "thick" and fast.pressure_transfer == pytest.approx(1.0 - math.exp(-6.0), rel=1e-6)
    slow = press.workpiece_load(spec, rise_time=5e-3)
    assert slow.regime == "thin" and slow.pressure_transfer < 0.4
    assert slow.delivered_pressure < 0.45 * fast.delivered_pressure
    # monotone: the slower the pulse, the less pressure
    transfers = [press.workpiece_load(spec, rise_time=t).pressure_transfer for t in (1e-5, 1e-4, 1e-3, 1e-2)]
    assert all(b < a for a, b in zip(transfers, transfers[1:]))
    with pytest.raises(InputError):
        press.workpiece_load(spec, rise_time=-1.0)
    with pytest.raises(InputError):
        press.workpiece_load("spec")


# ------------------------------------------------------------------ architectures
def test_architecture_order_and_shared_fields(ref4):
    archs = ref4.architectures
    assert tuple(a.key for a in archs) == press.ARCHITECTURES
    assert compare_architectures(reference_spec(4e9)) == archs                # same result standalone
    for a in archs:
        assert a.name and a.summary and a.limiting in a.components
        assert a.margin == a.pressure_fraction >= 0.0
        assert a.components[a.limiting] == a.status.value or a.key == "hts_dc"
        assert isinstance(a.details, dict)


def test_steady_magnet_is_no_press(ref4, ref16):
    for d in (ref4, ref16):
        dc = _by_key(d)["hts_dc"]
        assert dc.status is Status.INFEASIBLE and dc.pressure_fraction == 0.0 and dc.rise_time is None
        assert dc.limiting == "pressure on the workpiece"
        assert any("no forming pressure" in n for n in dc.notes)


def test_resistive_pulsed_magnet_is_the_buildable_option_for_a_billet(ref4, ref16):
    for d in (ref4, ref16):
        a = _by_key(d)["pulsed_resistive"]
        assert a.status is Status.EXTRAPOLATION and a.components["field and bore"] == "extrapolation"
        assert a.pressure_fraction == pytest.approx(1.062, rel=1e-2)
        assert a.relaxed_rise_time is None                                 # the supply keeps up with a solid billet
        assert a.details["fastest_practical_rise_time"] < a.rise_time
    a4 = _by_key(ref4)["pulsed_resistive"]
    assert a4.details["turns"] == pytest.approx(38, abs=2) and a4.details["charge_voltage"] <= 25e3
    assert a4.details["peak_current"] == pytest.approx(1.23e6, rel=5e-2)
    assert a4.energy == pytest.approx(101.9e6, rel=1e-2)


def test_thin_wall_exposes_the_time_conflict(tube4):
    a = _by_key(tube4)["pulsed_resistive"]
    assert a.status is Status.INFEASIBLE and a.rise_time == pytest.approx(2.565e-5, rel=1e-3)
    assert a.details["charge_voltage"] > 200e3 and a.details["peak_current"] > 40e6    # 1 turn, MV x MA class
    assert a.relaxed_rise_time == pytest.approx(2.15e-3, rel=5e-2) and a.relaxed_status is Status.EXTRAPOLATION
    assert a.relaxed_pressure_fraction == pytest.approx(0.51, abs=0.03)
    assert any("time conflict" in w for w in a.warnings)
    assert any("coil heating infeasible" in w for w in a.warnings)
    # nothing reusable delivers the pressure: the least-bad option destroys its coil
    assert tube4.best_architecture == "single_turn" and tube4.best_status is Status.ASPIRATIONAL
    assert any("consumed by every shot" in n for n in tube4.notes)


def test_hybrid_does_not_rescue_the_superconductor(ref4, ref16):
    h4, h16 = _by_key(ref4)["hybrid_dc_pulsed"], _by_key(ref16)["hybrid_dc_pulsed"]
    d_b = pulse_increment_for_pressure(4e9, 11.5)
    assert h4.details["pulse_increment"] == pytest.approx(d_b) and h4.field == pytest.approx(11.5 + d_b)
    assert h4.field > ref4.field                                             # the bias raises the total field
    assert h4.details["energy_saving"] == pytest.approx(1.0 - (d_b / ref4.field) ** 2)
    assert h4.components["insert field and bore"] == "aspirational"          # 100.9 T exceeds the 100.75 T record
    assert h4.details["flux_ratio"] == pytest.approx(2.66, rel=1e-2) and h4.details["field_dip"] > 11.5
    assert any("reversed" in w for w in h4.warnings)
    assert h4.status is Status.ASPIRATIONAL and h16.status is Status.ASPIRATIONAL
    assert h16.limiting == "background coupling" and h16.components["background magnet"] == "state_of_the_art"
    assert h16.details["energy_saving"] == pytest.approx(0.30, abs=0.01)


def test_background_coupling_closed_forms():
    insert = Solenoid(0.05, 0.25, 0.5)
    c = press.background_coupling(insert, 50.0, 11.5, 1e-2)
    flux = 50.0 * math.pi * (0.05 ** 2 + 0.05 * 0.2 + 0.2 ** 2 / 3.0)
    assert c.insert_flux == pytest.approx(flux) and c.background_radius == pytest.approx(0.275)
    assert c.background_flux == pytest.approx(11.5 * math.pi * 0.275 ** 2)
    assert c.flux_ratio == pytest.approx(c.insert_flux / c.background_flux)
    assert c.field_dip == pytest.approx(flux / (math.pi * 0.275 ** 2)) and c.reverses
    assert c.dip_rate == pytest.approx(c.field_dip / 1e-2) and c.volts_per_turn == pytest.approx(flux / 1e-2)
    assert c.status is Status.ASPIRATIONAL and c.ramp_rate_status is Status.ASPIRATIONAL
    # a small insert deep inside a wide background barely disturbs it in flux, but the ramp rate still counts
    small = press.background_coupling(Solenoid(0.0075, 0.06, 0.15), 70.0, 32.0, 1.0, background_radius=0.3)
    assert small.flux_ratio < 0.05 and not small.reverses and small.field_dip == pytest.approx(1.06, rel=1e-2)
    assert small.status is Status.STATE_OF_THE_ART                            # 1 T in 1 s: an ordinary ramp
    # a wider background coil links the same flux over a larger area
    wide = press.background_coupling(insert, 50.0, 11.5, 1e-2, background_radius=1.0)
    assert wide.field_dip == pytest.approx(c.field_dip * (0.275 / 1.0) ** 2)
    none = press.background_coupling(insert, 50.0, 0.0, 1e-2)
    assert none.status is Status.NOT_APPLICABLE
    for call in (lambda: press.background_coupling(insert, 0.0, 11.5, 1e-2),
                 lambda: press.background_coupling(insert, 50.0, -1.0, 1e-2),
                 lambda: press.background_coupling(insert, 50.0, 11.5, 0.0),
                 lambda: press.background_coupling(insert, 50.0, 11.5, 1e-2, background_radius=0.1),
                 lambda: press.background_coupling("coil", 50.0, 11.5, 1e-2)):
        with pytest.raises(InputError):
            call()


def test_background_magnet_status_envelope():
    assert press.background_magnet_status(11.5, 0.55) is Status.STATE_OF_THE_ART
    assert press.background_magnet_status(32.0, 0.034) is Status.STATE_OF_THE_ART
    assert press.background_magnet_status(32.0, 0.5) is Status.EXTRAPOLATION
    assert press.background_magnet_status(45.0, 0.05) is Status.EXTRAPOLATION
    assert press.background_magnet_status(50.0, 0.05) is Status.ASPIRATIONAL
    with pytest.raises(InputError):
        press.background_magnet_status(-1.0, 0.1)
    with pytest.raises(InputError):
        press.background_magnet_status(10.0, 0.0)


def test_single_turn_coil_closed_forms_and_small_bore(ref4):
    s = _by_key(ref4)["single_turn"]
    inductance = MU0 * math.pi * 0.05 ** 2 / 0.1
    current = ref4.field * 0.1 / MU0
    assert s.details["peak_current"] == pytest.approx(current) and not s.reusable
    assert s.energy == pytest.approx(0.5 * inductance * current ** 2)
    assert s.energy == pytest.approx(4e9 * math.pi * 0.05 ** 2 * 0.1)        # the bore field energy
    assert s.details["charge_voltage"] == pytest.approx(inductance * current * math.pi / (2 * press.SINGLE_TURN_RISE_TIME))
    assert s.status is Status.ASPIRATIONAL and s.limiting == "bank voltage"
    small = design_press(PressSpec(pressure=4e9, bore_diameter=0.010, length=0.15, winding_thickness=0.1))
    by = _by_key(small)
    assert by["single_turn"].status is Status.STATE_OF_THE_ART                 # megagauss practice, 10 mm bore
    assert by["pulsed_resistive"].components["field and bore"] == "state_of_the_art"   # the Los Alamos record
    assert small.stored_energy == pytest.approx(small.spec.solenoid.stored_energy(small.field))
    assert 1e6 < small.stored_energy < 1e7
    wider = design_press(PressSpec(pressure=4e9, bore_diameter=0.015, length=0.15, winding_thickness=0.1))
    assert _by_key(wider)["pulsed_resistive"].components["field and bore"] == "extrapolation"
    assert small.best_architecture == "single_turn" and small.status is Status.INFEASIBLE
    assert any("consumed by every shot" in n for n in small.notes)


def test_industrial_coil_is_capped_and_real_at_small_scale(ref4, ref16):
    e4, e16 = _by_key(ref4)["industrial_emf"], _by_key(ref16)["industrial_emf"]
    assert e4.field == 50.0 and e4.rise_time == press.INDUSTRIAL_RISE_TIME
    assert e4.pressure_fraction == pytest.approx(ref4.workpiece.surface_mid_ratio * (50.0 / ref4.field) ** 2, rel=1e-2)
    assert e16.pressure_fraction == pytest.approx(0.66, abs=0.02)
    assert any("capped at 50 T" in w for w in e4.warnings) and e4.status is Status.EXTRAPOLATION
    # a 25 mm coil at 30 T on a 1 mm aluminium tube is what industry does
    small = design_press(PressSpec(pressure=0.36e9, bore_diameter=0.025, length=0.05, winding_thickness=0.01,
                                   workpiece="aluminum_6061", workpiece_wall=1e-3))
    by = _by_key(small)
    assert by["industrial_emf"].status is Status.STATE_OF_THE_ART and by["industrial_emf"].field < 50.0
    assert by["pulsed_resistive"].status is Status.STATE_OF_THE_ART
    assert small.best_status is Status.STATE_OF_THE_ART
    assert small.status is Status.INFEASIBLE                                   # the superconducting concept still is not
    assert any("beta" in w for w in small.warnings)                            # short coil: long-coil estimates flagged


def test_status_never_improves_with_pressure():
    ranks = [design_press(reference_spec(p)).best_status.rank for p in (0.2e9, 0.8e9, 1.6e9, 4e9)]
    assert ranks == sorted(ranks)
    energies = [design_press(reference_spec(p)).stored_energy for p in (1e9, 2e9, 4e9)]
    assert energies[1] == pytest.approx(2 * energies[0]) and energies[2] == pytest.approx(4 * energies[0])


def test_custom_composite_and_background_are_used():
    stiff = conductor.CNSComposite(f_cns=0.6, f_hts=0.2, f_cu=0.1, f_insulation=0.1, cns_material="graphene_intrinsic")
    d = design_press(reference_spec(1.6e9, composite=stiff, background_field=32.0))
    assert d.structure.cns_material == "graphene_intrinsic"
    h = _by_key(d)["hybrid_dc_pulsed"]
    assert h.details["background_field"] == 32.0
    assert h.components["background magnet"] == "extrapolation"               # 32 T in a 550 mm bore is not demonstrated
    assert h.field == pytest.approx(math.hypot(32.0, field_for_pressure(1.6e9)))


# ------------------------------------------------------------------ serialisation and metadata
def test_design_round_trips_json_and_renders_text(ref4):
    data = json.loads(ref4.json())
    assert data["status"] == "infeasible" and data["best_architecture"] == "pulsed_resistive"
    assert data["spec"]["pressure"] == 4e9 and data["spec"]["composite"] is None
    assert [a["key"] for a in data["architectures"]] == list(press.ARCHITECTURES)
    assert data["architectures"][0]["details"]["loss_to_margin_ratio"] is None      # inf -> null keeps the JSON strict
    assert data["conductor"]["status"] == "aspirational" and data["network"]["turns"] == ref4.network.turns
    text = ref4.text()
    assert text.startswith("PressDesign") and "field [T] = 100.3" in text and "architectures:" in text
    assert "stored_energy [J]" in text
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref4.field = 1.0


def test_module_metadata():
    assert set(press.__all__) <= set(dir(press))
    assert press.REFERENCES and all(isinstance(v, str) and len(v) > 20 for v in press.REFERENCES.values())
    for name in press.__all__:
        obj = getattr(press, name)
        if callable(obj):
            assert obj.__doc__ and "Reference" in obj.__doc__, name
    assert press.ARCHITECTURES[0] == "hts_pulsed" and len(press.ARCHITECTURES) == 6
    import magform
    assert magform.PressSpec is PressSpec and magform.design_press is design_press
    assert magform.compare_architectures is compare_architectures
