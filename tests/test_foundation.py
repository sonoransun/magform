"""Tests for the shared foundation: status, result objects, benchmarks, materials."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from magform import benchmarks, materials
from magform._report import Assessment, ResultMixin, unit
from magform.errors import InputError, MagformError, UnknownMaterialError
from magform.status import Confidence, Status, Verdict, best, verdict_from_status, worst


# --------------------------------------------------------------------- status
def test_status_str_is_value_and_rollup():
    assert str(Status.EXTRAPOLATION) == "extrapolation"
    assert f"{Status.INFEASIBLE}" == "infeasible"
    assert worst(Status.STATE_OF_THE_ART, Status.ASPIRATIONAL, Status.NOT_APPLICABLE) is Status.ASPIRATIONAL
    assert worst([Status.NOT_APPLICABLE]) is Status.NOT_APPLICABLE
    assert best(Status.INFEASIBLE, Status.EXTRAPOLATION) is Status.EXTRAPOLATION
    assert verdict_from_status(Status.INFEASIBLE) is Verdict.INFEASIBLE
    assert json.dumps({"s": Status.STATE_OF_THE_ART}) == '{"s": "state_of_the_art"}'
    # 'assumed' is a premise, not a judgement: it never wins or loses a roll-up
    assert str(Status.ASSUMED) == "assumed" and Status.ASSUMED.assessed is False
    assert Status.INFEASIBLE.assessed and not Status.NOT_APPLICABLE.assessed
    assert worst(Status.ASSUMED, Status.EXTRAPOLATION) is Status.EXTRAPOLATION
    assert best(Status.ASSUMED, Status.EXTRAPOLATION) is Status.EXTRAPOLATION
    assert worst(Status.ASSUMED) is Status.NOT_APPLICABLE
    assert verdict_from_status(Status.ASSUMED) is Verdict.NOT_MODELLED
    assert {s.rank for s in Status} == {-2, -1, 0, 1, 2, 3}


# ------------------------------------------------------------- result objects
@dataclasses.dataclass(frozen=True, kw_only=True)
class _Demo(Assessment):
    pressure: float = unit("Pa")
    ratio: float = math.inf
    material: materials.Material | None = None
    series: tuple[float, ...] = ()


def test_result_mixin_json_is_strict_and_collapses_materials():
    d = _Demo(status=Status.EXTRAPOLATION, pressure=4e9, material=materials.get("cu"), series=(1.0, float("nan")))
    out = json.loads(d.json())
    assert out["status"] == "extrapolation"
    assert out["ratio"] is None  # inf -> None keeps JSON valid
    assert out["material"] == "copper_ofhc"
    assert out["series"] == [1.0, None]
    txt = d.text()
    assert "pressure [Pa] = 4e+09" in txt and "ratio = inf" in txt


# ------------------------------------------------------------------ errors
def test_error_hierarchy():
    assert issubclass(InputError, ValueError) and issubclass(InputError, MagformError)
    assert issubclass(UnknownMaterialError, KeyError)
    with pytest.raises(UnknownMaterialError) as exc:
        materials.get("unobtainium")
    assert "unknown material" in str(exc.value)


# ---------------------------------------------------------------- benchmarks
def test_benchmark_statuses():
    assert benchmarks.superconducting_field_status(32.0) is Status.STATE_OF_THE_ART
    assert benchmarks.superconducting_field_status(40.0) is Status.EXTRAPOLATION
    assert benchmarks.superconducting_field_status(48.7) is Status.EXTRAPOLATION    # NHMFL test coil, 2025
    assert benchmarks.superconducting_field_status(48.8) is Status.ASPIRATIONAL
    assert benchmarks.superconducting_field_status(63.4) is Status.ASPIRATIONAL
    assert benchmarks.PULSED_NONDESTRUCTIVE_BORE == 0.010                            # as operated, not the 15 mm design
    assert benchmarks.pulsed_field_status(100.27, 0.010) is Status.STATE_OF_THE_ART
    assert benchmarks.pulsed_field_status(100.27, 0.015) is Status.EXTRAPOLATION
    assert benchmarks.pulsed_field_status(60.0, 0.015) is Status.STATE_OF_THE_ART    # routine user magnets
    assert benchmarks.pulsed_field_status(100.27, 0.10) is Status.EXTRAPOLATION
    assert benchmarks.pulsed_field_status(150.0) is Status.ASPIRATIONAL
    assert benchmarks.energy_status(40e6) is Status.STATE_OF_THE_ART
    assert benchmarks.energy_status(500e6) is Status.EXTRAPOLATION
    assert benchmarks.energy_status(2e9) is Status.ASPIRATIONAL
    assert benchmarks.ramp_rate_status(2.0) is Status.STATE_OF_THE_ART
    assert benchmarks.ramp_rate_status(1e5) is Status.ASPIRATIONAL


def test_where_does_it_sit_4gpa():
    p = benchmarks.where_does_it_sit(4e9)
    assert p.field_equivalent == pytest.approx(100.27, rel=1e-3)
    assert "cold_forging_die" in p.below and "hpht_cubic" in p.above
    assert 0.5 < p.tnt_fraction < 0.7


def test_every_benchmark_is_cited():
    for ladder in benchmarks.ALL_LADDERS.values():
        for b in ladder:
            assert b.citation and b.year > 1900 and b.value > 0


# ---------------------------------------------------------------- materials
def test_registry_invariants_and_units():
    for key, m in materials.MATERIALS.items():
        assert m.key == key
        assert 500 < m.density < 25000
        assert 1e9 < m.youngs_modulus < 2e12
        assert 200 < m.melting_point < 5000
        for s in (m.yield_strength, m.ultimate_strength, m.compressive_yield, m.compressive_strength):
            assert s is None or 1e6 <= s <= 2e11, key
        if m.yield_strength is not None and m.ultimate_strength is not None:
            assert m.ultimate_strength >= m.yield_strength, key
        for mdl in m.constitutive:
            if isinstance(mdl, materials.JohnsonCook):
                assert 1e7 <= mdl.A <= 5e9, key
        if m.hall_petch is not None:
            assert 1e4 <= m.hall_petch.k <= 5e6, key
        assert m.shear_modulus is not None and m.bulk_modulus is not None
        assert isinstance(m.confidence, Confidence)


def test_aliases_resolve():
    for alias, key in materials.ALIASES.items():
        assert materials.get(alias).key == key
    assert materials.get("  Cu ").key == "copper_ofhc"
    assert materials.as_material(materials.get("W")) is materials.get("tungsten")


def test_constitutive_reference_values():
    w = materials.get("tungsten")
    lr, cg = w.constitutive
    assert lr.stress(0.3, 1e4, 293.15) == pytest.approx(2.05e9, rel=0.02)
    assert cg.stress(0.3, 1e4, 293.15) == pytest.approx(2.58e9, rel=0.02)
    wha = materials.get("tungsten_heavy_alloy_90w").constitutive[0]
    assert wha.stress(0.3, 1e4, 293.15) == pytest.approx(1.904e9, rel=0.01)
    mo = materials.get("molybdenum").constitutive[0]
    assert mo.stress(0.3, 1e4, 293.15) == pytest.approx(1.57e9, rel=0.02)
    tial = materials.get("gamma_tial").constitutive[0]
    assert tial.stress(0.05, 1e-3, 293.15) == pytest.approx(0.97e9, rel=0.02)
    assert tial.stress(0.05, 1e4, 293.15) == pytest.approx(1.46e9, rel=0.02)


def test_derived_properties():
    cu = materials.get("cu")
    assert cu.resistivity_at(77.0) == pytest.approx(2.7e-9, rel=0.1)
    assert cu.resistivity_at(4.2) == pytest.approx(0.02 * cu.resistivity)
    assert cu.hardness_estimate() == pytest.approx(440e6)
    re_ = materials.get("rhenium")
    assert re_.hugoniot_for().c0 == pytest.approx(re_.bulk_sound_speed())
    armco = materials.get("iron_armco")
    assert armco.hugoniot_for(5e9).c0 == 4630.0 and armco.hugoniot_for(20e9).c0 == 3574.0
    K, n = materials.get("tungsten").hollomon()
    assert 0.01 <= n <= 0.6 and K * 0.002 ** n == pytest.approx(1.3e9)
    assert materials.get("cu").volumetric_strain(4e9) == pytest.approx(4e9 / cu.bulk_modulus)
    with pytest.raises(InputError):
        materials.get("diamond").hollomon()
