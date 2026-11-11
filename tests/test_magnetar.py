"""Tests for magform.magnetar: the magnetar-mode thought experiment (premise assumed, consequences computed)."""

from __future__ import annotations

import json
import math

import pytest

from magform import benchmarks, eddy, materials
from magform import magnetar as mg
from magform._report import Assessment
from magform.constants import C_LIGHT, E_CHARGE, G_ACCEL, MU0
from magform.errors import InputError, UnknownMaterialError
from magform.magnetics import field_for_pressure, magnetic_pressure
from magform.status import Confidence, Status, worst

EV = E_CHARGE
MEV = 1e6 * E_CHARGE
YEAR = 3.15576e7
SOURCE = mg.MagnetarSource(1e10, 0.01)      # a 10 mm sphere at a typical magnetar surface field


# ------------------------------------------------------------------ characteristic scales
def test_characteristic_fields_follow_from_constants():
    assert mg.ATOMIC_FIELD == pytest.approx(2.35052e5, rel=1e-5)
    assert mg.ATOMIC_FIELD == pytest.approx(mg.HBAR / (E_CHARGE * mg.BOHR_RADIUS ** 2), rel=1e-8)
    assert mg.QED_CRITICAL_FIELD == pytest.approx(4.4140e9, rel=1e-4)
    assert 1.0 / mg.FINE_STRUCTURE == pytest.approx(137.036, rel=1e-6)
    # defining properties: cyclotron energy = Hartree at B0, = m_e c^2 at B_Q
    assert mg.cyclotron_energy(mg.ATOMIC_FIELD) == pytest.approx(mg.HARTREE, rel=1e-8)
    assert mg.cyclotron_energy(mg.QED_CRITICAL_FIELD) == pytest.approx(mg.ELECTRON_REST_ENERGY, rel=1e-12)
    assert mg.ELECTRON_REST_ENERGY / MEV == pytest.approx(0.51100, rel=1e-5)
    assert mg.cyclotron_energy(1.0) / EV == pytest.approx(1.1577e-4, rel=1e-4)
    assert mg.magnetic_length(mg.ATOMIC_FIELD) == pytest.approx(mg.BOHR_RADIUS, rel=1e-8)
    assert mg.magnetic_length(1e10) == pytest.approx(2.5656e-13, rel=1e-4)
    assert mg.magnetic_length(400.0) == pytest.approx(0.5 * mg.magnetic_length(100.0))
    assert mg.atomic_field_ratio(1e10) == pytest.approx(4.2544e4, rel=1e-4)
    assert mg.qed_field_ratio(1e10) == pytest.approx(2.2655, rel=1e-4)
    assert mg.atomic_field_ratio(-1e10) == mg.atomic_field_ratio(1e10)       # sign is irrelevant
    assert mg.ATOMIC_PRESSURE == pytest.approx(2.9421e13, rel=1e-4)
    assert mg.PRESSURE_IONISATION_FIELD == pytest.approx(8599.0, rel=1e-3)


def test_landau_excitation_energy_limits():
    # non-relativistic limit is the cyclotron energy; at B_Q it is (sqrt(3) - 1) m_e c^2
    assert mg.landau_excitation_energy(100.0) == pytest.approx(mg.cyclotron_energy(100.0), rel=1e-7)
    assert mg.landau_excitation_energy(1e-3) == pytest.approx(mg.cyclotron_energy(1e-3), rel=1e-9)
    assert mg.landau_excitation_energy(mg.QED_CRITICAL_FIELD) == pytest.approx(
        (math.sqrt(3.0) - 1.0) * mg.ELECTRON_REST_ENERGY, rel=1e-12)
    assert mg.landau_excitation_energy(1e10) / MEV == pytest.approx(0.6908, rel=1e-3)
    assert mg.cyclotron_energy(1e10) / MEV == pytest.approx(1.1577, rel=1e-3)
    assert mg.landau_excitation_energy(1e10) < mg.cyclotron_energy(1e10)
    fields = [10.0 ** k for k in range(1, 12)]
    values = [mg.landau_excitation_energy(b) for b in fields]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_scale_input_validation():
    for call in (lambda: mg.magnetic_length(0.0), lambda: mg.cyclotron_energy(float("nan")),
                 lambda: mg.atomic_field_ratio("strong"), lambda: mg.qed_field_ratio(float("inf")),
                 lambda: mg.landau_excitation_energy(None), lambda: mg.atomic_field_ratio(True)):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ the field ladder
def test_field_ladder_runs_from_forming_coils_to_heavy_ion_collisions():
    ladder = mg.EXTREME_FIELD_LADDER
    fields = [r.field for r in ladder]
    assert fields == sorted(fields) and len({r.key for r in ladder}) == len(ladder)
    by = {r.key: r for r in ladder}
    for b in benchmarks.FIELD_LADDER:                    # every laboratory record up to 2800 T is carried over
        assert by[b.key].field == b.value and by[b.key].citation == b.citation
    assert mg.DESTRUCTIVE_FIELD_RECORD == max(b.value for b in benchmarks.FIELD_LADDER)
    assert mg.DESTRUCTIVE_FIELD_RECORD == 2800.0
    assert by["magnetar_typical"].field == 1e10 and by["magnetar_upper"].field == 1e11
    assert by["magnetar_typical"].origin == "astrophysical" and by["magnetar_typical"].steady
    assert by["radio_pulsar"].field == 1e8 and by["magnetic_white_dwarf"].field == 1e5
    # computed rungs
    assert by["z_machine_surface"].field == pytest.approx(MU0 * benchmarks.PULSED_CURRENT_RECORD / (2 * math.pi * 1e-3))
    assert by["z_machine_surface"].field == pytest.approx(5200.0, rel=1e-6)
    assert by["laser_optical_field"].field == pytest.approx(math.sqrt(2 * MU0 * 1.1e27 / C_LIGHT))
    assert by["laser_optical_field"].field == pytest.approx(3.04e6, rel=2e-3)
    assert by["heavy_ion_collision"].field == pytest.approx(3.29e14, rel=2e-3)      # e B = m_pi^2
    assert by["laser_plasma_wagner"].field == 7.0e4 and by["laser_plasma_tatarakis"].field == 3.4e4
    for r in ladder:
        assert r.citation and isinstance(r.confidence, Confidence) and r.origin in ("laboratory", "astrophysical")
    # the honest pattern: above the non-destructive record every laboratory field is destructive or sub-picosecond
    for r in ladder:
        if r.origin == "laboratory" and r.field > benchmarks.PULSED_NONDESTRUCTIVE_RECORD:
            assert r.destructive or (r.duration is not None and r.duration < 1e-12), r.key
    # and nothing human-made above 1e5 T fills even a cubic micrometre for a nanosecond
    assert all(r.volume < 1e-15 and r.duration < 1e-9 for r in ladder
               if r.origin == "laboratory" and r.field > 1e5)


def test_laboratory_record_trades_field_against_volume_and_duration():
    assert mg.laboratory_record().key == "heavy_ion_collision"
    assert mg.laboratory_record(1e-9).key == "z_machine_surface"            # a cubic millimetre
    assert mg.laboratory_record(1e-7).key == "explosive_fc_2800t"
    assert mg.laboratory_record(5e-6, 1e-3).key == "pulsed_user_60t"
    steady = mg.laboratory_record(1e-6, 1.0)
    assert steady.steady and 45.0 <= steady.field < 50.0
    assert mg.laboratory_record(1.0) is None                                 # nobody has filled a cubic metre
    strongest = [mg.laboratory_record(v).field for v in (1e-42, 1e-17, 1e-15, 1e-8, 1e-6, 1e-5, 1e-4)]
    assert all(b <= a for a, b in zip(strongest, strongest[1:]))             # bigger volume, weaker record
    with pytest.raises(InputError):
        mg.laboratory_record(-1.0)
    with pytest.raises(InputError):
        mg.laboratory_record(0.0, float("nan"))


# ------------------------------------------------------------------ pressure, energy, mass equivalent
def test_pressure_energy_and_mass_equivalent_at_magnetar_fields():
    assert magnetic_pressure(1e10) == pytest.approx(3.9789e25, rel=1e-4)
    assert mg.mass_equivalent_density(1e10) == pytest.approx(4.427e8, rel=1e-3)        # denser than any solid
    assert mg.mass_equivalent_density(1e11) == pytest.approx(4.427e10, rel=1e-3)
    assert mg.mass_equivalent_density(100.0) == pytest.approx(4.427e-8, rel=1e-3)
    assert mg.mass_equivalent_density(1e10) == pytest.approx(magnetic_pressure(1e10) / C_LIGHT ** 2)
    cubic_mm = mg.field_energy(1e10, 1e-9)
    assert cubic_mm == pytest.approx(3.9789e16, rel=1e-4)
    assert mg.tnt_megatons(cubic_mm) == pytest.approx(9.510, rel=1e-3)                  # megatons per cubic millimetre
    assert mg.field_energy(1e10, 1e-18) == pytest.approx(3.9789e7, rel=1e-4)            # 40 MJ per cubic micrometre
    assert mg.field_energy(2e10, 1e-9) == pytest.approx(4.0 * cubic_mm)
    assert mg.field_energy(1e10, 0.0) == 0.0
    assert mg.MEGATON_TNT == 4.184e15


def test_energy_scales_and_ratios():
    energies = [s.energy for s in mg.ENERGY_SCALES]
    assert energies == sorted(energies)
    by = {s.key: s for s in mg.ENERGY_SCALES}
    assert by["largest_pulsed_store"].energy == benchmarks.ENERGY_ANY_STORE_RECORD
    assert by["megaton_tnt"].energy == mg.MEGATON_TNT and by["magnetar_giant_flare"].energy == 2.0e39
    ratios = mg.energy_ratios(mg.field_energy(1e10, 1e-9))
    assert set(ratios) == set(by)
    assert ratios["megaton_tnt"] == pytest.approx(9.510, rel=1e-3)
    assert ratios["largest_pulsed_store"] == pytest.approx(3.316e7, rel=1e-3)
    assert ratios["largest_nuclear_test"] < 1.0 < ratios["megaton_tnt"]
    assert mg.energy_ratios(SOURCE.energy())["world_annual_energy"] == pytest.approx(0.4167, rel=1e-3)
    for call in (lambda: mg.field_energy(1e10, -1.0), lambda: mg.tnt_megatons(-1.0), lambda: mg.energy_ratios("lots")):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ Chandrasekhar's equation of state
def test_degeneracy_pressure_limits_and_units():
    a = mg.DEGENERACY_PRESSURE_UNIT
    assert a == pytest.approx(6.0023e21, rel=1e-4)
    assert mg.DEGENERACY_DENSITY_UNIT == pytest.approx(9.7393e8, rel=1e-4)
    assert mg.LATTICE_COEFFICIENT == pytest.approx(1.25055, rel=1e-5)
    # non-relativistic: P -> (8/5) A x^5; ultra-relativistic: P -> 2 A x^4
    for x in (1e-4, 1e-3, 1e-2):
        assert mg.degeneracy_pressure(x) == pytest.approx(1.6 * a * x ** 5, rel=2e-4)
    for x in (1e2, 1e3, 1e4):
        assert mg.degeneracy_pressure(x) == pytest.approx(2.0 * a * x ** 4, rel=2e-4)
    assert mg.degeneracy_pressure(1e-2) / mg.degeneracy_pressure(1e-3) == pytest.approx(1e5, rel=1e-4)
    assert mg.degeneracy_pressure(1e4) / mg.degeneracy_pressure(1e3) == pytest.approx(1e4, rel=1e-5)
    # the series used below x = 0.1 meets the closed form
    assert mg.degeneracy_pressure(0.1 - 1e-9) == pytest.approx(mg.degeneracy_pressure(0.1 + 1e-9), rel=1e-7)
    closed = lambda x: a * (x * (2 * x * x - 3) * math.sqrt(1 + x * x) + 3 * math.asinh(x))  # noqa: E731
    assert mg.degeneracy_pressure(0.09) == pytest.approx(closed(0.09), rel=1e-9)
    assert mg.degeneracy_pressure(1.0) == pytest.approx(closed(1.0), rel=1e-14)
    xs = [10.0 ** (k / 4.0) for k in range(-16, 17)]
    ps = [mg.degeneracy_pressure(x) for x in xs]
    assert all(b > a_ for a_, b in zip(ps, ps[1:]))
    assert mg.degeneracy_pressure(0.0) == 0.0
    # density and the relativistic turning point
    assert mg.degenerate_density(1.0) == pytest.approx(1.9479e9, rel=1e-4)                 # mu_e = 2
    assert mg.degenerate_density(1.0, 1.0) == pytest.approx(mg.DEGENERACY_DENSITY_UNIT)
    assert mg.degenerate_density(2.0) == pytest.approx(8.0 * mg.degenerate_density(1.0))
    assert mg.RELATIVISTIC_DEGENERACY_FIELD == pytest.approx(1.3621e8, rel=1e-4)
    assert mg.RELATIVISTIC_DEGENERACY_FIELD == pytest.approx(field_for_pressure(mg.degeneracy_pressure(1.0)))


def test_equation_of_state_inverses_round_trip():
    for charge in (None, 1, 6, 26, 74):
        for x in (0.3, 1.0, 7.7, 50.0, 3e3):
            p = mg.degeneracy_pressure(x, charge)
            assert mg.fermi_parameter_for_pressure(p, charge) == pytest.approx(x, rel=1e-9), (charge, x)
    for x in (1e-3, 1e-2):                               # ideal gas only: far below the lattice model
        assert mg.fermi_parameter_for_pressure(mg.degeneracy_pressure(x)) == pytest.approx(x, rel=1e-9)
    for x in (0.01, 1.0, 100.0):
        assert mg.fermi_parameter_for_density(mg.degenerate_density(x, 2.154), 2.154) == pytest.approx(x, rel=1e-12)
    assert mg.fermi_parameter_for_density(0.0) == 0.0
    # Fermi kinetic energy: x^2/2 and x in units of m_e c^2
    assert mg.fermi_kinetic_energy(1e-5) == pytest.approx(0.5e-10 * mg.ELECTRON_REST_ENERGY, rel=1e-9)
    assert mg.fermi_kinetic_energy(1e4) == pytest.approx(1e4 * mg.ELECTRON_REST_ENERGY, rel=2e-4)
    assert mg.fermi_kinetic_energy(1e-4) == pytest.approx(mg.fermi_kinetic_energy(1.0000001e-4), rel=1e-6)
    assert mg.fermi_kinetic_energy(math.sqrt(3.0)) == pytest.approx(mg.ELECTRON_REST_ENERGY)


def test_lattice_correction_limits_and_bps_cross_check():
    alpha = mg.FINE_STRUCTURE
    for z in (1, 6, 26):
        z23 = z ** (2.0 / 3.0)
        assert mg.lattice_pressure_fraction(1e4, z) == pytest.approx(
            0.5 * mg.LATTICE_COEFFICIENT * alpha * z23, rel=1e-6)
        assert mg.lattice_pressure_fraction(0.05, z) == pytest.approx(
            mg.LATTICE_COEFFICIENT * alpha * z23 / (1.6 * 0.05), rel=2e-3)
    assert mg.lattice_pressure_fraction(1e4, 26) == pytest.approx(0.0400, rel=2e-3)          # 4 % for iron
    assert mg.degeneracy_pressure(1.0, 26) == pytest.approx(
        mg.degeneracy_pressure(1.0) * (1.0 - mg.lattice_pressure_fraction(1.0, 26)), rel=1e-12)
    assert mg.degeneracy_pressure(0.01, 26) == 0.0                   # below the model's zero-pressure point
    # with the lattice term this EOS reproduces the Baym-Pethick-Sutherland pressure at neutron drip (118Kr)
    x_drip = mg.fermi_parameter_for_density(mg.NEUTRON_DRIP_DENSITY, 118.0 / 36.0)
    assert mg.degeneracy_pressure(x_drip, 36) == pytest.approx(mg.NEUTRON_DRIP_PRESSURE, rel=2e-2)
    assert mg.NEUTRON_DRIP_FIELD == pytest.approx(4.428e11, rel=1e-3)
    assert mg.NEUTRON_DRIP_FIELD > 1e11                               # above the strongest magnetars
    for call in (lambda: mg.lattice_pressure_fraction(0.0, 26), lambda: mg.lattice_pressure_fraction(1.0, 0),
                 lambda: mg.lattice_pressure_fraction(1.0, 26.0), lambda: mg.degeneracy_pressure(-1.0),
                 lambda: mg.degenerate_density(1.0, 0.5), lambda: mg.fermi_parameter_for_pressure(0.0),
                 lambda: mg.degeneracy_pressure(1.0, True)):
        with pytest.raises(InputError):
            call()


def test_piston_state_is_offered_only_where_the_model_holds():
    # validity floors: field at which the lattice correction is 25 % of the ideal pressure
    assert mg.eos_floor_field(26) == pytest.approx(2.485e6, rel=2e-3)
    assert mg.eos_floor_field(29) == pytest.approx(2.994e6, rel=2e-3)
    assert mg.eos_floor_field(6) == pytest.approx(2.104e5, rel=2e-3)
    assert mg.eos_floor_field(1) == pytest.approx(1.058e4, rel=2e-3)
    assert mg.eos_floor_field() == pytest.approx(field_for_pressure(100.0 * mg.ATOMIC_PRESSURE))
    assert mg.eos_floor_field() == pytest.approx(8.599e4, rel=1e-3)
    assert mg.eos_floor_field(1) < mg.eos_floor_field(6) < mg.eos_floor_field(26) < mg.eos_floor_field(74)
    mu = 56.0 / 26.0
    for b in (100.0, 2800.0, 1e5, 2.4e6):          # the concept note's window and megagauss fields: not modelled
        assert mg.piston_state(b, mu, 26) is None and mg.piston_density(b, mu, 26) is None
    assert mg.piston_state(8e4) is None and mg.piston_state(9e4) is not None
    assert mg.piston_state(5e11, mu, 26) is None                   # beyond neutron drip
    at_floor = mg.piston_state(2.49e6, mu, 26)
    assert at_floor.lattice_fraction == pytest.approx(mg.EOS_MAX_LATTICE_FRACTION, rel=1e-2)
    # iron under a typical magnetar field: white-dwarf-core density
    s = mg.piston_state(1e10, mu, 26)
    assert s.pressure == pytest.approx(magnetic_pressure(1e10)) and s.field == 1e10 and s.charge == 26
    assert s.fermi_parameter == pytest.approx(7.698, rel=1e-3) and s.relativistic is True
    assert s.density == pytest.approx(9.568e11, rel=1e-3)
    assert s.fermi_energy / MEV == pytest.approx(3.456, rel=1e-3)
    assert s.lattice_fraction == pytest.approx(0.0407, rel=1e-2)
    assert mg.degeneracy_pressure(s.fermi_parameter, 26) == pytest.approx(s.pressure, rel=1e-9)
    low = mg.piston_state(1e8, mu, 26)
    assert low.density == pytest.approx(1.473e9, rel=1e-3) and low.relativistic is False
    # the lattice pulls the matter together: denser than the ideal gas at the same pressure
    ideal = mg.piston_density(1e10, mu)
    assert ideal == pytest.approx(9.277e11, rel=1e-3) and s.density > ideal
    assert mg.piston_state(1e10, mu).lattice_fraction is None
    densities = [mg.piston_density(b, mu, 26) for b in (3e6, 1e7, 1e8, 1e9, 1e10, 1e11)]
    assert all(b > a for a, b in zip(densities, densities[1:]))
    # ultra-relativistic scaling: P ~ rho^(4/3), so rho ~ B^(3/2)
    assert math.log10(densities[-1] / densities[-2]) == pytest.approx(1.5, abs=0.01)
    assert mg.piston_density(-1e10, mu, 26) == pytest.approx(s.density)
    for call in (lambda: mg.piston_state(1e10, 0.9), lambda: mg.piston_state(1e10, 2.0, 2.5),
                 lambda: mg.piston_state("x"), lambda: mg.eos_floor_field(-1)):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ electron capture (neutronisation)
@pytest.mark.parametrize("nuclide, density, field", [
    ("1H", 1.22e10, 8.72e8),
    ("4He", 1.37e14, 2.95e11),
    ("12C", 3.90e13, 1.269e11),
    ("16O", 1.90e13, 7.85e10),
    ("20Ne", 6.21e12, 3.71e10),
    ("24Mg", 3.16e12, 2.36e10),
    ("28Si", 1.97e12, 1.72e10),
    ("32S", 1.47e11, 2.99e9),
    ("56Fe", 1.14e12, 1.127e10),
])
def test_electron_capture_thresholds_reproduce_the_textbook_table(nuclide, density, field):
    c = mg.electron_capture_threshold(nuclide)
    assert c.density == pytest.approx(density, rel=5e-3)           # Shapiro & Teukolsky densities (kg/m^3)
    assert c.field == pytest.approx(field, rel=5e-3)
    assert c.field == pytest.approx(field_for_pressure(c.pressure))
    n = mg.NUCLIDES[nuclide]
    assert c.threshold_energy == n.threshold_energy and c.daughter == n.daughter
    assert mg.fermi_kinetic_energy(c.fermi_parameter) == pytest.approx(n.threshold_energy, rel=1e-12)
    assert c.density == pytest.approx(mg.degenerate_density(c.fermi_parameter, n.mu_e))
    assert c.pressure == pytest.approx(mg.degeneracy_pressure(c.fermi_parameter, n.charge))
    assert 0.0 < c.lattice_fraction < 0.05


def test_a_magnetar_class_piston_starts_transmuting_iron():
    iron = mg.electron_capture_threshold("56Fe")
    assert 1e10 < iron.field < 1.2e10                                  # between a typical and a strong magnetar
    assert iron.field > mg.QED_CRITICAL_FIELD
    assert iron.threshold_energy / MEV == pytest.approx(3.695)
    assert mg.piston_state(1e10, 56 / 26, 26).fermi_energy < iron.threshold_energy     # 1e10 T falls just short
    assert mg.piston_state(1.2e10, 56 / 26, 26).fermi_energy > iron.threshold_energy
    # copper is the odd one: 63Ni lies only 67 keV above 63Cu, so capture starts at tens of megatesla
    copper = mg.electron_capture_threshold("63Cu")
    assert copper.threshold_energy / MEV == pytest.approx(0.0669)
    assert copper.field == pytest.approx(2.84e7, rel=5e-3) and copper.density == pytest.approx(3.12e8, rel=5e-3)
    assert copper.field < mg.RELATIVISTIC_DEGENERACY_FIELD < iron.field
    assert copper.confidence is Confidence.REPRESENTATIVE and iron.confidence is Confidence.HANDBOOK
    aluminium = mg.electron_capture_threshold("27Al")
    assert aluminium.field == pytest.approx(6.14e9, rel=5e-3)
    # every recorded threshold lies below neutron drip
    assert all(mg.electron_capture_threshold(k).density < mg.NEUTRON_DRIP_DENSITY for k in mg.NUCLIDES)
    for bad in ("57Fe", "iron", 56, None):
        with pytest.raises(InputError):
            mg.electron_capture_threshold(bad)


def test_composition_of_registry_materials():
    fe = mg.composition("iron_armco")
    assert (fe.charge, fe.nuclide, fe.element) == (26, "56Fe", "Fe") and fe.mu_e == pytest.approx(55.845 / 26)
    assert mg.composition("steel_4340").nuclide == "56Fe"              # alloys use their base element
    cu = mg.composition("cu")
    assert cu.material == "copper_ofhc" and cu.charge == 29 and cu.nuclide == "63Cu"
    w = mg.composition("tungsten")
    assert w.charge == 74 and w.nuclide is None and w.mu_e == pytest.approx(183.84 / 74)
    wc = mg.composition("tungsten_carbide_co")
    assert wc.charge is None and wc.mass_number is None and wc.mu_e == 2.0 and wc.nuclide is None
    assert mg.composition("diamond").nuclide == "12C"
    for key, element in mg.ELEMENT_NUCLIDE.items():
        assert mg.NUCLIDES[element].charge == mg.ELEMENTS[key][0]
    for key, n in mg.NUCLIDES.items():
        assert n.key == key and n.mu_e == n.mass_number / n.charge and n.threshold_energy > 0
    with pytest.raises(UnknownMaterialError):
        mg.composition("unobtainium")


# ------------------------------------------------------------------ matter threaded by the field
def test_atoms_become_needles_above_the_atomic_unit_of_field():
    weak = mg.atom_shape(100.0)
    assert weak.aspect_ratio == 1.0 and weak.transverse_radius == mg.BOHR_RADIUS == weak.length
    assert mg.atom_shape(mg.ATOMIC_FIELD).aspect_ratio == 1.0
    strong = mg.atom_shape(1e10)
    assert strong.transverse_radius == pytest.approx(mg.magnetic_length(1e10), rel=1e-8)
    assert strong.length == pytest.approx(mg.BOHR_RADIUS / math.log(strong.atomic_field_ratio))
    assert strong.aspect_ratio == pytest.approx(19.35, rel=1e-3)
    assert mg.atom_shape(1e8).aspect_ratio == pytest.approx(3.408, rel=1e-3)
    assert mg.atom_shape(1e11).aspect_ratio == pytest.approx(50.3, rel=1e-2)
    ratios = [mg.atom_shape(b).aspect_ratio for b in (1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11)]
    assert all(b >= a for a, b in zip(ratios, ratios[1:]))
    assert all(mg.atom_shape(b).transverse_radius <= mg.BOHR_RADIUS for b in (1.0, 1e6, 1e11))


def test_hydrogen_binding_and_condensed_matter():
    assert mg.hydrogen_binding_energy(1e8) / EV == pytest.approx(159.5, rel=1e-3)        # ~160 eV at 1e12 G
    assert mg.hydrogen_binding_energy(1e10) / EV == pytest.approx(494.6, rel=1e-3)
    assert mg.hydrogen_binding_energy(1e10) == pytest.approx(0.16 * math.log(1e10 / mg.ATOMIC_FIELD) ** 2 * mg.HARTREE)
    assert mg.hydrogen_binding_energy(1e6) is None and mg.hydrogen_binding_energy(100.0) is None
    assert mg.hydrogen_binding_energy(mg.HYDROGEN_ASYMPTOTIC_MIN * mg.ATOMIC_FIELD * 1.0001) is not None
    assert mg.hydrogen_binding_energy(1e11) > mg.hydrogen_binding_energy(1e10) > 13.6 * EV
    # magnetically condensed matter (Lai 2001): 561 g/cm^3 A Z^-3/5 B12^6/5
    assert mg.condensed_density(1e8, 1, 1) == pytest.approx(561e3)
    assert mg.condensed_density(1e10, 56, 26) == pytest.approx(1.117e9, rel=1e-3)
    assert mg.condensed_density(1e11, 56, 26) / mg.condensed_density(1e10, 56, 26) == pytest.approx(10 ** 1.2)
    assert mg.condensed_density(1e10, 56, 26) / materials.get("iron_armco").density == pytest.approx(1.42e5, rel=1e-2)
    assert mg.condensed_density_valid(1e10, 26) and not mg.condensed_density_valid(1e7, 26)
    assert mg.condensed_density_valid(26 ** (4 / 3) * mg.ATOMIC_FIELD * 1.001, 26)
    assert mg.condensed_density_valid(mg.ATOMIC_FIELD, 1)
    # threaded matter is far less dense than matter crushed by the same field acting as a piston
    assert mg.condensed_density(1e10, 56, 26) < 1e-2 * mg.piston_density(1e10, 56 / 26, 26)
    for call in (lambda: mg.condensed_density(1e10, 0.0, 26), lambda: mg.condensed_density(1e10, 56, 26.5),
                 lambda: mg.condensed_density_valid(1e10, 0), lambda: mg.hydrogen_binding_energy("x")):
        with pytest.raises(InputError):
            call()


def test_synchrotron_cooling_time():
    assert mg.SYNCHROTRON_COEFFICIENT == pytest.approx(5.1587, rel=1e-4)
    assert mg.synchrotron_cooling_time(1.0) == pytest.approx(5.1587, rel=1e-4)
    assert mg.synchrotron_cooling_time(100.0) == pytest.approx(5.159e-4, rel=1e-3)
    assert mg.synchrotron_cooling_time(1e10) == pytest.approx(5.159e-20, rel=1e-3)       # Landau ground state at once
    assert mg.synchrotron_cooling_time(1e10, gamma=10.0) == pytest.approx(5.159e-21, rel=1e-3)
    for call in (lambda: mg.synchrotron_cooling_time(0.0), lambda: mg.synchrotron_cooling_time(1.0, 0.5)):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ the low-field side
def test_skin_boiling_and_vaporisation_continue_the_eddy_module():
    cu = materials.get("copper_ofhc")
    boil = mg.skin_boiling_field(cu)
    assert boil == pytest.approx(148.4, rel=1e-3)                       # Herlach: copper boils near 150 T
    assert abs(boil - 150.0) < 5.0
    assert boil == pytest.approx(math.sqrt(2 * MU0 * cu.density * cu.specific_heat * (2835.0 - 293.15)))
    assert mg.skin_boiling_field(cu, include_latent=True) == pytest.approx(163.5, rel=1e-3)
    assert mg.skin_vaporisation_field(cu) == pytest.approx(365.0, rel=1e-3)
    with_latent = mg.skin_boiling_field(cu, include_latent=True)
    assert eddy.skin_melting_field(cu) < boil < with_latent < mg.skin_vaporisation_field(cu)
    # same energy balance as eddy.skin_temperature_rise: at the boiling field the skin gains T_b - T_0
    assert eddy.skin_temperature_rise(cu, boil) == pytest.approx(2835.0 - 293.15, rel=1e-9)
    assert mg.skin_boiling_field(cu, kappa=4.0) == pytest.approx(0.5 * boil)
    assert mg.skin_boiling_field(cu, initial_temperature=77.0) > boil
    for key, b_boil, b_vap in (("aluminum_6061", 123.3, 303.0), ("iron_armco", 159.0, 388.1),
                               ("tungsten", 189.4, 510.0)):
        assert mg.skin_boiling_field(key) == pytest.approx(b_boil, rel=2e-3)
        assert mg.skin_vaporisation_field(key) == pytest.approx(b_vap, rel=2e-3)
    assert mg.skin_boiling_field("steel_1018") == pytest.approx(
        math.sqrt(2 * MU0 * 7870 * 486.0 * (3134.0 - 293.15)))          # alloys use their base element
    for call in (lambda: mg.skin_boiling_field("titanium_cp2"), lambda: mg.skin_vaporisation_field("diamond"),
                 lambda: mg.skin_boiling_field(cu, initial_temperature=2000.0),
                 lambda: mg.skin_boiling_field(cu, kappa=0.0),
                 lambda: mg.skin_vaporisation_field(cu, initial_temperature=-1.0)):
        with pytest.raises(InputError):
            call()


def test_yield_and_bulk_modulus_fields():
    cu = materials.get("copper_ofhc")
    assert mg.yield_field(cu) == pytest.approx(13.26, rel=1e-3)
    assert mg.yield_field("iron_armco") == pytest.approx(20.97, rel=1e-3)
    assert mg.yield_field("tungsten") == pytest.approx(57.16, rel=1e-3)
    assert magnetic_pressure(mg.yield_field(cu)) == pytest.approx(cu.yield_strength)
    assert mg.bulk_modulus_field(cu) == pytest.approx(553.4, rel=1e-3)
    assert mg.bulk_modulus_field("iron_armco") == pytest.approx(648.7, rel=1e-3)
    assert mg.bulk_modulus_field("tungsten") == pytest.approx(884.6, rel=1e-3)
    assert magnetic_pressure(mg.bulk_modulus_field("diamond")) == pytest.approx(443e9)
    # megagauss flux compression (1200-2800 T) works above the bulk modulus of every metal here
    assert all(mg.bulk_modulus_field(k) < 1200.0 for k in ("copper_ofhc", "iron_armco", "tungsten", "aluminum_6061"))
    with pytest.raises(InputError):
        mg.yield_field("diamond")


# ------------------------------------------------------------------ regimes
def test_regime_thresholds_for_iron_and_copper():
    iron = mg.regime_thresholds("iron_armco")
    assert [t.key for t in iron] == [
        "plastic_flow", "skin_melting", "skin_boiling", "skin_vaporised", "bulk_modulus", "pressure_ionisation",
        "atomic_field", "relativistic_degeneracy", "qed_critical", "electron_capture", "neutron_drip"]
    fields = [t.field for t in iron]
    assert fields == sorted(fields) and all(t.crossed is None for t in iron)
    by = {t.key: t for t in iron}
    assert by["skin_melting"].field == pytest.approx(eddy.skin_melting_field("iron_armco"))
    assert by["skin_melting"].field == pytest.approx(116.2, rel=1e-3)
    assert by["atomic_field"].field == mg.ATOMIC_FIELD and by["qed_critical"].field == mg.QED_CRITICAL_FIELD
    assert by["electron_capture"].field == pytest.approx(1.127e10, rel=2e-3)
    assert "56Fe" in by["electron_capture"].state and "56Mn" in by["electron_capture"].state
    assert by["neutron_drip"].field == mg.NEUTRON_DRIP_FIELD
    assert all(t.pressure == pytest.approx(magnetic_pressure(t.field)) and t.label and t.state for t in iron)
    assert {t.kind for t in iron} == {"strength", "heating", "pressure", "field"}
    assert [t.key for t in iron if t.kind == "field"] == ["atomic_field", "qed_critical"]
    # copper captures electrons before its electron gas is even relativistic
    copper = [t.key for t in mg.regime_thresholds("copper_ofhc")]
    assert copper.index("electron_capture") < copper.index("relativistic_degeneracy") < copper.index("qed_critical")
    # materials without data simply lack the rung
    assert "electron_capture" not in [t.key for t in mg.regime_thresholds("tungsten")]
    assert "skin_boiling" not in [t.key for t in mg.regime_thresholds("gamma_tial")]
    diamond = [t.key for t in mg.regime_thresholds("diamond")]
    assert diamond[0] == "bulk_modulus" and "plastic_flow" not in diamond and "skin_melting" not in diamond
    marked = mg.regime_thresholds("iron_armco", 1e10)
    assert [t.crossed for t in marked] == [True] * 9 + [False, False]


def test_classify_regime_walks_up_the_ladder():
    def at(field, material="iron_armco"):
        return mg.classify_regime(material, field)

    quiet = at(10.0)
    assert (quiet.key, quiet.tier, quiet.crossed, quiet.next_key) == ("elastic", "solid", (), "plastic_flow")
    concept = at(100.27)                                              # the concept note's 4 GPa
    assert (concept.key, concept.tier, concept.next_key) == ("plastic_flow", "solid", "skin_melting")
    assert concept.piston_density is None and concept.compression is None
    assert at(150.0).tier == "liquid_skin" and at(150.0).key == "skin_melting"
    assert at(500.0).key == "skin_vaporised" and at(500.0).tier == "liquid_skin"
    megagauss = at(2800.0)
    assert (megagauss.key, megagauss.tier) == ("bulk_modulus", "megabar")
    assert (at(1e4).key, at(1e4).tier) == ("pressure_ionisation", "plasma")
    landau = at(1e6)
    assert (landau.key, landau.matter_key, landau.tier) == ("atomic_field", "pressure_ionisation", "landau")
    assert "needles" in landau.state and "pressure-ionised" in landau.state
    magnetar = at(1e10)
    assert (magnetar.key, magnetar.matter_key) == ("qed_critical", "relativistic_degeneracy")
    assert magnetar.tier == "degenerate"
    assert "white-dwarf matter" in magnetar.state and "birefringent" in magnetar.state
    assert magnetar.next_key == "electron_capture" and magnetar.next_field == pytest.approx(1.127e10, rel=2e-3)
    assert magnetar.piston_density == pytest.approx(9.54e11, rel=2e-3)
    assert magnetar.compression == pytest.approx(magnetar.piston_density / 7870.0)
    assert magnetar.pressure == pytest.approx(magnetic_pressure(1e10)) and len(magnetar.thresholds) == 11
    strongest = at(1e11)
    assert strongest.key == strongest.matter_key == "electron_capture" and "transmuted" in strongest.state
    top = at(1e12)
    assert top.key == "neutron_drip" and top.next_key is None and top.next_field is None and top.piston_density is None
    # copper at 1e8 T is already being transmuted, and the state keeps saying so further up
    assert at(1e8, "copper_ofhc").matter_key == "electron_capture" and "63Ni" in at(1e8, "copper_ofhc").state
    hot_copper = at(1e10, "copper_ofhc")
    assert hot_copper.matter_key == "relativistic_degeneracy"
    assert all(word in hot_copper.state for word in ("white-dwarf matter", "63Ni", "birefringent"))
    assert "56Mn" not in magnetar.state                       # iron at 1e10 T has not reached its threshold
    # a non-conductor has no eddy-current skin tier
    assert at(500.0, "diamond").tier == "solid" and at(2000.0, "diamond").tier == "megabar"
    # tiers never go backwards as the field rises
    for key in ("iron_armco", "copper_ofhc", "tungsten", "diamond", "gamma_tial"):
        order = [mg.TIERS.index(at(b, key).tier) for b in mg.decade_fields()]
        assert order == sorted(order) and order[0] == 0 and order[-1] == len(mg.TIERS) - 1, key
    assert at(-1e10).key == magnetar.key
    with pytest.raises(UnknownMaterialError):
        at(1e10, "unobtainium")
    with pytest.raises(InputError):
        at(float("nan"))


def test_process_outcomes_cover_the_six_applications():
    assert mg.APPLICATIONS == ("forging", "refractory_forming", "welding", "synthesis", "densification", "containment")
    assert mg.TIERS == ("solid", "liquid_skin", "megabar", "plasma", "landau", "degenerate")
    ordinary = mg.process_outcomes("iron_armco", 50.0)
    assert tuple(ordinary) == mg.APPLICATIONS
    for name, module in (("forging", "magform.eddy"), ("refractory_forming", "magform.plasticity"),
                         ("welding", "magform.welding"), ("synthesis", "magform.synthesis"),
                         ("densification", "magform.porosity"), ("containment", "magform.containment")):
        assert module in ordinary[name]                                # below skin melting: the rest of the package
    exotic = mg.process_outcomes("iron_armco", 1e10)
    assert "Nuclear" in exotic["synthesis"] and "neutron star" in exotic["containment"]
    assert "white-dwarf" in exotic["forging"] and all(v.endswith(".") for v in exotic.values())
    megagauss = mg.process_outcomes("copper_ofhc", 2800.0)
    assert "bulk modulus" in megagauss["forging"] and "vaporise" in megagauss["welding"]
    seen = set()
    for b in mg.decade_fields(per_decade=2):
        seen.add(tuple(mg.process_outcomes("iron_armco", b).values()))
    assert len(seen) == len(mg.TIERS)                                  # one distinct answer set per tier


# ------------------------------------------------------------------ the assumed source
def test_source_energy_budget():
    u = magnetic_pressure(1e10)
    assert SOURCE.volume() == pytest.approx(4.18879e-6, rel=1e-6)
    assert SOURCE.interior_energy() == pytest.approx(u * SOURCE.volume())
    assert SOURCE.interior_energy() == pytest.approx(1.6667e20, rel=1e-4)
    assert SOURCE.exterior_energy() == pytest.approx(0.5 * SOURCE.interior_energy(), rel=1e-12)
    assert SOURCE.energy() == pytest.approx(math.pi * 0.01 ** 3 * 1e20 / MU0)
    assert SOURCE.energy() == pytest.approx(2.5e20, rel=1e-4)
    # exterior energy is that of the dipole outside the sphere, mu0 m^2 / (12 pi R^3)
    m = SOURCE.dipole_moment()
    assert m == pytest.approx(2.0 * math.pi * 1e-6 * 1e10 / MU0)
    assert SOURCE.exterior_energy() == pytest.approx(MU0 * m * m / (12.0 * math.pi * 1e-6), rel=1e-12)
    assert SOURCE.megatons() == pytest.approx(5.975e4, rel=1e-3)                 # sixty thousand megatons
    assert SOURCE.mass_equivalent() == pytest.approx(2782.0, rel=1e-3)           # 2.8 tonnes of mass-energy
    assert SOURCE.confinement_time() == pytest.approx(0.01 / C_LIGHT)
    assert SOURCE.confinement_time() == pytest.approx(3.336e-11, rel=1e-3)
    assert SOURCE.assembly_power() == pytest.approx(7.495e30, rel=1e-3)
    assert SOURCE.assembly_power() / mg.SOLAR_LUMINOSITY == pytest.approx(1.958e4, rel=1e-3)
    assert SOURCE.axial_field(0.01) == pytest.approx(1e10) and SOURCE.axial_field(0.02) == pytest.approx(1.25e9)
    assert SOURCE.pulse_power() is None and SOURCE.held() is False
    # energy goes as B^2 R^3
    assert mg.MagnetarSource(2e10, 0.01).energy() == pytest.approx(4.0 * SOURCE.energy())
    assert mg.MagnetarSource(1e10, 1e-3).megatons() == pytest.approx(59.75, rel=1e-3)
    assert mg.MagnetarSource(1e10, 1e-6).energy() == pytest.approx(2.5e8, rel=1e-4)
    held = mg.MagnetarSource(1e10, 1e-3, 1e-3)
    assert held.held() is True and held.pulse_power() == pytest.approx(held.energy() / 1e-3)
    assert mg.MagnetarSource(1e10, 1e-3, 1e-13).held() is False
    assert mg.MagnetarSource(-1e10, 0.01).field == 1e10
    for call in (lambda: mg.MagnetarSource(0.0, 0.01), lambda: mg.MagnetarSource(1e10, 0.0),
                 lambda: mg.MagnetarSource(1e10, 0.01, 0.0), lambda: mg.MagnetarSource(float("inf"), 0.01),
                 lambda: SOURCE.axial_field(0.001)):
        with pytest.raises(InputError):
            call()


def test_nothing_material_holds_the_field():
    fibre = SOURCE.virial_mass()
    assert fibre == pytest.approx(6.393e13, rel=1e-3)                            # of the best commercial carbon fibre
    t1100 = materials.get("carbon_fibre_t1100g")
    assert fibre == pytest.approx(t1100.density * SOURCE.energy() / t1100.ultimate_strength)
    assert SOURCE.virial_mass("graphene_intrinsic") == pytest.approx(4.346e12, rel=1e-3)
    assert mg.MagnetarSource(1e10, 1e-6).virial_mass() == pytest.approx(63.93, rel=1e-3)     # kg for a micrometre
    assert SOURCE.virial_mass("maraging_c350") > fibre                           # falls back to yield strength
    # about 10 kg of fibre per cubic micrometre of field, as the module docstring says
    assert t1100.density * mg.field_energy(1e10, 1e-18) / t1100.ultimate_strength == pytest.approx(10.2, rel=1e-2)
    with pytest.raises(InputError):
        SOURCE.virial_mass("hbn")


def test_chandrasekhar_fermi_limit_and_gravitational_holder():
    mass, radius = mg.CANONICAL_NEUTRON_STAR
    limit = mg.chandrasekhar_fermi_field(mass, radius)
    assert limit == pytest.approx(9.476e13, rel=1e-3)                            # ~1e18 G
    # definition: magnetic energy inside = gravitational binding energy of a uniform sphere
    e_mag = limit ** 2 / (2 * MU0) * (4.0 / 3.0) * math.pi * radius ** 3
    e_grav = 0.6 * mg.GRAVITATIONAL_CONSTANT * mass ** 2 / radius
    assert e_mag == pytest.approx(e_grav, rel=1e-12)
    assert limit > 1e3 * 1e10 and limit > 1e11                                   # room above every magnetar
    assert mg.chandrasekhar_fermi_field(mg.SOLAR_MASS, 6.957e8) == pytest.approx(2.014e4, rel=1e-3)
    assert mg.chandrasekhar_fermi_field(2 * mass, radius) == pytest.approx(2 * limit)
    assert mg.chandrasekhar_fermi_field(mass, 2 * radius) == pytest.approx(limit / 4)
    h = mg.gravitational_holder(1e10)
    assert h.virial_radius == pytest.approx(2.118, rel=1e-3) and h.virial_mass == pytest.approx(9.148e18, rel=1e-3)
    assert mg.chandrasekhar_fermi_field(h.virial_mass, h.virial_radius) == pytest.approx(1e10, rel=1e-12)
    assert h.self_bound is False and h.holder_mass == mg.MIN_NEUTRON_STAR_MASS
    assert h.holder_solar_masses == pytest.approx(0.1)
    assert h.neutron_star_limit == pytest.approx(limit)
    big = mg.gravitational_holder(9.4e13)
    assert big.self_bound is True and big.holder_mass == big.virial_mass
    assert big.holder_solar_masses == pytest.approx(3.82, rel=1e-2)
    for call in (lambda: mg.chandrasekhar_fermi_field(0.0, 1.0), lambda: mg.chandrasekhar_fermi_field(1.0, -1.0),
                 lambda: mg.gravitational_holder(0.0), lambda: mg.gravitational_holder(1e10, 0.0)):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ the space setting
def test_dipole_standoff_radii():
    assert mg.dipole_axial_field(1e10, 0.01, 1.0) == pytest.approx(1e4)
    assert mg.dipole_axial_field(1e10, 0.01, 10.0) == pytest.approx(10.0)
    radii = mg.standoff_radii(1e10, 0.01)
    assert list(radii) == [t.key for t in mg.FIELD_THRESHOLDS]
    expected = {"earth_field": 584.8, "pacemaker_line": 271.4, "public_limit": 29.24, "occupational_limit": 17.10,
                "limb_limit": 10.77, "atomic_field": 0.3491}
    for key, value in expected.items():
        assert radii[key] == pytest.approx(value, rel=1e-3), key
    assert radii == SOURCE.standoff()
    values = list(radii.values())
    assert all(b < a for a, b in zip(values, values[1:]))                        # stronger threshold, closer in
    for t in mg.FIELD_THRESHOLDS:
        assert mg.dipole_axial_field(1e10, 0.01, radii[t.key]) == pytest.approx(t.field, rel=1e-9)
    # a real magnetar is lethal by chemistry out to ~750 km
    assert mg.standoff_radius(1e11, 1e4, mg.ATOMIC_FIELD) == pytest.approx(7.52e5, rel=1e-3)
    # a weaker source never reaches the threshold outside itself
    assert mg.standoff_radius(1.0, 0.05, 8.0) == 0.05
    assert mg.standoff_radius(1e10, 0.02, 8.0) == pytest.approx(2.0 * radii["limb_limit"])
    for call in (lambda: mg.dipole_axial_field(1e10, 0.01, 0.005), lambda: mg.standoff_radius(1e10, 0.0, 1.0),
                 lambda: mg.standoff_radius(1e10, 0.01, 0.0), lambda: mg.dipole_axial_field(1e10, 0.01, -1.0)):
        with pytest.raises(InputError):
            call()


def test_gradient_force_on_water():
    gradient = mg.levitation_gradient()
    assert gradient == pytest.approx(MU0 * 1000.0 * G_ACCEL / 9.04e-6)
    assert gradient == pytest.approx(1363.2, rel=1e-3)
    assert gradient == pytest.approx(mg.WATER_LEVITATION_GRADIENT, rel=0.03)     # Berry & Geim's ~1400 T^2/m
    r_lev = mg.levitation_radius(1e10, 0.01)
    assert r_lev == pytest.approx(5.775, rel=1e-3)
    b = mg.dipole_axial_field(1e10, 0.01, r_lev)
    assert 3.0 * b * b / r_lev == pytest.approx(mg.WATER_LEVITATION_GRADIENT, rel=1e-9)      # B dB/dr on the axis
    # at the radius where B dB/dr equals the exact levitation gradient the push equals Earth gravity
    r_exact = mg.levitation_radius(1e10, 0.01, gradient)
    assert mg.diamagnetic_acceleration(1e10, 0.01, r_exact) == pytest.approx(G_ACCEL, rel=1e-9)
    assert mg.diamagnetic_acceleration(1e10, 0.01, 1.0) == pytest.approx(2.158e6, rel=1e-3)
    assert mg.diamagnetic_acceleration(1e10, 0.01, 2.0) == pytest.approx(
        mg.diamagnetic_acceleration(1e10, 0.01, 1.0) / 2 ** 7)                   # B dB/dr ~ r^-7
    assert mg.levitation_radius(1.0, 0.05) == 0.05 and mg.levitation_radius(0.0, 0.05) == 0.05
    for call in (lambda: mg.levitation_gradient(0.0), lambda: mg.levitation_radius(1e10, 0.01, 0.0),
                 lambda: mg.diamagnetic_acceleration(1e10, 0.01, 1.0, density=0.0)):
        with pytest.raises(InputError):
            call()


def test_energy_release_sets_the_real_standoff():
    energy = SOURCE.energy()
    damage = mg.fluence_radius(energy)
    assert damage == pytest.approx(4.460e6, rel=1e-3)                            # thousands of kilometres
    assert mg.radiant_fluence(energy, damage) == pytest.approx(mg.DAMAGE_FLUENCE, rel=1e-12)
    sun_second = mg.fluence_radius(energy, mg.SOLAR_CONSTANT)
    assert sun_second == pytest.approx(1.209e8, rel=1e-3)
    assert mg.radiant_fluence(energy, 2.0 * damage) == pytest.approx(mg.DAMAGE_FLUENCE / 4.0)
    assert damage > 1e4 * SOURCE.standoff()["pacemaker_line"]                    # radiation, not stray field, sets it
    assert mg.fluence_radius(mg.MagnetarSource(1e10, 1e-3).energy()) == pytest.approx(1.410e5, rel=1e-3)
    assert mg.fluence_radius(0.0) == 0.0
    # banking one cubic millimetre of field with a square kilometre of collector takes most of a year
    assert mg.solar_collection_time(mg.field_energy(1e10, 1e-9)) / YEAR == pytest.approx(0.9264, rel=1e-3)
    assert mg.solar_collection_time(1e9, 1.0, 0.5) == pytest.approx(1e9 / (1361.0 * 0.5))
    assert mg.solar_collection_time(SOURCE.energy()) / YEAR == pytest.approx(5.82e3, rel=1e-2)
    for call in (lambda: mg.radiant_fluence(1.0, 0.0), lambda: mg.fluence_radius(1.0, 0.0),
                 lambda: mg.solar_collection_time(1.0, 0.0), lambda: mg.solar_collection_time(1.0, 1.0, 1.5),
                 lambda: mg.solar_collection_time(-1.0)):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ the assessment
def test_assessment_is_assumed_not_assessed():
    a = mg.assess_magnetar_mode(1e10, 0.01)
    assert isinstance(a, Assessment)
    assert a.status is Status.ASSUMED and a.status.assessed is False and a.margin is None
    assert worst(a.status, Status.EXTRAPOLATION) is Status.EXTRAPOLATION          # a premise never wins a roll-up
    assert a.premise == mg.PREMISE == a.notes[0] and "assumed, not assessed" in a.premise
    assert any("flux compression" in n and "heavy-ion" in n and "laser" in n for n in a.notes)
    assert (a.material, a.field, a.radius, a.pulse_duration) == ("iron_armco", 1e10, 0.01, None)
    assert a.pressure == a.energy_density == pytest.approx(3.979e25, rel=1e-3)
    assert a.mass_equivalent_density == pytest.approx(4.427e8, rel=1e-3)
    assert a.atomic_field_ratio == pytest.approx(4.254e4, rel=1e-3)
    assert a.qed_field_ratio == pytest.approx(2.266, rel=1e-3)
    assert a.cyclotron_energy == mg.cyclotron_energy(1e10) and a.magnetic_length == mg.magnetic_length(1e10)
    assert a.landau_excitation_energy == mg.landau_excitation_energy(1e10)
    assert a.synchrotron_cooling_time == pytest.approx(5.159e-20, rel=1e-3)
    assert a.field_energy == pytest.approx(SOURCE.energy())
    assert a.interior_energy == pytest.approx(SOURCE.interior_energy())
    assert a.field_energy_megatons == pytest.approx(5.975e4, rel=1e-3)
    assert a.mass_equivalent == pytest.approx(2782.0, rel=1e-3)
    assert a.confinement_time == pytest.approx(3.336e-11, rel=1e-3)
    assert a.required_power == pytest.approx(7.495e30, rel=1e-3)
    assert a.pulse_power is None and a.held is False
    assert a.virial_mass_fibre == pytest.approx(6.393e13, rel=1e-3) and a.virial_mass_graphene < a.virial_mass_fibre
    assert a.holder.self_bound is False and a.holder.holder_solar_masses == pytest.approx(0.1)
    assert a.solar_collection_time == pytest.approx(mg.solar_collection_time(a.field_energy))
    assert a.regime.tier == "degenerate" and a.state == a.regime.state and a.thresholds_crossed == a.regime.crossed
    assert len(a.thresholds_crossed) == 9 and "electron_capture" not in a.thresholds_crossed
    assert a.compressed_density == pytest.approx(9.54e11, rel=2e-3)
    assert a.condensed_density == pytest.approx(mg.condensed_density(1e10, 55.845, 26))
    assert a.hydrogen_binding_energy / EV == pytest.approx(494.6, rel=1e-3)
    assert a.atom_aspect_ratio == pytest.approx(19.35, rel=1e-3)
    assert a.standoff == SOURCE.standoff() and a.levitation_radius == pytest.approx(5.775, rel=1e-3)
    assert a.damage_radius == pytest.approx(4.460e6, rel=1e-3)
    assert a.solar_second_radius == pytest.approx(1.209e8, rel=1e-3)
    assert tuple(a.processes) == mg.APPLICATIONS and a.processes == mg.process_outcomes("iron_armco", 1e10)
    # the consequences that would stop a real mission are in the warnings
    text = " | ".join(a.warnings)
    for phrase in ("non-destructive record", "no workpiece survives as a solid", "assembling the field",
                   "space-based", "0.5 mT line"):
        assert phrase in text, phrase


def test_assessment_gaps_to_the_record_book():
    a = mg.assess_magnetar_mode(1e10, 0.01)
    assert set(a.gaps) == {"field_over_nondestructive_record", "field_over_destructive_record",
                           "pressure_over_concept_note", "energy_over_largest_pulsed_store",
                           "power_over_pulsed_record", "power_over_largest_laser"}
    assert a.gaps["field_over_nondestructive_record"] == pytest.approx(1e10 / 100.75)
    assert a.gaps["field_over_destructive_record"] == pytest.approx(1e10 / 2800.0)
    assert a.gaps["pressure_over_concept_note"] == pytest.approx(9.947e15, rel=1e-3)
    assert a.gaps["energy_over_largest_pulsed_store"] == pytest.approx(2.083e11, rel=1e-3)
    assert a.gaps["power_over_pulsed_record"] == pytest.approx(a.required_power / benchmarks.PEAK_POWER_RECORD)
    assert a.gaps["power_over_largest_laser"] == pytest.approx(7.495e14, rel=1e-3)
    assert all(g > 1e6 for g in a.gaps.values())
    # the strongest laboratory field in the same volume: tens of tesla for 10 mm, megagauss for 1 mm,
    # a laser focus for 1 um -- always orders of magnitude short
    assert (a.laboratory_record, a.laboratory_record_field) == ("pulsed_user_60t", 60.0)
    mm = mg.assess_magnetar_mode(1e10, 1e-3)
    assert mm.laboratory_record == "z_machine_surface" and mm.field_energy_megatons == pytest.approx(59.75, rel=1e-3)
    um = mg.assess_magnetar_mode(1e10, 1e-6)
    assert um.laboratory_record == "laser_optical_field" and um.field_energy == pytest.approx(2.5e8, rel=1e-3)
    assert um.virial_mass_fibre == pytest.approx(63.93, rel=1e-3)
    assert um.required_power == pytest.approx(7.495e22, rel=1e-3) and um.gaps["power_over_largest_laser"] > 1e6
    assert not any("space-based" in w for w in um.warnings)                      # a 250 MJ shot is not megatons
    assert any("times that" in n for n in um.notes)


def test_assessment_variants():
    held = mg.assess_magnetar_mode(1e10, 1e-3, pulse_duration=1e-3)
    assert held.held is True and held.pulse_power == pytest.approx(held.field_energy / 1e-3)
    assert held.laboratory_record == "lanl_100t_nondestructive"                  # the strongest millisecond field
    assert any("something must hold the field" in w for w in held.warnings)
    strongest = mg.assess_magnetar_mode(1e11, 0.01)
    assert strongest.regime.key == "electron_capture" and strongest.field_energy == pytest.approx(100 * SOURCE.energy())
    assert "electron_capture" in strongest.thresholds_crossed
    beyond = mg.assess_magnetar_mode(1e12, 0.01)
    assert beyond.compressed_density is None and beyond.regime.key == "neutron_drip"
    # an ordinary field is still labelled assumed, but none of the exotic warnings fire
    ordinary = mg.assess_magnetar_mode(50.0, 0.05, "copper_ofhc")
    assert ordinary.status is Status.ASSUMED and ordinary.regime.tier == "solid"
    assert ordinary.gaps["field_over_nondestructive_record"] < 1.0 and ordinary.laboratory_record is None
    assert not any("assembling" in w or "no workpiece" in w or "space-based" in w for w in ordinary.warnings)
    assert any("within the non-destructive record" in n for n in ordinary.notes)
    assert ordinary.compressed_density is None and ordinary.condensed_density is None
    assert ordinary.hydrogen_binding_energy is None and ordinary.atom_aspect_ratio == 1.0
    # materials: key, alias and object agree; compounds work without a nuclide
    by_alias = mg.assess_magnetar_mode(1e10, 0.01, "fe")
    by_object = mg.assess_magnetar_mode(1e10, 0.01, materials.get("iron_armco"))
    assert by_alias.json() == by_object.json() == mg.assess_magnetar_mode(1e10, 0.01).json()
    carbide = mg.assess_magnetar_mode(1e10, 0.01, "tungsten_carbide_co")
    assert carbide.condensed_density is None and carbide.compressed_density == pytest.approx(mg.piston_density(1e10))
    for call in (lambda: mg.assess_magnetar_mode(0.0, 0.01), lambda: mg.assess_magnetar_mode(1e10, -1.0),
                 lambda: mg.assess_magnetar_mode(1e10, 0.01, pulse_duration=0.0),
                 lambda: mg.assess_magnetar_mode(float("nan"), 0.01)):
        with pytest.raises(InputError):
            call()
    with pytest.raises(UnknownMaterialError):
        mg.assess_magnetar_mode(1e10, 0.01, "unobtainium")


def test_a_real_magnetar_is_consistent_with_observation():
    # treat an actual magnetar (1e11 T, 10 km) as the source: the numbers should look like astrophysics
    star = mg.assess_magnetar_mode(1e11, 1.0e4)
    assert star.field_energy == pytest.approx(2.5e40, rel=1e-3)
    flare = next(s.energy for s in mg.ENERGY_SCALES if s.key == "magnetar_giant_flare")
    assert 5.0 < star.field_energy / flare < 50.0             # a giant flare spends a few per cent of the field
    assert star.standoff["atomic_field"] == pytest.approx(7.52e5, rel=1e-3)      # chemistry stops within ~750 km
    assert star.laboratory_record is None                      # nothing on Earth fills a 10 km sphere
    assert star.holder.neutron_star_limit > 1e11               # and a neutron star can hold it by gravity
    assert star.confinement_time == pytest.approx(1.0e4 / C_LIGHT)
    assert star.regime.key == "electron_capture" and star.status is Status.ASSUMED


# ------------------------------------------------------------------ sweep
def test_field_sweep_columns():
    fields = mg.decade_fields()
    assert len(fields) == 41 and fields[0] == pytest.approx(10.0) and fields[-1] == pytest.approx(1e11)
    assert all(b / a == pytest.approx(10 ** 0.25) for a, b in zip(fields, fields[1:]))
    assert len(mg.decade_fields(1.0, 1e3, 1)) == 4
    sweep = mg.field_sweep()
    assert sweep.material == "iron_armco" and sweep.field == fields
    columns = (sweep.pressure, sweep.mass_equivalent_density, sweep.cyclotron_energy, sweep.magnetic_length,
               sweep.atomic_field_ratio, sweep.piston_density, sweep.condensed_density, sweep.regime,
               sweep.matter_regime, sweep.tier)
    assert all(len(c) == 41 for c in columns)
    assert sweep.pressure == tuple(magnetic_pressure(b) for b in fields)
    assert sweep.regime[0] == "elastic" and sweep.regime[-1] == "electron_capture"
    assert sweep.tier[0] == "solid" and sweep.tier[-1] == "degenerate" and set(sweep.tier) == set(mg.TIERS)
    floor = mg.eos_floor_field(26)
    for b, rho, cond in zip(sweep.field, sweep.piston_density, sweep.condensed_density):
        assert (rho is None) == (b < floor)
        assert (cond is None) == (not mg.condensed_density_valid(b, 26))
    modelled = [rho for rho in sweep.piston_density if rho is not None]
    assert len(modelled) >= 15 and all(b > a for a, b in zip(modelled, modelled[1:]))
    custom = mg.field_sweep([100.0, 1e10], "copper_ofhc")
    assert custom.field == (100.0, 1e10) and custom.material == "copper_ofhc"
    assert custom.regime == ("skin_melting", "qed_critical") and custom.matter_regime[1] == "relativistic_degeneracy"
    for call in (lambda: mg.field_sweep([]), lambda: mg.field_sweep([0.0]), lambda: mg.decade_fields(10.0, 1.0),
                 lambda: mg.decade_fields(per_decade=0), lambda: mg.decade_fields(per_decade=2.5)):
        with pytest.raises(InputError):
            call()


# ------------------------------------------------------------------ serialisation and metadata
def test_results_round_trip_json_and_render_text():
    a = mg.assess_magnetar_mode(1e10, 0.01)
    data = json.loads(a.json())                                       # strict JSON: no inf/nan
    assert data["status"] == "assumed" and data["margin"] is None and data["pulse_duration"] is None
    assert data["field_energy_megatons"] == pytest.approx(a.field_energy_megatons)
    assert data["regime"]["tier"] == "degenerate" and len(data["regime"]["thresholds"]) == 11
    assert data["holder"]["self_bound"] is False and data["processes"]["synthesis"] == a.processes["synthesis"]
    assert data["gaps"]["field_over_destructive_record"] == pytest.approx(a.gaps["field_over_destructive_record"])
    text = a.text()
    assert text.startswith("Magnetar-mode thought experiment") and "status = assumed" in text
    assert "field_energy_megatons [Mt]" in text and "required_power [W]" in text and "processes:" in text
    for result, needle in ((SOURCE, "radius [m]"), (mg.classify_regime("iron_armco", 1e10), "piston_density [kg/m^3]"),
                           (mg.electron_capture_threshold("56Fe"), "threshold_energy [J]"),
                           (mg.piston_state(1e10, 2.154, 26), "fermi_energy [J]"),
                           (mg.atom_shape(1e10), "transverse_radius [m]"),
                           (mg.gravitational_holder(1e10), "virial_mass [kg]"),
                           (mg.EXTREME_FIELD_LADDER[-1], "duration [s]"), (mg.composition("iron_armco"), "mu_e"),
                           (mg.field_sweep([1e10]), "magnetic_length [m]")):
        assert needle in result.text(), needle
        assert isinstance(json.loads(result.json()), dict)
    assert json.loads(mg.piston_state(1e10).json())["lattice_fraction"] is None
    assert json.loads(mg.EXTREME_FIELD_LADDER[0].json())["confidence"] in {c.value for c in Confidence}
    with pytest.raises(Exception):
        a.field = 1.0  # frozen


def test_module_metadata():
    assert set(mg.__all__) <= set(dir(mg)) and len(set(mg.__all__)) == len(mg.__all__)
    assert len(mg.REFERENCES) >= 15
    assert all(isinstance(k, str) and isinstance(v, str) and len(v) > 20 for k, v in mg.REFERENCES.items())
    for name in mg.__all__:
        obj = getattr(mg, name)
        if callable(obj):
            assert obj.__doc__ and "Reference" in obj.__doc__, name
    assert mg.__doc__ and "thought experiment" in mg.__doc__.splitlines()[0]
    assert "ASSUMED" in mg.__doc__ and "not assessed" in mg.__doc__.lower()
    # handbook data used on the low-field side
    assert set(mg.VAPORISATION) == {"Cu", "Al", "Fe", "W"}
    for element, (t_boil, h_vap) in mg.VAPORISATION.items():
        assert 2000.0 < t_boil < 7000.0 and 3e6 < h_vap < 2e7, element
    assert mg.CONCEPT_PRESSURE == 4e9 and mg.SOLAR_CONSTANT == 1361.0 and mg.LASER_PEAK_POWER == 1e16
    assert [t.field for t in mg.FIELD_THRESHOLDS] == sorted(t.field for t in mg.FIELD_THRESHOLDS)
