"""The claims ledger: every claim of the concept note, quoted, tested and given a verdict.

The original concept note (``docs/archive/original-README.md``) makes twenty-eight
checkable claims and asks two questions.  This module holds each one verbatim
(:data:`CLAIMS`) and an evaluator that runs the relevant models and returns a
:class:`ClaimResult`: a :class:`~magform.status.Verdict`, a plain finding, the
numbers behind it and the functions that produced them.  Nothing in a result is
typed in: findings and evidence are assembled from model output at call time,
so the ledger cannot drift from the physics.

How verdicts are assigned
-------------------------
* Where a model returns a feasibility :class:`~magform.status.Status` for the
  claim's own conditions, the verdict is that status
  (:func:`~magform.status.verdict_from_status`).
* A statement of fact that the numbers show to be false is ``contradicted``.
* A statement that is true of today's practice is ``state_of_the_art``, with the
  conditions under which it is true (often far below the note's 63-100 T).
* Fracture toughness, fatigue life and service temperature are outside what the
  package computes; claims about them are ``not_modelled`` and say so.

The reference cases are those used throughout the documentation: the reference
press (100 mm bore, 0.5 m long, 0.2 m build) at 1.6 and 4 GPa, a 50 mm x 2 mm
copper tube, a 1 mm aluminium flyer over a 2 mm gap onto steel, a 2 mm copper
flyer over 10 mm onto graphite, cast gamma titanium aluminide with 0.5 %
porosity, and a 50 mm bore housing.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Callable, Iterable

from . import containment, eddy, fieldmap, microstructure, plasticity, porosity, pulse, synthesis, welding
from ._report import ResultMixin
from .benchmarks import (
    EMF_CONVENTIONAL_FIELD,
    PRESSURE_BENCHMARKS,
    PULSED_NONDESTRUCTIVE_BORE,
    PULSED_NONDESTRUCTIVE_RECORD,
    SC_CONDUCTOR_RECORD,
    SC_FIELD_RECORD,
    pulsed_field_status,
    where_does_it_sit,
)
from .constants import E_CHARGE, N_A, T_ROOM, ZERO_CELSIUS
from .errors import InputError
from .magnetics import field_for_pressure, magnetic_pressure, pressure_with_background
from .materials import as_material
from .press import PressDesign, design_press, reference_spec
from .status import Status, Verdict, best, verdict_from_status, worst

REFERENCES: dict[str, str] = {
    "concept_note": "The original concept note, archived verbatim at docs/archive/original-README.md.",
    "kittel2005": "C. Kittel, Introduction to Solid State Physics, 8th ed., Wiley, Hoboken (2005), "
                  "table of cohesive energies (Cu 3.49, W 8.90, Fe 4.28, Al 3.39 eV per atom).",
    "models": "Every number in the ledger comes from the magform module named in ClaimResult.models; "
              "the citations for each model are in that module's REFERENCES.",
}

P_LOW: float = 1.6e9
"""Pa.  Lower bound of the concept note's pressure window."""

P_HIGH: float = 4.0e9
"""Pa.  Upper bound of the concept note's pressure window."""

P_REFRACTORY: float = 2.0e9
"""Pa.  The '2,000+ MPa' the note quotes for refractory metals."""

ATOMIC_DATA: dict[str, tuple[float, float]] = {
    "copper_ofhc": (63.546e-3, 3.49),
    "tungsten": (183.84e-3, 8.90),
}
"""Material key -> (molar mass kg/mol, cohesive energy eV/atom; Kittel 2005).  Used to express a
pressure as an energy per atom, P x atomic volume."""

SECTIONS: tuple[tuple[str, str], ...] = (
    ("premise", "The premise"),
    ("forging", "Dynamic magneto-hydrodynamic forging"),
    ("refractory", "Refractory metals"),
    ("welding", "Cold welding of dissimilar alloys"),
    ("synthesis", "Quantum and superhard materials"),
    ("porosity", "Zero-porosity aerospace components"),
    ("containment", "Rail and launch systems"),
    ("questions", "The note's two questions"),
)
"""(key, title) of the ledger sections, in the order of the concept note."""

CHAPTERS: dict[str, str] = {
    "premise": "/story/pressure.html",
    "machine": "/story/machine.html",
    "power": "/story/power.html",
    "forging": "/story/forging.html",
    "refractory": "/story/refractory.html",
    "welding": "/story/welding.html",
    "synthesis": "/story/synthesis.html",
    "porosity": "/story/porosity.html",
    "containment": "/story/containment.html",
}
"""Documentation chapter that works through each topic (site-relative URL)."""


@dataclass(frozen=True)
class Claim(ResultMixin):
    """One claim (or question) of the concept note.

    ``quote`` is verbatim from the note; ``section`` one of the :data:`SECTIONS` keys; ``topic`` a
    short label; ``kind`` "claim" or "question"; ``chapter`` the key of the :data:`CHAPTERS` entry
    that discusses it.

    Reference: docs/archive/original-README.md.
    """

    id: str
    section: str
    quote: str
    topic: str
    kind: str = "claim"
    chapter: str = ""


CLAIMS: tuple[Claim, ...] = (
    Claim("C01", "premise", "generating 1,600 to over 4,000 MPa of magnetic pressure",
          "The pressure itself", chapter="premise"),
    Claim("C02", "premise", "using self-supporting, carbon-nanosheet-reinforced superconducting magnets",
          "The magnet", chapter="machine"),
    Claim("C03", "premise", "we move entirely out of the realm of traditional manufacturing and into extreme "
          "high-pressure materials science", "How extreme the pressure is", chapter="premise"),
    Claim("C04", "premise", "At these energy densities, you can manipulate matter at the atomic level across "
          "macro-scale components", "Atomic-level manipulation", chapter="premise"),
    Claim("C05", "premise", "without a single moving mechanical part", "No moving parts", chapter="forging"),
    Claim("C06", "forging", "Dynamic Magneto-Hydrodynamic (MHD) Forging", "The name", chapter="forging"),
    Claim("C07", "forging", "ultra-high magnetic fields are pulsed to induce massive eddy currents in a metal "
          "workpiece. The resulting Lorentz forces create an instantaneous, uniform inward crushing force.",
          "The mechanism", chapter="forging"),
    Claim("C08", "forging", "Because the pressure is applied purely through a magnetic field, it is perfectly "
          "uniform across the entire surface area.", "Uniform pressure", chapter="forging"),
    Claim("C09", "forging", "This completely eliminates surface friction, uneven structural cooling, and the tool "
          "wear associated with traditional heavy die forging.", "Friction, heat and tool wear", chapter="forging"),
    Claim("C10", "refractory", "Forging ultra-high-melting-point materials like tungsten, molybdenum, and rhenium "
          "typically requires heating them to near-melting temperatures", "How refractory metals are forged today",
          chapter="refractory"),
    Claim("C11", "refractory", "A 2,000+ MPa magnetic press generates enough force to plastically deform these "
          "hyper-hard refractory metals at room temperature or moderate heat.", "Cold deformation at 2 GPa",
          chapter="refractory"),
    Claim("C12", "refractory", "This preserves a pristine, ultra-fine crystalline grain structure",
          "Grain structure", chapter="refractory"),
    Claim("C13", "refractory", "resulting in parts with unprecedented fracture toughness and fatigue resistance",
          "Toughness and fatigue", chapter="refractory"),
    Claim("C14", "welding", "Room-Temperature Cold Welding of Dissimilar Alloys", "The capability",
          chapter="welding"),
    Claim("C15", "welding", "Driving two completely dissimilar metals (e.g., Titanium to Copper, or Steel to "
          "Aluminum) together with such extreme, instantaneous pressure that their atomic lattices are forced to "
          "interlock.", "How the bond forms", chapter="welding"),
    Claim("C16", "welding", "It creates a perfect atomic bond across large, complex geometric interfaces",
          "Large, complex interfaces", chapter="welding"),
    Claim("C17", "welding", "without creating a heat-affected zone (HAZ)", "No heat-affected zone",
          chapter="welding"),
    Claim("C18", "welding", "This bypasses the brittle intermetallic phases that usually ruin traditional thermal "
          "welding.", "Intermetallic phases", chapter="welding"),
    Claim("C19", "synthesis", "Diamond anvil cells can currently reach pressures to synthesize materials like bulk "
          "polycrystalline diamonds, lonsdaleite, and carbon nitrides, but only in microscopic microgram "
          "quantities.", "How superhard materials are made today", chapter="synthesis"),
    Claim("C20", "synthesis", "A large-bore, carbon-nanosheet HTS press scales these extreme pressures up to "
          "massive volumes.", "Scaling the pressure up", chapter="synthesis"),
    Claim("C21", "synthesis", "This enables the commercial, industrial-scale production of large-scale diamond "
          "semiconductor substrates, ultra-hard armor plating, and next-generation industrial cutting bits.",
          "Industrial products", chapter="synthesis"),
    Claim("C22", "synthesis", "Macroscopic Synthesis of Quantum & Superhard Materials", "Quantum materials",
          chapter="synthesis"),
    Claim("C23", "porosity", "These materials are notoriously difficult to forge without introducing microscopic "
          "internal voids or structural micro-cracks.", "Why TiAl and ODS alloys are hard to forge",
          chapter="porosity"),
    Claim("C24", "porosity", "A 4,000 MPa magnetic press can completely close all internal porosities in highly "
          "complex geometries", "Closing every pore", chapter="porosity"),
    Claim("C25", "porosity", "dramatically extending the operational lifespan and heat tolerances of rocket and "
          "turbine components", "Service life and heat tolerance", chapter="porosity"),
    Claim("C26", "containment", "Structural structural housing for hyper-velocity kinetic mass drivers, coilguns, "
          "or containment rings for compact fusion reactors.", "The loads to be contained", chapter="containment"),
    Claim("C27", "containment", "The exact same self-supporting carbon nanosheet + HTS tape architecture can be "
          "inverted.", "Inverting the architecture", chapter="containment"),
    Claim("C28", "containment", "it serves as the ultimate containment housing for high-energy physics "
          "applications, preventing plasma-containment fields or hyper-velocity projectile tracks from "
          "structurally tearing themselves apart under extreme internal electromagnetic stresses",
          "Whether the applications need it", chapter="containment"),
    Claim("Q1", "questions", "Should we map out the pulse-discharge electrical network (like homopolar "
          "generators or capacitor banks) required to drive these pressures?", "The pulse network",
          kind="question", chapter="power"),
    Claim("Q2", "questions", "Are you interested in the specifics of how room-temperature solid-state forging "
          "affects a specific alloy's crystalline microstructure?", "Microstructure after cold forging",
          kind="question", chapter="refractory"),
)
"""Every claim and question of the concept note, in its order, quoted verbatim."""


@dataclass(frozen=True, kw_only=True)
class ClaimResult(ResultMixin):
    """A claim with its verdict and the computed evidence for it.

    ``finding`` is one short paragraph in plain words; ``evidence`` the numbers it rests on, one
    statement each; ``models`` the functions that produced them (dotted names);
    ``chapter`` the site-relative URL of the chapter that works through it.

    Reference: docs/archive/original-README.md; the modules named in ``models``.
    """

    id: str
    section: str
    section_title: str
    quote: str
    topic: str
    kind: str
    verdict: Verdict
    finding: str
    evidence: tuple[str, ...]
    models: tuple[str, ...]
    chapter: str


@dataclass(frozen=True)
class _Evaluation:
    verdict: Verdict
    finding: str
    evidence: tuple[str, ...]
    models: tuple[str, ...]


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def _g(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}"


def _gpa(value: float, digits: int = 3) -> str:
    return f"{value / 1e9:.{digits}g} GPa"


def _mpa(value: float, digits: int = 3) -> str:
    return f"{value / 1e6:.{digits}g} MPa"


def _celsius(kelvin: float) -> str:
    return f"{kelvin - ZERO_CELSIUS:.0f} C"


def _pct(value: float, digits: int = 0) -> str:
    return f"{100.0 * value:.{digits}f} %"


def _sci(value: float) -> str:
    exponent = math.floor(math.log10(abs(value)))
    return f"{value / 10 ** exponent:.1f}e{exponent}"


# ---------------------------------------------------------------------------
# shared reference cases (computed once per process)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _design(pressure: float) -> PressDesign:
    return design_press(reference_spec(pressure))


@functools.lru_cache(maxsize=None)
def _tube(field: float) -> eddy.TubeSummary:
    """50 mm OD x 2 mm copper tube under a half-sine pulse matched to its wall."""
    rec = eddy.pulse_frequency_recommendation("copper_ofhc", 2e-3)
    half = 2.0 * rec.quarter_period
    tube = eddy.TubeCompression("copper_ofhc", 0.025, 0.002)
    return tube.simulate(eddy.half_sine_field(field, half), 4.0 * half).summary


@functools.lru_cache(maxsize=None)
def _weld(pressure: float) -> welding.WeldAssessment:
    """1 mm aluminium 6061 flyer over a 2 mm gap onto mild steel."""
    return welding.assess_weld("aluminum_6061", "steel_1018", 1e-3, 2e-3, pressure)


@functools.lru_cache(maxsize=None)
def _forge(material: str, pressure: float, temperature: float = T_ROOM) -> plasticity.ForgeAssessment:
    return plasticity.assess_forgeability(material, pressure, temperature=temperature)


@functools.lru_cache(maxsize=None)
def _micro(field: float | None) -> microstructure.MicrostructureAssessment:
    return microstructure.assess_microstructure("tungsten", peak_field=field)


@functools.lru_cache(maxsize=None)
def _pores() -> porosity.PoreClosureAssessment:
    return porosity.assess_pore_closure("gamma_tial", P_HIGH, initial_porosity=0.005)


_FLYER = synthesis.FlyerSpec("copper_ofhc", 0.002, 0.01)


def _bench(key: str) -> float:
    for item in PRESSURE_BENCHMARKS:
        if item.key == key:
            return float(item.value)
    raise InputError(f"unknown pressure benchmark {key!r}")


# ---------------------------------------------------------------------------
# evaluators: the premise
# ---------------------------------------------------------------------------
def _c01() -> _Evaluation:
    b_lo, b_hi = field_for_pressure(P_LOW), field_for_pressure(P_HIGH)
    small = pulsed_field_status(b_hi, PULSED_NONDESTRUCTIVE_BORE)
    large = pulsed_field_status(b_hi, 0.1)
    d = _design(P_HIGH)
    return _Evaluation(
        verdict_from_status(small),
        f"The pressures exist. {_gpa(P_LOW)} is the pressure of a {b_lo:.2f} T field and {_gpa(P_HIGH)} of "
        f"{b_hi:.2f} T, and a magnet has reached {PULSED_NONDESTRUCTIVE_RECORD:g} T without destroying itself. "
        f"It did so for milliseconds in a {PULSED_NONDESTRUCTIVE_BORE * 1e3:g} mm bore; the same field over a "
        f"100 mm bore is {large.label} and stores {d.stored_energy / 1e6:.0f} MJ.",
        (f"field for {_gpa(P_LOW)}: {b_lo:.2f} T; for {_gpa(P_HIGH)}: {b_hi:.2f} T",
         f"non-destructive pulsed record: {PULSED_NONDESTRUCTIVE_RECORD:g} T = "
         f"{_gpa(magnetic_pressure(PULSED_NONDESTRUCTIVE_RECORD))}, {PULSED_NONDESTRUCTIVE_BORE * 1e3:g} mm bore",
         f"{b_hi:.1f} T in a {PULSED_NONDESTRUCTIVE_BORE * 1e3:g} mm bore: {small.label}; in a 100 mm bore: {large.label}",
         f"stored energy of the reference press at {_gpa(P_HIGH)}: {d.stored_energy / 1e6:.1f} MJ"),
        ("magform.magnetics.field_for_pressure", "magform.benchmarks.pulsed_field_status",
         "magform.press.design_press"))


def _c02() -> _Evaluation:
    lo, hi = _design(P_LOW), _design(P_HIGH)
    status = worst(lo.status, hi.status)
    return _Evaluation(
        verdict_from_status(status),
        f"No. A superconducting magnet fails on three counts at once. The all-superconducting record is "
        f"{SC_FIELD_RECORD:g} T against the {lo.field:.0f} to {hi.field:.0f} T needed. The winding's "
        f"{_gpa(lo.peak_hoop_stress)} to {_gpa(hi.peak_hoop_stress)} of hoop stress needs a composite modulus of "
        f"{lo.structure.required_composite_modulus / 1e9:.0f} to {hi.structure.required_composite_modulus / 1e9:.0f} "
        "GPa to keep the superconductor within its 0.4 % strain limit. And because a steady field exerts no "
        "pressure the magnet must be pulsed, which at the bottom of the window deposits "
        f"{lo.pulsed_operation.loss_to_margin_ratio:.0f} times the winding's thermal margin on every shot; at "
        "the top the conductor has no margin left to lose.",
        (f"all-superconducting record {SC_FIELD_RECORD:g} T; highest field any superconductor has carried current "
         f"in: {SC_CONDUCTOR_RECORD:g} T",
         f"conductor at {_gpa(P_LOW)}: {lo.conductor.status.label} (margin {lo.conductor.margin:.2f}); at "
         f"{_gpa(P_HIGH)}: {hi.conductor.status.label} (margin {hi.conductor.margin:.2f})",
         f"structure at {_gpa(P_LOW)}: {lo.structure.status.label}; at {_gpa(P_HIGH)}: {hi.structure.status.label} "
         f"(graphene's own modulus is 1000 GPa)",
         f"pulsed operation at {_gpa(P_LOW)}: loss / enthalpy margin = "
         f"{lo.pulsed_operation.loss_to_margin_ratio:.3g} ({lo.pulsed_operation.status.label})",
         f"steady 100 T field on a workpiece: {pressure_with_background(100.0, 0.0):.0f} Pa",
         f"best buildable alternative: {hi.best_architecture} ({hi.best_status.label})"),
        ("magform.press.design_press", "magform.conductor.assess_conductor", "magform.conductor.assess_structure",
         "magform.conductor.assess_pulsed_operation", "magform.magnetics.pressure_with_background"))


def _c03() -> _Evaluation:
    die, hpht, hip = _bench("cold_forging_die"), _bench("hpht_cubic"), _bench("hip")
    place = where_does_it_sit(P_HIGH)
    false = die >= P_LOW and hpht > P_HIGH
    return _Evaluation(
        Verdict.CONTRADICTED if false else Verdict.EXTRAPOLATION,
        f"The window sits inside ordinary manufacturing. Cold-forging dies already see about {_gpa(die)} of "
        f"contact pressure, and industrial diamond presses hold {_gpa(hpht)} for hours. {_gpa(P_LOW)} to "
        f"{_gpa(P_HIGH)} lies between them: high for a forming process, low for high-pressure science.",
        (f"hot isostatic pressing {_gpa(hip)}; cold-forging die contact {_gpa(die)}; industrial diamond presses "
         f"{_gpa(hpht)}",
         f"benchmarks at or below {_gpa(P_HIGH)}: {len(place.below)}; above: {len(place.above)}",
         f"energy density at {_gpa(P_HIGH)}: {_pct(place.tnt_fraction)} of that of TNT"),
        ("magform.benchmarks.where_does_it_sit", "magform.benchmarks.PRESSURE_BENCHMARKS"))


def _c04() -> _Evaluation:
    evidence = []
    ratios = []
    for key, (molar_mass, cohesive) in ATOMIC_DATA.items():
        m = as_material(key)
        atomic_volume = molar_mass / (m.density * N_A)
        per_atom = P_HIGH * atomic_volume / E_CHARGE
        ratios.append(per_atom / cohesive)
        evidence.append(f"{m.base_element}: {_gpa(P_HIGH)} x atomic volume = {per_atom:.2f} eV per atom against "
                        f"{cohesive:.2f} eV cohesive energy; elastic volume change {_pct(m.volumetric_strain(P_HIGH), 1)}")
    evidence.append("the field at which atomic structure itself changes is the atomic unit, 2.35e5 T "
                    "(magform.magnetar)")
    return _Evaluation(
        Verdict.CONTRADICTED,
        f"At {_gpa(P_HIGH)} the energy available per atom is {_pct(min(ratios))} to {_pct(max(ratios))} of what "
        "holds the solid together. Applied evenly it squeezes a metal by one to three per cent, elastically, and "
        "the metal springs back. Shaping it takes uneven stress and ordinary plastic flow. Fields change atoms "
        "themselves only above about 235,000 T.",
        tuple(evidence),
        ("magform.materials.Material.volumetric_strain", "magform.claims.ATOMIC_DATA"))


def _c05() -> _Evaluation:
    tube, weld = _tube(30.0), _weld(0.1e9)
    shock = synthesis.assess_route("lonsdaleite_shock", P_HIGH, flyer=_FLYER)
    anvil = synthesis.anvil_feasibility(5e9)
    return _Evaluation(
        Verdict.CONTRADICTED,
        "The coil has no moving parts; the process does. In forming, the workpiece wall is the moving part "
        f"({tube.peak_velocity:.0f} m/s in a 30 T tube compression). Welding needs a flyer "
        f"({weld.impact_velocity:.0f} m/s), shock synthesis a faster one ({shock.flyer_velocity:.0f} m/s), and any "
        "static synthesis route needs anvils. Shape still comes from a die or mandrel.",
        (f"tube wall velocity at 30 T: {tube.peak_velocity:.0f} m/s",
         f"weld flyer velocity at {_gpa(0.1e9)} drive: {weld.impact_velocity:.0f} m/s",
         f"shock-synthesis flyer at {_gpa(P_HIGH)} drive: {shock.flyer_velocity:.0f} m/s",
         f"a static route at 5 GPa needs anvils: {anvil.note.split(';')[0].lower()}"),
        ("magform.eddy.TubeCompression", "magform.welding.assess_weld", "magform.synthesis.assess_route",
         "magform.synthesis.anvil_feasibility"))


# ---------------------------------------------------------------------------
# evaluators: forging
# ---------------------------------------------------------------------------
_SKIN_METALS = ("aluminum_6061", "copper_ofhc", "titanium_cp2", "steel_1018", "tungsten")


def _c06() -> _Evaluation:
    melt = {key: eddy.skin_melting_field(key) for key in _SKIN_METALS}
    lo, hi = EMF_CONVENTIONAL_FIELD
    return _Evaluation(
        Verdict.CONTRADICTED,
        "Magnetohydrodynamics is the dynamics of conducting fluids; this workpiece is solid. The process "
        f"described is electromagnetic forming, in industrial use since the 1960s at {lo:g} to {hi:g} T. The fluid "
        "name becomes literally true only where the eddy-current skin melts: above "
        f"{melt['aluminum_6061']:.0f} T for aluminium, {melt['copper_ofhc']:.0f} T for copper and "
        f"{melt['tungsten']:.0f} T for tungsten.",
        tuple(f"skin of {as_material(k).name} reaches its melting point at {v:.0f} T" for k, v in melt.items())
        + (f"industrial electromagnetic forming: {lo:g}-{hi:g} T, {_gpa(magnetic_pressure(lo))}-"
           f"{_gpa(magnetic_pressure(hi))}",),
        ("magform.eddy.skin_melting_field", "magform.benchmarks.EMF_CONVENTIONAL_FIELD"))


def _c07() -> _Evaluation:
    b_lo = field_for_pressure(P_LOW)
    t30, t63 = _tube(30.0), _tube(b_lo)
    rec = eddy.pulse_frequency_recommendation("copper_ofhc", 2e-3)
    status = Status.STATE_OF_THE_ART if b_lo <= EMF_CONVENTIONAL_FIELD[1] else Status.EXTRAPOLATION
    return _Evaluation(
        verdict_from_status(status),
        "The mechanism is real and routine up to about 50 T. A pulsed field drives eddy currents in the surface "
        "of the workpiece and the Lorentz force pushes it inward. Modelled on a 50 mm copper tube with a 2 mm "
        f"wall, a 30 T pulse closes the bore in {t30.time_to_stop * 1e6:.0f} microseconds. At {b_lo:.0f} T, the "
        f"bottom of the note's window, it closes in {t63.time_to_stop * 1e6:.0f} microseconds at "
        f"{t63.peak_velocity:.0f} m/s, beyond any industrial coil. The force is fast, not instantaneous, and not "
        "uniform (C08).",
        (f"matched pulse for a 2 mm copper wall: {rec.frequency / 1e3:.1f} kHz, rise {rec.quarter_period * 1e6:.0f} us",
         f"30 T: bore closed at {t30.time_to_stop * 1e6:.0f} us, peak wall speed {t30.peak_velocity:.0f} m/s, "
         f"strain rate {_sci(t30.peak_strain_rate)} 1/s",
         f"{b_lo:.1f} T: bore closed at {t63.time_to_stop * 1e6:.0f} us, {t63.peak_velocity:.0f} m/s, "
         f"{_sci(t63.peak_strain_rate)} 1/s",
         f"industrial coils stop at {EMF_CONVENTIONAL_FIELD[1]:g} T"),
        ("magform.eddy.TubeCompression", "magform.eddy.pulse_frequency_recommendation"))


def _c08() -> _Evaluation:
    spec = reference_spec(P_HIGH)
    sol = spec.solenoid
    mid = fieldmap.workpiece_pressure(sol, 0.04, 0.30)
    long_billet = fieldmap.workpiece_pressure(sol, 0.04, 0.60)
    rounded = fieldmap.workpiece_pressure(sol, 0.04, 0.30, 0.01)
    axis = sol.pressure_uniformity(1e8, 0.30)
    tube = reference_spec(P_HIGH, workpiece_wall=2e-3)
    slow = eddy.sheet_shielding_factor(2e-3, eddy.skin_depth_for_material("copper_ofhc", 0.25 / 2e-3))
    del tube
    return _Evaluation(
        Verdict.CONTRADICTED,
        "The pressure follows the field, and the field is shaped by the coil and by the workpiece. For a billet "
        "in the reference press the pressure is within "
        f"{_pct(1.0 - mid.uniformity)} of even over the middle 80 % of its length, then departs both ways: it "
        f"falls to {_pct(long_billet.central_min_ratio)} of nominal at the coil ends when the billet is as long as "
        f"the coil, and it peaks at {mid.peak_ratio:.1f} times nominal on the billet's own corners "
        f"({rounded.peak_ratio:.1f} times with a 10 mm corner radius). On a part with features the pressure is "
        "highest on convex edges and lowest in recesses.",
        (f"40 mm x 300 mm billet: mid-plane {mid.mid_ratio:.3f} x nominal, min/max over the central 80 % = "
         f"{mid.uniformity:.3f}",
         f"corner peak: {mid.peak_ratio:.2f} x nominal (2 mm radius), {rounded.peak_ratio:.2f} x (10 mm radius)",
         f"billet as long as the coil: minimum {long_billet.central_min_ratio:.2f} x nominal",
         f"free-space on-axis estimate for the same 300 mm: {axis.end_ratio:.2f} at the ends",
         f"a 2 mm copper wall passes {_pct(slow)} of the pressure if the field takes 2 ms to rise"),
        ("magform.fieldmap.workpiece_pressure", "magform.magnetics.Solenoid.pressure_uniformity",
         "magform.eddy.sheet_shielding_factor"))


def _c09() -> _Evaluation:
    b_lo, b_hi = field_for_pressure(P_LOW), field_for_pressure(P_HIGH)
    cu_lo, cu_hi = eddy.skin_temperature_rise("copper_ofhc", b_lo), eddy.skin_temperature_rise("copper_ofhc", b_hi)
    d = _design(P_HIGH)
    single = next(a for a in d.architectures if a.key == "single_turn")
    net = d.network
    evidence = [
        f"copper skin temperature rise: +{cu_lo:.0f} K at {b_lo:.0f} T, +{cu_hi:.0f} K at {b_hi:.0f} T "
        f"(melts above {eddy.skin_melting_field('copper_ofhc'):.0f} T)",
        f"aluminium 6061 skin: +{eddy.skin_temperature_rise('aluminum_6061', b_lo):.0f} K at {b_lo:.0f} T "
        f"(melts above {eddy.skin_melting_field('aluminum_6061'):.0f} T)",
        "single-turn coils, the simplest way to these fields, are destroyed by every shot "
        f"({single.status.label} at a 100 mm bore)",
    ]
    if net is not None:
        evidence.append(f"reference coil conductor ends each {d.rise_time * 1e3:.0f} ms pulse at "
                        f"{net.conductor_final_temperature:.0f} K from 77 K")
    return _Evaluation(
        Verdict.CONTRADICTED,
        "Friction goes only on the side facing the coil; a die or mandrel still rubs. Heating does not go at "
        f"all: the eddy currents that carry the force heat the surface layer by {cu_lo:.0f} K in copper at "
        f"{b_lo:.0f} T and melt it at {b_hi:.0f} T, while the interior stays cold. And the tool wear moves to the "
        "coil, which carries the same pressure as the workpiece and is the part that fails.",
        tuple(evidence),
        ("magform.eddy.skin_temperature_rise", "magform.eddy.skin_melting_field", "magform.press.design_press",
         "magform.pulse.assess_pulse_network"))


# ---------------------------------------------------------------------------
# evaluators: refractory metals
# ---------------------------------------------------------------------------
_REFRACTORY = ("tungsten", "molybdenum", "rhenium")


def _c10() -> _Evaluation:
    gaps = {key: plasticity.forging_practice_gap(key) for key in _REFRACTORY}
    false = all(g.readme_claim_contradicted for g in gaps.values())
    w, mo = gaps["tungsten"], gaps["molybdenum"]
    evidence = []
    for key, g in gaps.items():
        lo, hi = g.conventional_range
        m = as_material(key)
        cold = hi - lo < 1.0
        span = _celsius(lo) if cold else f"{lo - ZERO_CELSIUS:.0f}-{hi - ZERO_CELSIUS:.0f} C"
        share = f"{lo / m.melting_point:.2f}" if cold else f"{lo / m.melting_point:.2f}-{hi / m.melting_point:.2f}"
        evidence.append(f"{m.base_element}: worked at {span} ({share} of its {_celsius(m.melting_point)} "
                        f"melting point); {g.note}")
    return _Evaluation(
        Verdict.CONTRADICTED if false else Verdict.NOT_MODELLED,
        f"Tungsten is forged at {w.conventional_range[0] - ZERO_CELSIUS:.0f} to "
        f"{w.conventional_range[1] - ZERO_CELSIUS:.0f} C and molybdenum at "
        f"{mo.conventional_range[0] - ZERO_CELSIUS:.0f} to {mo.conventional_range[1] - ZERO_CELSIUS:.0f} C, about "
        "half their melting points; rhenium is rolled cold. Nobody forges them above 2,500 C, a temperature "
        f"within {as_material('molybdenum').melting_point - 2773.15:.0f} K of melting molybdenum.",
        tuple(evidence),
        ("magform.plasticity.forging_practice_gap", "magform.materials.MATERIALS"))


def _c11() -> _Evaluation:
    per_metal: list[Status] = []
    evidence: list[str] = []
    for key in _REFRACTORY:
        cold = _forge(key, P_REFRACTORY)
        options = [cold.status]
        text = (f"{as_material(key).base_element} at room temperature: {cold.status.label} (needs "
                f"{_gpa(cold.required_pressure)}; flow stress {_gpa(cold.flow_stress_min)}-{_gpa(cold.flow_stress_max)})")
        t_warm = cold.warm_forming_temperature
        if cold.status is Status.INFEASIBLE and t_warm is not None:
            warm = _forge(key, P_REFRACTORY, t_warm + 50.0)
            options.append(warm.status)
            text += f"; at {_celsius(t_warm + 50.0)}: {warm.status.label}"
        per_metal.append(best(options))
        evidence.append(text)
    w = _forge("tungsten", P_REFRACTORY)
    w4 = _forge("tungsten", P_HIGH)
    evidence.append(f"tungsten below its {_celsius(w.dbtt)} ductile-brittle transition fails at "
                    f"{_pct(w.failure_strain_limit)} strain unconfined; localisation strain {w.adiabatic_shear_strain:.3f}")
    evidence.append(f"tungsten at {_gpa(P_HIGH)}, room temperature: {w4.status.label} (margin {w4.margin:.2f})")
    status = worst(per_metal)
    return _Evaluation(
        verdict_from_status(status),
        f"Partly. At 2 GPa and forming rates tungsten's flow stress is {_gpa(w.flow_stress_min)} to "
        f"{_gpa(w.flow_stress_max)}, so at room temperature the press cannot move it; it can from about "
        f"{_celsius(w.warm_forming_temperature)}, which is moderate heat. Molybdenum just yields cold. In every "
        "case the metal is below or near its ductile-brittle transition, so it flows only if the pressure also "
        "confines it, and that has not been shown at this scale. Pressure alone shapes nothing: it needs a die.",
        tuple(evidence),
        ("magform.plasticity.assess_forgeability", "magform.plasticity.warm_forming_temperature",
         "magform.plasticity.adiabatic_shear_critical_strain"))


def _c12() -> _Evaluation:
    plain = _micro(None)
    b2 = field_for_pressure(P_REFRACTORY)
    at2, at4 = _micro(b2), _micro(field_for_pressure(P_HIGH))
    refine = microstructure.grain_refinement_strain("tungsten", microstructure.ULTRAFINE_GRAIN_SIZE)
    status = worst(plain.status, at2.status)
    return _Evaluation(
        verdict_from_status(status),
        "Half of this holds. A cold pulse leaves the interior far too cool to recrystallise, so the grains the "
        f"billet started with survive: {plain.grain_size_before * 1e6:.0f} micrometres, flattened to "
        f"{plain.grain_size_after * 1e6:.0f}. They are not ultra-fine and one forming strain does not make them "
        f"so; that takes a strain of about {refine:.1f}. Nor is the structure pristine. The dislocation density "
        f"rises {plain.dislocation_density / plain.initial_dislocation_density:.0f}-fold, and above about "
        f"{b2:.0f} T the eddy-current skin is heated to a large fraction of the melting point.",
        (f"bulk temperature rise at strain {plain.strain:g}: +{plain.bulk_temperature_rise:.0f} K to "
         f"{plain.bulk_homologous_temperature:.2f} of the melting point; recrystallised fraction "
         f"{plain.bulk_recrystallised_fraction:.1g}",
         f"grain size {plain.grain_size_before * 1e6:.0f} -> {plain.grain_size_after * 1e6:.1f} um; ultra-fine "
         f"(< 1 um) needs strain {refine:.2f}",
         f"dislocation density {_sci(plain.initial_dislocation_density)} -> {_sci(plain.dislocation_density)} "
         f"per m^2; cell size {plain.cell_size * 1e6:.2f} um",
         f"skin at {b2:.0f} T: {at2.skin.peak_temperature:.0f} K ({at2.skin.homologous_temperature:.2f} of melting); "
         f"at {at4.skin.peak_field:.0f} T: {at4.skin.peak_temperature:.0f} K "
         f"({at4.skin.homologous_temperature:.2f}) for {at4.skin.cooling_time * 1e3:.1f} ms"),
        ("magform.microstructure.assess_microstructure", "magform.microstructure.grain_refinement_strain",
         "magform.microstructure.skin_thermal_exposure"))


def _c13() -> _Evaluation:
    m = _micro(None)
    return _Evaluation(
        Verdict.NOT_MODELLED,
        "The package does not compute fracture toughness or fatigue life, so it cannot confirm this. What it does "
        f"compute points the other way for a single pulse: cold work raises tungsten's yield strength by "
        f"{_pct(m.hardening_ratio - 1.0)} and uses up ductility, and at forming rates the deformation concentrates "
        f"into shear bands from a strain of {m.shear_localisation_strain:.3f}. The low transition temperature "
        "of heavily rolled tungsten comes from rolling strains above three, not from one pulse.",
        (f"yield strength {_gpa(m.yield_strength_before)} -> {_gpa(m.yield_strength_after)}",
         f"adiabatic shear instability at strain {m.shear_localisation_strain:.3f}, target strain {m.strain:g}",
         f"strain to reach an ultra-fine structure: {m.strain_for_ultrafine:.2f}",
         "fracture toughness and fatigue life: not modelled"),
        ("magform.microstructure.assess_microstructure", "magform.plasticity.adiabatic_shear_critical_strain"))


# ---------------------------------------------------------------------------
# evaluators: welding
# ---------------------------------------------------------------------------
def _c14() -> _Evaluation:
    w = _weld(0.1e9)
    cu_ti = welding.assess_weld("copper_ofhc", "titanium_cp2", 1e-3, 2e-3, 0.4e9)
    status = worst(w.status, cu_ti.status)
    return _Evaluation(
        verdict_from_status(status),
        "This exists and is sold as magnetic pulse welding. A 1 mm aluminium sheet thrown across a 2 mm gap at "
        f"steel bonds at {w.impact_velocity:.0f} m/s, which takes a {w.drive_field:.0f} T coil. Copper onto "
        f"titanium takes {cu_ti.drive_field:.0f} T. Both sit inside industrial forming fields, a factor of four "
        "to forty below the note's pressures.",
        (f"Al 6061 onto steel: {_mpa(w.drive_pressure)} drive, {w.drive_field:.1f} T, {w.impact_velocity:.0f} m/s, "
         f"{w.status.label}, {w.interface_morphology} interface",
         f"Cu onto Ti: {_mpa(cu_ti.drive_pressure)} drive, {cu_ti.drive_field:.1f} T, {cu_ti.impact_velocity:.0f} m/s, "
         f"{cu_ti.status.label}",
         f"recorded pairs: {', '.join(sorted(welding.DEMONSTRATED_PAIRS))}"),
        ("magform.welding.assess_weld", "magform.welding.DEMONSTRATED_PAIRS"))


def _c15() -> _Evaluation:
    good, lo, hi = _weld(0.1e9), _weld(P_LOW), _weld(P_HIGH)
    window = welding.drive_pressure_window("aluminum_6061", "steel_1018", 1e-3, 2e-3)
    angle = welding.critical_collision_angle("aluminum_6061", "steel_1018", 400.0)
    false = lo.status is Status.INFEASIBLE and hi.status is Status.INFEASIBLE and good.in_window
    return _Evaluation(
        Verdict.CONTRADICTED if false else Verdict.EXTRAPOLATION,
        "Pressure does not make the bond; an oblique collision does. The plates must meet at an angle so that a "
        "jet strips the surfaces clean, and the impact speed has to fall in a window. For aluminium on steel "
        f"that window is {good.vp_min:.0f} to {good.vp_max:.0f} m/s, reached with {_mpa(window.pressure_min)} to "
        f"{_mpa(window.pressure_max)} of magnetic pressure. Driven at the note's {_gpa(P_LOW)} to {_gpa(P_HIGH)} "
        f"the sheet arrives at {lo.impact_velocity:.0f} to {hi.impact_velocity:.0f} m/s, the collision point "
        "outruns sound in both metals, no jet forms and nothing welds. The gap does the amplifying: the working "
        f"joint sees {_gpa(good.impact_pressure)} at impact from a {_mpa(good.drive_pressure)} drive.",
        (f"weld window at 10 degrees: {good.vp_min:.0f}-{good.vp_max:.0f} m/s; drive {_mpa(window.pressure_min)}-"
         f"{_mpa(window.pressure_max)} ({window.field_min:.1f}-{window.field_max:.1f} T)",
         f"at {_gpa(P_LOW)}: {lo.impact_velocity:.0f} m/s, collision point {lo.collision_velocity:.0f} m/s against a "
         f"{lo.vc_sonic:.0f} m/s sonic limit: {lo.status.label}",
         f"at {_gpa(P_HIGH)}: {hi.impact_velocity:.0f} m/s, collision point {hi.collision_velocity:.0f} m/s: "
         f"{hi.status.label}",
         f"impact pressure in the working joint: {_gpa(good.impact_pressure)}, {good.pressure_amplification:.0f} "
         "times the drive",
         f"below a collision angle of {math.degrees(angle):.1f} degrees at 400 m/s there is no jet: a flat-on "
         "impact does not weld"),
        ("magform.welding.assess_weld", "magform.welding.drive_pressure_window",
         "magform.welding.critical_collision_angle", "magform.shock.impedance_match"))


def _c16() -> _Evaluation:
    window = welding.weld_window("aluminum_6061", "steel_1018", 1e-3)
    narrow = welding.weld_window("aluminum_6061", "steel_1018", 1e-3, math.radians(5.0))
    good = _weld(0.1e9)
    return _Evaluation(
        Verdict.EXTRAPOLATION,
        "The bond forms only where the angle and the speed are both right, and they change along the joint. "
        f"At 10 degrees the usable speeds span a factor of {window.width:.2f}; at 5 degrees the window "
        f"{'is gone' if not narrow.exists else f'narrows to {narrow.width:.2f}'}. Pulse welds are therefore lap and "
        "tube joints a few millimetres wide. Large flat areas are joined by explosive cladding, which works on "
        "the same collision physics. Complex interfaces are not within either process.",
        (f"window at 10 degrees: {window.vp_min:.0f}-{window.vp_max:.0f} m/s (ratio {window.width:.2f}), limited "
         f"above by {window.upper_limited_by}",
         f"window at 5 degrees: {'none' if not narrow.exists else f'{narrow.vp_min:.0f}-{narrow.vp_max:.0f} m/s'}",
         f"estimated molten layer at the working point: {good.melt_layer_estimate * 1e6:.0f} um"),
        ("magform.welding.weld_window", "magform.welding.assess_weld"))


def _c17() -> _Evaluation:
    good, hi = _weld(0.1e9), _weld(P_HIGH)
    return _Evaluation(
        verdict_from_status(good.status),
        f"True in the bulk at the {good.drive_field:.0f} T a weld needs. The flyer's skin warms by "
        f"{good.flyer_skin_temperature_rise:.0f} K, and the heated interface layer is about "
        f"{good.melt_layer_estimate * 1e6:.0f} micrometres thick and cools in a microsecond. It stops being true "
        f"at the note's fields: the aluminium flyer's skin melts above {good.flyer_skin_melting_field:.0f} T.",
        (f"flyer skin temperature rise at {good.drive_field:.1f} T: +{good.flyer_skin_temperature_rise:.0f} K",
         f"interface layer {good.melt_layer_estimate * 1e6:.1f} um, quench time {good.quench_time * 1e6:.1f} us",
         f"flyer skin melting field {good.flyer_skin_melting_field:.1f} T; at {hi.drive_field:.0f} T the skin "
         f"rise would be +{hi.flyer_skin_temperature_rise:.0f} K"),
        ("magform.welding.assess_weld", "magform.eddy.skin_temperature_rise"))


def _c18() -> _Evaluation:
    good = _weld(0.1e9)
    solid = good.interdiffusion_solid if good.interdiffusion_solid is not None else math.nan
    ratio = good.fusion_weld_interdiffusion / solid if solid else math.inf
    return _Evaluation(
        verdict_from_status(good.status),
        "Largely true, and it is why the process is used. Brittle iron-aluminium phases grow by diffusion, and "
        f"a pulse weld gives them a microsecond: atoms move about {solid * 1e9:.1f} nanometres in the solid, "
        f"against {good.fusion_weld_interdiffusion * 1e6:.0f} micrometres in a second-long fusion weld. The phases "
        f"still appear ({', '.join(good.intermetallics)}) in the thin pockets where the interface melts, so they "
        "are reduced by orders of magnitude and not removed.",
        (f"solid-state interdiffusion over the {good.quench_time * 1e6:.1f} us quench: {solid * 1e9:.2f} nm",
         f"liquid interdiffusion over the same time: {good.interdiffusion_liquid * 1e9:.0f} nm",
         f"fusion weld, liquid for 1 s: {good.fusion_weld_interdiffusion * 1e6:.1f} um "
         f"(about {float(f'{ratio:.1g}'):,.0f} times more)",
         f"phases recorded for Al-Fe: {', '.join(good.intermetallics)}"),
        ("magform.welding.assess_weld", "magform.welding.interdiffusion_length"))


# ---------------------------------------------------------------------------
# evaluators: synthesis
# ---------------------------------------------------------------------------
def _c19() -> _Evaluation:
    routes = synthesis.SYNTHESIS_ROUTES
    industrial = [r for r in routes.values() if "industrial" in r.current_scale]
    dac = [r for r in routes.values() if r.kind == "laser_heated_dac"]
    false = len(industrial) >= 3
    return _Evaluation(
        Verdict.CONTRADICTED if false else Verdict.NOT_MODELLED,
        "Only the last of the three examples is right. Diamond grit, single crystals and polycrystalline "
        "compacts are made by the tonne in cubic and belt presses at 5 to 7 GPa, and nano-polycrystalline "
        "diamond comes out of multi-anvil presses in centimetre pieces at 15 GPa. Carbon nitrides are the "
        f"exception: they need {dac[0].p_min / 1e9:.0f} to {dac[0].p_max / 1e9:.0f} GPa, which only a diamond "
        "anvil cell reaches.",
        tuple(f"{r.product}: {r.p_min / 1e9:g}-{(r.p_max or r.p_min) / 1e9:g} GPa; {r.current_scale}"
              for r in routes.values() if r.is_static),
        ("magform.synthesis.SYNTHESIS_ROUTES", "magform.benchmarks.STATIC_PRESS_LADDER"))


def _c20() -> _Evaluation:
    plain = synthesis.assess_all(P_HIGH)
    static = [a for a in plain if a.kind != "shock"]
    status = best(a.status for a in static)
    routes = [r for r in synthesis.SYNTHESIS_ROUTES.values() if r.is_static]
    p_min = min(r.p_min for r in routes)
    dwell = min(r.min_dwell for r in routes)
    volume = synthesis.volume_scaling(0.05, 0.5, P_HIGH)
    return _Evaluation(
        verdict_from_status(status),
        f"No static route fits, for three independent reasons. The lowest pressure any of them needs is "
        f"{_gpa(p_min)}, above the top of the note's window. Every one needs that pressure held for at least "
        f"{dwell:g} second and usually minutes to days, and a magnetic pulse lasts microseconds to milliseconds. "
        "And every one needs 1,500 to 3,000 K, which the magnet does not supply.",
        (f"static routes assessed at {_gpa(P_HIGH)}, room temperature, 0.1 ms pulse: "
         f"{sum(a.status is Status.INFEASIBLE for a in static)} of {len(static)} infeasible",
         f"lowest route pressure: {_gpa(p_min)} (catalytic cubic boron nitride)",
         f"shortest dwell of any static route: {dwell:g} s",
         f"the reference bore would hold {volume.graphite_mass:.1f} kg of graphite; the field energy over it is "
         f"{volume.bore_magnetic_energy / 1e6:.0f} MJ"),
        ("magform.synthesis.assess_all", "magform.synthesis.volume_scaling"))


def _c21() -> _Evaluation:
    shock = synthesis.assess_route("lonsdaleite_shock", P_HIGH, flyer=_FLYER)
    dupont = synthesis.assess_route("diamond_shock_dupont", P_HIGH, flyer=_FLYER)
    porous = synthesis.assess_route("diamond_shock_dupont", P_HIGH, flyer=synthesis.FlyerSpec(
        "copper_ofhc", 0.002, 0.01, sample="graphite_polycrystalline"))
    cu_cu = synthesis.assess_route("lonsdaleite_shock", P_HIGH, flyer=synthesis.FlyerSpec(
        "copper_ofhc", 0.002, 0.01, sample="copper_ofhc"))
    return _Evaluation(
        Verdict.INFEASIBLE,
        "The only route a pulse can run is shock synthesis, and it does not make these products. A 2 mm copper "
        f"flyer driven at {_gpa(P_HIGH)} over 10 mm reaches {shock.flyer_velocity:.0f} m/s and puts "
        f"{_gpa(shock.pressure_sample)} into graphite for under a microsecond. That is at the edge of the diamond "
        f"window ({dupont.status.label}; {porous.status.label} with porous feedstock) and yields micrometre "
        "powder, as explosive shock synthesis has since 1961. Semiconductor substrates are grown as single "
        "crystals; a shock cannot make one.",
        (f"flyer velocity {shock.flyer_velocity:.0f} m/s; pressure in crystalline graphite "
         f"{_gpa(shock.pressure_sample)} (copper on copper would give {_gpa(cu_cu.pressure_sample)})",
         f"lonsdaleite route (>= 15 GPa): {shock.status.label}, margin {shock.pressure_margin:.2f}",
         f"shock diamond route (>= 20 GPa): {dupont.status.label}, margin {dupont.pressure_margin:.2f}; porous "
         f"graphite: {_gpa(porous.pressure_sample)}, {porous.status.label}",
         f"product form: {shock.product_form}"),
        ("magform.synthesis.assess_route", "magform.shock.impedance_match", "magform.shock.flyer_velocity"))


def _c22() -> _Evaluation:
    return _Evaluation(
        Verdict.INFEASIBLE,
        "No quantum material is waiting on 1.6 to 4 GPa. Quantum-grade diamond is grown by vapour deposition "
        "below atmospheric pressure. The superconducting hydrides need 150 to 170 GPa and do not survive release.",
        (synthesis.QUANTUM_MATERIALS_NOTE,),
        ("magform.synthesis.QUANTUM_MATERIALS_NOTE",))


# ---------------------------------------------------------------------------
# evaluators: porosity
# ---------------------------------------------------------------------------
def _c23() -> _Evaluation:
    tial, ods = as_material("gamma_tial"), as_material("ods_fecral")
    return _Evaluation(
        Verdict.STATE_OF_THE_ART,
        f"True. Gamma titanium aluminide stretches {_pct(tial.elongation, 1)} before it breaks at room temperature "
        f"and stays brittle up to about {_celsius(tial.dbtt)}, so it is forged isothermally at "
        f"{tial.conventional_working[0] - ZERO_CELSIUS:.0f} to {tial.conventional_working[1] - ZERO_CELSIUS:.0f} C "
        "or cast and then pressed hot. Oxide-dispersion-strengthened alloys cannot be melted without losing "
        "their dispersion and are consolidated from powder.",
        (f"gamma TiAl: elongation {_pct(tial.elongation, 1)}, ductile-brittle transition {_celsius(tial.dbtt)}, "
         f"worked at {tial.conventional_working_note}",
         f"ODS FeCrAl: elongation {_pct(ods.elongation, 0)}; {ods.notes}"),
        ("magform.materials.MATERIALS",))


def _c24() -> _Evaluation:
    p = _pores()
    hip = porosity.hip_cycle("gamma_tial")
    thin = porosity.assess_pore_closure("gamma_tial", P_HIGH, section_thickness=2e-3, drive_frequency=1e4)
    evidence = [
        f"pores of {_pct(p.initial_porosity, 1)} start to collapse at {_gpa(p.collapse_pressure_quasi_static)} "
        f"(slow) or {_gpa(p.collapse_pressure_dynamic)} (at pulse rates)",
        f"pressure to close them to 0.01 %: {_gpa(p.closure_pressure_quasi_static)} (slow), "
        f"{_gpa(p.closure_pressure_dynamic)} (pulse rates)",
        f"collapse takes {p.collapse_time * 1e9:.0f} ns; diffusion bonding of the closed faces at room "
        f"temperature would take {_sci(p.diffusion_bond_time)} s",
        f"strain at the pore wall {p.wall_strain:.1f} against {_pct(p.ductility, 1)} ductility",
        f"skin depth of TiAl at 10 kHz: {thin.skin_depth * 1e3:.1f} mm, so a bare 2 mm section feels "
        f"{_pct(thin.pressure_transfer)} of the pressure",
    ]
    if hip is not None:
        evidence.append(f"hot isostatic pressing does the job at {_mpa(hip.pressure)}, {_celsius(hip.temperature)}, "
                        f"{hip.time / 3600:.0f} h")
    return _Evaluation(
        verdict_from_status(p.status),
        "Closing a pore and healing it are different things, and the pulse does neither reliably. A small pore "
        f"is shielded by the metal around it: at pulse rates titanium aluminide's pores do not even begin to "
        f"yield below {_gpa(p.collapse_pressure_dynamic)}. Where a pore does close, its faces only touch. Bonding "
        "them takes diffusion, which at room temperature does not happen on any timescale, so the pore is left "
        "as a crack. A thin blade section also lets part of the field straight through, and on a shaped part "
        "magnetic pressure is not even. Hot isostatic pressing already closes and heals these pores at "
        f"{_mpa(hip.pressure) if hip is not None else '0.1-0.2 GPa'} because it is hot and slow.",
        tuple(evidence),
        ("magform.porosity.assess_pore_closure", "magform.porosity.hip_cycle",
         "magform.porosity.diffusion_bonding_time"))


def _c25() -> _Evaluation:
    hip = porosity.hip_cycle("gamma_tial")
    at_hip = None
    if hip is not None:
        at_hip = porosity.assess_pore_closure("gamma_tial", hip.pressure, temperature=hip.temperature,
                                              pulse_duration=hip.time)
    return _Evaluation(
        Verdict.NOT_MODELLED,
        "Fatigue life and service temperature are not computed here. The benefit of removing porosity is real "
        "and is already collected: cast turbine blades are hot isostatically pressed as a matter of course"
        + (f" ({at_hip.status.label} in this model)" if at_hip is not None else "")
        + ". Heat tolerance is set by the alloy, not by its porosity.",
        ((f"standard cycle for cast gamma TiAl: {_mpa(hip.pressure)}, {_celsius(hip.temperature)}, "
          f"{hip.time / 3600:.0f} h ({hip.source})",) if hip is not None else ())
        + ("fatigue life and service temperature: not modelled",),
        ("magform.porosity.HIP_PRACTICE", "magform.porosity.assess_pore_closure"))


# ---------------------------------------------------------------------------
# evaluators: containment
# ---------------------------------------------------------------------------
def _c26() -> _Evaluation:
    rail = containment.rail_launcher_load(4e6, 0.05, 0.05)
    fusion = containment.fusion_magnet_load(20.0, 12.2)
    rail_wall = containment.assess_containment(rail.bore_pressure, "maraging_c350", 0.025)
    fusion_wall = containment.assess_containment(fusion.coil_pressure, "steel_4340", 0.5)
    status = worst(rail_wall.status, fusion_wall.status)
    return _Evaluation(
        verdict_from_status(status),
        "These loads are real, and steel already carries them. A railgun at 4 MA puts "
        f"{_gpa(rail.bore_pressure)} on its bore, which a compound maraging-steel wall holds. The coils of a "
        f"20 T fusion magnet feel {_mpa(fusion.coil_pressure)}, held by a plain thick steel case. Neither needs "
        "a new material.",
        (f"rail launcher, 4 MA, 50 mm bore: {_gpa(rail.bore_pressure)} ({rail.effective_field:.1f} T equivalent); "
         f"maraging steel wall, b/a = {rail_wall.plastic_ratio:.2f}: {rail_wall.status.label}",
         f"fusion magnet, 20 T on the coil: {_mpa(fusion.coil_pressure)}; steel wall, b/a = "
         f"{fusion_wall.elastic_ratio:.2f}: {fusion_wall.status.label}"),
        ("magform.containment.rail_launcher_load", "magform.containment.fusion_magnet_load",
         "magform.containment.assess_containment"))


def _c27() -> _Evaluation:
    survey = {a.material: a for a in containment.containment_survey(P_HIGH, 0.025)}
    target = survey["cns_engineering_target"]
    fibre = survey["carbon_fibre_t1100g"]
    steel = survey["steel_4340"]
    return _Evaluation(
        verdict_from_status(target.status),
        f"At {_gpa(P_HIGH)} no real material makes a simple wall: the strength needed is "
        f"{_gpa(target.required_monolithic_strength)}, three times the strongest steel. A carbon-nanosheet "
        "laminate with the properties the note assumes would do it on paper, but no such material exists in "
        "bulk. Carbon laminates also have a specific weakness here. The bore presses on the wall sideways to "
        f"the fibres, where a laminate carries about {_gpa(containment.FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH)}, "
        "so a metal liner has to take the load anyway. The superconducting tape adds nothing to a housing.",
        (f"strength for any finite elastic wall at {_gpa(P_HIGH)} with a 1.5 safety factor: "
         f"{_gpa(target.required_monolithic_strength)}",
         f"steel 4340: {steel.status.label}; maraging steel: {survey['maraging_c350'].status.label} "
         f"(compound wall, b/a = {survey['maraging_c350'].plastic_ratio:.1f})",
         f"T1100G carbon fibre: {fibre.status.label}, transverse compression ok: {fibre.transverse_compression_ok}",
         f"carbon-nanosheet target: {target.status.label}; intrinsic graphene: "
         f"{survey['graphene_intrinsic'].status.label}"),
        ("magform.containment.containment_survey", "magform.containment.required_monolithic_strength"))


def _c28() -> _Evaluation:
    fusion = containment.fusion_magnet_load(20.0, 12.2)
    driver = containment.mass_driver(2400.0, "aluminum_6061", 1.0, 0.05, P_HIGH)
    false = fusion.ratio > 10.0 and driver.limited_by == "payload_strength"
    return _Evaluation(
        Verdict.CONTRADICTED if false else Verdict.EXTRAPOLATION,
        "The applications named do not need it. A fusion plasma pushes on its magnet with "
        f"{_mpa(fusion.plasma_pressure)}, {fusion.ratio:.0f} times less than the magnet pushes on itself, and "
        "the magnet's own load is within reach of steel. A mass driver is limited by what the payload "
        f"survives: an aluminium payload takes {_mpa(driver.allowed_pressure)} at its base, which sets a "
        f"{driver.launcher_length:.0f} m launcher for lunar escape. A bore able to hold {_gpa(P_HIGH)} would "
        "change neither number.",
        (f"fusion: coil {_mpa(fusion.coil_pressure)}, plasma {_mpa(fusion.plasma_pressure)}, ratio {fusion.ratio:.0f}",
         f"mass driver to 2.4 km/s: limited by {driver.limited_by.replace('_', ' ')} at "
         f"{_mpa(driver.allowed_pressure)}; length {driver.launcher_length:.1f} m; {driver.g_load:.0f} g",
         f"the {_gpa(P_HIGH)} bore capability is {P_HIGH / driver.payload_strength:.0f} times the payload's strength"),
        ("magform.containment.fusion_magnet_load", "magform.containment.mass_driver"))


# ---------------------------------------------------------------------------
# evaluators: the two questions
# ---------------------------------------------------------------------------
def _q1() -> _Evaluation:
    d = _design(P_HIGH)
    spec = reference_spec(P_HIGH)
    sol = spec.solenoid
    b = spec.field
    slow = d.network
    fast = pulse.assess_pulse_network(sol, b, 2.5e-5)
    hpg = pulse.homopolar_for_energy(d.stored_energy)
    hpg_rise = hpg.quarter_period(pulse.coil_electrical(sol, 1).inductance)
    product = pulse.voltage_current_product(d.stored_energy, 2.5e-5)
    status = slow.status if slow is not None else Status.INFEASIBLE
    evidence = [f"field energy of the reference press at {_gpa(P_HIGH)}: {d.stored_energy / 1e6:.1f} MJ",
                f"25 us rise: voltage x current = {_sci(product)} V A whatever the turn count "
                f"({fast.charge_voltage / 1e3:.0f} kV, {fast.peak_current / 1e6:.1f} MA, "
                f"{fast.peak_power / 1e12:.1f} TW): {fast.status.label}"]
    if slow is not None:
        evidence.append(f"{d.rise_time * 1e3:.0f} ms rise: {slow.turns} turns, {slow.charge_voltage / 1e3:.1f} kV, "
                        f"{slow.peak_current / 1e6:.2f} MA, {slow.peak_power / 1e9:.1f} GW, "
                        f"{slow.bank_energy / 1e6:.0f} MJ bank: {slow.status.label}")
    evidence.append(f"homopolar generator of the same energy: {hpg.voltage():.0f} V, fastest rise "
                    f"{hpg_rise * 1e3:.0f} ms into a single turn")
    return _Evaluation(
        verdict_from_status(status),
        "Yes, and it is mapped. The answer turns on how fast the field must rise. The energy is fixed by the "
        f"field and the bore, {d.stored_energy / 1e6:.0f} MJ for the reference press, and voltage times current "
        "is that energy over the rise time. A thick billet tolerates a "
        f"{d.rise_time * 1e3:.0f} ms rise, which a 25 kV capacitor bank twice the size of the largest "
        f"magnet bank can supply ({status.label}). A thin wall needs 25 microseconds, which means "
        f"{fast.peak_power / 1e12:.0f} TW and is out of reach. Homopolar generators hold the energy but release "
        f"it in tens of milliseconds at about a hundred volts: right for a slow outer coil, {hpg_rise / 2.5e-5:.0f} "
        "times too slow for a fast pulse.",
        tuple(evidence),
        ("magform.pulse.assess_pulse_network", "magform.pulse.voltage_current_product",
         "magform.pulse.homopolar_for_energy", "magform.press.design_press"))


def _q2() -> _Evaluation:
    m = _micro(None)
    cu = microstructure.assess_microstructure("copper_ofhc")
    return _Evaluation(
        verdict_from_status(m.status),
        f"Yes. For tungsten strained by {m.strain:g} in {m.strain / m.strain_rate * 1e6:.0f} microseconds, the "
        "model gives a dislocation density "
        f"up {m.dislocation_density / m.initial_dislocation_density:.0f}-fold arranged in cells about "
        f"{m.cell_size * 1e6:.1f} micrometres across, a yield strength up {_pct(m.hardening_ratio - 1.0)}, and a "
        f"temperature rise of {m.bulk_temperature_rise:.0f} K, far too little to recrystallise. The original "
        "grains are flattened and kept. The risk is localisation: tungsten at these rates concentrates strain "
        f"into shear bands almost at once. Copper under the same strain is ordinary cold work "
        f"({cu.status.label}).",
        (f"tungsten: dislocation density {_sci(m.dislocation_density)} per m^2, cells {m.cell_size * 1e6:.2f} um, "
         f"stored energy {m.stored_energy / 1e3:.0f} kJ/m^3",
         f"yield {_gpa(m.yield_strength_before)} -> {_gpa(m.yield_strength_after)}; hardness "
         f"{_gpa(m.hardness_before)} -> {_gpa(m.hardness_after)}",
         f"bulk heating +{m.bulk_temperature_rise:.0f} K; regime: {m.deformation_regime}; recrystallised fraction "
         f"{m.bulk_recrystallised_fraction:.1g}",
         f"shear localisation from strain {m.shear_localisation_strain:.3f}; status {m.status.label}",
         f"copper: {_sci(cu.dislocation_density)} per m^2, cells {cu.cell_size * 1e6:.2f} um, yield x "
         f"{cu.hardening_ratio:.1f}, {cu.status.label}"),
        ("magform.microstructure.assess_microstructure",))


_EVALUATORS: dict[str, Callable[[], _Evaluation]] = {
    "C01": _c01, "C02": _c02, "C03": _c03, "C04": _c04, "C05": _c05, "C06": _c06, "C07": _c07, "C08": _c08,
    "C09": _c09, "C10": _c10, "C11": _c11, "C12": _c12, "C13": _c13, "C14": _c14, "C15": _c15, "C16": _c16,
    "C17": _c17, "C18": _c18, "C19": _c19, "C20": _c20, "C21": _c21, "C22": _c22, "C23": _c23, "C24": _c24,
    "C25": _c25, "C26": _c26, "C27": _c27, "C28": _c28, "Q1": _q1, "Q2": _q2,
}
_SECTION_TITLES = dict(SECTIONS)
_BY_ID = {c.id: c for c in CLAIMS}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def get_claim(claim_id: str) -> Claim:
    """The :class:`Claim` with this id ("C01".."C28", "Q1", "Q2"; case-insensitive).

    Raises :class:`~magform.errors.InputError` for an unknown id.

    Reference: docs/archive/original-README.md.
    """
    key = str(claim_id).strip().upper()
    if key not in _BY_ID:
        raise InputError(f"unknown claim {claim_id!r}; ids run C01-C28, Q1, Q2")
    return _BY_ID[key]


@functools.lru_cache(maxsize=None)
def _evaluate(claim_id: str) -> ClaimResult:
    claim = _BY_ID[claim_id]
    result = _EVALUATORS[claim_id]()
    return ClaimResult(
        id=claim.id, section=claim.section, section_title=_SECTION_TITLES[claim.section], quote=claim.quote,
        topic=claim.topic, kind=claim.kind, verdict=result.verdict, finding=result.finding,
        evidence=result.evidence, models=result.models, chapter=CHAPTERS.get(claim.chapter, ""))


def evaluate(claim_id: str) -> ClaimResult:
    """Run the models behind one claim and return its verdict, finding and evidence.

    The verdict rules are in the module docstring; each evaluator's reference case is in its own
    docstring-free helper and named in the finding.  Results are cached for the life of the
    process (the models are deterministic).

    Reference: docs/archive/original-README.md; the modules listed in ``ClaimResult.models``.
    """
    return _evaluate(get_claim(claim_id).id)


def evaluate_all() -> tuple[ClaimResult, ...]:
    """Every claim and question of the concept note, evaluated, in the note's order.

    Thirty results: C01-C28 and Q1-Q2.  About a second of computation the first time.

    Reference: docs/archive/original-README.md.
    """
    return tuple(_evaluate(c.id) for c in CLAIMS)


def summary(results: Iterable[ClaimResult] | None = None) -> dict[str, int]:
    """Count of results per verdict value, in the order of :class:`~magform.status.Verdict`.

    ``results`` defaults to :func:`evaluate_all`.  Verdicts with no results are included with 0.

    Reference: magform.status.Verdict.
    """
    items = evaluate_all() if results is None else tuple(results)
    counts = {v.value: 0 for v in Verdict}
    for r in items:
        counts[r.verdict.value] += 1
    return counts


def by_section(results: Iterable[ClaimResult] | None = None) -> dict[str, tuple[ClaimResult, ...]]:
    """Results grouped by ledger section key, sections in the order of :data:`SECTIONS`.

    ``results`` defaults to :func:`evaluate_all`.

    Reference: docs/archive/original-README.md.
    """
    items = evaluate_all() if results is None else tuple(results)
    return {key: tuple(r for r in items if r.section == key) for key, _ in SECTIONS}


__all__ = [
    "REFERENCES", "P_LOW", "P_HIGH", "P_REFRACTORY", "ATOMIC_DATA", "SECTIONS", "CHAPTERS",
    "Claim", "CLAIMS", "ClaimResult", "get_claim", "evaluate", "evaluate_all", "summary", "by_section",
]
