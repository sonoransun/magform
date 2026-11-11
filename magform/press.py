"""The press as a system: field, coil, conductor, structure, power supply and workpiece.

Every other module answers one question.  This one asks them all of a single
machine: a solenoid of a given bore, length and build that must put a given
magnetic pressure on a workpiece.  :func:`design_press` sizes it and grades the
architecture the concept note proposes ("self-supporting, carbon-nanosheet-
reinforced superconducting magnets"); :func:`compare_architectures` grades the
other ways of making the same field so the reader can see which, if any, is
buildable.

What the integration shows that no single module does
-----------------------------------------------------
* **The time conflict.**  A steady field exerts no pressure
  (:func:`magform.magnetics.pressure_with_background`), so the field must rise
  within the workpiece's flux-diffusion time.  A thin wall needs tens of
  microseconds; the power supply and the coil want milliseconds.  The resistive
  pulsed-magnet option therefore reports two rise times: the one the workpiece
  needs and the fastest one the supply can deliver without leaving demonstrated
  practice, with the pressure that actually reaches the workpiece at each.
* **A superconducting background does not rescue the superconductor.**  A DC
  superconducting outsert around a pulsed resistive insert links the insert's
  flux.  A shorted (persistent) coil conserves its flux, so its field dips by
  ``dB * A_insert / A_background`` during the pulse, at kilotesla-per-second
  rates.  For inserts with a thick winding the dip exceeds the background field
  itself (:func:`background_coupling`), and the AC loss of that swing is checked
  against the enthalpy margin exactly as for the pulsed concept magnet.
* **The workpiece sees a different pressure from the naive one.**  The surface
  pressure on a finite billet comes from :mod:`magform.fieldmap` (flux squeezed
  into the gap, fall-off towards the coil ends, peaks at the billet's corners),
  multiplied by the eddy-current shielding of its wall at the pulse frequency.

Status of a :class:`PressDesign` is the status of the *concept* architecture
(pulsed carbon-nanosheet/REBCO magnet) as specified; ``best_architecture`` names
the least-bad alternative and ``best_status`` its status.

SI units throughout.  ``bore_diameter`` is the winding inner diameter.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from . import conductor, eddy, fieldmap, pulse
from ._numerics import logspace
from ._report import Assessment, ResultMixin, unit
from .benchmarks import (
    EMF_CONVENTIONAL_FIELD,
    PULSED_NONDESTRUCTIVE_BORE,
    PULSED_NONDESTRUCTIVE_RECORD,
    SC_CONDUCTOR_RECORD,
    SC_FIELD_RECORD,
    Placement,
    energy_status,
    peak_power_status,
    pulsed_current_status,
    pulsed_field_status,
    pulsed_voltage_status,
    ramp_rate_status,
    where_does_it_sit,
)
from .conductor import CNSComposite, ConductorAssessment, QuenchAssessment, StructureAssessment
from .constants import MU0, T_LIQUID_HELIUM, T_ROOM
from .errors import InputError
from .magnetics import Solenoid, field_for_pressure, magnetic_pressure, pulse_increment_for_pressure, virial_minimum_mass
from .materials import MaterialLike, as_material
from .pulse import PulseNetworkAssessment
from .status import Status, worst

REFERENCES: dict[str, str] = {
    "herlach2003": "F. Herlach and N. Miura (eds.), High Magnetic Fields: Science and Technology, Vol. 1, "
                   "World Scientific, Singapore (2003): pulsed, single-turn and flux-compression magnets.",
    "psyk2011": "V. Psyk, D. Risch, B. L. Kinsey, A. E. Tekkaya and M. Kleiner, 'Electromagnetic forming - "
                "a review', J. Mater. Process. Technol. 211 (2011) 787-829.",
    "miller2003": "J. R. Miller, 'The NHMFL 45-T hybrid magnet system: past, present, and future', "
                  "IEEE Trans. Appl. Supercond. 13 (2003) 1385-1390 (11.5 T superconducting outsert, 616 mm bore).",
    "hahn2019": "S. Hahn et al., '45.5-tesla direct-current magnetic field generated with a high-temperature "
                "superconducting magnet', Nature 570 (2019) 496-499.",
    "maglab2025": "National High Magnetic Field Laboratory, news release of 11 Sep 2025 and research highlight "
                  "of Jan 2026: REBCO test coil 'Little Big Coil 9', 48.7 T inside a 31 T resistive magnet "
                  "(August 2025).",
    "wilson1983": "M. N. Wilson, Superconducting Magnets, Clarendon Press, Oxford (1983): flux conservation in "
                  "coupled windings; AC loss and stability.",
    "iwasa2009": "Y. Iwasa, Case Studies in Superconducting Magnets, 2nd ed., Springer, New York (2009).",
    "montgomery1969": "D. B. Montgomery, Solenoid Magnet Design, Wiley-Interscience, New York (1969).",
    "knoepfel2000": "H. Knoepfel, Magnetic Fields: A Comprehensive Theoretical Treatise for Practical Use, "
                    "Wiley, New York (2000).",
    "benchmarks": "magform.benchmarks: cited record ladders for field, energy, current, voltage and power.",
}

ARCHITECTURES: tuple[str, ...] = (
    "hts_pulsed", "hts_dc", "pulsed_resistive", "hybrid_dc_pulsed", "single_turn", "industrial_emf",
)
"""Keys of :func:`compare_architectures`, in the order returned.  The first two are the concept
note's magnet, pulsed and steady."""

DEFAULT_WORKPIECE_RADIUS_FRACTION: float = 0.8
"""Workpiece radius as a fraction of the bore radius when none is given."""

DEFAULT_WORKPIECE_LENGTH_FRACTION: float = 0.6
"""Workpiece length as a fraction of the coil length when none is given."""

LARGE_BORE_SC_FIELD: float = 11.5
"""T.  Field of the superconducting outsert of the NHMFL 45 T hybrid in its 616 mm bore (Miller 2003):
the demonstrated class of large-bore DC superconducting background magnets."""

LARGE_BORE_SC_DIAMETER: float = 0.616
"""m.  Bore of that outsert."""

SMALL_BORE_SC_DIAMETER: float = 0.04
"""m.  Bore class (34-40 mm) of the 32-36 T all-superconducting records in benchmarks.FIELD_LADDER."""

BACKGROUND_CLEARANCE: float = 1.1
"""Inner radius of a background coil as a multiple of the insert's outer radius (cryostat wall and
clearance; estimated)."""

BACKGROUND_BUILD: float = 0.1
"""m.  Radial build assumed for the background winding when its current density is needed (estimated)."""

SINGLE_TURN_FIELD: float = 300.0
"""T.  Fields reached by destructive single-turn coils (benchmarks.FIELD_LADDER 'single_turn_300t')."""

SINGLE_TURN_BORE: float = 0.02
"""m.  Largest bore class of existing single-turn coil systems (5-20 mm; Herlach & Miura 2003)."""

SINGLE_TURN_RISE_TIME: float = 2.5e-6
"""s.  Rise time of single-turn coil discharges (2-3 microseconds; Herlach & Miura 2003)."""

SINGLE_TURN_WALL: float = 3e-3
"""m.  Wall of a single-turn coil (copper plate, about 3 mm; representative)."""

INDUSTRIAL_COIL_BUILD: float = 0.02
"""m.  Radial build assumed for an industrial forming coil (single layer, representative)."""

INDUSTRIAL_RISE_TIME: float = 5e-5
"""s.  Slowest field rise of industrial electromagnetic forming (discharge frequencies of 5-30 kHz,
i.e. quarter periods of 8-50 microseconds; Psyk et al. 2011)."""

DELIVERY_THRESHOLD: float = 0.9
"""Fraction of the asked pressure an architecture must put on the workpiece to count as delivering it."""

RISE_SCAN: tuple[float, float, int] = (-6.5, -1.0, 34)
"""log10 of the rise-time range (s) and the number of samples scanned for the fastest rise the
power supply can deliver without leaving demonstrated practice."""


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(v):
        raise InputError(f"{name} must be finite, got {value!r}")
    return v


def _positive(name: str, value: float) -> float:
    v = _finite(name, value)
    if v <= 0.0:
        raise InputError(f"{name} must be positive, got {v!r}")
    return v


def _gpa(value: float) -> str:
    return f"{value / 1e9:.3g} GPa"


def _mj(value: float) -> str:
    return f"{value / 1e6:.3g} MJ"


def _time(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.3g} us"
    if value < 1.0:
        return f"{value * 1e3:.3g} ms"
    return f"{value:.3g} s"


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PressSpec(ResultMixin):
    """What the press must do and the envelope of its coil.

    ``pressure`` [Pa] is the magnetic pressure ``B0^2/(2 mu0)`` of the coil's free-space central
    field (the number the concept note quotes: 1.6e9 to 4e9).  ``bore_diameter`` (winding inner
    diameter), ``length`` and ``winding_thickness`` in m.  Workpiece: ``workpiece`` material,
    ``workpiece_radius`` (default 0.8 x bore radius), ``workpiece_length`` (default 0.6 x coil
    length) and ``workpiece_wall`` (None = solid billet).  ``rise_time`` [s] of the field (None:
    the quarter period that puts the skin depth at one third of the wall,
    :func:`magform.eddy.pulse_frequency_recommendation`).  Superconducting winding:
    ``temperature`` [K], ``composite`` (None = :meth:`CNSComposite.default`), ``safety_factor``
    (>= 1) on strain and critical current.  ``background_field`` [T] is the DC field of the
    superconducting outsert in the hybrid option (default the demonstrated large-bore class, 11.5 T).

    Reference: Montgomery (1969); Psyk et al. (2011).
    """

    pressure: float = unit("Pa")
    bore_diameter: float = unit("m")
    length: float = unit("m")
    winding_thickness: float = unit("m")
    workpiece: MaterialLike = "copper_ofhc"
    workpiece_radius: float | None = unit("m", default=None)
    workpiece_length: float | None = unit("m", default=None)
    workpiece_wall: float | None = unit("m", default=None)
    rise_time: float | None = unit("s", default=None)
    temperature: float = unit("K", default=T_LIQUID_HELIUM)
    safety_factor: float = 1.5
    composite: CNSComposite | None = None
    background_field: float = unit("T", default=LARGE_BORE_SC_FIELD)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pressure", _positive("pressure", self.pressure))
        object.__setattr__(self, "bore_diameter", _positive("bore_diameter", self.bore_diameter))
        object.__setattr__(self, "length", _positive("length", self.length))
        object.__setattr__(self, "winding_thickness", _positive("winding_thickness", self.winding_thickness))
        mat = as_material(self.workpiece)
        if not mat.is_metal:
            raise InputError(f"workpiece '{mat.key}' is not a metal: eddy currents cannot carry the load")
        if isinstance(self.workpiece, str):
            object.__setattr__(self, "workpiece", mat.key)
        a1 = 0.5 * self.bore_diameter
        if self.workpiece_radius is not None:
            r = _positive("workpiece_radius", self.workpiece_radius)
            if r >= a1:
                raise InputError(f"workpiece_radius ({r!r} m) must be smaller than the bore radius ({a1!r} m)")
            object.__setattr__(self, "workpiece_radius", r)
        if self.workpiece_length is not None:
            object.__setattr__(self, "workpiece_length", _positive("workpiece_length", self.workpiece_length))
        if self.workpiece_wall is not None:
            w = _positive("workpiece_wall", self.workpiece_wall)
            if w > self.billet_radius * (1.0 + 1e-12):
                raise InputError("workpiece_wall cannot exceed the workpiece radius")
            object.__setattr__(self, "workpiece_wall", min(w, self.billet_radius))
        if self.rise_time is not None:
            object.__setattr__(self, "rise_time", _positive("rise_time", self.rise_time))
        object.__setattr__(self, "temperature", _positive("temperature", self.temperature))
        sf = _finite("safety_factor", self.safety_factor)
        if sf < 1.0:
            raise InputError(f"safety_factor must be >= 1, got {sf!r}")
        object.__setattr__(self, "safety_factor", sf)
        if self.composite is not None and not isinstance(self.composite, CNSComposite):
            raise InputError("composite must be a CNSComposite or None")
        bg = _finite("background_field", self.background_field)
        if bg < 0.0:
            raise InputError("background_field must be non-negative")
        object.__setattr__(self, "background_field", bg)

    @property
    def solenoid(self) -> Solenoid:
        """The coil envelope as a :class:`magform.magnetics.Solenoid`."""
        a1 = 0.5 * self.bore_diameter
        return Solenoid(a1, a1 + self.winding_thickness, self.length)

    @property
    def field(self) -> float:
        """Central field whose magnetic pressure is ``pressure``, ``sqrt(2 mu0 P)`` [T]."""
        return field_for_pressure(self.pressure)

    @property
    def billet_radius(self) -> float:
        """Workpiece outer radius [m] (given, or 0.8 x bore radius)."""
        if self.workpiece_radius is not None:
            return self.workpiece_radius
        return DEFAULT_WORKPIECE_RADIUS_FRACTION * 0.5 * self.bore_diameter

    @property
    def billet_length(self) -> float:
        """Workpiece length [m] (given, or 0.6 x coil length)."""
        if self.workpiece_length is not None:
            return self.workpiece_length
        return DEFAULT_WORKPIECE_LENGTH_FRACTION * self.length

    @property
    def wall(self) -> float:
        """Conducting wall thickness [m]: ``workpiece_wall``, or the radius for a solid billet."""
        return self.workpiece_wall if self.workpiece_wall is not None else self.billet_radius

    @property
    def matched_rise_time(self) -> float:
        """Rise time [s] the workpiece needs: given, or skin depth = wall/3 at room temperature."""
        if self.rise_time is not None:
            return self.rise_time
        return eddy.pulse_frequency_recommendation(self.workpiece, self.wall).quarter_period


# ---------------------------------------------------------------------------
# workpiece
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class WorkpieceLoad(ResultMixin):
    """What the pulse does to the workpiece.

    ``skin_depth`` at f = 1/(4 t_rise) and room temperature; ``pressure_transfer`` =
    ``1 - exp(-2 wall/skin_depth)`` (:func:`magform.eddy.sheet_shielding_factor`).  Surface
    distribution from :func:`magform.fieldmap.workpiece_pressure` (perfect-conductor limit):
    ``surface_mid_ratio`` (mid-plane pressure / naive ``B0^2/2mu0``), ``surface_uniformity``
    (min/max over the central 80 % of the length) and ``surface_peak_ratio`` (at the billet's
    corners); None when the billet does not fit the model.  ``delivered_pressure`` =
    pressure x transfer x mid ratio.  ``skin_temperature_rise`` is the eddy-current heating of the
    surface layer (kappa = 1, a lower estimate) and ``skin_melts`` whether it reaches the
    melting point from room temperature.  ``elastic_volumetric_strain`` = P/K.

    Reference: Psyk et al. (2011); Knoepfel (2000); Harrington (1968) via magform.fieldmap.
    """

    material: str
    radius: float = unit("m")
    length: float = unit("m")
    wall: float = unit("m")
    solid: bool
    rise_time: float = unit("s")
    frequency: float = unit("Hz")
    skin_depth: float = unit("m")
    regime: str
    pressure_transfer: float
    surface_mid_ratio: float | None
    surface_uniformity: float | None
    surface_peak_ratio: float | None
    delivered_pressure: float = unit("Pa")
    skin_temperature_rise: float = unit("K")
    skin_melting_field: float = unit("T")
    skin_melts: bool
    elastic_volumetric_strain: float


def _transfer(spec: PressSpec, rise_time: float) -> tuple[float, float]:
    """(pressure-transfer factor, skin depth) of the workpiece wall for a field rising in ``rise_time``."""
    delta = eddy.skin_depth_for_material(spec.workpiece, 0.25 / rise_time)
    return eddy.sheet_shielding_factor(spec.wall, delta), delta


def workpiece_load(spec: PressSpec, rise_time: float | None = None) -> WorkpieceLoad:
    """Pressure delivered to the workpiece of ``spec`` and the heating of its skin.

    ``delivered = P * (1 - exp(-2 h/delta)) * mid_ratio`` with ``delta`` the skin depth at
    ``f = 1/(4 t_rise)``, ``h`` the wall and ``mid_ratio`` from the boundary-element surface
    solution; skin heating ``dT = B^2/(2 mu0 rho c_p)``.  ``rise_time`` [s] defaults to
    ``spec.matched_rise_time``.  Validity: quasi-steady shielding factor (+-20 % near h ~ delta),
    rigid workpiece; the surface solution assumes a perfect conductor, so the product is an
    upper estimate for thin walls.

    Reference: Psyk et al. (2011); Knoepfel (2000).
    """
    if not isinstance(spec, PressSpec):
        raise InputError("spec must be a PressSpec")
    t_rise = spec.matched_rise_time if rise_time is None else _positive("rise_time", rise_time)
    mat = as_material(spec.workpiece)
    transfer, delta = _transfer(spec, t_rise)
    b = spec.field
    mid = uniformity = peak = None
    try:
        surface = fieldmap.workpiece_pressure(spec.solenoid, spec.billet_radius, spec.billet_length)
        mid, uniformity, peak = surface.mid_ratio, surface.uniformity, surface.peak_ratio
    except InputError:
        pass
    d_t = eddy.skin_temperature_rise(mat, b)
    b_melt = eddy.skin_melting_field(mat)
    return WorkpieceLoad(
        material=mat.key,
        radius=spec.billet_radius,
        length=spec.billet_length,
        wall=spec.wall,
        solid=spec.workpiece_wall is None,
        rise_time=t_rise,
        frequency=0.25 / t_rise,
        skin_depth=delta,
        regime="thin" if spec.wall <= delta else "thick",
        pressure_transfer=transfer,
        surface_mid_ratio=mid,
        surface_uniformity=uniformity,
        surface_peak_ratio=peak,
        delivered_pressure=spec.pressure * transfer * (mid if mid is not None else 1.0),
        skin_temperature_rise=d_t,
        skin_melting_field=b_melt,
        skin_melts=b >= b_melt,
        elastic_volumetric_strain=mat.volumetric_strain(spec.pressure),
    )


# ---------------------------------------------------------------------------
# hybrid coupling
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class BackgroundCoupling(ResultMixin):
    """Flux coupling between a pulsed insert and a superconducting background coil around it.

    ``insert_flux`` = ``dB * pi (a1^2 + a1 Delta + Delta^2/3)`` (long-coil field profile of the
    insert); ``background_flux`` = ``B_dc * pi R_bg^2``; ``flux_ratio`` their ratio.  A shorted
    (persistent) background conserves its flux linkage, so its field changes by ``field_dip`` =
    ``dB * A_insert / A_background`` during the pulse (``reverses`` when the dip exceeds B_dc);
    an open or supply-held winding sees instead ``volts_per_turn`` = insert flux / rise time on
    every turn.  ``loss`` and ``enthalpy_margin`` [J/m^3 of tape] are the AC loss of the dip
    (capped at 2 B_dc) and the adiabatic margin of the background conductor at B_dc;
    ``loss_to_margin_ratio`` > 1 means the background quenches unless the insert is
    flux-compensated.

    Reference: Wilson (1983); Iwasa (2009).
    """

    background_field: float = unit("T")
    background_radius: float = unit("m")
    pulse_increment: float = unit("T")
    rise_time: float = unit("s")
    insert_flux: float = unit("Wb")
    background_flux: float = unit("Wb")
    flux_ratio: float
    field_dip: float = unit("T")
    reverses: bool
    dip_rate: float = unit("T/s")
    volts_per_turn: float = unit("V")
    loss: float = unit("J/m^3")
    enthalpy_margin: float = unit("J/m^3")
    loss_to_margin_ratio: float
    ramp_rate_status: Status
    status: Status


def background_coupling(insert: Solenoid, pulse_increment: float, background_field: float, rise_time: float,
                        temperature: float = T_LIQUID_HELIUM, composite: CNSComposite | None = None,
                        background_radius: float | None = None) -> BackgroundCoupling:
    """Can a superconducting background coil sit around a pulsed insert?

    Flux of the insert through the background: ``Phi = dB pi (a1^2 + a1 Delta + Delta^2/3)``
    (bore plus the linearly falling winding field).  Flux conservation in a shorted coil of radius
    ``R_bg`` gives ``dip = Phi / (pi R_bg^2)``.  The AC loss of that swing
    (:func:`magform.conductor.tape_pulse_losses`) is compared with the adiabatic enthalpy margin of
    the background conductor at ``background_field`` (winding-average current density
    ``B_dc/(mu0 x 0.1 m)``).  ``pulse_increment`` dB [T] > 0, ``background_field`` [T] >= 0,
    ``rise_time`` [s] > 0; ``background_radius`` defaults to 1.1 x the insert's outer radius.

    Status: no background -> not applicable; loss/margin > 1 -> infeasible (quenches every pulse
    unless the insert is flux-compensated); > 0.3 -> at least extrapolation; then worst-combined
    with ``benchmarks.ramp_rate_status(dip / rise_time)``.  Validity: long coaxial coils, REBCO
    conductor (the generous case: low-temperature superconductors have far less margin), no eddy
    shield between the coils.

    Reference: Wilson (1983); Iwasa (2009).
    """
    if not isinstance(insert, Solenoid):
        raise InputError("insert must be a Solenoid")
    d_b = _positive("pulse_increment", pulse_increment)
    b_dc = _finite("background_field", background_field)
    if b_dc < 0.0:
        raise InputError("background_field must be non-negative")
    t_rise = _positive("rise_time", rise_time)
    t_op = _positive("temperature", temperature)
    comp = composite if composite is not None else CNSComposite.default()
    r_bg = BACKGROUND_CLEARANCE * insert.outer_radius if background_radius is None \
        else _positive("background_radius", background_radius)
    if r_bg < insert.outer_radius:
        raise InputError("background_radius must not be smaller than the insert's outer radius")
    a1, d = insert.inner_radius, insert.winding_thickness
    insert_flux = d_b * math.pi * (a1 * a1 + a1 * d + d * d / 3.0)
    area_bg = math.pi * r_bg * r_bg
    bg_flux = b_dc * area_bg
    dip = insert_flux / area_bg
    volts = insert_flux / t_rise
    if b_dc == 0.0:
        return BackgroundCoupling(
            background_field=0.0, background_radius=r_bg, pulse_increment=d_b, rise_time=t_rise,
            insert_flux=insert_flux, background_flux=0.0, flux_ratio=math.inf, field_dip=dip, reverses=False,
            dip_rate=dip / t_rise, volts_per_turn=volts, loss=0.0, enthalpy_margin=0.0, loss_to_margin_ratio=0.0,
            ramp_rate_status=Status.NOT_APPLICABLE, status=Status.NOT_APPLICABLE)
    swing = min(dip, 2.0 * b_dc)
    tape = comp.tape
    loss = conductor.tape_pulse_losses(swing, t_rise, t_op, tape).q_total
    j_tape = b_dc / (MU0 * BACKGROUND_BUILD) / comp.f_hts if comp.f_hts > 0.0 else math.inf
    if t_op >= tape.tc or not math.isfinite(j_tape):
        margin_h = 0.0
    else:
        t_cs = tape.current_sharing_temperature(b_dc, j_tape, "parallel", lowest_temperature=min(T_LIQUID_HELIUM, t_op))
        margin_h = conductor.enthalpy_margin(t_op, t_cs, tape) if t_cs > t_op else 0.0
    ratio = loss / margin_h if margin_h > 0.0 else math.inf
    r_status = ramp_rate_status(dip / t_rise)
    if ratio > 1.0:
        status = Status.INFEASIBLE
    elif ratio > conductor.LOSS_MARGIN_CAUTION:
        status = worst(Status.EXTRAPOLATION, r_status)
    else:
        status = r_status
    return BackgroundCoupling(
        background_field=b_dc, background_radius=r_bg, pulse_increment=d_b, rise_time=t_rise,
        insert_flux=insert_flux, background_flux=bg_flux, flux_ratio=insert_flux / bg_flux, field_dip=dip,
        reverses=dip > b_dc, dip_rate=dip / t_rise, volts_per_turn=volts, loss=loss, enthalpy_margin=margin_h,
        loss_to_margin_ratio=ratio, ramp_rate_status=r_status, status=status)


def background_magnet_status(field: float, bore_diameter: float) -> Status:
    """Status of a DC superconducting background magnet of a given field and bore.

    State of the art: <= 11.5 T in bores up to 616 mm (hybrid-magnet outserts, Miller 2003), or up
    to the all-superconducting record (``benchmarks.SC_FIELD_RECORD``) in bores <= 40 mm.
    Extrapolation: up to the highest field at which a superconducting coil has operated
    (``benchmarks.SC_CONDUCTOR_RECORD``, 48.7 T) at any bore.  Aspirational above.  ``field`` [T]
    >= 0, ``bore_diameter`` [m] > 0.  A two-point envelope, not a scaling law: stored energy and
    hoop force grow with bore, so "extrapolation" at large bore and high field is generous.

    Reference: Miller (2003); Hahn et al. (2019); NHMFL (2025); magform.benchmarks.
    """
    b = _finite("field", field)
    if b < 0.0:
        raise InputError("field must be non-negative")
    bore = _positive("bore_diameter", bore_diameter)
    if b <= LARGE_BORE_SC_FIELD and bore <= LARGE_BORE_SC_DIAMETER:
        return Status.STATE_OF_THE_ART
    if b <= SC_FIELD_RECORD and bore <= SMALL_BORE_SC_DIAMETER:
        return Status.STATE_OF_THE_ART
    if b <= SC_CONDUCTOR_RECORD:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


# ---------------------------------------------------------------------------
# architectures
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class ArchitectureAssessment(Assessment):
    """One way of building the press, graded.

    ``field`` is the peak field at the workpiece; ``rise_time`` the field rise it is assessed at;
    ``energy`` the energy one shot draws from its store; ``pressure_fraction`` the delivered
    mid-plane pressure divided by the pressure asked for (0 for a steady field; < 1 when the wall
    is transparent to a slow pulse or the field is capped); ``reusable`` False when the coil is
    consumed by the shot; ``components`` maps each subsystem to its status and ``limiting`` names
    the worst.  ``margin`` is ``pressure_fraction``.  ``details`` holds the architecture's own key
    numbers in SI units (turns, voltage, current, power; the hybrid's flux ratio and field dip).
    For the resistive pulsed magnet, when the supply cannot deliver the rise the workpiece needs,
    ``relaxed_rise_time`` is the fastest rise it can deliver within demonstrated practice, with
    ``relaxed_status`` and ``relaxed_pressure_fraction`` at that rise (None otherwise).

    Reference: Herlach & Miura (2003); Psyk et al. (2011); magform.benchmarks.
    """

    key: str
    name: str
    summary: str
    field: float = unit("T")
    rise_time: float | None = unit("s")
    energy: float | None = unit("J")
    pressure_fraction: float
    reusable: bool
    limiting: str
    components: dict[str, str]
    details: dict[str, float]
    relaxed_rise_time: float | None = unit("s", default=None)
    relaxed_status: Status | None = None
    relaxed_pressure_fraction: float | None = None


def _limiting(components: dict[str, Status]) -> tuple[Status, str]:
    status = worst(components.values())
    names = [name for name, s in components.items() if s is status]
    return status, (names[0] if names else "none")


def _network(solenoid: Solenoid, field: float, rise_time: float, **kwargs) -> PulseNetworkAssessment | None:
    try:
        return pulse.assess_pulse_network(solenoid, field, rise_time, **kwargs)
    except InputError:
        return None


def _network_limits(net: PulseNetworkAssessment) -> str:
    """The network's components that are not state of the art, worst first, as text."""
    parts = [("bank energy", net.energy_status), ("charge voltage", net.voltage_status),
             ("peak current", net.current_status), ("peak power", net.power_status),
             ("coil heating", net.thermal_status)]
    flagged = sorted((p for p in parts if p[1] is not Status.STATE_OF_THE_ART), key=lambda p: -p[1].rank)
    return ", ".join(f"{name} {status.label}" for name, status in flagged) or "all within demonstrated practice"


@dataclass(frozen=True)
class _Context:
    spec: PressSpec
    solenoid: Solenoid
    field: float
    current_density: float
    energy: float
    hoop_peak: float
    hoop_radius: float
    ends: fieldmap.EndField
    rise: float
    load: WorkpieceLoad
    composite: CNSComposite
    conductor: ConductorAssessment
    structure: StructureAssessment
    pulsed: QuenchAssessment
    network: PulseNetworkAssessment | None


def _context(spec: PressSpec) -> _Context:
    if not isinstance(spec, PressSpec):
        raise InputError("spec must be a PressSpec")
    sol = spec.solenoid
    b = spec.field
    j = sol.current_density_for_field(b)
    hoop_peak, hoop_r = sol.max_hoop_stress(j, b)
    ends = fieldmap.end_field(sol)
    rise = spec.matched_rise_time
    comp = spec.composite if spec.composite is not None else CNSComposite.default()
    cond = conductor.assess_conductor(ends.peak_field_ratio * b, spec.temperature, j, comp, spec.safety_factor,
                                      end_field_fraction=min(ends.fraction, 1.0))
    struct = conductor.assess_structure(hoop_peak, comp, spec.safety_factor)
    pulsed = conductor.assess_pulsed_operation(b, rise, j, spec.temperature, comp,
                                               perpendicular_fraction=min(ends.fraction, 1.0))
    return _Context(
        spec=spec, solenoid=sol, field=b, current_density=j, energy=sol.stored_energy(b), hoop_peak=hoop_peak,
        hoop_radius=hoop_r, ends=ends, rise=rise, load=workpiece_load(spec, rise), composite=comp, conductor=cond,
        structure=struct, pulsed=pulsed, network=_network(sol, b, rise))


def _hts_pulsed(ctx: _Context) -> ArchitectureAssessment:
    components = {"conductor": ctx.conductor.status, "structure": ctx.structure.status,
                  "pulsed operation": ctx.pulsed.status}
    status, limiting = _limiting(components)
    fraction = ctx.load.delivered_pressure / ctx.spec.pressure
    warnings = []
    if ctx.pulsed.status is Status.INFEASIBLE:
        ratio = ctx.pulsed.loss_to_margin_ratio
        warnings.append("the winding quenches every pulse: " + (
            f"per-pulse loss is {ratio:.3g}x its enthalpy margin" if math.isfinite(ratio)
            else "the tape cannot carry the current at this field even before it is pulsed (no enthalpy margin)"))
    if ctx.structure.status is Status.INFEASIBLE:
        warnings.append(f"no reinforcement keeps the REBCO within its strain limit at "
                        f"{_gpa(ctx.hoop_peak)} of hoop stress")
    notes = (
        f"needs {ctx.field:.4g} T from a superconducting winding: the all-superconducting record is "
        f"{SC_FIELD_RECORD:g} T and no superconductor has carried current above {SC_CONDUCTOR_RECORD:g} T",
        f"peak hoop stress {_gpa(ctx.hoop_peak)} needs a composite modulus of "
        f"{ctx.structure.required_composite_modulus / 1e9:.4g} GPa at the 0.4 % strain limit",
        f"field must rise in {_time(ctx.rise)} for this workpiece: {ctx.pulsed.ramp_rate:.3g} T/s",
    )
    return ArchitectureAssessment(
        status=status, margin=fraction, warnings=tuple(warnings), notes=notes,
        key="hts_pulsed", name="Concept magnet, pulsed",
        summary="Carbon-nanosheet-reinforced REBCO solenoid swung to full field every shot (the concept note's press).",
        field=ctx.field, rise_time=ctx.rise, energy=ctx.energy, pressure_fraction=fraction, reusable=True,
        limiting=limiting, components={k: v.value for k, v in components.items()},
        details={"hoop_stress": ctx.hoop_peak, "required_modulus": ctx.structure.required_composite_modulus,
                 "ramp_rate": ctx.pulsed.ramp_rate, "loss_to_margin_ratio": ctx.pulsed.loss_to_margin_ratio,
                 "conductor_margin": ctx.conductor.margin if ctx.conductor.margin is not None else math.nan})


def _hts_dc(ctx: _Context) -> ArchitectureAssessment:
    components = {"conductor": ctx.conductor.status, "structure": ctx.structure.status,
                  "pressure on the workpiece": Status.INFEASIBLE}
    notes = (
        "a steady field has diffused into the workpiece: (B_out^2 - B_in^2)/(2 mu0) = 0, so the magnet "
        "exerts no forming pressure however strong it is",
        f"as a magnet alone it would still be {worst(ctx.conductor.status, ctx.structure.status).label} "
        "(conductor and structure)",
    )
    return ArchitectureAssessment(
        status=Status.INFEASIBLE, margin=0.0,
        warnings=("a steady field exerts no pressure on a stationary, non-magnetic workpiece",), notes=notes,
        key="hts_dc", name="Concept magnet, steady",
        summary="The same superconducting solenoid held at constant field.",
        field=ctx.field, rise_time=None, energy=ctx.energy, pressure_fraction=0.0, reusable=True,
        limiting="pressure on the workpiece", components={k: v.value for k, v in components.items()},
        details={"delivered_pressure": 0.0})


def _fastest_practical_rise(ctx: _Context, field: float) -> tuple[float, PulseNetworkAssessment] | None:
    """Shortest scanned rise time at which the supply and coil are no worse than extrapolation."""
    lo, hi, n = RISE_SCAN
    for t in logspace(lo, hi, n):
        net = _network(ctx.solenoid, field, t)
        if net is not None and net.status in (Status.STATE_OF_THE_ART, Status.EXTRAPOLATION):
            return t, net
    return None


def _pulsed_resistive(ctx: _Context) -> ArchitectureAssessment:
    spec = ctx.spec
    field_status = pulsed_field_status(ctx.field, spec.bore_diameter)
    net = ctx.network
    components = {"field and bore": field_status,
                  "power supply and coil": net.status if net is not None else Status.INFEASIBLE}
    status, limiting = _limiting(components)
    fraction = ctx.load.delivered_pressure / spec.pressure
    warnings: list[str] = []
    notes: list[str] = [
        f"{ctx.field:.4g} T in a {spec.bore_diameter * 1e3:.4g} mm bore: {field_status.label} "
        f"(the {PULSED_NONDESTRUCTIVE_RECORD:g} T record is in a {PULSED_NONDESTRUCTIVE_BORE * 1e3:g} mm bore; "
        "stored energy grows with bore volume)",
    ]
    energy = ctx.energy
    if net is not None:
        energy = net.bank_energy
        notes.append(f"at the {_time(ctx.rise)} rise the workpiece needs: {net.turns} turn(s), "
                     f"{net.charge_voltage / 1e3:.4g} kV, {net.peak_current / 1e6:.4g} MA, "
                     f"{net.peak_power / 1e9:.4g} GW, {_mj(net.bank_energy)} bank ({net.status.label})")
        if net.status is Status.INFEASIBLE:
            warnings.append(f"at the rise time the workpiece needs, the supply or the coil fails "
                            f"({_network_limits(net)})")
    else:
        warnings.append("no capacitor discharge reaches this field in the rise time the workpiece needs")
    details: dict[str, float] = {}
    if net is not None:
        details = {"turns": float(net.turns), "charge_voltage": net.charge_voltage, "peak_current": net.peak_current,
                   "peak_power": net.peak_power, "bank_energy": net.bank_energy,
                   "conductor_final_temperature": net.conductor_final_temperature}
    fastest = _fastest_practical_rise(ctx, ctx.field)
    relaxed_rise = relaxed_fraction = None
    relaxed_status: Status | None = None
    if fastest is None:
        notes.append("no rise time between 0.3 us and 100 ms brings the supply within demonstrated practice")
    else:
        details["fastest_practical_rise_time"] = fastest[0]
        transfer, _ = _transfer(spec, fastest[0])
        mid = ctx.load.surface_mid_ratio if ctx.load.surface_mid_ratio is not None else 1.0
    if fastest is not None and (fastest[0] <= ctx.rise * (1.0 + 1e-9) or transfer * mid >= 0.95 * fraction):
        notes.append(f"the supply could go as fast as {_time(fastest[0])} within demonstrated practice: "
                     "it keeps up with this workpiece")
    elif fastest is not None:
        relaxed_rise, relaxed_net = fastest
        relaxed_fraction = transfer * mid
        relaxed_status = worst(field_status, relaxed_net.status)
        warnings.append(f"time conflict: the workpiece needs a {_time(ctx.rise)} rise, the supply's fastest within "
                        f"demonstrated practice is {_time(relaxed_rise)}, and at that rise the wall passes only "
                        f"{relaxed_fraction:.0%} of the pressure")
    return ArchitectureAssessment(
        status=status, margin=fraction, warnings=tuple(warnings), notes=tuple(notes),
        key="pulsed_resistive", name="Resistive pulsed magnet",
        summary="Reinforced copper-alloy coil driven by a capacitor bank, the technology of the 100 T record.",
        field=ctx.field, rise_time=ctx.rise, energy=energy, pressure_fraction=fraction, reusable=True,
        limiting=limiting, components={k: v.value for k, v in components.items()}, details=details,
        relaxed_rise_time=relaxed_rise, relaxed_status=relaxed_status, relaxed_pressure_fraction=relaxed_fraction)


def _hybrid(ctx: _Context) -> ArchitectureAssessment:
    spec = ctx.spec
    b_dc = spec.background_field
    d_b = pulse_increment_for_pressure(spec.pressure, b_dc)
    b_total = b_dc + d_b
    r_bg = BACKGROUND_CLEARANCE * ctx.solenoid.outer_radius
    field_status = pulsed_field_status(b_total, spec.bore_diameter)
    net = _network(ctx.solenoid, d_b, ctx.rise)
    bg_status = background_magnet_status(b_dc, 2.0 * r_bg) if b_dc > 0.0 else Status.NOT_APPLICABLE
    coupling = background_coupling(ctx.solenoid, d_b, b_dc, ctx.rise, spec.temperature, ctx.composite, r_bg)
    components = {"insert field and bore": field_status,
                  "insert power supply and coil": net.status if net is not None else Status.INFEASIBLE,
                  "background magnet": bg_status,
                  "background coupling": coupling.status}
    status, limiting = _limiting(components)
    fraction = ctx.load.delivered_pressure / spec.pressure
    saving = 1.0 - (d_b / ctx.field) ** 2
    warnings: list[str] = []
    if coupling.status is Status.INFEASIBLE:
        warnings.append(f"the background coil links {coupling.flux_ratio:.3g}x its own flux from the insert: "
                        f"its field would swing by {coupling.field_dip:.3g} T in {_time(ctx.rise)} and it quenches "
                        "unless the insert is flux-compensated")
    elif coupling.reverses:
        warnings.append(f"the insert's pulsed flux is {coupling.flux_ratio:.3g}x the background's own: a shorted "
                        f"background would be driven through zero and reversed ({coupling.field_dip:.3g} T swing at "
                        f"{coupling.dip_rate:.3g} T/s) unless the insert is flux-compensated")
    elif coupling.status is not Status.STATE_OF_THE_ART and coupling.status.assessed:
        warnings.append(f"the pulse swings the background field by {coupling.field_dip:.3g} T at "
                        f"{coupling.dip_rate:.3g} T/s ({coupling.status.label} for a superconducting winding)")
    notes = (
        f"{b_dc:g} T steady background plus a {d_b:.4g} T pulse: total {b_total:.4g} T at the insert "
        f"(more than the {ctx.field:.4g} T of an unbiased pulse) for the same pressure",
        f"the bias cuts the insert's own field energy by {saving:.0%}; the peak field and stress on the "
        "insert are not relieved",
        f"background magnet: {b_dc:g} T in a {2e3 * r_bg:.4g} mm bore ({bg_status.label})",
        f"open-circuit, the pulse induces {coupling.volts_per_turn:.3g} V on every background turn",
    )
    return ArchitectureAssessment(
        status=status, margin=fraction, warnings=tuple(warnings), notes=notes,
        key="hybrid_dc_pulsed", name="Hybrid: steady background plus pulsed insert",
        summary="DC superconducting outsert biasing a pulsed resistive insert that supplies the changing field.",
        field=b_total, rise_time=ctx.rise, energy=net.bank_energy if net is not None else None,
        pressure_fraction=fraction, reusable=True, limiting=limiting,
        components={k: v.value for k, v in components.items()},
        details={"background_field": b_dc, "pulse_increment": d_b, "energy_saving": saving,
                 "flux_ratio": coupling.flux_ratio, "field_dip": coupling.field_dip, "dip_rate": coupling.dip_rate,
                 "volts_per_turn": coupling.volts_per_turn, "loss_to_margin_ratio": coupling.loss_to_margin_ratio})


def _single_turn(ctx: _Context) -> ArchitectureAssessment:
    spec = ctx.spec
    a = 0.5 * spec.bore_diameter
    length = spec.bore_diameter  # single-turn coils are about as long as their bore
    t_rise = SINGLE_TURN_RISE_TIME
    inductance = MU0 * math.pi * a * a / length
    current = ctx.field * length / MU0
    energy = 0.5 * inductance * current * current
    voltage = inductance * current * (0.5 * math.pi / t_rise)
    power = pulse.discharge_peak_power(energy, t_rise)
    if ctx.field > SINGLE_TURN_FIELD:
        field_status = Status.ASPIRATIONAL
    elif spec.bore_diameter <= SINGLE_TURN_BORE:
        field_status = Status.STATE_OF_THE_ART
    else:
        field_status = Status.EXTRAPOLATION
    components = {"field and bore": field_status, "bank voltage": pulsed_voltage_status(voltage),
                  "peak current": pulsed_current_status(current), "peak power": peak_power_status(power),
                  "bank energy": energy_status(energy)}
    status, limiting = _limiting(components)
    transfer, _ = _transfer(spec, t_rise)
    notes = (
        f"one turn, {length * 1e3:.4g} mm long: {current / 1e6:.4g} MA in {_time(t_rise)}, "
        f"{voltage / 1e3:.4g} kV, {_mj(energy)} in the bore field",
        "the coil is destroyed by every shot (it explodes outward); samples in existing "
        f"{SINGLE_TURN_BORE * 1e3:g} mm systems usually survive",
        "the field lasts microseconds: the load on the workpiece is a shock, not a press stroke",
    )
    return ArchitectureAssessment(
        status=status, margin=transfer, warnings=("the coil is consumed by every shot",), notes=notes,
        key="single_turn", name="Single-turn destructive coil",
        summary="A copper strip driven by a fast, low-inductance bank; reaches 100-300 T for microseconds.",
        field=ctx.field, rise_time=t_rise, energy=energy, pressure_fraction=transfer, reusable=False,
        limiting=limiting, components={k: v.value for k, v in components.items()},
        details={"turns": 1.0, "charge_voltage": voltage, "peak_current": current, "peak_power": power,
                 "bank_energy": energy})


def _industrial_emf(ctx: _Context) -> ArchitectureAssessment:
    spec = ctx.spec
    cap = EMF_CONVENTIONAL_FIELD[1]
    b = min(ctx.field, cap)
    a1 = 0.5 * spec.bore_diameter
    coil = Solenoid(a1, a1 + INDUSTRIAL_COIL_BUILD, spec.length)
    t_rise = min(ctx.rise, INDUSTRIAL_RISE_TIME)  # forming machines are fast; a faster pulse only shields better
    net = _network(coil, b, t_rise, conductor="copper_ofhc", initial_temperature=T_ROOM)
    components = {"field": Status.STATE_OF_THE_ART,
                  "power supply and coil": net.status if net is not None else Status.INFEASIBLE}
    status, limiting = _limiting(components)
    transfer, _ = _transfer(spec, t_rise)
    mid = ctx.load.surface_mid_ratio if ctx.load.surface_mid_ratio is not None else 1.0
    fraction = transfer * mid * (b / ctx.field) ** 2
    warnings = []
    if b < ctx.field:
        warnings.append(f"capped at {cap:g} T: delivers {magnetic_pressure(b) / 1e9:.3g} GPa of the "
                        f"{_gpa(spec.pressure)} asked for")
    notes = [f"industrial electromagnetic forming works at {EMF_CONVENTIONAL_FIELD[0]:g}-{cap:g} T on areas of "
             "square centimetres with 10-100 kJ machines (Psyk et al. 2011)"]
    energy = None
    details: dict[str, float] = {"field_cap": cap}
    if net is not None:
        energy = net.bank_energy
        details.update({"turns": float(net.turns), "charge_voltage": net.charge_voltage,
                        "peak_current": net.peak_current, "peak_power": net.peak_power,
                        "bank_energy": net.bank_energy})
        notes.append(f"over this bore and length at a {_time(t_rise)} rise: {_mj(net.bank_energy)}, "
                     f"{net.charge_voltage / 1e3:.4g} kV, {net.peak_current / 1e6:.4g} MA ({net.status.label})")
        if net.status is not Status.STATE_OF_THE_ART:
            warnings.append(f"not an industrial machine at this size: {_network_limits(net)}")
    return ArchitectureAssessment(
        status=status, margin=fraction, warnings=tuple(warnings), notes=tuple(notes),
        key="industrial_emf", name="Industrial forming coil",
        summary="A conventional electromagnetic-forming coil at its field limit over the same bore.",
        field=b, rise_time=t_rise, energy=energy, pressure_fraction=fraction, reusable=True,
        limiting=limiting, components={k: v.value for k, v in components.items()}, details=details)


def _architectures(ctx: _Context) -> tuple[ArchitectureAssessment, ...]:
    return (_hts_pulsed(ctx), _hts_dc(ctx), _pulsed_resistive(ctx), _hybrid(ctx), _single_turn(ctx),
            _industrial_emf(ctx))


def compare_architectures(spec: PressSpec) -> tuple[ArchitectureAssessment, ...]:
    """Grade six ways of building the press of ``spec`` (order: :data:`ARCHITECTURES`).

    1. ``hts_pulsed``: the concept magnet swung to full field: worst of conductor, structure and
       per-pulse AC loss against the enthalpy margin (:mod:`magform.conductor`).
    2. ``hts_dc``: the same magnet held steady: infeasible as a press, because
       ``(B_out^2 - B_in^2)/(2 mu0) = 0`` once the field has diffused in.
    3. ``pulsed_resistive``: worst of ``benchmarks.pulsed_field_status(B, bore)`` and the
       capacitor-bank network (:func:`magform.pulse.assess_pulse_network`) at the rise time the
       workpiece needs, plus the fastest rise the supply can deliver within demonstrated practice
       and the pressure that then reaches the workpiece.
    4. ``hybrid_dc_pulsed``: a ``spec.background_field`` superconducting outsert plus a pulsed
       insert of ``dB = sqrt(B_dc^2 + 2 mu0 P) - B_dc``; adds :func:`background_magnet_status` and
       :func:`background_coupling`.
    5. ``single_turn``: a destructive single-turn coil (L = mu0 pi a^2/l with l = bore diameter,
       2.5 us rise); the coil is consumed.
    6. ``industrial_emf``: a forming coil capped at 50 T over the same bore and length.

    Each status is the worst of its components; ``pressure_fraction`` says how much of the asked
    pressure reaches the workpiece.  Validity: every model is the lumped estimate of its module;
    the comparison ranks orders of magnitude, it does not replace a design.

    Reference: Herlach & Miura (2003); Psyk et al. (2011); Wilson (1983); magform.benchmarks.
    """
    return _architectures(_context(spec))


# ---------------------------------------------------------------------------
# the design
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class PressDesign(Assessment):
    """A sized press and the verdict on the concept note's way of building it.

    Geometry and field: ``field`` B0, ``field_factor`` (Montgomery f), ``current_density`` (winding
    average), ``stored_energy`` (long-coil estimate), ``energy_density`` = pressure,
    ``tnt_fraction``, ``placement`` among cited pressures.  Winding loads: ``peak_hoop_stress`` at
    ``hoop_radius`` (non-interacting turns), ``hoop_ratio`` = peak / pressure,
    ``average_hoop_stress``, ``virial_mass`` (lower bound on structure mass at the composite's
    allowable stress), ``end_field_fraction`` and ``peak_conductor_field`` from
    :func:`magform.fieldmap.end_field`.  Then the three assessments of the concept architecture
    (``conductor``, ``structure``, ``pulsed_operation``), the capacitor-bank ``network`` for a
    resistive coil in the same envelope (None when no discharge fits), the ``workpiece`` load and
    the six ``architectures``.  ``status`` and ``margin`` are those of the concept architecture
    (worst of its three assessments; smallest of their margins); ``best_architecture`` is the key
    of the best-graded option that delivers at least 90 % of the pressure (else the best-graded
    one) and ``best_status`` its status.

    Reference: Montgomery (1969); Wilson (1983); Herlach & Miura (2003).
    """

    spec: PressSpec
    field: float = unit("T")
    field_factor: float
    current_density: float = unit("A/m^2")
    stored_energy: float = unit("J")
    energy_density: float = unit("J/m^3")
    tnt_fraction: float
    peak_hoop_stress: float = unit("Pa")
    hoop_radius: float = unit("m")
    hoop_ratio: float
    average_hoop_stress: float = unit("Pa")
    virial_mass: float = unit("kg")
    end_field_fraction: float
    peak_conductor_field: float = unit("T")
    rise_time: float = unit("s")
    placement: Placement
    workpiece: WorkpieceLoad
    conductor: ConductorAssessment
    structure: StructureAssessment
    pulsed_operation: QuenchAssessment
    network: PulseNetworkAssessment | None
    architectures: tuple[ArchitectureAssessment, ...]
    best_architecture: str
    best_status: Status


def _best(architectures: tuple[ArchitectureAssessment, ...]) -> ArchitectureAssessment:
    """Best-graded option that delivers the pressure; reusable coils win ties."""
    delivering = [a for a in architectures if a.pressure_fraction >= DELIVERY_THRESHOLD]
    pool = delivering if delivering else list(architectures)
    return min(pool, key=lambda a: (a.status.rank, not a.reusable, -a.pressure_fraction))


def design_press(spec: PressSpec) -> PressDesign:
    """Size the press of ``spec`` and grade the concept note's architecture for it.

    ``B0 = sqrt(2 mu0 P)``; ``J = B0/(mu0 Delta f)`` (Montgomery); energy, hoop stresses and the
    virial bound from :mod:`magform.magnetics`; coil-end and peak fields from
    :mod:`magform.fieldmap`; the rise time from the workpiece wall; then
    :func:`magform.conductor.assess_conductor` (at the peak conductor field and the computed end
    field), :func:`~magform.conductor.assess_structure` (at the peak hoop stress) and
    :func:`~magform.conductor.assess_pulsed_operation`, the workpiece load, the capacitor-bank
    network and :func:`compare_architectures`.

    Status: worst of conductor, structure and pulsed operation, i.e. the concept magnet as the
    note describes it.  Example: 4 GPa in a 100 mm bore, 0.5 m long, 0.2 m build -> 100.27 T,
    0.1 GJ, 3.1 GPa peak hoop stress, infeasible (the winding quenches every pulse and no
    reinforcement holds the strain); best alternative: a resistive pulsed magnet (extrapolation).

    Validity: long-coil energy and stress estimates (see ``warnings`` when beta < 3), lumped
    models throughout; an order-of-magnitude system study.

    Reference: Montgomery (1969); Wilson (1983); Herlach & Miura (2003); Psyk et al. (2011).
    """
    ctx = _context(spec)
    sol = ctx.solenoid
    architectures = _architectures(ctx)
    best = _best(architectures)
    status = worst(ctx.conductor.status, ctx.structure.status, ctx.pulsed.status)
    margins = [m for m in (ctx.conductor.margin, ctx.structure.margin, ctx.pulsed.margin) if m is not None]
    placement = where_does_it_sit(spec.pressure)
    allowable = ctx.composite.allowable_stress(spec.safety_factor)
    warnings: list[str] = []
    end_warning = sol.end_effect_warning()
    if end_warning:
        warnings.append(end_warning)
    for name, a in (("conductor", ctx.conductor), ("structure", ctx.structure), ("pulsed operation", ctx.pulsed)):
        if a.status in (Status.INFEASIBLE, Status.ASPIRATIONAL) and a.warnings:
            warnings.append(f"{name}: {a.warnings[0]}")
    if ctx.load.skin_melts:
        warnings.append(f"workpiece skin: {ctx.load.material} melts at the surface above "
                        f"{ctx.load.skin_melting_field:.3g} T (this press runs at {ctx.field:.4g} T)")
    if ctx.load.pressure_transfer < 0.8:
        warnings.append(f"the workpiece wall passes only {ctx.load.pressure_transfer:.0%} of the pressure at a "
                        f"{_time(ctx.rise)} rise: use a thicker wall, a faster pulse or a driver sheet")
    notes = (
        f"{_gpa(spec.pressure)} needs {ctx.field:.4g} T; the coil stores {_mj(ctx.energy)} "
        f"({placement.tnt_fraction:.0%} of the energy density of TNT in the bore)",
        f"peak hoop stress {_gpa(ctx.hoop_peak)} = {ctx.hoop_peak / spec.pressure:.3g} x the magnetic pressure; "
        "no winding thickness brings it below half",
        f"radial field at the coil ends is {ctx.ends.fraction:.3g} of the central field for this shape "
        f"(computed; the package default is {conductor.DEFAULT_END_FIELD_FRACTION:g})",
        f"concept architecture: {status.label}; best alternative: {best.name.lower()} ({best.status.label}; "
        f"{best.pressure_fraction:.0%} of the nominal pressure at the workpiece mid-plane"
        + ("" if best.reusable else "; the coil is consumed by every shot") + ")",
        "a steady field exerts no pressure: every option that works is pulsed",
    )
    return PressDesign(
        status=status, margin=min(margins) if margins else None, warnings=tuple(warnings), notes=notes,
        spec=spec, field=ctx.field, field_factor=sol.field_factor(), current_density=ctx.current_density,
        stored_energy=ctx.energy, energy_density=spec.pressure, tnt_fraction=placement.tnt_fraction,
        peak_hoop_stress=ctx.hoop_peak, hoop_radius=ctx.hoop_radius, hoop_ratio=ctx.hoop_peak / spec.pressure,
        average_hoop_stress=sol.average_hoop_stress(ctx.field),
        virial_mass=virial_minimum_mass(ctx.energy, ctx.composite.density(), allowable),
        end_field_fraction=ctx.ends.fraction, peak_conductor_field=ctx.ends.peak_field_ratio * ctx.field,
        rise_time=ctx.rise, placement=placement, workpiece=ctx.load, conductor=ctx.conductor,
        structure=ctx.structure, pulsed_operation=ctx.pulsed, network=ctx.network, architectures=architectures,
        best_architecture=best.key, best_status=best.status)


def reference_spec(pressure: float = 4.0e9, **changes) -> PressSpec:
    """The reference press used throughout the documentation: 100 mm bore, 0.5 m long, 0.2 m build.

    ``pressure`` [Pa] defaults to the concept note's upper bound; keyword ``changes`` replace any
    other :class:`PressSpec` field.  With the defaults the workpiece is a solid copper billet of
    40 mm radius and 0.3 m length.

    Reference: the worked design example in the documentation.
    """
    spec = PressSpec(pressure=pressure, bore_diameter=0.1, length=0.5, winding_thickness=0.2)
    return dataclasses.replace(spec, **changes) if changes else spec


__all__ = [
    "REFERENCES", "ARCHITECTURES", "DEFAULT_WORKPIECE_RADIUS_FRACTION", "DEFAULT_WORKPIECE_LENGTH_FRACTION",
    "LARGE_BORE_SC_FIELD", "LARGE_BORE_SC_DIAMETER", "SMALL_BORE_SC_DIAMETER", "BACKGROUND_CLEARANCE",
    "BACKGROUND_BUILD", "SINGLE_TURN_FIELD", "SINGLE_TURN_BORE", "SINGLE_TURN_RISE_TIME", "SINGLE_TURN_WALL",
    "INDUSTRIAL_COIL_BUILD", "INDUSTRIAL_RISE_TIME", "DELIVERY_THRESHOLD", "RISE_SCAN",
    "PressSpec", "WorkpieceLoad", "workpiece_load", "BackgroundCoupling", "background_coupling",
    "background_magnet_status", "ArchitectureAssessment", "compare_architectures", "PressDesign",
    "design_press", "reference_spec",
]
