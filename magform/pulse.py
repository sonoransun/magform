"""Pulse-power network: capacitor banks, rotating machines and the coil they drive.

The concept note closes with the question "Should we map out the pulse-discharge
electrical network (like homopolar generators or capacitor banks) required to
drive these pressures?".  This module is that map.  It takes a coil
(:class:`magform.magnetics.Solenoid`), a field and a rise time and returns the
circuit that would have to exist: capacitance, charge voltage, peak current,
peak power, switch count, bank mass, and what one pulse does to the conductor.

Physics in one paragraph
------------------------
A capacitor bank ringing into an inductive coil reaches its current peak after
a quarter period, so ``omega0 = pi / (2 t_rise)`` and the bank must satisfy

    ``V0 * I_pk = pi * E / t_rise``        (``E`` = magnetic energy at the peak)

whatever the winding looks like.  The number of turns only trades current
against voltage (``L ~ N^2``, ``I ~ 1/N``, ``V0 ~ N``); the product, and the peak
power ``pi E / (2 t_rise)``, are fixed by the field energy and the rise time.
The field energy is fixed by the bore: 4 GPa in a 100 mm bore, 0.5 m long coil is
~0.1 GJ.  Delivering that in the tens of microseconds an eddy-current forming
pulse needs is a terawatt-class problem (hundreds of kilovolts at tens of
mega-amperes), and at that speed the current cannot diffuse into the conductor:
the coil works as a current sheet whose surface melts near 90-100 T, which is
the destructive single-turn-coil regime.  In 10 ms the same energy is a 25 kV,
~1 MA capacitor bank twice the size of the largest one built to drive magnets.

Homopolar generators and motor-generators store far more energy per unit cost
but are low-voltage machines: a homopolar rotor is electrically a capacitor of
``C_eq = J / k^2`` (kilofarads at ~100 V), so it rings into the same coil in
tens of milliseconds to seconds.  They suit the slow outer "platform" coil of a
two-coil magnet and are wrong by three orders of magnitude or more for a
microsecond pulse; shortening their pulse through an inductive store loses at
least 75 % of the energy when the load is itself an inductor.

What the module does not model
------------------------------
Coil stresses (:mod:`magform.magnetics`, :mod:`magform.conductor`,
:mod:`magform.containment`), the workpiece's reflected impedance (a shielding
workpiece lowers the bore inductance and energy), transmission-line effects,
switch physics, magnetoresistance of the conductor (which raises its
resistance and heating at 60-100 T) and non-uniform current density in graded
windings.  All quantities are SI; the current density of a coil is the
winding-average value of :class:`~magform.magnetics.Solenoid` and the conductor
current density is that divided by the conductor fill factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ._numerics import bisect, linspace, trapz
from ._report import Assessment, ResultMixin, unit
from .benchmarks import (
    ENERGY_ANY_STORE_RECORD,
    ENERGY_LADDER,
    ENERGY_MAGNET_DRIVER_SOTA,
    PEAK_POWER_SOTA,
    PULSED_CURRENT_SOTA,
    PULSED_VOLTAGE_SOTA,
    energy_status,
    peak_power_status,
    pulsed_current_status,
    pulsed_voltage_status,
)
from .conductor import terminal_voltage
from .constants import T_LIQUID_NITROGEN
from .eddy import (
    adiabatic_joule_heating,
    skin_depth,
    skin_melting_field,
    skin_temperature_rise,
    specific_action_to_temperature,
)
from .errors import InputError
from .magnetics import Solenoid
from .materials import Material, MaterialLike, as_material
from .status import Status, worst

REFERENCES: dict[str, str] = {
    "knoepfel1970": "H. Knoepfel, Pulsed High Magnetic Fields, North-Holland, Amsterdam (1970): capacitor-bank "
                    "discharge circuits, inductive energy storage and transfer, action integral of conductors, "
                    "explosive flux compression.",
    "knoepfel2000": "H. E. Knoepfel, Magnetic Fields: A Comprehensive Theoretical Treatise for Practical Use, "
                    "Wiley, New York (2000): field diffusion, skin heating, energy transfer between circuits.",
    "herlach_miura2003": "F. Herlach and N. Miura (eds.), High Magnetic Fields: Science and Technology, Vol. 1, "
                         "World Scientific, Singapore (2003): pulsed-magnet power supplies, coil heating from a "
                         "liquid-nitrogen start, destructive single-turn coils.",
    "psyk2011": "V. Psyk, D. Risch, B. L. Kinsey, A. E. Tekkaya and M. Kleiner, 'Electromagnetic forming - a "
                "review', J. Mater. Process. Technol. 211 (2011) 787-829: capacitor-bank forming machines and "
                "their damped-sine discharge.",
    "montgomery1969": "D. B. Montgomery, Solenoid Magnet Design: The Magnetic and Mechanical Aspects of "
                      "Resistive and Superconducting Systems, Wiley-Interscience, New York (1969).",
    "wheeler1928": "H. A. Wheeler, 'Simple inductance formulas for radio coils', Proc. IRE 16 (1928) 1398-1400.",
    "woodson_melcher1968": "H. H. Woodson and J. R. Melcher, Electromechanical Dynamics, Part I: Discrete "
                           "Systems, Wiley, New York (1968): homopolar machines, V = k omega and torque = k I.",
    "timoshenko_goodier1970": "S. P. Timoshenko and J. N. Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill, "
                              "New York (1970): stresses in rotating disks.",
    "fowler1960": "C. M. Fowler, W. B. Garn and R. S. Caird, 'Production of very high magnetic fields by "
                  "implosion', J. Appl. Phys. 31 (1960) 588-594: explosive flux compression.",
    "benchmark_ladders": "magform.benchmarks ENERGY_LADDER, CURRENT_LADDER and the pulsed voltage/power "
                         "thresholds: HZDR 50 MJ capacitor bank, UT-CEM Balcones 60 MJ homopolar supply, ANU "
                         "500 MJ homopolar generator, LANL 1.4 GW motor-generator, Sandia Z (citations there).",
}
"""Short key -> full citation for every model in this module."""


def _ladder_value(key: str, fallback: float) -> float:
    """Value of an entry of :data:`magform.benchmarks.ENERGY_LADDER` (single source of the thresholds)."""
    for bench in ENERGY_LADDER:
        if bench.key == key:
            return float(bench.value)
    return fallback


# ---------------------------------------------------------------------------
# module constants (documented single sources)
# ---------------------------------------------------------------------------
REGIMES: tuple[str, str, str] = ("underdamped", "critical", "overdamped")
"""Damping regimes of the series RLC discharge, by ``zeta = (R/2) sqrt(C/L)`` < 1, = 1, > 1."""

WAVEFORMS: tuple[str, str] = ("crowbar", "half_sine")
"""Current waveforms of :func:`conductor_heating`: ``"crowbar"`` (quarter-sine rise, then the coil is
short-circuited and its field energy decays in the circuit resistance) and ``"half_sine"`` (the current
swings back to zero and the field energy returns to the reverse-charged bank)."""

CRITICAL_BAND: float = 1e-9
"""Half-width in ``zeta`` (and in ``alpha t_rise``) of the band treated as critically damped (numerical)."""

DEFAULT_FILL_FACTOR: float = 0.5
"""Conductor fraction of the winding cross-section (estimated: the rest is reinforcement and insulation;
0.4-0.6 in reinforced pulsed-magnet windings, Herlach & Miura 2003)."""

PRECOOL_TEMPERATURE: float = T_LIQUID_NITROGEN
"""K.  Start temperature of a liquid-nitrogen pre-cooled pulsed coil."""

COIL_TEMPERATURE_LIMIT: float = 400.0
"""K.  End-of-pulse conductor temperature treated as the limit of a reusable coil (estimated: organic
insulation and reinforcement; pulsed magnets are designed to end a full-energy pulse near room
temperature to ~400 K, Herlach & Miura 2003)."""

DEFAULT_MAX_VOLTAGE: float = PULSED_VOLTAGE_SOTA
"""V.  Charge-voltage ceiling used to choose the turn count: ``benchmarks.PULSED_VOLTAGE_SOTA`` (25 kV)."""

SWITCH_CURRENT_RATING: float = 1.0e5
"""A.  Peak current assigned to one closing-switch module (estimated: thyristor-stack / ignitron class;
spark gaps and rail gaps carry several times more per unit at shorter life)."""

CAPACITOR_SPECIFIC_ENERGY: float = 1.0e3
"""J/kg.  Representative energy per unit mass of pulse capacitors (estimated; see the range)."""

CAPACITOR_SPECIFIC_ENERGY_RANGE: tuple[float, float] = (0.2e3, 2.5e3)
"""J/kg.  Estimated span from paper/oil capacitors and fast, high-reversal units to the best
metallised-film millisecond-discharge units.  A complete bank (racks, bus work, switches, clearances)
is 2-3 times heavier than its capacitors."""

CAPACITOR_ENERGY_DENSITY: float = 1.0e6
"""J/m^3.  Representative energy per unit volume of pulse capacitors (estimated; 1 J/cm^3)."""

CAPACITOR_ENERGY_DENSITY_RANGE: tuple[float, float] = (0.2e6, 2.5e6)
"""J/m^3.  Estimated span, as :data:`CAPACITOR_SPECIFIC_ENERGY_RANGE`."""

INDUCTIVE_TRANSFER_LIMIT: float = 0.25
"""Largest fraction of the energy of a storage inductor that an opening switch can transfer into an
uncoupled load inductor (reached when the two inductances are equal; Knoepfel 1970)."""

SKIN_LIMIT_RATIO: float = 2.0
"""Conductor side / skin depth above which the current of a solid conductor is treated as non-uniform
(estimated: the field penetrates about one skin depth from each face during the rise)."""

ROTOR_DENSITY: float = 7850.0
"""kg/m^3.  Steel rotor of a homopolar generator (as the ``steel_4340`` record)."""

ROTOR_POISSON: float = 0.3
"""Poisson ratio used for the rotating-disk stress of a steel rotor."""

HPG_TIP_SPEED: float = 200.0
"""m/s.  Default rotor surface speed of a homopolar generator (estimated: limited by the sliding brush
contacts before the rotor stress; the solid-disk stress at 200 m/s is ~130 MPa)."""

HPG_FLUX_DENSITY: float = 2.0
"""T.  Default flux density through a homopolar rotor (estimated: iron-saturation level excitation)."""

HPG_VOLTAGE_CLASS: float = 600.0
"""V.  Largest terminal voltage of a homopolar supply, used to grade the source: the series connection
of the UT-CEM Balcones machines (``benchmarks.ENERGY_LADDER['balcones_60mj_hpg']``: 60 MJ, 1.5 MA at 600 V)."""

HPG_INTERNAL_INDUCTANCE: float = 0.5e-6
"""H.  Internal plus bus inductance assumed for a homopolar supply when estimating the fastest pulse it
can deliver even into a short circuit (estimated, order of magnitude)."""

HPG_ENERGY_SOTA: float = _ladder_value("balcones_60mj_hpg", 60e6)
"""J.  Largest homopolar pulsed-power supply operated as such (UT-CEM Balcones, 6 x 10 MJ)."""

HPG_ENERGY_RECORD: float = _ladder_value("anu_500mj_hpg", 500e6)
"""J.  Largest homopolar generator built (ANU, 1962-1986)."""

MOTOR_GENERATOR_POWER: float = 1.4e9
"""W.  Rating of the motor-generator of the NHMFL Pulsed Field Facility at Los Alamos, the largest
rotating supply driving a pulsed magnet (``benchmarks.FIELD_LADDER['lanl_100t_nondestructive']`` note)."""

MOTOR_GENERATOR_ENERGY: float = 0.5 * ENERGY_ANY_STORE_RECORD
"""J.  Energy one pulse can draw from that machine: about half of its 1.2 GJ rotational energy
(estimated: the rotor may only be slowed to ~70 % of its speed)."""

MOTOR_GENERATOR_MIN_RISE_TIME: float = 0.01
"""s.  Fastest rise a rectified alternator can follow: about one half-cycle of its 50-60 Hz output
(estimated; pulsed alternators built for railguns reach a few milliseconds at MJ scale)."""

FCG_MIN_RISE_TIME: float = 1.0e-6
"""s.  Shortest current rise of explosive flux-compression generators (estimated: microsecond class)."""

FCG_MAX_RISE_TIME: float = 1.0e-3
"""s.  Longest pulse of an explosive flux-compression generator (estimated: detonation at ~8 km/s
sweeps a metre-scale generator in ~0.1 ms)."""

FCG_ENERGY_DEMONSTRATED: float = 3.0e7
"""J.  Electrical energy that explosive flux-compression generators deliver to a load with confidence
(estimated order of magnitude: tens of MJ at tens of MA, single shot; ~100 MJ-class disk generators
have been reported and are graded extrapolation here)."""

SOURCE_EXTRAPOLATION_FACTOR: float = 10.0
"""A source asked for up to this factor beyond its demonstrated power or energy is graded
extrapolation, beyond it aspirational (judgement)."""

_HEATING_SAMPLES: int = 200       # current-waveform samples per rise time for the action integral
_ACTION_SAMPLES: int = 800        # samples of the analytic RLC current up to its peak
_GOLDEN: float = 0.5 * (math.sqrt(5.0) - 1.0)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise InputError(f"{name} must be a real number, got {value!r}")
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(v):
        raise InputError(f"{name} must be finite, got {value!r}")
    return v


def _positive(name: str, value: object) -> float:
    v = _finite(name, value)
    if v <= 0.0:
        raise InputError(f"{name} must be > 0, got {v!r}")
    return v


def _non_negative(name: str, value: object) -> float:
    v = _finite(name, value)
    if v < 0.0:
        raise InputError(f"{name} must be >= 0, got {v!r}")
    return v


def _fraction(name: str, value: object) -> float:
    v = _finite(name, value)
    if not 0.0 <= v <= 1.0:
        raise InputError(f"{name} must lie in [0, 1], got {v!r}")
    return v


def _fill(value: object) -> float:
    v = _finite("fill_factor", value)
    if not 0.0 < v <= 1.0:
        raise InputError(f"fill_factor must lie in (0, 1], got {v!r}")
    return v


def _count(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _solenoid(solenoid: object) -> Solenoid:
    if not isinstance(solenoid, Solenoid):
        raise InputError(f"solenoid must be a magform.magnetics.Solenoid, got {type(solenoid).__name__}")
    return solenoid


def _waveform(waveform: object) -> str:
    w = waveform.strip().lower() if isinstance(waveform, str) else waveform
    if w not in WAVEFORMS:
        raise InputError(f"waveform must be one of {WAVEFORMS}, got {waveform!r}")
    return w  # type: ignore[return-value]


def _conductor(conductor: MaterialLike, temperature: float) -> Material:
    m = as_material(conductor)
    if not m.resistivity > 0.0:
        raise InputError(f"conductor '{m.key}' has no positive resistivity")
    if temperature >= m.melting_point:
        raise InputError(f"temperature {temperature:g} K is at or above the melting point of {m.key} "
                         f"({m.melting_point:g} K)")
    return m


def _si(value: float, unit_: str) -> str:
    """Engineering-prefixed text, e.g. 2.67e5 V -> '267 kV'."""
    if value == 0.0 or not math.isfinite(value):
        return f"{value:g} {unit_}"
    for factor, prefix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"),
                           (1e-6, "u"), (1e-9, "n")):
        if abs(value) >= factor:
            return f"{value / factor:.3g} {prefix}{unit_}"
    return f"{value:.3g} {unit_}"


def _grade(value: float, demonstrated: float) -> Status:
    """State of the art up to ``demonstrated``, extrapolation up to 10x it, aspirational beyond."""
    if value <= demonstrated:
        return Status.STATE_OF_THE_ART
    if value <= SOURCE_EXTRAPOLATION_FACTOR * demonstrated:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


# ---------------------------------------------------------------------------
# the fixed-product law of a quarter-period rise
# ---------------------------------------------------------------------------
def angular_frequency_for_rise(rise_time: float) -> float:
    """Natural angular frequency of a discharge that peaks after ``rise_time``, ``omega0 = pi/(2 t)`` [rad/s].

    A lossless capacitor discharge into an inductor is ``I = I_pk sin(omega0 t)`` and reaches its
    peak after a quarter period.  ``rise_time`` in s (> 0).  The equivalent frequency
    ``f = 1/(4 t_rise)`` is the one used for skin depths (:func:`magform.eddy.skin_depth`).
    Validity: lossless (``R << 2 sqrt(L/C)``); with damping use :func:`capacitance_for_rise_time`.

    Reference: Knoepfel (1970); Psyk et al. (2011).
    """
    return math.pi / (2.0 * _positive("rise_time", rise_time))


def voltage_current_product(energy: float, rise_time: float) -> float:
    """Charge voltage times peak current of a quarter-period discharge, ``V0 I_pk = pi E / t_rise`` [W].

    From ``E = L I_pk^2 / 2`` and ``V0 = omega0 L I_pk`` with ``omega0 = pi/(2 t_rise)``.  ``energy`` E
    is the magnetic energy at the current peak (J, >= 0), ``rise_time`` in s (> 0).  The product does
    not depend on the number of turns: N turns give ``L ~ N^2``, ``I_pk ~ 1/N`` and ``V0 ~ N``, so the
    winding only chooses how the product is split.  Example: 99.5 MJ in 25 us is 1.25e13 V A
    (267 kV at 46.7 MA); in 10 ms it is 3.1e10 V A (25 kV at 1.25 MA).
    Validity: lossless discharge; resistance raises ``V0`` by ``exp(R t_rise / 2L)``.

    Reference: Knoepfel (1970); Herlach & Miura (2003).
    """
    e = _non_negative("energy", energy)
    return math.pi * e / _positive("rise_time", rise_time)


def discharge_peak_power(energy: float, rise_time: float) -> float:
    """Peak power flowing from the bank into the coil, ``P_pk = pi E / (2 t_rise)`` [W].

    The power into the inductor is ``d/dt (L I^2 / 2) = E omega0 sin(2 omega0 t)``, largest at half
    the rise time; it equals ``V0 I_pk / 2``.  ``energy`` in J (>= 0), ``rise_time`` in s (> 0).
    Example: 99.5 MJ in 25 us -> 6.25 TW; in 1 ms -> 156 GW; in 10 ms -> 15.6 GW.
    Validity: lossless quarter-sine rise.  Compare with ``benchmarks.peak_power_status``.

    Reference: Knoepfel (1970).
    """
    e = _non_negative("energy", energy)
    return 0.5 * math.pi * e / _positive("rise_time", rise_time)


def rise_time_for_product(energy: float, voltage: float, current: float) -> float:
    """Rise time a source of given voltage and current can achieve, ``t_rise = pi E / (V0 I_pk)`` [s].

    Inverse of :func:`voltage_current_product`.  ``energy`` in J (>= 0), ``voltage`` in V (> 0),
    ``current`` in A (> 0).  With the ampere-turns of a magnet as ``current`` and a voltage ceiling as
    ``voltage`` it is the fastest rise a single-turn winding allows: 0.27 ms for 99.5 MJ at 25 kV and
    46.7 MA.  Validity: lossless quarter-sine rise.

    Reference: Knoepfel (1970).
    """
    e = _non_negative("energy", energy)
    v = _positive("voltage", voltage)
    i = _positive("current", current)
    return math.pi * e / (v * i)


def inductive_transfer_efficiency(store_inductance: float, load_inductance: float) -> float:
    """Fraction of a storage inductor's energy that reaches an uncoupled load inductor, ``L1 L2/(L1+L2)^2``.

    An opening switch forces the current ``I0`` of the store ``L1`` to share with the load ``L2``; flux
    conservation gives ``I = I0 L1/(L1 + L2)`` and a load energy ``E0 L1 L2/(L1 + L2)^2``.  The maximum
    is 1/4 at ``L1 = L2`` (:data:`INDUCTIVE_TRANSFER_LIMIT`); the fraction ``L2/(L1 + L2)`` is
    dissipated in the switch.  Inductances in H (> 0).  This is what compressing the pulse of a
    homopolar or motor-generator through an inductive store costs when the load is a magnet coil.
    Validity: uncoupled lumped inductors, ideal opening switch, no resistance; a transfer capacitor
    or flux compression can do better at the price of more components.

    Reference: Knoepfel (1970).
    """
    l1 = _positive("store_inductance", store_inductance)
    l2 = _positive("load_inductance", load_inductance)
    return l1 * l2 / (l1 + l2) ** 2


# ---------------------------------------------------------------------------
# series RLC discharge (analytic, all three regimes)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Circuit:
    """Solved series RLC discharge (private): ``rate`` is omega_d (underdamped) or gamma (overdamped)."""

    c: float
    v: float
    l: float
    r: float
    regime: str
    alpha: float
    omega0: float
    rate: float
    t_peak: float

    def current(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        a, w = self.alpha, self.rate
        if self.regime == "underdamped":
            return self.v / (self.l * w) * math.exp(-a * t) * math.sin(w * t)
        if self.regime == "critical":
            return self.v / self.l * t * math.exp(-a * t)
        slow = self.omega0 * self.omega0 / (a + w)     # alpha - gamma without cancellation
        return self.v / (2.0 * self.l * w) * (math.exp(-slow * t) - math.exp(-(a + w) * t))

    def voltage(self, t: float) -> float:
        if t <= 0.0:
            return self.v
        a, w = self.alpha, self.rate
        if self.regime == "underdamped":
            return self.v * math.exp(-a * t) * (math.cos(w * t) + a / w * math.sin(w * t))
        if self.regime == "critical":
            return self.v * math.exp(-a * t) * (1.0 + a * t)
        slow = self.omega0 * self.omega0 / (a + w)
        return self.v / (2.0 * w) * ((w + a) * math.exp(-slow * t) + (w - a) * math.exp(-(a + w) * t))


def _circuit(capacitance: float, voltage: float, inductance: float, resistance: float) -> _Circuit:
    c = _positive("capacitance", capacitance)
    v = _non_negative("voltage", voltage)
    ind = _positive("inductance", inductance)
    r = _non_negative("resistance", resistance)
    alpha = r / (2.0 * ind)
    omega0 = 1.0 / math.sqrt(ind * c)
    zeta = alpha / omega0
    if zeta < 1.0 - CRITICAL_BAND:
        rate = math.sqrt((omega0 - alpha) * (omega0 + alpha))
        return _Circuit(c, v, ind, r, "underdamped", alpha, omega0, rate, math.atan2(rate, alpha) / rate)
    if zeta <= 1.0 + CRITICAL_BAND:
        return _Circuit(c, v, ind, r, "critical", alpha, omega0, 0.0, 1.0 / alpha)
    rate = math.sqrt((alpha - omega0) * (alpha + omega0))
    return _Circuit(c, v, ind, r, "overdamped", alpha, omega0, rate, math.log((alpha + rate) / omega0) / rate)


def _golden_max(f: Callable[[float], float], a: float, b: float, n: int = 90) -> float:
    """Maximum of a unimodal function on [a, b] by golden-section search."""
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(n):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = f(d)
    return max(fc, fd)


@dataclass(frozen=True, kw_only=True)
class RLCDischarge(ResultMixin):
    """Analytic summary of a capacitor discharging through a series inductance and resistance.

    ``damping_ratio`` zeta = (R/2) sqrt(C/L); ``natural_frequency`` omega0 = 1/sqrt(LC);
    ``damped_frequency`` omega_d = sqrt(omega0^2 - alpha^2) (0 unless underdamped); ``decay_rate``
    alpha = R/(2L).  ``peak_current`` and ``time_to_peak`` refer to the first current maximum;
    ``peak_di_dt`` = V0/L occurs at t = 0.  ``capacitor_voltage_at_peak`` = R I_pk (the inductor
    voltage is zero there), so ``energy_dissipated_to_peak`` = C V0^2/2 - C (R I_pk)^2/2 - L I_pk^2/2.
    ``transfer_efficiency`` = L I_pk^2 / (C V0^2).  ``peak_power`` is the largest V_C I leaving the
    capacitor before the peak.  ``voltage_reversal`` is the first reverse capacitor voltage as a
    fraction of V0 when the circuit is left to ring (0 unless underdamped); ``crowbar_time_constant``
    = L/R governs the decay if the coil is short-circuited at the peak (inf for R = 0).

    Reference: Knoepfel (1970); Psyk et al. (2011).
    """

    capacitance: float = unit("F")
    voltage: float = unit("V")
    inductance: float = unit("H")
    resistance: float = unit("ohm")
    regime: str
    damping_ratio: float
    natural_frequency: float = unit("rad/s")
    damped_frequency: float = unit("rad/s")
    decay_rate: float = unit("1/s")
    peak_current: float = unit("A")
    time_to_peak: float = unit("s")
    peak_di_dt: float = unit("A/s")
    stored_energy: float = unit("J")
    magnetic_energy_at_peak: float = unit("J")
    capacitor_voltage_at_peak: float = unit("V")
    energy_dissipated_to_peak: float = unit("J")
    transfer_efficiency: float
    peak_power: float = unit("W")
    voltage_reversal: float
    crowbar_time_constant: float = unit("s")


def rlc_current(capacitance: float, voltage: float, inductance: float, resistance: float = 0.0,
                crowbar: bool = False) -> Callable[[float], float]:
    """Current waveform ``I(t)`` [A] of a capacitor discharged into a series R-L load.

    With ``alpha = R/(2L)`` and ``omega0 = 1/sqrt(LC)``::

        underdamped (alpha < omega0):  I = V0/(omega_d L) exp(-alpha t) sin(omega_d t),  omega_d^2 = omega0^2 - alpha^2
        critical    (alpha = omega0):  I = (V0/L) t exp(-alpha t)
        overdamped  (alpha > omega0):  I = V0/(gamma L) exp(-alpha t) sinh(gamma t),     gamma^2 = alpha^2 - omega0^2

    and ``I = 0`` for t <= 0.  With ``crowbar=True`` the coil is short-circuited at the first current
    peak and ``I = I_pk exp(-(t - t_pk) R/L)`` afterwards (constant for R = 0).  ``capacitance`` in F
    (> 0), ``voltage`` V0 in V (>= 0), ``inductance`` in H (> 0), ``resistance`` in ohm (>= 0); the
    returned callable takes time in s.
    Validity: lumped, linear, constant R and L (no conductor heating, no moving workpiece, no
    transit-time effects); ideal switch closing at t = 0.

    Reference: Knoepfel (1970); Psyk et al. (2011).
    """
    k = _circuit(capacitance, voltage, inductance, resistance)
    if not crowbar:
        return k.current
    t_pk = k.t_peak
    i_pk = k.current(t_pk)
    rate = k.r / k.l

    def crowbarred(t: float) -> float:
        if t <= t_pk:
            return k.current(t)
        return i_pk * math.exp(-(t - t_pk) * rate)

    return crowbarred


def rlc_capacitor_voltage(capacitance: float, voltage: float, inductance: float,
                          resistance: float = 0.0) -> Callable[[float], float]:
    """Capacitor voltage ``V_C(t)`` [V] during the series RLC discharge of :func:`rlc_current`.

    ``V_C = L dI/dt + R I``::

        underdamped:  V_C = V0 exp(-alpha t) [cos(omega_d t) + (alpha/omega_d) sin(omega_d t)]
        critical:     V_C = V0 exp(-alpha t) (1 + alpha t)
        overdamped:   V_C = V0 exp(-alpha t) [cosh(gamma t) + (alpha/gamma) sinh(gamma t)]

    and ``V_C = V0`` for t <= 0 (no crowbar).  Units and validity as :func:`rlc_current`.  The energy
    balance ``C V_C^2/2 + L I^2/2 + integral R I^2 dt = C V0^2/2`` holds at every instant.

    Reference: Knoepfel (1970).
    """
    return _circuit(capacitance, voltage, inductance, resistance).voltage


def rlc_discharge(capacitance: float, voltage: float, inductance: float, resistance: float = 0.0) -> RLCDischarge:
    """Peak current, time to peak, losses and regime of a series RLC capacitor discharge.

    ``zeta = (R/2) sqrt(C/L)`` selects the regime; the time to the first current maximum is::

        underdamped:  t_pk = atan(omega_d/alpha)/omega_d      (-> pi/(2 omega0) for R -> 0)
        critical:     t_pk = 1/alpha = 2L/R
        overdamped:   t_pk = atanh(gamma/alpha)/gamma

    and in every regime ``I_pk = V0 sqrt(C/L) exp(-alpha t_pk)``.  ``capacitance`` in F (> 0),
    ``voltage`` in V (>= 0), ``inductance`` in H (> 0), ``resistance`` in ohm (>= 0).  Check: 1 mF at
    10 kV into 1 uH without resistance gives 316 kA after 49.7 us and 50 kJ in the field.
    Validity: as :func:`rlc_current`.

    Reference: Knoepfel (1970); Psyk et al. (2011).
    """
    k = _circuit(capacitance, voltage, inductance, resistance)
    t_pk = k.t_peak
    i_pk = k.current(t_pk)
    stored = 0.5 * k.c * k.v * k.v
    magnetic = 0.5 * k.l * i_pk * i_pk
    v_pk = k.r * i_pk
    dissipated = max(stored - 0.5 * k.c * v_pk * v_pk - magnetic, 0.0) if k.r > 0.0 else 0.0
    if k.v > 0.0:
        power = _golden_max(lambda t: k.voltage(t) * k.current(t), 0.0, t_pk)
    else:
        power = 0.0
    reversal = math.exp(-k.alpha * math.pi / k.rate) if k.regime == "underdamped" else 0.0
    return RLCDischarge(
        capacitance=k.c,
        voltage=k.v,
        inductance=k.l,
        resistance=k.r,
        regime=k.regime,
        damping_ratio=k.alpha / k.omega0,
        natural_frequency=k.omega0,
        damped_frequency=k.rate if k.regime == "underdamped" else 0.0,
        decay_rate=k.alpha,
        peak_current=i_pk,
        time_to_peak=t_pk,
        peak_di_dt=k.v / k.l,
        stored_energy=stored,
        magnetic_energy_at_peak=magnetic,
        capacitor_voltage_at_peak=v_pk,
        energy_dissipated_to_peak=dissipated,
        transfer_efficiency=magnetic / stored if stored > 0.0 else 0.0,
        peak_power=power,
        voltage_reversal=reversal,
        crowbar_time_constant=k.l / k.r if k.r > 0.0 else math.inf,
    )


def capacitance_for_rise_time(inductance: float, rise_time: float, resistance: float = 0.0) -> float:
    """Capacitance that makes a series RLC discharge peak after ``rise_time`` [F].

    With ``x = alpha t_rise = R t_rise/(2L)``::

        x = 0:   C = (2 t_rise/pi)^2 / L
        x < 1:   underdamped; u = omega_d t_rise solves tan(u) = u/x on (0, pi/2), omega0^2 = (u^2 + x^2)/t_rise^2
        x = 1:   critical; C = 4L/R^2
        x > 1:   overdamped; w = gamma t_rise solves tanh(w) = w/x, omega0 = alpha/cosh(w)

    then ``C = 1/(L omega0^2)``.  ``inductance`` in H (> 0), ``rise_time`` in s (> 0), ``resistance`` in
    ohm (>= 0).  The capacitance grows as ``exp(R t_rise / L)`` once the rise time exceeds a few L/R (the
    current then merely creeps towards V/R): beyond ~700 L/R the result overflows and
    :class:`InputError` is raised - use a controlled rectifier ramp, not a capacitor discharge.
    Validity: as :func:`rlc_current` (constant R, L).

    Reference: Knoepfel (1970).
    """
    ind = _positive("inductance", inductance)
    t = _positive("rise_time", rise_time)
    r = _non_negative("resistance", resistance)
    alpha = r / (2.0 * ind)
    x = alpha * t
    if x == 0.0:
        omega0_sq = (math.pi / (2.0 * t)) ** 2
    elif x < 1.0 - CRITICAL_BAND:
        # delta = pi/2 - u is well conditioned for weak damping: (pi/2 - delta) sin(delta) = x cos(delta)
        delta = bisect(lambda d: (0.5 * math.pi - d) * math.sin(d) - x * math.cos(d),
                       0.0, 0.5 * math.pi - 1e-9, tol=1e-15, max_iter=400)
        u = 0.5 * math.pi - delta
        omega0_sq = (u * u + x * x) / (t * t)
    elif x <= 1.0 + CRITICAL_BAND:
        omega0_sq = alpha * alpha
    else:
        lo = 0.5 * min(math.sqrt(3.0 * (1.0 - 1.0 / x)), 1.0)
        w = bisect(lambda z: math.tanh(z) - z / x, lo, x, tol=1e-15, max_iter=400)
        try:
            omega0_sq = alpha * alpha / math.cosh(w) ** 2
        except OverflowError:
            omega0_sq = 0.0
        if omega0_sq <= 0.0:
            raise InputError(f"rise_time {t:g} s is {x:.3g} times 2L/R: no finite capacitor discharge peaks that "
                             "late; use a controlled (rectifier-driven) ramp")
    return 1.0 / (ind * omega0_sq)


def voltage_for_peak_current(capacitance: float, inductance: float, peak_current: float,
                             resistance: float = 0.0) -> float:
    """Charge voltage that drives ``peak_current`` through a series R-L load [V].

    ``V0 = I_pk sqrt(L/C) exp(alpha t_pk)`` with ``alpha = R/(2L)`` and ``t_pk`` from
    :func:`rlc_discharge` (the peak current is proportional to V0 in a linear circuit).  For R = 0
    this is ``I_pk sqrt(L/C) = omega0 L I_pk``; in general the stored energy ``C V0^2/2`` exceeds the
    field energy ``L I_pk^2/2`` by the factor ``exp(R t_pk / L)``.  ``capacitance`` in F (> 0), ``inductance`` in H (> 0),
    ``peak_current`` in A (>= 0), ``resistance`` in ohm (>= 0).
    Validity: as :func:`rlc_current`.

    Reference: Knoepfel (1970).
    """
    i = _non_negative("peak_current", peak_current)
    k = _circuit(capacitance, 1.0, inductance, resistance)
    return i / k.current(k.t_peak)


# ---------------------------------------------------------------------------
# coil electrical parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class CoilElectrical(ResultMixin):
    """Lumped electrical parameters of a uniform-current-density solenoid wound with ``turns`` turns.

    ``inductance`` is energy-consistent (2E/I^2, the value the bank must fill); ``inductance_wheeler``
    includes fringing flux.  ``resistance`` = rho_e N 2 pi a_m / A_c with the conductor cross-section
    of one turn ``conductor_area`` = fill L Delta / N; ``time_constant`` = L/R; ``current_per_tesla``
    = terminal current for 1 T at the centre; ``layers`` is the number of radial layers if the turns
    are square in cross-section (>= 1, estimated); ``conductor_side`` = sqrt(conductor_area).

    Reference: Montgomery (1969); Wheeler (1928).
    """

    turns: float
    layers: int
    conductor: str
    fill_factor: float
    temperature: float = unit("K")
    resistivity: float = unit("ohm m")
    inductance: float = unit("H")
    inductance_wheeler: float = unit("H")
    resistance: float = unit("ohm")
    time_constant: float = unit("s")
    current_per_tesla: float = unit("A/T")
    mean_turn_length: float = unit("m")
    conductor_area: float = unit("m^2")
    conductor_side: float = unit("m")
    conductor_volume: float = unit("m^3")
    conductor_mass: float = unit("kg")


def coil_electrical(solenoid: Solenoid, turns: float, conductor: MaterialLike = "cuag_conductor",
                    fill_factor: float = DEFAULT_FILL_FACTOR,
                    temperature: float = PRECOOL_TEMPERATURE) -> CoilElectrical:
    """Inductance, resistance and L/R time of a solenoid wound with ``turns`` turns of ``conductor``.

    * ``L = 2 E / I^2`` (:meth:`Solenoid.inductance_from_energy`, ~ N^2), Wheeler's value alongside;
    * ``R = rho_e(T) N^2 2 pi a_m / (f L Delta)``: N turns of mean length ``2 pi a_m`` in series, each
      with conductor cross-section ``f L Delta / N`` (exact Joule power for uniform current density);
    * ``tau = L/R`` is independent of N.

    ``turns`` > 0 (fractional allowed in a smeared model), ``fill_factor`` f in (0, 1] is the conductor
    fraction of the winding cross-section, ``temperature`` in K sets ``rho_e`` through
    ``Material.resistivity_at`` (linear in T).  Reference coil (a1 = 50 mm, a2 = 250 mm, L = 0.5 m),
    Cu-Ag at 77 K, f = 0.5: L = 91.0 nH N^2, R = 0.215 uohm N^2, tau = 0.42 s.
    Validity: uniform current density (conductor thinner than its skin depth during the pulse, or
    finely stranded and transposed); long-coil stored energy (beta >= 3); no magnetoresistance, which
    at 60-100 T raises the resistance of cold copper severalfold; no workpiece in the bore.

    Reference: Montgomery (1969); Wheeler (1928); Herlach & Miura (2003).
    """
    s = _solenoid(solenoid)
    n = _positive("turns", turns)
    f = _fill(fill_factor)
    t = _positive("temperature", temperature)
    m = _conductor(conductor, t)
    rho = m.resistivity_at(t)
    turn_length = 2.0 * math.pi * s.mean_radius
    area = f * s.length * s.winding_thickness / n
    resistance = rho * n * turn_length / area
    inductance = s.inductance_from_energy(n)
    volume = f * s.winding_volume
    layers = max(1, round(math.sqrt(n * s.winding_thickness / s.length)))
    return CoilElectrical(
        turns=n,
        layers=layers,
        conductor=m.key,
        fill_factor=f,
        temperature=t,
        resistivity=rho,
        inductance=inductance,
        inductance_wheeler=s.inductance(n),
        resistance=resistance,
        time_constant=inductance / resistance,
        current_per_tesla=s.current_for_field(1.0, n),
        mean_turn_length=turn_length,
        conductor_area=area,
        conductor_side=math.sqrt(area),
        conductor_volume=volume,
        conductor_mass=volume * m.density,
    )


def _charge_voltage(solenoid: Solenoid, field: float, rise_time: float, turns: float, conductor: Material,
                    fill_factor: float, temperature: float, stray_inductance: float) -> float:
    coil = coil_electrical(solenoid, turns, conductor, fill_factor, temperature)
    total = coil.inductance + stray_inductance
    cap = capacitance_for_rise_time(total, rise_time, coil.resistance)
    return voltage_for_peak_current(cap, total, solenoid.current_for_field(field, turns), coil.resistance)


def turns_for_voltage(solenoid: Solenoid, field: float, rise_time: float,
                      max_voltage: float = DEFAULT_MAX_VOLTAGE, conductor: MaterialLike = "cuag_conductor",
                      fill_factor: float = DEFAULT_FILL_FACTOR, temperature: float = PRECOOL_TEMPERATURE,
                      stray_inductance: float = 0.0) -> int:
    """Largest whole number of turns whose capacitor-bank charge voltage stays within ``max_voltage``.

    The charge voltage of a quarter-period discharge is proportional to the turn count,
    ``V0(N) = N V0(1)`` with ``V0(1) = pi E exp(R t/2L) / (N I t_rise)`` (the damping factor does not
    depend on N because both L and R scale as N^2), so ``N = floor(max_voltage / V0(1))``.  More turns
    mean less current, so the largest admissible N minimises the current the switches carry.  Returns
    1 when even a single turn needs more than ``max_voltage`` (the caller must then accept a higher
    voltage or a slower rise).  With a ``stray_inductance`` (H) the scaling is no longer exact and
    the count is reduced until the voltage fits.  ``field`` in T (> 0), ``rise_time`` in s (> 0),
    ``max_voltage`` in V (> 0).  Reference coil (100 mm bore) at 100.27 T and 25 kV: 1 turn for a 25 us
    rise (which really needs 267 kV), 3 turns for 1 ms, 37 for 10 ms.
    Validity: as :func:`coil_electrical` and :func:`rlc_current`.

    Reference: Knoepfel (1970); Herlach & Miura (2003).
    """
    s = _solenoid(solenoid)
    b = _positive("field", field)
    t = _positive("rise_time", rise_time)
    v_max = _positive("max_voltage", max_voltage)
    f = _fill(fill_factor)
    temp = _positive("temperature", temperature)
    m = _conductor(conductor, temp)
    stray = _non_negative("stray_inductance", stray_inductance)
    v_one = _charge_voltage(s, b, t, 1.0, m, f, temp, 0.0)
    n = max(1, int(math.floor(v_max / v_one * (1.0 + 1e-12))))
    if stray > 0.0:
        while n > 1 and _charge_voltage(s, b, t, float(n), m, f, temp, stray) > v_max:
            n -= 1
    return n


# ---------------------------------------------------------------------------
# capacitor-bank sizing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class BankDesign(Assessment):
    """Capacitor bank that drives ``peak_current`` into an R-L load in ``rise_time``.

    ``inductance`` is the load (coil) inductance and ``stray_inductance`` the bank/bus inductance in
    series with it; ``magnetic_energy`` = L_load I^2/2 is the useful field energy and ``bank_energy``
    = C V0^2/2 what must be stored; ``transfer_efficiency`` is their ratio.  ``action_to_peak`` =
    integral of I^2 dt up to the peak and ``action_crowbar`` adds the L/R decay I^2 L/(2R) (inf for
    R = 0); ``charge_to_peak`` is the charge the switches pass before the peak.  ``switch_count`` =
    ceil(I_pk / switch_current).  Mass and volume use the representative capacitor energy densities;
    the ``*_range`` tuples (low, high) span the estimated density ranges.  ``status`` is the worst of
    the four benchmark statuses and ``margin`` the smallest ratio of a state-of-the-art limit
    (50 MJ, 25 kV, 5 MA, 1 TW) to the requirement, so margin >= 1 exactly when all four are state of
    the art.

    Reference: Knoepfel (1970); magform.benchmarks.
    """

    inductance: float = unit("H")
    stray_inductance: float = unit("H")
    resistance: float = unit("ohm")
    peak_current: float = unit("A")
    rise_time: float = unit("s")
    capacitance: float = unit("F")
    charge_voltage: float = unit("V")
    bank_energy: float = unit("J")
    magnetic_energy: float = unit("J")
    energy_dissipated_to_peak: float = unit("J")
    transfer_efficiency: float
    peak_power: float = unit("W")
    peak_di_dt: float = unit("A/s")
    action_to_peak: float = unit("A^2 s")
    action_crowbar: float = unit("A^2 s")
    charge_to_peak: float = unit("C")
    regime: str
    damping_ratio: float
    voltage_reversal: float
    crowbar_time_constant: float = unit("s")
    switch_current: float = unit("A")
    switch_count: int
    bank_mass: float = unit("kg")
    bank_volume: float = unit("m^3")
    bank_mass_range: tuple[float, float] = unit("kg")
    bank_volume_range: tuple[float, float] = unit("m^3")
    energy_status: Status
    voltage_status: Status
    current_status: Status
    power_status: Status


def _sota_margin(energy: float, voltage: float, current: float, power: float) -> float:
    ratios = [limit / need for limit, need in ((ENERGY_MAGNET_DRIVER_SOTA, energy), (PULSED_VOLTAGE_SOTA, voltage),
                                               (PULSED_CURRENT_SOTA, current), (PEAK_POWER_SOTA, power))
              if need > 0.0]
    return min(ratios) if ratios else math.inf


def size_capacitor_bank(inductance: float, peak_current: float, rise_time: float, resistance: float = 0.0,
                        stray_inductance: float = 0.0, switch_current: float = SWITCH_CURRENT_RATING,
                        specific_energy: float = CAPACITOR_SPECIFIC_ENERGY,
                        energy_density: float = CAPACITOR_ENERGY_DENSITY) -> BankDesign:
    """Size the capacitor bank for a peak current, a rise time and an R-L load.

    ``C`` from :func:`capacitance_for_rise_time` with ``L_tot = L + L_stray``; ``V0`` from
    :func:`voltage_for_peak_current`; ``E_bank = C V0^2/2``; ``P_pk`` = max of ``V_C I``; switch modules
    ``ceil(I_pk / I_switch)``; mass ``E_bank / e_m`` and volume ``E_bank / e_v``.  Because
    ``I_pk = V0 sqrt(C/L) exp(-R t_rise / 2L)`` in every damping regime, the bank energy is exactly
    ``E_bank = (L_tot I_pk^2 / 2) exp(R t_rise / L_tot)``: the field energy of coil and stray
    inductance times ``exp(t_rise / tau)`` with ``tau = L/R``.  For a lossless circuit this reduces to
    ``V0 I_pk = pi E / t_rise`` and ``P_pk = pi E / (2 t_rise)`` (:func:`voltage_current_product`,
    :func:`discharge_peak_power`).

    Inputs: ``inductance`` L (H, > 0), ``peak_current`` (A, > 0), ``rise_time`` (s, > 0),
    ``resistance`` (ohm, >= 0), ``stray_inductance`` (H, >= 0), ``switch_current`` (A per module, > 0),
    ``specific_energy`` (J/kg, > 0), ``energy_density`` (J/m^3, > 0).
    Status: worst of ``benchmarks.energy_status(E_bank)``, ``pulsed_voltage_status(V0)``,
    ``pulsed_current_status(I_pk)`` and ``peak_power_status(P_pk)``.
    Validity: as :func:`rlc_current`; the capacitor energy densities are estimates
    (:data:`CAPACITOR_SPECIFIC_ENERGY_RANGE`), and a complete bank is 2-3 times heavier.

    Reference: Knoepfel (1970); Herlach & Miura (2003); magform.benchmarks.
    """
    ind = _positive("inductance", inductance)
    i_pk = _positive("peak_current", peak_current)
    t = _positive("rise_time", rise_time)
    r = _non_negative("resistance", resistance)
    stray = _non_negative("stray_inductance", stray_inductance)
    i_sw = _positive("switch_current", switch_current)
    e_m = _positive("specific_energy", specific_energy)
    e_v = _positive("energy_density", energy_density)

    total = ind + stray
    cap = capacitance_for_rise_time(total, t, r)
    v0 = voltage_for_peak_current(cap, total, i_pk, r)
    d = rlc_discharge(cap, v0, total, r)
    k = _circuit(cap, v0, total, r)
    ts = linspace(0.0, k.t_peak, _ACTION_SAMPLES + 1)
    action = trapz([k.current(x) ** 2 for x in ts], ts)
    action_crowbar = action + i_pk * i_pk * total / (2.0 * r) if r > 0.0 else math.inf
    bank_energy = d.stored_energy
    magnetic = 0.5 * ind * i_pk * i_pk

    s_energy = energy_status(bank_energy)
    s_voltage = pulsed_voltage_status(v0)
    s_current = pulsed_current_status(i_pk)
    s_power = peak_power_status(d.peak_power)
    status = worst(s_energy, s_voltage, s_current, s_power)

    lo_m, hi_m = CAPACITOR_SPECIFIC_ENERGY_RANGE
    lo_v, hi_v = CAPACITOR_ENERGY_DENSITY_RANGE
    warnings: list[str] = []
    notes: list[str] = [
        f"quarter-period law: V0 I_pk = pi E / t_rise = {voltage_current_product(0.5 * total * i_pk * i_pk, t):.3g} V A "
        f"for the {_si(0.5 * total * i_pk * i_pk, 'J')} in the circuit inductance; with losses the bank holds "
        f"{_si(bank_energy, 'J')} at {_si(v0, 'V')} and delivers {_si(i_pk, 'A')} ({d.regime}, zeta = "
        f"{d.damping_ratio:.3g})",
        f"peak power {_si(d.peak_power, 'W')}; {_si(d.energy_dissipated_to_peak, 'J')} is dissipated before the peak "
        f"and {d.transfer_efficiency:.1%} of the bank energy is in the load inductance at the peak",
        f"bank of ~{bank_energy / e_m / 1e3:.3g} t and ~{bank_energy / e_v:.3g} m^3 at {e_m / 1e3:g} kJ/kg and "
        f"{e_v / 1e6:g} MJ/m^3 (estimated capacitor-level densities; range {bank_energy / hi_m / 1e3:.3g}-"
        f"{bank_energy / lo_m / 1e3:.3g} t; a complete bank is 2-3x heavier)",
    ]
    if stray > 0.0:
        notes.append(f"stray inductance {_si(stray, 'H')} takes {stray / total:.1%} of the magnetic energy")
    if d.regime != "underdamped":
        warnings.append(f"{d.regime} discharge (rise time {t / d.crowbar_time_constant:.3g} L/R): most of the bank "
                        "energy is burnt in the resistance; a controlled rectifier ramp suits this time scale")
    if s_energy is not Status.STATE_OF_THE_ART:
        warnings.append(f"bank energy {_si(bank_energy, 'J')} exceeds the largest magnet-driver capacitor bank "
                        f"({_si(ENERGY_MAGNET_DRIVER_SOTA, 'J')}): {s_energy.label}")
    if s_voltage is not Status.STATE_OF_THE_ART:
        warnings.append(f"charge voltage {_si(v0, 'V')} exceeds pulsed-magnet bank practice "
                        f"({_si(PULSED_VOLTAGE_SOTA, 'V')}): {s_voltage.label}")
    if s_current is not Status.STATE_OF_THE_ART:
        warnings.append(f"peak current {_si(i_pk, 'A')} exceeds routine single-load currents "
                        f"({_si(PULSED_CURRENT_SOTA, 'A')}): {s_current.label}")
    if s_power is not Status.STATE_OF_THE_ART:
        warnings.append(f"peak power {_si(d.peak_power, 'W')} exceeds {_si(PEAK_POWER_SOTA, 'W')}: {s_power.label}")
    return BankDesign(
        status=status,
        margin=_sota_margin(bank_energy, v0, i_pk, d.peak_power),
        warnings=tuple(warnings),
        notes=tuple(notes),
        inductance=ind,
        stray_inductance=stray,
        resistance=r,
        peak_current=i_pk,
        rise_time=t,
        capacitance=cap,
        charge_voltage=v0,
        bank_energy=bank_energy,
        magnetic_energy=magnetic,
        energy_dissipated_to_peak=d.energy_dissipated_to_peak,
        transfer_efficiency=magnetic / bank_energy,
        peak_power=d.peak_power,
        peak_di_dt=d.peak_di_dt,
        action_to_peak=action,
        action_crowbar=action_crowbar,
        charge_to_peak=cap * (v0 - d.capacitor_voltage_at_peak),
        regime=d.regime,
        damping_ratio=d.damping_ratio,
        voltage_reversal=d.voltage_reversal,
        crowbar_time_constant=d.crowbar_time_constant,
        switch_current=i_sw,
        switch_count=int(math.ceil(i_pk / i_sw * (1.0 - 1e-12))),
        bank_mass=bank_energy / e_m,
        bank_volume=bank_energy / e_v,
        bank_mass_range=(bank_energy / hi_m, bank_energy / lo_m),
        bank_volume_range=(bank_energy / hi_v, bank_energy / lo_v),
        energy_status=s_energy,
        voltage_status=s_voltage,
        current_status=s_current,
        power_status=s_power,
    )


# ---------------------------------------------------------------------------
# conductor heating
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class ConductorHeating(ResultMixin):
    """Adiabatic temperature of a pulsed-coil conductor after one pulse.

    ``current_density`` is the peak value in the conductor (winding average / fill factor);
    ``action`` the specific action of the current waveform that is integrated explicitly (the
    quarter-sine rise, plus the symmetric fall for ``"half_sine"``); ``rise_phase_temperature_rise``
    the heating up to the field peak; ``stored_energy_temperature_rise`` what dissipating
    ``(1 - dump_fraction)`` of the peak field energy in the conductor adds (``"crowbar"`` only);
    ``self_energy_temperature_rise`` the rise if the whole field energy ended in the conductor
    (E / (m c_p), waveform-independent); ``max_rise_time`` the longest half-sine rise time that keeps
    the conductor below ``temperature_limit``.  Heating stops at the melting point (``melted``).

    Reference: Knoepfel (1970); Herlach & Miura (2003).
    """

    conductor: str
    waveform: str
    field: float = unit("T")
    rise_time: float = unit("s")
    fill_factor: float
    dump_fraction: float
    current_density: float = unit("A/m^2")
    action: float = unit("A^2 s/m^4")
    conductor_mass: float = unit("kg")
    magnetic_energy: float = unit("J")
    initial_temperature: float = unit("K")
    final_temperature: float = unit("K")
    temperature_rise: float = unit("K")
    rise_phase_temperature_rise: float = unit("K")
    stored_energy_temperature_rise: float = unit("K")
    self_energy_temperature_rise: float = unit("K")
    temperature_limit: float = unit("K")
    within_limit: bool
    melted: bool
    max_rise_time: float = unit("s")


def conductor_heating(solenoid: Solenoid, field: float, rise_time: float, waveform: str = "crowbar",
                      conductor: MaterialLike = "cuag_conductor", fill_factor: float = DEFAULT_FILL_FACTOR,
                      initial_temperature: float = PRECOOL_TEMPERATURE, dump_fraction: float = 0.0,
                      temperature_limit: float = COIL_TEMPERATURE_LIMIT) -> ConductorHeating:
    """Adiabatic heating of the coil conductor by one pulse to ``field`` (uniform current density).

    The conductor carries ``J_c = J_w / f`` with ``J_w`` the Montgomery winding-average density for
    the field.  Its temperature follows the action integral ``dT = rho_e(T) J_c^2 dt / (rho_m c_p)``
    (:func:`magform.eddy.adiabatic_joule_heating`):

    * ``"half_sine"``: ``J = J_c sin(pi t / (2 t_rise))`` for 0..2 t_rise (specific action
      ``J_c^2 t_rise``); the field energy returns to the bank - the lower bound of the heating;
    * ``"crowbar"``: the quarter-sine rise (action ``J_c^2 t_rise / 2``), then the coil is
      short-circuited and ``(1 - dump_fraction)`` of the peak field energy ``E`` is dissipated in the
      winding, adding ``(1 - d) E / (m_c c_p)`` whatever the decay time - the upper bound when
      ``dump_fraction`` = 0 (no external dump resistor).

    ``max_rise_time = G(T0 -> T_lim) / J_c^2`` with ``G`` from
    :func:`magform.eddy.specific_action_to_temperature`.  Inputs: ``field`` (T, >= 0), ``rise_time``
    (s, > 0), ``fill_factor`` in (0, 1], ``initial_temperature`` (K, default 77 K pre-cooled),
    ``dump_fraction`` in [0, 1], ``temperature_limit`` (K, above the initial temperature).
    Reference coil at 100.27 T, 10 ms, Cu-Ag from 77 K: +34 K for the half-sine but +666 K when the
    coil must absorb its own 99.5 MJ - a compact 100 T coil cannot swallow its field energy.
    Validity: uniform current density and temperature (no skin effect, no grading, no cooling during
    the pulse); registry resistivity law (linear in T, no magnetoresistance: at 60-100 T the true
    resistance of cold copper is severalfold higher, so the rise-phase heating is a lower estimate)
    and constant room-temperature c_p (about 15 % too high between 77 K and 200 K); long-coil field
    energy.

    Reference: Knoepfel (1970); Herlach & Miura (2003).
    """
    s = _solenoid(solenoid)
    b = _non_negative("field", field)
    t_r = _positive("rise_time", rise_time)
    wave = _waveform(waveform)
    f = _fill(fill_factor)
    t0 = _positive("initial_temperature", initial_temperature)
    m = _conductor(conductor, t0)
    dump = _fraction("dump_fraction", dump_fraction)
    t_lim = _positive("temperature_limit", temperature_limit)
    if t_lim <= t0:
        raise InputError(f"temperature_limit ({t_lim:g} K) must exceed initial_temperature ({t0:g} K)")

    j_c = s.current_density_for_field(b) / f
    volume = f * s.winding_volume
    mass = volume * m.density
    energy = s.stored_energy(b)
    rho_c = m.density * m.specific_heat
    omega = math.pi / (2.0 * t_r)

    ts_rise = linspace(0.0, t_r, _HEATING_SAMPLES + 1)
    rise = adiabatic_joule_heating([j_c * math.sin(omega * x) for x in ts_rise], ts_rise, m, t0)
    d_rise = rise.delta_t
    self_rise = energy / (volume * rho_c)

    if wave == "half_sine":
        ts = linspace(0.0, 2.0 * t_r, 2 * _HEATING_SAMPLES + 1)
        full = adiabatic_joule_heating([j_c * math.sin(omega * x) for x in ts], ts, m, t0)
        final, melted, action, d_store = full.final_temperature, full.melted, full.action, 0.0
    else:
        action = rise.action
        d_store = (1.0 - dump) * self_rise
        final = rise.final_temperature + d_store
        melted = rise.melted or final >= m.melting_point
        if melted:
            final = m.melting_point
            d_store = final - rise.final_temperature

    if j_c > 0.0:
        longest = specific_action_to_temperature(m, t0, t_lim) / (j_c * j_c)
    else:
        longest = math.inf
    return ConductorHeating(
        conductor=m.key,
        waveform=wave,
        field=b,
        rise_time=t_r,
        fill_factor=f,
        dump_fraction=dump,
        current_density=j_c,
        action=action,
        conductor_mass=mass,
        magnetic_energy=energy,
        initial_temperature=t0,
        final_temperature=final,
        temperature_rise=final - t0,
        rise_phase_temperature_rise=d_rise,
        stored_energy_temperature_rise=d_store,
        self_energy_temperature_rise=self_rise,
        temperature_limit=t_lim,
        within_limit=final <= t_lim,
        melted=melted,
        max_rise_time=longest,
    )


# ---------------------------------------------------------------------------
# homopolar generator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HomopolarGenerator(ResultMixin):
    """Homopolar (Faraday-disk / drum) generator as a pulsed source.

    A conducting rotor of outer radius ``rotor_radius`` ro, inner radius ``inner_radius`` ri and axial
    length ``rotor_length`` spins at surface speed ``tip_speed`` v in an axial flux density ``field`` B.
    With ``omega = v/ro``: mass ``m = rho pi (ro^2 - ri^2) l``, inertia ``J = m (ro^2 + ri^2)/2``,
    energy ``E = J omega^2/2`` (= m v^2/4 for a solid rotor), terminal voltage
    ``V = B omega (ro^2 - ri^2)/2``.  Because ``V = k omega`` and the braking torque is ``k I``
    (``k = B (ro^2 - ri^2)/2``), the machine is electrically a capacitor ``C_eq = J/k^2`` charged to V:
    it rings into an inductance L in a quarter period ``(pi/2) sqrt(L C_eq)``.  SI units; defaults
    200 m/s, 2 T, steel.
    Validity: rigid rotor, constant excitation, ideal brushes (the ~1 V brush drop and the internal
    resistance matter at ~100 V), no armature reaction.

    Reference: Woodson & Melcher (1968); Knoepfel (1970); Timoshenko & Goodier (1970).
    """

    rotor_radius: float = unit("m")
    rotor_length: float = unit("m")
    tip_speed: float = unit("m/s", default=HPG_TIP_SPEED)
    field: float = unit("T", default=HPG_FLUX_DENSITY)
    inner_radius: float = unit("m", default=0.0)
    density: float = unit("kg/m^3", default=ROTOR_DENSITY)
    internal_resistance: float = unit("ohm", default=0.0)

    def __post_init__(self) -> None:
        ro = _positive("rotor_radius", self.rotor_radius)
        _positive("rotor_length", self.rotor_length)
        _positive("tip_speed", self.tip_speed)
        _positive("field", self.field)
        ri = _non_negative("inner_radius", self.inner_radius)
        if ri >= ro:
            raise InputError(f"inner_radius ({ri!r} m) must be smaller than rotor_radius ({ro!r} m)")
        _positive("density", self.density)
        _non_negative("internal_resistance", self.internal_resistance)

    # -- mechanics ------------------------------------------------------------
    def mass(self) -> float:
        """Rotor mass ``rho pi (ro^2 - ri^2) l`` [kg].

        Reference: Woodson & Melcher (1968).
        """
        return self.density * math.pi * (self.rotor_radius ** 2 - self.inner_radius ** 2) * self.rotor_length

    def moment_of_inertia(self) -> float:
        """Polar moment of inertia ``m (ro^2 + ri^2)/2`` [kg m^2].

        Reference: Woodson & Melcher (1968).
        """
        return 0.5 * self.mass() * (self.rotor_radius ** 2 + self.inner_radius ** 2)

    def angular_speed(self) -> float:
        """Angular speed ``omega = v_tip / ro`` [rad/s].

        Reference: Woodson & Melcher (1968).
        """
        return self.tip_speed / self.rotor_radius

    def stored_energy(self) -> float:
        """Rotational energy ``J omega^2 / 2`` [J] (``m v^2 / 4`` for a solid rotor: 10 kJ/kg at 200 m/s).

        Reference: Woodson & Melcher (1968).
        """
        return 0.5 * self.moment_of_inertia() * self.angular_speed() ** 2

    def rotor_stress(self) -> float:
        """Largest hoop stress of the spinning rotor [Pa].

        Solid disk (centre): ``(3 + nu)/8 rho v^2``; with a bore of radius ri (at the bore):
        ``(3 + nu)/4 rho v^2 [1 + (1 - nu)/(3 + nu) (ri/ro)^2]`` - a pin hole doubles the stress.
        nu = 0.3.  130 MPa for a solid steel rotor at 200 m/s.  Validity: thin elastic disk.

        Reference: Timoshenko & Goodier (1970).
        """
        nu = ROTOR_POISSON
        rho_v2 = self.density * self.tip_speed ** 2
        if self.inner_radius == 0.0:
            return (3.0 + nu) / 8.0 * rho_v2
        ratio = self.inner_radius / self.rotor_radius
        return (3.0 + nu) / 4.0 * rho_v2 * (1.0 + (1.0 - nu) / (3.0 + nu) * ratio * ratio)

    # -- electrics ------------------------------------------------------------
    def voltage_constant(self) -> float:
        """``k = B (ro^2 - ri^2)/2`` [V s/rad]: ``V = k omega`` and torque ``= k I`` (flux cut per radian).

        Reference: Woodson & Melcher (1968).
        """
        return 0.5 * self.field * (self.rotor_radius ** 2 - self.inner_radius ** 2)

    def voltage(self) -> float:
        """Open-circuit terminal voltage ``V = B omega (ro^2 - ri^2)/2`` [V] (``B v ro / 2`` for a solid rotor).

        Of order 100 V for any practical rotor (2 T, 200 m/s, 0.5 m radius).

        Reference: Woodson & Melcher (1968).
        """
        return self.voltage_constant() * self.angular_speed()

    def equivalent_capacitance(self) -> float:
        """Capacitance the rotor presents to its load, ``C_eq = J / k^2 = 2 E / V^2`` [F].

        From ``J d(omega)/dt = -k I`` and ``V = k omega``: ``dV/dt = -I k^2/J``.  Kilofarads for MJ
        machines.

        Reference: Woodson & Melcher (1968); Knoepfel (1970).
        """
        k = self.voltage_constant()
        return self.moment_of_inertia() / (k * k)

    def quarter_period(self, inductance: float) -> float:
        """Rise time of the lossless discharge into ``inductance`` [H], ``(pi/2) sqrt(L C_eq)`` [s].

        Reference: Knoepfel (1970).
        """
        return 0.5 * math.pi * math.sqrt(_positive("inductance", inductance) * self.equivalent_capacitance())

    def peak_current(self, inductance: float) -> float:
        """Lossless peak current into ``inductance`` [H], ``V sqrt(C_eq / L)`` [A] (all rotor energy in L).

        Reference: Knoepfel (1970).
        """
        return self.voltage() * math.sqrt(self.equivalent_capacitance() / _positive("inductance", inductance))

    def discharge(self, inductance: float, resistance: float = 0.0) -> RLCDischarge:
        """Discharge into an R-L load as the equivalent RLC circuit (:func:`rlc_discharge`).

        ``inductance`` in H (> 0); ``resistance`` in ohm (>= 0) is added to ``internal_resistance``.

        Reference: Woodson & Melcher (1968); Knoepfel (1970).
        """
        r = _non_negative("resistance", resistance) + self.internal_resistance
        return rlc_discharge(self.equivalent_capacitance(), self.voltage(), inductance, r)


def rotor_tip_speed_limit(allowable_stress: float, density: float = ROTOR_DENSITY) -> float:
    """Surface speed at which a solid rotating disk reaches ``allowable_stress`` at its centre [m/s].

    ``v = sqrt(8 sigma / ((3 + nu) rho))`` with nu = 0.3; the energy stored per unit rotor mass is then
    ``v^2/4 = 2 sigma / ((3 + nu) rho)``.  ``allowable_stress`` in Pa (> 0), ``density`` in kg/m^3 (> 0).
    Steel at 500 MPa: 393 m/s and 39 kJ/kg - the mechanical ceiling; sliding brush contacts hold
    homopolar machines to about half that speed (:data:`HPG_TIP_SPEED`), a quarter of the energy.
    Validity: thin, solid, elastic disk; a central bore doubles the stress
    (:meth:`HomopolarGenerator.rotor_stress`).

    Reference: Timoshenko & Goodier (1970).
    """
    sigma = _positive("allowable_stress", allowable_stress)
    rho = _positive("density", density)
    return math.sqrt(8.0 * sigma / ((3.0 + ROTOR_POISSON) * rho))


def homopolar_for_energy(energy: float, tip_speed: float = HPG_TIP_SPEED, field: float = HPG_FLUX_DENSITY,
                         aspect_ratio: float = 1.0, density: float = ROTOR_DENSITY) -> HomopolarGenerator:
    """Solid-rotor homopolar generator that stores ``energy`` at ``tip_speed``.

    ``m = 4 E / v^2`` (solid cylinder), ``ro = (m / (2 pi rho A))^(1/3)`` and length ``l = 2 A ro`` with
    ``A`` = ``aspect_ratio`` (length / diameter).  ``energy`` in J (> 0), ``tip_speed`` in m/s (> 0),
    ``field`` in T (> 0), ``density`` in kg/m^3 (> 0).  Example: 99.5 MJ at 200 m/s and 2 T is a 9.9 t
    rotor of 0.59 m radius giving 117 V, i.e. a 14.5 kF equivalent capacitor.
    Validity: as :class:`HomopolarGenerator`; a single rotor (real supplies stack several).

    Reference: Woodson & Melcher (1968).
    """
    e = _positive("energy", energy)
    v = _positive("tip_speed", tip_speed)
    b = _positive("field", field)
    a = _positive("aspect_ratio", aspect_ratio)
    rho = _positive("density", density)
    mass = 4.0 * e / (v * v)
    radius = (mass / (2.0 * math.pi * rho * a)) ** (1.0 / 3.0)
    return HomopolarGenerator(rotor_radius=radius, rotor_length=2.0 * a * radius, tip_speed=v, field=b,
                              density=rho)


# ---------------------------------------------------------------------------
# source comparison
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class SourceOption(ResultMixin):
    """One class of pulsed energy source graded for an (energy, rise time) requirement.

    ``status`` grades delivering one pulse; ``fits`` is False when the status is infeasible;
    ``reusable`` is False for a source consumed by the shot.  ``fastest_rise_time`` is the shortest
    rise the source class delivers at this energy without leaving demonstrated practice (estimated);
    ``peak_power`` = pi E / (2 t_rise) is the requirement.  ``reason`` says why it does or does not
    suit.

    Reference: Knoepfel (1970); magform.benchmarks.
    """

    key: str
    name: str
    status: Status
    fits: bool
    reusable: bool
    energy: float = unit("J")
    rise_time: float = unit("s")
    peak_power: float = unit("W")
    fastest_rise_time: float = unit("s")
    reason: str
    reference: str


def source_comparison(energy: float, rise_time: float) -> tuple[SourceOption, ...]:
    """Grade the four classes of pulsed source for delivering ``energy`` with a ``rise_time``.

    The requirement is the peak power ``P = pi E / (2 t_rise)``.  In order:

    1. ``capacitor_bank`` - any rise time; status worst(``energy_status(E)``, ``peak_power_status(P)``);
       fastest state-of-the-art rise ``pi E / (2 x 1 TW)``.
    2. ``motor_generator`` (flywheel + rectifier) - infeasible below ~10 ms
       (:data:`MOTOR_GENERATOR_MIN_RISE_TIME`); else graded against 1.4 GW and ~0.6 GJ per pulse
       (LANL): state of the art within them, extrapolation to 10x, aspirational beyond.
    3. ``homopolar_generator`` - electrically a capacitor ``C = 2E/V^2`` at ~600 V at most; even into
       its own ~0.5 uH it cannot ring faster than ``(pi/2) sqrt(L_int 2E/V^2)``, so it is infeasible for
       shorter rises; else graded by energy against 60 MJ (Balcones) and 500 MJ (ANU).  An inductive
       store with an opening switch can shorten the pulse to milliseconds but passes at most 25 % of
       the energy to an uncoupled load inductor (:func:`inductive_transfer_efficiency`).
    4. ``flux_compression_generator`` (explosive) - microseconds to ~1 ms only, graded by energy
       against ~30 MJ; single shot: it and usually the load are destroyed (``reusable`` False).

    ``energy`` in J (> 0), ``rise_time`` in s (> 0).
    Validity: order-of-magnitude grading against the cited machines; the windows and the factor of
    10 are judgement (documented module constants).

    Reference: Knoepfel (1970); Herlach & Miura (2003); Fowler et al. (1960); magform.benchmarks.
    """
    e = _positive("energy", energy)
    t = _positive("rise_time", rise_time)
    power = discharge_peak_power(e, t)
    ladders = REFERENCES["benchmark_ladders"]

    # 1. capacitor bank
    cap_status = worst(energy_status(e), peak_power_status(power))
    cap_fast = 0.5 * math.pi * e / PEAK_POWER_SOTA   # rise time at which pi E/(2 t) reaches 1 TW
    cap_reason = (f"capacitors discharge on any time scale; {_si(e, 'J')} in {_si(t, 's')} is {_si(power, 'W')} peak "
                  f"(largest magnet-driver bank {_si(ENERGY_MAGNET_DRIVER_SOTA, 'J')}; TW-class power stays state "
                  f"of the art down to a {_si(cap_fast, 's')} rise at this energy)")
    options = [SourceOption(
        key="capacitor_bank", name="capacitor bank", status=cap_status, fits=True, reusable=True, energy=e,
        rise_time=t, peak_power=power, fastest_rise_time=cap_fast,
        reason=cap_reason, reference=ladders)]

    # 2. motor-generator
    mg_fast = max(MOTOR_GENERATOR_MIN_RISE_TIME, 0.5 * math.pi * e / MOTOR_GENERATOR_POWER)
    if t < MOTOR_GENERATOR_MIN_RISE_TIME:
        mg_status = Status.INFEASIBLE
        mg_reason = (f"a rectified alternator cannot rise in less than about a half-cycle of its output "
                     f"({_si(MOTOR_GENERATOR_MIN_RISE_TIME, 's')}); the request is {MOTOR_GENERATOR_MIN_RISE_TIME / t:.3g}x "
                     "faster")
    else:
        mg_status = worst(_grade(power, MOTOR_GENERATOR_POWER), _grade(e, MOTOR_GENERATOR_ENERGY))
        mg_reason = (f"needs {_si(power, 'W')} and {_si(e, 'J')} per pulse against the {_si(MOTOR_GENERATOR_POWER, 'W')}, "
                     f"~{_si(MOTOR_GENERATOR_ENERGY, 'J')}-per-pulse motor-generator at Los Alamos: {mg_status.label}; "
                     "right for a slow outer platform coil, not for an eddy-current pulse")
    options.append(SourceOption(
        key="motor_generator", name="motor-generator (flywheel and rectifier)", status=mg_status,
        fits=mg_status is not Status.INFEASIBLE, reusable=True, energy=e, rise_time=t, peak_power=power,
        fastest_rise_time=mg_fast, reason=mg_reason, reference=ladders))

    # 3. homopolar generator
    c_eq = 2.0 * e / HPG_VOLTAGE_CLASS ** 2
    hpg_fast = 0.5 * math.pi * math.sqrt(HPG_INTERNAL_INDUCTANCE * c_eq)
    if t < hpg_fast:
        hpg_status = Status.INFEASIBLE
        hpg_reason = (f"at <= {HPG_VOLTAGE_CLASS:g} V a rotor storing {_si(e, 'J')} is a {_si(c_eq, 'F')} capacitor: even "
                      f"into its own ~{_si(HPG_INTERNAL_INDUCTANCE, 'H')} it needs {_si(hpg_fast, 's')} to peak, "
                      f"{hpg_fast / t:.3g}x slower than asked; an inductive store with an opening switch shortens "
                      f"the pulse but passes at most {INDUCTIVE_TRANSFER_LIMIT:.0%} of the energy to a load inductor")
    else:
        hpg_status = Status.STATE_OF_THE_ART if e <= HPG_ENERGY_SOTA else (
            Status.EXTRAPOLATION if e <= HPG_ENERGY_RECORD else Status.ASPIRATIONAL)
        hpg_reason = (f"low-voltage (~100 V per rotor), mega-ampere source with a natural rise of >= "
                      f"{_si(hpg_fast, 's')} at this energy; {_si(e, 'J')} against the {_si(HPG_ENERGY_SOTA, 'J')} "
                      f"Balcones supply and the {_si(HPG_ENERGY_RECORD, 'J')} ANU machine: {hpg_status.label}")
    options.append(SourceOption(
        key="homopolar_generator", name="homopolar generator", status=hpg_status,
        fits=hpg_status is not Status.INFEASIBLE, reusable=True, energy=e, rise_time=t, peak_power=power,
        fastest_rise_time=hpg_fast, reason=hpg_reason,
        reference=REFERENCES["woodson_melcher1968"] + " " + ladders))

    # 4. explosive flux-compression generator
    if t > FCG_MAX_RISE_TIME:
        fcg_status = Status.INFEASIBLE
        fcg_reason = (f"an explosive generator is swept by its detonation in <~ {_si(FCG_MAX_RISE_TIME, 's')}; it "
                      f"cannot stretch its pulse to {_si(t, 's')}")
    else:
        fcg_status = _grade(e, FCG_ENERGY_DEMONSTRATED)
        fcg_reason = (f"delivers tens of MA in microseconds ({_si(e, 'J')} against ~{_si(FCG_ENERGY_DEMONSTRATED, 'J')} "
                      f"demonstrated: {fcg_status.label}) but is consumed by every shot, usually with its load: an "
                      "experiment, not a production source")
    options.append(SourceOption(
        key="flux_compression_generator", name="explosive flux-compression generator", status=fcg_status,
        fits=fcg_status is not Status.INFEASIBLE, reusable=False, energy=e, rise_time=t, peak_power=power,
        fastest_rise_time=FCG_MIN_RISE_TIME, reason=fcg_reason,
        reference=REFERENCES["fowler1960"] + " " + REFERENCES["knoepfel1970"]))
    return tuple(options)


def _recommended(options: tuple[SourceOption, ...]) -> str:
    """Key of the best-graded reusable source that fits ('none' if there is none)."""
    usable = [o for o in options if o.fits and o.reusable]
    if not usable:
        return "none"
    return min(usable, key=lambda o: o.status.rank).key


# ---------------------------------------------------------------------------
# network assessment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class PulseNetworkAssessment(Assessment):
    """Capacitor-bank network that drives a solenoid to ``field`` in ``rise_time``, with its limits.

    Electrical: ``turns`` (chosen by :func:`turns_for_voltage` unless given; ``voltage_limited`` is
    True when the charge voltage exceeds ``max_voltage``, which with the automatic choice means that
    even one turn is too many), ``ampere_turns`` N I, ``inductance`` (coil,
    energy-consistent), ``resistance``, ``time_constant`` L/R, ``peak_current``, ``charge_voltage``,
    ``capacitance``, ``magnetic_energy`` (coil), ``bank_energy``, ``peak_power``, damping, and
    ``ramp_voltage`` = 2E/(I t), the terminal voltage of a linear rectifier-driven ramp to the same
    current.  ``minimum_rise_time`` is the fastest lossless rise ``max_voltage`` allows with a single
    turn, and ``homopolar_rise_time`` the quarter period of an energy-matched homopolar generator
    into the single-turn coil (its fastest).
    Thermal: ``conductor_temperature_rise`` for the chosen ``waveform`` and ``dump_fraction``;
    ``half_sine_temperature_rise`` (field energy returned to the bank: lower bound);
    ``self_energy_temperature_rise`` (all of it dissipated in the winding); ``max_rise_time_thermal``
    (longest half-sine rise below ``temperature_limit``); ``skin_depth`` at f = 1/(4 t_rise),
    ``skin_limited`` (solid conductor thicker than 2 skin depths), ``layer_field_step`` = B / layers
    and ``surface_temperature_rise`` (skin heating of a skin-limited conductor by that field step).
    ``status`` = worst of the five component statuses; ``margin`` = smallest ratio of a
    state-of-the-art electrical limit (50 MJ, 25 kV, 5 MA, 1 TW) to the requirement.

    Reference: Knoepfel (1970); Herlach & Miura (2003); magform.benchmarks.
    """

    field: float = unit("T")
    rise_time: float = unit("s")
    bore_diameter: float = unit("m")
    winding_thickness: float = unit("m")
    coil_length: float = unit("m")
    turns: int
    layers: int
    max_voltage: float = unit("V")
    voltage_limited: bool
    ampere_turns: float = unit("A")
    inductance: float = unit("H")
    inductance_wheeler: float = unit("H")
    stray_inductance: float = unit("H")
    resistance: float = unit("ohm")
    time_constant: float = unit("s")
    peak_current: float = unit("A")
    charge_voltage: float = unit("V")
    ramp_voltage: float = unit("V")
    capacitance: float = unit("F")
    magnetic_energy: float = unit("J")
    bank_energy: float = unit("J")
    energy_dissipated_to_peak: float = unit("J")
    transfer_efficiency: float
    peak_power: float = unit("W")
    peak_di_dt: float = unit("A/s")
    damping_regime: str
    damping_ratio: float
    voltage_reversal: float
    minimum_rise_time: float = unit("s")
    homopolar_rise_time: float = unit("s")
    energy_status: Status
    voltage_status: Status
    current_status: Status
    power_status: Status
    thermal_status: Status
    conductor: str
    fill_factor: float
    waveform: str
    dump_fraction: float
    transposed: bool
    conductor_current_density: float = unit("A/m^2")
    initial_temperature: float = unit("K")
    temperature_limit: float = unit("K")
    conductor_temperature_rise: float = unit("K")
    conductor_final_temperature: float = unit("K")
    half_sine_temperature_rise: float = unit("K")
    self_energy_temperature_rise: float = unit("K")
    max_rise_time_thermal: float = unit("s")
    skin_depth: float = unit("m")
    conductor_side: float = unit("m")
    skin_limited: bool
    layer_field_step: float = unit("T")
    surface_temperature_rise: float = unit("K")
    surface_melts: bool
    bank_mass: float = unit("kg")
    bank_volume: float = unit("m^3")
    switch_count: int
    switch_current: float = unit("A")
    recommended_source: str
    sources: tuple[SourceOption, ...]


def assess_pulse_network(
    solenoid: Solenoid,
    field: float,
    rise_time: float,
    turns: int | None = None,
    max_voltage: float = DEFAULT_MAX_VOLTAGE,
    conductor: MaterialLike = "cuag_conductor",
    fill_factor: float = DEFAULT_FILL_FACTOR,
    initial_temperature: float = PRECOOL_TEMPERATURE,
    waveform: str = "crowbar",
    dump_fraction: float = 0.0,
    stray_inductance: float = 0.0,
    switch_current: float = SWITCH_CURRENT_RATING,
    temperature_limit: float = COIL_TEMPERATURE_LIMIT,
    transposed: bool = False,
) -> PulseNetworkAssessment:
    """Size and grade the pulse-discharge network for a coil, a field and a rise time.

    Chain: ``I = solenoid.current_for_field(B, N)``; :func:`coil_electrical` (L ~ N^2, R ~ N^2);
    :func:`size_capacitor_bank` (C, V0, E_bank, P_pk, switches, mass); :func:`conductor_heating`;
    :func:`source_comparison`.  ``turns`` None -> :func:`turns_for_voltage` (largest N within
    ``max_voltage``, default 25 kV).  Governing law: ``V0 I_pk = pi E / t_rise`` - the winding can
    only trade current for voltage.

    Thermal grading of a reusable coil (``thermal_status``): state of the art when the conductor
    ends the pulse below ``temperature_limit`` (400 K); extrapolation when hotter but solid (needs
    an external dump resistor, energy recovery or a bigger winding); infeasible when it melts.  Two
    mechanisms are checked: the bulk heating of the chosen ``waveform``, and - unless ``transposed``
    is True (finely stranded, transposed conductor) - the skin heating of a solid conductor thicker
    than two skin depths, ``dT = (B/n_layers)^2 / (2 mu0 rho_m c_p)``
    (:func:`magform.eddy.skin_temperature_rise` with the field step across one layer and
    kappa = 1, its lower estimate).  A
    single-layer coil of massive turns at >= 89 T (Cu-Ag from 77 K) melts its bore surface: the
    destructive single-turn-coil regime.

    Status: worst of the energy, voltage, current and power benchmark statuses and the thermal
    status.  Inputs: ``field`` (T, > 0), ``rise_time`` (s, > 0), ``turns`` (integer >= 1 or None),
    ``max_voltage`` (V, > 0), ``fill_factor`` in (0, 1], ``initial_temperature`` (K), ``waveform``
    "crowbar" | "half_sine", ``dump_fraction`` in [0, 1], ``stray_inductance`` (H, >= 0),
    ``switch_current`` (A, > 0), ``temperature_limit`` (K).
    Reference coil (100 mm bore, 0.2 m build, 0.5 m long) at 100.27 T: 99.5 MJ of field energy;
    25 us -> one turn, 46.7 MA at 267 kV, 6.25 TW (infeasible as a reusable coil); 10 ms -> 37 turns,
    1.26 MA at 24.9 kV, a 102 MJ bank (extrapolation).
    Validity: as :func:`coil_electrical`, :func:`rlc_current` and :func:`conductor_heating`; no
    workpiece in the bore (a shielding workpiece lowers L and E); coil stresses are assessed in
    :mod:`magform.magnetics` / :mod:`magform.conductor`, not here.

    Reference: Knoepfel (1970); Herlach & Miura (2003); Psyk et al. (2011); magform.benchmarks.
    """
    s = _solenoid(solenoid)
    b = _positive("field", field)
    t_r = _positive("rise_time", rise_time)
    v_max = _positive("max_voltage", max_voltage)
    f = _fill(fill_factor)
    t0 = _positive("initial_temperature", initial_temperature)
    m = _conductor(conductor, t0)
    wave = _waveform(waveform)
    dump = _fraction("dump_fraction", dump_fraction)
    stray = _non_negative("stray_inductance", stray_inductance)
    i_sw = _positive("switch_current", switch_current)
    t_lim = _positive("temperature_limit", temperature_limit)
    if not isinstance(transposed, bool):
        raise InputError(f"transposed must be a bool, got {transposed!r}")
    if turns is None:
        n = turns_for_voltage(s, b, t_r, v_max, m, f, t0, stray)
    else:
        n = _count("turns", turns)

    coil = coil_electrical(s, n, m, f, t0)
    i_pk = s.current_for_field(b, n)
    ampere_turns = i_pk * n
    bank = size_capacitor_bank(coil.inductance, i_pk, t_r, coil.resistance, stray, i_sw)
    energy = s.stored_energy(b)
    heat = conductor_heating(s, b, t_r, wave, m, f, t0, dump, t_lim)
    half = heat if wave == "half_sine" else conductor_heating(s, b, t_r, "half_sine", m, f, t0, 0.0, t_lim)

    # skin effect in the coil conductor
    delta = skin_depth(coil.resistivity, 1.0 / (4.0 * t_r))
    skin_limited = (not transposed) and coil.conductor_side > SKIN_LIMIT_RATIO * delta
    layer_step = b / coil.layers
    surface_rise = skin_temperature_rise(m, layer_step) if skin_limited else 0.0
    surface_melts = skin_limited and layer_step >= skin_melting_field(m, t0)

    hottest = max(heat.final_temperature, t0 + surface_rise)
    if heat.melted or surface_melts:
        s_thermal = Status.INFEASIBLE
    elif hottest > t_lim:
        s_thermal = Status.EXTRAPOLATION
    else:
        s_thermal = Status.STATE_OF_THE_ART

    one_turn = coil_electrical(s, 1.0, m, f, t0)
    minimum_rise = rise_time_for_product(energy, v_max, ampere_turns)
    hpg = homopolar_for_energy(energy)
    hpg_rise = hpg.quarter_period(one_turn.inductance + stray)
    sources = source_comparison(bank.bank_energy, t_r)
    recommended = _recommended(sources)
    voltage_limited = bank.charge_voltage > v_max * (1.0 + 1e-9)
    status = worst(bank.energy_status, bank.voltage_status, bank.current_status, bank.power_status, s_thermal)

    warnings: list[str] = [w for w in bank.warnings]
    notes: list[str] = [
        f"field energy {_si(energy, 'J')} at {b:.4g} T in a {2e3 * s.inner_radius:.4g} mm bore, {s.length:.3g} m long "
        f"coil ({_si(ampere_turns, 'A')}-turns); quarter-period law V0 I_pk = pi E / t_rise = "
        f"{voltage_current_product(energy, t_r):.3g} V A whatever the winding",
        f"{n} turn{'s' if n != 1 else ''} ({coil.layers} layer{'s' if coil.layers != 1 else ''}): L = "
        f"{_si(coil.inductance, 'H')}, R = {_si(coil.resistance, 'ohm')} ({m.key} at {t0:g} K, fill {f:g}), L/R = "
        f"{_si(coil.time_constant, 's')}; bank {_si(bank.capacitance, 'F')} at {_si(bank.charge_voltage, 'V')}, "
        f"{_si(bank.bank_energy, 'J')}, {_si(i_pk, 'A')} peak, {_si(bank.peak_power, 'W')} peak, "
        f"{bank.switch_count} switch modules of {_si(i_sw, 'A')}",
        f"with one turn a {_si(v_max, 'V')} bank cannot rise faster than {_si(minimum_rise, 's')} in this magnet; an "
        f"energy-matched homopolar generator ({hpg.voltage():.0f} V, {_si(hpg.equivalent_capacitance(), 'F')} "
        f"equivalent) needs {_si(hpg_rise, 's')} even into the single-turn coil, {hpg_rise / t_r:.3g}x the requested "
        "rise",
        f"conductor ({_si(heat.current_density, 'A/m^2')} peak): +{half.temperature_rise:.3g} K if the field energy "
        f"returns to the bank (half-sine), +{heat.self_energy_temperature_rise:.3g} K if the winding absorbs all of "
        f"it; longest half-sine rise below {t_lim:g} K: {_si(heat.max_rise_time, 's')}",
        f"recommended reusable source for {_si(bank.bank_energy, 'J')} in {_si(t_r, 's')}: {recommended.replace('_', ' ')}",
        "eddy-current forming needs the rise to be shorter than the workpiece's flux-diffusion time "
        "(magform.eddy.shell_time_constant); a slower pulse leaks through the wall and delivers less pressure",
        "resistance uses the registry's linear rho(T) without magnetoresistance and the heating a constant c_p: "
        "both make the temperatures lower estimates at 60-100 T",
    ]
    if voltage_limited and n == 1:
        warnings.append(f"even a single turn needs {_si(bank.charge_voltage, 'V')} (> {_si(v_max, 'V')}): the rise time "
                        "is below what the voltage ceiling allows for this field energy")
    elif voltage_limited:
        warnings.append(f"{n} turns need {_si(bank.charge_voltage, 'V')} (> {_si(v_max, 'V')}); the ceiling allows "
                        f"{turns_for_voltage(s, b, t_r, v_max, m, f, t0, stray)} turn(s) at this rise time")
    if skin_limited:
        text = (f"skin-limited conductor: {1e3 * coil.conductor_side:.3g} mm solid turns against a "
                f"{1e3 * delta:.3g} mm skin depth - the current is not uniform, so resistance and heating are "
                f"underestimated; skin heating +{surface_rise:.0f} K for the {layer_step:.3g} T step across one layer")
        if surface_melts:
            text += (f", above the {skin_melting_field(m, t0):.3g} T at which the surface of {m.key} melts from "
                     f"{t0:g} K: the destructive single-turn-coil regime, not a reusable coil")
        else:
            text += "; a finely stranded, transposed conductor (transposed=True) avoids it at a lower fill factor"
        warnings.append(text)
    if heat.melted:
        warnings.append(f"the conductor reaches its melting point ({m.melting_point:g} K) within one {wave} pulse")
    elif not heat.within_limit:
        warnings.append(f"conductor ends the {wave} pulse at {heat.final_temperature:.0f} K (> {t_lim:g} K): the coil "
                        "cannot absorb its own field energy; an external dump resistor, energy recovery or a larger "
                        "winding is needed")
    if recommended == "none":
        warnings.append("no reusable source class fits this energy and rise time")
    return PulseNetworkAssessment(
        status=status,
        margin=bank.margin,
        warnings=tuple(warnings),
        notes=tuple(notes),
        field=b,
        rise_time=t_r,
        bore_diameter=2.0 * s.inner_radius,
        winding_thickness=s.winding_thickness,
        coil_length=s.length,
        turns=n,
        layers=coil.layers,
        max_voltage=v_max,
        voltage_limited=voltage_limited,
        ampere_turns=ampere_turns,
        inductance=coil.inductance,
        inductance_wheeler=coil.inductance_wheeler,
        stray_inductance=stray,
        resistance=coil.resistance,
        time_constant=coil.time_constant,
        peak_current=i_pk,
        charge_voltage=bank.charge_voltage,
        ramp_voltage=terminal_voltage(energy, i_pk, t_r),
        capacitance=bank.capacitance,
        magnetic_energy=energy,
        bank_energy=bank.bank_energy,
        energy_dissipated_to_peak=bank.energy_dissipated_to_peak,
        transfer_efficiency=bank.transfer_efficiency,
        peak_power=bank.peak_power,
        peak_di_dt=bank.peak_di_dt,
        damping_regime=bank.regime,
        damping_ratio=bank.damping_ratio,
        voltage_reversal=bank.voltage_reversal,
        minimum_rise_time=minimum_rise,
        homopolar_rise_time=hpg_rise,
        energy_status=bank.energy_status,
        voltage_status=bank.voltage_status,
        current_status=bank.current_status,
        power_status=bank.power_status,
        thermal_status=s_thermal,
        conductor=m.key,
        fill_factor=f,
        waveform=wave,
        dump_fraction=dump,
        transposed=transposed,
        conductor_current_density=heat.current_density,
        initial_temperature=t0,
        temperature_limit=t_lim,
        conductor_temperature_rise=heat.temperature_rise,
        conductor_final_temperature=heat.final_temperature,
        half_sine_temperature_rise=half.temperature_rise,
        self_energy_temperature_rise=heat.self_energy_temperature_rise,
        max_rise_time_thermal=heat.max_rise_time,
        skin_depth=delta,
        conductor_side=coil.conductor_side,
        skin_limited=skin_limited,
        layer_field_step=layer_step,
        surface_temperature_rise=surface_rise,
        surface_melts=surface_melts,
        bank_mass=bank.bank_mass,
        bank_volume=bank.bank_volume,
        switch_count=bank.switch_count,
        switch_current=i_sw,
        recommended_source=recommended,
        sources=sources,
    )


__all__ = [
    "REFERENCES", "REGIMES", "WAVEFORMS", "CRITICAL_BAND", "DEFAULT_FILL_FACTOR", "PRECOOL_TEMPERATURE",
    "COIL_TEMPERATURE_LIMIT", "DEFAULT_MAX_VOLTAGE", "SWITCH_CURRENT_RATING", "CAPACITOR_SPECIFIC_ENERGY",
    "CAPACITOR_SPECIFIC_ENERGY_RANGE", "CAPACITOR_ENERGY_DENSITY", "CAPACITOR_ENERGY_DENSITY_RANGE",
    "INDUCTIVE_TRANSFER_LIMIT", "SKIN_LIMIT_RATIO", "ROTOR_DENSITY", "ROTOR_POISSON", "HPG_TIP_SPEED",
    "HPG_FLUX_DENSITY", "HPG_VOLTAGE_CLASS", "HPG_INTERNAL_INDUCTANCE", "HPG_ENERGY_SOTA", "HPG_ENERGY_RECORD",
    "MOTOR_GENERATOR_POWER", "MOTOR_GENERATOR_ENERGY", "MOTOR_GENERATOR_MIN_RISE_TIME", "FCG_MIN_RISE_TIME",
    "FCG_MAX_RISE_TIME",
    "FCG_ENERGY_DEMONSTRATED", "SOURCE_EXTRAPOLATION_FACTOR",
    "angular_frequency_for_rise", "voltage_current_product", "discharge_peak_power", "rise_time_for_product",
    "inductive_transfer_efficiency",
    "RLCDischarge", "rlc_current", "rlc_capacitor_voltage", "rlc_discharge", "capacitance_for_rise_time",
    "voltage_for_peak_current",
    "CoilElectrical", "coil_electrical", "turns_for_voltage",
    "BankDesign", "size_capacitor_bank",
    "ConductorHeating", "conductor_heating",
    "HomopolarGenerator", "rotor_tip_speed_limit", "homopolar_for_energy",
    "SourceOption", "source_comparison",
    "PulseNetworkAssessment", "assess_pulse_network",
]
