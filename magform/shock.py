"""Planar shock physics: impedance matching and magnetically driven flyer kinematics.

Shared by :mod:`magform.welding` (impact welding) and :mod:`magform.synthesis`
(shock synthesis of diamond/lonsdaleite), which both need to know how fast a
magnetically accelerated flyer plate travels and what pressure its impact
creates in the target.

Model summary
-------------
* **Hugoniot**: linear shock-velocity/particle-velocity fit ``Us = c0 + s*up``
  per material (``materials.Material.hugoniot_for``) with the Rankine-Hugoniot
  momentum jump ``P = rho0*Us*up`` from an unshocked, stress-free state
  [marsh_1980, mcqueen_1970].
* **Impedance match** of a flyer hitting a stationary target at velocity ``v``:
  pressure and particle velocity are continuous across the contact, so
  ``P_flyer(v - u) = P_target(u)`` [meyers_1994].
* **Phase transitions** (alpha-epsilon iron at 13 GPa [bancroft_1956]): a
  material that records ``phase_transition_pressure`` and ``hugoniot_high``
  is evaluated on a continuous piecewise Hugoniot - low-pressure branch up to
  the transition, a mixed-phase plateau at the transition pressure, and the
  high-pressure branch beyond.  This reproduces the "solve on the low branch,
  re-solve on the high branch if P exceeds the transition" rule whenever that
  rule is self-consistent, and pins the state at the transition pressure in
  the band where it is not.  Between 0.8x and 1.5x the transition pressure the
  real wave splits into two (two-wave region) and the single-shock answer is
  approximate; :class:`ShockState` says so in ``branch_note``
  [duvall_graham_1977].
* **Flyer kinematics**: a sheet of areal mass ``rho*h`` pushed by a constant
  pressure ``eta*P`` over a gap ``g`` reaches ``v = sqrt(2*eta*P*g/(rho*h))``
  (work-energy theorem) [psyk_2011].

Validity: one-dimensional planar impact, strength (Hugoniot elastic limit,
elastic precursor) neglected, linear Us-up fits inside their recorded range
(``Hugoniot.p_max``), constant drive pressure during the flight.  All SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._numerics import bisect, solve_quadratic
from ._report import ResultMixin, unit
from .errors import InputError
from .materials import Hugoniot, Material, MaterialLike, as_material
from .status import Confidence

REFERENCES: dict[str, str] = {
    "marsh_1980": "Marsh, S. P. (ed.), LASL Shock Hugoniot Data, University of California Press, "
                  "Berkeley (1980).",
    "mcqueen_1970": "McQueen, R. G., Marsh, S. P., Taylor, J. W., Fritz, J. N. & Carter, W. J., "
                    "'The equation of state of solids from shock wave studies', in Kinslow, R. (ed.), "
                    "High-Velocity Impact Phenomena, Academic Press, New York (1970) 293-417.",
    "meyers_1994": "Meyers, M. A., Dynamic Behavior of Materials, Wiley, New York (1994), chapters 4-7 "
                   "(Rankine-Hugoniot relations, impedance matching, pulse duration).",
    "bancroft_1956": "Bancroft, D., Peterson, E. L. & Minshall, S., 'Polymorphism of iron at high pressure', "
                     "J. Appl. Phys. 27 (1956) 291-298.",
    "duvall_graham_1977": "Duvall, G. E. & Graham, R. A., 'Phase transitions under shock-wave loading', "
                          "Rev. Mod. Phys. 49 (1977) 523-579.",
    "zeldovich_raizer_1967": "Zel'dovich, Ya. B. & Raizer, Yu. P., Physics of Shock Waves and "
                             "High-Temperature Hydrodynamic Phenomena, Academic Press, New York (1967), "
                             "vol. II, ch. XI (shock compression of solids).",
    "psyk_2011": "Psyk, V., Risch, D., Kinsey, B. L., Tekkaya, A. E. & Kleiner, M., 'Electromagnetic "
                 "forming - a review', J. Mater. Process. Technol. 211 (2011) 787-829.",
}

TWO_WAVE_BAND: tuple[float, float] = (0.8, 1.5)
"""Pressure band, as multiples of a material's phase-transition pressure, in which the
shock splits into a transition wave and a transformation wave (two-wave region) and a
single-shock impedance match is only approximate [duvall_graham_1977]."""

_CONFIDENCE_RANK = {
    Confidence.HANDBOOK: 0,
    Confidence.REPRESENTATIVE: 1,
    Confidence.REPORTED: 2,
    Confidence.ESTIMATED: 3,
}


# ---------------------------------------------------------------------------
# input validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(x):
        raise InputError(f"{name} must be finite, got {value!r}")
    return x


def _positive(name: str, value: float) -> float:
    x = _finite(name, value)
    if x <= 0.0:
        raise InputError(f"{name} must be > 0, got {value!r}")
    return x


def _non_negative(name: str, value: float) -> float:
    x = _finite(name, value)
    if x < 0.0:
        raise InputError(f"{name} must be >= 0, got {value!r}")
    return x


def _efficiency(value: float) -> float:
    x = _finite("efficiency", value)
    if not 0.0 < x <= 1.0:
        raise InputError(f"efficiency must be in (0, 1], got {value!r}")
    return x


# ---------------------------------------------------------------------------
# piecewise Hugoniot (low branch, mixed-phase plateau, high branch)
# ---------------------------------------------------------------------------
def _particle_velocity_on(hugoniot: Hugoniot, density: float, pressure: float) -> float:
    """Positive root u of rho*(c0 + s*u)*u = P on one linear branch (0 for P = 0)."""
    if pressure <= 0.0:
        return 0.0
    roots = solve_quadratic(density * hugoniot.s, density * hugoniot.c0, -pressure)
    positive = [r for r in roots if r >= 0.0]
    if not positive:
        raise InputError(
            f"Hugoniot (c0={hugoniot.c0}, s={hugoniot.s}) has no positive particle velocity "
            f"for P = {pressure:.4g} Pa"
        )
    return max(positive)


@dataclass(frozen=True)
class _Curve:
    """Continuous, monotone P(u) of one material (private helper)."""

    key: str
    density: float
    low: Hugoniot
    high: Hugoniot | None
    transition: float | None
    u_low_end: float      # low(u) = transition here
    u_high_start: float   # high(u) = transition here

    def evaluate(self, u: float) -> tuple[float, float, str]:
        """(pressure, shock speed, branch) at particle-velocity jump ``u`` >= 0."""
        if self.high is None or self.transition is None or u <= self.u_low_end:
            return self.low.pressure(self.density, u), self.low.shock_speed(u), "low"
        if u >= self.u_high_start:
            return self.high.pressure(self.density, u), self.high.shock_speed(u), "high"
        # mixed phase: the pressure stays at the transition while u grows; the
        # reported shock speed is the effective single-wave (Rayleigh-line) speed
        p = self.transition
        return p, p / (self.density * u), "mixed"

    def branches_used(self, branch: str) -> tuple[Hugoniot, ...]:
        if branch == "high" and self.high is not None:
            return (self.high,)
        if branch == "mixed" and self.high is not None:
            return (self.low, self.high)
        return (self.low,)


def _curve(material: Material) -> _Curve:
    low = material.hugoniot_for(None)
    if low.c0 <= 0.0:
        raise InputError(f"material '{material.key}' has a non-positive Hugoniot c0")
    pt = material.phase_transition_pressure
    if pt is not None and pt > 0.0:
        high = material.hugoniot_for(math.inf)  # branch the material declares above its transition
        if high != low:
            if high.c0 <= 0.0:
                raise InputError(f"material '{material.key}' has a non-positive high-pressure Hugoniot c0")
            return _Curve(
                key=material.key, density=material.density, low=low, high=high, transition=pt,
                u_low_end=_particle_velocity_on(low, material.density, pt),
                u_high_start=_particle_velocity_on(high, material.density, pt),
            )
    return _Curve(material.key, material.density, low, None, None, math.inf, math.inf)


def _worst_confidence(hugoniots: tuple[Hugoniot, ...]) -> Confidence:
    return max((h.confidence for h in hugoniots), key=lambda c: _CONFIDENCE_RANK.get(c, 3))


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class ShockState(ResultMixin):
    """Shocked state at the flyer/target contact after a planar impact.

    Rankine-Hugoniot jump conditions on each side (momentum, from rest):
    ``P = rho_t * Us_t * u`` (target) and ``P = rho_f * Us_f * (v - u)`` (flyer),
    with ``Us = c0 + s * up`` on the branch in use [marsh_1980, meyers_1994].

    Fields (SI): ``pressure`` [Pa] at the contact (equal in both bodies);
    ``particle_velocity`` [m/s] of the target material behind its shock
    (lab frame); ``flyer_particle_velocity`` = ``impact_velocity - particle_velocity``
    [m/s], the velocity jump across the shock running back into the flyer;
    ``shock_speed_target`` / ``shock_speed_flyer`` [m/s] relative to the
    unshocked material (effective single-wave speed ``P/(rho0*up)`` on a
    mixed-phase plateau); ``confidence`` is the weakest provenance of the
    Hugoniot branches used (ESTIMATED if either is estimated);
    ``branch_note`` lists phase-transition branch choices, two-wave warnings
    and validity-limit exceedances (empty when none apply).
    """

    flyer: str
    target: str
    impact_velocity: float = unit("m/s")
    pressure: float = unit("Pa")
    particle_velocity: float = unit("m/s")
    flyer_particle_velocity: float = unit("m/s")
    shock_speed_target: float = unit("m/s")
    shock_speed_flyer: float = unit("m/s")
    confidence: Confidence
    branch_note: str = ""


# ---------------------------------------------------------------------------
# impedance matching
# ---------------------------------------------------------------------------
def impedance_match(flyer: MaterialLike, target: MaterialLike, impact_velocity: float) -> ShockState:
    """Pressure and particle velocity produced by a flyer hitting a stationary target.

    Governing equation (continuity of pressure and particle velocity at the
    contact, Rankine-Hugoniot momentum jump on each side) [meyers_1994]::

        rho_f * (c_f + s_f*(v - u)) * (v - u)  =  rho_t * (c_t + s_t*u) * u,   0 <= u <= v

    solved for the target particle velocity ``u`` by bisection (the left side
    falls and the right side rises monotonically in ``u``).  For identical
    materials ``u = v/2`` exactly.

    Hugoniots come from ``Material.hugoniot_for``: the low-pressure branch
    first; a material with ``phase_transition_pressure`` and ``hugoniot_high``
    uses the high branch above the transition, with a mixed-phase plateau at
    the transition pressure where neither branch alone is self-consistent
    (Fe alpha-epsilon at 13 GPa [bancroft_1956]).  Materials without a measured
    Hugoniot use ``c0 = sqrt(K/rho)``, ``s = 1.4`` (estimated).

    Args:
        flyer: flyer material (key or :class:`~magform.materials.Material`).
        target: target material.
        impact_velocity: flyer velocity at impact, m/s (>= 0).

    Returns:
        :class:`ShockState` (Pa, m/s).  ``confidence`` is ESTIMATED if either
        Hugoniot is estimated.

    Validity: 1-D planar impact; material strength and the elastic precursor
    neglected (good for P well above the Hugoniot elastic limit, ~0.1-3 GPa);
    linear Us-up fits are extrapolated beyond ``Hugoniot.p_max`` (noted in
    ``branch_note``); between 0.8x and 1.5x a transition pressure the real
    wave is split in two and the answer is approximate [duvall_graham_1977].
    """
    f = as_material(flyer)
    t = as_material(target)
    v = _non_negative("impact_velocity", impact_velocity)
    cf = _curve(f)
    ct = _curve(t)

    if v == 0.0:
        u = 0.0
    else:
        def mismatch(u_t: float) -> float:
            return ct.evaluate(u_t)[0] - cf.evaluate(v - u_t)[0]

        u = bisect(mismatch, 0.0, v, tol=1e-14, max_iter=400)
    w = v - u
    p_t, us_t, branch_t = ct.evaluate(u)
    p_f, us_f, branch_f = cf.evaluate(w)
    pressure = p_t

    notes: list[str] = []
    used: list[Hugoniot] = []
    for role, mat, curve, branch in (("flyer", f, cf, branch_f), ("target", t, ct, branch_t)):
        branches = curve.branches_used(branch)
        used.extend(branches)
        pt = curve.transition
        if pt is not None:
            if branch == "high":
                notes.append(f"{role} {mat.key}: high-pressure Hugoniot branch used "
                             f"(P above its {pt / 1e9:.3g} GPa phase transition)")
            elif branch == "mixed":
                notes.append(f"{role} {mat.key}: state pinned at its {pt / 1e9:.3g} GPa phase transition "
                             f"(mixed phase; neither branch alone is self-consistent)")
            lo, hi = TWO_WAVE_BAND
            if lo * pt <= pressure <= hi * pt:
                notes.append(f"{role} {mat.key}: P = {pressure / 1e9:.3g} GPa is within {lo:g}-{hi:g}x of its "
                             f"{pt / 1e9:.3g} GPa phase transition - two-wave region; the single-shock "
                             f"impedance match is approximate")
        if branch != "mixed":
            h = branches[0]
            if h.p_max is not None and pressure > h.p_max * (1.0 + 1e-9):
                notes.append(f"{role} {mat.key}: P = {pressure / 1e9:.3g} GPa exceeds the {h.p_max / 1e9:.3g} GPa "
                             f"validity limit of its Hugoniot fit (extrapolated)")
        for h in branches:
            if h.confidence is Confidence.ESTIMATED:
                notes.append(f"{role} {mat.key}: estimated Hugoniot ({h.source})")
    return ShockState(
        flyer=f.key,
        target=t.key,
        impact_velocity=v,
        pressure=pressure,
        particle_velocity=u,
        flyer_particle_velocity=w,
        shock_speed_target=us_t,
        shock_speed_flyer=us_f,
        confidence=_worst_confidence(tuple(used)),
        branch_note="; ".join(notes),
    )


def shock_state_from_pressure(material: MaterialLike, pressure: float) -> tuple[float, float]:
    """Particle velocity and shock speed ``(u_p, Us)`` of a material shocked to ``pressure``.

    Governing equation: the positive root of the Rankine-Hugoniot quadratic
    ``rho0 * (c0 + s*u) * u = P`` on the branch ``Material.hugoniot_for(P)``
    (high-pressure branch above a recorded phase transition), then
    ``Us = c0 + s*u`` [marsh_1980, mcqueen_1970].

    Args:
        material: material key or object.
        pressure: shock pressure, Pa (>= 0).

    Returns:
        ``(u_p [m/s], Us [m/s])``.  At P = 0 this is ``(0, c0)``.

    Validity: linear Us-up fit inside its calibrated range; at exactly the
    transition pressure the low-branch state (start of the mixed-phase
    plateau) is returned.
    """
    m = as_material(material)
    p = _non_negative("pressure", pressure)
    h = m.hugoniot_for(p)
    u = _particle_velocity_on(h, m.density, p)
    return u, h.shock_speed(u)


# ---------------------------------------------------------------------------
# flyer kinematics (constant drive pressure over the gap)
# ---------------------------------------------------------------------------
def flyer_velocity(pressure: float, gap: float, flyer: MaterialLike, thickness: float,
                   efficiency: float = 1.0) -> float:
    """Velocity of a flyer sheet accelerated across a gap by a constant pressure.

    Governing equation (work-energy theorem per unit area,
    ``1/2 * rho*h * v**2 = eta * P * g``) [psyk_2011]::

        v = sqrt(2 * eta * P * g / (rho * h))

    Args:
        pressure: drive (magnetic) pressure on the flyer, Pa (>= 0).
        gap: acceleration distance (standoff), m (> 0).
        flyer: flyer material (density rho).
        thickness: flyer thickness h, m (> 0).
        efficiency: fraction eta of the pressure that accelerates the flyer, (0, 1].

    Returns:
        Impact velocity, m/s.  Check: 4 GPa, 10 mm, Cu, 2 mm -> 2113 m/s.

    Validity: the drive pulse must outlast the flight
    (:func:`flyer_transit_time`); rigid, unbent sheet; in a real coil the
    pressure falls as the gap opens (inductance change, field leakage), so
    this is an upper bound folded into ``efficiency``.
    """
    p = _non_negative("pressure", pressure)
    g = _positive("gap", gap)
    h = _positive("thickness", thickness)
    eta = _efficiency(efficiency)
    m = as_material(flyer)
    return math.sqrt(2.0 * eta * p * g / (m.density * h))


def flyer_transit_time(gap: float, pressure: float, flyer: MaterialLike, thickness: float,
                       efficiency: float = 1.0) -> float:
    """Time for a flyer to cross the gap under constant pressure (uniform acceleration).

    Governing equation (``g = a t**2 / 2`` with ``a = eta*P/(rho*h)``) [psyk_2011]::

        t = sqrt(2 * g * rho * h / (eta * P))      (= 2 g / v)

    Args:
        gap: acceleration distance, m (> 0).
        pressure: drive pressure, Pa (>= 0); returns ``math.inf`` for P = 0.
        flyer: flyer material.
        thickness: flyer thickness, m (> 0).
        efficiency: pressure efficiency eta, (0, 1].

    Returns:
        Transit time, s.  If it exceeds the pressure-pulse duration the
        constant-pressure velocity of :func:`flyer_velocity` is not reached.

    Validity: as :func:`flyer_velocity`.
    """
    g = _positive("gap", gap)
    p = _non_negative("pressure", pressure)
    h = _positive("thickness", thickness)
    eta = _efficiency(efficiency)
    m = as_material(flyer)
    if p == 0.0:
        return math.inf
    return math.sqrt(2.0 * g * m.density * h / (eta * p))


def required_standoff(target_velocity: float, pressure: float, flyer: MaterialLike, thickness: float,
                      efficiency: float = 1.0) -> float:
    """Gap needed for a constant pressure to accelerate a flyer to ``target_velocity``.

    Governing equation (inverse of :func:`flyer_velocity`) [psyk_2011]::

        g = rho * h * v**2 / (2 * eta * P)

    Args:
        target_velocity: desired impact velocity, m/s (>= 0).
        pressure: drive pressure, Pa (>= 0); ``math.inf`` if P = 0 and v > 0.
        flyer: flyer material.
        thickness: flyer thickness, m (> 0).
        efficiency: pressure efficiency, (0, 1].

    Returns:
        Standoff gap, m.

    Validity: as :func:`flyer_velocity`.
    """
    v = _non_negative("target_velocity", target_velocity)
    p = _non_negative("pressure", pressure)
    h = _positive("thickness", thickness)
    eta = _efficiency(efficiency)
    m = as_material(flyer)
    if v == 0.0:
        return 0.0
    if p == 0.0:
        return math.inf
    return m.density * h * v * v / (2.0 * eta * p)


def required_pressure_for_velocity(velocity: float, gap: float, flyer: MaterialLike, thickness: float,
                                   efficiency: float = 1.0) -> float:
    """Constant drive pressure that brings a flyer to ``velocity`` within ``gap``.

    Governing equation (inverse of :func:`flyer_velocity`) [psyk_2011]::

        P = rho * h * v**2 / (2 * eta * g)

    Args:
        velocity: desired impact velocity, m/s (>= 0).
        gap: acceleration distance, m (> 0).
        flyer: flyer material.
        thickness: flyer thickness, m (> 0).
        efficiency: pressure efficiency, (0, 1].

    Returns:
        Drive pressure, Pa.

    Validity: as :func:`flyer_velocity`.
    """
    v = _non_negative("velocity", velocity)
    g = _positive("gap", gap)
    h = _positive("thickness", thickness)
    eta = _efficiency(efficiency)
    m = as_material(flyer)
    return m.density * h * v * v / (2.0 * eta * g)


# ---------------------------------------------------------------------------
# order-of-magnitude estimates (confidence: estimated)
# ---------------------------------------------------------------------------
def shock_duration(flyer_thickness: float, shock_speed_flyer: float) -> float:
    """Duration of the impact pressure pulse at the contact (estimate).

    Governing equation: the shock runs back through the flyer and the release
    from its free rear surface returns to the contact [meyers_1994]::

        t ~= 2 * h / Us_flyer

    Args:
        flyer_thickness: flyer thickness h, m (> 0).
        shock_speed_flyer: shock speed in the flyer, m/s (> 0), e.g.
            ``ShockState.shock_speed_flyer``.

    Returns:
        Pulse duration, s.  Confidence: estimated.

    Validity: thick target (no reflection from its rear surface first); the
    release travels at the compressed-state sound speed (> Us), so the true
    flat-top duration is somewhat shorter - an upper-end estimate.
    """
    h = _positive("flyer_thickness", flyer_thickness)
    us = _positive("shock_speed_flyer", shock_speed_flyer)
    return 2.0 * h / us


def shock_temperature_rise(particle_velocity: float, material: MaterialLike) -> float:
    """Upper-bound order estimate of the shock heating of a material.

    Governing equation: the Rankine-Hugoniot energy jump from rest is
    ``E - E0 = up**2 / 2`` per unit mass [zeldovich_raizer_1967]; treating all
    of it as heat at constant specific heat gives::

        dT ~= up**2 / (2 * c_p)

    Args:
        particle_velocity: particle velocity behind the shock, m/s (>= 0),
            e.g. ``ShockState.particle_velocity`` for the target.
        material: shocked material (specific heat c_p).

    Returns:
        Temperature rise, K.  Confidence: estimated (upper bound).

    Validity: real shock heating is lower - part of the Hugoniot energy is
    elastic (cold-compression) energy recovered on release, so residual
    heating is a fraction of this; porous materials heat more for the same
    pressure but the formula uses only ``up``.  Ignores phase-change enthalpy
    and the temperature dependence of c_p.
    """
    u = _non_negative("particle_velocity", particle_velocity)
    m = as_material(material)
    return u * u / (2.0 * m.specific_heat)


__all__ = [
    "REFERENCES",
    "TWO_WAVE_BAND",
    "ShockState",
    "impedance_match",
    "shock_state_from_pressure",
    "flyer_velocity",
    "flyer_transit_time",
    "required_standoff",
    "required_pressure_for_velocity",
    "shock_duration",
    "shock_temperature_rise",
]
