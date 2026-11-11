"""Magnetic pressure, thick-solenoid fields, stored energy and winding stresses.

Conventions
-----------
SI throughout: fields in T, pressures and stresses in Pa, lengths in m,
current densities in A/m^2, energies in J, inductances in H.  The
``current_density`` of a :class:`Solenoid` is the *winding-average*
engineering current density (ampere-turns per unit area of the whole winding
cross-section, insulation and reinforcement included), so that the long-coil
central field is ``B0 = mu0 J Delta``.

Physics in one paragraph
------------------------
Magnetic pressure is ``P = B^2/(2 mu0)``: 1.6 GPa needs 63.41 T, 2.0 GPa
70.90 T and 4.0 GPa 100.27 T.  A *static* field exerts no force on a
stationary, current-free conductor: electromagnetic-forming pressure
``(B_out^2 - B_in^2)/(2 mu0)`` exists only while the outside field changes
faster than the workpiece's flux-diffusion time, so a DC magnet can at most
bias a pulsed insert (:func:`pressure_with_background`).  The winding that
makes the field feels the same pressure: for non-interacting turns its hoop
stress maximum never falls below ``P/2`` however thick the coil
(:func:`max_hoop_stress_ratio`), and the virial theorem sets a minimum
structural mass ``M >= rho E / sigma`` (:func:`virial_minimum_mass`).

Approximations
--------------
The central field uses Montgomery's exact on-axis result for a uniform-J
thick solenoid.  The radial winding-field profile, the stored energy and the
hoop stresses use the *long-coil* approximation (field ``B0`` in the bore,
falling linearly to zero across the winding), which is within ~10 % only for
``beta = L/(2 a1) >= 3`` (see :meth:`Solenoid.end_effect_warning`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ._numerics import linspace, trapz
from ._report import ResultMixin, unit
from .constants import MU0
from .errors import InputError

REFERENCES: dict[str, str] = {
    "jackson1999": "J. D. Jackson, Classical Electrodynamics, 3rd ed., Wiley, New York (1999): "
                   "sec. 5.6 (magnetic dipole field) and sec. 6.7 (Maxwell stress tensor).",
    "knoepfel2000": "H. Knoepfel, Magnetic Fields: A Comprehensive Theoretical Treatise for Practical Use, "
                    "Wiley, New York (2000).",
    "psyk2011": "V. Psyk, D. Risch, B. L. Kinsey, A. E. Tekkaya and M. Kleiner, 'Electromagnetic forming - "
                "a review', J. Mater. Process. Technol. 211 (2011) 787-829.",
    "montgomery1969": "D. B. Montgomery, Solenoid Magnet Design: The Magnetic and Mechanical Aspects of "
                      "Resistive and Superconducting Systems, Wiley-Interscience, New York (1969).",
    "wilson1983": "M. N. Wilson, Superconducting Magnets, Clarendon Press, Oxford (1983).",
    "iwasa2009": "Y. Iwasa, Case Studies in Superconducting Magnets: Design and Operational Issues, "
                 "2nd ed., Springer, New York (2009).",
    "wheeler1928": "H. A. Wheeler, 'Simple inductance formulas for radio coils', Proc. IRE 16 (1928) 1398-1400.",
    "moon1982": "F. C. Moon, 'The virial theorem and scaling laws for superconducting magnet systems', "
                "J. Appl. Phys. 53 (1982) 9112-9121.",
    "moon1984": "F. C. Moon, Magneto-Solid Mechanics, Wiley, New York (1984).",
    "herlach2003": "F. Herlach and N. Miura (eds.), High Magnetic Fields: Science and Technology, Vol. 1, "
                   "World Scientific, Singapore (2003).",
}

WHEELER_COEFFICIENT: float = 0.8e-6 / 0.0254
"""H/m.  Wheeler's 0.8 uH/inch coefficient converted to SI (= 31.50e-6 H/m)."""

LONG_COIL_BETA: float = 3.0
"""beta = L/(2 a1) above which the long-coil profile and energy are within ~10 %
(field factor f >= 0.89 at alpha = 2; Montgomery 1969)."""


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


def _non_negative(name: str, value: float) -> float:
    v = _finite(name, value)
    if v < 0.0:
        raise InputError(f"{name} must be non-negative, got {v!r}")
    return v


def _count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# magnetic pressure
# ---------------------------------------------------------------------------
def magnetic_pressure(field: float) -> float:
    """Magnetic pressure of a field, ``P = B^2 / (2 mu0)`` [Pa].

    ``field`` in T (its sign is irrelevant).  This is the Maxwell-stress
    pressure on a perfect conductor with the field on one side only, and equals
    the field energy density.  Exact for non-magnetic media (mu_r = 1).
    Examples: 63.41 T -> 1.6 GPa, 70.90 T -> 2.0 GPa, 100.27 T -> 4.0 GPa.

    Reference: Jackson (1999) sec. 6.7; Knoepfel (2000).
    """
    b = _finite("field", field)
    return b * b / (2.0 * MU0)


def field_for_pressure(pressure: float) -> float:
    """Field whose magnetic pressure is ``pressure``: ``B = sqrt(2 mu0 P)`` [T].

    ``pressure`` in Pa, must be >= 0 (:class:`InputError` otherwise).
    Inverse of :func:`magnetic_pressure`; 4.0 GPa -> 100.27 T.

    Reference: Jackson (1999) sec. 6.7.
    """
    p = _non_negative("pressure", pressure)
    return math.sqrt(2.0 * MU0 * p)


def energy_density(field: float) -> float:
    """Magnetic energy density ``u = B^2/(2 mu0)`` [J/m^3], numerically equal to the pressure.

    ``field`` in T; valid for mu_r = 1.  At 100 T, u = 4.0 GJ/m^3 (about 58 % of
    the detonation energy density of TNT).

    Reference: Jackson (1999) sec. 6.7.
    """
    return magnetic_pressure(field)


def pressure_with_background(background_field: float, pulse_outside: float,
                             pulse_inside: float = 0.0) -> float:
    """Net magnetic pressure on a conducting shell in a DC background with a pulsed increment [Pa].

    ``P = ((B_dc + dB_out)^2 - (B_dc + dB_in)^2) / (2 mu0)``

    ``background_field`` B_dc is the steady field present on *both* sides of the
    shell (it has diffused through long ago); ``pulse_outside``/``pulse_inside``
    are the pulsed increments on the coil side and the screened side (T, signed:
    positive is parallel to B_dc).  A static field exerts no force on a
    stationary, current-free conductor: with both increments zero the pressure is
    exactly 0 whatever B_dc.  Eddy-current pressure exists only while the
    outside field changes faster than the shell's flux-diffusion time constant
    (``pulse_inside`` << ``pulse_outside``); a DC superconducting magnet can
    therefore only *bias* a pulsed insert, which reduces the pulse amplitude
    needed (:func:`pulse_increment_for_pressure`) but not the total gap field.
    A negative result means the net force pushes the shell towards the coil.

    Reference: Knoepfel (2000); Psyk et al. (2011).
    """
    b0 = _finite("background_field", background_field)
    d_out = _finite("pulse_outside", pulse_outside)
    d_in = _finite("pulse_inside", pulse_inside)
    b_out = b0 + d_out
    b_in = b0 + d_in
    return (b_out * b_out - b_in * b_in) / (2.0 * MU0)


def pulse_increment_for_pressure(pressure: float, background_field: float) -> float:
    """Pulsed field increment that yields ``pressure`` on a fully screening shell in a DC background [T].

    ``dB = sqrt(B_dc^2 + 2 mu0 P) - B_dc``  (inverse of :func:`pressure_with_background`
    with ``pulse_inside = 0``).  ``pressure`` in Pa (>= 0), ``background_field`` in T
    (signed, positive parallel to the pulse).  The *total* outside field
    ``B_dc + dB`` still equals ``sqrt(B_dc^2 + 2 mu0 P)`` >= ``field_for_pressure(P)``,
    so the background does not relieve the insert's peak field or stress.

    Reference: Knoepfel (2000).
    """
    p = _non_negative("pressure", pressure)
    b0 = _finite("background_field", background_field)
    return math.sqrt(b0 * b0 + 2.0 * MU0 * p) - b0


def current_sheet_pressure(surface_current_density: float, field: float) -> float:
    """Lorentz (J x B) pressure on a current sheet, ``P = K B`` [Pa].

    ``surface_current_density`` K in A/m, ``field`` B in T: the transverse field
    acting on the sheet, i.e. the mean of the fields on its two faces.  For a
    sheet that screens a field B_out completely (K = B_out/mu0, mean field
    B_out/2) this reproduces ``B_out^2/(2 mu0)``.  Signed: positive when K x B
    points away from the field side.

    Reference: Jackson (1999) sec. 5.7 and 6.7.
    """
    k = _finite("surface_current_density", surface_current_density)
    b = _finite("field", field)
    return k * b


def surface_current_for_pressure(pressure: float, field: float) -> float:
    """Surface current density needed for a J x B pressure in a given field, ``K = P / B`` [A/m].

    ``pressure`` in Pa (>= 0), ``field`` in T (mean field on the sheet).  Returns
    ``math.inf`` when ``field`` is 0 and ``pressure`` > 0 (no finite current can
    push against nothing) and 0 for zero pressure.

    Reference: Jackson (1999) sec. 5.7.
    """
    p = _non_negative("pressure", pressure)
    b = _finite("field", field)
    if p == 0.0:
        return 0.0
    if b == 0.0:
        return math.inf
    return p / b


# ---------------------------------------------------------------------------
# winding stresses and the virial bound
# ---------------------------------------------------------------------------
def max_hoop_stress_ratio(alpha: float) -> float:
    """Maximum non-interacting-turn hoop stress of a long coil in units of the bore pressure.

    With the long-coil profile ``B(r) = B0 (a2 - r)/Delta`` and ``J = B0/(mu0 Delta)``,
    ``sigma(r) = r J B(r) = 2 P r (a2 - r)/Delta^2`` and

    * ``sigma_max / P = 2/(alpha - 1)``            for alpha <= 2 (maximum at r = a1),
    * ``sigma_max / P = alpha^2/(2 (alpha - 1)^2)`` for alpha > 2 (maximum at r = a2/2),

    with ``alpha = a2/a1`` (> 1, dimensionless).  The ratio tends to 1/2 as
    alpha -> infinity and is < 1 only for alpha > 2 + sqrt(2) = 3.414: no winding
    thickness brings the peak hoop stress below half the magnetic pressure.
    Validity: long coil (beta >= 3), turns free to slide (no radial load
    transfer), elastic.

    Reference: Wilson (1983); Iwasa (2009); Montgomery (1969).
    """
    a = _finite("alpha", alpha)
    if a <= 1.0:
        raise InputError(f"alpha = a2/a1 must exceed 1, got {a!r}")
    if a <= 2.0:
        return 2.0 / (a - 1.0)
    return a * a / (2.0 * (a - 1.0) ** 2)


def thin_shell_hoop_stress(pressure: float, radius: float, thickness: float) -> float:
    """Hoop stress of a thin cylindrical shell under internal pressure, ``sigma = P a / t`` [Pa].

    ``pressure`` in Pa (>= 0), ``radius`` a and ``thickness`` t in m (> 0).
    Validity: t << a (error ~ t/(2a)); use a Lame thick-wall solution otherwise.

    Reference: Timoshenko & Goodier, Theory of Elasticity (1951); Montgomery (1969).
    """
    p = _non_negative("pressure", pressure)
    a = _positive("radius", radius)
    t = _positive("thickness", thickness)
    return p * a / t


def virial_minimum_mass(stored_energy: float, density: float, allowable_stress: float) -> float:
    """Virial-theorem lower bound on the structural mass of a magnet, ``M = rho E / sigma`` [kg].

    ``stored_energy`` E in J (>= 0), ``density`` rho in kg/m^3 (> 0),
    ``allowable_stress`` sigma in Pa (> 0).  The virial theorem
    ``integral(trace sigma) dV = E`` means any self-supported magnet needs at
    least this mass in tension at ``sigma``; it is a tension-only bound that real
    solenoids exceed by 2-3x because their radial and axial stresses are
    compressive.

    Reference: Moon (1982) J. Appl. Phys. 53, 9112; Moon (1984).
    """
    e = _non_negative("stored_energy", stored_energy)
    rho = _positive("density", density)
    s = _positive("allowable_stress", allowable_stress)
    return rho * e / s


def virial_mean_stress(stored_energy: float, structure_volume: float) -> float:
    """Virial mean tensile stress of a load-bearing structure, ``sigma = E / V`` [Pa].

    ``stored_energy`` in J (>= 0), ``structure_volume`` in m^3 (> 0).  A lower
    bound on the volume-averaged tension; the actual hoop average of a solenoid
    is about twice this (see :meth:`Solenoid.average_hoop_stress`).

    Reference: Moon (1982); Moon (1984).
    """
    e = _non_negative("stored_energy", stored_energy)
    v = _positive("structure_volume", structure_volume)
    return e / v


def graded_winding_thickness(central_field: float, current_density_of_field: Callable[[float], float],
                             n: int = 2000) -> float:
    """Minimum winding thickness of an ideally graded long solenoid [m].

    ``Delta = integral_0^B dB' / (mu0 J(B'))``

    from ``dB/dr = -mu0 J(B(r))`` in the long-coil limit, where every layer runs at
    the largest current density ``J(B')`` [A/m^2] its local field allows.
    ``central_field`` in T (>= 0); ``current_density_of_field`` maps field (T) to the
    winding-average current density (A/m^2).  Midpoint rule with ``n`` intervals
    (error O(n^-2)).  Returns ``math.inf`` if ``J`` is <= 0 (or not finite) at any
    sample or at the end points: no finite winding reaches the field.
    Validity: long coil, perfect grading, current-limited (stress is checked
    separately).

    Reference: Iwasa (2009); Wilson (1983).
    """
    b_top = _non_negative("central_field", central_field)
    _count("n", n, 1)
    if b_top == 0.0:
        return 0.0

    def j_of(b: float) -> float:
        j = float(current_density_of_field(b))
        return j if (math.isfinite(j) and j > 0.0) else 0.0

    if j_of(0.0) <= 0.0 or j_of(b_top) <= 0.0:
        return math.inf
    h = b_top / n
    total = 0.0
    for i in range(n):
        j = j_of((i + 0.5) * h)
        if j <= 0.0:
            return math.inf
        total += h / (MU0 * j)
    return total


def dipole_field_on_equator(moment: float, distance: float) -> float:
    """Far field of a magnetic dipole in its equatorial plane, ``B = mu0 m / (4 pi r^3)`` [T].

    ``moment`` m in A m^2, ``distance`` r in m (> 0).  On the axis the field is
    twice this.  Validity: r much larger than the source size (r > ~3 a2 for a
    solenoid).  Used for stray-field and containment estimates.

    Reference: Jackson (1999) eq. 5.56.
    """
    m = _finite("moment", moment)
    r = _positive("distance", distance)
    return MU0 * m / (4.0 * math.pi * r ** 3)


# ---------------------------------------------------------------------------
# result records
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class PressureUniformity(ResultMixin):
    """On-axis magnetic-pressure uniformity over a workpiece centred in a solenoid.

    ``ratio[i] = P(z_i)/P(0) = (B_z(z_i)/B0)^2`` for ``z`` spanning
    [-l/2, l/2]; ``end_ratio`` is the value at ``z = l/2``, ``mean_ratio`` the
    length average (trapezoid), ``max_deviation = 1 - min(ratio)``.  On-axis
    values only; the pressure on a workpiece surface off the axis differs.

    Reference: Montgomery (1969).
    """

    workpiece_length: float = unit("m")
    end_ratio: float
    mean_ratio: float
    max_deviation: float
    z: tuple[float, ...] = unit("m")
    ratio: tuple[float, ...]


@dataclass(frozen=True, kw_only=True)
class HoopProfile(ResultMixin):
    """Non-interacting-turn hoop stress across a winding, ``sigma(r) = r J B(r)``.

    ``r`` and ``sigma`` are the sampled profile, ``max``/``r_at`` the analytic
    maximum from :meth:`Solenoid.max_hoop_stress`.  Long-coil field profile.

    Reference: Wilson (1983); Iwasa (2009).
    """

    r: tuple[float, ...] = unit("m")
    sigma: tuple[float, ...] = unit("Pa")
    max: float = unit("Pa")
    r_at: float = unit("m")


# ---------------------------------------------------------------------------
# solenoid geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Solenoid(ResultMixin):
    """Uniform-current-density thick solenoid (geometry only).

    ``inner_radius`` a1, ``outer_radius`` a2 and ``length`` L in m, with
    ``0 < a1 < a2`` and ``L > 0`` (:class:`InputError` otherwise).  Shape
    parameters: ``alpha = a2/a1``, ``beta = L/(2 a1)``.  Field methods take the
    winding-average current density J [A/m^2] or the central field B0 [T].
    Central and on-axis fields are exact for uniform J (Montgomery 1969);
    profile, energy and stresses use the long-coil approximation, good to ~10 %
    for beta >= 3 (:meth:`end_effect_warning`).

    Reference: Montgomery (1969); Wilson (1983).
    """

    inner_radius: float = unit("m")
    outer_radius: float = unit("m")
    length: float = unit("m")

    def __post_init__(self) -> None:
        a1 = _finite("inner_radius", self.inner_radius)
        a2 = _finite("outer_radius", self.outer_radius)
        length = _finite("length", self.length)
        if a1 <= 0.0:
            raise InputError(f"inner_radius must be positive, got {a1!r}")
        if a2 <= a1:
            raise InputError(f"outer_radius ({a2!r}) must exceed inner_radius ({a1!r})")
        if length <= 0.0:
            raise InputError(f"length must be positive, got {length!r}")
        object.__setattr__(self, "inner_radius", a1)
        object.__setattr__(self, "outer_radius", a2)
        object.__setattr__(self, "length", length)

    # -- geometry -----------------------------------------------------------
    @property
    def winding_thickness(self) -> float:
        """Radial build ``Delta = a2 - a1`` [m]."""
        return self.outer_radius - self.inner_radius

    @property
    def mean_radius(self) -> float:
        """Mean winding radius ``(a1 + a2)/2`` [m]."""
        return 0.5 * (self.inner_radius + self.outer_radius)

    @property
    def alpha(self) -> float:
        """Radius ratio ``a2/a1`` (dimensionless, > 1)."""
        return self.outer_radius / self.inner_radius

    @property
    def beta(self) -> float:
        """Shape ratio ``L/(2 a1)`` (dimensionless)."""
        return self.length / (2.0 * self.inner_radius)

    @property
    def winding_volume(self) -> float:
        """Winding volume ``pi (a2^2 - a1^2) L`` [m^3]."""
        return math.pi * (self.outer_radius ** 2 - self.inner_radius ** 2) * self.length

    @property
    def bore_volume(self) -> float:
        """Bore volume ``pi a1^2 L`` [m^3]."""
        return math.pi * self.inner_radius ** 2 * self.length

    def _log_term(self) -> float:
        al, be = self.alpha, self.beta
        return math.log((al + math.sqrt(al * al + be * be)) / (1.0 + math.sqrt(1.0 + be * be)))

    # -- fields ---------------------------------------------------------------
    def field_factor(self) -> float:
        """Montgomery field factor normalised to the long-coil field, ``f = B0/(mu0 J Delta)``.

        ``f = (beta/(alpha - 1)) ln[(alpha + sqrt(alpha^2 + beta^2)) / (1 + sqrt(1 + beta^2))]``
        (dimensionless, 0 < f < 1, -> 1 as beta -> infinity).  At alpha = 2:
        f = 0.33 (beta 0.5), 0.56 (1), 0.80 (2), 0.89 (3), 0.96 (5).

        Reference: Montgomery (1969).
        """
        return self.beta / (self.alpha - 1.0) * self._log_term()

    def central_field(self, current_density: float) -> float:
        """Central field of a uniform-J thick solenoid (Montgomery) [T].

        ``B0 = mu0 J a1 beta ln[(alpha + sqrt(alpha^2 + beta^2)) / (1 + sqrt(1 + beta^2))]``

        ``current_density`` J in A/m^2 (winding average, >= 0).  Exact for uniform
        J at the geometric centre.

        Reference: Montgomery (1969).
        """
        j = _non_negative("current_density", current_density)
        return MU0 * j * self.inner_radius * self.beta * self._log_term()

    def current_density_for_field(self, central_field: float) -> float:
        """Winding-average current density giving ``central_field`` (inverse Montgomery) [A/m^2].

        ``J = B0 / (mu0 a1 beta ln[...]) = B0 / (mu0 Delta f)``; ``central_field`` in
        T (>= 0).  For short coils (f < 1) this exceeds the long-coil value
        ``B0/(mu0 Delta)``.

        Reference: Montgomery (1969).
        """
        b0 = _non_negative("central_field", central_field)
        return b0 / (MU0 * self.inner_radius * self.beta * self._log_term())

    def field_from_current(self, current: float, turns: float) -> float:
        """Central field for a terminal current ``current`` [A] in ``turns`` turns [T].

        ``J = N I / (L Delta)`` then :meth:`central_field`.  ``current`` >= 0,
        ``turns`` > 0 (may be fractional in a smeared model).

        Reference: Montgomery (1969).
        """
        i = _non_negative("current", current)
        n = _positive("turns", turns)
        return self.central_field(n * i / (self.length * self.winding_thickness))

    def current_for_field(self, central_field: float, turns: float) -> float:
        """Terminal current needed for ``central_field`` with ``turns`` turns [A].

        ``I = J L Delta / N`` with ``J`` from :meth:`current_density_for_field`.
        ``central_field`` in T (>= 0), ``turns`` > 0.

        Reference: Montgomery (1969).
        """
        n = _positive("turns", turns)
        j = self.current_density_for_field(central_field)
        return j * self.length * self.winding_thickness / n

    def axial_field(self, z: float, current_density: float) -> float:
        """On-axis field of the uniform-J thick solenoid at axial position ``z`` [T].

        With ``l = L/2``:
        ``Bz = (mu0 J/2) [ (l+z) ln((a2 + sqrt(a2^2+(l+z)^2)) / (a1 + sqrt(a1^2+(l+z)^2)))
        + (l-z) ln((a2 + sqrt(a2^2+(l-z)^2)) / (a1 + sqrt(a1^2+(l-z)^2))) ]``

        ``z`` in m from the centre (any value, inside or outside the coil),
        ``current_density`` in A/m^2 (>= 0).  Equals :meth:`central_field` at z = 0.

        Reference: Montgomery (1969).
        """
        zz = _finite("z", z)
        j = _non_negative("current_density", current_density)
        a1, a2 = self.inner_radius, self.outer_radius
        half = 0.5 * self.length

        def term(u: float) -> float:
            return u * math.log((a2 + math.sqrt(a2 * a2 + u * u)) / (a1 + math.sqrt(a1 * a1 + u * u)))

        return 0.5 * MU0 * j * (term(half + zz) + term(half - zz))

    def pressure_uniformity(self, current_density: float, workpiece_length: float,
                            n: int = 41) -> PressureUniformity:
        """Axial uniformity of the magnetic pressure over a centred workpiece.

        ``ratio(z) = (Bz(z)/Bz(0))^2`` sampled at ``n`` points on [-l/2, l/2]
        (``workpiece_length`` l in m, 0 < l <= L; ``current_density`` > 0 A/m^2,
        which cancels in the ratio).  Contradicts the concept note's 'perfectly
        uniform' pressure: e.g. a1 = 50 mm, a2 = 150 mm, L = 0.30 m over a
        0.15 m workpiece gives end_ratio = 0.84.  On-axis field only.

        Reference: Montgomery (1969).
        """
        j = _positive("current_density", current_density)
        ell = _positive("workpiece_length", workpiece_length)
        _count("n", n, 2)
        if ell > self.length * (1.0 + 1e-12):
            raise InputError(f"workpiece_length ({ell!r} m) exceeds the coil length ({self.length!r} m)")
        b0 = self.axial_field(0.0, j)
        zs = linspace(-0.5 * ell, 0.5 * ell, n)
        ratios = [(self.axial_field(z, j) / b0) ** 2 for z in zs]
        mean_ratio = trapz(ratios, zs) / ell
        return PressureUniformity(
            workpiece_length=ell,
            end_ratio=ratios[-1],
            mean_ratio=mean_ratio,
            max_deviation=1.0 - min(ratios),
            z=tuple(zs),
            ratio=tuple(ratios),
        )

    def winding_field(self, r: float, central_field: float) -> float:
        """Axial field at radius ``r`` in the long-coil approximation [T].

        ``B0`` for r <= a1; ``B0 (a2 - r)/Delta`` for a1 < r < a2; 0 for r >= a2.
        ``r`` in m (>= 0), ``central_field`` in T (>= 0).  Ignores end effects
        (valid for beta >= 3) and the field depression in short thick coils.

        Reference: Wilson (1983).
        """
        rr = _non_negative("r", r)
        b0 = _non_negative("central_field", central_field)
        if rr <= self.inner_radius:
            return b0
        if rr >= self.outer_radius:
            return 0.0
        return b0 * (self.outer_radius - rr) / self.winding_thickness

    def stored_energy(self, central_field: float) -> float:
        """Magnetic energy of the long-coil field distribution [J].

        ``E = (B0^2/(2 mu0)) pi L [a1^2 + 2 Delta (a2/3 - Delta/4)]``
        (bore plus linearly decaying winding field; fringe field ignored).
        ``central_field`` in T (>= 0).  Within ~10 % for beta >= 3.

        Reference: Wilson (1983); Montgomery (1969).
        """
        b0 = _non_negative("central_field", central_field)
        a1, a2, d = self.inner_radius, self.outer_radius, self.winding_thickness
        return magnetic_pressure(b0) * math.pi * self.length * (a1 * a1 + 2.0 * d * (a2 / 3.0 - d / 4.0))

    def end_effect_warning(self) -> str | None:
        """Warning text when the long-coil approximation is poor (beta < 3), else ``None``.

        Basis: the Montgomery field factor f = B0/(mu0 J Delta) is 0.80 at beta = 2
        and 0.89 at beta = 3 (alpha = 2).

        Reference: Montgomery (1969).
        """
        if self.beta < LONG_COIL_BETA * (1.0 - 1e-9):   # tolerance: beta = 3 by construction passes
            return (f"beta = L/(2 a1) = {self.beta:.3g} < {LONG_COIL_BETA:g}: long-coil energy/profile within "
                    f"~10 % only for beta >= 3 (field factor f = {self.field_factor():.3g}); use the Montgomery "
                    "field and inductance_from_energy with caution")
        return None

    def inductance(self, turns: float) -> float:
        """Wheeler (1928) multilayer air-core inductance [H].

        ``L = 31.50e-6 a_m^2 N^2 / (6 a_m + 9 L + 10 Delta)`` with a_m the mean
        radius, L the length and Delta the build, all in m (coefficient
        0.8e-6/0.0254 H/m).  ``turns`` > 0.  Accurate to ~1 % when the three
        denominator terms are comparable; it includes fringing flux and so
        differs from :meth:`inductance_from_energy` (by > 10 % when beta < 3;
        about 10 % low for very long thin coils).

        Reference: Wheeler (1928).
        """
        n = _positive("turns", turns)
        am = self.mean_radius
        return WHEELER_COEFFICIENT * am * am * n * n / (6.0 * am + 9.0 * self.length + 10.0 * self.winding_thickness)

    def inductance_from_energy(self, turns: float) -> float:
        """Energy-consistent inductance ``L = 2 E / I^2`` [H].

        ``E`` from :meth:`stored_energy`, ``I`` from :meth:`current_for_field` at the
        same central field (the ratio is independent of B0).  ``turns`` > 0.
        Primary inductance for circuit design because it is consistent with the
        stored energy the bank must deliver; report :meth:`inductance` alongside.

        Reference: Wilson (1983).
        """
        n = _positive("turns", turns)
        e = self.stored_energy(1.0)
        i = self.current_for_field(1.0, n)
        return 2.0 * e / (i * i)

    # -- stresses -------------------------------------------------------------
    def hoop_stress_at(self, r: float, current_density: float, central_field: float) -> float:
        """Non-interacting-turn hoop stress ``sigma = r J B(r)`` at radius ``r`` in the winding [Pa].

        ``r`` in m with a1 <= r <= a2 (:class:`InputError` outside the winding),
        ``current_density`` J in A/m^2 (>= 0), ``central_field`` B0 in T (>= 0);
        ``B(r)`` from :meth:`winding_field`.  Each turn carries its own
        Lorentz load (no radial load sharing): an upper bound for the peak of a
        bonded winding.

        Reference: Wilson (1983); Iwasa (2009).
        """
        rr = _finite("r", r)
        tol = 1e-12 * self.outer_radius
        if rr < self.inner_radius - tol or rr > self.outer_radius + tol:
            raise InputError(f"r = {rr!r} m lies outside the winding [{self.inner_radius!r}, {self.outer_radius!r}]")
        rr = min(max(rr, self.inner_radius), self.outer_radius)
        j = _non_negative("current_density", current_density)
        return rr * j * self.winding_field(rr, central_field)

    def max_hoop_stress(self, current_density: float, central_field: float) -> tuple[float, float]:
        """Peak non-interacting-turn hoop stress and its radius, ``(sigma_max [Pa], r_at [m])``.

        ``sigma(r) = r J B0 (a2 - r)/Delta`` peaks at ``r_at = a1`` if a2 <= 2 a1,
        else at ``r_at = a2/2``.  With the long-coil J = B0/(mu0 Delta) this is
        ``P * max_hoop_stress_ratio(alpha)``: e.g. a1 = 0.05 m, Delta = 0.20 m,
        P = 4 GPa gives 3.125 GPa at r = 0.125 m.  The Montgomery J of a short coil
        is larger than the long-coil J (by 1/f), and so is the stress.
        ``current_density`` in A/m^2, ``central_field`` in T (both >= 0).

        Reference: Wilson (1983); Iwasa (2009).
        """
        a1, a2 = self.inner_radius, self.outer_radius
        r_at = a1 if a2 <= 2.0 * a1 else 0.5 * a2
        return self.hoop_stress_at(r_at, current_density, central_field), r_at

    def average_hoop_stress(self, central_field: float) -> float:
        """Thickness-averaged hoop stress from radial force balance [Pa].

        ``sigma_avg = (1/Delta) integral_{a1}^{a2} r J B(r) dr = P (a2 + 2 a1)/(3 Delta)``
        with the long-coil ``J = B0/(mu0 Delta)`` and ``P = B0^2/(2 mu0)``
        (``central_field`` in T, >= 0).  Tends to ``P a/Delta`` for thin coils and
        P/3 for very thick ones; about twice the virial value E/V (which is a
        lower bound because the radial and axial stresses are compressive).

        Reference: Wilson (1983); Moon (1982).
        """
        b0 = _non_negative("central_field", central_field)
        a1, a2 = self.inner_radius, self.outer_radius
        return magnetic_pressure(b0) * (a2 + 2.0 * a1) / (3.0 * self.winding_thickness)

    def hoop_stress_profile(self, current_density: float, central_field: float, n: int = 51) -> HoopProfile:
        """Sampled hoop-stress profile ``sigma(r) = r J B(r)`` across the winding.

        ``n`` >= 2 equally spaced radii from a1 to a2; ``current_density`` in A/m^2,
        ``central_field`` in T.  ``max``/``r_at`` are the analytic maximum.

        Reference: Wilson (1983).
        """
        _count("n", n, 2)
        rs = linspace(self.inner_radius, self.outer_radius, n)
        rs[-1] = self.outer_radius
        sig = [self.hoop_stress_at(r, current_density, central_field) for r in rs]
        smax, r_at = self.max_hoop_stress(current_density, central_field)
        return HoopProfile(r=tuple(rs), sigma=tuple(sig), max=smax, r_at=r_at)

    def dipole_moment(self, current_density: float) -> float:
        """Magnetic dipole moment of the winding, ``m = pi J L (a2^3 - a1^3)/3`` [A m^2].

        Sum of ``pi r^2 dI`` over the turns with ``dI = J L dr``; ``current_density``
        in A/m^2 (>= 0).  Use with :func:`dipole_field_on_equator` for the stray
        field at r >> a2.

        Reference: Jackson (1999) sec. 5.6.
        """
        j = _non_negative("current_density", current_density)
        return math.pi * j * self.length * (self.outer_radius ** 3 - self.inner_radius ** 3) / 3.0


__all__ = [
    "REFERENCES", "WHEELER_COEFFICIENT", "LONG_COIL_BETA",
    "magnetic_pressure", "field_for_pressure", "energy_density", "pressure_with_background",
    "pulse_increment_for_pressure", "current_sheet_pressure", "surface_current_for_pressure",
    "max_hoop_stress_ratio", "thin_shell_hoop_stress", "virial_minimum_mass", "virial_mean_stress",
    "graded_winding_thickness", "dipole_field_on_equator",
    "PressureUniformity", "HoopProfile", "Solenoid",
]
