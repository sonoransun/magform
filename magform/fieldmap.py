"""Off-axis field of a thick solenoid and the pressure it really puts on a workpiece.

:mod:`magform.magnetics` lives on the axis and in the long-coil limit.  Two
numbers the rest of the package needs live off the axis, and this module
computes them instead of assuming them:

1. **The radial field at the coil ends.**  It is perpendicular to a REBCO tape
   and so sets the critical current of the end turns.  :mod:`magform.conductor`
   uses ``DEFAULT_END_FIELD_FRACTION = 0.4`` ("0.3-0.5 typical", estimated);
   :func:`end_field` integrates the winding and finds 0.25-0.43 of the central
   field for the coil shapes in this package (0.28 for the reference press),
   always on the end face near mid-build.  The same function reports the peak
   field on the conductor, which sits on the bore surface at the mid-plane and
   exceeds the central field in short coils.

2. **The pressure on the workpiece surface.**  The concept note calls it
   "perfectly uniform".  :meth:`magform.magnetics.Solenoid.pressure_uniformity`
   tests that with the free-space field on the axis, but a workpiece that
   screens a pulsed field is not a probe: its eddy currents reshape the field.
   For a pulse much shorter than the workpiece's flux-diffusion time the metal
   behaves as a perfect conductor, the field is tangential at its surface and
   the pressure there is ``B_t^2/(2 mu0)`` with ``B_t = mu0 K`` set by the
   induced sheet current ``K``.  :func:`surface_pressure` solves for ``K`` on
   an axisymmetric body (a boundary-element / method-of-moments solve of
   "zero flux through every ring of the surface") and returns the pressure
   along its outline.  The honest results for a billet in the reference press:
   the pressure over the central 80 % of its length is within ~3 % of uniform
   and a few per cent *above* the naive ``B0^2/(2 mu0)`` (the flux is squeezed
   into the gap), it falls to less than half towards the coil ends, and it
   spikes 2-6x at the billet's own corners (edges concentrate the field).  The
   load is close to uniform only where the part is a plain cylinder well inside
   the coil, and it is never isostatic.

Model
-----
The winding is a uniform-current-density rectangle integrated as circular
loops (complete elliptic integrals by the arithmetic-geometric mean), with an
adaptive subdivision that follows the field point so that points on and inside
the winding converge.  The radial field uses an exact reduction to a
one-dimensional integral.  On the axis everything agrees with Montgomery's
closed form used in :mod:`magform.magnetics`.

Validity: axisymmetric, non-magnetic, quasi-static; uniform winding current
density (no grading, no leads); the workpiece is a perfect conductor (skin
depth much smaller than its size and than the gap, pulse much shorter than
its diffusion time) that started field-free.  SI units; ``r`` is the distance
from the axis and ``z`` the axial distance from the coil mid-plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from ._numerics import gauss_legendre, solve_linear
from ._report import ResultMixin, unit
from .constants import MU0
from .errors import InputError
from .magnetics import Solenoid

REFERENCES: dict[str, str] = {
    "smythe1968": "W. R. Smythe, Static and Dynamic Electricity, 3rd ed., McGraw-Hill, New York (1968), "
                  "sec. 7.10 (vector potential and field of a circular loop).",
    "jackson1999": "J. D. Jackson, Classical Electrodynamics, 3rd ed., Wiley, New York (1999), sec. 5.5 "
                   "(circular current loop) and sec. 5.13 (perfectly conducting sphere in a uniform field: "
                   "surface field (3/2) B0 sin(theta)).",
    "abramowitz1964": "M. Abramowitz and I. A. Stegun (eds.), Handbook of Mathematical Functions, National "
                      "Bureau of Standards, Washington (1964), sec. 17.6 (elliptic integrals by the "
                      "arithmetic-geometric mean).",
    "montgomery1969": "D. B. Montgomery, Solenoid Magnet Design: The Magnetic and Mechanical Aspects of "
                      "Resistive and Superconducting Systems, Wiley-Interscience, New York (1969).",
    "grover1946": "F. W. Grover, Inductance Calculations: Working Formulas and Tables, Van Nostrand, "
                  "New York (1946) (flux of a circular filament near itself; geometric mean distances).",
    "harrington1968": "R. F. Harrington, Field Computation by Moment Methods, Macmillan, New York (1968).",
    "knoepfel2000": "H. Knoepfel, Magnetic Fields: A Comprehensive Theoretical Treatise for Practical Use, "
                    "Wiley, New York (2000) (flux exclusion by conductors, field concentration at edges).",
    "wilson1983": "M. N. Wilson, Superconducting Magnets, Clarendon Press, Oxford (1983) (peak and radial "
                  "fields in solenoid windings).",
}

QUADRATURE_ORDER: int = 4
"""Gauss-Legendre points per direction on each cell of the adaptive winding integration."""

SUBDIVISION_DEPTH: int = 22
"""Maximum number of bisections of a winding cell that contains (or touches) the field point."""

SUBDIVISION_RATIO: float = 1.0
"""A winding cell is bisected while its larger side exceeds this multiple of its distance to the field point."""

SINGULAR_ORDER: int = 16
"""Gauss-Legendre points on each side of a logarithmic singularity (after the ``s = t^2`` substitution)."""

NEAR_PANEL_FACTOR: float = 6.0
"""A source panel closer than this multiple of its own length is integrated (not lumped into one loop)."""

NEAR_PANEL_ORDER: int = 6
"""Gauss-Legendre points along a near source panel."""

DEFAULT_CENTRAL_FRACTION: float = 0.8
"""Fraction of the workpiece half-length over which the 'central' pressure statistics are taken."""

_M_MAX: float = 1.0 - 1e-15


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


def _solenoid(solenoid: Solenoid) -> Solenoid:
    if not isinstance(solenoid, Solenoid):
        raise InputError("solenoid must be a magform.magnetics.Solenoid")
    return solenoid


# ---------------------------------------------------------------------------
# elliptic integrals and the circular loop
# ---------------------------------------------------------------------------
def ellipke(m: float) -> tuple[float, float]:
    """Complete elliptic integrals ``(K(m), E(m))`` of the first and second kind.

    ``K = integral_0^{pi/2} (1 - m sin^2 t)^(-1/2) dt``, ``E = integral_0^{pi/2} (1 - m sin^2 t)^(1/2) dt``
    with the *parameter* ``m = k^2`` (0 <= m < 1).  Computed with the
    arithmetic-geometric mean, which converges quadratically (double precision
    in at most 8 steps).  Checks: m = 0 -> (pi/2, pi/2); m = 0.5 -> (1.8540747, 1.3506439).

    Reference: Abramowitz & Stegun (1964) sec. 17.6.
    """
    mm = _finite("m", m)
    if not 0.0 <= mm < 1.0:
        raise InputError(f"m must lie in [0, 1), got {mm!r}")
    return _ellipke(mm)


def _ellipke(m: float) -> tuple[float, float]:
    a = 1.0
    b = math.sqrt(1.0 - m)
    c = math.sqrt(m)
    total = 0.5 * c * c
    power = 0.5
    for _ in range(40):
        if abs(c) < 1e-17:
            break
        a_next = 0.5 * (a + b)
        c = 0.5 * (a - b)
        b = math.sqrt(a * b)
        a = a_next
        power *= 2.0
        total += power * c * c
    k = math.pi / (2.0 * a)
    return k, k * (1.0 - total)


def _loop_aphi(a: float, r: float, zeta: float) -> float:
    """A_phi per unit current of a loop of radius ``a`` at axial distance ``zeta`` (r > 0)."""
    q = (a + r) * (a + r) + zeta * zeta
    m = 4.0 * a * r / q
    if m > _M_MAX:
        m = _M_MAX
    if m < 1e-3:  # far from the loop or close to the axis: series in m, avoids the K - E cancellation
        return MU0 * a * a * r / (4.0 * q ** 1.5) * (1.0 + 0.75 * m + 0.5859375 * m * m)
    k, e = _ellipke(m)
    return MU0 / (math.pi * math.sqrt(m)) * math.sqrt(a / r) * ((1.0 - 0.5 * m) * k - e)


def _loop_psi(a: float, r: float, zeta: float) -> float:
    return r * _loop_aphi(a, r, zeta) if r > 0.0 else 0.0


def _loop_bz(a: float, r: float, zeta: float) -> float:
    if r < 1e-12 * a:
        return MU0 * a * a / (2.0 * (a * a + zeta * zeta) ** 1.5)
    q = (a + r) * (a + r) + zeta * zeta
    m = 4.0 * a * r / q
    if m > _M_MAX:
        m = _M_MAX
    k, e = _ellipke(m)
    d = (a - r) * (a - r) + zeta * zeta
    if d <= 0.0:
        d = 1e-300
    return MU0 / (2.0 * math.pi * math.sqrt(q)) * (k + (a * a - r * r - zeta * zeta) / d * e)


def _loop_br(a: float, r: float, zeta: float) -> float:
    if r < 1e-12 * a or zeta == 0.0:
        return 0.0
    q = (a + r) * (a + r) + zeta * zeta
    m = 4.0 * a * r / q
    if m > _M_MAX:
        m = _M_MAX
    k, e = _ellipke(m)
    d = (a - r) * (a - r) + zeta * zeta
    return MU0 * zeta / (2.0 * math.pi * r * math.sqrt(q)) * (-k + (a * a + r * r + zeta * zeta) / d * e)


def loop_field(loop_radius: float, r: float, z: float, current: float = 1.0) -> tuple[float, float]:
    """Field ``(B_r, B_z)`` [T] of a circular current loop at the point (r, z).

    The loop of radius ``a`` lies in the plane z = 0 and carries ``current`` [A].  With
    ``k^2 = 4 a r / ((a + r)^2 + z^2)``::

        B_z = mu0 I / (2 pi sqrt((a+r)^2 + z^2)) [ K + (a^2 - r^2 - z^2)/((a-r)^2 + z^2) E ]
        B_r = mu0 I z / (2 pi r sqrt((a+r)^2 + z^2)) [ -K + (a^2 + r^2 + z^2)/((a-r)^2 + z^2) E ]

    ``loop_radius`` > 0, ``r`` >= 0 (m).  On the axis ``B_z = mu0 I a^2 / (2 (a^2 + z^2)^(3/2))``.
    The point must not lie on the wire itself.

    Reference: Smythe (1968) sec. 7.10; Jackson (1999) sec. 5.5.
    """
    a = _positive("loop_radius", loop_radius)
    rr = _non_negative("r", r)
    zz = _finite("z", z)
    i = _finite("current", current)
    if rr == a and zz == 0.0:
        raise InputError("the field is singular on the loop itself")
    return i * _loop_br(a, rr, zz), i * _loop_bz(a, rr, zz)


def loop_flux_function(loop_radius: float, r: float, z: float, current: float = 1.0) -> float:
    """Flux function ``psi = r A_phi`` [Wb/rad] of a circular loop at the point (r, z).

    ``A_phi = mu0 I / (pi k) sqrt(a/r) [ (1 - k^2/2) K - E ]`` with ``k^2 = 4 a r / ((a + r)^2 + z^2)``.
    ``2 pi psi`` is the flux through the coaxial circle of radius r at height z, so lines of
    constant ``psi`` are field lines, and the mutual inductance of two coaxial loops is
    ``2 pi psi / I``.  ``loop_radius`` > 0, ``r`` >= 0 (m); 0 on the axis.

    Reference: Smythe (1968) sec. 7.10.
    """
    a = _positive("loop_radius", loop_radius)
    rr = _non_negative("r", r)
    zz = _finite("z", z)
    i = _finite("current", current)
    if rr == a and zz == 0.0:
        raise InputError("the flux function is singular on the loop itself")
    return i * _loop_psi(a, rr, zz)


# ---------------------------------------------------------------------------
# adaptive integration over the winding cross-section
# ---------------------------------------------------------------------------
def _winding_integral(kernel: Callable[[float, float, float], float], sol: Solenoid, r: float, z: float) -> float:
    """Integral of ``kernel(a, r, z - z')`` over the winding rectangle (per unit current density).

    Cells are bisected along their longer side while they are large compared with their
    distance to the field point, then integrated with a tensor Gauss-Legendre rule, so the
    mesh grades towards (and into) the point where the loop kernels are singular.
    """
    xs, ws = gauss_legendre(QUADRATURE_ORDER)
    half = 0.5 * sol.length
    total = 0.0
    stack = [(sol.inner_radius, sol.outer_radius, -half, half, 0)]
    while stack:
        r0, r1, z0, z1, depth = stack.pop()
        dr = r1 - r0
        dz = z1 - z0
        gap_r = r0 - r if r < r0 else (r - r1 if r > r1 else 0.0)
        gap_z = z0 - z if z < z0 else (z - z1 if z > z1 else 0.0)
        dist = math.hypot(gap_r, gap_z)
        size = dr if dr > dz else dz
        if depth < SUBDIVISION_DEPTH and size > SUBDIVISION_RATIO * dist:
            if dr > dz:
                mid = 0.5 * (r0 + r1)
                stack.append((r0, mid, z0, z1, depth + 1))
                stack.append((mid, r1, z0, z1, depth + 1))
            else:
                mid = 0.5 * (z0 + z1)
                stack.append((r0, r1, z0, mid, depth + 1))
                stack.append((r0, r1, mid, z1, depth + 1))
            continue
        rc = 0.5 * (r0 + r1)
        zc = 0.5 * (z0 + z1)
        cell = 0.0
        for xi, wi in zip(xs, ws):
            a = rc + 0.5 * dr * xi
            row = 0.0
            for xj, wj in zip(xs, ws):
                row += wj * kernel(a, r, z - (zc + 0.5 * dz * xj))
            cell += wi * row
        total += cell * 0.25 * dr * dz
    return total


def _singular_integral(f: Callable[[float], float], lo: float, hi: float, singular: float) -> float:
    """Integral of ``f`` on [lo, hi] with an integrable (logarithmic) singularity at ``singular``.

    The interval is split at the singular point (clamped into it) and each side is mapped
    with ``s = t^2`` so that the integrand vanishes at the singular end.
    """
    xs, ws = gauss_legendre(SINGULAR_ORDER)
    split = lo if singular < lo else (hi if singular > hi else singular)
    total = 0.0
    for start, length, sign in ((split, split - lo, -1.0), (split, hi - split, 1.0)):
        if length <= 0.0:
            continue
        for x, w in zip(xs, ws):
            u = 0.5 * (x + 1.0)
            total += w * u * length * f(start + sign * length * u * u)
    return total


@dataclass(frozen=True, kw_only=True)
class FieldPoint(ResultMixin):
    """Field of a solenoid at one point: components, magnitude and angle to the axis.

    ``angle`` is ``atan2(|B_r|, |B_z|)`` in radians (0 on the axis and at the mid-plane).

    Reference: Smythe (1968); Montgomery (1969).
    """

    r: float = unit("m")
    z: float = unit("m")
    br: float = unit("T")
    bz: float = unit("T")
    magnitude: float = unit("T")
    angle: float = unit("rad")


def solenoid_radial_field(solenoid: Solenoid, current_density: float, r: float, z: float) -> float:
    """Radial field ``B_r`` [T] of a uniform-J thick solenoid at the point (r, z).

    Because ``B_r = -dA_phi/dz``, the axial integration over the winding is exact and only the
    radial one remains::

        B_r(r, z) = J * integral_{a1}^{a2} [ a_phi(a; r, z - L/2) - a_phi(a; r, z + L/2) ] da

    where ``a_phi(a; r, zeta)`` is the vector potential per unit current of a loop of radius ``a``
    at axial distance ``zeta``.  ``current_density`` J in A/m^2 (winding average, >= 0), ``r`` >= 0,
    any ``z`` (m).  Zero on the axis and on the mid-plane; valid inside the winding (the
    logarithmic singularity on an end face is integrated exactly).

    Reference: Smythe (1968) sec. 7.10; Wilson (1983).
    """
    sol = _solenoid(solenoid)
    j = _non_negative("current_density", current_density)
    rr = _non_negative("r", r)
    zz = _finite("z", z)
    return j * _radial_unit(sol, rr, zz)


def _radial_unit(sol: Solenoid, r: float, z: float) -> float:
    if r <= 0.0 or z == 0.0:
        return 0.0
    half = 0.5 * sol.length

    def integrand(a: float) -> float:
        return _loop_aphi(a, r, z - half) - _loop_aphi(a, r, z + half)

    return _singular_integral(integrand, sol.inner_radius, sol.outer_radius, r)


def _axial_unit(sol: Solenoid, r: float, z: float) -> float:
    if r <= 0.0:
        return sol.axial_field(z, 1.0)
    return _winding_integral(_loop_bz, sol, r, z)


def solenoid_field(solenoid: Solenoid, current_density: float, r: float, z: float) -> FieldPoint:
    """Field of a uniform-J thick solenoid at any point (r, z), inside the winding included.

    ``B = J * integral over the winding cross-section of the loop field`` (adaptive
    Gauss-Legendre for ``B_z``; :func:`solenoid_radial_field` for ``B_r``).  On the axis the
    closed form of :meth:`magform.magnetics.Solenoid.axial_field` is returned.
    ``current_density`` in A/m^2 (>= 0), ``r`` >= 0 and ``z`` in m.  Accuracy: better than
    1e-4 of the central field away from the winding edges, ~1e-3 on them.

    Reference: Smythe (1968) sec. 7.10; Montgomery (1969).
    """
    sol = _solenoid(solenoid)
    j = _non_negative("current_density", current_density)
    rr = _non_negative("r", r)
    zz = _finite("z", z)
    br = j * _radial_unit(sol, rr, zz)
    bz = j * _axial_unit(sol, rr, zz)
    return FieldPoint(r=rr, z=zz, br=br, bz=bz, magnitude=math.hypot(br, bz),
                      angle=math.atan2(abs(br), abs(bz)))


def solenoid_flux_function(solenoid: Solenoid, current_density: float, r: float, z: float) -> float:
    """Flux function ``psi = r A_phi`` [Wb/rad] of a uniform-J thick solenoid at (r, z).

    ``psi = J * integral over the winding of the loop flux function``; ``2 pi psi`` is the flux
    through the circle of radius r at height z, so contours of ``psi`` are the field lines.
    ``current_density`` in A/m^2 (>= 0), ``r`` >= 0 and ``z`` in m.

    Reference: Smythe (1968) sec. 7.10.
    """
    sol = _solenoid(solenoid)
    j = _non_negative("current_density", current_density)
    rr = _non_negative("r", r)
    zz = _finite("z", z)
    if rr == 0.0:
        return 0.0
    return j * _winding_integral(_loop_psi, sol, rr, zz)


# ---------------------------------------------------------------------------
# peak and radial fields on the conductor
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class EndField(ResultMixin):
    """Radial (tape-perpendicular) and peak fields in a solenoid winding, as fractions of ``B0``.

    ``fraction`` = max |B_r| / B0 over the winding, found on the end face at ``radius``;
    ``peak_field_ratio`` = |B| / B0 at the bore surface on the mid-plane (the highest field on
    the conductor); ``field_factor`` is Montgomery's f = B0/(mu0 J Delta).

    Reference: Wilson (1983); Montgomery (1969).
    """

    fraction: float
    radius: float = unit("m")
    z: float = unit("m")
    peak_field_ratio: float
    field_factor: float
    alpha: float
    beta: float


def end_field(solenoid: Solenoid, n_scan: int = 48) -> EndField:
    """Largest radial field in the winding and the peak field on the conductor, relative to ``B0``.

    ``fraction = max_{a1 <= r <= a2} |B_r(r, L/2)| / B0``: the maximum over the whole winding lies
    on the end faces (checked on a grid for 1 <= beta <= 10, 2 <= alpha <= 8), near mid-build.
    The end face is scanned at ``n_scan`` + 1 radii and the maximum refined by golden-section
    search.  ``peak_field_ratio = B_z(a1, 0)/B0`` is the field on the bore surface at the
    mid-plane.  Both are independent of the current density.  Examples: a1 = 50 mm, a2 = 250 mm,
    L = 0.5 m -> fraction 0.282, peak ratio 1.005; alpha = 2, beta = 3 -> 0.387; a short coil
    (alpha = 5, beta = 1) -> 0.433 with a peak ratio of 1.107; a long thin coil (alpha = 1.4,
    beta = 10) -> 0.461.

    Use it in place of ``conductor.DEFAULT_END_FIELD_FRACTION`` (0.4, estimated) when the
    geometry is known.

    Reference: Wilson (1983); Smythe (1968) sec. 7.10.
    """
    sol = _solenoid(solenoid)
    n = _count("n_scan", n_scan, 4)
    j = sol.current_density_for_field(1.0)  # B0 = 1 T
    half = 0.5 * sol.length
    a1, a2 = sol.inner_radius, sol.outer_radius

    def br(radius: float) -> float:
        return abs(j * _radial_unit(sol, radius, half))

    radii = [a1 + (a2 - a1) * i / n for i in range(n + 1)]
    values = [br(x) for x in radii]
    k = max(range(n + 1), key=lambda i: values[i])
    lo = radii[max(k - 1, 0)]
    hi = radii[min(k + 1, n)]
    golden = 0.5 * (math.sqrt(5.0) - 1.0)
    x1 = hi - golden * (hi - lo)
    x2 = lo + golden * (hi - lo)
    f1, f2 = br(x1), br(x2)
    for _ in range(40):
        if hi - lo <= 1e-6 * (a2 - a1):
            break
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + golden * (hi - lo)
            f2 = br(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - golden * (hi - lo)
            f1 = br(x1)
    best_r, best = (x1, f1) if f1 >= f2 else (x2, f2)
    if values[k] > best:
        best_r, best = radii[k], values[k]
    peak = j * _winding_integral(_loop_bz, sol, a1, 0.0)
    return EndField(fraction=best, radius=best_r, z=half, peak_field_ratio=peak,
                    field_factor=sol.field_factor(), alpha=sol.alpha, beta=sol.beta)


# ---------------------------------------------------------------------------
# workpiece outlines
# ---------------------------------------------------------------------------
Profile = tuple[tuple[float, float], ...]


def cylinder_profile(radius: float, length: float, corner_radius: float = 0.0, n_side: int = 60,
                     n_cap: int = 14, n_corner: int = 8) -> Profile:
    """Outline ``((r, z), ...)`` of a solid cylinder, from the axis at +L/2 to the axis at -L/2.

    End cap (``n_cap`` panels, refined towards the rim), optional rounded corner of
    ``corner_radius`` (``n_corner`` panels per corner; 0 = sharp edge), side (``n_side``
    panels, refined towards both ends), then the mirror image.  ``radius`` and ``length`` in m;
    ``0 <= corner_radius <= min(radius, length/2)``.  A sharp edge concentrates the field
    without limit (surface field ~ distance^(-1/3)), so its panels report a mesh-dependent
    peak; give real parts their corner radius.

    Reference: Knoepfel (2000) (field concentration at edges); Harrington (1968).
    """
    rad = _positive("radius", radius)
    ell = _positive("length", length)
    rc = _non_negative("corner_radius", corner_radius)
    ns = _count("n_side", n_side, 4)
    nc = _count("n_cap", n_cap, 2)
    nk = _count("n_corner", n_corner, 2)
    half = 0.5 * ell
    if rc > min(rad, half) * (1.0 + 1e-12):
        raise InputError("corner_radius cannot exceed the cylinder radius or half its length")
    rc = min(rc, rad, half)

    def cosine(i: int, n: int) -> float:  # 0..1, clustered at both ends
        return 0.5 * (1.0 - math.cos(math.pi * i / n))

    def toward_one(i: int, n: int) -> float:  # 0..1, clustered at 1
        return math.sin(0.5 * math.pi * i / n)

    top: list[tuple[float, float]] = []
    flat = rad - rc
    if flat > 0.0:
        top.extend((flat * toward_one(i, nc), half) for i in range(nc + 1))
    else:
        top.append((0.0, half))
    if rc > 0.0:
        top.extend((flat + rc * math.sin(0.5 * math.pi * i / nk), half - rc + rc * math.cos(0.5 * math.pi * i / nk))
                   for i in range(1, nk + 1))
    side_half = half - rc
    points = list(top)
    if side_half > 0.0:
        points.extend((rad, side_half - 2.0 * side_half * cosine(i, ns)) for i in range(1, ns + 1))
    points.extend((r_, -z_) for r_, z_ in reversed(top[:-1]))
    return tuple(points)


def sphere_profile(radius: float, n: int = 64) -> Profile:
    """Outline of a sphere of ``radius`` [m] as ``n`` equal-angle panels from pole to pole.

    Used to validate :func:`surface_pressure` against the textbook result for a perfectly
    conducting sphere in a uniform field, ``B_surface = (3/2) B0 sin(theta)``.

    Reference: Jackson (1999) sec. 5.13.
    """
    rad = _positive("radius", radius)
    nn = _count("n", n, 4)
    return tuple((rad * math.sin(math.pi * i / nn) if 0 < i < nn else 0.0, rad * math.cos(math.pi * i / nn))
                 for i in range(nn + 1))


def _check_profile(profile: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    try:
        pts = [(float(p[0]), float(p[1])) for p in profile]
    except (TypeError, ValueError, IndexError):
        raise InputError("profile must be a sequence of (r, z) pairs") from None
    if len(pts) < 5:
        raise InputError("profile needs at least 5 points")
    for r_, z_ in pts:
        if not (math.isfinite(r_) and math.isfinite(z_)) or r_ < 0.0:
            raise InputError("profile points must be finite with r >= 0")
    if pts[0][0] != 0.0 or pts[-1][0] != 0.0:
        raise InputError("profile must start and end on the axis (r = 0): a closed body of revolution")
    for (r0, z0), (r1, z1) in zip(pts, pts[1:]):
        if math.hypot(r1 - r0, z1 - z0) <= 0.0:
            raise InputError("profile has a zero-length segment")
    return pts


# ---------------------------------------------------------------------------
# perfectly conducting workpiece: induced sheet current and surface pressure
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class SurfacePressure(ResultMixin):
    """Magnetic pressure along the outline of a perfectly conducting workpiece in a solenoid.

    Per panel (ordered from the +z pole to the -z pole): midpoint ``r``, ``z``; arc length
    ``s`` from the +z pole; ``surface_field`` = mu0 |K| (tangential field just outside the
    metal); ``pressure`` = B^2/(2 mu0); ``ratio`` = pressure / ``reference_pressure``.
    ``panel_current`` is the induced current of each panel (A, sign opposing the coil).

    ``reference_field`` is the free-space central field B0 of the coil at the same current
    and ``reference_pressure`` its B0^2/(2 mu0) -- the number a 'pressure = B^2/2mu0'
    estimate would quote.  Central statistics cover panels with |z| <= ``central_fraction`` x
    the body's half-length: ``mid_ratio`` (panel nearest z = 0), ``central_min_ratio``,
    ``central_max_ratio``, ``central_mean_ratio`` (area-weighted) and ``uniformity`` =
    min/max.  ``peak_ratio`` at (``peak_r``, ``peak_z``) is the largest value anywhere,
    usually at a corner.  ``excluded_flux_fraction`` is the largest |psi| left on the surface
    relative to the coil's own flux function there (solver residual; 0 for an exact solve).

    Reference: Harrington (1968); Knoepfel (2000); Jackson (1999) sec. 5.13.
    """

    reference_field: float = unit("T")
    reference_pressure: float = unit("Pa")
    mid_ratio: float
    central_min_ratio: float
    central_max_ratio: float
    central_mean_ratio: float
    uniformity: float
    central_fraction: float
    peak_ratio: float
    peak_r: float = unit("m")
    peak_z: float = unit("m")
    half_length: float = unit("m")
    max_radius: float = unit("m")
    radial_gap: float = unit("m")
    n_panels: int
    excluded_flux_fraction: float
    s: tuple[float, ...] = unit("m")
    r: tuple[float, ...] = unit("m")
    z: tuple[float, ...] = unit("m")
    surface_field: tuple[float, ...] = unit("T")
    pressure: tuple[float, ...] = unit("Pa")
    ratio: tuple[float, ...]
    panel_current: tuple[float, ...] = unit("A")
    panel_length: tuple[float, ...] = unit("m")


def _panel_flux(ra: float, za: float, rb: float, zb: float, r: float, z: float) -> float:
    """psi at (r, z) per unit total current spread uniformly along the panel a-b (not on it)."""
    rm = 0.5 * (ra + rb)
    zm = 0.5 * (za + zb)
    length = math.hypot(rb - ra, zb - za)
    dist = math.hypot(r - rm, z - zm)
    if dist > NEAR_PANEL_FACTOR * length:
        return _loop_psi(rm, r, z - zm) if rm > 0.0 else 0.0
    xs, ws = gauss_legendre(NEAR_PANEL_ORDER)
    total = 0.0
    for x, w in zip(xs, ws):
        t = 0.5 * (x + 1.0)
        a = ra + (rb - ra) * t
        if a > 0.0:
            total += w * _loop_psi(a, r, z - (za + (zb - za) * t))
    return 0.5 * total


def _panel_self_flux(ra: float, za: float, rb: float, zb: float) -> float:
    """psi at a panel's own midpoint per unit total current spread uniformly along it."""
    rm = 0.5 * (ra + rb)
    zm = 0.5 * (za + zb)
    xs, ws = gauss_legendre(SINGULAR_ORDER)
    total = 0.0
    for end_r, end_z in ((ra, za), (rb, zb)):
        for x, w in zip(xs, ws):
            u = 0.5 * (x + 1.0)
            t = u * u  # 0 at the midpoint (singular), 1 at the panel end
            a = rm + (end_r - rm) * t
            if a > 0.0:
                total += w * u * _loop_psi(a, rm, zm - (zm + (end_z - zm) * t))
    # each half: integral_0^1 f(t) dt = integral_0^1 2 u f(u^2) du, GL weight w/2; average of the two halves
    return 0.5 * total


def surface_pressure(solenoid: Solenoid, current_density: float, profile: Sequence[Sequence[float]],
                     central_fraction: float = DEFAULT_CENTRAL_FRACTION) -> SurfacePressure:
    """Magnetic pressure on a perfectly conducting body of revolution centred in a solenoid.

    The body's surface carries a sheet current ``K`` that keeps the field out of the metal.
    With the surface split into ring panels (panel j: total current ``I_j``, length ``w_j``),
    excluding the flux means the flux function vanishes on every panel::

        psi_coil(r_i, z_i) + sum_j G_ij I_j = 0,      G_ij = psi at panel i per unit current in panel j

    (``G_ij`` from the loop flux function, integrated along near panels and analytically
    regularised on the panel itself).  The dense system is solved directly; then
    ``K_j = I_j / w_j``, the field just outside the metal is tangential with
    ``B = mu0 |K|``, and the pressure is ``B^2/(2 mu0)``.

    ``profile`` is the outline ``((r, z), ...)`` from the axis to the axis (see
    :func:`cylinder_profile`), entirely inside the bore; ``current_density`` in A/m^2 (> 0);
    ``central_fraction`` in (0, 1].  Checks: a sphere in a long coil gives (3/2) B0 at its
    equator (pressure ratio 2.25); a long cylinder in a long coil gives the gap field
    ``mu0 J Delta`` (ratio 1/f^2 with f the Montgomery field factor).

    Validity: perfect conductor -- skin depth << body size and << gap, pulse << flux-diffusion
    time -- that was field-free before the pulse; rigid body; axisymmetric.  A real workpiece
    lets some flux in (lower pressure, see :mod:`magform.eddy`) and moves.

    Reference: Harrington (1968); Smythe (1968) sec. 7.10; Jackson (1999) sec. 5.13.
    """
    sol = _solenoid(solenoid)
    j = _positive("current_density", current_density)
    frac = _positive("central_fraction", central_fraction)
    if frac > 1.0:
        raise InputError(f"central_fraction must lie in (0, 1], got {frac!r}")
    pts = _check_profile(profile)
    r_max = max(p[0] for p in pts)
    z_half = max(abs(p[1]) for p in pts)
    if r_max >= sol.inner_radius:
        raise InputError(f"the workpiece (r_max = {r_max!r} m) does not fit inside the bore "
                         f"(inner radius {sol.inner_radius!r} m)")

    panels = [(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) for i in range(len(pts) - 1)]
    n = len(panels)
    mids = [(0.5 * (ra + rb), 0.5 * (za + zb)) for ra, za, rb, zb in panels]
    lengths = [math.hypot(rb - ra, zb - za) for ra, za, rb, zb in panels]

    psi_coil = [j * _winding_integral(_loop_psi, sol, rm, zm) for rm, zm in mids]
    matrix = []
    for i, (ri, zi) in enumerate(mids):
        row = []
        for k, (ra, za, rb, zb) in enumerate(panels):
            row.append(_panel_self_flux(ra, za, rb, zb) if k == i else _panel_flux(ra, za, rb, zb, ri, zi))
        matrix.append(row)
    currents = solve_linear(matrix, [-p for p in psi_coil])

    residual = 0.0
    for i in range(n):
        left = psi_coil[i] + sum(matrix[i][k] * currents[k] for k in range(n))
        scale = abs(psi_coil[i])
        if scale > 0.0:
            residual = max(residual, abs(left) / scale)

    b_ref = sol.central_field(j)
    p_ref = b_ref * b_ref / (2.0 * MU0)
    fields = [MU0 * abs(currents[i]) / lengths[i] for i in range(n)]
    pressures = [b * b / (2.0 * MU0) for b in fields]
    ratios = [p / p_ref for p in pressures]

    s_mid: list[float] = []
    run = 0.0
    for length in lengths:
        s_mid.append(run + 0.5 * length)
        run += length

    central = [i for i in range(n) if abs(mids[i][1]) <= frac * z_half and mids[i][0] > 0.5 * r_max]
    if not central:
        central = [min(range(n), key=lambda i: abs(mids[i][1]))]
    i_mid = min(central, key=lambda i: abs(mids[i][1]))
    weights = [mids[i][0] * lengths[i] for i in central]
    mean = sum(w * ratios[i] for w, i in zip(weights, central)) / sum(weights)
    c_min = min(ratios[i] for i in central)
    c_max = max(ratios[i] for i in central)
    i_peak = max(range(n), key=lambda i: ratios[i])
    return SurfacePressure(
        reference_field=b_ref,
        reference_pressure=p_ref,
        mid_ratio=ratios[i_mid],
        central_min_ratio=c_min,
        central_max_ratio=c_max,
        central_mean_ratio=mean,
        uniformity=c_min / c_max if c_max > 0.0 else 0.0,
        central_fraction=frac,
        peak_ratio=ratios[i_peak],
        peak_r=mids[i_peak][0],
        peak_z=mids[i_peak][1],
        half_length=z_half,
        max_radius=r_max,
        radial_gap=sol.inner_radius - r_max,
        n_panels=n,
        excluded_flux_fraction=residual,
        s=tuple(s_mid),
        r=tuple(m[0] for m in mids),
        z=tuple(m[1] for m in mids),
        surface_field=tuple(fields),
        pressure=tuple(pressures),
        ratio=tuple(ratios),
        panel_current=tuple(currents),
        panel_length=tuple(lengths),
    )


def workpiece_pressure(solenoid: Solenoid, radius: float, length: float, corner_radius: float | None = None,
                       central_fraction: float = DEFAULT_CENTRAL_FRACTION, n_side: int = 60) -> SurfacePressure:
    """Pressure distribution on a solid cylindrical workpiece centred in the coil (unit central field).

    Convenience wrapper: :func:`cylinder_profile` + :func:`surface_pressure` at the current
    density that gives a free-space central field of 1 T, so ``ratio`` is the pressure in
    units of the naive ``B0^2/(2 mu0)`` (ratios do not depend on the current).
    ``corner_radius`` defaults to 5 % of the workpiece radius (a broken edge).  Example: a
    40 mm radius, 0.3 m long billet in the reference press (a1 = 50 mm, a2 = 250 mm,
    L = 0.5 m) sees 1.06x the naive pressure at its mid-plane and stays within 3 % of that
    over the central 80 % of its length, but its 2 mm corners see 5.7x (2.4x with a 10 mm
    corner radius); a billet as long as the coil drops to 0.43x at the coil ends.

    Reference: Harrington (1968); Knoepfel (2000).
    """
    sol = _solenoid(solenoid)
    rad = _positive("radius", radius)
    rc = 0.05 * rad if corner_radius is None else _non_negative("corner_radius", corner_radius)
    profile = cylinder_profile(rad, length, min(rc, rad, 0.5 * _positive("length", length)), n_side=n_side)
    return surface_pressure(sol, sol.current_density_for_field(1.0), profile, central_fraction)


def flux_function_with_workpiece(solenoid: Solenoid, current_density: float, solution: SurfacePressure,
                                 r: float, z: float) -> float:
    """Total flux function ``psi`` [Wb/rad] at (r, z): coil plus the workpiece's induced currents.

    ``psi = psi_coil + sum_j I_j psi_loop(r_j; r, z - z_j)`` with the panel currents of
    ``solution`` (from :func:`surface_pressure` at the same ``current_density``).  Its contours
    are the field lines squeezed around the workpiece; it is ~0 inside the body.  Points on
    the body's surface use the lumped-loop value and are approximate.

    Reference: Smythe (1968) sec. 7.10; Harrington (1968).
    """
    sol = _solenoid(solenoid)
    j = _non_negative("current_density", current_density)
    rr = _non_negative("r", r)
    zz = _finite("z", z)
    if not isinstance(solution, SurfacePressure):
        raise InputError("solution must be a SurfacePressure from surface_pressure()")
    if rr == 0.0:
        return 0.0
    total = j * _winding_integral(_loop_psi, sol, rr, zz)
    for rm, zm, current in zip(solution.r, solution.z, solution.panel_current):
        if rm > 0.0 and not (rm == rr and zm == zz):
            total += current * _loop_psi(rm, rr, zz - zm)
    return total


__all__ = [
    "REFERENCES", "QUADRATURE_ORDER", "SUBDIVISION_DEPTH", "SUBDIVISION_RATIO", "SINGULAR_ORDER",
    "NEAR_PANEL_FACTOR", "NEAR_PANEL_ORDER", "DEFAULT_CENTRAL_FRACTION",
    "ellipke", "loop_field", "loop_flux_function",
    "FieldPoint", "solenoid_radial_field", "solenoid_field", "solenoid_flux_function",
    "EndField", "end_field",
    "Profile", "cylinder_profile", "sphere_profile",
    "SurfacePressure", "surface_pressure", "workpiece_pressure", "flux_function_with_workpiece",
]
