"""Electromagnetic (magnetic-pulse) forming: eddy currents, field diffusion,
skin heating and rigid-plastic tube compression.

Process naming
--------------
The concept README calls its first process "Dynamic Magneto-Hydrodynamic (MHD)
Forging".  For a *solid* workpiece that is a misnomer: magnetohydrodynamics is
the dynamics of electrically conducting fluids.  Pulsing a field around a solid
conductor so that the induced (eddy) current and the field produce the Lorentz
pressure ``(B_out**2 - B_in**2) / (2 mu0)`` is **electromagnetic
(magnetic-pulse) forming** (EMF), an industrial process since the 1960s that
operates at 20-50 T, i.e. 0.16-1 GPa (Psyk et al. 2011;
``benchmarks.EMF_CONVENTIONAL_FIELD``).  The README's 63-100 T window is 1.6-25x
higher in pressure.  The "MHD" label becomes literally true only where the
eddy-current skin melts during the pulse and the surface layer is a conducting
liquid (:func:`skin_melting_field`, kappa = 1): above ~60 T for aluminium,
~96 T for copper, ~96-99 T for titanium, ~117 T for mild steel and ~149 T for
tungsten.

What this module computes
-------------------------
* skin depth, shell time constant, magnetic diffusion time, thin/thick regime,
  quasi-steady sheet shielding and the net pressure on a shell;
* skin-layer Joule heating ``dT = kappa B**2 / (2 mu0 rho_m c_p)``, which is
  independent of the pulse length and contradicts the README's "room
  temperature" and "no heat-affected zone" wording at 63-100 T (Cu heats by
  ~1150 K at 100 T; Knoepfel 1970, Herlach 1986);
* the lumped interior field of a shell (exact exponential update), action
  integrals and adiabatic Joule heating of conductors;
* a 1-D nonlinear magnetic-diffusion solver (Crank-Nicolson) for a slab whose
  resistivity rises as it heats;
* drive waveforms (damped sine, half sine), a pulse-frequency recommendation,
  and :class:`TubeCompression`, a rigid-plastic thin-shell model of tube
  compression with flux compression, wall thickening, an optional sacrificial
  driver sheet, a mandrel stop and no elastic rebound.

All quantities are SI: T, Pa, m, s, K, Ohm m, J.  Ferromagnetic workpieces are
treated with relative permeability 1 (fully saturated at the fields of
interest, B >> 2 T).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Callable, Sequence

from ._numerics import bisect, solve_quadratic, solve_tridiagonal, trapz
from ._report import ResultMixin, unit
from .constants import MU0, T_ROOM
from .errors import InputError
from .materials import Material, MaterialLike, as_material

REFERENCES: dict[str, str] = {
    "psyk_2011": (
        "V. Psyk, D. Risch, B. L. Kinsey, A. E. Tekkaya and M. Kleiner, Electromagnetic forming - "
        "A review, J. Mater. Process. Technol. 211 (2011) 787-829."
    ),
    "knoepfel_1970": "H. Knoepfel, Pulsed High Magnetic Fields, North-Holland, Amsterdam (1970).",
    "knoepfel_2000": (
        "H. E. Knoepfel, Magnetic Fields: A Comprehensive Theoretical Treatise for Practical Use, "
        "Wiley, New York (2000), ch. 4 (magnetic diffusion into conductors, surface heating)."
    ),
    "herlach_1986": (
        "F. Herlach, Europhysics News 17(2) (1986) 29: under adiabatic skin-effect conditions copper "
        "warms by about 1200 K per (100 T)^2 and boils near 150 T."
    ),
    "herlach_miura_2003": (
        "F. Herlach and N. Miura (eds.), High Magnetic Fields: Science and Technology, Vol. 1, "
        "World Scientific, Singapore (2003)."
    ),
    "jackson_1999": "J. D. Jackson, Classical Electrodynamics, 3rd ed., Wiley, New York (1999), sec. 5.18 (skin depth).",
    "landau_lifshitz_1984": (
        "L. D. Landau, E. M. Lifshitz and L. P. Pitaevskii, Electrodynamics of Continuous Media, "
        "2nd ed., Pergamon, Oxford (1984) (skin effect, field penetration into conductors)."
    ),
    "carslaw_jaeger_1959": (
        "H. S. Carslaw and J. C. Jaeger, Conduction of Heat in Solids, 2nd ed., Oxford University "
        "Press (1959) (erfc solution of the step-boundary diffusion problem)."
    ),
    "crank_nicolson_1947": (
        "J. Crank and P. Nicolson, A practical method for numerical evaluation of solutions of "
        "partial differential equations of the heat-conduction type, Proc. Cambridge Philos. Soc. "
        "43 (1947) 50-67."
    ),
    "fowler_1960": (
        "C. M. Fowler, W. B. Garn and R. S. Caird, Production of very high magnetic fields by "
        "implosion, J. Appl. Phys. 31 (1960) 588-594 (flux compression)."
    ),
    "johnson_1972": "W. Johnson, Impact Strength of Materials, Edward Arnold, London (1972) (rigid-plastic ring and tube dynamics).",
    "taylor_quinney_1934": (
        "G. I. Taylor and H. Quinney, The latent energy remaining in a metal after cold working, "
        "Proc. R. Soc. Lond. A 143 (1934) 307-326."
    ),
    "wheeler_1928": "H. A. Wheeler, Simple inductance formulas for radio coils, Proc. IRE 16 (1928) 1398-1400.",
}

KAPPA_HALF_SINE: float = 1.0
"""Knoepfel surface-heating factor for a sinusoidal half-pulse (exactly 1 for an
exponentially rising field; Knoepfel 1970)."""

KAPPA_STEP: float = 2.4
"""Upper surface-heating factor for a step field: for linear diffusion the Joule
energy per area is sqrt(2)/(2 - sqrt(2)) = 2.41 times the magnetic energy per
area stored in the diffused layer (Knoepfel 2000)."""

STRAIN_RATE_FLOOR: float = 1e-3
"""1/s.  Strain rate passed to the flow-stress function while the tube is at rest."""

DEFAULT_TUBE_STEPS: int = 20_000
"""Default number of fixed time steps of :meth:`TubeCompression.simulate` (dt = t_end/20000)."""

ARREST_STEPS: int = 50
"""Consecutive at-rest steps (after the field peak) required to declare the tube arrested."""

HISTORY_MAX_POINTS: int = 2000
"""Maximum number of samples kept in :class:`TubeHistory`."""

DEFAULT_DIFFUSION_STEPS: int = 2000
"""Default number of Crank-Nicolson steps of :func:`slab_field_diffusion` (dt = t_end/2000)."""

_JOULE_SUBSTEP_K: float = 1.0      # target temperature increment per RK4 sub-step (K)
_JOULE_MAX_SUBSTEPS: int = 5000    # cap on RK4 sub-steps per sample interval
_INV_E: float = math.exp(-1.0)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a real number, got {value!r}") from None
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


def _count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise InputError(f"{name} must be >= {minimum}, got {value}")
    return value


def _callable(name: str, fn: object) -> None:
    if not callable(fn):
        raise InputError(f"{name} must be a callable of time (s) returning field (T)")


def _solid_temperature(m: Material, name: str, value: float) -> float:
    x = _positive(name, value)
    if x >= m.melting_point:
        raise InputError(f"{name} = {x} K is at or above the melting point of {m.key} ({m.melting_point} K)")
    return x


def _shield_update(d_prev: float, b0: float, b1: float, x: float) -> float:
    """Exact update of the shielding difference d = b_out - b_in over one step.

    Solves dd/dt = db_out/dt - d/tau exactly for b_out varying linearly from
    ``b0`` to ``b1`` across the step (x = dt/tau):
    d1 = d0 exp(-x) + (b1 - b0) (1 - exp(-x))/x.  Unconditionally stable; exact
    for piecewise-linear drives, so b_in = b_out - d.
    """
    if x == math.inf:
        return 0.0
    if x <= 0.0:
        return d_prev + (b1 - b0)
    em = math.exp(-x)
    phi = -math.expm1(-x) / x
    return d_prev * em + (b1 - b0) * phi


# ---------------------------------------------------------------------------
# skin depth, time constants, regimes
# ---------------------------------------------------------------------------
def skin_depth(resistivity: float, frequency: float, relative_permeability: float = 1.0) -> float:
    """Electromagnetic skin depth delta = sqrt(rho / (pi f mu0 mu_r)), m.

    Parameters: ``resistivity`` rho (Ohm m), ``frequency`` f (Hz),
    ``relative_permeability`` mu_r (-).  Validity: good conductor
    (rho << 1/(2 pi f eps0)), linear, time-harmonic field; for a pulse use
    f = 1/(4 t_rise).  Ferromagnets saturate above ~2 T, so mu_r = 1 is the
    right value for 63-100 T forming.  Check: Cu (1.68e-8 Ohm m) at 10 kHz ->
    0.652 mm.  Reference: Jackson (1999) sec. 5.18.
    """
    rho = _positive("resistivity", resistivity)
    f = _positive("frequency", frequency)
    mur = _positive("relative_permeability", relative_permeability)
    return math.sqrt(rho / (math.pi * f * MU0 * mur))


def skin_depth_for_material(material: MaterialLike, frequency: float, temperature: float = T_ROOM) -> float:
    """Skin depth (m) of a registry material at ``temperature`` (K).

    delta = sqrt(rho_e(T) / (pi f mu0)) with rho_e(T) from
    ``Material.resistivity_at`` (linear in T, floored at 2 % of rho_293).
    mu_r = 1 (saturated at forming fields).  Validity as :func:`skin_depth`;
    the solid-phase resistivity law is extrapolated above the melting point.
    Reference: Jackson (1999) sec. 5.18.
    """
    m = as_material(material)
    t = _positive("temperature", temperature)
    return skin_depth(m.resistivity_at(t), frequency)


def frequency_for_skin_depth(resistivity: float, depth: float) -> float:
    """Frequency (Hz) at which the skin depth equals ``depth``: f = rho/(pi mu0 delta^2).

    ``resistivity`` in Ohm m, ``depth`` in m; mu_r = 1.  Inverse of
    :func:`skin_depth`; same validity.  Reference: Jackson (1999) sec. 5.18.
    """
    rho = _positive("resistivity", resistivity)
    d = _positive("depth", depth)
    return rho / (math.pi * MU0 * d * d)


def shell_time_constant(resistivity: float, radius: float, thickness: float) -> float:
    """Flux-diffusion time constant of a long thin cylindrical shell, tau = mu0 a h / (2 rho), s.

    ``radius`` a is the shell mid-surface radius (m), ``thickness`` h (m),
    ``resistivity`` rho (Ohm m).  It is the L/R time of the shell as a
    single-turn sheet in an axial field: dB_in/dt = (B_out - B_in)/tau.
    Validity: long shell (length >> a; Wheeler's finite-length factor
    l/(l + 0.9 a) reduces tau otherwise), uniform current through the wall,
    i.e. h <= skin depth.  Reference: Knoepfel (2000) ch. 4.
    """
    rho = _positive("resistivity", resistivity)
    a = _positive("radius", radius)
    h = _positive("thickness", thickness)
    return MU0 * a * h / (2.0 * rho)


def diffusion_time(resistivity: float, thickness: float) -> float:
    """Magnetic diffusion time through a wall, tau_d = mu0 h^2 / rho, s.

    ``resistivity`` rho (Ohm m), ``thickness`` h (m).  The field crosses the
    wall in ~tau_d; a pulse shorter than tau_d sees a "thick" wall.  Note
    shell_time_constant/diffusion_time = a/(2h).  Reference: Knoepfel (2000) ch. 4.
    """
    rho = _positive("resistivity", resistivity)
    h = _positive("thickness", thickness)
    return MU0 * h * h / rho


def shell_regime(resistivity: float, thickness: float, frequency: float) -> str:
    """``"thin"`` if the wall is no thicker than the skin depth (h <= delta), else ``"thick"``.

    delta = sqrt(rho/(pi f mu0)).  In the thin regime the current is nearly
    uniform through the wall and the lumped shell model (tau, uniform-current
    Joule heating) applies; in the thick regime the current crowds into the
    outer skin and the skin-heating estimate applies.  Reference: Knoepfel (2000) ch. 4.
    """
    h = _positive("thickness", thickness)
    return "thin" if h <= skin_depth(resistivity, frequency) else "thick"


def pressure_on_shell(b_out: float, b_in: float) -> float:
    """Net magnetic pressure on a conducting shell, P = (B_out^2 - B_in^2) / (2 mu0), Pa.

    ``b_out`` is the field just outside the wall and ``b_in`` the field inside
    (T).  Positive P pushes the wall inward (compression).  Exact for a long
    cylinder in an axial field (Maxwell stress across the current sheet);
    a static field that has fully diffused (b_in = b_out) exerts no pressure.
    Check: 63.41 T outside, 0 inside -> 1.6 GPa.  Reference: Psyk et al. (2011).
    """
    bo = _finite("b_out", b_out)
    bi = _finite("b_in", b_in)
    return (bo * bo - bi * bi) / (2.0 * MU0)


def sheet_shielding_factor(thickness: float, skin_depth: float) -> float:
    """Quasi-steady pressure-transfer factor of a conducting sheet, 1 - exp(-2h/delta).

    The transmitted field decays roughly as exp(-h/delta), so the pressure
    difference (B_out^2 - B_in^2)/(2 mu0) is the incident pressure times
    1 - exp(-2h/delta).  ``thickness`` h and ``skin_depth`` delta in m.
    Validity: quasi-steady (pulse many skin times long), ignores the phase of
    the transmitted wave and the exact thin-sheet transmission; an
    order-of-magnitude (+-20 %) estimate for h ~ delta.  Reference:
    Knoepfel (2000) ch. 4; Psyk et al. (2011).
    """
    h = _non_negative("thickness", thickness)
    d = _positive("skin_depth", skin_depth)
    return -math.expm1(-2.0 * h / d)


def transmitted_pressure(pressure: float, thickness: float, skin_depth: float) -> float:
    """Pressure delivered to a sheet of thickness h: P (1 - exp(-2h/delta)), Pa.

    ``pressure`` is the incident magnetic pressure B^2/(2 mu0) (Pa).  See
    :func:`sheet_shielding_factor` for validity.  Reference: Psyk et al. (2011).
    """
    p = _non_negative("pressure", pressure)
    return p * sheet_shielding_factor(thickness, skin_depth)


def effective_current_depth(thickness: float, skin_depth: float) -> float:
    """Depth that carries the shielding current, min(h, delta), m.

    For h <= delta the current is spread through the wall; for h > delta it is
    concentrated in ~delta (the field decays as exp(-z/delta)).  Use it for
    current-density and Joule-heating estimates J ~ (B_out - B_in)/(mu0 d_eff).
    Reference: Knoepfel (2000) ch. 4.
    """
    h = _positive("thickness", thickness)
    d = _positive("skin_depth", skin_depth)
    return min(h, d)


# ---------------------------------------------------------------------------
# skin heating
# ---------------------------------------------------------------------------
def surface_heating_energy_density(peak_field: float, kappa: float = 1.0) -> float:
    """Joule heat per unit volume deposited in the eddy-current skin, q = kappa B^2 / (2 mu0), J/m^3.

    ``peak_field`` B (T) is the peak surface field.  ``kappa`` = 1 for a
    sinusoidal half-pulse (exactly 1 for an exponentially rising field), up to
    2.4 for a step field (Knoepfel 1970/2000).  Validity: conductor thick
    compared with the skin depth (h > delta); independent of the pulse length
    to first order because both the dissipated energy per area and the heated
    depth scale with delta.  Order-of-magnitude (factor ~2): kappa = 1 is the
    *lower* estimate for a half-sine -- the exact constant-property surface
    value is 1.64 at the field peak and 2.2-2.4 by the end of the pulse, and
    the rising resistivity of the hot skin brings Cu at 100 T to an effective
    ~1.8 (:func:`slab_field_diffusion`).  Herlach (1986): Cu warms ~1200 K per
    (100 T)^2.  References: Knoepfel (1970); Herlach (1986).
    """
    b = _finite("peak_field", peak_field)
    k = _positive("kappa", kappa)
    return k * b * b / (2.0 * MU0)


def skin_temperature_rise(material: MaterialLike, peak_field: float, kappa: float = 1.0) -> float:
    """Skin-layer temperature rise dT = kappa B^2 / (2 mu0 rho_m c_p), K.

    ``peak_field`` B (T); ``kappa`` as in :func:`surface_heating_energy_density`;
    rho_m (kg/m^3) and c_p (J/(kg K)) from the material.  Equivalently
    dT = kappa P_mag / (rho_m c_p): the skin warms by the magnetic pressure
    divided by the volumetric heat capacity.  Validity: h > delta; adiabatic
    (ignores conduction, which only matters for pulses >> 1 ms) and latent heat;
    constant c_p.  Checks: Cu at 100 T -> ~1160 K; Al-6061 at 63.4 T -> ~660 K.
    References: Knoepfel (1970); Herlach (1986).
    """
    m = as_material(material)
    return surface_heating_energy_density(peak_field, kappa) / (m.density * m.specific_heat)


def skin_melting_field(
    material: MaterialLike,
    initial_temperature: float = T_ROOM,
    kappa: float = 1.0,
    include_latent: bool = False,
) -> float:
    """Peak field at which the eddy-current skin reaches (or completes) melting, B = sqrt(2 mu0 H / kappa), T.

    H = rho_m (c_p (T_m - T_0) + L_f if ``include_latent`` else 0), J/m^3, with
    T_0 = ``initial_temperature`` (K) and L_f the latent heat of fusion (J/kg).
    ``include_latent=False`` gives melting onset, True complete melting.
    Validity as :func:`skin_temperature_rise` (h > delta, adiabatic, constant
    c_p).  Checks: Cu ~96 T, Al-6061 (solidus 855 K) ~58 T, Al-1100 ~62 T.
    Raises InputError if latent heat is requested but not recorded.
    References: Knoepfel (1970); Herlach (1986).
    """
    m = as_material(material)
    t0 = _solid_temperature(m, "initial_temperature", initial_temperature)
    k = _positive("kappa", kappa)
    specific = m.specific_heat * (m.melting_point - t0)
    if include_latent:
        m.require("latent_heat_fusion")
        specific += float(m.latent_heat_fusion)  # type: ignore[arg-type]
    return math.sqrt(2.0 * MU0 * m.density * specific / k)


# ---------------------------------------------------------------------------
# lumped interior field, action, adiabatic Joule heating
# ---------------------------------------------------------------------------
def interior_field(
    b_out: Callable[[float], float], tau: float, t_end: float, n_steps: int
) -> tuple[list[float], list[float]]:
    """Field inside a thin shell, dB_in/dt = (B_out(t) - B_in)/tau, sampled on n_steps+1 times.

    Returns ``(t, b_in)`` lists (s, T).  Each step uses the exact exponential
    update for B_out varying linearly across the step:
    d_{n+1} = d_n e^{-dt/tau} + (B_out,n+1 - B_out,n)(1 - e^{-dt/tau})/(dt/tau),
    B_in = B_out - d; it is unconditionally stable and exact for step-free
    piecewise-linear drives (second-order accurate otherwise).  The interior
    starts in equilibrium with the drive, B_in(0) = B_out(0): a field that was
    steady before t = 0 has fully diffused in.  ``tau`` (s) from
    :func:`shell_time_constant`; validity: thin, static shell (h <= delta).
    Reference: Knoepfel (2000) ch. 4.
    """
    _callable("b_out", b_out)
    tau_ = _positive("tau", tau)
    t_end_ = _positive("t_end", t_end)
    n = _count("n_steps", n_steps, 1)
    dt = t_end_ / n
    x = dt / tau_
    b_prev = _finite("b_out(0)", b_out(0.0))
    ts = [0.0]
    bins = [b_prev]
    d = 0.0
    for k in range(1, n + 1):
        t = t_end_ if k == n else k * dt
        b_now = _finite(f"b_out({t})", b_out(t))
        d = _shield_update(d, b_prev, b_now, x)
        ts.append(t)
        bins.append(b_now - d)
        b_prev = b_now
    return ts, bins


def _check_time_series(name: str, values: Sequence[float], t: Sequence[float]) -> tuple[list[float], list[float]]:
    if len(values) != len(t):
        raise InputError(f"{name} and t must have the same length ({len(values)} != {len(t)})")
    if len(t) == 0:
        raise InputError(f"{name} and t must not be empty")
    vs = [_finite(name, v) for v in values]
    ts = [_finite("t", x) for x in t]
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1]:
            raise InputError("t must be non-decreasing")
    return vs, ts


def action_integral(current: Sequence[float], t: Sequence[float]) -> float:
    """Action integral G = integral of I^2 dt (trapezoid rule), A^2 s.

    ``current`` I (A) sampled at ``t`` (s, non-decreasing).  With current
    *density* samples (A/m^2) the result is the specific action (A^2 s/m^4)
    that sets the adiabatic temperature of a conductor
    (:func:`specific_action_to_temperature`).  Reference: Knoepfel (1970).
    """
    vs, ts = _check_time_series("current", current, t)
    return trapz([v * v for v in vs], ts)


@dataclass(frozen=True, kw_only=True)
class JouleResult(ResultMixin):
    """Adiabatic Joule heating of a conductor carrying a current-density waveform."""

    material: str
    initial_temperature: float = unit("K")
    final_temperature: float = unit("K")
    delta_t: float = unit("K")
    action: float = unit("A^2 s/m^4")
    melted: bool
    time_to_melt: float | None = unit("s")


def _time_for_action(a: float, b: float, dt: float, target: float) -> float:
    """Time s in [0, dt] at which the cumulative integral of a linear J^2 reaches ``target``."""
    if dt <= 0.0 or target <= 0.0:
        return 0.0
    c2 = (b - a) / (2.0 * dt)
    c1 = a
    if abs(c2) * dt * dt <= 1e-12 * max(c1 * dt, 1e-300):
        s = target / c1 if c1 > 0.0 else dt
    else:
        roots = [r for r in solve_quadratic(c2, c1, -target) if -1e-12 * dt <= r <= dt * (1.0 + 1e-12)]
        s = min(roots) if roots else dt
    return min(max(s, 0.0), dt)


def adiabatic_joule_heating(
    current_density: Sequence[float],
    t: Sequence[float],
    material: MaterialLike,
    initial_temperature: float = T_ROOM,
) -> JouleResult:
    """Adiabatic heating of a conductor by a current-density waveform, dT = rho_e(T) J^2 dt / (rho_m c_p).

    ``current_density`` J (A/m^2) sampled at ``t`` (s).  Because J^2 dt is the
    only time dependence, the explicit march integrates dT/dA = rho_e(T)/(rho_m c_p)
    in the specific action A = integral J^2 dt (trapezoidal J^2 between samples)
    with RK4 sub-steps of ~1 K, so coarse sampling does not degrade accuracy.
    rho_e(T) from ``Material.resistivity_at``; constant c_p.  Heating stops at
    the melting point: ``melted`` and ``time_to_melt`` record the event and
    ``action`` is always the full integral.  Validity: uniform current (wire
    thinner than its skin depth), no conduction, no latent heat, solid-phase
    properties.  Reference: Knoepfel (1970) (action integral of conductors).
    """
    m = as_material(material)
    t0 = _solid_temperature(m, "initial_temperature", initial_temperature)
    js, ts = _check_time_series("current_density", current_density, t)
    rho_c = m.density * m.specific_heat
    tm = m.melting_point

    def rate(temp: float) -> float:
        return m.resistivity_at(temp) / rho_c

    temp = t0
    melted = False
    t_melt: float | None = None
    total = 0.0
    for i in range(len(ts) - 1):
        dti = ts[i + 1] - ts[i]
        a = js[i] * js[i]
        b = js[i + 1] * js[i + 1]
        d_a = 0.5 * (a + b) * dti
        total += d_a
        if melted or d_a <= 0.0:
            continue
        n_sub = int(min(_JOULE_MAX_SUBSTEPS, max(1, math.ceil(rate(temp) * d_a / _JOULE_SUBSTEP_K))))
        h = d_a / n_sub
        done = 0.0
        for _ in range(n_sub):
            k1 = rate(temp)
            k2 = rate(temp + 0.5 * h * k1)
            k3 = rate(temp + 0.5 * h * k2)
            k4 = rate(temp + h * k3)
            new = temp + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if new >= tm:
                frac = (tm - temp) / (new - temp) if new > temp else 0.0
                t_melt = ts[i] + _time_for_action(a, b, dti, done + frac * h)
                temp = tm
                melted = True
                break
            temp = new
            done += h
    return JouleResult(
        material=m.key,
        initial_temperature=t0,
        final_temperature=temp,
        delta_t=temp - t0,
        action=total,
        melted=melted,
        time_to_melt=t_melt,
    )


def specific_action_to_temperature(
    material: MaterialLike, temperature_from: float, temperature_to: float, n: int = 400
) -> float:
    """Specific action needed to heat a conductor adiabatically, integral rho_m c_p / rho_e(T) dT, A^2 s/m^4.

    Trapezoid rule on ``n`` intervals between ``temperature_from`` and
    ``temperature_to`` (K); rho_e(T) from ``Material.resistivity_at``.  Divide
    by J^2 to get the allowed pulse duration at current density J.  Validity:
    adiabatic, uniform current, constant c_p; above the melting point the
    solid-phase law is extrapolated and latent heat is ignored.  Check: Cu
    293 -> 1356 K gives ~8.5e16 A^2 s/m^4 (order check 8e16-1.2e17).
    Reference: Knoepfel (1970).
    """
    m = as_material(material)
    t1 = _positive("temperature_from", temperature_from)
    t2 = _positive("temperature_to", temperature_to)
    if t2 < t1:
        raise InputError("temperature_to must be >= temperature_from")
    nn = _count("n", n, 1)
    if t2 == t1:
        return 0.0
    rho_c = m.density * m.specific_heat
    temps = [t1 + (t2 - t1) * i / nn for i in range(nn + 1)]
    return trapz([rho_c / m.resistivity_at(x) for x in temps], temps)


# ---------------------------------------------------------------------------
# 1-D nonlinear magnetic diffusion into a slab
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class DiffusionResult(ResultMixin):
    """Result of :func:`slab_field_diffusion`.

    ``z``/``final_temperature_profile``: the surface (z = 0, heated at the rate
    E_s^2/rho_e(T_s) with E = rho_e J extrapolated quadratically from the first
    cell faces) followed by the cell centres (i + 1/2) dz.  Temperatures never
    fall (no conduction), so the final profile is also the peak temperature at
    each depth.
    """

    material: str
    thickness: float = unit("m")
    t_end: float = unit("s")
    surface_peak_temperature: float = unit("K")
    max_temperature: float = unit("K")
    melted: bool
    melt_depth: float = unit("m")
    penetration_depth: float = unit("m")
    joule_energy_per_area: float = unit("J/m^2")
    n_nodes: int
    n_steps: int
    z: tuple[float, ...] = unit("m")
    final_temperature_profile: tuple[float, ...] = unit("K")


def _penetration_depth(b: Sequence[float], dz: float) -> float:
    """Depth where B falls to B(0)/e (linear interpolation between nodes)."""
    b0 = b[0]
    if b0 == 0.0:
        return 0.0
    prev = 1.0
    for i in range(1, len(b)):
        ratio = b[i] / b0
        if ratio <= _INV_E:
            frac = (prev - _INV_E) / (prev - ratio) if prev != ratio else 0.0
            return (i - 1 + frac) * dz
        prev = ratio
    return (len(b) - 1) * dz


def slab_field_diffusion(
    material: MaterialLike,
    surface_field: Callable[[float], float],
    thickness: float,
    t_end: float,
    n_nodes: int = 60,
    n_steps: int | None = None,
    initial_temperature: float = T_ROOM,
) -> DiffusionResult:
    """Nonlinear 1-D magnetic diffusion into a heating slab, dB/dt = d/dz((rho_e(T)/mu0) dB/dz).

    Boundary conditions B(0, t) = ``surface_field(t)`` (T) and B(h, t) = 0
    (shielded back face); the interior starts field-free.  Crank-Nicolson in
    time on a uniform staggered grid: B on ``n_nodes`` nodes, temperature and
    resistivity on the n_nodes - 1 cells between them; the tridiagonal system
    is solved with ``_numerics.solve_tridiagonal`` and rho_e is lagged one step.
    Current density J = -(1/mu0) dB/dz at the half step; the cell heating
    rho_e J^2 dt/(rho_m c_p) uses the same rho_e as the field step, so the Joule
    energy equals the discrete Poynting input minus the stored field energy.
    The surface (z = 0) temperature is tracked separately, heated at
    E_s^2/rho_e(T_s) with the electric field E = rho_e J extrapolated
    quadratically to the surface (E stays smooth where rho_e(T) is steep, so
    this converges with ~60 nodes).  ``n_steps`` defaults to 2000
    (dt = t_end/2000).  ``penetration_depth`` is the depth where B = B_s/e at
    the (last) time of the peak |surface field|; for a step and constant
    properties it is 0.90 sqrt(2 rho t/mu0) (erfc profile).

    Surface heating vs :func:`skin_temperature_rise`: for constant properties
    the exact surface heating of a half-sine is 1.64 B^2/(2 mu0) at the field
    peak, 2.19 at the end of the pulse and 2.42 after it (Duhamel solution;
    reproduced by this solver), i.e. kappa = 1 is a lower estimate.  For a
    field step the surface heating diverges logarithmically,
    dT(z) = B^2 E1(z^2/(2 D t))/(mu0 pi rho_m c_p), so the surface value is
    grid-dependent; use a finite rise time.  Validity: adiabatic (thermal
    diffusion length << magnetic: true for Cu over < 1 ms), solid-phase
    properties extrapolated above the melting point, latent heat ignored (so
    ``melt_depth`` marks where T_m was reached, an upper bound on the fully
    molten layer), resistivity linear in T, mu_r = 1.  References: Knoepfel
    (2000) ch. 4; Crank & Nicolson (1947); Carslaw & Jaeger (1959).
    """
    m = as_material(material)
    _callable("surface_field", surface_field)
    h = _positive("thickness", thickness)
    t_end_ = _positive("t_end", t_end)
    nn = _count("n_nodes", n_nodes, 3)
    steps = DEFAULT_DIFFUSION_STEPS if n_steps is None else _count("n_steps", n_steps, 1)
    t0 = _solid_temperature(m, "initial_temperature", initial_temperature)

    dz = h / (nn - 1)
    dt = t_end_ / steps
    rho_c = m.density * m.specific_heat
    n_cells = nn - 1
    n_unknown = nn - 2
    coef = dt / (2.0 * MU0 * dz * dz)
    jfac = 1.0 / (2.0 * MU0 * dz)

    b = [0.0] * nn
    b[0] = _finite("surface_field(0)", surface_field(0.0))
    temps = [t0] * n_cells
    t_surf = t0
    energy = 0.0
    peak_bs = abs(b[0])
    penetration = _penetration_depth(b, dz) if b[0] != 0.0 else 0.0

    lower = [0.0] * n_unknown
    diag = [0.0] * n_unknown
    upper = [0.0] * n_unknown
    rhs = [0.0] * n_unknown
    for step in range(1, steps + 1):
        t_new = t_end_ if step == steps else step * dt
        bs_new = _finite(f"surface_field({t_new})", surface_field(t_new))
        rhos = [m.resistivity_at(x) for x in temps]
        lam = [r * coef for r in rhos]
        for j in range(n_unknown):
            i = j + 1
            lw = lam[i - 1]
            up = lam[i]
            lower[j] = -lw
            diag[j] = 1.0 + lw + up
            upper[j] = -up
            rhs[j] = lw * b[i - 1] + (1.0 - lw - up) * b[i] + up * b[i + 1]
        rhs[0] += lam[0] * bs_new
        interior = solve_tridiagonal(lower, diag, upper, rhs)
        b_new = [bs_new] + interior + [0.0]
        e_face = [0.0] * n_cells
        for f in range(n_cells):
            jf = ((b_new[f] + b[f]) - (b_new[f + 1] + b[f + 1])) * jfac
            q = rhos[f] * jf * jf * dt
            temps[f] += q / rho_c
            energy += q * dz
            e_face[f] = rhos[f] * jf
        # surface (z = 0): extrapolate the electric field E = rho_e J (smooth even where rho_e(T)
        # varies steeply) from the first cell faces, then heat at the surface rate E^2 / rho_e(T_s)
        if n_cells >= 3:
            e_s = (15.0 * e_face[0] - 10.0 * e_face[1] + 3.0 * e_face[2]) / 8.0
        else:
            e_s = 1.5 * e_face[0] - 0.5 * e_face[1]
        t_surf += e_s * e_s / m.resistivity_at(t_surf) * dt / rho_c
        b = b_new
        abs_bs = abs(bs_new)
        if abs_bs > 0.0 and abs_bs >= peak_bs * (1.0 - 1e-12):
            peak_bs = max(peak_bs, abs_bs)
            penetration = _penetration_depth(b, dz)

    z = (0.0,) + tuple((i + 0.5) * dz for i in range(n_cells))
    profile = (t_surf,) + tuple(temps)
    tm = m.melting_point
    melt_depth = 0.0
    if profile[0] >= tm:
        melt_depth = h
        for k in range(1, len(profile)):
            if profile[k] < tm:
                frac = (profile[k - 1] - tm) / (profile[k - 1] - profile[k])
                melt_depth = z[k - 1] + frac * (z[k] - z[k - 1])
                break
    max_t = max(profile)
    return DiffusionResult(
        material=m.key,
        thickness=h,
        t_end=t_end_,
        surface_peak_temperature=t_surf,
        max_temperature=max_t,
        melted=max_t >= tm,
        melt_depth=melt_depth,
        penetration_depth=penetration,
        joule_energy_per_area=energy,
        n_nodes=nn,
        n_steps=steps,
        z=z,
        final_temperature_profile=profile,
    )


# ---------------------------------------------------------------------------
# drive waveforms
# ---------------------------------------------------------------------------
def damped_sine_field(
    peak_field: float, frequency: float, damping_time: float | None = None
) -> Callable[[float], float]:
    """Damped-sine drive normalised so that max_t B(t) = ``peak_field`` (T).

    B(t) = B_pk e^{-t/tau} sin(omega t)/N for t >= 0, 0 for t < 0, with
    omega = 2 pi f, first peak at t_p = atan(omega tau)/omega and
    N = e^{-t_p/tau} sin(omega t_p); undamped (``damping_time`` None):
    B = B_pk sin(omega t), t_p = 1/(4f).  Represents an underdamped RLC
    discharge into a coil (no crowbar).  ``frequency`` in Hz, ``damping_time``
    tau = 2L/R in s.  Reference: Psyk et al. (2011).
    """
    b_pk = _non_negative("peak_field", peak_field)
    f = _positive("frequency", frequency)
    omega = 2.0 * math.pi * f
    if damping_time is None:
        def field(t: float) -> float:
            return b_pk * math.sin(omega * t) if t >= 0.0 else 0.0

        return field
    tau = _positive("damping_time", damping_time)
    t_p = math.atan(omega * tau) / omega
    norm = math.exp(-t_p / tau) * math.sin(omega * t_p)
    amp = b_pk / norm

    def damped(t: float) -> float:
        return amp * math.exp(-t / tau) * math.sin(omega * t) if t >= 0.0 else 0.0

    return damped


def half_sine_field(peak_field: float, half_period: float) -> Callable[[float], float]:
    """Single half-sine pulse B(t) = B_pk sin(pi t / T_half) for 0 <= t <= T_half, else 0 (T).

    Idealised crowbarred/clipped discharge: peak at T_half/2, equivalent
    frequency 1/(2 T_half).  ``peak_field`` in T, ``half_period`` in s.
    Reference: Psyk et al. (2011).
    """
    b_pk = _non_negative("peak_field", peak_field)
    t_half = _positive("half_period", half_period)
    w = math.pi / t_half

    def field(t: float) -> float:
        return b_pk * math.sin(w * t) if 0.0 <= t <= t_half else 0.0

    return field


# ---------------------------------------------------------------------------
# pulse-frequency recommendation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class PulseRecommendation(ResultMixin):
    """Drive frequency that places the skin depth at wall/depth_ratio."""

    material: str
    wall: float = unit("m")
    frequency: float = unit("Hz")
    quarter_period: float = unit("s")
    skin_depth: float = unit("m")
    diffusion_time: float = unit("s")
    regime: str
    depth_ratio: float
    note: str


def pulse_frequency_recommendation(
    material: MaterialLike, wall: float, depth_ratio: float = 3.0, temperature: float = T_ROOM
) -> PulseRecommendation:
    """Drive frequency f = rho_e(T) / (pi mu0 delta^2) with delta = wall/depth_ratio.

    ``wall`` (m) is the workpiece wall, ``depth_ratio`` = h/delta (-),
    ``temperature`` (K) sets rho_e.  Quarter period 1/(4f) is the field rise
    time.  delta = h/3 shields well (transmitted pressure fraction
    1 - e^{-6} ~ 1) but places the wall in the thick regime: use the skin-heating
    estimate for the surface and the lumped shell time constant only for the
    late-time flux decay.  Example: Cu, 2 mm wall -> ~9.7 kHz, quarter period
    ~26 us.  Reference: Psyk et al. (2011); Knoepfel (2000) ch. 4.
    """
    m = as_material(material)
    h = _positive("wall", wall)
    ratio = _positive("depth_ratio", depth_ratio)
    temp = _positive("temperature", temperature)
    rho = m.resistivity_at(temp)
    delta = h / ratio
    f = frequency_for_skin_depth(rho, delta)
    # h <= delta by construction when depth_ratio <= 1; recomputing delta from f (shell_regime) would
    # misclassify depth_ratio == 1 through round-off
    regime = "thin" if h <= delta else "thick"
    if regime == "thick":
        note = (
            f"delta = h/{ratio:g} shields well (pressure transfer 1 - exp(-2h/delta) = "
            f"{sheet_shielding_factor(h, delta):.3f}) but places the wall in the thick regime: "
            "use skin heating (skin_temperature_rise) for the surface and the lumped shell time "
            "constant only for the late-time flux decay."
        )
    else:
        note = (
            f"delta >= h: thin regime, current nearly uniform through the wall; pressure transfer "
            f"only {sheet_shielding_factor(h, delta):.3f} of B^2/(2 mu0) (field leaks through the wall)."
        )
    return PulseRecommendation(
        material=m.key,
        wall=h,
        frequency=f,
        quarter_period=0.25 / f,
        skin_depth=delta,
        diffusion_time=diffusion_time(rho, h),
        regime=regime,
        depth_ratio=ratio,
        note=note,
    )


# ---------------------------------------------------------------------------
# rigid-plastic tube compression
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class TubeSummary(ResultMixin):
    """Scalar outcome of :meth:`TubeCompression.simulate`.

    ``final_radius`` is the wall mid-surface radius; ``peak_velocity`` the peak
    inward speed |v|; ``transfer_efficiency`` = peak net pressure /
    (peak B_out^2/(2 mu0)); ``plastic_work_per_length`` is per metre of the
    original tube length; ``stopped_by`` is one of "arrested", "mandrel",
    "closed", "melted", "t_end"; ``time_to_stop`` is the last time the wall
    moved (inf if still moving at t_end).  ``skin_temperature_rise_estimate``
    is :func:`skin_temperature_rise` (kappa = 1) of the outermost conductor for
    the peak field swing |B_out(t) - B_out(0)|, and ``regime`` ("thin" or
    "thick") compares the wall with the skin depth at f = 1/(4 t_swing): eddy
    currents follow the change of the field, so both ignore a steady
    background (zero rise and "thin" for a static drive).
    """

    final_radius: float = unit("m")
    final_inner_radius: float = unit("m")
    final_wall: float = unit("m")
    final_strain: float
    peak_velocity: float = unit("m/s")
    peak_strain_rate: float = unit("1/s")
    peak_pressure: float = unit("Pa")
    peak_field: float = unit("T")
    transfer_efficiency: float
    plastic_work_per_length: float = unit("J/m")
    bulk_joule_temperature_rise: float = unit("K")
    plastic_temperature_rise: float = unit("K")
    final_temperature: float = unit("K")
    skin_temperature_rise_estimate: float = unit("K")
    regime: str
    stopped_by: str
    time_to_stop: float = unit("s")
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TubeHistory(ResultMixin):
    """Columnar time history (<= 2000 samples) of :meth:`TubeCompression.simulate`.

    ``radius`` is the mid-surface radius, ``velocity`` is signed (negative =
    inward), ``strain`` the equivalent plastic strain, ``p_net`` =
    (B_out^2 - B_in^2)/(2 mu0), ``temperature`` the workpiece bulk temperature.
    """

    t: tuple[float, ...] = unit("s")
    radius: tuple[float, ...] = unit("m")
    velocity: tuple[float, ...] = unit("m/s")
    strain: tuple[float, ...]
    strain_rate: tuple[float, ...] = unit("1/s")
    p_net: tuple[float, ...] = unit("Pa")
    b_out: tuple[float, ...] = unit("T")
    b_in: tuple[float, ...] = unit("T")
    temperature: tuple[float, ...] = unit("K")
    flow_stress: tuple[float, ...] = unit("Pa")


@dataclass(frozen=True, kw_only=True)
class TubeResult(ResultMixin):
    """Summary and history of a tube-compression simulation."""

    material: str
    summary: TubeSummary
    history: TubeHistory


@dataclass(frozen=True)
class TubeCompression(ResultMixin):
    """Electromagnetic compression of a long conducting tube (rigid-plastic thin shell).

    Geometry: ``outer_radius`` and ``wall`` (m) of the tube; the state radius r
    is the wall mid-surface, r0 = outer_radius - wall/2.  ``length`` (m,
    optional) only triggers a warning when the tube is short enough for the
    long-shell time constant to be overestimated by more than 25 % (Wheeler's
    factor (l + 0.9 r)/l).  ``mandrel_radius`` (m) stops the bore.
    ``temperature`` (K) is the initial bulk temperature.  ``end_condition``:
    "plane_strain" (length fixed: h = h0 r0/r, equivalent strain
    (2/sqrt3) ln(r0/r), yield hoop stress (2/sqrt3) sigma) or "free_ends"
    (uniaxial hoop: h = h0 sqrt(r0/r), k = 1).  An optional sacrificial
    ``driver_material``/``driver_thickness`` sheet outside the tube adds sheet
    conductance and mass (its strength is neglected).  ``taylor_quinney`` beta is
    the fraction of plastic work converted to heat.

    Equations (per unit wall area, k = 2/sqrt3 or 1):
    m dv/dt = -(p_net - P_y) while moving inward or while p_net > P_y, else 0;
    P_y = k sigma h / r, m = rho h + rho_d h_d; no outward motion (rigid-plastic,
    no elastic rebound).  Interior field of the lumped shell:
    dB_in/dt = (B_out - B_in)/tau - 2 (dr/dt / r) B_in with tau = mu0 r G / 2 and
    G = h/rho_e(T) + h_d/rho_e,d(T_d) (exact exponential update plus flux
    compression B_in r^2 = const).  Joule power per area K^2/G with
    K = (B_out - B_in)/mu0, shared between tube and driver in proportion to
    their conductances (uniform current through each sheet); plastic heating
    beta sigma d(eps)/(rho c_p).  Validity: long thin shell, uniform axial
    field, current uniform through the wall (lumped model; in the thick regime
    the skin heating is reported separately), no buckling.  References:
    Psyk et al. (2011); Johnson (1972); Knoepfel (2000); Fowler et al. (1960);
    Taylor & Quinney (1934).
    """

    material: MaterialLike
    outer_radius: float
    wall: float
    length: float | None = None
    mandrel_radius: float | None = None
    temperature: float = T_ROOM
    end_condition: str = "plane_strain"
    driver_material: MaterialLike | None = None
    driver_thickness: float = 0.0
    taylor_quinney: float = 0.9

    def __post_init__(self) -> None:
        m = as_material(self.material)
        ro = _positive("outer_radius", self.outer_radius)
        h = _positive("wall", self.wall)
        if h >= ro:
            raise InputError(f"wall ({h} m) must be smaller than outer_radius ({ro} m)")
        if self.length is not None:
            _positive("length", self.length)
        if self.mandrel_radius is not None:
            rm = _non_negative("mandrel_radius", self.mandrel_radius)
            if rm >= ro - h:
                raise InputError(f"mandrel_radius ({rm} m) must be smaller than the inner radius ({ro - h} m)")
        _solid_temperature(m, "temperature", self.temperature)
        if self.end_condition not in ("plane_strain", "free_ends"):
            raise InputError(f"end_condition must be 'plane_strain' or 'free_ends', got {self.end_condition!r}")
        hd = _non_negative("driver_thickness", self.driver_thickness)
        if hd > 0.0:
            if self.driver_material is None:
                raise InputError("driver_thickness > 0 requires a driver_material")
            dm = as_material(self.driver_material)
            if self.temperature >= dm.melting_point:
                raise InputError(f"temperature is above the melting point of the driver {dm.key}")
        elif self.driver_material is not None:
            as_material(self.driver_material)
        beta = _finite("taylor_quinney", self.taylor_quinney)
        if not 0.0 <= beta <= 1.0:
            raise InputError(f"taylor_quinney must be in [0, 1], got {beta}")

    @property
    def inner_radius(self) -> float:
        """Initial bore radius outer_radius - wall, m."""
        return self.outer_radius - self.wall

    @property
    def mean_radius(self) -> float:
        """Initial wall mid-surface radius outer_radius - wall/2, m."""
        return self.outer_radius - 0.5 * self.wall

    def simulate(
        self,
        b_out: Callable[[float], float],
        t_end: float,
        dt: float | None = None,
        flow_stress_fn: Callable[[float, float, float], float] | None = None,
        max_steps: int = 400_000,
    ) -> TubeResult:
        """Integrate the tube response to the outside field ``b_out(t)`` (T) up to ``t_end`` (s).

        Fixed step ``dt`` (default t_end/20000).  ``flow_stress_fn(strain,
        strain_rate, temperature) -> Pa`` defaults to
        ``functools.partial(plasticity.flow_stress, material)``; it is called
        once per step at the start-of-step state with the strain rate floored at
        1e-3 1/s.  Each step: (1) exact exponential update of the shielding
        difference d = B_out - B_in for B_out linear across the step, with tau
        from the start-of-step radius, wall and temperatures; (2) RK4 on (r, v)
        with B_in interpolated in time and compressed with the stage radius;
        no outward motion (v > 0 -> 0, r never increases); mandrel or bore
        closure clamps the inner radius and ends the run; (3) flux compression
        B_in <- B_in (r_prev/r)^2; (4) Joule and plastic heating.  The interior
        starts in equilibrium, B_in(0) = B_out(0), so a steady background field
        exerts no pressure and induces no eddy current (the regime and the skin
        estimate use the swing |B_out(t) - B_out(0)|).  A background trapped in
        the bore is flux-compressed as the tube collapses and opposes the
        implosion.  The run ends as "arrested" once the tube has been
        at rest with p_net <= P_y for 50 consecutive steps after the field peak
        and the remaining drive can no longer exceed P_y (max future
        B_out^2/(2 mu0) <= P_y); as "melted" if the bulk reaches the melting
        point.  Heating is integrated up to the stop time.  Raises InputError if
        t_end/dt exceeds ``max_steps``.  References: Psyk et al. (2011);
        Johnson (1972).
        """
        _callable("b_out", b_out)
        t_end_ = _positive("t_end", t_end)
        step = t_end_ / DEFAULT_TUBE_STEPS if dt is None else _positive("dt", dt)
        if step > t_end_:
            raise InputError("dt must not exceed t_end")
        limit = _count("max_steps", max_steps, 1)
        n_total = max(1, math.ceil(t_end_ / step - 1e-9))
        if n_total > limit:
            raise InputError(f"t_end/dt = {n_total} steps exceeds max_steps = {limit}; increase dt")
        if flow_stress_fn is not None and not callable(flow_stress_fn):
            raise InputError("flow_stress_fn must be callable (strain, strain_rate, temperature) -> Pa")

        m = as_material(self.material)
        drv = as_material(self.driver_material) if self.driver_thickness > 0.0 else None
        if flow_stress_fn is None:
            from . import plasticity  # lazy: plasticity sits left of eddy in the module DAG

            flow_stress_fn = functools.partial(plasticity.flow_stress, m)

        # drive sampled on the step grid (reused by the integrator)
        bo = [_finite(f"b_out({k * step})", b_out(k * step)) for k in range(n_total + 1)]
        future = [0.0] * (n_total + 1)
        running = 0.0
        for k in range(n_total, -1, -1):
            running = max(running, bo[k] * bo[k])
            future[k] = running
        k_peak = max(range(n_total + 1), key=lambda k: abs(bo[k]))
        t_peak_field = k_peak * step
        # eddy currents respond to the change of B_out from its initial, fully diffused value (the swing),
        # not to the field itself: a steady background induces no current, pressure or skin heating
        k_swing = max(range(n_total + 1), key=lambda k: abs(bo[k] - bo[0]))
        peak_swing = abs(bo[k_swing] - bo[0])

        plane = self.end_condition == "plane_strain"
        kf = 2.0 / math.sqrt(3.0) if plane else 1.0
        h0 = self.wall
        r0 = self.mean_radius
        hd0 = self.driver_thickness if drv is not None else 0.0
        rho_w = m.density
        cp_w = m.specific_heat
        rho_d = drv.density if drv is not None else 0.0
        cp_d = drv.specific_heat if drv is not None else 0.0
        beta = self.taylor_quinney
        vol_per_length = 2.0 * math.pi * r0 * h0
        tm = m.melting_point
        rm = self.mandrel_radius

        def thick(rr: float) -> float:
            return h0 * r0 / rr if plane else h0 * math.sqrt(r0 / rr)

        def drv_thick(rr: float) -> float:
            if hd0 == 0.0:
                return 0.0
            return hd0 * r0 / rr if plane else hd0 * math.sqrt(r0 / rr)

        def bore(rr: float) -> float:
            return rr - 0.5 * thick(rr)

        stop_radius = rm if rm is not None else 0.0

        def contact_radius(r_prev: float) -> float:
            """Mid-surface radius at which the bore r - h(r)/2 equals the stop radius."""
            if plane:  # r - h0 r0/(2r) = s  ->  r^2 - s r - h0 r0/2 = 0
                return 0.5 * (stop_radius + math.sqrt(stop_radius * stop_radius + 2.0 * h0 * r0))
            lo = max(stop_radius, 1e-9 * r0)  # bore(lo) < stop_radius; bore(r_prev) > stop_radius
            return bisect(lambda rr: bore(rr) - stop_radius, lo, r_prev, tol=1e-14)
        warnings: list[str] = []

        r = r0
        v = 0.0
        b_in = bo[0]
        d = 0.0
        temp = self.temperature
        temp_d = self.temperature
        joule_rise = 0.0
        plastic_rise = 0.0
        plastic_work = 0.0
        peak_v = 0.0
        peak_rate = 0.0
        peak_p = 0.0
        peak_b = abs(bo[0])
        t_last_motion = 0.0
        moved = False
        rest_count = 0
        stopped_by = "t_end"
        stop_time = None
        expansion_flagged = False
        driver_melt_flagged = False

        stride = max(1, math.ceil(n_total / (HISTORY_MAX_POINTS - 2)))
        hist: dict[str, list[float]] = {key: [] for key in (
            "t", "radius", "velocity", "strain", "strain_rate", "p_net", "b_out", "b_in", "temperature", "flow_stress")}

        def record(tt: float, rr: float, vv: float, bout: float, bin_: float, dd: float, te: float, sig: float) -> None:
            hist["t"].append(tt)
            hist["radius"].append(rr)
            hist["velocity"].append(vv)
            hist["strain"].append(kf * math.log(r0 / rr))
            hist["strain_rate"].append(kf * abs(vv) / rr)
            hist["p_net"].append(dd * (bout + bin_) / (2.0 * MU0))
            hist["b_out"].append(bout)
            hist["b_in"].append(bin_)
            hist["temperature"].append(te)
            hist["flow_stress"].append(sig)

        def flow(eps: float, rate: float, te: float) -> float:
            s = flow_stress_fn(eps, max(rate, STRAIN_RATE_FLOOR), te)  # type: ignore[misc]
            s = _finite("flow_stress_fn result", s)
            if s < 0.0:
                raise InputError(f"flow_stress_fn returned a negative stress ({s} Pa)")
            return s

        sigma = flow(0.0, 0.0, temp)
        record(0.0, r, v, bo[0], b_in, d, temp, sigma)
        last_recorded = 0
        n_done = 0

        for n in range(n_total):
            t_a = n * step
            t_b = t_a + step
            b_a = bo[n]
            b_b = bo[n + 1]
            if temp >= tm:
                stopped_by = "melted"
                stop_time = t_a
                warnings.append(f"workpiece bulk reached its melting point ({tm:g} K) at t = {t_a:.3g} s; run stopped")
                break
            h = thick(r)
            hd = drv_thick(r)
            eps = kf * math.log(r0 / r)
            sigma = flow(eps, kf * abs(v) / r, temp)

            # (1) field diffusion through the wall (lumped shell, exact exponential update)
            g_w = h / m.resistivity_at(temp)
            g_d = hd / drv.resistivity_at(temp_d) if drv is not None else 0.0
            g = g_w + g_d
            tau = MU0 * r * g / 2.0
            x = step / tau if tau > 0.0 else math.inf
            d_star = _shield_update(d, b_a, b_b, x)
            bin_star = b_b - d_star

            # (2) rigid-plastic motion, RK4 on (r, v)
            b_mid = _finite(f"b_out({t_a + 0.5 * step})", b_out(t_a + 0.5 * step))
            b_in_a = b_in
            r_a = r

            def accel(frac: float, bout: float, rr: float, vv: float) -> float:
                if rr <= 0.0:
                    return 0.0
                bi = (b_in_a + (bin_star - b_in_a) * frac) * (r_a / rr) ** 2
                p = (bout * bout - bi * bi) / (2.0 * MU0)
                hh = thick(rr)
                p_y = kf * sigma * hh / rr
                if vv < 0.0 or p > p_y:
                    return -(p - p_y) / (rho_w * hh + rho_d * drv_thick(rr))
                return 0.0

            k1r = v
            k1v = accel(0.0, b_a, r, v)
            k2r = v + 0.5 * step * k1v
            k2v = accel(0.5, b_mid, r + 0.5 * step * k1r, k2r)
            k3r = v + 0.5 * step * k2v
            k3v = accel(0.5, b_mid, r + 0.5 * step * k2r, k3r)
            k4r = v + step * k3v
            k4v = accel(1.0, b_b, r + step * k3r, k4r)
            r_new = r + step * (k1r + 2.0 * k2r + 2.0 * k3r + k4r) / 6.0
            v_new = v + step * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
            if v_new > 0.0:
                v_new = 0.0  # rigid-plastic: no elastic rebound, no outward flow
            if r_new > r:
                r_new = r
            contact = None
            if r_new <= 0.0 or bore(r_new) <= stop_radius:
                r_new = contact_radius(r)
                v_new = 0.0
                contact = "mandrel" if rm is not None else "closed"

            # (3) flux compression of the trapped field, B_in r^2 = const
            c = (r / r_new) ** 2
            b_in_new = c * bin_star
            d_new = d_star + (1.0 - c) * bin_star

            # (4) heating: Joule (lumped sheets) and plastic work
            q = 0.5 * (d * d + d_new * d_new) / (MU0 * MU0 * g) * step if g > 0.0 else 0.0
            d_tj = (g_w / g) * q / (rho_w * h * cp_w) if g > 0.0 else 0.0
            if drv is not None and g > 0.0:
                temp_d += (g_d / g) * q / (rho_d * hd * cp_d)
                if temp_d >= drv.melting_point and not driver_melt_flagged:
                    driver_melt_flagged = True
                    warnings.append(
                        f"driver sheet ({drv.key}) reaches its melting point (lumped bulk) at t = {t_b:.3g} s")
            d_eps = kf * math.log(r / r_new)
            d_tp = beta * sigma * d_eps / (rho_w * cp_w)
            plastic_work += sigma * d_eps * vol_per_length
            joule_rise += d_tj
            plastic_rise += d_tp
            temp += d_tj + d_tp
            if r_new < r or v_new != 0.0:
                moved = True
                t_last_motion = t_b

            r, v, b_in, d = r_new, v_new, b_in_new, d_new
            n_done = n + 1
            p_now = d * (b_b + b_in) / (2.0 * MU0)
            peak_v = max(peak_v, abs(v))
            peak_rate = max(peak_rate, kf * abs(v) / r)
            peak_p = max(peak_p, p_now)
            peak_b = max(peak_b, abs(b_b))
            if n_done - last_recorded >= stride:
                record(t_b, r, v, b_b, b_in, d, temp, sigma)
                last_recorded = n_done
            if contact is not None:
                stopped_by = contact
                stop_time = t_b
                break
            p_y_now = kf * sigma * thick(r) / r
            if v == 0.0 and p_now < -p_y_now and not expansion_flagged:
                expansion_flagged = True
                warnings.append(
                    "trapped flux pushes outward beyond the yield pressure; outward plastic flow is not modelled")
            if t_b > t_peak_field and v == 0.0 and p_now <= p_y_now:
                rest_count += 1
            else:
                rest_count = 0
            if rest_count >= ARREST_STEPS and future[n + 1] / (2.0 * MU0) <= p_y_now:
                stopped_by = "arrested"
                stop_time = t_last_motion
                break

        if n_done != last_recorded:
            record(n_done * step, r, v, bo[n_done], b_in, d, temp, sigma)
        if stopped_by == "t_end":
            stop_time = t_last_motion if v == 0.0 else math.inf
            if v != 0.0:
                warnings.append("tube still moving at t_end: final strain is a lower bound; increase t_end")
        elif stopped_by == "mandrel":
            warnings.append(f"bore reached the mandrel (r = {rm:g} m) at t = {stop_time:.3g} s")
        elif stopped_by == "closed":
            warnings.append(f"bore closed completely at t = {stop_time:.3g} s (no mandrel)")
        if not moved:
            warnings.append("net pressure never exceeded the yield pressure k sigma h / r: no plastic deformation")

        # regime and skin heating of the outermost conductor at the drive's characteristic frequency; both
        # follow the field swing |B_out(t) - B_out(0)| (equal to |B_out| for a pulse that starts from zero)
        skin_mat = drv if drv is not None else m
        skin_rise = skin_temperature_rise(skin_mat, peak_swing)
        if peak_swing > 0.0:
            f_char = 0.25 / max(k_swing * step, step)
            electrical = h0 / skin_depth(m.resistivity_at(self.temperature), f_char)
            if drv is not None:
                electrical += hd0 / skin_depth(drv.resistivity_at(self.temperature), f_char)
        else:  # static drive: the field has fully diffused through the wall (infinite skin depth)
            f_char = 0.0
            electrical = 0.0
        regime = "thin" if electrical <= 1.0 else "thick"
        if peak_swing == 0.0:
            warnings.append(
                "static drive (B_out never changes): no eddy currents, so no magnetic pressure and no skin heating")
        elif regime == "thick":
            warnings.append(
                f"thick wall (sum h/delta = {electrical:.2g} at f ~ 1/(4 t_peak) = {f_char:.3g} Hz): the lumped "
                "uniform-current Joule heating underestimates the skin heating; see skin_temperature_rise_estimate")
        else:
            warnings.append(
                "thin wall (h <= delta): skin_temperature_rise_estimate (thick-conductor formula) is an upper "
                "bound; the bulk Joule rise applies")
        if self.temperature + skin_rise >= skin_mat.melting_point:
            warnings.append(
                f"outer skin of {skin_mat.key} reaches its melting point (skin estimate +{skin_rise:.0f} K for a "
                f"{peak_swing:.3g} T field swing, kappa = 1)")
        if drv is not None:
            warnings.append(
                f"skin_temperature_rise_estimate refers to the driver sheet ({drv.key}), which shields the "
                "workpiece surface; driver strength is neglected")
        if self.length is not None:
            wheeler = (self.length + 0.9 * r0) / self.length
            if wheeler > 1.25:
                warnings.append(
                    f"short tube (length {self.length:g} m vs radius {r0:g} m): the long-shell time constant "
                    f"is overestimated by ~{wheeler:.2f}x (Wheeler 1928) and end effects are not modelled")

        h_final = thick(r)
        summary = TubeSummary(
            final_radius=r,
            final_inner_radius=r - 0.5 * h_final,
            final_wall=h_final,
            final_strain=kf * math.log(r0 / r),
            peak_velocity=peak_v,
            peak_strain_rate=peak_rate,
            peak_pressure=peak_p,
            peak_field=peak_b,
            transfer_efficiency=(peak_p / (peak_b * peak_b / (2.0 * MU0))) if peak_b > 0.0 else 0.0,
            plastic_work_per_length=plastic_work,
            bulk_joule_temperature_rise=joule_rise,
            plastic_temperature_rise=plastic_rise,
            final_temperature=temp,
            skin_temperature_rise_estimate=skin_rise,
            regime=regime,
            stopped_by=stopped_by,
            time_to_stop=stop_time if stop_time is not None else math.inf,
            warnings=tuple(warnings),
        )
        history = TubeHistory(**{key: tuple(vals) for key, vals in hist.items()})
        return TubeResult(material=m.key, summary=summary, history=history)


__all__ = [
    "REFERENCES",
    "KAPPA_HALF_SINE",
    "KAPPA_STEP",
    "STRAIN_RATE_FLOOR",
    "DEFAULT_TUBE_STEPS",
    "ARREST_STEPS",
    "HISTORY_MAX_POINTS",
    "DEFAULT_DIFFUSION_STEPS",
    "skin_depth",
    "skin_depth_for_material",
    "frequency_for_skin_depth",
    "shell_time_constant",
    "diffusion_time",
    "shell_regime",
    "pressure_on_shell",
    "sheet_shielding_factor",
    "transmitted_pressure",
    "effective_current_depth",
    "surface_heating_energy_density",
    "skin_temperature_rise",
    "skin_melting_field",
    "interior_field",
    "action_integral",
    "JouleResult",
    "adiabatic_joule_heating",
    "specific_action_to_temperature",
    "DiffusionResult",
    "slab_field_diffusion",
    "damped_sine_field",
    "half_sine_field",
    "PulseRecommendation",
    "pulse_frequency_recommendation",
    "TubeSummary",
    "TubeHistory",
    "TubeResult",
    "TubeCompression",
]
