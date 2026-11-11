"""Containment: the magnet architecture inverted into a structural housing.

The concept note's last application ("Deep-Space Electromagnetic Rail & Launch
Systems") proposes that the self-supporting carbon-nanosheet + HTS magnet be
*inverted* so that it contains electromagnetic loads: the housing of a mass
driver, coilgun or railgun, or a containment ring for a compact fusion reactor.
This module puts numbers on that claim.

Models
------
* Thick-walled (Lame) cylinders under internal pressure and the yield criteria
  that bound a monolithic elastic wall: no finite wall exists unless
  ``sigma_allow > sqrt(3) p`` (von Mises) or ``> 2 p`` (Tresca).
* Fully plastic / autofrettaged / compound walls (Hill 1950), including the
  shakedown limit of a single autofrettaged tube.
* The dynamic regime: Lame statics is valid only for pulses longer than a few
  hoop acoustic transits ``2 pi a / c_L``.
* Electromagnetic load cases: rail launcher (``F = L' I^2 / 2``), coil launcher
  (``B^2 / 2 mu0``) and fusion TF coil (coil vs plasma pressure).
* Mass-driver kinematics for lunar/orbital launch, where the *payload's*
  strength, not the magnet, limits the acceleration.
* :func:`assess_containment` / :func:`containment_survey`, which label a
  containment pressure state of the art / extrapolation / aspirational /
  infeasible for each candidate wall material.

Conventions: SI units; stresses are tension-positive; pressures are positive
compressive tractions; ``a``/``b`` are the inner/outer radii and ``k = b/a``.
Magnetic pressure is computed here as ``B^2/(2 mu0)`` from
:data:`magform.constants.MU0`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ._report import Assessment, ResultMixin, unit
from .benchmarks import (
    PRESSURE_BENCHMARKS,
    PULSED_NONDESTRUCTIVE_BORE,
    PULSED_NONDESTRUCTIVE_RECORD,
    pulsed_current_status,
    superconducting_field_status,
)
from .constants import G_ACCEL, MU0
from .errors import InputError
from .materials import Material, MaterialLike, as_material
from .status import Status, worst

REFERENCES: dict[str, str] = {
    "lame_1852": "G. Lame, Lecons sur la theorie mathematique de l'elasticite des corps solides, "
                 "Bachelier, Paris (1852).",
    "timoshenko_1970": "S. P. Timoshenko & J. N. Goodier, Theory of Elasticity, 3rd ed., McGraw-Hill, "
                       "New York (1970), sec. 33 (thick-walled cylinder).",
    "hill_1950": "R. Hill, The Mathematical Theory of Plasticity, Clarendon Press, Oxford (1950), ch. V "
                 "(plastic expansion of a thick-walled tube; autofrettage).",
    "chakrabarty_2006": "J. Chakrabarty, Theory of Plasticity, 3rd ed., Elsevier Butterworth-Heinemann, "
                        "Oxford (2006): thick-walled tubes, residual stresses and shakedown.",
    "harvey_1985": "J. F. Harvey, Theory and Design of Pressure Vessels, Van Nostrand Reinhold, New York (1985).",
    "asme_viii_3": "ASME Boiler and Pressure Vessel Code, Section VIII, Division 3: Alternative Rules for "
                   "Construction of High Pressure Vessels, ASME, New York (2023).",
    "lange_1985": "K. Lange (ed.), Handbook of Metal Forming, McGraw-Hill, New York (1985): shrink-ring "
                  "(compound) cold-forging dies, contact pressures up to ~2.5 GPa.",
    "daniel_ishai_2006": "I. M. Daniel & O. Ishai, Engineering Mechanics of Composite Materials, 2nd ed., "
                         "Oxford University Press (2006): transverse compressive strength of unidirectional "
                         "carbon/epoxy ~0.15-0.3 GPa.",
    "graff_1975": "K. F. Graff, Wave Motion in Elastic Solids, Ohio State University Press (1975): breathing "
                  "mode of a ring, period 2 pi a / c.",
    "biggs_1964": "J. M. Biggs, Introduction to Structural Dynamics, McGraw-Hill, New York (1964): dynamic "
                  "load factors for ramp and step loads.",
    "marshall_wang_2004": "R. A. Marshall & Y. Wang, Railguns: Their Science and Technology, China Machine "
                          "Press, Beijing (2004): F = L'I^2/2, practical L' = 0.4-0.6 uH/m.",
    "kolm_mongeau_1984": "H. Kolm & P. Mongeau, 'Basic principles of coaxial launch technology', "
                         "IEEE Trans. Magn. 20 (1984) 227-230.",
    "mcnab_2003": "I. R. McNab, 'Launch to space with an electromagnetic railgun', IEEE Trans. Magn. 39 "
                  "(2003) 295-304.",
    "oneill_1974": "G. K. O'Neill, 'The colonization of space', Physics Today 27(9) (1974) 32-40 "
                   "(lunar mass driver).",
    "moon_1984": "F. C. Moon, Magneto-Solid Mechanics, Wiley, New York (1984).",
    "freidberg_2007": "J. P. Freidberg, Plasma Physics and Fusion Energy, Cambridge University Press (2007): "
                      "plasma beta and magnetic pressure.",
    "mitchell_2008": "N. Mitchell et al., 'The ITER magnet system', IEEE Trans. Appl. Supercond. 18 (2008) "
                     "435-440.",
    "creely_2020": "A. J. Creely et al., 'Overview of the SPARC tokamak', J. Plasma Phys. 86 (2020) 865860502.",
    "nhmfl_100t": "NHMFL Pulsed Field Facility, Los Alamos, 100.75 T multi-shot magnet (22 March 2012); "
                  "about 10 mm bore as operated (magform.benchmarks.FIELD_LADDER).",
}
"""Short key -> full citation for every model in this module."""

_SQRT3: float = math.sqrt(3.0)


def _benchmark_value(key: str, fallback: float) -> float:
    """Value of a pressure benchmark from :data:`magform.benchmarks.PRESSURE_BENCHMARKS`."""
    for bench in PRESSURE_BENCHMARKS:
        if bench.key == key:
            return float(bench.value)
    return fallback


# ---------------------------------------------------------------------------
# Module constants (documented thresholds)
# ---------------------------------------------------------------------------
CRITERIA: tuple[str, ...] = ("von_mises", "tresca", "hoop")
"""Bore failure criteria accepted by the thick-wall functions ('hoop' ignores sigma_r and is non-conservative)."""

FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH: float = 0.25e9
"""Pa (estimated). Transverse compressive strength of a unidirectional carbon-fibre laminate
(0.15-0.3 GPa for carbon/epoxy; Daniel & Ishai 2006).  The bore of a fibre-wound housing is
loaded radially by -p; above this value a metallic liner must carry the radial stress."""

ELASTIC_RATIO_PRACTICAL: float = 3.0
"""Largest wall ratio b/a treated as a conventional monolithic thick wall (judgement): beyond
k ~ 3 the outer material is barely stressed (sigma_theta ~ 1/r^2) and such sections cannot be
through-hardened (Harvey 1985)."""

PLASTIC_RATIO_LIMIT: float = 10.0
"""Largest b/a accepted for an autofrettaged / compound wall (estimated boundary): a wall ten
bore radii thick is the practical ceiling of shrink-fit and wire-wound construction."""

SHAKEDOWN_RATIO: float = 2.2184574899167
"""b/a above which a single autofrettaged tube shakes down (reverse yielding) before plastic
collapse: root of 1 - 1/k^2 = ln k (Hill 1950; Chakrabarty 2006)."""

QUASI_STATIC_FACTOR: float = 3.0
"""Pulse length, in hoop acoustic transits 2 pi a / c_L, above which the wall responds
quasi-statically (ramp-load dynamic load factor <= ~1.1; Biggs 1964)."""

COMPOUND_DEMONSTRATED_PRESSURE: float = _benchmark_value("cold_forging_die", 2.5e9)
"""Pa. Pressure contained routinely by shrink-ring (compound) cold-forging dies:
``magform.benchmarks.PRESSURE_BENCHMARKS['cold_forging_die']`` (Lange 1985).  A compound
wall at or below it, in a commercial metal with b/a <= 3, is labelled state of the art."""

PULSED_MAGNET_CONTAINED_PRESSURE: float = PULSED_NONDESTRUCTIVE_RECORD ** 2 / (2.0 * MU0)
"""Pa (~4.04 GPa). Magnetic pressure carried for millisecond pulses by the LANL 100.75 T
non-destructive magnet (about 10 mm bore) with reinforcement distributed through the winding."""

RAIL_INDUCTANCE_GRADIENT_RANGE: tuple[float, float] = (0.4e-6, 0.6e-6)
"""H/m. Inductance gradient of practical railguns (Marshall & Wang 2004)."""

TOKAMAK_TF_STRUCTURAL_STRESS: tuple[float, float] = (0.5e9, 1.0e9)
"""Pa. Order of the structural stresses carried by the steel cases of high-field tokamak TF
coils (ITER, SPARC class; Mitchell et al. 2008, Creely et al. 2020): state of the art."""

COMMERCIAL_METAL_CATEGORIES: tuple[str, ...] = ("metal", "superalloy", "conductor")
"""Material categories treated as commercial pressure-vessel metals."""

LAMINATE_CATEGORIES: tuple[str, ...] = ("fibre", "carbon")
"""Material categories checked for bore transverse compression (with :data:`LAMINATE_STRUCTURES`)."""

LAMINATE_STRUCTURES: tuple[str, ...] = ("fibre", "composite")
"""Crystal-structure tags identifying fibre/laminate materials."""

DEFAULT_CANDIDATES: tuple[str, ...] = (
    "steel_4340", "maraging_c350", "mp35n", "carbon_fibre_t1100g",
    "cnt_fibre_record", "cns_engineering_target", "graphene_intrinsic",
)
"""Wall materials surveyed by :func:`containment_survey` (commercial steel to graphene ceiling)."""

_K_CRIT: dict[str, float] = {"von_mises": _SQRT3, "tresca": 2.0, "hoop": 1.0}
_K_LABEL: dict[str, str] = {"von_mises": "sqrt(3)", "tresca": "2", "hoop": "1x"}
_CRITERION_ALIASES: dict[str, str] = {
    "von_mises": "von_mises", "mises": "von_mises", "vm": "von_mises",
    "tresca": "tresca", "hoop": "hoop", "max_hoop": "hoop",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if math.isnan(v):
        raise InputError(f"{name} must not be NaN")
    return v


def _finite(name: str, value: object) -> float:
    v = _real(name, value)
    if not math.isfinite(v):
        raise InputError(f"{name} must be finite, got {v!r}")
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


def _safety_factor(value: object) -> float:
    v = _finite("safety_factor", value)
    if v < 1.0:
        raise InputError(f"safety_factor must be >= 1 (a factor below 1 is not conservative), got {v!r}")
    return v


def _wall_ratio(name: str, value: object) -> float:
    """b/a in [1, inf] (inf allowed: the infinitely thick wall is a meaningful limit)."""
    v = _real(name, value)
    if v < 1.0:
        raise InputError(f"{name} (b/a) must be >= 1, got {v!r}")
    return v


def _radii(a: object, b: object) -> tuple[float, float]:
    ra = _positive("inner radius", a)
    rb = _positive("outer radius", b)
    if rb <= ra:
        raise InputError(f"outer radius ({rb!r} m) must exceed inner radius ({ra!r} m)")
    return ra, rb


def _criterion(criterion: object, allow_hoop: bool = True) -> str:
    if not isinstance(criterion, str):
        raise InputError(f"criterion must be a string, one of {CRITERIA}")
    c = _CRITERION_ALIASES.get(criterion.strip().lower().replace("-", "_").replace(" ", "_"))
    if c is None:
        raise InputError(f"unknown criterion {criterion!r}; use one of {CRITERIA}")
    if c == "hoop" and not allow_hoop:
        raise InputError("the 'hoop' criterion has no fully plastic counterpart; use 'von_mises' or 'tresca'")
    return c


def _gpa(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value / 1e9:.3g} GPa"


def _containment_strength(m: Material) -> tuple[float, str]:
    """Strength of a hoop-loaded wall: tensile yield, else ultimate (fibres, brittle solids)."""
    if m.yield_strength is not None and m.yield_strength > 0.0:
        return float(m.yield_strength), "yield_strength"
    if m.ultimate_strength is not None and m.ultimate_strength > 0.0:
        return float(m.ultimate_strength), "ultimate_strength"
    raise InputError(f"material '{m.key}' has no yield_strength or ultimate_strength for a containment wall")


def _payload_strength(m: Material) -> tuple[float, str]:
    """Strength of a base-loaded payload: smaller recorded yield (tension/compression), else
    the smaller of compressive/ultimate strength for materials that do not yield."""
    yields = [(float(v), n) for n, v in (("yield_strength", m.yield_strength),
                                         ("compressive_yield", m.compressive_yield))
              if v is not None and v > 0.0]
    if yields:
        return min(yields)
    others = [(float(v), n) for n, v in (("compressive_strength", m.compressive_strength),
                                         ("ultimate_strength", m.ultimate_strength))
              if v is not None and v > 0.0]
    if others:
        return min(others)
    raise InputError(f"material '{m.key}' has no strength data for a launch payload")


def _transit(a: float, m: Material) -> float:
    return 2.0 * math.pi * a / m.longitudinal_sound_speed()


# ---------------------------------------------------------------------------
# Thick-walled cylinder (Lame) and yield criteria at the bore
# ---------------------------------------------------------------------------
def lame_stresses(
    internal_pressure: float,
    inner_radius: float,
    outer_radius: float,
    r: float,
    external_pressure: float = 0.0,
    closed_ends: bool = True,
) -> tuple[float, float, float]:
    """Lame stresses ``(sigma_r, sigma_theta, sigma_z)`` [Pa] at radius ``r`` of a thick cylinder.

    With ``A = (p_i a^2 - p_o b^2)/(b^2 - a^2)`` and ``B = (p_i - p_o) a^2 b^2/(b^2 - a^2)``::

        sigma_r     = A - B/r^2        (= -p_i at r = a, -p_o at r = b)
        sigma_theta = A + B/r^2
        sigma_z     = A  (closed ends: end-cap force balance), 0 (open ends)

    Units: pressures in Pa (positive = compressive traction), radii in m.
    Validity: linear-elastic, isotropic, homogeneous long cylinder away from its ends, small
    strain, static load (pulse longer than ~3 hoop transits, see :func:`dynamic_regime`), no
    thermal gradient.  ``r`` must lie in ``[a, b]``.
    Reference: Lame (1852); Timoshenko & Goodier (1970) sec. 33.
    """
    p_i = _non_negative("internal_pressure", internal_pressure)
    p_o = _non_negative("external_pressure", external_pressure)
    a, b = _radii(inner_radius, outer_radius)
    rr = _positive("r", r)
    if rr < a * (1.0 - 1e-9) or rr > b * (1.0 + 1e-9):
        raise InputError(f"r = {rr!r} m lies outside the wall [{a!r}, {b!r}] m")
    rr = min(max(rr, a), b)
    d = b * b - a * a
    coef_a = (p_i * a * a - p_o * b * b) / d
    coef_b = (p_i - p_o) * a * a * b * b / d
    sigma_r = coef_a - coef_b / (rr * rr)
    sigma_theta = coef_a + coef_b / (rr * rr)
    sigma_z = coef_a if closed_ends else 0.0
    return sigma_r, sigma_theta, sigma_z


def max_hoop_stress(p: float, a: float, b: float) -> float:
    """Bore hoop stress of a thick cylinder under internal pressure [Pa].

    ``sigma_theta(a) = p (b^2 + a^2)/(b^2 - a^2)`` -> p as b/a -> inf.
    Units: p in Pa, a and b (inner/outer radius) in m.
    Validity: Lame assumptions (see :func:`lame_stresses`); the hoop stress alone ignores the
    bore radial stress -p and is NOT a failure criterion (use :func:`von_mises_bore`).
    Reference: Timoshenko & Goodier (1970) sec. 33.
    """
    p = _non_negative("p", p)
    ra, rb = _radii(a, b)
    return p * (rb * rb + ra * ra) / (rb * rb - ra * ra)


def von_mises_bore(p: float, a: float, b: float, closed_ends: bool = True) -> float:
    """von Mises equivalent stress at the bore of a thick cylinder [Pa].

    closed ends (sigma_z = p a^2/(b^2 - a^2)): ``sigma_vm = sqrt(3) p b^2/(b^2 - a^2)``;
    open ends (sigma_z = 0): ``sigma_vm = p sqrt(3 b^4 + a^4)/(b^2 - a^2)``.
    Both tend to ``sqrt(3) p`` as b/a -> inf, so no finite monolithic elastic wall contains p
    unless the allowable exceeds sqrt(3) p.
    Units: Pa, m.  Validity: Lame assumptions; isotropic ductile wall.
    Reference: Lame (1852); Hill (1950) ch. V.
    """
    p = _non_negative("p", p)
    ra, rb = _radii(a, b)
    d = rb * rb - ra * ra
    if closed_ends:
        return _SQRT3 * p * rb * rb / d
    return p * math.sqrt(3.0 * rb ** 4 + ra ** 4) / d


def tresca_bore(p: float, a: float, b: float) -> float:
    """Tresca (maximum shear) stress intensity at the bore [Pa].

    ``sigma_theta - sigma_r = 2 p b^2/(b^2 - a^2)`` (open or closed ends) -> 2p as b/a -> inf.
    Units: Pa, m.  Validity: Lame assumptions.  Reference: Timoshenko & Goodier (1970); Harvey (1985).
    """
    p = _non_negative("p", p)
    ra, rb = _radii(a, b)
    return 2.0 * p * rb * rb / (rb * rb - ra * ra)


def required_monolithic_strength(p: float, safety_factor: float = 1.0, criterion: str = "von_mises") -> float:
    """Minimum strength [Pa] for which ANY finite monolithic elastic wall contains ``p``.

    ``sigma_min = k_crit * SF * p`` with ``k_crit = sqrt(3)`` (von Mises, closed ends), 2 (Tresca)
    or 1 (hoop only, non-conservative).  At 4 GPa: 6.93 GPa (von Mises), 8.0 GPa (Tresca);
    10.4 / 12.0 GPa with SF 1.5.
    Units: Pa.  Validity: static Lame wall of one homogeneous material (no prestress).
    Reference: Hill (1950) ch. V; Harvey (1985).
    """
    p = _non_negative("p", p)
    sf = _safety_factor(safety_factor)
    c = _criterion(criterion)
    return _K_CRIT[c] * sf * p


def required_radius_ratio(p: float, sigma_allow: float, criterion: str = "von_mises") -> float:
    """Wall ratio k = b/a at which the bore stress reaches ``sigma_allow`` (monolithic, elastic).

    * hoop:      ``k^2 = (sigma + p)/(sigma - p)``   needs sigma > p
    * tresca:    ``k^2 = sigma/(sigma - 2p)``        needs sigma > 2p
    * von_mises: ``k^2 = sigma/(sigma - sqrt(3) p)`` needs sigma > sqrt(3) p (closed ends)

    Returns ``math.inf`` when no finite wall works (never raises for that); returns 1.0 for
    p = 0.  Raises :class:`InputError` for negative inputs or an unknown criterion.
    Units: Pa in, dimensionless out.  Validity: Lame assumptions, single homogeneous wall.
    Reference: Timoshenko & Goodier (1970); Hill (1950).
    """
    p = _non_negative("p", p)
    s = _non_negative("sigma_allow", sigma_allow)
    c = _criterion(criterion)
    if p == 0.0:
        return 1.0
    if c == "hoop":
        return math.sqrt((s + p) / (s - p)) if s > p else math.inf
    limit = _K_CRIT[c] * p
    return math.sqrt(s / (s - limit)) if s > limit else math.inf


def elastic_limit_pressure(sigma_y: float, k: float, criterion: str = "von_mises") -> float:
    """Internal pressure [Pa] at which a monolithic wall of ratio k = b/a first yields at the bore.

    * von_mises (closed ends): ``p_y = sigma_y (k^2 - 1)/(sqrt(3) k^2)`` -> sigma_y/sqrt(3)
    * tresca:                  ``p_y = sigma_y (k^2 - 1)/(2 k^2)``       -> sigma_y/2
    * hoop:                    ``p_y = sigma_y (k^2 - 1)/(k^2 + 1)``     -> sigma_y

    ``k = math.inf`` returns the asymptote.  Units: Pa.  Validity: Lame assumptions.
    Reference: Hill (1950) ch. V; Chakrabarty (2006).
    """
    s = _non_negative("sigma_y", sigma_y)
    kk = _wall_ratio("k", k)
    c = _criterion(criterion)
    inv = 1.0 / (kk * kk)
    frac = 1.0 - inv  # (k^2 - 1)/k^2, exactly 1 for k = inf
    if c == "von_mises":
        return s * frac / _SQRT3
    if c == "tresca":
        return s * frac / 2.0
    return s * frac / (1.0 + inv)


def fully_plastic_pressure(sigma_y: float, k: float, criterion: str = "von_mises") -> float:
    """Plastic-collapse pressure [Pa] of a wall of ratio k: the autofrettaged / compound limit.

    von Mises (closed ends, plane-strain incompressible flow): ``p_c = (2/sqrt(3)) sigma_y ln k``;
    Tresca: ``p_c = sigma_y ln k``.  It is also the limit of an infinitely layered, optimally
    shrink-fitted compound cylinder, which reaches it elastically.
    Units: Pa.  Validity: elastic-perfectly plastic wall, no hardening or Bauschinger effect,
    small strain; a SINGLE autofrettaged tube reaches it cyclically only for
    k <= :data:`SHAKEDOWN_RATIO` (see :func:`shakedown_pressure`).
    Reference: Hill (1950) ch. V.
    """
    s = _non_negative("sigma_y", sigma_y)
    kk = _wall_ratio("k", k)
    c = _criterion(criterion, allow_hoop=False)
    if s == 0.0 or kk == 1.0:
        return 0.0
    factor = 2.0 / _SQRT3 if c == "von_mises" else 1.0
    return factor * s * math.log(kk)


def shakedown_pressure(sigma_y: float, k: float, criterion: str = "von_mises") -> float:
    """Largest cyclic working pressure [Pa] of a single autofrettaged tube of ratio k.

    ``p_sd = min(2 p_y(k), p_c(k))`` with p_y = :func:`elastic_limit_pressure` and
    p_c = :func:`fully_plastic_pressure`: unloading from overstrain re-yields the bore in reverse
    once the elastic stress range exceeds twice the yield stress.  For k > 2.218 the tube shakes
    down below collapse and is bounded by ``(2/sqrt(3)) sigma_y`` (von Mises) however thick it is.
    Units: Pa.  Validity: elastic-perfectly plastic, no Bauschinger effect (which lowers it further).
    Reference: Hill (1950) ch. V; Chakrabarty (2006).
    """
    c = _criterion(criterion, allow_hoop=False)
    return min(2.0 * elastic_limit_pressure(sigma_y, k, c), fully_plastic_pressure(sigma_y, k, c))


def autofrettage_ratio(p: float, sigma_y: float, safety_factor: float = 1.0, criterion: str = "von_mises") -> float:
    """Wall ratio k = b/a whose fully plastic (compound) capacity equals ``SF * p``.

    von Mises: ``k = exp(sqrt(3) SF p / (2 sigma_y))``; Tresca: ``k = exp(SF p / sigma_y)``.
    Check: 4 GPa, sigma_y = 2 GPa -> k = 5.65.  Returns ``math.inf`` on overflow (never raises
    for an impossible design).
    Units: Pa in, dimensionless out.  Validity: see :func:`fully_plastic_pressure`; beyond
    k = 2.218 only multi-layer shrink-fit / wire-wound construction reaches it cyclically.
    Reference: Hill (1950) ch. V.
    """
    p = _non_negative("p", p)
    s = _positive("sigma_y", sigma_y)
    sf = _safety_factor(safety_factor)
    c = _criterion(criterion, allow_hoop=False)
    factor = _SQRT3 / 2.0 if c == "von_mises" else 1.0
    try:
        return math.exp(factor * sf * p / s)
    except OverflowError:
        return math.inf


# ---------------------------------------------------------------------------
# Dynamic regime
# ---------------------------------------------------------------------------
def hoop_transit_time(inner_radius: float, material: MaterialLike) -> float:
    """Hoop acoustic transit time ``2 pi a / c_L`` [s] around the bore.

    c_L is the dilatational wave speed (:meth:`Material.longitudinal_sound_speed`).  It sets the
    breathing-mode period of the wall; the thin-ring period uses the bar speed sqrt(E/rho),
    10-20 % lower, which :data:`QUASI_STATIC_FACTOR` absorbs.
    Units: m in, s out.  Validity: order-of-magnitude regime indicator, not a modal analysis.
    Reference: Graff (1975).
    """
    a = _positive("inner_radius", inner_radius)
    m = as_material(material)
    return _transit(a, m)


def dynamic_regime(inner_radius: float, material: MaterialLike, pulse_duration: float) -> str:
    """Classify the wall response to a pressure pulse: ``"quasi_static"`` or ``"inertial"``.

    ``"quasi_static"`` if ``pulse_duration > 3 * 2 pi a / c_L`` (dynamic load factor of a ramp
    <= ~1.1), else ``"inertial"``: the wall is loaded by stress waves, a step load can amplify
    the static stress up to 2x, and Lame statics is not adequate (stress-wave analysis needed).
    Units: m, s.  Validity: regime indicator only.  Reference: Graff (1975); Biggs (1964).
    """
    a = _positive("inner_radius", inner_radius)
    m = as_material(material)
    t = _positive("pulse_duration", pulse_duration)
    return "quasi_static" if t > QUASI_STATIC_FACTOR * _transit(a, m) else "inertial"


# ---------------------------------------------------------------------------
# Electromagnetic launcher and fusion load cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class RailLoad(ResultMixin):
    """Structural load case of a rail launcher (railgun) firing at constant current.

    ``force = L' I^2/2`` on the armature; ``bore_pressure = force/(w h)``; ``effective_field =
    sqrt(2 mu0 P)``; ``rail_repulsion_per_length = L' I^2/(2 w)`` (= P h) pushing the rails
    apart; ``ideal_sheet_field = mu0 I/h`` (current spread over the rail height h, parallel-plate
    limit, equal to the effective field only when L' = mu0 w/h).
    Reference: Marshall & Wang (2004); McNab (2003).
    """

    current: float = unit("A")
    inductance_gradient: float = unit("H/m")
    force: float = unit("N")
    bore_pressure: float = unit("Pa")
    effective_field: float = unit("T")
    rail_repulsion_per_length: float = unit("N/m")
    ideal_sheet_field: float = unit("T")
    bore_height: float = unit("m")
    bore_width: float = unit("m")
    current_status: Status
    notes: tuple[str, ...] = ()


def ideal_inductance_gradient(bore_height: float, bore_width: float) -> float:
    """Inductance gradient [H/m] of ideal parallel-plate rails: ``L' = mu0 w / h``.

    w = rail separation (bore width), h = rail height (bore height); fringing neglected, so it
    over-estimates real rails (0.4-0.6 uH/m).  For a square bore L' = mu0 = 1.257 uH/m.
    Units: m in, H/m out.  Validity: thin plates with h >> skin depth.
    Reference: Marshall & Wang (2004).
    """
    h = _positive("bore_height", bore_height)
    w = _positive("bore_width", bore_width)
    return MU0 * w / h


def rail_launcher_load(
    current: float,
    bore_height: float,
    bore_width: float,
    inductance_gradient: float = 0.5e-6,
) -> RailLoad:
    """Rail-launcher (railgun) structural loads at peak current.

    ``F = L' I^2/2`` [N]; ``P = F/(w h)`` [Pa]; ``B_eff = sqrt(2 mu0 P)`` [T];
    rail repulsion per length ``L' I^2/(2 w)`` [N/m]; ideal sheet field ``mu0 I/h`` [T].
    Check: 4 MA, 50 mm x 50 mm, L' = 0.5 uH/m -> F = 4.0 MN, P = 1.6 GPa, B_eff = 63.4 T;
    the ideal-sheet bound (L' = mu0) gives 4.02 GPa / 100.5 T.
    Units: A, m, H/m.  Validity: quasi-steady current, uniform current over the rail height,
    rigid rails; ignores velocity skin effect and armature contact physics.
    Reference: Marshall & Wang (2004); McNab (2003).
    """
    i = _positive("current", current)
    h = _positive("bore_height", bore_height)
    w = _positive("bore_width", bore_width)
    lp = _positive("inductance_gradient", inductance_gradient)
    force = 0.5 * lp * i * i
    pressure = force / (w * h)
    b_eff = math.sqrt(2.0 * MU0 * pressure)
    repulsion = lp * i * i / (2.0 * w)
    b_sheet = MU0 * i / h
    lp_ideal = MU0 * w / h
    p_ideal = b_sheet * b_sheet / (2.0 * MU0)
    current_status = pulsed_current_status(i)
    notes = [
        f"F = L'I^2/2 with L' = {lp * 1e6:.3g} uH/m; practical rails have L' = 0.4-0.6 uH/m "
        "(Marshall & Wang 2004).",
        f"ideal parallel-plate bound: L' = mu0 w/h = {lp_ideal * 1e6:.3g} uH/m gives B = mu0 I/h = "
        f"{b_sheet:.4g} T and P = {_gpa(p_ideal)}; B_eff equals it only when L' = mu0 w/h.",
        f"the rail housing must contain {repulsion / 1e6:.3g} MN/m of rail repulsion (= P h); "
        "size it with assess_containment(bore_pressure, ...).",
        f"peak current {i / 1e6:.3g} MA: {current_status.label} (benchmarks.pulsed_current_status).",
    ]
    lo, hi = RAIL_INDUCTANCE_GRADIENT_RANGE
    if not lo <= lp <= hi:
        notes.append(f"inductance gradient {lp * 1e6:.3g} uH/m lies outside the practical "
                     f"{lo * 1e6:.1f}-{hi * 1e6:.1f} uH/m range.")
    return RailLoad(
        current=i,
        inductance_gradient=lp,
        force=force,
        bore_pressure=pressure,
        effective_field=b_eff,
        rail_repulsion_per_length=repulsion,
        ideal_sheet_field=b_sheet,
        bore_height=h,
        bore_width=w,
        current_status=current_status,
        notes=tuple(notes),
    )


def coil_launcher_bore_pressure(field: float) -> float:
    """Magnetic pressure [Pa] on a coil-launcher (coilgun) armature: ``P = B^2/(2 mu0)``.

    B is the drive field at the armature surface with the field excluded from inside the
    armature (thick, highly conducting sleeve).  Check: 63.41 T -> 1.6 GPa, 100.27 T -> 4.0 GPa.
    Units: T in, Pa out.  Validity: armature thicker than the skin depth; upper bound otherwise.
    Reference: Kolm & Mongeau (1984); Moon (1984).
    """
    b = _finite("field", field)
    return b * b / (2.0 * MU0)


@dataclass(frozen=True, kw_only=True)
class FusionMagnetLoad(ResultMixin):
    """Fusion-magnet load case: magnetic pressure on the TF coil vs the plasma pressure.

    ``coil_pressure = B_coil^2/(2 mu0)``; ``plasma_pressure = beta B_axis^2/(2 mu0)``;
    ``ratio = coil_pressure/plasma_pressure`` (inf when beta = 0).  ``field_status`` places the
    on-coil field on the superconducting ladder (benchmarks).
    Reference: Freidberg (2007); Mitchell et al. (2008); Creely et al. (2020).
    """

    field_on_coil: float = unit("T")
    coil_pressure: float = unit("Pa")
    plasma_beta: float
    field_on_axis: float = unit("T")
    plasma_pressure: float = unit("Pa")
    ratio: float
    field_status: Status
    notes: tuple[str, ...] = ()


def fusion_magnet_load(field_on_coil: float, field_on_axis: float, plasma_beta: float = 0.05) -> FusionMagnetLoad:
    """Structural load of a fusion TF magnet compared with the plasma it confines.

    ``P_coil = B_coil^2/(2 mu0)``, ``P_plasma = beta B_axis^2/(2 mu0)``, ``ratio = P_coil/P_plasma``.
    Compact high-field tokamak TF coils (e.g. SPARC class, ~20 T on coil) run structural
    stresses of order 0.5-1 GPa in steel cases -- containment at that level is state of the
    art; the 'containment ring' carries the magnet's own Lorentz forces, not the plasma.
    Units: T, dimensionless beta (0 <= beta <= 1).  Validity: pressure scales only; the real
    TF-case stress depends on the centering force, bending and the bucking/wedging design.
    Reference: Freidberg (2007); Mitchell et al. (2008); Creely et al. (2020).
    """
    b_coil = _positive("field_on_coil", field_on_coil)
    b_axis = _positive("field_on_axis", field_on_axis)
    beta = _finite("plasma_beta", plasma_beta)
    if not 0.0 <= beta <= 1.0:
        raise InputError(f"plasma_beta must lie in [0, 1], got {beta!r}")
    p_coil = b_coil * b_coil / (2.0 * MU0)
    p_plasma = beta * b_axis * b_axis / (2.0 * MU0)
    ratio = p_coil / p_plasma if p_plasma > 0.0 else math.inf
    status = superconducting_field_status(b_coil)
    notes = [
        f"magnetic pressure on the TF winding B^2/2mu0 = {p_coil / 1e6:.4g} MPa; TF-case structural "
        "stresses are a few times this (centering force, bending) and reach ~0.5-1 GPa in ITER/SPARC-class "
        "steel cases: state of the art (Mitchell et al. 2008; Creely et al. 2020).",
        f"the plasma pushes with only beta B_axis^2/2mu0 = {p_plasma / 1e6:.4g} MPa (ratio {ratio:.3g}): "
        "the 'containment ring' contains the magnet's own Lorentz forces, not the plasma.",
        f"on-coil field {b_coil:g} T: {status.label} for a superconducting magnet "
        "(benchmarks.superconducting_field_status).",
    ]
    if p_coil > TOKAMAK_TF_STRUCTURAL_STRESS[1]:
        notes.append(f"the coil magnetic pressure alone ({_gpa(p_coil)}) exceeds the ~1 GPa case stresses of "
                     "present TF coils: assess the structure with assess_containment(coil_pressure, ...).")
    if b_axis > b_coil:
        notes.append("on-axis field exceeds the on-coil field, unusual for a tokamak (B ~ 1/R): check inputs.")
    return FusionMagnetLoad(
        field_on_coil=b_coil,
        coil_pressure=p_coil,
        plasma_beta=beta,
        field_on_axis=b_axis,
        plasma_pressure=p_plasma,
        ratio=ratio,
        field_status=status,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Mass driver (space launch) kinematics
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class MassDriverResult(ResultMixin):
    """Kinematics of an electromagnetic mass driver accelerating a rigid payload.

    ``allowed_pressure`` is the base pressure actually applied (the binding constraint is
    ``limited_by``: "bore_pressure" | "payload_strength" | "acceleration_limit");
    ``acceleration = P/(rho L)``, ``launcher_length = v^2/(2a)``, ``launch_time = v/a``,
    ``payload_mass = rho pi r^2 L``, ``kinetic_energy = m v^2/2``, ``bore_field = sqrt(2 mu0 P)``
    (coilgun-equivalent drive field), ``g_load = a/g0``.
    Reference: O'Neill (1974); McNab (2003).
    """

    exit_velocity: float = unit("m/s")
    payload_material: str
    payload_length: float = unit("m")
    bore_radius: float = unit("m")
    allowed_pressure: float = unit("Pa")
    limited_by: str
    acceleration: float = unit("m/s^2")
    launcher_length: float = unit("m")
    launch_time: float = unit("s")
    payload_mass: float = unit("kg")
    kinetic_energy: float = unit("J")
    bore_field: float = unit("T")
    bore_pressure: float | None = unit("Pa", default=None)
    payload_strength: float = unit("Pa")
    payload_pressure_limit: float = unit("Pa")
    safety_factor: float
    g_load: float
    notes: tuple[str, ...] = ()


def mass_driver(
    exit_velocity: float,
    payload_material: MaterialLike,
    payload_length: float,
    bore_radius: float,
    bore_pressure: float | None = None,
    safety_factor: float = 1.5,
    acceleration_limit: float | None = None,
) -> MassDriverResult:
    """Space-launch kinematics of an electromagnetic mass driver (coilgun / railgun).

    ``P = min(bore_pressure, sigma_p/SF)`` where sigma_p is the payload strength (smaller
    recorded tensile/compressive yield; the smaller of compressive/ultimate strength if the
    material does not yield); ``a = P/(rho L)`` (capped by ``acceleration_limit``);
    ``length = v^2/(2a)``; ``t = v/a``; ``m = rho pi r^2 L``; ``E_k = m v^2/2``.
    The payload, not the magnet, limits acceleration: a 4 GPa bore capability is irrelevant
    to launch and only matters for containing the magnet itself.
    Check: 2400 m/s (lunar escape), aluminum_6061, 1 m payload, 4 GPa bore -> limited by
    payload strength, length = v^2 rho L/(2 sigma_y/1.5) = 42.3 m.
    Units: m/s, m, Pa, m/s^2.  Validity: rigid payload of uniform density filling the bore,
    pushed at its base by a constant pressure (peak stress = P at the base); no armature/sabot
    mass, no atmosphere (airless body), no electrical efficiency or energy supply.
    Reference: O'Neill (1974); McNab (2003); Kolm & Mongeau (1984).
    """
    v = _positive("exit_velocity", exit_velocity)
    m = as_material(payload_material)
    length = _positive("payload_length", payload_length)
    r = _positive("bore_radius", bore_radius)
    p_bore = None if bore_pressure is None else _positive("bore_pressure", bore_pressure)
    sf = _safety_factor(safety_factor)
    a_cap = None if acceleration_limit is None else _positive("acceleration_limit", acceleration_limit)

    sigma, basis = _payload_strength(m)
    p_payload = sigma / sf
    if p_bore is not None and p_bore < p_payload:
        pressure, limited_by = p_bore, "bore_pressure"
    else:
        pressure, limited_by = p_payload, "payload_strength"
    areal_mass = m.density * length
    accel = pressure / areal_mass
    if a_cap is not None and a_cap < accel:
        accel, limited_by = a_cap, "acceleration_limit"
        pressure = areal_mass * accel
    launcher_length = v * v / (2.0 * accel)
    launch_time = v / accel
    mass = m.density * math.pi * r * r * length
    energy = 0.5 * mass * v * v
    b_bore = math.sqrt(2.0 * MU0 * pressure)

    notes: list[str] = []
    if limited_by == "payload_strength":
        notes.append(f"the payload, not the magnet, limits acceleration: base pressure capped at "
                     f"{_gpa(p_payload)} = {basis.replace('_', ' ')}/SF.")
        if p_bore is not None:
            notes.append(f"the {_gpa(p_bore)} bore-pressure capability would stress the payload base to "
                         f"{p_bore / sigma:.3g}x its strength: it is irrelevant to launch and matters only for "
                         "containing the launcher's own coils/rails.")
    elif limited_by == "bore_pressure":
        notes.append(f"the launcher's bore pressure ({_gpa(pressure)}) binds below the payload limit "
                     f"({_gpa(p_payload)}).")
    else:
        notes.append(f"the imposed acceleration limit binds; base pressure {_gpa(pressure)} is below the "
                     f"payload limit {_gpa(p_payload)}.")
    notes.append(f"peak acceleration {accel:.3g} m/s^2 = {accel / G_ACCEL:.3g} g0 over {launch_time:.3g} s; "
                 f"launcher length {launcher_length:.3g} m.")
    notes.append("model: rigid payload filling the bore, constant base pressure (uniform acceleration); "
                 "no armature/sabot mass, no atmosphere (airless lunar launch; Earth orbit also needs "
                 "atmospheric transit), no electrical efficiency.")
    return MassDriverResult(
        exit_velocity=v,
        payload_material=m.key,
        payload_length=length,
        bore_radius=r,
        allowed_pressure=pressure,
        limited_by=limited_by,
        acceleration=accel,
        launcher_length=launcher_length,
        launch_time=launch_time,
        payload_mass=mass,
        kinetic_energy=energy,
        bore_field=b_bore,
        bore_pressure=p_bore,
        payload_strength=sigma,
        payload_pressure_limit=p_payload,
        safety_factor=sf,
        g_load=accel / G_ACCEL,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Containment assessment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class ContainmentAssessment(Assessment):
    """Feasibility of containing an internal pressure in a cylindrical wall of one material.

    ``sigma_allow = strength/SF`` (yield, or ultimate for materials that do not yield);
    ``required_monolithic_strength = k_crit SF p`` (k_crit = sqrt(3) von Mises, 2 Tresca, 1 hoop);
    ``elastic_ratio`` = monolithic b/a (inf if impossible); ``plastic_ratio`` = autofrettaged /
    compound b/a (:func:`autofrettage_ratio`); ``mass_per_length = rho pi a^2 (k^2 - 1)`` for the
    smaller finite design; ``radial_stress_at_bore = -p``; ``transverse_compression_ok`` (fibre/
    carbon laminates only, else None); ``dynamic_regime`` (None without a pulse duration);
    ``route`` = "elastic_monolithic" | "compound" | "none".
    ``margin`` = compound capacity at b/a = :data:`PLASTIC_RATIO_LIMIT` divided by p, so
    margin >= 1 exactly when some admissible construction exists.
    Reference: Lame (1852); Hill (1950); Harvey (1985).
    """

    pressure: float = unit("Pa")
    material: str
    inner_radius: float = unit("m")
    safety_factor: float
    criterion: str
    sigma_allow: float = unit("Pa")
    required_monolithic_strength: float = unit("Pa")
    elastic_ratio: float
    plastic_ratio: float
    outer_radius_elastic: float = unit("m")
    outer_radius_plastic: float = unit("m")
    mass_per_length: float = unit("kg/m")
    radial_stress_at_bore: float = unit("Pa")
    transverse_compression_ok: bool | None = None
    dynamic_regime: str | None = None
    route: str
    strength: float = unit("Pa")
    strength_basis: str
    pulse_duration: float | None = unit("s", default=None)
    hoop_transit_time: float = unit("s")


def assess_containment(
    pressure: float,
    material: MaterialLike,
    inner_radius: float,
    safety_factor: float = 1.5,
    criterion: str = "von_mises",
    pulse_duration: float | None = None,
) -> ContainmentAssessment:
    """Assess containing ``pressure`` in a cylindrical housing of ``material`` (README: 'inverted
    architecture ... the ultimate containment housing').

    Equations: Lame bore criteria (:func:`required_radius_ratio`), the fully plastic / compound
    limit ``p = (2/sqrt(3)) (sigma/SF) ln k`` (:func:`autofrettage_ratio`), and for fibre/carbon
    laminates a bore transverse-compression screen ``p <= 0.25 GPa``
    (:data:`FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH`, estimated).
    Status (first match, then worst-combined with the dynamic check):

    1. elastic monolithic wall with b/a <= 3 -> state of the art for a commercial metal
       (category metal/superalloy/conductor), extrapolation otherwise;
    2. only an autofrettaged/compound (fully plastic) wall with b/a <= 10 -> extrapolation
       (state of the art for a commercial metal with b/a <= 3 at p <= 2.5 GPa, the shrink-ring
       cold-forging-die benchmark);
    3. required strength met only by a carbon-nanosheet-class material (category carbon) ->
       aspirational;
    4. nothing works (no finite elastic wall and compound b/a > 10) -> infeasible;
    5. a pulse shorter than 3 hoop transits (inertial loading) -> at least extrapolation.

    Units: Pa, m, s.  Validity: static, long, homogeneous cylinder; isotropic criteria (fibre
    laminates are only screened); no thermal, fatigue or end effects.
    Reference: Lame (1852); Hill (1950); Harvey (1985); Daniel & Ishai (2006); Lange (1985).
    """
    p = _positive("pressure", pressure)
    m = as_material(material)
    a = _positive("inner_radius", inner_radius)
    sf = _safety_factor(safety_factor)
    crit = _criterion(criterion)
    plastic_crit = "tresca" if crit == "tresca" else "von_mises"
    tau = None if pulse_duration is None else _positive("pulse_duration", pulse_duration)

    strength, basis = _containment_strength(m)
    sigma_allow = strength / sf
    required = _K_CRIT[crit] * sf * p
    k_el = required_radius_ratio(p, sigma_allow, crit)
    k_pl = autofrettage_ratio(p, strength, sf, plastic_crit)
    b_el = a * k_el
    b_pl = a * k_pl
    k_min = min(k_el, k_pl)
    mass = m.density * math.pi * a * a * (k_min * k_min - 1.0) if math.isfinite(k_min) else math.inf
    margin = fully_plastic_pressure(sigma_allow, PLASTIC_RATIO_LIMIT, plastic_crit) / p

    warnings: list[str] = []
    notes: list[str] = [
        f"Lame thick wall (static, linear elastic, closed ends): a monolithic elastic wall exists only if "
        f"sigma_allow > {_K_LABEL[crit]} p; at p = {_gpa(p)} with SF {sf:g} the strength must exceed "
        f"{_gpa(required)} ({m.key} {basis.replace('_', ' ')}: {_gpa(strength)}).",
    ]
    if crit == "hoop":
        warnings.append("the 'hoop' criterion ignores the bore radial stress -p and is non-conservative; "
                        "von Mises (sqrt(3) p) or Tresca (2 p) govern a real wall.")

    commercial = m.category in COMMERCIAL_METAL_CATEGORIES
    carbon = m.category == "carbon"
    if k_el <= ELASTIC_RATIO_PRACTICAL:
        route = "elastic_monolithic"
        status = Status.STATE_OF_THE_ART if commercial else Status.EXTRAPOLATION
        notes.append(f"elastic monolithic wall: b/a = {k_el:.4g} (outer radius {b_el:.4g} m).")
        if commercial:
            notes.append("commercial pressure-vessel metal in a conventional thick wall: state of the art.")
        elif not carbon:
            notes.append(f"elastic wall possible, but category '{m.category}' is not a commercial "
                         "pressure-vessel metal: extrapolation.")
    elif k_pl <= PLASTIC_RATIO_LIMIT:
        route = "compound"
        demonstrated = (commercial and k_pl <= ELASTIC_RATIO_PRACTICAL
                        and p <= COMPOUND_DEMONSTRATED_PRESSURE)
        status = Status.STATE_OF_THE_ART if demonstrated else Status.EXTRAPOLATION
        notes.append(f"no practical monolithic wall (elastic b/a = {k_el:.4g}); a fully plastic/compound wall "
                     f"needs b/a = {k_pl:.4g} (outer radius {b_pl:.4g} m) from p = (2/sqrt(3)) sigma ln(b/a) "
                     "(Hill 1950).")
        if demonstrated:
            notes.append(f"compound wall with b/a <= {ELASTIC_RATIO_PRACTICAL:g} at p <= "
                         f"{_gpa(COMPOUND_DEMONSTRATED_PRESSURE)} (shrink-ring cold-forging dies, Lange 1985): "
                         "state of the art.")
        elif not carbon:
            notes.append(f"compound/autofrettaged containment beyond demonstrated practice (shrink-ring dies to "
                         f"{_gpa(COMPOUND_DEMONSTRATED_PRESSURE)}; HPHT presses reach ~6 GPa only with WC "
                         "anvils and pressure-gradient gaskets over cm^3): extrapolation.")
        if basis == "yield_strength":
            if k_pl > SHAKEDOWN_RATIO:
                cap = shakedown_pressure(strength, k_pl, plastic_crit) / sf
                notes.append(f"a single autofrettaged tube of this ratio shakes down (reverse yielding) at "
                             f"{_gpa(cap)} < p (b/a > {SHAKEDOWN_RATIO:.3f}); multi-layer shrink-fit or "
                             "wire-wound compound construction is required to approach the ln(b/a) limit.")
        else:
            notes.append(f"{m.key} does not yield: the compound route means an ideally prestressed "
                         "(tension-wound or shrink-fit) construction on a ductile liner, not autofrettage.")
    else:
        route = "none"
        status = Status.INFEASIBLE
        warnings.append(f"no finite monolithic elastic wall (sigma_allow {_gpa(sigma_allow)} <= "
                        f"{_K_LABEL[crit]} p) and the compound wall would need b/a = {k_pl:.4g} > "
                        f"{PLASTIC_RATIO_LIMIT:g}: {m.key} cannot contain {_gpa(p)} with SF {sf:g}.")

    if carbon and status is not Status.INFEASIBLE:
        status = Status.ASPIRATIONAL
        notes.append(f"the required strength is met only by a carbon-nanosheet-class material ({m.name}); "
                     "no macroscopic containment structure of it exists: aspirational.")
        if m.notes:
            notes.append(f"material note: {m.notes}")

    laminate = m.category in LAMINATE_CATEGORIES and m.crystal_structure in LAMINATE_STRUCTURES
    transverse_ok: bool | None = None
    if laminate:
        transverse_ok = p <= FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH
        if not transverse_ok:
            warnings.append(f"bore radial compression {_gpa(p)} exceeds the laminate transverse compressive "
                            f"strength ~{_gpa(FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH)} (estimated): a metallic "
                            "liner must carry the radial stress.")
        elif sf * p > FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH:
            notes.append(f"bore radial compression is within the laminate transverse strength "
                         f"(~{_gpa(FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH)}) but not with SF {sf:g}.")
        notes.append("fibre laminate: von Mises/Tresca are isotropic metal criteria and the strength used is "
                     "the bare-fibre value; a wound composite (~60 % fibre) carries ~0.6 of it in the hoop "
                     "direction and almost nothing radially.")

    if COMPOUND_DEMONSTRATED_PRESSURE < p <= PULSED_MAGNET_CONTAINED_PRESSURE:
        notes.append(f"for comparison, the LANL 100.75 T pulsed magnet carries {_gpa(PULSED_MAGNET_CONTAINED_PRESSURE)} "
                     f"for millisecond pulses in a {PULSED_NONDESTRUCTIVE_BORE * 1e3:g} mm bore with reinforcement "
                     f"distributed through the winding (not one wall); this bore is {2e3 * a:.3g} mm.")

    transit = _transit(a, m)
    regime: str | None = None
    if tau is None:
        notes.append(f"dynamic regime not evaluated (no pulse_duration); hoop transit 2 pi a/c_L = {transit:.3g} s.")
    else:
        regime = "quasi_static" if tau > QUASI_STATIC_FACTOR * transit else "inertial"
        if regime == "inertial":
            status = worst(status, Status.EXTRAPOLATION)
            warnings.append(f"pulse {tau:.3g} s is shorter than {QUASI_STATIC_FACTOR:g} hoop transits "
                            f"({QUASI_STATIC_FACTOR * transit:.3g} s): inertial loading, Lame statics is not "
                            "adequate (step-load amplification up to 2x; stress-wave analysis needed).")
        else:
            notes.append(f"quasi-static: pulse {tau:.3g} s exceeds {QUASI_STATIC_FACTOR:g} hoop transits "
                         f"({QUASI_STATIC_FACTOR * transit:.3g} s); Lame statics applies.")

    return ContainmentAssessment(
        status=status,
        margin=margin,
        warnings=tuple(warnings),
        notes=tuple(notes),
        pressure=p,
        material=m.key,
        inner_radius=a,
        safety_factor=sf,
        criterion=crit,
        sigma_allow=sigma_allow,
        required_monolithic_strength=required,
        elastic_ratio=k_el,
        plastic_ratio=k_pl,
        outer_radius_elastic=b_el,
        outer_radius_plastic=b_pl,
        mass_per_length=mass,
        radial_stress_at_bore=-p,
        transverse_compression_ok=transverse_ok,
        dynamic_regime=regime,
        route=route,
        strength=strength,
        strength_basis=basis,
        pulse_duration=tau,
        hoop_transit_time=transit,
    )


def containment_survey(
    pressure: float,
    inner_radius: float,
    safety_factor: float = 1.5,
    candidates: Sequence[MaterialLike] = DEFAULT_CANDIDATES,
    criterion: str = "von_mises",
    pulse_duration: float | None = None,
) -> tuple[ContainmentAssessment, ...]:
    """Run :func:`assess_containment` for each candidate wall material (input order kept).

    Same equations and status rules as :func:`assess_containment`.  The default candidates span
    commercial steel (steel_4340), maraging steel, MP35N, T1100G carbon fibre, the record CNT
    fibre, the carbon-nanosheet engineering target and the graphene intrinsic ceiling.  At
    4 GPa with SF 1.5 no candidate below graphene has a finite monolithic elastic wall.
    Units: Pa, m, s.  Validity: as :func:`assess_containment`.
    Reference: Lame (1852); Hill (1950).
    """
    p = _positive("pressure", pressure)
    a = _positive("inner_radius", inner_radius)
    sf = _safety_factor(safety_factor)
    if isinstance(candidates, (str, Material)):
        candidates = (candidates,)
    items = tuple(candidates)
    if not items:
        raise InputError("containment_survey needs at least one candidate material")
    return tuple(assess_containment(p, c, a, sf, criterion, pulse_duration) for c in items)


__all__ = [
    "REFERENCES", "CRITERIA", "FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH", "ELASTIC_RATIO_PRACTICAL",
    "PLASTIC_RATIO_LIMIT", "SHAKEDOWN_RATIO", "QUASI_STATIC_FACTOR", "COMPOUND_DEMONSTRATED_PRESSURE",
    "PULSED_MAGNET_CONTAINED_PRESSURE", "RAIL_INDUCTANCE_GRADIENT_RANGE", "TOKAMAK_TF_STRUCTURAL_STRESS",
    "COMMERCIAL_METAL_CATEGORIES", "LAMINATE_CATEGORIES", "LAMINATE_STRUCTURES", "DEFAULT_CANDIDATES",
    "lame_stresses", "max_hoop_stress", "von_mises_bore", "tresca_bore", "required_monolithic_strength",
    "required_radius_ratio", "elastic_limit_pressure", "fully_plastic_pressure", "shakedown_pressure",
    "autofrettage_ratio", "hoop_transit_time", "dynamic_regime",
    "RailLoad", "ideal_inductance_gradient", "rail_launcher_load", "coil_launcher_bore_pressure",
    "FusionMagnetLoad", "fusion_magnet_load", "MassDriverResult", "mass_driver",
    "ContainmentAssessment", "assess_containment", "containment_survey",
]
