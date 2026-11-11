"""Flow stress, forging pressure, adiabatic heating and forgeability of metals.

This is the constitutive backbone of magform.  It turns the concept note's
refractory-metal claim -- "a 2,000+ MPa magnetic press generates enough force
to plastically deform these hyper-hard refractory metals at room temperature
or moderate heat" -- into numbers with an honest status, and it supplies the
flow stress used by :mod:`magform.eddy` (tube compression),
:mod:`magform.microstructure`, :mod:`magform.porosity` and :mod:`magform.press`.

Flow-stress contract
--------------------
``flow_stress(material, strain, strain_rate=1e-3, temperature=T_ROOM)`` returns
the von Mises equivalent flow stress in Pa.  This module owns every guard, so
callers never pre-filter their arguments:

* ``strain < 0``, ``strain_rate < 0``, ``temperature <= 0``,
  ``temperature >= melting_point`` or any non-finite input raise
  :class:`~magform.errors.InputError`;
* a zero (or tiny) strain rate is the quasi-static limit: rates are floored at
  :data:`RATE_FLOOR` inside every logarithm;
* the primary published fit ``material.constitutive[0]`` (Johnson-Cook or
  Zerilli-Armstrong BCC, stored in :mod:`magform.materials`) is used when it
  exists, otherwise the Hollomon fallback :func:`fallback_flow_stress`;
* the result is never below :data:`MIN_FLOW_STRESS` (1e5 Pa), so pressure
  ratios and root brackets stay finite close to the melting point.

Physics that the assessment reports explicitly
----------------------------------------------
* **Hydrostatic pressure does not forge.**  A uniform pressure on a closed
  surface only compresses the volume elastically (dV/V = P/K, 1.3 % for W at
  4 GPa); plastic shape change needs a deviatoric stress, supplied by a die,
  a mandrel or a free surface.  Magnetic loading removes friction only on the
  coil side; the die or mandrel side keeps it.
* **Brittleness below the DBTT.**  Polycrystalline tungsten fails at 5-10 %
  strain in unconfined dynamic compression (Duemmer et al. 1998).  A
  superimposed hydrostatic pressure of the order of the flow stress
  suppresses fracture (pressure-induced ductility, Bridgman 1952), but this
  has not been shown at component scale at ~1e4 1/s, so it is labelled
  "extrapolation".
* **Adiabatic heating and shear localisation.**  Above ~1e2 1/s plastic work
  stays in the part (Taylor-Quinney fraction 0.9).  The Culver (1973)
  criterion gives the strain at which thermal softening overtakes strain
  hardening on the adiabatic path.  For W and its heavy alloy it is a few
  per cent at most, far below the 0.3 strain the concept note implies.
* **Practice gap.**  Conventional working temperatures (W 1200-1600 C, Mo
  1000-1300 C; Re, Ta and Nb cold worked) contradict the concept note's
  "near-melting temperatures (>2,500 C)" (:func:`forging_practice_gap`).

Pressures given to this module are the *mechanical* pressures delivered to the
workpiece surface; only :mod:`magform.eddy` and :mod:`magform.press` convert
fields into delivered pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from ._numerics import bisect
from ._report import Assessment, ResultMixin, unit
from .constants import T_ROOM, ZERO_CELSIUS
from .errors import InputError
from .materials import Material, MaterialLike, as_material
from .status import Status, worst

REFERENCES: dict[str, str] = {
    "johnson1983": "G. R. Johnson and W. H. Cook, 'A constitutive model and data for metals subjected to "
                   "large strains, high strain rates and high temperatures', Proc. 7th Int. Symp. Ballistics, "
                   "The Hague (1983) 541-547.",
    "zerilli1987": "F. J. Zerilli and R. W. Armstrong, 'Dislocation-mechanics-based constitutive relations for "
                   "material dynamics calculations', J. Appl. Phys. 61 (1987) 1816-1825.",
    "hollomon1945": "J. H. Hollomon, 'Tensile deformation', Trans. AIME 162 (1945) 268-290.",
    "taylor1934": "G. I. Taylor and H. Quinney, 'The latent energy remaining in a metal after cold working', "
                  "Proc. R. Soc. Lond. A 143 (1934) 307-326.",
    "culver1973": "R. S. Culver, 'Thermal instability strain in dynamic plastic deformation', in R. W. Rohde, "
                  "B. M. Butcher, J. R. Holland and C. H. Karnes (eds.), Metallurgical Effects at High Strain "
                  "Rates, Plenum, New York (1973).",
    "bai1992": "Y. Bai and B. Dodd, Adiabatic Shearing: Theory and Applications, Pergamon, Oxford (1992).",
    "wright2002": "T. W. Wright, The Physics and Mathematics of Adiabatic Shear Bands, Cambridge University "
                  "Press (2002).",
    "meyers1994": "M. A. Meyers, Dynamic Behavior of Materials, Wiley, New York (1994).",
    "bridgman1952": "P. W. Bridgman, Studies in Large Plastic Flow and Fracture with Special Emphasis on the "
                    "Effects of Hydrostatic Pressure, McGraw-Hill, New York (1952).",
    "duemmer1998": "T. Duemmer, J. C. LaSalvia, G. Ravichandran and M. A. Meyers, 'Effect of strain rate on "
                   "plastic flow and failure in polycrystalline tungsten', Acta Mater. 46 (1998) 6267-6290.",
    "wei2006": "Q. Wei, T. Jiao, K. T. Ramesh, E. Ma et al., 'Mechanical behavior and dynamic failure of "
               "high-strength ultrafine grained tungsten under uniaxial compression', Acta Mater. 54 (2006) "
               "77-87.",
    "chen1996": "S. R. Chen and G. T. Gray III, 'Constitutive behavior of tantalum and tantalum-tungsten "
                "alloys', Metall. Mater. Trans. A 27 (1996) 2994-3006.",
    "hill1950": "R. Hill, The Mathematical Theory of Plasticity, Clarendon Press, Oxford (1950).",
    "tabor1951": "D. Tabor, The Hardness of Metals, Clarendon Press, Oxford (1951).",
    "hosford2011": "W. F. Hosford and R. M. Caddell, Metal Forming: Mechanics and Metallurgy, 4th ed., "
                   "Cambridge University Press (2011).",
    "carslaw1959": "H. S. Carslaw and J. C. Jaeger, Conduction of Heat in Solids, 2nd ed., Oxford University "
                   "Press (1959).",
    "psyk2011": "V. Psyk, D. Risch, B. L. Kinsey, A. E. Tekkaya and M. Kleiner, 'Electromagnetic forming - "
                "a review', J. Mater. Process. Technol. 211 (2011) 787-829.",
    "lassner1999": "E. Lassner and W.-D. Schubert, Tungsten: Properties, Chemistry, Technology of the Element, "
                   "Alloys, and Chemical Compounds, Kluwer Academic/Plenum, New York (1999).",
    "asm14a": "ASM Handbook Vol. 14A, Metalworking: Bulk Forming, ASM International, Materials Park OH (2005): "
              "forging of refractory metals.",
    "astm_e8": "ASTM E8/E8M, Standard Test Methods for Tension Testing of Metallic Materials, ASTM "
               "International: 'room temperature' is 10-38 C.",
}

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
RATE_FLOOR: float = 1e-6
"""1/s.  Strain rates below this are evaluated at this value inside every logarithm
(quasi-static limit; a zero rate is not an error)."""

FALLBACK_REFERENCE_RATE: float = 1e-3
"""1/s.  Reference rate of the Hollomon fallback: the quasi-static tensile tests the
yield/UTS/elongation data come from (ASTM E8)."""

FALLBACK_RATE_SENSITIVITY: dict[str, float] = {"fcc": 0.015, "bcc": 0.04, "hcp": 0.025, "L1_0": 0.03}
"""Default rate coefficient C of the fallback, ``1 + C ln(rate/1e-3)``, by crystal structure
(confidence: estimated).  Representative of Johnson-Cook C values (fcc Cu 0.025, Ni alloys
~0.013; hcp Ti 0.014-0.03) and of the rate sensitivity m ~ 0.04 of coarse-grained bcc W
(Johnson & Cook 1983; Meyers 1994; Wei et al. 2006)."""

DEFAULT_RATE_SENSITIVITY: float = 0.02
"""Fallback rate coefficient C for crystal structures not in :data:`FALLBACK_RATE_SENSITIVITY`."""

TAYLOR_QUINNEY: float = 0.9
"""Fraction of plastic work converted to heat (Taylor & Quinney 1934)."""

MIN_FLOW_STRESS: float = 1e5
"""Pa.  Floor on every returned flow stress (keeps ratios and brackets finite near T_melt)."""

RATE_EXTRAPOLATION_FACTOR: float = 10.0
"""A fit is flagged as extrapolated in rate only beyond this factor outside its recorded
rate range (one decade; constitutive fits are routinely used that far)."""

FALLBACK_CALIBRATION_TEMPERATURES: tuple[float, float] = (283.15, 311.15)
"""K.  Temperature window of the fallback's tensile data: ASTM E8 'room temperature', 10-38 C."""

CULVER_START_STRAIN: float = 0.002
"""Strain at which the Culver march starts comparing stresses: the 0.2 % offset that defines
yield.  The fits do not resolve the elastic-plastic transition below it and the Hollomon
fallback is flat (sigma_y) there, which would make every fallback material spuriously
unstable at zero strain.  Heating from 0 to this strain is still integrated."""

ADIABATIC_STOP_FRACTION: float = 0.95
"""Adiabatic marches stop when the temperature reaches this fraction of T_melt."""

WARM_FORMING_MAX_FRACTION: float = 0.9
"""Upper end, as a fraction of T_melt, of the warm-forming temperature search."""

DEFAULT_LENGTH_SCALE: float = 1e-3
"""m.  Default conduction length for the adiabatic fraction (mm-scale parts and walls)."""

README_REFRACTORY_FORGING_CLAIM: float = 2500.0 + ZERO_CELSIUS
"""K.  The concept note's 'near-melting temperatures (>2,500 C)' for forging W, Mo and Re
(docs/archive/original-README.md)."""

PRACTICE_TOLERANCE: float = 10.0
"""K.  Tolerance when comparing a temperature with a recorded conventional working range
(the ranges are coarse; room-temperature ranges are recorded as 293 K)."""

FALLBACK_LABEL: str = "fallback"
"""Key of the Hollomon fallback in :attr:`FlowStressBand.values`."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class FlowStressBand(ResultMixin):
    """Flow stress of every model recorded for a material at one (strain, rate, T) state.

    ``values`` holds every published constitutive fit (keyed by its source label) plus
    ``"fallback"`` (Hollomon power law from tensile data) whenever strength data exist.
    ``minimum``/``maximum`` span the published fits listed in ``band_models``; the
    fallback defines the band only when no fit exists.  (It is a quasi-static power law
    extrapolated far beyond its uniform strain with an assumed rate coefficient: for
    90W-7Ni-3Fe it gives 2.94 GPa against the published 1.90 GPa at 0.3, 1e4 1/s.)
    ``primary`` is exactly :func:`flow_stress`.  ``extrapolated`` is True when the
    primary model's ``in_range()`` rejects the temperature or the rate lies more than
    :data:`RATE_EXTRAPOLATION_FACTOR` outside its recorded rate range; the fallback is
    calibrated only at room temperature and 1e-3 1/s.
    """

    _title: ClassVar[str] = "Flow-stress band"

    material: str
    strain: float
    strain_rate: float = unit("1/s")
    temperature: float = unit("K")
    minimum: float = unit("Pa")
    maximum: float = unit("Pa")
    primary: float = unit("Pa")
    primary_model: str
    band_models: tuple[str, ...]
    values: dict[str, float]
    extrapolated: bool
    extrapolation_note: str = ""

    @property
    def spread(self) -> float:
        """maximum/minimum (1.0 for a single model)."""
        return self.maximum / self.minimum


@dataclass(frozen=True, kw_only=True)
class PracticeGap(ResultMixin):
    """Where a proposed forming temperature sits relative to conventional working practice.

    ``conventional_range`` is ``material.conventional_working`` (K) and ``note`` its
    ``conventional_working_note``.  ``ratio_to_conventional_min`` is T / T_min.
    ``readme_claim_contradicted`` is set for refractory metals: True when conventional
    practice tops out below the concept note's '>2,500 C'.
    """

    _title: ClassVar[str] = "Forging practice gap"

    material: str
    temperature: float = unit("K")
    homologous_temperature: float
    conventional_range: tuple[float, float] | None = unit("K")
    conventional_homologous_range: tuple[float, float] | None
    below_conventional: bool
    ratio_to_conventional_min: float | None
    note: str
    verdict: str
    readme_claim_contradicted: bool | None = None


@dataclass(frozen=True, kw_only=True)
class ForgeAssessment(Assessment):
    """Can the delivered pressure plastically forge this material at this state?

    ``required_pressure`` = Q * sigma_max; ``margin`` = ``margin_conservative`` =
    P/(Q sigma_max); ``margin_optimistic`` = P/(Q sigma_min).  ``confinement_ratio`` =
    hydrostatic pressure / primary flow stress (>= 1: Bridgman-type confinement).
    ``failure_strain_limit`` is the strain used for the brittle-fracture rule
    (``dynamic_failure_strain`` when recorded, else tensile elongation).  See
    :func:`assess_forgeability` for the status rules.
    """

    _title: ClassVar[str] = "Forgeability assessment"

    material: str
    pressure: float = unit("Pa")
    target_strain: float
    strain_rate: float = unit("1/s")
    temperature: float = unit("K")
    constraint_factor: float
    flow_stress: float = unit("Pa")
    flow_stress_min: float = unit("Pa")
    flow_stress_max: float = unit("Pa")
    primary_model: str
    flow_stress_values: dict[str, float]
    required_pressure: float = unit("Pa")
    margin_conservative: float
    margin_optimistic: float
    max_strain_at_pressure: float
    adiabatic_fraction: float
    adiabatic_temperature_rise: float = unit("K")
    final_temperature: float = unit("K")
    dbtt: float | None = unit("K")
    below_dbtt: bool | None
    dynamic_failure_strain: float | None
    failure_strain_limit: float | None
    hydrostatic_pressure: float = unit("Pa")
    elastic_volumetric_strain: float
    confinement_ratio: float
    adiabatic_shear_strain: float | None
    practice_gap: PracticeGap
    warm_forming_temperature: float | None = unit("K")
    warm_forming_temperature_conservative: float | None = unit("K")
    extrapolated_model: bool


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(v):
        raise InputError(f"{name} must be finite, got {value!r}")
    return v


def _nonnegative(name: str, value: Any) -> float:
    v = _finite(name, value)
    if v < 0.0:
        raise InputError(f"{name} must be >= 0, got {v:g}")
    return v


def _positive(name: str, value: Any) -> float:
    v = _finite(name, value)
    if v <= 0.0:
        raise InputError(f"{name} must be > 0, got {v:g}")
    return v


def _fraction(name: str, value: Any) -> float:
    v = _finite(name, value)
    if not 0.0 <= v <= 1.0:
        raise InputError(f"{name} must lie in [0, 1], got {v:g}")
    return v


def _count(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise InputError(f"{name} must be a positive integer, got {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        raise InputError(f"{name} must be a positive integer, got {value!r}") from None
    if n != value or n < 1:
        raise InputError(f"{name} must be a positive integer, got {value!r}")
    return n


def _temperature(mat: Material, value: Any, name: str = "temperature") -> float:
    t = _finite(name, value)
    if t <= 0.0:
        raise InputError(f"{name} must be > 0 K, got {t:g}")
    if t >= mat.melting_point:
        raise InputError(
            f"{name} {t:.1f} K is at or above the melting point of '{mat.key}' "
            f"({mat.melting_point:.0f} K): no solid flow stress")
    return t


def _state(mat: Material, strain: Any, strain_rate: Any, temperature: Any) -> tuple[float, float, float]:
    return (_nonnegative("strain", strain), _nonnegative("strain_rate", strain_rate),
            _temperature(mat, temperature))


def _heat_capacity(mat: Material) -> float:
    rho_c = mat.density * mat.specific_heat
    if not (math.isfinite(rho_c) and rho_c > 0.0):
        raise InputError(f"material '{mat.key}' needs a positive density and specific heat")
    return rho_c


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Hollomon:
    """Fallback: max(sigma_y, K eps^n) * max(1, 1 + C ln(rate/1e-3)) * (1 - T*)."""

    sigma_y: float
    K: float
    n: float
    C: float
    t_melt: float

    def stress(self, strain: float, strain_rate: float, temperature: float) -> float:
        hard = self.K * strain ** self.n if strain > 0.0 else 0.0
        hard = hard if hard > self.sigma_y else self.sigma_y
        rate = strain_rate if strain_rate > RATE_FLOOR else RATE_FLOOR
        rate_factor = max(1.0, 1.0 + self.C * math.log(rate / FALLBACK_REFERENCE_RATE))
        if temperature <= T_ROOM:
            t_star = 0.0
        else:  # here T_ROOM < T < T_melt, so the denominator is positive
            t_star = min((temperature - T_ROOM) / (self.t_melt - T_ROOM), 0.999)
        return hard * rate_factor * (1.0 - t_star)


def _hollomon_fallback(mat: Material) -> _Hollomon:
    try:
        K, n = mat.hollomon()
    except InputError:
        raise InputError(
            f"material '{mat.key}' has no strength data for the Hollomon fallback "
            "(needs yield_strength, ultimate_strength and elongation, or hollomon_override)") from None
    K, n = float(K), float(n)
    if not (math.isfinite(K) and K > 0.0 and math.isfinite(n) and n >= 0.0):
        raise InputError(f"material '{mat.key}': invalid Hollomon parameters K={K!r}, n={n!r}")
    sigma_y = float(mat.yield_strength) if mat.yield_strength is not None else K * 0.002 ** n
    c = FALLBACK_RATE_SENSITIVITY.get(mat.crystal_structure, DEFAULT_RATE_SENSITIVITY)
    return _Hollomon(sigma_y, K, n, c, mat.melting_point)


def _checked(value: float, mat: Material) -> float:
    if not math.isfinite(value):
        raise InputError(f"flow stress of '{mat.key}' is not finite at these conditions")
    return value if value > MIN_FLOW_STRESS else MIN_FLOW_STRESS


def _fit_stress(model: Any, mat: Material, strain: float, rate: float, temperature: float) -> float:
    try:
        s = float(model.stress(strain, rate, temperature, T_ROOM, mat.melting_point))
    except ArithmeticError:
        raise InputError(
            f"constitutive model of '{mat.key}' cannot be evaluated at strain={strain:g}, "
            f"rate={rate:g} 1/s, T={temperature:g} K") from None
    return _checked(s, mat)


def _fallback_stress(fb: _Hollomon, mat: Material, strain: float, rate: float, temperature: float) -> float:
    try:
        s = fb.stress(strain, rate, temperature)
    except ArithmeticError:
        raise InputError(f"Hollomon fallback of '{mat.key}' cannot be evaluated at strain={strain:g}") from None
    return _checked(s, mat)


def _primary_curve(mat: Material, strain_rate: float) -> Callable[[float, float], float]:
    """sigma(strain, temperature) of the primary model at a fixed rate (no validation)."""
    rate = strain_rate if strain_rate > RATE_FLOOR else RATE_FLOOR
    if mat.constitutive:
        model = mat.constitutive[0]

        def curve(strain: float, temperature: float) -> float:
            return _fit_stress(model, mat, strain, rate, temperature)
    else:
        fb = _hollomon_fallback(mat)

        def curve(strain: float, temperature: float) -> float:
            return _fallback_stress(fb, mat, strain, rate, temperature)
    return curve


def _labelled_fits(mat: Material) -> list[tuple[str, Any]]:
    """(label, model) for each published fit; labels are the model sources, made unique."""
    out: list[tuple[str, Any]] = []
    used = {FALLBACK_LABEL}
    for i, model in enumerate(mat.constitutive):
        base = str(getattr(model, "source", "") or "").strip() or f"{getattr(model, 'kind', 'model')} #{i}"
        label = base if base not in used else f"{base} #{i}"
        used.add(label)
        out.append((label, model))
    return out


def _band_values(mat: Material, strain: float, strain_rate: float,
                 temperature: float) -> tuple[dict[str, float], tuple[str, ...], Any | None]:
    """(values, labels spanning the band, primary model or None for the fallback)."""
    rate = strain_rate if strain_rate > RATE_FLOOR else RATE_FLOOR
    fits = _labelled_fits(mat)
    values = {label: _fit_stress(model, mat, strain, rate, temperature) for label, model in fits}
    try:
        fb: _Hollomon | None = _hollomon_fallback(mat)
    except InputError:
        if not fits:
            raise
        fb = None
    if fb is not None:
        values[FALLBACK_LABEL] = _fallback_stress(fb, mat, strain, rate, temperature)
    if fits:
        return values, tuple(label for label, _ in fits), fits[0][1]
    return values, (FALLBACK_LABEL,), None


def _calibration(model: Any | None, strain_rate: float, temperature: float) -> tuple[bool, str]:
    """(extrapolated, reason) for the primary model (None = Hollomon fallback)."""
    rate = strain_rate if strain_rate > RATE_FLOOR else RATE_FLOOR
    if model is None:
        name = "Hollomon fallback (room-temperature quasi-static tensile data, assumed rate coefficient)"
        rate_range: tuple[float, float] | None = (FALLBACK_REFERENCE_RATE, FALLBACK_REFERENCE_RATE)
        t_range: tuple[float, float] | None = FALLBACK_CALIBRATION_TEMPERATURES
        t_ok = t_range[0] <= temperature <= t_range[1]
    else:
        name = str(getattr(model, "kind", "constitutive model"))
        rate_range = getattr(model, "rate_range", None)
        t_range = getattr(model, "temperature_range", None)
        probe = rate if rate_range is None else min(max(rate, rate_range[0]), rate_range[1])
        in_range = getattr(model, "in_range", None)
        t_ok = True if in_range is None else bool(in_range(probe, temperature))
    r_ok = True
    if rate_range is not None:
        lo, hi = rate_range[0] / RATE_EXTRAPOLATION_FACTOR, rate_range[1] * RATE_EXTRAPOLATION_FACTOR
        r_ok = lo <= rate <= hi
    reasons: list[str] = []
    if not t_ok:
        window = f"{t_range[0]:.0f}-{t_range[1]:.0f} K" if t_range is not None else "recorded"
        reasons.append(f"T = {temperature:.0f} K is outside its {window} calibration")
    if not r_ok and rate_range is not None:
        reasons.append(f"strain rate {rate:.3g} 1/s is more than {RATE_EXTRAPOLATION_FACTOR:g}x outside its "
                       f"{rate_range[0]:.3g}-{rate_range[1]:.3g} 1/s calibration")
    if not reasons:
        return False, ""
    return True, f"{name}: " + "; ".join(reasons)


def _adiabatic_fraction(mat: Material, strain: float, strain_rate: float, length_scale: float) -> float:
    if strain_rate == 0.0:
        return 0.0
    if strain == 0.0:
        return 1.0
    alpha = mat.thermal_conductivity / _heat_capacity(mat)
    if alpha <= 0.0:
        return 1.0
    t_def = strain / strain_rate
    t_cond = length_scale * length_scale / alpha
    return -math.expm1(-t_cond / t_def)


def _lowest_temperature(excess: Callable[[float], float], lo: float, hi: float) -> float | None:
    """Smallest T in [lo, hi] with excess(T) <= 0 for a decreasing excess(T)."""
    if excess(lo) <= 0.0:
        return lo
    if hi <= lo or excess(hi) > 0.0:
        return None
    for _ in range(200):
        if hi - lo <= 1e-7 * hi:
            break
        mid = 0.5 * (lo + hi)
        if excess(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def _gpa(value: float) -> str:
    return f"{value / 1e9:.3g} GPa"


# ---------------------------------------------------------------------------
# Flow stress
# ---------------------------------------------------------------------------
def fallback_rate_sensitivity(material: MaterialLike) -> float:
    """Rate coefficient C used by the Hollomon fallback for this material's crystal structure.

    ``C = FALLBACK_RATE_SENSITIVITY[crystal_structure]`` (fcc 0.015, bcc 0.04, hcp 0.025,
    L1_0 0.03), else :data:`DEFAULT_RATE_SENSITIVITY` (0.02).  Dimensionless.  These are
    estimates representative of Johnson-Cook fits (Johnson & Cook 1983; Meyers 1994), not
    measurements for the specific alloy.
    """
    mat = as_material(material)
    return FALLBACK_RATE_SENSITIVITY.get(mat.crystal_structure, DEFAULT_RATE_SENSITIVITY)


def fallback_flow_stress(material: MaterialLike, strain: float, strain_rate: float = FALLBACK_REFERENCE_RATE,
                         temperature: float = T_ROOM) -> float:
    """Hollomon fallback flow stress (Pa) from tensile data.

    sigma = max(sigma_y, K eps^n) * max(1, 1 + C ln(rate/1e-3)) * (1 - T*),
    T* = (T - T_ROOM)/(T_melt - T_ROOM) clamped to [0, 0.999],

    with (K, n) = ``material.hollomon()`` (two-point power law through the 0.2 % yield and
    the true necking point, Hollomon 1945) and C from :func:`fallback_rate_sensitivity`.
    Units: strain dimensionless, rate 1/s, T in K.

    Validity: quasi-static room-temperature tensile data only.  The rate term is an
    assumed log-linear sensitivity (no strengthening below 1e-3 1/s), the thermal term is
    linear Johnson-Cook softening with m = 1, no low-temperature strengthening is modelled
    below T_ROOM, and strains beyond the uniform elongation are extrapolations.  Never
    returns less than :data:`MIN_FLOW_STRESS`.  Raises :class:`InputError` for invalid
    states or when the material has no yield/UTS/elongation (or ``hollomon_override``).
    Reference: Hollomon (1945); Johnson & Cook (1983) for the rate/thermal form.
    """
    mat = as_material(material)
    eps, rate, t = _state(mat, strain, strain_rate, temperature)
    return _fallback_stress(_hollomon_fallback(mat), mat, eps, rate, t)


def flow_stress(material: MaterialLike, strain: float, strain_rate: float = 1e-3,
                temperature: float = T_ROOM) -> float:
    """Equivalent (von Mises) flow stress in Pa: the constitutive backbone of magform.

    sigma = ``material.constitutive[0].stress(strain, rate, T, T_ROOM, T_melt)`` when the
    material has a published fit, i.e. Johnson-Cook
    ``(A + B eps^n)(1 + C ln(rate/rate0))(1 - T*^m)`` or Zerilli-Armstrong BCC
    ``c0 + c1 exp(-c3 T + c4 T ln rate) + c5 eps^n``; otherwise
    :func:`fallback_flow_stress`.  Units: strain dimensionless, ``strain_rate`` 1/s,
    ``temperature`` K, result Pa.

    Guards (owned here; callers never pre-filter): strain < 0, rate < 0, T <= 0,
    T >= T_melt or non-finite inputs raise :class:`InputError`; the rate is floored at
    :data:`RATE_FLOOR` (zero rate = quasi-static limit); the result is >= 1e5 Pa.
    Validity: each fit only within its calibration window (see
    :func:`flow_stress_band` ``extrapolated``); isothermal at the given T.
    References: Johnson & Cook (1983); Zerilli & Armstrong (1987); Hollomon (1945).
    """
    mat = as_material(material)
    eps, rate, t = _state(mat, strain, strain_rate, temperature)
    return _primary_curve(mat, rate)(eps, t)


def flow_stress_band(material: MaterialLike, strain: float, strain_rate: float = 1e-3,
                     temperature: float = T_ROOM) -> FlowStressBand:
    """Flow stress (Pa) of every recorded model at one state, as an uncertainty band.

    values[label] = sigma_model(strain, rate, T) for each published fit in
    ``material.constitutive`` plus ``values["fallback"]`` (:func:`fallback_flow_stress`)
    when strength data exist.  minimum/maximum span the published fits (the fallback only
    when no fit exists); primary = :func:`flow_stress`.  ``extrapolated`` is True when the
    primary model's ``in_range()`` rejects T or the rate is more than 10x outside its
    recorded rate range (fallback window: 1e-3 1/s, 283-311 K).

    Units: strain dimensionless, rate 1/s, T in K, stresses in Pa.  Validity: the spread
    between published fits (e.g. 2.05-2.58 GPa for wrought W at 0.3, 1e4 1/s, 293 K) is
    material-batch uncertainty, not a confidence interval.  References: Zerilli &
    Armstrong (1987); Johnson & Cook (1983); Chen & Gray (1996).
    """
    mat = as_material(material)
    eps, rate, t = _state(mat, strain, strain_rate, temperature)
    values, band_labels, primary_model = _band_values(mat, eps, rate, t)
    band = [values[label] for label in band_labels]
    extrapolated, why = _calibration(primary_model, rate, t)
    return FlowStressBand(
        material=mat.key, strain=eps, strain_rate=rate, temperature=t,
        minimum=min(band), maximum=max(band), primary=values[band_labels[0]],
        primary_model=band_labels[0], band_models=band_labels, values=values,
        extrapolated=extrapolated, extrapolation_note=why,
    )


def forging_pressure(material: MaterialLike, strain: float, strain_rate: float = 1e-3,
                     temperature: float = T_ROOM, constraint_factor: float = 1.0, friction: float = 0.0,
                     diameter_to_height: float = 0.0) -> float:
    """Average die pressure (Pa) needed to keep the workpiece flowing.

    p = Q * sigma * (1 + mu * (d/h) / 3)

    sigma = :func:`flow_stress`; Q = constraint factor (1 uniaxial, 2/sqrt(3) = 1.155 plane
    strain, ~3 for indentation-like constraint; Hill 1950, Tabor 1951); the bracket is the
    slab-method friction hill for upsetting a cylinder of diameter d and height h with
    Coulomb friction mu (Hosford & Caddell 2011).  Units: Pa, dimensionless mu and d/h.

    Physics caveat: a uniform pressure on a closed surface is hydrostatic and causes no
    shape change (only an elastic volume change P/K); a die, mandrel or free surface must
    make the stress deviatoric.  Magnetic loading removes friction only on the coil side;
    the die or mandrel side keeps its friction.  Validity: sliding friction with
    mu*d/h <~ 1 (sticking friction saturates the friction hill).
    """
    mat = as_material(material)
    q = _positive("constraint_factor", constraint_factor)
    mu = _nonnegative("friction", friction)
    dh = _nonnegative("diameter_to_height", diameter_to_height)
    return q * flow_stress(mat, strain, strain_rate, temperature) * (1.0 + mu * dh / 3.0)


# ---------------------------------------------------------------------------
# Adiabatic heating and localisation
# ---------------------------------------------------------------------------
def adiabatic_fraction(material: MaterialLike, strain: float, strain_rate: float,
                       length_scale: float = DEFAULT_LENGTH_SCALE) -> float:
    """Fraction of the plastic heat retained in the deforming region (0 = isothermal, 1 = adiabatic).

    f = 1 - exp(-t_cond / t_def),  t_def = strain/strain_rate,  t_cond = L^2/alpha_th,
    alpha_th = k/(rho c_p).

    Units: L in m, rate 1/s, result dimensionless.  For mm-scale W (alpha ~ 6.7e-5 m^2/s,
    t_cond ~ 15 ms) a 0.3 strain at 1e4 1/s (30 us) is fully adiabatic, at 1e-3 1/s
    isothermal.  Limits: rate = 0 gives 0, strain = 0 (instantaneous) gives 1.  Validity:
    order-of-magnitude conduction scaling (Carslaw & Jaeger 1959); the crossover is
    usually placed at 1e2-1e3 1/s (Meyers 1994).
    """
    mat = as_material(material)
    eps = _nonnegative("strain", strain)
    rate = _nonnegative("strain_rate", strain_rate)
    length = _positive("length_scale", length_scale)
    return _adiabatic_fraction(mat, eps, rate, length)


def adiabatic_temperature_rise(material: MaterialLike, strain: float, strain_rate: float,
                               temperature: float = T_ROOM, taylor_quinney: float = TAYLOR_QUINNEY,
                               length_scale: float = DEFAULT_LENGTH_SCALE, n_steps: int = 200) -> float:
    """Bulk temperature rise (K) from plastic work up to ``strain``.

    dT = f * beta * sigma(eps, rate, T) d eps / (rho c_p),

    integrated in ``n_steps`` equal strain steps (trapezoidal predictor-corrector, T
    updated every step so thermal softening feeds back), with f = :func:`adiabatic_fraction`
    and beta the Taylor-Quinney coefficient (0.9; Taylor & Quinney 1934).  The march stops
    at 0.95 T_melt.  Units: K.

    Validity: uniform deformation (no localisation, see
    :func:`adiabatic_shear_critical_strain`); bulk heating only (eddy-current skin heating
    is :func:`magform.eddy.skin_temperature_rise`); f is a lumped conduction estimate.
    """
    mat = as_material(material)
    eps, rate, t0 = _state(mat, strain, strain_rate, temperature)
    beta = _fraction("taylor_quinney", taylor_quinney)
    length = _positive("length_scale", length_scale)
    n = _count("n_steps", n_steps)
    rho_c = _heat_capacity(mat)
    f = _adiabatic_fraction(mat, eps, rate, length)
    t_cap = ADIABATIC_STOP_FRACTION * mat.melting_point
    if eps == 0.0 or beta == 0.0 or f == 0.0 or t0 >= t_cap:
        return 0.0
    curve = _primary_curve(mat, rate)
    k = f * beta / rho_c
    d = eps / n
    t = t0
    s = curve(0.0, t)
    for i in range(n):
        e1 = eps if i == n - 1 else (i + 1) * d
        t_pred = min(t + k * s * d, t_cap)
        s_pred = curve(e1, t_pred)
        t_new = t + 0.5 * k * (s + s_pred) * d
        if t_new >= t_cap:
            t = t_cap
            break
        t = t_new
        s = curve(e1, t)
    return t - t0


def adiabatic_shear_critical_strain(material: MaterialLike, strain_rate: float, temperature: float = T_ROOM,
                                    taylor_quinney: float = TAYLOR_QUINNEY, max_strain: float = 5.0,
                                    d_eps: float = 1e-3) -> float | None:
    """Culver (1973) adiabatic-instability strain, or None if none occurs before ``max_strain``.

    Along the fully adiabatic path (f = 1) dT/d eps = beta sigma/(rho c_p), the
    instability is where d sigma/d eps = (d sigma/d eps)_T + (d sigma/d T)(dT/d eps) = 0.
    The path is marched in steps ``d_eps`` and the first eps with
    sigma(eps + d_eps, T + dT) <= sigma(eps, T) is returned.  Comparison starts at
    :data:`CULVER_START_STRAIN` (0.2 % offset yield); heating below it is integrated.  If
    the path reaches 0.95 T_melt first, that strain is returned (thermal runaway).  For
    Johnson-Cook with m = 1 the criterion reduces to
    B n eps^(n-1) = beta R (A + B eps^n)^2 / (rho c_p (T_m - T_r)).

    Units: strain dimensionless, rate 1/s, T in K.  Validity: Culver's strain is the
    maximum of the adiabatic stress-strain curve, a necessary condition for shear banding;
    bands usually form at several times this strain (Bai & Dodd 1992; Wright 2002).
    Johnson-Cook fits with m < 1 have an infinite softening slope at T* = 0, which makes
    the value small when T starts at the fit's reference temperature.  Results at 1e4 1/s,
    293 K: W heavy alloy 90W-7Ni-3Fe 0.017 (0.0165 from the closed form; 1723 K matrix
    solidus as T_m), wrought W (Lennon-Ramesh Zerilli-Armstrong) 0.004.
    """
    mat = as_material(material)
    rate = _nonnegative("strain_rate", strain_rate)
    t0 = _temperature(mat, temperature)
    beta = _fraction("taylor_quinney", taylor_quinney)
    e_max = _positive("max_strain", max_strain)
    de = _positive("d_eps", d_eps)
    if de >= e_max:
        raise InputError(f"d_eps ({de:g}) must be smaller than max_strain ({e_max:g})")
    return _culver(mat, rate, t0, beta, e_max, de)[0]


def _culver(mat: Material, rate: float, t0: float, beta: float, e_max: float,
            de: float) -> tuple[float | None, float]:
    """(Culver strain or None, temperature reached there or at e_max); inputs validated."""
    rho_c = _heat_capacity(mat)
    curve = _primary_curve(mat, rate)
    t_cap = ADIABATIC_STOP_FRACTION * mat.melting_point
    if t0 >= t_cap:
        return 0.0, t0
    k = beta / rho_c

    def advance(e0: float, e1: float, t: float, s: float) -> float:
        h = e1 - e0
        t_pred = min(t + k * s * h, t_cap)
        return t + 0.5 * k * (s + curve(e1, t_pred)) * h

    e_start = min(CULVER_START_STRAIN, 0.5 * e_max)
    e = e_start
    t = advance(0.0, e, t0, curve(0.0, t0))
    if t >= t_cap:
        return e, t
    s = curve(e, t)
    i = 0
    while e < e_max:
        i += 1
        e1 = min(e_start + i * de, e_max)
        t1 = advance(e, e1, t, s)
        if t1 >= t_cap:
            return e, t
        s1 = curve(e1, t1)
        if s1 <= s:
            return e, t
        e, t, s = e1, t1, s1
    return None, t


def _softening_singularity(mat: Material, temperature: float) -> str:
    """Explanation when the primary fit is Johnson-Cook with m < 1 and T is within 1 % of T* = 0."""
    if not mat.constitutive:
        return ""
    model = mat.constitutive[0]
    m = getattr(model, "m", None)
    if getattr(model, "kind", "") != "johnson_cook" or m is None or m >= 1.0:
        return ""
    t_ref = model.t_ref if model.t_ref is not None else T_ROOM
    t_melt = model.t_melt if model.t_melt is not None else mat.melting_point
    if (temperature - t_ref) / (t_melt - t_ref) >= 0.01:
        return ""
    return (f" (the primary Johnson-Cook fit has m = {m:g} < 1, whose softening slope is infinite at its "
            f"{t_ref:.0f} K reference temperature; this early instability is probably a fitting artifact, "
            "compare the other fits)")


# ---------------------------------------------------------------------------
# Pressure / temperature windows
# ---------------------------------------------------------------------------
def max_strain_at_pressure(material: MaterialLike, pressure: float, strain_rate: float = 1e4,
                           temperature: float = T_ROOM, constraint_factor: float = 1.0,
                           max_strain: float = 3.0) -> float:
    """Largest strain whose flow stress the pressure can still overcome: Q sigma(eps) = P.

    Returns 0 if P < Q sigma(0) (no plastic flow), ``max_strain`` if
    P >= Q sigma(max_strain), else the bisection root of Q sigma(eps, rate, T) - P.
    sigma = :func:`flow_stress` (isothermal at ``temperature``; strain hardening is
    monotone for all models).  Units: Pa, 1/s, K; result dimensionless.

    Validity: isothermal hardening curve; adiabatic softening (see
    :func:`adiabatic_temperature_rise`) raises the reachable strain, localisation
    (:func:`adiabatic_shear_critical_strain`) caps it.  The pressure is the delivered
    mechanical pressure.  Reference: Hill (1950) for the constraint factor.
    """
    mat = as_material(material)
    p = _nonnegative("pressure", pressure)
    q = _positive("constraint_factor", constraint_factor)
    e_max = _positive("max_strain", max_strain)
    _, rate, t = _state(mat, 0.0, strain_rate, temperature)
    curve = _primary_curve(mat, rate)

    def excess(e: float) -> float:
        return q * curve(e, t) - p

    if excess(0.0) > 0.0:
        return 0.0
    if excess(e_max) <= 0.0:
        return e_max
    return bisect(excess, 0.0, e_max, tol=1e-10)


def warm_forming_temperature(material: MaterialLike, pressure: float, strain: float, strain_rate: float,
                             constraint_factor: float = 1.0, *, conservative: bool = False) -> float | None:
    """Lowest temperature (K) in [T_ROOM, 0.9 T_melt] at which Q sigma(strain, rate, T) <= P.

    Returns T_ROOM if the pressure already suffices at room temperature and None if even
    0.9 T_melt is not enough; otherwise bisects the monotone decreasing function
    Q sigma(T) - P (to 1e-7 relative).  sigma = :func:`flow_stress` (primary model), or the
    band maximum of :func:`flow_stress_band` when ``conservative`` is True.  Units: Pa, K.

    Validity: isothermal preheat (ignores adiabatic heating during the stroke and die
    chilling); the fits are calibrated only up to their recorded temperatures (W
    Zerilli-Armstrong: 1273 K), beyond which the answer is an extrapolation.  Reference:
    Johnson & Cook (1983); Zerilli & Armstrong (1987).
    """
    mat = as_material(material)
    p = _nonnegative("pressure", pressure)
    q = _positive("constraint_factor", constraint_factor)
    eps, rate, t_lo = _state(mat, strain, strain_rate, T_ROOM)
    t_hi = WARM_FORMING_MAX_FRACTION * mat.melting_point
    if conservative:
        def sigma(t: float) -> float:
            values, labels, _ = _band_values(mat, eps, rate, t)
            return max(values[label] for label in labels)
    else:
        curve = _primary_curve(mat, rate)

        def sigma(t: float) -> float:
            return curve(eps, t)
    return _lowest_temperature(lambda t: q * sigma(t) - p, t_lo, t_hi)


def forging_practice_gap(material: MaterialLike, temperature: float = T_ROOM) -> PracticeGap:
    """Compare a proposed forming temperature with conventional working practice.

    ratio = T / T_conv,min and homologous temperature T/T_melt, with the conventional
    range from ``material.conventional_working`` (W 1473-1873 K = 1200-1600 C,
    Lassner & Schubert 1999; Mo 1273-1573 K; Re, Ta, Nb cold worked; ASM Handbook 14A).
    ``below_conventional`` is T < T_min - 10 K (False when no range is recorded).  For
    refractory metals ``readme_claim_contradicted`` records whether practice tops out
    below the concept note's '>2,500 C' (2773 K).

    Units: K.  Validity: the recorded ranges are coarse industrial practice, not limits;
    a temperature below them is a departure from practice, not an impossibility.
    """
    mat = as_material(material)
    t = _temperature(mat, temperature)
    tm = mat.melting_point
    rng = mat.conventional_working
    note = mat.conventional_working_note
    homologous = t / tm
    if rng is None:
        return PracticeGap(
            material=mat.key, temperature=t, homologous_temperature=homologous, conventional_range=None,
            conventional_homologous_range=None, below_conventional=False, ratio_to_conventional_min=None,
            note=note, verdict=f"no conventional working range is recorded for {mat.name}; practice gap not "
                               "evaluated", readme_claim_contradicted=None)
    lo, hi = float(rng[0]), float(rng[1])
    if hi - lo < 1.0:
        span = f"{lo:.0f} K ({lo - ZERO_CELSIUS:.0f} C, {lo / tm:.2f} Tm)"
    else:
        span = (f"{lo:.0f}-{hi:.0f} K ({lo - ZERO_CELSIUS:.0f}-{hi - ZERO_CELSIUS:.0f} C, "
                f"{lo / tm:.2f}-{hi / tm:.2f} Tm)")
    below = t < lo - PRACTICE_TOLERANCE
    ratio = t / lo
    if below:
        verdict = (f"{t:.0f} K is {lo - t:.0f} K below the conventional working range of {mat.name}, {span}: "
                   f"T/T_min = {ratio:.2f}; forming here departs from industrial practice")
    elif t <= hi + PRACTICE_TOLERANCE:
        verdict = f"{t:.0f} K lies within conventional practice for {mat.name}, {span}: no practice gap"
    else:
        verdict = f"{t:.0f} K is above the conventional working range of {mat.name}, {span}"
    if note:
        verdict += f" (practice: {note})"
    contradicted: bool | None = None
    if mat.category == "refractory":
        contradicted = hi < README_REFRACTORY_FORGING_CLAIM
        if contradicted:
            practice = ("works it cold, at room temperature" if hi <= T_ROOM + PRACTICE_TOLERANCE
                        else f"tops out at {hi - ZERO_CELSIUS:.0f} C")
            verdict += (f"; conventional practice {practice}, so the concept note's 'near-melting temperatures "
                        "(>2,500 C)' overstates it")
    return PracticeGap(
        material=mat.key, temperature=t, homologous_temperature=homologous, conventional_range=(lo, hi),
        conventional_homologous_range=(lo / tm, hi / tm), below_conventional=below,
        ratio_to_conventional_min=ratio, note=note, verdict=verdict, readme_claim_contradicted=contradicted)


# ---------------------------------------------------------------------------
# Forgeability assessment
# ---------------------------------------------------------------------------
def assess_forgeability(material: MaterialLike, pressure: float, target_strain: float = 0.3,
                        strain_rate: float = 1e4, temperature: float = T_ROOM, constraint_factor: float = 1.2,
                        hydrostatic_pressure: float | None = None) -> ForgeAssessment:
    """Assess whether a delivered pressure can forge ``material`` to ``target_strain``.

    Required pressure Q * sigma_max with sigma from :func:`flow_stress_band` at
    (target_strain, rate, T); margins P/(Q sigma_max) (conservative, = ``margin``) and
    P/(Q sigma_min) (optimistic); confinement ratio = P_hydro/sigma_flow
    (``hydrostatic_pressure`` defaults to ``pressure``).  Units: Pa, 1/s, K.

    Status (first rule that fires, worst-combined with every later rule that also fires):

    1. margin_optimistic < 1 -> infeasible ("pressure insufficient at this temperature";
       the warm-forming temperature is given in the notes);
    2. below the DBTT, target strain > failure strain (``dynamic_failure_strain`` or
       elongation) and confinement ratio < 1 -> infeasible ("fractures unless confined";
       Duemmer et al. 1998);
    3. below the DBTT but confined (ratio >= 1) -> extrapolation (pressure-induced
       ductility, Bridgman 1952; not demonstrated at component scale);
    4. margin_conservative < 1 <= margin_optimistic -> extrapolation (data band straddles);
    5. Culver strain < target strain -> extrapolation + warning (localisation risk);
    6. primary model extrapolated -> extrapolation + warning;
    7. otherwise state of the art.

    The hydrostatic/deviatoric caveat and the practice-gap verdict are always in
    ``notes``.  Validity: bulk, uniform deformation; eddy-current skin heating is not
    included (:mod:`magform.eddy`); the recorded DBTT is quasi-static and rises with rate.
    """
    mat = as_material(material)
    p = _nonnegative("pressure", pressure)
    eps, rate, t = _state(mat, target_strain, strain_rate, temperature)
    q = _positive("constraint_factor", constraint_factor)
    p_hydro = p if hydrostatic_pressure is None else _nonnegative("hydrostatic_pressure", hydrostatic_pressure)

    band = flow_stress_band(mat, eps, rate, t)
    sigma, s_min, s_max = band.primary, band.minimum, band.maximum
    required = q * s_max
    m_cons = p / required
    m_opt = p / (q * s_min)
    eps_reach = max_strain_at_pressure(mat, p, rate, t, q)
    f_ad = adiabatic_fraction(mat, eps, rate)
    d_t = adiabatic_temperature_rise(mat, eps, rate, t)
    t_final = t + d_t
    dbtt = mat.dbtt
    below = None if dbtt is None else t < dbtt
    dfs = mat.dynamic_failure_strain
    fail = dfs if dfs is not None else mat.elongation
    confinement = p_hydro / sigma
    eps_c, t_c = _culver(mat, rate, t, TAYLOR_QUINNEY, 5.0, 1e-3)
    gap = forging_practice_gap(mat, t)
    t_warm = warm_forming_temperature(mat, p, eps, rate, q)
    t_warm_cons = warm_forming_temperature(mat, p, eps, rate, q, conservative=True)
    vol = mat.volumetric_strain(p_hydro)

    fired: list[Status] = []
    warnings: list[str] = []
    notes: list[str] = [
        (f"Uniform pressure on a closed surface is hydrostatic: it compresses {mat.key} elastically by "
         f"dV/V = P/K = {vol:.2%} but causes no shape change. Plastic forming needs a deviatoric stress from a "
         "die, mandrel or free surface; magnetic loading removes friction only on the coil side, the die or "
         "mandrel side keeps it."),
        f"Practice gap: {gap.verdict}.",
        (f"Required pressure = Q x sigma_max = {q:g} x {_gpa(s_max)} = {_gpa(required)} "
         f"(flow-stress band {_gpa(s_min)}-{_gpa(s_max)}, primary {_gpa(sigma)}; Q = 1 uniaxial, 1.155 plane "
         "strain, ~3 indentation-like)."),
        (f"Adiabatic fraction {f_ad:.2f} over a 1 mm conduction length; bulk plastic heating "
         f"+{d_t:.0f} K (Taylor-Quinney 0.9) to {t_final:.0f} K. Eddy-current skin heating is not included "
         "(magform.eddy.skin_temperature_rise)."),
    ]

    # rule 1 -- pressure versus the weakest model
    if m_opt < 1.0:
        fired.append(Status.INFEASIBLE)
        warnings.append(f"pressure insufficient at this temperature: {_gpa(p)} < Q x sigma_min = "
                        f"{_gpa(q * s_min)} (margin_optimistic {m_opt:.2f})")
        if t_warm is None:
            notes.append(f"No preheat up to {WARM_FORMING_MAX_FRACTION:g} Tm "
                         f"({WARM_FORMING_MAX_FRACTION * mat.melting_point:.0f} K) brings Q x sigma below the "
                         "pressure.")
        else:
            text = (f"Warm forming: Q x sigma (primary model) falls to the pressure at {t_warm:.0f} K "
                    f"({t_warm - ZERO_CELSIUS:.0f} C)")
            if t_warm_cons is not None and t_warm_cons > t_warm + 0.5:
                text += f"; with the strongest published fit at {t_warm_cons:.0f} K"
            elif t_warm_cons is None:
                text += f"; the strongest published fit never does below {WARM_FORMING_MAX_FRACTION:g} Tm"
            warm_extrap, warm_why = _calibration(mat.constitutive[0] if mat.constitutive else None, rate, t_warm)
            if warm_extrap:
                text += f" (extrapolated: {warm_why})"
            notes.append(text + ".")

    # rules 2/3 -- brittleness below the ductile-brittle transition
    if below:
        if fail is not None and eps > fail and confinement < 1.0:
            fired.append(Status.INFEASIBLE)
            warnings.append(f"fractures unless confined: {t:.0f} K is below the DBTT ({dbtt:.0f} K), the target "
                            f"strain {eps:g} exceeds the failure strain {fail:g} and P_hydro/sigma = "
                            f"{confinement:.2f} < 1")
        elif confinement >= 1.0:
            fired.append(Status.EXTRAPOLATION)
            warnings.append(f"below the DBTT ({dbtt:.0f} K) but confined (P_hydro/sigma = {confinement:.2f} >= 1): "
                            "relies on pressure-induced ductility (Bridgman), not demonstrated at component scale "
                            "and ~1e4 1/s")
        elif fail is None:
            warnings.append(f"below the DBTT ({dbtt:.0f} K), unconfined, and no failure-strain data: brittle "
                            "fracture possible")
        else:
            warnings.append(f"below the DBTT ({dbtt:.0f} K) and unconfined: the target strain {eps:g} is within the "
                            f"failure strain {fail:g}, little ductility reserve")
        notes.append("The recorded DBTT is a quasi-static value; bcc transition temperatures rise with strain rate.")

    # rule 4 -- the flow-stress data band straddles the pressure
    if m_cons < 1.0 <= m_opt:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"flow-stress data band straddles the pressure: margin {m_cons:.2f} (strongest fit) to "
                        f"{m_opt:.2f} (weakest fit)")

    # rule 5 -- adiabatic shear localisation
    if eps_c is not None and eps_c < eps:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"adiabatic shear localisation risk: Culver instability at strain {eps_c:.3g} < target "
                        f"{eps:g} at {rate:.3g} 1/s" + _softening_singularity(mat, t_c))

    # rule 6 -- primary model outside its calibration
    if band.extrapolated:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"flow-stress model extrapolated: {band.extrapolation_note}")

    status = worst(fired) if fired else Status.STATE_OF_THE_ART
    return ForgeAssessment(
        status=status, margin=m_cons, warnings=tuple(warnings), notes=tuple(notes),
        material=mat.key, pressure=p, target_strain=eps, strain_rate=rate, temperature=t,
        constraint_factor=q, flow_stress=sigma, flow_stress_min=s_min, flow_stress_max=s_max,
        primary_model=band.primary_model, flow_stress_values=dict(band.values), required_pressure=required,
        margin_conservative=m_cons, margin_optimistic=m_opt, max_strain_at_pressure=eps_reach,
        adiabatic_fraction=f_ad, adiabatic_temperature_rise=d_t, final_temperature=t_final, dbtt=dbtt,
        below_dbtt=below, dynamic_failure_strain=dfs, failure_strain_limit=fail, hydrostatic_pressure=p_hydro,
        elastic_volumetric_strain=vol, confinement_ratio=confinement, adiabatic_shear_strain=eps_c,
        practice_gap=gap, warm_forming_temperature=t_warm, warm_forming_temperature_conservative=t_warm_cons,
        extrapolated_model=band.extrapolated,
    )


__all__ = [
    "REFERENCES",
    "RATE_FLOOR", "FALLBACK_REFERENCE_RATE", "FALLBACK_RATE_SENSITIVITY", "DEFAULT_RATE_SENSITIVITY",
    "TAYLOR_QUINNEY", "MIN_FLOW_STRESS", "RATE_EXTRAPOLATION_FACTOR", "FALLBACK_CALIBRATION_TEMPERATURES",
    "CULVER_START_STRAIN", "ADIABATIC_STOP_FRACTION", "WARM_FORMING_MAX_FRACTION", "DEFAULT_LENGTH_SCALE",
    "README_REFRACTORY_FORGING_CLAIM", "PRACTICE_TOLERANCE", "FALLBACK_LABEL",
    "FlowStressBand", "PracticeGap", "ForgeAssessment",
    "fallback_rate_sensitivity", "fallback_flow_stress", "flow_stress", "flow_stress_band", "forging_pressure",
    "adiabatic_fraction", "adiabatic_temperature_rise", "adiabatic_shear_critical_strain",
    "max_strain_at_pressure", "warm_forming_temperature", "forging_practice_gap", "assess_forgeability",
]
