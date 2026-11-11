"""Pore closure and densification: can a magnetic pressure pulse make "zero-porosity" parts?

The concept note's "Zero-Porosity Aerospace Components" section claims that "a 4,000 MPa
magnetic press can completely close all internal porosities in highly complex geometries"
of titanium-aluminide (TiAl) and oxide-dispersion-strengthened (ODS) alloys.  That one
sentence needs four separate things to be true.  This module puts a number on each and
labels it honestly.

1. **Collapse** (:func:`plastic_collapse_pressure`, :func:`porosity_under_pressure`,
   :func:`matrix_strength`).  An isolated pore in a plastic matrix does not shrink until the
   hydrostatic pressure exceeds ``(2/3) sigma ln(1/f)`` (Torre 1948; Carroll & Holt 1972):
   *small* porosity is *harder* to collapse (0.5 % needs 3.5 sigma, 0.01 % needs 6.1 sigma).
   For cast gamma-TiAl with 1 % porosity the threshold is 1.69 GPa at the quasi-static
   compressive yield (0.55 GPa) but 4.30 GPa at the 1e4 1/s flow stress (1.40 GPa): at the
   strain rate of a 0.1 ms pulse a 4 GPa press does not even start, and at 1.6 GPa nothing
   happens on any reading.  Ti-6Al-4V and Inconel 718 are too strong to be closed at 4 GPa
   even quasi-statically.
2. **Time** (:func:`rayleigh_collapse_time`).  Once a pore yields, inertia lets it close in
   tens of nanoseconds; time is not the obstacle to *closure*.
3. **Healing** (:func:`self_diffusivity`, :func:`diffusion_bonding_time`).  Closure is not
   bonding.  Fatigue life improves only when the collapsed surfaces interdiffuse, and at
   room temperature that takes ~1e43 s for gamma-TiAl (atoms move ~1e-31 m during the
   pulse): the "closed" pore is an unbonded, crack-like flaw.  Below the ductile-brittle
   transition the pore wall must also survive strains of order 3, two orders of magnitude
   beyond the tensile ductility.
4. **Delivery**.  A poor conductor thinner than its skin depth (gamma-TiAl: 4.2 mm at
   10 kHz) is transparent to the pulse and needs a conducting driver or can, and the
   magnetic pressure ``B_t^2/(2 mu0)`` on a complex shape follows the local tangential field:
   it is not isostatic.  (Documented limitation: there is no field solver here; pressures
   given to this module are mechanical pressures already delivered to the part.)

The comparator is hot isostatic pressing (:data:`HIP_PRACTICE`): industry closes *and heals*
casting porosity at 0.1-0.2 GPa because the part is hot (creep, diffusion) and the dwell is
hours.  The nearest real magnetic process is dynamic magnetic compaction of *powder*
(:func:`assess_powder_compaction`): demonstrated, but its product is a green compact that
still has to be sintered.

Validity (whole module): isolated spherical pores, rigid-perfectly-plastic matrix evaluated
at its yield stress (strain hardening during collapse is neglected, so every threshold is a
lower bound on the pressure for complete closure), rate-independent plasticity (creep is
not modelled: for multi-hour hot dwells the thresholds are upper bounds), isothermal bulk
(local heating at the collapsing pore wall is not modelled), volume self-diffusion with an
estimated prefactor for bonding (adhesion of clean ductile surfaces and local melting are
not modelled).  SI units throughout; porosity and relative density are volume fractions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from ._numerics import bisect
from ._report import Assessment, ResultMixin, unit
from .benchmarks import Benchmark
from .constants import R_GAS, T_ROOM, ZERO_CELSIUS
from .eddy import sheet_shielding_factor, skin_depth_for_material
from .errors import InputError
from .magnetics import field_for_pressure
from .materials import Material, MaterialLike, as_material
from .plasticity import FALLBACK_LABEL, MIN_FLOW_STRESS, flow_stress_band
from .status import Confidence, Status, worst

REFERENCES: dict[str, str] = {
    "torre_1948": "C. Torre, 'Theorie und Verhalten der zusammengepressten Pulver', Berg- und "
                  "Huettenmaennische Monatshefte 93 (1948) 62-67.",
    "carroll_holt_1972": "M. M. Carroll and A. C. Holt, 'Static and dynamic pore-collapse relations for "
                         "ductile porous materials', J. Appl. Phys. 43 (1972) 1626-1636.",
    "gurson_1977": "A. L. Gurson, 'Continuum theory of ductile rupture by void nucleation and growth: "
                   "Part I - Yield criteria and flow rules for porous ductile media', J. Eng. Mater. "
                   "Technol. 99 (1977) 2-15.",
    "hill_1950": "R. Hill, The Mathematical Theory of Plasticity, Clarendon Press, Oxford (1950), ch. V "
                 "(expansion of a thick spherical shell).",
    "rayleigh_1917": "Lord Rayleigh, 'On the pressure developed in a liquid during the collapse of a "
                     "spherical cavity', Philos. Mag. 34 (1917) 94-98.",
    "helle_1985": "A. S. Helle, K. E. Easterling and M. F. Ashby, 'Hot-isostatic pressing diagrams: new "
                  "developments', Acta Metall. 33 (1985) 2163-2174.",
    "arzt_1983": "E. Arzt, M. F. Ashby and K. E. Easterling, 'Practical applications of hot-isostatic "
                 "pressing diagrams: four case studies', Metall. Trans. A 14 (1983) 211-221.",
    "atkinson_davies_2000": "H. V. Atkinson and S. Davies, 'Fundamental aspects of hot isostatic pressing: "
                            "an overview', Metall. Mater. Trans. A 31 (2000) 2981-3000.",
    "derby_wallach_1982": "B. Derby and E. R. Wallach, 'Theoretical model for diffusion bonding', "
                          "Metal Science 16 (1982) 49-56.",
    "brown_ashby_1980": "A. M. Brown and M. F. Ashby, 'Correlations for diffusion constants', "
                        "Acta Metall. 28 (1980) 1085-1101.",
    "appel_2011": "F. Appel, J. D. H. Paul and M. Oehring, Gamma Titanium Aluminide Alloys: Science and "
                  "Technology, Wiley-VCH, Weinheim (2011).",
    "astm_f2924": "ASTM F2924, Standard Specification for Additive Manufacturing Titanium-6 Aluminum-4 "
                  "Vanadium with Powder Bed Fusion, ASTM International: hot isostatic pressing at not less "
                  "than 100 MPa within 895-955 C for 180 +/- 60 min.",
    "astm_f3055": "ASTM F3055, Standard Specification for Additive Manufacturing Nickel Alloy (UNS N07718) "
                  "with Powder Bed Fusion, ASTM International: hot isostatic pressing at not less than "
                  "100 MPa within 1120-1185 C for 240 +/- 60 min.",
    "bridgman_1952": "P. W. Bridgman, Studies in Large Plastic Flow and Fracture with Special Emphasis on "
                     "the Effects of Hydrostatic Pressure, McGraw-Hill, New York (1952).",
    "chelluri_barber_1999": "B. Chelluri and J. P. Barber, 'Full-density, net-shape powder consolidation "
                            "using dynamic magnetic pulse pressures', JOM 51(7) (1999) 36-37.",
    "chelluri_knoth_2010": "B. Chelluri and E. Knoth, 4th International Conference on High Speed Forming "
                           "(2010), hdl.handle.net/2003/27201: pulse pressures in the GPa range applied for "
                           "under a millisecond; green densities above 95 % in most steels.",
    "kaygorodov_2006": "A. S. Kaygorodov et al. (2006), koreascience.kr/article/CFKO200617317788902: radial "
                       "magnetic pulsed press, pulses up to 0.3 GPa lasting about 100 microseconds.",
    "ge_ep2641984": "General Electric Co., European patent application EP 2 641 984 A2: existing practice for "
                    "cast Ti-48Al-2Cr-2Nb is hot isostatic pressing at 25 ksi (about 172 MPa) or more and "
                    "about 1185 C for about 3 h.",
    "yu_2021": "W. Yu et al., Metals 11 (2021) 1156, doi:10.3390/met11081156: 1260 C / 172 MPa / 4 h described "
               "as the most commonly performed HIP cycle for cast TiAl (as summarised; full text not checked).",
    "psyk_2011": "V. Psyk, D. Risch, B. L. Kinsey, A. E. Tekkaya and M. Kleiner, 'Electromagnetic forming - "
                 "a review', J. Mater. Process. Technol. 211 (2011) 787-829 (driver sheets for poorly "
                 "conducting workpieces).",
}
"""Short key -> full citation for every model and data source in this module."""

# ---------------------------------------------------------------------------
# Module constants (documented single sources)
# ---------------------------------------------------------------------------
QUASI_STATIC_RATE: float = 1e-3
"""1/s.  Strain rate of the quasi-static (optimistic) strength: the rate of the tensile and
compression tests behind handbook yield strengths."""

FALLBACK_BASIS: str = "Hollomon fallback (handbook tensile data, assumed rate sensitivity)"
"""Label used in :class:`MatrixStrength` when the strength comes from the Hollomon fallback of
:mod:`magform.plasticity` (materials without a published constitutive fit)."""

DEFAULT_POROSITY: float = 0.01
"""Default initial porosity (1 %): the order of the micro-shrinkage of an un-HIPed investment
casting (representative; castings span ~0.1-2 %)."""

DEFAULT_PORE_RADIUS: float = 50e-6
"""m.  Default pore radius (50 um): the order of interdendritic shrinkage pores (representative)."""

CLOSURE_POROSITY: float = 1e-4
"""Residual porosity at or below which a pore population is treated as geometrically closed
(judgement: 0.01 %, an order of magnitude below the ~0.1 % resolution of an Archimedes density
measurement).  A perfectly plastic matrix reaches it only at ``(2/3) sigma ln(1e4) = 6.14 sigma``."""

RAYLEIGH_COEFFICIENT: float = 0.91468
"""Coefficient of the Rayleigh (1917) collapse time ``t = 0.915 a sqrt(rho/dP)`` of an empty cavity."""

DIFFUSION_PREFACTOR: float = 1e-4
"""m^2/s.  Pre-exponential factor D0 of lattice self-diffusion used with the registry's
activation energies (estimated: D0 of metals clusters around 1e-5-1e-4 m^2/s; Brown & Ashby 1980)."""

DEFAULT_BOND_LENGTH: float = 1e-7
"""m.  Width of the interface zone (0.1 um) that must interdiffuse before clean pore surfaces
pressed into contact count as healed (estimated; machined faying surfaces in diffusion bonding
are rougher, ~1 um: Derby & Wallach 1982).  With it the volume-diffusion estimate lands within
one order of magnitude of industrial HIP holds; at room temperature the conclusion does not
depend on it: the bonding time exceeds 1e14 s (millions of years) for every metal in the
registry and is 3e43 s for gamma-TiAl."""

HEALING_UNCERTAINTY_FACTOR: float = 10.0
"""Order-of-magnitude uncertainty of the bonding-time estimate (prefactor, mechanism, interface
scale).  A dwell >= 10 x the estimate counts as healed, <= 1/10 as not healed; in between the
answer is inside the model uncertainty (extrapolation)."""

CREEP_HOMOLOGOUS_TEMPERATURE: float = 0.5
"""T/T_melt above which, for dwells longer than 1/:data:`QUASI_STATIC_RATE` (1000 s), time-dependent
creep (not modelled) closes pores below the plastic threshold (Helle et al. 1985)."""

PORE_WALL_CONFINEMENT: float = 2.0 / 3.0
"""Mean compressive stress at the wall of a collapsing pore in units of the flow stress.  The
wall is a free surface (radial stress zero, hoop stresses about -sigma), so the far-field
pressure does not confine it: locally the ratio is 2/3, below the ~1 at which
:mod:`magform.plasticity` invokes pressure-induced ductility (Bridgman 1952)."""

HIP_PRESSURE_RANGE: tuple[float, float] = (100e6, 200e6)
"""Pa.  Gas pressures of industrial hot isostatic pressing (Atkinson & Davies 2000)."""

HIP_TEMPERATURE_TOLERANCE: float = 15.0
"""K.  Tolerance when comparing a temperature with a tabulated HIP window (judgement: the order
of furnace hold tolerances)."""

MAGNETIC_PULSE_DURATION_RANGE: tuple[float, float] = (1e-6, 1e-2)
"""s.  Durations of magnetic pressure pulses, from single-turn coils to long-pulse magnets.  A
static field exerts no pressure on a stationary, current-free part, so a longer dwell needs a
gas or mechanical press."""

RANDOM_DENSE_PACKING: float = 0.64
"""Relative density of randomly packed equal spheres: the default initial powder density
(Helle et al. 1985)."""

STAGE_TRANSITION_DENSITY: float = 0.9
"""Relative density at which the Helle-Easterling-Ashby model switches from stage 1 (growing
particle contacts, connected porosity) to stage 2 (isolated spherical pores)."""

DEFAULT_GREEN_DENSITY_TARGET: float = 0.95
"""Default target relative density of a pressed (green) compact (judgement: above ~0.92-0.95 the
porosity is closed and the compact can be sintered or HIPed without a can)."""

DMC_BENCHMARK: Benchmark = Benchmark(
    "dynamic_magnetic_compaction",
    "Dynamic magnetic compaction of metal powders: peak compaction pressure class, sub-millisecond pulse",
    2.0e9, "Pa", 1999, REFERENCES["chelluri_barber_1999"], Confidence.ESTIMATED,
    "Order of magnitude of the pressures reported for the process ('GPa range' for under a millisecond, "
    "Chelluri & Knoth 2010; 0.3 GPa for about 100 microseconds in a radial press, Kaygorodov et al. 2006), not "
    "a sharp limit; the field collapses a conducting armature onto the powder, which can raise the powder "
    "pressure above the magnetic pressure.",
)
"""Cited benchmark for powder compaction by magnetic pulses (the nearest real process).  The
2 GPa value is an order-of-magnitude class boundary (confidence: estimated), not a record."""

DMC_PRESSURE_CLASS: float = DMC_BENCHMARK.value
"""Pa.  Compaction pressures at or below this are labelled state of the art for powder
(:data:`DMC_BENCHMARK`); above it the process is an extrapolation."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise InputError(f"{name} must be a real number, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(v):
        raise InputError(f"{name} must be finite, got {value!r}")
    return v


def _positive(name: str, value: Any) -> float:
    v = _finite(name, value)
    if v <= 0.0:
        raise InputError(f"{name} must be > 0, got {v!r}")
    return v


def _non_negative(name: str, value: Any) -> float:
    v = _finite(name, value)
    if v < 0.0:
        raise InputError(f"{name} must be >= 0, got {v!r}")
    return v


def _closed_fraction(name: str, value: Any) -> float:
    v = _finite(name, value)
    if not 0.0 <= v <= 1.0:
        raise InputError(f"{name} must lie in [0, 1], got {v!r}")
    return v


def _open_fraction(name: str, value: Any) -> float:
    v = _finite(name, value)
    if not 0.0 < v < 1.0:
        raise InputError(f"{name} must lie strictly between 0 and 1, got {v!r}")
    return v


def _solid_temperature(mat: Material, value: Any, name: str = "temperature") -> float:
    t = _positive(name, value)
    if t >= mat.melting_point:
        raise InputError(f"{name} {t:g} K is at or above the melting point of '{mat.key}' "
                         f"({mat.melting_point:g} K): no solid-state model")
    return t


def _gpa(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value / 1e9:.3g} GPa"


def _pct(value: float) -> str:
    return f"{100.0 * value:.3g} %"


def _density_pct(value: float) -> str:
    return f"{100.0 * value:.2f} %"


# ---------------------------------------------------------------------------
# Plastic collapse of an isolated pore (closed forms)
# ---------------------------------------------------------------------------
def plastic_collapse_pressure(flow_stress: float, porosity: float) -> float:
    """Hydrostatic pressure at which an isolated spherical pore collapses plastically [Pa].

    ``P_c = (2/3) sigma ln(1/f)``

    the fully plastic limit of a hollow sphere of void fraction ``f`` (inner/outer radius
    ``(a/b)^3 = f``) in a rigid-perfectly-plastic von Mises matrix of flow stress ``sigma``.
    It is also exactly the hydrostatic limit of the Gurson yield surface.  ``flow_stress`` in
    Pa (>= 0), ``porosity`` in [0, 1].  Limits: f -> 0 gives ``math.inf`` (the last trace of a
    pore needs unbounded pressure), f = 1 gives 0.  Small porosity is harder to collapse:
    f = 10 %, 1 %, 0.5 %, 0.01 % need 1.54, 3.07, 3.53 and 6.14 sigma.

    Validity: isolated spherical pores (f <~ 0.1), no strain hardening (a hardening matrix
    needs more), rate-independent (creep closes pores below P_c during long hot dwells), no
    gas in the pore.  The wall first yields earlier, at ``(2/3) sigma (1 - f)``, without a
    measurable change of porosity.

    Reference: Torre (1948); Carroll & Holt (1972); Hill (1950) ch. V; Gurson (1977).
    """
    s = _non_negative("flow_stress", flow_stress)
    f = _closed_fraction("porosity", porosity)
    if s == 0.0 or f == 1.0:
        return 0.0
    if f == 0.0:
        return math.inf
    return (2.0 / 3.0) * s * math.log(1.0 / f)


def gas_limited_porosity(initial_porosity: float, gas_pressure: float, pressure: float) -> float:
    """Porosity at which the gas trapped in a pore alone balances the applied pressure.

    Isothermal ideal gas in a pore whose volume per unit matrix volume is ``x = f/(1 - f)``:
    ``p_gas = p0 x0/x``, so ``x = (p0/P) x0`` and ``f = x/(1 + x)`` (= ``f0 p0/P`` for small
    porosity).  ``initial_porosity`` f0 in (0, 1), ``gas_pressure`` p0 and ``pressure`` P in Pa
    (>= 0).  Returns 0 for a vacuum pore (p0 = 0: no gas limit) and f0 when P <= p0 (the gas
    already carries the load).  Check: 1 % porosity sealed at 1 atm under 4 GPa -> 2.6e-7.

    Validity: insoluble gas (argon in powder-metallurgy and additively manufactured material),
    isothermal compression, no leakage; vacuum-cast shrinkage pores hold no gas.  Gas that was
    compressed this way re-expands when the part is later heated (thermally induced porosity).

    Reference: Atkinson & Davies (2000); Helle et al. (1985).
    """
    f0 = _open_fraction("initial_porosity", initial_porosity)
    pg = _non_negative("gas_pressure", gas_pressure)
    p = _non_negative("pressure", pressure)
    if pg == 0.0:
        return 0.0
    if p <= pg:
        return f0
    x = (pg / p) * f0 / (1.0 - f0)
    return x / (1.0 + x)


def porosity_under_pressure(flow_stress: float, pressure: float, initial_porosity: float,
                            gas_pressure: float = 0.0) -> float:
    """Residual porosity of a perfectly plastic matrix under hydrostatic pressure.

    The pore shrinks until its resistance equals the applied pressure::

        P = p_gas(f) + (2/3) sigma ln(1/f),      p_gas(f) = p0 [f0/(1 - f0)] / [f/(1 - f)]

    For a vacuum pore (``gas_pressure`` p0 = 0, the default) this is the inverse of
    :func:`plastic_collapse_pressure`: ``f = min(f0, exp(-3 P / (2 sigma)))``; with gas the
    root is found by bisection in ln f.  The pore does not move while
    ``P <= p0 + (2/3) sigma ln(1/f0)`` (returns f0).  ``flow_stress`` and pressures in Pa
    (>= 0), ``initial_porosity`` f0 in (0, 1).  A matrix without strength (sigma = 0) is
    limited only by the gas (:func:`gas_limited_porosity`).

    Validity: as :func:`plastic_collapse_pressure`; because hardening is neglected the result
    is a lower bound on the porosity that remains.  Elastic recovery of the cavity on
    unloading, ``9 (1 - nu) P / (2 E)`` of the *pore volume* (8 % for gamma-TiAl at 4 GPa), is
    neglected.

    Reference: Torre (1948); Carroll & Holt (1972).
    """
    s = _non_negative("flow_stress", flow_stress)
    p = _non_negative("pressure", pressure)
    f0 = _open_fraction("initial_porosity", initial_porosity)
    pg = _non_negative("gas_pressure", gas_pressure)
    if pg == 0.0:
        if s == 0.0:
            return 0.0 if p > 0.0 else f0
        return min(f0, math.exp(-1.5 * p / s))
    if p <= pg + plastic_collapse_pressure(s, f0):
        return f0
    f_gas = gas_limited_porosity(f0, pg, p)  # the gas alone balances P here
    if s == 0.0:
        return f_gas
    x0 = f0 / (1.0 - f0)

    def excess(log_f: float) -> float:
        f = math.exp(log_f)
        return pg * x0 * (1.0 - f) / f - (2.0 / 3.0) * s * log_f - p

    lo = math.log(f_gas)
    if excess(lo) <= 0.0:  # only by rounding, for a matrix of negligible strength
        return f_gas
    return math.exp(bisect(excess, lo, math.log(f0), tol=1e-13))


def pore_wall_strain(initial_porosity: float, final_porosity: float) -> float:
    """Equivalent plastic strain at the wall of a pore that shrinks from f0 to f.

    Incompressible radial flow around a shrinking cavity of radius a gives the hoop strain
    ``ln(a/a0)`` at the wall and the equivalent strain ``2 ln(a0/a)``; with the matrix volume
    conserved, ``a^3`` is proportional to ``f/(1 - f)``, so

    ``eps_wall = (2/3) ln[ f0 (1 - f) / (f (1 - f0)) ]``

    Dimensionless; f0 in (0, 1), 0 <= f <= f0 (``math.inf`` for f = 0).  Halving the pore
    radius costs 1.39; closing 1 % porosity to 0.01 % costs 3.1.  The strain falls off as
    ``(a/r)^3`` away from the wall, so it is a local, not a bulk, strain.

    Validity: spherically symmetric collapse (real pores collapse asymmetrically and jet);
    the wall is a free surface with mean stress of only -(2/3) sigma
    (:data:`PORE_WALL_CONFINEMENT`), so far-field pressure does not confine it.

    Reference: Hill (1950) ch. V; Carroll & Holt (1972).
    """
    f0 = _open_fraction("initial_porosity", initial_porosity)
    f = _closed_fraction("final_porosity", final_porosity)
    if f > f0:
        raise InputError(f"final_porosity ({f!r}) must not exceed initial_porosity ({f0!r})")
    if f == 0.0:
        return math.inf
    return (2.0 / 3.0) * math.log(f0 * (1.0 - f) / (f * (1.0 - f0)))


def rayleigh_collapse_time(pore_radius: float, material: MaterialLike, overpressure: float) -> float:
    """Inertial collapse time of a pore under a constant net pressure [s].

    ``t = 0.915 a sqrt(rho / dP)``

    (Rayleigh's result for an empty spherical cavity in an inviscid incompressible medium),
    with ``pore_radius`` a in m (> 0), rho the matrix density and ``overpressure`` dP in Pa:
    the applied pressure minus the plastic resistance of :func:`plastic_collapse_pressure`.
    Returns ``math.inf`` for dP <= 0 (the pore does not yield).  Check: a 50 um pore in
    gamma-TiAl under 2.3 GPa net pressure closes in 60 ns.

    Validity: a lower bound.  The resistance grows as the pore shrinks (ln(1/f)) and rate
    hardening slows the last stage, so the true time is longer and the pore stops at the
    residual porosity of :func:`porosity_under_pressure`; in practice the pore simply follows
    the pressure rise, because this time is 3-4 orders of magnitude below any pulse rise time.
    Time is not the obstacle to closure.

    Reference: Rayleigh (1917); Carroll & Holt (1972).
    """
    a = _positive("pore_radius", pore_radius)
    m = as_material(material)
    dp = _finite("overpressure", overpressure)
    if dp <= 0.0:
        return math.inf
    return RAYLEIGH_COEFFICIENT * a * math.sqrt(m.density / dp)


# ---------------------------------------------------------------------------
# Matrix strength: optimistic (quasi-static) and conservative (dynamic) bounds
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class MatrixStrength(ResultMixin):
    """The two matrix flow stresses between which pore collapse is bracketed.

    ``quasi_static`` (optimistic): the smallest recorded quasi-static value at zero strain,
    i.e. the handbook compressive yield (tensile yield when no compressive value is recorded)
    and every published constitutive fit at 1e-3 1/s; ``quasi_static_basis`` names the winner.
    ``dynamic`` (conservative): the strongest published fit at ``strain_rate`` (the Hollomon
    fallback only when the material has no fit), never below ``quasi_static``;
    ``dynamic_model`` names it.  ``extrapolated`` is the calibration flag of
    :func:`magform.plasticity.flow_stress_band` at the dynamic state.

    Reference: Johnson & Cook (1983) and Zerilli & Armstrong (1987) via magform.plasticity.
    """

    _title: ClassVar[str] = "Matrix strength"

    material: str
    temperature: float = unit("K")
    strain_rate: float = unit("1/s")
    quasi_static: float = unit("Pa")
    dynamic: float = unit("Pa")
    quasi_static_basis: str
    dynamic_model: str
    extrapolated: bool
    extrapolation_note: str = ""


def _model_label(label: str) -> str:
    """Readable name of a flow-stress-band entry (the fallback's key is terse)."""
    return FALLBACK_BASIS if label == FALLBACK_LABEL else label


def _handbook_strength(mat: Material, temperature: float) -> tuple[float, str] | None:
    """Handbook quasi-static yield (compressive when recorded), softened linearly to T_melt."""
    for field_name, label in (("compressive_yield", "handbook compressive yield"),
                              ("yield_strength", "handbook yield strength")):
        value = getattr(mat, field_name)
        if value is not None and value > 0.0:
            base = float(value)
            if temperature > T_ROOM:
                t_star = min((temperature - T_ROOM) / (mat.melting_point - T_ROOM), 0.999)
                base *= 1.0 - t_star
            return max(base, MIN_FLOW_STRESS), label
    return None


def matrix_strength(material: MaterialLike, strain_rate: float = 1e4,
                    temperature: float = T_ROOM) -> MatrixStrength:
    """Optimistic and conservative yield stress of the matrix around a pore.

    ``quasi_static = min(handbook yield, published fits at 1e-3 1/s)`` and
    ``dynamic = max(published fits at strain_rate)``, all at zero plastic strain and
    ``temperature`` (K).  The handbook value (``compressive_yield`` if recorded, else
    ``yield_strength``) is softened as ``1 - (T - 293)/(T_melt - 293)`` above room temperature,
    the same linear assumption as the Hollomon fallback of :mod:`magform.plasticity`.
    ``strain_rate`` in 1/s (>= 0; rates below 1e-3 1/s are evaluated at 1e-3 1/s, the
    quasi-static limit).  Gamma-TiAl at 293 K: 0.55 GPa (handbook compressive yield) and
    1.40 GPa at 1e4 1/s (Johnson-Cook, split-Hopkinson compression).

    Validity: yield stress only; the matrix around a collapsing pore hardens (wall strains of
    order 1, :func:`pore_wall_strain`), so both values are lower bounds on the resistance.
    Raises :class:`InputError` for T <= 0, T >= T_melt, or a material without strength data.

    Reference: Johnson & Cook (1983) and Zerilli & Armstrong (1987) via
    :func:`magform.plasticity.flow_stress_band`; Carroll & Holt (1972) for the rate dependence
    of pore collapse.
    """
    mat = as_material(material)
    rate = _non_negative("strain_rate", strain_rate)
    band_qs = flow_stress_band(mat, 0.0, QUASI_STATIC_RATE, temperature)
    t = band_qs.temperature
    candidates: list[tuple[float, str]] = []
    handbook = _handbook_strength(mat, t)
    if handbook is not None:
        candidates.append(handbook)  # first, so that a tie is credited to the handbook value
    candidates.extend((band_qs.values[label], _model_label(label)) for label in band_qs.band_models)
    quasi_static, basis = min(candidates, key=lambda item: item[0])
    band_dyn = flow_stress_band(mat, 0.0, max(rate, QUASI_STATIC_RATE), t)
    strongest = max(band_dyn.band_models, key=lambda label: band_dyn.values[label])
    dynamic, dynamic_model = band_dyn.values[strongest], _model_label(strongest)
    if dynamic < quasi_static:
        dynamic, dynamic_model = quasi_static, basis
    return MatrixStrength(
        material=mat.key, temperature=t, strain_rate=rate, quasi_static=quasi_static, dynamic=dynamic,
        quasi_static_basis=basis, dynamic_model=dynamic_model, extrapolated=band_dyn.extrapolated,
        extrapolation_note=band_dyn.extrapolation_note,
    )


def _strength_for_rate(strength: MatrixStrength) -> float:
    return strength.quasi_static if strength.strain_rate <= QUASI_STATIC_RATE else strength.dynamic


def collapse_pressure(material: MaterialLike, porosity: float, strain_rate: float = QUASI_STATIC_RATE,
                      temperature: float = T_ROOM) -> float:
    """Plastic collapse pressure of a pore in ``material`` [Pa].

    ``P_c = (2/3) sigma ln(1/f)`` (:func:`plastic_collapse_pressure`) with ``sigma`` from
    :func:`matrix_strength`: the quasi-static (optimistic) value for ``strain_rate`` <= 1e-3 1/s,
    the dynamic (conservative) value above.  ``porosity`` in [0, 1], rate in 1/s, T in K.
    Gamma-TiAl at 293 K: 1.69 GPa (1 %) and 1.94 GPa (0.5 %) quasi-statically; 4.30 GPa and
    4.95 GPa at 1e4 1/s, i.e. above the concept note's 4 GPa.

    Validity: as :func:`plastic_collapse_pressure` and :func:`matrix_strength`.

    Reference: Torre (1948); Carroll & Holt (1972).
    """
    return plastic_collapse_pressure(_strength_for_rate(matrix_strength(material, strain_rate, temperature)),
                                     porosity)


def residual_porosity(material: MaterialLike, pressure: float, initial_porosity: float,
                      strain_rate: float = QUASI_STATIC_RATE, temperature: float = T_ROOM,
                      gas_pressure: float = 0.0) -> float:
    """Porosity left in ``material`` after a hydrostatic pressure, by plastic collapse alone.

    :func:`porosity_under_pressure` with ``sigma`` from :func:`matrix_strength` (quasi-static
    value for ``strain_rate`` <= 1e-3 1/s, dynamic value above): for a vacuum pore
    ``f = min(f0, exp(-3 P / (2 sigma)))``.  Pressures in Pa, ``initial_porosity`` in (0, 1),
    rate in 1/s, T in K.  Gamma-TiAl, 1 % porosity, 4 GPa, 293 K: 1.8e-5 quasi-statically
    (closed), unchanged 1 % at 1e4 1/s (the pore never yields).

    Validity: as :func:`porosity_under_pressure`; says nothing about bonding
    (:func:`diffusion_bonding_time`).

    Reference: Torre (1948); Carroll & Holt (1972).
    """
    sigma = _strength_for_rate(matrix_strength(material, strain_rate, temperature))
    return porosity_under_pressure(sigma, pressure, initial_porosity, gas_pressure)


# ---------------------------------------------------------------------------
# Healing: diffusion bonding of the collapsed pore surfaces
# ---------------------------------------------------------------------------
def self_diffusivity(material: MaterialLike, temperature: float,
                     prefactor: float = DIFFUSION_PREFACTOR) -> float:
    """Lattice self-diffusivity ``D = D0 exp(-Q / (R T))`` [m^2/s].

    ``Q`` is ``Material.self_diffusion_activation`` (J/mol), ``prefactor`` D0 in m^2/s
    (default :data:`DIFFUSION_PREFACTOR`, 1e-4, estimated), ``temperature`` in K
    (0 < T < T_melt).  Gamma-TiAl: 6e-15 m^2/s at the 1533 K HIP temperature, 3.5e-58 m^2/s at
    293 K.  Raises :class:`InputError` when the material records no activation energy.

    Validity: order of magnitude.  One Arrhenius term with a generic prefactor; real alloys
    add grain-boundary and surface paths (faster at low temperature) and phase changes.

    Reference: Brown & Ashby (1980); activation energies from the materials registry.
    """
    m = as_material(material)
    t = _solid_temperature(m, temperature)
    d0 = _positive("prefactor", prefactor)
    m.require("self_diffusion_activation")
    return d0 * math.exp(-float(m.self_diffusion_activation) / (R_GAS * t))  # type: ignore[arg-type]


def diffusion_length(material: MaterialLike, temperature: float, time: float) -> float:
    """Distance atoms diffuse in ``time``, ``x = sqrt(D t)`` [m].

    ``D`` from :func:`self_diffusivity`; ``time`` in s (>= 0), ``temperature`` in K.  During a
    0.1 ms pulse at room temperature gamma-TiAl atoms move 2e-31 m, twenty-one orders of magnitude
    less than one lattice spacing; during a 4 h HIP hold at 1533 K, 9 um.

    Validity: as :func:`self_diffusivity`.

    Reference: Brown & Ashby (1980).
    """
    t = _non_negative("time", time)
    return math.sqrt(self_diffusivity(material, temperature) * t)


def diffusion_bonding_time(material: MaterialLike, temperature: float,
                           length_scale: float = DEFAULT_BOND_LENGTH) -> float:
    """Time for collapsed pore surfaces to interdiffuse, ``t = L^2 / D`` [s].

    ``length_scale`` L in m (> 0, default :data:`DEFAULT_BOND_LENGTH`, 0.1 um), ``D`` from
    :func:`self_diffusivity` at ``temperature`` (K).  Returns ``math.inf`` when D underflows to
    zero.  Gamma-TiAl: 1.7 s at 1533 K (a 4 h HIP hold is ample), 3e43 s at 293 K.  Closure
    is not bonding: until this time has passed the collapsed pore is an unbonded, crack-like
    interface.

    Validity: order of magnitude (:data:`HEALING_UNCERTAINTY_FACTOR`).  Volume diffusion only;
    real diffusion bonds also close by creep of asperities and by grain-boundary and surface
    diffusion, and need the surface films to dissolve.  Mechanical adhesion of clean, ductile
    surfaces (cold pressure welding) and local melting are not modelled, so a ductile metal
    may stick sooner than this says; gas-contaminated or oxidised pore surfaces do not.

    Reference: Derby & Wallach (1982); Atkinson & Davies (2000).
    """
    length = _positive("length_scale", length_scale)
    d = self_diffusivity(material, temperature)
    return math.inf if d == 0.0 else length * length / d


# ---------------------------------------------------------------------------
# The comparator: hot isostatic pressing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HipCycle(ResultMixin):
    """A representative industrial hot-isostatic-pressing cycle.

    ``pressure`` [Pa], ``temperature`` [K] and ``time`` [s] are the typical gas pressure, hold
    temperature and hold time; ``temperature_window`` [K] the range used in practice;
    ``medium`` the pressure-transmitting gas; ``source`` the citation.  Representative of
    practice, not a specification.

    Reference: Atkinson & Davies (2000) and the per-cycle ``source``.
    """

    material: str
    pressure: float = unit("Pa")
    temperature: float = unit("K")
    time: float = unit("s")
    temperature_window: tuple[float, float] = unit("K")
    medium: str = "argon"
    source: str = ""
    confidence: Confidence = Confidence.REPRESENTATIVE
    note: str = ""


def _celsius(t_c: float) -> float:
    return t_c + ZERO_CELSIUS


HIP_PRACTICE: dict[str, HipCycle] = {
    "gamma_tial": HipCycle(
        "gamma_tial", 172e6, _celsius(1260.0), 4.0 * 3600.0, (_celsius(1185.0), _celsius(1260.0)),
        source=REFERENCES["appel_2011"],
        note="representative of published practice for investment-cast Ti-48Al-2Cr-2Nb (about 1185-1260 C, "
             "172 MPa = 25 ksi, 3-4 h: GE patent application EP 2 641 984 A2 gives about 1185 C for about "
             "3 h; Yu et al. 2021 report 1260 C for 4 h as the commonest cycle); 0.88 T_melt; closes casting "
             "micro-shrinkage before the stabilising heat treatment; the cycle used in production for any "
             "particular engine part is not public"),
    "ti6al4v": HipCycle(
        "ti6al4v", 100e6, _celsius(920.0), 2.0 * 3600.0, (_celsius(895.0), _celsius(955.0)),
        source=REFERENCES["astm_f2924"],
        note="the usual cycle for castings and powder-bed parts, 920 C / 100 MPa / 2 h (the short end of "
             "the ASTM hold); below the beta transus"),
    "inconel_718": HipCycle(
        "inconel_718", 100e6, _celsius(1163.0), 3.0 * 3600.0, (_celsius(1120.0), _celsius(1185.0)),
        source=REFERENCES["astm_f3055"],
        note="hold recorded at the short end of the ASTM window; followed by solution treatment and ageing"),
}
"""Typical industrial HIP cycles by registry key: the state-of-the-art way to close and heal
internal porosity.  All work at 0.10-0.17 GPa because the part is hot and the dwell is hours."""


def hip_cycle(material: MaterialLike) -> HipCycle | None:
    """The tabulated industrial HIP cycle of ``material``, or ``None`` if none is recorded.

    Lookup in :data:`HIP_PRACTICE` by registry key (aliases accepted).  Gamma-TiAl:
    172 MPa, 1533 K (1260 C), 4 h in argon - a pressure 23 times below the concept note's
    4 GPa.  HIP works at 0.1-0.2 GPa and roughly 0.6-0.9 T_melt because pores close by
    power-law creep and heal by diffusion during a hold of hours; the gas makes the pressure
    truly isostatic, so the shape is preserved.

    Reference: Atkinson & Davies (2000); Helle et al. (1985).
    """
    return HIP_PRACTICE.get(as_material(material).key)


def matches_hip_practice(material: MaterialLike, pressure: float, temperature: float, dwell: float) -> bool:
    """True if (pressure, temperature, dwell) is at least an industrial HIP cycle for ``material``.

    ``pressure >= 100 MPa`` (lower end of :data:`HIP_PRESSURE_RANGE`), ``temperature`` inside
    the cycle's ``temperature_window`` +- :data:`HIP_TEMPERATURE_TOLERANCE` (15 K) and
    ``dwell >= cycle.time``.  Units: Pa, K, s.  False when no cycle is tabulated.  More
    pressure than HIP still counts (it can only help the material), but no magnetic pulse
    lasts hours and a static field exerts no pressure: a match means a gas press is doing
    the work.

    Reference: Atkinson & Davies (2000); the cycle's ``source``.
    """
    cycle = hip_cycle(material)
    p = _non_negative("pressure", pressure)
    t = _positive("temperature", temperature)
    hold = _positive("dwell", dwell)
    if cycle is None:
        return False
    lo, hi = cycle.temperature_window
    return (p >= HIP_PRESSURE_RANGE[0]
            and lo - HIP_TEMPERATURE_TOLERANCE <= t <= hi + HIP_TEMPERATURE_TOLERANCE
            and hold >= cycle.time)


# ---------------------------------------------------------------------------
# Pore-closure assessment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class PoreClosureAssessment(Assessment):
    """Can a pressure pulse close *and heal* the internal porosity of a part?

    Strengths and thresholds come in pairs: ``*_quasi_static`` (optimistic) and ``*_dynamic``
    (conservative), see :class:`MatrixStrength`.  ``collapse_pressure_*`` is the pressure at
    which the pores start to shrink, ``closure_pressure_*`` the pressure that brings them to
    :data:`CLOSURE_POROSITY`; ``collapses_*`` / ``closed_*`` compare them with ``pressure``;
    ``residual_porosity_*`` is what remains.  ``collapse_time`` is the inertial (Rayleigh)
    lower bound.  ``wall_strain`` is the pore-wall strain needed to reach the closure level,
    ``ductility`` the failure strain it is compared with.  ``diffusion_bond_time`` is compared
    with ``pulse_duration`` to give ``bonded``.  ``skin_depth`` / ``transparent`` /
    ``pressure_transfer`` describe magnetic delivery to a bare section (``None`` when not
    asked).  ``hip`` is the industrial comparator.  ``status`` is the worst of
    ``closure_status`` and ``healing_status`` (state of the art when HIP practice is matched);
    ``margin`` = ``pressure / collapse_pressure_dynamic`` (``pressure / 100 MPa``, the lower end of
    industrial HIP, when HIP practice is matched); ``field_equivalent`` = ``sqrt(2 mu0 P)``.  See
    :func:`assess_pore_closure`.

    Reference: Carroll & Holt (1972); Atkinson & Davies (2000); Derby & Wallach (1982).
    """

    _title: ClassVar[str] = "Pore-closure assessment"

    material: str
    pressure: float = unit("Pa")
    field_equivalent: float = unit("T")
    temperature: float = unit("K")
    pulse_duration: float = unit("s")
    strain_rate: float = unit("1/s")
    initial_porosity: float
    pore_radius: float = unit("m")
    gas_pressure: float = unit("Pa")
    flow_stress_quasi_static: float = unit("Pa")
    flow_stress_dynamic: float = unit("Pa")
    quasi_static_basis: str
    dynamic_model: str
    collapse_pressure_quasi_static: float = unit("Pa")
    collapse_pressure_dynamic: float = unit("Pa")
    closure_pressure_quasi_static: float = unit("Pa")
    closure_pressure_dynamic: float = unit("Pa")
    collapses_quasi_static: bool
    collapses_dynamic: bool
    residual_porosity_quasi_static: float
    residual_porosity_dynamic: float
    closed_quasi_static: bool
    closed_dynamic: bool
    gas_limited_porosity: float
    collapse_time: float = unit("s")
    wall_strain: float
    ductility: float | None
    wall_strain_to_ductility: float | None
    dbtt: float | None = unit("K")
    below_dbtt: bool | None
    confinement_ratio: float
    creep_regime: bool
    diffusivity: float | None = unit("m^2/s")
    diffusion_length: float | None = unit("m")
    bond_length: float = unit("m")
    diffusion_bond_time: float | None = unit("s")
    bonded: bool | None
    drive_frequency: float | None = unit("Hz")
    skin_depth: float | None = unit("m")
    section_thickness: float | None = unit("m")
    transparent: bool | None
    pressure_transfer: float | None
    hip: HipCycle | None
    matches_hip_practice: bool
    closure_status: Status
    healing_status: Status
    extrapolated_model: bool


def _hip_text(cycle: HipCycle) -> str:
    return (f"{cycle.pressure / 1e6:.0f} MPa, {cycle.temperature:.0f} K "
            f"({cycle.temperature - ZERO_CELSIUS:.0f} C), {cycle.time / 3600.0:g} h in {cycle.medium}")


def assess_pore_closure(material: MaterialLike, pressure: float, initial_porosity: float = DEFAULT_POROSITY,
                        pore_radius: float = DEFAULT_PORE_RADIUS, temperature: float = T_ROOM,
                        pulse_duration: float = 1e-4, strain_rate: float | None = None,
                        gas_pressure: float = 0.0, section_thickness: float | None = None,
                        drive_frequency: float | None = None, *,
                        bond_length: float = DEFAULT_BOND_LENGTH) -> PoreClosureAssessment:
    """Assess whether a pressure pulse closes and heals internal porosity (README: 'a 4,000 MPa
    magnetic press can completely close all internal porosities').

    Equations: collapse threshold ``p0 + (2/3) sigma ln(1/f0)`` and residual porosity from
    :func:`porosity_under_pressure`, each at the quasi-static and the dynamic strength of
    :func:`matrix_strength`; "closed" means residual porosity <= :data:`CLOSURE_POROSITY`
    (0.01 %); inertial time :func:`rayleigh_collapse_time`; bonding time
    :func:`diffusion_bonding_time` against ``pulse_duration``; delivery
    ``skin depth = sqrt(rho_e / (pi f mu0))`` against ``section_thickness``.

    Arguments: ``pressure`` [Pa] is the mechanical hydrostatic pressure delivered to the part;
    ``initial_porosity`` in (0, 1); ``pore_radius`` [m]; ``temperature`` [K] of the part during
    the pulse; ``pulse_duration`` [s] the time at pressure and temperature; ``strain_rate``
    [1/s] of the matrix (default ``1/pulse_duration``); ``gas_pressure`` [Pa] of gas sealed in
    the pore (0 = vacuum shrinkage pore; pass ``constants.P_ATM`` for argon porosity);
    ``section_thickness`` [m] and ``drive_frequency`` [Hz] (default ``1/(2 pulse_duration)``
    when a section is given) switch on the delivery check; ``bond_length`` [m] (keyword).

    Status.  If the conditions match the material's industrial HIP cycle
    (:func:`matches_hip_practice`) the result is **state of the art**: the conventional press
    already does it (by creep and diffusion, which the plastic thresholds here do not
    describe; ``margin`` is then ``pressure / 100 MPa``).  Otherwise ``margin`` is
    ``pressure / collapse_pressure_dynamic`` and
    ``status = worst(closure_status, healing_status)`` with

    closure (first match, then worst-combined with the flags below):

    1. not closed even at the quasi-static strength (the pores do not yield, or shrink
       without reaching 0.01 %) -> infeasible (extrapolation instead in the creep regime,
       T >= 0.5 T_melt and dwell > 1000 s, where creep is not modelled);
    2. closed at the quasi-static strength only (the strength band straddles) -> extrapolation;
    3. closed at the dynamic strength -> state of the art (plain plasticity);

    flags: below the DBTT with a wall strain above the failure strain -> at least
    extrapolation (pressure-induced ductility, Bridgman 1952; undemonstrated, and the pore
    wall itself is a free surface); flow-stress model outside its calibration -> at least
    extrapolation; pulse shorter than the inertial collapse time -> at least extrapolation.

    healing: ``pulse_duration >= 10 t_bond`` -> state of the art; within a factor 10 of
    ``t_bond`` -> extrapolation (model uncertainty); shorter -> infeasible ("closed but not
    healed": an unbonded, crack-like flaw); no diffusion data -> not applicable.

    The delivery check and the non-isostatic nature of magnetic pressure are reported as
    warnings and notes; they do not change the status because ``pressure`` is by definition
    the delivered mechanical pressure.  Gamma-TiAl, 1 % porosity, 4 GPa, 293 K, 0.1 ms:
    closure extrapolation (1.69 GPa threshold quasi-static, 4.30 GPa dynamic), healing
    infeasible (3e43 s), overall infeasible; the HIP cycle (172 MPa, 1533 K, 4 h) is state of
    the art.

    Validity: see the module docstring.  Reference: Torre (1948); Carroll & Holt (1972);
    Rayleigh (1917); Derby & Wallach (1982); Atkinson & Davies (2000); Helle et al. (1985);
    Bridgman (1952); Psyk et al. (2011).
    """
    mat = as_material(material)
    p = _non_negative("pressure", pressure)
    f0 = _open_fraction("initial_porosity", initial_porosity)
    radius = _positive("pore_radius", pore_radius)
    tp = _positive("pulse_duration", pulse_duration)
    rate = 1.0 / tp if strain_rate is None else _non_negative("strain_rate", strain_rate)
    pg = _non_negative("gas_pressure", gas_pressure)
    section = None if section_thickness is None else _positive("section_thickness", section_thickness)
    freq = None if drive_frequency is None else _positive("drive_frequency", drive_frequency)
    l_bond = _positive("bond_length", bond_length)

    strength = matrix_strength(mat, rate, temperature)
    t = strength.temperature
    s_qs, s_dyn = strength.quasi_static, strength.dynamic
    pc_qs = pg + plastic_collapse_pressure(s_qs, f0)
    pc_dyn = pg + plastic_collapse_pressure(s_dyn, f0)
    already_closed = f0 <= CLOSURE_POROSITY
    if already_closed:
        p_close_qs = p_close_dyn = 0.0
        wall_strain = 0.0
    else:
        gas_at_closure = pg * (f0 / (1.0 - f0)) * (1.0 - CLOSURE_POROSITY) / CLOSURE_POROSITY
        p_close_qs = gas_at_closure + plastic_collapse_pressure(s_qs, CLOSURE_POROSITY)
        p_close_dyn = gas_at_closure + plastic_collapse_pressure(s_dyn, CLOSURE_POROSITY)
        wall_strain = pore_wall_strain(f0, CLOSURE_POROSITY)
    f_qs = porosity_under_pressure(s_qs, p, f0, pg)
    f_dyn = porosity_under_pressure(s_dyn, p, f0, pg)
    collapses_qs, collapses_dyn = p > pc_qs, p > pc_dyn
    closed_qs, closed_dyn = f_qs <= CLOSURE_POROSITY, f_dyn <= CLOSURE_POROSITY
    f_gas = gas_limited_porosity(f0, pg, p)
    net = p - pc_dyn if collapses_dyn else p - pc_qs
    t_collapse = rayleigh_collapse_time(radius, mat, net)

    ductility = mat.dynamic_failure_strain if mat.dynamic_failure_strain is not None else mat.elongation
    strain_ratio = wall_strain / ductility if ductility is not None and ductility > 0.0 else None
    dbtt = mat.dbtt
    below = None if dbtt is None else t < dbtt
    confinement = p / s_dyn
    creep = t >= CREEP_HOMOLOGOUS_TEMPERATURE * mat.melting_point and tp > 1.0 / QUASI_STATIC_RATE

    if mat.self_diffusion_activation is None:
        diffusivity = x_diff = t_bond = None
        bonded: bool | None = None
    else:
        diffusivity = self_diffusivity(mat, t)
        x_diff = math.sqrt(diffusivity * tp)
        t_bond = math.inf if diffusivity == 0.0 else l_bond * l_bond / diffusivity
        bonded = tp >= t_bond

    skin = transparent = transfer = None
    if section is not None and freq is None:
        freq = 1.0 / (2.0 * tp)
    if freq is not None:
        skin = skin_depth_for_material(mat, freq, t)
        if section is not None:
            transparent = section <= skin
            transfer = sheet_shielding_factor(section, skin)

    cycle = HIP_PRACTICE.get(mat.key)
    hip_match = matches_hip_practice(mat, p, t, tp)

    warnings: list[str] = []
    notes: list[str] = [
        (f"Plastic collapse of an isolated pore needs P > (2/3) sigma ln(1/f): small porosity is harder to "
         f"collapse. At f0 = {_pct(f0)} the threshold is {_gpa(pc_qs)} at the quasi-static strength "
         f"({_gpa(s_qs)}, {strength.quasi_static_basis}) and {_gpa(pc_dyn)} at the dynamic strength "
         f"({_gpa(s_dyn)} at {max(rate, QUASI_STATIC_RATE):.3g} 1/s); closing to {_pct(CLOSURE_POROSITY)} "
         f"needs {_gpa(p_close_qs)} / {_gpa(p_close_dyn)}. Strain hardening is neglected, so these are "
         "lower bounds."),
    ]
    if math.isfinite(t_collapse):
        notes.append(f"Inertial (Rayleigh) collapse time of a {radius * 1e6:.3g} um pore: {t_collapse:.3g} s, "
                     f"against a {tp:.3g} s pulse: time is not the obstacle to closure.")
    else:
        notes.append("The pores never yield, so there is no collapse time.")
    if pg > 0.0:
        notes.append(f"Gas sealed at {pg / 1e3:.4g} kPa limits the porosity to {f_gas:.3g} by itself "
                     "(isothermal); compressed gas re-expands when the part is later heated.")
    else:
        notes.append("Vacuum (shrinkage) pore assumed; gas porosity from powder or additive processing "
                     "adds the resistance of the trapped gas (gas_pressure).")
    if t_bond is not None:
        notes.append(f"Closure is not bonding: at {t:.0f} K self-diffusion (D = {diffusivity:.3g} m^2/s, "
                     f"estimated prefactor) moves atoms {x_diff:.3g} m during the pulse, while bonding a "
                     f"{l_bond * 1e6:.3g} um interface zone takes about {t_bond:.3g} s. Adhesion of clean "
                     "ductile surfaces and local melting are not modelled.")
    if cycle is not None:
        notes.append(f"Industrial comparator: hot isostatic pressing closes and heals {mat.name} at "
                     f"{_hip_text(cycle)} - {p / cycle.pressure:.3g} x less pressure than assessed here, "
                     "because the part is hot and the dwell is hours.")
    else:
        notes.append(f"Industrial comparator: hot isostatic pressing at {HIP_PRESSURE_RANGE[0] / 1e6:.0f}-"
                     f"{HIP_PRESSURE_RANGE[1] / 1e6:.0f} MPa and roughly 0.6-0.9 T_melt for hours "
                     "(Atkinson & Davies 2000); no cycle is tabulated for this material.")
    notes.append("Magnetic pressure B_t^2/(2 mu0) follows the local tangential field: on a complex shape it is "
                 "not isostatic (no field solver here) and it distorts the part; HIP uses a gas precisely "
                 "because gas pressure is isostatic.")
    if mat.category == "ods":
        notes.append("ODS alloys are made by mechanical alloying and hot consolidation of powder (melting "
                     "destroys the oxide dispersion), so their porosity is a powder-consolidation question: "
                     "see assess_powder_compaction.")
    if tp > MAGNETIC_PULSE_DURATION_RANGE[1]:
        notes.append(f"A dwell of {tp:.3g} s is longer than any magnetic pressure pulse "
                     f"({MAGNETIC_PULSE_DURATION_RANGE[0]:g}-{MAGNETIC_PULSE_DURATION_RANGE[1]:g} s) and a "
                     "static field exerts no pressure on a stationary part: this is a gas or mechanical press.")

    # delivery (reported, not graded: `pressure` is the delivered mechanical pressure)
    if transparent:
        warnings.append(f"section {section * 1e3:.3g} mm is thinner than the skin depth {skin * 1e3:.3g} mm at "
                        f"{freq:.3g} Hz: the bare part is transparent to the pulse (pressure transfer "
                        f"{transfer:.2f}); a conducting driver or can must carry the current")

    margin = p / pc_dyn if pc_dyn > 0.0 else math.inf
    if hip_match:
        closure_status = healing_status = status = Status.STATE_OF_THE_ART
        margin = p / HIP_PRESSURE_RANGE[0]
        notes.append("Conditions match industrial HIP practice: state of the art, the conventional press "
                     "already does this. HIP closes pores by power-law creep and diffusion over hours (Helle "
                     "et al. 1985), not by time-independent yielding, so the plastic thresholds above do not "
                     "apply to it.")
    else:
        # --- closure --------------------------------------------------------------------------
        if already_closed:
            closure_status = Status.STATE_OF_THE_ART
            notes.append(f"Initial porosity is already at or below the {_pct(CLOSURE_POROSITY)} closure level.")
        elif not closed_qs:
            if creep:
                closure_status = Status.EXTRAPOLATION
                warnings.append(f"plastic collapse is insufficient at {_gpa(p)} but the dwell is hot and long "
                                f"(T/T_melt = {t / mat.melting_point:.2f}, {tp:.3g} s): creep closure is not "
                                "modelled here and the conditions are outside tabulated HIP practice")
            elif not collapses_qs:
                closure_status = Status.INFEASIBLE
                warnings.append(f"pores do not yield even at the quasi-static strength: {_gpa(p)} <= "
                                f"{_gpa(pc_qs)} = (2/3) sigma ln(1/f0)")
            else:
                closure_status = Status.INFEASIBLE
                warnings.append(f"pores shrink but do not close even at the quasi-static strength: residual "
                                f"porosity {_pct(f_qs)} > {_pct(CLOSURE_POROSITY)} (closing needs "
                                f"{_gpa(p_close_qs)})")
        elif not closed_dyn:
            closure_status = Status.EXTRAPOLATION
            if collapses_dyn:
                detail = f"at the dynamic strength {_pct(f_dyn)} porosity remains"
            else:
                detail = f"at the dynamic strength the pores never yield (threshold {_gpa(pc_dyn)})"
            warnings.append(f"closure only at the quasi-static strength (strength band straddles the pressure): "
                            f"{detail}")
        else:
            closure_status = Status.STATE_OF_THE_ART
        if closure_status is not Status.INFEASIBLE and not already_closed:
            if below and strain_ratio is not None and strain_ratio > 1.0:
                closure_status = worst(closure_status, Status.EXTRAPOLATION)
                warnings.append(f"below the DBTT ({dbtt:.0f} K): the pore wall must strain by {wall_strain:.2f}, "
                                f"{strain_ratio:.0f} x the failure strain {ductility:g}; relies on "
                                "pressure-induced ductility (Bridgman), undemonstrated here, and the wall "
                                f"itself is a free surface (local confinement {PORE_WALL_CONFINEMENT:.2f}): "
                                "micro-cracking around collapsed pores is likely")
            if strength.extrapolated:
                closure_status = worst(closure_status, Status.EXTRAPOLATION)
                warnings.append(f"flow-stress model extrapolated: {strength.extrapolation_note}")
            if math.isfinite(t_collapse) and t_collapse > tp:
                closure_status = worst(closure_status, Status.EXTRAPOLATION)
                warnings.append(f"the {tp:.3g} s pulse is shorter than the inertial collapse time "
                                f"({t_collapse:.3g} s): closure is incomplete")
        # --- healing --------------------------------------------------------------------------
        if t_bond is None:
            healing_status = Status.NOT_APPLICABLE
            warnings.append(f"no self-diffusion data for {mat.key}: healing of the collapsed pores not assessed")
        elif tp >= HEALING_UNCERTAINTY_FACTOR * t_bond:
            healing_status = Status.STATE_OF_THE_ART
        elif tp * HEALING_UNCERTAINTY_FACTOR >= t_bond:
            healing_status = Status.EXTRAPOLATION
            warnings.append(f"bonding time ~{t_bond:.3g} s is within a factor {HEALING_UNCERTAINTY_FACTOR:g} of "
                            f"the {tp:.3g} s dwell: inside the uncertainty of the diffusion estimate")
        else:
            healing_status = Status.INFEASIBLE
            if closed_qs:
                warnings.append(f"closed but not healed: diffusion bonding needs ~{t_bond:.3g} s, the pulse "
                                f"lasts {tp:.3g} s; the collapsed pore remains an unbonded, crack-like flaw")
            else:
                warnings.append(f"no healing either: diffusion bonding needs ~{t_bond:.3g} s, the pulse lasts "
                                f"{tp:.3g} s; any pore that did collapse would remain an unbonded, crack-like "
                                "flaw")
        status = worst(closure_status, healing_status)

    return PoreClosureAssessment(
        status=status, margin=margin, warnings=tuple(warnings), notes=tuple(notes), material=mat.key,
        pressure=p, field_equivalent=field_for_pressure(p), temperature=t, pulse_duration=tp, strain_rate=rate,
        initial_porosity=f0, pore_radius=radius, gas_pressure=pg, flow_stress_quasi_static=s_qs,
        flow_stress_dynamic=s_dyn, quasi_static_basis=strength.quasi_static_basis,
        dynamic_model=strength.dynamic_model, collapse_pressure_quasi_static=pc_qs,
        collapse_pressure_dynamic=pc_dyn, closure_pressure_quasi_static=p_close_qs,
        closure_pressure_dynamic=p_close_dyn, collapses_quasi_static=collapses_qs,
        collapses_dynamic=collapses_dyn, residual_porosity_quasi_static=f_qs, residual_porosity_dynamic=f_dyn,
        closed_quasi_static=closed_qs, closed_dynamic=closed_dyn, gas_limited_porosity=f_gas,
        collapse_time=t_collapse, wall_strain=wall_strain, ductility=ductility,
        wall_strain_to_ductility=strain_ratio, dbtt=dbtt, below_dbtt=below, confinement_ratio=confinement,
        creep_regime=creep, diffusivity=diffusivity, diffusion_length=x_diff, bond_length=l_bond,
        diffusion_bond_time=t_bond, bonded=bonded, drive_frequency=freq, skin_depth=skin,
        section_thickness=section, transparent=transparent, pressure_transfer=transfer, hip=cycle,
        matches_hip_practice=hip_match, closure_status=closure_status, healing_status=healing_status,
        extrapolated_model=strength.extrapolated,
    )


# ---------------------------------------------------------------------------
# Powder densification (the nearest real magnetic process)
# ---------------------------------------------------------------------------
def _stage_one_pressure(density: float, flow_stress: float, initial_density: float) -> float:
    return 3.0 * density * density * (density - initial_density) / (1.0 - initial_density) * flow_stress


def powder_yield_pressure(relative_density: float, flow_stress: float,
                          initial_density: float = RANDOM_DENSE_PACKING) -> float:
    """Pressure that densifies a powder to ``relative_density`` by plastic yielding alone [Pa].

    Helle-Easterling-Ashby two-stage model of a packing of spheres::

        stage 1 (D < 0.9):   P = 3 D^2 (D - D0) / (1 - D0) sigma_y     growing particle contacts
        stage 2 (D >= 0.9):  P = (2/3) sigma_y ln(1 / (1 - D))         isolated spherical pores

    ``relative_density`` D in [D0, 1] (``math.inf`` at D = 1), ``flow_stress`` sigma_y of the
    particles in Pa (>= 0), ``initial_density`` D0 in (0, 1) (default 0.64, random dense
    packing).  Stage 2 is :func:`plastic_collapse_pressure` with f = 1 - D.  The stages do
    not join: at D = 0.9 stage 1 gives 1.755 sigma_y and stage 2 1.535 sigma_y for D0 = 0.64.
    Check (D0 = 0.64): 80 % needs 0.85 sigma_y, 95 % needs 2.00 sigma_y, 99 % needs 3.07 sigma_y.

    Validity: rate-independent plasticity of ductile, equal spheres, no particle
    rearrangement or fracture, no hardening (lower bound on the pressure), no gas.

    Reference: Helle, Easterling & Ashby (1985); Arzt, Ashby & Easterling (1983); Torre (1948).
    """
    d = _closed_fraction("relative_density", relative_density)
    s = _non_negative("flow_stress", flow_stress)
    d0 = _open_fraction("initial_density", initial_density)
    if d < d0:
        raise InputError(f"relative_density ({d!r}) must not be below initial_density ({d0!r})")
    if d >= STAGE_TRANSITION_DENSITY:
        return plastic_collapse_pressure(s, 1.0 - d)
    return _stage_one_pressure(d, s, d0)


def powder_density_at_pressure(pressure: float, flow_stress: float,
                               initial_density: float = RANDOM_DENSE_PACKING) -> float:
    """Relative density a powder reaches under ``pressure`` by plastic yielding.

    Densification proceeds while the pressure exceeds :func:`powder_yield_pressure`: the
    stage-1 root of ``3 D^2 (D - D0)/(1 - D0) sigma_y = P`` (bisection) while it stays below
    0.9; once the pressure carries the compact to 0.9 the stage-2 density
    ``max(0.9, 1 - exp(-3 P / (2 sigma_y)))`` applies.  For D0 >= 0.9 only stage 2 exists:
    ``max(D0, 1 - exp(-3 P / (2 sigma_y)))``.  ``pressure`` and ``flow_stress`` in Pa (>= 0),
    ``initial_density`` D0 in (0, 1).  Non-decreasing in P; because the stages do not join,
    the result steps from 0.900 to 0.928 at P = 1.755 sigma_y (D0 = 0.64).  A powder without
    strength (sigma_y = 0) is fully dense under any pressure.

    Validity: as :func:`powder_yield_pressure`; this is the density under load of a *green*
    compact whose particles are not yet bonded.

    Reference: Helle, Easterling & Ashby (1985).
    """
    p = _non_negative("pressure", pressure)
    s = _non_negative("flow_stress", flow_stress)
    d0 = _open_fraction("initial_density", initial_density)
    if p == 0.0:
        return d0
    if s == 0.0:
        return 1.0
    stage_two = 1.0 - math.exp(-1.5 * p / s)
    if d0 >= STAGE_TRANSITION_DENSITY:
        return max(d0, stage_two)
    if p < _stage_one_pressure(STAGE_TRANSITION_DENSITY, s, d0):
        return bisect(lambda d: _stage_one_pressure(d, s, d0) - p, d0, STAGE_TRANSITION_DENSITY, tol=1e-14)
    return max(STAGE_TRANSITION_DENSITY, stage_two)


@dataclass(frozen=True, kw_only=True)
class PowderCompactionAssessment(Assessment):
    """Green density of a metal powder compacted by a pressure pulse.

    ``density_quasi_static`` / ``density_dynamic`` bracket the relative density under load
    (optimistic / conservative strength, :class:`MatrixStrength`);
    ``pressure_for_target_*`` are the pressures that reach ``target_density``;
    ``full_density_pressure_*`` those that reach 1 - :data:`CLOSURE_POROSITY`;
    ``field_equivalent`` is ``sqrt(2 mu0 P)``; ``within_dmc_class`` compares the pressure with
    :data:`DMC_PRESSURE_CLASS`; ``sintering_required`` is True when the particles cannot
    diffusion-bond during the pulse (always, at room temperature).  ``margin`` =
    ``pressure / pressure_for_target_dynamic``.  The status grades the compaction step only.

    Reference: Helle, Easterling & Ashby (1985); Chelluri & Barber (1999).
    """

    _title: ClassVar[str] = "Powder-compaction assessment"

    material: str
    pressure: float = unit("Pa")
    field_equivalent: float = unit("T")
    temperature: float = unit("K")
    pulse_duration: float = unit("s")
    strain_rate: float = unit("1/s")
    initial_density: float
    target_density: float
    flow_stress_quasi_static: float = unit("Pa")
    flow_stress_dynamic: float = unit("Pa")
    quasi_static_basis: str
    dynamic_model: str
    density_quasi_static: float
    density_dynamic: float
    pressure_for_target_quasi_static: float = unit("Pa")
    pressure_for_target_dynamic: float = unit("Pa")
    full_density_pressure_quasi_static: float = unit("Pa")
    full_density_pressure_dynamic: float = unit("Pa")
    within_dmc_class: bool
    dbtt: float | None = unit("K")
    below_dbtt: bool | None
    diffusion_bond_time: float | None = unit("s")
    sintering_required: bool | None
    extrapolated_model: bool


def assess_powder_compaction(material: MaterialLike, pressure: float,
                             initial_density: float = RANDOM_DENSE_PACKING, temperature: float = T_ROOM,
                             pulse_duration: float = 1e-4, strain_rate: float | None = None,
                             target_density: float = DEFAULT_GREEN_DENSITY_TARGET) -> PowderCompactionAssessment:
    """Assess compacting a metal powder with a pressure pulse (dynamic magnetic compaction).

    Equations: green density :func:`powder_density_at_pressure` at the quasi-static and the
    dynamic particle strength of :func:`matrix_strength`; pressures for ``target_density`` and
    for full density from :func:`powder_yield_pressure`; bonding time
    :func:`diffusion_bonding_time` against ``pulse_duration``.  ``pressure`` [Pa] is the
    mechanical pressure in the powder, ``initial_density`` D0 and ``target_density`` relative
    densities in (0, 1), ``temperature`` [K], ``pulse_duration`` [s], ``strain_rate`` [1/s]
    (default ``1/pulse_duration``).

    Status (first match, then worst-combined with the rules that also fire):

    1. target not reached even at the quasi-static strength -> infeasible;
    2. reached at the quasi-static strength only (band straddles) -> extrapolation;
    3. pressure above :data:`DMC_PRESSURE_CLASS` (2 GPa, the order of the pressures demonstrated
       by dynamic magnetic compaction) -> extrapolation;
    4. particles below their DBTT (they fracture and rearrange instead of flowing: the
       plastic model does not describe them) -> extrapolation;
    5. flow-stress model outside its calibration -> extrapolation;
    6. otherwise state of the art: dynamic magnetic compaction of ductile powders at
       GPa-class, sub-millisecond pressures is demonstrated practice.

    The status grades the *compaction* step.  The product is a green compact: its particles
    are pressed together, not diffusion-bonded (``sintering_required``), so sintering or HIP
    must follow, and the pressure reaches the powder through a conducting armature that the
    field accelerates - a moving solid part.  Iron powder at 1.6 GPa: more than 99.9 % green
    density, state of the art; gamma-TiAl powder at 4 GPa: 98.6-100 %, extrapolation.

    Validity: as :func:`powder_yield_pressure`; no shock heating, no interparticle melting.
    Reference: Helle, Easterling & Ashby (1985); Chelluri & Barber (1999); Psyk et al. (2011).
    """
    mat = as_material(material)
    p = _non_negative("pressure", pressure)
    d0 = _open_fraction("initial_density", initial_density)
    tp = _positive("pulse_duration", pulse_duration)
    rate = 1.0 / tp if strain_rate is None else _non_negative("strain_rate", strain_rate)
    target = _open_fraction("target_density", target_density)

    strength = matrix_strength(mat, rate, temperature)
    t = strength.temperature
    s_qs, s_dyn = strength.quasi_static, strength.dynamic
    d_qs = powder_density_at_pressure(p, s_qs, d0)
    d_dyn = powder_density_at_pressure(p, s_dyn, d0)
    if target <= d0:
        p_target_qs = p_target_dyn = 0.0
    else:
        p_target_qs = powder_yield_pressure(target, s_qs, d0)
        p_target_dyn = powder_yield_pressure(target, s_dyn, d0)
    p_full_qs = plastic_collapse_pressure(s_qs, CLOSURE_POROSITY)
    p_full_dyn = plastic_collapse_pressure(s_dyn, CLOSURE_POROSITY)
    within = p <= DMC_PRESSURE_CLASS
    dbtt = mat.dbtt
    below = None if dbtt is None else t < dbtt
    if mat.self_diffusion_activation is None:
        t_bond = None
        sinter: bool | None = None
    else:
        t_bond = diffusion_bonding_time(mat, t)
        sinter = tp < t_bond

    fired: list[Status] = []
    warnings: list[str] = []
    notes: list[str] = [
        (f"Green density under load by plastic yielding (Helle-Easterling-Ashby): {_density_pct(d_qs)} at the "
         f"quasi-static particle strength ({_gpa(s_qs)}, {strength.quasi_static_basis}), {_density_pct(d_dyn)} "
         f"at the dynamic strength ({_gpa(s_dyn)}); {_density_pct(target)} needs {_gpa(p_target_qs)} / "
         f"{_gpa(p_target_dyn)}, {_density_pct(1.0 - CLOSURE_POROSITY)} needs {_gpa(p_full_qs)} / "
         f"{_gpa(p_full_dyn)}."),
        ("The product is a green compact: particles are pressed into contact, not diffusion-bonded"
         + (f" (bonding takes ~{t_bond:.3g} s at {t:.0f} K against a {tp:.3g} s pulse)" if t_bond is not None
            else "")
         + "; sintering or HIP must follow."),
        ("The field does not press on the powder directly: it collapses a conducting armature (sleeve or "
         "punch) onto it, a moving solid part, and the armature's momentum can raise the powder pressure "
         "above the magnetic pressure."),
        (f"Field equivalent of the pressure: {field_for_pressure(p):.3g} T; demonstrated class of dynamic "
         f"magnetic compaction: {_gpa(DMC_PRESSURE_CLASS)} (order of magnitude; Chelluri & Barber 1999)."),
    ]
    if d_qs < target:
        fired.append(Status.INFEASIBLE)
        warnings.append(f"pressure insufficient for the target green density even at the quasi-static "
                        f"strength: {_density_pct(d_qs)} < {_density_pct(target)} (needs {_gpa(p_target_qs)})")
    elif d_dyn < target:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"target reached at the quasi-static strength only (strength band straddles the "
                        f"pressure): {_density_pct(d_dyn)} at the dynamic strength (needs "
                        f"{_gpa(p_target_dyn)})")
    if not within:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"{_gpa(p)} is above the {_gpa(DMC_PRESSURE_CLASS)} class demonstrated for dynamic "
                        "magnetic compaction")
    if below:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"particles are below their DBTT ({dbtt:.0f} K): they fracture and rearrange rather than "
                        "flow, which the plastic-yield model does not describe")
    if strength.extrapolated:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"flow-stress model extrapolated: {strength.extrapolation_note}")
    status = worst(fired) if fired else Status.STATE_OF_THE_ART
    return PowderCompactionAssessment(
        status=status, margin=p / p_target_dyn if p_target_dyn > 0.0 else math.inf, warnings=tuple(warnings),
        notes=tuple(notes), material=mat.key, pressure=p, field_equivalent=field_for_pressure(p),
        temperature=t, pulse_duration=tp, strain_rate=rate, initial_density=d0, target_density=target,
        flow_stress_quasi_static=s_qs, flow_stress_dynamic=s_dyn,
        quasi_static_basis=strength.quasi_static_basis, dynamic_model=strength.dynamic_model,
        density_quasi_static=d_qs, density_dynamic=d_dyn, pressure_for_target_quasi_static=p_target_qs,
        pressure_for_target_dynamic=p_target_dyn, full_density_pressure_quasi_static=p_full_qs,
        full_density_pressure_dynamic=p_full_dyn, within_dmc_class=within, dbtt=dbtt, below_dbtt=below,
        diffusion_bond_time=t_bond, sintering_required=sinter, extrapolated_model=strength.extrapolated,
    )


__all__ = [
    "REFERENCES",
    "QUASI_STATIC_RATE", "FALLBACK_BASIS", "DEFAULT_POROSITY", "DEFAULT_PORE_RADIUS", "CLOSURE_POROSITY",
    "RAYLEIGH_COEFFICIENT", "DIFFUSION_PREFACTOR", "DEFAULT_BOND_LENGTH", "HEALING_UNCERTAINTY_FACTOR",
    "CREEP_HOMOLOGOUS_TEMPERATURE", "PORE_WALL_CONFINEMENT", "HIP_PRESSURE_RANGE", "HIP_TEMPERATURE_TOLERANCE",
    "MAGNETIC_PULSE_DURATION_RANGE", "RANDOM_DENSE_PACKING", "STAGE_TRANSITION_DENSITY",
    "DEFAULT_GREEN_DENSITY_TARGET", "DMC_BENCHMARK", "DMC_PRESSURE_CLASS",
    "plastic_collapse_pressure", "gas_limited_porosity", "porosity_under_pressure", "pore_wall_strain",
    "rayleigh_collapse_time", "MatrixStrength", "matrix_strength", "collapse_pressure", "residual_porosity",
    "self_diffusivity", "diffusion_length", "diffusion_bonding_time",
    "HipCycle", "HIP_PRACTICE", "hip_cycle", "matches_hip_practice",
    "PoreClosureAssessment", "assess_pore_closure",
    "powder_yield_pressure", "powder_density_at_pressure", "PowderCompactionAssessment",
    "assess_powder_compaction",
]
