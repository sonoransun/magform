"""Microstructure after a cold, fast magnetic forming pulse: dislocations, grains and heat.

The concept note closes by asking "how room-temperature solid-state forging
affects a specific alloy's crystalline microstructure", and claims that cold
magnetic forging of tungsten, molybdenum and rhenium "preserves a pristine,
ultra-fine crystalline grain structure, resulting in parts with unprecedented
fracture toughness and fatigue resistance".  This module turns that sentence
into four separate, checkable statements and grades each one with the package's
own flow-stress and skin-heating models.

1. **"Preserves" - true in the bulk.**  Recrystallisation is thermally
   activated (:func:`recrystallisation_time`, an Arrhenius-scaled JMAK law
   anchored to the rule of thumb "complete in about 1 h at 0.4 T_m").  Plastic
   work heats wrought tungsten by 189 K at a strain of 0.3 and 1e4 1/s; at the
   resulting 482 K (0.13 T_m) the recrystallisation time is ~1e46 s.  The bulk
   grain structure is frozen: whatever grains the feedstock had, it keeps.
2. **"Ultra-fine" - not created by one pulse.**  A forming strain of 0.3 builds
   a dislocation cell structure *inside the original grains*
   (:func:`dislocation_density`, :func:`dislocation_cell_size`: ~4e13 1/m^2 and
   ~1.7 um cells for tungsten) and flattens those grains geometrically by
   exp(-0.3) = 0.74 (:func:`grain_size_after_strain`).  Sub-micrometre grains
   need the equivalent strains of severe plastic deformation, about 3 (to reach
   1 um) to 5 (to approach the saturation size) for 20 um tungsten
   (:func:`grain_refinement_strain`; ECAP and high-pressure torsion work at
   strains of 4-8).  The pulse preserves the feedstock grain size; it does not
   make an ultra-fine one.
3. **"Pristine" - not in the eddy-current skin.**  The field that presses the
   part also heats its skin by ``kappa B^2/(2 mu0 rho c_p)`` whatever the pulse
   length (:func:`skin_thermal_exposure`).  Two estimates bracket it: the layer
   average (kappa = 1, the lower estimate) and the outermost surface
   (kappa = 2.4).  At 100 T bare copper melts on the lower estimate (onset
   96 T; aluminium from 58 T); molybdenum reaches 0.64 T_m for about 6 ms and
   recrystallises; tungsten's layer average reaches 0.50 T_m for about 5 ms
   (recrystallised fraction below 0.1 %, but within 5 % in temperature of the
   1 % threshold) while its surface estimate melts.  The kinetic anchor is
   known only to 0.3-0.5 T_m, which moves these fractions between nil and
   complete, so the firm statement is the thermal one: the skin, a layer about
   half a skin depth thick, is a heat-affected zone at these fields unless a
   sacrificial driver sheet takes the current instead of the workpiece.
4. **"Unprecedented fracture toughness and fatigue resistance" - not modelled.**
   Cold work raises strength (Taylor hardening, tungsten 1.23 -> 1.42 GPa) and
   normally lowers ductility.  Heavily worked tungsten does have a lower
   ductile-brittle transition temperature than recrystallised tungsten, but that
   comes from the rolling strains of severe working (> 3), not from one pulse.
   This module computes no toughness and no fatigue life and says so in every
   assessment.

Two further cautions are reported by :func:`assess_microstructure`: the Culver
adiabatic-instability strain of tungsten (0.004) is far below any forming
strain, so the deformation tends to concentrate in shear bands whose interior
recrystallises dynamically (:func:`derby_grain_size`), and below the
ductile-brittle transition temperature the predicted structure exists only if
hydrostatic confinement suppresses fracture (:func:`magform.plasticity.assess_forgeability`).

Model limits: every relation here is a scaling law with estimated constants
(Taylor alpha 0.2-0.5, cell constant 10-20, recrystallisation anchor 0.3-0.5 T_m);
results are order-of-magnitude estimates that are consistent with the flow
stress the rest of the package uses, not microstructure simulations.  Twinning,
texture, recovery kinetics, grain growth and phase transformations are not
modelled.  SI units throughout (dislocation density in 1/m^2, sizes in m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from ._report import Assessment, ResultMixin, unit
from .constants import R_GAS, T_ROOM
from .eddy import KAPPA_STEP, skin_depth_for_material, skin_melting_field, skin_temperature_rise
from .errors import InputError
from .materials import Material, MaterialLike, as_material
from .plasticity import (
    RATE_FLOOR,
    adiabatic_shear_critical_strain,
    adiabatic_temperature_rise,
    flow_stress,
    flow_stress_band,
)
from .status import Status, worst

REFERENCES: dict[str, str] = {
    "taylor1934": "G. I. Taylor, 'The mechanism of plastic deformation of crystals. Part I. Theoretical', "
                  "Proc. R. Soc. Lond. A 145 (1934) 362-387 (flow stress proportional to the square root of the "
                  "dislocation density).",
    "taylor1938": "G. I. Taylor, 'Plastic strain in metals', J. Inst. Metals 62 (1938) 307-324 (Taylor factor "
                  "M = 3.06 of a random fcc polycrystal).",
    "kocks1970": "U. F. Kocks, 'The relation between polycrystal deformation and single-crystal deformation', "
                 "Metall. Trans. 1 (1970) 1121-1143 (Taylor factors, including bcc pencil glide).",
    "kocks_mecking2003": "U. F. Kocks and H. Mecking, 'Physics and phenomenology of strain hardening: the FCC "
                         "case', Prog. Mater. Sci. 48 (2003) 171-273.",
    "holt1970": "D. L. Holt, 'Dislocation cell formation in metals', J. Appl. Phys. 41 (1970) 3197-3201 "
                "(cell size proportional to the inverse square root of the dislocation density).",
    "hall1951": "E. O. Hall, 'The deformation and ageing of mild steel: III Discussion of results', "
                "Proc. Phys. Soc. B 64 (1951) 747-753.",
    "petch1953": "N. J. Petch, 'The cleavage strength of polycrystals', J. Iron Steel Inst. 174 (1953) 25-28.",
    "meyers2006": "M. A. Meyers, A. Mishra and D. J. Benson, 'Mechanical properties of nanocrystalline "
                  "materials', Prog. Mater. Sci. 51 (2006) 427-556 (breakdown of Hall-Petch strengthening below "
                  "about 10-20 nm).",
    "humphreys2004": "F. J. Humphreys and M. Hatherly, Recrystallization and Related Annealing Phenomena, "
                     "2nd ed., Elsevier, Oxford (2004): stored energy of cold work (ch. 2), JMAK kinetics and "
                     "measured Avrami exponents (ch. 7), continuous recrystallisation at large strain (ch. 14).",
    "humphreys1999": "F. J. Humphreys, P. B. Prangnell, J. R. Bowen, A. Gholinia and C. Harris, 'Developing "
                     "stable fine-grain microstructures by large strain deformation', Phil. Trans. R. Soc. "
                     "Lond. A 357 (1999) 1663-1681 (high-angle boundary spacing reduced geometrically by strain "
                     "until it reaches the subgrain size).",
    "hughes_hansen2000": "D. A. Hughes and N. Hansen, 'Microstructure and strength of nickel at large strains', "
                         "Acta Mater. 48 (2000) 2985-3004.",
    "valiev2000": "R. Z. Valiev, R. K. Islamgaliev and I. V. Alexandrov, 'Bulk nanostructured materials from "
                  "severe plastic deformation', Prog. Mater. Sci. 45 (2000) 103-189.",
    "pippan2010": "R. Pippan, S. Scheriau, A. Taylor, M. Hafok, A. Hohenwarter and A. Bachmaier, 'Saturation "
                  "of fragmentation during severe plastic deformation', Annu. Rev. Mater. Res. 40 (2010) 319-343.",
    "estrin_vinogradov2013": "Y. Estrin and A. Vinogradov, 'Extreme grain refinement by severe plastic "
                             "deformation: a wealth of challenging science', Acta Mater. 61 (2013) 782-817.",
    "avrami1939": "M. Avrami, 'Kinetics of phase change. I. General theory', J. Chem. Phys. 7 (1939) 1103-1112.",
    "johnson_mehl1939": "W. A. Johnson and R. F. Mehl, 'Reaction kinetics in processes of nucleation and "
                        "growth', Trans. AIME 135 (1939) 416-458.",
    "dieter1986": "G. E. Dieter, Mechanical Metallurgy, 3rd ed., McGraw-Hill, New York (1986): recrystallisation "
                  "temperature (complete in about one hour) of 0.3-0.5 T_m; cold, warm and hot working.",
    "zener_hollomon1944": "C. Zener and J. H. Hollomon, 'Effect of strain rate upon plastic flow of steel', "
                          "J. Appl. Phys. 15 (1944) 22-32 (temperature-compensated strain rate).",
    "derby1991": "B. Derby, 'The dependence of grain size on stress during dynamic recrystallisation', "
                 "Acta Metall. Mater. 39 (1991) 955-962.",
    "tabor1951": "D. Tabor, The Hardness of Metals, Clarendon Press, Oxford (1951) (H ~ 3 x flow stress at a "
                 "representative strain of 8 %).",
    "culver1973": "R. S. Culver, 'Thermal instability strain in dynamic plastic deformation', in R. W. Rohde, "
                  "B. M. Butcher, J. R. Holland and C. H. Karnes (eds.), Metallurgical Effects at High Strain "
                  "Rates, Plenum, New York (1973).",
    "meyers1994": "M. A. Meyers, Dynamic Behavior of Materials, Wiley, New York (1994) (adiabatic shear bands "
                  "and the fine recrystallised grains inside them).",
    "wei2006": "Q. Wei, T. Jiao, K. T. Ramesh, E. Ma et al., 'Mechanical behavior and dynamic failure of "
               "high-strength ultrafine grained tungsten under uniaxial compression', Acta Mater. 54 (2006) "
               "77-87 (ultrafine-grained W made by severe plastic deformation, ~3 GPa dynamic strength).",
    "lassner1999": "E. Lassner and W.-D. Schubert, Tungsten: Properties, Chemistry, Technology of the Element, "
                   "Alloys, and Chemical Compounds, Kluwer Academic/Plenum, New York (1999) (worked tungsten has "
                   "a lower ductile-brittle transition temperature than recrystallised tungsten).",
    "knoepfel1970": "H. Knoepfel, Pulsed High Magnetic Fields, North-Holland, Amsterdam (1970) (skin heating "
                    "B^2/(2 mu0 rho c_p)).",
    "carslaw1959": "H. S. Carslaw and J. C. Jaeger, Conduction of Heat in Solids, 2nd ed., Oxford University "
                   "Press (1959) (cooling of an exponentially heated surface layer).",
}

# ---------------------------------------------------------------------------
# Module constants (documented single sources)
# ---------------------------------------------------------------------------
TAYLOR_FACTOR: dict[str, float] = {"fcc": 3.06, "bcc": 2.75}
"""Taylor factor M (tensile flow stress / resolved shear stress) of a randomly textured
polycrystal: 3.06 for fcc {111}<110> slip (Taylor 1938), 2.75 for bcc pencil glide
(Kocks 1970).  Textured material differs by up to ~20 %."""

DEFAULT_TAYLOR_FACTOR: float = 3.06
"""Taylor factor used for crystal structures not in :data:`TAYLOR_FACTOR` (hcp, L1_0, ...).
Estimated: hexagonal metals have no single value (2.5-6.5 with texture and active systems)."""

TAYLOR_ALPHA: float = 0.3
"""Dislocation-interaction constant alpha of the Taylor relation tau = alpha G b sqrt(rho)
(confidence: estimated; see :data:`TAYLOR_ALPHA_RANGE`; Kocks & Mecking 2003)."""

TAYLOR_ALPHA_RANGE: tuple[float, float] = (0.2, 0.5)
"""Range of alpha reported for metals.  Because rho ~ 1/alpha^2, this range alone moves a
dislocation density inferred from a stress increment by a factor of about six."""

INITIAL_DISLOCATION_DENSITY: float = 1e12
"""1/m^2.  Default starting dislocation density: a well-annealed metal (1e10-1e12 1/m^2).
Wrought or stress-relieved stock starts higher (1e13-1e14 1/m^2); pass ``initial_density``."""

CELL_SIZE_CONSTANT: float = 10.0
"""Similitude constant K of the dislocation cell size d = K/sqrt(rho) (Holt 1970); measured
values lie in :data:`CELL_SIZE_CONSTANT_RANGE`."""

CELL_SIZE_CONSTANT_RANGE: tuple[float, float] = (10.0, 20.0)
"""Range of the cell-size constant K (Holt 1970; Kocks & Mecking 2003)."""

STORED_ENERGY_FACTOR: float = 0.5
"""Line energy of a dislocation in units of G b^2 (about 1/2; Humphreys & Hatherly 2004 ch. 2)."""

ULTRAFINE_GRAIN_SIZE: float = 1e-6
"""m.  Conventional upper bound of an 'ultrafine-grained' metal (< 1 um; nanocrystalline
< 100 nm; Valiev, Islamgaliev & Alexandrov 2000)."""

HALL_PETCH_FLOOR: float = 20e-9
"""m.  Grain size below which Hall-Petch strengthening is not extended: the relation breaks
down (plateau or inverse Hall-Petch softening) near 10-20 nm (Meyers, Mishra & Benson 2006).
:func:`hall_petch_strength` evaluates finer grains at this size."""

DEFAULT_SATURATION_GRAIN_SIZE: float = 200e-9
"""m.  Saturation grain size of severe plastic deformation assumed for materials without a
recorded ``spd_saturation_grain_size`` (estimated; pure metals and alloys saturate at
100-500 nm, Pippan et al. 2010)."""

SATURATION_APPROACH_FACTOR: float = 2.0
"""Default target of :func:`grain_refinement_strain`, as a multiple of the saturation grain
size: 'approaching saturation' means within a factor of two of it."""

RECRYSTALLISATION_HOMOLOGOUS: float = 0.4
"""Anchor of the recrystallisation kinetics: a heavily cold-worked metal recrystallises
(:data:`RECRYSTALLISATION_REFERENCE_FRACTION`) in :data:`RECRYSTALLISATION_REFERENCE_TIME`
at this fraction of its melting point (rule of thumb; confidence: estimated; Dieter 1986).
Reproduces W ~1200 C, Mo ~890 C, Cu ~270 C."""

RECRYSTALLISATION_HOMOLOGOUS_RANGE: tuple[float, float] = (0.3, 0.5)
"""Range of the one-hour recrystallisation temperature in units of T_m (pure, heavily worked
metals near 0.3; alloys, lightly worked or coarse-grained stock near 0.5).  Used for the
uncertainty band of :func:`skin_thermal_exposure` and for the cold/warm/hot regimes."""

RECRYSTALLISATION_REFERENCE_TIME: float = 3600.0
"""s.  Anneal time of the recrystallisation-temperature rule of thumb (one hour)."""

RECRYSTALLISATION_REFERENCE_FRACTION: float = 0.95
"""Recrystallised fraction counted as 'complete' at the anchor (the recrystallisation
temperature is defined by completion in one hour; Dieter 1986)."""

AVRAMI_EXPONENT: float = 2.0
"""JMAK (Avrami) exponent n.  Estimated: measured exponents for recrystallisation are mostly
1-3, below the ideal 3-4 of random nucleation (Humphreys & Hatherly 2004 ch. 7)."""

RECRYSTALLISED_THRESHOLD: float = 0.01
"""Recrystallised fraction above which a region is counted as heat-affected (and below which
the grain structure is counted as preserved): 1 %, a detection-level judgement."""

BULK_HOLD_TIME: float = 60.0
"""s.  Default time the bulk is assumed to stay at its peak adiabatic temperature before the
part cools (generous for cm-scale parts, whose conduction time is seconds)."""

DEFAULT_PULSE_FREQUENCY: float = 1e4
"""Hz.  Pulse frequency f = 1/(4 t_rise) assumed for the skin when none is given: 10 kHz
(25 us rise), typical of electromagnetic-forming discharges."""

SURFACE_KAPPA: float = KAPPA_STEP
"""Surface-heating factor of the *upper* skin estimate (2.4, :data:`magform.eddy.KAPPA_STEP`):
the outermost surface of a conductor ends a half-sine pulse at 2.2-2.4 x B^2/(2 mu0 rho c_p),
whereas kappa = 1 (the default of :func:`skin_thermal_exposure`) is the lower, layer-average
estimate (Knoepfel 1970)."""

REGIME_REFERENCE_RATE: float = 1.0
"""1/s.  Strain rate at which the cold/warm/hot working boundaries are quoted (conventional
press forging); :func:`equivalent_temperature` maps other rates onto it."""

QUASI_STATIC_RATE: float = 1e-3
"""1/s.  Rate at which the as-formed yield strength and hardness are evaluated (tensile-test rate)."""

TABOR_FACTOR: float = 3.0
"""Tabor (1951) constraint factor: indentation hardness ~ 3 x flow stress."""

TABOR_REPRESENTATIVE_STRAIN: float = 0.08
"""Representative plastic strain a Vickers indenter adds to the current state (Tabor 1951)."""

DERBY_K_RANGE: tuple[float, float] = (1.0, 10.0)
"""Range of the constant K in Derby's (1991) piezometer (sigma/G)(d/b)^(2/3) = K across metals
and minerals.  No single value is adopted: results are reported for both ends."""

CULVER_MAX_STRAIN: float = 5.0
"""Largest strain searched for the Culver adiabatic instability (as in :mod:`magform.plasticity`)."""

SURFACE_EXCESS_AT_COOLING_TIME: float = math.e * math.erfc(1.0)
"""Fraction (0.428) of its peak excess temperature that the surface of an exponentially
heated layer retains one conduction time L^2/alpha after the pulse: exp(1) erfc(1), from
T_s(t)/dT = exp(tau) erfc(sqrt(tau)), tau = alpha t/L^2 (Carslaw & Jaeger 1959)."""

_EXP_OVERFLOW: float = 700.0
_LN10: float = math.log(10.0)


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
        raise InputError(f"{name} must be > 0, got {v:g}")
    return v


def _non_negative(name: str, value: Any) -> float:
    v = _finite(name, value)
    if v < 0.0:
        raise InputError(f"{name} must be >= 0, got {v:g}")
    return v


def _open_fraction(name: str, value: Any) -> float:
    v = _finite(name, value)
    if not 0.0 < v < 1.0:
        raise InputError(f"{name} must lie strictly between 0 and 1, got {v:g}")
    return v


def _solid_temperature(mat: Material, value: Any, name: str = "temperature") -> float:
    t = _finite(name, value)
    if t <= 0.0:
        raise InputError(f"{name} must be > 0 K, got {t:g}")
    if t >= mat.melting_point:
        raise InputError(f"{name} {t:.1f} K is at or above the melting point of '{mat.key}' "
                         f"({mat.melting_point:.0f} K): no solid microstructure")
    return t


def _activation_energy(mat: Material) -> float:
    mat.require("self_diffusion_activation")
    q = float(mat.self_diffusion_activation)  # type: ignore[arg-type]
    if not (math.isfinite(q) and q > 0.0):
        raise InputError(f"material '{mat.key}' needs a positive self_diffusion_activation")
    return q


def _burgers(mat: Material) -> float:
    mat.require("burgers_vector")
    b = float(mat.burgers_vector)  # type: ignore[arg-type]
    if not (math.isfinite(b) and b > 0.0):
        raise InputError(f"material '{mat.key}' needs a positive burgers_vector")
    return b


def _shear_modulus(mat: Material) -> float:
    return float(mat.shear_modulus)  # type: ignore[arg-type]


def _initial_grain_size(mat: Material, initial_grain_size: Any) -> float:
    if initial_grain_size is not None:
        return _positive("initial_grain_size", initial_grain_size)
    if mat.typical_grain_size is None:
        raise InputError(f"material '{mat.key}' has no typical_grain_size; pass initial_grain_size")
    return float(mat.typical_grain_size)


def _saturation_grain_size(mat: Material, saturation_grain_size: Any) -> float:
    if saturation_grain_size is not None:
        return _positive("saturation_grain_size", saturation_grain_size)
    if mat.spd_saturation_grain_size is not None:
        return float(mat.spd_saturation_grain_size)
    return DEFAULT_SATURATION_GRAIN_SIZE


def _um(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value * 1e6:.3g} um"


def _gpa(value: float) -> str:
    return f"{value / 1e9:.3g} GPa"


# ---------------------------------------------------------------------------
# Dislocations: Taylor hardening, cells, stored energy
# ---------------------------------------------------------------------------
def taylor_factor(material: MaterialLike) -> float:
    """Taylor factor M of the material's crystal structure (dimensionless).

    ``M = TAYLOR_FACTOR[crystal_structure]`` (fcc 3.06, bcc 2.75), else
    :data:`DEFAULT_TAYLOR_FACTOR` (3.06).  M converts the resolved shear stress on the
    slip systems into the polycrystal's uniaxial flow stress, ``sigma = M tau``.

    Validity: random texture; rolled or drawn stock differs by up to ~20 %; hexagonal
    and intermetallic structures have no single value (estimated).
    Reference: Taylor (1938); Kocks (1970).
    """
    mat = as_material(material)
    return TAYLOR_FACTOR.get(mat.crystal_structure, DEFAULT_TAYLOR_FACTOR)


def taylor_hardening(material: MaterialLike, density: float,
                     initial_density: float = INITIAL_DISLOCATION_DENSITY,
                     alpha: float = TAYLOR_ALPHA) -> float:
    """Flow-stress increment of a dislocation density, ``d sigma = M alpha G b (sqrt(rho) - sqrt(rho0))`` [Pa].

    From the Taylor relation ``sigma = sigma_0 + M alpha G b sqrt(rho)`` written for the
    deformed (``density`` rho, 1/m^2) and the starting state (``initial_density`` rho0).
    M from :func:`taylor_factor`, G the shear modulus (Pa) and b the Burgers vector (m) of
    the record, ``alpha`` (default 0.3) dimensionless.  Negative when rho < rho0.

    Validity: forest hardening only (no solute, precipitate or grain-size change); alpha is
    known to within 0.2-0.5; order-of-magnitude link between stress and density.
    Reference: Taylor (1934); Kocks & Mecking (2003).
    """
    mat = as_material(material)
    rho = _non_negative("density", density)
    rho0 = _non_negative("initial_density", initial_density)
    a = _positive("alpha", alpha)
    return taylor_factor(mat) * a * _shear_modulus(mat) * _burgers(mat) * (math.sqrt(rho) - math.sqrt(rho0))


def _density_from_increment(mat: Material, d_sigma: float, rho0: float, alpha: float) -> float:
    root = math.sqrt(rho0) + max(d_sigma, 0.0) / (taylor_factor(mat) * alpha * _shear_modulus(mat) * _burgers(mat))
    return root * root


def dislocation_density(material: MaterialLike, strain: float, strain_rate: float = 1e-3,
                        temperature: float = T_ROOM,
                        initial_density: float = INITIAL_DISLOCATION_DENSITY,
                        alpha: float = TAYLOR_ALPHA) -> float:
    """Dislocation density after a plastic strain, from the package's own flow stress [1/m^2].

    The Taylor relation inverted for the strain-hardening increment of the primary
    constitutive model::

        sqrt(rho) = sqrt(rho0) + (sigma(eps) - sigma(0)) / (M alpha G b)

    with ``sigma`` = :func:`magform.plasticity.flow_stress` at the same ``strain_rate``
    (1/s) and ``temperature`` (K), so that only the isothermal hardening, not thermal
    softening, is counted.  ``initial_density`` rho0 in 1/m^2 (default 1e12, annealed).
    Check: wrought W, strain 0.3, 1e4 1/s, 293 K -> 3.7e13 1/m^2 (hardening 183 MPa).

    Validity: all strain hardening is attributed to forest dislocations; the answer scales
    as 1/alpha^2 (alpha 0.2-0.5) and inherits the hardening exponent of the published fit
    (see :func:`dislocation_density_band`); twinning and recovery are ignored, so above
    ~0.3 T_m it is an upper bound.  Raises :class:`InputError` for materials without a
    Burgers vector or flow-stress data.
    Reference: Taylor (1934); Kocks & Mecking (2003).
    """
    mat = as_material(material)
    rho0 = _non_negative("initial_density", initial_density)
    a = _positive("alpha", alpha)
    _burgers(mat)
    eps = _non_negative("strain", strain)
    s0 = flow_stress(mat, 0.0, strain_rate, temperature)
    s1 = flow_stress(mat, eps, strain_rate, temperature)
    return _density_from_increment(mat, s1 - s0, rho0, a)


def dislocation_density_band(material: MaterialLike, strain: float, strain_rate: float = 1e-3,
                             temperature: float = T_ROOM,
                             initial_density: float = INITIAL_DISLOCATION_DENSITY,
                             alpha: float = TAYLOR_ALPHA) -> tuple[float, float]:
    """(minimum, maximum) dislocation density over every published flow-stress fit [1/m^2].

    :func:`dislocation_density` evaluated with the hardening increment
    ``sigma(eps) - sigma(0)`` of each model that spans
    :func:`magform.plasticity.flow_stress_band` (the Hollomon fallback only when no
    published fit exists).  For wrought W at 0.3, 1e4 1/s the two Zerilli-Armstrong fits
    give 3.7e13 and 1.4e14 1/m^2: the spread is batch and fitting uncertainty, not a
    confidence interval.

    Validity: as :func:`dislocation_density`.
    Reference: Taylor (1934); Kocks & Mecking (2003).
    """
    mat = as_material(material)
    rho0 = _non_negative("initial_density", initial_density)
    a = _positive("alpha", alpha)
    _burgers(mat)
    eps = _non_negative("strain", strain)
    start = flow_stress_band(mat, 0.0, strain_rate, temperature)
    end = flow_stress_band(mat, eps, strain_rate, temperature)
    values = [_density_from_increment(mat, end.values[label] - start.values[label], rho0, a)
              for label in end.band_models]
    return min(values), max(values)


def dislocation_cell_size(density: float, k: float = CELL_SIZE_CONSTANT) -> float:
    """Dislocation cell (subgrain) size from similitude, ``d = K / sqrt(rho)`` [m].

    ``density`` rho in 1/m^2 (> 0), ``k`` the similitude constant (default 10; 10-20
    measured).  Check: 1e14 1/m^2 -> 1.0 um; 1e15 1/m^2 -> 0.32 um.

    Validity: cell-forming metals at low homologous temperature after enough strain for a
    cell structure to exist (>~ 0.05-0.1); the walls are *low-angle* boundaries inside the
    original grains, not new grains.
    Reference: Holt (1970); Kocks & Mecking (2003).
    """
    rho = _positive("density", density)
    kk = _positive("k", k)
    return kk / math.sqrt(rho)


def stored_energy_density(material: MaterialLike, density: float) -> float:
    """Stored energy of cold work carried by the dislocations, ``E = (1/2) G b^2 rho`` [J/m^3].

    ``density`` rho in 1/m^2 (>= 0); G (Pa) and b (m) from the record.  It is the driving
    pressure for recovery and recrystallisation (1 J/m^3 = 1 Pa).  Check: W at 3.7e13 1/m^2
    -> 2.2e5 J/m^3, about 0.04 % of the plastic work done at a strain of 0.3.

    Validity: line energy of isolated dislocations (factor ~1/2); boundaries, point defects
    and long-range stresses add to it, so calorimetric stored energies are higher.
    Reference: Humphreys & Hatherly (2004) ch. 2.
    """
    mat = as_material(material)
    rho = _non_negative("density", density)
    b = _burgers(mat)
    return STORED_ENERGY_FACTOR * _shear_modulus(mat) * b * b * rho


# ---------------------------------------------------------------------------
# Hall-Petch grain-size strengthening
# ---------------------------------------------------------------------------
def hall_petch_friction_stress(material: MaterialLike) -> float:
    """Friction stress sigma_0 of the Hall-Petch relation [Pa].

    ``Material.hall_petch.sigma0`` when recorded; otherwise calibrated so that the recorded
    yield strength is reproduced at the record's reference grain size::

        sigma_0 = sigma_y - k / sqrt(d_ref)

    (k in Pa m^0.5, d_ref in m, default 50 um).  Raises :class:`InputError` if the material
    has no Hall-Petch record, or no yield strength to calibrate against.

    Validity: the calibration assumes the recorded yield strength belongs to a grain size
    near ``reference_grain_size``; many k values in the registry are estimates.
    Reference: Hall (1951); Petch (1953).
    """
    mat = as_material(material)
    mat.require("hall_petch")
    hp = mat.hall_petch
    if hp.sigma0 is not None:  # type: ignore[union-attr]
        return float(hp.sigma0)  # type: ignore[union-attr]
    mat.require("yield_strength")
    return float(mat.yield_strength) - hp.k / math.sqrt(hp.reference_grain_size)  # type: ignore[arg-type,union-attr]


def hall_petch_strength(material: MaterialLike, grain_size: float) -> float:
    """Yield strength at a grain size, ``sigma_y = sigma_0 + k d^(-1/2)`` [Pa].

    ``grain_size`` d in m (> 0); sigma_0 from :func:`hall_petch_friction_stress`, k from
    the record.  Grain sizes below :data:`HALL_PETCH_FLOOR` (20 nm) are evaluated at the
    floor: the relation breaks down there (plateau or inverse Hall-Petch softening), so
    the strength is not extrapolated further.  Check: W at its 100 nm saturation size ->
    3.7 GPa, against 1.3 GPa at 50 um.

    Validity: equiaxed, recrystallised-type grains from ~20 nm to ~1 mm; deformation-induced
    (non-equilibrium) boundaries follow it only roughly.
    Reference: Hall (1951); Petch (1953); Meyers, Mishra & Benson (2006).
    """
    mat = as_material(material)
    d = _positive("grain_size", grain_size)
    sigma0 = hall_petch_friction_stress(mat)
    return sigma0 + mat.hall_petch.k / math.sqrt(max(d, HALL_PETCH_FLOOR))  # type: ignore[union-attr]


def grain_size_for_strength(material: MaterialLike, strength: float) -> float | None:
    """Grain size that gives a target yield strength, ``d = (k / (sigma - sigma_0))^2`` [m].

    Inverse of :func:`hall_petch_strength`.  ``strength`` in Pa (> 0).  Returns
    ``math.inf`` when the target does not exceed the friction stress sigma_0 (any grain
    size is at least that strong) and ``None`` when it would need grains finer than
    :data:`HALL_PETCH_FLOOR` (not reachable by grain refinement).

    Validity: as :func:`hall_petch_strength`.
    Reference: Hall (1951); Petch (1953).
    """
    mat = as_material(material)
    s = _positive("strength", strength)
    sigma0 = hall_petch_friction_stress(mat)
    if s <= sigma0:
        return math.inf
    d = (mat.hall_petch.k / (s - sigma0)) ** 2  # type: ignore[union-attr]
    if d < HALL_PETCH_FLOOR * (1.0 - 1e-9):
        return None
    return d


# ---------------------------------------------------------------------------
# Grain refinement with strain
# ---------------------------------------------------------------------------
def grain_size_after_strain(material: MaterialLike, strain: float, initial_grain_size: float | None = None,
                            saturation_grain_size: float | None = None) -> float:
    """High-angle-boundary spacing along the compression direction after a strain [m].

    ``d(eps) = d_sat + (d_0 - d_sat) exp(-eps)``

    The exponential is geometry, not a fitted rate: plastic flow thins a grain of size d_0
    to d_0 exp(-eps) in the compression (or rolling-normal) direction, and the original
    boundaries are the only high-angle boundaries until their spacing approaches the
    subgrain size; d_sat is the saturation grain size of severe plastic deformation
    (``Material.spd_saturation_grain_size``, else 200 nm estimated).  ``strain`` is the
    equivalent plastic strain (>= 0); ``initial_grain_size`` defaults to
    ``Material.typical_grain_size``.  A feedstock already finer than d_sat is returned
    unchanged.  Check: W, 20 um, strain 0.3 -> 14.8 um (not ultrafine); strain 6 -> 0.15 um.

    Validity: this is the *smallest* grain dimension of pancaked original grains - the
    grain volume is unchanged and the new cell walls inside (:func:`dislocation_cell_size`)
    are low-angle; monotonic strain below ~0.3 T_m (no recovery or recrystallisation);
    redundant strain paths (ECAP routes, torsion) refine at a similar equivalent strain to
    within a factor of about two.
    Reference: Humphreys et al. (1999); Humphreys & Hatherly (2004) ch. 14; Pippan et al. (2010).
    """
    mat = as_material(material)
    eps = _non_negative("strain", strain)
    d0 = _initial_grain_size(mat, initial_grain_size)
    ds = _saturation_grain_size(mat, saturation_grain_size)
    if d0 <= ds:
        return d0
    return ds + (d0 - ds) * math.exp(-eps)


def grain_refinement_strain(material: MaterialLike, target_grain_size: float | None = None,
                            initial_grain_size: float | None = None,
                            saturation_grain_size: float | None = None) -> float:
    """Equivalent strain needed to refine the grain size to a target (dimensionless).

    Inverse of :func:`grain_size_after_strain`::

        eps = ln((d_0 - d_sat) / (d_target - d_sat))

    ``target_grain_size`` (m) defaults to twice the saturation size ('approaching
    saturation', :data:`SATURATION_APPROACH_FACTOR`).  Returns 0 when the feedstock is
    already at or below the target and ``math.inf`` when the target is at or below the
    saturation size (not reachable by deformation alone).  Checks: W, 20 um -> 1 um needs
    3.1; W -> 0.2 um needs 5.3; Cu, 30 um -> 0.4 um needs 5.0.  A forming strain of 0.3 is
    an order of magnitude short: ultrafine grains need the strains of ECAP or high-pressure
    torsion (4-8), not of one forming pulse.

    Validity: as :func:`grain_size_after_strain`.
    Reference: Humphreys et al. (1999); Valiev, Islamgaliev & Alexandrov (2000); Pippan et al. (2010).
    """
    mat = as_material(material)
    d0 = _initial_grain_size(mat, initial_grain_size)
    ds = _saturation_grain_size(mat, saturation_grain_size)
    target = (SATURATION_APPROACH_FACTOR * ds if target_grain_size is None
              else _positive("target_grain_size", target_grain_size))
    if d0 <= target:
        return 0.0
    if target <= ds:
        return math.inf
    return math.log((d0 - ds) / (target - ds))


def hardness_after_strain(material: MaterialLike, strain: float = 0.0, temperature: float = T_ROOM) -> float:
    """Vickers-type hardness of the worked state, ``H = 3 sigma(eps + 0.08)`` [Pa].

    Tabor's relation with the flow stress from :func:`magform.plasticity.flow_stress` at the
    quasi-static rate (1e-3 1/s) and ``temperature`` (K): an indenter adds a representative
    strain of 8 % to the state it tests.  ``strain`` is the prior plastic strain (>= 0).
    Check: wrought W, strain 0 -> 4.2 GPa (recorded 4.3 GPa); strain 0.3 -> 4.3 GPa.

    Validity: Tabor's factor 3 holds for metals that are not strongly pressure-sensitive;
    the worked state is assumed to follow the same flow curve when reloaded.
    Reference: Tabor (1951).
    """
    mat = as_material(material)
    eps = _non_negative("strain", strain)
    return TABOR_FACTOR * flow_stress(mat, eps + TABOR_REPRESENTATIVE_STRAIN, QUASI_STATIC_RATE, temperature)


# ---------------------------------------------------------------------------
# Recrystallisation kinetics (Arrhenius-scaled JMAK)
# ---------------------------------------------------------------------------
def _completion_time(mat: Material, temperature: float, anchor: float) -> float:
    """Time (s) to the reference fraction at ``temperature``; inf when it overflows."""
    q = _activation_energy(mat)
    t_anchor = anchor * mat.melting_point
    exponent = (q / R_GAS) * (1.0 / temperature - 1.0 / t_anchor)
    if exponent > _EXP_OVERFLOW:
        return math.inf
    return RECRYSTALLISATION_REFERENCE_TIME * math.exp(exponent)


_JMAK_REFERENCE: float = -math.log1p(-RECRYSTALLISATION_REFERENCE_FRACTION)  # ln(1/(1 - 0.95)) = ln 20


def recrystallisation_time(material: MaterialLike, temperature: float, fraction: float = 0.5, *,
                           anchor: float = RECRYSTALLISATION_HOMOLOGOUS) -> float:
    """Time for a cold-worked metal to recrystallise to a given fraction at a temperature [s].

    Arrhenius-scaled Johnson-Mehl-Avrami-Kolmogorov kinetics anchored to the rule of thumb
    that recrystallisation is complete (95 %) in one hour at ``anchor`` x T_m (0.4)::

        t_95(T) = 3600 s * exp[(Q/R) (1/T - 1/(anchor T_m))]
        t_X(T)  = t_95(T) * [ln(1/(1 - X)) / ln 20]^(1/n),      n = 2

    Q is ``Material.self_diffusion_activation`` (J/mol), R the gas constant, ``temperature``
    in K (0 < T < T_m), ``fraction`` X strictly between 0 and 1.  Returns ``math.inf`` when
    the time exceeds the floating-point range.  Checks: W at 0.4 T_m (1478 K): 3600 s to
    95 %; W at 482 K (after forming): ~1e46 s; W at 1836 K: 0.34 s to 95 %.

    Validity: heavily cold-worked single-phase metal; order of magnitude only - the anchor
    spans 0.3-0.5 T_m with purity and strain (:data:`RECRYSTALLISATION_HOMOLOGOUS_RANGE`),
    the activation energy of boundary migration is 0.5-1x that of self-diffusion, and
    recovery (faster, not modelled) precedes recrystallisation.  Confidence: estimated.
    Reference: Avrami (1939); Johnson & Mehl (1939); Humphreys & Hatherly (2004) ch. 7; Dieter (1986).
    """
    mat = as_material(material)
    t = _solid_temperature(mat, temperature)
    x = _open_fraction("fraction", fraction)
    a = _open_fraction("anchor", anchor)
    t95 = _completion_time(mat, t, a)
    if math.isinf(t95):
        return math.inf
    return t95 * (-math.log1p(-x) / _JMAK_REFERENCE) ** (1.0 / AVRAMI_EXPONENT)


def recrystallised_fraction(material: MaterialLike, temperature: float, time: float, *,
                            anchor: float = RECRYSTALLISATION_HOMOLOGOUS) -> float:
    """Fraction recrystallised after holding a cold-worked metal at a temperature (0-1).

    ``X = 1 - exp(-ln 20 * (t / t_95(T))^n)`` with t_95 and n = 2 as in
    :func:`recrystallisation_time` (so X = 0.95 after one hour at ``anchor`` x T_m).
    ``temperature`` in K (0 < T < T_m), ``time`` in s (>= 0).  Checks: W, 60 s at 482 K ->
    0; W, 5 ms at 1836 K -> 7e-4; Mo, 6.3 ms at 1844 K -> 1.0.

    Validity: isothermal hold of heavily worked metal; see :func:`recrystallisation_time`
    for the (large) uncertainty of the anchor and activation energy.
    Reference: Avrami (1939); Johnson & Mehl (1939); Humphreys & Hatherly (2004) ch. 7.
    """
    mat = as_material(material)
    t = _solid_temperature(mat, temperature)
    hold = _non_negative("time", time)
    a = _open_fraction("anchor", anchor)
    t95 = _completion_time(mat, t, a)
    if hold == 0.0 or math.isinf(t95):
        return 0.0
    if t95 <= 0.0:
        return 1.0
    ratio = hold / t95
    if ratio > 1e6:
        return 1.0
    return -math.expm1(-_JMAK_REFERENCE * ratio ** AVRAMI_EXPONENT)


# ---------------------------------------------------------------------------
# Temperature-compensated strain rate and working regime
# ---------------------------------------------------------------------------
def zener_hollomon(material: MaterialLike, strain_rate: float, temperature: float) -> float:
    """Zener-Hollomon parameter (temperature-compensated strain rate), ``Z = rate * exp(Q/(R T))`` [1/s].

    ``strain_rate`` in 1/s (>= 0), ``temperature`` in K (0 < T < T_m), Q the self-diffusion
    activation energy of the record (J/mol).  Large Z means cold working (no dynamic
    restoration), small Z hot working.  Returns ``math.inf`` when the exponential exceeds
    the floating-point range; use :func:`log10_zener_hollomon` for cold conditions.

    Validity: Q of self-diffusion stands in for the activation energy of hot deformation,
    which is close to it for pure metals above ~0.5 T_m and differs for alloys.
    Reference: Zener & Hollomon (1944).
    """
    mat = as_material(material)
    rate = _non_negative("strain_rate", strain_rate)
    t = _solid_temperature(mat, temperature)
    exponent = _activation_energy(mat) / (R_GAS * t)
    if rate == 0.0:
        return 0.0
    if exponent > _EXP_OVERFLOW:
        return math.inf
    return rate * math.exp(exponent)


def log10_zener_hollomon(material: MaterialLike, strain_rate: float, temperature: float) -> float:
    """Decimal logarithm of the Zener-Hollomon parameter, ``log10 Z = log10(rate) + Q/(R T ln 10)``.

    Overflow-free form of :func:`zener_hollomon` (Z in 1/s); ``strain_rate`` must be > 0.
    Check: W at 1e4 1/s and 482 K -> 67.3; conventional hot forging of W (1 1/s, 1673 K) -> 18.3.

    Validity: as :func:`zener_hollomon`.
    Reference: Zener & Hollomon (1944).
    """
    mat = as_material(material)
    rate = _positive("strain_rate", strain_rate)
    t = _solid_temperature(mat, temperature)
    return math.log10(rate) + _activation_energy(mat) / (R_GAS * t * _LN10)


def equivalent_temperature(material: MaterialLike, strain_rate: float, temperature: float,
                           reference_rate: float = REGIME_REFERENCE_RATE) -> float:
    """Temperature at the reference rate that has the same Zener-Hollomon parameter [K].

    ``1/T_eq = 1/T + (R/Q) ln(rate / rate_ref)``

    A fast deformation acts 'colder' than its temperature: W formed at 1e4 1/s and 482 K
    is equivalent to 454 K at 1 1/s.  ``strain_rate`` in 1/s (floored at 1e-6 1/s, the
    quasi-static limit), ``temperature`` in K (0 < T < T_m), ``reference_rate`` in 1/s (> 0,
    default 1 1/s).  Returns ``math.inf`` when the right-hand side is not positive
    (deformation so slow and hot that it is equivalent to any temperature).

    Validity: one thermally activated mechanism with the self-diffusion activation energy;
    a regime indicator, not a constitutive law.
    Reference: Zener & Hollomon (1944).
    """
    mat = as_material(material)
    rate = max(_non_negative("strain_rate", strain_rate), RATE_FLOOR)
    t = _solid_temperature(mat, temperature)
    ref = _positive("reference_rate", reference_rate)
    inverse = 1.0 / t + (R_GAS / _activation_energy(mat)) * math.log(rate / ref)
    if inverse <= 0.0:
        return math.inf
    return 1.0 / inverse


def deformation_regime(material: MaterialLike, strain_rate: float, temperature: float) -> str:
    """Working regime of a deformation: ``"cold"``, ``"warm"`` or ``"hot"``.

    Based on the rate-compensated homologous temperature ``T_eq / T_m`` with T_eq from
    :func:`equivalent_temperature` (reference 1 1/s): cold below 0.3, hot at or above 0.5,
    warm between (the bounds are :data:`RECRYSTALLISATION_HOMOLOGOUS_RANGE`).  Cold working
    stores dislocations (this module's Taylor estimate applies); hot working restores the
    structure dynamically while it deforms.

    Validity: classification only; the boundaries are conventions.
    Reference: Dieter (1986); Zener & Hollomon (1944).
    """
    mat = as_material(material)
    homologous = equivalent_temperature(mat, strain_rate, temperature) / mat.melting_point
    lo, hi = RECRYSTALLISATION_HOMOLOGOUS_RANGE
    if homologous < lo:
        return "cold"
    if homologous >= hi:
        return "hot"
    return "warm"


def derby_grain_size(material: MaterialLike, stress: float, k: float) -> float:
    """Dynamically recrystallised grain size at a flow stress, ``d = b (K G / sigma)^(3/2)`` [m].

    Derby's piezometer ``(sigma/G) (d/b)^(2/3) = K`` solved for d.  ``stress`` in Pa (> 0);
    G and b from the record; ``k`` is Derby's constant K, which has no default because the
    published range :data:`DERBY_K_RANGE` (1-10) is all that is known - evaluate both ends.
    Check: W at 2.05 GPa -> 0.19 um (K = 1) to 6.0 um (K = 10); the 0.05-0.3 um grains
    observed inside adiabatic shear bands sit at the K = 1 end.

    Validity: steady-state dynamic recrystallisation; applying it inside an adiabatic
    shear band extrapolates far beyond the hot-working data it summarises (estimated).
    Reference: Derby (1991); Meyers (1994).
    """
    mat = as_material(material)
    s = _positive("stress", stress)
    kk = _positive("k", k)
    return _burgers(mat) * (kk * _shear_modulus(mat) / s) ** 1.5


# ---------------------------------------------------------------------------
# Eddy-current skin: thermal exposure
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class SkinExposure(ResultMixin):
    """Thermal exposure of the eddy-current skin of a bare workpiece during a field pulse.

    Geometry and time: ``skin_depth`` delta at the initial temperature; ``heated_depth`` =
    delta/2 (Joule heating ~ J^2 ~ exp(-2z/delta)); ``cooling_time`` = heated_depth^2/alpha_th
    (surface excess down to 43 %).  Lower, layer-average estimate (``kappa`` as given, 1 by
    default): ``temperature_rise`` (uncapped kappa B^2/(2 mu0 rho c_p)), ``peak_temperature``
    (capped at the melting point), ``homologous_temperature``, ``melted``, ``melting_field``
    (melting onset at this kappa), ``recrystallisation_time`` (to 95 % at the peak
    temperature), ``recrystallised_fraction`` after ``cooling_time`` at the peak temperature
    (1.0 when melted, None when the record has no activation energy) and
    ``recrystallised_fraction_band`` = (anchor 0.5 T_m, anchor 0.3 T_m).  Upper estimate for
    the outermost surface (``surface_kappa`` = max(kappa, 2.4)): ``surface_temperature_rise``,
    ``surface_peak_temperature``, ``surface_homologous_temperature``, ``surface_melted`` and
    ``surface_recrystallised_fraction``.  ``heat_affected`` = melted or recrystallised >= 1 %
    on the lower estimate; ``surface_heat_affected`` the same on the upper estimate;
    ``haz_depth`` = heated_depth when either is True (an upper bound when only the surface
    estimate is), else 0.

    Reference: Knoepfel (1970); Carslaw & Jaeger (1959); Humphreys & Hatherly (2004).
    """

    _title: ClassVar[str] = "Skin thermal exposure"

    material: str
    peak_field: float = unit("T")
    frequency: float = unit("Hz")
    kappa: float
    initial_temperature: float = unit("K")
    skin_depth: float = unit("m")
    heated_depth: float = unit("m")
    cooling_time: float = unit("s")
    temperature_rise: float = unit("K")
    peak_temperature: float = unit("K")
    homologous_temperature: float
    melted: bool
    melting_field: float = unit("T")
    recrystallisation_time: float | None = unit("s")
    recrystallised_fraction: float | None
    recrystallised_fraction_band: tuple[float, float] | None
    surface_kappa: float
    surface_temperature_rise: float = unit("K")
    surface_peak_temperature: float = unit("K")
    surface_homologous_temperature: float
    surface_melted: bool
    surface_recrystallised_fraction: float | None
    heat_affected: bool
    surface_heat_affected: bool
    haz_depth: float = unit("m")


def skin_thermal_exposure(material: MaterialLike, peak_field: float,
                          frequency: float = DEFAULT_PULSE_FREQUENCY,
                          initial_temperature: float = T_ROOM, kappa: float = 1.0) -> SkinExposure:
    """How hot the eddy-current skin gets, for how long, and what that does to its grains.

    Chain of estimates for a bare conductor in a pulsed field ``peak_field`` B (T)::

        delta   = sqrt(rho_e / (pi f mu0))              skin depth at the initial temperature
        dT      = kappa B^2 / (2 mu0 rho c_p)           eddy.skin_temperature_rise
        L       = delta / 2                             heated depth (Joule heat ~ exp(-2z/delta))
        t_cool  = L^2 / alpha_th                        conduction time; surface excess -> 43 %
        X       = recrystallised_fraction(T_0 + dT, t_cool)

    evaluated twice: with ``kappa`` (default 1, the lower, layer-average estimate) and with
    max(kappa, 2.4) for the outermost surface (:data:`SURFACE_KAPPA`; the surface of a
    conductor ends a half-sine pulse at 2.2-2.4 x B^2/(2 mu0 rho c_p), see
    :func:`magform.eddy.surface_heating_energy_density`).  ``frequency`` f = 1/(4 t_rise) in
    Hz (default 10 kHz), ``initial_temperature`` T_0 in K.  When an estimate reaches the
    melting point it is reported as melted (fraction 1.0, temperature capped at T_m; latent
    heat ignored).  The band evaluates the lower-estimate X with the recrystallisation anchor
    at 0.5 and 0.3 T_m.  Checks at 100 T, 10 kHz: Cu melts on the lower estimate (onset
    96 T); W layer average 1836 K = 0.50 T_m for 5.0 ms, X = 7e-4 (band 4e-12 to 1.0), but
    its surface estimate melts; Mo 0.64 T_m for 6.3 ms, X = 1.0 (band 2e-5 to 1.0).

    Validity: wall thicker than the skin depth; adiabatic heating then pure conduction into
    cold metal; isothermal hold at the peak temperature for t_cool (an upper estimate of the
    exposure at that temperature, since the layer cools during it, and more so for the
    surface value, whose excess sits in a thinner film); the hot skin's higher resistivity
    deepens the real layer; a sacrificial driver sheet moves all of this out of the
    workpiece.  Order of magnitude: the tungsten layer value at 100 T is within 5 % in
    temperature of the 1 % recrystallisation threshold.
    Reference: Knoepfel (1970); Carslaw & Jaeger (1959); Humphreys & Hatherly (2004) ch. 7.
    """
    mat = as_material(material)
    b = abs(_finite("peak_field", peak_field))
    f = _positive("frequency", frequency)
    t0 = _solid_temperature(mat, initial_temperature, "initial_temperature")
    kap = _positive("kappa", kappa)
    kap_surface = max(kap, SURFACE_KAPPA)
    tm = mat.melting_point
    has_kinetics = mat.self_diffusion_activation is not None

    delta = skin_depth_for_material(mat, f, t0)
    heated = 0.5 * delta
    cooling = heated * heated / mat.thermal_diffusivity()

    def estimate(k: float) -> tuple[float, float, bool, float | None]:
        """(uncapped rise, capped peak temperature, melted, recrystallised fraction) at factor k."""
        rise_k = skin_temperature_rise(mat, b, k)
        melted_k = t0 + rise_k >= tm
        peak_k = tm if melted_k else t0 + rise_k
        if melted_k:
            return rise_k, peak_k, True, 1.0
        return rise_k, peak_k, False, (recrystallised_fraction(mat, peak_k, cooling) if has_kinetics else None)

    rise, peak, melted, fraction = estimate(kap)
    rise_s, peak_s, melted_s, fraction_s = estimate(kap_surface)

    t95: float | None = None
    band: tuple[float, float] | None = None
    if melted:
        band = (1.0, 1.0)
    elif has_kinetics:
        lo_anchor, hi_anchor = RECRYSTALLISATION_HOMOLOGOUS_RANGE
        t95 = recrystallisation_time(mat, peak, RECRYSTALLISATION_REFERENCE_FRACTION)
        band = (recrystallised_fraction(mat, peak, cooling, anchor=hi_anchor),
                recrystallised_fraction(mat, peak, cooling, anchor=lo_anchor))
    affected = melted or (fraction is not None and fraction >= RECRYSTALLISED_THRESHOLD)
    affected_s = melted_s or (fraction_s is not None and fraction_s >= RECRYSTALLISED_THRESHOLD)
    return SkinExposure(
        material=mat.key, peak_field=b, frequency=f, kappa=kap, initial_temperature=t0,
        skin_depth=delta, heated_depth=heated, cooling_time=cooling,
        temperature_rise=rise, peak_temperature=peak, homologous_temperature=peak / tm, melted=melted,
        melting_field=skin_melting_field(mat, t0, kap), recrystallisation_time=t95,
        recrystallised_fraction=fraction, recrystallised_fraction_band=band,
        surface_kappa=kap_surface, surface_temperature_rise=rise_s, surface_peak_temperature=peak_s,
        surface_homologous_temperature=peak_s / tm, surface_melted=melted_s,
        surface_recrystallised_fraction=fraction_s,
        heat_affected=affected, surface_heat_affected=affected_s,
        haz_depth=heated if (affected or affected_s) else 0.0,
    )


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class MicrostructureAssessment(Assessment):
    """Predicted microstructure of a part after one cold forming pulse.

    Dislocations: ``initial_dislocation_density`` -> ``dislocation_density`` (primary fit)
    with ``dislocation_density_band`` over the published fits, ``cell_size`` and
    ``stored_energy``.  Grains: ``grain_size_before`` -> ``grain_size_after`` (thickness of
    the flattened original grains), ``ultrafine`` (< 1 um), ``saturation_grain_size``,
    ``strain_for_ultrafine`` and ``saturation_strain`` (strains needed for 1 um and for
    twice the saturation size).  Strength at room temperature and 1e-3 1/s:
    ``yield_strength_before`` -> ``yield_strength_after``, ``hardening_ratio``,
    ``hardness_before`` -> ``hardness_after`` (Tabor), ``ultrafine_yield_strength``
    (Hall-Petch at the saturation size, for comparison; None without a record).  Heat:
    ``bulk_temperature_rise``, ``bulk_final_temperature``, ``bulk_homologous_temperature``,
    ``deformation_regime`` and ``log10_zener_hollomon`` at the final temperature,
    ``bulk_recrystallised_fraction`` after ``bulk_hold_time`` (these three None without an
    activation energy) and ``grain_structure_preserved``.  Localisation and
    fracture: ``shear_localisation_strain`` (Culver; None if none below 5),
    ``localisation_exceeded``, ``shear_band_grain_size`` (Derby, K = 1 and 10),
    ``dbtt``, ``below_dbtt``, ``failure_strain_limit``, ``ductility_exceeded``.  Skin:
    ``skin`` (None when no field is given) and ``haz_depth``.  ``margin`` = Culver strain /
    target strain (5/strain when no instability is found; None at zero strain).  See
    :func:`assess_microstructure` for the status rules.

    Reference: Taylor (1934); Holt (1970); Humphreys & Hatherly (2004); Culver (1973).
    """

    _title: ClassVar[str] = "Microstructure assessment"

    material: str
    strain: float
    strain_rate: float = unit("1/s")
    temperature: float = unit("K")
    initial_dislocation_density: float = unit("1/m^2")
    dislocation_density: float = unit("1/m^2")
    dislocation_density_band: tuple[float, float] = unit("1/m^2")
    cell_size: float = unit("m")
    stored_energy: float = unit("J/m^3")
    grain_size_before: float = unit("m")
    grain_size_after: float = unit("m")
    ultrafine: bool
    saturation_grain_size: float = unit("m")
    strain_for_ultrafine: float
    saturation_strain: float
    yield_strength_before: float = unit("Pa")
    yield_strength_after: float = unit("Pa")
    hardening_ratio: float
    hardness_before: float = unit("Pa")
    hardness_after: float = unit("Pa")
    ultrafine_yield_strength: float | None = unit("Pa")
    bulk_temperature_rise: float = unit("K")
    bulk_final_temperature: float = unit("K")
    bulk_homologous_temperature: float
    deformation_regime: str | None
    log10_zener_hollomon: float | None
    bulk_hold_time: float = unit("s")
    bulk_recrystallised_fraction: float | None
    grain_structure_preserved: bool
    shear_localisation_strain: float | None
    localisation_exceeded: bool
    shear_band_grain_size: tuple[float, float] | None = unit("m")
    dbtt: float | None = unit("K")
    below_dbtt: bool | None
    failure_strain_limit: float | None
    ductility_exceeded: bool
    skin: SkinExposure | None
    haz_depth: float = unit("m")
    extrapolated_model: bool


def assess_microstructure(material: MaterialLike, strain: float = 0.3, strain_rate: float = 1e4,
                          temperature: float = T_ROOM, peak_field: float | None = None,
                          frequency: float | None = None, initial_grain_size: float | None = None, *,
                          kappa: float = 1.0, bulk_hold_time: float = BULK_HOLD_TIME,
                          initial_density: float = INITIAL_DISLOCATION_DENSITY) -> MicrostructureAssessment:
    """What one cold, fast forming pulse does to the grains and dislocations of ``material``.

    Combines :func:`dislocation_density` (Taylor, ``sqrt(rho) = sqrt(rho0) + d sigma/(M alpha G b)``),
    :func:`dislocation_cell_size` (``K/sqrt(rho)``), :func:`grain_size_after_strain`
    (``d_sat + (d_0 - d_sat) exp(-eps)``), the as-formed room-temperature yield strength
    (flow stress at the final strain) and hardness (Tabor), the bulk adiabatic temperature
    rise (:func:`magform.plasticity.adiabatic_temperature_rise`) with its recrystallised
    fraction after ``bulk_hold_time`` s, the Culver localisation strain, and - when
    ``peak_field`` (T) is given - :func:`skin_thermal_exposure` of the bare workpiece at
    ``frequency`` (Hz, default 10 kHz; ignored without a field) and ``kappa``.  Units:
    strain dimensionless, ``strain_rate`` 1/s, ``temperature`` K, sizes m, ``initial_density``
    1/m^2.

    Status (state of the art unless a rule fires; the rules that fire are worst-combined):

    1. bulk recrystallised fraction >= 1 % -> extrapolation (not cold work any more: the
       Taylor estimate becomes an upper bound and the grain structure is not preserved);
    2. Culver instability strain below the target strain -> extrapolation (deformation
       concentrates in adiabatic shear bands; the uniform prediction is not valid there);
    3. constitutive model outside its calibration -> extrapolation;
    4. below the DBTT with the target strain beyond the unconfined failure strain ->
       extrapolation (the structure exists only if confinement suppresses fracture, which
       :func:`magform.plasticity.assess_forgeability` decides, not this module);
    5. skin heat-affected: recrystallised >= 1 % on the layer estimate (``kappa``), or
       recrystallised >= 1 % or melted on the surface estimate (kappa 2.4) -> extrapolation;
    6. skin melted on the layer estimate -> infeasible *as a solid-state process on the bare
       workpiece*: a layer about half a skin depth deep re-solidifies.  Shield the part
       with a sacrificial driver sheet (then assess with ``peak_field=None``) or machine
       the layer off.

    Otherwise this is ordinary physical metallurgy: cold work stores dislocations in cells
    inside the original grains and nothing recrystallises (state of the art).  Every
    assessment notes that fracture toughness and fatigue resistance are NOT modelled.
    Check: W, 0.3, 1e4 1/s, 293 K -> extrapolation (rules 2 and 4), 3.7e13 1/m^2, 1.7 um
    cells, 20 -> 14.8 um grains, yield 1.23 -> 1.42 GPa, +189 K, nothing recrystallised.

    Validity: uniform bulk deformation; scaling laws with estimated constants (see the
    module docstring); no twinning, texture, recovery or phase transformation.
    Reference: Taylor (1934); Holt (1970); Humphreys & Hatherly (2004); Culver (1973); Tabor (1951).
    """
    mat = as_material(material)
    eps = _non_negative("strain", strain)
    rate = _non_negative("strain_rate", strain_rate)
    t = _solid_temperature(mat, temperature)
    hold = _non_negative("bulk_hold_time", bulk_hold_time)
    rho0 = _non_negative("initial_density", initial_density)
    d0 = _initial_grain_size(mat, initial_grain_size)
    d_sat = _saturation_grain_size(mat, None)
    # the as-formed strength is quoted at room temperature whatever the forming temperature
    t_ref = T_ROOM if T_ROOM < mat.melting_point else t

    # --- dislocations -------------------------------------------------------
    band = flow_stress_band(mat, eps, rate, t)
    rho = dislocation_density(mat, eps, rate, t, rho0)
    rho_lo, rho_hi = dislocation_density_band(mat, eps, rate, t, rho0)
    cell = dislocation_cell_size(rho) if rho > 0.0 else math.inf
    stored = stored_energy_density(mat, rho)

    # --- grains ---------------------------------------------------------------
    d_after = grain_size_after_strain(mat, eps, d0)
    ultrafine = d_after < ULTRAFINE_GRAIN_SIZE
    eps_ultrafine = grain_refinement_strain(mat, ULTRAFINE_GRAIN_SIZE, d0)
    eps_saturation = grain_refinement_strain(mat, None, d0)

    # --- strength of the as-formed state -----------------------------------------
    y_before = flow_stress(mat, 0.0, QUASI_STATIC_RATE, t_ref)
    y_after = flow_stress(mat, eps, QUASI_STATIC_RATE, t_ref)
    h_before = hardness_after_strain(mat, 0.0, t_ref)
    h_after = hardness_after_strain(mat, eps, t_ref)
    y_ultrafine = hall_petch_strength(mat, d_sat) if mat.hall_petch is not None and (
        mat.hall_petch.sigma0 is not None or mat.yield_strength is not None) else None

    # --- bulk heat ---------------------------------------------------------------
    d_t = adiabatic_temperature_rise(mat, eps, rate, t)
    t_final = t + d_t
    homologous = t_final / mat.melting_point
    has_kinetics = mat.self_diffusion_activation is not None
    if has_kinetics:
        x_bulk: float | None = recrystallised_fraction(mat, t_final, hold)
        regime: str | None = deformation_regime(mat, rate, t_final)
        log_z: float | None = log10_zener_hollomon(mat, max(rate, RATE_FLOOR), t_final)
        preserved = x_bulk < RECRYSTALLISED_THRESHOLD  # type: ignore[operator]
    else:
        x_bulk, regime, log_z = None, None, None
        preserved = homologous < RECRYSTALLISATION_HOMOLOGOUS_RANGE[0]

    # --- localisation and fracture -------------------------------------------------
    eps_c = adiabatic_shear_critical_strain(mat, rate, t, max_strain=CULVER_MAX_STRAIN)
    localised = eps_c is not None and eps_c < eps
    band_grains: tuple[float, float] | None = None
    if localised and mat.burgers_vector is not None:
        band_grains = (derby_grain_size(mat, band.primary, DERBY_K_RANGE[0]),
                       derby_grain_size(mat, band.primary, DERBY_K_RANGE[1]))
    dbtt = mat.dbtt
    below = None if dbtt is None else t < dbtt
    fail = mat.dynamic_failure_strain if mat.dynamic_failure_strain is not None else mat.elongation
    brittle = bool(below) and fail is not None and eps > fail

    # --- skin -------------------------------------------------------------------------
    skin: SkinExposure | None = None
    if peak_field is not None:
        skin = skin_thermal_exposure(mat, peak_field,
                                     DEFAULT_PULSE_FREQUENCY if frequency is None else frequency, t, kappa)

    # --- status ---------------------------------------------------------------------------
    fired: list[Status] = []
    warnings: list[str] = []
    notes: list[str] = [
        (f"Cold work stores dislocations: {rho0:.2g} -> {rho:.2g} 1/m^2 (Taylor, alpha = {TAYLOR_ALPHA:g}; "
         f"{rho_lo:.2g}-{rho_hi:.2g} across the published fits), arranged in cells of about {_um(cell)} "
         f"(K = {CELL_SIZE_CONSTANT:g}) bounded by low-angle walls inside the original grains."),
        (f"Grain size: {_um(d0)} -> {_um(d_after)} thick (the original grains flattened by exp(-{eps:g}) = "
         f"{math.exp(-eps):.2f}; grain volume unchanged). Reaching 1 um would take an equivalent strain of "
         f"{eps_ultrafine:.3g} and approaching the {_um(d_sat)} saturation size {eps_saturation:.3g}: the strains "
         "of ECAP or high-pressure torsion (4-8), not of one forming pulse."),
        (f"As-formed room-temperature yield strength {_gpa(y_before)} -> {_gpa(y_after)} "
         f"(x{y_after / y_before:.2f}); hardness {_gpa(h_before)} -> {_gpa(h_after)} (Tabor)."),
        (f"Bulk plastic heating +{d_t:.0f} K to {t_final:.0f} K ({homologous:.2f} T_m)"
         + (f", {regime} working" if regime is not None else "")
         + (f"; recrystallised fraction after {hold:g} s: {x_bulk:.2g}." if x_bulk is not None
            else "; no activation energy recorded, recrystallisation judged from the homologous temperature.")),
        ("Fracture toughness and fatigue resistance are NOT modelled. Cold work raises strength and normally "
         "lowers ductility; the lower ductile-brittle transition temperature of heavily worked tungsten comes "
         "from the rolling strains of severe working (> 3), not from one pulse (Lassner & Schubert 1999; "
         "Wei et al. 2006)."),
    ]
    if ultrafine:
        notes.append(f"The predicted grain thickness is ultrafine (< {_um(ULTRAFINE_GRAIN_SIZE)}): "
                     + ("the feedstock already was." if d0 < ULTRAFINE_GRAIN_SIZE
                        else "this needed the large strain supplied, not the pulse as such."))
    if y_ultrafine is not None:
        notes.append(f"For comparison, Hall-Petch at the {_um(d_sat)} saturation grain size would give "
                     f"{_gpa(y_ultrafine)}: the strength an ultra-fine structure would have if severe plastic "
                     "deformation, not this pulse, had produced it.")
    if mat.spd_saturation_grain_size is None:
        notes.append(f"No SPD saturation grain size is recorded for {mat.key}; {_um(d_sat)} assumed (estimated).")

    if preserved:
        notes.append("The bulk grain structure is preserved: nothing recrystallises at this temperature, so the "
                     "part keeps the grains of its feedstock (this part of the concept note's claim holds).")
    else:
        fired.append(Status.EXTRAPOLATION)
        detail = (f"recrystallised fraction {x_bulk:.2g} after {hold:g} s at {t_final:.0f} K" if x_bulk is not None
                  else f"{homologous:.2f} T_m is above {RECRYSTALLISATION_HOMOLOGOUS_RANGE[0]:g} T_m")
        warnings.append(f"bulk grain structure not preserved ({detail}): this is warm or hot working, the Taylor "
                        "dislocation density is an upper bound")
    if localised:
        fired.append(Status.EXTRAPOLATION)
        text = (f"adiabatic shear localisation: Culver instability at strain {eps_c:.3g} < target {eps:g} at "
                f"{rate:.3g} 1/s; the deformation tends to concentrate in shear bands")
        if band_grains is not None:
            text += (f" whose interior recrystallises dynamically to about {_um(band_grains[0])}-"
                     f"{_um(band_grains[1])} (Derby, K = 1-10; estimated)")
        warnings.append(text)
    if band.extrapolated:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"flow-stress model extrapolated: {band.extrapolation_note}")
    if brittle:
        fired.append(Status.EXTRAPOLATION)
        warnings.append(f"below the DBTT ({dbtt:.0f} K) the target strain {eps:g} exceeds the unconfined failure "
                        f"strain {fail:g}: this microstructure exists only if hydrostatic confinement suppresses "
                        "fracture (see plasticity.assess_forgeability)")
    if skin is not None:
        layer = (f"+{skin.temperature_rise:.0f} K to {skin.peak_temperature:.0f} K "
                 f"({skin.homologous_temperature:.2f} T_m) for about {skin.cooling_time * 1e3:.2g} ms over "
                 f"{_um(skin.heated_depth)}")
        if skin.recrystallised_fraction is not None and not skin.melted:
            lo, hi = skin.recrystallised_fraction_band  # type: ignore[misc]
            layer += (f", recrystallised fraction {skin.recrystallised_fraction:.2g} ({lo:.2g}-{hi:.2g} across "
                      "the 0.3-0.5 T_m anchor band)")
        if skin.surface_melted:
            surface = f"the outermost surface melts on the upper estimate (kappa = {skin.surface_kappa:g})"
        else:
            surface = (f"the outermost surface reaches {skin.surface_peak_temperature:.0f} K "
                       f"({skin.surface_homologous_temperature:.2f} T_m) on the upper estimate "
                       f"(kappa = {skin.surface_kappa:g})")
            if skin.surface_recrystallised_fraction is not None:
                surface += f", recrystallised fraction {skin.surface_recrystallised_fraction:.2g}"
        if skin.melted:
            fired.append(Status.INFEASIBLE)
            warnings.append(f"eddy-current skin melts: +{skin.temperature_rise:.0f} K at {skin.peak_field:.4g} T "
                            f"(melting onset {skin.melting_field:.3g} T at kappa = {skin.kappa:g}); a layer about "
                            f"{_um(skin.heated_depth)} deep re-solidifies, so the bare workpiece is not processed "
                            "in the solid state - use a sacrificial driver sheet or remove the layer")
        elif skin.heat_affected:
            fired.append(Status.EXTRAPOLATION)
            warnings.append(f"eddy-current skin is a heat-affected zone at {skin.peak_field:.4g} T: {layer}; "
                            f"{surface}")
        elif skin.surface_heat_affected:
            fired.append(Status.EXTRAPOLATION)
            warnings.append(f"eddy-current skin is heat-affected at its surface at {skin.peak_field:.4g} T: "
                            f"{surface}, although the layer average (kappa = {skin.kappa:g}, the lower estimate) "
                            f"is {layer}")
        else:
            text = f"Eddy-current skin at {skin.peak_field:.4g} T: {layer}; {surface}"
            if skin.homologous_temperature >= RECRYSTALLISATION_HOMOLOGOUS_RANGE[0]:
                text += "; hot enough for recovery, which is not modelled"
            notes.append(text + ".")
    else:
        notes.append("No field given: eddy-current skin heating not assessed (pass peak_field for a bare "
                     "workpiece; a part behind a sacrificial driver sheet sees none).")

    status = worst(fired) if fired else Status.STATE_OF_THE_ART
    if eps > 0.0:
        margin: float | None = (eps_c if eps_c is not None else CULVER_MAX_STRAIN) / eps
    else:
        margin = None
    return MicrostructureAssessment(
        status=status, margin=margin, warnings=tuple(warnings), notes=tuple(notes),
        material=mat.key, strain=eps, strain_rate=rate, temperature=t,
        initial_dislocation_density=rho0, dislocation_density=rho, dislocation_density_band=(rho_lo, rho_hi),
        cell_size=cell, stored_energy=stored, grain_size_before=d0, grain_size_after=d_after,
        ultrafine=ultrafine, saturation_grain_size=d_sat, strain_for_ultrafine=eps_ultrafine,
        saturation_strain=eps_saturation, yield_strength_before=y_before, yield_strength_after=y_after,
        hardening_ratio=y_after / y_before, hardness_before=h_before, hardness_after=h_after,
        ultrafine_yield_strength=y_ultrafine, bulk_temperature_rise=d_t, bulk_final_temperature=t_final,
        bulk_homologous_temperature=homologous, deformation_regime=regime, log10_zener_hollomon=log_z,
        bulk_hold_time=hold,
        bulk_recrystallised_fraction=x_bulk, grain_structure_preserved=preserved,
        shear_localisation_strain=eps_c, localisation_exceeded=localised, shear_band_grain_size=band_grains,
        dbtt=dbtt, below_dbtt=below, failure_strain_limit=fail, ductility_exceeded=brittle,
        skin=skin, haz_depth=skin.haz_depth if skin is not None else 0.0, extrapolated_model=band.extrapolated,
    )


__all__ = [
    "REFERENCES",
    "TAYLOR_FACTOR", "DEFAULT_TAYLOR_FACTOR", "TAYLOR_ALPHA", "TAYLOR_ALPHA_RANGE",
    "INITIAL_DISLOCATION_DENSITY", "CELL_SIZE_CONSTANT", "CELL_SIZE_CONSTANT_RANGE", "STORED_ENERGY_FACTOR",
    "ULTRAFINE_GRAIN_SIZE", "HALL_PETCH_FLOOR", "DEFAULT_SATURATION_GRAIN_SIZE", "SATURATION_APPROACH_FACTOR",
    "RECRYSTALLISATION_HOMOLOGOUS", "RECRYSTALLISATION_HOMOLOGOUS_RANGE", "RECRYSTALLISATION_REFERENCE_TIME",
    "RECRYSTALLISATION_REFERENCE_FRACTION", "AVRAMI_EXPONENT", "RECRYSTALLISED_THRESHOLD", "BULK_HOLD_TIME",
    "DEFAULT_PULSE_FREQUENCY", "SURFACE_KAPPA", "REGIME_REFERENCE_RATE", "QUASI_STATIC_RATE", "TABOR_FACTOR",
    "TABOR_REPRESENTATIVE_STRAIN", "DERBY_K_RANGE", "CULVER_MAX_STRAIN", "SURFACE_EXCESS_AT_COOLING_TIME",
    "taylor_factor", "taylor_hardening", "dislocation_density", "dislocation_density_band",
    "dislocation_cell_size", "stored_energy_density",
    "hall_petch_friction_stress", "hall_petch_strength", "grain_size_for_strength",
    "grain_size_after_strain", "grain_refinement_strain", "hardness_after_strain",
    "recrystallisation_time", "recrystallised_fraction",
    "zener_hollomon", "log10_zener_hollomon", "equivalent_temperature", "deformation_regime",
    "derby_grain_size",
    "SkinExposure", "skin_thermal_exposure",
    "MicrostructureAssessment", "assess_microstructure",
]
