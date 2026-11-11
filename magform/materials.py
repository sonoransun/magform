"""Materials database: properties, constitutive models and Hugoniots.

Values are *representative* engineering data (SI units) with a provenance tag
per entry; they are not datasheet guarantees.  Strength fields are ``None``
for materials that do not yield (ceramics, fibres store their tensile
strength in ``ultimate_strength``).

Constitutive models are plain data classes with a ``stress()`` method so this
module has no dependency on :mod:`magform.plasticity`:

* :class:`JohnsonCook` (Johnson & Cook 1983 form)
* :class:`ZerilliArmstrongBCC` (Zerilli & Armstrong 1987, BCC form)

The first entry of ``Material.constitutive`` is the primary model; any further
entries are alternative published fits used for uncertainty bands.

Public API
----------
``MATERIALS``  dict key -> Material
``ALIASES``    dict alias -> key (case-insensitive lookup)
``get(key_or_alias)``, ``as_material(material_like)``, ``list_materials(category=None)``,
``search(text)``, ``keys()``
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field, replace
from typing import Union

from .constants import T_ROOM, gpa, mpa
from .errors import InputError, UnknownMaterialError
from .status import Confidence

_RATE_FLOOR = 1e-6  # 1/s; constitutive logs are evaluated at max(rate, this)


# ---------------------------------------------------------------------------
# Shock Hugoniot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hugoniot:
    """Linear shock-velocity/particle-velocity Hugoniot ``Us = c0 + s * up``.

    ``p_max`` is the upper pressure of validity (e.g. the 13 GPa alpha->epsilon
    transition of iron); ``None`` means no limit is recorded.
    """

    c0: float  # m/s
    s: float   # dimensionless
    p_max: float | None = None
    source: str = ""
    confidence: Confidence = Confidence.HANDBOOK

    def shock_speed(self, particle_velocity: float) -> float:
        """Shock speed Us (m/s) for particle velocity ``up`` (m/s)."""
        return self.c0 + self.s * particle_velocity

    def pressure(self, density: float, particle_velocity: float) -> float:
        """Rankine-Hugoniot pressure ``rho0 * Us * up`` (Pa)."""
        return density * self.shock_speed(particle_velocity) * particle_velocity


# ---------------------------------------------------------------------------
# Constitutive models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JohnsonCook:
    """Johnson-Cook flow stress.

    sigma = (A + B eps^n) (1 + C ln(rate/eps0)) (1 - T*^m),
    T* = (T - t_ref)/(t_melt - t_ref) clipped to [0, 0.999].

    ``t_ref``/``t_melt`` default to room temperature and the material's
    melting point when ``None``.  The rate factor is floored at 0.2 (the fit
    is not meaningful far below its calibration range).
    """

    A: float
    B: float
    n: float
    C: float
    m: float
    eps0: float = 1.0
    t_ref: float | None = None
    t_melt: float | None = None
    source: str = ""
    condition: str = ""
    rate_range: tuple[float, float] | None = None
    temperature_range: tuple[float, float] | None = None
    confidence: Confidence = Confidence.REPRESENTATIVE
    kind: str = "johnson_cook"

    def stress(self, strain: float, strain_rate: float, temperature: float,
               t_ref_default: float = T_ROOM, t_melt_default: float | None = None) -> float:
        tr = self.t_ref if self.t_ref is not None else t_ref_default
        tm = self.t_melt if self.t_melt is not None else t_melt_default
        if tm is None:
            raise InputError("JohnsonCook.stress needs a melting temperature")
        eps = max(strain, 0.0)
        rate = max(strain_rate, _RATE_FLOOR)
        hard = self.A + self.B * (eps ** self.n if eps > 0.0 else 0.0)
        rate_factor = max(1.0 + self.C * math.log(rate / self.eps0), 0.2)
        t_star = min(max((temperature - tr) / (tm - tr), 0.0), 0.999)
        thermal = 1.0 - t_star ** self.m if t_star > 0.0 else 1.0
        return hard * rate_factor * thermal

    def in_range(self, strain_rate: float, temperature: float) -> bool:
        """True if (rate, T) lies inside the recorded calibration window."""
        ok = True
        if self.rate_range is not None:
            ok &= self.rate_range[0] <= strain_rate <= self.rate_range[1]
        if self.temperature_range is not None:
            ok &= self.temperature_range[0] <= temperature <= self.temperature_range[1]
        return ok


@dataclass(frozen=True)
class ZerilliArmstrongBCC:
    """Zerilli-Armstrong BCC flow stress (rate in 1/s, T in K).

    sigma = c0 + c1 exp(-c3 T + c4 T ln(rate)) + c5 eps^n
    """

    c0: float
    c1: float
    c3: float
    c4: float
    c5: float
    n: float
    source: str = ""
    condition: str = ""
    rate_range: tuple[float, float] | None = None
    temperature_range: tuple[float, float] | None = None
    confidence: Confidence = Confidence.REPRESENTATIVE
    kind: str = "zerilli_armstrong_bcc"

    def stress(self, strain: float, strain_rate: float, temperature: float,
               t_ref_default: float = T_ROOM, t_melt_default: float | None = None) -> float:
        eps = max(strain, 0.0)
        rate = max(strain_rate, _RATE_FLOOR)
        thermal = self.c1 * math.exp(-self.c3 * temperature + self.c4 * temperature * math.log(rate))
        return self.c0 + thermal + self.c5 * (eps ** self.n if eps > 0.0 else 0.0)

    def in_range(self, strain_rate: float, temperature: float) -> bool:
        ok = True
        if self.rate_range is not None:
            ok &= self.rate_range[0] <= strain_rate <= self.rate_range[1]
        if self.temperature_range is not None:
            ok &= self.temperature_range[0] <= temperature <= self.temperature_range[1]
        return ok


ConstitutiveModel = Union[JohnsonCook, ZerilliArmstrongBCC]


@dataclass(frozen=True)
class HallPetch:
    """Hall-Petch parameters.  ``sigma0`` None -> calibrated so that the
    yield strength is reproduced at ``reference_grain_size``."""

    k: float  # Pa m^0.5
    sigma0: float | None = None
    reference_grain_size: float = 50e-6
    confidence: Confidence = Confidence.REPRESENTATIVE


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Material:
    """A material record.  Derived moduli are filled in ``__post_init__``."""

    key: str
    name: str
    category: str  # metal|refractory|intermetallic|ods|superalloy|conductor|ceramic|carbon|fibre
    base_element: str
    crystal_structure: str  # fcc|bcc|hcp|L1_0|graphite|diamond|hexagonal|cubic|fibre|composite
    condition: str = ""

    density: float
    youngs_modulus: float
    poisson: float
    shear_modulus: float | None = None
    bulk_modulus: float | None = None

    yield_strength: float | None = None
    ultimate_strength: float | None = None
    elongation: float | None = None
    uniform_elongation: float | None = None
    compressive_yield: float | None = None
    compressive_strength: float | None = None
    hollomon_override: tuple[float, float] | None = None  # (K [Pa], n)

    melting_point: float
    specific_heat: float
    thermal_conductivity: float
    latent_heat_fusion: float | None = None
    thermal_expansion: float | None = None

    resistivity: float
    resistivity_temp_coeff: float = 0.0

    vickers_hardness: float | None = None
    hugoniot: Hugoniot | None = None
    hugoniot_high: Hugoniot | None = None
    phase_transition_pressure: float | None = None
    constitutive: tuple[ConstitutiveModel, ...] = ()
    hall_petch: HallPetch | None = None
    burgers_vector: float | None = None
    dbtt: float | None = None
    dynamic_failure_strain: float | None = None
    spd_saturation_grain_size: float | None = None
    typical_grain_size: float | None = None
    self_diffusion_activation: float | None = None  # J/mol
    conventional_working: tuple[float, float] | None = None  # K
    conventional_working_note: str = ""

    notes: str = ""
    sources: tuple[str, ...] = ()
    confidence: Confidence = Confidence.REPRESENTATIVE

    def __post_init__(self) -> None:
        if self.density <= 0 or self.youngs_modulus <= 0:
            raise InputError(f"{self.key}: density and Young's modulus must be positive")
        if not -1.0 < self.poisson < 0.5:
            raise InputError(f"{self.key}: Poisson ratio must be in (-1, 0.5)")
        if self.shear_modulus is None:
            object.__setattr__(self, "shear_modulus", self.youngs_modulus / (2.0 * (1.0 + self.poisson)))
        if self.bulk_modulus is None:
            object.__setattr__(self, "bulk_modulus", self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson)))

    # -- convenience -------------------------------------------------------
    @property
    def is_metal(self) -> bool:
        return self.category in ("metal", "refractory", "intermetallic", "ods", "superalloy", "conductor")

    @property
    def volumetric_heat_capacity(self) -> float:
        """rho * cp, J/(m^3 K)."""
        return self.density * self.specific_heat

    def require(self, *fields: str) -> None:
        """Raise :class:`InputError` if any named field is ``None``."""
        missing = [f for f in fields if getattr(self, f) is None]
        if missing:
            raise InputError(f"material '{self.key}' has no data for: {', '.join(missing)}")

    def resistivity_at(self, temperature: float) -> float:
        """Linear resistivity model rho(T) = rho293 [1 + alpha (T - 293.15)], floored at 2 % of rho293."""
        if temperature <= 0:
            raise InputError("temperature must be positive (K)")
        r = self.resistivity * (1.0 + self.resistivity_temp_coeff * (temperature - T_ROOM))
        return max(r, 0.02 * self.resistivity)

    def homologous_temperature(self, temperature: float) -> float:
        return temperature / self.melting_point

    def thermal_diffusivity(self) -> float:
        """k/(rho cp), m^2/s."""
        return self.thermal_conductivity / (self.density * self.specific_heat)

    def bulk_sound_speed(self) -> float:
        """sqrt(K/rho), m/s (K from the record, or E/(3(1-2 nu)))."""
        return math.sqrt(self.bulk_modulus / self.density)  # type: ignore[operator]

    def longitudinal_sound_speed(self) -> float:
        """Dilatational wave speed sqrt(E(1-nu)/(rho(1+nu)(1-2nu))), m/s."""
        nu = self.poisson
        return math.sqrt(self.youngs_modulus * (1 - nu) / (self.density * (1 + nu) * (1 - 2 * nu)))

    def volumetric_strain(self, pressure: float) -> float:
        """Linear-elastic volume change P/K (fraction) under hydrostatic pressure."""
        return pressure / self.bulk_modulus  # type: ignore[operator]

    def hugoniot_for(self, pressure: float | None = None) -> Hugoniot:
        """Hugoniot branch valid at ``pressure`` (Pa); estimated from the bulk
        sound speed (s = 1.4) when no measured Hugoniot is recorded."""
        if self.hugoniot is None:
            return Hugoniot(self.bulk_sound_speed(), 1.4, None,
                            "estimated: c0 = sqrt(K/rho), s = 1.4", Confidence.ESTIMATED)
        if (pressure is not None and self.hugoniot_high is not None
                and self.phase_transition_pressure is not None
                and pressure > self.phase_transition_pressure):
            return self.hugoniot_high
        return self.hugoniot

    def hollomon(self) -> tuple[float, float]:
        """Power-law fit sigma = K eps^n (K in Pa).

        Uses ``hollomon_override`` when given.  Otherwise a two-point fit
        through (0.002, yield) and the true necking point
        (eps_u = ln(1+e_u), sigma = UTS (1+e_u)), where e_u is
        ``uniform_elongation`` or 0.6 x total elongation (capped at 0.5).
        n is clipped to [0.01, 0.6].
        """
        if self.hollomon_override is not None:
            return self.hollomon_override
        self.require("yield_strength", "ultimate_strength", "elongation")
        sy = float(self.yield_strength)  # type: ignore[arg-type]
        uts = float(self.ultimate_strength)  # type: ignore[arg-type]
        e_u = self.uniform_elongation if self.uniform_elongation is not None else min(0.6 * float(self.elongation), 0.5)  # type: ignore[arg-type]
        eps_u = math.log1p(e_u)
        s_u = uts * (1.0 + e_u)
        if eps_u <= 0.002 or s_u <= sy:
            n = 0.01
        else:
            n = math.log(s_u / sy) / math.log(eps_u / 0.002)
        n = min(max(n, 0.01), 0.6)
        K = sy / 0.002 ** n
        return K, n

    def hardness_estimate(self) -> float:
        """Vickers hardness (Pa): recorded value, else Tabor HV ~ 3 sigma(eps = 0.08)."""
        if self.vickers_hardness is not None:
            return self.vickers_hardness
        self.require("yield_strength")
        try:
            K, n = self.hollomon()
            flow = max(float(self.yield_strength), K * 0.08 ** n)  # type: ignore[arg-type]
        except InputError:
            flow = float(self.yield_strength)  # type: ignore[arg-type]
        return 3.0 * flow

    def with_changes(self, **changes) -> "Material":
        """Copy with fields replaced (derived moduli recomputed unless given)."""
        if "youngs_modulus" in changes or "poisson" in changes:
            changes.setdefault("shear_modulus", None)
            changes.setdefault("bulk_modulus", None)
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Shared constitutive fits and Hugoniots
# ---------------------------------------------------------------------------
_JC1983 = "Johnson & Cook, Proc. 7th Int. Symp. Ballistics, The Hague (1983) 541-547"
_MARSH = "Marsh (ed.), LASL Shock Hugoniot Data, Univ. California Press (1980)"
_STEINBERG = "Steinberg, Equation of State and Strength Properties of Selected Materials, LLNL UCRL-MA-106439 (1996)"

_ALPHA_FE_HUGONIOT = Hugoniot(4630.0, 1.33, gpa(13.0), "alpha-Fe branch estimated from bulk sound speed; below the 13 GPa alpha-epsilon transition", Confidence.ESTIMATED)
_EPS_FE_HUGONIOT = Hugoniot(3574.0, 1.92, None, f"epsilon-Fe branch, {_STEINBERG}", Confidence.HANDBOOK)

W_ZA_LENNON_RAMESH = ZerilliArmstrongBCC(
    c0=mpa(49.91), c1=mpa(2749.0), c3=2.25e-3, c4=9.05e-5, c5=mpa(194.5), n=0.0505,
    source="Lennon & Ramesh, Acta Mater. 48 (2000) / tabulated in Skoro et al., arXiv:1105.5514 (2011)",
    condition="HIPed commercially pure W", rate_range=(1e-3, 4e3), temperature_range=(293.0, 1273.0),
)
W_ZA_CHEN_GRAY = ZerilliArmstrongBCC(
    c0=0.0, c1=mpa(3000.0), c3=2.0e-3, c4=1.0e-4, c5=mpa(800.0), n=0.6,
    source="Chen & Gray, rolled CP W plate / tabulated in Skoro et al., arXiv:1105.5514 (2011)",
    condition="rolled commercially pure W plate", rate_range=(1e-3, 4e3), temperature_range=(293.0, 1273.0),
)
TA_ZA_CHEN_GRAY = ZerilliArmstrongBCC(
    c0=mpa(140.0), c1=mpa(1750.0), c3=9.75e-3, c4=6.75e-4, c5=mpa(650.0), n=0.65,
    source="Chen & Gray, Metall. Mater. Trans. A 27 (1996) 2994, Table II",
    condition="unalloyed Ta", rate_range=(1e-3, 4e3), temperature_range=(77.0, 1273.0),
)


def _m(**kw) -> Material:
    return Material(**kw)


_ENTRIES: list[Material] = [
    # ---------------------------------------------------------------- metals
    _m(key="copper_ofhc", name="Copper, OFHC (annealed)", category="metal", base_element="Cu",
       crystal_structure="fcc", condition="annealed",
       density=8960, youngs_modulus=gpa(117), poisson=0.34,
       yield_strength=mpa(70), ultimate_strength=mpa(220), elongation=0.45, uniform_elongation=0.35,
       melting_point=1356.0, specific_heat=385.0, thermal_conductivity=391.0,
       latent_heat_fusion=2.09e5, thermal_expansion=16.5e-6,
       resistivity=1.71e-8, resistivity_temp_coeff=3.9e-3, vickers_hardness=mpa(440),
       hugoniot=Hugoniot(3940.0, 1.489, None, _MARSH),
       constitutive=(JohnsonCook(mpa(90), mpa(292), 0.31, 0.025, 1.09, 1.0, t_melt=1356.0,
                                 source=_JC1983, condition="annealed OFHC", rate_range=(1e-3, 1e3)),),
       hall_petch=HallPetch(k=0.12e6), burgers_vector=2.56e-10,
       spd_saturation_grain_size=200e-9, typical_grain_size=30e-6, self_diffusion_activation=197e3,
       conventional_working=(293.0, 1223.0), conventional_working_note="cold or hot worked",
       sources=(_JC1983, _MARSH, "ASM Handbook Vol. 2"), confidence=Confidence.HANDBOOK),
    _m(key="aluminum_1100", name="Aluminium 1100-O", category="metal", base_element="Al",
       crystal_structure="fcc", condition="O (annealed)",
       density=2710, youngs_modulus=gpa(69), poisson=0.33,
       yield_strength=mpa(34), ultimate_strength=mpa(90), elongation=0.35,
       hollomon_override=(mpa(180), 0.20),
       melting_point=916.0, specific_heat=904.0, thermal_conductivity=222.0,
       latent_heat_fusion=3.97e5, thermal_expansion=23.6e-6,
       resistivity=2.99e-8, resistivity_temp_coeff=4.2e-3, vickers_hardness=mpa(225),
       hugoniot=Hugoniot(5386.0, 1.339, None, f"{_MARSH}; {_STEINBERG}"),
       hall_petch=HallPetch(k=0.04e6), burgers_vector=2.86e-10,
       spd_saturation_grain_size=600e-9, typical_grain_size=50e-6, self_diffusion_activation=142e3,
       conventional_working=(293.0, 773.0),
       sources=("Kalpakjian & Schmid, Manufacturing Engineering and Technology, Table 2.3", _MARSH),
       confidence=Confidence.HANDBOOK),
    _m(key="aluminum_6061", name="Aluminium 6061-T6", category="metal", base_element="Al",
       crystal_structure="fcc", condition="T6",
       density=2700, youngs_modulus=gpa(68.9), poisson=0.33,
       yield_strength=mpa(276), ultimate_strength=mpa(310), elongation=0.12, uniform_elongation=0.08,
       hollomon_override=(mpa(410), 0.05),
       melting_point=855.0, specific_heat=896.0, thermal_conductivity=167.0,
       latent_heat_fusion=3.9e5, thermal_expansion=23.6e-6,
       resistivity=3.99e-8, resistivity_temp_coeff=2.9e-3, vickers_hardness=mpa(1050),
       hugoniot=Hugoniot(5240.0, 1.40, None, f"{_STEINBERG} (Marsh linear fit 5350/1.34)"),
       constitutive=(JohnsonCook(mpa(324), mpa(114), 0.42, 0.002, 1.34, 1.0, t_melt=855.0,
                                 source="Lesuer, Kay & LeBlanc, LLNL UCRL-JC-134118 (2001), via Al-Badour et al. (2014)",
                                 condition="T6"),),
       hall_petch=HallPetch(k=0.07e6, confidence=Confidence.ESTIMATED), burgers_vector=2.86e-10,
       spd_saturation_grain_size=200e-9, typical_grain_size=50e-6, self_diffusion_activation=142e3,
       conventional_working=(293.0, 773.0), notes="solidus 855 K used as melting point",
       sources=("ASM Handbook Vol. 2",), confidence=Confidence.HANDBOOK),
    _m(key="magnesium_az31", name="Magnesium AZ31B", category="metal", base_element="Mg",
       crystal_structure="hcp", condition="rolled sheet (H24)",
       density=1770, youngs_modulus=gpa(45), poisson=0.35,
       yield_strength=mpa(200), ultimate_strength=mpa(260), elongation=0.15, compressive_yield=mpa(140),
       melting_point=878.0, specific_heat=1000.0, thermal_conductivity=96.0,
       latent_heat_fusion=3.4e5, thermal_expansion=26e-6,
       resistivity=9.2e-8, resistivity_temp_coeff=2.0e-3, vickers_hardness=mpa(640),
       hugoniot=Hugoniot(4520.0, 1.242, None, _STEINBERG),
       constitutive=(JohnsonCook(mpa(224), mpa(380), 0.761, 0.012, 1.55, 1.0, t_melt=905.0,
                                 source="Hasenpouth, MASc thesis, Univ. Waterloo (2010); Ulacia et al., JMPT 211 (2011) 830",
                                 condition="AZ31B sheet"),),
       hall_petch=HallPetch(k=0.28e6), burgers_vector=3.21e-10,
       spd_saturation_grain_size=500e-9, typical_grain_size=10e-6, self_diffusion_activation=135e3,
       conventional_working=(473.0, 723.0), conventional_working_note="warm forming; few slip systems at RT",
       notes="tension/compression asymmetry from twinning", confidence=Confidence.REPRESENTATIVE),
    _m(key="steel_1018", name="Steel AISI 1018 (hot rolled)", category="metal", base_element="Fe",
       crystal_structure="bcc", condition="hot rolled",
       density=7870, youngs_modulus=gpa(205), poisson=0.29,
       yield_strength=mpa(310), ultimate_strength=mpa(440), elongation=0.25,
       hollomon_override=(mpa(530), 0.26),
       melting_point=1723.0, specific_heat=486.0, thermal_conductivity=51.9,
       latent_heat_fusion=2.47e5, thermal_expansion=12e-6,
       resistivity=1.59e-7, resistivity_temp_coeff=4.0e-3, vickers_hardness=mpa(1250),
       hugoniot=_ALPHA_FE_HUGONIOT, hugoniot_high=_EPS_FE_HUGONIOT, phase_transition_pressure=gpa(13.0),
       constitutive=(JohnsonCook(mpa(350), mpa(275), 0.36, 0.022, 1.00, 1.0, t_melt=1811.0,
                                 source=_JC1983, condition="1006 steel (mild-steel proxy)",
                                 confidence=Confidence.ESTIMATED),),
       hall_petch=HallPetch(k=0.6e6), burgers_vector=2.48e-10, dbtt=250.0,
       spd_saturation_grain_size=200e-9, typical_grain_size=20e-6, self_diffusion_activation=251e3,
       conventional_working=(293.0, 1523.0), conventional_working_note="cold heading or hot forging",
       confidence=Confidence.REPRESENTATIVE),
    _m(key="steel_4340", name="Steel AISI 4340 (Q&T, HRC 30)", category="metal", base_element="Fe",
       crystal_structure="bcc", condition="quenched and tempered, HRC 30",
       density=7850, youngs_modulus=gpa(205), poisson=0.29,
       yield_strength=mpa(792), ultimate_strength=mpa(1000), elongation=0.16,
       melting_point=1793.0, specific_heat=477.0, thermal_conductivity=44.5,
       latent_heat_fusion=2.47e5, thermal_expansion=12.3e-6,
       resistivity=2.48e-7, resistivity_temp_coeff=2.5e-3, vickers_hardness=mpa(2950),
       hugoniot=Hugoniot(4578.0, 1.33, gpa(13.0), _STEINBERG), hugoniot_high=_EPS_FE_HUGONIOT,
       phase_transition_pressure=gpa(13.0),
       constitutive=(JohnsonCook(mpa(792), mpa(510), 0.26, 0.014, 1.03, 1.0, t_melt=1793.0,
                                 source=_JC1983, condition="HRC 30"),),
       hall_petch=HallPetch(k=0.6e6, confidence=Confidence.ESTIMATED), burgers_vector=2.48e-10, dbtt=230.0,
       spd_saturation_grain_size=150e-9, typical_grain_size=15e-6, self_diffusion_activation=251e3,
       conventional_working=(1123.0, 1473.0), confidence=Confidence.HANDBOOK),
    _m(key="steel_304", name="Stainless steel 304 (annealed)", category="metal", base_element="Fe",
       crystal_structure="fcc", condition="annealed",
       density=8000, youngs_modulus=gpa(193), poisson=0.29,
       yield_strength=mpa(215), ultimate_strength=mpa(505), elongation=0.70, uniform_elongation=0.55,
       hollomon_override=(mpa(1275), 0.45),
       melting_point=1673.0, specific_heat=500.0, thermal_conductivity=16.2,
       latent_heat_fusion=2.6e5, thermal_expansion=17.3e-6,
       resistivity=7.2e-7, resistivity_temp_coeff=1.0e-3, vickers_hardness=mpa(1500),
       hugoniot=Hugoniot(4570.0, 1.49, None, _STEINBERG),
       hall_petch=HallPetch(k=0.4e6), burgers_vector=2.54e-10,
       spd_saturation_grain_size=80e-9, typical_grain_size=40e-6, self_diffusion_activation=284e3,
       conventional_working=(293.0, 1423.0),
       notes="no published Johnson-Cook set with traceable provenance is used; Hollomon fallback",
       confidence=Confidence.HANDBOOK),
    _m(key="iron_armco", name="Iron, Armco (annealed)", category="metal", base_element="Fe",
       crystal_structure="bcc", condition="annealed",
       density=7870, youngs_modulus=gpa(211), poisson=0.29,
       yield_strength=mpa(175), ultimate_strength=mpa(300), elongation=0.40,
       melting_point=1811.0, specific_heat=450.0, thermal_conductivity=80.0,
       latent_heat_fusion=2.47e5, thermal_expansion=11.8e-6,
       resistivity=9.7e-8, resistivity_temp_coeff=6.5e-3, vickers_hardness=mpa(785),
       hugoniot=_ALPHA_FE_HUGONIOT, hugoniot_high=_EPS_FE_HUGONIOT, phase_transition_pressure=gpa(13.0),
       constitutive=(JohnsonCook(mpa(175), mpa(380), 0.32, 0.060, 0.55, 1.0, t_melt=1811.0,
                                 source=_JC1983, condition="Armco iron"),),
       hall_petch=HallPetch(k=0.6e6), burgers_vector=2.48e-10, dbtt=250.0,
       spd_saturation_grain_size=200e-9, typical_grain_size=80e-6, self_diffusion_activation=251e3,
       conventional_working=(293.0, 1523.0), confidence=Confidence.HANDBOOK),
    _m(key="titanium_cp2", name="Titanium CP grade 2", category="metal", base_element="Ti",
       crystal_structure="hcp", condition="annealed",
       density=4510, youngs_modulus=gpa(105), poisson=0.34,
       yield_strength=mpa(345), ultimate_strength=mpa(450), elongation=0.20, uniform_elongation=0.132,
       melting_point=1941.0, specific_heat=523.0, thermal_conductivity=16.4,
       latent_heat_fusion=2.96e5, thermal_expansion=8.6e-6,
       resistivity=5.2e-7, resistivity_temp_coeff=3.5e-3, vickers_hardness=mpa(1500),
       hugoniot=Hugoniot(5220.0, 0.767, gpa(15.0), f"{_MARSH}; anomalously low s from alpha-omega transition"),
       constitutive=(JohnsonCook(mpa(330), mpa(753), 0.42, 0.03, 1.0, 8e-3, t_melt=1941.0,
                                 source="Siegel, Laporte & Sauter-Starace, Materials 14 (2021) 3887 (A, B, n); C, m estimated",
                                 condition="grade 2, tension", rate_range=(8e-3, 8e-2),
                                 confidence=Confidence.ESTIMATED),),
       hall_petch=HallPetch(k=0.4e6, confidence=Confidence.ESTIMATED), burgers_vector=2.95e-10,
       spd_saturation_grain_size=150e-9, typical_grain_size=30e-6, self_diffusion_activation=242e3,
       conventional_working=(293.0, 1173.0), confidence=Confidence.REPRESENTATIVE),
    _m(key="ti6al4v", name="Titanium Ti-6Al-4V (annealed)", category="metal", base_element="Ti",
       crystal_structure="hcp", condition="mill annealed",
       density=4430, youngs_modulus=gpa(113.8), poisson=0.342,
       yield_strength=mpa(880), ultimate_strength=mpa(950), elongation=0.14, uniform_elongation=0.08,
       melting_point=1878.0, specific_heat=526.0, thermal_conductivity=6.7,
       latent_heat_fusion=2.9e5, thermal_expansion=8.6e-6,
       resistivity=1.78e-6, resistivity_temp_coeff=3e-4, vickers_hardness=mpa(3350),
       hugoniot=Hugoniot(5130.0, 1.028, None, _MARSH),
       constitutive=(JohnsonCook(mpa(1098), mpa(1092), 0.93, 0.014, 1.1, 1.0, t_melt=1878.0,
                                 source="Lesuer, DOT/FAA/AR-00/25 (2000)", condition="annealed"),),
       hall_petch=HallPetch(k=0.4e6, confidence=Confidence.ESTIMATED), burgers_vector=2.95e-10,
       spd_saturation_grain_size=100e-9, typical_grain_size=10e-6, self_diffusion_activation=330e3,
       conventional_working=(1173.0, 1253.0), conventional_working_note="alpha+beta forging",
       confidence=Confidence.HANDBOOK),
    # ------------------------------------------------------------ refractory
    _m(key="tungsten", name="Tungsten, commercially pure (wrought)", category="refractory", base_element="W",
       crystal_structure="bcc", condition="wrought/stress relieved",
       density=19250, youngs_modulus=gpa(411), poisson=0.28,
       yield_strength=mpa(1300), ultimate_strength=mpa(1400), elongation=0.02,
       melting_point=3695.0, specific_heat=134.0, thermal_conductivity=173.0,
       latent_heat_fusion=2.85e5, thermal_expansion=4.5e-6,
       resistivity=5.28e-8, resistivity_temp_coeff=4.5e-3, vickers_hardness=mpa(4300),
       hugoniot=Hugoniot(4029.0, 1.237, None, _MARSH),
       constitutive=(W_ZA_LENNON_RAMESH, W_ZA_CHEN_GRAY),
       hall_petch=HallPetch(k=0.8e6, confidence=Confidence.ESTIMATED), burgers_vector=2.74e-10, dbtt=573.0,
       dynamic_failure_strain=0.07,
       spd_saturation_grain_size=100e-9, typical_grain_size=20e-6, self_diffusion_activation=585e3,
       conventional_working=(1473.0, 1873.0), conventional_working_note="forged/swaged/rolled at 1200-1600 C",
       notes="yield is compressive, wrought; annealed tensile yield ~0.75 GPa. Unconfined dynamic compression "
             "fails at 5-10 % strain (Duemmer et al., Acta Mater. 46 (1998) 6267).",
       sources=("Lassner & Schubert, Tungsten (1999)", "Duemmer, LaSalvia, Ravichandran & Meyers, Acta Mater. 46 (1998) 6267"),
       confidence=Confidence.REPRESENTATIVE),
    _m(key="tungsten_heavy_alloy_90w", name="Tungsten heavy alloy 90W-7Ni-3Fe (swaged)", category="refractory",
       base_element="W", crystal_structure="bcc", condition="swaged",
       density=17100, youngs_modulus=gpa(350), poisson=0.29,
       yield_strength=mpa(1500), ultimate_strength=mpa(1600), elongation=0.08,
       melting_point=1723.0, specific_heat=140.0, thermal_conductivity=75.0,
       latent_heat_fusion=2.8e5, thermal_expansion=5.0e-6,
       resistivity=1.4e-7, resistivity_temp_coeff=3e-3, vickers_hardness=mpa(3800),
       hugoniot=Hugoniot(4010.0, 1.27, None, "estimated from pure W", Confidence.ESTIMATED),
       constitutive=(JohnsonCook(mpa(1506), mpa(177), 0.12, 0.016, 1.0, 1.0, t_melt=1723.0,
                                 source=_JC1983, condition="W-0.07Ni-0.03Fe"),),
       burgers_vector=2.74e-10, typical_grain_size=30e-6, self_diffusion_activation=585e3,
       conventional_working=(293.0, 873.0), conventional_working_note="swaged at room to moderate temperature",
       notes="melting point is the Ni-Fe matrix solidus used by Johnson & Cook (1983)",
       confidence=Confidence.REPRESENTATIVE),
    _m(key="molybdenum", name="Molybdenum, commercially pure", category="refractory", base_element="Mo",
       crystal_structure="bcc", condition="stress relieved",
       density=10220, youngs_modulus=gpa(329), poisson=0.31,
       yield_strength=mpa(600), ultimate_strength=mpa(750), elongation=0.15,
       melting_point=2896.0, specific_heat=251.0, thermal_conductivity=138.0,
       latent_heat_fusion=3.91e5, thermal_expansion=4.8e-6,
       resistivity=5.34e-8, resistivity_temp_coeff=4.7e-3, vickers_hardness=mpa(2200),
       hugoniot=Hugoniot(5124.0, 1.233, None, _MARSH),
       constitutive=(JohnsonCook(mpa(926), mpa(369), 0.51, 0.043, 0.45, 1.0, t_ref=297.0, t_melt=2895.0,
                                 source="Chen et al., Materials 14 (2021) 4847 (modified JC, Mo-C grade), converted to eps0 = 1/s",
                                 condition="commercially pure", rate_range=(3e-4, 6.2e3),
                                 temperature_range=(297.0, 1273.0)),),
       hall_petch=HallPetch(k=1.0e6, confidence=Confidence.ESTIMATED), burgers_vector=2.73e-10, dbtt=300.0,
       spd_saturation_grain_size=150e-9, typical_grain_size=20e-6, self_diffusion_activation=405e3,
       conventional_working=(1273.0, 1573.0), conventional_working_note="hot worked at 1000-1300 C",
       notes="high-purity P/M Mo yields at 411-498 MPa", confidence=Confidence.REPRESENTATIVE),
    _m(key="rhenium", name="Rhenium (annealed)", category="refractory", base_element="Re",
       crystal_structure="hcp", condition="annealed",
       density=21020, youngs_modulus=gpa(463), poisson=0.30,
       yield_strength=mpa(290), ultimate_strength=mpa(1070), elongation=0.25,
       melting_point=3459.0, specific_heat=137.0, thermal_conductivity=48.0,
       latent_heat_fusion=3.25e5, thermal_expansion=6.2e-6,
       resistivity=1.93e-7, resistivity_temp_coeff=4.5e-3, vickers_hardness=mpa(2450),
       hall_petch=HallPetch(k=0.6e6, confidence=Confidence.ESTIMATED), burgers_vector=2.76e-10,
       spd_saturation_grain_size=100e-9, typical_grain_size=20e-6, self_diffusion_activation=500e3,
       conventional_working=(293.0, 293.0),
       conventional_working_note="cold rolled with intermediate anneals near 1700-2000 C; hot work limited by oxidation",
       notes="no measured Hugoniot recorded; estimated from sqrt(K/rho)", confidence=Confidence.REPRESENTATIVE),
    _m(key="tantalum", name="Tantalum, unalloyed (annealed)", category="refractory", base_element="Ta",
       crystal_structure="bcc", condition="annealed",
       density=16690, youngs_modulus=gpa(186), poisson=0.34,
       yield_strength=mpa(170), ultimate_strength=mpa(285), elongation=0.30,
       melting_point=3290.0, specific_heat=140.0, thermal_conductivity=57.5,
       latent_heat_fusion=2.02e5, thermal_expansion=6.3e-6,
       resistivity=1.31e-7, resistivity_temp_coeff=3.8e-3, vickers_hardness=mpa(1000),
       hugoniot=Hugoniot(3414.0, 1.201, None, _MARSH),
       constitutive=(JohnsonCook(mpa(340), mpa(750), 0.70, 0.0575, 0.40, 1.0, t_ref=298.0, t_melt=3250.0,
                                 source="Chen & Gray, Metall. Mater. Trans. A 27 (1996) 2994, Table III",
                                 condition="unalloyed Ta", rate_range=(1e-3, 4e3), temperature_range=(77.0, 1273.0)),
                     TA_ZA_CHEN_GRAY),
       hall_petch=HallPetch(k=0.5e6, confidence=Confidence.ESTIMATED), burgers_vector=2.86e-10,
       spd_saturation_grain_size=100e-9, typical_grain_size=40e-6, self_diffusion_activation=413e3,
       conventional_working=(293.0, 293.0), conventional_working_note="cold workable at room temperature",
       confidence=Confidence.HANDBOOK),
    _m(key="niobium", name="Niobium (annealed)", category="refractory", base_element="Nb",
       crystal_structure="bcc", condition="annealed",
       density=8570, youngs_modulus=gpa(105), poisson=0.40,
       yield_strength=mpa(150), ultimate_strength=mpa(275), elongation=0.30,
       melting_point=2750.0, specific_heat=265.0, thermal_conductivity=53.7,
       latent_heat_fusion=3.23e5, thermal_expansion=7.3e-6,
       resistivity=1.52e-7, resistivity_temp_coeff=3.9e-3, vickers_hardness=mpa(900),
       hugoniot=Hugoniot(4440.0, 1.207, None, _MARSH),
       hall_petch=HallPetch(k=0.1e6, confidence=Confidence.ESTIMATED), burgers_vector=2.86e-10,
       spd_saturation_grain_size=150e-9, typical_grain_size=50e-6, self_diffusion_activation=400e3,
       conventional_working=(293.0, 293.0), conventional_working_note="cold workable at room temperature",
       confidence=Confidence.REPRESENTATIVE),
    # ------------------------------------------------- aerospace alloys
    _m(key="gamma_tial", name="Gamma titanium aluminide Ti-48Al-2Cr-2Nb (cast + HIP)", category="intermetallic",
       base_element="Ti", crystal_structure="L1_0", condition="cast + HIP",
       density=3970, youngs_modulus=gpa(170), poisson=0.23,
       yield_strength=mpa(400), ultimate_strength=mpa(480), elongation=0.015, compressive_yield=mpa(550),
       melting_point=1733.0, specific_heat=610.0, thermal_conductivity=22.0,
       latent_heat_fusion=3.5e5, thermal_expansion=11e-6,
       resistivity=7e-7, resistivity_temp_coeff=1e-3, vickers_hardness=mpa(2750),
       constitutive=(JohnsonCook(mpa(927), mpa(1394.5), 1.1816, 0.0318, 1.6635, 1e-3, t_ref=293.0, t_melt=1733.0,
                                 source="Materials 16 (2023) 6715 (PMC10608512), SHPB compression",
                                 condition="Ti-48Al-2Cr-2Nb, compression; upper bound on flow stress",
                                 rate_range=(3e3, 1.1e4), temperature_range=(293.0, 773.0)),),
       hall_petch=HallPetch(k=1.0e6, confidence=Confidence.ESTIMATED), burgers_vector=2.83e-10, dbtt=973.0,
       typical_grain_size=200e-6, self_diffusion_activation=300e3,
       conventional_working=(1373.0, 1573.0), conventional_working_note="isothermal forging at 1100-1300 C",
       notes="tensile yield 330-480 MPa; compressive dynamic flow stress 1.0-1.5 GPa", confidence=Confidence.REPRESENTATIVE),
    _m(key="ods_fecral", name="ODS FeCrAl (MA956 / PM2000 class)", category="ods", base_element="Fe",
       crystal_structure="bcc", condition="recrystallised bar",
       density=7250, youngs_modulus=gpa(269), poisson=0.30,
       yield_strength=mpa(517), ultimate_strength=mpa(645), elongation=0.10,
       melting_point=1753.0, specific_heat=470.0, thermal_conductivity=10.9,
       latent_heat_fusion=2.7e5, thermal_expansion=11e-6,
       resistivity=1.3e-6, resistivity_temp_coeff=1e-4, vickers_hardness=mpa(2350),
       hugoniot=_ALPHA_FE_HUGONIOT, hugoniot_high=_EPS_FE_HUGONIOT, phase_transition_pressure=gpa(13.0),
       burgers_vector=2.49e-10, dbtt=300.0, typical_grain_size=500e-6, self_diffusion_activation=300e3,
       conventional_working=(1273.0, 1473.0), notes="oxide-dispersion strengthened; coarse recrystallised grains",
       confidence=Confidence.REPRESENTATIVE),
    _m(key="ods_ni", name="ODS nickel alloy (MA754 class)", category="ods", base_element="Ni",
       crystal_structure="fcc", condition="hot worked, recrystallised",
       density=8300, youngs_modulus=gpa(200), poisson=0.30,
       yield_strength=mpa(586), ultimate_strength=mpa(965), elongation=0.21,
       melting_point=1673.0, specific_heat=440.0, thermal_conductivity=14.3,
       latent_heat_fusion=2.9e5, thermal_expansion=12.2e-6,
       resistivity=1.07e-6, resistivity_temp_coeff=1e-4, vickers_hardness=mpa(2800),
       burgers_vector=2.49e-10, typical_grain_size=300e-6, self_diffusion_activation=285e3,
       conventional_working=(1273.0, 1473.0), notes="turbine vane ODS alloy", confidence=Confidence.ESTIMATED),
    _m(key="inconel_718", name="Inconel 718 (solution treated + aged)", category="superalloy", base_element="Ni",
       crystal_structure="fcc", condition="STA",
       density=8190, youngs_modulus=gpa(200), poisson=0.29,
       yield_strength=mpa(1100), ultimate_strength=mpa(1375), elongation=0.12,
       melting_point=1533.0, specific_heat=435.0, thermal_conductivity=11.4,
       latent_heat_fusion=2.7e5, thermal_expansion=13e-6,
       resistivity=1.25e-6, resistivity_temp_coeff=2e-4, vickers_hardness=mpa(4300),
       constitutive=(JohnsonCook(mpa(1241), mpa(622), 0.6522, 0.0134, 1.3, 1.0, t_melt=1570.0,
                                 source="Pereira & Lerch, Int. J. Impact Eng. 25 (2001) 715", condition="STA"),),
       burgers_vector=2.54e-10, spd_saturation_grain_size=100e-9, typical_grain_size=30e-6,
       self_diffusion_activation=400e3, conventional_working=(1223.0, 1393.0),
       notes="solidus 1533 K used as melting point", confidence=Confidence.HANDBOOK),
    _m(key="hastelloy_c276", name="Hastelloy C-276 (annealed)", category="superalloy", base_element="Ni",
       crystal_structure="fcc", condition="annealed",
       density=8890, youngs_modulus=gpa(205), poisson=0.30,
       yield_strength=mpa(355), ultimate_strength=mpa(790), elongation=0.50,
       melting_point=1596.0, specific_heat=427.0, thermal_conductivity=10.2, thermal_expansion=11.2e-6,
       resistivity=1.30e-6, resistivity_temp_coeff=1e-4, vickers_hardness=mpa(1950),
       burgers_vector=2.54e-10, typical_grain_size=50e-6, conventional_working=(1144.0, 1505.0),
       notes="REBCO tape substrate material", confidence=Confidence.HANDBOOK),
    # ------------------------------------------- magnet conductors/structure
    _m(key="cuag_conductor", name="Cu-Ag microcomposite wire (pulsed-magnet conductor)", category="conductor",
       base_element="Cu", crystal_structure="fcc", condition="heavily drawn",
       density=9285, youngs_modulus=gpa(120), poisson=0.34,
       yield_strength=mpa(850), ultimate_strength=mpa(1000), elongation=0.03,
       melting_point=1052.0, specific_heat=350.0, thermal_conductivity=330.0, latent_heat_fusion=1.84e5,
       resistivity=2.6e-8, resistivity_temp_coeff=2.6e-3, vickers_hardness=mpa(2900),
       burgers_vector=2.56e-10, conventional_working=(293.0, 293.0),
       notes="melting point is the Cu-Ag eutectic (779 C); strength depends on wire size",
       sources=("Han et al., NHMFL Cu-Ag conductor development",), confidence=Confidence.REPRESENTATIVE),
    _m(key="mp35n", name="MP35N (cold worked + aged)", category="superalloy", base_element="Co",
       crystal_structure="fcc", condition="cold worked + aged",
       density=8430, youngs_modulus=gpa(233), poisson=0.30,
       yield_strength=mpa(1900), ultimate_strength=mpa(2050), elongation=0.08,
       melting_point=1713.0, specific_heat=418.0, thermal_conductivity=11.2, latent_heat_fusion=2.7e5,
       resistivity=1.03e-6, resistivity_temp_coeff=1e-4, vickers_hardness=mpa(5000),
       notes="reinforcement shells of pulsed magnets", confidence=Confidence.REPRESENTATIVE),
    _m(key="maraging_c350", name="Maraging steel C350 (aged)", category="metal", base_element="Fe",
       crystal_structure="bcc", condition="aged",
       density=8080, youngs_modulus=gpa(200), poisson=0.30,
       yield_strength=mpa(2390), ultimate_strength=mpa(2420), elongation=0.07,
       melting_point=1686.0, specific_heat=452.0, thermal_conductivity=25.5, latent_heat_fusion=2.6e5,
       resistivity=6e-7, resistivity_temp_coeff=1e-3, vickers_hardness=mpa(5900),
       hugoniot=_ALPHA_FE_HUGONIOT, hugoniot_high=_EPS_FE_HUGONIOT, phase_transition_pressure=gpa(13.0),
       burgers_vector=2.48e-10,
       notes="high-pressure vessel and autofrettage material", confidence=Confidence.REPRESENTATIVE),
    _m(key="zylon_fibre", name="Zylon HM (PBO) fibre", category="fibre", base_element="C",
       crystal_structure="fibre",
       density=1560, youngs_modulus=gpa(270), poisson=0.3,
       ultimate_strength=gpa(5.8), elongation=0.025,
       melting_point=923.0, specific_heat=1000.0, thermal_conductivity=20.0,
       resistivity=1e14, notes="pulsed-magnet reinforcement; decomposes near 650 C; electrical insulator",
       confidence=Confidence.HANDBOOK),
    # ----------------------------------------------------- carbon structure
    _m(key="carbon_fibre_t1100g", name="Carbon fibre, Toray T1100G", category="fibre", base_element="C",
       crystal_structure="fibre", density=1790, youngs_modulus=gpa(324), poisson=0.25,
       ultimate_strength=gpa(7.0), elongation=0.022,
       melting_point=3900.0, specific_heat=710.0, thermal_conductivity=10.0, resistivity=1.4e-5,
       notes="strongest commercial carbon fibre", confidence=Confidence.HANDBOOK),
    _m(key="carbon_fibre_m60j", name="Carbon fibre, Toray M60J (high modulus)", category="fibre", base_element="C",
       crystal_structure="fibre", density=1930, youngs_modulus=gpa(588), poisson=0.25,
       ultimate_strength=gpa(3.82), elongation=0.007,
       melting_point=3900.0, specific_heat=710.0, thermal_conductivity=150.0, resistivity=7e-6,
       notes="highest-modulus PAN fibre", confidence=Confidence.HANDBOOK),
    _m(key="cnt_fibre_wetspun", name="CNT fibre, continuous wet-spun (state of the art)", category="carbon",
       base_element="C", crystal_structure="fibre", density=1500, youngs_modulus=gpa(260), poisson=0.25,
       ultimate_strength=gpa(4.2), elongation=0.016,
       melting_point=3900.0, specific_heat=710.0, thermal_conductivity=100.0, resistivity=9.2e-8,
       sources=("Taylor et al., Carbon 171 (2021) 689",), confidence=Confidence.HANDBOOK),
    _m(key="cnt_fibre_record", name="CNT/graphitic fibre, laboratory record", category="carbon",
       base_element="C", crystal_structure="fibre", density=1900, youngs_modulus=gpa(629), poisson=0.25,
       ultimate_strength=gpa(6.57), elongation=0.011,
       melting_point=3900.0, specific_heat=710.0, thermal_conductivity=482.0, resistivity=4.5e-7,
       sources=("Lee et al., Sci. Adv. 8 (2022) eabn0939",),
       notes="short gauge length, lab scale", confidence=Confidence.HANDBOOK),
    _m(key="graphene_fibre_sota", name="Graphene fibre, state of the art (carbon-nanosheet benchmark)",
       category="carbon", base_element="C", crystal_structure="fibre", density=1900,
       youngs_modulus=gpa(282), poisson=0.25, ultimate_strength=gpa(1.45), elongation=0.005,
       melting_point=3900.0, specific_heat=710.0, thermal_conductivity=1000.0, resistivity=4.5e-6,
       sources=("Xu et al., Adv. Mater. 28 (2016) 6449",),
       notes="best macroscopic carbon-nanosheet (graphene) fibres and films", confidence=Confidence.REPRESENTATIVE),
    _m(key="cns_engineering_target", name="Carbon-nanosheet laminate, engineering target (aspirational)",
       category="carbon", base_element="C", crystal_structure="composite", density=2000,
       youngs_modulus=gpa(300), poisson=0.2, ultimate_strength=gpa(10.0), elongation=0.033,
       melting_point=3900.0, specific_heat=710.0, thermal_conductivity=500.0, resistivity=1e-6,
       notes="hypothetical; 3.3 % failure strain is far above any layered carbon material; "
             "at the 0.4 % REBCO strain limit only 1.2 GPa is usable",
       confidence=Confidence.ESTIMATED),
    _m(key="graphene_intrinsic", name="Graphene, intrinsic monolayer (theoretical ceiling)", category="carbon",
       base_element="C", crystal_structure="graphite", density=2260, youngs_modulus=gpa(1000), poisson=0.17,
       ultimate_strength=gpa(130), elongation=0.25,
       melting_point=3900.0, specific_heat=710.0, thermal_conductivity=3000.0, resistivity=1e-8,
       sources=("Lee, Wei, Kysar & Hone, Science 321 (2008) 385",),
       notes="defect-free monolayer; not attainable in macroscopic laminates", confidence=Confidence.HANDBOOK),
    # --------------------------------------------- synthesis / anvil phases
    _m(key="graphite_crystal", name="Graphite, crystalline (thermodynamic reference)", category="carbon",
       base_element="C", crystal_structure="graphite", density=2260, youngs_modulus=gpa(36.5), poisson=0.2,
       bulk_modulus=gpa(33.8), melting_point=3900.0, specific_heat=710.0, thermal_conductivity=400.0,
       resistivity=4e-7,
       hugoniot=Hugoniot(3700.0, 1.6, gpa(20.0), f"estimated; range c0 2.5-4.5 km/s ({_MARSH})", Confidence.ESTIMATED),
       notes="used for phase-boundary and volume-change calculations", confidence=Confidence.REPRESENTATIVE),
    _m(key="graphite_polycrystalline", name="Graphite, isostatic polycrystalline (feedstock)", category="carbon",
       base_element="C", crystal_structure="graphite", density=1800, youngs_modulus=gpa(10), poisson=0.2,
       compressive_strength=mpa(150), melting_point=3900.0, specific_heat=710.0, thermal_conductivity=100.0,
       resistivity=1e-5,
       hugoniot=Hugoniot(2000.0, 1.8, gpa(20.0), "estimated for porous graphite", Confidence.ESTIMATED),
       notes="porous feedstock; Hugoniot highly uncertain", confidence=Confidence.REPRESENTATIVE),
    _m(key="diamond", name="Diamond", category="ceramic", base_element="C", crystal_structure="diamond",
       density=3515, youngs_modulus=gpa(1050), poisson=0.07, bulk_modulus=gpa(443),
       melting_point=4000.0, specific_heat=509.0, thermal_conductivity=2000.0, resistivity=1e12,
       vickers_hardness=gpa(100),
       hugoniot=Hugoniot(12160.0, 1.0, None, "estimated", Confidence.ESTIMATED),
       notes="graphitises above ~1800 K at low pressure; 'melting point' nominal",
       confidence=Confidence.HANDBOOK),
    _m(key="hbn", name="Hexagonal boron nitride", category="ceramic", base_element="BN", crystal_structure="hexagonal",
       density=2280, youngs_modulus=gpa(34), poisson=0.2, melting_point=3200.0, specific_heat=800.0,
       thermal_conductivity=30.0, resistivity=1e13, confidence=Confidence.ESTIMATED),
    _m(key="cbn", name="Cubic boron nitride", category="ceramic", base_element="BN", crystal_structure="cubic",
       density=3480, youngs_modulus=gpa(909), poisson=0.12, bulk_modulus=gpa(369),
       melting_point=3200.0, specific_heat=800.0, thermal_conductivity=740.0, resistivity=1e10,
       vickers_hardness=gpa(47), confidence=Confidence.HANDBOOK),
    _m(key="tungsten_carbide_co", name="Cemented tungsten carbide WC-6Co (anvil grade)", category="ceramic",
       base_element="WC", crystal_structure="hexagonal",
       density=14950, youngs_modulus=gpa(620), poisson=0.22, compressive_strength=gpa(5.5),
       ultimate_strength=gpa(2.5), melting_point=1768.0, specific_heat=200.0, thermal_conductivity=100.0,
       resistivity=2e-7, vickers_hardness=gpa(15),
       notes="ultimate_strength is transverse rupture strength; melting point is the Co binder",
       confidence=Confidence.REPRESENTATIVE),
]

MATERIALS: dict[str, Material] = {m.key: m for m in _ENTRIES}

ALIASES: dict[str, str] = {
    "cu": "copper_ofhc", "copper": "copper_ofhc", "ofhc": "copper_ofhc",
    "al": "aluminum_6061", "aluminium": "aluminum_6061", "aluminum": "aluminum_6061",
    "6061": "aluminum_6061", "al6061": "aluminum_6061", "1100": "aluminum_1100", "al1100": "aluminum_1100",
    "aluminium_6061": "aluminum_6061", "aluminium_1100": "aluminum_1100",
    "mg": "magnesium_az31", "az31": "magnesium_az31", "az31b": "magnesium_az31",
    "steel": "steel_1018", "1018": "steel_1018", "mild_steel": "steel_1018", "4340": "steel_4340",
    "304": "steel_304", "ss304": "steel_304", "stainless": "steel_304", "fe": "iron_armco", "iron": "iron_armco",
    "armco": "iron_armco", "ti": "titanium_cp2", "titanium": "titanium_cp2", "cp_ti": "titanium_cp2",
    "ti64": "ti6al4v", "ti-6al-4v": "ti6al4v", "w": "tungsten", "wha": "tungsten_heavy_alloy_90w",
    "mo": "molybdenum", "re": "rhenium", "ta": "tantalum", "nb": "niobium",
    "tial": "gamma_tial", "gamma-tial": "gamma_tial", "ti4822": "gamma_tial",
    "ods": "ods_fecral", "ma956": "ods_fecral", "pm2000": "ods_fecral", "ma754": "ods_ni",
    "in718": "inconel_718", "inconel": "inconel_718", "718": "inconel_718", "c276": "hastelloy_c276",
    "hastelloy": "hastelloy_c276", "cuag": "cuag_conductor", "zylon": "zylon_fibre", "pbo": "zylon_fibre",
    "t1100g": "carbon_fibre_t1100g", "m60j": "carbon_fibre_m60j", "cnt": "cnt_fibre_wetspun",
    "graphene": "graphene_fibre_sota", "cns": "cns_engineering_target", "cns_ideal": "graphene_intrinsic",
    "graphite": "graphite_crystal", "wc": "tungsten_carbide_co", "c350": "maraging_c350",
    "maraging": "maraging_c350",
}

MaterialLike = Union[Material, str]


def keys() -> tuple[str, ...]:
    """All registry keys in definition order."""
    return tuple(MATERIALS)


def get(key_or_alias: str) -> Material:
    """Case-insensitive lookup by key or alias.

    Raises :class:`UnknownMaterialError` (a ``KeyError``) with suggestions.
    """
    if not isinstance(key_or_alias, str):
        raise UnknownMaterialError(f"material key must be a string, got {type(key_or_alias).__name__}")
    k = key_or_alias.strip().lower()
    if k in MATERIALS:
        return MATERIALS[k]
    if k in ALIASES:
        return MATERIALS[ALIASES[k]]
    close = difflib.get_close_matches(k, list(MATERIALS) + list(ALIASES), n=4, cutoff=0.5)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise UnknownMaterialError(f"unknown material '{key_or_alias}'{hint}")


def as_material(material: MaterialLike) -> Material:
    """Accept a :class:`Material` or a registry key/alias."""
    return material if isinstance(material, Material) else get(material)


def list_materials(category: str | None = None) -> list[Material]:
    """All materials, optionally filtered by ``category``."""
    if category is None:
        return list(MATERIALS.values())
    return [m for m in MATERIALS.values() if m.category == category]


def search(text: str) -> list[Material]:
    """Materials whose key, name, notes or condition contain ``text`` (case-insensitive)."""
    t = text.lower()
    return [m for m in MATERIALS.values() if t in m.key or t in m.name.lower() or t in m.notes.lower()
            or t in m.condition.lower()]


CATEGORIES: tuple[str, ...] = tuple(sorted({m.category for m in _ENTRIES}))

__all__ = [
    "Hugoniot", "JohnsonCook", "ZerilliArmstrongBCC", "ConstitutiveModel", "HallPetch", "Material",
    "MaterialLike", "MATERIALS", "ALIASES", "CATEGORIES", "W_ZA_LENNON_RAMESH", "W_ZA_CHEN_GRAY",
    "TA_ZA_CHEN_GRAY", "keys", "get", "as_material", "list_materials", "search",
]
