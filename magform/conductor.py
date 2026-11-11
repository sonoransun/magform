"""REBCO conductor, carbon-nanosheet (CNS) reinforced winding, and pulsed-operation limits.

The concept note proposes "self-supporting, carbon-nanosheet-reinforced
superconducting magnets" producing 63-100 T.  This module quantifies the three
things such a winding must do and labels each honestly:

1. **Carry current** (:class:`REBCOTape`, :func:`assess_conductor`,
   :func:`je_limited_graded_thickness`).  A parametric engineering-current-density
   fit representative of 2020s commercial 4 mm REBCO tape (NOT a datasheet),
   calibrated at 4.2 K against published short-sample data measured to 31 T.
   Above 31 T the Je numbers are model extrapolations (:func:`je_data_status`).
2. **Carry the magnetic load** (:class:`CNSComposite`, :func:`assess_structure`).
   An iso-strain rule of mixtures: every constituent shares the strain, and the
   REBCO layer tolerates only ~0.4 % tension.  The binding constraint is
   stiffness, not strength: at 4 GPa hoop stress and 0.4 % strain the winding
   needs E >= 1 TPa (graphene's intrinsic modulus); because the hoop-stress
   peak never drops below P/2 for any winding thickness
   (:func:`magform.magnetics.max_hoop_stress_ratio`), thick coils only halve it.
3. **Survive being pulsed** (:func:`tape_pulse_losses`, :func:`enthalpy_margin`,
   :func:`assess_pulsed_operation`).  A static field exerts no force on a
   stationary workpiece, so a forming magnet must swing its field in
   microseconds to milliseconds.  Sweeping a REBCO winding by ~100 T that fast
   dissipates 1e7-1e8 J/m^3 per pulse against an enthalpy margin (4.2 K to the
   current-sharing temperature) of order 1e5-1e6 J/m^3: it quenches every pulse.  The physically consistent layout is a
   DC superconducting background plus a pulsed resistive insert
   (:func:`magform.magnetics.pressure_with_background`).

SI units throughout (A/m^2 for current densities; 1 A/mm^2 = 1e6 A/m^2).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from ._numerics import bisect, interp1, linspace, trapz
from ._report import Assessment, ResultMixin, unit
from .benchmarks import SC_CONDUCTOR_RECORD, ramp_rate_status, superconducting_field_status
from .constants import MU0, T_LIQUID_HELIUM
from .errors import InputError
from .magnetics import graded_winding_thickness
from .materials import Material, MaterialLike, as_material
from .status import Status, worst

REFERENCES: dict[str, str] = {
    "tsuchiya2021": "K. Tsuchiya et al., 'Superconducting properties of commercial REBCO-coated conductors "
                    "with artificial pinning centers', Supercond. Sci. Technol. 34 (2021) 105005.",
    "braccini2011": "V. Braccini et al., 'Properties of recent IBAD-MOCVD coated conductors relevant to their "
                    "high field, low temperature magnet use', Supercond. Sci. Technol. 24 (2011) 035001.",
    "xu2014": "A. Xu, J. J. Jaroszynski, F. Kametani, Z. Chen, D. C. Larbalestier et al., 'Strongly enhanced "
              "vortex pinning from 4 to 77 K in magnetic fields up to 31 T in 15 mol.% Zr-added (Gd, Y)-Ba-Cu-O "
              "superconducting tapes', APL Mater. 2 (2014) 046111.",
    "senatore2016": "C. Senatore, C. Barth, M. Bonura, M. Kulich and G. Mondonico, 'Field and temperature "
                    "scaling of the critical current density in commercial REBCO coated conductors', "
                    "Supercond. Sci. Technol. 29 (2016) 014002.",
    "barth2015": "C. Barth, G. Mondonico and C. Senatore, 'Electro-mechanical properties of REBCO coated "
                 "conductors from various industrial manufacturers at 77 K, self-field and 4.2 K, 19 T', "
                 "Supercond. Sci. Technol. 28 (2015) 045011.",
    "hahn2019": "S. Hahn et al., '45.5-tesla direct-current magnetic field generated with a high-temperature "
                "superconducting magnet', Nature 570 (2019) 496-499.",
    "maglab2025": "National High Magnetic Field Laboratory, news release of 11 Sep 2025 and research highlight "
                  "of Jan 2026: REBCO test coil 'Little Big Coil 9', 48.7 T inside a 31 T resistive magnet "
                  "(August 2025).",
    "kim1962": "Y. B. Kim, C. F. Hempstead and A. R. Strnad, 'Critical persistent currents in hard "
               "superconductors', Phys. Rev. Lett. 9 (1962) 306-309.",
    "bean1964": "C. P. Bean, 'Magnetization of high-field superconductors', Rev. Mod. Phys. 36 (1964) 31-39.",
    "norris1970": "W. T. Norris, 'Calculation of hysteresis losses in hard superconductors carrying ac: "
                  "isolated conductors and edges of thin sheets', J. Phys. D: Appl. Phys. 3 (1970) 489-507.",
    "brandt1993": "E. H. Brandt and M. Indenbom, 'Type-II-superconductor strip with current in a "
                  "perpendicular magnetic field', Phys. Rev. B 48 (1993) 12893-12906.",
    "wilson1983": "M. N. Wilson, Superconducting Magnets, Clarendon Press, Oxford (1983), ch. 8 (AC losses).",
    "iwasa2009": "Y. Iwasa, Case Studies in Superconducting Magnets: Design and Operational Issues, "
                 "2nd ed., Springer, New York (2009) (stability, enthalpy margin, current sharing).",
    "ekin2006": "J. W. Ekin, Experimental Techniques for Low-Temperature Measurements, Oxford University "
                "Press (2006), appendix A6 (specific heat of technical materials).",
    "nist_cu": "NIST Cryogenic Technologies Group, Material Properties: OFHC Copper (UNS C10100/C10200), "
               "specific heat and resistivity fits (trc.nist.gov/cryogenics).",
    "lu2008": "J. Lu, E. S. Choi and H. D. Zhou, 'Physical properties of Hastelloy C-276 at cryogenic "
              "temperatures', J. Appl. Phys. 103 (2008) 064908.",
    "toray": "Toray Industries, TORAYCA M60J and T1100G technical data sheets.",
    "lee2008": "C. Lee, X. Wei, J. W. Kysar and J. Hone, 'Measurement of the elastic properties and intrinsic "
               "strength of monolayer graphene', Science 321 (2008) 385-388.",
    "lee2022": "D. Lee et al., 'Ultrahigh strength, modulus, and conductivity of graphitic fibers by "
               "macromolecular coalescence', Sci. Adv. 8 (2022) eabn0939.",
    "knoepfel2000": "H. Knoepfel, Magnetic Fields: A Comprehensive Theoretical Treatise for Practical Use, "
                    "Wiley, New York (2000) (skin depth, eddy-current losses).",
    "herlach2003": "F. Herlach and N. Miura (eds.), High Magnetic Fields: Science and Technology, Vol. 1, "
                   "World Scientific, Singapore (2003) (all >60 T non-destructive magnets are resistive).",
}

# ---------------------------------------------------------------------------
# module constants (documented single sources)
# ---------------------------------------------------------------------------
ORIENTATIONS: tuple[str, str] = ("parallel", "perpendicular")
"""Field orientations of the Je model: in the tape plane, or normal to the tape face (B || c)."""

JE_MEASURED_FIELD: float = 31.0
"""T.  Highest DC field at which commercial-tape Je(B) is routinely measured on short samples
(31 T resistive magnets; Xu et al. 2014, Braccini et al. 2011).  Above it the Je model is extrapolated;
up to ``benchmarks.SC_CONDUCTOR_RECORD`` (48.7 T) single test coils have operated inside resistive magnets."""

DEFAULT_END_FIELD_FRACTION: float = 0.4
"""Radial (tape-perpendicular) field at the coil ends as a fraction of the central field; 0.3-0.5 is typical
of solenoids (estimated; Iwasa 2009)."""

ADIABATIC_RISE_TIME: float = 1.0
"""s.  Pulses shorter than this are treated as adiabatic: heat cannot leave a winding pack (thermal diffusion
time of order seconds) during one pulse, so the per-pulse loss must fit inside the enthalpy margin
(order-of-magnitude judgement; Wilson 1983, Iwasa 2009)."""

LOSS_MARGIN_CAUTION: float = 0.3
"""Loss-to-enthalpy-margin ratio above which pulsed operation is labelled extrapolation (estimated design
margin)."""

CU_MAGNETORESISTANCE_FIELD: float = 10.0
"""T.  rho_Cu(4 K, B) = rho_Cu(4 K) (1 + B/10 T): linear magnetoresistance of RRR ~ 50-100 copper
(Kohler-rule estimate; NIST copper data)."""

INSULATION_DENSITY: float = 1420.0
"""kg/m^3.  Polyimide (Kapton HN) turn insulation."""

REBCO_DENSITY: float = 6380.0
"""kg/m^3.  YBa2Cu3O7 X-ray density (REBCO layer of the tape)."""

SILVER_DENSITY: float = 10490.0
"""kg/m^3.  Density assigned to the tape thickness not in REBCO, copper or substrate (silver cap, buffers)."""

TABLE_T_MAX: float = 300.0
"""K.  Upper end of the specific-heat tables."""

CU_SPECIFIC_HEAT: tuple[tuple[float, float], ...] = (
    (4.0, 0.09), (6.0, 0.23), (8.0, 0.47), (10.0, 0.86), (15.0, 2.7), (20.0, 7.4), (25.0, 15.5),
    (30.0, 27.0), (40.0, 59.0), (50.0, 98.0), (60.0, 137.0), (77.0, 192.0), (100.0, 252.0),
    (150.0, 323.0), (200.0, 356.0), (250.0, 374.0), (300.0, 385.0),
)
"""(K, J/(kg K)).  Specific heat of OFHC copper (NIST cryogenic data; Ekin 2006)."""

HASTELLOY_SPECIFIC_HEAT: tuple[tuple[float, float], ...] = (
    (4.0, 0.6), (10.0, 1.6), (20.0, 4.0), (30.0, 10.0), (40.0, 20.0), (50.0, 40.0), (77.0, 150.0),
    (100.0, 220.0), (200.0, 380.0), (300.0, 410.0),
)
"""(K, J/(kg K)).  Specific heat of Hastelloy C-276 (estimated; cf. Lu, Choi & Zhou 2008)."""

STRUCTURE_BENCHMARKS: tuple[str, ...] = (
    "carbon_fibre_m60j", "carbon_fibre_t1100g", "cnt_fibre_wetspun", "cnt_fibre_record",
    "graphene_fibre_sota", "cns_engineering_target", "graphene_intrinsic",
)
"""Reinforcements against which :func:`assess_structure` compares the requirement."""

STRUCTURE_LADDER: tuple[tuple[Status, tuple[str, ...]], ...] = (
    (Status.STATE_OF_THE_ART, ("carbon_fibre_m60j", "carbon_fibre_t1100g")),
    (Status.EXTRAPOLATION, ("cnt_fibre_record",)),
    (Status.ASPIRATIONAL, ("cns_engineering_target", "graphene_intrinsic")),
)
"""Status ladder of :func:`assess_structure`: commercial fibre -> laboratory-record CNT fibre ->
aspirational CNS laminate / intrinsic graphene.  The other benchmarks are reported for reference."""


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


def _fraction(name: str, value: float) -> float:
    v = _finite(name, value)
    if not 0.0 <= v <= 1.0:
        raise InputError(f"{name} must lie in [0, 1], got {v!r}")
    return v


def _safety_factor(value: float) -> float:
    v = _finite("safety_factor", value)
    if v < 1.0:
        raise InputError(f"safety_factor must be >= 1, got {v!r}")
    return v


def _orientation(orientation: str) -> str:
    o = orientation.strip().lower() if isinstance(orientation, str) else orientation
    if o not in ORIENTATIONS:
        raise InputError(f"orientation must be 'parallel' or 'perpendicular', got {orientation!r}")
    return o


def _cp(table: tuple[tuple[float, float], ...], temperature: float) -> float:
    """Tabulated specific heat (J/(kg K)), linear interpolation; below the first point cp ~ T."""
    t0, c0 = table[0]
    if temperature < t0:
        return c0 * temperature / t0
    return interp1(temperature, [p[0] for p in table], [p[1] for p in table])


# ---------------------------------------------------------------------------
# REBCO tape
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class REBCOTape(ResultMixin):
    """REBCO coated-conductor tape with a parametric Je(B, T, orientation) model.

    Defaults: 4 mm x 0.1 mm commercial tape (1.6 um REBCO, 2 x 20 um Cu, 50 um
    Hastelloy) referenced to 4.2 K.  Model (parametric fit representative of
    2020s commercial REBCO; NOT a datasheet)::

        h(T)    = (1 - T/Tc) / (1 - 4.2/Tc)            (0 for T >= Tc)
        g(T)    = h^p                                    p = temperature_exponent
        Birr(T) = Birr_4k,o * h^q                        q = birr_temperature_exponent
        Je      = Je_sf * g(T) * (1 + B/B0_o)^(-alpha_o) * max(0, 1 - B/Birr_o(T))

    with o in {"parallel" (B in the tape plane), "perpendicular" (B || c, worst
    case)}; Je in A/m^2 over the whole tape cross-section.  Reproduces at 4.2 K
    (A/mm^2): perpendicular 1416 at 12 T, 1106 at 18 T, 740 at 30 T (measured
    1100-2500, 750-1850, 500-1300), parallel/perpendicular = 2.07 at 18 T
    (measured 1.8-3.5) and 360 at 77 K self-field (measured 300-500).
    Validity: 4.2 K data to 31 T; above 31 T (295 A/mm^2 perpendicular at 63 T)
    the values are extrapolations; in-field values at 77 K are overestimated
    because the field dependence is calibrated at 4.2 K.  Mechanical limits:
    0.4 % irreversible tensile strain, -0.6 % compressive.

    References: Tsuchiya et al. (2021); Braccini et al. (2011); Xu et al. (2014);
    Senatore et al. (2016); Kim et al. (1962) for the (1 + B/B0)^-alpha form;
    Barth et al. (2015) for the strain limit.
    """

    width: float = unit("m", default=4e-3)
    thickness: float = unit("m", default=0.1e-3)
    rebco_thickness: float = unit("m", default=1.6e-6)
    copper_thickness: float = unit("m", default=40e-6)
    copper_layer_thickness: float = unit("m", default=20e-6)
    substrate_thickness: float = unit("m", default=50e-6)
    tc: float = unit("K", default=92.0)
    je_sf_4k: float = unit("A/m^2", default=3.0e9)
    b0_perp: float = unit("T", default=8.8)
    alpha_perp: float = 0.75
    birr_perp_4k: float = unit("T", default=120.0)
    b0_par: float = unit("T", default=60.0)
    alpha_par: float = 0.75
    birr_par_4k: float = unit("T", default=250.0)
    temperature_exponent: float = 1.2
    birr_temperature_exponent: float = 1.5
    youngs_modulus: float = unit("Pa", default=160e9)
    irreversible_strain: float = 0.004
    compressive_strain_limit: float = -0.006

    def __post_init__(self) -> None:
        for name in ("width", "thickness", "rebco_thickness", "je_sf_4k", "b0_perp", "birr_perp_4k",
                     "b0_par", "birr_par_4k", "youngs_modulus", "irreversible_strain"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        for name in ("copper_thickness", "copper_layer_thickness", "substrate_thickness", "alpha_perp",
                     "alpha_par", "temperature_exponent", "birr_temperature_exponent"):
            object.__setattr__(self, name, _non_negative(name, getattr(self, name)))
        object.__setattr__(self, "tc", _finite("tc", self.tc))
        object.__setattr__(self, "compressive_strain_limit",
                           _finite("compressive_strain_limit", self.compressive_strain_limit))
        if self.tc <= T_LIQUID_HELIUM:
            raise InputError(f"tc must exceed the 4.2 K reference temperature, got {self.tc!r}")
        if self.compressive_strain_limit >= 0.0:
            raise InputError("compressive_strain_limit must be negative")
        layers = self.rebco_thickness + self.copper_thickness + self.substrate_thickness
        if layers > self.thickness * (1.0 + 1e-9):
            raise InputError(f"REBCO + copper + substrate ({layers!r} m) exceed the tape thickness "
                             f"({self.thickness!r} m)")
        if self.copper_layer_thickness > self.copper_thickness * (1.0 + 1e-9):
            raise InputError("copper_layer_thickness cannot exceed copper_thickness")

    # -- model ----------------------------------------------------------------
    def _reduced(self, temperature: float) -> float:
        t = _positive("temperature", temperature)
        if t >= self.tc:
            return 0.0
        return (1.0 - t / self.tc) / (1.0 - T_LIQUID_HELIUM / self.tc)

    def _params(self, orientation: str) -> tuple[float, float, float]:
        if _orientation(orientation) == "parallel":
            return self.b0_par, self.alpha_par, self.birr_par_4k
        return self.b0_perp, self.alpha_perp, self.birr_perp_4k

    def birr(self, temperature: float, orientation: str = "parallel") -> float:
        """Irreversibility field ``Birr(T) = Birr_4k,o ((1 - T/Tc)/(1 - 4.2/Tc))^q`` [T].

        ``temperature`` in K (> 0; Birr = 0 at and above Tc), ``orientation``
        "parallel" or "perpendicular".  Defaults: 250 T parallel and 120 T
        perpendicular at 4.2 K (YBCO Hc2 for H || c extrapolates to ~140 T).

        Reference: Senatore et al. (2016); Tsuchiya et al. (2021).
        """
        h = self._reduced(temperature)
        _, _, b_irr_4k = self._params(orientation)
        return b_irr_4k * h ** self.birr_temperature_exponent if h > 0.0 else 0.0

    def je(self, field: float, temperature: float, orientation: str = "parallel") -> float:
        """Engineering critical current density over the tape cross-section [A/m^2].

        ``Je = Je_sf g(T) (1 + B/B0_o)^(-alpha_o) max(0, 1 - B/Birr_o(T))``
        (see the class docstring).  ``field`` in T (>= 0), ``temperature`` in K
        (> 0; Je = 0 at and above Tc), ``orientation`` "parallel" or
        "perpendicular".  Measured basis to 31 T at 4.2 K; beyond that an
        extrapolation (:func:`je_data_status`).

        Reference: Tsuchiya et al. (2021); Braccini et al. (2011); Kim et al. (1962).
        """
        b = _non_negative("field", field)
        h = self._reduced(temperature)
        b0, alpha, b_irr_4k = self._params(orientation)
        if h <= 0.0:
            return 0.0
        b_irr = b_irr_4k * h ** self.birr_temperature_exponent
        cutoff = 1.0 - b / b_irr
        if cutoff <= 0.0:
            return 0.0
        return self.je_sf_4k * h ** self.temperature_exponent * (1.0 + b / b0) ** (-alpha) * cutoff

    def ic(self, field: float, temperature: float, orientation: str = "parallel") -> float:
        """Critical current of one tape, ``Ic = Je w t`` [A].

        ``field`` in T, ``temperature`` in K, ``orientation`` as :meth:`je`.
        Default tape: ~915 A at 4.2 K, 18 T in-plane; ~440 A perpendicular.

        Reference: Tsuchiya et al. (2021).
        """
        return self.je(field, temperature, orientation) * self.width * self.thickness

    def layer_jc(self, field: float, temperature: float, orientation: str = "parallel") -> float:
        """Critical current density of the REBCO layer itself, ``Jc = Je t / t_REBCO`` [A/m^2].

        Used for critical-state AC losses.  Default tape: 1.9e11 A/m^2 at 4.2 K, self-field;
        6.9e10 A/m^2 at 18 T perpendicular.

        Reference: Tsuchiya et al. (2021) (layer Jc up to ~60 kA/mm^2 at 4.2 K, 12-18 T).
        """
        return self.je(field, temperature, orientation) * self.thickness / self.rebco_thickness

    def current_sharing_temperature(self, field: float, j_operating: float,
                                    orientation: str = "parallel", *,
                                    lowest_temperature: float = T_LIQUID_HELIUM) -> float:
        """Current-sharing temperature: the T at which Je(B, T) falls to ``j_operating`` [K].

        Solves ``Je(B, T_cs) = J_op`` by bisection on [4.2 K, Tc] (Je is monotone
        in T).  ``field`` in T (>= 0), ``j_operating`` the tape current density in
        A/m^2 (> 0).  Returns 4.2 K when ``Je(B, 4.2 K) < J_op``: the tape is
        already current-sharing at 4.2 K and has no temperature margin there.
        ``lowest_temperature`` (keyword, default 4.2 K) moves the lower end of the
        search for sub-4.2 K operation (e.g. 1.8 K superfluid helium).

        Reference: Iwasa (2009); Wilson (1983).
        """
        b = _non_negative("field", field)
        j = _positive("j_operating", j_operating)
        o = _orientation(orientation)
        t_low = _positive("lowest_temperature", lowest_temperature)
        if t_low >= self.tc:
            raise InputError(f"lowest_temperature must be below Tc = {self.tc:g} K")
        if self.je(b, t_low, o) <= j:
            return t_low
        return bisect(lambda t: self.je(b, t, o) - j, t_low, self.tc, tol=1e-10)

    def density(self) -> float:
        """Tape density from its layers [kg/m^3] (estimated).

        ``rho = (t_Cu rho_Cu + t_sub rho_Hastelloy + t_REBCO rho_REBCO + t_rest rho_Ag)/t``, the
        remainder (silver cap, buffers) taken as silver.  Default tape ~9.0e3 kg/m^3.

        Reference: materials registry (copper_ofhc, hastelloy_c276); Ekin (2006).
        """
        cu = as_material("copper_ofhc").density
        sub = as_material("hastelloy_c276").density
        rest = max(self.thickness - self.copper_thickness - self.substrate_thickness - self.rebco_thickness, 0.0)
        return (self.copper_thickness * cu + self.substrate_thickness * sub + self.rebco_thickness * REBCO_DENSITY
                + rest * SILVER_DENSITY) / self.thickness


def je_data_status(field: float) -> Status:
    """Status of the *data* behind the Je model at ``field`` [T] (>= 0).

    <= 31 T (:data:`JE_MEASURED_FIELD`): state of the art (measured on commercial
    tape in DC magnets); above: extrapolation.  Between 31 T and 48.7 T
    (``benchmarks.SC_CONDUCTOR_RECORD``) REBCO test coils have operated inside
    resistive magnets; above it no superconductor has carried current, so the
    model is pure extrapolation.  Magnet-level field status is a separate axis
    (``benchmarks.superconducting_field_status``).

    Reference: Xu et al. (2014); Hahn et al. (2019); NHMFL (2025).
    """
    b = _non_negative("field", field)
    return Status.STATE_OF_THE_ART if b <= JE_MEASURED_FIELD else Status.EXTRAPOLATION


def _je_data_note(field: float) -> str:
    if field <= JE_MEASURED_FIELD:
        return f"Je model within the measured range (<= {JE_MEASURED_FIELD:g} T at 4.2 K)"
    if field <= SC_CONDUCTOR_RECORD:
        return (f"Je model extrapolated above {JE_MEASURED_FIELD:g} T; REBCO test coils have operated to "
                f"{SC_CONDUCTOR_RECORD:g} T inside resistive magnets (Hahn et al. 2019; NHMFL 2025)")
    return (f"Je model extrapolated beyond any superconductor operation (record {SC_CONDUCTOR_RECORD:g} T); "
            "treat values as model output only")


# ---------------------------------------------------------------------------
# CNS-reinforced composite winding
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CNSComposite(ResultMixin):
    """Winding composite of CNS reinforcement, REBCO tape, extra copper and insulation (volume fractions).

    ``f_cns + f_hts + f_cu + f_insulation = 1`` (+-1e-6), each in [0, 1].
    ``cns_material`` is a registry key/alias or a :class:`Material` (default the
    aspirational "cns_engineering_target": E = 300 GPa, 10 GPa strength);
    ``insulation_modulus`` in Pa; ``tape`` the conductor.  Iso-strain (Voigt)
    mechanics: all constituents share the hoop strain, so
    ``E_c = sum f_i E_i`` and the strain limit is the smallest constituent limit
    (REBCO's 0.4 %).  Validity: continuous constituents aligned with the hoop
    direction; no transverse/shear load, no thermal pre-strain.

    Reference: rule of mixtures (Voigt); Barth et al. (2015) for the REBCO
    strain limit; Toray data sheets; Lee et al. (2008, 2022).
    """

    f_cns: float = 0.5
    f_hts: float = 0.3
    f_cu: float = 0.1
    f_insulation: float = 0.1
    cns_material: MaterialLike = "cns_engineering_target"
    insulation_modulus: float = unit("Pa", default=5e9)
    tape: REBCOTape = dataclasses.field(default_factory=REBCOTape)

    def __post_init__(self) -> None:
        for name in ("f_cns", "f_hts", "f_cu", "f_insulation"):
            object.__setattr__(self, name, _fraction(name, getattr(self, name)))
        total = self.f_cns + self.f_hts + self.f_cu + self.f_insulation
        if abs(total - 1.0) > 1e-6:
            raise InputError(f"volume fractions must sum to 1, got {total!r}")
        mat = as_material(self.cns_material)
        if isinstance(self.cns_material, str):
            object.__setattr__(self, "cns_material", mat.key)
        object.__setattr__(self, "insulation_modulus", _positive("insulation_modulus", self.insulation_modulus))
        if not isinstance(self.tape, REBCOTape):
            raise InputError("tape must be a REBCOTape")

    @classmethod
    def default(cls) -> CNSComposite:
        """The reference composite: 50 % CNS, 30 % REBCO tape, 10 % Cu, 10 % insulation."""
        return cls()

    @property
    def cns(self) -> Material:
        """The reinforcement :class:`Material`."""
        return as_material(self.cns_material)

    def _rest_modulus_sum(self) -> float:
        e_cu = as_material("copper_ofhc").youngs_modulus
        return self.f_hts * self.tape.youngs_modulus + self.f_cu * e_cu + self.f_insulation * self.insulation_modulus

    def modulus(self) -> float:
        """Iso-strain composite modulus ``E_c = f_cns E_cns + f_hts E_tape + f_cu E_Cu + f_ins E_ins`` [Pa].

        E_Cu = 117 GPa (copper_ofhc record).  Default composite: 210.2 GPa.

        Reference: Voigt rule of mixtures.
        """
        return self.f_cns * self.cns.youngs_modulus + self._rest_modulus_sum()

    def matrix_modulus(self) -> float:
        """Modulus of the non-reinforcement constituents in their own proportions [Pa].

        ``E_other = (f_hts E_tape + f_cu E_Cu + f_ins E_ins)/(1 - f_cns)`` (0 if f_cns = 1),
        so that ``E_c = f_cns E_cns + (1 - f_cns) E_other``.  Default: 120.4 GPa.

        Reference: Voigt rule of mixtures.
        """
        if self.f_cns >= 1.0 - 1e-12:
            return 0.0
        return self._rest_modulus_sum() / (1.0 - self.f_cns)

    def density(self) -> float:
        """Composite density ``sum f_i rho_i`` [kg/m^3] (tape from :meth:`REBCOTape.density`, polyimide insulation).

        Reference: rule of mixtures.
        """
        rho_cu = as_material("copper_ofhc").density
        return (self.f_cns * self.cns.density + self.f_hts * self.tape.density() + self.f_cu * rho_cu
                + self.f_insulation * INSULATION_DENSITY)

    def strain_at_stress(self, stress: float) -> float:
        """Iso-strain hoop strain ``eps = sigma / E_c`` under a winding-average stress [-].

        ``stress`` in Pa (signed).

        Reference: Voigt rule of mixtures.
        """
        return _finite("stress", stress) / self.modulus()

    def strain_limit(self) -> float:
        """Allowed tensile strain ``min(tape.irreversible_strain, cns elongation)`` [-].

        0.004 for every carbon reinforcement in the registry (REBCO limits).

        Reference: Barth et al. (2015).
        """
        elong = self.cns.elongation
        lim = self.tape.irreversible_strain
        return min(lim, elong) if elong is not None and elong > 0.0 else lim

    def allowable_stress(self, safety_factor: float = 1.0) -> float:
        """Largest winding-average stress at the strain limit, ``E_c eps_lim / SF`` [Pa].

        ``safety_factor`` > 0.  Default composite: 0.84 GPa at SF = 1 (the
        concept's 1.6-4 GPa magnetic pressures are 2-5x beyond it).

        Reference: Voigt rule of mixtures.
        """
        sf = _positive("safety_factor", safety_factor)
        return self.modulus() * self.strain_limit() / sf

    def reinforcement_stress(self, stress: float) -> float:
        """Stress carried by the reinforcement at the composite strain, ``E_cns sigma / E_c`` [Pa].

        ``stress`` in Pa (winding average).  Compare with the reinforcement's strength.

        Reference: Voigt rule of mixtures.
        """
        return self.cns.youngs_modulus * _finite("stress", stress) / self.modulus()


def required_composite_modulus(hoop_stress: float, eps_allow: float = 0.004, safety_factor: float = 1.0) -> float:
    """Composite modulus that keeps the hoop strain at the allowed value, ``E = SF sigma / eps`` [Pa].

    ``hoop_stress`` in Pa (>= 0), ``eps_allow`` (> 0, default REBCO's 0.4 %),
    ``safety_factor`` > 0.  At 4 GPa and 0.4 % this is 1 TPa, graphene's intrinsic
    modulus.

    Reference: Voigt rule of mixtures; Lee et al. (2008).
    """
    s = _non_negative("hoop_stress", hoop_stress)
    e = _positive("eps_allow", eps_allow)
    sf = _positive("safety_factor", safety_factor)
    return sf * s / e


def required_cns_fraction(hoop_stress: float, e_cns: float, e_other: float, eps_allow: float = 0.004,
                          safety_factor: float = 1.0) -> float:
    """Reinforcement volume fraction needed to reach the required modulus (not clipped).

    ``f = (SF sigma/eps - E_other) / (E_cns - E_other)`` with stresses/moduli in Pa.
    Values > 1 mean no fraction of this reinforcement suffices (callers map that
    to infeasible); negative values mean the other constituents alone suffice.
    If ``e_cns <= e_other`` adding reinforcement cannot raise the modulus: returns
    0.0 when ``E_other`` already suffices, else ``math.inf``.

    Reference: Voigt rule of mixtures.
    """
    e_req = required_composite_modulus(hoop_stress, eps_allow, safety_factor)
    ec = _positive("e_cns", e_cns)
    eo = _non_negative("e_other", e_other)
    if ec <= eo:
        return 0.0 if e_req <= eo else math.inf
    return (e_req - eo) / (ec - eo)


# ---------------------------------------------------------------------------
# conductor assessment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class ConductorAssessment(Assessment):
    """Je margin of a REBCO winding: body (B in the tape plane) and coil ends (B perpendicular).

    ``j_allow_* = f_hts Je_*`` is the winding-average current density the tape
    fraction can carry; ``margin_* = j_allow_* / j_required``; ``margin`` is the
    smaller of the two.  ``je_data_status`` labels the Je model's data basis.

    Reference: Iwasa (2009); Tsuchiya et al. (2021).
    """

    field: float = unit("T")
    temperature: float = unit("K")
    j_required: float = unit("A/m^2")
    je_parallel: float = unit("A/m^2")
    je_perpendicular: float = unit("A/m^2")
    j_allow_parallel: float = unit("A/m^2")
    j_allow_perpendicular: float = unit("A/m^2")
    margin_parallel: float
    margin_perpendicular: float
    end_field_fraction: float
    je_data_status: Status


def assess_conductor(field: float, temperature: float, winding_current_density: float,
                     composite: CNSComposite | None = None, safety_factor: float = 1.0,
                     end_field_fraction: float = DEFAULT_END_FIELD_FRACTION) -> ConductorAssessment:
    """Assess whether the composite's REBCO fraction carries the winding current at ``field``.

    Body: ``Je(B, T, parallel)``; coil ends: ``Je(f_end B, T, perpendicular)``,
    where ``end_field_fraction`` f_end (0.3-0.5 typical for solenoids) is the radial
    field component at the coil ends.  ``j_allow = f_hts Je``;
    ``margin = min(j_allow/J)``.  Inputs: ``field`` [T] >= 0 (peak field on the
    conductor), ``temperature`` [K] > 0, ``winding_current_density`` J [A/m^2] > 0
    (winding average), ``safety_factor`` >= 1, ``end_field_fraction`` in [0, 1].

    Status: margin >= SF -> state of the art; 1 <= margin < SF -> extrapolation;
    margin < 1 -> aspirational ("needs a conductor that does not exist"); then
    worst-combined with :func:`je_data_status` (field).  The magnet-level field
    status (``benchmarks.superconducting_field_status``) is a separate axis and is
    only noted.

    Reference: Iwasa (2009); Tsuchiya et al. (2021); Hahn et al. (2019).
    """
    b = _non_negative("field", field)
    t = _positive("temperature", temperature)
    j = _positive("winding_current_density", winding_current_density)
    sf = _safety_factor(safety_factor)
    f_end = _fraction("end_field_fraction", end_field_fraction)
    comp = composite if composite is not None else CNSComposite.default()
    tape = comp.tape

    je_par = tape.je(b, t, "parallel")
    je_perp = tape.je(f_end * b, t, "perpendicular")
    ja_par = comp.f_hts * je_par
    ja_perp = comp.f_hts * je_perp
    m_par = ja_par / j
    m_perp = ja_perp / j
    margin = min(m_par, m_perp)
    data_status = je_data_status(b)

    warnings: list[str] = []
    if margin >= sf:
        m_status = Status.STATE_OF_THE_ART
    elif margin >= 1.0:
        m_status = Status.EXTRAPOLATION
        warnings.append(f"Je margin {margin:.3g} is below the safety factor {sf:g}")
    else:
        m_status = Status.ASPIRATIONAL
        warnings.append(f"winding current density {j:.3g} A/m^2 exceeds f_hts*Je ({min(ja_par, ja_perp):.3g} A/m^2): "
                        "needs a conductor that does not exist")
    if t >= tape.tc:
        warnings.append(f"temperature {t:g} K is at or above Tc = {tape.tc:g} K: no superconducting current")
    limiting = "coil ends (perpendicular)" if m_perp < m_par else "coil body (parallel)"
    notes = (
        f"body: B parallel to the tape at {b:.4g} T; ends: perpendicular component {f_end:g} x B = {f_end * b:.4g} T "
        "(radial field at the coil ends, 0.3-0.5 typical for solenoids)",
        f"allowable winding J = f_hts x Je with f_hts = {comp.f_hts:g}; limiting region: {limiting}",
        _je_data_note(b),
        f"magnet-level field status (separate axis): {superconducting_field_status(b)}",
        "Je model is a parametric fit representative of 2020s commercial REBCO, not a datasheet",
    )
    return ConductorAssessment(
        status=worst(m_status, data_status),
        margin=margin,
        warnings=tuple(warnings),
        notes=notes,
        field=b,
        temperature=t,
        j_required=j,
        je_parallel=je_par,
        je_perpendicular=je_perp,
        j_allow_parallel=ja_par,
        j_allow_perpendicular=ja_perp,
        margin_parallel=m_par,
        margin_perpendicular=m_perp,
        end_field_fraction=f_end,
        je_data_status=data_status,
    )


# ---------------------------------------------------------------------------
# structure assessment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class StructureAssessment(Assessment):
    """Strain-limited structural check of a CNS-reinforced winding at a given hoop stress.

    ``strain = sigma/E_c`` for the composite as specified;
    ``required_composite_modulus = SF sigma/eps_lim``; ``required_reinforcement_modulus``
    is the reinforcement modulus that would meet it at the composite's f_cns (0 if the
    other constituents suffice); ``required_cns_fraction`` the fraction of the
    composite's own reinforcement that would.  ``benchmark`` maps each reference
    reinforcement to its check at the same f_cns (keys: tier, modulus, strength,
    composite_modulus, strain, strain_limit, reinforcement_stress, modulus_ok,
    strength_ok, meets, required_cns_fraction).  ``margin`` is the specified
    composite's min(strain margin, reinforcement strength margin).

    Reference: Voigt rule of mixtures; Barth et al. (2015).
    """

    hoop_stress: float = unit("Pa")
    composite_modulus: float = unit("Pa")
    strain: float
    strain_limit: float
    required_composite_modulus: float = unit("Pa")
    required_reinforcement_modulus: float = unit("Pa")
    required_cns_fraction: float
    reinforcement_stress: float = unit("Pa")
    reinforcement_strength: float | None = unit("Pa")
    cns_material: str
    benchmark: dict[str, dict]


def _reinforcement_check(hoop_stress: float, mat: Material, comp: CNSComposite, sf: float,
                         rest_sum: float, e_matrix: float) -> dict:
    e_m = mat.youngs_modulus
    e_c = comp.f_cns * e_m + rest_sum
    strain = hoop_stress / e_c if e_c > 0.0 else math.inf
    elong = mat.elongation
    lim = min(comp.tape.irreversible_strain, elong) if elong is not None and elong > 0.0 else comp.tape.irreversible_strain
    stress_m = e_m * strain if math.isfinite(strain) else math.inf
    strength = mat.ultimate_strength
    modulus_ok = sf * strain <= lim
    strength_ok = strength is not None and sf * stress_m <= strength
    return {
        "modulus": e_m,
        "strength": strength,
        "composite_modulus": e_c,
        "strain": strain,
        "strain_limit": lim,
        "reinforcement_stress": stress_m,
        "modulus_ok": modulus_ok,
        "strength_ok": strength_ok,
        "meets": modulus_ok and strength_ok,
        "required_cns_fraction": required_cns_fraction(hoop_stress, e_m, e_matrix, lim, sf),
    }


def assess_structure(hoop_stress: float, composite: CNSComposite | None = None,
                     safety_factor: float = 1.5) -> StructureAssessment:
    """Can any reinforcement keep a REBCO winding below its strain limit at ``hoop_stress``?

    Iso-strain composite: ``E_c = f_cns E_r + sum_others f_i E_i``; a reinforcement
    meets the need if ``SF sigma/E_c <= min(0.4 %, elongation_r)`` (stiffness) and
    ``SF E_r sigma/E_c <= strength_r`` (strength), at the composite's f_cns.
    ``hoop_stress`` [Pa] >= 0 (use the peak, e.g. ``Solenoid.max_hoop_stress``),
    ``safety_factor`` >= 1 (default 1.5).

    Status ladder: met by a commercial fibre (M60J or T1100G) -> state of the art;
    by the laboratory-record CNT fibre -> extrapolation; by the CNS engineering
    target or intrinsic graphene -> aspirational; otherwise infeasible (including
    when even intrinsic graphene would need f_cns > 1).  At 4 GPa and 0.4 % the
    composite needs E >= 1 TPa (graphene's intrinsic modulus) unless the coil is
    thick, and sigma_max >= P/2 for any thickness
    (:func:`magform.magnetics.max_hoop_stress_ratio`), so 4 GPa coils need >= 0.5 TPa
    even before a safety factor.

    Reference: Voigt rule of mixtures; Barth et al. (2015); Toray data sheets;
    Lee et al. (2008, 2022).
    """
    s = _non_negative("hoop_stress", hoop_stress)
    sf = _safety_factor(safety_factor)
    comp = composite if composite is not None else CNSComposite.default()
    cns = comp.cns
    e_c = comp.modulus()
    eps_lim = comp.strain_limit()
    strain = s / e_c
    rest_sum = comp._rest_modulus_sum()
    e_matrix = comp.matrix_modulus()
    e_req = required_composite_modulus(s, eps_lim, sf)
    if comp.f_cns > 0.0:
        e_r_req = max((e_req - rest_sum) / comp.f_cns, 0.0)
    else:
        e_r_req = 0.0 if e_req <= rest_sum else math.inf
    f_req = required_cns_fraction(s, cns.youngs_modulus, e_matrix, eps_lim, sf)
    r_stress = comp.reinforcement_stress(s)
    r_strength = cns.ultimate_strength

    tiers = {key: status for status, keys in STRUCTURE_LADDER for key in keys}
    bench: dict[str, dict] = {}
    for key in STRUCTURE_BENCHMARKS:
        entry = _reinforcement_check(s, as_material(key), comp, sf, rest_sum, e_matrix)
        tier = tiers.get(key)
        bench[key] = {"tier": tier.value if tier is not None else "reference", **entry}

    status = Status.INFEASIBLE
    met_by: str | None = None
    for tier_status, keys in STRUCTURE_LADDER:
        winners = [k for k in keys if bench[k]["meets"]]
        if winners:
            status, met_by = tier_status, winners[0]
            break

    strain_margin = (e_c * eps_lim) / (sf * s) if s > 0.0 else math.inf
    if s == 0.0:
        strength_margin = math.inf
    elif r_strength is None:
        strength_margin = 0.0
    else:
        strength_margin = r_strength / (sf * r_stress)
    margin = min(strain_margin, strength_margin)

    warnings: list[str] = []
    notes: list[str] = [
        "iso-strain rule of mixtures: all constituents share the hoop strain; the REBCO irreversible strain "
        f"({comp.tape.irreversible_strain:.2%}) limits every reinforcement, so stiffness, not strength, binds",
        f"required composite modulus {e_req / 1e9:.4g} GPa (SF {sf:g}); at 4 GPa and 0.4 % strain E >= 1 TPa is "
        "needed (graphene's intrinsic modulus); sigma_max >= P/2 for any winding thickness "
        "(magnetics.max_hoop_stress_ratio)",
    ]
    if met_by is not None:
        notes.append(f"requirement met at f_cns = {comp.f_cns:g} by {met_by} ({status.label})")
    graphene = bench["graphene_intrinsic"]
    if status is Status.INFEASIBLE:
        f_g = graphene["required_cns_fraction"]
        if f_g > 1.0:
            notes.append(f"no reinforcement fraction suffices even with intrinsic graphene "
                         f"(required f_cns = {f_g:.3g} > 1)")
        else:
            notes.append(f"intrinsic graphene would need f_cns >= {f_g:.3g} (leaving <= {1.0 - f_g:.3g} for conductor, "
                         "copper and insulation)")
        warnings.append("no existing or hypothetical reinforcement meets the strain limit at this fraction")
    if margin < 1.0:
        warnings.append(f"the specified reinforcement '{comp.cns.key}' at f_cns = {comp.f_cns:g} does not meet the "
                        f"requirement (margin {margin:.3g}); the status refers to the best reinforcement class that does")
    target = as_material("cns_engineering_target")
    m60j = as_material("carbon_fibre_m60j")
    if cns.key == target.key and target.youngs_modulus < m60j.youngs_modulus:
        notes.append(f"the CNS engineering target ({target.youngs_modulus / 1e9:.0f} GPa) is less stiff than "
                     f"commercial M60J ({m60j.youngs_modulus / 1e9:.0f} GPa): it adds strength, not the stiffness "
                     "the strain limit needs")
    return StructureAssessment(
        status=status,
        margin=margin,
        warnings=tuple(warnings),
        notes=tuple(notes),
        hoop_stress=s,
        composite_modulus=e_c,
        strain=strain,
        strain_limit=eps_lim,
        required_composite_modulus=e_req,
        required_reinforcement_modulus=e_r_req,
        required_cns_fraction=f_req,
        reinforcement_stress=r_stress,
        reinforcement_strength=r_strength,
        cns_material=cns.key,
        benchmark=bench,
    )


def je_limited_graded_thickness(central_field: float, temperature: float = 4.2,
                                composite: CNSComposite | None = None, orientation: str = "parallel",
                                n: int = 2000) -> float:
    """Minimum (current-limited) winding thickness of an ideally graded REBCO solenoid [m].

    ``Delta = integral_0^B dB' / (mu0 f_hts Je(B', T, o))`` via
    :func:`magform.magnetics.graded_winding_thickness`; ``central_field`` [T] >= 0,
    ``temperature`` [K] > 0, ``orientation`` "parallel" (body) or "perpendicular"
    (worst case everywhere).  ``math.inf`` if Je reaches 0 below the field (e.g.
    above Birr).  Stress is not considered: the structural limit usually needs a
    thicker coil (:func:`assess_structure`).

    Reference: Iwasa (2009); Wilson (1983).
    """
    _non_negative("central_field", central_field)
    t = _positive("temperature", temperature)
    o = _orientation(orientation)
    comp = composite if composite is not None else CNSComposite.default()
    f, tape = comp.f_hts, comp.tape
    return graded_winding_thickness(central_field, lambda b: f * tape.je(b, t, o), n=n)


# ---------------------------------------------------------------------------
# pulsed operation: AC loss versus enthalpy margin
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class PulseLoss(ResultMixin):
    """Per-pulse dissipation in a REBCO tape for a unipolar field pulse 0 -> dB -> 0.

    All ``q`` in J per m^3 of tape per pulse: critical-state hysteresis with the
    field perpendicular to the tape (coil ends) and parallel to it (body), eddy
    loss in the copper stabiliser, and ``q_total`` weighted by
    ``perpendicular_fraction``.  ``copper_resistivity`` is the magnetoresistive
    rho_Cu used (ohm m).

    Reference: Wilson (1983) ch. 8; Bean (1964); Brandt & Indenbom (1993).
    """

    field_swing: float = unit("T")
    rise_time: float = unit("s")
    ramp_rate: float = unit("T/s")
    q_hysteresis_perpendicular: float = unit("J/m^3")
    q_hysteresis_parallel: float = unit("J/m^3")
    q_eddy_copper: float = unit("J/m^3")
    q_total: float = unit("J/m^3")
    perpendicular_fraction: float
    copper_resistivity: float = unit("ohm m")


def _strip_loss_function(x: float) -> float:
    """Brandt-Indenbom thin-strip loss function g(x) = (2/x) ln cosh x - tanh x (-> 1 for x >> 1)."""
    if x <= 0.0:
        return 0.0
    if math.isinf(x):
        return 1.0
    if x < 1e-3:
        return x ** 3 / 6.0
    ln_cosh = x + math.log1p(math.exp(-2.0 * x)) - math.log(2.0)
    return (2.0 / x) * ln_cosh - math.tanh(x)


def _bean_slab_loss(bm: float, bp: float) -> float:
    """Bean slab hysteresis loss per cycle per unit volume (Wilson 1983), J/m^3."""
    if bm <= 0.0:
        return 0.0
    if bp > 0.0 and bm <= bp:
        return 2.0 * bm ** 3 / (3.0 * MU0 * bp)
    return (2.0 * bm * bp / MU0) * (1.0 - 2.0 * bp / (3.0 * bm))


def tape_pulse_losses(field_swing: float, rise_time: float, temperature: float = 4.2,
                      tape: REBCOTape | None = None,
                      perpendicular_fraction: float = DEFAULT_END_FIELD_FRACTION) -> PulseLoss:
    """Energy dissipated per unit tape volume by one unipolar pulse 0 -> dB -> 0 [J/m^3].

    A unipolar excursion is one critical-state cycle of amplitude ``Bm = dB/2``.

    * Perpendicular (coil ends; thin strip of width w, REBCO thickness d,
      ``dB_perp = perpendicular_fraction * dB``, Jc = ``layer_jc(dB_perp/2, perpendicular)``):
      ``q_SC = Jc (w/2) dB_perp g(x)``, ``x = (dB_perp/2)/(mu0 Jc d/pi)``,
      ``g(x) = (2/x) ln cosh x - tanh x`` (Brandt & Indenbom 1993).  For
      dB_perp >> mu0 Jc d/pi (~0.1 T) g -> 1, the Norris (1970) high-field limit
      ``Jc (w/2) dB_perp``.
    * Parallel (coil body; Bean slab of half-thickness a = d/2, ``Bp = mu0 Jc a``,
      Jc = ``layer_jc(dB/2, parallel)``, Bm = dB/2):
      ``q_SC = 2 Bm^3/(3 mu0 Bp)`` (Bm <= Bp), else ``(2 Bm Bp/mu0)(1 - 2 Bp/(3 Bm))``
      (Wilson 1983).
    * Both are averaged over the tape: ``x rebco_thickness/thickness``.
    * Copper eddy loss (field parallel to the 20 um layers): ``q = (dB/dt)^2 d_Cu^2
      (2 t_rise)/(12 rho_Cu) x copper_thickness/thickness`` with
      ``rho_Cu = rho_Cu(T) (1 + (dB/2)/10 T)`` (magnetoresistance, estimated).
    * ``q_total = f q_perp + (1 - f) q_par + q_eddy`` with f = ``perpendicular_fraction``:
      a fraction f of the winding (the ends) sees a perpendicular component f*dB,
      the rest sees the full swing in the tape plane.  This is a bracketing
      weighting, not a field map; the perpendicular-field eddy loss of the copper
      (which scales with w^2) is ignored, so q_total is a lower estimate.

    Inputs: ``field_swing`` dB [T] > 0, ``rise_time`` [s] > 0, ``temperature`` [K] > 0,
    ``perpendicular_fraction`` in [0, 1].  Adiabatic, uniform-field, no transport
    current (transport current increases the loss).

    Reference: Bean (1964); Norris (1970); Brandt & Indenbom (1993); Wilson (1983) ch. 8.
    """
    db = _positive("field_swing", field_swing)
    tr = _positive("rise_time", rise_time)
    t = _positive("temperature", temperature)
    f = _fraction("perpendicular_fraction", perpendicular_fraction)
    tp = tape if tape is not None else REBCOTape()
    sc_fraction = tp.rebco_thickness / tp.thickness
    ramp = db / tr

    # perpendicular: thin strip (Brandt & Indenbom)
    db_perp = f * db
    q_perp = 0.0
    if db_perp > 0.0:
        jc = tp.layer_jc(0.5 * db_perp, t, "perpendicular")
        if jc > 0.0:
            b_d = MU0 * jc * tp.rebco_thickness / math.pi
            x = 0.5 * db_perp / b_d
            q_perp = jc * 0.5 * tp.width * db_perp * _strip_loss_function(x) * sc_fraction

    # parallel: Bean slab of half-thickness rebco/2
    bm = 0.5 * db
    jc_par = tp.layer_jc(bm, t, "parallel")
    bp = MU0 * jc_par * 0.5 * tp.rebco_thickness
    q_par = _bean_slab_loss(bm, bp) * sc_fraction

    # copper stabiliser eddy loss
    rho = as_material("copper_ofhc").resistivity_at(t) * (1.0 + bm / CU_MAGNETORESISTANCE_FIELD)
    d = tp.copper_layer_thickness
    q_eddy = ramp ** 2 * d ** 2 * (2.0 * tr) / (12.0 * rho) * (tp.copper_thickness / tp.thickness)

    q_total = f * q_perp + (1.0 - f) * q_par + q_eddy
    return PulseLoss(
        field_swing=db,
        rise_time=tr,
        ramp_rate=ramp,
        q_hysteresis_perpendicular=q_perp,
        q_hysteresis_parallel=q_par,
        q_eddy_copper=q_eddy,
        q_total=q_total,
        perpendicular_fraction=f,
        copper_resistivity=rho,
    )


def enthalpy_margin(temperature_from: float, temperature_to: float, tape: REBCOTape | None = None) -> float:
    """Heat per unit tape volume to warm the tape from ``temperature_from`` to ``temperature_to`` [J/m^3].

    ``H = f_Cu rho_Cu int cp_Cu dT + f_sub rho_Hast int cp_Hast dT`` with the copper
    and substrate volume fractions of the tape (the REBCO, silver and buffer
    layers, ~8 % of the default tape, are neglected: conservative).  cp(T) from
    :data:`CU_SPECIFIC_HEAT` and :data:`HASTELLOY_SPECIFIC_HEAT` with linear
    interpolation (cp ~ T below 4 K); trapezoid on a grid that contains every table
    point, so the integral of the tabulated model is exact.  Temperatures in K,
    0 < from <= to <= 300 K.  Adiabatic: no helium or conduction cooling.
    Default tape, 4.2 -> 20 K: 2.8e5 J/m^3; 4.2 -> 77 K: 3.3e7 J/m^3.

    Reference: NIST OFHC copper data; Ekin (2006); Lu, Choi & Zhou (2008); Iwasa (2009).
    """
    t0 = _positive("temperature_from", temperature_from)
    t1 = _positive("temperature_to", temperature_to)
    if t1 < t0:
        raise InputError(f"temperature_to ({t1!r} K) must not be below temperature_from ({t0!r} K)")
    if t1 > TABLE_T_MAX:
        raise InputError(f"temperature_to must be <= {TABLE_T_MAX:g} K (end of the specific-heat tables)")
    tp = tape if tape is not None else REBCOTape()
    if t1 == t0:
        return 0.0
    knots = {p[0] for p in CU_SPECIFIC_HEAT} | {p[0] for p in HASTELLOY_SPECIFIC_HEAT}
    grid = sorted(set(linspace(t0, t1, 401)) | {k for k in knots if t0 < k < t1})
    rho_cu = as_material("copper_ofhc").density
    rho_sub = as_material("hastelloy_c276").density
    f_cu = tp.copper_thickness / tp.thickness
    f_sub = tp.substrate_thickness / tp.thickness
    vol_cp = [f_cu * rho_cu * _cp(CU_SPECIFIC_HEAT, x) + f_sub * rho_sub * _cp(HASTELLOY_SPECIFIC_HEAT, x)
              for x in grid]
    return trapz(vol_cp, grid)


@dataclass(frozen=True, kw_only=True)
class QuenchAssessment(Assessment):
    """Per-pulse AC loss of a pulsed REBCO winding against its adiabatic enthalpy margin.

    ``loss_to_margin_ratio = loss.q_total / enthalpy_margin`` (inf when there is no
    margin); ``margin`` is its inverse.  ``ramp_rate_status`` is the cited
    benchmark for sweeping superconducting magnets.

    Reference: Wilson (1983); Iwasa (2009).
    """

    field_swing: float = unit("T")
    rise_time: float = unit("s")
    ramp_rate: float = unit("T/s")
    ramp_rate_status: Status
    loss: PulseLoss
    operating_temperature: float = unit("K")
    current_sharing_temperature: float = unit("K")
    enthalpy_margin: float = unit("J/m^3")
    loss_to_margin_ratio: float


def assess_pulsed_operation(field_swing: float, rise_time: float, j_operating: float, temperature: float = 4.2,
                            composite: CNSComposite | None = None,
                            perpendicular_fraction: float = DEFAULT_END_FIELD_FRACTION) -> QuenchAssessment:
    """Can a REBCO winding be pulsed 0 -> dB -> 0 without quenching?

    Loss: :func:`tape_pulse_losses`.  Margin: :func:`enthalpy_margin` from the
    operating temperature to the current-sharing temperature at the peak field
    (parallel orientation, tape current density ``j_operating / f_hts``).
    ``ratio = q_total / H``.  Inputs: ``field_swing`` [T] > 0 (peak field, starting
    from zero), ``rise_time`` [s] > 0, ``j_operating`` [A/m^2] > 0 winding average,
    ``temperature`` [K] > 0, ``perpendicular_fraction`` in [0, 1].

    Status: the adiabatic comparison applies only when ``rise_time`` <= 1 s
    (:data:`ADIABATIC_RISE_TIME`, shorter than the winding's thermal time constant);
    for slower ramps the ratio is information and the status is
    ``benchmarks.ramp_rate_status``.  Otherwise ratio > 1 -> infeasible ("quenches
    every pulse"); ratio > 0.3 -> worst(extrapolation, ramp status); else ramp
    status.  If the tape cannot carry the current at the peak field even in DC
    (T_cs <= T_op) the status is at least aspirational.  Example: 100 T in 25 us
    is infeasible by orders of magnitude; the consistent architecture is a DC
    superconducting background plus a pulsed resistive insert
    (:func:`magform.magnetics.pressure_with_background`).

    Reference: Wilson (1983) ch. 8; Iwasa (2009); Herlach & Miura (2003).
    """
    db = _positive("field_swing", field_swing)
    tr = _positive("rise_time", rise_time)
    j = _positive("j_operating", j_operating)
    t_op = _positive("temperature", temperature)
    f = _fraction("perpendicular_fraction", perpendicular_fraction)
    comp = composite if composite is not None else CNSComposite.default()
    if comp.f_hts <= 0.0:
        raise InputError("composite has no superconductor (f_hts = 0)")
    tape = comp.tape
    if t_op >= tape.tc:
        raise InputError(f"operating temperature {t_op:g} K is not below Tc = {tape.tc:g} K")

    loss = tape_pulse_losses(db, tr, t_op, tape, f)
    j_tape = j / comp.f_hts
    t_cs = tape.current_sharing_temperature(db, j_tape, "parallel",
                                            lowest_temperature=min(T_LIQUID_HELIUM, t_op))
    h = enthalpy_margin(t_op, t_cs, tape) if t_cs > t_op else 0.0
    q = loss.q_total
    if h > 0.0:
        ratio = q / h
    else:
        ratio = math.inf if q > 0.0 else 0.0
    margin = h / q if q > 0.0 else math.inf
    ramp = db / tr
    r_status = ramp_rate_status(ramp)

    warnings: list[str] = []
    notes: list[str] = [
        f"per-pulse loss {q:.3g} J/m^3 of tape (hysteresis perpendicular {loss.q_hysteresis_perpendicular:.3g}, "
        f"parallel {loss.q_hysteresis_parallel:.3g}, copper eddy {loss.q_eddy_copper:.3g}; weighting f = {f:g})",
        f"enthalpy margin {h:.3g} J/m^3 from {t_op:g} K to T_cs = {t_cs:.4g} K (adiabatic, Cu + Hastelloy only)",
        "a static field exerts no force on a stationary workpiece, so a forming magnet must swing its field; "
        "the consistent architecture is a DC superconducting background plus a pulsed resistive insert "
        "(magnetics.pressure_with_background)",
    ]
    if tr > ADIABATIC_RISE_TIME:
        status = r_status
        notes.append(f"rise time {tr:g} s > {ADIABATIC_RISE_TIME:g} s: heat leaves the winding during the ramp, so the "
                     f"adiabatic ratio {ratio:.3g} is information only; status from the ramp-rate benchmark")
    elif ratio > 1.0:
        status = Status.INFEASIBLE
        warnings.append(f"quenches every pulse: loss/margin = {ratio:.3g} (> 1)")
    elif ratio > LOSS_MARGIN_CAUTION:
        status = worst(Status.EXTRAPOLATION, r_status)
        warnings.append(f"loss uses {ratio:.0%} of the enthalpy margin (> {LOSS_MARGIN_CAUTION:.0%})")
    else:
        status = r_status
    if t_cs <= t_op:
        status = worst(status, Status.ASPIRATIONAL)
        warnings.append(f"tape current density {j_tape:.3g} A/m^2 exceeds Je at {db:g} T and {t_op:g} K: no temperature "
                        "margin even without pulsing")
    if r_status is not Status.STATE_OF_THE_ART:
        notes.append(f"ramp rate {ramp:.3g} T/s is beyond routinely swept superconducting magnets ({r_status.label})")
    return QuenchAssessment(
        status=status,
        margin=margin,
        warnings=tuple(warnings),
        notes=tuple(notes),
        field_swing=db,
        rise_time=tr,
        ramp_rate=ramp,
        ramp_rate_status=r_status,
        loss=loss,
        operating_temperature=t_op,
        current_sharing_temperature=t_cs,
        enthalpy_margin=h,
        loss_to_margin_ratio=ratio,
    )


def terminal_voltage(stored_energy: float, peak_current: float, rise_time: float) -> float:
    """Inductive terminal voltage to ramp a magnet to ``stored_energy`` in ``rise_time``, ``V = 2E/(I t)`` [V].

    From ``V = L dI/dt ~ L I/t`` with ``L = 2E/I^2`` (linear ramp).  ``stored_energy``
    [J] >= 0, ``peak_current`` [A] > 0, ``rise_time`` [s] > 0.  Compare with
    ``benchmarks.sc_voltage_status`` (superconducting) or ``pulsed_voltage_status``.

    Reference: Wilson (1983); Iwasa (2009).
    """
    e = _non_negative("stored_energy", stored_energy)
    i = _positive("peak_current", peak_current)
    t = _positive("rise_time", rise_time)
    return 2.0 * e / (i * t)


@dataclass(frozen=True, kw_only=True)
class ReinforcementEddy(ResultMixin):
    """Eddy-current heating of the (electrically conducting) carbon reinforcement during a pulse.

    ``loss_per_pulse_density`` in J per m^3 of reinforcement per unipolar pulse;
    ``thick`` is True when the lamina is thicker than the skin depth (the field
    does not fully penetrate and the loss estimate uses d = skin depth).

    Reference: Knoepfel (2000); Wilson (1983).
    """

    material: str
    frequency: float = unit("Hz")
    skin_depth: float = unit("m")
    lamina_thickness: float = unit("m")
    thick: bool
    loss_per_pulse_density: float = unit("J/m^3")


def cns_eddy_check(composite: CNSComposite | None, field_swing: float, rise_time: float,
                   lamina_thickness: float = 1e-3) -> ReinforcementEddy:
    """Eddy loss in the carbon reinforcement, which is itself a conductor.

    Skin depth ``delta = sqrt(rho/(pi f mu0))`` with ``f = 1/(4 t_rise)``; loss per
    pulse ``q = (dB/dt)^2 d^2 (2 t_rise)/(12 rho)`` with ``d = min(lamina, delta)``
    (thin-slab formula, rise and fall).  ``composite`` (None -> default),
    ``field_swing`` [T] > 0, ``rise_time`` [s] > 0, ``lamina_thickness`` [m] > 0;
    rho is the reinforcement's room-temperature resistivity (carbon's varies
    weakly with temperature).  Example: the CNS target (1e-6 ohm m) in 1 mm laminae
    swept 100 T in 25 us absorbs ~7e7 J/m^3 per pulse.

    Reference: Knoepfel (2000); Wilson (1983) ch. 8.
    """
    comp = composite if composite is not None else CNSComposite.default()
    db = _positive("field_swing", field_swing)
    tr = _positive("rise_time", rise_time)
    lam = _positive("lamina_thickness", lamina_thickness)
    mat = comp.cns
    rho = mat.resistivity
    freq = 1.0 / (4.0 * tr)
    delta = math.sqrt(rho / (math.pi * freq * MU0))
    d = min(lam, delta)
    q = (db / tr) ** 2 * d ** 2 * (2.0 * tr) / (12.0 * rho)
    return ReinforcementEddy(
        material=mat.key,
        frequency=freq,
        skin_depth=delta,
        lamina_thickness=lam,
        thick=lam > delta,
        loss_per_pulse_density=q,
    )


__all__ = [
    "REFERENCES", "ORIENTATIONS", "JE_MEASURED_FIELD", "DEFAULT_END_FIELD_FRACTION", "ADIABATIC_RISE_TIME",
    "LOSS_MARGIN_CAUTION", "CU_MAGNETORESISTANCE_FIELD", "INSULATION_DENSITY", "REBCO_DENSITY",
    "SILVER_DENSITY", "TABLE_T_MAX", "CU_SPECIFIC_HEAT", "HASTELLOY_SPECIFIC_HEAT", "STRUCTURE_BENCHMARKS",
    "STRUCTURE_LADDER",
    "REBCOTape", "je_data_status", "CNSComposite", "required_composite_modulus", "required_cns_fraction",
    "ConductorAssessment", "assess_conductor", "StructureAssessment", "assess_structure",
    "je_limited_graded_thickness", "PulseLoss", "tape_pulse_losses", "enthalpy_margin", "QuenchAssessment",
    "assess_pulsed_operation", "terminal_voltage", "ReinforcementEddy", "cns_eddy_check",
]
