"""Magnetar-class fields on demand: a thought experiment granted by premise.

Everything else in magform asks whether a 63-100 T press can be built and what it
would do.  This module answers a different question, posed as an assumption:

    "Assume an exotic mode of operation where magnetar-style fields are applied
    in a space-based setting on demand."

The source is **not assessed**.  It is granted, and :func:`assess_magnetar_mode`
says so with ``Status.ASSUMED`` (a premise, not a judgement).  What the module
does compute, with established physics, is everything that follows from the
premise:

1. **How far the premise is from the record book** (:data:`EXTREME_FIELD_LADDER`,
   :func:`laboratory_record`).  Magnetars carry 1e10-1e11 T.  The strongest field
   made without destroying the magnet is 100.75 T; the strongest made at all in
   a macroscopic volume is 2800 T for microseconds.  Stronger human-made fields
   exist only where they are tiny and fleeting: ~7e4 T over micrometres for
   picoseconds in laser plasmas, ~3e14 T over femtometres for 1e-23 s in
   heavy-ion collisions.
2. **What a shot costs** (:class:`MagnetarSource`).  The energy density at
   1e10 T is 4e25 J/m^3, a mass-equivalent of 4.4e8 kg/m^3: one cubic
   millimetre holds 9.5 megatons, one cubic micrometre 40 MJ.  Nothing holds
   such a field: the virial theorem asks for about 10 kg of the best carbon
   fibre for every cubic micrometre of it, and the only static holder nature
   provides is a neutron star (:func:`chandrasekhar_fermi_field`).  Unheld, the
   field disperses in the light-crossing time, so "on demand" means assembling
   megatons in picoseconds.
3. **What such fields do to matter** (:func:`classify_regime`).  As a piston the
   pressure crushes cold matter to white-dwarf densities
   (:func:`piston_state`, Chandrasekhar's equation of state) and, at
   1.1e10 T, forces electrons into iron nuclei (:func:`electron_capture_threshold`).
   Where the field threads matter instead, atoms above 2.35e5 T become needles
   and ordinary chemistry stops (:func:`atom_shape`, :func:`condensed_density`).
4. **What the concept note's six processes become**, and why the mode has to be
   space-based: a 10 mm, 1e10 T source has a 0.5 mT line at 270 m, pushes water
   away harder than gravity inside 6 m, and releases 6e4 megatons per shot
   (:func:`standoff_radii`, :func:`levitation_radius`, :func:`fluence_radius`).

Validity: cold (zero-temperature) matter; classical field energy ``B^2/(2 mu0)``,
whose vacuum-polarisation correction is ``(alpha/45 pi)(B/B_Q)^2`` for B << B_Q
and stays below 1 % up to 1e11 T (Heisenberg & Euler 1936; Schwinger 1951); the
degenerate equation of state is offered only where it is within ~25 % (see
:func:`piston_state`); atomic and condensed-matter scalings in strong fields are
asymptotic, order-of-magnitude results (Lai 2001).  SI units throughout:
fields in T, pressures in Pa, energies in J, densities in kg/m^3.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import ClassVar, Sequence

from . import benchmarks
from ._numerics import bisect
from ._report import Assessment, ResultMixin, unit
from .constants import C_LIGHT, E_CHARGE, EPS0, G_ACCEL, MU0, T_ROOM
from .eddy import skin_melting_field
from .errors import InputError
from .magnetics import field_for_pressure, magnetic_pressure, virial_minimum_mass
from .materials import Material, MaterialLike, as_material
from .status import Confidence, Status

REFERENCES: dict[str, str] = {
    "lai2001": "D. Lai, 'Matter in strong magnetic fields', Rev. Mod. Phys. 73 (2001) 629-662.",
    "harding_lai2006": "A. K. Harding and D. Lai, 'Physics of strongly magnetized neutron stars', "
                       "Rep. Prog. Phys. 69 (2006) 2631-2708.",
    "duncan_thompson1992": "R. C. Duncan and C. Thompson, 'Formation of very strongly magnetized neutron stars: "
                           "implications for gamma-ray bursts', Astrophys. J. 392 (1992) L9-L13.",
    "kaspi_beloborodov2017": "V. M. Kaspi and A. M. Beloborodov, 'Magnetars', Annu. Rev. Astron. Astrophys. 55 "
                             "(2017) 261-301.",
    "schwinger1951": "J. Schwinger, 'On gauge invariance and vacuum polarization', Phys. Rev. 82 (1951) 664-679.",
    "heisenberg_euler1936": "W. Heisenberg and H. Euler, 'Folgerungen aus der Diracschen Theorie des Positrons', "
                            "Z. Phys. 98 (1936) 714-732.",
    "chandrasekhar1939": "S. Chandrasekhar, An Introduction to the Study of Stellar Structure, University of "
                         "Chicago Press (1939) (degenerate electron gas).",
    "shapiro_teukolsky1983": "S. L. Shapiro and S. A. Teukolsky, Black Holes, White Dwarfs, and Neutron Stars: "
                             "The Physics of Compact Objects, Wiley, New York (1983), ch. 2 (cold equation of state "
                             "below neutron drip: degenerate electrons, lattice correction, inverse beta decay).",
    "bps1971": "G. Baym, C. Pethick and P. Sutherland, 'The ground state of matter at high densities: equation "
               "of state and stellar models', Astrophys. J. 170 (1971) 299-317.",
    "chandrasekhar_fermi1953": "S. Chandrasekhar and E. Fermi, 'Problems of gravitational stability in the "
                               "presence of a magnetic field', Astrophys. J. 118 (1953) 116-141.",
    "tatarakis2002": "M. Tatarakis et al., 'Measuring huge magnetic fields', Nature 415 (2002) 280.",
    "wagner2004": "U. Wagner et al., 'Laboratory measurements of 0.7 GG magnetic fields generated during "
                  "high-intensity laser interactions with dense plasmas', Phys. Rev. E 70 (2004) 026401.",
    "skokov2009": "V. Skokov, A. Yu. Illarionov and V. Toneev, 'Estimate of the magnetic field strength in "
                  "heavy-ion collisions', Int. J. Mod. Phys. A 24 (2009) 5925-5932.",
    "yoon2021": "J. W. Yoon et al., 'Realization of laser intensity over 10^23 W/cm^2', Optica 8 (2021) 630-635.",
    "hurley2005": "K. Hurley et al., 'An exceptionally bright flare from SGR 1806-20 and the origins of "
                  "short-duration gamma-ray bursts', Nature 434 (2005) 1098-1103.",
    "horowitz_kadau2009": "C. J. Horowitz and K. Kadau, 'Breaking strain of neutron star crust and gravitational "
                          "waves', Phys. Rev. Lett. 102 (2009) 191102.",
    "berry_geim1997": "M. V. Berry and A. K. Geim, 'Of flying frogs and levitrons', Eur. J. Phys. 18 (1997) "
                      "307-313.",
    "icnirp2009": "International Commission on Non-Ionizing Radiation Protection, 'Guidelines on limits of "
                  "exposure to static magnetic fields', Health Phys. 96 (2009) 504-514.",
    "fowler1960": "C. M. Fowler, W. B. Garn and R. S. Caird, 'Production of very high magnetic fields by "
                  "implosion', J. Appl. Phys. 31 (1960) 588-594.",
    "nakamura2018": "D. Nakamura, A. Ikeda, H. Sawabe, Y. H. Matsuda and S. Takeyama, 'Record indoor magnetic "
                    "field of 1200 T generated by electromagnetic flux-compression', Rev. Sci. Instrum. 89 (2018) "
                    "095106.",
    "herlach1986": "F. Herlach, Europhysics News 17(2) (1986) 29: under adiabatic skin-effect conditions copper "
                   "boils near 150 T.",
    "jackson1999": "J. D. Jackson, Classical Electrodynamics, 3rd ed., Wiley, New York (1999), sec. 5.6 "
                   "(magnetic dipole), 5.10 (uniformly magnetised sphere) and ch. 14 (synchrotron radiation).",
    "moon1982": "F. C. Moon, 'The virial theorem and scaling laws for superconducting magnet systems', "
                "J. Appl. Phys. 53 (1982) 9112-9121.",
    "field_ladder": "magform.benchmarks.FIELD_LADDER (laboratory records up to 2800 T, each with its citation).",
}

# ---------------------------------------------------------------------------
# constants this module needs and constants.py does not carry (CODATA 2018 unless noted)
# ---------------------------------------------------------------------------
HBAR: float = 1.054571817e-34
"""Reduced Planck constant, J s."""

PLANCK: float = 6.62607015e-34
"""Planck constant, J s."""

ELECTRON_MASS: float = 9.1093837015e-31
"""Electron rest mass, kg."""

ATOMIC_MASS_UNIT: float = 1.66053906660e-27
"""Unified atomic mass unit, kg."""

GRAVITATIONAL_CONSTANT: float = 6.67430e-11
"""Newtonian constant of gravitation, m^3/(kg s^2)."""

BOHR_RADIUS: float = 5.29177210903e-11
"""Bohr radius a0, m."""

HARTREE: float = 4.3597447222071e-18
"""Hartree energy (27.211 eV), J."""

ELECTRON_VOLT: float = E_CHARGE
"""One electronvolt, J."""

ELECTRON_REST_ENERGY: float = ELECTRON_MASS * C_LIGHT ** 2
"""m_e c^2 = 0.511 MeV, J."""

FINE_STRUCTURE: float = E_CHARGE ** 2 / (4.0 * math.pi * EPS0 * HBAR * C_LIGHT)
"""Fine-structure constant alpha = 1/137.036."""

SOLAR_MASS: float = 1.98847e30
"""Solar mass, kg (IAU nominal GM divided by G)."""

SOLAR_CONSTANT: float = 1361.0
"""W/m^2.  Total solar irradiance at 1 au."""

SOLAR_LUMINOSITY: float = 3.828e26
"""W.  IAU 2015 nominal solar luminosity."""

MEGATON_TNT: float = 4.184e15
"""J.  One megaton of TNT (by definition 4.184e15 J)."""

PION_REST_ENERGY: float = 139.57039e6 * E_CHARGE
"""J.  Charged-pion rest energy (139.57 MeV), the scale of heavy-ion collision fields, e B ~ m_pi^2."""

# ---------------------------------------------------------------------------
# characteristic fields and pressures
# ---------------------------------------------------------------------------
ATOMIC_FIELD: float = ELECTRON_MASS ** 2 * E_CHARGE ** 3 / ((4.0 * math.pi * EPS0) ** 2 * HBAR ** 3)
"""T (2.3505e5).  Atomic unit of magnetic field B0 = m_e^2 e^3/((4 pi eps0)^2 hbar^3) = hbar/(e a0^2):
the field whose electron cyclotron energy equals the Hartree.  Above it the field, not the
nucleus, sets the shape of an atom (Lai 2001)."""

QED_CRITICAL_FIELD: float = ELECTRON_MASS ** 2 * C_LIGHT ** 2 / (E_CHARGE * HBAR)
"""T (4.414e9).  Quantum-electrodynamic critical field B_Q = m_e^2 c^2/(e hbar): the electron
cyclotron energy equals its rest energy (Schwinger 1951)."""

ATOMIC_PRESSURE: float = HARTREE / BOHR_RADIUS ** 3
"""Pa (2.94e13, 294 Mbar).  Atomic unit of pressure E_h/a0^3: the scale at which outer electron
shells are crushed (pressure ionisation; order of magnitude, heavier elements need ~Z^(10/3) more)."""

DEGENERACY_PRESSURE_UNIT: float = math.pi * ELECTRON_MASS ** 4 * C_LIGHT ** 5 / (3.0 * PLANCK ** 3)
"""Pa (6.002e21).  A = pi m_e^4 c^5/(3 h^3) of Chandrasekhar's equation of state P = A f(x)."""

DEGENERACY_DENSITY_UNIT: float = (8.0 * math.pi / 3.0) * ATOMIC_MASS_UNIT * (ELECTRON_MASS * C_LIGHT / PLANCK) ** 3
"""kg/m^3 (9.739e8).  B = (8 pi/3) m_u (m_e c/h)^3 of rho = B mu_e x^3."""

LATTICE_COEFFICIENT: float = 0.4 * (96.0 / math.pi) ** (1.0 / 3.0)
"""1.2505.  Coulomb (ion-lattice) correction of the cold equation of state in units of A:
P = A [f(x) - LATTICE_COEFFICIENT alpha Z^(2/3) x^4], from the Wigner-Seitz energy
-(9/10) Z^(5/3) e^2/r_e per ion (Shapiro & Teukolsky 1983 sec. 2.4)."""

EOS_MAX_LATTICE_FRACTION: float = 0.25
"""Largest lattice correction (as a fraction of the ideal degeneracy pressure) for which the
equation of state is offered; below it electron shell structure matters and the model is not used
(for iron this is ~2.5e6 T).  Judgement."""

EOS_GENERIC_FLOOR: float = 100.0
"""When the nuclear charge is not given, the equation of state is offered only above this multiple
of :data:`ATOMIC_PRESSURE` (2.9e15 Pa, i.e. ~8.6e4 T).  Judgement."""

NEUTRON_DRIP_DENSITY: float = 4.3e14
"""kg/m^3 (4.3e11 g/cm^3).  Density at which neutrons drip out of nuclei (Baym, Pethick &
Sutherland 1971)."""

NEUTRON_DRIP_PRESSURE: float = 7.8e28
"""Pa (7.8e29 dyn/cm^2).  Pressure at neutron drip (Baym, Pethick & Sutherland 1971); this module's
lattice-corrected equation of state gives 7.9e28 Pa for their drip nuclide 118Kr."""

NUCLEAR_DENSITY: float = 2.3e17
"""kg/m^3.  Nuclear saturation density (0.16 nucleons/fm^3 x m_u, rounded)."""

MIN_NEUTRON_STAR_MASS: float = 0.1 * SOLAR_MASS
"""kg.  Order of the smallest mass whose gravity holds matter at nuclear density (about 0.09 solar
masses in Baym, Pethick & Sutherland 1971); confidence estimated."""

CANONICAL_NEUTRON_STAR: tuple[float, float] = (1.4 * SOLAR_MASS, 12.0e3)
"""(kg, m).  Mass and radius of the canonical neutron star used for the Chandrasekhar-Fermi limit."""

HYDROGEN_ASYMPTOTIC_MIN: float = 100.0
"""Smallest b = B/B0 for which the asymptotic hydrogen binding energy 0.16 (ln b)^2 Hartree is quoted
(Lai 2001); below it :func:`hydrogen_binding_energy` returns None."""

CONDENSED_DENSITY_COEFFICIENT: float = 5.61e5
"""kg/m^3.  rho_s = 561 g/cm^3 x A Z^(-3/5) B12^(6/5) with B12 = B/(1e8 T): zero-pressure density of
magnetically condensed matter (Lai 2001, uniform-gas model; Wigner-Seitz radius 1.90 Z^(1/5) b^(-2/5) a0)."""

SYNCHROTRON_COEFFICIENT: float = 6.0 * math.pi * EPS0 * ELECTRON_MASS ** 3 * C_LIGHT ** 3 / E_CHARGE ** 4
"""s T^2 (5.159).  Classical synchrotron cooling time of an electron is this / (B^2 gamma)."""

# ---------------------------------------------------------------------------
# comparison data
# ---------------------------------------------------------------------------
CONCEPT_PRESSURE: float = 4.0e9
"""Pa.  The top of the concept note's window, the yardstick for 'pressure gap'."""

DESTRUCTIVE_FIELD_RECORD: float = max(b.value for b in benchmarks.FIELD_LADDER)
"""T (2800).  Strongest laboratory field in a macroscopic volume (explosive flux compression;
``benchmarks.FIELD_LADDER``)."""

LASER_PEAK_POWER: float = 1.0e16
"""W.  Peak power of the largest lasers (10 PW class); order of magnitude, confidence estimated."""

LASER_RECORD_INTENSITY: float = 1.1e27
"""W/m^2 (1.1e23 W/cm^2).  Highest reported focused laser intensity (Yoon et al. 2021)."""

Z_FIELD_RADIUS: float = 1.0e-3
"""m.  Radius at which the Z machine's peak current is converted to a field, B = mu0 I/(2 pi r)."""

WATER_SUSCEPTIBILITY: float = -9.04e-6
"""SI volume magnetic susceptibility of water (diamagnetic)."""

WATER_DENSITY: float = 1000.0
"""kg/m^3."""

WATER_LEVITATION_GRADIENT: float = 1400.0
"""T^2/m.  B dB/dz that levitates water (and frogs) against Earth gravity (Berry & Geim 1997);
mu0 rho g/|chi| = 1363 T^2/m with the constants above."""

DAMAGE_FLUENCE: float = 1.0e6
"""J/m^2.  Radiant fluence taken as damaging to an unshielded spacecraft surface: of the order
needed to melt a millimetre of aluminium (~2.5e6 J/m^2).  Confidence estimated."""

REFERENCE_COLLECTOR_AREA: float = 1.0e6
"""m^2.  A one-square-kilometre solar collector, the unit used for 'time to bank one shot'."""

VAPORISATION: dict[str, tuple[float, float]] = {
    "Cu": (2835.0, 4.73e6),
    "Al": (2792.0, 1.09e7),
    "Fe": (3134.0, 6.09e6),
    "W": (5828.0, 4.35e6),
}
"""Base element -> (boiling point [K], heat of vaporisation [J/kg]) at 1 atm.  Handbook values
(300, 294, 340 and 800 kJ/mol); sources differ by ~5 % (10 % for tungsten).  Alloys use their base
element."""

ELEMENTS: dict[str, tuple[int, float]] = {
    "H": (1, 1.008), "C": (6, 12.011), "Mg": (12, 24.305), "Al": (13, 26.982), "Ti": (22, 47.867),
    "Fe": (26, 55.845), "Co": (27, 58.933), "Ni": (28, 58.693), "Cu": (29, 63.546), "Nb": (41, 92.906),
    "Mo": (42, 95.95), "Ta": (73, 180.948), "W": (74, 183.84), "Re": (75, 186.207),
}
"""Base element -> (atomic number Z, standard atomic weight).  Gives mu_e = A/Z and the lattice
correction for registry materials."""


@dataclass(frozen=True)
class Nuclide(ResultMixin):
    """A nuclide and the electron kinetic energy at which it starts capturing electrons.

    ``threshold_energy`` [J] is the electron Fermi *kinetic* energy at which
    (A, Z) + e- -> (A, Z-1) + nu becomes energetically allowed, i.e. the beta-decay energy of the
    daughter.  ``mu_e = A/Z`` is the mean molecular weight per electron.

    Reference: Shapiro & Teukolsky (1983) ch. 2 (inverse beta decay thresholds).
    """

    key: str
    mass_number: int
    charge: int
    daughter: str
    threshold_energy: float = unit("J")
    confidence: Confidence = Confidence.HANDBOOK
    note: str = ""

    @property
    def mu_e(self) -> float:
        """Mean molecular weight per electron, A/Z."""
        return self.mass_number / self.charge


def _nuclides() -> dict[str, Nuclide]:
    mev = 1.0e6 * ELECTRON_VOLT
    table = "Shapiro & Teukolsky (1983) table of neutronisation thresholds"
    rows = (
        Nuclide("1H", 1, 1, "n", 0.782 * mev, note=f"(m_n - m_p - m_e) c^2; {table}"),
        Nuclide("4He", 4, 2, "3H + n", 20.596 * mev, note=table),
        Nuclide("12C", 12, 6, "12B", 13.370 * mev, note=table),
        Nuclide("16O", 16, 8, "16N", 10.419 * mev, note=table),
        Nuclide("20Ne", 20, 10, "20F", 7.026 * mev, note=table),
        Nuclide("24Mg", 24, 12, "24Na", 5.513 * mev, note=table),
        Nuclide("27Al", 27, 13, "27Mg", 2.610 * mev, Confidence.REPRESENTATIVE,
                "beta-decay energy of 27Mg (ground state to ground state)"),
        Nuclide("28Si", 28, 14, "28Al", 4.643 * mev, note=table),
        Nuclide("32S", 32, 16, "32P", 1.710 * mev, note=table),
        Nuclide("56Fe", 56, 26, "56Mn", 3.695 * mev, note=table),
        Nuclide("63Cu", 63, 29, "63Ni", 0.0669 * mev, Confidence.REPRESENTATIVE,
                "beta-decay energy of 63Ni (66.9 keV); 63Cu is 69 % of natural copper"),
    )
    return {n.key: n for n in rows}


NUCLIDES: dict[str, Nuclide] = _nuclides()
"""Nuclides with an electron-capture threshold energy, keyed '56Fe' etc."""

ELEMENT_NUCLIDE: dict[str, str] = {"H": "1H", "C": "12C", "Mg": "24Mg", "Al": "27Al", "Fe": "56Fe", "Cu": "63Cu"}
"""Base element -> the nuclide used for its electron-capture threshold."""


@dataclass(frozen=True)
class FieldRecord(ResultMixin):
    """One rung of the field ladder: how strong, how large, how long.

    ``volume`` [m^3] is the order of the high-field region and ``duration`` [s] the time near peak
    (``None`` = steady); both are order-of-magnitude estimates unless the note says otherwise.
    ``origin`` is "laboratory" or "astrophysical"; ``destructive`` is True when the source does not
    survive the shot (None for astrophysical objects).

    Reference: ``magform.benchmarks.FIELD_LADDER`` and the citation of each record.
    """

    key: str
    description: str
    field: float = unit("T")
    volume: float | None = unit("m^3")
    duration: float | None = unit("s")
    origin: str = "laboratory"
    destructive: bool | None = False
    citation: str = ""
    confidence: Confidence = Confidence.REPRESENTATIVE
    note: str = ""

    @property
    def steady(self) -> bool:
        """True when the field is held indefinitely (``duration`` is None)."""
        return self.duration is None


_BENCHMARK_SCALES: dict[str, tuple[float, float | None, bool]] = {
    # key: (volume m^3, duration s or None for steady, destructive)
    "nhmfl_32t_all_sc": (1e-5, None, False),
    "secuf_35p6t_all_sc": (1e-5, None, False),
    "rebco_insert_45p5t": (1e-6, None, False),
    "lbc9_48p7t": (1e-6, None, False),
    "chmfl_resistive_42t": (1e-5, None, False),
    "chmfl_hybrid_45t": (1e-5, None, False),
    "lanl_100t_nondestructive": (3e-6, 1e-2, False),
    "single_turn_300t": (1e-6, 5e-6, True),
    "emfc_1200t": (1e-8, 1e-6, True),
    "explosive_fc_2800t": (1e-7, 1e-6, True),
}


def _field_ladder() -> tuple[FieldRecord, ...]:
    rows: list[FieldRecord] = []
    for b in benchmarks.FIELD_LADDER:
        volume, duration, destructive = _BENCHMARK_SCALES.get(b.key, (None, None, False))
        rows.append(FieldRecord(b.key, b.description, float(b.value), volume, duration, "laboratory",
                                destructive, b.citation, b.confidence,
                                "volume and duration are order-of-magnitude estimates"))
    z_current = benchmarks.PULSED_CURRENT_RECORD
    rows.extend((
        FieldRecord("pulsed_user_60t", "Non-destructive pulsed user magnets (60 T class, about 20 mm bore)",
                    60.0, 5e-6, 1e-2, "laboratory", False,
                    "F. Herlach and N. Miura (eds.), High Magnetic Fields: Science and Technology, Vol. 1, "
                    "World Scientific, Singapore (2003).", Confidence.REPRESENTATIVE,
                    "the field class benchmarks.pulsed_field_status treats as routine"),
        FieldRecord("emf_industrial", "Industrial electromagnetic-forming coil (upper end)",
                    benchmarks.EMF_CONVENTIONAL_FIELD[1], 1e-4, 1e-4, "laboratory", False,
                    "Psyk et al., J. Mater. Process. Technol. 211, 787 (2011).", Confidence.REPRESENTATIVE,
                    "benchmarks.EMF_CONVENTIONAL_FIELD"),
        FieldRecord("z_machine_surface", "Z-pinch driver: field at 1 mm radius for the Z machine's peak current",
                    MU0 * z_current / (2.0 * math.pi * Z_FIELD_RADIUS), 1e-8, 1e-7, "laboratory", True,
                    "Computed, B = mu0 I/(2 pi r), from benchmarks.PULSED_CURRENT_RECORD (Savage et al. 2007).",
                    Confidence.ESTIMATED, "the load is destroyed each shot; the driver is not"),
        FieldRecord("laser_plasma_tatarakis", "Quasi-static field in a laser-produced plasma (340 MG)",
                    3.4e4, 1e-15, 1e-12, "laboratory", True, REFERENCES["tatarakis2002"]),
        FieldRecord("laser_plasma_wagner", "Quasi-static field in a laser-produced plasma (0.7 GG)",
                    7.0e4, 1e-15, 1e-12, "laboratory", True, REFERENCES["wagner2004"]),
        FieldRecord("magnetic_white_dwarf", "Strongest magnetic white dwarfs (1e9 G)",
                    1.0e5, 1e21, None, "astrophysical", None, REFERENCES["lai2001"]),
        FieldRecord("laser_optical_field", "Oscillating field at the focus of the most intense laser (1.1e23 W/cm^2)",
                    math.sqrt(2.0 * MU0 * LASER_RECORD_INTENSITY / C_LIGHT), 1e-17, 2e-14, "laboratory", False,
                    REFERENCES["yoon2021"], Confidence.ESTIMATED,
                    "computed, B = sqrt(2 mu0 I/c); reverses every half optical cycle (~1.3 fs)"),
        FieldRecord("radio_pulsar", "Typical radio pulsar surface field (1e12 G)",
                    1.0e8, 4e12, None, "astrophysical", None, REFERENCES["harding_lai2006"]),
        FieldRecord("magnetar_typical", "Magnetar surface dipole field (1e14 G)",
                    1.0e10, 4e12, None, "astrophysical", None, REFERENCES["kaspi_beloborodov2017"]),
        FieldRecord("magnetar_upper", "Upper range of magnetar surface dipole fields (1e15 G)",
                    1.0e11, 4e12, None, "astrophysical", None, REFERENCES["kaspi_beloborodov2017"]),
        FieldRecord("heavy_ion_collision", "Peripheral heavy-ion collision, e B ~ m_pi^2",
                    PION_REST_ENERGY ** 2 / (E_CHARGE * HBAR * C_LIGHT ** 2), 1e-42, 1e-23, "laboratory", True,
                    REFERENCES["skokov2009"], Confidence.REPRESENTATIVE,
                    "computed from e B = m_pi^2; femtometre volume, gone in ~1e-23 s"),
    ))
    return tuple(sorted(rows, key=lambda r: r.field))


EXTREME_FIELD_LADDER: tuple[FieldRecord, ...] = _field_ladder()
"""Field records from industrial forming coils to heavy-ion collisions, ascending.  The
laboratory entries up to 2800 T come from ``benchmarks.FIELD_LADDER``."""


@dataclass(frozen=True)
class EnergyScale(ResultMixin):
    """A reference energy for placing the cost of a shot.

    Reference: the citation of each entry.
    """

    key: str
    description: str
    energy: float = unit("J")
    citation: str = ""
    confidence: Confidence = Confidence.REPRESENTATIVE


ENERGY_SCALES: tuple[EnergyScale, ...] = (
    EnergyScale("largest_pulsed_store", "Largest pulsed energy store in a magnet laboratory",
                benchmarks.ENERGY_ANY_STORE_RECORD, "magform.benchmarks.ENERGY_ANY_STORE_RECORD", Confidence.REPORTED),
    EnergyScale("megaton_tnt", "One megaton of TNT", MEGATON_TNT, "By definition, 4.184e15 J.", Confidence.HANDBOOK),
    EnergyScale("largest_nuclear_test", "Largest nuclear device ever detonated (about 50 Mt, 1961)",
                50.0 * MEGATON_TNT, "Order of magnitude of the 1961 Novaya Zemlya test.", Confidence.REPRESENTATIVE),
    EnergyScale("world_annual_energy", "World primary energy consumption in one year (about 600 EJ)",
                6.0e20, "Order of magnitude for the early 2020s.", Confidence.ESTIMATED),
    EnergyScale("solar_output_second", "Energy radiated by the Sun in one second",
                SOLAR_LUMINOSITY, "IAU 2015 nominal solar luminosity, 3.828e26 W.", Confidence.HANDBOOK),
    EnergyScale("magnetar_giant_flare", "Giant flare of the magnetar SGR 1806-20 (27 December 2004)",
                2.0e39, REFERENCES["hurley2005"], Confidence.REPORTED),
)
"""Reference energies, ascending."""


@dataclass(frozen=True)
class FieldThreshold(ResultMixin):
    """A field level that matters to people and hardware near the source.

    Reference: the citation of each entry.
    """

    key: str
    description: str
    field: float = unit("T")
    citation: str = ""
    confidence: Confidence = Confidence.HANDBOOK


FIELD_THRESHOLDS: tuple[FieldThreshold, ...] = (
    FieldThreshold("earth_field", "Comparable to the Earth's surface field (25-65 uT): compasses and "
                   "magnetometers are disturbed", 5.0e-5, "Order of magnitude.", Confidence.ESTIMATED),
    FieldThreshold("pacemaker_line", "0.5 mT ('5 gauss') line for implanted electronic medical devices",
                   5.0e-4, REFERENCES["icnirp2009"]),
    FieldThreshold("public_limit", "Static-field exposure limit for the general public (400 mT)",
                   0.4, REFERENCES["icnirp2009"]),
    FieldThreshold("occupational_limit", "Occupational static-field limit for head and trunk (2 T)",
                   2.0, REFERENCES["icnirp2009"]),
    FieldThreshold("limb_limit", "Occupational static-field limit for limbs (8 T)",
                   8.0, REFERENCES["icnirp2009"]),
    FieldThreshold("atomic_field", "Atomic unit of field (2.35e5 T): atoms are reshaped and ordinary chemistry stops",
                   ATOMIC_FIELD, REFERENCES["lai2001"]),
)
"""Stand-off thresholds, ascending in field (descending in distance)."""

APPLICATIONS: tuple[str, ...] = ("forging", "refractory_forming", "welding", "synthesis", "densification",
                                 "containment")
"""The six applications of the concept note, in its order."""

TIERS: tuple[str, ...] = ("solid", "liquid_skin", "megabar", "plasma", "landau", "degenerate")
"""Process tiers used by :func:`process_outcomes`: below skin melting; skin molten or vaporised;
pressure above the bulk modulus; pressure-ionised; beyond the atomic unit of field; relativistically
degenerate matter."""

_PROCESS_TEXT: dict[str, tuple[str, ...]] = {
    "forging": (
        "Electromagnetic forming as modelled in magform.eddy: the workpiece stays solid and flows where the "
        "pressure exceeds its flow stress.",
        "The driven surface melts or boils within the pulse, so the part is shaped through a liquid-metal skin "
        "and a shock: 'magnetohydrodynamic' finally describes it, and surface finish and dimensions are lost.",
        "The pressure exceeds the bulk modulus: the metal is shock-compressed to a fraction of its volume and "
        "comes back as hot liquid or vapour. No forged shape survives.",
        "The workpiece is a pressure-ionised plasma for the length of the pulse. There is no solid to forge.",
        "The piston pressure makes degenerate matter and the field, where it gets in, squeezes atoms into "
        "needles. Forging has no meaning.",
        "The sample becomes white-dwarf matter and explodes when the field lets go. Forging has no meaning.",
    ),
    "refractory_forming": (
        "Flow stress decides (magform.plasticity): tungsten needs roughly 2.5-3 GPa at room temperature.",
        "Strength stops mattering: the pressure is several to hundreds of times any flow stress, and tungsten "
        "fails the same way copper does, by losing its skin.",
        "Refractory and soft metals behave alike: neither strength nor melting point registers against "
        "megabar pressure.",
        "No property of the solid survives; only density and atomic number distinguish one target from another.",
        "Only nuclear charge and mass number distinguish materials.",
        "Only nuclear charge and mass number distinguish materials, and above the capture threshold not even "
        "the charge is conserved.",
    ),
    "welding": (
        "Impact welding needs only 10-40 T (magform.welding); more field overshoots the weld window.",
        "Flyers leave at several to tens of kilometres per second, far beyond any weld window: the joint is a "
        "hypervelocity impact crater.",
        "Flyers reach tens of kilometres per second and both parts vaporise on impact. Nothing is joined.",
        "Both parts are plasma before they meet.",
        "Both parts are plasma before they meet.",
        "Both parts are degenerate plasma before they meet.",
    ),
    "synthesis": (
        "Below every static route to diamond or cubic boron nitride; a flyer-driven shock is marginal "
        "(magform.synthesis).",
        "Shock and ramp compression to tens and hundreds of gigapascals: diamond, lonsdaleite and other dense "
        "phases form in nanoseconds, as in flux-compression and Z-pinch experiments today. Recovering the product "
        "is the hard part.",
        "Planetary-core pressures, megabars and up: metallic hydrogen and other dense phases exist while the "
        "pulse lasts and not after.",
        "Warm dense matter: every chemical bond is broken and whatever is recovered is quenched debris.",
        "'Magnetic chemistry': where the field threads matter, atoms bind into chains along it with tens to "
        "hundreds of electronvolts per atom, but only while the field is on. As a piston it makes degenerate "
        "matter.",
        "Nuclear, not chemical, synthesis: degenerate electrons are forced into nuclei above each capture "
        "threshold and light elements fuse. The product is neutron-rich isotopes and a thermonuclear yield.",
    ),
    "densification": (
        "Pores collapse only above (2/3) sigma ln(1/f) and never bond cold (magform.porosity).",
        "Every pore is crushed, together with the surface of the part.",
        "Every pore is crushed, and so is the part.",
        "Porosity has no meaning in a plasma.",
        "Porosity has no meaning in degenerate matter.",
        "Porosity has no meaning in degenerate matter.",
    ),
    "containment": (
        "Compound steel walls hold about 2.5 GPa; no monolithic wall holds 4 GPa (magform.containment).",
        "No structure contains the field statically: the pressure exceeds every material strength, so the "
        "source is destroyed each shot, as single-turn coils and flux-compression generators are.",
        "No structure contains the field: it is held only by the inertia of an imploding liner, for "
        "microseconds.",
        "Inertial confinement only, for nanoseconds or less.",
        "Nothing material holds the field; unheld it disperses in the light-crossing time. Nature's only static "
        "holder is a neutron star.",
        "Nothing material holds the field; unheld it disperses in the light-crossing time. Nature's only static "
        "holder is a neutron star.",
    ),
}

_STATE_TEXT: dict[str, str] = {
    "elastic": "solid and elastic: the pressure is below the yield strength and nothing permanent happens",
    "plastic_flow": "solid, flowing plastically wherever the load is uneven (the regime of the rest of magform)",
    "skin_melting": "solid inside a molten eddy-current skin: the driven surface is a conducting liquid",
    "skin_boiling": "the skin reaches its boiling point within the pulse and starts to blow off as vapour",
    "skin_vaporised": "the whole skin depth is vaporised and expands as a conducting plasma",
    "bulk_modulus": "compressed to a fraction of its volume (megabar dynamic compression) and released as hot "
                    "liquid or vapour",
    "pressure_ionisation": "pressure-ionised: electron shells are crushed and the workpiece is a dense plasma, "
                           "degenerate like the outer layers of a white dwarf as the pressure rises",
    "atomic_field": "where the field threads the matter, atoms are squeezed into needles along it and ordinary "
                    "chemistry stops",
    "electron_capture": "degenerate electrons are forced into nuclei: the workpiece is being transmuted",
    "relativistic_degeneracy": "white-dwarf matter: a relativistically degenerate electron gas denser than "
                               "1e9 kg/m^3",
    "qed_critical": "the field is beyond the quantum-electrodynamic critical value: an electron's cyclotron energy "
                    "exceeds its rest energy and the vacuum itself is birefringent",
    "neutron_drip": "neutrons drip out of the nuclei: the matter of a neutron star's inner crust",
}


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise InputError(f"{name} must be a real number, got {value!r}")
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(v):
        raise InputError(f"{name} must be finite, got {value!r}")
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


def _field(value: object, name: str = "field") -> float:
    """A field magnitude in T (the sign is irrelevant everywhere in this module)."""
    return abs(_finite(name, value))


def _positive_field(value: object, name: str = "field") -> float:
    v = _field(value, name)
    if v == 0.0:
        raise InputError(f"{name} must be non-zero")
    return v


def _charge(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputError(f"charge must be a positive integer (atomic number), got {value!r}")
    return value


def _mu_e(value: object) -> float:
    v = _positive("mu_e", value)
    if v < 1.0:
        raise InputError(f"mu_e = A/Z cannot be below 1 (hydrogen), got {v!r}")
    return v


def _sci(value: float) -> str:
    return f"{value:.3g}"


def _human_time(seconds: float) -> str:
    """A duration in the unit a reader would use."""
    year = 3.15576e7
    if seconds >= year:
        return f"{seconds / year:.3g} years"
    if seconds >= 86400.0:
        return f"{seconds / 86400.0:.3g} days"
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.3g} hours"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.3g} minutes"
    return f"{seconds:.3g} s"


# ---------------------------------------------------------------------------
# 1. characteristic fields
# ---------------------------------------------------------------------------
def atomic_field_ratio(field: float) -> float:
    """Field in atomic units, ``b = B/B0`` with ``B0 = hbar/(e a0^2) = 2.3505e5 T``.

    ``field`` in T.  For b >> 1 the cyclotron radius of an electron is smaller than the Bohr
    radius and the field, not the nucleus, sets the transverse size of an atom.  Magnetars:
    b ~ 4e4-4e5.

    Reference: Lai (2001) sec. I.
    """
    return _field(field) / ATOMIC_FIELD


def qed_field_ratio(field: float) -> float:
    """Field in units of the QED critical field, ``B/B_Q`` with ``B_Q = m_e^2 c^2/(e hbar) = 4.414e9 T``.

    ``field`` in T.  At ``B_Q`` the electron cyclotron energy equals its rest energy; above it
    the vacuum is birefringent and photons can split.  Magnetars: 2-20.

    Reference: Schwinger (1951); Harding & Lai (2006).
    """
    return _field(field) / QED_CRITICAL_FIELD


def cyclotron_energy(field: float) -> float:
    """Electron cyclotron energy ``hbar omega_c = hbar e B/m_e`` [J] (non-relativistic).

    ``field`` in T.  1.158e-4 eV per tesla: 11.6 meV at 100 T, one Hartree at 2.35e5 T, 0.511 MeV
    at 4.41e9 T.  It is the spacing of Landau levels for B << B_Q; above that use
    :func:`landau_excitation_energy`.

    Reference: Lai (2001) sec. II.
    """
    return HBAR * E_CHARGE * _field(field) / ELECTRON_MASS


def landau_excitation_energy(field: float) -> float:
    """Energy of the first excited Landau level above the ground level [J], relativistically exact.

    ``E_1 - E_0 = m_e c^2 (sqrt(1 + 2 B/B_Q) - 1)``, which reduces to ``hbar e B/m_e`` for
    B << B_Q.  ``field`` in T.  At 1e10 T it is 0.69 MeV (the non-relativistic formula gives
    1.16 MeV).

    Reference: Harding & Lai (2006) sec. 2; Lai (2001).
    """
    b = _field(field) / QED_CRITICAL_FIELD
    if b < 1e-6:
        return ELECTRON_REST_ENERGY * b * (1.0 - 0.5 * b)
    return ELECTRON_REST_ENERGY * (math.sqrt(1.0 + 2.0 * b) - 1.0)


def magnetic_length(field: float) -> float:
    """Magnetic length ``lambda_B = sqrt(hbar/(e B))`` [m]: the cyclotron radius of the lowest Landau level.

    ``field`` in T (non-zero).  Equals the Bohr radius at B0 = 2.35e5 T and ``a0/sqrt(b)`` in
    general: 2.57e-9 m at 100 T, 2.57e-13 m at 1e10 T.

    Reference: Lai (2001) sec. II.
    """
    return math.sqrt(HBAR / (E_CHARGE * _positive_field(field)))


# ---------------------------------------------------------------------------
# 2. the field ladder
# ---------------------------------------------------------------------------
def laboratory_record(min_volume: float = 0.0, min_duration: float = 0.0) -> FieldRecord | None:
    """Strongest laboratory field that fills at least ``min_volume`` for at least ``min_duration``.

    Picks from :data:`EXTREME_FIELD_LADDER` the largest laboratory ``field`` with
    ``volume >= min_volume`` [m^3] and (steady or ``duration >= min_duration`` [s]); ``None`` if
    nothing qualifies.  It shows the trade the premise ignores: 3e14 T exists for 1e-23 s in a
    femtometre volume, 2800 T for a microsecond in a tenth of a cubic centimetre, 45 T for as
    long as the power bill is paid.  Volumes and durations are order-of-magnitude estimates.

    Reference: ``magform.benchmarks.FIELD_LADDER``; Tatarakis et al. (2002); Wagner et al. (2004);
    Skokov et al. (2009).
    """
    v = _non_negative("min_volume", min_volume)
    t = _non_negative("min_duration", min_duration)
    best: FieldRecord | None = None
    for record in EXTREME_FIELD_LADDER:
        if record.origin != "laboratory" or record.volume is None or record.volume < v:
            continue
        if record.duration is not None and record.duration < t:
            continue
        if best is None or record.field > best.field:
            best = record
    return best


# ---------------------------------------------------------------------------
# 3. pressure, energy and mass equivalent
# ---------------------------------------------------------------------------
def mass_equivalent_density(field: float) -> float:
    """Mass-equivalent density of the field energy, ``u/c^2 = B^2/(2 mu0 c^2)`` [kg/m^3].

    ``field`` in T.  4.4e-8 kg/m^3 at 100 T; 4.43e8 kg/m^3 at 1e10 T (the field outweighs lead
    by 4e4); 4.43e10 kg/m^3 at 1e11 T.  Classical energy density; the vacuum-polarisation
    correction is below 1 % up to 1e11 T.

    Reference: Jackson (1999) sec. 6.7; Heisenberg & Euler (1936) for the correction.
    """
    return magnetic_pressure(_field(field)) / C_LIGHT ** 2


def field_energy(field: float, volume: float) -> float:
    """Energy of a uniform field filling a volume, ``E = (B^2/(2 mu0)) V`` [J].

    ``field`` in T, ``volume`` in m^3 (>= 0).  At 1e10 T: 3.98e16 J (9.5 megatons) per cubic
    millimetre, 39.8 MJ per cubic micrometre.

    Reference: Jackson (1999) sec. 6.7.
    """
    return magnetic_pressure(_field(field)) * _non_negative("volume", volume)


def tnt_megatons(energy: float) -> float:
    """Energy in megatons of TNT, ``E / 4.184e15 J``.

    ``energy`` in J (>= 0).

    Reference: definition of the megaton, 4.184e15 J.
    """
    return _non_negative("energy", energy) / MEGATON_TNT


def energy_ratios(energy: float) -> dict[str, float]:
    """Ratio of an energy to each entry of :data:`ENERGY_SCALES`, keyed by the entry's key.

    ``energy`` in J (>= 0).  A ratio above 1 means the shot exceeds that reference.

    Reference: the citations in :data:`ENERGY_SCALES`.
    """
    e = _non_negative("energy", energy)
    return {scale.key: e / scale.energy for scale in ENERGY_SCALES}


# ---------------------------------------------------------------------------
# 4. cold degenerate matter under a magnetic piston
# ---------------------------------------------------------------------------
def _f(x: float) -> float:
    """Chandrasekhar's f(x) = x (2x^2 - 3) sqrt(1 + x^2) + 3 asinh x = 8 int_0^x t^4/sqrt(1 + t^2) dt."""
    if x < 0.1:  # series: the closed form cancels to O(x^5)
        x2 = x * x
        return x ** 5 * (8.0 / 5.0 + x2 * (-4.0 / 7.0 + x2 * (1.0 / 3.0 + x2 * (-5.0 / 22.0 + x2 * (35.0 / 208.0)))))
    return x * (2.0 * x * x - 3.0) * math.sqrt(1.0 + x * x) + 3.0 * math.asinh(x)


def _lattice(charge: int | None) -> float:
    """Coefficient c of the lattice term, P = A [f(x) - c x^4] (0 without a charge)."""
    return 0.0 if charge is None else LATTICE_COEFFICIENT * FINE_STRUCTURE * charge ** (2.0 / 3.0)


def degeneracy_pressure(fermi_parameter: float, charge: int | None = None) -> float:
    """Pressure of cold degenerate matter at relativity parameter ``x = p_F/(m_e c)`` [Pa].

    Chandrasekhar's exact zero-temperature electron gas::

        P = A f(x),   A = pi m_e^4 c^5/(3 h^3) = 6.002e21 Pa,
        f(x) = x (2 x^2 - 3) sqrt(1 + x^2) + 3 asinh(x)

    with ``f -> (8/5) x^5`` for x << 1 and ``f -> 2 x^4`` for x >> 1.  With ``charge`` Z the
    Coulomb attraction of the ion lattice is subtracted, ``P = A [f(x) - 1.2505 alpha Z^(2/3) x^4]``
    (zero below the model's zero-pressure point).  ``fermi_parameter`` >= 0.

    Validity: T = 0, fully ionised, electrons only (ions contribute no pressure); the lattice
    term is the uniform-electron Wigner-Seitz estimate and fails at low density, where shell
    structure matters (:func:`lattice_pressure_fraction`).

    Reference: Chandrasekhar (1939); Shapiro & Teukolsky (1983) sec. 2.3-2.4.
    """
    x = _non_negative("fermi_parameter", fermi_parameter)
    c = _lattice(None if charge is None else _charge(charge))
    return max(DEGENERACY_PRESSURE_UNIT * (_f(x) - c * x ** 4), 0.0)


def degenerate_density(fermi_parameter: float, mu_e: float = 2.0) -> float:
    """Mass density of cold matter at relativity parameter x, ``rho = B mu_e x^3`` [kg/m^3].

    ``B = (8 pi/3) m_u (m_e c/h)^3 = 9.739e8 kg/m^3`` and ``mu_e = A/Z`` is the mean molecular
    weight per electron (2 for carbon and oxygen, 2.154 for 56Fe, 1 for hydrogen).  x = 1
    (electrons turning relativistic) is 1.95e9 kg/m^3 for mu_e = 2.

    Reference: Chandrasekhar (1939); Shapiro & Teukolsky (1983) sec. 2.3.
    """
    x = _non_negative("fermi_parameter", fermi_parameter)
    return DEGENERACY_DENSITY_UNIT * _mu_e(mu_e) * x ** 3


def fermi_parameter_for_density(density: float, mu_e: float = 2.0) -> float:
    """Relativity parameter ``x = (rho/(B mu_e))^(1/3)`` of cold matter at a mass density.

    ``density`` in kg/m^3 (>= 0).  Inverse of :func:`degenerate_density`.

    Reference: Chandrasekhar (1939).
    """
    rho = _non_negative("density", density)
    return (rho / (DEGENERACY_DENSITY_UNIT * _mu_e(mu_e))) ** (1.0 / 3.0)


def fermi_kinetic_energy(fermi_parameter: float) -> float:
    """Electron Fermi kinetic energy ``m_e c^2 (sqrt(1 + x^2) - 1)`` [J].

    ``fermi_parameter`` x >= 0.  It is what a degenerate electron brings to a nucleus:
    electron capture starts when it reaches the beta-decay energy of the daughter.

    Reference: Shapiro & Teukolsky (1983) sec. 2.3.
    """
    x = _non_negative("fermi_parameter", fermi_parameter)
    if x < 1e-4:
        return ELECTRON_REST_ENERGY * 0.5 * x * x * (1.0 - 0.25 * x * x)
    return ELECTRON_REST_ENERGY * (math.sqrt(1.0 + x * x) - 1.0)


def _fermi_parameter_for_energy(energy: float) -> float:
    e = energy / ELECTRON_REST_ENERGY
    return math.sqrt(e * (e + 2.0))


def lattice_pressure_fraction(fermi_parameter: float, charge: int) -> float:
    """Fraction of the ideal degeneracy pressure cancelled by the ion lattice.

    ``1.2505 alpha Z^(2/3) x^4 / f(x)``: about ``0.78 alpha Z^(2/3)/x`` for x << 1 and
    ``0.625 alpha Z^(2/3)`` for x >> 1 (4 % for iron).  ``fermi_parameter`` > 0, ``charge`` the
    atomic number.  The equation of state is used only where this is below
    :data:`EOS_MAX_LATTICE_FRACTION`; a value >= 1 means the model's zero-pressure density has
    been passed.

    Reference: Shapiro & Teukolsky (1983) sec. 2.4.
    """
    x = _positive("fermi_parameter", fermi_parameter)
    return _lattice(_charge(charge)) * x ** 4 / _f(x)


def _solve_x(pressure: float, charge: int | None) -> float:
    """x with A [f(x) - c x^4] = pressure (pressure > 0), on the rising branch."""
    c = _lattice(charge)
    target = pressure / DEGENERACY_PRESSURE_UNIT
    lo = math.log(max(0.625 * c, 1e-12))  # zero-pressure point of the model: f ~ (8/5) x^5 = c x^4
    hi = math.log(1e12)

    def excess(log_x: float) -> float:
        x = math.exp(log_x)
        return _f(x) - c * x ** 4 - target

    if excess(lo) >= 0.0:
        return math.exp(lo)
    return math.exp(bisect(excess, lo, hi, tol=1e-14, max_iter=400))


def fermi_parameter_for_pressure(pressure: float, charge: int | None = None) -> float:
    """Relativity parameter x at which cold degenerate matter supports ``pressure``.

    Solves ``A [f(x) - c x^4] = P`` (c = 0 without ``charge``) by bisection on the rising
    branch.  ``pressure`` in Pa (> 0).  No validity check is applied here; use
    :func:`piston_state` for the checked result.

    Reference: Chandrasekhar (1939); Shapiro & Teukolsky (1983) sec. 2.4.
    """
    p = _positive("pressure", pressure)
    return _solve_x(p, None if charge is None else _charge(charge))


@dataclass(frozen=True, kw_only=True)
class PistonState(ResultMixin):
    """Cold matter held against a magnetic piston: the degenerate state that balances B^2/(2 mu0).

    ``fermi_parameter`` x = p_F/(m_e c); ``density`` = B mu_e x^3; ``fermi_energy`` the electron
    Fermi kinetic energy; ``lattice_fraction`` the share of the ideal pressure cancelled by the
    ion lattice (None without a charge); ``relativistic`` is x >= 1; ``compression`` is filled in
    by :func:`classify_regime` when a material is known.

    Reference: Chandrasekhar (1939); Shapiro & Teukolsky (1983) ch. 2.
    """

    _title: ClassVar[str] = "Matter under a magnetic piston"

    field: float = unit("T")
    pressure: float = unit("Pa")
    mu_e: float
    charge: int | None
    fermi_parameter: float
    density: float = unit("kg/m^3")
    fermi_energy: float = unit("J")
    lattice_fraction: float | None
    relativistic: bool


def eos_floor_field(charge: int | None = None) -> float:
    """Field below which :func:`piston_state` declines to answer [T].

    With ``charge`` Z: the field whose pressure is supported at the x where the lattice
    correction is :data:`EOS_MAX_LATTICE_FRACTION` of the ideal pressure (2.5e6 T for iron,
    3.0e6 T for copper, 2.1e5 T for carbon, 1.1e4 T for hydrogen).  Without it:
    ``sqrt(2 mu0 x 100 E_h/a0^3)`` = 8.6e4 T.

    Reference: Shapiro & Teukolsky (1983) sec. 2.4.
    """
    return _floor_field(None if charge is None else _charge(charge))


@functools.lru_cache(maxsize=None)
def _floor_field(charge: int | None) -> float:
    if charge is None:
        return field_for_pressure(EOS_GENERIC_FLOOR * ATOMIC_PRESSURE)
    c = _lattice(charge)

    def excess(log_x: float) -> float:
        x = math.exp(log_x)
        return c * x ** 4 / _f(x) - EOS_MAX_LATTICE_FRACTION

    # the fraction falls monotonically from 1 at the zero-pressure point to 0.625 alpha Z^(2/3) at large x
    lo, hi = math.log(0.625 * c * 1.000001), math.log(1e6)
    if excess(hi) >= 0.0:
        raise InputError(f"lattice correction never falls below {EOS_MAX_LATTICE_FRACTION:g} for Z = {charge}")
    x_floor = math.exp(bisect(excess, lo, hi, tol=1e-13, max_iter=400))
    return field_for_pressure(DEGENERACY_PRESSURE_UNIT * (_f(x_floor) - c * x_floor ** 4))


def piston_state(field: float, mu_e: float = 2.0, charge: int | None = None) -> PistonState | None:
    """Cold degenerate state that balances the magnetic pressure of ``field``, or None where the
    equation of state should not be used.

    The field is assumed excluded from the matter (a magnetic piston, as for a conducting
    target in a pulse much shorter than its flux-diffusion time), so the matter carries
    ``P = B^2/(2 mu0)``.  x solves ``A [f(x) - c x^4] = P`` and ``rho = B mu_e x^3``.

    Returns ``None`` below :func:`eos_floor_field` (the matter is pressure-ionised but shell
    structure still shapes its equation of state, which this module does not model) and above
    :data:`NEUTRON_DRIP_FIELD` (free neutrons, not electrons, start to carry the pressure).
    ``field`` in T, ``mu_e`` = A/Z >= 1, ``charge`` the atomic number (enables the lattice
    correction).  Examples (iron, mu_e = 2.154, Z = 26): 1e8 T -> 1.4e9 kg/m^3; 1e10 T ->
    9.6e11 kg/m^3.

    Validity: T = 0 (real pulses heat the matter, lowering the density reached), fixed
    composition (above :func:`electron_capture_threshold` mu_e rises and the real density is
    higher), no field inside the matter.

    Reference: Chandrasekhar (1939); Shapiro & Teukolsky (1983) ch. 2.
    """
    b = _field(field)
    mu = _mu_e(mu_e)
    z = None if charge is None else _charge(charge)
    if b < eos_floor_field(z) or b > NEUTRON_DRIP_FIELD:
        return None
    pressure = magnetic_pressure(b)
    x = _solve_x(pressure, z)
    return PistonState(
        field=b, pressure=pressure, mu_e=mu, charge=z, fermi_parameter=x,
        density=DEGENERACY_DENSITY_UNIT * mu * x ** 3, fermi_energy=fermi_kinetic_energy(x),
        lattice_fraction=None if z is None else _lattice(z) * x ** 4 / _f(x), relativistic=x >= 1.0,
    )


def piston_density(field: float, mu_e: float = 2.0, charge: int | None = None) -> float | None:
    """Density of cold matter held by a magnetic piston at ``field`` [kg/m^3], or None below the floor.

    Shorthand for ``piston_state(field, mu_e, charge).density``; same equation and validity.

    Reference: Chandrasekhar (1939); Shapiro & Teukolsky (1983) ch. 2.
    """
    state = piston_state(field, mu_e, charge)
    return None if state is None else state.density


RELATIVISTIC_DEGENERACY_FIELD: float = field_for_pressure(DEGENERACY_PRESSURE_UNIT * _f(1.0))
"""T (1.362e8).  Field whose pressure is supported by an ideal degenerate electron gas at x = 1,
where the electrons turn relativistic (1.95e9 kg/m^3 for mu_e = 2)."""

NEUTRON_DRIP_FIELD: float = field_for_pressure(NEUTRON_DRIP_PRESSURE)
"""T (4.43e11).  Field whose pressure equals the neutron-drip pressure: above the strongest
magnetars."""

PRESSURE_IONISATION_FIELD: float = field_for_pressure(ATOMIC_PRESSURE)
"""T (8.60e3).  Field whose pressure equals the atomic unit of pressure, E_h/a0^3."""


@dataclass(frozen=True, kw_only=True)
class CaptureThreshold(ResultMixin):
    """Where cold compression starts forcing electrons into a nuclide.

    ``threshold_energy`` is the Fermi kinetic energy required, ``fermi_parameter`` and
    ``density`` the state at which the degenerate electrons reach it, ``pressure`` the
    lattice-corrected pressure there and ``field`` the magnetic piston that supplies it.

    Reference: Shapiro & Teukolsky (1983) ch. 2; Baym, Pethick & Sutherland (1971).
    """

    _title: ClassVar[str] = "Electron-capture threshold"

    nuclide: str
    daughter: str
    threshold_energy: float = unit("J")
    fermi_parameter: float
    density: float = unit("kg/m^3")
    pressure: float = unit("Pa")
    field: float = unit("T")
    lattice_fraction: float
    confidence: Confidence


def electron_capture_threshold(nuclide: str) -> CaptureThreshold:
    """Density, pressure and piston field at which a nuclide starts capturing electrons.

    Capture (A, Z) + e- -> (A, Z-1) + nu is allowed once the electron Fermi kinetic energy
    reaches the threshold ``E_t`` of :data:`NUCLIDES`::

        x = sqrt((E_t/m_e c^2 + 1)^2 - 1),  rho = B (A/Z) x^3,
        P = A [f(x) - c x^4],               B_field = sqrt(2 mu0 P)

    ``nuclide`` is a key of :data:`NUCLIDES` ("56Fe", "12C", ...).  Results: 56Fe at
    1.14e12 kg/m^3 and 1.13e10 T (a magnetar-class field); 12C at 3.90e13 kg/m^3 and 1.27e11 T;
    hydrogen at 1.22e10 kg/m^3 and 8.7e8 T; 63Cu already at 3.1e8 kg/m^3 and 2.8e7 T.

    Validity: cold matter, ground-state to ground-state threshold, equilibrium with respect to
    capture (the rate just above threshold is slow); the daughter usually captures again at once
    when its own threshold is lower.

    Reference: Shapiro & Teukolsky (1983) ch. 2; Baym, Pethick & Sutherland (1971).
    """
    if not isinstance(nuclide, str) or nuclide not in NUCLIDES:
        raise InputError(f"unknown nuclide {nuclide!r}; choose one of {', '.join(NUCLIDES)}")
    n = NUCLIDES[nuclide]
    x = _fermi_parameter_for_energy(n.threshold_energy)
    pressure = degeneracy_pressure(x, n.charge)
    return CaptureThreshold(
        nuclide=n.key, daughter=n.daughter, threshold_energy=n.threshold_energy, fermi_parameter=x,
        density=DEGENERACY_DENSITY_UNIT * n.mu_e * x ** 3, pressure=pressure,
        field=field_for_pressure(pressure), lattice_fraction=lattice_pressure_fraction(x, n.charge),
        confidence=n.confidence,
    )


@dataclass(frozen=True, kw_only=True)
class Composition(ResultMixin):
    """Nuclear make-up assumed for a registry material: Z, A, mu_e and its capture nuclide.

    ``charge`` and ``mass_number`` come from the material's base element (:data:`ELEMENTS`);
    both are None for compounds and unknown elements, where ``mu_e`` defaults to 2.
    ``nuclide`` is the key of the electron-capture record (None when none is recorded).

    Reference: :data:`ELEMENTS`; Shapiro & Teukolsky (1983) ch. 2.
    """

    material: str
    element: str
    charge: int | None
    mass_number: float | None
    mu_e: float
    nuclide: str | None


def composition(material: MaterialLike) -> Composition:
    """Atomic number, atomic weight, ``mu_e = A/Z`` and capture nuclide for a registry material.

    Uses ``Material.base_element``; alloys are treated as their base element.  Unknown elements
    and compounds return ``charge=None`` and ``mu_e = 2`` (exact for nuclei with equal numbers
    of protons and neutrons, within 15 % for anything heavier than hydrogen).

    Reference: :data:`ELEMENTS`; Shapiro & Teukolsky (1983) ch. 2.
    """
    m = as_material(material)
    entry = ELEMENTS.get(m.base_element)
    if entry is None:
        return Composition(material=m.key, element=m.base_element, charge=None, mass_number=None, mu_e=2.0,
                           nuclide=None)
    z, a = entry
    return Composition(material=m.key, element=m.base_element, charge=z, mass_number=a, mu_e=a / z,
                       nuclide=ELEMENT_NUCLIDE.get(m.base_element))


# ---------------------------------------------------------------------------
# 5. matter threaded by the field
# ---------------------------------------------------------------------------
def hydrogen_binding_energy(field: float) -> float | None:
    """Ground-state binding energy of hydrogen in a superstrong field [J], or None below validity.

    ``|E| ~ 0.16 (ln b)^2`` Hartree with ``b = B/B0``, the asymptotic result for b >> 1.
    Returns ``None`` for b < :data:`HYDROGEN_ASYMPTOTIC_MIN` (100, i.e. B < 2.35e7 T), where the
    formula is not accurate.  ``field`` in T.  13.6 eV at zero field becomes ~160 eV at 1e8 T and
    ~490 eV at 1e10 T: chemistry in a magnetar field deals in X-ray energies.

    Validity: order of magnitude (tens of per cent) for b ~ 1e2-1e6; infinitely heavy proton.

    Reference: Lai (2001) sec. III.A; Harding & Lai (2006).
    """
    b = atomic_field_ratio(field)
    if b < HYDROGEN_ASYMPTOTIC_MIN:
        return None
    return 0.16 * math.log(b) ** 2 * HARTREE


@dataclass(frozen=True, kw_only=True)
class AtomShape(ResultMixin):
    """Size and shape of an atom threaded by a strong field.

    ``transverse_radius`` ~ magnetic length (never more than a0); ``length`` ~ a0/ln b along the
    field (never more than a0); ``aspect_ratio`` = length / transverse radius (1 for b <= 1).

    Reference: Lai (2001) sec. III.
    """

    field: float = unit("T")
    atomic_field_ratio: float
    transverse_radius: float = unit("m")
    length: float = unit("m")
    aspect_ratio: float


def atom_shape(field: float) -> AtomShape:
    """Order-of-magnitude shape of a hydrogen-like atom in a field: a sphere below B0, a needle above.

    For ``b = B/B0 <= 1`` the atom keeps the Bohr radius.  For b > 1 the electron is confined
    across the field to the magnetic length ``a0/sqrt(b)`` and bound along it over
    ``a0/ln b`` (capped at a0), so the aspect ratio is ``sqrt(b)/max(ln b, 1)``: 3.4 at 1e8 T,
    19 at 1e10 T, 54 at 1e11 T.  ``field`` in T.

    Validity: scaling estimates for the ground state; factors of order one are dropped.

    Reference: Lai (2001) sec. III.A.
    """
    f = _field(field)
    b = f / ATOMIC_FIELD
    if b <= 1.0:
        return AtomShape(field=f, atomic_field_ratio=b, transverse_radius=BOHR_RADIUS, length=BOHR_RADIUS,
                         aspect_ratio=1.0)
    radius = BOHR_RADIUS / math.sqrt(b)
    length = BOHR_RADIUS / max(math.log(b), 1.0)
    return AtomShape(field=f, atomic_field_ratio=b, transverse_radius=radius, length=length,
                     aspect_ratio=length / radius)


def condensed_density_valid(field: float, charge: int) -> bool:
    """True where :func:`condensed_density` applies: ``B/B0 >= Z^(4/3)`` (ground Landau level only).

    ``field`` in T, ``charge`` the atomic number.  For iron the condition is B >= 1.8e7 T; it is
    a necessary condition, and the formula is quantitatively good only well above it.

    Reference: Lai (2001) sec. V.
    """
    return atomic_field_ratio(field) >= _charge(charge) ** (4.0 / 3.0)


def condensed_density(field: float, mass_number: float, charge: int) -> float:
    """Zero-pressure density of magnetically condensed matter [kg/m^3].

    ``rho_s ~ 561 g/cm^3 x A Z^(-3/5) B12^(6/5)`` with ``B12 = B/(1e8 T)``: the density at which
    matter threaded by the field holds itself together with no external pressure (uniform
    electron gas in the lowest Landau level, Wigner-Seitz radius ``1.90 Z^(1/5) b^(-2/5) a0``).
    ``field`` in T, ``mass_number`` A > 0, ``charge`` Z.  Iron at 1e10 T: 1.1e9 kg/m^3, a
    hundred thousand times its ordinary density; hydrogen at 1e8 T: 561 kg/m^3.

    Validity: needs ``B/B0 >> Z^(4/3)`` (:func:`condensed_density_valid`); a factor-of-two
    estimate; whether such matter is actually bound relative to isolated atoms depends on Z and B.

    Reference: Lai (2001) sec. V.B.
    """
    f = _field(field)
    a = _positive("mass_number", mass_number)
    z = _charge(charge)
    return CONDENSED_DENSITY_COEFFICIENT * a * z ** (-0.6) * (f / 1.0e8) ** 1.2


def synchrotron_cooling_time(field: float, gamma: float = 1.0) -> float:
    """Classical synchrotron cooling time of an electron gyrating in a field [s].

    ``tau = 6 pi eps0 m_e^3 c^3/(e^4 B^2 gamma) = 5.16 s/(B^2 gamma)`` with B in T and ``gamma``
    the Lorentz factor (>= 1; for gamma ~ 1 it is the decay time of the transverse energy).
    0.5 ms at 100 T; 5e-20 s at 1e10 T: in a magnetar-class field every electron drops to the
    lowest Landau level at once and can only move along the field.

    Validity: classical radiation reaction; for B approaching B_Q the transitions are quantum
    and this is an order-of-magnitude figure.

    Reference: Jackson (1999) ch. 14; Harding & Lai (2006).
    """
    f = _positive_field(field)
    g = _finite("gamma", gamma)
    if g < 1.0:
        raise InputError(f"gamma must be >= 1, got {g!r}")
    return SYNCHROTRON_COEFFICIENT / (f * f * g)


# ---------------------------------------------------------------------------
# 6. the low-field side: where the solid workpiece of the rest of the package ends
# ---------------------------------------------------------------------------
def _vaporisation(material: Material) -> tuple[float, float]:
    data = VAPORISATION.get(material.base_element)
    if data is None:
        raise InputError(f"no boiling data for '{material.key}' (base element {material.base_element!r}); "
                         f"available: {', '.join(VAPORISATION)}")
    return data


def _initial_temperature(material: Material, value: object) -> float:
    t = _positive("initial_temperature", value)
    if t >= material.melting_point:
        raise InputError(f"initial_temperature {t:g} K is not below the melting point of {material.key}")
    return t


def skin_boiling_field(material: MaterialLike, initial_temperature: float = T_ROOM, kappa: float = 1.0,
                       include_latent: bool = False) -> float:
    """Peak field at which the eddy-current skin of a metal reaches its boiling point [T].

    ``B = sqrt(2 mu0 H / kappa)`` with ``H = rho (c_p (T_b - T_0) + L_f)`` (the latent heat of
    fusion only when ``include_latent``), the skin-heating balance of
    :func:`magform.eddy.skin_temperature_rise` carried on from melting to boiling.  ``kappa`` as
    in :func:`magform.eddy.surface_heating_energy_density` (1 = lower estimate).  Copper: 148 T
    (164 T with the latent heat), matching Herlach's "boils near 150 T"; aluminium 123 T; iron
    159 T; tungsten 189 T.

    Validity: adiabatic thick-conductor skin, constant solid-phase c_p extended through the
    liquid, boiling point at 1 atm (the magnetic pressure raises it); data for Cu, Al, Fe and W
    base metals only (:data:`VAPORISATION`).

    Reference: Herlach (1986); Knoepfel as cited in :mod:`magform.eddy`.
    """
    m = as_material(material)
    t0 = _initial_temperature(m, initial_temperature)
    k = _positive("kappa", kappa)
    t_boil, _ = _vaporisation(m)
    specific = m.specific_heat * (t_boil - t0)
    if include_latent:
        m.require("latent_heat_fusion")
        specific += float(m.latent_heat_fusion)  # type: ignore[arg-type]
    return math.sqrt(2.0 * MU0 * m.density * specific / k)


def skin_vaporisation_field(material: MaterialLike, initial_temperature: float = T_ROOM, kappa: float = 1.0) -> float:
    """Peak field at which the eddy-current skin is completely vaporised [T].

    ``B = sqrt(2 mu0 rho (c_p (T_b - T_0) + L_f + H_vap) / kappa)``: the skin has absorbed
    enough to heat, melt and boil off entirely.  Copper 365 T, aluminium 303 T, iron 388 T,
    tungsten 510 T: above a few hundred tesla every metal surface leaves as vapour within the
    pulse, which is why megagauss coils are single-shot.

    Validity: as :func:`skin_boiling_field`; latent heat of fusion taken as 0 when not recorded.

    Reference: Herlach (1986); Fowler, Garn & Caird (1960).
    """
    m = as_material(material)
    t0 = _initial_temperature(m, initial_temperature)
    k = _positive("kappa", kappa)
    t_boil, h_vap = _vaporisation(m)
    fusion = float(m.latent_heat_fusion) if m.latent_heat_fusion is not None else 0.0
    specific = m.specific_heat * (t_boil - t0) + fusion + h_vap
    return math.sqrt(2.0 * MU0 * m.density * specific / k)


def bulk_modulus_field(material: MaterialLike) -> float:
    """Field whose magnetic pressure equals the bulk modulus, ``B = sqrt(2 mu0 K)`` [T].

    Above it the linear-elastic picture of a slightly squeezed solid is gone: the pressure
    compresses the metal by a large fraction of its volume.  Copper 553 T, iron 649 T,
    tungsten 885 T.  Megagauss flux compression (1200-2800 T) works here.

    Reference: Fowler, Garn & Caird (1960); Nakamura et al. (2018).
    """
    m = as_material(material)
    return field_for_pressure(float(m.bulk_modulus))  # type: ignore[arg-type]


def yield_field(material: MaterialLike) -> float:
    """Field whose magnetic pressure equals the static yield strength, ``B = sqrt(2 mu0 sigma_y)`` [T].

    The bottom rung of the ladder: annealed copper 13 T, Armco iron 21 T, wrought tungsten 57 T.
    Raises :class:`InputError` for materials that do not yield.  Flow under a real die needs
    the constraint factor and rate hardening of :mod:`magform.plasticity` on top.

    Reference: Psyk et al. (2011) as cited in :mod:`magform.eddy`.
    """
    m = as_material(material)
    m.require("yield_strength")
    return field_for_pressure(float(m.yield_strength))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. regime classifier
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Threshold(ResultMixin):
    """One boundary on the way from a formed part to neutron-star matter.

    ``field`` is where the boundary sits for this material, ``pressure`` its B^2/(2 mu0),
    ``crossed`` whether the assessed field reaches it (None when no field was given) and
    ``state`` the condition of the workpiece just above it.  ``kind`` says what is being
    compared: "strength" (pressure against yield), "heating" (eddy-current skin energy),
    "pressure" (pressure against a property of matter) or "field" (the field against a
    quantum scale, whatever the matter).

    Reference: see :func:`regime_thresholds`.
    """

    key: str
    kind: str
    label: str
    field: float = unit("T")
    pressure: float = unit("Pa")
    crossed: bool | None
    state: str


_THRESHOLD_KINDS: dict[str, str] = {
    "plastic_flow": "strength", "skin_melting": "heating", "skin_boiling": "heating", "skin_vaporised": "heating",
    "bulk_modulus": "pressure", "pressure_ionisation": "pressure", "electron_capture": "pressure",
    "relativistic_degeneracy": "pressure", "neutron_drip": "pressure", "atomic_field": "field",
    "qed_critical": "field",
}

_THRESHOLD_LABELS: dict[str, str] = {
    "plastic_flow": "pressure reaches the static yield strength",
    "skin_melting": "eddy-current skin starts to melt",
    "skin_boiling": "skin reaches its boiling point",
    "skin_vaporised": "skin completely vaporised",
    "bulk_modulus": "pressure equals the bulk modulus",
    "pressure_ionisation": "pressure equals the atomic unit E_h/a0^3",
    "atomic_field": "field equals the atomic unit B0",
    "electron_capture": "degenerate electrons reach the capture threshold",
    "relativistic_degeneracy": "degenerate electrons turn relativistic (x = 1)",
    "qed_critical": "field equals the QED critical field",
    "neutron_drip": "pressure equals the neutron-drip pressure",
}


def regime_thresholds(material: MaterialLike = "iron_armco", field: float | None = None) -> tuple[Threshold, ...]:
    """Ordered boundaries a workpiece crosses as the field rises from tens of tesla to 1e11 T and beyond.

    For ``material`` (default Armco iron) the fields of, where data exist: yield
    (:func:`yield_field`), skin melting (:func:`magform.eddy.skin_melting_field`), skin boiling
    and vaporisation (:func:`skin_boiling_field`, :func:`skin_vaporisation_field`), bulk-modulus
    compression (:func:`bulk_modulus_field`), pressure ionisation (8.6e3 T), the atomic unit
    of field (2.35e5 T), electron capture on the material's nuclide
    (:func:`electron_capture_threshold`), relativistic degeneracy (1.36e8 T), the QED critical
    field (4.41e9 T) and neutron drip (4.43e11 T), sorted by field.  With ``field`` each is
    marked crossed or not.

    Validity: each boundary is an order-of-magnitude marker computed from a single balance;
    real transitions are gradual and depend on pulse length and geometry.

    Reference: Herlach (1986); Lai (2001); Shapiro & Teukolsky (1983); Baym, Pethick &
    Sutherland (1971); Schwinger (1951).
    """
    m = as_material(material)
    f = None if field is None else _field(field)
    comp = composition(m)
    marks: list[tuple[str, float, str]] = []
    if m.yield_strength is not None and m.yield_strength > 0.0:
        marks.append(("plastic_flow", yield_field(m), _STATE_TEXT["plastic_flow"]))
    if m.is_metal and T_ROOM < m.melting_point:
        marks.append(("skin_melting", skin_melting_field(m), _STATE_TEXT["skin_melting"]))
    if m.base_element in VAPORISATION:
        marks.append(("skin_boiling", skin_boiling_field(m), _STATE_TEXT["skin_boiling"]))
        marks.append(("skin_vaporised", skin_vaporisation_field(m), _STATE_TEXT["skin_vaporised"]))
    marks.append(("bulk_modulus", bulk_modulus_field(m), _STATE_TEXT["bulk_modulus"]))
    marks.append(("pressure_ionisation", PRESSURE_IONISATION_FIELD, _STATE_TEXT["pressure_ionisation"]))
    marks.append(("atomic_field", ATOMIC_FIELD, _STATE_TEXT["atomic_field"]))
    if comp.nuclide is not None:
        capture = electron_capture_threshold(comp.nuclide)
        state = (f"degenerate electrons are forced into nuclei: {capture.nuclide} starts turning into "
                 f"{capture.daughter} and the workpiece is being transmuted")
        marks.append(("electron_capture", capture.field, state))
    marks.append(("relativistic_degeneracy", RELATIVISTIC_DEGENERACY_FIELD, _STATE_TEXT["relativistic_degeneracy"]))
    marks.append(("qed_critical", QED_CRITICAL_FIELD, _STATE_TEXT["qed_critical"]))
    marks.append(("neutron_drip", NEUTRON_DRIP_FIELD, _STATE_TEXT["neutron_drip"]))
    marks.sort(key=lambda item: item[1])
    return tuple(Threshold(key=key, kind=_THRESHOLD_KINDS[key], label=_THRESHOLD_LABELS[key], field=b,
                           pressure=magnetic_pressure(b),
                           crossed=None if f is None else f >= b, state=state)
                 for key, b, state in marks)


@dataclass(frozen=True, kw_only=True)
class Regime(ResultMixin):
    """Where a workpiece stands on the ladder at a given field.

    ``key`` is the highest boundary crossed ("elastic" if none); ``matter_key`` the highest one
    that concerns the matter rather than the field itself; ``state`` describes the workpiece in
    plain language (its matter state, plus what the field does where it threads the matter once
    the atomic unit or the QED critical field is passed); ``tier`` is the coarser process tier of
    :data:`TIERS`; ``crossed`` lists every
    boundary passed; ``next_key``/``next_field`` the next one up (None at the top);
    ``piston_density`` is the cold degenerate density if the field acts as a piston (None where
    the equation of state is not offered) and ``compression`` its ratio to the ordinary density.

    Reference: see :func:`regime_thresholds`.
    """

    _title: ClassVar[str] = "Workpiece regime"

    material: str
    field: float = unit("T")
    pressure: float = unit("Pa")
    key: str
    matter_key: str
    state: str
    tier: str
    crossed: tuple[str, ...]
    next_key: str | None
    next_field: float | None = unit("T")
    piston_density: float | None = unit("kg/m^3")
    compression: float | None
    thresholds: tuple[Threshold, ...]


def _tier(material: Material, field: float) -> str:
    """Process tier: how many of the five tier boundaries the field has passed."""
    bulk = bulk_modulus_field(material)
    # a non-conductor has no eddy-current skin: it goes straight from solid to megabar compression
    first = min(skin_melting_field(material), bulk) if material.is_metal else bulk
    boundaries = (first, bulk, PRESSURE_IONISATION_FIELD, ATOMIC_FIELD, RELATIVISTIC_DEGENERACY_FIELD)
    return TIERS[sum(field >= b for b in boundaries)]


def classify_regime(material: MaterialLike, field: float) -> Regime:
    """State of a workpiece of ``material`` exposed to ``field`` (as a piston pressure B^2/(2 mu0)).

    Marks the boundaries of :func:`regime_thresholds`, reports the highest one crossed with its
    plain-language state, the process tier (:data:`TIERS`: solid, liquid_skin, megabar, plasma,
    landau, degenerate) and, where the equation of state applies, the cold degenerate density.
    ``field`` in T.  Iron: 100 T -> plastic flow below the skin-melting field (the concept
    note's regime); 2800 T -> megabar compression; 1e10 T -> relativistically degenerate matter
    at 9.5e11 kg/m^3 in a field beyond the QED critical value, just short of turning into
    manganese.

    Validity: as :func:`regime_thresholds` and :func:`piston_state`.

    Reference: Lai (2001); Shapiro & Teukolsky (1983); Herlach (1986).
    """
    m = as_material(material)
    f = _field(field)
    marks = regime_thresholds(m, f)
    passed = [t for t in marks if t.crossed]
    crossed = tuple(t.key for t in passed)
    key = passed[-1].key if passed else "elastic"
    matter = [t for t in passed if t.kind != "field"]
    matter_key = matter[-1].key if matter else "elastic"
    state = matter[-1].state if matter else _STATE_TEXT["elastic"]
    capture = next((item for item in passed if item.key == "electron_capture"), None)
    if capture is not None and matter_key not in ("electron_capture", "neutron_drip"):
        state += "; " + capture.state          # capture began lower down (copper) and is still going on
    if "qed_critical" in crossed:
        state += "; " + _STATE_TEXT["qed_critical"]
    elif "atomic_field" in crossed:
        state += "; " + _STATE_TEXT["atomic_field"]
    upcoming = next((t for t in marks if not t.crossed), None)
    comp = composition(m)
    piston = piston_state(f, comp.mu_e, comp.charge)
    density = None if piston is None else piston.density
    return Regime(
        material=m.key, field=f, pressure=magnetic_pressure(f), key=key, matter_key=matter_key, state=state,
        tier=_tier(m, f), crossed=crossed, next_key=None if upcoming is None else upcoming.key,
        next_field=None if upcoming is None else upcoming.field,
        piston_density=density, compression=None if density is None else density / m.density,
        thresholds=marks,
    )


def process_outcomes(material: MaterialLike, field: float) -> dict[str, str]:
    """What each of the concept note's six applications becomes at ``field``, one sentence each.

    Keys are :data:`APPLICATIONS` (forging, refractory_forming, welding, synthesis,
    densification, containment); the sentence is chosen by the process tier of
    :func:`classify_regime`.  Below the skin-melting field the answer is the rest of the
    package; above it every solid-state process is replaced by its violent successor.

    Reference: Fowler, Garn & Caird (1960); Lai (2001); Shapiro & Teukolsky (1983); the
    modules named in each sentence.
    """
    regime = classify_regime(material, field)
    index = TIERS.index(regime.tier)
    return {name: _PROCESS_TEXT[name][index] for name in APPLICATIONS}


# ---------------------------------------------------------------------------
# 8. the assumed source
# ---------------------------------------------------------------------------
def chandrasekhar_fermi_field(mass: float, radius: float) -> float:
    """Largest field a self-gravitating body can hold, from the virial theorem [T].

    Magnetic energy inside a uniform sphere must not exceed its gravitational binding energy,
    ``(B^2/(2 mu0)) (4 pi R^3/3) <= (3/5) G M^2/R``, so::

        B_max = sqrt(18 mu0 G/(20 pi)) M/R^2 = 4.90e-9 T m^2/kg x M/R^2

    ``mass`` in kg, ``radius`` in m.  A 1.4 solar-mass, 12 km neutron star: 9.5e13 T (1e18 G),
    a thousand times a magnetar's surface field.  The Sun: 2e4 T.  The Earth: 720 T.

    Validity: uniform density and field, interior energy only; an order-of-magnitude ceiling.

    Reference: Chandrasekhar & Fermi (1953); Lai (2001) sec. I.
    """
    m = _positive("mass", mass)
    r = _positive("radius", radius)
    return math.sqrt(18.0 * MU0 * GRAVITATIONAL_CONSTANT / (20.0 * math.pi)) * m / (r * r)


@dataclass(frozen=True, kw_only=True)
class GravitationalHolder(ResultMixin):
    """The smallest self-gravitating body that could hold a field statically.

    ``virial_radius``/``virial_mass``: the ball of matter at ``density`` whose
    Chandrasekhar-Fermi limit equals ``field``.  ``self_bound`` is False when that ball is
    lighter than the minimum neutron-star mass (its own gravity could not keep it at that
    density), in which case ``holder_mass`` is the minimum neutron-star mass.
    ``neutron_star_limit`` is the Chandrasekhar-Fermi field of the canonical neutron star.

    Reference: Chandrasekhar & Fermi (1953); Baym, Pethick & Sutherland (1971).
    """

    field: float = unit("T")
    density: float = unit("kg/m^3")
    virial_radius: float = unit("m")
    virial_mass: float = unit("kg")
    self_bound: bool
    holder_mass: float = unit("kg")
    holder_solar_masses: float
    neutron_star_limit: float = unit("T")


def gravitational_holder(field: float, density: float = NUCLEAR_DENSITY) -> GravitationalHolder:
    """The body gravity would need to hold ``field`` statically: in practice, a neutron star.

    At fixed ``density`` rho the Chandrasekhar-Fermi limit is
    ``B_max = k (4 pi/3) rho R`` (k = 4.90e-9 T m^2/kg), so the ball that just holds ``field``
    has ``R = B/(k (4 pi/3) rho)``.  At nuclear density a 1e10 T field needs only a 2 m, 9e18 kg
    ball by this arithmetic, but matter stays at nuclear density only under the gravity of
    about 0.1 solar masses or more, so the smallest real holder is a neutron star.  Even there
    the field is held by gravity, not by strength: the crust, the strongest material known,
    cracks under magnetar fields, which is what giant flares are thought to be.

    ``field`` in T (> 0), ``density`` in kg/m^3 (> 0, default nuclear saturation density).

    Reference: Chandrasekhar & Fermi (1953); Baym, Pethick & Sutherland (1971); Horowitz &
    Kadau (2009); Hurley et al. (2005).
    """
    f = _positive_field(field)
    rho = _positive("density", density)
    k = math.sqrt(18.0 * MU0 * GRAVITATIONAL_CONSTANT / (20.0 * math.pi))
    radius = f / (k * (4.0 * math.pi / 3.0) * rho)
    mass = (4.0 * math.pi / 3.0) * rho * radius ** 3
    bound = mass >= MIN_NEUTRON_STAR_MASS
    holder = mass if bound else MIN_NEUTRON_STAR_MASS
    return GravitationalHolder(
        field=f, density=rho, virial_radius=radius, virial_mass=mass, self_bound=bound, holder_mass=holder,
        holder_solar_masses=holder / SOLAR_MASS,
        neutron_star_limit=chandrasekhar_fermi_field(*CANONICAL_NEUTRON_STAR),
    )


def solar_collection_time(energy: float, area: float = REFERENCE_COLLECTOR_AREA, efficiency: float = 1.0) -> float:
    """Time for a solar collector at 1 au to bank an energy, ``t = E/(S A eta)`` [s].

    ``energy`` in J (>= 0), ``area`` in m^2 (> 0, default one square kilometre), ``efficiency``
    in (0, 1]; S = 1361 W/m^2.  One cubic millimetre of 1e10 T field (4e16 J) takes a square
    kilometre 0.93 years at perfect efficiency.

    Reference: total solar irradiance, 1361 W/m^2.
    """
    e = _non_negative("energy", energy)
    a = _positive("area", area)
    eta = _finite("efficiency", efficiency)
    if not 0.0 < eta <= 1.0:
        raise InputError(f"efficiency must lie in (0, 1], got {eta!r}")
    return e / (SOLAR_CONSTANT * a * eta)


def dipole_axial_field(surface_field: float, radius: float, distance: float) -> float:
    """Field on the axis of a dipole source at ``distance`` from its centre, ``B = B_s (R/r)^3`` [T].

    ``surface_field`` is the polar surface field of a uniformly magnetised sphere of ``radius``
    (equal to its interior field); ``distance`` >= radius, in m.  On the equator the field is
    half this.

    Reference: Jackson (1999) sec. 5.6 and 5.10.
    """
    b = _field(surface_field, "surface_field")
    r0 = _positive("radius", radius)
    r = _positive("distance", distance)
    if r < r0 * (1.0 - 1e-12):
        raise InputError(f"distance ({r!r} m) is inside the source (radius {r0!r} m)")
    return b * (r0 / r) ** 3


def standoff_radius(surface_field: float, radius: float, threshold_field: float) -> float:
    """Distance at which the axial dipole field falls to ``threshold_field``, ``r = R (B_s/B_th)^(1/3)`` [m].

    Never less than ``radius`` (the threshold is not reached outside a weaker source).  A real
    magnetar (1e11 T, 10 km) holds the atomic-unit field out to 750 km; a 10 mm, 1e10 T source
    holds the 0.5 mT pacemaker line at 270 m.

    Reference: Jackson (1999) sec. 5.6; ICNIRP (2009).
    """
    b = _field(surface_field, "surface_field")
    r0 = _positive("radius", radius)
    th = _positive("threshold_field", threshold_field)
    if b <= th:
        return r0
    return r0 * (b / th) ** (1.0 / 3.0)


def standoff_radii(surface_field: float, radius: float) -> dict[str, float]:
    """Stand-off distance for every entry of :data:`FIELD_THRESHOLDS`, keyed by its key [m].

    ``r = R (B_s/B_th)^(1/3)`` on the dipole axis (:func:`standoff_radius`).  For a 10 mm,
    1e10 T source: Earth-field 585 m, pacemaker line 271 m, public limit 29 m, occupational
    limits 17 m and 11 m, atomic unit 0.35 m.

    Reference: ICNIRP (2009); Lai (2001); Jackson (1999) sec. 5.6.
    """
    return {t.key: standoff_radius(surface_field, radius, t.field) for t in FIELD_THRESHOLDS}


def levitation_gradient(susceptibility: float = WATER_SUSCEPTIBILITY, density: float = WATER_DENSITY) -> float:
    """Field-gradient product that balances Earth gravity on a weakly magnetic body [T^2/m].

    ``B |dB/dz| = mu0 rho g0/|chi|``: 1363 T^2/m for water with the constants of this module,
    the ~1400 T^2/m of the levitating-frog experiment.  ``susceptibility`` is the SI volume
    susceptibility (non-zero), ``density`` in kg/m^3.

    Reference: Berry & Geim (1997).
    """
    chi = _finite("susceptibility", susceptibility)
    if chi == 0.0:
        raise InputError("susceptibility must be non-zero")
    return MU0 * _positive("density", density) * G_ACCEL / abs(chi)


def diamagnetic_acceleration(surface_field: float, radius: float, distance: float,
                             susceptibility: float = WATER_SUSCEPTIBILITY, density: float = WATER_DENSITY) -> float:
    """Acceleration of a weakly magnetic body on the axis of a dipole source [m/s^2].

    Force density ``(chi/mu0) B dB/dr`` with ``B = B_s (R/r)^3`` gives
    ``a = 3 |chi| B(r)^2/(mu0 rho r)``, directed away from the source for a diamagnet
    (chi < 0, e.g. water, tissue) and towards it for a paramagnet.  ``susceptibility`` is the SI
    volume susceptibility, ``density`` in kg/m^3.  Water 1 m from a 10 mm, 1e10 T source (where
    the field is still 1e4 T): 2.2e6 m/s^2 outward, 2e5 times Earth gravity.

    Validity: |chi| << 1 (no saturation), body small compared with ``distance``.

    Reference: Berry & Geim (1997).
    """
    chi = _finite("susceptibility", susceptibility)
    rho = _positive("density", density)
    r = _positive("distance", distance)
    b = dipole_axial_field(surface_field, radius, r)
    return 3.0 * abs(chi) * b * b / (MU0 * rho * r)


def levitation_radius(surface_field: float, radius: float, gradient: float = WATER_LEVITATION_GRADIENT) -> float:
    """Distance inside which ``B |dB/dr|`` on the dipole axis exceeds ``gradient`` [m].

    ``B dB/dr = 3 B_s^2 R^6/r^7``, so ``r = (3 B_s^2 R^6/gradient)^(1/7)`` (never less than
    ``radius``).  With the default 1400 T^2/m (water levitates against Earth gravity) a 10 mm,
    1e10 T source pushes water away harder than the Earth pulls inside 5.8 m.

    Reference: Berry & Geim (1997).
    """
    b = _field(surface_field, "surface_field")
    r0 = _positive("radius", radius)
    g = _positive("gradient", gradient)
    if b == 0.0:
        return r0
    return max(r0, (3.0 * b * b * r0 ** 6 / g) ** (1.0 / 7.0))


def radiant_fluence(energy: float, distance: float) -> float:
    """Energy per unit area at ``distance`` if ``energy`` is released isotropically, ``E/(4 pi d^2)`` [J/m^2].

    ``energy`` in J (>= 0), ``distance`` in m (> 0).  In vacuum nothing absorbs the release on
    the way, so this is what a surface facing the source receives.

    Reference: inverse-square law.
    """
    e = _non_negative("energy", energy)
    d = _positive("distance", distance)
    return e / (4.0 * math.pi * d * d)


def fluence_radius(energy: float, fluence: float = DAMAGE_FLUENCE) -> float:
    """Distance at which an isotropic release of ``energy`` falls to ``fluence``, ``sqrt(E/(4 pi F))`` [m].

    Default ``fluence`` is :data:`DAMAGE_FLUENCE` (1e6 J/m^2, estimated).  With F = 1361 J/m^2
    it is the distance at which the shot delivers one second of sunlight.

    Reference: inverse-square law; total solar irradiance 1361 W/m^2.
    """
    e = _non_negative("energy", energy)
    f = _positive("fluence", fluence)
    return math.sqrt(e / (4.0 * math.pi * f))


@dataclass(frozen=True)
class MagnetarSource(ResultMixin):
    """The assumed source: a uniformly magnetised sphere with a magnetar-class field inside.

    ``field`` [T] is the uniform interior field, which equals the polar surface field of the
    dipole outside; ``radius`` [m] the size of the working volume; ``pulse_duration`` [s] how
    long the field is applied (None = no longer than it takes to fly apart, R/c).  The class
    makes no claim that such a source can exist; its methods compute what the premise implies.

    Reference: Jackson (1999) sec. 5.10 (uniformly magnetised sphere); Moon (1982);
    Chandrasekhar & Fermi (1953).
    """

    field: float = unit("T")
    radius: float = unit("m")
    pulse_duration: float | None = unit("s", default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _positive_field(self.field))
        object.__setattr__(self, "radius", _positive("radius", self.radius))
        if self.pulse_duration is not None:
            object.__setattr__(self, "pulse_duration", _positive("pulse_duration", self.pulse_duration))

    def volume(self) -> float:
        """Working volume ``(4/3) pi R^3`` [m^3].

        Reference: geometry.
        """
        return 4.0 * math.pi * self.radius ** 3 / 3.0

    def dipole_moment(self) -> float:
        """Magnetic moment ``m = 2 pi R^3 B/mu0`` [A m^2] of the equivalent dipole.

        Reference: Jackson (1999) sec. 5.10.
        """
        return 2.0 * math.pi * self.radius ** 3 * self.field / MU0

    def interior_energy(self) -> float:
        """Field energy inside the sphere, ``(B^2/(2 mu0)) (4/3) pi R^3`` [J].

        Reference: Jackson (1999) sec. 6.7.
        """
        return magnetic_pressure(self.field) * self.volume()

    def exterior_energy(self) -> float:
        """Energy of the dipole field outside the sphere, ``mu0 m^2/(12 pi R^3) = pi R^3 B^2/(3 mu0)`` [J].

        Exactly half the interior energy.

        Reference: Jackson (1999) sec. 5.6.
        """
        return math.pi * self.radius ** 3 * self.field ** 2 / (3.0 * MU0)

    def energy(self) -> float:
        """Total field energy, interior plus exterior, ``pi R^3 B^2/mu0`` [J].

        A 10 mm, 1e10 T source: 2.5e20 J (6e4 megatons, 40 % of a year of world energy use);
        1 mm: 2.5e17 J (60 megatons); 1 um: 2.5e8 J.

        Reference: Jackson (1999) sec. 5.6 and 6.7.
        """
        return self.interior_energy() + self.exterior_energy()

    def megatons(self) -> float:
        """Total field energy in megatons of TNT.

        Reference: definition of the megaton.
        """
        return self.energy() / MEGATON_TNT

    def mass_equivalent(self) -> float:
        """Mass equivalent of the field energy, ``E/c^2`` [kg].

        Reference: E = m c^2.
        """
        return self.energy() / C_LIGHT ** 2

    def confinement_time(self) -> float:
        """Light-crossing time ``R/c`` [s]: how long the field lasts if nothing holds it.

        A field in vacuum is not self-confining (virial theorem), so without a holder it
        disperses at the speed of light.  33 ps for 10 mm, 3.3 fs for 1 um.

        Reference: Chandrasekhar & Fermi (1953); Moon (1982).
        """
        return self.radius / C_LIGHT

    def assembly_power(self) -> float:
        """Power needed to assemble the field within its own confinement time, ``E c/R`` [W].

        For 10 mm and 1e10 T: 7.5e30 W, twenty thousand solar luminosities.

        Reference: derived from :meth:`energy` and :meth:`confinement_time`.
        """
        return self.energy() / self.confinement_time()

    def pulse_power(self) -> float | None:
        """Mean power over ``pulse_duration``, ``E/t`` [W] (None when no duration is given).

        Reference: derived from :meth:`energy`.
        """
        return None if self.pulse_duration is None else self.energy() / self.pulse_duration

    def held(self) -> bool:
        """True when ``pulse_duration`` exceeds R/c, i.e. something has to hold the field.

        Reference: Moon (1982).
        """
        return self.pulse_duration is not None and self.pulse_duration > self.confinement_time()

    def virial_mass(self, material: MaterialLike = "carbon_fibre_t1100g") -> float:
        """Least structural mass that could hold the field statically, ``rho E/sigma`` [kg].

        Virial theorem (:func:`magform.magnetics.virial_minimum_mass`) with the total field
        energy and the ultimate (or yield) strength and density of ``material``.  Default: the
        strongest commercial carbon fibre.  One cubic-micrometre-scale source (R = 1 um) at
        1e10 T needs 64 kg of it; R = 10 mm needs 6e13 kg.

        Reference: Moon (1982).
        """
        m = as_material(material)
        strength = m.ultimate_strength if m.ultimate_strength is not None else m.yield_strength
        if strength is None:
            raise InputError(f"material '{m.key}' has no strength for a virial estimate")
        return virial_minimum_mass(self.energy(), m.density, float(strength))

    def axial_field(self, distance: float) -> float:
        """Dipole field on the axis at ``distance`` from the centre [T] (:func:`dipole_axial_field`).

        Reference: Jackson (1999) sec. 5.6.
        """
        return dipole_axial_field(self.field, self.radius, distance)

    def standoff(self) -> dict[str, float]:
        """Stand-off radii for :data:`FIELD_THRESHOLDS` [m] (:func:`standoff_radii`).

        Reference: ICNIRP (2009); Lai (2001).
        """
        return standoff_radii(self.field, self.radius)


# ---------------------------------------------------------------------------
# 10. the assessment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class MagnetarAssessment(Assessment):
    """Consequences of an assumed on-demand magnetar-class field (``status`` is always ASSUMED).

    Field scales: ``pressure``, ``energy_density``, ``mass_equivalent_density``,
    ``atomic_field_ratio`` (b), ``qed_field_ratio``, ``cyclotron_energy`` (non-relativistic),
    ``landau_excitation_energy``, ``magnetic_length``, ``synchrotron_cooling_time``.
    Source: ``interior_energy``, ``field_energy`` (interior + exterior) and
    ``field_energy_megatons``, ``mass_equivalent``, ``confinement_time`` (R/c),
    ``required_power`` (E c/R), ``pulse_power``/``held`` when a duration was given,
    ``virial_mass_fibre``/``virial_mass_graphene``, ``holder`` (gravity), ``solar_collection_time``
    for one square kilometre.  Gaps to the record book in ``gaps`` (ratios; > 1 = beyond the
    record) and ``laboratory_record`` (strongest laboratory field in at least this volume for at
    least the pulse duration, or R/c when none is given).  Matter: ``regime`` (state, tier,
    thresholds), ``compressed_density`` (piston; None where not modelled), ``condensed_density``
    (threaded; None where not valid),
    ``hydrogen_binding_energy``, ``atom_aspect_ratio``.  Space: ``standoff`` radii,
    ``levitation_radius``, ``damage_radius``, ``solar_second_radius``.  ``processes`` maps the
    six applications to what they become.

    Reference: Lai (2001); Shapiro & Teukolsky (1983); Chandrasekhar & Fermi (1953);
    Kaspi & Beloborodov (2017).
    """

    _title: ClassVar[str] = "Magnetar-mode thought experiment"

    premise: str
    material: str
    field: float = unit("T")
    radius: float = unit("m")
    pulse_duration: float | None = unit("s")
    pressure: float = unit("Pa")
    energy_density: float = unit("J/m^3")
    mass_equivalent_density: float = unit("kg/m^3")
    atomic_field_ratio: float
    qed_field_ratio: float
    cyclotron_energy: float = unit("J")
    landau_excitation_energy: float = unit("J")
    magnetic_length: float = unit("m")
    synchrotron_cooling_time: float = unit("s")
    interior_energy: float = unit("J")
    field_energy: float = unit("J")
    field_energy_megatons: float = unit("Mt")
    mass_equivalent: float = unit("kg")
    confinement_time: float = unit("s")
    required_power: float = unit("W")
    pulse_power: float | None = unit("W")
    held: bool
    virial_mass_fibre: float = unit("kg")
    virial_mass_graphene: float = unit("kg")
    holder: GravitationalHolder
    solar_collection_time: float = unit("s")
    gaps: dict[str, float]
    laboratory_record: str | None
    laboratory_record_field: float | None = unit("T")
    regime: Regime
    state: str
    thresholds_crossed: tuple[str, ...]
    compressed_density: float | None = unit("kg/m^3")
    condensed_density: float | None = unit("kg/m^3")
    hydrogen_binding_energy: float | None = unit("J")
    atom_aspect_ratio: float
    standoff: dict[str, float]
    levitation_radius: float = unit("m")
    damage_radius: float = unit("m")
    solar_second_radius: float = unit("m")
    processes: dict[str, str]


PREMISE: str = ("A magnetar-class field is available on demand at a space-based source. This is assumed, "
                "not assessed.")
"""The premise every :class:`MagnetarAssessment` carries."""


def assess_magnetar_mode(field: float, radius: float, material: MaterialLike = "iron_armco",
                         pulse_duration: float | None = None) -> MagnetarAssessment:
    """Work out what follows if a field of ``field`` tesla could be applied on demand in space.

    The source (a uniformly magnetised sphere of ``radius`` metres, :class:`MagnetarSource`) is
    granted by premise: ``status`` is ``Status.ASSUMED`` and ``margin`` is None.  Everything else
    is computed: the field scales (``B^2/(2 mu0)``, ``B/B0``, ``B/B_Q``), the energy of a shot
    (``pi R^3 B^2/mu0``) and the power to assemble it in R/c, the gaps to today's records, what
    would have to hold it (virial mass; a neutron star), what the field does to ``material``
    (:func:`classify_regime`, :func:`piston_state`, :func:`condensed_density`), the stand-off
    distances that make the space setting a requirement, and what the concept note's six
    applications turn into (:func:`process_outcomes`).

    ``field`` in T (non-zero), ``radius`` in m (> 0), ``material`` a registry key or Material
    (default Armco iron), ``pulse_duration`` in s (> 0) or None.

    Notes always state that the premise is assumed and name the nearest real things: megagauss
    flux compression and Z-pinch compression at 1e3-1e4 T, laser plasmas at ~1e5 T, heavy-ion
    collisions at ~1e14 T.  Warnings carry the consequences that would stop a real mission.

    Validity: order-of-magnitude physics of cold matter in a prescribed field; no model of the
    source, of the pulse shape, of heating or of radiation transport.

    Reference: Lai (2001); Harding & Lai (2006); Shapiro & Teukolsky (1983); Chandrasekhar &
    Fermi (1953); Kaspi & Beloborodov (2017); Duncan & Thompson (1992).
    """
    m = as_material(material)
    source = MagnetarSource(field, radius, pulse_duration)
    b, r0 = source.field, source.radius
    pressure = magnetic_pressure(b)
    energy = source.energy()
    power = source.assembly_power()
    regime = classify_regime(m, b)
    comp = composition(m)
    condensed = None
    if comp.charge is not None and comp.mass_number is not None and condensed_density_valid(b, comp.charge):
        condensed = condensed_density(b, comp.mass_number, comp.charge)
    applied_for = source.pulse_duration if source.pulse_duration is not None else source.confinement_time()
    record = laboratory_record(source.volume(), applied_for)
    holder = gravitational_holder(b)
    gaps = {
        "field_over_nondestructive_record": b / benchmarks.PULSED_NONDESTRUCTIVE_RECORD,
        "field_over_destructive_record": b / DESTRUCTIVE_FIELD_RECORD,
        "pressure_over_concept_note": pressure / CONCEPT_PRESSURE,
        "energy_over_largest_pulsed_store": energy / benchmarks.ENERGY_ANY_STORE_RECORD,
        "power_over_pulsed_record": power / benchmarks.PEAK_POWER_RECORD,
        "power_over_largest_laser": power / LASER_PEAK_POWER,
    }
    fibre = source.virial_mass("carbon_fibre_t1100g")
    graphene = source.virial_mass("graphene_intrinsic")
    damage = fluence_radius(energy, DAMAGE_FLUENCE)
    solar_second = fluence_radius(energy, SOLAR_CONSTANT)
    standoff = source.standoff()
    shape = atom_shape(b)

    notes: list[str] = [
        PREMISE,
        "Nearest real things: megagauss flux compression (1200-2800 T) and Z-pinch isentropic compression "
        "(~1e3-1e4 T) for microseconds in millimetre volumes; laser-produced plasmas (~3e4-7e4 T) for "
        "picoseconds over micrometres; heavy-ion collisions (~3e14 T) for ~1e-23 s over femtometres; and "
        "magnetars themselves (1e10-1e11 T), the nearest thousands of light-years away.",
        f"Field scales: {_sci(b)} T is {_sci(shape.atomic_field_ratio)} atomic units and "
        f"{_sci(b / QED_CRITICAL_FIELD)} of the QED critical field; pressure {_sci(pressure)} Pa, "
        f"{_sci(pressure / CONCEPT_PRESSURE)} times the concept note's 4 GPa; mass-equivalent density "
        f"{_sci(pressure / C_LIGHT ** 2)} kg/m^3.",
        f"One shot holds {_sci(energy)} J ({_sci(energy / MEGATON_TNT)} Mt, {_sci(energy / C_LIGHT ** 2)} kg of "
        f"mass-energy); a 1 km^2 solar collector at 1 au banks it in {_human_time(solar_collection_time(energy))}.",
        f"Workpiece ({m.key}): {regime.state}.",
        f"Holding the field statically needs at least {_sci(fibre)} kg of T1100G carbon fibre or {_sci(graphene)} kg "
        f"of ideal graphene (virial theorem); gravity can do it only as a neutron star of about "
        f"{holder.holder_solar_masses:.2g} solar masses or more.",
    ]
    if record is not None:
        notes.append(f"Strongest field on the laboratory ladder in at least this volume for at least this long: "
                     f"{_sci(record.field)} T ({record.description}); the premise is {_sci(b / record.field)} "
                     "times that.")
    else:
        notes.append("Nothing on the laboratory ladder fills this volume for this long (EXTREME_FIELD_LADDER).")

    warnings: list[str] = []
    if b > DESTRUCTIVE_FIELD_RECORD:
        warnings.append(f"the field is {_sci(b / DESTRUCTIVE_FIELD_RECORD)} times the strongest macroscopic "
                        f"laboratory field ({DESTRUCTIVE_FIELD_RECORD:g} T) and "
                        f"{_sci(b / benchmarks.PULSED_NONDESTRUCTIVE_RECORD)} times the non-destructive record")
    if regime.tier != "solid":
        warnings.append(f"no workpiece survives as a solid: {regime.state}")
    if source.held() and b > benchmarks.PULSED_NONDESTRUCTIVE_RECORD:
        warnings.append(f"a pulse of {_sci(source.pulse_duration or 0.0)} s is longer than the free-expansion time "
                        f"R/c = {_sci(source.confinement_time())} s, so something must hold the field: no material "
                        "can (see the virial masses)")
    beyond_magnets = b > benchmarks.PULSED_NONDESTRUCTIVE_RECORD
    if beyond_magnets and power > benchmarks.PEAK_POWER_RECORD:
        warnings.append(f"assembling the field before it disperses takes {_sci(power)} W, "
                        f"{_sci(power / benchmarks.PEAK_POWER_RECORD)} times the largest pulsed-power machine")
    if not beyond_magnets:
        notes.append(f"At {_sci(b)} T the field is within the non-destructive record "
                     f"({benchmarks.PULSED_NONDESTRUCTIVE_RECORD:g} T): a pulsed magnet can hold it, so the "
                     "free-expansion time and the power to beat it do not apply; see magform.press.")
    if energy >= MEGATON_TNT:
        warnings.append(f"each shot releases {_sci(energy / MEGATON_TNT)} Mt unless the field energy is "
                        f"recovered: unshielded hardware inside about {_sci(damage)} m is damaged "
                        "(estimated fluence 1e6 J/m^2), which is why the mode has to be space-based")
    if standoff["pacemaker_line"] > 10.0 * r0:
        warnings.append(f"stray field: the 0.5 mT line is at {_sci(standoff['pacemaker_line'])} m and water is "
                        f"pushed away harder than Earth gravity inside {_sci(levitation_radius(b, r0))} m")

    return MagnetarAssessment(
        status=Status.ASSUMED, margin=None, warnings=tuple(warnings), notes=tuple(notes),
        premise=PREMISE, material=m.key, field=b, radius=r0, pulse_duration=source.pulse_duration,
        pressure=pressure, energy_density=pressure, mass_equivalent_density=pressure / C_LIGHT ** 2,
        atomic_field_ratio=shape.atomic_field_ratio, qed_field_ratio=b / QED_CRITICAL_FIELD,
        cyclotron_energy=cyclotron_energy(b), landau_excitation_energy=landau_excitation_energy(b),
        magnetic_length=magnetic_length(b), synchrotron_cooling_time=synchrotron_cooling_time(b),
        interior_energy=source.interior_energy(), field_energy=energy,
        field_energy_megatons=energy / MEGATON_TNT, mass_equivalent=energy / C_LIGHT ** 2,
        confinement_time=source.confinement_time(), required_power=power, pulse_power=source.pulse_power(),
        held=source.held(), virial_mass_fibre=fibre, virial_mass_graphene=graphene, holder=holder,
        solar_collection_time=solar_collection_time(energy), gaps=gaps,
        laboratory_record=None if record is None else record.key,
        laboratory_record_field=None if record is None else record.field,
        regime=regime, state=regime.state, thresholds_crossed=regime.crossed,
        compressed_density=regime.piston_density, condensed_density=condensed,
        hydrogen_binding_energy=hydrogen_binding_energy(b), atom_aspect_ratio=shape.aspect_ratio,
        standoff=standoff, levitation_radius=levitation_radius(b, r0), damage_radius=damage,
        solar_second_radius=solar_second, processes=process_outcomes(m, b),
    )


# ---------------------------------------------------------------------------
# sweep for plotting
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class FieldSweep(ResultMixin):
    """Columnar data across a range of fields, for one material.

    Per field: ``pressure`` [Pa], ``mass_equivalent_density`` [kg/m^3], ``cyclotron_energy`` [J]
    (non-relativistic), ``magnetic_length`` [m], ``atomic_field_ratio``, ``piston_density``
    [kg/m^3] (None where the equation of state is not offered), ``condensed_density`` [kg/m^3]
    (None where not valid), ``regime`` (highest boundary crossed), ``matter_regime`` (highest
    boundary that concerns the matter) and ``tier``.

    Reference: see :func:`classify_regime`.
    """

    material: str
    field: tuple[float, ...] = unit("T")
    pressure: tuple[float, ...] = unit("Pa")
    mass_equivalent_density: tuple[float, ...] = unit("kg/m^3")
    cyclotron_energy: tuple[float, ...] = unit("J")
    magnetic_length: tuple[float, ...] = unit("m")
    atomic_field_ratio: tuple[float, ...]
    piston_density: tuple[float | None, ...] = unit("kg/m^3")
    condensed_density: tuple[float | None, ...] = unit("kg/m^3")
    regime: tuple[str, ...]
    matter_regime: tuple[str, ...]
    tier: tuple[str, ...]


def decade_fields(lowest: float = 10.0, highest: float = 1.0e11, per_decade: int = 4) -> tuple[float, ...]:
    """Log-spaced fields from ``lowest`` to ``highest`` inclusive, ``per_decade`` points per decade [T].

    Defaults span 10 T to 1e11 T (41 points): from a forming coil to the strongest magnetars.

    Reference: none (plotting helper for :func:`field_sweep`).
    """
    lo = _positive("lowest", lowest)
    hi = _positive("highest", highest)
    if hi <= lo:
        raise InputError("highest must exceed lowest")
    if isinstance(per_decade, bool) or not isinstance(per_decade, int) or per_decade < 1:
        raise InputError(f"per_decade must be an integer >= 1, got {per_decade!r}")
    span = math.log10(hi / lo)
    n = max(1, round(span * per_decade))
    return tuple(lo * (hi / lo) ** (i / n) for i in range(n + 1))


def field_sweep(fields: Sequence[float] | None = None, material: MaterialLike = "iron_armco") -> FieldSweep:
    """Field scales and the state of ``material`` across a list of fields, as columns for plotting.

    ``fields`` in T (non-zero; default :func:`decade_fields`, 10 T to 1e11 T).  Each column is
    the corresponding scalar function of this module evaluated at every field
    (``B^2/(2 mu0)``, ``u/c^2``, ``hbar e B/m_e``, ``sqrt(hbar/(e B))``, ``B/B0``,
    :func:`piston_density`, :func:`condensed_density`, :func:`classify_regime`).

    Reference: Lai (2001); Shapiro & Teukolsky (1983).
    """
    m = as_material(material)
    values = decade_fields() if fields is None else tuple(_positive_field(f) for f in fields)
    if not values:
        raise InputError("fields must not be empty")
    comp = composition(m)
    piston: list[float | None] = []
    condensed: list[float | None] = []
    regimes: list[str] = []
    matter: list[str] = []
    tiers: list[str] = []
    for b in values:
        regime = classify_regime(m, b)
        piston.append(regime.piston_density)
        regimes.append(regime.key)
        matter.append(regime.matter_key)
        tiers.append(regime.tier)
        if comp.charge is not None and comp.mass_number is not None and condensed_density_valid(b, comp.charge):
            condensed.append(condensed_density(b, comp.mass_number, comp.charge))
        else:
            condensed.append(None)
    return FieldSweep(
        material=m.key, field=tuple(values),
        pressure=tuple(magnetic_pressure(b) for b in values),
        mass_equivalent_density=tuple(mass_equivalent_density(b) for b in values),
        cyclotron_energy=tuple(cyclotron_energy(b) for b in values),
        magnetic_length=tuple(magnetic_length(b) for b in values),
        atomic_field_ratio=tuple(b / ATOMIC_FIELD for b in values),
        piston_density=tuple(piston), condensed_density=tuple(condensed),
        regime=tuple(regimes), matter_regime=tuple(matter), tier=tuple(tiers),
    )


__all__ = [
    "REFERENCES",
    "HBAR", "PLANCK", "ELECTRON_MASS", "ATOMIC_MASS_UNIT", "GRAVITATIONAL_CONSTANT", "BOHR_RADIUS", "HARTREE",
    "ELECTRON_VOLT", "ELECTRON_REST_ENERGY", "FINE_STRUCTURE", "SOLAR_MASS", "SOLAR_CONSTANT", "SOLAR_LUMINOSITY",
    "MEGATON_TNT", "PION_REST_ENERGY",
    "ATOMIC_FIELD", "QED_CRITICAL_FIELD", "ATOMIC_PRESSURE", "DEGENERACY_PRESSURE_UNIT", "DEGENERACY_DENSITY_UNIT",
    "LATTICE_COEFFICIENT", "EOS_MAX_LATTICE_FRACTION", "EOS_GENERIC_FLOOR", "NEUTRON_DRIP_DENSITY",
    "NEUTRON_DRIP_PRESSURE", "NUCLEAR_DENSITY", "MIN_NEUTRON_STAR_MASS", "CANONICAL_NEUTRON_STAR",
    "HYDROGEN_ASYMPTOTIC_MIN", "CONDENSED_DENSITY_COEFFICIENT", "SYNCHROTRON_COEFFICIENT",
    "RELATIVISTIC_DEGENERACY_FIELD", "NEUTRON_DRIP_FIELD", "PRESSURE_IONISATION_FIELD",
    "CONCEPT_PRESSURE", "DESTRUCTIVE_FIELD_RECORD", "LASER_PEAK_POWER", "LASER_RECORD_INTENSITY", "Z_FIELD_RADIUS",
    "WATER_SUSCEPTIBILITY", "WATER_DENSITY", "WATER_LEVITATION_GRADIENT", "DAMAGE_FLUENCE",
    "REFERENCE_COLLECTOR_AREA", "VAPORISATION", "ELEMENTS", "NUCLIDES", "ELEMENT_NUCLIDE",
    "EXTREME_FIELD_LADDER", "ENERGY_SCALES", "FIELD_THRESHOLDS", "APPLICATIONS", "TIERS", "PREMISE",
    "Nuclide", "FieldRecord", "EnergyScale", "FieldThreshold",
    "atomic_field_ratio", "qed_field_ratio", "cyclotron_energy", "landau_excitation_energy", "magnetic_length",
    "laboratory_record",
    "mass_equivalent_density", "field_energy", "tnt_megatons", "energy_ratios",
    "degeneracy_pressure", "degenerate_density", "fermi_parameter_for_density", "fermi_kinetic_energy",
    "lattice_pressure_fraction", "fermi_parameter_for_pressure", "PistonState", "eos_floor_field", "piston_state",
    "piston_density", "CaptureThreshold", "electron_capture_threshold", "Composition", "composition",
    "hydrogen_binding_energy", "AtomShape", "atom_shape", "condensed_density_valid", "condensed_density",
    "synchrotron_cooling_time",
    "skin_boiling_field", "skin_vaporisation_field", "bulk_modulus_field", "yield_field",
    "Threshold", "regime_thresholds", "Regime", "classify_regime", "process_outcomes",
    "chandrasekhar_fermi_field", "GravitationalHolder", "gravitational_holder", "solar_collection_time",
    "dipole_axial_field", "standoff_radius", "standoff_radii", "levitation_gradient", "diamagnetic_acceleration",
    "levitation_radius",
    "radiant_fluence", "fluence_radius", "MagnetarSource",
    "MagnetarAssessment", "assess_magnetar_mode",
    "FieldSweep", "decade_fields", "field_sweep",
]
