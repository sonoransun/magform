"""Cited benchmark ladders: what has actually been demonstrated.

Every feasibility status in magform is assigned by comparing a computed
quantity with one of the ladders below, so each label traces to a dated,
cited achievement.  Values are SI (T, Pa, J, T/s).

Ladders
-------
FIELD_LADDER        magnet fields by technology (DC superconducting, hybrid,
                    resistive, pulsed non-destructive, destructive)
RAMP_RATE_LADDER    how fast superconducting magnets have been swept
ENERGY_LADDER       largest pulsed energy stores
PRESSURE_BENCHMARKS where 1.6-4 GPa sits among manufacturing and
                    high-pressure-science pressures
STATIC_PRESS_LADDER anvil/press technologies by pressure and sample volume

Status helpers
--------------
``superconducting_field_status``, ``pulsed_field_status``,
``ramp_rate_status``, ``energy_status``, ``where_does_it_sit``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._report import ResultMixin
from .constants import MU0
from .status import Confidence, Status


@dataclass(frozen=True)
class Benchmark(ResultMixin):
    """One demonstrated (or reported) achievement with its citation."""

    key: str
    description: str
    value: float
    unit: str
    year: int
    citation: str
    confidence: Confidence = Confidence.HANDBOOK
    note: str = ""


# ---------------------------------------------------------------------------
# Magnetic field records by technology
# ---------------------------------------------------------------------------
FIELD_LADDER: tuple[Benchmark, ...] = (
    Benchmark(
        "nhmfl_32t_all_sc", "All-superconducting user magnet (LTS outsert + REBCO insert), NHMFL",
        32.0, "T", 2017,
        "National High Magnetic Field Laboratory news release, 12 Dec 2017: 32 T reached on 8 Dec 2017.",
    ),
    Benchmark(
        "secuf_35p6t_all_sc", "All-superconducting user magnet, 35 mm usable aperture, CAS SECUF (Beijing)",
        35.6, "T", 2026,
        "Institute of Physics, Chinese Academy of Sciences, release on the Synergetic Extreme Condition User "
        "Facility: 35.6 T with a usable aperture of 35 mm on 24 Jan 2026 (built with the Institute of "
        "Electrical Engineering, CAS).",
        Confidence.REPORTED,
        "Institutional announcement. Record for all-superconducting user magnets; between it and the 32 T "
        "NHMFL magnet lie 32.35 T (Institute of Electrical Engineering, CAS, announced Dec 2019) and 35.1 T "
        "(Institute of Plasma Physics, Hefei, announced Sep 2025).",
    ),
    Benchmark(
        "rebco_insert_45p5t", "REBCO test coil (14.4 T) operated inside a 31.1 T resistive background, NHMFL",
        45.5, "T", 2019,
        "Hahn et al., Nature 570, 496-499 (2019).",
        note="The peer-reviewed record; DC, small test coil. Exceeded in 2025 (see lbc9_48p7t).",
    ),
    Benchmark(
        "lbc9_48p7t", "REBCO test coil (Little Big Coil 9) operated inside a 31 T resistive background, NHMFL",
        48.7, "T", 2025,
        "National High Magnetic Field Laboratory news release, 11 Sep 2025, and research highlight, Jan 2026: "
        "48.7 T in August 2025, the test coil adding 17.6-17.7 T to a 31 T resistive magnet.",
        Confidence.REPORTED,
        "Highest field produced (in part) by a superconductor; a DC test coil, not a user magnet; no journal "
        "paper found as of November 2026.",
    ),
    Benchmark(
        "chmfl_resistive_42t", "All-resistive DC magnet, CHMFL Hefei (32.3 MW)",
        42.02, "T", 2024,
        "Hefei Institutes of Physical Science, CAS, release: 42.02 T on 22 Sep 2024 at 32.3 MW.",
        Confidence.REPORTED,
    ),
    Benchmark(
        "chmfl_hybrid_45t", "Hybrid DC magnet (superconducting outsert + resistive insert), CHMFL",
        45.22, "T", 2022,
        "Chinese Academy of Sciences release, 15 Aug 2022: 45.22 T on 12 Aug 2022 in a 32 mm bore.",
        Confidence.REPORTED,
    ),
    Benchmark(
        "lanl_100t_nondestructive", "Non-destructive pulsed magnet (generator-driven outsert + capacitor-driven insert), LANL",
        100.75, "T", 2012,
        "NHMFL Pulsed Field Facility, Los Alamos: 100.75 T multi-shot magnet, 22 March 2012 (LANL and US "
        "Department of Energy releases).",
        note="User bore about 10 mm as operated (10 mm at 77 K, MagLab user page; 10.5 mm, IEEE Trans. Appl. "
             "Supercond., doi:10.1109/TASC.2023.3338580; the design figure was 15 mm). Outsert coils in a "
             "225 mm bore driven by a 1.4 GW motor-generator supply 40-44 T and store 120 MJ; the insert "
             "(55-60 T, 1.4 MJ) is fired from a 2-2.2 MJ capacitor bank and takes the field from 40 T to "
             "100 T in 8 ms; the field lasts about 15 ms inside an outsert pulse of 2.5-3 s, as often as "
             "once an hour (MagLab; Battesti et al., arXiv:1803.07547). Offline since 2019 while the generator rotor "
             "is replaced. No later record found (Toulouse 98.8 T, Dresden 95.6 T, Wuhan 94.8 T).",
    ),
    Benchmark(
        "single_turn_300t", "Destructive single-turn coil (microsecond pulses)",
        300.0, "T", 2000,
        "ISSP Tokyo / LNCMI single-turn coil systems, of order 300 T (Herlach & Miura, High Magnetic Fields, 2003).",
        Confidence.REPRESENTATIVE,
    ),
    Benchmark(
        "emfc_1200t", "Electromagnetic flux compression (destructive), ISSP Tokyo",
        1200.0, "T", 2018,
        "Nakamura et al., Rev. Sci. Instrum. 89, 095106 (2018).",
    ),
    Benchmark(
        "explosive_fc_2800t", "Explosive-driven flux compression (destructive), VNIIEF",
        2800.0, "T", 1999,
        "Boyko et al., Proc. 12th IEEE Int. Pulsed Power Conf., Monterey (1999), p. 746; reviewed in "
        "Boriskov et al., Phys.-Usp. 54, 421 (2011): about 28 MG.",
        Confidence.REPRESENTATIVE,
        "Year of the report; the date of the shot itself is not established in the sources consulted.",
    ),
)

# Single-source thresholds derived from the ladder
SC_FIELD_ESTABLISHED: float = 32.0
"""T. All-superconducting user magnets established (NHMFL, December 2017)."""
SC_FIELD_RECORD: float = 35.6
"""T. All-superconducting record (CAS SECUF, January 2026, reported)."""
SC_CONDUCTOR_RECORD: float = 48.7
"""T. Highest field at which a superconducting coil has operated, inside a resistive background
(NHMFL Little Big Coil 9, August 2025, reported; the peer-reviewed predecessor is 45.5 T, Hahn et al. 2019)."""
PULSED_NONDESTRUCTIVE_RECORD: float = 100.75
"""T. Non-destructive pulsed record (LANL, 22 March 2012, about 10 mm bore)."""
PULSED_NONDESTRUCTIVE_BORE: float = 0.010
"""m. User bore of the 100 T LANL magnet as operated (10 mm at 77 K per the MagLab user page; 10.5 mm in
LANL's 2024 paper; the design figure, still quoted on MagLab's overview page, was 15 mm)."""
PULSED_ROUTINE_FIELD: float = 60.0
"""T. Field class of routine non-destructive pulsed user magnets (many laboratories run 60-75 T)."""
PULSED_ROUTINE_BORE: float = 0.025
"""m. Largest bore treated as routine for that class (10-20 mm is typical; judgement)."""
PULSED_RECORD_INSERT_BANK: float = 2.2e6
"""J. Capacitor bank that fires the insert coil of the LANL 100 T magnet (2 MJ per MagLab; 2.2 MJ per
Battesti et al., arXiv:1803.07547)."""
PULSED_RECORD_GENERATOR_POWER: float = 1.4e9
"""W. Motor-generator that drives the outsert coils of the LANL 100 T magnet for 2.5-3 s."""
PULSED_RECORD_OUTSERT_ENERGY: float = 120e6
"""J. Magnetic energy stored in the outsert coils of the LANL 100 T magnet at full field (MagLab);
the insert stores 1.4 MJ."""
EMF_CONVENTIONAL_FIELD: tuple[float, float] = (20.0, 50.0)
"""T. Field range of conventional industrial electromagnetic (magnetic-pulse) forming coils."""


def superconducting_field_status(field: float) -> Status:
    """Status of a field produced by a superconducting (DC) magnet.

    <= 35.6 T state of the art (all-SC record); <= 48.7 T extrapolation
    (a superconducting coil has operated there, in a resistive background);
    above, aspirational.
    """
    if field <= SC_FIELD_RECORD:
        return Status.STATE_OF_THE_ART
    if field <= SC_CONDUCTOR_RECORD:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


def pulsed_field_status(field: float, bore_diameter: float | None = None) -> Status:
    """Status of a field produced by a pulsed resistive (non-destructive) magnet.

    <= 100.75 T is demonstrated, but only in a bore of about 10 mm
    (``PULSED_NONDESTRUCTIVE_BORE``): a larger bore at the same field is an
    extrapolation (stored energy scales with bore volume).
    Above the record the status is aspirational for non-destructive coils.
    """
    if field > PULSED_NONDESTRUCTIVE_RECORD:
        return Status.ASPIRATIONAL
    if field <= PULSED_ROUTINE_FIELD:
        # many labs operate 60-75 T non-destructive pulsed magnets with 10-20 mm bores
        if bore_diameter is None or bore_diameter <= PULSED_ROUTINE_BORE:
            return Status.STATE_OF_THE_ART
        return Status.EXTRAPOLATION
    if bore_diameter is None or bore_diameter <= PULSED_NONDESTRUCTIVE_BORE * 1.05:
        return Status.STATE_OF_THE_ART
    return Status.EXTRAPOLATION


# ---------------------------------------------------------------------------
# Ramp rate of superconducting magnets
# ---------------------------------------------------------------------------
RAMP_RATE_LADDER: tuple[Benchmark, ...] = (
    Benchmark(
        "sis100_superferric", "Fast-ramped superconducting accelerator dipoles (FAIR SIS100, Nuclotron type)",
        4.0, "T/s", 2019,
        "Fischer et al., Proc. IPAC2019, paper THPTS010: the SIS100 main dipoles ramp at 4 T/s up to 1.9 T.",
        Confidence.REPRESENTATIVE,
    ),
)
SC_RAMP_RATE_SOTA: float = 4.0
"""T/s. Fastest routinely operated superconducting magnet ramps."""
SC_RAMP_RATE_EXTRAPOLATION: float = 100.0
"""T/s. Boundary beyond which pulsed superconducting operation is treated as aspirational
(order-of-magnitude judgement; small HTS test coils; confidence estimated)."""


def ramp_rate_status(ramp_rate: float) -> Status:
    """Status of sweeping a superconducting winding at ``ramp_rate`` (T/s)."""
    r = abs(ramp_rate)
    if r <= SC_RAMP_RATE_SOTA:
        return Status.STATE_OF_THE_ART
    if r <= SC_RAMP_RATE_EXTRAPOLATION:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


# ---------------------------------------------------------------------------
# Pulsed energy stores
# ---------------------------------------------------------------------------
ENERGY_LADDER: tuple[Benchmark, ...] = (
    Benchmark(
        "lanl_insert_bank_2mj", "Capacitor bank firing the insert coil of the LANL 100 T magnet",
        2.2e6, "J", 2012,
        "Battesti et al., arXiv:1803.07547 (2018): 2.2 MJ; National High Magnetic Field Laboratory, "
        "'Meet the 100 tesla pulsed magnet': 2 MJ.",
        Confidence.REPRESENTATIVE,
    ),
    Benchmark(
        "hld_50mj_bank", "Capacitor bank driving pulsed magnets, HZDR Dresden High Magnetic Field Lab",
        50e6, "J", 2007,
        "Helmholtz-Zentrum Dresden-Rossendorf publication records 13017 and 17042: 50 MJ modular capacitor "
        "bank, 24 kV maximum charging voltage, up to 500 kA and about 5 GW.",
    ),
    Benchmark(
        "balcones_60mj_hpg", "Homopolar generator supply (6 x 10 MJ), UT Center for Electromechanics",
        60e6, "J", 1985,
        "UT-CEM Balcones homopolar generator power supply, 60 MJ, 1.5 MA at 600 V.",
        Confidence.REPRESENTATIVE,
    ),
    Benchmark(
        "lanl_outsert_120mj", "Field energy of the outsert coils of the LANL 100 T magnet at full field (225 mm bore)",
        120e6, "J", 2012,
        "National High Magnetic Field Laboratory, 'Meet the 100 tesla pulsed magnet': stored magnetic energy at "
        "100 T, 120 MJ in the outsert and 1.4 MJ in the insert.",
        Confidence.REPRESENTATIVE,
        "Charged over about a second by the 1.4 GW motor-generator, not by a capacitor bank.",
    ),
    Benchmark(
        "nif_400mj_bank", "Capacitor bank energy, National Ignition Facility power conditioning system",
        400e6, "J", 2009,
        "LLNL National Ignition Facility power conditioning system (~400 MJ).",
        Confidence.REPRESENTATIVE,
    ),
    Benchmark(
        "anu_500mj_hpg", "Homopolar generator, Australian National University",
        500e6, "J", 1962,
        "ANU homopolar generator (1962-1986), 500 MJ stored rotational energy.",
        Confidence.REPRESENTATIVE,
    ),
    Benchmark(
        "lanl_mg_1200mj", "Motor-generator rotational energy, NHMFL Pulsed Field Facility (1.4 GW)",
        1.2e9, "J", 2012,
        "LANL release (2012): a 1,200 MJ motor-generator powers the 100 T outsert; LANL's 2024 paper "
        "(doi:10.1109/TASC.2023.3338580) calls it a 1 GJ, 1.4 GW generator.",
        Confidence.REPORTED,
        "Battesti et al. (arXiv:1803.07547) say the same 1.4 GW machine stores up to 600 MJ of rotational "
        "energy in a 300-ton shaft; the sources disagree by a factor of two.",
    ),
)
ENERGY_MAGNET_DRIVER_SOTA: float = 50e6
"""J. Largest capacitor bank built as a pulsed-magnet driver (HZDR)."""
ENERGY_ANY_STORE_RECORD: float = 1.2e9
"""J. Largest pulsed rotating store in a magnet facility (LANL motor-generator)."""


def energy_status(energy: float) -> Status:
    """Status of the energy a single shot must draw from a pulsed store."""
    if energy <= ENERGY_MAGNET_DRIVER_SOTA:
        return Status.STATE_OF_THE_ART
    if energy <= ENERGY_ANY_STORE_RECORD:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


# ---------------------------------------------------------------------------
# Pulsed currents and powers
# ---------------------------------------------------------------------------
CURRENT_LADDER: tuple[Benchmark, ...] = (
    Benchmark("emf_industrial_current", "Industrial electromagnetic-forming machines (peak coil current)",
              1e6, "A", 2011, "Psyk et al., J. Mater. Process. Technol. 211, 787 (2011): 10^5-10^6 A class machines.",
              Confidence.REPRESENTATIVE),
    Benchmark("z_machine_current", "Sandia Z machine (refurbished), peak load current",
              26e6, "A", 2007, "Sandia National Laboratories, Z machine overview (OSTI 1429410): 22 MJ stored, "
              "26 MA peak current, 80 TW; Hutsel et al., Phys. Rev. Accel. Beams 21, 030401 (2018).",
              Confidence.REPRESENTATIVE),
)
PULSED_CURRENT_SOTA: float = 5e6
"""A. Peak currents routinely delivered by large capacitor-bank facilities to a single load."""
PULSED_CURRENT_RECORD: float = 26e6
"""A. Z machine peak load current."""
PEAK_POWER_SOTA: float = 1e12
"""W. Order of magnitude of the peak electrical power of the largest capacitor-bank installations
(hundreds of megajoules released in under a millisecond; an estimated boundary).  Pulsed-magnet banks
are far below it: the 50 MJ Dresden bank delivers about 5 GW."""
PEAK_POWER_RECORD: float = 8e13
"""W. Sandia Z machine electrical power (~80 TW)."""


def pulsed_current_status(current: float) -> Status:
    """Status of a peak current delivered to a single load by a pulsed source."""
    i = abs(current)
    if i <= PULSED_CURRENT_SOTA:
        return Status.STATE_OF_THE_ART
    if i <= PULSED_CURRENT_RECORD:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


def peak_power_status(power: float) -> Status:
    """Status of the peak electrical power a pulsed source must deliver."""
    w = abs(power)
    if w <= PEAK_POWER_SOTA:
        return Status.STATE_OF_THE_ART
    if w <= PEAK_POWER_RECORD:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


# ---------------------------------------------------------------------------
# Terminal voltages
# ---------------------------------------------------------------------------
SC_VOLTAGE_SOTA: float = 5e3
"""V. Upper end of terminal/quench-protection voltages of large superconducting magnets
(ITER-class coils are designed for a few kV inside the cryostat)."""
SC_VOLTAGE_EXTRAPOLATION: float = 20e3
"""V. Beyond this, insulation inside a cryostat is treated as aspirational (estimated boundary)."""
PULSED_VOLTAGE_SOTA: float = 25e3
"""V. Capacitor-bank charging voltages of pulsed-magnet facilities (LANL/HZDR 10-24 kV)."""
PULSED_VOLTAGE_EXTRAPOLATION: float = 100e3
"""V. Marx-generator class; beyond this, coil insulation is aspirational (estimated boundary)."""


def sc_voltage_status(voltage: float) -> Status:
    """Status of a terminal voltage across a superconducting winding."""
    v = abs(voltage)
    if v <= SC_VOLTAGE_SOTA:
        return Status.STATE_OF_THE_ART
    if v <= SC_VOLTAGE_EXTRAPOLATION:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


def pulsed_voltage_status(voltage: float) -> Status:
    """Status of a capacitor-bank / pulsed-coil terminal voltage."""
    v = abs(voltage)
    if v <= PULSED_VOLTAGE_SOTA:
        return Status.STATE_OF_THE_ART
    if v <= PULSED_VOLTAGE_EXTRAPOLATION:
        return Status.EXTRAPOLATION
    return Status.ASPIRATIONAL


# ---------------------------------------------------------------------------
# Pressure benchmarks (Pa)
# ---------------------------------------------------------------------------
PRESSURE_BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark("hip", "Hot isostatic pressing (industrial, gas)", 0.2e9, "Pa", 2020,
              "Atkinson & Davies, Metall. Mater. Trans. A 31, 2981 (2000): HIP at 100-200 MPa.",
              Confidence.REPRESENTATIVE),
    Benchmark("mpw_typical", "Magnetic pulse welding/forming, typical interface/working pressures", 0.5e9, "Pa", 2011,
              "Psyk et al., J. Mater. Process. Technol. 211, 787 (2011).", Confidence.REPRESENTATIVE),
    Benchmark("emf_50t", "Magnetic pressure of a 50 T forming coil (upper end of industrial EMF)",
              50.0 ** 2 / (2 * MU0), "Pa", 2011, "B^2/(2 mu0) at 50 T.", Confidence.ESTIMATED),
    Benchmark("cold_forging_die", "Contact pressure on cold-forging dies (steels)", 2.5e9, "Pa", 2005,
              "Lange (ed.), Handbook of Metal Forming (1985): cold-forging punch pressures up to ~2.5 GPa.",
              Confidence.REPRESENTATIVE),
    Benchmark("lanl_100t_pressure", "Magnetic pressure at 100.75 T (LANL record)",
              100.75 ** 2 / (2 * MU0), "Pa", 2012, "B^2/(2 mu0) at 100.75 T.", Confidence.ESTIMATED),
    Benchmark("hpht_cubic", "Industrial HPHT diamond/cBN synthesis (cubic and belt presses)", 6.0e9, "Pa", 1955,
              "Bundy et al., Nature 176, 51 (1955); industrial cubic presses 5-6.5 GPa.",
              Confidence.REPRESENTATIVE),
    Benchmark("npd_multianvil", "Nano-polycrystalline diamond, Kawai multi-anvil (15 GPa)", 15e9, "Pa", 2003,
              "Irifune et al., Nature 421, 599 (2003).", Confidence.HANDBOOK),
    Benchmark("dupont_shock", "Shock synthesis of diamond powder (DuPont process)", 30e9, "Pa", 1961,
              "DeCarli & Jamieson, Science 133, 1821 (1961).", Confidence.REPRESENTATIVE),
    Benchmark("carbon_nitride_dac", "Carbon nitride synthesis, laser-heated diamond anvil cell", 124e9, "Pa", 2024,
              "Laniel et al., Adv. Mater. 36, 2308030 (2024): 72-134 GPa, ~2500 K.", Confidence.HANDBOOK),
    Benchmark("dac_static", "Static diamond-anvil-cell pressures (routine upper range)", 400e9, "Pa", 2020,
              "Dubrovinsky et al., Nat. Commun. 3, 1163 (2012) and later toroidal-anvil work.",
              Confidence.REPRESENTATIVE),
)

STATIC_PRESS_LADDER: tuple[Benchmark, ...] = (
    Benchmark("belt_cubic_wc", "Belt/cubic presses with tungsten-carbide anvils; cm^3 samples, industrial",
              7e9, "Pa", 1960, "Hall, Rev. Sci. Instrum. 31, 125 (1960); industrial practice.",
              Confidence.REPRESENTATIVE),
    Benchmark("kawai_wc", "Kawai multi-anvil with WC cubes; mm^3 samples", 28e9, "Pa", 2003,
              "Ito, Treatise on Geophysics 2 (2007): WC-anvil multi-anvil to ~28 GPa.",
              Confidence.REPRESENTATIVE),
    Benchmark("kawai_sintered_diamond", "Kawai multi-anvil with sintered-diamond cubes; ~0.1 mm^3", 100e9, "Pa", 2014,
              "Yamazaki et al., Phys. Earth Planet. Inter. 228, 262 (2014): ~100 GPa class.",
              Confidence.REPRESENTATIVE),
    Benchmark("dac", "Diamond anvil cell; microgram samples", 400e9, "Pa", 2012,
              "Dubrovinsky et al., Nat. Commun. 3, 1163 (2012).", Confidence.REPRESENTATIVE),
)

TNT_ENERGY_DENSITY: float = 6.9e9
"""J/m^3. Detonation energy of TNT per unit volume (4.184 MJ/kg x 1654 kg/m^3), for scale."""


@dataclass(frozen=True)
class Placement(ResultMixin):
    """Where a pressure sits relative to the cited benchmarks."""

    pressure: float
    field_equivalent: float
    energy_density: float
    below: tuple[str, ...]
    above: tuple[str, ...]
    nearest: str
    tnt_fraction: float


def where_does_it_sit(pressure: float) -> Placement:
    """Place a pressure (Pa) among :data:`PRESSURE_BENCHMARKS`."""
    import math

    below = tuple(b.key for b in PRESSURE_BENCHMARKS if b.value <= pressure)
    above = tuple(b.key for b in PRESSURE_BENCHMARKS if b.value > pressure)
    nearest = min(PRESSURE_BENCHMARKS, key=lambda b: abs(math.log(b.value / pressure))).key
    field = math.sqrt(2.0 * MU0 * pressure)
    return Placement(
        pressure=pressure,
        field_equivalent=field,
        energy_density=pressure,
        below=below,
        above=above,
        nearest=nearest,
        tnt_fraction=pressure / TNT_ENERGY_DENSITY,
    )


ALL_LADDERS: dict[str, tuple[Benchmark, ...]] = {
    "field": FIELD_LADDER,
    "current": CURRENT_LADDER,
    "ramp_rate": RAMP_RATE_LADDER,
    "energy": ENERGY_LADDER,
    "pressure": PRESSURE_BENCHMARKS,
    "static_press": STATIC_PRESS_LADDER,
}

__all__ = [
    "Benchmark", "FIELD_LADDER", "RAMP_RATE_LADDER", "ENERGY_LADDER", "PRESSURE_BENCHMARKS",
    "STATIC_PRESS_LADDER", "ALL_LADDERS", "SC_FIELD_ESTABLISHED", "SC_FIELD_RECORD",
    "SC_CONDUCTOR_RECORD", "PULSED_NONDESTRUCTIVE_RECORD", "PULSED_NONDESTRUCTIVE_BORE",
    "PULSED_RECORD_INSERT_BANK", "PULSED_RECORD_GENERATOR_POWER", "PULSED_RECORD_OUTSERT_ENERGY",
    "PULSED_ROUTINE_FIELD", "PULSED_ROUTINE_BORE",
    "EMF_CONVENTIONAL_FIELD", "SC_RAMP_RATE_SOTA", "SC_RAMP_RATE_EXTRAPOLATION",
    "ENERGY_MAGNET_DRIVER_SOTA", "ENERGY_ANY_STORE_RECORD", "TNT_ENERGY_DENSITY",
    "superconducting_field_status", "pulsed_field_status", "ramp_rate_status", "energy_status",
    "SC_VOLTAGE_SOTA", "SC_VOLTAGE_EXTRAPOLATION", "PULSED_VOLTAGE_SOTA", "PULSED_VOLTAGE_EXTRAPOLATION",
    "sc_voltage_status", "pulsed_voltage_status",
    "CURRENT_LADDER", "PULSED_CURRENT_SOTA", "PULSED_CURRENT_RECORD", "PEAK_POWER_SOTA", "PEAK_POWER_RECORD",
    "pulsed_current_status", "peak_power_status",
    "Placement", "where_does_it_sit",
]
