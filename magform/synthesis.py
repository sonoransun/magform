"""High-pressure synthesis of superhard phases: phase boundaries, routes and pulse compatibility.

The concept note claims that a 1.6-4 GPa magnetic "press" enables macroscopic,
industrial-scale synthesis of diamond, lonsdaleite, carbon nitrides and
"quantum" materials, and that diamond anvil cells make such phases only in
microgram quantities.  This module makes the comparison quantitative:

* equilibrium phase boundaries, two cited models each: graphite-diamond
  (Kennedy & Kennedy 1976; Berman & Simon 1955) and hBN-cBN (Solozhenko et al.
  1999; Corrigan & Bundy 1975 - the two disagree by ~5 GPa at 1800 K);
* :data:`SYNTHESIS_ROUTES`: how these phases are made today (pressure,
  temperature, minimum dwell, heating, product form, present scale);
* :func:`assess_route` / :func:`assess_all`: can a magnetic pressure pulse
  execute a route?  :func:`anvil_feasibility`: what an anvil intensifier implies;
* :func:`volume_scaling`: masses and energies of a hypothetical large-bore charge.

Central honest results (pinned in ``tests/test_synthesis.py``):

1. Static HPHT routes need 1 s to ~10 days of dwell at 1500-3000 K.  A
   magnetic pressure pulse lasts microseconds to milliseconds, and a DC field
   exerts no pressure on a stationary, current-free sample (eddy currents
   decay), so every static route is INFEASIBLE for a pulsed magnetic drive,
   whatever the pressure.  The magnet also supplies no heating.
2. 1.6-4 GPa is below the minimum pressure of every static diamond and cBN
   route (cBN catalytic 4.5 GPa, diamond catalytic 5 GPa, PCD 5.5 GPa, NPD 15 GPa).
3. The only time-compatible route is shock synthesis.  A 2 mm Cu flyer driven
   over a 10 mm gap at 4 GPa reaches 2.11 km/s; Cu-on-Cu that is 52 GPa, but
   impedance-matched into graphite the sample sees ~22 GPa (crystal density) or
   ~16 GPa (porous isostatic graphite): marginal for lonsdaleite (>= 15 GPa)
   and at or below the DuPont diamond window (>= 20 GPa).  Magnetically driven
   flyer shock synthesis is not industrial; explosive shock synthesis was.
4. The README's premise is wrong for most of its examples: HPHT diamond, PCD
   and cBN are made industrially at tonne scale in cubic/belt presses, NPD
   compacts >= 1 cm in multi-anvil presses; only the > 50 GPa phases (carbon
   nitrides, hydrides) are DAC-limited.  See :data:`QUANTUM_MATERIALS_NOTE`.

All pressures Pa, temperatures K, times s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._numerics import linspace
from ._report import Assessment, ResultMixin, unit
from .benchmarks import STATIC_PRESS_LADDER
from .constants import T_ROOM, ZERO_CELSIUS, gpa
from .errors import InputError
from .materials import MaterialLike, as_material, get
from .shock import (
    ShockState,
    flyer_transit_time,
    flyer_velocity,
    impedance_match,
    shock_duration,
    shock_temperature_rise,
)
from .status import Confidence, Status, worst

REFERENCES: dict[str, str] = {
    "kennedy_1976": "Kennedy, C. S. & Kennedy, G. C., 'The equilibrium boundary between graphite and "
                    "diamond', J. Geophys. Res. 81 (1976) 2467-2470.",
    "berman_simon_1955": "Berman, R. & Simon, F., 'On the graphite-diamond equilibrium', "
                         "Z. Elektrochem. 59 (1955) 333-338.",
    "solozhenko_1999": "Solozhenko, V. L., Turkevich, V. Z. & Holzapfel, W. B., 'Refined phase diagram of "
                       "boron nitride', J. Phys. Chem. B 103 (1999) 2903-2905.",
    "corrigan_bundy_1975": "Corrigan, F. R. & Bundy, F. P., 'Direct transitions among the allotropic forms "
                           "of boron nitride at high pressures and temperatures', J. Chem. Phys. 63 (1975) "
                           "3812-3820.",
    "bundy_1955": "Bundy, F. P., Hall, H. T., Strong, H. M. & Wentorf, R. H., 'Man-made diamonds', "
                  "Nature 176 (1955) 51-55.",
    "strong_wentorf_1972": "Strong, H. M. & Wentorf, R. H., 'The growth of large diamond crystals', "
                           "Naturwissenschaften 59 (1972) 1-7.",
    "hall_1970": "Hall, H. T., 'Sintered diamond: a synthetic carbonado', Science 169 (1970) 868-869.",
    "irifune_2003": "Irifune, T., Kurio, A., Sakamoto, S., Inoue, T. & Sumiya, H., 'Ultrahard "
                    "polycrystalline diamond from graphite', Nature 421 (2003) 599-600.",
    "irifune_2019": "Irifune, T. et al., C. R. Geoscience (2019), doi:10.1016/j.crte.2018.07.005: "
                    "nano-polycrystalline diamond rods up to 1 cm in diameter and length; routine 9 mm x 8 mm "
                    "rods at 15 GPa and 2600 K for 20 min.",
    "wentorf_1957": "Wentorf, R. H., 'Cubic form of boron nitride', J. Chem. Phys. 26 (1957) 956.",
    "bundy_wentorf_1963": "Bundy, F. P. & Wentorf, R. H., 'Direct transformation of hexagonal boron nitride "
                          "to denser forms', J. Chem. Phys. 38 (1963) 1144-1149.",
    "hanneman_1967": "Hanneman, R. E., Strong, H. M. & Bundy, F. P., 'Hexagonal diamonds in meteorites: "
                     "implications', Science 155 (1967) 995-997.",
    "erskine_nellis_1991": "Erskine, D. J. & Nellis, W. J., 'Shock-induced martensitic phase transformation "
                           "of oriented graphite to diamond', Nature 349 (1991) 317-319.",
    "decarli_jamieson_1961": "DeCarli, P. S. & Jamieson, J. C., 'Formation of diamond by explosive shock', "
                             "Science 133 (1961) 1821-1822.",
    "kraus_2016": "Kraus, D. et al., 'Nanosecond formation of diamond and lonsdaleite by shock compression "
                  "of graphite', Nat. Commun. 7 (2016) 10970.",
    "laniel_2024": "Laniel, D. et al., Adv. Mater. 36 (2024) 2308030: carbon nitrides synthesised at "
                   "72-134 GPa and ~2500 K in laser-heated diamond anvil cells, recoverable to ambient.",
    "drozdov_2015": "Drozdov, A. P., Eremets, M. I., Troyan, I. A., Ksenofontov, V. & Shylin, S. I., "
                    "'Conventional superconductivity at 203 kelvin at high pressures in the sulfur hydride "
                    "system', Nature 525 (2015) 73-76.",
    "drozdov_2019": "Drozdov, A. P. et al., 'Superconductivity at 250 K in lanthanum hydride under high "
                    "pressures', Nature 569 (2019) 528-531.",
    "balasubramanian_2009": "Balasubramanian, G. et al., 'Ultralong spin coherence time in isotopically "
                            "engineered diamond', Nat. Mater. 8 (2009) 383-387.",
    "static_press_ladder": "magform.benchmarks.STATIC_PRESS_LADDER: Hall, Rev. Sci. Instrum. 31 (1960) 125; "
                           "Ito, Treatise on Geophysics 2 (2007); Yamazaki et al., Phys. Earth Planet. Inter. "
                           "228 (2014) 262; Dubrovinsky et al., Nat. Commun. 3 (2012) 1163.",
}

# ---------------------------------------------------------------------------
# phase-boundary model constants
# ---------------------------------------------------------------------------
GRAPHITE_DIAMOND_MODELS: tuple[str, ...] = ("kennedy_1976", "berman_simon")
"""Graphite-diamond equilibrium models; the first is the default."""

HBN_CBN_MODELS: tuple[str, ...] = ("solozhenko_1999", "corrigan_bundy_1975")
"""hBN-cBN equilibrium models; the first (modern) is the default."""

KENNEDY_1976_CALIBRATION: tuple[float, float] = (1373.0, 1898.0)
"""K. Temperature range of the Kennedy & Kennedy (1976) measurements (1100-1625 C); the
line is extrapolated outside it [kennedy_1976]."""

BERMAN_SIMON_MIN_TEMPERATURE: float = 1200.0
"""K. Berman & Simon (1955) linear form is intended for T >~ 1200 K [berman_simon_1955]."""

SOLOZHENKO_ZERO_PRESSURE_TEMPERATURE: float = 1660.0
"""K. Below this cBN is the thermodynamically stable BN phase at ambient pressure
(Solozhenko et al. 1999, the 1988 line shifted by +60 K) [solozhenko_1999]."""

SOLOZHENKO_TRIPLE_POINT: tuple[float, float] = (3480.0, 5.9e9)
"""(K, Pa). hBN-cBN-liquid triple point [solozhenko_1999]."""

CORRIGAN_BUNDY_TRIPLE_POINT: tuple[float, float] = (3500.0, 9.5e9)
"""(K, Pa). hBN-cBN-liquid triple point of the historical line [corrigan_bundy_1975]."""

ROUTE_KINDS: tuple[str, ...] = ("static_catalytic", "static_direct", "static_sintering", "shock",
                                "laser_heated_dac")
"""Allowed :attr:`SynthesisRoute.kind` values; all but ``"shock"`` need a sustained dwell."""


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number, got {value!r}") from None
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


def _model(model: str, allowed: tuple[str, ...], what: str) -> str:
    if model not in allowed:
        raise InputError(f"unknown {what} model {model!r}; choose one of {', '.join(allowed)}")
    return model


# ---------------------------------------------------------------------------
# equilibrium phase boundaries
# ---------------------------------------------------------------------------
def graphite_diamond_boundary(temperature: float, model: str = "kennedy_1976") -> float:
    """Graphite-diamond equilibrium pressure (Pa) at ``temperature`` (K).

    Governing equations (linear fits)::

        kennedy_1976:  P = 1.94 GPa + 0.0025 GPa/K * (T - 273.15 K)   [kennedy_1976]
                       (P[kbar] = 19.4 + T[C]/40)
        berman_simon:  P = 0.71 GPa + 0.0027 GPa/K * T                 [berman_simon_1955]

    Checks: 1500 K -> 5.01 / 4.76 GPa; 2000 K -> 6.26 / 6.11 GPa.

    Validity: Kennedy & Kennedy measured 1373-1898 K (:data:`KENNEDY_1976_CALIBRATION`)
    and the line is extrapolated outside; Berman & Simon is intended for
    T >~ 1200 K.  Equilibrium only: at room temperature diamond is
    thermodynamically stable above ~2 GPa, but conversion needs ~1500+ K and
    a metal catalyst (or > 12 GPa and ~2500 K uncatalysed) - kinetics, not
    thermodynamics, set the synthesis conditions.
    """
    t = _positive("temperature", temperature)
    m = _model(model, GRAPHITE_DIAMOND_MODELS, "graphite-diamond")
    if m == "kennedy_1976":
        return gpa(1.94 + 0.0025 * (t - ZERO_CELSIUS))
    return gpa(0.71 + 0.0027 * t)


def diamond_stable(pressure: float, temperature: float, model: str = "kennedy_1976") -> bool:
    """True if diamond (not graphite) is the stable carbon phase at (P, T).

    Governing relation: ``P >= graphite_diamond_boundary(T, model)`` (on or
    above the equilibrium line) [kennedy_1976, berman_simon_1955].

    Args:
        pressure: Pa (>= 0).  temperature: K (> 0).  model: see
            :data:`GRAPHITE_DIAMOND_MODELS`.

    Validity: thermodynamic stability only - says nothing about conversion
    kinetics (see :func:`graphite_diamond_boundary`).
    """
    p = _non_negative("pressure", pressure)
    return p >= graphite_diamond_boundary(temperature, model)


def hbn_cbn_boundary(temperature: float, model: str = "solozhenko_1999") -> float:
    """hBN-cBN equilibrium pressure (Pa) at ``temperature`` (K).

    Governing equations (straight lines through cited anchor points)::

        solozhenko_1999:      P = max(0, 0.003242 GPa/K * (T - 1660 K))     [solozhenko_1999]
                              (through 1660 K at P = 0 and the triple point 3480 K, 5.9 GPa)
        corrigan_bundy_1975:  P = 1.4 GPa + 0.002314 GPa/K * T              [corrigan_bundy_1975]
                              (through 1.4 GPa at 0 K and the triple point 3500 K, 9.5 GPa)

    Check (1800 K): 0.45 GPa (Solozhenko) vs 5.57 GPa (Corrigan-Bundy).

    The literature disagrees: the modern calorimetric diagram makes cBN the
    stable phase at ambient pressure below ~1660 K (so the boundary is 0 there),
    whereas the historical line never reaches P = 0.  Industrial catalytic cBN
    (4.5-6 GPa, 1500-2000 K) lies inside the cBN field of the modern line and
    straddles the historical one.  Validity: up to the triple point (~3500 K).
    """
    t = _positive("temperature", temperature)
    m = _model(model, HBN_CBN_MODELS, "hBN-cBN")
    if m == "solozhenko_1999":
        return gpa(max(0.0, 0.003242 * (t - SOLOZHENKO_ZERO_PRESSURE_TEMPERATURE)))
    return gpa(1.4 + 0.002314 * t)


def cbn_stable(pressure: float, temperature: float, model: str = "solozhenko_1999") -> bool:
    """True if cubic BN (not hexagonal) is the stable boron-nitride phase at (P, T).

    Governing relation: ``P >= hbn_cbn_boundary(T, model)`` [solozhenko_1999,
    corrigan_bundy_1975].  With the default model cBN is stable at ambient
    pressure below 1660 K (``cbn_stable(0, 300)`` is True).

    Args:
        pressure: Pa (>= 0).  temperature: K (> 0).  model: see :data:`HBN_CBN_MODELS`.

    Validity: thermodynamic stability only; the hBN -> cBN conversion is
    kinetically blocked without a solvent-catalyst or > ~8 GPa and > 2000 K.
    """
    p = _non_negative("pressure", pressure)
    return p >= hbn_cbn_boundary(temperature, model)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SynthesisRoute(ResultMixin):
    """One established synthesis route and the conditions it needs.

    Fields (SI): ``p_min``/``p_max`` sample pressure window [Pa] (``p_max``
    None = open-ended); ``t_min``/``t_max`` sample temperature window [K]
    (None for shock-heated routes); ``min_dwell`` [s] the shortest time the
    sample must be held at pressure and temperature (the order of the
    shortest successful runs); ``typical_dwell`` [s] the usual production
    range (order of magnitude; None when not established); ``kind`` one of
    :data:`ROUTE_KINDS`; ``heating``, ``product_form``, ``current_scale`` and
    ``reference`` describe practice.  Sources in :data:`REFERENCES`.
    """

    key: str
    product: str
    kind: str
    p_min: float = unit("Pa")
    p_max: float | None = unit("Pa")
    t_min: float | None = unit("K")
    t_max: float | None = unit("K")
    min_dwell: float = unit("s")
    typical_dwell: tuple[float, float] | None = unit("s")
    heating: str
    product_form: str
    current_scale: str
    reference: str
    confidence: Confidence = Confidence.REPRESENTATIVE

    def __post_init__(self) -> None:
        if self.kind not in ROUTE_KINDS:
            raise InputError(f"route {self.key!r}: kind must be one of {', '.join(ROUTE_KINDS)}")
        if not (math.isfinite(self.p_min) and self.p_min > 0.0):
            raise InputError(f"route {self.key!r}: p_min must be > 0 Pa")
        if self.p_max is not None and not self.p_max >= self.p_min:
            raise InputError(f"route {self.key!r}: p_max must be >= p_min")
        for name in ("t_min", "t_max"):
            t = getattr(self, name)
            if t is not None and not (math.isfinite(t) and t > 0.0):
                raise InputError(f"route {self.key!r}: {name} must be > 0 K")
        if self.t_min is not None and self.t_max is not None and self.t_max < self.t_min:
            raise InputError(f"route {self.key!r}: t_max must be >= t_min")
        if not (math.isfinite(self.min_dwell) and self.min_dwell > 0.0):
            raise InputError(f"route {self.key!r}: min_dwell must be > 0 s")
        if self.typical_dwell is not None:
            lo, hi = self.typical_dwell
            if not (0.0 < lo <= hi):
                raise InputError(f"route {self.key!r}: typical_dwell must be (lo, hi) with 0 < lo <= hi")

    @property
    def is_static(self) -> bool:
        """True for routes that need a sustained dwell (every kind except ``"shock"``)."""
        return self.kind != "shock"


def _route_table() -> dict[str, SynthesisRoute]:
    rows = (
        SynthesisRoute(
            key="diamond_hpht_catalytic", product="diamond (grit and single crystals)",
            kind="static_catalytic", p_min=gpa(5.0), p_max=gpa(6.5), t_min=1573.0, t_max=1873.0,
            min_dwell=600.0, typical_dwell=(3.6e3, 8.6e4),
            heating="metal-solvent catalyst, resistive heating",
            product_form="grit and single crystals",
            current_scale="industrial, tonnes per year (cubic/belt presses)",
            reference=REFERENCES["bundy_1955"]),
        SynthesisRoute(
            key="hpht_single_crystal", product="diamond (large single crystals)",
            kind="static_catalytic", p_min=gpa(5.0), p_max=gpa(6.0), t_min=1573.0, t_max=1773.0,
            min_dwell=1e5, typical_dwell=(1e5, 1e6),
            heating="temperature-gradient growth (metal solvent, resistive heating)",
            product_form="single crystals up to ~15 mm",
            current_scale="industrial",
            reference=REFERENCES["strong_wentorf_1972"]),
        SynthesisRoute(
            key="pcd_sintering", product="polycrystalline diamond (PCD) compacts",
            kind="static_sintering", p_min=gpa(5.5), p_max=gpa(7.0), t_min=1673.0, t_max=2073.0,
            min_dwell=60.0, typical_dwell=(60.0, 900.0),
            heating="resistive heating, Co infiltration from a WC-Co substrate",
            product_form="Co-bonded compacts (cutting tools)",
            current_scale="industrial",
            reference=REFERENCES["hall_1970"]),
        SynthesisRoute(
            key="diamond_direct_npd", product="nano-polycrystalline diamond (NPD)",
            kind="static_direct", p_min=gpa(15.0), p_max=gpa(25.0), t_min=2573.0, t_max=2773.0,
            min_dwell=60.0, typical_dwell=(60.0, 1200.0),
            heating="direct conversion (resistive heating, no catalyst)",
            product_form="nano-polycrystalline rods up to about 1 cm",
            current_scale="laboratory/pilot (Kawai multi-anvil; Irifune 2003, 2019)",
            reference=REFERENCES["irifune_2003"]),
        SynthesisRoute(
            key="cbn_catalytic", product="cubic boron nitride (cBN)",
            kind="static_catalytic", p_min=gpa(4.5), p_max=gpa(6.0), t_min=1500.0, t_max=2000.0,
            min_dwell=300.0, typical_dwell=(300.0, 1e4),
            heating="solvent-catalyst (Li3N/Mg3N2 class), resistive heating",
            product_form="grit",
            current_scale="industrial",
            reference=REFERENCES["wentorf_1957"]),
        SynthesisRoute(
            key="cbn_direct", product="cubic boron nitride (cBN), binderless",
            kind="static_direct", p_min=gpa(7.7), p_max=gpa(10.0), t_min=2000.0, t_max=3000.0,
            min_dwell=1.0, typical_dwell=(1.0, 1e3),
            heating="direct conversion (resistive heating, no catalyst)",
            product_form="compacts",
            current_scale="laboratory",
            reference=REFERENCES["bundy_wentorf_1963"] + "; " + REFERENCES["corrigan_bundy_1975"]),
        SynthesisRoute(
            key="lonsdaleite_shock", product="lonsdaleite (hexagonal diamond)",
            kind="shock", p_min=gpa(15.0), p_max=None, t_min=None, t_max=None,
            min_dwell=1e-7, typical_dwell=(1e-7, 1e-6),
            heating="shock-heated",
            product_form="micrometre-scale grains in shocked graphite",
            current_scale="laboratory (shock recovery)",
            reference=REFERENCES["hanneman_1967"] + "; " + REFERENCES["erskine_nellis_1991"]),
        SynthesisRoute(
            key="diamond_shock_dupont", product="diamond (shock-synthesised powder)",
            kind="shock", p_min=gpa(20.0), p_max=None, t_min=None, t_max=None,
            min_dwell=1e-7, typical_dwell=(1e-7, 1e-5),
            heating="shock-heated",
            product_form="nm-to-micrometre diamond powder",
            current_scale="industrial historically (explosive, DuPont, DeCarli & Jamieson 1961)",
            reference=REFERENCES["decarli_jamieson_1961"]),
        SynthesisRoute(
            key="c3n4_dac", product="carbon nitrides (C3N4, CN, CN2 polymorphs)",
            kind="laser_heated_dac", p_min=gpa(72.0), p_max=gpa(134.0), t_min=2000.0, t_max=3000.0,
            min_dwell=60.0, typical_dwell=None,
            heating="laser-heated diamond anvil cell (~2500 K)",
            product_form="microgram single crystals",
            current_scale="laboratory (Laniel et al., Adv. Mater. 36 (2024) 2308030)",
            reference=REFERENCES["laniel_2024"], confidence=Confidence.HANDBOOK),
    )
    return {r.key: r for r in rows}


SYNTHESIS_ROUTES: dict[str, SynthesisRoute] = _route_table()
"""Established routes keyed by name (pressures Pa, temperatures K, dwell s)."""

QUANTUM_MATERIALS_NOTE: str = (
    "Quantum-grade diamond (NV centres, isotopically engineered) is grown by chemical vapour deposition "
    "(CVD) at < 1 bar (Balasubramanian et al., Nat. Mater. 8 (2009) 383); superconducting hydrides "
    "(H3S ~150 GPa, Drozdov et al. 2015; LaH10 ~170 GPa, Drozdov et al. 2019) are not recoverable to "
    "ambient; no pressure-limited quantum material is enabled by the 1.6-4 GPa window."
)
"""Why the README's 'quantum materials' do not follow from a 1.6-4 GPa press."""


def _route(route: str | SynthesisRoute) -> SynthesisRoute:
    if isinstance(route, SynthesisRoute):
        return route
    if isinstance(route, str) and route in SYNTHESIS_ROUTES:
        return SYNTHESIS_ROUTES[route]
    raise InputError(f"unknown synthesis route {route!r}; choose one of {', '.join(SYNTHESIS_ROUTES)}")


# ---------------------------------------------------------------------------
# anvils
# ---------------------------------------------------------------------------
_MOVING_PARTS = ("anvils and pistons are moving mechanical parts that must themselves carry the sample "
                 "pressure (against the README's 'without a single moving mechanical part')")

_ANVIL_VOLUME_CLASS: dict[str, str] = {
    "belt_cubic_wc": "cm^3 (industrial production)",
    "kawai_wc": "up to ~0.5 cm^3 (laboratory; NPD rods up to about 1 cm at 15 GPa)",
    "kawai_sintered_diamond": "~0.1 mm^3 (laboratory)",
    "dac": "micrograms (laboratory)",
}


@dataclass(frozen=True, kw_only=True)
class AnvilClass(ResultMixin):
    """Static-press technology needed to hold a sample at ``sample_pressure``.

    ``technology`` is the :data:`magform.benchmarks.STATIC_PRESS_LADDER` key
    (``belt_cubic_wc``, ``kawai_wc``, ``kawai_sintered_diamond``, ``dac``) or
    ``"none"`` above the ladder; ``max_pressure`` [Pa] is that technology's
    demonstrated limit; ``sample_volume_class`` the sample size it holds at
    that pressure; ``status`` refers to that volume class; ``reference`` the
    ladder citation; ``note`` the moving-parts caveat [static_press_ladder].
    """

    sample_pressure: float = unit("Pa")
    technology: str
    max_pressure: float = unit("Pa")
    sample_volume_class: str
    status: Status
    reference: str = ""
    note: str = ""


def anvil_feasibility(sample_pressure: float) -> AnvilClass:
    """Smallest demonstrated static-press class that reaches ``sample_pressure``.

    Ladder (:data:`magform.benchmarks.STATIC_PRESS_LADDER`) [static_press_ladder]::

        belt/cubic presses, WC anvils      <= 7 GPa     cm^3, industrial        state of the art
        Kawai multi-anvil, WC cubes        <= 28 GPa    mm^3                    state of the art
        Kawai multi-anvil, sintered diamond<= 100 GPa   ~0.1 mm^3               state of the art
        diamond anvil cell                 <= 400 GPa   micrograms              state of the art
        above                              -            none demonstrated       aspirational

    The status refers to the stated sample-volume class: the volume falls by
    ~7 orders of magnitude from 7 to 400 GPa, so a "massive volume" at
    > 7 GPa would itself be an extrapolation.  Anvils are moving mechanical
    parts that must carry the sample pressure - contrary to the README's
    "without a single moving mechanical part".

    Args:
        sample_pressure: Pa (>= 0).

    Returns:
        :class:`AnvilClass`.
    """
    p = _non_negative("sample_pressure", sample_pressure)
    ladder = sorted(STATIC_PRESS_LADDER, key=lambda b: b.value)
    for b in ladder:
        if p <= b.value:
            return AnvilClass(
                sample_pressure=p, technology=b.key, max_pressure=b.value,
                sample_volume_class=_ANVIL_VOLUME_CLASS.get(b.key, b.description),
                status=Status.STATE_OF_THE_ART, reference=b.citation,
                note=f"{b.description}; {_MOVING_PARTS}",
            )
    top = ladder[-1]
    return AnvilClass(
        sample_pressure=p, technology="none", max_pressure=top.value,
        sample_volume_class="none demonstrated", status=Status.ASPIRATIONAL, reference=top.citation,
        note=f"above the {top.value / 1e9:.0f} GPa static record ({top.key}); {_MOVING_PARTS}",
    )


# ---------------------------------------------------------------------------
# flyer specification and assessment result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlyerSpec(ResultMixin):
    """A magnetically driven flyer plate for shock synthesis.

    ``material`` flyer material; ``thickness`` [m] and ``gap`` [m] (standoff
    over which the drive pressure accelerates it); ``sample`` the material it
    hits (default ``graphite_crystal``; ``graphite_polycrystalline`` for porous
    feedstock); ``efficiency`` fraction of the drive pressure that accelerates
    the flyer, (0, 1].  Velocity from :func:`magform.shock.flyer_velocity`
    ``v = sqrt(2*eta*P*g/(rho*h))`` [psyk_2011 in magform.shock].
    """

    material: MaterialLike
    thickness: float = unit("m")
    gap: float = unit("m")
    sample: MaterialLike = "graphite_crystal"
    efficiency: float = 1.0

    def __post_init__(self) -> None:
        _positive("thickness", self.thickness)
        _positive("gap", self.gap)
        eta = _finite("efficiency", self.efficiency)
        if not 0.0 < eta <= 1.0:
            raise InputError(f"efficiency must be in (0, 1], got {self.efficiency!r}")
        as_material(self.material)  # raise early on unknown keys
        as_material(self.sample)


@dataclass(frozen=True, kw_only=True)
class SynthesisAssessment(Assessment):
    """Can a magnetic pressure pulse execute a synthesis route?

    Fields (SI): ``pressure_drive`` [Pa] magnetic pressure applied;
    ``intensification`` anvil area ratio (static routes); ``pressure_sample``
    [Pa] pressure reaching the sample (``pressure_drive * intensification``
    for static routes, the impact pressure for shock routes, the drive
    pressure itself for a shock route assessed without a flyer); ``temperature``
    [K] sample (initial) temperature; ``pulse_duration`` [s];
    ``pressure_margin`` = ``pressure_sample / p_min`` (also ``margin``);
    ``temperature_ok`` (None for shock-heated routes); ``dwell_ok`` pulse (or
    shock) duration >= route minimum dwell; ``phase_stable`` product is the
    thermodynamically stable phase at the sample state (None when no boundary
    is modelled); ``anvil`` press class (static routes); ``flyer_velocity``
    [m/s] and ``shock`` (shock routes).
    """

    route: str
    product: str
    kind: str
    pressure_drive: float = unit("Pa")
    intensification: float
    pressure_sample: float = unit("Pa")
    temperature: float = unit("K")
    pulse_duration: float = unit("s")
    pressure_margin: float
    temperature_ok: bool | None
    dwell_ok: bool
    phase_stable: bool | None
    anvil: AnvilClass | None
    flyer_velocity: float | None = unit("m/s")
    shock: ShockState | None
    product_form: str
    current_scale: str


def _phase_system(route: SynthesisRoute) -> str | None:
    p = route.product.lower()
    if "boron nitride" in p or "cbn" in p:
        return "bn"
    if "diamond" in p or "lonsdaleite" in p:
        return "carbon"
    return None


def _phase_check(route: SynthesisRoute, pressure: float,
                 temperature: float) -> tuple[bool | None, bool | None, list[str]]:
    """Product stability at (P, T): (default model, alternative model, explanatory notes)."""
    system = _phase_system(route)
    if system == "carbon":
        pk = graphite_diamond_boundary(temperature, "kennedy_1976")
        pb = graphite_diamond_boundary(temperature, "berman_simon")
        notes = [f"graphite-diamond equilibrium at {temperature:.0f} K: {pk / 1e9:.2f} GPa (Kennedy & Kennedy "
                 f"1976) / {pb / 1e9:.2f} GPa (Berman & Simon 1955)"]
        lo, hi = KENNEDY_1976_CALIBRATION
        if not lo <= temperature <= hi:
            notes.append(f"Kennedy & Kennedy line extrapolated outside its {lo:.0f}-{hi:.0f} K calibration")
        if (pressure >= pk) != (pressure >= pb):
            notes.append("the two graphite-diamond models disagree at this state")
        if route.kind == "shock" or "lonsdaleite" in route.product.lower():
            notes.append("lonsdaleite and shock diamond are metastable products; the graphite-diamond line "
                         "is used as the sp3-carbon stability proxy")
        return pressure >= pk, pressure >= pb, notes
    if system == "bn":
        ps = hbn_cbn_boundary(temperature, "solozhenko_1999")
        pc = hbn_cbn_boundary(temperature, "corrigan_bundy_1975")
        notes = [f"hBN-cBN equilibrium at {temperature:.0f} K: {ps / 1e9:.2f} GPa (Solozhenko et al. 1999) / "
                 f"{pc / 1e9:.2f} GPa (Corrigan & Bundy 1975)"]
        if (pressure >= ps) != (pressure >= pc):
            notes.append("the two hBN-cBN models disagree at this state (literature disagreement)")
        return pressure >= ps, pressure >= pc, notes
    return None, None, ["no equilibrium phase boundary is modelled for this product"]


def assess_route(route: str | SynthesisRoute, pressure: float, temperature: float = T_ROOM,
                 pulse_duration: float = 1e-4, intensification: float = 1.0,
                 flyer: FlyerSpec | None = None) -> SynthesisAssessment:
    """Assess whether a magnetic pressure pulse can execute one synthesis route.

    Static and laser-heated-DAC routes (status precedence in this order; every
    failing check is listed in ``warnings``) [bundy_1955, irifune_2003]:

    1. ``pulse_duration < min_dwell`` -> INFEASIBLE: the pulse cannot supply
       the dwell, and a DC field exerts no pressure on a stationary,
       current-free sample (induced currents decay), so the magnet cannot
       simply hold the pressure;
    2. ``P_sample = pressure * intensification < p_min`` -> INFEASIBLE;
    3. ``temperature`` outside ``[t_min, t_max]`` -> INFEASIBLE (the magnet
       supplies no heating);
    4. product not thermodynamically stable at (P_sample, T) under both
       boundary models -> INFEASIBLE (catalysts change kinetics, not the
       equilibrium); unstable on the default model only -> at most
       EXTRAPOLATION (inside the literature disagreement).  This rule is an
       addition to the spec's list;
    5. ``intensification > 1`` -> ``worst(STATE_OF_THE_ART, anvil status)``
       plus a note that anvils are moving parts;
    6. otherwise STATE_OF_THE_ART: this is industrial/laboratory practice -
       the conventional press already does it.

    Shock routes need ``flyer`` (else NOT_APPLICABLE).  Governing chain
    [magform.shock]: ``v = sqrt(2*eta*P*g/(rho*h))``, then the impedance match
    ``rho_f(c_f + s_f(v-u))(v-u) = rho_s(c_s + s_s u)u`` gives ``P_sample``;
    ``P_sample >= p_min`` -> EXTRAPOLATION (magnetically driven flyer shock
    synthesis is not industrial; explosive shock synthesis is
    [decarli_jamieson_1961]), else INFEASIBLE.  Check: Cu 2 mm, 10 mm gap,
    4 GPa -> 2113 m/s -> ~22 GPa in graphite_crystal (not the 52 GPa Cu/Cu).

    Args:
        route: key of :data:`SYNTHESIS_ROUTES` or a :class:`SynthesisRoute`.
        pressure: magnetic drive pressure, Pa (>= 0).
        temperature: sample (initial) temperature, K (> 0).
        pulse_duration: duration of the pressure pulse, s (> 0).
        intensification: anvil area ratio for static routes (> 0; > 1 means anvils).
        flyer: :class:`FlyerSpec` for shock routes.

    Returns:
        :class:`SynthesisAssessment` with ``margin = pressure_sample / p_min``.

    Validity: route windows are representative of practice, not sharp
    limits; the shock path inherits the planar impedance-match limits of
    :func:`magform.shock.impedance_match` and assumes the drive pressure lasts
    the flight (a warning is issued otherwise).
    """
    r = _route(route)
    p_drive = _non_negative("pressure", pressure)
    t = _positive("temperature", temperature)
    tp = _positive("pulse_duration", pulse_duration)
    k = _positive("intensification", intensification)
    if flyer is not None and not isinstance(flyer, FlyerSpec):
        raise InputError("flyer must be a FlyerSpec or None")
    if r.is_static:
        return _assess_static(r, p_drive, t, tp, k)
    return _assess_shock(r, p_drive, t, tp, k, flyer)


def _dwell_text(r: SynthesisRoute) -> str:
    if r.typical_dwell is None:
        return f">= {r.min_dwell:.3g} s"
    lo, hi = r.typical_dwell
    return f">= {r.min_dwell:.3g} s (typical {lo:.3g}-{hi:.3g} s)"


def _assess_static(r: SynthesisRoute, p_drive: float, t: float, tp: float, k: float) -> SynthesisAssessment:
    p_sample = p_drive * k
    margin = p_sample / r.p_min
    dwell_ok = tp >= r.min_dwell
    t_ok = (r.t_min is None or t >= r.t_min) and (r.t_max is None or t <= r.t_max)
    anvil = anvil_feasibility(p_sample)
    stable, stable_alt, phase_notes = _phase_check(r, p_sample, t)

    reasons: list[str] = []   # each one makes the route INFEASIBLE
    cautions: list[str] = []  # each one caps the status at EXTRAPOLATION
    notes: list[str] = []
    if not dwell_ok:
        reasons.append(f"pulse cannot supply the dwell: a {tp:.3g} s pressure pulse vs {_dwell_text(r)} at "
                       f"pressure and temperature; a DC field exerts no pressure on a stationary sample, so "
                       f"the magnet cannot hold the pressure")
    if p_sample < r.p_min:
        reasons.append(f"sample pressure {p_sample / 1e9:.3g} GPa is below the route minimum "
                       f"{r.p_min / 1e9:.3g} GPa")
    if not t_ok:
        lo = f"{r.t_min:.0f}" if r.t_min is not None else "-"
        hi = f"{r.t_max:.0f}" if r.t_max is not None else "-"
        reasons.append(f"temperature {t:.0f} K is outside the route window {lo}-{hi} K; the magnet supplies "
                       f"no heating (route heating: {r.heating})")
    if stable is False and stable_alt is False:
        reasons.append(f"{r.product} is not the thermodynamically stable phase at {p_sample / 1e9:.3g} GPa and "
                       f"{t:.0f} K under either boundary model: the route converts nothing (catalysts change "
                       f"kinetics, not equilibrium)")
    elif stable is False:
        cautions.append(f"{r.product} is unstable at {p_sample / 1e9:.3g} GPa and {t:.0f} K on the default "
                        f"boundary but stable on the alternative one: the state sits on the equilibrium line "
                        f"within the literature disagreement")
    if r.p_max is not None and p_sample > r.p_max:
        notes.append(f"sample pressure {p_sample / 1e9:.3g} GPa is above this route's practice window "
                     f"({r.p_max / 1e9:.3g} GPa)")
    if k > 1.0:
        notes.append(f"intensification x{k:.3g} needs anvils ({anvil.technology}, sample volume "
                     f"{anvil.sample_volume_class}): {_MOVING_PARTS}")
    notes.extend(phase_notes)

    if reasons:
        status = Status.INFEASIBLE
    else:
        status = worst(Status.STATE_OF_THE_ART, anvil.status) if k > 1.0 else Status.STATE_OF_THE_ART
        if cautions:
            status = worst(status, Status.EXTRAPOLATION)
        if status is Status.STATE_OF_THE_ART and k <= 1.0:
            notes.append(f"matches established practice ({r.current_scale}): the conventional press already "
                         f"does this")
    warnings = reasons + cautions
    return SynthesisAssessment(
        status=status, margin=margin, warnings=tuple(warnings), notes=tuple(notes),
        route=r.key, product=r.product, kind=r.kind, pressure_drive=p_drive, intensification=k,
        pressure_sample=p_sample, temperature=t, pulse_duration=tp, pressure_margin=margin,
        temperature_ok=t_ok, dwell_ok=dwell_ok, phase_stable=stable, anvil=anvil,
        flyer_velocity=None, shock=None, product_form=r.product_form, current_scale=r.current_scale,
    )


def _assess_shock(r: SynthesisRoute, p_drive: float, t: float, tp: float, k: float,
                  flyer: FlyerSpec | None) -> SynthesisAssessment:
    notes: list[str] = []
    warnings: list[str] = []
    if k != 1.0:
        notes.append("intensification is not used for shock routes (the flyer impact sets the sample pressure)")
    if flyer is None:
        notes.append("shock route needs a FlyerSpec (flyer material, thickness, gap, sample); without a flyer "
                     "the sample sees only the drive pressure, far below any shock-synthesis window")
        margin = p_drive / r.p_min
        return SynthesisAssessment(
            status=Status.NOT_APPLICABLE, margin=margin, warnings=tuple(warnings), notes=tuple(notes),
            route=r.key, product=r.product, kind=r.kind, pressure_drive=p_drive, intensification=k,
            pressure_sample=p_drive, temperature=t, pulse_duration=tp, pressure_margin=margin,
            temperature_ok=None, dwell_ok=tp >= r.min_dwell, phase_stable=None, anvil=None,
            flyer_velocity=None, shock=None, product_form=r.product_form, current_scale=r.current_scale,
        )

    fl = as_material(flyer.material)
    sample = as_material(flyer.sample)
    v = flyer_velocity(p_drive, flyer.gap, fl, flyer.thickness, flyer.efficiency)
    state = impedance_match(fl, sample, v)
    p_sample = state.pressure
    margin = p_sample / r.p_min
    duration = shock_duration(flyer.thickness, state.shock_speed_flyer)
    dwell_ok = duration >= r.min_dwell
    d_t = shock_temperature_rise(state.particle_velocity, sample)
    stable, _, phase_notes = _phase_check(r, p_sample, t + d_t)

    notes.append(f"{fl.key} flyer {flyer.thickness * 1e3:.3g} mm over {flyer.gap * 1e3:.3g} mm at "
                 f"{p_drive / 1e9:.3g} GPa drive -> {v:.0f} m/s -> {p_sample / 1e9:.3g} GPa in {sample.key}")
    if sample.key != fl.key:
        p_sym = impedance_match(fl, fl, v).pressure
        side = "lower" if p_sample < p_sym else "higher"
        notes.append(f"a symmetric {fl.key}-on-{fl.key} impact would give {p_sym / 1e9:.3g} GPa; the sample's "
                     f"{side} shock impedance gives {p_sample / 1e9:.3g} GPa")
    notes.append(f"shock duration ~{duration:.2g} s (2h/Us, estimate); shock heating <= {d_t:.0f} K "
                 f"(u_p^2/(2 c_p) upper bound, estimate)")
    if state.branch_note:
        notes.append(f"Hugoniot: {state.branch_note}")
    if p_drive > 0.0:
        transit = flyer_transit_time(flyer.gap, p_drive, fl, flyer.thickness, flyer.efficiency)
        if transit > tp:
            v_imp = flyer.efficiency * p_drive * tp / (fl.density * flyer.thickness)
            warnings.append(f"the {tp:.3g} s drive pulse ends before the flyer crosses the gap ({transit:.3g} s): "
                            f"the constant-pressure velocity is an upper bound (impulse-limited ~{v_imp:.0f} m/s)")
    if not dwell_ok:
        warnings.append(f"shock duration {duration:.2g} s is below the {r.min_dwell:.2g} s typical of recovery "
                        f"experiments; formation can be nanosecond-fast (Kraus et al. 2016) but recovery is "
                        f"uncertain")
    if stable is False:
        warnings.append(f"at the estimated shock temperature ({t + d_t:.0f} K) the product is not "
                        f"thermodynamically stable; back-conversion on release is likely")
    notes.extend(phase_notes)

    if p_sample >= r.p_min:
        status = Status.EXTRAPOLATION
        notes.append("magnetically driven flyer shock synthesis is not industrial; explosive shock synthesis is "
                     "(DeCarli & Jamieson 1961; DuPont)")
    else:
        status = Status.INFEASIBLE
        warnings.append(f"impact pressure {p_sample / 1e9:.3g} GPa in {sample.key} is below the route minimum "
                        f"{r.p_min / 1e9:.3g} GPa")
    return SynthesisAssessment(
        status=status, margin=margin, warnings=tuple(warnings), notes=tuple(notes),
        route=r.key, product=r.product, kind=r.kind, pressure_drive=p_drive, intensification=k,
        pressure_sample=p_sample, temperature=t, pulse_duration=tp, pressure_margin=margin,
        temperature_ok=None, dwell_ok=dwell_ok, phase_stable=stable, anvil=None,
        flyer_velocity=v, shock=state, product_form=r.product_form, current_scale=r.current_scale,
    )


def assess_all(pressure: float, temperature: float = T_ROOM, pulse_duration: float = 1e-4,
               intensification: float = 1.0, flyer: FlyerSpec | None = None) -> tuple[SynthesisAssessment, ...]:
    """:func:`assess_route` for every route in :data:`SYNTHESIS_ROUTES` (table order).

    Same governing rules, units and validity as :func:`assess_route`
    [bundy_1955, decarli_jamieson_1961].  At 1.6-4 GPa, room temperature and
    a 0.1 ms pulse every static route is INFEASIBLE (dwell, pressure and
    heating all fail); shock routes are NOT_APPLICABLE without a flyer and
    EXTRAPOLATION/INFEASIBLE with one.
    """
    return tuple(assess_route(r, pressure, temperature, pulse_duration, intensification, flyer)
                 for r in SYNTHESIS_ROUTES.values())


# ---------------------------------------------------------------------------
# volume scaling
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class VolumeScaling(ResultMixin):
    """Masses and energies of a hypothetical graphite charge filling a magnet bore.

    ``sample_volume`` [m^3] = fill_fraction * pi a^2 L; ``graphite_mass`` [kg]
    at crystal density; ``diamond_mass_if_converted`` [kg] (mass is conserved);
    ``mechanical_work`` [J] = P * dV with dV = V (1 - rho_graphite/rho_diamond);
    ``bore_magnetic_energy`` [J] = (B^2/2 mu0) * pi a^2 L = P * bore volume;
    ``pressure`` [Pa] the magnetic pressure used.
    """

    bore_radius: float = unit("m")
    length: float = unit("m")
    fill_fraction: float
    sample_volume: float = unit("m^3")
    graphite_mass: float = unit("kg")
    diamond_mass_if_converted: float = unit("kg")
    mechanical_work: float = unit("J")
    bore_magnetic_energy: float = unit("J")
    pressure: float = unit("Pa")


def volume_scaling(bore_radius: float, length: float, pressure: float,
                   fill_fraction: float = 0.5) -> VolumeScaling:
    """What a large-bore press would hold if it could convert its charge.

    Governing equations::

        V_bore = pi * a**2 * L,      V = f * V_bore
        m_graphite = rho_g * V = m_diamond (mass conserved)
        W = P * dV,  dV = V * (1 - rho_g / rho_d)      (rho_g 2260, rho_d 3515 kg/m^3 -> dV/V = 0.357)
        E_bore = (B**2 / (2 mu0)) * V_bore = P * V_bore

    Args:
        bore_radius: a, m (> 0).  length: L, m (> 0).
        pressure: magnetic pressure P = B^2/(2 mu0), Pa (>= 0).
        fill_fraction: f, sample fraction of the bore volume, (0, 1].

    Returns:
        :class:`VolumeScaling`.

    Validity: bookkeeping only - it does not imply conversion happens (the
    dwell and temperature of :func:`assess_route` still rule static routes
    out).  Uses crystalline graphite (thermodynamic reference); porous
    isostatic feedstock (1800 kg/m^3) would give dV/V = 0.49.  The field
    energy in the winding and fringe adds to ``bore_magnetic_energy``.
    References: densities from the CRC Handbook via :mod:`magform.materials`
    [kennedy_1976 for the phase volume change context].
    """
    a = _positive("bore_radius", bore_radius)
    length_ = _positive("length", length)
    p = _non_negative("pressure", pressure)
    f = _finite("fill_fraction", fill_fraction)
    if not 0.0 < f <= 1.0:
        raise InputError(f"fill_fraction must be in (0, 1], got {fill_fraction!r}")
    graphite = get("graphite_crystal")
    diamond = get("diamond")
    v_bore = math.pi * a * a * length_
    v = f * v_bore
    mass = graphite.density * v
    d_v = v * (1.0 - graphite.density / diamond.density)
    return VolumeScaling(
        bore_radius=a, length=length_, fill_fraction=f, sample_volume=v, graphite_mass=mass,
        diamond_mass_if_converted=mass, mechanical_work=p * d_v, bore_magnetic_energy=p * v_bore,
        pressure=p,
    )


# ---------------------------------------------------------------------------
# plotting helper
# ---------------------------------------------------------------------------
def phase_boundary_curves(t_min: float = 300, t_max: float = 3500,
                          n: int = 50) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """Equilibrium lines sampled for plotting: ``{name: (temperatures [K], pressures [Pa])}``.

    Keys: ``graphite_diamond_kennedy_1976``, ``graphite_diamond_berman_simon``,
    ``hbn_cbn_solozhenko_1999``, ``hbn_cbn_corrigan_bundy_1975``; equations as
    in :func:`graphite_diamond_boundary` and :func:`hbn_cbn_boundary`
    [kennedy_1976, berman_simon_1955, solozhenko_1999, corrigan_bundy_1975].
    The one dict-returning public function (spec exception).

    Args:
        t_min, t_max: temperature range, K (0 < t_min < t_max).
        n: number of samples (>= 2).

    Validity: as the underlying lines (Kennedy measured 1373-1898 K; the
    hBN-cBN lines end at their triple points near 3500 K).
    """
    lo = _positive("t_min", t_min)
    hi = _positive("t_max", t_max)
    if hi <= lo:
        raise InputError("t_max must be greater than t_min")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise InputError(f"n must be an integer >= 2, got {n!r}")
    ts = tuple(linspace(lo, hi, n))
    return {
        "graphite_diamond_kennedy_1976": (ts, tuple(graphite_diamond_boundary(x, "kennedy_1976") for x in ts)),
        "graphite_diamond_berman_simon": (ts, tuple(graphite_diamond_boundary(x, "berman_simon") for x in ts)),
        "hbn_cbn_solozhenko_1999": (ts, tuple(hbn_cbn_boundary(x, "solozhenko_1999") for x in ts)),
        "hbn_cbn_corrigan_bundy_1975": (ts, tuple(hbn_cbn_boundary(x, "corrigan_bundy_1975") for x in ts)),
    }


__all__ = [
    "REFERENCES",
    "GRAPHITE_DIAMOND_MODELS",
    "HBN_CBN_MODELS",
    "KENNEDY_1976_CALIBRATION",
    "BERMAN_SIMON_MIN_TEMPERATURE",
    "SOLOZHENKO_ZERO_PRESSURE_TEMPERATURE",
    "SOLOZHENKO_TRIPLE_POINT",
    "CORRIGAN_BUNDY_TRIPLE_POINT",
    "ROUTE_KINDS",
    "graphite_diamond_boundary",
    "diamond_stable",
    "hbn_cbn_boundary",
    "cbn_stable",
    "SynthesisRoute",
    "SYNTHESIS_ROUTES",
    "QUANTUM_MATERIALS_NOTE",
    "AnvilClass",
    "anvil_feasibility",
    "FlyerSpec",
    "SynthesisAssessment",
    "assess_route",
    "assess_all",
    "VolumeScaling",
    "volume_scaling",
    "phase_boundary_curves",
]
