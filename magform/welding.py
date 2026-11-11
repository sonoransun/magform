"""Impact (magnetic-pulse) welding of dissimilar metals: the weldability window.

The concept note's third process is "Room-Temperature Cold Welding of Dissimilar
Alloys": driving e.g. titanium onto copper or steel onto aluminium "with such
extreme, instantaneous pressure that their atomic lattices are forced to
interlock", giving "a perfect atomic bond across large, complex geometric
interfaces without creating a heat-affected zone (HAZ)" and bypassing "the
brittle intermetallic phases".  This module grades that paragraph.

What is true
------------
Solid-state welding by high-velocity oblique impact is industrial practice:
explosive welding since the 1960s (Crossland 1982) and magnetic pulse welding
(MPW) of Al-Cu, Al-steel, Cu-steel and similar pairs (Aizawa et al. 2007;
Kapil & Sharma 2015).  The bulk of both parts stays cold and the intermetallic
layer is thin.  For the demonstrated pairs in :data:`DEMONSTRATED_PAIRS` the
capability is state of the art.

What the note gets wrong (each point is a number in this module)
----------------------------------------------------------------
1. **The bond is made by an oblique collision, not by pressure.**  A re-entrant
   jet must strip the oxide films, which needs a collision point that travels
   slower than sound in both metals (:func:`jetting_sonic_limit`; Walsh et al.
   1953), an impact fast enough to make both surfaces flow
   (:func:`lower_limit_velocity`; Deribas & Zakharenko 1974) and slow enough
   that the interface does not stay molten (:func:`upper_limit_velocity`;
   Wittman 1973).  Together these bound a *window* (:func:`weld_window`), about
   370-650 m/s for 1 mm aluminium 6061 on steel at a 10 degree collision angle.
   A flat-on impact (zero angle) never welds (:func:`critical_collision_angle`).
2. **The standoff gap, not the field, makes the gigapascals.**  A 1 mm
   aluminium flyer driven across 2 mm by 0.1 GPa (15.9 T) lands at 385 m/s and
   puts 4.1 GPa on the steel: the gap amplifies the drive pressure ~41x
   (:func:`assess_weld`).  The whole drive window is 0.09-0.28 GPa (15-27 T;
   :func:`drive_pressure_window`), inside industrial electromagnetic forming.
3. **1.6-4 GPa overshoots.**  The same joint driven at the note's pressures
   arrives at 1.5-2.4 km/s: the collision point is supersonic (no jet) and far
   beyond the melting limit, so there is no weld; the gap would have to shrink
   to 0.05-0.36 mm, crossed in 0.25-1.1 us (:func:`standoff_window`), and the
   aluminium flyer's eddy-current skin melts above ~58 T.
4. **Rise time, not peak pressure, decides.**  Under a 5 ms half-sine pulse
   peaking at 4 GPa the flyer leaves early and crosses the gap in ~80 us, when
   the pressure is still ~10 MPa: it lands at ~100 m/s, below the bonding
   threshold (:func:`flyer_impact`).  Welding needs ~10 us rise times.
5. **Not zero heat, not zero intermetallics.**  The collision dissipates
   ~10^5 J/m^2; if a tenth of it stays at the interface it melts a layer of
   order 10 um that quenches in ~1 us (:func:`equivalent_melt_layer`,
   :func:`thermal_quench_time`).  Solid-state interdiffusion in that time is
   below a nanometre and liquid interdiffusion tens of nanometres, against
   tens of micrometres in a fusion weld that stays liquid for a second
   (:func:`interdiffusion_length`): intermetallics are confined to the melted
   pockets, roughly a thousand times thinner than in fusion welding, not absent.

Conventions: SI units, angles in radians.  ``Vp`` is the flyer (impact)
velocity, ``beta`` the dynamic collision angle between the two surfaces and
``Vc = Vp / sin(beta)`` the velocity of the collision point along the target.
The weld-window limits are empirical correlations calibrated on explosive
welding of similar metals; for dissimilar pairs their constants are uncertain
by tens of per cent, so every limit here is an engineering estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from ._numerics import bisect
from ._report import Assessment, ResultMixin, unit
from .benchmarks import EMF_CONVENTIONAL_FIELD
from .constants import R_GAS, T_ROOM
from .eddy import sheet_shielding_factor, skin_depth_for_material, skin_melting_field, skin_temperature_rise
from .errors import InputError
from .magnetics import field_for_pressure
from .materials import Material, MaterialLike, as_material
from .shock import (
    flyer_transit_time,
    flyer_velocity,
    impedance_match,
    required_pressure_for_velocity,
    required_standoff,
)
from .status import Status, worst

REFERENCES: dict[str, str] = {
    "crossland_1982": "B. Crossland, Explosive Welding of Metals and its Application, Clarendon Press, "
                      "Oxford (1982).",
    "deribas_zakharenko_1974": "A. A. Deribas and I. D. Zakharenko, 'Surface effects with oblique collisions "
                               "between metallic plates', Combust. Explos. Shock Waves 10 (1974) 358-367.",
    "wittman_1973": "R. H. Wittman, 'The influence of collision parameters on the strength and microstructure "
                    "of an explosion welded aluminium alloy', Proc. 2nd Int. Symp. on Use of Explosive Energy "
                    "in Manufacturing Metallic Materials, Marianske Lazne (1973) 153-168.",
    "de_rosset_2006": "W. S. de Rosset, 'Analysis of explosive bonding parameters', Mater. Manuf. Process. 21 "
                      "(2006) 634-638.",
    "cowan_1971": "G. R. Cowan, O. R. Bergmann and A. H. Holtzman, 'Mechanism of bond zone wave formation in "
                  "explosion-clad metals', Metall. Trans. 2 (1971) 3145-3155.",
    "walsh_1953": "J. M. Walsh, R. G. Shreffler and F. J. Willig, 'Limiting conditions for jet formation in "
                  "high velocity collisions', J. Appl. Phys. 24 (1953) 349-359.",
    "ribeiro_2014": "J. B. Ribeiro, R. Mendes and A. Loureiro, 'Review of the weldability window concept and "
                    "equations for explosive welding', J. Phys.: Conf. Ser. 500 (2014) 052038.",
    "lysak_kuzmin_2012": "V. I. Lysak and S. V. Kuzmin, 'Lower boundary in metal explosive welding. Evolution "
                         "of ideas', J. Mater. Process. Technol. 212 (2012) 150-156.",
    "findik_2011": "F. Findik, 'Recent developments in explosive welding', Mater. Des. 32 (2011) 1081-1093.",
    "kapil_sharma_2015": "A. Kapil and A. Sharma, 'Magnetic pulse welding: an efficient and environmentally "
                         "friendly multi-material joining technique', J. Clean. Prod. 100 (2015) 35-58.",
    "psyk_2011": "V. Psyk, D. Risch, B. L. Kinsey, A. E. Tekkaya and M. Kleiner, 'Electromagnetic forming - "
                 "a review', J. Mater. Process. Technol. 211 (2011) 787-829.",
    "aizawa_2007": "T. Aizawa, M. Kashani and K. Okagawa, 'Application of magnetic pulse welding for aluminum "
                   "alloys and SPCC steel sheet joints', Weld. J. 86 (2007) 119s-124s.",
    "zhang_2011": "Y. Zhang, S. S. Babu, C. Prothe, M. Blakely, J. Kwasegroch, M. LaHa and G. S. Daehn, "
                  "'Application of high velocity impact welding at varied different length scales', "
                  "J. Mater. Process. Technol. 211 (2011) 944-952.",
    "zhang_2010": "Y. Zhang, S. S. Babu and G. S. Daehn, 'Interfacial ultrafine-grained structures on aluminum "
                  "alloy 6061 joint and copper alloy 110 joint fabricated by magnetic pulse welding', "
                  "J. Mater. Sci. 45 (2010) 4645-4651.",
    "kore_2008": "S. D. Kore, P. P. Date and S. V. Kulkarni, 'Electromagnetic impact welding of aluminum to "
                 "stainless steel sheets', J. Mater. Process. Technol. 208 (2008) 486-493.",
    "kore_2009": "S. D. Kore, J. Imbert, M. J. Worswick and Y. Zhou, 'Electromagnetic impact welding of Mg to "
                 "Al sheets', Sci. Technol. Weld. Join. 14 (2009) 549-553.",
    "raoelison_2014": "R. N. Raoelison, D. Racine, Z. Zhang, N. Buiron, D. Marceau and M. Rachik, 'Magnetic "
                      "pulse welding: interface of Al/Cu joint and investigation of intermetallic formation "
                      "effect on the weld features', J. Manuf. Process. 16 (2014) 427-434.",
    "kahraman_2005": "N. Kahraman and B. Gulenc, 'Microstructural and mechanical properties of Cu-Ti plates "
                     "bonded through explosive welding process', J. Mater. Process. Technol. 169 (2005) 67-71.",
    "brown_ashby_1980": "A. M. Brown and M. F. Ashby, 'Correlations for diffusion constants', Acta Metall. 28 "
                        "(1980) 1085-1101.",
    "iida_guthrie_1988": "T. Iida and R. I. L. Guthrie, The Physical Properties of Liquid Metals, Clarendon "
                         "Press, Oxford (1988) (liquid-metal diffusivities of order 1e-9 m^2/s).",
    "carslaw_jaeger_1959": "H. S. Carslaw and J. C. Jaeger, Conduction of Heat in Solids, 2nd ed., Oxford "
                           "University Press (1959).",
    "meyers_1994": "M. A. Meyers, Dynamic Behavior of Materials, Wiley, New York (1994), ch. 17 (explosive "
                   "welding, jetting, wave formation).",
    "herlach_miura_2003": "F. Herlach and N. Miura (eds.), High Magnetic Fields: Science and Technology, Vol. 1, "
                          "World Scientific, Singapore (2003) (single-turn coils: megagauss fields with "
                          "microsecond rise times).",
}
"""Short key -> full citation for every model in this module."""

# ---------------------------------------------------------------------------
# module constants (documented single sources)
# ---------------------------------------------------------------------------
DERIBAS_K_CLEAN: float = 0.6
"""Lower-limit coefficient k for carefully cleaned surfaces: ``beta_min = k sqrt(HV/(rho Vc^2))``
(Deribas & Zakharenko 1974; Ribeiro et al. 2014)."""

DERIBAS_K_TECHNICAL: float = 1.2
"""Lower-limit coefficient k for imperfectly cleaned (technical) surfaces (Deribas & Zakharenko 1974)."""

COWAN_REYNOLDS_TRANSITION: float = 10.6
"""Collision Reynolds number ``(rho_f + rho_t) Vc^2 / (2 (HV_f + HV_t))`` at which the bond line
changes from flat to wavy; mean of the metal pairs studied by Cowan, Bergmann & Holtzman (1971)."""

WITTMAN_N: float = 0.11
"""Empirical constant N of the upper (interface-melting) limit (Wittman 1973, as used by de Rosset
2006).  Confidence: estimated - values from roughly 0.04 to 0.14 are quoted for other metal
combinations in the weld-window literature (Ribeiro et al. 2014), and the limit scales as 1/N."""

DEFAULT_COLLISION_ANGLE: float = math.radians(10.0)
"""rad.  Default dynamic collision angle (middle of the magnetic-pulse-welding range)."""

MPW_TYPICAL_ANGLE: tuple[float, float] = (math.radians(5.0), math.radians(20.0))
"""rad.  Collision angles typical of magnetic pulse welding (representative; Kapil & Sharma 2015)."""

MPW_TYPICAL_VELOCITY: tuple[float, float] = (200.0, 500.0)
"""m/s.  Impact velocities typical of magnetic pulse welding (representative; Zhang et al. 2011;
Kapil & Sharma 2015)."""

README_PRESSURE_RANGE: tuple[float, float] = (1.6e9, 4.0e9)
"""Pa.  The concept note's '1,600 to over 4,000 MPa of magnetic pressure' (docs/archive/original-README.md)."""

INTERFACE_ENERGY_FRACTION: float = 0.1
"""Fraction of the collision energy assumed to stay at the interface (estimated).  The rest goes
into the jet, bulk plastic work and stress waves.  A tenth reproduces the micrometre-scale melt
pockets and intermetallic seams observed in impact welds; treat the result as an order of magnitude."""

DIFFUSION_PREFACTOR: float = 1.0e-4
"""m^2/s (estimated).  Pre-exponential D0 of ``D = D0 exp(-Q/(R T))``; lattice self-diffusion in
metals has D0 ~ 5e-5 to 2e-4 m^2/s (Brown & Ashby 1980).  Q is ``Material.self_diffusion_activation``."""

LIQUID_DIFFUSIVITY: float = 1.0e-9
"""m^2/s (representative).  Interdiffusion coefficient in liquid metals near their melting point
(Iida & Guthrie 1988)."""

FUSION_WELD_TIME: float = 1.0
"""s (order of magnitude).  Time a fusion-weld pool stays liquid, used only for the comparison
length :attr:`WeldAssessment.fusion_weld_interdiffusion`."""

DRIVER_TRANSFER_THRESHOLD: float = 0.8
"""Pressure-transfer factor ``1 - exp(-2 h/delta)`` below which the flyer shields the field too
poorly to be driven directly and needs a copper or aluminium driver sheet or a faster pulse
(estimated design boundary)."""

FASTEST_HIGH_FIELD_RISE: float = 2.5e-6
"""s (representative).  Rise time of destructive single-turn coils, the fastest sources of
100 T-class fields (2-3 us; Herlach & Miura 2003).  Flights shorter than this cannot see a
constant drive pressure from any existing 50 T+ magnet."""

PULSE_SHAPES: tuple[str, str] = ("constant", "half_sine")
"""Drive-pressure waveforms of :func:`flyer_impact`: a constant (rectangular) pressure, or the
``sin^2`` pressure of a half-sine field pulse."""

WELDABLE_CATEGORIES: tuple[str, ...] = ("metal", "conductor")
"""Material categories (commercial ductile alloys) for which :data:`DEMONSTRATED_PAIRS` is consulted."""

_SMALL_THETA: float = 0.1  # below this the half-sine kinematics use their power series


# ---------------------------------------------------------------------------
# demonstrated pairs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeldPair(ResultMixin):
    """A base-element pair that has been impact welded, with the intermetallics it can form.

    ``elements`` are the two base elements in alphabetical order; ``intermetallics`` the
    compounds of the binary system that appear when the interface melts (empty for similar
    metals and for systems without compounds); ``processes`` the impact-welding processes
    that have produced sound joints (``"magnetic_pulse"``, ``"explosive"``); ``citation``
    the source; ``note`` any qualifier.  The physics is the same for both processes: an
    oblique collision inside the weldability window.

    Reference: Crossland (1982); Kapil & Sharma (2015).
    """

    elements: tuple[str, str]
    intermetallics: tuple[str, ...]
    processes: tuple[str, ...]
    citation: str
    note: str = ""

    @property
    def key(self) -> str:
        """Registry key, e.g. ``"Al-Fe"``."""
        return "-".join(self.elements)


def _pairs() -> dict[str, WeldPair]:
    r = REFERENCES
    rows = (
        WeldPair(("Al", "Al"), (), ("magnetic_pulse", "explosive"),
                 f"{r['aizawa_2007']} {r['zhang_2010']}", "similar metals: no intermetallics"),
        WeldPair(("Al", "Cu"), ("Al2Cu", "AlCu", "Al4Cu9"), ("magnetic_pulse", "explosive"),
                 f"{r['raoelison_2014']} {r['kapil_sharma_2015']}",
                 "intermetallics appear as thin, discontinuous pockets where the interface melted"),
        WeldPair(("Al", "Fe"), ("Fe4Al13", "Fe2Al5", "FeAl"), ("magnetic_pulse", "explosive"),
                 f"{r['aizawa_2007']} {r['kore_2008']}", "the concept note's 'Steel to Aluminum'"),
        WeldPair(("Al", "Mg"), ("Al3Mg2", "Al12Mg17"), ("magnetic_pulse",), r["kore_2009"]),
        WeldPair(("Al", "Ti"), ("TiAl3", "TiAl", "Ti3Al"), ("explosive", "magnetic_pulse"),
                 f"{r['findik_2011']} {r['kapil_sharma_2015']}"),
        WeldPair(("Cu", "Cu"), (), ("magnetic_pulse", "explosive"), r["zhang_2010"],
                 "similar metals: no intermetallics"),
        WeldPair(("Cu", "Fe"), (), ("magnetic_pulse", "explosive"),
                 f"{r['kapil_sharma_2015']} {r['crossland_1982']}",
                 "Cu and Fe are nearly immiscible and form no intermetallic compound"),
        WeldPair(("Cu", "Ti"), ("CuTi2", "CuTi", "Cu4Ti3", "Cu4Ti"), ("explosive",), r["kahraman_2005"],
                 "the concept note's 'Titanium to Copper'; demonstrated by explosive welding, "
                 "magnetic-pulse reports are sparse"),
        WeldPair(("Fe", "Fe"), (), ("explosive",), r["crossland_1982"],
                 "stainless-clad steel plate; similar metals"),
        WeldPair(("Fe", "Ti"), ("TiFe", "TiFe2"), ("explosive",), r["crossland_1982"],
                 "titanium-clad steel plate is the classic explosive-cladding product"),
    )
    return {p.key: p for p in rows}


DEMONSTRATED_PAIRS: dict[str, WeldPair] = _pairs()
"""Impact-welded base-element pairs keyed ``"A-B"`` with A <= B alphabetically (see :class:`WeldPair`)."""


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise InputError(f"{name} must be a real number, got {value!r}")
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(x):
        raise InputError(f"{name} must be finite, got {value!r}")
    return x


def _positive(name: str, value: object) -> float:
    x = _finite(name, value)
    if x <= 0.0:
        raise InputError(f"{name} must be > 0, got {value!r}")
    return x


def _non_negative(name: str, value: object) -> float:
    x = _finite(name, value)
    if x < 0.0:
        raise InputError(f"{name} must be >= 0, got {value!r}")
    return x


def _fraction(name: str, value: object) -> float:
    x = _finite(name, value)
    if not 0.0 <= x <= 1.0:
        raise InputError(f"{name} must lie in [0, 1], got {value!r}")
    return x


def _efficiency(value: object) -> float:
    x = _finite("efficiency", value)
    if not 0.0 < x <= 1.0:
        raise InputError(f"efficiency must be in (0, 1], got {value!r}")
    return x


def _angle(value: object) -> float:
    b = _finite("collision_angle", value)
    if not 0.0 < b < 0.5 * math.pi:
        raise InputError(f"collision_angle must lie in (0, pi/2) rad, got {value!r}")
    return b


def _pulse_shape(value: object) -> str:
    s = value.strip().lower() if isinstance(value, str) else value
    if s not in PULSE_SHAPES:
        raise InputError(f"pulse_shape must be one of {PULSE_SHAPES}, got {value!r}")
    return s  # type: ignore[return-value]


def _metal(material: MaterialLike, role: str) -> Material:
    m = as_material(material)
    if not m.is_metal:
        raise InputError(f"{role} '{m.key}' is not a metal (category '{m.category}'): the impact-welding "
                         "window applies to metals")
    return m


def _initial_temperature(value: object, *mats: Material) -> float:
    t = _positive("initial_temperature", value)
    for m in mats:
        if t >= m.melting_point:
            raise InputError(f"initial_temperature {t:g} K is at or above the melting point of '{m.key}' "
                             f"({m.melting_point:g} K)")
    return t


def _flow_speed(m: Material) -> float:
    """sqrt(HV/rho), m/s: the impact speed scale at which the dynamic pressure equals the hardness."""
    return math.sqrt(m.hardness_estimate() / m.density)


def _melting_member(f: Material, t: Material) -> Material:
    return f if f.melting_point <= t.melting_point else t


def _wittman_group(member: Material, thickness: float, n: float) -> float:
    """W = (1/N) sqrt(Tm c0) (kappa cp c0 / (rho h))^(1/4), m^2/s^2, so that Vp,max = W / Vc."""
    c0 = member.bulk_sound_speed()
    return (math.sqrt(member.melting_point * c0)
            * (member.thermal_conductivity * member.specific_heat * c0 / (member.density * thickness)) ** 0.25 / n)


def _deg(angle: float) -> str:
    return f"{math.degrees(angle):.3g} deg"


def _gpa(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value / 1e9:.3g} GPa"


# ---------------------------------------------------------------------------
# pair lookup
# ---------------------------------------------------------------------------
def pair_key(flyer: MaterialLike, target: MaterialLike) -> str:
    """Base-element key of a flyer/target pair, e.g. ``"Al-Fe"`` (alphabetical, order-independent).

    Uses ``Material.base_element`` of each member, so every steel is "Fe" and every
    titanium alloy "Ti".  Dimensionless label; valid for any registry material.

    Reference: Kapil & Sharma (2015) (pairs are reported by base metal).
    """
    f = as_material(flyer)
    t = as_material(target)
    return "-".join(sorted((f.base_element, t.base_element)))


def demonstrated_pair(flyer: MaterialLike, target: MaterialLike) -> WeldPair | None:
    """The :class:`WeldPair` record of a flyer/target pair, or ``None`` if none is recorded.

    A pair counts only when both members are commercial ductile alloys
    (``category`` in :data:`WELDABLE_CATEGORIES`) and their base elements are in
    :data:`DEMONSTRATED_PAIRS`.  ``None`` means "no impact weld recorded here", not
    "impossible": e.g. tantalum-clad steel exists but is not tabulated.

    Reference: Crossland (1982); Kapil & Sharma (2015); Aizawa et al. (2007).
    """
    f = as_material(flyer)
    t = as_material(target)
    if f.category not in WELDABLE_CATEGORIES or t.category not in WELDABLE_CATEGORIES:
        return None
    return DEMONSTRATED_PAIRS.get(pair_key(f, t))


# ---------------------------------------------------------------------------
# collision kinematics
# ---------------------------------------------------------------------------
def collision_point_velocity(impact_velocity: float, collision_angle: float) -> float:
    """Velocity of the collision point along the target, ``Vc = Vp / sin(beta)`` [m/s].

    ``impact_velocity`` Vp in m/s (>= 0), the flyer velocity normal to its own surface;
    ``collision_angle`` beta in rad, 0 < beta < pi/2, the dynamic angle between the two
    surfaces.  Other set-ups give ``Vp / tan(beta)`` (flyer moving normal to the target) or
    ``Vp / (2 sin(beta/2))`` (parallel explosive welding); the three agree within 7 % for
    beta <= 20 degrees, well inside the scatter of the window limits.  Vc diverges as
    beta -> 0: a flat-on impact has no travelling collision point and cannot jet.

    Reference: Crossland (1982); Meyers (1994) ch. 17.
    """
    vp = _non_negative("impact_velocity", impact_velocity)
    b = _angle(collision_angle)
    return vp / math.sin(b)


def impact_velocity_for_collision(collision_velocity: float, collision_angle: float) -> float:
    """Impact velocity that gives a collision-point velocity, ``Vp = Vc sin(beta)`` [m/s].

    Inverse of :func:`collision_point_velocity`; ``collision_velocity`` Vc in m/s (>= 0),
    ``collision_angle`` in rad, 0 < beta < pi/2.  Same validity.

    Reference: Crossland (1982).
    """
    vc = _non_negative("collision_velocity", collision_velocity)
    b = _angle(collision_angle)
    return vc * math.sin(b)


def jetting_sonic_limit(flyer: MaterialLike, target: MaterialLike) -> float:
    """Largest collision-point velocity that still forms a jet: the smaller bulk sound speed [m/s].

    ``Vc < min(c0_flyer, c0_target)`` with ``c0 = sqrt(K/rho)``
    (:meth:`Material.bulk_sound_speed`).  If the collision point outruns sound in either
    metal the flow ahead of it cannot be turned, an attached oblique shock forms instead of
    a re-entrant jet, the oxide films are not removed and no bond results.  This is the
    right-hand boundary of the weldability window.  Validity: the usual engineering
    criterion; the exact jetting limit depends on the angle (Walsh et al. 1953) and is
    slightly supersonic at large angles.

    Reference: Walsh, Shreffler & Willig (1953); Cowan, Bergmann & Holtzman (1971).
    """
    f = _metal(flyer, "flyer")
    t = _metal(target, "target")
    return min(f.bulk_sound_speed(), t.bulk_sound_speed())


def critical_collision_angle(flyer: MaterialLike, target: MaterialLike, impact_velocity: float) -> float | None:
    """Smallest collision angle with a subsonic collision point, ``beta = asin(Vp / c_min)`` [rad].

    ``impact_velocity`` Vp in m/s (>= 0); ``c_min`` = :func:`jetting_sonic_limit`.  Below this
    angle the collision point is supersonic and nothing bonds, which is why the centre of a
    magnetic-pulse lap joint (where the sheets meet flat-on) is always unwelded and why a
    large or complex interface bonds only along the band where the angle is right.  Returns
    ``None`` when Vp >= c_min: no angle below 90 degrees gives a subsonic collision point.
    Check: aluminium 6061 on steel at 400 m/s -> 5.05 degrees.

    Reference: Walsh, Shreffler & Willig (1953); Zhang et al. (2011).
    """
    vp = _non_negative("impact_velocity", impact_velocity)
    c = jetting_sonic_limit(flyer, target)
    if vp >= c:
        return None
    return math.asin(vp / c)


# ---------------------------------------------------------------------------
# weldability-window limits
# ---------------------------------------------------------------------------
def lower_limit_velocity(flyer: MaterialLike, target: MaterialLike, coefficient: float = DERIBAS_K_CLEAN) -> float:
    """Smallest impact velocity that bonds, ``Vp,min = k max_i sqrt(HV_i / rho_i)`` [m/s].

    Deribas & Zakharenko give the lower boundary as ``beta_min = k sqrt(HV / (rho Vc^2))``
    (beta in rad); with ``Vp = Vc sin(beta) ~ Vc beta`` this is a constant impact velocity.
    HV is the Vickers hardness in Pa (:meth:`Material.hardness_estimate`), rho the density;
    ``coefficient`` k = 0.6 for carefully cleaned surfaces, 1.2 for technical ones.  Both
    surfaces must flow to feed the jet, so the larger of the two members' values is used.
    Checks: Al 6061-T6 on steel 374 m/s (aluminium governs), Al 1100 on steel 239 m/s
    (steel governs), Cu on Ti 346 m/s.  Validity: small-angle form (2 % high at 20 degrees);
    calibrated on similar-metal explosive welds; the dissimilar-pair rule is a convention.

    Reference: Deribas & Zakharenko (1974); Lysak & Kuzmin (2012); Ribeiro et al. (2014).
    """
    f = _metal(flyer, "flyer")
    t = _metal(target, "target")
    k = _positive("coefficient", coefficient)
    return k * max(_flow_speed(f), _flow_speed(t))


def collision_reynolds_number(flyer: MaterialLike, target: MaterialLike, collision_velocity: float) -> float:
    """Cowan's collision Reynolds number ``Re = (rho_f + rho_t) Vc^2 / (2 (HV_f + HV_t))`` [-].

    The ratio of the dynamic pressure of the colliding streams to their hardness (HV in Pa,
    rho in kg/m^3, ``collision_velocity`` Vc in m/s, >= 0).  Above about 10.6
    (:data:`COWAN_REYNOLDS_TRANSITION`) the bond line is wavy, below it flat.

    Reference: Cowan, Bergmann & Holtzman (1971).
    """
    f = _metal(flyer, "flyer")
    t = _metal(target, "target")
    vc = _non_negative("collision_velocity", collision_velocity)
    return (f.density + t.density) * vc * vc / (2.0 * (f.hardness_estimate() + t.hardness_estimate()))


def wavy_transition_velocity(flyer: MaterialLike, target: MaterialLike,
                             reynolds: float = COWAN_REYNOLDS_TRANSITION) -> float:
    """Collision-point velocity at which the bond line turns wavy [m/s].

    ``Vc,T = sqrt(2 Re_T (HV_f + HV_t) / (rho_f + rho_t))`` with ``reynolds`` Re_T = 10.6
    (inverse of :func:`collision_reynolds_number`).  Flat interfaces below it are still
    bonded; the wavy regime is the one usually aimed for.  Checks: Al 6061/steel 2148 m/s,
    Cu/Ti 1747 m/s.  Validity: Re_T is a mean over the pairs Cowan et al. studied
    (scatter roughly +-30 %).

    Reference: Cowan, Bergmann & Holtzman (1971).
    """
    f = _metal(flyer, "flyer")
    t = _metal(target, "target")
    re = _positive("reynolds", reynolds)
    return math.sqrt(2.0 * re * (f.hardness_estimate() + t.hardness_estimate()) / (f.density + t.density))


def upper_limit_velocity(flyer: MaterialLike, target: MaterialLike, flyer_thickness: float,
                         collision_velocity: float, n: float = WITTMAN_N) -> float:
    """Largest impact velocity before the interface stays molten (Wittman's upper limit) [m/s].

    ``Vp,max = (1/N) sqrt(Tm c0) (kappa cp c0 / (rho h))^(1/4) / Vc``

    with the melting temperature Tm [K], bulk sound speed c0 [m/s], thermal conductivity
    kappa [W/(m K)], specific heat cp [J/(kg K)] and density rho [kg/m^3] of the
    lower-melting member, the flyer thickness h [m], the collision-point velocity Vc [m/s]
    and N = 0.11 (:data:`WITTMAN_N`).  Equivalent to ``Vp^2 = (cp Tm/N^2) (c0/Vc)^2
    sqrt(alpha/(c0 h))``: the collision energy must not exceed what conduction
    (diffusivity alpha) removes from the interface before the release wave returns
    (time ~ h/c0), or the tensile wave tears a still-liquid bond.  Check: 1 mm Al 6061 on
    steel at Vc = 2500 m/s -> 970 m/s.  Validity: empirical, calibrated on similar-metal
    explosive welds; using the lower-melting member and Tm in kelvin are conventions, and
    N is uncertain by a factor ~2 (the limit scales as 1/N).

    Reference: Wittman (1973); de Rosset (2006); Ribeiro et al. (2014).
    """
    f = _metal(flyer, "flyer")
    t = _metal(target, "target")
    h = _positive("flyer_thickness", flyer_thickness)
    vc = _positive("collision_velocity", collision_velocity)
    nn = _positive("n", n)
    return _wittman_group(_melting_member(f, t), h, nn) / vc


@dataclass(frozen=True, kw_only=True)
class WeldWindow(ResultMixin):
    """Weldability window of a flyer/target pair at one collision angle, as impact velocities.

    ``vp_min`` lower (bonding) limit; ``vp_melt`` Wittman's upper limit made self-consistent
    at this angle, ``sqrt(W sin(beta))``; ``vp_sonic`` = ``vc_sonic sin(beta)`` the impact
    velocity at which the collision point turns supersonic; ``vp_max`` the smaller of the
    two (``upper_limited_by`` says which: "melting" or "sonic"); ``vc_wavy``/``vp_wavy``
    the flat-to-wavy transition; ``exists`` is ``vp_min < vp_max``.
    ``lower_governed_by`` is the member with the larger sqrt(HV/rho), ``melting_member``
    the lower-melting one used in the upper limit.

    Reference: Wittman (1973); Deribas & Zakharenko (1974); Cowan et al. (1971);
    Ribeiro et al. (2014).
    """

    _title: ClassVar[str] = "Weldability window"

    flyer: str
    target: str
    flyer_thickness: float = unit("m")
    collision_angle: float = unit("rad")
    lower_coefficient: float
    wittman_n: float
    vp_min: float = unit("m/s")
    vp_melt: float = unit("m/s")
    vp_sonic: float = unit("m/s")
    vp_max: float = unit("m/s")
    vp_wavy: float = unit("m/s")
    vc_sonic: float = unit("m/s")
    vc_wavy: float = unit("m/s")
    exists: bool
    upper_limited_by: str
    lower_governed_by: str
    melting_member: str

    @property
    def width(self) -> float:
        """``vp_max / vp_min`` (> 1 when the window exists)."""
        return self.vp_max / self.vp_min

    def contains(self, impact_velocity: float) -> bool:
        """True if ``vp_min <= impact_velocity <= vp_max`` (m/s)."""
        return self.vp_min <= impact_velocity <= self.vp_max


def weld_window(flyer: MaterialLike, target: MaterialLike, flyer_thickness: float,
                collision_angle: float = DEFAULT_COLLISION_ANGLE,
                lower_coefficient: float = DERIBAS_K_CLEAN, wittman_n: float = WITTMAN_N) -> WeldWindow:
    """Weldability window of a pair at a fixed collision angle (impact-velocity limits).

    * lower: ``vp_min = k max sqrt(HV/rho)`` (:func:`lower_limit_velocity`);
    * upper, melting: ``Vp = W / Vc`` with ``Vc = Vp / sin(beta)`` gives
      ``vp_melt = sqrt(W sin(beta))`` (:func:`upper_limit_velocity`);
    * upper, jetting: ``vp_sonic = min(c0) sin(beta)`` (:func:`jetting_sonic_limit`);
    * morphology: wavy for ``Vp >= vc_wavy sin(beta)`` (:func:`wavy_transition_velocity`).

    ``flyer_thickness`` in m (> 0), ``collision_angle`` in rad (0 < beta < pi/2; magnetic
    pulse welds are reported at about 5-30 degrees and 200-600 m/s).  At small angles the sonic limit closes the
    window (Al 6061 on steel: no window below 4.7 degrees); at 10 degrees and 1 mm it is
    374-649 m/s (melting-limited), for Cu on Ti 346-560 m/s.  The window narrows slowly
    with flyer thickness (vp_melt ~ h^(-1/8)).  Validity: empirical limits (see each
    function); the collision angle is an input here, whereas in a real coil it follows
    from how the flyer deforms across the gap.

    Reference: Wittman (1973); Deribas & Zakharenko (1974); Cowan et al. (1971);
    Ribeiro, Mendes & Loureiro (2014).
    """
    f = _metal(flyer, "flyer")
    t = _metal(target, "target")
    h = _positive("flyer_thickness", flyer_thickness)
    b = _angle(collision_angle)
    k = _positive("lower_coefficient", lower_coefficient)
    n = _positive("wittman_n", wittman_n)
    sin_b = math.sin(b)
    s_f, s_t = _flow_speed(f), _flow_speed(t)
    vp_min = k * max(s_f, s_t)
    member = _melting_member(f, t)
    vp_melt = math.sqrt(_wittman_group(member, h, n) * sin_b)
    vc_sonic = min(f.bulk_sound_speed(), t.bulk_sound_speed())
    vp_sonic = vc_sonic * sin_b
    vc_wavy = math.sqrt(2.0 * COWAN_REYNOLDS_TRANSITION * (f.hardness_estimate() + t.hardness_estimate())
                        / (f.density + t.density))
    vp_max = min(vp_melt, vp_sonic)
    return WeldWindow(
        flyer=f.key, target=t.key, flyer_thickness=h, collision_angle=b, lower_coefficient=k, wittman_n=n,
        vp_min=vp_min, vp_melt=vp_melt, vp_sonic=vp_sonic, vp_max=vp_max, vp_wavy=vc_wavy * sin_b,
        vc_sonic=vc_sonic, vc_wavy=vc_wavy, exists=vp_min < vp_max,
        upper_limited_by="melting" if vp_melt <= vp_sonic else "sonic",
        lower_governed_by=f.key if s_f >= s_t else t.key, melting_member=member.key,
    )


# ---------------------------------------------------------------------------
# flyer flight under a pressure pulse
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class FlyerImpact(ResultMixin):
    """Flight of a flyer sheet across the standoff gap under a drive-pressure pulse.

    ``impact_velocity`` and ``transit_time`` at the moment the gap closes;
    ``pressure_at_impact`` the drive pressure still acting then; ``limited_by`` is
    ``"gap"`` (the gap closes while the pulse is on), ``"pulse"`` (the pulse ends first and
    the flyer coasts) or ``"no_drive"`` (zero pressure); ``constant_pressure_velocity`` is
    the upper bound ``sqrt(2 eta P g/(rho h))`` for the same peak pressure.

    Reference: Psyk et al. (2011); Zhang et al. (2011).
    """

    _title: ClassVar[str] = "Flyer impact"

    flyer: str
    thickness: float = unit("m")
    gap: float = unit("m")
    peak_pressure: float = unit("Pa")
    efficiency: float
    pulse_shape: str
    pulse_duration: float | None = unit("s", default=None)
    impact_velocity: float = unit("m/s")
    transit_time: float = unit("s")
    pressure_at_impact: float = unit("Pa")
    limited_by: str
    constant_pressure_velocity: float = unit("m/s")


def _theta_minus_sin(theta: float) -> float:
    """theta - sin(theta), by series for small theta (avoids cancellation)."""
    if theta < _SMALL_THETA:
        t2 = theta * theta
        return theta * t2 * (1.0 / 6.0 - t2 * (1.0 / 120.0 - t2 * (1.0 / 5040.0 - t2 / 362880.0)))
    return theta - math.sin(theta)


def _half_square_minus_versine(theta: float) -> float:
    """theta^2/2 - (1 - cos(theta)), by series for small theta (avoids cancellation)."""
    if theta < _SMALL_THETA:
        t2 = theta * theta
        return t2 * t2 * (1.0 / 24.0 - t2 * (1.0 / 720.0 - t2 * (1.0 / 40320.0 - t2 / 3628800.0)))
    return 0.5 * theta * theta - (1.0 - math.cos(theta))


def flyer_impact(pressure: float, gap: float, flyer: MaterialLike, thickness: float, efficiency: float = 1.0,
                 pulse_duration: float | None = None, pulse_shape: str = "constant") -> FlyerImpact:
    """Impact velocity of a flyer driven across a gap by a pressure pulse of given shape.

    With the peak acceleration ``a = eta P / (rho h)``:

    * ``"constant"``, no ``pulse_duration``: ``v = sqrt(2 a g)``, ``t = sqrt(2 g/a)``
      (:func:`magform.shock.flyer_velocity`);
    * ``"constant"`` with ``pulse_duration`` T (rectangular pulse): the same if
      ``a T^2/2 >= g``; otherwise the pulse ends first, ``v = a T`` (impulse-limited) and the
      flyer coasts the rest of the gap;
    * ``"half_sine"`` (field ``B = B_pk sin(pi t/T)``, so ``P = P_pk sin^2(pi t/T)`` for
      0 <= t <= T, where T = ``pulse_duration`` is required)::

          v(t) = a [t/2 - (T/4pi) sin(2 pi t/T)]
          x(t) = a [t^2/4 - (T^2/8pi^2)(1 - cos(2 pi t/T))]

      solved for ``x(t) = g`` by bisection; if ``x(T) = a T^2/4 < g`` the flyer leaves the
      pulse at ``v = a T/2`` and coasts.  For T much longer than the flight the pressure is
      still rising at impact, ``x ~ a pi^2 t^4/(12 T^2)``: a 4 GPa, 5 ms pulse moves a 1 mm
      aluminium flyer across 2 mm in 80 us and lands it at only ~100 m/s.

    ``pressure`` peak drive pressure, Pa (>= 0); ``gap`` m (> 0); ``thickness`` m (> 0);
    ``efficiency`` eta in (0, 1]; ``pulse_duration`` s (> 0).  Validity: rigid sheet, no
    strength (the flyer must first yield to bend), no air cushion, pressure independent of
    the opening gap (fold the loss into ``efficiency``).

    Reference: Psyk et al. (2011); Zhang et al. (2011).
    """
    p = _non_negative("pressure", pressure)
    g = _positive("gap", gap)
    h = _positive("thickness", thickness)
    eta = _efficiency(efficiency)
    m = as_material(flyer)
    shape = _pulse_shape(pulse_shape)
    tp = None if pulse_duration is None else _positive("pulse_duration", pulse_duration)
    if shape == "half_sine" and tp is None:
        raise InputError("pulse_shape 'half_sine' needs a pulse_duration (the half-period of the field)")

    def result(v: float, t: float, p_imp: float, limited_by: str, v_const: float) -> FlyerImpact:
        return FlyerImpact(flyer=m.key, thickness=h, gap=g, peak_pressure=p, efficiency=eta, pulse_shape=shape,
                           pulse_duration=tp, impact_velocity=v, transit_time=t, pressure_at_impact=p_imp,
                           limited_by=limited_by, constant_pressure_velocity=v_const)

    if p == 0.0:
        return result(0.0, math.inf, 0.0, "no_drive", 0.0)
    v_const = flyer_velocity(p, g, m, h, eta)
    a = eta * p / (m.density * h)
    if shape == "constant":
        t_const = flyer_transit_time(g, p, m, h, eta)
        if tp is None or tp >= t_const:
            return result(v_const, t_const, p, "gap", v_const)
        v = a * tp
        return result(v, tp + (g - 0.5 * a * tp * tp) / v, 0.0, "pulse", v_const)

    period = float(tp)  # type: ignore[arg-type]  # half_sine always has a duration (checked above)
    two_pi = 2.0 * math.pi
    x_scale = a * period * period / (8.0 * math.pi * math.pi)   # x = x_scale [theta^2/2 - (1 - cos theta)]
    v_scale = a * period / (4.0 * math.pi)                      # v = v_scale [theta - sin theta]
    x_end = 0.25 * a * period * period                          # theta = 2 pi t / T = 2 pi
    if x_end >= g:
        target = g / x_scale
        theta = bisect(lambda th: _half_square_minus_versine(th) - target, 0.0, two_pi, tol=1e-14, max_iter=400)
        t_hit = theta * period / two_pi
        return result(v_scale * _theta_minus_sin(theta), t_hit, p * math.sin(0.5 * theta) ** 2, "gap", v_const)
    v = 0.5 * a * period
    return result(v, period + (g - x_end) / v, 0.0, "pulse", v_const)


# ---------------------------------------------------------------------------
# drive-pressure and standoff windows (constant pressure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class DriveWindow(ResultMixin):
    """Constant drive pressures (and fields) that land a flyer inside the weldability window.

    ``pressure_min``/``pressure_max`` give ``velocity_min = vp_min`` and
    ``velocity_max = vp_max`` across ``gap``; ``field_*`` are ``sqrt(2 mu0 P)``;
    ``flight_time_max``/``flight_time_min`` are the gap-crossing times ``2 g / v`` at the two
    ends, i.e. the time within which the pulse must deliver its pressure; ``exists`` mirrors
    ``window.exists``.

    Reference: Psyk et al. (2011); Kapil & Sharma (2015).
    """

    _title: ClassVar[str] = "Drive-pressure window"

    flyer: str
    target: str
    flyer_thickness: float = unit("m")
    gap: float = unit("m")
    collision_angle: float = unit("rad")
    efficiency: float
    velocity_min: float = unit("m/s")
    velocity_max: float = unit("m/s")
    pressure_min: float = unit("Pa")
    pressure_max: float = unit("Pa")
    field_min: float = unit("T")
    field_max: float = unit("T")
    flight_time_min: float = unit("s")
    flight_time_max: float = unit("s")
    exists: bool
    window: WeldWindow


def drive_pressure_window(flyer: MaterialLike, target: MaterialLike, flyer_thickness: float, gap: float,
                          collision_angle: float = DEFAULT_COLLISION_ANGLE, efficiency: float = 1.0,
                          lower_coefficient: float = DERIBAS_K_CLEAN, wittman_n: float = WITTMAN_N) -> DriveWindow:
    """Drive-pressure (and field) window that welds a joint: what a coil must deliver.

    ``P = rho h v^2 / (2 eta g)`` (:func:`magform.shock.required_pressure_for_velocity`)
    evaluated at the two ends of :func:`weld_window`, and ``B = sqrt(2 mu0 P)``.
    ``flyer_thickness`` h and ``gap`` g in m (> 0), ``collision_angle`` rad, ``efficiency``
    eta in (0, 1].  Checks (1 mm flyer, 2 mm gap, 10 degrees): Al 6061 on steel
    0.095-0.284 GPa (15.4-26.7 T); Al 1100 on steel 0.039-0.317 GPa (9.9-28.2 T); Cu on Ti
    0.27-0.70 GPa (26-42 T).  Every one lies inside industrial electromagnetic forming
    (20-50 T) and 2-100x below the concept note's 1.6-4 GPa.  Validity: constant pressure
    during the flight (an upper bound on velocity, so real coils need somewhat higher peak
    fields); the flight times (6-11 us for the aluminium joint) are the rise times the
    pulsed supply must beat.

    Reference: Psyk et al. (2011); Kapil & Sharma (2015); Aizawa et al. (2007).
    """
    g = _positive("gap", gap)
    eta = _efficiency(efficiency)
    f = _metal(flyer, "flyer")
    w = weld_window(f, target, flyer_thickness, collision_angle, lower_coefficient, wittman_n)
    p_min = required_pressure_for_velocity(w.vp_min, g, f, w.flyer_thickness, eta)
    p_max = required_pressure_for_velocity(w.vp_max, g, f, w.flyer_thickness, eta)
    return DriveWindow(
        flyer=w.flyer, target=w.target, flyer_thickness=w.flyer_thickness, gap=g,
        collision_angle=w.collision_angle, efficiency=eta, velocity_min=w.vp_min, velocity_max=w.vp_max,
        pressure_min=p_min, pressure_max=p_max, field_min=field_for_pressure(p_min),
        field_max=field_for_pressure(p_max), flight_time_min=2.0 * g / w.vp_max,
        flight_time_max=2.0 * g / w.vp_min, exists=w.exists, window=w,
    )


@dataclass(frozen=True, kw_only=True)
class StandoffWindow(ResultMixin):
    """Standoff gaps that land a flyer inside the weldability window at a fixed drive pressure.

    ``gap_min``/``gap_max`` give ``velocity_min``/``velocity_max``; ``flight_time_*`` are the
    corresponding flight times ``rho h v / (eta P)``; ``drive_field = sqrt(2 mu0 P)``.

    Reference: Psyk et al. (2011).
    """

    _title: ClassVar[str] = "Standoff window"

    flyer: str
    target: str
    flyer_thickness: float = unit("m")
    drive_pressure: float = unit("Pa")
    drive_field: float = unit("T")
    collision_angle: float = unit("rad")
    efficiency: float
    velocity_min: float = unit("m/s")
    velocity_max: float = unit("m/s")
    gap_min: float = unit("m")
    gap_max: float = unit("m")
    flight_time_min: float = unit("s")
    flight_time_max: float = unit("s")
    exists: bool


def standoff_window(flyer: MaterialLike, target: MaterialLike, flyer_thickness: float, drive_pressure: float,
                    collision_angle: float = DEFAULT_COLLISION_ANGLE, efficiency: float = 1.0,
                    lower_coefficient: float = DERIBAS_K_CLEAN, wittman_n: float = WITTMAN_N) -> StandoffWindow:
    """Standoff-gap window for a given constant drive pressure (inverse of :func:`drive_pressure_window`).

    ``g = rho h v^2 / (2 eta P)`` (:func:`magform.shock.required_standoff`) at the two ends
    of :func:`weld_window`, with flight times ``t = rho h v / (eta P)``.  ``drive_pressure``
    in Pa (> 0).  At the concept note's pressures the gap must shrink until it is no longer
    a practical standoff: 1 mm Al 6061 on steel needs 0.12-0.36 mm at 1.6 GPa and
    47-142 um at 4 GPa, crossed in 0.25-1.1 us - shorter than any 60-100 T magnet's rise
    and comparable to sheet flatness.  Validity: constant pressure during the flight.

    Reference: Psyk et al. (2011); Zhang et al. (2011).
    """
    p = _positive("drive_pressure", drive_pressure)
    eta = _efficiency(efficiency)
    f = _metal(flyer, "flyer")
    w = weld_window(f, target, flyer_thickness, collision_angle, lower_coefficient, wittman_n)
    areal = f.density * w.flyer_thickness
    return StandoffWindow(
        flyer=w.flyer, target=w.target, flyer_thickness=w.flyer_thickness, drive_pressure=p,
        drive_field=field_for_pressure(p), collision_angle=w.collision_angle, efficiency=eta,
        velocity_min=w.vp_min, velocity_max=w.vp_max,
        gap_min=required_standoff(w.vp_min, p, f, w.flyer_thickness, eta),
        gap_max=required_standoff(w.vp_max, p, f, w.flyer_thickness, eta),
        flight_time_min=areal * w.vp_min / (eta * p), flight_time_max=areal * w.vp_max / (eta * p),
        exists=w.exists,
    )


# ---------------------------------------------------------------------------
# interface heating and interdiffusion (estimates)
# ---------------------------------------------------------------------------
def interface_energy(flyer: MaterialLike, flyer_thickness: float, impact_velocity: float,
                     target: MaterialLike | None = None, target_thickness: float | None = None) -> float:
    """Kinetic energy per unit area dissipated when the flyer sticks to the target [J/m^2].

    Perfectly inelastic collision of areal masses ``m = rho h``:
    ``E = m_f m_t / (2 (m_f + m_t)) Vp^2``; with no ``target``/``target_thickness`` the
    target is massive or rigidly backed and ``E = rho_f h_f Vp^2 / 2``.  Thicknesses in m,
    ``impact_velocity`` in m/s (>= 0).  Check: 1 mm Al 6061 at 385 m/s -> 2.0e5 J/m^2.
    Validity: energy bookkeeping only; where it goes (jet, bulk plastic work, stress waves,
    interface heat) is not resolved, see :func:`equivalent_melt_layer`.

    Reference: Crossland (1982); Meyers (1994) ch. 17.
    """
    f = as_material(flyer)
    h = _positive("flyer_thickness", flyer_thickness)
    vp = _non_negative("impact_velocity", impact_velocity)
    m_f = f.density * h
    if target is None and target_thickness is None:
        return 0.5 * m_f * vp * vp
    if target is None or target_thickness is None:
        raise InputError("give both target and target_thickness for a free target, or neither for a backed one")
    m_t = as_material(target).density * _positive("target_thickness", target_thickness)
    return 0.5 * m_f * m_t / (m_f + m_t) * vp * vp


def equivalent_melt_layer(material: MaterialLike, energy_per_area: float,
                          fraction: float = INTERFACE_ENERGY_FRACTION,
                          initial_temperature: float = T_ROOM) -> float:
    """Thickness of ``material`` that a fraction of the collision energy would melt [m] (estimate).

    ``delta = f E / (rho (cp (Tm - T0) + L_f))`` with ``energy_per_area`` E in J/m^2 (>= 0),
    ``fraction`` f in [0, 1] (default 0.1, :data:`INTERFACE_ENERGY_FRACTION`), T0 =
    ``initial_temperature`` in K and L_f the latent heat of fusion (omitted when the record
    has none).  Check: Al 6061, 2.0e5 J/m^2 -> 8.3 um.  Confidence: estimated - the
    fraction is not known a priori; the result is the order of the melt pockets and
    intermetallic seams seen in impact welds (micrometres), not a prediction of them.  It
    does show that the heat-affected layer is small but not zero.

    Reference: Crossland (1982); Kapil & Sharma (2015).
    """
    m = as_material(material)
    e = _non_negative("energy_per_area", energy_per_area)
    f = _fraction("fraction", fraction)
    t0 = _initial_temperature(initial_temperature, m)
    specific = m.specific_heat * (m.melting_point - t0) + (m.latent_heat_fusion or 0.0)
    return f * e / (m.density * specific)


def thermal_quench_time(material: MaterialLike, layer_thickness: float) -> float:
    """Time for a heated layer to cool by conduction into the cold metal, ``t = delta^2 / alpha`` [s].

    ``alpha = kappa / (rho cp)`` is the thermal diffusivity (:meth:`Material.thermal_diffusivity`),
    ``layer_thickness`` delta in m (>= 0).  Check: 8.3 um of Al 6061 -> 1.0 us, i.e. cooling
    rates of order 1e8-1e9 K/s.  Validity: order of magnitude (a layer between two
    half-spaces cools in ~delta^2/(4 alpha) to delta^2/alpha); constant properties; latent
    heat ignored.

    Reference: Carslaw & Jaeger (1959).
    """
    m = as_material(material)
    d = _non_negative("layer_thickness", layer_thickness)
    return d * d / m.thermal_diffusivity()


def solid_diffusivity(material: MaterialLike, temperature: float, prefactor: float = DIFFUSION_PREFACTOR) -> float:
    """Lattice diffusivity ``D = D0 exp(-Q / (R T))`` [m^2/s] (estimate).

    Q = ``Material.self_diffusion_activation`` [J/mol] (:class:`InputError` if not recorded),
    ``temperature`` T in K (> 0), ``prefactor`` D0 = 1e-4 m^2/s (:data:`DIFFUSION_PREFACTOR`).
    Check: aluminium at its melting point -> ~2e-13 m^2/s.  Confidence: estimated - a
    generic D0 and the self-diffusion activation energy stand in for the interdiffusion
    coefficients of a specific couple (good to an order of magnitude; Brown & Ashby 1980).

    Reference: Brown & Ashby (1980).
    """
    m = as_material(material)
    t = _positive("temperature", temperature)
    d0 = _positive("prefactor", prefactor)
    m.require("self_diffusion_activation")
    return d0 * math.exp(-float(m.self_diffusion_activation) / (R_GAS * t))  # type: ignore[arg-type]


def diffusion_length(diffusivity: float, time: float) -> float:
    """Diffusion length ``x = sqrt(2 D t)`` [m].

    ``diffusivity`` D in m^2/s (>= 0), ``time`` t in s (>= 0).  The scale of an interdiffusion
    zone, and so of a diffusion-grown intermetallic layer.  Check: liquid metal
    (1e-9 m^2/s) for 1 s -> 45 um; for 1 us -> 45 nm.

    Reference: Carslaw & Jaeger (1959).
    """
    d = _non_negative("diffusivity", diffusivity)
    t = _non_negative("time", time)
    return math.sqrt(2.0 * d * t)


def interdiffusion_length(material: MaterialLike, temperature: float, time: float,
                          prefactor: float = DIFFUSION_PREFACTOR) -> float:
    """Solid-state interdiffusion length ``sqrt(2 D(T) t)`` [m] (estimate).

    ``D(T)`` from :func:`solid_diffusivity`; ``temperature`` in K, ``time`` in s.  Evaluated
    at the melting point it is the most the solid can do: aluminium held at 855 K for the
    1 us an impact-weld interface stays hot diffuses 0.65 nm, against ~45 um for a fusion
    weld that is liquid for a second.  Intermetallics in impact welds therefore come from
    local melting and mechanical mixing, not from solid-state diffusion.

    Reference: Brown & Ashby (1980); Kapil & Sharma (2015).
    """
    return diffusion_length(solid_diffusivity(material, temperature, prefactor), time)


# ---------------------------------------------------------------------------
# assessment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class WeldAssessment(Assessment):
    """Does this flyer, gap and drive pressure produce an impact weld?

    Kinematics: ``impact_velocity`` (from :func:`flyer_impact`; ``velocity_limited_by``
    "gap" | "pulse" | "no_drive"), ``collision_velocity = Vp / sin(beta)``, ``transit_time``.
    Window (impact velocities at this angle): ``vp_min`` lower limit, ``vp_melt`` Wittman's
    upper limit, ``vp_sonic`` jetting limit, ``vp_max = min(vp_melt, vp_sonic)``;
    ``vc_sonic`` and ``vc_wavy`` as collision-point velocities.  ``in_window`` is
    ``vp_min <= Vp <= vp_max`` and ``margin = min(Vp/vp_min, vp_max/Vp)`` (>= 1 exactly
    when in the window).  ``interface_morphology`` is "wavy", "flat" or "none" (no bond);
    ``reynolds_number`` is Cowan's.  ``impact_pressure`` is the planar impedance-match
    pressure at the normal velocity ``Vp cos(beta)`` and ``pressure_amplification`` its
    ratio to the drive pressure.  Flyer: ``flyer_skin_temperature_rise`` at the drive field
    (thick-conductor estimate), ``flyer_skin_melts``, ``pressure_transfer`` =
    ``1 - exp(-2 h/delta)`` and ``driver_needed``.  Interface estimates:
    ``interface_energy``, ``melt_layer_estimate`` and ``quench_time`` (in
    ``melting_member``), ``interdiffusion_solid``/``interdiffusion_liquid`` over the quench
    time and ``fusion_weld_interdiffusion`` (liquid for 1 s) for comparison.

    Reference: Crossland (1982); Wittman (1973); Deribas & Zakharenko (1974);
    Kapil & Sharma (2015).
    """

    _title: ClassVar[str] = "Impact-weld assessment"

    flyer: str
    target: str
    pair: str
    pair_demonstrated: bool
    intermetallics: tuple[str, ...]
    flyer_thickness: float = unit("m")
    gap: float = unit("m")
    drive_pressure: float = unit("Pa")
    drive_field: float = unit("T")
    collision_angle: float = unit("rad")
    impact_velocity: float = unit("m/s")
    collision_velocity: float = unit("m/s")
    velocity_limited_by: str
    transit_time: float = unit("s")
    vp_min: float = unit("m/s")
    vp_melt: float = unit("m/s")
    vp_sonic: float = unit("m/s")
    vp_max: float = unit("m/s")
    vc_sonic: float = unit("m/s")
    vc_wavy: float = unit("m/s")
    in_window: bool
    interface_morphology: str
    reynolds_number: float
    impact_pressure: float = unit("Pa")
    pressure_amplification: float | None
    flyer_skin_temperature_rise: float = unit("K")
    flyer_skin_melting_field: float = unit("T")
    flyer_skin_melts: bool
    pressure_transfer: float | None
    driver_needed: bool
    melting_member: str
    interface_energy: float = unit("J/m^2")
    melt_layer_estimate: float = unit("m")
    quench_time: float = unit("s")
    interdiffusion_solid: float | None = unit("m")
    interdiffusion_liquid: float = unit("m")
    fusion_weld_interdiffusion: float = unit("m")


def assess_weld(flyer: MaterialLike, target: MaterialLike, flyer_thickness: float, gap: float,
                drive_pressure: float, collision_angle: float = DEFAULT_COLLISION_ANGLE,
                efficiency: float = 1.0, pulse_duration: float | None = None, pulse_shape: str = "constant",
                target_thickness: float | None = None, lower_coefficient: float = DERIBAS_K_CLEAN,
                wittman_n: float = WITTMAN_N, initial_temperature: float = T_ROOM) -> WeldAssessment:
    """Assess a magnetically driven impact weld of ``flyer`` onto ``target``.

    Chain: drive pressure -> flight across the gap (:func:`flyer_impact`; constant pressure
    unless a ``pulse_duration``/``pulse_shape`` is given) -> ``Vc = Vp / sin(beta)`` ->
    comparison with :func:`weld_window` -> interface pressure
    (:func:`magform.shock.impedance_match` at ``Vp cos(beta)``) and thermal estimates.

    Status (every failing check is listed in ``warnings``):

    1. ``Vp < vp_min`` (surfaces do not flow, no jet), ``Vp > vp_sonic`` (supersonic
       collision point, no jet) or ``Vp > vp_melt`` (interface still molten when the release
       wave arrives) -> INFEASIBLE: no sound weld, whatever the pressure;
    2. otherwise inside the window: STATE_OF_THE_ART when the pair is in
       :data:`DEMONSTRATED_PAIRS`, the flyer can be driven directly and the drive field is
       within industrial electromagnetic forming (<= 50 T,
       ``benchmarks.EMF_CONVENTIONAL_FIELD``);
    3. worst-combined with EXTRAPOLATION for each of: pair not recorded as impact welded;
       flyer shielding too poor (``pressure_transfer`` < 0.8, needs a driver sheet); drive
       field above industrial coils; flyer skin reaching its melting point at the drive
       field (``eddy.skin_melting_field``).  These cautions are listed in ``warnings``
       even when rule 1 already makes the weld infeasible.

    ``margin = min(Vp/vp_min, vp_max/Vp)``.  Inputs: thicknesses and ``gap`` in m (> 0);
    ``drive_pressure`` peak magnetic pressure in Pa (>= 0); ``collision_angle`` rad;
    ``efficiency`` in (0, 1]; ``pulse_duration`` s (duration of the pressure pulse, i.e. the
    half-period of the field) and ``pulse_shape`` "constant" or "half_sine";
    ``target_thickness`` m for a free target (default: backed/massive);
    ``initial_temperature`` K.  The skin-depth frequency is ``1/(2 T)`` for a given pulse,
    else ``1/(4 t_flight)`` (a drive that peaks as the gap closes).

    Checks: 1 mm Al 6061 on steel 1018, 2 mm gap, 10 degrees: 0.1 GPa (15.9 T) -> 385 m/s,
    4.1 GPa on the interface (41x the drive), state of the art; 1.6 GPa -> 1540 m/s and
    4 GPa -> 2434 m/s, both infeasible (supersonic collision point, beyond the melting
    limit, flyer skin molten).  Validity: 1-D kinematics with a prescribed collision
    angle; empirical window limits; thermal quantities are order-of-magnitude estimates;
    weld strength, fatigue and the unbonded zone where the sheets first touch are not
    modelled.

    Reference: Crossland (1982); Wittman (1973); Deribas & Zakharenko (1974); Cowan et al.
    (1971); Kapil & Sharma (2015); Psyk et al. (2011).
    """
    f = _metal(flyer, "flyer")
    t = _metal(target, "target")
    h = _positive("flyer_thickness", flyer_thickness)
    g = _positive("gap", gap)
    p = _non_negative("drive_pressure", drive_pressure)
    b = _angle(collision_angle)
    eta = _efficiency(efficiency)
    t0 = _initial_temperature(initial_temperature, f, t)
    if target_thickness is not None:
        _positive("target_thickness", target_thickness)

    w = weld_window(f, t, h, b, lower_coefficient, wittman_n)
    flight = flyer_impact(p, g, f, h, eta, pulse_duration, pulse_shape)
    vp = flight.impact_velocity
    sin_b = math.sin(b)
    vc = vp / sin_b
    drive_field = field_for_pressure(p)
    record = demonstrated_pair(f, t)
    key = pair_key(f, t)

    # interface state
    state = impedance_match(f, t, vp * math.cos(b))
    p_impact = state.pressure
    amplification = p_impact / p if p > 0.0 else None
    reynolds = (f.density + t.density) * vc * vc / (2.0 * (f.hardness_estimate() + t.hardness_estimate()))

    # flyer electromagnetics
    skin_rise = skin_temperature_rise(f, drive_field)
    b_melt = skin_melting_field(f, t0)
    skin_melts = drive_field >= b_melt
    if flight.pulse_duration is not None:
        frequency: float | None = 0.5 / flight.pulse_duration
    elif math.isfinite(flight.transit_time) and flight.transit_time > 0.0:
        frequency = 0.25 / flight.transit_time
    else:
        frequency = None
    if frequency is None:
        transfer: float | None = None
        delta = math.inf
    else:
        delta = skin_depth_for_material(f, frequency, t0)
        transfer = sheet_shielding_factor(h, delta)
    driver_needed = transfer is not None and transfer < DRIVER_TRANSFER_THRESHOLD

    # interface heating and interdiffusion (estimates, in the lower-melting member)
    member = _melting_member(f, t)
    if target_thickness is None:
        energy = interface_energy(f, h, vp)
    else:
        energy = interface_energy(f, h, vp, t, target_thickness)
    melt_layer = equivalent_melt_layer(member, energy, INTERFACE_ENERGY_FRACTION, t0)
    quench = thermal_quench_time(member, melt_layer)
    if member.self_diffusion_activation is not None:
        x_solid: float | None = interdiffusion_length(member, member.melting_point, quench)
    else:
        x_solid = None
    x_liquid = diffusion_length(LIQUID_DIFFUSIVITY, quench)
    x_fusion = diffusion_length(LIQUID_DIFFUSIVITY, FUSION_WELD_TIME)

    in_window = w.contains(vp)
    if vp > 0.0:
        margin = min(vp / w.vp_min, w.vp_max / vp)
    else:
        margin = 0.0
    if not in_window:
        morphology = "none"
    else:
        morphology = "wavy" if vc >= w.vc_wavy else "flat"

    warnings: list[str] = []
    notes: list[str] = []

    # ---- kinematics
    if flight.limited_by == "no_drive":
        notes.append("no drive pressure: the flyer does not move")
    else:
        text = (f"{f.key} flyer {h * 1e3:.3g} mm over {g * 1e3:.3g} mm at {_gpa(p)} peak drive "
                f"({drive_field:.3g} T) -> {vp:.0f} m/s in {flight.transit_time * 1e6:.3g} us; collision point "
                f"{vc:.0f} m/s at beta = {_deg(b)}")
        if flight.pulse_shape == "half_sine":
            text += (f" (half-sine field pulse of {flight.pulse_duration:.3g} s; drive pressure at impact "
                     f"{_gpa(flight.pressure_at_impact)})")
        notes.append(text)
    if flight.limited_by == "pulse":
        warnings.append(f"the {flight.pulse_duration:.3g} s drive pulse ends before the flyer crosses the gap: "
                        f"the impact velocity {vp:.0f} m/s is impulse-limited (constant pressure would give "
                        f"{flight.constant_pressure_velocity:.0f} m/s)")
    elif flight.pulse_shape == "half_sine" and p > 0.0 and flight.pressure_at_impact < 0.5 * p:
        warnings.append(f"slow pulse: the flyer crosses the gap in {flight.transit_time * 1e6:.3g} us, when the drive "
                        f"pressure has reached only {_gpa(flight.pressure_at_impact)} of its {_gpa(p)} peak - "
                        "rise time, not peak pressure, sets the impact velocity")

    # ---- window
    upper = ("Wittman's interface-melting limit" if w.upper_limited_by == "melting"
             else "the sonic (jetting) limit")
    notes.append(f"weldability window at beta = {_deg(b)}: {w.vp_min:.0f}-{w.vp_max:.0f} m/s (lower limit "
                 f"k = {w.lower_coefficient:g} x sqrt(HV/rho) of {w.lower_governed_by}; upper limit: {upper}, "
                 f"evaluated for {w.melting_member}); collision point must stay below {w.vc_sonic:.0f} m/s")
    if not w.exists:
        warnings.append(f"no weldability window at beta = {_deg(b)}: the lower limit {w.vp_min:.0f} m/s exceeds "
                        f"the upper limit {w.vp_max:.0f} m/s ({w.upper_limited_by}-limited)")
    failures: list[str] = []
    if vp < w.vp_min:
        failures.append(f"below the bonding threshold: {vp:.0f} m/s < {w.vp_min:.0f} m/s, the surfaces do not "
                        "flow and no jet forms")
    if vp > w.vp_sonic:
        failures.append(f"supersonic collision point: Vc = {vc:.0f} m/s exceeds the bulk sound speed "
                        f"{w.vc_sonic:.0f} m/s, so no re-entrant jet strips the oxide films")
    if vp > w.vp_melt:
        failures.append(f"beyond the upper limit: {vp:.0f} m/s > {w.vp_melt:.0f} m/s, the interface is still "
                        "molten when the release wave arrives (continuous melt layer, cracking)")
    warnings.extend(failures)
    lo_ang, hi_ang = MPW_TYPICAL_ANGLE
    if not lo_ang <= b <= hi_ang:
        notes.append(f"collision angle {_deg(b)} lies outside the {_deg(lo_ang)}-{_deg(hi_ang)} typical of "
                     "magnetic pulse welding")

    # ---- what the gap does, and what the drive window is
    if amplification is not None and vp > 0.0:
        notes.append(f"impact pressure {_gpa(p_impact)} at the interface = {amplification:.3g}x the drive "
                     "pressure: the standoff gap stores the impulse as momentum, so the interface pressure is set "
                     "by the impact velocity, not by the field")
    if w.exists and flight.limited_by != "no_drive":
        p_lo = required_pressure_for_velocity(w.vp_min, g, f, h, eta)
        p_hi = required_pressure_for_velocity(w.vp_max, g, f, h, eta)
        text = (f"with this gap a constant drive of {p_lo / 1e9:.3g}-{p_hi / 1e9:.3g} GPa "
                f"({field_for_pressure(p_lo):.3g}-{field_for_pressure(p_hi):.3g} T) lands in the window")
        if p_hi < README_PRESSURE_RANGE[0]:
            text += (f"; the concept note's {README_PRESSURE_RANGE[0] / 1e9:g}-{README_PRESSURE_RANGE[1] / 1e9:g} "
                     f"GPa is {README_PRESSURE_RANGE[0] / p_hi:.3g}-{README_PRESSURE_RANGE[1] / p_hi:.3g}x above "
                     "its upper end")
        notes.append(text)
    notes.append("bonding needs an oblique collision with a travelling, subsonic collision point; a flat-on "
                 "impact (beta -> 0) never welds, so only the band of the interface where angle and velocity are "
                 "in the window is bonded")

    # ---- pair
    if record is not None:
        how = " and ".join(s.replace("_", "-") for s in record.processes)
        imc = ", ".join(record.intermetallics) if record.intermetallics else "none (no compounds in this system)"
        notes.append(f"{key}: impact welded by {how} welding; intermetallics that form where the interface "
                     f"melts: {imc}" + (f" ({record.note})" if record.note else ""))
    else:
        notes.append(f"{key}: no impact weld of this pair is recorded in DEMONSTRATED_PAIRS")

    # ---- flyer electromagnetics
    if transfer is not None:
        notes.append(f"flyer skin depth {delta * 1e3:.3g} mm at {frequency:.3g} Hz: pressure transfer "
                     f"1 - exp(-2h/delta) = {transfer:.2f}; eddy-current skin heating +{skin_rise:.0f} K at "
                     f"{drive_field:.3g} T (thick-conductor estimate, kappa = 1)")

    # ---- interface heat and intermetallics
    if vp > 0.0:
        solid = f"{x_solid * 1e9:.2g} nm" if x_solid is not None else "not evaluated (no activation energy)"
        notes.append(f"collision energy {energy / 1e3:.3g} kJ/m^2; if {INTERFACE_ENERGY_FRACTION:.0%} stays at "
                     f"the interface it melts ~{melt_layer * 1e6:.2g} um of {member.key} (estimate) that quenches "
                     f"in ~{quench:.2g} s; interdiffusion in that time: solid {solid}, liquid "
                     f"{x_liquid * 1e9:.2g} nm, against ~{x_fusion * 1e6:.0f} um in a fusion weld liquid for "
                     f"{FUSION_WELD_TIME:g} s - intermetallics are confined to melted pockets, not eliminated")

    # ---- status (cautions are reported whatever the outcome)
    cautions: list[str] = []
    if record is None:
        cautions.append(f"pair {key} ({f.key} on {t.key}) is not a recorded impact-welded combination")
    if driver_needed:
        cautions.append(f"the {h * 1e3:.3g} mm {f.key} flyer shields poorly at {frequency:.3g} Hz (skin depth "
                        f"{delta * 1e3:.3g} mm, pressure transfer {transfer:.2f} < {DRIVER_TRANSFER_THRESHOLD:g}): "
                        "a copper or aluminium driver sheet (or a faster pulse) is needed, or make the better "
                        "conductor the flyer")
    if drive_field > EMF_CONVENTIONAL_FIELD[1]:
        text = (f"drive field {drive_field:.3g} T is above industrial electromagnetic-forming coils "
                f"({EMF_CONVENTIONAL_FIELD[0]:g}-{EMF_CONVENTIONAL_FIELD[1]:g} T)")
        if flight.pulse_duration is None and flight.transit_time < FASTEST_HIGH_FIELD_RISE:
            text += (f"; the {flight.transit_time * 1e6:.2g} us flight is shorter than the "
                     f"~{FASTEST_HIGH_FIELD_RISE * 1e6:g} us rise of the fastest 100 T-class (single-turn, "
                     "destructive) coils, so the constant-pressure velocity is an upper bound - give "
                     "pulse_duration and pulse_shape for the real waveform")
        cautions.append(text)
    if skin_melts:
        cautions.append(f"flyer skin melts: the drive field {drive_field:.3g} T exceeds the {b_melt:.3g} T at "
                        f"which the eddy-current skin of {f.key} reaches its melting point (+{skin_rise:.0f} K): "
                        "a heat-affected (molten) layer on the coil side")
    warnings.extend(cautions)
    if failures:
        status = Status.INFEASIBLE
    else:
        status = worst(Status.STATE_OF_THE_ART, [Status.EXTRAPOLATION] * len(cautions))

    return WeldAssessment(
        status=status, margin=margin, warnings=tuple(warnings), notes=tuple(notes),
        flyer=f.key, target=t.key, pair=key, pair_demonstrated=record is not None,
        intermetallics=record.intermetallics if record is not None else (),
        flyer_thickness=h, gap=g, drive_pressure=p, drive_field=drive_field, collision_angle=b,
        impact_velocity=vp, collision_velocity=vc, velocity_limited_by=flight.limited_by,
        transit_time=flight.transit_time, vp_min=w.vp_min, vp_melt=w.vp_melt, vp_sonic=w.vp_sonic,
        vp_max=w.vp_max, vc_sonic=w.vc_sonic, vc_wavy=w.vc_wavy, in_window=in_window,
        interface_morphology=morphology, reynolds_number=reynolds, impact_pressure=p_impact,
        pressure_amplification=amplification, flyer_skin_temperature_rise=skin_rise,
        flyer_skin_melting_field=b_melt, flyer_skin_melts=skin_melts, pressure_transfer=transfer,
        driver_needed=driver_needed, melting_member=member.key, interface_energy=energy,
        melt_layer_estimate=melt_layer, quench_time=quench, interdiffusion_solid=x_solid,
        interdiffusion_liquid=x_liquid, fusion_weld_interdiffusion=x_fusion,
    )


__all__ = [
    "REFERENCES",
    "DERIBAS_K_CLEAN", "DERIBAS_K_TECHNICAL", "COWAN_REYNOLDS_TRANSITION", "WITTMAN_N",
    "DEFAULT_COLLISION_ANGLE", "MPW_TYPICAL_ANGLE", "MPW_TYPICAL_VELOCITY", "README_PRESSURE_RANGE",
    "INTERFACE_ENERGY_FRACTION", "DIFFUSION_PREFACTOR", "LIQUID_DIFFUSIVITY", "FUSION_WELD_TIME",
    "DRIVER_TRANSFER_THRESHOLD", "FASTEST_HIGH_FIELD_RISE", "PULSE_SHAPES", "WELDABLE_CATEGORIES",
    "WeldPair", "DEMONSTRATED_PAIRS", "pair_key", "demonstrated_pair",
    "collision_point_velocity", "impact_velocity_for_collision", "jetting_sonic_limit",
    "critical_collision_angle", "lower_limit_velocity", "collision_reynolds_number",
    "wavy_transition_velocity", "upper_limit_velocity", "WeldWindow", "weld_window",
    "FlyerImpact", "flyer_impact", "DriveWindow", "drive_pressure_window", "StandoffWindow",
    "standoff_window", "interface_energy", "equivalent_melt_layer", "thermal_quench_time",
    "solid_diffusivity", "diffusion_length", "interdiffusion_length", "WeldAssessment", "assess_weld",
]
