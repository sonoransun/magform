"""Home page and explorer: the press cross-section drawing and the data the browser kernel reads.

Everything the two interactive pages need from the package is written here, so that
``docs/assets/js/magform.js`` holds formulas only and no thresholds or material data of its own.
"""

from __future__ import annotations

import math

from docs_common import P_HI, P_LO, REF, Site, fx, sig

from magform import benchmarks, eddy, fieldmap, magnetar, materials, pulse
from magform.constants import C_LIGHT, MU0
from magform.magnetics import field_for_pressure

B_HI = field_for_pressure(P_HI)

# ---------------------------------------------------------------------------
# field lines of the reference press around a billet that keeps the field out
# ---------------------------------------------------------------------------
BILLET_RADIUS, BILLET_LENGTH, BILLET_CORNER = 0.04, 0.30, 0.004
R_MAX, Z_MAX = 0.36, 0.52          # half-extent of the drawing [m]
N_Z = 40                           # axial grid of one quadrant; the other three are mirror images
N_LINES = 11


def _radial_grid() -> list[float]:
    """Radii of the grid [m]: 2.5 mm steps across the gap between billet and coil, coarser elsewhere."""
    inner = [0.0, 0.01, 0.02]
    gap = [0.03 + 0.0025 * i for i in range(13)]             # 30 mm ... 60 mm
    outer = [0.07 + (R_MAX - 0.07) * i / 17 for i in range(18)]
    return inner + gap + outer
SCALE = 500.0                      # px per metre


def _segments(xs: list[float], ys: list[float], v: list[list[float]], level: float) -> list[tuple]:
    """Marching squares: line segments of the contour ``v = level`` on a rectangular grid."""
    out = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            corners = ((xs[i], ys[j], v[i][j]), (xs[i + 1], ys[j], v[i + 1][j]),
                       (xs[i + 1], ys[j + 1], v[i + 1][j + 1]), (xs[i], ys[j + 1], v[i][j + 1]))
            pts = []
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                (x0, y0, v0), (x1, y1, v1) = corners[a], corners[b]
                if (v0 - level) * (v1 - level) < 0.0:
                    t = (level - v0) / (v1 - v0)
                    pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            if len(pts) == 2:
                out.append((pts[0], pts[1]))
            elif len(pts) == 4:   # saddle: join the pairs on the side of the cell-centre value
                centre = sum(c[2] for c in corners) / 4.0
                if (centre > level) == (corners[0][2] > level):
                    out += [(pts[0], pts[1]), (pts[2], pts[3])]
                else:
                    out += [(pts[3], pts[0]), (pts[1], pts[2])]
    return out


def _chains(segments: list[tuple]) -> list[list[tuple[float, float]]]:
    """Join segments that share end points into polylines."""
    def key(p: tuple[float, float]) -> tuple[int, int]:
        return (round(p[0] * 1e7), round(p[1] * 1e7))

    ends: dict[tuple[int, int], list[int]] = {}
    for n, (a, b) in enumerate(segments):
        ends.setdefault(key(a), []).append(n)
        ends.setdefault(key(b), []).append(n)
    used = [False] * len(segments)
    chains = []

    def extend(chain: list, forward: bool) -> None:
        while True:
            tip = chain[-1] if forward else chain[0]
            nxt = next((n for n in ends[key(tip)] if not used[n]), None)
            if nxt is None:
                return
            used[nxt] = True
            a, b = segments[nxt]
            other = b if key(a) == key(tip) else a
            chain.append(other) if forward else chain.insert(0, other)

    for n, (a, b) in enumerate(segments):
        if used[n]:
            continue
        used[n] = True
        chain = [a, b]
        extend(chain, True)
        extend(chain, False)
        chains.append(chain)
    return chains


def press_section_svg() -> tuple[str, dict[str, float]]:
    """Inline SVG of the reference press in section, with field lines, and a few facts about it."""
    current_density = REF.current_density_for_field(B_HI)
    profile = fieldmap.cylinder_profile(BILLET_RADIUS, BILLET_LENGTH, BILLET_CORNER, n_side=24, n_cap=8, n_corner=4)
    solution = fieldmap.surface_pressure(REF, current_density, profile)
    rs = _radial_grid()
    zs = [Z_MAX * j / (N_Z - 1) for j in range(N_Z)]
    psi = [[fieldmap.flux_function_with_workpiece(REF, current_density, solution, max(r, 1e-6), z) for z in zs]
           for r in rs]
    top = max(max(row) for row in psi)
    half_w, half_h = Z_MAX * SCALE, R_MAX * SCALE

    def px(z: float, r: float) -> str:
        return f"{z * SCALE:.1f} {-r * SCALE:.1f}"

    paths = []
    for k in range(N_LINES):
        level = top * (k + 0.5) / N_LINES
        # grid axes are (r, z); the drawing has z along x and r up
        for chain in _chains(_segments(rs, zs, psi, level)):
            if len(chain) < 3:
                continue
            paths.append("M" + " L".join(px(z, r) for r, z in chain))
    lines = "".join(f'<path d="{d}"/>' for d in paths)

    a1, a2, half = REF.inner_radius * SCALE, REF.outer_radius * SCALE, 0.5 * REF.length * SCALE
    br, bh, bc = BILLET_RADIUS * SCALE, 0.5 * BILLET_LENGTH * SCALE, BILLET_CORNER * SCALE
    w, h = 2.0 * half_w, 2.0 * half_h
    svg = (
        f'<svg viewBox="{-half_w:.0f} {-half_h:.0f} {w:.0f} {h:.0f}" role="img" '
        'aria-label="Section through the reference press: a thick coil around a solid billet. Field lines '
        'run through the gap between billet and coil and through the winding, and close outside it. '
        'None enters the billet.">'
        f'<rect class="coil" x="{-half:.0f}" y="{-a2:.0f}" width="{2 * half:.0f}" height="{a2 - a1:.0f}"/>'
        f'<rect class="coil" x="{-half:.0f}" y="{a1:.0f}" width="{2 * half:.0f}" height="{a2 - a1:.0f}"/>'
        f'<rect class="work" x="{-bh:.0f}" y="{-br:.0f}" width="{2 * bh:.0f}" height="{2 * br:.0f}" rx="{bc:.0f}"/>'
        f'<g class="fl"><g id="ps-q">{lines}</g>'
        '<use href="#ps-q" transform="scale(-1 1)"/><use href="#ps-q" transform="scale(1 -1)"/>'
        '<use href="#ps-q" transform="scale(-1 -1)"/></g>'
        f'<line class="axis" x1="{-half_w:.0f}" y1="0" x2="{half_w:.0f}" y2="0" stroke-dasharray="14 4 2 4"/>'
        f'<text x="{-half + 8:.0f}" y="{-a2 + 16:.0f}">winding</text>'
        f'<text x="0" y="4" text-anchor="middle">billet</text>'
        f'<text x="{half_w - 6:.0f}" y="-6" text-anchor="end">axis</text>'
        "</svg>\n"
    )
    facts = {"lines": float(len(paths)), "mid_ratio": solution.mid_ratio if hasattr(solution, "mid_ratio") else float("nan")}
    return svg, facts


# ---------------------------------------------------------------------------
# data for the browser kernel
# ---------------------------------------------------------------------------
_WORKPIECES = ("copper_ofhc", "aluminum_6061", "steel_1018", "ti6al4v", "tungsten", "gamma_tial")
_WORKPIECE_NAMES = {"copper_ofhc": "Copper", "aluminum_6061": "Aluminium 6061", "steel_1018": "Mild steel",
                    "ti6al4v": "Titanium Ti-6Al-4V", "tungsten": "Tungsten", "gamma_tial": "Titanium aluminide"}
_PRESSURE_NAMES = {
    "hip": "hot isostatic pressing", "mpw_typical": "magnetic pulse welding", "emf_50t": "a 50 T forming coil",
    "cold_forging_die": "cold-forging dies", "lanl_100t_pressure": "the 100.75 T pulsed record",
    "hpht_cubic": "industrial diamond presses", "npd_multianvil": "multi-anvil diamond synthesis",
    "dupont_shock": "shock synthesis of diamond", "carbon_nitride_dac": "carbon nitride synthesis",
    "dac_static": "diamond anvil cells",
}
_TIER_NAMES = {"solid": "solid metal", "liquid_skin": "molten or boiling skin", "megabar": "megabar compression",
               "plasma": "dense plasma", "landau": "needle atoms", "degenerate": "white-dwarf matter"}
_PROCESS_NAMES = {"forging": "Forging", "refractory_forming": "Refractory metals", "welding": "Welding",
                  "synthesis": "Synthesis", "densification": "Densification", "containment": "Containment"}


def explorer_data() -> dict:
    """Constants, thresholds, materials and magnetar-regime data read by ``assets/js/explorer.js``."""
    b = benchmarks
    tier_fields = (1.0, 300.0, 3000.0, 1e5, 1e6, 1e10)
    tiers = {}
    for f in tier_fields:
        regime = magnetar.classify_regime("iron_armco", f)
        tiers.setdefault(regime.tier, {"name": _TIER_NAMES[regime.tier],
                                       "processes": magnetar.process_outcomes("iron_armco", f)})
    return {
        "constants": {"mu0": MU0, "c": C_LIGHT, "megaton": magnetar.MEGATON_TNT, "tnt_per_kg": magnetar.MEGATON_TNT / 1e9,
                      "depth_ratio": 3.0},
        "window": {"p_lo": P_LO, "p_hi": P_HI},
        "reference": {"bore": 2.0 * REF.inner_radius, "build": REF.winding_thickness, "length": REF.length},
        "thresholds": {
            "sc_record": b.SC_FIELD_RECORD, "sc_conductor": b.SC_CONDUCTOR_RECORD,
            "pulsed_record": b.PULSED_NONDESTRUCTIVE_RECORD, "pulsed_bore": b.PULSED_NONDESTRUCTIVE_BORE,
            "pulsed_routine_field": b.PULSED_ROUTINE_FIELD, "pulsed_routine_bore": b.PULSED_ROUTINE_BORE,
            "energy_sota": b.ENERGY_MAGNET_DRIVER_SOTA, "energy_record": b.ENERGY_ANY_STORE_RECORD,
            "power_sota": b.PEAK_POWER_SOTA, "power_record": b.PEAK_POWER_RECORD,
            "voltage_sota": b.PULSED_VOLTAGE_SOTA, "voltage_extrapolation": b.PULSED_VOLTAGE_EXTRAPOLATION,
            "current_sota": b.PULSED_CURRENT_SOTA, "current_record": b.PULSED_CURRENT_RECORD,
            "max_voltage": pulse.DEFAULT_MAX_VOLTAGE,
            "coil_skin_melting_field": sig(eddy.skin_melting_field("cuag_conductor", pulse.PRECOOL_TEMPERATURE), 4),
        },
        "materials": [{"key": k, "name": _WORKPIECE_NAMES[k], "resistivity": materials.get(k).resistivity,
                       "density": materials.get(k).density, "specific_heat": materials.get(k).specific_heat,
                       "melting_point": materials.get(k).melting_point,
                       "skin_melting_field": sig(eddy.skin_melting_field(k), 4)} for k in _WORKPIECES],
        "pressures": [{"key": x.key, "name": _PRESSURE_NAMES.get(x.key, x.description), "value": x.value}
                      for x in sorted(b.PRESSURE_BENCHMARKS, key=lambda x: x.value)],
        "magnetar": {
            "atomic_field": magnetar.ATOMIC_FIELD, "qed_field": magnetar.QED_CRITICAL_FIELD,
            "thresholds": [{"key": t.key, "field": t.field, "label": t.label, "state": t.state}
                           for t in sorted(magnetar.regime_thresholds("iron_armco"), key=lambda t: t.field)],
            "tier_bounds": _tier_bounds(),
            "tiers": tiers,
            "process_names": _PROCESS_NAMES,
            "standoff": [{"key": t.key, "field": t.field, "label": t.description} for t in magnetar.FIELD_THRESHOLDS],
            "destructive_record": magnetar.DESTRUCTIVE_FIELD_RECORD,
        },
    }


def _tier_bounds() -> list[dict]:
    """Field at which each regime tier begins for iron, found by bisection on ``classify_regime``."""
    order = []
    lo = 1.0
    current = magnetar.classify_regime("iron_armco", lo).tier
    order.append({"tier": current, "from": 0.0})
    hi_limit = 1e12
    while True:
        # find the next field at which the tier changes
        a, b_ = lo, lo
        while b_ < hi_limit and magnetar.classify_regime("iron_armco", b_).tier == current:
            a, b_ = b_, b_ * 1.5
        if b_ >= hi_limit:
            break
        for _ in range(60):
            mid = math.sqrt(a * b_)
            if magnetar.classify_regime("iron_armco", mid).tier == current:
                a = mid
            else:
                b_ = mid
        current = magnetar.classify_regime("iron_armco", b_).tier
        order.append({"tier": current, "from": sig(b_, 6)})
        lo = b_
    return order


def chapter_home(site: Site) -> None:
    svg, _ = press_section_svg()
    site.includes["press-section.svg"] = svg
    site.data["explorer"] = explorer_data()
    copper = materials.get("copper_ofhc")
    ranked = sorted(benchmarks.PRESSURE_BENCHMARKS, key=lambda x: x.value)
    rows = [{"label": _PRESSURE_NAMES[x.key][0].upper() + _PRESSURE_NAMES[x.key][1:], "value": sig(x.value / 1e9),
             "text": f"{x.value / 1e9:.3g} GPa", "tone": "muted"} for x in ranked]
    rows.append({"label": "This press", "value": P_HI / 1e9, "text": f"{P_HI / 1e9:.2f} GPa", "tone": "accent", "emphasis": True})
    rows.sort(key=lambda r: r["value"])
    site.figure("explorer_ladder", {
        "type": "dot",
        "title": "Where this pressure sits",
        "subtitle": "Working pressures of existing processes, with the press set by the controls above.",
        "x": {"label": "Pressure (GPa, log scale)", "scale": "log", "min": 0.05, "max": 600.0, "short": "Pressure"},
        "rows": rows,
        "bands": [{"axis": "x", "from": P_LO / 1e9, "to": P_HI / 1e9, "label": "concept note"}],
    })
    below = [x for x in ranked if x.value <= P_HI][-1]
    above = [x for x in ranked if x.value > P_HI][0]
    site.nums(
        h_energy_tnt=fx(REF.stored_energy(B_HI) / (magnetar.MEGATON_TNT / 1e9), 0),
        h_copper_melt=fx(eddy.skin_melting_field(copper), 0),
        h_sc_status=benchmarks.superconducting_field_status(B_HI).value,
        h_pulsed_status=benchmarks.pulsed_field_status(B_HI, 2.0 * REF.inner_radius).value,
        h_below=f"{_PRESSURE_NAMES[below.key]} ({below.value / 1e9:.3g} GPa)",
        h_above=f"{_PRESSURE_NAMES[above.key]} ({above.value / 1e9:.3g} GPa)",
    )
