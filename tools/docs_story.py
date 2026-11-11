"""Data for the story chapters: named numbers and chart specs, computed from the models.

Each ``chapter_*`` function fills a :class:`docs_common.Site` with the numbers its page quotes
(``site.num``) and the charts it shows (``site.figure``).  ``CHAPTERS`` lists them in reading order.
"""

from __future__ import annotations

import math

from docs_common import (CONCEPT_BAND_GPA, CONCEPT_BAND_T, P_HI, P_LO, REF, Site, area, axis, bench, fx, gpa,
                         label, line, mpa, pct, plain, points, sg, sig, sigs, table)

from magform import (benchmarks, claims, conductor, containment, eddy, fieldmap, magnetar, magnetics, materials,
                     microstructure, plasticity, porosity, press, pulse, synthesis, welding)
from magform._numerics import linspace, logspace
from magform.constants import MU0, T_ROOM, ZERO_CELSIUS
from magform.magnetics import Solenoid, field_for_pressure, magnetic_pressure
from magform.press import design_press, reference_spec
from magform.status import Status

B_LO, B_HI = field_for_pressure(P_LO), field_for_pressure(P_HI)
SOLAR_LUMINOSITY = 3.828e26  # W, IAU 2015 nominal solar luminosity


def _us(seconds: float, digits: int = 0) -> str:
    return fx(seconds * 1e6, digits)


def _ms(seconds: float, digits: int = 1) -> str:
    return fx(seconds * 1e3, digits)


def _c(kelvin: float) -> str:
    return fx(kelvin - ZERO_CELSIUS, 0)


# ---------------------------------------------------------------------------
# chapter 1: what four gigapascals means
# ---------------------------------------------------------------------------
def chapter_pressure(site: Site) -> None:
    b_lo, b_hi, b_2 = field_for_pressure(P_LO), field_for_pressure(P_HI), field_for_pressure(2e9)
    emf_lo, emf_hi = benchmarks.EMF_CONVENTIONAL_FIELD
    place = benchmarks.where_does_it_sit(P_HI)
    tungsten, copper, aluminium = materials.get("tungsten"), materials.get("copper_ofhc"), materials.get("aluminum_6061")
    site.nums(
        p_field_lo=fx(b_lo, 2), p_field_hi=fx(b_hi, 2), p_field_2=fx(b_2, 2),
        p_field_lo_round=fx(b_lo, 0), p_field_hi_round=fx(b_hi, 0),
        p_tnt_lo=pct(P_LO / benchmarks.TNT_ENERGY_DENSITY), p_tnt_hi=pct(P_HI / benchmarks.TNT_ENERGY_DENSITY),
        p_emf_field_lo=fx(emf_lo, 0), p_emf_field_hi=fx(emf_hi, 0),
        p_emf_pressure_lo=gpa(magnetic_pressure(emf_lo), 2), p_emf_pressure_hi=gpa(magnetic_pressure(emf_hi), 2),
        p_over_emf_lo=fx(P_LO / magnetic_pressure(emf_hi), 1), p_over_emf_hi=fx(P_HI / magnetic_pressure(emf_lo), 0),
        p_static_100=fx(magnetics.pressure_with_background(100.0, 0.0, 0.0), 0),
        p_compress_w=pct(tungsten.volumetric_strain(P_HI), 1),
        p_compress_cu=pct(copper.volumetric_strain(P_HI), 1),
        p_compress_al=pct(aluminium.volumetric_strain(P_HI), 1),
        p_nearest=place.nearest,
        p_die=gpa(bench("pressure", "cold_forging_die").value, 1),
        p_hpht=gpa(bench("pressure", "hpht_cubic").value, 0),
        p_hip=gpa(bench("pressure", "hip").value, 1),
        p_lanl_pressure=gpa(bench("pressure", "lanl_100t_pressure").value, 2),
    )
    fields = linspace(0.0, 120.0, 61)
    site.figure("pressure_field", {
        "type": "xy",
        "title": "Magnetic pressure rises with the square of the field",
        "subtitle": "P = B²/2μ₀. The concept note's 1.6–4 GPa is a 63–100 T problem.",
        "x": axis("Field at the workpiece surface (T)", lo=0, hi=120, unit="T", short="Field"),
        "y": axis("Magnetic pressure (GPa)", lo=0, hi=6, unit="GPa", short="Pressure"),
        "series": [line("B²/2μ₀", fields, [magnetic_pressure(b) / 1e9 for b in fields], slot=1)],
        "bands": [CONCEPT_BAND_T],
        "marks": [
            {"x": sig(emf_hi), "y": sig(magnetic_pressure(emf_hi) / 1e9), "label": "industrial forming coils top out here",
             "tone": "ink", "tip": "Upper end of industrial electromagnetic forming"},
            {"x": sig(benchmarks.PULSED_NONDESTRUCTIVE_RECORD), "y": sig(magnetic_pressure(benchmarks.PULSED_NONDESTRUCTIVE_RECORD) / 1e9),
             "label": "non-destructive record", "tone": "ink", "anchor": "end", "dy": -12,
             "tip": f"Los Alamos {benchmarks.PULSED_NONDESTRUCTIVE_RECORD:g} T pulsed magnet, "
                    f"{benchmarks.PULSED_NONDESTRUCTIVE_BORE * 1e3:g} mm bore (2012)"},
            {"x": sig(benchmarks.SC_FIELD_RECORD), "y": sig(magnetic_pressure(benchmarks.SC_FIELD_RECORD) / 1e9),
             "label": "all-superconducting record", "tone": "ink", "dy": 14,
             "tip": "Highest field from an all-superconducting magnet"},
        ],
        "table": table(["Field (T)", "Pressure (GPa)", "What it is"], [
            [emf_lo, magnetic_pressure(emf_lo) / 1e9, "lower end of industrial electromagnetic forming"],
            [benchmarks.SC_FIELD_RECORD, magnetic_pressure(benchmarks.SC_FIELD_RECORD) / 1e9, "all-superconducting record"],
            [emf_hi, magnetic_pressure(emf_hi) / 1e9, "upper end of industrial electromagnetic forming"],
            [b_lo, P_LO / 1e9, "concept note, lower bound"],
            [b_2, 2.0, "concept note, refractory forging"],
            [b_hi, P_HI / 1e9, "concept note, upper bound"],
            [benchmarks.PULSED_NONDESTRUCTIVE_RECORD, magnetic_pressure(benchmarks.PULSED_NONDESTRUCTIVE_RECORD) / 1e9,
             f"non-destructive pulsed record ({benchmarks.PULSED_NONDESTRUCTIVE_BORE * 1e3:g} mm bore)"],
        ]),
    })
    rows = []
    for b in sorted(benchmarks.PRESSURE_BENCHMARKS, key=lambda item: item.value):
        rows.append({"label": _PRESSURE_LABELS.get(b.key, b.description), "value": sig(b.value / 1e9),
                     "text": f"{sg(b.value / 1e9, 2).replace('&nbsp;', ' ')} GPa", "tone": "ink",
                     "detail": b.citation})
    site.figure("pressure_ladder", {
        "type": "dot",
        "title": "Where 1.6–4 GPa sits among working pressures",
        "subtitle": "Inside the range of cold-forging dies, below every static route to diamond.",
        "x": axis("Pressure (GPa, log scale)", scale="log", lo=0.1, hi=1000, short="Pressure"),
        "rows": rows,
        "bands": [CONCEPT_BAND_GPA],
        "table": table(["Process", "Pressure (GPa)", "Source"],
                       [[r["label"], r["value"], r["detail"]] for r in rows]),
    })


_PRESSURE_LABELS = {
    "hip": "Hot isostatic pressing",
    "mpw_typical": "Magnetic pulse welding (typical)",
    "emf_50t": "50 T forming coil",
    "cold_forging_die": "Cold-forging die contact",
    "lanl_100t_pressure": "100.75 T pulsed magnet",
    "hpht_cubic": "Industrial diamond presses",
    "npd_multianvil": "Nano-polycrystalline diamond",
    "dupont_shock": "Shock-synthesised diamond",
    "carbon_nitride_dac": "Carbon nitrides (diamond anvil cell)",
    "dac_static": "Diamond anvil cell, upper range",
}



# ---------------------------------------------------------------------------
# chapter 2: the machine
# ---------------------------------------------------------------------------
_ARCH_BLURB = {
    "hts_pulsed": "The note's own magnet, swung to full field on every shot.",
    "hts_dc": "The same magnet held at constant field.",
    "pulsed_resistive": "A reinforced copper-alloy coil fired from a capacitor bank: the technology of the 100 T record.",
    "hybrid_dc_pulsed": "A steady superconducting outsert with a pulsed copper insert inside it.",
    "single_turn": "A copper strip and a very fast bank. Reaches these fields for microseconds and is destroyed doing it.",
    "industrial_emf": "Today's forming coil, run at its field limit over the same bore.",
}


def _architecture_rows(design) -> list[dict]:
    rows = []
    for a in design.architectures:
        rows.append({
            "key": a.key, "name": a.name, "blurb": _ARCH_BLURB[a.key], "status": a.status.value,
            "field": fx(a.field, 1), "rise": _rise_text(a.rise_time), "delivered": pct(a.pressure_fraction),
            "reusable": a.reusable, "limiting": a.limiting,
            "energy": fx(a.energy / 1e6, 1) if a.energy is not None else None,
            "components": [{"name": k, "status": v} for k, v in a.components.items()],
            "warnings": list(a.warnings), "notes": list(a.notes),
            "relaxed_rise": _rise_text(a.relaxed_rise_time) if a.relaxed_rise_time else None,
            "relaxed_delivered": pct(a.relaxed_pressure_fraction) if a.relaxed_pressure_fraction is not None else None,
            "relaxed_status": a.relaxed_status.value if a.relaxed_status is not None else None,
        })
    return rows


def _rise_text(seconds: float | None) -> str:
    if seconds is None:
        return "steady"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds:.3g} s"


def chapter_machine(site: Site) -> None:
    d4, d16 = design_press(reference_spec(P_HI)), design_press(reference_spec(P_LO))
    t4, t16 = design_press(reference_spec(P_HI, workpiece_wall=2e-3)), design_press(reference_spec(P_LO, workpiece_wall=2e-3))
    tape = conductor.REBCOTape()
    comp = conductor.CNSComposite.default()
    by4 = {a.key: a for a in d4.architectures}
    by16 = {a.key: a for a in d16.architectures}
    tube4 = {a.key: a for a in t4.architectures}
    hyb = by16["hybrid_dc_pulsed"]
    bench_m = d4.structure.benchmark
    site.nums(
        m_sc_record=fx(benchmarks.SC_FIELD_RECORD, 1), m_sc_conductor=fx(benchmarks.SC_CONDUCTOR_RECORD, 1),
        m_sc_established=fx(benchmarks.SC_FIELD_ESTABLISHED, 0),
        m_pulsed_record=fx(benchmarks.PULSED_NONDESTRUCTIVE_RECORD, 2),
        m_pulsed_bore=fx(benchmarks.PULSED_NONDESTRUCTIVE_BORE * 1e3, 0),
        m_energy_hi=fx(d4.stored_energy / 1e6, 0), m_energy_lo=fx(d16.stored_energy / 1e6, 0),
        m_hoop_hi=gpa(d4.peak_hoop_stress, 2), m_hoop_lo=gpa(d16.peak_hoop_stress, 2),
        m_hoop_ratio=fx(d4.hoop_ratio, 2),
        m_modulus_hi=fx(d4.structure.required_composite_modulus / 1e9, 0),
        m_modulus_lo=fx(d16.structure.required_composite_modulus / 1e9, 0),
        m_modulus_default=fx(comp.modulus() / 1e9, 0),
        m_modulus_m60j=fx(bench_m["carbon_fibre_m60j"]["composite_modulus"] / 1e9, 0),
        m_modulus_graphene=fx(bench_m["graphene_intrinsic"]["composite_modulus"] / 1e9, 0),
        m_allowable=gpa(comp.allowable_stress(), 2),
        m_strain_limit=pct(tape.irreversible_strain, 1),
        m_je_lo=fx(tape.je(B_LO, 4.2, "parallel") / 1e6, 0), m_je_hi=fx(tape.je(B_HI, 4.2, "parallel") / 1e6, 0),
        m_je_measured=fx(conductor.JE_MEASURED_FIELD, 0),
        m_j_hi=fx(d4.current_density / 1e6, 0), m_j_lo=fx(d16.current_density / 1e6, 0),
        m_cond_margin_lo=fx(d16.conductor.margin, 2), m_cond_margin_hi=fx(d4.conductor.margin, 2),
        m_end_fraction=fx(d4.end_field_fraction, 2), m_end_default=fx(conductor.DEFAULT_END_FIELD_FRACTION, 1),
        m_peak_field=fx(d4.peak_conductor_field, 1),
        m_rise_billet=_ms(d4.rise_time, 0), m_rise_tube=_us(t4.rise_time, 0),
        m_ramp_lo=sg(d16.pulsed_operation.ramp_rate, 2), m_ramp_tube=sg(t16.pulsed_operation.ramp_rate, 2),
        m_ramp_sota=fx(benchmarks.SC_RAMP_RATE_SOTA, 0),
        m_quench_ratio=fx(d16.pulsed_operation.loss_to_margin_ratio, 0),
        m_quench_ratio_tube=fx(t16.pulsed_operation.loss_to_margin_ratio, 0),
        m_loss=sg(d16.pulsed_operation.loss.q_total / 1e6, 2),
        m_margin=sg(d16.pulsed_operation.enthalpy_margin / 1e6, 2),
        m_resistive_turns=fx(by4["pulsed_resistive"].details["turns"], 0),
        m_resistive_kv=fx(by4["pulsed_resistive"].details["charge_voltage"] / 1e3, 0),
        m_resistive_ma=fx(by4["pulsed_resistive"].details["peak_current"] / 1e6, 1),
        m_resistive_bank=fx(by4["pulsed_resistive"].energy / 1e6, 0),
        m_tube_relaxed_rise=_ms(tube4["pulsed_resistive"].relaxed_rise_time, 1),
        m_tube_relaxed_pct=pct(tube4["pulsed_resistive"].relaxed_pressure_fraction),
        m_tube_kv=fx(tube4["pulsed_resistive"].details["charge_voltage"] / 1e3, 0),
        m_tube_ma=fx(tube4["pulsed_resistive"].details["peak_current"] / 1e6, 0),
        m_hybrid_bg=fx(hyb.details["background_field"], 1), m_hybrid_pulse=fx(hyb.details["pulse_increment"], 1),
        m_hybrid_total=fx(hyb.field, 1), m_hybrid_saving=pct(hyb.details["energy_saving"]),
        m_hybrid_flux=fx(hyb.details["flux_ratio"], 1), m_hybrid_dip=fx(hyb.details["field_dip"], 0),
        m_hybrid_total_hi=fx(by4["hybrid_dc_pulsed"].field, 1),
        m_single_ma=fx(by4["single_turn"].details["peak_current"] / 1e6, 0),
        m_single_kv=fx(by4["single_turn"].details["charge_voltage"] / 1e3, 0),
        m_emf_pct_lo=pct(by16["industrial_emf"].pressure_fraction), m_emf_pct_hi=pct(by4["industrial_emf"].pressure_fraction),
        m_best=by4["pulsed_resistive"].name.lower(),
    )
    site.data["architectures"] = {
        "billet_4": _architecture_rows(d4), "billet_16": _architecture_rows(d16),
        "tube_4": _architecture_rows(t4), "tube_16": _architecture_rows(t16),
        "best": {"billet_4": d4.best_architecture, "billet_16": d16.best_architecture,
                 "tube_4": t4.best_architecture, "tube_16": t16.best_architecture},
    }

    # record fields by technology
    rows = []
    for b in sorted(benchmarks.FIELD_LADDER, key=lambda item: item.value):
        destructive = "estructive" in b.description and "Non-destructive" not in b.description
        rows.append({"label": _FIELD_LABELS.get(b.key, b.description), "value": sig(b.value),
                     "text": f"{b.value:g} T ({b.year})", "tone": "muted" if destructive else "ink",
                     "detail": b.citation + (" The magnet is destroyed." if destructive else "")})
    site.figure("field_ladder", {
        "type": "dot",
        "title": "Record fields, by how the field is made",
        "subtitle": "Superconducting magnets stop at about a third of the field the note needs. Grey dots destroy the coil.",
        "x": axis("Field (T, log scale)", scale="log", lo=10, hi=5000, short="Field"),
        "rows": rows, "bands": [CONCEPT_BAND_T],
        "table": table(["Technology", "Field (T)", "Year", "Source"],
                       [[_FIELD_LABELS.get(b.key, b.description), b.value, b.year, b.citation]
                        for b in sorted(benchmarks.FIELD_LADDER, key=lambda item: item.value)]),
    })

    fields = linspace(1.0, 120.0, 60)
    site.figure("je_field", {
        "type": "xy",
        "title": "What the superconductor can carry",
        "subtitle": "Engineering current density of REBCO tape at 4.2 K. Measured to 31 T; everything beyond is the model's extrapolation.",
        "x": axis("Field on the tape (T)", lo=0, hi=120, unit="T", short="Field"),
        "y": axis("Critical current density (A/mm², log scale)", scale="log", lo=50, hi=4000, unit="A/mm²",
                  short="Current density"),
        "series": [
            line("Field in the tape plane", fields, [tape.je(b, 4.2, "parallel") / 1e6 for b in fields], slot=1, end_label=False),
            line("Field across the tape", fields, [tape.je(b, 4.2, "perpendicular") / 1e6 for b in fields], slot=2),
        ],
        "bands": [CONCEPT_BAND_T],
        "rules": [{"axis": "x", "at": conductor.JE_MEASURED_FIELD, "label": "measured to here"},
                  {"axis": "x", "at": benchmarks.SC_CONDUCTOR_RECORD, "label": "operated to here", "dy": 30}],
        "table": table(["Field (T)", "In plane (A/mm²)", "Across (A/mm²)"],
                       [[b, tape.je(b, 4.2, "parallel") / 1e6, tape.je(b, 4.2, "perpendicular") / 1e6]
                        for b in (12.0, 18.0, 31.0, benchmarks.SC_CONDUCTOR_RECORD, B_LO, B_HI)]),
        "note": "A parametric fit representative of commercial 4 mm tape, not a datasheet.",
    })

    alphas = linspace(1.2, 8.0, 69)
    site.figure("hoop_ratio", {
        "type": "xy",
        "title": "A thicker coil does not escape the load",
        "subtitle": "Peak hoop stress in the winding, as a multiple of the magnetic pressure it produces.",
        "x": axis("Outer radius / inner radius", lo=1, hi=8, short="Radius ratio"),
        "y": axis("Peak hoop stress / magnetic pressure", lo=0, hi=5, short="Stress ratio"),
        "series": [line("Peak hoop stress", alphas, [magnetics.max_hoop_stress_ratio(a) for a in alphas], slot=1)],
        "rules": [{"axis": "y", "at": 0.5, "label": "floor: one half, however thick"}],
        "marks": [{"x": 5.0, "y": sig(magnetics.max_hoop_stress_ratio(5.0)), "label": "reference press", "tone": "ink",
                   "dy": -12}],
        "table": table(["Radius ratio", "Stress ratio"], [[a, magnetics.max_hoop_stress_ratio(a)]
                                                          for a in (1.2, 1.5, 2.0, 3.0, 3.414, 5.0, 8.0)]),
        "note": "Long coil, turns carrying their own load. The reference press is 7 % above this curve because it is short.",
    })

    names = {"carbon_fibre_t1100g": "T1100G carbon fibre", "carbon_fibre_m60j": "M60J carbon fibre",
             "cnt_fibre_record": "Record nanotube fibre", "cns_engineering_target": "Nanosheet target (hypothetical)",
             "graphene_intrinsic": "Perfect graphene (ceiling)"}
    tones = {"carbon_fibre_t1100g": "state_of_the_art", "carbon_fibre_m60j": "state_of_the_art",
             "cnt_fibre_record": "extrapolation", "cns_engineering_target": "aspirational",
             "graphene_intrinsic": "aspirational"}
    bar_rows = [{"label": names[k], "value": sig(bench_m[k]["composite_modulus"] / 1e9),
                 "text": f"{bench_m[k]['composite_modulus'] / 1e9:.0f} GPa", "tone": tones[k],
                 "detail": f"Reinforcement modulus {bench_m[k]['modulus'] / 1e9:.0f} GPa at half the winding volume"}
                for k in names]
    site.figure("modulus_gap", {
        "type": "bars",
        "title": "The winding has to be stiffer than anything that exists",
        "subtitle": "Stiffness of a winding that is half reinforcement, against what the superconductor's 0.4 % strain limit demands.",
        "x": axis("Winding modulus (GPa)", lo=0, hi=1500, short="Modulus"),
        "rows": bar_rows,
        "rules": [{"axis": "x", "at": sig(d16.structure.required_composite_modulus / 1e9), "label": "needed at 1.6 GPa"},
                  {"axis": "x", "at": sig(d4.structure.required_composite_modulus / 1e9), "label": "needed at 4 GPa"}],
        "table": table(["Reinforcement", "Winding modulus (GPa)", "Reinforcement modulus (GPa)"],
                       [[names[k], bench_m[k]["composite_modulus"] / 1e9, bench_m[k]["modulus"] / 1e9] for k in names]),
        "note": "Reference press, safety factor 1.5. Colour marks how real the reinforcement is.",
    })

    times = logspace(-6.0, 0.0, 49)
    j_tape = d16.current_density / comp.f_hts
    t_cs = tape.current_sharing_temperature(B_LO, j_tape, "parallel")
    margin = conductor.enthalpy_margin(4.2, t_cs, tape)
    site.figure("quench_margin", {
        "type": "xy",
        "title": "Every pulse quenches the superconductor",
        "subtitle": "Heat left in the tape by one field pulse, against the heat it can absorb before it stops superconducting.",
        "x": axis("Field rise time (log scale)", scale="log", lo=1e-6, hi=1.0, short="Rise time", time=True),
        "y": axis("Heat per pulse (MJ per m³ of tape, log scale)", scale="log", lo=0.05, hi=3000, unit="MJ/m³", short="Heat"),
        "series": [
            line("Pulse to 100 T", times, [conductor.tape_pulse_losses(B_HI, t, 4.2, tape, d4.end_field_fraction).q_total / 1e6
                                           for t in times], slot=2),
            line("Pulse to 63 T", times, [conductor.tape_pulse_losses(B_LO, t, 4.2, tape, d4.end_field_fraction).q_total / 1e6
                                          for t in times], slot=1),
        ],
        "rules": [{"axis": "y", "at": sig(margin / 1e6), "label": "all the heat the tape can absorb (63 T design)", "side": "left"}],
        "table": table(["Rise time (s)", "Heat, 63 T (MJ/m³)", "Heat, 100 T (MJ/m³)"],
                       [[t, conductor.tape_pulse_losses(B_LO, t, 4.2, tape, d4.end_field_fraction).q_total / 1e6,
                         conductor.tape_pulse_losses(B_HI, t, 4.2, tape, d4.end_field_fraction).q_total / 1e6]
                        for t in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)]),
        "note": f"Margin {margin / 1e6:.2g} MJ/m³: 4.2 K to the current-sharing temperature at 63 T. "
                "Slower pulses lose less to eddy currents, but the hysteresis floor stays far above the margin.",
    })


_FIELD_LABELS = {
    "nhmfl_32t_all_sc": "All-superconducting (2017)",
    "secuf_35p6t_all_sc": "All-superconducting record",
    "rebco_insert_45p5t": "Superconducting insert in a resistive magnet (2019)",
    "lbc9_48p7t": "Superconducting insert in a resistive magnet",
    "chmfl_resistive_42t": "Resistive, steady",
    "chmfl_hybrid_45t": "Hybrid, steady",
    "lanl_100t_nondestructive": "Pulsed, coil survives",
    "single_turn_300t": "Single-turn coil",
    "emfc_1200t": "Electromagnetic flux compression",
    "explosive_fc_2800t": "Explosive flux compression",
}


# ---------------------------------------------------------------------------
# chapter 3: the pulse
# ---------------------------------------------------------------------------
def chapter_power(site: Site) -> None:
    e_hi, e_lo = REF.stored_energy(B_HI), REF.stored_energy(B_LO)
    record_bore = benchmarks.PULSED_NONDESTRUCTIVE_BORE
    bore_label = f"{record_bore * 1e3:g} mm bore"
    insert = Solenoid(0.5 * record_bore, 0.11, 0.15)     # one thick coil with the record's bore, as a comparison
    e_insert = insert.stored_energy(benchmarks.PULSED_NONDESTRUCTIVE_RECORD)
    rows = []
    for field in (B_HI, B_LO):
        for rise in (2.5e-5, 1e-3, 1e-2):
            n = pulse.assess_pulse_network(REF, field, rise)
            rows.append({"field": fx(field, 0), "rise": _rise_text(rise), "turns": n.turns,
                         "kv": fx(n.charge_voltage / 1e3, 0), "ma": fx(n.peak_current / 1e6, 2),
                         "power": sg(n.peak_power / 1e9, 3), "bank": fx(n.bank_energy / 1e6, 0),
                         "status": n.status.value, "thermal": n.thermal_status.value})
    slow = pulse.assess_pulse_network(REF, B_HI, 1e-2)
    fast = pulse.assess_pulse_network(REF, B_HI, 2.5e-5)
    lanl = pulse.assess_pulse_network(insert, benchmarks.PULSED_NONDESTRUCTIVE_RECORD, 5e-3)
    hpg = pulse.homopolar_for_energy(e_hi)
    hpg_rise = hpg.quarter_period(pulse.coil_electrical(REF, 1).inductance)
    heat = pulse.conductor_heating(REF, B_HI, 1e-2)
    heat_lo = pulse.conductor_heating(REF, B_LO, 1e-2)
    heat_sine = pulse.conductor_heating(REF, B_HI, 1e-2, waveform="half_sine")
    sources_fast = pulse.source_comparison(e_hi, 2.5e-5)
    sources_slow = pulse.source_comparison(e_hi, 1e-2)
    site.nums(
        e_energy_hi=fx(e_hi / 1e6, 1), e_energy_lo=fx(e_lo / 1e6, 1), e_energy_insert=fx(e_insert / 1e6, 1),
        e_energy_ratio=fx(e_hi / e_insert, 0),
        e_record_bank=fx(benchmarks.PULSED_RECORD_INSERT_BANK / 1e6, 1),
        e_bank_ratio=fx(slow.bank_energy / benchmarks.PULSED_RECORD_INSERT_BANK, 0),
        e_generator_gw=fx(benchmarks.PULSED_RECORD_GENERATOR_POWER / 1e9, 1),
        e_record_outsert=fx(benchmarks.PULSED_RECORD_OUTSERT_ENERGY / 1e6, 0),
        e_area_ratio=fx((REF.inner_radius * 2.0 / record_bore) ** 2, 0),
        e_scaled_record=fx(REF.stored_energy(B_HI) * (record_bore / (2.0 * REF.inner_radius)) ** 3 / 1e6, 2),
        e_half_metre=fx(REF.stored_energy(B_HI) * (0.5 / (2.0 * REF.inner_radius)) ** 3 / 1e9, 1),
        e_product=sg(pulse.voltage_current_product(e_hi, 2.5e-5), 2),
        e_fast_kv=fx(fast.charge_voltage / 1e3, 0), e_fast_ma=fx(fast.peak_current / 1e6, 0),
        e_fast_tw=fx(fast.peak_power / 1e12, 1),
        e_slow_turns=str(slow.turns), e_slow_kv=fx(slow.charge_voltage / 1e3, 1), e_slow_ma=fx(slow.peak_current / 1e6, 2),
        e_slow_gw=fx(slow.peak_power / 1e9, 0), e_slow_bank=fx(slow.bank_energy / 1e6, 0),
        e_slow_mass=fx(slow.bank_mass / 1e3, 0), e_slow_switches=str(slow.switch_count),
        e_min_rise=_us(slow.minimum_rise_time, 0),
        e_bank_sota=fx(benchmarks.ENERGY_MAGNET_DRIVER_SOTA / 1e6, 0),
        e_store_record=fx(benchmarks.ENERGY_ANY_STORE_RECORD / 1e9, 1),
        e_power_sota=fx(benchmarks.PEAK_POWER_SOTA / 1e12, 0), e_power_record=fx(benchmarks.PEAK_POWER_RECORD / 1e12, 0),
        e_lanl_turns=str(lanl.turns), e_lanl_kv=fx(lanl.charge_voltage / 1e3, 1), e_lanl_ka=fx(lanl.peak_current / 1e3, 0),
        e_lanl_gw=fx(lanl.peak_power / 1e9, 2),
        e_hpg_mass=fx(hpg.mass() / 1e3, 1), e_hpg_volts=fx(hpg.voltage(), 0), e_hpg_rise=_ms(hpg_rise, 0),
        e_hpg_slow=fx(hpg_rise / 2.5e-5, 0),
        e_transfer_max=pct(pulse.inductive_transfer_efficiency(1.0, 1.0)),
        e_heat_crowbar=fx(heat.final_temperature, 0), e_heat_rise=fx(heat.temperature_rise, 0),
        e_heat_sine=fx(heat_sine.temperature_rise, 0), e_heat_lo=fx(heat_lo.final_temperature, 0),
        e_heat_limit=fx(pulse.COIL_TEMPERATURE_LIMIT, 0),
    )
    site.data["power"] = {
        "networks": rows,
        "sources_fast": [{"name": s.name, "status": s.status.value, "reusable": s.reusable, "reason": s.reason,
                          "fastest": _rise_text(s.fastest_rise_time)} for s in sources_fast],
        "sources_slow": [{"name": s.name, "status": s.status.value, "reusable": s.reusable, "reason": s.reason,
                          "fastest": _rise_text(s.fastest_rise_time)} for s in sources_slow],
    }

    times = logspace(-6.0, -1.0, 51)
    site.figure("power_rise", {
        "type": "xy",
        "title": "The faster the pulse, the larger the power station",
        "subtitle": "Peak electrical power needed to raise the field in a given time: πE/2t.",
        "x": axis("Field rise time (log scale)", scale="log", lo=1e-6, hi=1e-1, short="Rise time", time=True),
        "y": axis("Peak power (GW, log scale)", scale="log", lo=1e-2, hi=1e6, unit="GW", short="Peak power"),
        "series": [
            line("Reference press, 100 T", times, [pulse.discharge_peak_power(e_hi, t) / 1e9 for t in times], slot=1),
            line("Reference press, 63 T", times, [pulse.discharge_peak_power(e_lo, t) / 1e9 for t in times], slot=2),
            line(f"{bore_label}, 100 T", times, [pulse.discharge_peak_power(e_insert, t) / 1e9 for t in times], slot=3),
        ],
        "rules": [{"axis": "y", "at": benchmarks.PEAK_POWER_SOTA / 1e9, "label": "largest capacitor banks"},
                  {"axis": "y", "at": benchmarks.PEAK_POWER_RECORD / 1e9, "label": "Sandia Z machine"}],
        "bands": [{"axis": "x", "from": 1e-5, "to": 5e-5, "label": "thin wall"},
                  {"axis": "x", "from": 5e-3, "to": 2e-2, "label": "solid billet", "tone": "muted"}],
        "table": table(["Rise time (s)", "Reference 100 T (GW)", "Reference 63 T (GW)", f"{bore_label} 100 T (GW)"],
                       [[t, pulse.discharge_peak_power(e_hi, t) / 1e9, pulse.discharge_peak_power(e_lo, t) / 1e9,
                         pulse.discharge_peak_power(e_insert, t) / 1e9] for t in (2.5e-5, 1e-4, 1e-3, 1e-2, 1e-1)]),
    })

    bores = logspace(math.log10(0.007), 0.0, 44)

    def scaled_energy(d: float, field: float) -> float:
        return Solenoid(0.5 * d, 0.5 * d + 2.0 * d, 5.0 * d).stored_energy(field) / 1e6

    site.figure("energy_bore", {
        "type": "xy",
        "title": "Energy grows with the cube of the bore",
        "subtitle": "Field energy of a coil scaled from the reference press (build twice the bore, length five times).",
        "x": axis("Bore diameter (m, log scale)", scale="log", lo=0.007, hi=1.0, unit="m", short="Bore"),
        "y": axis("Stored energy (MJ, log scale)", scale="log", unit="MJ", short="Energy"),
        "series": [line("100 T", bores, [scaled_energy(d, B_HI) for d in bores], slot=1),
                   line("63 T", bores, [scaled_energy(d, B_LO) for d in bores], slot=2)],
        "rules": [{"axis": "y", "at": benchmarks.ENERGY_MAGNET_DRIVER_SOTA / 1e6, "label": "largest magnet bank", "side": "left"},
                  {"axis": "y", "at": benchmarks.ENERGY_ANY_STORE_RECORD / 1e6, "label": "largest pulsed store", "side": "left"}],
        "marks": [{"x": 0.1, "y": sig(scaled_energy(0.1, B_HI)), "label": "reference press", "tone": "ink", "dy": 18},
                  {"x": record_bore, "y": sig(scaled_energy(record_bore, B_HI)), "label": "the record's bore", "tone": "ink", "dy": 18}],
        "table": table(["Bore (m)", "100 T (MJ)", "63 T (MJ)"],
                       [[d, scaled_energy(d, B_HI), scaled_energy(d, B_LO)] for d in (record_bore, 0.05, 0.1, 0.2, 0.5, 1.0)]),
    })

    wave_t = linspace(0.0, 0.2, 201)
    with_crowbar = pulse.rlc_current(slow.capacitance, slow.charge_voltage, slow.inductance, slow.resistance, crowbar=True)
    ringing = pulse.rlc_current(slow.capacitance, slow.charge_voltage, slow.inductance, slow.resistance)
    site.figure("discharge", {
        "type": "xy",
        "title": "One shot of the 10 ms press",
        "subtitle": f"Current in the {slow.turns}-turn coil. The bank is shorted at the peak so the current decays instead of ringing back.",
        "x": axis("Time (ms)", lo=0, hi=200, unit="ms", short="Time"),
        "y": axis("Coil current (MA)", unit="MA", short="Current"),
        "series": [line("Shorted at the peak", [t * 1e3 for t in wave_t], [with_crowbar(t) / 1e6 for t in wave_t], slot=1),
                   line("Left to ring", [t * 1e3 for t in wave_t], [ringing(t) / 1e6 for t in wave_t], slot=0, thin=True)],
        "table": table(["Time (ms)", "Shorted at the peak (MA)", "Left to ring (MA)"],
                       [[t * 1e3, with_crowbar(t) / 1e6, ringing(t) / 1e6] for t in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)]),
        "note": f"{slow.capacitance * 1e3:.0f} mF at {slow.charge_voltage / 1e3:.1f} kV into "
                f"{slow.inductance * 1e6:.0f} µH and {slow.resistance * 1e6:.0f} µΩ.",
    })


# ---------------------------------------------------------------------------
# chapter 4: forging without a die
# ---------------------------------------------------------------------------
_SKIN_SET = (("aluminum_6061", "Aluminium 6061", 1), ("copper_ofhc", "Copper", 2), ("titanium_cp2", "Titanium", 3),
             ("steel_1018", "Mild steel", 4), ("tungsten", "Tungsten", 5))


def chapter_forging(site: Site) -> None:
    rec = eddy.pulse_frequency_recommendation("copper_ofhc", 2e-3)
    half = 2.0 * rec.quarter_period
    tube = eddy.TubeCompression("copper_ofhc", 0.025, 0.002)
    runs = {b: tube.simulate(eddy.half_sine_field(b, half), 4.0 * half) for b in (20.0, 30.0, B_LO, B_HI)}
    s20, s30, s63, s100 = (runs[b].summary for b in (20.0, 30.0, B_LO, B_HI))
    billet = fieldmap.workpiece_pressure(REF, 0.04, 0.30)
    rounded = fieldmap.workpiece_pressure(REF, 0.04, 0.30, 0.01)
    long_billet = fieldmap.workpiece_pressure(REF, 0.04, 0.60)
    axis_est = REF.pressure_uniformity(1e8, 0.30)
    melt = {k: eddy.skin_melting_field(k) for k, _, _ in _SKIN_SET}
    slab = eddy.slab_field_diffusion("copper_ofhc", eddy.half_sine_field(B_HI, 1e-4), 10 * eddy.skin_depth(1.71e-8, 5e3), 1e-4)
    site.nums(
        f_freq=fx(rec.frequency / 1e3, 1), f_rise=_us(rec.quarter_period, 0), f_skin=fx(rec.skin_depth * 1e3, 2),
        f_strain_20=fx(s20.final_strain, 2), f_v20=fx(s20.peak_velocity, 0), f_v30=fx(s30.peak_velocity, 0),
        f_v63=fx(s63.peak_velocity, 0), f_v100=fx(s100.peak_velocity, 0),
        f_close_30=_us(s30.time_to_stop, 0), f_close_63=_us(s63.time_to_stop, 0), f_close_100=_us(s100.time_to_stop, 0),
        f_rate_30=sg(s30.peak_strain_rate, 2), f_rate_63=sg(s63.peak_strain_rate, 2),
        f_plastic_heat=fx(s63.plastic_temperature_rise, 0),
        f_skin_cu_30=fx(eddy.skin_temperature_rise("copper_ofhc", 30.0), 0),
        f_skin_cu_lo=fx(eddy.skin_temperature_rise("copper_ofhc", B_LO), 0),
        f_skin_cu_hi=fx(eddy.skin_temperature_rise("copper_ofhc", B_HI), 0),
        f_skin_al_lo=fx(eddy.skin_temperature_rise("aluminum_6061", B_LO), 0),
        f_melt_al=fx(melt["aluminum_6061"], 0), f_melt_cu=fx(melt["copper_ofhc"], 0), f_melt_ti=fx(melt["titanium_cp2"], 0),
        f_melt_steel=fx(melt["steel_1018"], 0), f_melt_w=fx(melt["tungsten"], 0),
        f_slab_surface=fx(slab.surface_peak_temperature, 0), f_slab_melt_depth=fx(slab.melt_depth * 1e3, 2),
        f_mid=fx(billet.mid_ratio, 2), f_uniformity=pct(1.0 - billet.uniformity),
        f_corner=fx(billet.peak_ratio, 1), f_corner_round=fx(rounded.peak_ratio, 1),
        f_long_min=pct(long_billet.central_min_ratio), f_axis_end=fx(axis_est.end_ratio, 2),
        f_emf_hi=fx(benchmarks.EMF_CONVENTIONAL_FIELD[1], 0),
    )

    series = []
    for (b, name, slot) in ((20.0, "20 T", 3), (30.0, "30 T", 1), (B_LO, "63 T", 2), (B_HI, "100 T", 8)):
        h = runs[b].history
        step = max(1, len(h.t) // 120)
        ts = [t * 1e6 for t in h.t[::step]] + [h.t[-1] * 1e6]
        rs = [r * 1e3 for r in h.radius[::step]] + [h.radius[-1] * 1e3]
        series.append(line(name, ts, rs, slot=slot, end_label=name))
    site.figure("tube_radius", {
        "type": "xy",
        "title": "A copper tube under a magnetic pulse",
        "subtitle": "Radius of a 50 mm tube with a 2 mm wall during a pulse matched to the wall. Three of the four close the bore.",
        "x": axis("Time (µs)", lo=0, hi=130, unit="µs", short="Time"),
        "y": axis("Wall mid-radius (mm)", lo=0, hi=26, unit="mm", short="Radius"),
        "series": series,
        "table": table(["Peak field (T)", "Outcome", "Peak wall speed (m/s)", "Peak strain rate (1/s)", "Skin heating (K)"],
                       [[b, "bore closed at %.0f µs" % (r.summary.time_to_stop * 1e6) if r.summary.stopped_by == "closed"
                         else "stops at strain %.2f" % r.summary.final_strain, r.summary.peak_velocity,
                         r.summary.peak_strain_rate, r.summary.skin_temperature_rise_estimate] for b, r in runs.items()]),
        "note": "Rigid-plastic thin-shell model with field diffusion through the wall; no die or mandrel.",
    })

    fields = linspace(0.0, 150.0, 76)
    site.figure("skin_heating", {
        "type": "xy",
        "title": "The surface pays for the pressure in heat",
        "subtitle": "Temperature rise of the eddy-current skin. Each dot is where that metal's surface reaches its melting point.",
        "x": axis("Peak field at the surface (T)", lo=0, hi=150, unit="T", short="Field"),
        "y": axis("Skin temperature rise (K)", lo=0, hi=3600, unit="K", short="Rise"),
        "series": [line(name, fields, [eddy.skin_temperature_rise(key, b) for b in fields], slot=slot)
                   for key, name, slot in _SKIN_SET],
        "bands": [CONCEPT_BAND_T],
        "marks": [{"x": sig(melt[key]), "y": sig(materials.get(key).melting_point - T_ROOM), "slot": slot,
                   "tip": f"{name} skin melts at {melt[key]:.0f} T"} for key, name, slot in _SKIN_SET],
        "table": table(["Metal", "Rise at 63 T (K)", "Rise at 100 T (K)", "Skin melts at (T)"],
                       [[name, eddy.skin_temperature_rise(key, B_LO), eddy.skin_temperature_rise(key, B_HI), melt[key]]
                        for key, name, _ in _SKIN_SET]),
        "note": "A lower estimate (it ignores the resistivity rise of the hot skin). The interior stays near room temperature.",
    })

    def side(result, cut):
        zs, rs = [], []
        for r, z, ratio in zip(result.r, result.z, result.ratio):
            if r > 0.999 * result.max_radius and abs(z) <= cut:
                zs.append(z * 1e3)
                rs.append(ratio)
        order = sorted(range(len(zs)), key=lambda i: zs[i])
        return [zs[i] for i in order], [rs[i] for i in order]

    z_short, r_short = side(billet, 0.147)
    z_long, r_long = side(long_billet, 0.297)
    z_axis = linspace(-300.0, 300.0, 61)
    j_unit = REF.current_density_for_field(1.0)
    site.figure("surface_pressure", {
        "type": "xy",
        "title": "Pressure along a billet is level in the middle only",
        "subtitle": "Surface pressure on a 40 mm billet in the reference press, as a multiple of the nominal B²/2μ₀. The coil ends at ±250 mm.",
        "x": axis("Position along the billet (mm)", lo=-300, hi=300, unit="mm", short="Position"),
        "y": axis("Pressure / nominal", lo=0, hi=1.6, short="Pressure"),
        "series": [line("300 mm billet", z_short, r_short, slot=1),
                   line("600 mm billet", z_long, r_long, slot=2),
                   line("Empty-bore estimate on the axis", z_axis,
                        [(REF.axial_field(z / 1e3, j_unit)) ** 2 for z in z_axis], slot=0, thin=True)],
        "rules": [{"axis": "x", "at": -250, "label": "coil end"}, {"axis": "x", "at": 250, "label": "coil end"}],
        "table": table(["Case", "Mid-plane", "Min/max, central 80 %", "Peak (corner)"],
                       [["300 mm billet, 2 mm corners", billet.mid_ratio, billet.uniformity, billet.peak_ratio],
                        ["300 mm billet, 10 mm corners", rounded.mid_ratio, rounded.uniformity, rounded.peak_ratio],
                        ["600 mm billet, 2 mm corners", long_billet.mid_ratio, long_billet.uniformity, long_billet.peak_ratio]]),
        "note": "Perfect-conductor limit (the field stays out of the metal). The corner peaks, 2 to 6 times nominal, are off the top of the plot.",
    })


# ---------------------------------------------------------------------------
# chapter 5: tungsten, cold
# ---------------------------------------------------------------------------
_REFRACTORY = (("tungsten", "Tungsten", 1), ("molybdenum", "Molybdenum", 2), ("rhenium", "Rhenium", 3),
               ("tantalum", "Tantalum", 4), ("niobium", "Niobium", 5))


def chapter_refractory(site: Site) -> None:
    p2 = claims.P_REFRACTORY
    q = 1.2
    forge = {key: plasticity.assess_forgeability(key, p2) for key, _, _ in _REFRACTORY}
    forge4 = {key: plasticity.assess_forgeability(key, P_HI) for key, _, _ in _REFRACTORY}
    w = forge["tungsten"]
    w_warm = plasticity.assess_forgeability("tungsten", p2, temperature=w.warm_forming_temperature + 50.0)
    w_bare = plasticity.assess_forgeability("tungsten", P_HI, hydrostatic_pressure=0.0)
    m = microstructure.assess_microstructure("tungsten")
    m2 = microstructure.assess_microstructure("tungsten", peak_field=field_for_pressure(p2))
    m4 = microstructure.assess_microstructure("tungsten", peak_field=B_HI)
    cu = microstructure.assess_microstructure("copper_ofhc")
    refine = microstructure.grain_refinement_strain("tungsten", microstructure.ULTRAFINE_GRAIN_SIZE)
    tungsten = materials.get("tungsten")
    site.nums(
        r_field_2=fx(field_for_pressure(p2), 1),
        r_w_stress_lo=gpa(w.flow_stress_min, 2), r_w_stress_hi=gpa(w.flow_stress_max, 2),
        r_w_needed=gpa(w.required_pressure, 2), r_w_margin=fx(w.margin, 2),
        r_w_warm_c=_c(w.warm_forming_temperature), r_w_warm_k=fx(w.warm_forming_temperature, 0),
        r_w_warm_status=label(w_warm.status).replace("_", " "),
        r_w_dbtt_c=_c(w.dbtt), r_w_fail=pct(w.failure_strain_limit), r_w_culver=fx(w.adiabatic_shear_strain, 3),
        r_w_compress=pct(w.elastic_volumetric_strain, 1),
        r_w_margin_4=fx(forge4["tungsten"].margin, 2), r_w_status_4=label(forge4["tungsten"].status).replace("_", " "),
        r_w_bare_status=label(w_bare.status), r_w_heat=fx(w.adiabatic_temperature_rise, 0),
        r_mo_margin=fx(forge["molybdenum"].margin, 2), r_mo_stress=gpa(forge["molybdenum"].flow_stress, 2),
        r_ta_margin=fx(forge["tantalum"].margin, 2), r_nb_margin=fx(forge["niobium"].margin, 2),
        r_re_warm_c=_c(forge["rhenium"].warm_forming_temperature),
        r_w_conv_lo=_c(tungsten.conventional_working[0]), r_w_conv_hi=_c(tungsten.conventional_working[1]),
        r_w_conv_frac_lo=fx(tungsten.conventional_working[0] / tungsten.melting_point, 2),
        r_w_conv_frac_hi=fx(tungsten.conventional_working[1] / tungsten.melting_point, 2),
        r_mo_melt_c=_c(materials.get("molybdenum").melting_point),
        r_grain_before=fx(m.grain_size_before * 1e6, 0), r_grain_after=fx(m.grain_size_after * 1e6, 0),
        r_refine_strain=fx(refine, 1), r_saturation_strain=fx(m.saturation_strain, 1),
        r_dislocation=sg(m.dislocation_density, 2), r_dislocation_factor=fx(m.dislocation_density / m.initial_dislocation_density, 0),
        r_cell=fx(m.cell_size * 1e6, 1), r_yield_before=gpa(m.yield_strength_before, 2),
        r_yield_after=gpa(m.yield_strength_after, 2), r_hardening=pct(m.hardening_ratio - 1.0),
        r_bulk_frac=fx(m.bulk_homologous_temperature, 2), r_rex=sg(m.bulk_recrystallised_fraction, 1),
        r_skin_2=fx(m2.skin.peak_temperature, 0), r_skin_2_frac=fx(m2.skin.homologous_temperature, 2),
        r_skin_4=fx(m4.skin.peak_temperature, 0), r_skin_4_frac=fx(m4.skin.homologous_temperature, 2),
        r_skin_4_ms=_ms(m4.skin.cooling_time, 0), r_haz_depth=fx(m4.haz_depth * 1e3, 1),
        r_ufg_strength=gpa(microstructure.hall_petch_strength("tungsten", 100e-9), 1),
        r_cu_cells=fx(cu.cell_size * 1e6, 2), r_cu_hardening=fx(cu.hardening_ratio, 1),
    )
    site.data["refractory"] = [
        {"key": key, "name": name, "melting_c": _c(materials.get(key).melting_point),
         "practice": materials.get(key).conventional_working_note,
         "practice_range": (_c(materials.get(key).conventional_working[0]) if materials.get(key).conventional_working[1]
                            - materials.get(key).conventional_working[0] < 1 else
                            f"{_c(materials.get(key).conventional_working[0])}–{_c(materials.get(key).conventional_working[1])}"),
         "stress": gpa(forge[key].flow_stress, 2), "needed": gpa(forge[key].required_pressure, 2),
         "status_2": forge[key].status.value, "margin_2": fx(forge[key].margin, 2),
         "status_4": forge4[key].status.value, "margin_4": fx(forge4[key].margin, 2),
         "warm": (_c(forge[key].warm_forming_temperature) if forge[key].warm_forming_temperature
                  and forge[key].warm_forming_temperature > T_ROOM + 1 else None),
         "model": "published fit" if materials.get(key).constitutive else "estimated from tensile data"}
        for key, name, _ in _REFRACTORY]

    temps = linspace(T_ROOM, 1500.0, 50)
    lo_band, hi_band = [], []
    for t in temps:
        band = plasticity.flow_stress_band("tungsten", 0.3, 1e4, t)
        lo_band.append(q * band.minimum / 1e9)
        hi_band.append(q * band.maximum / 1e9)
    series = [area("Tungsten (range of published fits)", [t - ZERO_CELSIUS for t in temps], hi_band, lo_band, slot=1,
                   outline=False)]
    for key, name, slot in _REFRACTORY[1:4]:
        series.append(line(name, [t - ZERO_CELSIUS for t in temps],
                           [q * plasticity.flow_stress(key, 0.3, 1e4, t) / 1e9 for t in temps], slot=slot))
    site.figure("flow_stress_temperature", {
        "type": "xy",
        "title": "Pressure needed to keep a refractory metal flowing",
        "subtitle": "Required pressure at a strain of 0.3 and forming rates, against the temperature of the billet.",
        "x": axis("Billet temperature (°C)", lo=0, hi=1250, unit="°C", short="Temperature"),
        "y": axis("Required pressure (GPa)", lo=0, hi=4.5, unit="GPa", short="Pressure"),
        "series": series,
        "rules": [{"axis": "y", "at": 2.0, "label": "2 GPa press"}, {"axis": "y", "at": 4.0, "label": "4 GPa press"}],
        "table": table(["Temperature (°C)", "Tungsten, weakest fit (GPa)", "Tungsten, strongest fit (GPa)",
                        "Molybdenum (GPa)", "Rhenium (GPa)", "Tantalum (GPa)"],
                       [[t - ZERO_CELSIUS, q * plasticity.flow_stress_band("tungsten", 0.3, 1e4, t).minimum / 1e9,
                         q * plasticity.flow_stress_band("tungsten", 0.3, 1e4, t).maximum / 1e9]
                        + [q * plasticity.flow_stress(k, 0.3, 1e4, t) / 1e9 for k, _, _ in _REFRACTORY[1:4]]
                        for t in (T_ROOM, 473.15, 673.15, 873.15, 1273.15)]),
        "note": "Constraint factor 1.2. Tungsten fits are calibrated to 1,000 °C; the rhenium curve is an estimate from tensile data.",
    })

    strains = linspace(0.0, 8.0, 81)
    site.figure("grain_strain", {
        "type": "xy",
        "title": "One forming pulse does not make a fine grain",
        "subtitle": "Grain thickness against accumulated strain. Ultra-fine structures come from strains of three and more.",
        "x": axis("Accumulated strain", lo=0, hi=8, short="Strain"),
        "y": axis("Grain size (µm, log scale)", scale="log", lo=0.05, hi=50, unit="µm", short="Grain size"),
        "series": [line("Tungsten", strains, [microstructure.grain_size_after_strain("tungsten", e) * 1e6 for e in strains], slot=1),
                   line("Copper", strains, [microstructure.grain_size_after_strain("copper_ofhc", e) * 1e6 for e in strains], slot=2)],
        "rules": [{"axis": "y", "at": 1.0, "label": "ultra-fine below this line", "side": "right"}],
        "bands": [{"axis": "x", "from": 0.0, "to": 0.3, "label": "one pulse"}],
        "table": table(["Strain", "Tungsten (µm)", "Copper (µm)"],
                       [[e, microstructure.grain_size_after_strain("tungsten", e) * 1e6,
                         microstructure.grain_size_after_strain("copper_ofhc", e) * 1e6] for e in (0.0, 0.3, 1.0, 2.0, 3.1, 5.0, 8.0)]),
        "note": "Geometric thinning towards the saturation size of severe plastic deformation; an estimate, not a fitted law.",
    })



# ---------------------------------------------------------------------------
# chapter 6: welding by impact
# ---------------------------------------------------------------------------
_PAIRS = (("aluminum_6061", "steel_1018", "Aluminium 6061 onto steel"), ("aluminum_1100", "steel_1018", "Soft aluminium onto steel"),
          ("aluminum_6061", "copper_ofhc", "Aluminium 6061 onto copper"), ("copper_ofhc", "steel_1018", "Copper onto steel"),
          ("copper_ofhc", "titanium_cp2", "Copper onto titanium"))


def chapter_welding(site: Site) -> None:
    al, st = "aluminum_6061", "steel_1018"
    good = welding.assess_weld(al, st, 1e-3, 2e-3, 0.1e9)
    lo, hi = welding.assess_weld(al, st, 1e-3, 2e-3, P_LO), welding.assess_weld(al, st, 1e-3, 2e-3, P_HI)
    window = welding.weld_window(al, st, 1e-3)
    drive = welding.drive_pressure_window(al, st, 1e-3, 2e-3)
    gap = welding.standoff_window(al, st, 1e-3, P_HI)
    cu_ti = welding.assess_weld("copper_ofhc", "titanium_cp2", 1e-3, 2e-3, 0.4e9)
    angle = welding.critical_collision_angle(al, st, 400.0)
    site.nums(
        w_field=fx(good.drive_field, 0), w_field_1=fx(good.drive_field, 1), w_drive=mpa(good.drive_pressure),
        w_velocity=fx(good.impact_velocity, 0), w_vc=fx(good.collision_velocity, 0),
        w_impact=gpa(good.impact_pressure, 1), w_amplification=fx(good.pressure_amplification, 0),
        w_flight=_us(good.transit_time, 0),
        w_vmin=fx(window.vp_min, 0), w_vmax=fx(window.vp_max, 0), w_width=fx(window.width, 2),
        w_sonic=fx(window.vc_sonic, 0),
        w_drive_min=mpa(drive.pressure_min), w_drive_max=mpa(drive.pressure_max),
        w_field_min=fx(drive.field_min, 0), w_field_max=fx(drive.field_max, 0),
        w_v_lo=fx(lo.impact_velocity, 0), w_v_hi=fx(hi.impact_velocity, 0),
        w_vc_lo=fx(lo.collision_velocity, 0), w_vc_hi=fx(hi.collision_velocity, 0),
        w_gap_min=fx(gap.gap_min * 1e6, 0), w_gap_max=fx(gap.gap_max * 1e6, 0),
        w_flight_hi=fx(gap.flight_time_max * 1e6, 2),
        w_angle_min=fx(math.degrees(angle), 0),
        w_skin=fx(good.flyer_skin_temperature_rise, 0), w_skin_melt=fx(good.flyer_skin_melting_field, 0),
        w_skin_hi=fx(hi.flyer_skin_temperature_rise, 0),
        w_melt_layer=fx(good.melt_layer_estimate * 1e6, 0), w_quench=fx(good.quench_time * 1e6, 0),
        w_diff_solid=fx(good.interdiffusion_solid * 1e9, 1), w_diff_liquid=fx(good.interdiffusion_liquid * 1e9, 0),
        w_diff_fusion=fx(good.fusion_weld_interdiffusion * 1e6, 0),
        w_phases=", ".join(good.intermetallics),
        w_cuti_field=fx(cu_ti.drive_field, 0), w_cuti_velocity=fx(cu_ti.impact_velocity, 0),
        w_cuti_impact=gpa(cu_ti.impact_pressure, 1),
        w_pairs=", ".join(sorted(welding.DEMONSTRATED_PAIRS)),
    )

    angles = linspace(3.0, 25.0, 45)
    lower, melting, sonic, top, base = [], [], [], [], []
    for a in angles:
        w = welding.weld_window(al, st, 1e-3, math.radians(a))
        lower.append(w.vp_min)
        melting.append(w.vp_melt)
        sonic.append(w.vp_sonic)
        top.append(max(w.vp_max, w.vp_min))
        base.append(w.vp_min)
    site.figure("weld_window", {
        "type": "xy",
        "title": "A weld needs a speed inside a window, at an angle",
        "subtitle": "Aluminium 6061, 1 mm, onto steel. Too slow and nothing bonds; too fast and the interface melts or the collision outruns sound.",
        "x": axis("Collision angle (degrees)", lo=3, hi=25, unit="°", short="Angle"),
        "y": axis("Impact speed (m/s)", lo=0, hi=2600, unit="m/s", short="Speed"),
        "series": [
            area("Weld window", angles, top, base, slot=3, outline=False),
            line("Bonding threshold", angles, lower, slot=3),
            line("Interface melts above", angles, melting, slot=2),
            line("Collision point supersonic above", angles, sonic, slot=7),
        ],
        "marks": [
            {"x": 10.0, "y": sig(good.impact_velocity), "label": f"{good.drive_field:.0f} T drive: welds", "tone": "state_of_the_art", "dy": -13},
            {"x": 10.0, "y": sig(lo.impact_velocity), "label": "1.6 GPa drive", "tone": "infeasible"},
            {"x": 10.0, "y": sig(hi.impact_velocity), "label": "4 GPa drive", "tone": "infeasible"},
        ],
        "table": table(["Angle (°)", "Bonding threshold (m/s)", "Melting limit (m/s)", "Sonic limit (m/s)"],
                       [[a] + [getattr(welding.weld_window(al, st, 1e-3, math.radians(a)), k)
                               for k in ("vp_min", "vp_melt", "vp_sonic")] for a in (4.0, 5.0, 7.5, 10.0, 15.0, 20.0)]),
        "note": "Empirical limits from explosive welding (Deribas, Wittman, Cowan); drive across a 2 mm gap.",
    })

    rows = []
    for flyer, target, name in _PAIRS:
        w = welding.drive_pressure_window(flyer, target, 1e-3, 2e-3)
        if not w.exists:
            continue
        rows.append({"label": name, "value": sig(w.pressure_min / 1e9), "value_hi": sig(w.pressure_max / 1e9),
                     "text": f"{w.field_min:.0f}–{w.field_max:.0f} T", "slot": 3,
                     "detail": f"{w.pressure_min / 1e6:.0f}–{w.pressure_max / 1e6:.0f} MPa drive; lands at "
                               f"{w.velocity_min:.0f}–{w.velocity_max:.0f} m/s"})
    site.nums(w_factor_lo=fx(P_LO / 1e9 / max(r["value_hi"] for r in rows), 1),
              w_factor_hi=fx(P_HI / 1e9 / min(r["value"] for r in rows), 0),
              w_drive_top=fx(1e3 * max(r["value_hi"] for r in rows), 0))
    site.figure("weld_drive", {
        "type": "dot",
        "title": "The pressures that weld sit well below the note's window",
        "subtitle": "Drive pressure that lands a 1 mm flyer inside its weld window across a 2 mm gap.",
        "x": axis("Magnetic drive pressure (GPa, log scale)", scale="log", lo=0.02, hi=6, short="Drive pressure"),
        "rows": rows, "bands": [CONCEPT_BAND_GPA],
        "table": table(["Pair", "Drive pressure, low (GPa)", "Drive pressure, high (GPa)", "Field"],
                       [[r["label"], r["value"], r["value_hi"], r["text"]] for r in rows]),
    })


# ---------------------------------------------------------------------------
# chapter 7: diamond by the tonne
# ---------------------------------------------------------------------------
_ROUTE_LABELS = {
    "diamond_hpht_catalytic": "Diamond grit (metal catalyst)",
    "hpht_single_crystal": "Large diamond crystals",
    "pcd_sintering": "Polycrystalline diamond compacts",
    "diamond_direct_npd": "Nano-polycrystalline diamond",
    "cbn_catalytic": "Cubic boron nitride (catalyst)",
    "cbn_direct": "Cubic boron nitride (direct)",
    "lonsdaleite_shock": "Lonsdaleite (shock)",
    "diamond_shock_dupont": "Diamond powder (shock)",
    "c3n4_dac": "Carbon nitrides",
}


def chapter_synthesis(site: Site) -> None:
    routes = synthesis.SYNTHESIS_ROUTES
    static = [r for r in routes.values() if r.is_static]
    flyer = synthesis.FlyerSpec("copper_ofhc", 0.002, 0.01)
    porous = synthesis.FlyerSpec("copper_ofhc", 0.002, 0.01, sample="graphite_polycrystalline")
    cu_cu = synthesis.FlyerSpec("copper_ofhc", 0.002, 0.01, sample="copper_ofhc")
    lons = synthesis.assess_route("lonsdaleite_shock", P_HI, flyer=flyer)
    dupont = synthesis.assess_route("diamond_shock_dupont", P_HI, flyer=flyer)
    dupont_porous = synthesis.assess_route("diamond_shock_dupont", P_HI, flyer=porous)
    sym = synthesis.assess_route("lonsdaleite_shock", P_HI, flyer=cu_cu)
    lons_lo = synthesis.assess_route("lonsdaleite_shock", P_LO, flyer=flyer)
    plain_results = synthesis.assess_all(P_HI)
    volume = synthesis.volume_scaling(0.05, 0.5, P_HI)
    anvil = synthesis.anvil_feasibility(6e9)
    site.nums(
        s_p_min=gpa(min(r.p_min for r in static), 1),
        s_dwell_min=fx(min(r.min_dwell for r in static), 0),
        s_static_count=str(len(static)),
        s_static_infeasible=str(sum(a.status is Status.INFEASIBLE for a in plain_results if a.kind != "shock")),
        s_boundary_1500=gpa(synthesis.graphite_diamond_boundary(1500.0), 1),
        s_boundary_300=gpa(synthesis.graphite_diamond_boundary(300.0), 1),
        s_flyer_v=fx(lons.flyer_velocity, 0), s_flyer_kms=fx(lons.flyer_velocity / 1e3, 1),
        s_shock_graphite=gpa(lons.pressure_sample, 1), s_shock_porous=gpa(dupont_porous.pressure_sample, 1),
        s_shock_cucu=gpa(sym.pressure_sample, 0), s_shock_lo=gpa(lons_lo.pressure_sample, 1),
        s_lons_margin=fx(lons.pressure_margin, 2), s_dupont_margin=fx(dupont.pressure_margin, 2),
        s_lons_status_lo=label(lons_lo.status),
        s_volume_kg=fx(volume.graphite_mass, 1), s_volume_mj=fx(volume.bore_magnetic_energy / 1e6, 0),
        s_anvil=fx(anvil.max_pressure / 1e9, 0),
        s_cn_lo=fx(routes["c3n4_dac"].p_min / 1e9, 0), s_cn_hi=fx(routes["c3n4_dac"].p_max / 1e9, 0),
        s_npd=fx(routes["diamond_direct_npd"].p_min / 1e9, 0),
    )
    site.data["synthesis"] = [
        {"key": r.key, "name": _ROUTE_LABELS[r.key], "product": r.product, "kind": r.kind.replace("_", " "),
         "pressure": (f"{r.p_min / 1e9:g}–{r.p_max / 1e9:g}" if r.p_max else f"≥ {r.p_min / 1e9:g}"),
         "temperature": (f"{r.t_min:.0f}–{r.t_max:.0f}" if r.t_min else "shock-heated"),
         "dwell": plain(sg(r.min_dwell, 2)), "scale": r.current_scale, "form": r.product_form,
         "status": next(a.status.value for a in synthesis.assess_all(P_HI, flyer=flyer) if a.route == r.key)}
        for r in routes.values()]

    curves = synthesis.phase_boundary_curves(300.0, 3000.0, 28)
    kt, kp = curves["graphite_diamond_kennedy_1976"]
    bt, bp = curves["graphite_diamond_berman_simon"]
    st_, sp = curves["hbn_cbn_solozhenko_1999"]
    ct, cp = curves["hbn_cbn_corrigan_bundy_1975"]
    marks = []
    for key in ("diamond_hpht_catalytic", "pcd_sintering", "cbn_catalytic", "cbn_direct"):
        r = routes[key]
        marks.append({"x": sig(0.5 * (r.t_min + r.t_max)), "y": sig(0.5 * (r.p_min + r.p_max) / 1e9),
                      "label": _ROUTE_LABELS[key], "tone": "ink", "anchor": "end",
                      "tip": f"{_ROUTE_LABELS[key]}: {r.p_min / 1e9:g}–{r.p_max / 1e9:g} GPa, {r.t_min:.0f}–{r.t_max:.0f} K"})
    site.figure("phase_diagram", {
        "type": "xy",
        "title": "Where diamond and cubic boron nitride are made",
        "subtitle": "Equilibrium lines (two published models each) and the working points of industrial routes. Above a line, the hard phase is stable.",
        "x": axis("Temperature (K)", lo=300, hi=3000, unit="K", short="Temperature"),
        "y": axis("Pressure (GPa)", lo=0, hi=10, unit="GPa", short="Pressure"),
        "series": [
            line("Graphite to diamond (Kennedy 1976)", kt, [v / 1e9 for v in kp], slot=1),
            line("Graphite to diamond (Berman–Simon 1955)", bt, [v / 1e9 for v in bp], slot=1, thin=True),
            line("Boron nitride (Solozhenko 1999)", st_, [v / 1e9 for v in sp], slot=2),
            line("Boron nitride (Corrigan–Bundy 1975)", ct, [v / 1e9 for v in cp], slot=2, thin=True),
        ],
        "bands": [{"axis": "y", "from": 1.6, "to": 4.0, "label": "concept note: pressure only, no heat, no dwell", "side": "left"}],
        "marks": marks,
        "table": table(["Route", "Pressure (GPa)", "Temperature (K)", "Shortest dwell (s)", "Made today"],
                       [[_ROUTE_LABELS[r.key], f"{r.p_min / 1e9:g}-{(r.p_max or r.p_min) / 1e9:g}",
                         f"{r.t_min:.0f}-{r.t_max:.0f}" if r.t_min else "shock-heated", r.min_dwell, r.current_scale]
                        for r in routes.values()]),
        "note": "Nano-polycrystalline diamond (15 GPa and up) and carbon nitrides (72 GPa and up) are off the top of the scale.",
    })

    rows = []
    for r in routes.values():
        hi_dwell = r.typical_dwell[1] if r.typical_dwell else None
        rows.append({"label": _ROUTE_LABELS[r.key], "value": sig(r.min_dwell),
                     "value_hi": sig(hi_dwell) if hi_dwell and hi_dwell > r.min_dwell else None,
                     "text": _dwell_text(r.min_dwell, hi_dwell), "tone": "ink" if r.is_static else "state_of_the_art",
                     "detail": f"{r.heating}; {r.current_scale}"})
    site.figure("route_dwell", {
        "type": "dot",
        "title": "Synthesis takes time that a pulse does not have",
        "subtitle": "How long each route holds its pressure and temperature. Only the two shock routes fit inside a magnetic pulse.",
        "x": axis("Time at pressure (log scale)", scale="log", lo=1e-8, hi=1e7, short="Time at pressure", time=True,
                  ticks=[{"v": 1e-6, "label": "1 µs"}, {"v": 1e-3, "label": "1 ms"}, {"v": 1.0, "label": "1 s"},
                         {"v": 60.0, "label": "1 min"}, {"v": 3600.0, "label": "1 h"}, {"v": 86400.0, "label": "1 day"},
                         {"v": 2592000.0, "label": "30 days"}]),
        "rows": rows,
        "bands": [{"axis": "x", "from": 1e-6, "to": 1e-2, "label": "a magnetic pulse"}],
        "table": table(["Route", "Shortest dwell (s)", "Typical longest (s)", "How it is heated"],
                       [[_ROUTE_LABELS[r.key], r.min_dwell, r.typical_dwell[1] if r.typical_dwell else None, r.heating]
                        for r in routes.values()]),
    })

    drives = linspace(0.2e9, 6e9, 30)

    def shocked(spec):
        return [synthesis.assess_route("lonsdaleite_shock", p_, flyer=spec).pressure_sample / 1e9 for p_ in drives]

    site.figure("flyer_shock", {
        "type": "xy",
        "title": "The one route a pulse can run: throw a plate at it",
        "subtitle": "Shock pressure in the sample when a 2 mm copper plate, driven over 10 mm, strikes it.",
        "x": axis("Magnetic drive pressure (GPa)", lo=0, hi=6, unit="GPa", short="Drive"),
        "y": axis("Pressure in the sample (GPa)", lo=0, hi=70, unit="GPa", short="Shock pressure"),
        "series": [line("into copper (for scale)", [p_ / 1e9 for p_ in drives], shocked(cu_cu), slot=0, thin=True),
                   line("into dense graphite", [p_ / 1e9 for p_ in drives], shocked(flyer), slot=1),
                   line("into porous graphite", [p_ / 1e9 for p_ in drives], shocked(porous), slot=2)],
        "bands": [{"axis": "x", "from": 1.6, "to": 4.0, "label": "concept note"}],
        "rules": [{"axis": "y", "at": 20.0, "label": "shock diamond needs 20 GPa", "side": "left"},
                  {"axis": "y", "at": 15.0, "label": "lonsdaleite needs 15 GPa", "side": "left"}],
        "table": table(["Drive (GPa)", "Flyer speed (m/s)", "Dense graphite (GPa)", "Porous graphite (GPa)", "Copper (GPa)"],
                       [[p_ / 1e9, synthesis.assess_route("lonsdaleite_shock", p_, flyer=flyer).flyer_velocity,
                         synthesis.assess_route("lonsdaleite_shock", p_, flyer=flyer).pressure_sample / 1e9,
                         synthesis.assess_route("lonsdaleite_shock", p_, flyer=porous).pressure_sample / 1e9,
                         synthesis.assess_route("lonsdaleite_shock", p_, flyer=cu_cu).pressure_sample / 1e9]
                        for p_ in (0.5e9, 1.6e9, 2.5e9, 4e9, 6e9)]),
        "note": "Planar impedance match on linear Hugoniots; the graphite Hugoniots are estimates.",
    })


def _dwell_text(lo: float, hi: float | None) -> str:
    def one(seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds * 1e6:.3g} µs"
        if seconds < 60:
            return f"{seconds:.3g} s"
        if seconds < 7200:
            return f"{seconds / 60:.3g} min"
        if seconds < 2 * 86400:
            return f"{seconds / 3600:.3g} h"
        return f"{seconds / 86400:.3g} days"
    return one(lo) if not hi or hi <= lo else f"{one(lo)} to {one(hi)}"


# ---------------------------------------------------------------------------
# chapter 8: zero porosity
# ---------------------------------------------------------------------------
def chapter_porosity(site: Site) -> None:
    tial = "gamma_tial"
    a = porosity.assess_pore_closure(tial, P_HI, initial_porosity=0.005)
    a_lo = porosity.assess_pore_closure(tial, P_LO, initial_porosity=0.005)
    hip = porosity.hip_cycle(tial)
    at_hip = porosity.assess_pore_closure(tial, hip.pressure, temperature=hip.temperature, pulse_duration=hip.time)
    strength = porosity.matrix_strength(tial)
    thin = porosity.assess_pore_closure(tial, P_HI, section_thickness=2e-3, drive_frequency=1e4)
    ti64 = porosity.assess_pore_closure("ti6al4v", P_HI)
    in718 = porosity.assess_pore_closure("inconel_718", P_HI)
    fe_powder = porosity.assess_powder_compaction("iron_armco", P_LO)
    tial_powder = porosity.assess_powder_compaction(tial, P_HI)
    mat = materials.get(tial)
    site.nums(
        z_strength_slow=gpa(strength.quasi_static, 2), z_strength_fast=gpa(strength.dynamic, 2),
        z_threshold_slow=gpa(a.collapse_pressure_quasi_static, 2), z_threshold_fast=gpa(a.collapse_pressure_dynamic, 2),
        z_close_slow=gpa(a.closure_pressure_quasi_static, 2), z_close_fast=gpa(a.closure_pressure_dynamic, 2),
        z_factor=fx(a.collapse_pressure_quasi_static / strength.quasi_static, 1),
        z_collapse_ns=fx(a.collapse_time * 1e9, 0), z_bond_time=sg(a.diffusion_bond_time, 2),
        z_wall_strain=fx(a.wall_strain, 1), z_ductility=pct(a.ductility, 1), z_dbtt_c=_c(mat.dbtt),
        z_status_lo=label(a_lo.status), z_closure_lo=label(a_lo.closure_status).replace("_", " "),
        z_closure_hi=label(a.closure_status).replace("_", " "), z_healing_hi=label(a.healing_status),
        z_hip_mpa=mpa(hip.pressure), z_hip_c=_c(hip.temperature), z_hip_h=fx(hip.time / 3600, 0),
        z_hip_bond_s=fx(porosity.diffusion_bonding_time(tial, hip.temperature), 1),
        z_hip_status=label(at_hip.status).replace("_", " "),
        z_skin_mm=fx(thin.skin_depth * 1e3, 1), z_transfer=pct(thin.pressure_transfer),
        z_resistivity=fx(mat.resistivity * 1e6, 1),
        z_ti64_left=pct(ti64.residual_porosity_dynamic, 2), z_ti64_need=gpa(ti64.closure_pressure_dynamic, 1),
        z_in718_left=pct(in718.residual_porosity_dynamic, 2), z_in718_need=gpa(in718.closure_pressure_dynamic, 1),
        z_fe_density=("100" if fe_powder.density_dynamic >= 0.9995 else pct(fe_powder.density_dynamic, 1)), z_fe_status=label(fe_powder.status).replace("_", " "),
        z_tial_density=pct(tial_powder.density_dynamic, 1), z_tial_powder_status=label(tial_powder.status),
    )

    fractions = logspace(-4.0, math.log10(0.3), 40)
    site.figure("pore_collapse", {
        "type": "xy",
        "title": "Small pores are the hardest to crush",
        "subtitle": "Pressure at which an isolated pore in titanium aluminide starts to collapse, against how much porosity there is.",
        "x": axis("Porosity (fraction of volume, log scale)", scale="log", lo=1e-4, hi=0.3, short="Porosity"),
        "y": axis("Pressure to start collapse (GPa)", lo=0, hi=10, unit="GPa", short="Pressure"),
        "series": [line("at pulse rates", fractions, [porosity.plastic_collapse_pressure(strength.dynamic, f) / 1e9 for f in fractions], slot=2),
                   line("loaded slowly", fractions, [porosity.plastic_collapse_pressure(strength.quasi_static, f) / 1e9 for f in fractions], slot=1)],
        "bands": [{"axis": "y", "from": 1.6, "to": 4.0, "label": "concept note"}],
        "rules": [{"axis": "x", "at": 0.005, "label": "typical casting, 0.5 %"}],
        "table": table(["Porosity", "Loaded slowly (GPa)", "At pulse rates (GPa)"],
                       [[f, porosity.plastic_collapse_pressure(strength.quasi_static, f) / 1e9,
                         porosity.plastic_collapse_pressure(strength.dynamic, f) / 1e9] for f in (0.1, 0.03, 0.01, 0.005, 0.001, 0.0001)]),
        "note": "Hollow-sphere collapse, (2/3) σ ln(1/f), with no strain hardening: a lower bound on the pressure.",
    })

    temps = linspace(300.0, 1600.0, 53)
    site.figure("bond_time", {
        "type": "xy",
        "title": "Closing a pore is quick; healing it needs heat",
        "subtitle": "Time for diffusion to bond the faces of a closed pore in titanium aluminide, against temperature.",
        "x": axis("Temperature (°C)", lo=0, hi=1350, unit="°C", short="Temperature"),
        "y": axis("Bonding time (s, log scale)", scale="log", lo=1e-6, hi=1e45, unit="s", short="Bonding time"),
        "series": [line("Diffusion bonding time", [t - ZERO_CELSIUS for t in temps],
                        [porosity.diffusion_bonding_time(tial, t) for t in temps], slot=1)],
        "rules": [{"axis": "y", "at": 1e-4, "label": "a magnetic pulse lasts this long", "side": "left"},
                  {"axis": "y", "at": hip.time, "label": "a hot isostatic pressing hold", "side": "left"}],
        "marks": [{"x": sig(hip.temperature - ZERO_CELSIUS), "y": sig(porosity.diffusion_bonding_time(tial, hip.temperature)),
                   "label": "at pressing temperature", "tone": "state_of_the_art", "anchor": "end"}],
        "table": table(["Temperature (°C)", "Bonding time (s)"],
                       [[t - ZERO_CELSIUS, porosity.diffusion_bonding_time(tial, t)] for t in (293.15, 673.15, 1073.15, 1273.15, hip.temperature)]),
        "note": "Volume diffusion across 0.1 µm with an estimated prefactor; good to an order of magnitude at best.",
    })

    pressures = linspace(0.05e9, 5e9, 60)
    fe = porosity.matrix_strength("iron_armco")
    site.figure("powder_density", {
        "type": "xy",
        "title": "What a pressure pulse does densify: powder",
        "subtitle": "Density of a loose metal powder after one pressure pulse, as a share of the solid.",
        "x": axis("Pressure (GPa)", lo=0, hi=5, unit="GPa", short="Pressure"),
        "y": axis("Density (% of solid)", lo=60, hi=100, unit="%", short="Density"),
        "series": [line("Iron powder", [p_ / 1e9 for p_ in pressures],
                        [100 * porosity.powder_density_at_pressure(p_, fe.dynamic) for p_ in pressures], slot=1),
                   line("Titanium aluminide powder", [p_ / 1e9 for p_ in pressures],
                        [100 * porosity.powder_density_at_pressure(p_, strength.dynamic) for p_ in pressures], slot=2)],
        "bands": [{"axis": "x", "from": 1.6, "to": 4.0, "label": "concept note"}],
        "table": table(["Pressure (GPa)", "Iron (%)", "Titanium aluminide (%)"],
                       [[p_ / 1e9, 100 * porosity.powder_density_at_pressure(p_, fe.dynamic),
                         100 * porosity.powder_density_at_pressure(p_, strength.dynamic)] for p_ in (0.5e9, 1e9, 1.6e9, 2.5e9, 4e9)]),
        "note": "Plastic yielding of particles at pulse-rate strength, from random packing at 64 %. The step near 90 % is where the model changes from touching particles to isolated pores; the two laws do not quite join. The compact is not bonded and still needs sintering.",
    })


# ---------------------------------------------------------------------------
# chapter 9: the inverted magnet
# ---------------------------------------------------------------------------
_WALLS = (("steel_4340", "Alloy steel 4340"), ("maraging_c350", "Maraging steel C350"), ("mp35n", "MP35N"),
          ("carbon_fibre_t1100g", "T1100G carbon fibre"), ("cnt_fibre_record", "Record nanotube fibre"),
          ("cns_engineering_target", "Nanosheet target (hypothetical)"), ("graphene_intrinsic", "Perfect graphene (ceiling)"))


def chapter_containment(site: Site) -> None:
    survey = {a.material: a for a in containment.containment_survey(P_HI, 0.025)}
    rail = containment.rail_launcher_load(4e6, 0.05, 0.05)
    fusion = containment.fusion_magnet_load(20.0, 12.2)
    rail_wall = containment.assess_containment(rail.bore_pressure, "maraging_c350", 0.025)
    fusion_wall = containment.assess_containment(fusion.coil_pressure, "steel_4340", 0.5)
    driver = containment.mass_driver(2400.0, "aluminum_6061", 1.0, 0.05, P_HI)
    leo = containment.mass_driver(7800.0, "aluminum_6061", 1.0, 0.05, P_HI)
    steel = materials.get("maraging_c350")
    site.nums(
        c_required=gpa(survey["steel_4340"].required_monolithic_strength, 1),
        c_required_plain=gpa(containment.required_monolithic_strength(P_HI), 1),
        c_maraging_strength=gpa(steel.yield_strength, 1),
        c_maraging_ratio=fx(survey["maraging_c350"].plastic_ratio, 1),
        c_transverse=gpa(containment.FIBRE_TRANSVERSE_COMPRESSIVE_STRENGTH, 2),
        c_die=gpa(containment.COMPOUND_DEMONSTRATED_PRESSURE, 1),
        c_rail_pressure=gpa(rail.bore_pressure, 1), c_rail_field=fx(rail.effective_field, 0),
        c_rail_force=fx(rail.force / 1e6, 0), c_rail_ratio=fx(rail_wall.plastic_ratio, 2),
        c_rail_status=label(rail_wall.status).replace("_", " "),
        c_fusion_coil=mpa(fusion.coil_pressure), c_fusion_plasma=mpa(fusion.plasma_pressure, 1),
        c_fusion_ratio=fx(fusion.ratio, 0), c_fusion_wall=fx(fusion_wall.elastic_ratio, 2),
        c_fusion_status=label(fusion_wall.status).replace("_", " "),
        c_driver_pressure=mpa(driver.allowed_pressure), c_driver_length=fx(driver.launcher_length, 0),
        c_driver_g=fx(driver.g_load, 0), c_driver_field=fx(driver.bore_field, 0),
        c_driver_over=fx(P_HI / driver.payload_strength, 0),
        c_leo_length=fx(leo.launcher_length, 0),
        c_transit_us=_us(survey["steel_4340"].hoop_transit_time, 0),
    )
    site.data["containment"] = [
        {"key": key, "name": name, "status": survey[key].status.value, "route": survey[key].route.replace("_", " "),
         "strength": gpa(survey[key].strength, 1),
         "elastic": fx(survey[key].elastic_ratio, 2) if math.isfinite(survey[key].elastic_ratio) else None,
         "compound": fx(survey[key].plastic_ratio, 1) if survey[key].plastic_ratio < 1e3 else None,
         "transverse": survey[key].transverse_compression_ok}
        for key, name in _WALLS]

    pressures = linspace(0.1e9, 5e9, 50)

    def capped(value: float) -> float | None:
        return value if math.isfinite(value) and value <= 12.0 else None

    sf = 1.5
    site.figure("wall_ratio", {
        "type": "xy",
        "title": "How thick a steel wall must be",
        "subtitle": "Outer radius over bore radius for a wall that contains the pressure with a 1.5 safety factor.",
        "x": axis("Contained pressure (GPa)", lo=0, hi=5, unit="GPa", short="Pressure"),
        "y": axis("Outer radius / bore radius", lo=1, hi=12, short="Wall ratio"),
        "series": [
            line("Alloy steel, one piece", [p_ / 1e9 for p_ in pressures],
                 [capped(containment.required_radius_ratio(p_, materials.get("steel_4340").yield_strength / sf)) for p_ in pressures], slot=1),
            line("Maraging steel, one piece", [p_ / 1e9 for p_ in pressures],
                 [capped(containment.required_radius_ratio(p_, steel.yield_strength / sf)) for p_ in pressures], slot=2),
            line("Maraging steel, layered and prestressed", [p_ / 1e9 for p_ in pressures],
                 [capped(containment.autofrettage_ratio(p_, steel.yield_strength, sf)) for p_ in pressures], slot=3),
        ],
        "bands": [{"axis": "x", "from": 1.6, "to": 4.0, "label": "concept note"}],
        "rules": [{"axis": "y", "at": 3.0, "label": "practical limit of a one-piece wall", "side": "left"},
                  {"axis": "y", "at": 10.0, "label": "limit of layered construction", "side": "left"}],
        "table": table(["Pressure (GPa)", "Alloy steel, one piece", "Maraging, one piece", "Maraging, layered"],
                       [[p_ / 1e9, capped(containment.required_radius_ratio(p_, materials.get("steel_4340").yield_strength / sf)),
                         capped(containment.required_radius_ratio(p_, steel.yield_strength / sf)),
                         capped(containment.autofrettage_ratio(p_, steel.yield_strength, sf))]
                        for p_ in (0.16e9, 0.5e9, 0.9e9, 1.6e9, 2.5e9, 4e9)]),
        "note": "A one-piece wall has no solution once the pressure passes its strength divided by 1.73: the curve ends there.",
    })

    velocities = logspace(math.log10(500.0), 4.0, 40)
    payload = materials.get("aluminum_6061")
    limit = payload.yield_strength / 1.5

    def length(v: float, p_: float) -> float:
        return v * v * payload.density * 1.0 / (2.0 * p_)

    site.figure("launcher_length", {
        "type": "xy",
        "title": "A launcher is as long as its payload is weak",
        "subtitle": "Length of a mass driver for a one-metre aluminium payload. The payload's strength sets the push, not the bore's.",
        "x": axis("Exit speed (m/s, log scale)", scale="log", lo=500, hi=10000, unit="m/s", short="Exit speed"),
        "y": axis("Launcher length (m, log scale)", scale="log", unit="m", short="Length"),
        "series": [line("limited by the payload", velocities, [length(v, limit) for v in velocities], slot=1),
                   line("if the payload could take 4 GPa", velocities, [length(v, P_HI) for v in velocities], slot=0, thin=True)],
        "marks": [{"x": 2400.0, "y": sig(length(2400.0, limit)), "label": "lunar escape", "tone": "ink"},
                  {"x": 7800.0, "y": sig(length(7800.0, limit)), "label": "orbital speed", "tone": "ink", "anchor": "end"}],
        "table": table(["Exit speed (m/s)", "Payload-limited length (m)", "At 4 GPa (m)"],
                       [[v, length(v, limit), length(v, P_HI)] for v in (1000.0, 2400.0, 5000.0, 7800.0, 10000.0)]),
        "note": f"Base pressure capped at {limit / 1e6:.0f} MPa (aluminium 6061 yield over 1.5). No atmosphere, no sabot.",
    })


# ---------------------------------------------------------------------------
# verdict, ledger and home page
# ---------------------------------------------------------------------------
_VERDICT_ORDER = ("state_of_the_art", "extrapolation", "aspirational", "infeasible", "contradicted", "not_modelled")
_VERDICT_MEANING = {
    "state_of_the_art": "True of today's technology, under the conditions stated.",
    "extrapolation": "Beyond what has been shown, with no known barrier: a matter of scaling.",
    "aspirational": "Needs a material or technology that does not exist.",
    "infeasible": "Conflicts with physics as modelled.",
    "contradicted": "A statement of fact that the numbers show to be false.",
    "not_modelled": "Outside what the package computes.",
}


def chapter_ledger(site: Site) -> None:
    results = claims.evaluate_all()
    counts = claims.summary(results)
    by_id = {}
    for r in results:
        by_id[r.id] = {"id": r.id, "section": r.section_title, "section_key": r.section, "quote": r.quote,
                       "topic": r.topic, "kind": r.kind, "verdict": r.verdict.value, "finding": r.finding,
                       "evidence": list(r.evidence), "models": list(r.models), "chapter": r.chapter}
    sections = [{"key": key, "title": title, "ids": [r.id for r in results if r.section == key]}
                for key, title in claims.SECTIONS]
    total = len(results)
    site.data["claims"] = {
        "by_id": by_id, "order": [r.id for r in results], "sections": sections, "total": total,
        "counts": [{"verdict": v, "count": counts[v], "share": round(100.0 * counts[v] / total, 2),
                    "meaning": _VERDICT_MEANING[v]} for v in _VERDICT_ORDER],
    }
    site.nums(
        l_total=str(total), l_claims=str(sum(r.kind == "claim" for r in results)),
        l_questions=str(sum(r.kind == "question" for r in results)),
        l_sota=str(counts["state_of_the_art"]), l_extrapolation=str(counts["extrapolation"]),
        l_aspirational=str(counts["aspirational"]), l_infeasible=str(counts["infeasible"]),
        l_contradicted=str(counts["contradicted"]), l_not_modelled=str(counts["not_modelled"]),
        l_false=str(counts["infeasible"] + counts["contradicted"]),
        l_holds=str(counts["state_of_the_art"] + counts["extrapolation"]),
    )



# ---------------------------------------------------------------------------
# chapter 11: the magnetar mode (a thought experiment; the source is assumed)
# ---------------------------------------------------------------------------
_APPLICATION_LABELS = {"forging": "Forging", "refractory_forming": "Refractory metals", "welding": "Welding",
                       "synthesis": "Synthesis", "densification": "Densification", "containment": "Containment"}
_TIER_LABELS = {"solid": "solid metal", "liquid_skin": "molten or boiling skin", "megabar": "megabar compression",
                "plasma": "dense plasma", "landau": "needle atoms", "degenerate": "white-dwarf matter"}
_STANDOFF_LABELS = {
    "earth_field": "Field falls to the Earth's own", "pacemaker_line": "Pacemaker line (0.5 mT)",
    "public_limit": "Public exposure limit (0.4 T)", "occupational_limit": "Occupational limit (2 T)",
    "limb_limit": "Limb exposure limit (8 T)", "atomic_field": "Chemistry stops (235,000 T)",
}


def _field_text(field: float) -> str:
    if field < 1e4:
        return f"{float(f'{field:.3g}'):,.0f} T" if field >= 100 else f"{field:.3g} T"
    exponent = math.floor(math.log10(field))
    mantissa = field / 10 ** exponent
    sup = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return (f"{mantissa:.2g} × " if abs(mantissa - 1.0) > 0.05 else "") + f"10{sup} T"


def _size_text(volume: float | None) -> str:
    if volume is None:
        return ""
    side = volume ** (1.0 / 3.0)
    for scale, unit in ((1e3, "km"), (1.0, "m"), (1e-3, "mm"), (1e-6, "µm"), (1e-9, "nm"), (1e-15, "fm")):
        if side >= scale:
            return f"{side / scale:.2g} {unit} across"
    return f"{side:.1g} m across"


def _duration_text(duration: float | None) -> str:
    if duration is None:
        return "steady"
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns"), (1e-12, "ps"), (1e-15, "fs"), (1e-24, "ys")):
        if duration >= scale:
            return f"{duration / scale:.2g} {unit}"
    return f"{duration:.1g} s"


_LADDER_LABELS = {
    "secuf_35p6t_all_sc": "All-superconducting record", "chmfl_hybrid_45t": "Strongest steady magnet",
    "emf_industrial": "Industrial forming coil", "lanl_100t_nondestructive": "Pulsed record, coil survives",
    "single_turn_300t": "Single-turn coil", "emfc_1200t": "Electromagnetic flux compression",
    "explosive_fc_2800t": "Explosive flux compression", "z_machine_surface": "Z machine, at 1 mm from the current",
    "laser_plasma_wagner": "Laser-produced plasma", "magnetic_white_dwarf": "Magnetic white dwarf",
    "laser_optical_field": "Focus of the most intense laser", "radio_pulsar": "Radio pulsar",
    "magnetar_typical": "Magnetar", "magnetar_upper": "Strongest magnetars", "heavy_ion_collision": "Heavy-ion collision",
}


def chapter_magnetar(site: Site) -> None:
    mg = magnetar
    b = 1e10
    cm, mm, um = mg.MagnetarSource(b, 1e-2), mg.MagnetarSource(b, 1e-3), mg.MagnetarSource(b, 1e-6)
    a = mg.assess_magnetar_mode(b, 1e-2)
    ratios = mg.energy_ratios(cm.energy())
    fe, cu = mg.electron_capture_threshold("56Fe"), mg.electron_capture_threshold("63Cu")
    standoff = cm.standoff()
    shape = mg.atom_shape(b)
    ev = 1.602176634e-19
    ns_mass, ns_radius = mg.CANONICAL_NEUTRON_STAR
    r_hi, r_lo = mg.standoff_radius(b, 1e-2, B_HI), mg.standoff_radius(b, 1e-2, B_LO)
    press_energy = REF.stored_energy(B_HI)
    forming = eddy.pulse_frequency_recommendation("copper_ofhc", 2e-3).quarter_period
    site.nums(
        x_atomic=sg(mg.ATOMIC_FIELD, 3), x_qed=sg(mg.QED_CRITICAL_FIELD, 3),
        x_pressure=sg(magnetic_pressure(b), 2), x_gap_pressure=sg(a.gaps["pressure_over_concept_note"], 1),
        x_mass_density=sg(mg.mass_equivalent_density(b), 2),
        x_lead_ratio=sg(mg.mass_equivalent_density(b) / 11340.0, 1),
        x_mm3_mt=fx(mg.tnt_megatons(mg.field_energy(b, 1e-9)), 1), x_um3_mj=fx(mg.field_energy(b, 1e-18) / 1e6, 0),
        x_cm_mt=fx(cm.megatons(), 0), x_cm_world=pct(ratios["world_annual_energy"]), x_cm_mass=fx(cm.mass_equivalent(), 0),
        x_mm_mt=fx(mm.megatons(), 0), x_um_mj=fx(um.energy() / 1e6, 0),
        x_confinement_ps=fx(cm.confinement_time() * 1e12, 0), x_power=sg(cm.assembly_power(), 2),
        x_um_power=sg(um.assembly_power(), 2),
        x_gap_field=sg(a.gaps["field_over_nondestructive_record"], 1),
        x_gap_destructive=sg(a.gaps["field_over_destructive_record"], 2),
        x_gap_energy=sg(a.gaps["energy_over_largest_pulsed_store"], 1),
        x_gap_power=sg(a.gaps["power_over_pulsed_record"], 1),
        x_virial=sg(cm.virial_mass("carbon_fibre_t1100g"), 2), x_virial_um=fx(um.virial_mass("carbon_fibre_t1100g"), 0),
        x_cf_field=sg(mg.chandrasekhar_fermi_field(ns_mass, ns_radius), 2),
        x_capture_fe=sg(fe.field, 3), x_capture_fe_density=sg(fe.density, 3), x_capture_cu=sg(cu.field, 3),
        x_piston_density=sg(mg.piston_density(b, 56 / 26, 26), 2),
        x_piston_density_1e8=sg(mg.piston_density(1e8, 56 / 26, 26), 2),
        x_condensed=sg(mg.condensed_density(b, 56, 26), 2),
        x_h_binding=fx(mg.hydrogen_binding_energy(b) / ev, 0), x_aspect=fx(shape.aspect_ratio, 0),
        x_cyclotron=fx(mg.cyclotron_energy(b) / ev / 1e6, 2), x_synchrotron=sg(mg.synchrotron_cooling_time(b), 1),
        x_magnetic_length=sg(mg.magnetic_length(b), 2),
        x_r_earth=fx(standoff["earth_field"], 0), x_r_pacemaker=fx(standoff["pacemaker_line"], 0),
        x_r_public=fx(standoff["public_limit"], 0), x_r_limb=fx(standoff["limb_limit"], 0),
        x_r_atomic=fx(standoff["atomic_field"] * 100, 0),
        x_r_100t=fx(mg.standoff_radius(b, 1e-2, B_HI), 1),
        x_levitation=fx(mg.levitation_radius(b, 1e-2), 1),
        x_damage_km=fx(mg.fluence_radius(cm.energy()) / 1e3, 0),
        x_magnetar_km=fx(mg.standoff_radius(1e11, 1e4, mg.ATOMIC_FIELD) / 1e3, 0),
        x_boil_cu=fx(mg.skin_boiling_field("copper_ofhc"), 0), x_vapour_cu=fx(mg.skin_vaporisation_field("copper_ofhc"), 0),
        x_bulk_cu=fx(mg.bulk_modulus_field("copper_ofhc"), 0),
        x_ionisation=sg(mg.PRESSURE_IONISATION_FIELD, 2),
        x_solar_years=fx(mg.solar_collection_time(mg.field_energy(b, 1e-9)) / 3.156e7, 2),
        x_lab_cm=fx(mg.laboratory_record(1e-6).field, 0),
        x_destructive=fx(mg.DESTRUCTIVE_FIELD_RECORD, 0),
        # the working distance: where the dipole field of the source has fallen into the note's window
        x_r_63=fx(r_lo, 1), x_shell_cm=fx((r_lo - r_hi) * 100, 0),
        x_part_variation=pct(6.0 * 0.1 / r_hi),
        x_energy_over_press=sg(cm.energy() / press_energy, 1),
        x_press_mj=fx(press_energy / 1e6, 0),
        x_power_suns=sg(cm.assembly_power() / SOLAR_LUMINOSITY, 1),
        x_electron_rest=fx(0.51099895, 3),
        x_iron_ratio=sg(mg.piston_density(b, 56 / 26, 26) / 7870.0, 1),
        x_matter_over_field=sg(mg.piston_density(b, 56 / 26, 26) / mg.mass_equivalent_density(b), 1),
        # a forming pulse (the rise that keeps the field out of a 2 mm copper wall) against the light-crossing time
        x_forming_us=_us(forming, 0), x_hold_ratio=sg(forming / cm.confinement_time(), 1),
        x_flash_impulse=fx(P_HI * cm.confinement_time(), 2),
        x_flash_speed_mm=fx(1e3 * P_HI * cm.confinement_time() / (materials.get("copper_ofhc").density * 1e-3), 0),
    )

    fields_shown = (300.0, 3000.0, 1e6, 1e10)
    regimes = [mg.classify_regime("iron_armco", f) for f in fields_shown]
    outcomes = [mg.process_outcomes("iron_armco", f) for f in fields_shown]
    site.data["magnetar"] = {
        "premise": a.premise,
        "columns": [{"field": _field_text(f), "tier": _TIER_LABELS[r.tier]} for f, r in zip(fields_shown, regimes)],
        "processes": [{"name": _APPLICATION_LABELS[key], "cells": [o[key] for o in outcomes]} for key in mg.APPLICATIONS],
        "thresholds": [{"label": t.label, "iron": _field_text(t.field),
                        "copper": _field_text(next(c.field for c in mg.regime_thresholds("copper_ofhc") if c.key == t.key)),
                        "state": t.state, "kind": t.kind}
                       for t in sorted(mg.regime_thresholds("iron_armco"), key=lambda t: t.field)],
        "standoff": [{"label": _STANDOFF_LABELS[k], "metres": fx(v, 1) if v < 100 else fx(v, 0)} for k, v in standoff.items()],
        "gaps": [{"label": k.replace("_", " "), "value": plain(sg(v, 2))} for k, v in a.gaps.items()],
    }

    rows = []
    for r in mg.EXTREME_FIELD_LADDER:
        if r.key not in _LADDER_LABELS:
            continue
        natural = r.origin == "astrophysical"
        rows.append({"label": _LADDER_LABELS[r.key], "value": sig(r.field), "text": _field_text(r.field),
                     "tone": "accent" if natural else ("muted" if r.destructive else "ink"),
                     "detail": f"{_size_text(r.volume)}; {_duration_text(r.duration)}. {r.citation}"})
    site.figure("x_ladder", {
        "type": "dot",
        "title": "From the laboratory to a magnetar",
        "subtitle": "Bold dots: made by people, and the source survives. Faint dots: made by people, and the source is destroyed. Blue: stars. Hover for the size and duration of each.",
        "x": axis("Field (T, log scale)", scale="log", lo=10, hi=1e15, short="Field",
                  ticks=[{"v": 10.0 ** e, "label": plain(_field_text(10.0 ** e))} for e in (1, 3, 5, 7, 9, 11, 13, 15)]),
        "rows": rows,
        "bands": [CONCEPT_BAND_T],
        "rules": [{"axis": "x", "at": sig(mg.ATOMIC_FIELD), "label": "atoms deform"},
                  {"axis": "x", "at": sig(mg.QED_CRITICAL_FIELD), "label": "QED critical field"}],
        "table": table(["Source", "Field (T)", "Size", "Lasts", "Reference"],
                       [[_LADDER_LABELS[r.key], r.field, _size_text(r.volume), _duration_text(r.duration), r.citation]
                        for r in mg.EXTREME_FIELD_LADDER if r.key in _LADDER_LABELS]),
        "note": "Sizes and durations of the laboratory entries are order-of-magnitude estimates.",
    })

    fields = logspace(6.5, 11.6, 52)
    densities = [mg.piston_density(f, 56 / 26, 26) for f in fields]
    site.figure("x_density", {
        "type": "xy",
        "title": "What a magnetar-class field would squeeze iron into",
        "subtitle": "Density of cold iron held by the field's pressure, and the mass-equivalent density of the field itself.",
        "x": axis("Field (T, log scale)", scale="log", lo=3e6, hi=4e11, short="Field",
                  ticks=[{"v": 10.0 ** e, "label": plain(_field_text(10.0 ** e))} for e in (7, 8, 9, 10, 11)]),
        "y": axis("Density (kg/m³, log scale)", scale="log", lo=1e3, hi=1e15, unit="kg/m³", short="Density"),
        "series": [line("Iron under the field's pressure", fields, densities, slot=1),
                   line("The field's own mass-equivalent", fields, [mg.mass_equivalent_density(f) for f in fields], slot=2)],
        "rules": [{"axis": "y", "at": 7870.0, "label": "iron on the bench", "side": "left"},
                  {"axis": "x", "at": sig(fe.field), "label": "iron starts capturing electrons"}],
        "marks": [{"x": 1e10, "y": sig(mg.piston_density(1e10, 56 / 26, 26)), "label": "magnetar field", "tone": "ink", "anchor": "end"}],
        "table": table(["Field (T)", "Iron density (kg/m³)", "Field mass-equivalent (kg/m³)", "Pressure (Pa)"],
                       [[f, mg.piston_density(f, 56 / 26, 26), mg.mass_equivalent_density(f), magnetic_pressure(f)]
                        for f in (1e7, 1e8, 1e9, 1e10, 1e11)]),
        "note": "Chandrasekhar's zero-temperature equation of state with the ion-lattice correction. Valid above about 2.5 million tesla.",
    })

    radii = logspace(-7.0, -1.0, 37)
    megaton = 4.184e15
    site.figure("x_energy", {
        "type": "xy",
        "title": "Energy to fill a region with the field",
        "subtitle": "Field energy of a source of a given radius. At magnetar strength a pinhead holds megatons.",
        "x": axis("Radius of the source (m, log scale)", scale="log", lo=1e-7, hi=1e-1, short="Radius",
                  ticks=[{"v": 1e-7, "label": "0.1 µm"}, {"v": 1e-6, "label": "1 µm"}, {"v": 1e-5, "label": "10 µm"},
                         {"v": 1e-4, "label": "0.1 mm"}, {"v": 1e-3, "label": "1 mm"}, {"v": 1e-2, "label": "1 cm"},
                         {"v": 1e-1, "label": "10 cm"}]),
        "y": axis("Energy (J, log scale)", scale="log", lo=1e-3, hi=1e24, unit="J", short="Energy"),
        "series": [line("10¹⁰ T (magnetar)", radii, [mg.MagnetarSource(1e10, r).energy() for r in radii], slot=1),
                   line("10⁸ T (pulsar)", radii, [mg.MagnetarSource(1e8, r).energy() for r in radii], slot=2),
                   line("10⁶ T", radii, [mg.MagnetarSource(1e6, r).energy() for r in radii], slot=3)],
        "rules": [{"axis": "y", "at": benchmarks.ENERGY_ANY_STORE_RECORD, "label": "largest pulsed store on Earth", "side": "left"},
                  {"axis": "y", "at": megaton, "label": "one megaton", "side": "left"},
                  {"axis": "y", "at": 6e20, "label": "a year of world energy use", "side": "left"}],
        "table": table(["Radius (m)", "at 1e10 T (J)", "at 1e8 T (J)", "at 1e6 T (J)"],
                       [[r, mg.MagnetarSource(1e10, r).energy(), mg.MagnetarSource(1e8, r).energy(),
                         mg.MagnetarSource(1e6, r).energy()] for r in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]),
        "note": "A uniformly magnetised sphere: interior field energy plus half as much again outside.",
    })

    stand_rows = [{"label": _STANDOFF_LABELS[k], "value": sig(v), "text": (f"{v:.0f} m" if v >= 10 else f"{v:.2g} m"),
                   "tone": "ink", "detail": next(t.description for t in mg.FIELD_THRESHOLDS if t.key == k)}
                  for k, v in sorted(standoff.items(), key=lambda kv: kv[1])]
    stand_rows.append({"label": "Water pushed harder than Earth gravity holds it", "value": sig(mg.levitation_radius(b, 1e-2)),
                       "text": f"{mg.levitation_radius(b, 1e-2):.1f} m", "tone": "ink",
                       "detail": "Diamagnetic force on water exceeds its weight on Earth inside this radius"})
    stand_rows.append({"label": "Radiant damage if the shot's energy escapes", "value": sig(mg.fluence_radius(cm.energy())),
                       "text": f"{mg.fluence_radius(cm.energy()) / 1e3:,.0f} km", "tone": "infeasible",
                       "detail": "Distance at which the released field energy falls to 1 MJ per square metre (estimated damage level)"})
    stand_rows.sort(key=lambda r: r["value"])
    site.figure("x_standoff", {
        "type": "dot",
        "title": "Why it would have to be in space",
        "subtitle": "Keep-out distances around a one-centimetre source at 10¹⁰ T. The field alone clears half a kilometre; the energy clears a continent.",
        "x": axis("Distance from the source (m, log scale)", scale="log", lo=0.1, hi=1e7, short="Distance",
                  ticks=[{"v": 0.1, "label": "10 cm"}, {"v": 1.0, "label": "1 m"}, {"v": 10.0, "label": "10 m"},
                         {"v": 100.0, "label": "100 m"}, {"v": 1e3, "label": "1 km"}, {"v": 1e4, "label": "10 km"},
                         {"v": 1e5, "label": "100 km"}, {"v": 1e6, "label": "1,000 km"}, {"v": 1e7, "label": "10,000 km"}]),
        "rows": stand_rows,
        "table": table(["Threshold", "Distance (m)"], [[r["label"], r["value"]] for r in stand_rows]),
        "note": "Dipole field falling as the cube of distance along the axis.",
    })


CHAPTERS = [chapter_pressure, chapter_machine, chapter_power, chapter_forging, chapter_refractory, chapter_welding,
            chapter_synthesis, chapter_porosity, chapter_containment, chapter_magnetar, chapter_ledger]
