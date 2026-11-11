#!/usr/bin/env python3
"""Reference section of the documentation site: materials, benchmarks, references, methods.

``chapter_reference(site)`` is registered in ``tools/build_docs.py``.  It fills

    site.data["materials"]    the registry as table rows (display units, rounded)
    site.data["benchmarks"]   every cited ladder, the named thresholds and the status bands
    site.data["references"]   every module's REFERENCES plus a de-duplicated bibliography
    site.data["methods"]      one record per model module: purpose, relations, limits, entry points

and registers the chart specs ``ref_field_ladder`` and ``ref_energy_ladder``.

Almost everything is read from the package by introspection (docstrings, ``__all__``, the
attribute docstrings of module constants, the status helpers probed for their thresholds), so
the reference pages cannot drift from the code.  The only hand-written text is in the
dictionaries at the top of this file.  Standard library only, deterministic output.

Run it on its own for a smoke test::

    python tools/docs_reference.py            # print a summary of what would be written
    python tools/docs_reference.py --write    # also write the four JSON files to docs/_data
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import importlib
import inspect
import json
import math
import re
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run from a checkout without installing
    sys.path.insert(0, str(ROOT))

import magform  # noqa: E402
from magform import benchmarks, materials  # noqa: E402
from magform.magnetics import Solenoid, field_for_pressure  # noqa: E402
from magform import status as status_module  # noqa: E402
from magform.status import Confidence, Status, Verdict  # noqa: E402

# ---------------------------------------------------------------------------
# hand-written text (everything else on the reference pages is introspected)
# ---------------------------------------------------------------------------
METHODS: dict[str, dict[str, Any]] = {
    # module -> answers (one sentence), relations (3-6 plain-text relations), limits (validity),
    #           main (entry points to feature; names are checked against the module), chapter (site URL)
    "magnetics": {
        "answers": "What field a pressure needs, and what that field does to the coil that makes it.",
        "relations": [
            "Magnetic pressure: P = B²/(2μ₀); with field on both sides of a wall, "
            "P = (B_out² − B_in²)/(2μ₀)",
            "Central field of a uniform-current thick solenoid (Montgomery): B₀ = μ₀ J a₁ β "
            "ln[(α + √(α² + β²)) / (1 + √(1 + β²))], α = a₂/a₁, β = L/(2a₁)",
            "Stored energy of a long coil: E = (B₀²/2μ₀) π L [a₁² + 2Δ(a₂/3 − Δ/4)]",
            "Hoop stress of a free turn: σ(r) = r J B(r); its maximum never falls below P/2, however thick the winding",
            "Virial bound on structural mass: M ≥ ρE/σ",
        ],
        "limits": [
            "Non-magnetic media. The on-axis field is exact for a uniform current density.",
            "The field profile in the winding, the stored energy and the stresses use the long-coil "
            "approximation, good to about 10 % for β ≥ 3.",
            "Each turn carries its own Lorentz load (no radial load sharing), which is an upper bound on "
            "the peak stress of a bonded winding.",
        ],
        "main": ["magnetic_pressure", "field_for_pressure", "pressure_with_background", "max_hoop_stress_ratio",
                 "virial_minimum_mass", "Solenoid"],
        "chapter": "/story/pressure.html",
    },
    "fieldmap": {
        "answers": "What the field is away from the axis, and how the pressure is really spread over a workpiece.",
        "relations": [
            "Field of a circular loop from the complete elliptic integrals K and E (arithmetic-geometric mean)",
            "Winding field: B(r, z) = J ∬ b_loop dA over the winding cross-section, with a mesh that grades "
            "towards the field point",
            "Radial field, reduced exactly to one dimension: B_r = J ∫ [a_φ(a; r, z − L/2) − "
            "a_φ(a; r, z + L/2)] da",
            "Perfectly conducting workpiece: ψ_coil + Σ G_ij I_j = 0 on every surface panel; then "
            "B_surface = μ₀|K| and P = B²/(2μ₀)",
        ],
        "limits": [
            "Axisymmetric, non-magnetic and quasi-static, with a uniform winding current density.",
            "The workpiece is a perfect conductor that was field-free before the pulse: skin depth much "
            "smaller than the part and the gap, pulse much shorter than the flux-diffusion time.",
            "A sharp edge gives a mesh-dependent pressure peak. Give real parts their corner radius.",
            "Checked against two closed forms: Montgomery's on-axis field, and the 3/2 B₀ equatorial "
            "field of a conducting sphere in a uniform field.",
        ],
        "main": ["solenoid_field", "end_field", "surface_pressure", "workpiece_pressure", "cylinder_profile"],
        "chapter": "/story/forging.html",
    },
    "conductor": {
        "answers": "Whether a REBCO winding can carry the current, hold the load and survive being pulsed.",
        "relations": [
            "Engineering critical current density: J_e = J_e,sf g(T) (1 + B/B₀)^(−α) max(0, 1 − B/B_irr(T))",
            "Iso-strain winding composite: E_c = Σ f_i E_i, allowed stress = E_c ε_lim with ε_lim = 0.4 % (REBCO)",
            "Required modulus: E ≥ SF σ/ε_lim, which is 1 TPa at 4 GPa and 0.4 %",
            "Loss per pulse: critical-state hysteresis (thin strip and Bean slab) plus copper eddy loss "
            "(dB/dt)² d² 2t/(12ρ)",
            "Quench test: loss per pulse against the enthalpy from the operating temperature to the "
            "current-sharing temperature",
        ],
        "limits": [
            "J_e is a parametric fit representative of 2020s commercial tape, calibrated at 4.2 K on data "
            "to 31 T. Above 31 T it is an extrapolation.",
            "The rule of mixtures assumes continuous constituents aligned with the hoop direction and no "
            "thermal pre-strain.",
            "Pulse losses are adiabatic and leave out the transport current, so they are lower estimates.",
        ],
        "main": ["REBCOTape", "CNSComposite", "assess_conductor", "assess_structure", "assess_pulsed_operation",
                 "tape_pulse_losses", "enthalpy_margin"],
        "chapter": "/story/machine.html",
    },
    "pulse": {
        "answers": "What power supply the pulse needs: voltage, current, power, bank size and coil heating.",
        "relations": [
            "Quarter-period law: ω₀ = π/(2 t_rise), V₀ I_pk = πE/t_rise, peak power = πE/(2 t_rise)",
            "Turns trade current against voltage: L ∝ N², I ∝ 1/N, V₀ ∝ N",
            "Series RLC discharge in all three damping regimes; the bank must hold E_field exp(R t_rise/L)",
            "A homopolar rotor is a capacitor: C_eq = J/k² with k = V/ω",
            "Conductor heating from the action integral ∫J² dt, starting at 77 K",
            "Energy transfer between uncoupled inductors: at most 25 %",
        ],
        "limits": [
            "A lumped circuit: no transmission-line effects, switch physics or coupling to the workpiece.",
            "Coil stresses are handled by magnetics, conductor and containment.",
            "Heating uses a linear resistivity law and a constant specific heat without magnetoresistance, "
            "so it is a lower estimate.",
        ],
        "main": ["assess_pulse_network", "size_capacitor_bank", "rlc_discharge", "coil_electrical",
                 "conductor_heating", "HomopolarGenerator", "source_comparison"],
        "chapter": "/story/power.html",
    },
    "eddy": {
        "answers": "How a pulsed field loads, heats and moves a metal workpiece.",
        "relations": [
            "Skin depth: δ = √(ρ/(π f μ₀))",
            "Interior field of a shell: dB_in/dt = (B_out − B_in)/τ, τ = μ₀ a h/(2ρ); "
            "pressure P = (B_out² − B_in²)/(2μ₀)",
            "Skin heating: ΔT = κ B²/(2μ₀ ρ_m c_p), whatever the pulse length "
            "(κ = 1 is the lower estimate, up to 2.4 at the surface)",
            "Magnetic diffusion into a heating slab: ∂B/∂t = ∂/∂z[(ρ(T)/μ₀) ∂B/∂z], Crank-Nicolson",
            "Rigid-plastic tube: m dv/dt = −(P_net − k σ h/r), with the trapped flux compressed as B_in r² = const",
        ],
        "limits": [
            "A solid, non-magnetic (saturated) workpiece: a long thin shell in a uniform axial field.",
            "The lumped shell model assumes the current is uniform through the wall. For a wall thicker "
            "than the skin depth the skin estimate applies to the surface.",
            "Adiabatic. Solid-phase properties are extrapolated above the melting point and latent heat is ignored.",
            "The tube model has no elastic rebound and no buckling.",
        ],
        "main": ["skin_depth", "skin_temperature_rise", "skin_melting_field", "slab_field_diffusion",
                 "pulse_frequency_recommendation", "TubeCompression"],
        "chapter": "/story/forging.html",
    },
    "plasticity": {
        "answers": "Whether a given pressure can deform a given metal, and what limits the deformation.",
        "relations": [
            "Johnson-Cook: σ = (A + B εⁿ)(1 + C ln(ε̇/ε̇₀))(1 − T*ᵐ)",
            "Zerilli-Armstrong (bcc): σ = c₀ + c₁ exp(−c₃T + c₄T ln ε̇) + c₅ εⁿ",
            "Forging pressure: p = Q σ (1 + μ d/(3h)), with Q from 1 (uniaxial) to about 3 (indentation)",
            "Adiabatic heating: dT = f β σ dε/(ρ c_p), β = 0.9",
            "Culver instability: dσ/dε = 0 along the adiabatic path",
        ],
        "limits": [
            "Each published fit holds only inside its recorded strain-rate and temperature window. The "
            "assessment says when it is extrapolating.",
            "Materials without a published fit use a Hollomon power law from tensile data with an assumed "
            "rate sensitivity.",
            "Bulk, uniform deformation. Hydrostatic pressure alone changes no shape, so a die, a mandrel "
            "or a free surface is assumed.",
            "The recorded ductile-brittle transition temperatures are quasi-static values.",
        ],
        "main": ["flow_stress", "flow_stress_band", "forging_pressure", "adiabatic_temperature_rise",
                 "adiabatic_shear_critical_strain", "warm_forming_temperature", "forging_practice_gap",
                 "assess_forgeability"],
        "chapter": "/story/refractory.html",
    },
    "microstructure": {
        "answers": "What a cold, fast forming pulse leaves behind in the grains.",
        "relations": [
            "Taylor hardening: Δσ = M α G b (√ρ − √ρ₀)",
            "Dislocation cell size: d = K/√ρ, K about 10",
            "Hall-Petch: σ_y = σ₀ + k d^(−1/2)",
            "Grain refinement with strain: d = d_sat + (d₀ − d_sat) exp(−ε)",
            "Recrystallisation (JMAK, n = 2): X = 1 − exp[−ln 20 (t/t₉₅)²], "
            "t₉₅ = 1 h × exp[(Q/R)(1/T − 1/(0.4 T_m))]",
            "Skin exposure: ΔT = κ B²/(2μ₀ ρ c_p), held for about (δ/2)²/α_thermal",
        ],
        "limits": [
            "Scaling laws with estimated constants (Taylor α 0.2 to 0.5, cell constant 10 to 20, "
            "recrystallisation anchor 0.3 to 0.5 T_m). Results are order-of-magnitude estimates.",
            "Fracture toughness and fatigue life are not modelled.",
            "Twinning, texture, recovery, grain growth and phase transformations are not modelled.",
        ],
        "main": ["dislocation_density", "dislocation_cell_size", "hall_petch_strength", "grain_size_after_strain",
                 "grain_refinement_strain", "recrystallised_fraction", "skin_thermal_exposure",
                 "assess_microstructure"],
        "chapter": "/story/refractory.html",
    },
    "shock": {
        "answers": "How fast a magnetically driven flyer goes and what pressure its impact makes.",
        "relations": [
            "Linear Hugoniot: U_s = c₀ + s u_p, with the momentum jump P = ρ₀ U_s u_p",
            "Impedance match at the contact: ρ_f (c_f + s_f (v − u))(v − u) = ρ_t (c_t + s_t u) u",
            "Flyer velocity across a gap: v = √(2 η P g/(ρ h))",
            "Pressure pulse at the contact: t ≈ 2h/U_s",
        ],
        "limits": [
            "One-dimensional planar impact. Material strength and the elastic precursor are neglected.",
            "Linear fits are used inside their recorded pressure range. A phase transition is a low branch, "
            "a mixed-phase plateau and a high branch.",
            "The drive pressure is constant during the flight, so flyer velocities are upper bounds.",
        ],
        "main": ["impedance_match", "flyer_velocity", "flyer_transit_time", "required_pressure_for_velocity",
                 "shock_duration"],
        "chapter": "/story/welding.html",
    },
    "welding": {
        "answers": "Which impact conditions weld two metals, and what drive pressure reaches them.",
        "relations": [
            "Collision point: V_c = V_p/sin β",
            "Lower (bonding) limit: V_p,min = k √(HV/ρ), k from 0.6 to 1.2",
            "Jetting limit: V_c below the bulk sound speed of both metals",
            "Upper (melting) limit, Wittman: V_p,max = (1/N) √(T_m c₀) (κ c_p c₀/(ρ h))^(1/4) / V_c",
            "Wavy interface above a collision Reynolds number (ρ_f + ρ_t) V_c²/(2(HV_f + HV_t)) of about 10.6",
            "Interdiffusion: x = √(2Dt), D = D₀ exp(−Q/RT)",
        ],
        "limits": [
            "The window limits are empirical correlations calibrated on explosive welding of similar "
            "metals. For dissimilar pairs their constants are uncertain by tens of per cent.",
            "The collision angle is an input. It is not solved from the flyer's deformation.",
            "Interface heating, melt layer and diffusion figures are order-of-magnitude estimates.",
        ],
        "main": ["weld_window", "drive_pressure_window", "standoff_window", "flyer_impact", "assess_weld",
                 "interdiffusion_length"],
        "chapter": "/story/welding.html",
    },
    "synthesis": {
        "answers": "Whether a pressure pulse can run any known route to diamond, cubic boron nitride or carbon nitrides.",
        "relations": [
            "Graphite-diamond equilibrium (Kennedy and Kennedy): P = 1.94 GPa + 0.0025 GPa/K × (T − 273 K)",
            "hBN-cBN equilibrium (Solozhenko): P = max(0, 0.00324 GPa/K × (T − 1660 K))",
            "A static route runs only if the pulse outlasts the minimum dwell, the sample pressure reaches "
            "the route minimum, the temperature is inside the route window and the product is the stable phase",
            "Shock route: flyer velocity, then impedance match, then sample pressure",
            "Volume bookkeeping: W = P ΔV with ΔV/V = 1 − ρ_graphite/ρ_diamond = 0.357",
        ],
        "limits": [
            "Route windows are representative of practice. They are not sharp limits.",
            "An equilibrium line says where a phase is stable. Conversion is set by kinetics: catalyst, "
            "temperature and time.",
            "The shock path inherits the limits of the planar impedance match.",
        ],
        "main": ["graphite_diamond_boundary", "hbn_cbn_boundary", "assess_route", "assess_all",
                 "anvil_feasibility", "FlyerSpec", "volume_scaling"],
        "chapter": "/story/synthesis.html",
    },
    "porosity": {
        "answers": "Whether pressure alone closes internal pores, and whether closing them heals them.",
        "relations": [
            "Collapse threshold of an isolated pore: P = (2/3) σ ln(1/f)",
            "Residual porosity: f = min(f₀, exp(−3P/(2σ)))",
            "Trapped gas: f = f₀ p₀/P",
            "Inertial collapse time (Rayleigh): t = 0.915 a √(ρ/ΔP)",
            "Diffusion bonding: t ≈ L²/D, D = D₀ exp(−Q/RT)",
            "Powder yielding (Helle, Easterling and Ashby): P = 3D²(D − D₀)/(1 − D₀) σ_y below "
            "90 % density, (2/3) σ_y ln(1/(1 − D)) above",
        ],
        "limits": [
            "Isolated spherical pores in a rigid, perfectly plastic matrix at its yield stress, so the "
            "thresholds are lower bounds on the pressure for full closure.",
            "Creep is not modelled. Hot isostatic pressing enters as a cited benchmark, not as a model result.",
            "Pressures are mechanical pressures already delivered to the part. There is no field solver "
            "for complex shapes here.",
            "Bonding uses volume self-diffusion with an estimated prefactor.",
        ],
        "main": ["collapse_pressure", "residual_porosity", "rayleigh_collapse_time", "diffusion_bonding_time",
                 "hip_cycle", "assess_pore_closure", "assess_powder_compaction"],
        "chapter": "/story/porosity.html",
    },
    "containment": {
        "answers": "What wall contains a given pressure, and what launchers and fusion coils actually ask of it.",
        "relations": [
            "Lamé thick wall: σ_θ(a) = p (b² + a²)/(b² − a²); von Mises stress at the bore "
            "√3 p b²/(b² − a²)",
            "No finite monolithic elastic wall exists unless σ_allow > √3 p (von Mises) or 2p (Tresca)",
            "Fully plastic or compound wall: p = (2/√3) σ_y ln(b/a)",
            "Rail launcher: F = L′I²/2. Coil launcher and fusion coil: P = B²/(2μ₀)",
            "Mass driver: a = P/(ρL), length = v²/(2a), with P capped by the strength of the payload",
        ],
        "limits": [
            "A static, long, homogeneous cylinder with isotropic yield criteria. Fibre laminates are only "
            "screened for transverse compression at the bore.",
            "Lamé statics holds when the pulse is longer than about three hoop acoustic transits. "
            "Shorter pulses are flagged as inertial.",
            "No thermal, fatigue or end effects.",
        ],
        "main": ["required_radius_ratio", "autofrettage_ratio", "shakedown_pressure", "rail_launcher_load",
                 "fusion_magnet_load", "mass_driver", "assess_containment", "containment_survey"],
        "chapter": "/story/containment.html",
    },
    "press": {
        "answers": "What a complete press looks like for a given pressure and workpiece, and which way of "
                   "building it is closest to possible.",
        "relations": [
            "Field from the pressure, P = B²/(2μ₀), then coil current density, energy and hoop stress from "
            "magnetics",
            "Rise time from the workpiece: skin depth equal to a third of the wall, t_rise = 1/(4f)",
            "Delivered pressure = nominal × (1 − e^(−2h/δ)) × surface ratio from the field solver",
            "Hybrid coupling: a pulsed insert of radius a inside a shorted superconducting loop of radius b "
            "changes its field by −ΔB (a/b)² unless the loop quenches",
            "Each architecture takes the worst status of its components; the best architecture is the "
            "least bad, preferring coils that survive",
        ],
        "limits": [
            "Six architectures on one specification: a ranking of options, not a design of any of them.",
            "The workpiece is a cylinder on the axis, treated as a perfect conductor for the surface "
            "pressure and as a diffusing wall for the transfer.",
            "Thresholds for what counts as demonstrated come from the benchmark ladders and move when "
            "records do.",
        ],
        "main": ["PressSpec", "design_press", "compare_architectures", "workpiece_load", "background_coupling",
                 "reference_spec"],
        "chapter": "/story/machine.html",
    },
    "magnetar": {
        "answers": "What would follow if a magnetar-strength field were available on demand in space. The "
                   "source is assumed, never graded.",
        "relations": [
            "Energy density and pressure B²/(2μ₀); mass equivalent B²/(2μ₀c²); source energy 1.5 × interior "
            "for a uniformly magnetised sphere",
            "Free-expansion time R/c; virial mass M ≥ ρE/σ for any structure that holds the field",
            "Atomic unit of field B₀ = 2.35 × 10⁵ T; QED critical field m²c²/(eħ) = 4.41 × 10⁹ T",
            "Cold degenerate matter under a magnetic piston: Chandrasekhar's equation of state with the "
            "ion-lattice correction; electron capture where the Fermi energy reaches the threshold",
            "Dipole stand-off: r = R (B_s/B_th)^(1/3)",
        ],
        "limits": [
            "A thought experiment. The status of every result is 'assumed' and never enters the ledger.",
            "The degenerate equation of state is for cold matter above about 2.5 million tesla of piston "
            "field; below it the tabulated thresholds apply.",
            "Sizes and durations of laboratory entries on the field ladder are order-of-magnitude estimates.",
        ],
        "main": ["assess_magnetar_mode", "MagnetarSource", "regime_thresholds", "classify_regime",
                 "process_outcomes", "electron_capture_threshold", "standoff_radius"],
        "chapter": "/story/magnetar.html",
    },
    "materials": {
        "answers": "The property data every model draws on, with its provenance.",
        "relations": [
            "Derived moduli: G = E/(2(1 + ν)), K = E/(3(1 − 2ν))",
            "Resistivity: ρ(T) = ρ₂₉₃ [1 + α(T − 293 K)], floored at 2 % of ρ₂₉₃",
            "Hollomon power law through the yield point and the necking point: σ = K εⁿ",
            "Where no Hugoniot is measured: c₀ = √(K/ρ), s = 1.4, tagged as estimated",
        ],
        "limits": [
            "Representative engineering data with a provenance tag on every entry. They are not datasheet guarantees.",
            "Strength depends on condition. The recorded condition is listed with each material.",
        ],
        "main": ["get", "list_materials", "search", "Material", "JohnsonCook", "ZerilliArmstrongBCC", "Hugoniot"],
        "chapter": "/reference/materials.html",
    },
    "benchmarks": {
        "answers": "What has been demonstrated, with citations, so that every status label can be traced.",
        "relations": [
            "A status is the position of a computed quantity on a cited ladder: demonstrated, beyond "
            "demonstration, or beyond the record",
            "Magnetic pressure of a record field: P = B²/(2μ₀)",
            "Placement of a pressure: the benchmarks below and above it, the nearest one on a log scale, "
            "and its fraction of the energy density of TNT",
        ],
        "limits": [
            "Institutional announcements are tagged as reported until they are published.",
            "Where the line between extrapolation and aspirational rests on judgement, the source marks it as estimated.",
        ],
        "main": ["superconducting_field_status", "pulsed_field_status", "energy_status", "where_does_it_sit"],
        "chapter": "/reference/benchmarks.html",
    },
}
"""Hand-written text of the methods page.  Module order here is the order on the page."""

CATEGORY_LABELS: dict[str, str] = {
    "metal": "Metals and alloys",
    "refractory": "Refractory metals",
    "intermetallic": "Intermetallics",
    "ods": "Oxide-dispersion-strengthened alloys",
    "superalloy": "Superalloys",
    "conductor": "Magnet conductors",
    "fibre": "Reinforcing fibres",
    "carbon": "Carbon and carbon-nanosheet materials",
    "ceramic": "Superhard phases and anvil materials",
}
"""Display names of the registry categories, in page order."""

MODEL_LABELS: dict[str, str] = {"johnson_cook": "Johnson-Cook", "zerilli_armstrong_bcc": "Zerilli-Armstrong"}
MODEL_SHORT: dict[str, str] = {"Johnson-Cook": "JC", "Zerilli-Armstrong": "ZA"}
"""Names of the constitutive fits and the abbreviations used in the materials table."""

LADDERS: dict[str, dict[str, Any]] = {
    "field": {"title": "Magnetic field records", "unit": "T", "scale": 1.0,
              "intro": "The strongest field made by each kind of magnet, from steady superconducting magnets "
                       "to explosive flux compression, which destroys the experiment."},
    "current": {"title": "Pulsed currents", "unit": "MA", "scale": 1e-6,
                "intro": "Peak currents delivered to a single load."},
    "ramp_rate": {"title": "Sweep rates of superconducting magnets", "unit": "T/s", "scale": 1.0,
                  "intro": "How fast a superconducting winding has been ramped in routine operation."},
    "energy": {"title": "Pulsed energy stores", "unit": "MJ", "scale": 1e-6,
               "intro": "The largest stores that deliver their energy in one pulse."},
    "pressure": {"title": "Working pressures", "unit": "GPa", "scale": 1e-9,
                 "intro": "Pressures that manufacturing and high-pressure science already use."},
    "static_press": {"title": "Static presses", "unit": "GPa", "scale": 1e-9,
                     "intro": "The highest pressure each anvil technology holds. Sample volume falls by about "
                              "seven orders of magnitude from the first row to the last."},
}
"""Title, display unit and one-line introduction of each ladder (unknown ladders get generic text)."""

STATUS_HELPERS: tuple[tuple[str, str, str, float, tuple[float, float], dict[str, Any]], ...] = (
    # function, what is graded, display unit, scale to the display unit, probe range (SI), keyword arguments
    ("superconducting_field_status", "Field from a superconducting magnet", "T", 1.0, (1.0, 1e3), {}),
    ("pulsed_field_status", "Field from a pulsed magnet that survives, 10 mm bore", "T", 1.0, (1.0, 1e3),
     {"bore_diameter": 0.010}),
    ("pulsed_field_status", "Field from a pulsed magnet that survives, 100 mm bore", "T", 1.0, (1.0, 1e3),
     {"bore_diameter": 0.1}),
    ("ramp_rate_status", "Sweep rate of a superconducting winding", "T/s", 1.0, (1e-2, 1e6), {}),
    ("energy_status", "Energy drawn from a pulsed store in one shot", "MJ", 1e-6, (1e5, 1e11), {}),
    ("pulsed_current_status", "Peak current into one load", "MA", 1e-6, (1e4, 1e9), {}),
    ("peak_power_status", "Peak electrical power", "TW", 1e-12, (1e6, 1e16), {}),
    ("sc_voltage_status", "Terminal voltage across a superconducting winding", "kV", 1e-3, (10.0, 1e7), {}),
    ("pulsed_voltage_status", "Charging voltage of a capacitor bank", "kV", 1e-3, (10.0, 1e7), {}),
)
"""Status helpers of ``magform.benchmarks`` shown on the benchmarks page; their thresholds are probed."""

FIELD_LABELS: dict[str, str] = {
    "nhmfl_32t_all_sc": "All-superconducting user magnet",
    "secuf_35p6t_all_sc": "All-superconducting record",
    "rebco_insert_45p5t": "REBCO coil in a resistive magnet (2019)",
    "lbc9_48p7t": "REBCO coil in a resistive magnet",
    "chmfl_resistive_42t": "All-resistive DC magnet",
    "chmfl_hybrid_45t": "Hybrid DC magnet",
    "lanl_100t_nondestructive": "Pulsed magnet that survives",
    "single_turn_300t": "Single-turn coil (destroyed)",
    "emfc_1200t": "Electromagnetic flux compression",
    "explosive_fc_2800t": "Explosive flux compression",
}
ENERGY_LABELS: dict[str, str] = {
    "hld_50mj_bank": "Dresden capacitor bank",
    "balcones_60mj_hpg": "Balcones homopolar supply",
    "lanl_insert_bank_2mj": "LANL insert coil, capacitor bank",
    "lanl_outsert_120mj": "LANL outsert coils, field energy",
    "nif_400mj_bank": "NIF capacitor bank",
    "anu_500mj_hpg": "ANU homopolar generator",
    "lanl_mg_1200mj": "LANL motor-generator",
}
"""Short row labels for the two ladder charts (fallback: the start of the benchmark description)."""

CONCEPT_PRESSURES: tuple[float, float] = (1.6e9, 4.0e9)
"""The concept note's pressure window, Pa."""

SUPPORT_MODULES: tuple[str, ...] = ("constants", "errors", "status")
"""Modules listed as infrastructure on the methods page (no physics of their own)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _sig(x: Any, n: int = 6) -> Any:
    """Round to ``n`` significant digits for JSON (None for missing or non-finite)."""
    if x is None or isinstance(x, (bool, str)):
        return x
    x = float(x)
    if not math.isfinite(x):
        return None
    return 0.0 if x == 0.0 else float(f"{x:.{n}g}")


def _fmt(x: float | None) -> str | None:
    """Table number: thousands separators above 1000, about three significant digits below."""
    if x is None or not math.isfinite(x):
        return None
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    if a >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    if a == 0:
        return "0"
    return f"{x:.3g}"


def _fmt_value(x: float | None) -> str | None:
    """A cited value exactly as recorded (100.75, 42.02); a computed one to three significant digits."""
    if x is None or not math.isfinite(x):
        return None
    text = f"{x:.5g}"
    if "e" in text or not math.isclose(float(text), x, rel_tol=1e-9):
        return _fmt(x)
    value = float(text)
    if abs(value) >= 1 and value == int(value):
        return f"{int(value):,}"
    return text


_PREFIXES = ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"))
_PREFIXED_UNITS = {"J", "W", "A", "V", "m", "Pa", "J/m^3"}


def _eng(value: float, unit: str) -> str:
    """Value with an SI prefix for the units that take one: 5e7 J -> '50 MJ'; tesla stays as it is."""
    shown = unit.replace("^3", "³").replace("^2", "²")
    if unit not in _PREFIXED_UNITS or value == 0:
        return f"{_fmt_value(value)} {shown}"
    for factor, prefix in _PREFIXES:
        if abs(value) >= factor * 0.9995:
            return f"{_fmt_value(value / factor)} {prefix}{shown}"
    return f"{_fmt_value(value)} {shown}"


def _plain(text: str) -> str:
    """Docstring markup to plain text: drop reST roles and backticks, collapse whitespace."""
    text = re.sub(r":(?:func|class|meth|data|attr|exc|obj):`~?(?:[\w.]*\.)?([\w]+)`", r"\1", text)
    text = re.sub(r":mod:`~?([\w.]+)`", r"\1", text)
    text = text.replace("``", "").replace("`", "")
    return " ".join(text.split())


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _pretty(text: str | None) -> str | None:
    """ASCII maths in source text to typographic form: ^2 becomes a superscript, mu0 becomes μ₀."""
    if not text:
        return text
    text = re.sub(r"\^(-?\d+)", lambda m: m.group(1).translate(_SUPERSCRIPT), text)
    return re.sub(r"\bmu0\b", "μ₀", text)


def _sentence(text: str) -> str:
    """Capitalise a docstring fragment and close it with a full stop."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    return _plain(inspect.cleandoc(doc).split("\n\n")[0])


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _first_sentence(doc: str | None) -> str:
    """First sentence of a docstring's first paragraph, as plain text."""
    paragraph = _first_paragraph(doc)
    if not paragraph:
        return ""
    paragraph = re.sub(r"\s*\(README:[^)]*\)", "", paragraph)  # drop "(README: '...')" asides
    parts = _SENTENCE_END.split(paragraph)
    sentence = parts[0]
    # do not stop on an abbreviation such as "e.g." or "approx." or an initial
    i = 1
    while i < len(parts) and re.search(r"(?:\b(?:e\.g|i\.e|vs|cf|approx|sec|ch|eq|Fig|no)\.|\b[A-Z]\.)$", sentence):
        sentence += " " + parts[i]
        i += 1
    return sentence


def _attribute_docs(module: types.ModuleType) -> dict[str, str]:
    """Docstrings written under module-level constants (``NAME = value`` followed by a string)."""
    try:
        body = ast.parse(inspect.getsource(module)).body
    except (OSError, SyntaxError, TypeError):
        return {}
    docs: dict[str, str] = {}
    for node, following in zip(body, body[1:]):
        name = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if (name and isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)):
            docs[name] = " ".join(following.value.value.split())
    return docs


def _enum_comments(enum_cls: type[Enum]) -> dict[str, str]:
    """Trailing comments of an enum's members: value -> comment."""
    try:
        source = inspect.getsource(enum_cls)
    except (OSError, TypeError):
        return {}
    out: dict[str, str] = {}
    for member in enum_cls:
        match = re.search(rf"^\s*{member.name}\s*=.*?#\s*(.+?)\s*$", source, flags=re.MULTILINE)
        if match:
            out[str(member.value)] = match.group(1)
    return out


def _status_meanings() -> list[dict[str, str]]:
    """The 'Status meanings' block of ``magform.status``: one {key, label, meaning} per Status member."""
    doc = inspect.cleandoc(status_module.__doc__ or "")
    found: dict[str, str] = {}
    for match in re.finditer(r"^``(\w+)``[ \t]+(.+?)(?=^``\w+``|\Z)", doc, flags=re.MULTILINE | re.DOTALL):
        found[match.group(1)] = _plain(match.group(2))
    return [{"key": str(member.value), "label": str(member.value).replace("_", " "),
             "meaning": _sentence(found.get(str(member.value), ""))} for member in Status]


def discover_modules() -> list[tuple[str, types.ModuleType]]:
    """Importable public modules of the package: ``magform._SUBMODULES`` order, then any others on disk.

    Modules that are listed but not written yet, or that fail to import, are skipped.
    """
    package_dir = Path(magform.__file__).resolve().parent
    listed = [n for n in getattr(magform, "_SUBMODULES", ()) if (package_dir / f"{n}.py").exists()]
    extra = sorted(p.stem for p in package_dir.glob("*.py") if not p.stem.startswith("_") and p.stem not in listed)
    found: list[tuple[str, types.ModuleType]] = []
    for name in listed + extra:
        try:
            found.append((name, importlib.import_module(f"magform.{name}")))
        except Exception:  # noqa: BLE001 - a module still being written must not break the docs build
            continue
    return found


def _reference_solenoid() -> Solenoid:
    """The site's reference press (build_docs.REF when this runs inside the generator)."""
    for name in ("build_docs", "__main__"):
        candidate = getattr(sys.modules.get(name), "REF", None)
        if isinstance(candidate, Solenoid):
            return candidate
    return Solenoid(0.05, 0.25, 0.5)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------
def build_materials() -> dict[str, Any]:
    aliases: dict[str, list[str]] = {}
    for alias, key in materials.ALIASES.items():
        aliases.setdefault(key, []).append(alias)
    confidence_notes = _enum_comments(Confidence)

    rows = []
    for m in materials.MATERIALS.values():
        kinds = [MODEL_LABELS.get(getattr(c, "kind", ""), getattr(c, "kind", "model")) for c in m.constitutive]
        fit_parts, fit_short = [], []
        for kind in dict.fromkeys(kinds):
            n = kinds.count(kind)
            fit_parts.append(kind if n == 1 else f"{kind} ×{n}")
            short = MODEL_SHORT.get(kind, kind)
            fit_short.append(short if n == 1 else f"{short} ×{n}")
        condition = m.condition or None
        condition_extra = condition if condition and condition.lower() not in m.name.lower() else None
        if m.hugoniot is None:
            hugoniot = None
        elif m.hugoniot.confidence is Confidence.ESTIMATED:
            hugoniot = "estimated"
        else:
            hugoniot = "measured"
        insulator = m.resistivity >= 1.0
        working = None
        if m.conventional_working is not None:
            lo, hi = m.conventional_working
            working = f"{lo:.0f} K" if hi - lo < 1.0 else f"{lo:.0f} to {hi:.0f} K"
        rows.append({
            "key": m.key,
            "name": m.name,
            "category": m.category,
            "category_label": CATEGORY_LABELS.get(m.category, m.category.replace("_", " ").capitalize()),
            "base_element": m.base_element,
            "structure": m.crystal_structure,
            "condition": condition,
            "condition_extra": condition_extra,
            "density": _fmt(m.density),
            "modulus": _fmt(m.youngs_modulus / 1e9),
            "yield": _fmt(m.yield_strength / 1e6) if m.yield_strength is not None else None,
            "ultimate": _fmt(m.ultimate_strength / 1e6) if m.ultimate_strength is not None else None,
            "elongation": _fmt(100.0 * m.elongation) if m.elongation is not None else None,
            "melting": _fmt(m.melting_point),
            "resistivity": None if insulator else _fmt(m.resistivity * 1e8),
            "insulator": insulator,
            "hardness": _fmt(m.vickers_hardness / 1e9) if m.vickers_hardness is not None else None,
            "dbtt": _fmt(m.dbtt) if m.dbtt is not None else None,
            "working": working,
            "working_note": m.conventional_working_note or None,
            "fit": ", ".join(fit_parts) or None,
            "fit_short": ", ".join(fit_short) or None,
            "fit_count": len(m.constitutive),
            "hugoniot": hugoniot,
            "hall_petch": _fmt(m.hall_petch.k / 1e6) if m.hall_petch is not None else None,
            "hall_petch_estimated": bool(m.hall_petch is not None and m.hall_petch.confidence is Confidence.ESTIMATED),
            "confidence": str(m.confidence.value),
            "aliases": sorted(aliases.get(m.key, [])),
            "notes": m.notes or None,
            "sources": list(m.sources),
            "si": {
                "density": _sig(m.density), "youngs_modulus": _sig(m.youngs_modulus),
                "yield_strength": _sig(m.yield_strength), "ultimate_strength": _sig(m.ultimate_strength),
                "melting_point": _sig(m.melting_point), "resistivity": _sig(m.resistivity),
                "vickers_hardness": _sig(m.vickers_hardness), "dbtt": _sig(m.dbtt),
            },
        })

    present = list(dict.fromkeys(r["category"] for r in rows))
    order = [c for c in CATEGORY_LABELS if c in present] + [c for c in present if c not in CATEGORY_LABELS]
    categories = [{"key": c,
                   "label": CATEGORY_LABELS.get(c, c.replace("_", " ").capitalize()),
                   "count": sum(1 for r in rows if r["category"] == c)} for c in order]
    confidence = [{"key": str(c.value), "meaning": _sentence(confidence_notes.get(str(c.value), "")),
                   "count": sum(1 for r in rows if r["confidence"] == c.value)} for c in Confidence]
    return {
        "count": len(rows),
        "alias_count": len(materials.ALIASES),
        "with_fit": sum(1 for r in rows if r["fit"]),
        "with_measured_hugoniot": sum(1 for r in rows if r["hugoniot"] == "measured"),
        "with_hall_petch": sum(1 for r in rows if r["hall_petch"]),
        "categories": categories,
        "confidence": confidence,
        "units": {"density": "kg/m³", "modulus": "GPa", "yield": "MPa", "ultimate": "MPa", "elongation": "%",
                  "melting": "K", "resistivity": "µΩ·cm", "hardness": "GPa", "dbtt": "K",
                  "hall_petch": "MPa·√m"},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------
def _ladder_meta(key: str, ladder: Sequence[benchmarks.Benchmark]) -> dict[str, Any]:
    meta = LADDERS.get(key)
    if meta is not None:
        return meta
    unit = ladder[0].unit if ladder else ""
    return {"title": key.replace("_", " ").capitalize(), "unit": unit, "scale": 1.0, "intro": ""}


def _status_bands(fn: Callable[..., Status], lo: float, hi: float, kwargs: dict[str, Any]) -> list[tuple[Status, float | None]]:
    """Probe a status helper on a log grid and refine each change by bisection: [(status, upper bound)]."""
    n = 600
    xs = [lo * (hi / lo) ** (i / n) for i in range(n + 1)]
    bands: list[tuple[Status, float | None]] = []
    previous = fn(xs[0], **kwargs)
    for a, b in zip(xs, xs[1:]):
        if fn(b, **kwargs) is previous:
            continue
        for _ in range(80):
            mid = math.sqrt(a * b)
            if fn(mid, **kwargs) is previous:
                a = mid
            else:
                b = mid
        bands.append((previous, a))
        previous = fn(b, **kwargs)
    bands.append((previous, None))
    return bands


def build_benchmarks() -> dict[str, Any]:
    ladders = []
    total = 0
    for key, ladder in benchmarks.ALL_LADDERS.items():
        meta = _ladder_meta(key, ladder)
        rows = []
        for b in sorted(ladder, key=lambda item: item.value):
            shown = b.value * meta["scale"]
            rows.append({
                "key": b.key,
                "description": _pretty(b.description),
                "value": _fmt_value(shown),
                "unit": meta["unit"],
                "text": f"{_fmt_value(shown)} {meta['unit']}",
                "si": _sig(b.value),
                "si_unit": b.unit,
                "year": b.year,
                "citation": _pretty(b.citation),
                "confidence": str(b.confidence.value),
                "note": _pretty(b.note) or None,
            })
        total += len(rows)
        ladders.append({"key": key, "title": meta["title"], "unit": meta["unit"], "intro": meta["intro"],
                        "count": len(rows), "rows": rows})

    docs = _attribute_docs(benchmarks)
    thresholds = []
    for name, doc in docs.items():
        value = getattr(benchmarks, name, None)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        pair = (isinstance(value, tuple) and len(value) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value))
        if not name.isupper() or not (numeric or pair):
            continue
        unit, _, meaning = doc.partition(". ")
        if not meaning or " " in unit or len(unit) > 6:
            unit, meaning = "", doc
        if numeric:
            text = _eng(float(value), unit) if unit else _fmt_value(float(value))
            si: Any = _sig(float(value))
        else:
            text = (f"{_fmt_value(float(value[0]))} to {_eng(float(value[1]), unit)}" if unit
                    else f"{_fmt_value(float(value[0]))} to {_fmt_value(float(value[1]))}")
            si = [_sig(float(v)) for v in value]
        thresholds.append({"name": name, "value": text, "si": si, "si_unit": unit or None,
                           "meaning": _pretty(_plain(meaning))})

    rules = []
    for name, what, unit, scale, (lo, hi), kwargs in STATUS_HELPERS:
        fn = getattr(benchmarks, name, None)
        if fn is None:
            continue
        bands = []
        for status, upper in _status_bands(fn, lo, hi, kwargs):
            bands.append({"status": str(status.value),
                          "upto": None if upper is None else f"{_fmt_value(float(f'{upper * scale:.5g}'))} {unit}"})
        rules.append({"function": name, "what": what, "summary": _pretty(_first_sentence(fn.__doc__)), "bands": bands})

    return {"count": total, "ladders": ladders, "thresholds": thresholds, "rules": rules}


def _short_label(b: benchmarks.Benchmark, labels: dict[str, str]) -> str:
    if b.key in labels:
        return labels[b.key]
    text = re.split(r" \(|, ", b.description)[0]
    return text if len(text) <= 36 else text[:35].rstrip() + "…"


_NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def _is_destructive(b: benchmarks.Benchmark) -> bool:
    text = b.description.lower().replace("non-destructive", "")
    return "destructive" in text


def field_ladder_figure() -> dict[str, Any]:
    lo, hi = (field_for_pressure(p) for p in CONCEPT_PRESSURES)
    ordered = sorted(benchmarks.FIELD_LADDER, key=lambda item: item.value)
    rows = [{"label": _short_label(b, FIELD_LABELS), "value": _sig(b.value, 5), "text": f"{_fmt_value(b.value)} T",
             "tone": "ink", "detail": _pretty(f"{b.description} ({b.year}). {b.citation}")} for b in ordered]
    flags = [_is_destructive(b) for b in ordered]
    n_destroyed = sum(flags)
    note = ""
    if 0 < n_destroyed < len(flags) <= 12 and flags == sorted(flags):  # the destructive methods are the top rows
        note = (f"The first {_NUMBER_WORDS[len(flags) - n_destroyed]} magnets survive the shot. "
                f"The last {_NUMBER_WORDS[n_destroyed]} destroy the coil or the sample space.")
    return {
        "type": "dot",
        "title": "Record fields by kind of magnet",
        "subtitle": "The concept note's window lies above every superconducting and steady-field magnet.",
        "x": {"label": "Field (T, log scale)", "scale": "log", "min": 10, "max": 5000, "short": "Field"},
        "rows": rows,
        "bands": [{"axis": "x", "from": _sig(lo, 4), "to": _sig(hi, 4), "label": "concept note"}],
        "note": note,
        "table": {"columns": ["Magnet", "Field (T)", "Year", "Source"],
                  "rows": [[_pretty(b.description), _sig(b.value, 5), b.year, _pretty(b.citation)] for b in ordered]},
    }


def energy_ladder_figure() -> dict[str, Any]:
    ref = _reference_solenoid()
    lo, hi = (ref.stored_energy(field_for_pressure(p)) / 1e6 for p in CONCEPT_PRESSURES)
    ordered = sorted(benchmarks.ENERGY_LADDER, key=lambda item: item.value)
    rows = [{"label": _short_label(b, ENERGY_LABELS), "value": _sig(b.value / 1e6, 5),
             "text": f"{_fmt_value(b.value / 1e6)} MJ", "tone": "ink",
             "detail": _pretty(f"{b.description} ({b.year}). {b.citation}")} for b in ordered]
    bore_mm = 2e3 * ref.inner_radius
    return {
        "type": "dot",
        "title": "Pulsed energy stores",
        "subtitle": f"The field energy of a {bore_mm:.0f} mm bore press at 63 to 100 T is the size of the largest "
                    "capacitor bank built to drive magnets.",
        "x": {"label": "Energy (MJ, log scale)", "scale": "log", "min": 10, "max": 10000, "short": "Energy"},
        "rows": rows,
        "bands": [{"axis": "x", "from": _sig(lo, 4), "to": _sig(hi, 4), "label": f"{bore_mm:.0f} mm press"}],
        "table": {"columns": ["Store", "Energy (MJ)", "Year", "Source"],
                  "rows": [[_pretty(b.description), _sig(b.value / 1e6, 5), b.year, _pretty(b.citation)]
                           for b in ordered]},
    }


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------
_YEAR = re.compile(r"\((?:[A-Za-z]+\.? )?((?:1[89]|20)\d\d)\)")
_INITIAL = re.compile(r"^(?:[A-Z][a-z]?\.-?)+$")
_PAGES = re.compile(r"\s*,?\s*(?:pp?\.\s*)?[A-Za-z]{0,4}\d[\dA-Za-z]*(?:\s*[-–]\s*[A-Za-z]?\d+[A-Za-z]?)?")
_PARTICLES = {"lord", "sir", "de", "van", "von", "der", "and", "&"}


def _first_author(text: str) -> str:
    """Surname of the first author (or the organisation) of a citation string."""
    head = text.split("(")[0]
    for token in head.replace(",", " ").split():
        t = token.strip(";:'\"")
        if not t or _INITIAL.match(t) or t.lower() in _PARTICLES:
            continue
        return t
    return text.split()[0] if text.split() else ""


def _citation_handle(text: str) -> tuple[str, str | None]:
    """(label author, year) used to sort and name a work: ('Jackson', '1999'), ('ASTM F2924', None)."""
    author = _first_author(text)
    match = _YEAR.search(text)
    year = match.group(1) if match else None
    if author.isupper() and len(author) >= 3:  # an organisation: keep a designation such as "E8/E8M"
        tokens = text.replace(",", " ").split()
        if len(tokens) > 1 and any(ch.isdigit() for ch in tokens[1]):
            author = f"{author} {tokens[1]}"
    return author, year


def _trim_citation(text: str) -> str:
    """The bibliographic part of a reference: cut the per-module annotation after year and pages."""
    match = _YEAR.search(text)
    if not match:
        return text
    end = match.end()
    pages = _PAGES.match(text[end:])
    if pages:
        end += pages.end()
    return text[:end].rstrip(" ,;:.") + "."


_STOPWORDS = {"the", "of", "and", "in", "on", "for", "to", "at", "by", "an", "ed", "eds", "et", "al", "vol"}


def _normalise(text: str) -> str:
    match = _YEAR.search(text)
    core = text[:match.end()] if match else text
    words = [w for w in re.sub(r"[^0-9A-Za-z]+", " ", core).lower().split() if len(w) > 1 and w not in _STOPWORDS]
    return " ".join(words)


def _same_work(a: str, b: str) -> bool:
    """True if two citations with the same first author and year describe the same work."""
    na, nb = _normalise(a), _normalise(b)
    if difflib.SequenceMatcher(None, na, nb).ratio() >= 0.7:
        return True
    ta, tb = set(na.split()), set(nb.split())
    return bool(ta and tb) and len(ta & tb) / min(len(ta), len(tb)) >= 0.8


def build_references(modules: Sequence[tuple[str, types.ModuleType]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    by_module = []
    order = [name for name in METHODS] + [name for name, _ in modules if name not in METHODS]
    rank = {name: i for i, name in enumerate(order)}
    for name, module in sorted(modules, key=lambda item: rank[item[0]]):
        refs = getattr(module, "REFERENCES", None)
        if not isinstance(refs, dict) or not refs:
            continue
        items = []
        for key, text in refs.items():
            if not isinstance(text, str):
                continue
            item = {"module": name, "key": str(key), "text": " ".join(text.split()),
                    "internal": text.lstrip().startswith("magform.")}
            items.append(item)
            entries.append(item)
        by_module.append({"module": name, "count": len(items),
                          "entries": [{"key": i["key"], "text": i["text"]} for i in items]})

    # cluster the same work cited by several modules (same first author and year, similar text)
    buckets: dict[tuple[str, str | None], list[list[dict[str, Any]]]] = {}
    for item in entries:
        if item["internal"]:
            continue
        author, year = _citation_handle(item["text"])
        clusters = buckets.setdefault((author.lower(), year), [])
        for cluster in clusters:
            if _same_work(item["text"], cluster[0]["text"]):
                cluster.append(item)
                break
        else:
            clusters.append([item])

    works = []
    for (author_key, year), clusters in buckets.items():
        for cluster in clusters:
            trimmed = [_trim_citation(i["text"]) for i in cluster]
            # canonical text: the commonest trimmed form, then the "Initials Surname" style, then the longest
            canonical = max(trimmed, key=lambda t: (trimmed.count(t), bool(_INITIAL.match(t.split()[0])), len(t)))
            author, _ = _citation_handle(cluster[0]["text"])
            cited_in = list(dict.fromkeys(i["module"] for i in cluster))
            works.append({
                "label": f"{author} {year}" if year else author,
                "text": canonical,
                "year": int(year) if year else None,
                "modules": cited_in,
                "sort": f"{author_key} {year or '0000'} {canonical.lower()}",
            })
    works.sort(key=lambda w: w["sort"])
    for w in works:
        del w["sort"]
    return {
        "count": len(entries),
        "unique": len(works),
        "module_count": len(by_module),
        "entries": [{"module": e["module"], "key": e["key"], "text": e["text"]} for e in entries],
        "modules": by_module,
        "bibliography": works,
    }


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------
def _short_default(value: Any) -> str:
    if value is None or isinstance(value, (bool, int)):
        return repr(value)
    if isinstance(value, float):  # 1e+12 -> 1e12, 5e-05 -> 5e-5
        return re.sub(r"e([+-])0*(\d+)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), f"{value:g}")
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, tuple) and len(value) <= 3 and all(isinstance(v, (int, float, str)) for v in value):
        return "(" + ", ".join(_short_default(v) for v in value) + ")"
    return "…"


def _signature(name: str, obj: Any) -> str:
    """Compact call signature: parameter names with short defaults, no annotations."""
    try:
        parameters = list(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return ""
    if inspect.isclass(obj) and len(parameters) > 8:
        return ""
    parts = []
    keyword_marker = False
    for p in parameters:
        if p.kind is p.VAR_POSITIONAL:
            parts.append("*" + p.name)
            keyword_marker = True
            continue
        if p.kind is p.VAR_KEYWORD:
            parts.append("**" + p.name)
            continue
        if p.kind is p.KEYWORD_ONLY and not keyword_marker:
            parts.append("*")
            keyword_marker = True
        parts.append(p.name if p.default is p.empty else f"{p.name}={_short_default(p.default)}")
    return f"{name}({', '.join(parts)})"


def _public_callables(module: types.ModuleType) -> list[tuple[str, Any]]:
    """Functions and classes named in ``__all__`` (type aliases and constants are left out)."""
    out = []
    for name in getattr(module, "__all__", ()):
        obj = getattr(module, name, None)
        if isinstance(obj, types.GenericAlias) or getattr(obj, "__module__", "") == "typing":
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            out.append((name, obj))
    return out


def _entry(name: str, obj: Any) -> dict[str, Any]:
    return {"name": name, "kind": "class" if inspect.isclass(obj) else "function",
            "signature": _signature(name, obj), "summary": _pretty(_first_sentence(obj.__doc__)) or ""}


def _is_model_module(name: str, module: types.ModuleType) -> bool:
    if name in METHODS:
        return True
    if name in SUPPORT_MODULES or name in ("cli", "claims"):
        return False
    return isinstance(getattr(module, "REFERENCES", None), dict) or any(
        n.startswith(("assess_", "design_", "compare_")) for n, _ in _public_callables(module))


def build_methods(modules: Sequence[tuple[str, types.ModuleType]]) -> dict[str, Any]:
    available = dict(modules)
    order = [n for n in METHODS if n in available] + [n for n, m in modules if n not in METHODS and _is_model_module(n, m)]
    records = []
    missing: list[str] = []
    for name in order:
        module = available[name]
        hand = METHODS.get(name, {})
        callables = _public_callables(module)
        lookup = dict(callables)
        wanted = hand.get("main")
        if wanted is None:  # no hand-picked list: the assessments and designers are the entry points
            wanted = [n for n, _ in callables if n.startswith(("assess_", "design_", "compare_"))]
        main = []
        for entry_name in wanted:
            if entry_name in lookup:
                main.append(_entry(entry_name, lookup[entry_name]))
            else:
                missing.append(f"{name}.{entry_name}")
        refs = getattr(module, "REFERENCES", None)
        public = getattr(module, "__all__", ())
        records.append({
            "module": name,
            "title": _first_paragraph(module.__doc__).rstrip("."),
            "answers": hand.get("answers", ""),
            "relations": list(hand.get("relations", [])),
            "limits": list(hand.get("limits", [])),
            "chapter": hand.get("chapter"),
            "entry_points": main,
            "all_entry_points": [_entry(n, o) for n, o in callables],
            "public_count": len(public),
            "callable_count": len(callables),
            "constant_count": len(public) - len(callables),
            "reference_count": len(refs) if isinstance(refs, dict) else 0,
            "has_tests": (ROOT / "tests" / f"test_{name}.py").exists(),
        })
    support = []
    for name, module in modules:
        if name in order:
            continue
        support.append({"module": name, "title": _first_paragraph(module.__doc__).rstrip("."),
                        "callable_count": len(_public_callables(module))})
    verdict_notes = _enum_comments(Verdict)
    return {
        "statuses": _status_meanings(),
        "verdicts": [{"key": str(v.value), "label": str(v.value).replace("_", " "),
                      "meaning": _sentence(verdict_notes.get(str(v.value), ""))} for v in Verdict],
        "count": len(records),
        "callable_count": sum(r["callable_count"] for r in records),
        "modules": records,
        "support": support,
        "missing_entry_points": missing,
    }


# ---------------------------------------------------------------------------
# chapter entry point
# ---------------------------------------------------------------------------
def chapter_reference(site: Any) -> None:
    """Fill the reference data sets and chart specs on a ``build_docs.Site``."""
    modules = discover_modules()
    data_materials = build_materials()
    data_benchmarks = build_benchmarks()
    data_references = build_references(modules)
    data_methods = build_methods(modules)
    if data_methods["missing_entry_points"]:
        print("docs_reference: entry points named in METHODS but not found: "
              + ", ".join(data_methods["missing_entry_points"]), file=sys.stderr)
    site.data["materials"] = data_materials
    site.data["benchmarks"] = data_benchmarks
    site.data["references"] = data_references
    site.data["methods"] = data_methods
    site.figure("ref_field_ladder", field_ladder_figure())
    site.figure("ref_energy_ladder", energy_ladder_figure())
    site.nums(
        ref_material_count=str(data_materials["count"]),
        ref_category_count=str(len(data_materials["categories"])),
        ref_benchmark_count=str(data_benchmarks["count"]),
        ref_citation_count=str(data_references["count"]),
        ref_reference_count=str(data_references["unique"]),
        ref_module_count=str(data_methods["count"]),
        ref_callable_count=str(data_methods["callable_count"]),
    )


class _StandaloneSite:
    """The part of ``build_docs.Site`` this chapter uses, for running the file on its own."""

    def __init__(self) -> None:
        self.n: dict[str, str] = {}
        self.figures: dict[str, dict] = {}
        self.data: dict[str, Any] = {}

    def num(self, key: str, value: str) -> None:
        self.n[key] = value

    def nums(self, **values: str) -> None:
        self.n.update(values)

    def figure(self, key: str, spec: dict) -> None:
        self.figures[key] = spec


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-run the reference chapter of the docs generator.")
    parser.add_argument("--write", action="store_true",
                        help="write materials/benchmarks/references/methods.json to docs/_data")
    args = parser.parse_args(list(argv) if argv is not None else None)
    site = _StandaloneSite()
    chapter_reference(site)
    m, b, r, me = (site.data[k] for k in ("materials", "benchmarks", "references", "methods"))
    print(f"materials : {m['count']} rows in {len(m['categories'])} categories, {m['alias_count']} aliases")
    print(f"benchmarks: {b['count']} entries in {len(b['ladders'])} ladders, {len(b['thresholds'])} thresholds, "
          f"{len(b['rules'])} status rules")
    print(f"references: {r['count']} citations in {r['module_count']} modules, {r['unique']} distinct works")
    print(f"methods   : {me['count']} model modules, {me['callable_count']} public callables; "
          f"support: {[s['module'] for s in me['support']]}")
    print(f"figures   : {sorted(site.figures)}")
    print(f"numbers   : {site.n}")
    if args.write:
        out = ROOT / "docs" / "_data"
        for name in ("materials", "benchmarks", "references", "methods"):
            path = out / f"{name}.json"
            path.write_text(json.dumps(site.data[name], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
