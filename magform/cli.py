"""Command-line interface: ``magform <command> ...`` (or ``python -m magform``).

Units on the command line follow the concept note and shop-floor habit:
pressures in **MPa**, fields in tesla, lengths in metres, times in seconds,
temperatures in kelvin, angles in degrees.  Every command accepts ``--json``
for machine-readable output (strict JSON; non-finite numbers become null).

Exit codes: 0 on success, 2 for an invalid input or unknown material (the
message goes to standard error), as for argparse usage errors.

Commands
--------
``field``          pressure <-> field, and where the pressure sits among cited benchmarks
``press``          size a press and grade the concept note's magnet for it
``architectures``  compare six ways of building that press
``pulse``          capacitor-bank network for a coil, field and rise time
``forge``          can a pressure forge a metal? (flow stress, brittleness, heating)
``micro``          what a cold forming pulse does to grains and dislocations
``weld``           impact-weld assessment for a flyer, target, gap and drive pressure
``synth``          can a pressure pulse run a superhard-material synthesis route?
``pores``          pore closure and healing under a pressure pulse
``contain``        wall needed to contain a pressure, by material
``magnetar``       thought experiment: a magnetar-class field applied on demand in space
``claims``         the ledger: every claim of the concept note with its verdict
``materials``      list the materials registry or show one record
``benchmarks``     the cited record ladders behind every status
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from typing import Any, Callable, Sequence

from . import __version__
from ._report import ResultMixin, to_jsonable
from .errors import MagformError

_STATUS_MARK = {
    "state_of_the_art": "[ok]", "extrapolation": "[~]", "aspirational": "[*]", "infeasible": "[x]",
    "contradicted": "[!=]", "not_modelled": "[-]", "not_applicable": "[ ]", "assumed": "[if]",
}


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def _emit_json(value: Any) -> None:
    print(json.dumps(to_jsonable(value), indent=2, allow_nan=False))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cells[0], widths)).rstrip(),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in cells[1:]]
    return "\n".join(lines)


def _status(value: Any) -> str:
    text = str(getattr(value, "value", value))
    return f"{_STATUS_MARK.get(text, '')} {text.replace('_', ' ')}".strip()


def _g(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}g}"


def _time(value: float | None) -> str:
    if value is None:
        return "steady"
    if value < 1e-3:
        return f"{value * 1e6:.3g} us"
    if value < 1.0:
        return f"{value * 1e3:.3g} ms"
    return f"{value:.3g} s"


def _wrap(text: str, indent: str = "    ", width: int = 100) -> str:
    import textwrap

    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _notes(result: Any) -> list[str]:
    lines: list[str] = []
    for w in getattr(result, "warnings", ()) or ():
        lines.append(_wrap(f"! {w}", "  "))
    for n in getattr(result, "notes", ()) or ():
        lines.append(_wrap(f"- {n}", "  "))
    return lines


def _show(result: Any, as_json: bool) -> None:
    if as_json:
        _emit_json(result)
    elif isinstance(result, ResultMixin):
        print(result.text())
    else:
        print(result)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _cmd_field(args: argparse.Namespace) -> int:
    from . import benchmarks
    from .magnetics import field_for_pressure, magnetic_pressure

    if (args.pressure is None) == (args.field is None):
        raise MagformError("give exactly one of --pressure (MPa) or --field (T)")
    if args.pressure is not None:
        pressure = args.pressure * 1e6
        field = field_for_pressure(pressure)
    else:
        field = args.field
        pressure = magnetic_pressure(field)
    place = benchmarks.where_does_it_sit(pressure) if pressure > 0 else None
    data = {
        "pressure": pressure, "field": field, "energy_density": pressure,
        "tnt_fraction": place.tnt_fraction if place else 0.0,
        "superconducting_status": benchmarks.superconducting_field_status(field),
        "pulsed_status_record_bore": benchmarks.pulsed_field_status(field, benchmarks.PULSED_NONDESTRUCTIVE_BORE),
        "pulsed_status_100mm_bore": benchmarks.pulsed_field_status(field, 0.1),
        "nearest_benchmark": place.nearest if place else None,
        "below": place.below if place else (), "above": place.above if place else (),
    }
    if args.json:
        _emit_json(data)
        return 0
    print(f"pressure          {pressure / 1e6:.6g} MPa  ({pressure / 1e9:.4g} GPa)")
    print(f"field             {field:.4f} T          (P = B^2 / 2 mu0)")
    print(f"energy density    {pressure / 1e9:.4g} GJ/m^3  ({data['tnt_fraction']:.0%} of TNT)")
    print(f"from a superconducting magnet:      {_status(data['superconducting_status'])}")
    record_bore = f"{benchmarks.PULSED_NONDESTRUCTIVE_BORE * 1e3:g} mm bore:"
    print(f"from a pulsed magnet, {record_bore:<13} {_status(data['pulsed_status_record_bore'])}")
    print(f"from a pulsed magnet, 100 mm bore:  {_status(data['pulsed_status_100mm_bore'])}")
    if place:
        by_key = {b.key: b for b in benchmarks.PRESSURE_BENCHMARKS}
        print("\namong cited working pressures:")
        rows = [(f"{by_key[k].value / 1e9:.3g} GPa", by_key[k].description, "<= this" if k in place.below else "")
                for k in list(place.below) + list(place.above)]
        print(_table(("pressure", "process", ""), rows))
    print("\nnote: a steady field exerts no pressure on a stationary workpiece; the field must be pulsed.")
    return 0


def _spec(args: argparse.Namespace):
    from .press import PressSpec

    return PressSpec(
        pressure=args.pressure * 1e6, bore_diameter=args.bore, length=args.length, winding_thickness=args.winding,
        workpiece=args.workpiece, workpiece_wall=args.wall, workpiece_radius=args.workpiece_radius,
        workpiece_length=args.workpiece_length, rise_time=args.rise, temperature=args.temperature,
        safety_factor=args.safety_factor, background_field=args.background)


def _architecture_rows(architectures) -> list[tuple]:
    return [(a.name, _status(a.status), f"{a.field:.4g} T", _time(a.rise_time), f"{a.pressure_fraction:.0%}",
             "yes" if a.reusable else "no", a.limiting) for a in architectures]


_ARCH_HEADERS = ("architecture", "status", "field", "rise", "pressure delivered", "coil reusable", "limited by")


def _cmd_press(args: argparse.Namespace) -> int:
    from .press import design_press

    d = design_press(_spec(args))
    if args.json:
        _emit_json(d)
        return 0
    s, w = d.spec, d.workpiece
    print(f"Press: {s.pressure / 1e9:.4g} GPa in a {s.bore_diameter * 1e3:.4g} mm bore, {s.length:.4g} m long, "
          f"{s.winding_thickness * 1e3:.4g} mm build")
    print(f"  field                   {d.field:.2f} T   (field factor {d.field_factor:.3f})")
    print(f"  winding current density {d.current_density / 1e6:.4g} A/mm^2")
    print(f"  stored energy           {d.stored_energy / 1e6:.4g} MJ   ({d.tnt_fraction:.0%} of TNT energy density)")
    print(f"  peak hoop stress        {d.peak_hoop_stress / 1e9:.3g} GPa  ({d.hoop_ratio:.2f} x pressure)")
    print(f"  coil-end radial field   {d.end_field_fraction:.3f} x B0;  peak field on conductor {d.peak_conductor_field:.2f} T")
    print(f"  field rise time         {_time(d.rise_time)}  (set by the workpiece)")
    print(f"\nWorkpiece: {w.material}, {'solid billet' if w.solid else f'{w.wall * 1e3:.3g} mm wall'}, "
          f"radius {w.radius * 1e3:.4g} mm, length {w.length * 1e3:.4g} mm")
    print(f"  pressure passed by wall {w.pressure_transfer:.0%}   (skin depth {w.skin_depth * 1e3:.3g} mm)")
    if w.surface_mid_ratio is not None:
        print(f"  surface pressure        {w.surface_mid_ratio:.3f} x nominal at mid-plane; min/max {w.surface_uniformity:.3f} "
              f"over the central 80 %; {w.surface_peak_ratio:.1f} x at the corners")
    print(f"  delivered pressure      {w.delivered_pressure / 1e9:.3g} GPa")
    print(f"  skin temperature rise   +{w.skin_temperature_rise:.0f} K  ({'melts' if w.skin_melts else 'solid'}; "
          f"melting field {w.skin_melting_field:.0f} T)")
    print(f"\nConcept magnet (carbon-nanosheet REBCO, pulsed): {_status(d.status)}")
    print(f"  conductor         {_status(d.conductor.status)}   margin {_g(d.conductor.margin, 3)}")
    print(f"  structure         {_status(d.structure.status)}   needs composite modulus "
          f"{d.structure.required_composite_modulus / 1e9:.4g} GPa")
    ratio = d.pulsed_operation.loss_to_margin_ratio
    print(f"  pulsed operation  {_status(d.pulsed_operation.status)}   "
          + (f"per-pulse loss is {ratio:.3g} x the enthalpy margin" if math.isfinite(ratio)
             else "no enthalpy margin at this field"))
    if d.network is not None:
        n = d.network
        print(f"\nCapacitor bank for a resistive coil in the same envelope: {_status(n.status)}")
        print(f"  {n.turns} turns, {n.charge_voltage / 1e3:.4g} kV, {n.peak_current / 1e6:.4g} MA, "
              f"{n.peak_power / 1e9:.4g} GW peak, {n.bank_energy / 1e6:.4g} MJ bank")
    print("\nWays of building it:")
    print(_table(_ARCH_HEADERS, _architecture_rows(d.architectures)))
    best = next(a for a in d.architectures if a.key == d.best_architecture)
    print(f"\nBest option: {best.name} ({_status(d.best_status)})")
    print("\n".join(_notes(d)))
    return 0


def _cmd_architectures(args: argparse.Namespace) -> int:
    from .press import compare_architectures

    archs = compare_architectures(_spec(args))
    if args.json:
        _emit_json(archs)
        return 0
    print(_table(_ARCH_HEADERS, _architecture_rows(archs)))
    for a in archs:
        print(f"\n{a.name}: {_status(a.status)}")
        print(_wrap(a.summary, "  "))
        print("\n".join(_notes(a)))
    return 0


def _cmd_pulse(args: argparse.Namespace) -> int:
    from . import pulse
    from .magnetics import Solenoid, field_for_pressure

    if (args.pressure is None) == (args.field is None):
        raise MagformError("give exactly one of --pressure (MPa) or --field (T)")
    field = args.field if args.field is not None else field_for_pressure(args.pressure * 1e6)
    sol = Solenoid(0.5 * args.bore, 0.5 * args.bore + args.winding, args.length)
    net = pulse.assess_pulse_network(sol, field, args.rise, turns=args.turns, max_voltage=args.max_voltage * 1e3)
    if args.json:
        _emit_json(net)
        return 0
    print(f"Coil: {args.bore * 1e3:.4g} mm bore, {args.length:.4g} m long, {args.winding * 1e3:.4g} mm build; "
          f"{field:.2f} T in {_time(args.rise)}")
    print(f"  status           {_status(net.status)}")
    print(f"  turns            {net.turns}   ({net.layers} layers)")
    print(f"  inductance       {net.inductance:.4g} H   resistance {net.resistance:.4g} ohm   L/R {_time(net.time_constant)}")
    print(f"  charge voltage   {net.charge_voltage / 1e3:.4g} kV   {_status(net.voltage_status)}")
    print(f"  peak current     {net.peak_current / 1e6:.4g} MA   {_status(net.current_status)}")
    print(f"  peak power       {net.peak_power / 1e9:.4g} GW   {_status(net.power_status)}")
    print(f"  bank energy      {net.bank_energy / 1e6:.4g} MJ   {_status(net.energy_status)}   "
          f"(field energy {net.magnetic_energy / 1e6:.4g} MJ)")
    print(f"  conductor        +{net.conductor_temperature_rise:.0f} K to {net.conductor_final_temperature:.0f} K   "
          f"{_status(net.thermal_status)}")
    print(f"  fastest rise a {args.max_voltage:g} kV bank allows (one turn): {_time(net.minimum_rise_time)}")
    print(f"  homopolar generator of the same energy, fastest rise: {_time(net.homopolar_rise_time)}")
    print("\nSources for this energy and rise time:")
    print(_table(("source", "status", "fastest rise", "reusable"),
                 [(s.name, _status(s.status), _time(s.fastest_rise_time), "yes" if s.reusable else "no")
                  for s in net.sources]))
    print("\n".join(_notes(net)))
    return 0


def _cmd_forge(args: argparse.Namespace) -> int:
    from .plasticity import assess_forgeability

    a = assess_forgeability(args.material, args.pressure * 1e6, target_strain=args.strain, strain_rate=args.rate,
                            temperature=args.temperature, constraint_factor=args.constraint)
    if args.json:
        _emit_json(a)
        return 0
    print(f"{a.material} at {a.pressure / 1e9:.4g} GPa, strain {a.target_strain:g}, {a.strain_rate:g} 1/s, "
          f"{a.temperature:.0f} K: {_status(a.status)}")
    print(f"  flow stress         {a.flow_stress_min / 1e9:.3g}-{a.flow_stress_max / 1e9:.3g} GPa  ({a.primary_model})")
    print(f"  required pressure   {a.required_pressure / 1e9:.3g} GPa   margin {a.margin_conservative:.2f} "
          f"(strongest fit) to {a.margin_optimistic:.2f}")
    print(f"  reachable strain    {a.max_strain_at_pressure:.3g}")
    print(f"  bulk heating        +{a.adiabatic_temperature_rise:.0f} K")
    if a.dbtt is not None:
        print(f"  ductile-brittle     {a.dbtt:.0f} K ({'below' if a.below_dbtt else 'above'}); confinement ratio "
              f"{a.confinement_ratio:.2f}")
    if a.adiabatic_shear_strain is not None:
        print(f"  shear localisation  from strain {a.adiabatic_shear_strain:.3g}")
    if a.warm_forming_temperature is not None:
        print(f"  warm forming from   {a.warm_forming_temperature:.0f} K")
    print("\n".join(_notes(a)))
    return 0


def _cmd_micro(args: argparse.Namespace) -> int:
    from .microstructure import assess_microstructure

    a = assess_microstructure(args.material, strain=args.strain, strain_rate=args.rate, temperature=args.temperature,
                              peak_field=args.field, frequency=args.frequency)
    if args.json:
        _emit_json(a)
        return 0
    print(f"{a.material}, strain {a.strain:g} at {a.strain_rate:g} 1/s, {a.temperature:.0f} K: {_status(a.status)}")
    print(f"  dislocation density {a.initial_dislocation_density:.2g} -> {a.dislocation_density:.2g} per m^2; "
          f"cells {a.cell_size * 1e6:.3g} um")
    print(f"  grain size          {a.grain_size_before * 1e6:.3g} -> {a.grain_size_after * 1e6:.3g} um  "
          f"(ultra-fine: {'yes' if a.ultrafine else 'no'}; needs strain {_g(a.strain_for_ultrafine, 3)})")
    print(f"  yield strength      {a.yield_strength_before / 1e9:.3g} -> {a.yield_strength_after / 1e9:.3g} GPa")
    print(f"  bulk heating        +{a.bulk_temperature_rise:.0f} K; recrystallised fraction "
          f"{a.bulk_recrystallised_fraction:.2g}; grains preserved: {'yes' if a.grain_structure_preserved else 'no'}")
    if a.shear_localisation_strain is not None:
        print(f"  shear localisation  from strain {a.shear_localisation_strain:.3g}")
    if a.skin is not None:
        k = a.skin
        print(f"  skin at {k.peak_field:.4g} T      {k.peak_temperature:.0f} K ({k.homologous_temperature:.2f} of melting) "
              f"for {_time(k.cooling_time)}; heat-affected depth {a.haz_depth * 1e3:.3g} mm"
              f"{'; melted' if k.melted else ''}")
    print("\n".join(_notes(a)))
    return 0


def _cmd_weld(args: argparse.Namespace) -> int:
    from . import welding

    kwargs = dict(collision_angle=math.radians(args.angle))
    if args.pressure is None:
        w = welding.drive_pressure_window(args.flyer, args.target, args.thickness, args.gap, **kwargs)
        if args.json:
            _emit_json(w)
            return 0
        print(f"{w.flyer} onto {w.target}, {args.thickness * 1e3:g} mm flyer over a {args.gap * 1e3:g} mm gap, "
              f"{args.angle:g} degrees:")
        if not w.exists:
            print("  no weld window at this angle")
            return 0
        print(f"  impact velocity window  {w.velocity_min:.0f}-{w.velocity_max:.0f} m/s")
        print(f"  drive pressure window   {w.pressure_min / 1e6:.4g}-{w.pressure_max / 1e6:.4g} MPa")
        print(f"  drive field window      {w.field_min:.3g}-{w.field_max:.3g} T")
        print(f"  flight time             {w.flight_time_min * 1e6:.3g}-{w.flight_time_max * 1e6:.3g} us")
        return 0
    a = welding.assess_weld(args.flyer, args.target, args.thickness, args.gap, args.pressure * 1e6, **kwargs)
    if args.json:
        _emit_json(a)
        return 0
    print(f"{a.flyer} onto {a.target} at {a.drive_pressure / 1e6:.4g} MPa drive ({a.drive_field:.3g} T): {_status(a.status)}")
    print(f"  impact velocity     {a.impact_velocity:.0f} m/s   (window {a.vp_min:.0f}-{a.vp_max:.0f} m/s)")
    print(f"  collision point     {a.collision_velocity:.0f} m/s  (sonic limit {a.vc_sonic:.0f} m/s)")
    print(f"  interface           {a.interface_morphology}")
    print(f"  impact pressure     {a.impact_pressure / 1e9:.3g} GPa  ({_g(a.pressure_amplification, 3)} x the drive)")
    print(f"  flyer skin          +{a.flyer_skin_temperature_rise:.0f} K ({'melts' if a.flyer_skin_melts else 'solid'})")
    print(f"  interdiffusion      {_g(a.interdiffusion_solid * 1e9 if a.interdiffusion_solid else None, 3)} nm solid, "
          f"against {a.fusion_weld_interdiffusion * 1e6:.3g} um in a fusion weld")
    print("\n".join(_notes(a)))
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from . import synthesis

    flyer = None
    if args.flyer:
        flyer = synthesis.FlyerSpec(args.flyer, args.thickness, args.gap, sample=args.sample)
    results = synthesis.assess_all(args.pressure * 1e6, temperature=args.temperature, pulse_duration=args.pulse,
                                   intensification=args.intensification, flyer=flyer)
    if args.json:
        _emit_json(results)
        return 0
    rows = [(a.product, a.kind.replace("_", " "), _status(a.status), f"{a.pressure_sample / 1e9:.3g} GPa",
             f"{a.pressure_margin:.2f}", "yes" if a.dwell_ok else "no") for a in results]
    print(_table(("product", "route", "status", "sample pressure", "pressure margin", "dwell met"), rows))
    for a in results:
        if a.warnings:
            print(f"\n{a.product}:")
            for w in a.warnings:
                print(_wrap(f"! {w}", "  "))
    print("\n" + _wrap(synthesis.QUANTUM_MATERIALS_NOTE, ""))
    return 0


def _cmd_pores(args: argparse.Namespace) -> int:
    from . import porosity

    a = porosity.assess_pore_closure(args.material, args.pressure * 1e6, initial_porosity=args.porosity,
                                     pore_radius=args.pore_radius, temperature=args.temperature,
                                     pulse_duration=args.pulse, section_thickness=args.section,
                                     drive_frequency=args.frequency)
    if args.json:
        _emit_json(a)
        return 0
    print(f"{a.material}, {a.initial_porosity:.2%} porosity at {a.pressure / 1e9:.4g} GPa, {a.temperature:.0f} K, "
          f"{_time(a.pulse_duration)}: {_status(a.status)}")
    print(f"  closure             {_status(a.closure_status)}")
    print(f"  healing             {_status(a.healing_status)}")
    print(f"  pores start to collapse at {a.collapse_pressure_quasi_static / 1e9:.3g} GPa (slow), "
          f"{a.collapse_pressure_dynamic / 1e9:.3g} GPa (at this rate)")
    print(f"  residual porosity   {a.residual_porosity_quasi_static:.2e} (slow), {a.residual_porosity_dynamic:.2e} (at this rate)")
    print(f"  collapse time       {_time(a.collapse_time)};  diffusion bonding time {_g(a.diffusion_bond_time, 3)} s")
    if a.skin_depth is not None:
        print(f"  skin depth          {a.skin_depth * 1e3:.3g} mm ({'section transparent' if a.transparent else 'section shields'})")
    if a.hip is not None:
        print(f"  HIP comparator      {a.hip.pressure / 1e6:.0f} MPa, {a.hip.temperature:.0f} K, {a.hip.time / 3600:.1f} h")
    print("\n".join(_notes(a)))
    return 0


def _cmd_contain(args: argparse.Namespace) -> int:
    from . import containment

    pressure = args.pressure * 1e6
    radius = 0.5 * args.bore
    if args.material:
        results = (containment.assess_containment(pressure, args.material, radius, args.safety_factor,
                                                  args.criterion, args.pulse),)
    else:
        results = containment.containment_survey(pressure, radius, args.safety_factor, criterion=args.criterion,
                                                 pulse_duration=args.pulse)
    if args.json:
        _emit_json(results)
        return 0
    print(f"Containing {pressure / 1e9:.4g} GPa in a {args.bore * 1e3:.4g} mm bore (safety factor {args.safety_factor:g}, "
          f"{args.criterion}):")
    rows = [(a.material, _status(a.status), a.route.replace("_", " "), _g(a.elastic_ratio, 3), _g(a.plastic_ratio, 3),
             f"{a.strength / 1e9:.3g} GPa") for a in results]
    print(_table(("wall material", "status", "route", "elastic b/a", "compound b/a", "strength"), rows))
    print(f"\nstrength needed for any finite elastic wall: {results[0].required_monolithic_strength / 1e9:.3g} GPa")
    if len(results) == 1:
        print("\n".join(_notes(results[0])))
    return 0


def _cmd_magnetar(args: argparse.Namespace) -> int:
    from . import magnetar

    a = magnetar.assess_magnetar_mode(args.field, args.radius, material=args.material)
    if args.json:
        _emit_json(a)
        return 0
    print("Thought experiment: the source is assumed, not assessed.\n")
    print(a.text())
    return 0


def _cmd_claims(args: argparse.Namespace) -> int:
    from . import claims

    if args.id:
        results = tuple(claims.evaluate(i) for i in args.id)
    else:
        results = claims.evaluate_all()
    if args.verdict:
        results = tuple(r for r in results if r.verdict.value == args.verdict)
    if args.json:
        _emit_json({"summary": claims.summary(results), "claims": results})
        return 0
    if args.markdown:
        print("| id | claim | verdict | finding |")
        print("|---|---|---|---|")
        for r in results:
            print(f"| {r.id} | {r.quote} | {r.verdict.label} | {r.finding} |")
        return 0
    detail = bool(args.id) or args.full
    for r in results:
        print(f"{r.id}  {_status(r.verdict)}  {r.topic}")
        if detail:
            print(_wrap(f'"{r.quote}"', "    "))
            print(_wrap(r.finding, "    "))
            for e in r.evidence:
                print(_wrap(f"- {e}", "      "))
            print(_wrap("models: " + ", ".join(r.models), "      "))
            print()
    if not args.id:
        counts = claims.summary(results)
        print("\n" + ", ".join(f"{v.replace('_', ' ')}: {n}" for v, n in counts.items() if n))
        if not detail:
            print("show one in full: magform claims C08      everything: magform claims --full")
    return 0


def _cmd_materials(args: argparse.Namespace) -> int:
    from . import materials

    if args.key:
        m = materials.get(args.key)
        if args.json:
            _emit_json({f.name: getattr(m, f.name) for f in dataclasses.fields(m)})
            return 0
        for f in dataclasses.fields(m):
            v = getattr(m, f.name)
            if v in (None, "", ()) or f.name in ("constitutive", "hugoniot", "hugoniot_high", "hall_petch"):
                continue
            print(f"{f.name:28s} {v}")
        for model in m.constitutive:
            print(f"{'constitutive':28s} {model.kind}: {model.source}")
        if m.hugoniot is not None:
            print(f"{'hugoniot':28s} Us = {m.hugoniot.c0:g} + {m.hugoniot.s:g} up m/s ({m.hugoniot.source})")
        return 0
    rows = materials.list_materials(args.category)
    if args.json:
        _emit_json([{"key": m.key, "name": m.name, "category": m.category} for m in rows])
        return 0
    print(_table(("key", "name", "category", "yield or strength", "melting"),
                 [(m.key, m.name, m.category,
                   f"{(m.yield_strength or m.ultimate_strength or 0) / 1e6:.0f} MPa" if (m.yield_strength or m.ultimate_strength) else "",
                   f"{m.melting_point:.0f} K") for m in rows]))
    print(f"\n{len(rows)} materials; categories: {', '.join(materials.CATEGORIES)}")
    return 0


def _cmd_benchmarks(args: argparse.Namespace) -> int:
    from . import benchmarks

    ladders = benchmarks.ALL_LADDERS
    if args.ladder:
        if args.ladder not in ladders:
            raise MagformError(f"unknown ladder '{args.ladder}'; choose one of {', '.join(ladders)}")
        ladders = {args.ladder: ladders[args.ladder]}
    if args.json:
        _emit_json(ladders)
        return 0
    for name, ladder in ladders.items():
        print(f"\n{name}")
        print(_table(("value", "year", "what", "source"),
                     [(f"{b.value:.5g} {b.unit}", b.year, b.description, b.citation) for b in ladder]))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def _add_press_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pressure", type=float, required=True, help="magnetic pressure to deliver, MPa (the note: 1600-4000)")
    p.add_argument("--bore", type=float, default=0.1, help="winding inner diameter, m (default 0.1, the reference press)")
    p.add_argument("--length", type=float, default=0.5, help="coil length, m (default 0.5)")
    p.add_argument("--winding", type=float, default=0.2, help="radial build of the winding, m (default 0.2)")
    p.add_argument("--workpiece", default="copper_ofhc", help="workpiece material key or alias (default copper_ofhc)")
    p.add_argument("--wall", type=float, default=None, help="workpiece wall, m (default: solid billet)")
    p.add_argument("--workpiece-radius", type=float, default=None, help="workpiece radius, m (default 0.8 x bore radius)")
    p.add_argument("--workpiece-length", type=float, default=None, help="workpiece length, m (default 0.6 x coil length)")
    p.add_argument("--rise", type=float, default=None, help="field rise time, s (default: matched to the workpiece wall)")
    p.add_argument("--temperature", type=float, default=4.2, help="superconducting winding temperature, K (default 4.2)")
    p.add_argument("--safety-factor", type=float, default=1.5, help="safety factor on strain and critical current (default 1.5)")
    p.add_argument("--background", type=float, default=11.5, help="DC background field of the hybrid option, T (default 11.5)")


def build_parser() -> argparse.ArgumentParser:
    """The ``magform`` argument parser (one sub-command per model family)."""
    parser = argparse.ArgumentParser(
        prog="magform",
        description="Engineering models for 63-100 T magnetic-pressure processing, with a feasibility verdict on "
                    "every result. Pressures in MPa, fields in T, lengths in m, times in s, temperatures in K.",
        epilog="Every command accepts --json. Documentation: the docs/ site of the repository.")
    parser.add_argument("--version", action="version", version=f"magform {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    def add(name: str, func: Callable[[argparse.Namespace], int], help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.add_argument("--json", action="store_true", help="print strict JSON instead of text")
        p.set_defaults(func=func)
        return p

    p = add("field", _cmd_field, "Convert between magnetic pressure and field and place the pressure among benchmarks.")
    p.add_argument("--pressure", type=float, help="pressure, MPa")
    p.add_argument("--field", type=float, help="field, T")

    p = add("press", _cmd_press, "Size a press and grade the concept note's superconducting magnet for it.")
    _add_press_arguments(p)

    p = add("architectures", _cmd_architectures, "Compare six ways of building the press.")
    _add_press_arguments(p)

    p = add("pulse", _cmd_pulse, "Capacitor-bank network that drives a coil to a field in a given rise time.")
    p.add_argument("--pressure", type=float, help="magnetic pressure, MPa")
    p.add_argument("--field", type=float, help="central field, T")
    p.add_argument("--bore", type=float, required=True, help="winding inner diameter, m")
    p.add_argument("--length", type=float, required=True, help="coil length, m")
    p.add_argument("--winding", type=float, required=True, help="radial build, m")
    p.add_argument("--rise", type=float, required=True, help="field rise time, s")
    p.add_argument("--turns", type=int, default=None, help="number of turns (default: most that a --max-voltage bank can drive)")
    p.add_argument("--max-voltage", type=float, default=25.0, help="charge-voltage ceiling, kV (default 25)")

    p = add("forge", _cmd_forge, "Can a delivered pressure forge this metal?")
    p.add_argument("material")
    p.add_argument("--pressure", type=float, required=True, help="delivered pressure, MPa")
    p.add_argument("--strain", type=float, default=0.3, help="target strain (default 0.3)")
    p.add_argument("--rate", type=float, default=1e4, help="strain rate, 1/s (default 1e4)")
    p.add_argument("--temperature", type=float, default=293.15, help="workpiece temperature, K (default 293.15)")
    p.add_argument("--constraint", type=float, default=1.2, help="constraint factor Q (default 1.2)")

    p = add("micro", _cmd_micro, "What a cold forming pulse does to the grain and dislocation structure.")
    p.add_argument("material")
    p.add_argument("--strain", type=float, default=0.3, help="strain (default 0.3)")
    p.add_argument("--rate", type=float, default=1e4, help="strain rate, 1/s (default 1e4)")
    p.add_argument("--temperature", type=float, default=293.15, help="initial temperature, K")
    p.add_argument("--field", type=float, default=None, help="peak surface field, T (adds the eddy-current skin)")
    p.add_argument("--frequency", type=float, default=None, help="pulse frequency 1/(4 t_rise), Hz (default 1e4)")

    p = add("weld", _cmd_weld, "Impact-weld assessment; without --pressure, the drive window that welds.")
    p.add_argument("flyer")
    p.add_argument("target")
    p.add_argument("--pressure", type=float, default=None, help="drive (magnetic) pressure, MPa")
    p.add_argument("--thickness", type=float, default=1e-3, help="flyer thickness, m (default 0.001)")
    p.add_argument("--gap", type=float, default=2e-3, help="standoff gap, m (default 0.002)")
    p.add_argument("--angle", type=float, default=10.0, help="collision angle, degrees (default 10)")

    p = add("synth", _cmd_synth, "Can a pressure pulse run each superhard-material synthesis route?")
    p.add_argument("--pressure", type=float, required=True, help="magnetic drive pressure, MPa")
    p.add_argument("--temperature", type=float, default=293.15, help="sample temperature, K")
    p.add_argument("--pulse", type=float, default=1e-4, help="pulse duration, s (default 1e-4)")
    p.add_argument("--intensification", type=float, default=1.0, help="anvil area ratio (default 1: no anvils)")
    p.add_argument("--flyer", default=None, help="flyer material for the shock routes (e.g. copper_ofhc)")
    p.add_argument("--thickness", type=float, default=2e-3, help="flyer thickness, m (default 0.002)")
    p.add_argument("--gap", type=float, default=1e-2, help="flyer standoff, m (default 0.01)")
    p.add_argument("--sample", default="graphite_crystal", help="shocked sample material (default graphite_crystal)")

    p = add("pores", _cmd_pores, "Pore closure and healing under a pressure pulse.")
    p.add_argument("material")
    p.add_argument("--pressure", type=float, required=True, help="delivered pressure, MPa")
    p.add_argument("--porosity", type=float, default=0.01, help="initial porosity as a fraction (default 0.01)")
    p.add_argument("--pore-radius", type=float, default=50e-6, help="pore radius, m (default 5e-5)")
    p.add_argument("--temperature", type=float, default=293.15, help="temperature, K")
    p.add_argument("--pulse", type=float, default=1e-4, help="time at pressure, s (default 1e-4)")
    p.add_argument("--section", type=float, default=None, help="section thickness, m (adds the skin-depth check)")
    p.add_argument("--frequency", type=float, default=None, help="drive frequency, Hz (with --section)")

    p = add("contain", _cmd_contain, "Wall needed to contain a pressure, for one material or a survey.")
    p.add_argument("--pressure", type=float, required=True, help="internal pressure, MPa")
    p.add_argument("--bore", type=float, required=True, help="bore diameter, m")
    p.add_argument("--material", default=None, help="wall material (default: survey steel to graphene)")
    p.add_argument("--safety-factor", type=float, default=1.5)
    p.add_argument("--criterion", default="von_mises", choices=("von_mises", "tresca", "hoop"))
    p.add_argument("--pulse", type=float, default=None, help="pulse duration, s (adds the dynamic check)")

    p = add("magnetar", _cmd_magnetar, "Thought experiment: a magnetar-class field applied on demand in space.")
    p.add_argument("--field", type=float, default=1e10, help="field at the surface of the source, T (default 1e10)")
    p.add_argument("--radius", type=float, default=1e-3, help="radius of the source, m (default 0.001)")
    p.add_argument("--material", default="iron_armco", help="workpiece material (default iron_armco)")

    p = add("claims", _cmd_claims, "The ledger: every claim of the concept note with its verdict and evidence.")
    p.add_argument("id", nargs="*", help="claim ids to show in full (C01-C28, Q1, Q2)")
    p.add_argument("--verdict", default=None, help="only this verdict (e.g. contradicted)")
    p.add_argument("--full", action="store_true", help="findings and evidence for every claim")
    p.add_argument("--markdown", action="store_true", help="a Markdown table")

    p = add("materials", _cmd_materials, "List the materials registry or show one record.")
    p.add_argument("key", nargs="?", help="material key or alias")
    p.add_argument("--category", default=None, help="filter the list by category")

    p = add("benchmarks", _cmd_benchmarks, "The cited record ladders behind every status.")
    p.add_argument("ladder", nargs="?", help="field, current, ramp_rate, energy, pressure or static_press")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except MagformError as exc:
        print(f"magform: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # e.g. piped into head
        return 0


__all__ = ["build_parser", "main"]
