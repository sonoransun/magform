---
title: The inverted magnet
kicker: Chapter 9 · Case 6
doc: MF-09
description: Turn the magnet's structure inside out and it becomes a pressure vessel for railguns, mass drivers and fusion machines. The loads are real. Steel already carries them, and the applications are limited by something else.
pitch: The same structure, inverted, as a housing for railguns and fusion.
brief: Steel already carries these loads. The proposed wall material does not exist, and the applications are limited elsewhere.
models: [containment, benchmarks]
verdict: aspirational
claims: [C26, C27, C28]
---
{% assign n = site.data.n %}

The note's last application is the only one that does not use the press as a press. The winding that resists a magnet's own bursting force is, seen from the bore, a pressure vessel. So the same structure could be "inverted" into a housing for things that push outward: railguns, coilguns, the magnets of a fusion reactor.

There are three questions here. Are the loads real? Can the proposed wall hold 4 GPa? And do the applications need it?

## The loads are real, and steel holds them

A railgun carrying 4 MA across a 50 mm bore pushes its rails apart with {{ n.c_rail_pressure }} GPa, the pressure of a {{ n.c_rail_field }} T field and the bottom of the note's window. The force on the projectile is {{ n.c_rail_force }} MN. The coil of a 20 T fusion magnet is pushed outward by its own field with {{ n.c_fusion_coil }} MPa.

Both are contained today. The railgun bore needs a layered, prestressed wall of maraging steel with an outer radius {{ n.c_rail_ratio }} times the bore radius. The fusion coil needs a plain steel case with a ratio of {{ n.c_fusion_wall }}. The model rates both {{ n.c_rail_status }}. No new material is involved.

## A wall for four gigapascals

At 4 GPa the picture changes, and the reason is a property of thick cylinders that no amount of material gets around.

A wall under internal pressure is stressed most at the bore, and the stress falls away quickly with depth. Adding thickness adds material where there is little stress to carry. Past a certain point more wall does nothing: a one-piece wall of any thickness yields at the bore once the pressure passes its strength divided by 1.73.

{% include figure.html id="wall_ratio" caption="<b>Wall thickness against pressure.</b> Each one-piece curve runs out where the wall would have to be infinitely thick. Layering and prestressing extend the range. The note's window lies beyond every one-piece wall in steel." %}

With a safety factor of 1.5, containing 4 GPa in one piece takes a material with a strength of {{ n.c_required }} GPa. The strongest steel, maraging C350, has {{ n.c_maraging_strength }} GPa.

<aside class="aside">
<p class="label">Built up in layers</p>
<span class="big">{{ n.c_die }} GPa</span>
<p>is about what layered, shrink-fitted steel dies hold in routine service. The belt presses that make diamond go higher by using tungsten carbide in compression.</p>
</aside>

The standard remedy is to build the wall in layers, each shrunk onto the last so the bore starts out in compression. That is how high-pressure dies are made, and it is why the survey below has any entries that are not red. For maraging steel at 4 GPa the outer radius comes to {{ n.c_maraging_ratio }} times the bore: a wall 20 cm thick around a 5 cm hole.

<div class="table-scroll">
<table>
<thead><tr><th>Wall material</th><th class="r">Strength</th><th>How it could hold 4 GPa</th><th class="r">Outer / bore radius</th><th>Verdict</th></tr></thead>
<tbody>
{% for w in site.data.containment %}
<tr><td>{{ w.name }}</td><td class="r">{{ w.strength }} GPa</td><td>{% if w.route == "none" %}it cannot{% elsif w.route == "compound" %}layered and prestressed{% else %}one piece{% endif %}{% if w.transverse == false %}<br><span class="muted small">needs a metal liner</span>{% endif %}</td><td class="r">{% if w.route == "none" %}n/a{% elsif w.elastic %}{{ w.elastic }}{% else %}{{ w.compound }}{% endif %}</td><td>{% include stamp.html v=w.status %}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

The carbon rows look best and carry a catch. A wound carbon wall is strong along its fibres, which run round the bore. The bore pressure pushes on it *across* the fibres, where a laminate is held together by resin and carries about {{ n.c_transverse }} GPa. Sixteen times too little. So a carbon wall needs a metal liner to take the pressure and spread it, and at 4 GPa that liner is the layered steel wall again.

The nanosheet composite the note relies on would do the job on paper, with the properties the note assumes. Those properties are a target, not a material: nobody has made it in bulk ([chapter 2](machine.md)). And the superconducting tape, the other half of the "exact same architecture", contributes nothing to a housing.

One more thing applies to all the rows. These loads are pulses. A stress wave crosses the wall in about {{ n.c_transit_us }} µs, so a wall loaded faster than that does not act as a whole, and static formulas flatter it.

## Whether the applications need it

Suppose the wall existed. The note says it would stop "plasma-containment fields or hyper-velocity projectile tracks from structurally tearing themselves apart". Take each in turn.

**Fusion.** A plasma is a thin gas. In a reactor-class machine it pushes on its surroundings with {{ n.c_fusion_plasma }} MPa. The magnet that holds it is pushed {{ n.c_fusion_ratio }} times harder by its own field, {{ n.c_fusion_coil }} MPa, and that is within the reach of steel. Fusion magnets are limited by their superconductor and by cost, not by a shortage of gigapascals.

**Mass drivers.** The pressure that accelerates a payload acts on the payload. An aluminium payload can take {{ n.c_driver_pressure }} MPa on its base before it yields. That fixes the acceleration at {{ n.c_driver_g }} g, and the acceleration fixes the length.

{% include figure.html id="launcher_length" caption="<b>Length of a mass driver.</b> The upper line is set by what the payload survives. The lower line is what a 4 GPa bore would allow if the payload were not there to object." %}

To reach lunar escape speed a one-metre aluminium payload needs {{ n.c_driver_length }} m of track at {{ n.c_driver_field }} T. For orbital speed from an airless body, {{ n.c_leo_length }} m. A bore that could hold 4 GPa offers {{ n.c_driver_over }} times more push than the payload can accept, and changes neither length.

## What survives

The observation behind the claim is sound: a magnet's reinforcement and a pressure vessel are the same structural problem. But the problem is solved in steel up to about {{ n.c_die }} GPa, the proposed material does not exist, and the machines named in the note are limited by their plasmas and their payloads. A stronger wall would be welcome in a 100 T magnet. That is the subject of chapter 2, and it is where this idea belongs.

## Check it yourself

```console
$ magform contain --pressure 4000 --bore 0.05
$ magform contain --pressure 1600 --bore 0.05 --material maraging_c350
```

```python
from magform import containment

containment.required_monolithic_strength(4e9) / 1e9       # 6.9 GPa before the safety factor
wall = containment.assess_containment(4e9, "maraging_c350", 0.025)
wall.status, wall.route, wall.plastic_ratio               # extrapolation, compound, 8.8

fusion = containment.fusion_magnet_load(20.0, 12.2)
fusion.coil_pressure / fusion.plasma_pressure             # 54

containment.mass_driver(2400.0, "aluminum_6061", 1.0, 0.05, 4e9).launcher_length   # 42 m
```
