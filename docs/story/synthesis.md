---
title: Diamond by the tonne
kicker: Chapter 7 · Case 4
doc: MF-07
description: The note imagines a large-bore magnetic press making diamond, lonsdaleite and carbon nitrides in bulk. Synthesis needs three things at once, and a magnetic pulse supplies part of one.
pitch: Diamond-anvil pressures, scaled up to large volumes.
brief: Diamond needs pressure, heat and time together. A pulse offers part of the first, except through shock.
models: [synthesis, shock, benchmarks]
verdict: infeasible
claims: [C19, C20, C21, C22]
---
{% assign n = site.data.n %}

The first of the note's applications is the grandest: "macroscopic synthesis of quantum and superhard materials". Its argument runs in two steps. Today these materials can be made "only in microscopic microgram quantities" in diamond anvil cells. A large magnetic press would scale the same pressures to "massive volumes".

Both steps fail, and for reasons that have little to do with magnets.

## What is made today, and how

Synthetic diamond is not a laboratory curiosity. It is an industry older than the transistor radio.

<div class="table-scroll wide">
<table>
<thead><tr><th>Product</th><th class="r">Pressure</th><th class="r">Temperature</th><th class="r">Shortest hold</th><th>Made today</th><th>By a 4 GPa pulse</th></tr></thead>
<tbody>
{% for r in site.data.synthesis %}
<tr><td>{{ r.name }}<br><span class="muted small">{{ r.form }}</span></td><td class="r">{{ r.pressure }} GPa</td><td class="r">{{ r.temperature }}{% unless r.temperature == "shock-heated" %} K{% endunless %}</td><td class="r">{{ r.dwell }} s</td><td class="small">{{ r.scale }}</td><td>{% include stamp.html v=r.status %}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

Diamond grit, large single crystals and the polycrystalline compacts on drill bits are made by the tonne in presses with tungsten-carbide anvils, at 5 to 7 GPa. Cubic boron nitride likewise. Nano-polycrystalline diamond, harder than single crystal, comes out of multi-anvil presses in pieces up to a centimetre across. Of the note's three examples only carbon nitrides are truly confined to the diamond anvil cell, because they need {{ n.s_cn_lo }} to {{ n.s_cn_hi }} GPa.

So there is no microgram bottleneck for a magnetic press to relieve, except for materials that need thirty times its pressure.

## Three requirements

Every one of the industrial routes needs three things together.

{% include figure.html id="phase_diagram" caption="<b>Pressure and temperature of the industrial routes.</b> Each dot sits above its equilibrium line, where the hard phase is the stable one, and at a temperature where atoms can rearrange. The note's window is the band at the bottom, and it has no temperature." %}

**Pressure.** The lowest pressure any static route uses is {{ n.s_p_min }} GPa. The top of the note's window is 4.

**Temperature.** They all run at 1,500 to 3,000 K. Pressure makes diamond the stable form of carbon; heat is what lets graphite turn into it. At room temperature graphite will sit at 4 GPa indefinitely. A magnet heats nothing but the skin of a conductor.

**Time.** The shortest hold of any static route is {{ n.s_dwell_min }} second, and most take minutes to days.

{% include figure.html id="route_dwell" caption="<b>How long each route takes.</b> The band is the life of a magnetic pressure pulse." %}

Time is the one that cannot be engineered around. From [chapter 1](pressure.md): a field presses only while it is changing. Hold the field steady to hold the pressure, and the pressure is gone. A magnetic press is a hammer, and these are oven recipes.

Run through the model at 4 GPa, room temperature and a tenth of a millisecond, all {{ n.s_static_count }} static routes come out infeasible, each on all three counts.

## The one route a hammer can run

Two entries in the table take microseconds: shock synthesis. A shock wave compresses and heats in the same instant, and for a moment the conditions for diamond exist. DuPont made industrial diamond powder this way from 1961, with explosives.

A magnet can launch a plate instead. This is the flyer of [chapter 6](welding.md) with the gap opened up: a 2 mm copper plate driven across 10 mm.

{% include figure.html id="flyer_shock" caption="<b>Shock pressure from a magnetically driven plate.</b> The grey line is the tempting number, copper striking copper. Graphite is softer and the shock in it is less than half as strong." %}

At 4 GPa the plate reaches {{ n.s_flyer_kms }} km/s. Striking copper it would make {{ n.s_shock_cucu }} GPa. Striking graphite, which is what matters, it makes {{ n.s_shock_graphite }} GPa, and in the porous graphite used as feedstock {{ n.s_shock_porous }} GPa. Shock diamond needs about 20 and lonsdaleite about 15. At the bottom of the note's window the same plate makes {{ n.s_shock_lo }} GPa, and neither forms.

So the top edge of the window reaches the bottom edge of shock synthesis. That is rated an extrapolation, and it is the closest this application gets. What it would produce is what shock synthesis has always produced: micrometre powder, for polishing. Not "large-scale diamond semiconductor substrates", which are grown slowly as single crystals, and not armour plate.

## "Quantum materials"

The heading promises quantum materials too. Diamond for quantum devices is grown from gas at less than atmospheric pressure. The superconducting hydrides need 150 GPa and more and fall apart when it is released. Nothing in that field is waiting for 4 GPa.

## What survives

A magnetically driven flyer is a clean, repeatable way to do shock experiments, and at the top of the note's window it could make diamond powder. That is a laboratory tool with an explosive competitor, not a new industry.

## Check it yourself

```console
$ magform synth --pressure 4000
$ magform synth --pressure 4000 --flyer copper --thickness 0.002 --gap 0.01
```

```python
from magform import synthesis

plate = synthesis.FlyerSpec("copper_ofhc", thickness=0.002, gap=0.01)
shot = synthesis.assess_route("diamond_shock_dupont", 4e9, flyer=plate)
shot.flyer_velocity, shot.pressure_sample / 1e9, shot.status     # 2113 m/s, 22.3 GPa, extrapolation
```
