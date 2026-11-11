---
title: The magnetar mode
kicker: Chapter 11 · Thought experiment
doc: MF-11
description: Suppose the field were not the problem. Grant a source in space that delivers the field of a magnetar on demand, and work out what follows for energy, for matter and for each of the note's six processes.
brief: Grant a magnetar-strength source in space. A pinhead of its field holds megatons, and no process in the note survives contact with it.
models: [magnetar, magnetics, benchmarks]
verdict: assumed
---
{% assign n = site.data.n %}
{% assign x = site.data.magnetar %}

Every chapter so far has asked whether a 63 to 100 T machine can be built and what it would do. This one asks a different kind of question. Suppose a far stronger field were simply available:

<div class="note" data-label="Premise">
<p>{{ x.premise }}</p>
</div>

A magnetar is a neutron star with a surface field of about 10¹⁰ to 10¹¹ T. Nothing known could make such a field on demand, so the premise is not graded. It carries the stamp **assumed**, which the package uses for nothing else, and it never counts for or against the note. Everything downstream of it is computed in the usual way.

## How far away that is

{% include figure.html id="x_ladder" wide=true caption="<b>Fields that exist, on one axis.</b> Each step to the right is a factor of a hundred. The note's window is the narrow band on the left. Laboratory fields beyond it last microseconds or less and fill less than a thimble." %}

Take 10¹⁰ T as the working figure. It is {{ n.x_gap_field }} times the strongest field any magnet has survived, and {{ n.x_gap_destructive }} times the {{ n.x_destructive }} T reached by explosive flux compression, which destroys the apparatus on every shot. Pressure goes as the square of the field, so the magnetic pressure is {{ n.x_pressure }} Pa: {{ n.x_gap_pressure }} times the top of the note's window.

Two natural yardsticks fall inside that gap. At {{ n.x_atomic }} T the magnetic force on an electron matches the electric force that holds an atom together. At {{ n.x_qed }} T, the critical field of quantum electrodynamics, an electron's orbital energy in the field equals its rest energy and empty space itself becomes birefringent. A magnetar field is beyond both.

## The energy bill

A field stores energy in proportion to its pressure. At 10¹⁰ T one cubic micrometre of field holds {{ n.x_um3_mj }} MJ. One cubic millimetre holds {{ n.x_mm3_mt }} megatons of TNT.

<aside class="aside">
<p class="label">The field has weight</p>
<span class="big">{{ n.x_lead_ratio }} × lead</span>
<p>By E = mc², field energy at 10¹⁰ T has a mass of {{ n.x_mass_density }} kg per cubic metre.</p>
</aside>

{% include figure.html id="x_energy" caption="<b>Energy to fill a region with field.</b> At magnetar strength, a source the size of a grain of sand already exceeds every pulsed energy store on Earth." %}

A source one centimetre in radius holds {{ n.x_cm_mt }} megatons, {{ n.x_cm_world }} % of the energy the world uses in a year, and by E = mc² that energy weighs {{ n.x_cm_mass }} kg. That is {{ n.x_gap_energy }} times the largest pulsed store ever built. A square kilometre of perfect solar collector near the Earth would need {{ n.x_solar_years }} years to bank the energy for a single cubic millimetre.

This is the first reason the premise says "in space". Nothing on a planet's surface can be allowed to hold, or lose, that much energy.

## Nothing holds it

A 100 T magnet is hard to build because its field tries to burst it with 4 GPa. This field pushes outward with {{ n.x_pressure }} Pa, and the strongest chemical bond gives way near 10¹¹. By the virial theorem a carbon-fibre structure able to hold the one-centimetre source would weigh {{ n.x_virial }} kg, and that sum assumes the fibre keeps its strength, which it does not.

Unheld, the field leaves at the speed of light. The one-centimetre source lasts {{ n.x_confinement_ps }} picoseconds. Assembling it in that time takes {{ n.x_power }} W, the output of {{ n.x_power_suns }} Suns, and its release is the same power going the other way.

Nature's only way of holding such a field is gravity. A neutron star can carry up to about {{ n.x_cf_field }} T before the field would unbind it, which is why magnetars exist and why nothing smaller does the job. So "on demand" has to mean a flash, and each shot spends its source.

## What it does to a workpiece

Put iron or copper in a rising field and it passes through a series of thresholds. The first five are within reach of today's destructive experiments.

<div class="table-scroll wide">
<table>
<thead><tr><th>Threshold</th><th class="r">Iron</th><th class="r">Copper</th><th>The workpiece is then</th></tr></thead>
<tbody>
{% for t in x.thresholds %}
<tr><td>{{ t.label | capitalize }}</td><td class="r">{{ t.iron }}</td><td class="r">{{ t.copper }}</td><td class="small">{{ t.state }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

Everything the rest of this site analyses happens in the first two rows. By {{ n.x_vapour_cu }} T the driven surface of copper is vapour. By {{ n.x_bulk_cu }} T the pressure exceeds the metal's stiffness and it is compressed like a gas. At {{ n.x_ionisation }} T the pressure crushes the electron shells and the metal is a dense plasma. Strength, melting point, grain size: the properties that fill the earlier chapters stop mattering in turn, and only the atomic number is left.

{% capture density_caption %}<b>Iron under a magnetar-class field.</b> The pressure of the field squeezes cold iron to the density of a white dwarf. The lower line is the field's own weight, which at 10¹⁰ T is still {{ n.x_matter_over_field }} times less than that of the matter it has compressed.{% endcapture %}
{% include figure.html id="x_density" caption=density_caption %}

At 10¹⁰ T the pressure holds iron at {{ n.x_piston_density }} kg/m³, {{ n.x_iron_ratio }} times its density on the bench. The electrons are squeezed so hard that at {{ n.x_capture_fe }} T they are forced into the nuclei, and iron begins turning into manganese. Copper starts lower, at {{ n.x_capture_cu }} T. The workpiece is being transmuted.

Where the field threads the matter instead of pushing on it, the atoms change shape. An electron can no longer circle the nucleus; it is pinned to a field line and can only slide along it. At 10¹⁰ T a hydrogen atom is a needle {{ n.x_aspect }} times longer than it is wide, bound by {{ n.x_h_binding }} eV in place of the usual 13.6. An electron's orbital energy in the field is {{ n.x_cyclotron }} MeV, more than twice its rest energy of {{ n.x_electron_rest }} MeV. Chemistry as we know it does not exist there.

## The six processes, revisited

With that ladder in hand, each of the note's applications can be followed up the field.

<div class="table-scroll wide">
<table class="fixed">
<thead><tr><th style="width:11%">Process</th>{% for c in x.columns %}<th>{{ c.field }}<br><span class="muted small">{{ c.tier }}</span></th>{% endfor %}</tr></thead>
<tbody>
{% for p in x.processes %}
<tr><td><b>{{ p.name }}</b></td>{% for c in p.cells %}<td class="small">{{ c }}</td>{% endfor %}</tr>
{% endfor %}
</tbody>
</table>
</div>

The pattern is the one the earlier chapters found, carried to its end. Each process has a best field, and it is low: {{ n.w_field }} T to weld, 30 T to close a tube, {{ n.z_hip_mpa }} MPa of gas to press a casting. More field never helps the process. It replaces it with a different and more violent one. Only synthesis improves for a while, since megabar pressures do make new phases, before it too becomes nuclear physics.

## Why it has to be in space

{% include figure.html id="x_standoff" wide=true caption="<b>Keep-out distances around a one-centimetre source.</b> The field itself falls off quickly. The energy does not." %}

The field of a small source falls with the cube of distance. Around the one-centimetre source it drops to the Earth's own field at {{ n.x_r_earth }} m and passes the pacemaker limit at {{ n.x_r_pacemaker }} m. Inside {{ n.x_levitation }} m it pushes water away harder than the Earth's gravity would hold it. Inside {{ n.x_r_atomic }} cm chemistry stops.

Those are modest distances. The energy sets a larger one. If the shot's {{ n.x_cm_mt }} megatons escapes as radiation, it does damage out to {{ n.x_damage_km }} km. That figure, not the field, is why the premise needs open space. For comparison, a real magnetar holds a chemistry-stopping field out to {{ n.x_magnetar_km }} km from its surface.

## The one useful place to stand

There is a sensible way to use such a source, and it is to keep well back.

<aside class="aside">
<p class="label">Working distance</p>
<span class="big">{{ n.x_r_100t }} to {{ n.x_r_63 }} m</span>
<p>from a one-centimetre, 10¹⁰ T source is where the field equals the note's 100 to 63 T.</p>
</aside>

At {{ n.x_r_100t }} m from the one-centimetre source the field has fallen to 100 T. At {{ n.x_r_63 }} m it is 63 T. So the whole of the note's forming window is a shell {{ n.x_shell_cm }} cm thick, several metres out, with no coil around the workpiece and nothing to burst. Switch the source on and any conductor in that shell feels a pressure pulse of the size the note asks for, give or take a factor for its shape. In this reading the magnetar mode does deliver what the note wanted from its magnet.

The reading leans on the premise twice over. A source left to itself is gone in {{ n.x_confinement_ps }} picoseconds, and a flash that short pushes on the part for too little time to move it: a 1 mm copper sheet would leave at {{ n.x_flash_speed_mm }} mm/s. A forming pulse lasts about {{ n.x_forming_us }} µs. So "on demand" has to include holding the source {{ n.x_hold_ratio }} times longer than light takes to cross it, by means the premise does not name.

It delivers it badly. The pressure changes by {{ n.x_part_variation }} % across a part 100 mm deep, and it pushes from one side only, so the part is thrown as well as squeezed. The source spends {{ n.x_energy_over_press }} times the {{ n.x_press_mj }} MJ that a coil around the same workpiece would store. And everything in the earlier chapters still applies inside the shell: the molten skin, the weld window with a top, the pores that close without healing. Standing back from a magnetar gives the same 4 GPa, with the same limits.

## What the thought experiment shows

The note's difficulty was never a shortage of field. Granting a hundred million times more does not rescue one of its applications; it removes them. What made the note's ideas interesting was pressure applied to a solid that stays a solid, and that regime ends near a few hundred tesla, where the surface boils.

What an on-demand source of this kind would be good for is physics: dense matter, atoms in needle form, the vacuum itself. That is how the fields beyond 1,000 T are used today, for microseconds, at great expense, in instruments that do not survive.

<div class="table-scroll">
<table>
<thead><tr><th>Gap between the premise and the record</th><th class="r">Factor</th></tr></thead>
<tbody>
{% for g in x.gaps %}
<tr><td>{{ g.label | capitalize }}</td><td class="r">{{ g.value }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

## Check it yourself

```console
$ magform magnetar --field 1e10 --radius 0.01
$ magform magnetar --field 300 --radius 0.01 --material copper
```

```python
from magform import magnetar

source = magnetar.MagnetarSource(1e10, radius=0.01)
source.megatons(), source.confinement_time()          # 59751 Mt, 3.3e-11 s
magnetar.standoff_radius(1e10, 0.01, 100.27)          # 4.6 m: where the field is back to 100 T

a = magnetar.assess_magnetar_mode(1e10, 0.01)
a.status                                              # assumed, never graded
magnetar.classify_regime("iron_armco", 1e10).tier     # degenerate
magnetar.electron_capture_threshold("56Fe").field     # 1.13e10 T
```
