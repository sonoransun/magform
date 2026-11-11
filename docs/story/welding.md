---
title: Welding by impact
kicker: Chapter 6 · Case 3
doc: MF-06
description: Joining metals that will not weld by heat is the note's strongest claim, because it is already an industry. The catch is that it works well below the note's pressures and fails at the note's own.
pitch: Dissimilar metals driven together until their lattices interlock.
brief: Dissimilar metals already weld cold, well below the note's pressures. At the note's own pressures the same joint fails.
models: [welding, shock, eddy]
verdict: state_of_the_art
claims: [C14, C15, C16, C17, C18]
---
{% assign n = site.data.n %}

Aluminium and steel will not fusion-weld. Melt them together and they form iron-aluminium compounds as brittle as glass. The note proposes driving dissimilar metals together "with such extreme, instantaneous pressure that their atomic lattices are forced to interlock", leaving no heat-affected zone and none of those compounds.

Something very like that is sold today under the name magnetic pulse welding. It is worth being exact about what it is, because the working process and the note's description part company on the one thing the note is about: pressure.

## How a pulse weld is made

A coil throws a thin sheet, the flyer, across a small gap at a stationary part. The sheet strikes at an angle, so the line of contact sweeps along the joint. Ahead of that line, metal from both surfaces is squirted out as a jet, taking oxides and dirt with it, and clean metal meets clean metal under several gigapascals. The bond is formed in the solid.

For a 1 mm aluminium sheet thrown across 2 mm at steel, the model gives this.

<ul class="stats wide">
  <li class="stat"><div class="stat__label">Coil field</div><div class="stat__value">{{ n.w_field }}<span class="stat__unit">T</span></div><div class="stat__note">{{ n.w_drive }} MPa of magnetic pressure</div></li>
  <li class="stat"><div class="stat__label">Impact speed</div><div class="stat__value">{{ n.w_velocity }}<span class="stat__unit">m/s</span></div><div class="stat__note">after {{ n.w_flight }} µs of flight</div></li>
  <li class="stat"><div class="stat__label">Pressure at impact</div><div class="stat__value">{{ n.w_impact }}<span class="stat__unit">GPa</span></div><div class="stat__note">{{ n.w_amplification }} times the pressure that drove it</div></li>
</ul>

The gap is the trick. Magnetic pressure of {{ n.w_drive }} MPa acts on the sheet for ten microseconds and gives it momentum; the impact spends that momentum in a fraction of a microsecond. The joint sees {{ n.w_impact }} GPa, squarely in the note's window, from a coil at {{ n.w_field }} T. The extreme pressure is real. It comes from a modest magnet and a 2 mm gap.

## A window, with a top

Impact welds form only inside a window of speed and angle, mapped for explosive welding half a century ago.

{% include figure.html id="weld_window" caption="<b>The weld window for aluminium on steel.</b> The green band is where a bond forms. The lowest dot is a working weld. The two above it are the same joint driven at the note's pressures." %}

Too slow, below {{ n.w_vmin }} m/s here, and the surfaces do not flow. Too fast, above {{ n.w_vmax }} m/s, and the interface melts. And the contact line must travel slower than sound in both metals, {{ n.w_sonic }} m/s here, or no jet can form ahead of it.

Drive the same sheet at the note's 1.6 GPa and it lands at {{ n.w_v_lo }} m/s. At 4 GPa, {{ n.w_v_hi }} m/s. The contact line then moves at {{ n.w_vc_lo }} to {{ n.w_vc_hi }} m/s, two to three times the speed of sound. There is no jet, the interface melts, and nothing welds.

{% capture drive_caption %}<b>Drive pressures that weld.</b> The highest of them, {{ n.w_drive_top }} MPa for a copper flyer, is still {{ n.w_factor_lo }} times below the bottom of the note's window.{% endcapture %}
{% include figure.html id="weld_drive" caption=drive_caption %}

One could shrink the gap to slow the sheet down. At 4 GPa the gap that lands it in the window is {{ n.w_gap_min }} to {{ n.w_gap_max }} µm, the thickness of a hair, crossed in under half a microsecond. That is not a process anyone can set up.

So the note's mechanism is backwards. More pressure does not force lattices to interlock. Past the top of the window, more pressure destroys the weld.

## Heat and brittle phases

On the two metallurgical claims the note does better.

**No heat-affected zone.** At {{ n.w_field }} T the flyer's surface warms by {{ n.w_skin }} K. The layer heated at the interface is about {{ n.w_melt_layer }} µm thick and cools in a microsecond. Against the millimetres of a fusion weld that is as near to "none" as makes no difference. The claim fails only at the note's own fields: above {{ n.w_skin_melt }} T an aluminium flyer's skin melts before it arrives.

**No brittle compounds.** The compounds grow by diffusion, and diffusion needs time.

<aside class="aside">
<p class="label">Atoms on the move</p>
<span class="big">{{ n.w_diff_solid }} nm</span>
<p>of interdiffusion in the solid during a pulse weld. A fusion weld that stays liquid for a second manages {{ n.w_diff_fusion }} µm.</p>
</aside>

In the microsecond the interface stays hot, atoms move about {{ n.w_diff_solid }} nm in the solid. In a fusion weld they move {{ n.w_diff_fusion }} µm. The brittle layer is thinner by four or five orders of magnitude. It is not absent: the same phases ({{ n.w_phases }}) turn up in the small pockets where the interface did melt, and a weld run too hot has a continuous layer of them. "Bypasses" is fair for a weld inside its window.

## Large and complex

The note says the bond forms "across large, complex geometric interfaces". At a 10° angle the usable speeds span a factor of {{ n.w_width }}, and below {{ n.w_angle_min }}° there is no jet at all, so a flat-on impact never welds. Speed and angle both change along a joint as the flyer bends. Pulse welds are therefore narrow: lap seams and tube joints a few millimetres wide. Large flat areas are clad with explosives, which run the same physics at a larger scale. Complex shapes are beyond both.

## What survives

Almost everything, at the wrong pressure. Dissimilar metals are welded cold, with no heat-affected zone to speak of and brittle phases cut by orders of magnitude. It takes {{ n.w_field_min }} to {{ n.w_field_max }} T for aluminium on steel and {{ n.w_cuti_field }} T for copper on titanium. The 63 to 100 T machine of the note would be the wrong tool for the one job in its list that already works.

## Check it yourself

```console
$ magform weld al steel                   # the drive window that welds
$ magform weld al steel --pressure 100    # a working weld
$ magform weld al steel --pressure 4000   # the note's pressure
```

```python
from magform import welding

good = welding.assess_weld("aluminum_6061", "steel_1018", 1e-3, 2e-3, 0.1e9)
good.status, good.impact_velocity, good.impact_pressure / 1e9    # state of the art, 385 m/s, 4.1 GPa

welding.assess_weld("aluminum_6061", "steel_1018", 1e-3, 2e-3, 4e9).status    # infeasible
```
