---
title: Forging without a die
kicker: Chapter 4 · Case 1
doc: MF-04
description: The note's first process pulses a field around a workpiece and lets eddy currents crush it. The process is real and decades old. The three advantages claimed for it at these fields are not.
pitch: Eddy currents crush the part evenly, with no die and no friction.
brief: Electromagnetic forming is real and old. Uniform pressure, even heating and no tool wear are not what it gives at these fields.
models: [eddy, fieldmap, plasticity]
verdict: contradicted
claims: [C06, C07, C08, C09, C05]
---
{% assign n = site.data.n %}

Pulse a strong field around a piece of metal and currents flow in its surface to keep the field out. Those currents sit in the field that drives them, so the surface is pushed inward. The concept note calls this "Dynamic Magneto-Hydrodynamic Forging" and presents it as new. Industry calls it electromagnetic forming and has used it since the 1960s to crimp tubes, shape sheet and join parts.

## What already exists

Industrial coils run at {{ n.p_emf_field_lo }} to {{ n.p_emf_field_hi }} T. The package models the textbook case: a copper tube 50 mm across with a 2 mm wall, in a field that rises in {{ n.f_rise }} µs, the rise that keeps the field out of a wall that thick.

{% include figure.html id="tube_radius" caption="<b>Tube compression at four fields.</b> At 20 T the wall moves, slows and stops. From 30 T upward the bore closes completely. The note's window begins at the third curve." %}

At 20 T the tube shrinks to a strain of {{ n.f_strain_20 }} and stops. At 30 T the bore closes in {{ n.f_close_30 }} µs with the wall moving at {{ n.f_v30 }} m/s. So the mechanism the note describes works, and works well below its window.

At {{ n.p_field_lo_round }} T, the bottom of that window, the same tube closes in {{ n.f_close_63 }} µs at {{ n.f_v63 }} m/s. At {{ n.p_field_hi_round }} T it reaches {{ n.f_v100 }} m/s. These are not forging speeds. They are the speeds of a shaped charge, and the tube arrives at its own axis as a jet.

<aside class="aside">
<p class="label">The name</p>
<p>Magnetohydrodynamics is about conducting <em>fluids</em>. The workpiece here is solid, unless its surface melts, which for copper happens above {{ n.f_melt_cu }} T. Only there does the name describe the process.</p>
</aside>

## "Perfectly uniform"

The note's first advantage is that magnetic pressure is "perfectly uniform across the entire surface area". The pressure at each point is set by the field at that point, and the field is shaped by the coil and by the workpiece itself. The package solves for the field around a billet that keeps the field out, and reads the pressure off its surface.

{% include figure.html id="surface_pressure" caption="<b>Pressure along a billet.</b> Level through the middle, falling towards the ends of the coil, and rising steeply at the billet's own corners, where the field crowds round the edge." %}

For a billet well inside the coil the middle is good: within {{ n.f_uniformity }} % over the central four-fifths, and slightly above the nominal value because the billet squeezes the flux into the gap. Away from the middle it is not uniform at all. A billet as long as the coil sees {{ n.f_long_min }} % of nominal pressure at the coil's ends. And the billet's corners see {{ n.f_corner }} times nominal, or {{ n.f_corner_round }} times with a generous 10 mm radius.

That last number is the general rule. Magnetic pressure is highest on convex edges and lowest in recesses. On a part with any shape to it, the load is nothing like even, and it is never the all-round squeeze of a fluid.

## "Eliminates uneven cooling"

The currents that carry the force also heat the metal they flow in, and they flow only in a thin surface layer. The energy dumped in that layer is about equal to the energy density of the field, which is the pressure. So the temperature rise of the skin is the magnetic pressure divided by the metal's heat capacity, and it does not depend on how long the pulse lasts.

{% include figure.html id="skin_heating" caption="<b>Skin heating against field.</b> In the note's window the surface of aluminium is already molten and copper is on the way. The inside of the part stays cold." %}

Copper's skin heats by {{ n.f_skin_cu_30 }} K at 30 T, which is why industrial forming is a cold process. At {{ n.p_field_lo_round }} T it heats by {{ n.f_skin_cu_lo }} K and at {{ n.p_field_hi_round }} T by {{ n.f_skin_cu_hi }} K. Aluminium's skin melts at {{ n.f_melt_al }} T, below the bottom of the window. A full diffusion calculation for copper at 100 T puts the surface at {{ n.f_slab_surface }} K and the molten layer {{ n.f_slab_melt_depth }} mm deep.

A cold core under a surface heated by a thousand degrees in microseconds is the most uneven heating a forging could have.

## "Eliminates friction and tool wear"

There is no friction against the field. But a uniform squeeze does not shape metal ([chapter 1](pressure.md)); a die or a mandrel does, and the workpiece slides on it as it always has.

Tool wear does not vanish either. It moves. The coil feels the same pressure as the workpiece on every shot, and the coil is the expensive part. Industrial coils last because they work at a few hundred megapascals. The magnets that reach 100 T are rebuilt after hundreds of shots, and the simplest ones are single-use by design ([chapter 2](machine.md)).

## What survives

The process is real, and the note is right that it needs no punch. Everything it claims as an edge over die forging belongs to the low-field version that already exists. Raising the field from 50 T into the note's window buys closing speeds past a kilometre per second, a molten surface, and a coil that wears out, in exchange for pressure that most forming does not need.

## Check it yourself

```python
from magform import eddy, fieldmap
from magform.magnetics import Solenoid

tube = eddy.TubeCompression("copper_ofhc", outer_radius=0.025, wall=0.002)
pulse = eddy.half_sine_field(63.41, 51.3e-6)
tube.simulate(pulse, 205e-6).summary.peak_velocity          # 1260 m/s

eddy.skin_temperature_rise("copper_ofhc", 100.27)           # +1160 K
eddy.skin_melting_field("aluminum_6061")                    # 58 T

press = Solenoid(0.05, 0.25, 0.5)
fieldmap.workpiece_pressure(press, 0.04, 0.30).peak_ratio   # 5.7 at the corners
```
