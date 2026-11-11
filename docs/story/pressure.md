---
title: Four gigapascals
kicker: Chapter 1
doc: MF-01
description: The concept note rests on one number, 1,600 to 4,000 MPa of magnetic pressure. Here is what that number is, where it sits, and the condition attached to it.
pitch: 1,600 to over 4,000 MPa of magnetic pressure.
brief: The note's pressures exist. The top of its window is the world record for a magnet that survives the shot, and the pressure acts only while the field is changing.
models: [magnetics, benchmarks]
claims: [C01, C03, C04]
verdict: state_of_the_art
---
{% assign n = site.data.n %}

A magnetic field that cannot get into a piece of metal pushes on it. The push is a pressure, and it depends on nothing but the strength of the field. The concept note asks for 1,600 to 4,000 MPa of it, and every process it proposes inherits that range. So the first job is to turn the pressure into the field that makes it.

## The field behind the pressure

For a conductor with the field on one side only, the pressure is the energy density of the field:

$$ P = \frac{B^{2}}{2\mu_{0}} $$

<aside class="aside">
<p class="label">Conversion</p>
<span class="big">{{ n.p_field_hi }} T</span>
<p>is the field whose pressure is 4 GPa. For 1.6 GPa it is {{ n.p_field_lo }} T, for 2 GPa {{ n.p_field_2 }} T.</p>
</aside>

Solve it for the concept note's range and you get {{ n.p_field_lo }} T at the bottom and {{ n.p_field_hi }} T at the top. That is the whole translation. From here on, "1.6 to 4 GPa" and "{{ n.p_field_lo_round }} to {{ n.p_field_hi_round }} tesla" are the same statement.

{% include figure.html id="pressure_field" caption="<b>Pressure against field.</b> Because pressure goes as the square of the field, the step from industrial coils to the concept note's window is larger than it looks: doubling the field quadruples the load on everything, the coil included." %}

Two reference points frame the window. Industrial electromagnetic forming, the process that already shapes and joins metal with pulsed fields, runs at {{ n.p_emf_field_lo }} to {{ n.p_emf_field_hi }} T, which is {{ n.p_emf_pressure_lo }} to {{ n.p_emf_pressure_hi }} GPa. The concept note wants {{ n.p_over_emf_lo }} to {{ n.p_over_emf_hi }} times more pressure than that. At the other end, the strongest field ever produced by a magnet that survived the shot is {{ n.m_pulsed_record }} T, at Los Alamos in 2012, in a bore {{ n.m_pulsed_bore }} mm across. Its pressure is {{ n.p_lanl_pressure }} GPa. The top of the concept note's window is the world record, and the record fits around a pencil.

## How extreme is it?

The note says that at these pressures "we move entirely out of the realm of traditional manufacturing and into extreme high-pressure materials science." The ladder below places the window among pressures that people already work with.

{% include figure.html id="pressure_ladder" caption="<b>Working pressures on a log scale.</b> Each dot is a cited figure from the package's benchmark table. Hover or use the arrow keys for sources." %}

Cold-forging dies see about {{ n.p_die }} GPa of contact pressure in ordinary production. Industrial diamond presses hold 5 to 6.5 GPa for hours. The concept note's window sits between the two. It is a high pressure for a forming process and a low one for high-pressure science: it is below every static route to diamond and cubic boron nitride, a point [chapter 7](synthesis.md) returns to.

It also helps to know what such a pressure does when it acts evenly on all sides of a solid. It squeezes it, elastically, and the solid springs back. Under 4 GPa tungsten loses {{ n.p_compress_w }} % of its volume and copper {{ n.p_compress_cu }} %. Nothing is rearranged. Shaping metal needs the pressure to be uneven, which is the job of a die, a mandrel or a free surface, and that requirement comes back in [chapter 4](forging.md) and [chapter 5](refractory.md).

## The condition attached

The formula above has a quiet assumption in it: the field is on one side of the metal and not the other. The general form is

$$ P = \frac{B_{\text{out}}^{2} - B_{\text{in}}^{2}}{2\mu_{0}} $$

A field holds itself out of a conductor by driving eddy currents in its surface, and those currents die away. Give the field time and it soaks through. Once the field inside equals the field outside, the pressure is zero, however strong the field is.

<aside class="aside">
<p class="label">A steady 100 T field</p>
<span class="big">{{ n.p_static_100 }} Pa</span>
<p>on a stationary, non-magnetic workpiece, once the field has diffused in.</p>
</aside>

A steady magnet is therefore no press at all. To push, the field must rise faster than it can leak into the workpiece, which for millimetre walls of copper means tens of microseconds. That one condition decides most of what follows:

- The magnet has to be **pulsed**, and superconductors do not tolerate fast field changes ([chapter 2](machine.md)).
- The energy in the field has to arrive within the pulse, which sets the size of the **power supply** ([chapter 3](power.md)).
- Any process that needs pressure to be **held**, for minutes or hours, is out of reach ([chapter 7](synthesis.md), [chapter 8](porosity.md)).

## Check it yourself

```python
from magform import magnetics

magnetics.field_for_pressure(4e9)                  # 100.27 T
magnetics.magnetic_pressure(50.0) / 1e9            # 0.99 GPa, the top of industrial forming
magnetics.pressure_with_background(100.0, 0.0)     # 0.0 Pa: a steady field does not push
```

The same from a shell: `magform field --pressure 4000`.
