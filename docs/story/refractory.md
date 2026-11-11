---
title: Tungsten, cold
kicker: Chapter 5 · Case 2
doc: MF-05
description: Can two gigapascals forge tungsten, molybdenum and rhenium at room temperature, and what would that do to the metal? This chapter also answers the note's question about microstructure.
pitch: Tungsten, molybdenum and rhenium forged at room temperature.
brief: Two gigapascals does not move cold tungsten, but it does move warm tungsten. The grains survive and the skin does not.
models: [plasticity, microstructure, eddy]
verdict: extrapolation
claims: [C10, C11, C12, C13, Q2]
---
{% assign n = site.data.n %}

The note's second process is the one with the most to it. Refractory metals are strong, brittle when cold, and awkward to work. If pressure alone could shape them at room temperature, that would be useful. The claim has four parts: how these metals are forged today, whether 2 GPa can do it cold, what happens to the grain structure, and what that does to toughness.

## How they are forged today

The note says forging tungsten, molybdenum and rhenium "typically requires heating them to near-melting temperatures (>2,500 °C)". It does not.

<div class="table-scroll">
<table>
<thead><tr><th>Metal</th><th class="r">Melts at</th><th>Worked at</th><th>Practice</th></tr></thead>
<tbody>
{% for m in site.data.refractory %}
<tr><td>{{ m.name }}</td><td class="r">{{ m.melting_c }} °C</td><td>{{ m.practice_range }} °C</td><td class="small">{{ m.practice }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

Tungsten is forged at {{ n.r_w_conv_lo }} to {{ n.r_w_conv_hi }} °C, which is {{ n.r_w_conv_frac_lo }} to {{ n.r_w_conv_frac_hi }} of its melting point. Rhenium, tantalum and niobium are rolled cold. Nothing is worked above 2,500 °C; molybdenum melts at {{ n.r_mo_melt_c }} °C. The problem the note sets out to solve is smaller than it says.

## Is two gigapascals enough?

Forging needs the applied pressure to exceed the metal's flow stress, with a margin for the constraint of the die. Flow stress rises with speed and falls with temperature, and a magnetic pulse is fast.

{% include figure.html id="flow_stress_temperature" caption="<b>Pressure needed against billet temperature.</b> The band for tungsten spans two published sets of measurements. At room temperature it lies wholly above the 2 GPa line." %}

At room temperature and forming rates, tungsten's flow stress is {{ n.r_w_stress_lo }} to {{ n.r_w_stress_hi }} GPa. With the die's constraint the press needs {{ n.r_w_needed }} GPa. Two gigapascals is {{ n.r_w_margin }} of that: the billet does not move.

<aside class="aside">
<p class="label">Moderate heat</p>
<span class="big">{{ n.r_w_warm_c }} °C</span>
<p>is where the weaker of the two tungsten data sets comes down to 2 GPa. An oven, not a furnace.</p>
</aside>

The note hedges with "or moderate heat", and that hedge is right. From about {{ n.r_w_warm_c }} °C tungsten begins to yield at 2 GPa. Molybdenum just yields cold, with a margin of {{ n.r_mo_margin }}. Tantalum and niobium yield easily, which is no news: they are cold-worked already. At 4 GPa tungsten yields at room temperature with a margin of {{ n.r_w_margin_4 }}.

<div class="table-scroll">
<table>
<thead><tr><th>Metal</th><th class="r">Flow stress</th><th class="r">Pressure needed</th><th>At 2 GPa, cold</th><th>At 4 GPa, cold</th><th class="r">Yields at 2 GPa from</th></tr></thead>
<tbody>
{% for m in site.data.refractory %}
<tr><td>{{ m.name }}<br><span class="muted small">{{ m.model }}</span></td><td class="r">{{ m.stress }} GPa</td><td class="r">{{ m.needed }} GPa</td><td>{% include stamp.html v=m.status_2 %}</td><td>{% include stamp.html v=m.status_4 %}</td><td class="r">{% if m.warm %}{{ m.warm }} °C{% else %}room temperature{% endif %}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

Nothing in that table is rated "state of the art", for two reasons that pressure does not remove.

**Brittleness.** Below about {{ n.r_w_dbtt_c }} °C tungsten is brittle. Compressed at high rate without support it cracks at {{ n.r_w_fail }} % strain, long before a forging strain of 30 %. A large all-round pressure suppresses cracking, and the model credits that when the pressure exceeds the flow stress. It is a real effect, known since Bridgman, and it has not been shown on a component at these rates. Without the confinement the same billet is rated {{ n.r_w_bare_status }}.

**Localisation.** A fast deformation has no time to shed its heat. The metal softens where it is deforming, deforms more there, and the strain collapses into narrow bands. For tungsten at these rates the model puts the onset at a strain of {{ n.r_w_culver }}.

And as always, the pressure has to be uneven to shape anything. Evenly applied, 2 GPa compresses tungsten by {{ n.r_w_compress }} % and lets it spring back.

## What it does to the metal

The note's last question asks how cold forging affects the microstructure. For tungsten strained by 0.3 in thirty microseconds, the model gives this.

<ul class="stats wide">
  <li class="stat"><div class="stat__label">Dislocation density</div><div class="stat__value">×{{ n.r_dislocation_factor }}</div><div class="stat__note">to {{ n.r_dislocation }} per m², in cells {{ n.r_cell }} µm across</div></li>
  <li class="stat"><div class="stat__label">Yield strength</div><div class="stat__value">+{{ n.r_hardening }}<span class="stat__unit">%</span></div><div class="stat__note">{{ n.r_yield_before }} to {{ n.r_yield_after }} GPa</div></li>
  <li class="stat"><div class="stat__label">Bulk temperature rise</div><div class="stat__value">{{ n.r_w_heat }}<span class="stat__unit">K</span></div><div class="stat__note">to {{ n.r_bulk_frac }} of the melting point</div></li>
  <li class="stat"><div class="stat__label">Fraction recrystallised</div><div class="stat__value">0</div><div class="stat__note">the model says {{ n.r_rex }}</div></li>
</ul>

So the note is half right about grain structure. The interior never gets warm enough to recrystallise, and the grains the billet started with survive. That is the "preserves" part, and it is ordinary cold work.

{% capture grain_caption %}<b>Grain size against strain.</b> A forming pulse flattens the original grains from {{ n.r_grain_before }} to {{ n.r_grain_after }} µm. Ultra-fine grains are made by straining a metal many times over, in rolling or in special presses.{% endcapture %}
{% include figure.html id="grain_strain" caption=grain_caption %}

They are not "ultra-fine", though, unless the billet was. One pulse flattens a {{ n.r_grain_before }} µm grain to {{ n.r_grain_after }} µm. Getting below a micrometre takes a strain of about {{ n.r_refine_strain }}, ten forming pulses' worth, which is how ultra-fine tungsten is actually made.

Nor are they "pristine". The skin of the billet is heated by the eddy currents of [chapter 4](forging.md). At the {{ n.r_field_2 }} T of a 2 GPa press tungsten's skin reaches {{ n.r_skin_2 }} K, {{ n.r_skin_2_frac }} of its melting point. At 100 T it reaches {{ n.r_skin_4 }} K, half the melting point, for {{ n.r_skin_4_ms }} ms, over a depth of about {{ n.r_haz_depth }} mm. That layer is a heat-affected zone by any definition.

## Toughness and fatigue

The package does not compute fracture toughness or fatigue life, and the ledger says so. What it does compute does not point towards "unprecedented". Cold work raises strength and spends ductility. Shear bands are damage. The well-known case of tough tungsten, heavily rolled sheet with a low transition temperature, gets there through rolling strains above three, not through one pulse.

## Check it yourself

```console
$ magform forge tungsten --pressure 2000
$ magform forge tungsten --pressure 2000 --temperature 520
$ magform micro tungsten --field 100
```

```python
from magform import plasticity, microstructure

cold = plasticity.assess_forgeability("tungsten", 2e9)
cold.status, cold.warm_forming_temperature        # infeasible, 460 K

grains = microstructure.assess_microstructure("tungsten", peak_field=100.27)
grains.grain_structure_preserved, grains.ultrafine    # True, False
```
