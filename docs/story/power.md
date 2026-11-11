---
title: The pulse
kicker: Chapter 3
doc: MF-03
description: The concept note asks whether the pulse-discharge network should be mapped out. Here it is, and it turns on one trade between energy and time.
pitch: Homopolar generators or capacitor banks to drive it.
brief: The energy is fixed by field and bore. How fast it must arrive decides whether a power supply exists.
models: [pulse, magnetics, benchmarks]
verdict: extrapolation
claims: [Q1]
---
{% assign n = site.data.n %}

The concept note closes with a question: should the pulse-discharge network, "like homopolar generators or capacitor banks", be mapped out? It should, because the network sets the limit that [chapter 2](machine.md) ran into. This chapter sizes it for the reference press and shows why the answer changes by five orders of magnitude depending on how fast the field has to rise.

## Energy is set by the field and the bore

The energy stored in the field is its pressure times the volume it fills. For the reference press that is {{ n.e_energy_lo }} MJ at 1.6 GPa and {{ n.e_energy_hi }} MJ at 4 GPa. The supply has to deliver all of it on every shot, whatever the workpiece.

{% capture bore_caption %}<b>Stored energy against bore.</b> Shrunk to the record's {{ n.m_pulsed_bore }} mm bore, this design would store {{ n.e_scaled_record }} MJ. At 100 mm it stores {{ n.e_energy_hi }} MJ. At half a metre it would store {{ n.e_half_metre }} GJ, more than the largest pulsed store ever built.{% endcapture %}
{% include figure.html id="energy_bore" caption=bore_caption %}

<aside class="aside">
<p class="label">Largest magnet bank</p>
<span class="big">{{ n.e_bank_sota }} MJ</span>
<p>is the biggest capacitor bank built to drive pulsed magnets. The reference press at 4 GPa needs twice that.</p>
</aside>

Energy alone does not decide much. A hundred megajoules is the energy in three litres of petrol. What matters is how fast it has to arrive.

## Power is energy over rise time

Discharge a capacitor bank into a coil and the current rises along a quarter of a sine wave. For that shape there is a fixed relation between what the bank must supply and the time allowed:

$$ V \times I = \frac{\pi E}{t_{\text{rise}}} \qquad\qquad P_{\text{peak}} = \frac{\pi E}{2\, t_{\text{rise}}} $$

The number of turns in the coil trades voltage against current and leaves their product alone. Wind more turns and the bank needs more volts and fewer amps. Nothing in the coil's design changes the product.

{% capture power_caption %}<b>Peak power against rise time.</b> At a {{ n.m_rise_billet }} ms rise the reference press needs {{ n.e_slow_gw }} GW, more than any magnet bank has delivered and well within what large capacitor banks do. At 25 µs it needs {{ n.e_fast_tw }} TW.{% endcapture %}
{% include figure.html id="power_rise" caption=power_caption %}

<div class="table-scroll wide">
<table>
<thead><tr><th class="r">Field</th><th class="r">Rise</th><th class="r">Turns</th><th class="r">Voltage</th><th class="r">Current</th><th class="r">Peak power</th><th class="r">Bank</th><th>Verdict</th></tr></thead>
<tbody>
{% for r in site.data.power.networks %}
<tr><td class="r">{{ r.field }} T</td><td class="r">{{ r.rise }}</td><td class="r">{{ r.turns }}</td><td class="r">{{ r.kv }} kV</td><td class="r">{{ r.ma }} MA</td><td class="r">{{ r.power }} GW</td><td class="r">{{ r.bank }} MJ</td><td>{% include stamp.html v=r.status %}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

The slow case is a recognisable machine: {{ n.e_slow_turns }} turns, {{ n.e_slow_kv }} kV, {{ n.e_slow_ma }} MA, a {{ n.e_slow_bank }} MJ bank weighing about {{ n.e_slow_mass }} tonnes and switched by {{ n.e_slow_switches }} parallel switches. The fast case needs {{ n.e_fast_kv }} kV and {{ n.e_fast_ma }} MA at once, {{ n.e_fast_tw }} TW. With a 25 kV bank and a single turn, the fastest this coil can be driven is {{ n.e_min_rise }} µs. A 2 mm wall needs ten times faster.

For comparison, the same sizing applied to a single coil with the record's {{ n.m_pulsed_bore }} mm bore gives {{ n.e_energy_insert }} MJ, {{ n.e_lanl_kv }} kV and {{ n.e_lanl_ka }} kA. The real record magnet splits the job in two. Large outer coils, driven for seconds by a {{ n.e_generator_gw }} GW motor-generator, supply the first 40 T. Their field holds {{ n.e_record_outsert }} MJ, against {{ n.e_energy_hi }} MJ for this press, so that much energy is within reach when there is a second in which to deliver it. A billet allows {{ n.m_rise_billet }} milliseconds. A small inner coil, fired from a {{ n.e_record_bank }} MJ capacitor bank, adds the rest in a few milliseconds. The bank for the reference press is {{ n.e_bank_ratio }} times the size of that one.

## What one shot looks like

{% include figure.html id="discharge" caption="<b>Coil current through one shot.</b> At the peak the bank is short-circuited and the current dies away through the coil's own resistance over a few tenths of a second." %}

Short-circuiting the bank at the peak protects the capacitors and gives a single clean pulse. It also means the coil swallows its own field energy as heat. Starting from liquid nitrogen, the winding of the reference press ends a 4 GPa shot at {{ n.e_heat_crowbar }} K, well past the {{ n.e_heat_limit }} K a reinforced coil tolerates; at 1.6 GPa it ends at {{ n.e_heat_lo }} K. At full field the energy has to be taken out of the coil, into a resistor or back into the bank, and that hardware is part of the supply.

## Homopolar generators, and the rest

The note mentions homopolar generators. They store energy in a spinning rotor and deliver very large currents at very low voltage. One sized for the reference press would have a {{ n.e_hpg_mass }} tonne rotor and produce {{ n.e_hpg_volts }} V. Into a single-turn coil, the best case, its current takes {{ n.e_hpg_rise }} ms to peak. That suits a slow outer coil, which is how the record magnet uses a rotating machine. It is {{ n.e_hpg_slow }} times too slow for a thin-wall pulse, and compressing its pulse through an intermediate inductor loses at least three quarters of the energy.

<div class="table-scroll wide">
<table>
<thead><tr><th>Source, for 100 MJ in 25 µs</th><th>Verdict</th><th class="r">Fastest rise at this energy</th><th>Why</th></tr></thead>
<tbody>
{% for s in site.data.power.sources_fast %}
<tr><td><strong>{{ s.name }}</strong>{% unless s.reusable %}<br><span class="muted small">consumed by the shot</span>{% endunless %}</td><td>{% include stamp.html v=s.status %}</td><td class="r">{{ s.fastest }}</td><td class="small">{{ s.reason }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

## The answer to the question

Yes, the network can be mapped, and the map has one feature. Rise time is everything. A press for thick billets, with a field that rises in ten milliseconds, runs from a capacitor bank about twice the size of the largest one built for magnets: an extrapolation. A press for thin walls needs the same energy a thousand times faster, and nothing reusable supplies that.

## Check it yourself

```console
$ magform pulse --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2 --rise 0.01
$ magform pulse --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2 --rise 25e-6
```

```python
from magform import pulse
from magform.magnetics import Solenoid

coil = Solenoid(0.05, 0.25, 0.5)
net = pulse.assess_pulse_network(coil, 100.27, 1e-2)
net.turns, net.charge_voltage, net.peak_current     # 37, 24.9 kV, 1.26 MA
pulse.voltage_current_product(99.5e6, 25e-6)        # 1.3e13 V A, whatever the winding
```
