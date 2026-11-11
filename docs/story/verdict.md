---
title: Verdict and roadmap
kicker: Chapter 10
doc: MF-10
description: Thirty statements from the concept note, each graded against a model. What holds, what does not, and what a programme built on the parts that hold would look like.
brief: Twelve of thirty statements hold or would hold with more engineering. Fifteen do not. The parts that hold point to a different machine at lower fields.
models: [claims, press]
---
{% assign n = site.data.n %}
{% assign cl = site.data.claims %}

The concept note makes {{ n.l_claims }} statements that can be checked and asks {{ n.l_questions }} questions. Each one has an entry in the [claims ledger](../ledger.md): the passage as written, a verdict, the numbers behind the verdict, and the functions that produced them.

## The count

{% include segbar.html %}

<div class="table-scroll wide">
<table>
<thead><tr><th>Verdict</th><th class="r">Entries</th><th>What it means</th><th>Which</th></tr></thead>
<tbody>
{% for v in cl.counts %}
<tr><td>{% include stamp.html v=v.verdict %}</td><td class="r">{{ v.count }}</td><td class="small">{{ v.meaning }}</td><td class="small">{% for id in cl.order %}{% assign c = cl.by_id[id] %}{% if c.verdict == v.verdict %}<a href="{{ '/ledger.html' | relative_url }}#claim-{{ id }}" title="{{ c.topic }}">{{ id }}</a> {% endif %}{% endfor %}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

{{ n.l_holds }} of the {{ n.l_total }} hold today or would hold with more engineering. {{ n.l_false }} are false as stated or conflict with the physics. One needs a material that does not exist, and two are about properties this package does not compute.

Those totals flatter the note a little. The statements that hold are mostly about processes that exist already, at lower fields. The statements that fail are mostly the ones that made the note new.

## Three findings

**The pressure is real, and a steady field supplies none of it.** A {{ n.p_field_hi_round }} T field does exert 4 GPa, but only on metal it has not yet soaked into. Hold the field and the pressure decays to nothing ([chapter 1](pressure.md)). Everything else follows from that. The magnet must be pulsed, the energy must arrive within the pulse, and no process that needs pressure *held* is available.

**The machine that fits is not the machine proposed.** A superconducting magnet fails three independent tests: field, stiffness and heat ([chapter 2](machine.md)). What reaches 100 T is a set of resistive coils driven in pulses, and it does so in a {{ n.m_pulsed_bore }} mm bore. Widening that to 100 mm is engineering on a large scale, not new physics, and it is the best-rated route in the comparison below.

{% include arch-table.html rows=site.data.architectures.billet_4 %}

**The applications want less pressure, or heat, or time.** Welds form at {{ n.w_field_min }} to {{ n.w_field_max }} T and are destroyed above that. Forming works at {{ n.p_emf_field_lo }} to {{ n.p_emf_field_hi }} T. Pores are healed by {{ n.z_hip_mpa }} MPa held hot for hours. Diamond needs at least {{ n.s_p_min }} GPa together with 1,500 K or more, held for a second or longer. Launchers are limited by their payloads. In no case is the missing ingredient a stronger field.

## What the note gets right

It would be a poor review that only counted errors. The note is right about these.

- **Magnetic pressure of this size exists.** The top of its window is the world record for a magnet that survives ([C01]({{ '/ledger.html' | relative_url }}#claim-C01)).
- **Dissimilar metals can be welded cold**, with almost no heat-affected zone and with the brittle phases cut by orders of magnitude ([C14]({{ '/ledger.html' | relative_url }}#claim-C14), [C17]({{ '/ledger.html' | relative_url }}#claim-C17), [C18]({{ '/ledger.html' | relative_url }}#claim-C18)).
- **"Or moderate heat."** The hedge on tungsten is correct. At 2 GPa it begins to yield from about {{ n.r_w_warm_c }} °C ([C11]({{ '/ledger.html' | relative_url }}#claim-C11)).
- **Cold forming keeps the grain structure** the billet started with ([C12]({{ '/ledger.html' | relative_url }}#claim-C12)).
- **A magnet's reinforcement and a pressure vessel are the same problem** ([C26]({{ '/ledger.html' | relative_url }}#claim-C26)).
- **Its two closing questions were the right ones.** The pulse network decides what can be built, and the microstructure decides what the product is worth ([Q1]({{ '/ledger.html' | relative_url }}#claim-Q1), [Q2]({{ '/ledger.html' | relative_url }}#claim-Q2)).

## A programme that follows from the numbers

If the aim is to get the most out of magnetic pressure in manufacturing, the results sort into steps. Each is rated the way the claims are.

<div class="table-scroll wide">
<table>
<thead><tr><th>Step</th><th>What the model says</th><th>Rating</th></tr></thead>
<tbody>
<tr><td><b>Use what works.</b> Pulse welding and electromagnetic forming.</td><td class="small">Aluminium to steel welds at {{ n.w_field_min }} to {{ n.w_field_max }} T. Tubes close at 30 T. No new magnet is needed. <a href="{{ '/story/welding.html' | relative_url }}">Chapter 6</a></td><td>{% include stamp.html v="state_of_the_art" %}</td></tr>
<tr><td><b>Compact powders</b> at the bottom of the window.</td><td class="small">Iron powder reaches {{ n.z_fe_density }} % of solid density at 1.6 GPa, then goes to a furnace to be sintered. <a href="{{ '/story/porosity.html' | relative_url }}">Chapter 8</a></td><td>{% include stamp.html v="state_of_the_art" %}</td></tr>
<tr><td><b>Warm-form refractory metals</b> with a pulse.</td><td class="small">Tungsten yields at 2 GPa from {{ n.r_w_warm_c }} °C, against {{ n.r_w_conv_lo }} to {{ n.r_w_conv_hi }} °C in today's practice. Needs proof on a real part, at rate, without cracking. <a href="{{ '/story/refractory.html' | relative_url }}">Chapter 5</a></td><td>{% include stamp.html v="extrapolation" %}</td></tr>
<tr><td><b>Build the wide-bore pulsed press.</b> Resistive, 100 mm, 100 T.</td><td class="small">{{ n.m_resistive_turns }} turns, {{ n.m_resistive_kv }} kV, {{ n.m_resistive_ma }} MA and a {{ n.m_resistive_bank }} MJ bank, twice the largest that drives a magnet today. It rises in {{ n.m_rise_billet }} ms, which suits thick billets only, and at full field the surfaces of aluminium, copper and titanium melt ({{ n.f_melt_al }}, {{ n.f_melt_cu }} and {{ n.f_melt_ti }} T), so its top end is for steel and the refractory metals. <a href="{{ '/story/power.html' | relative_url }}">Chapter 3</a></td><td>{% include stamp.html v="extrapolation" %}</td></tr>
<tr><td><b>Shock synthesis by magnetic flyer.</b></td><td class="small">At 4 GPa a copper plate makes {{ n.s_shock_graphite }} GPa in graphite, the threshold for shock diamond. The product is powder. <a href="{{ '/story/synthesis.html' | relative_url }}">Chapter 7</a></td><td>{% include stamp.html v="extrapolation" %}</td></tr>
<tr><td><b>Research the materials</b> the note assumed.</td><td class="small">A winding stiffness of {{ n.m_modulus_lo }} to {{ n.m_modulus_hi }} GPa and a superconductor that carries current above {{ n.m_sc_conductor }} T. Neither exists. <a href="{{ '/story/machine.html' | relative_url }}">Chapter 2</a></td><td>{% include stamp.html v="aspirational" %}</td></tr>
</tbody>
</table>
</div>

Four things are on no path, because the obstacle is physics and not engineering: pressing with a steady field, pulsing an all-superconducting magnet to 100 T, making diamond with a pressure pulse and no heat, and healing pores cold.

## One further premise

After the note was graded, a different question was put to the model: what if a field of magnetar strength were available on demand, in space? That premise is assumed, not assessed, and it changes none of the verdicts above. What follows from it is in [chapter 11](magnetar.md).

## How to disagree with this review

Every number on this site is computed by the package and can be regenerated with one command. So the useful way to disagree is to find the assumption you reject and change it.

- The models, their equations and their limits are listed under [models and validity](../reference/methods.md).
- Material data, with sources and confidence levels, are under [materials](../reference/materials.md).
- The records each verdict is measured against are under [benchmarks](../reference/benchmarks.md).

Three things the review does not cover at all: cost, fatigue and fracture toughness, and anything that depends on the detailed design of a real coil. A verdict of "extrapolation" means no barrier was found. It does not mean the thing is cheap or soon.

```console
$ magform claims                  # all thirty verdicts
$ magform claims C02         # one entry with its evidence
$ python tools/build_docs.py      # regenerate every number on this site
```
