---
title: The machine
kicker: Chapter 2
doc: MF-02
description: What kind of magnet makes 63 to 100 tesla, why it cannot be the one the concept note describes, and which one it could be.
pitch: A self-supporting superconducting magnet, reinforced with carbon nanosheets.
brief: A superconducting magnet fails on field, stiffness and heat. A resistive pulsed coil is the route that could be built.
models: [press, conductor, fieldmap, magnetics]
verdict: infeasible
claims: [C02]
---
{% assign n = site.data.n %}

The concept note names its magnet in one phrase: self-supporting, carbon-nanosheet-reinforced, superconducting. Each of those words meets a number. This chapter takes them in turn for one concrete machine, a coil with a 100 mm bore, half a metre long, wound 200 mm thick, and then asks what else could make the same field.

## Where superconductors stop

{% capture ladder_caption %}<b>Record fields.</b> Everything above {{ n.m_sc_conductor }} T has been made with copper, in pulses. Above about 100 T the coil does not survive.{% endcapture %}
{% include figure.html id="field_ladder" caption=ladder_caption %}

The strongest magnet built entirely from superconductor makes {{ n.m_sc_record }} T. A superconducting coil has carried current in a field as high as {{ n.m_sc_conductor }} T, but only as an insert inside a large resistive magnet. The note needs {{ n.p_field_lo_round }} to {{ n.p_field_hi_round }} T.

The tape itself is not the first thing to fail. Its ability to carry current falls slowly with field, and a fit to commercial tape gives {{ n.m_je_lo }} A/mm² at {{ n.p_field_lo_round }} T and {{ n.m_je_hi }} at {{ n.p_field_hi_round }} T. Those figures are extrapolations: measurements stop at {{ n.m_je_measured }} T.

{% capture je_caption %}<b>Critical current of the tape.</b> The reference press needs a winding average of {{ n.m_j_lo }} A/mm² at the bottom of the note's window and {{ n.m_j_hi }} at the top. With tape filling 30 % of the winding, the first is just possible on paper and the second is not.{% endcapture %}
{% include figure.html id="je_field" caption=je_caption %}

<aside class="aside">
<p class="label">Computed, not assumed</p>
<span class="big">{{ n.m_end_fraction }}</span>
<p>of the central field cuts across the tape at the ends of this coil, where the tape is weakest. A common rule of thumb says {{ n.m_end_default }}; the field map of the actual winding gives the number.</p>
</aside>

With the tape's margin at {{ n.m_cond_margin_lo }} for 1.6 GPa and {{ n.m_cond_margin_hi }} for 4 GPa, the conductor on its own would rate "extrapolation" at the low end. Two other things rule the magnet out.

## The winding has to hold itself together

A field pushes outward on the coil that makes it with the same pressure it would put on a workpiece. The winding carries that as hoop tension, and making the coil thicker helps less than one would hope.

{% include figure.html id="hoop_ratio" caption="<b>Hoop stress against coil thickness.</b> The peak stress in the winding never falls below half the magnetic pressure. That floor is geometry, not materials." %}

For the reference press the peak hoop stress is {{ n.m_hoop_lo }} to {{ n.m_hoop_hi }} GPa. Strong fibres can carry that. The trouble is what they are wound together with. The superconducting layer in the tape stops working if it is stretched by more than {{ n.m_strain_limit }} %, and everything in a winding stretches together. So the winding must be stiff enough to carry gigapascals while stretching less than half a per cent. That takes a modulus of {{ n.m_modulus_lo }} to {{ n.m_modulus_hi }} GPa.

{% include figure.html id="modulus_gap" caption="<b>Stiffness available against stiffness needed.</b> Half the winding is reinforcement; the rest is tape, copper and insulation. Perfect graphene, which cannot be made in bulk, reaches the 1.6 GPa line and falls far short of the 4 GPa one." %}

A winding reinforced with the stiffest carbon fibre on the market comes to {{ n.m_modulus_m60j }} GPa. The carbon-nanosheet material the note imagines, taken at an optimistic 10 GPa strength, makes a winding of only {{ n.m_modulus_default }} GPa, because strength is not what is missing. Stiffness is, and there the hypothetical material is behind the fibre you can buy.

## It has to be pulsed

[Chapter 1](pressure.md) ended on a condition: a steady field does not push. To press, the field must rise before it soaks into the workpiece, which means about {{ n.m_rise_billet }} ms for a solid copper billet of this size and {{ n.m_rise_tube }} µs for a tube with a 2 mm wall.

For the billet that is a ramp of {{ n.m_ramp_lo }} T/s. The fastest superconducting magnets in service are swept at {{ n.m_ramp_sota }} T/s. The gap matters because a changing field heats a superconductor, and a superconductor near its limit has almost no heat to spare.

{% capture quench_caption %}<b>Heat per pulse against the heat the tape can take.</b> At {{ n.p_field_lo_round }} T each pulse leaves {{ n.m_loss }} MJ in every cubic metre of tape. The tape stops superconducting after {{ n.m_margin }} MJ.{% endcapture %}
{% include figure.html id="quench_margin" caption=quench_caption %}

Each pulse deposits {{ n.m_quench_ratio }} times the heat the winding can absorb. It stops superconducting, a quench, on every shot. Stretching the pulse does not help: the larger part of the loss depends only on how far the field swings.

So the note's magnet fails three separate tests, and the pulsed one would fail it even with a perfect conductor and a perfect structure. A steady version of the same magnet passes that test and exerts no pressure at all.

## Six ways to make the field

The field can still be made. The table grades six ways of doing it for the reference press at 4 GPa, with a solid copper billet as the workpiece.

{% include arch-table.html rows=site.data.architectures.billet_4 %}

**The resistive pulsed magnet** is the one that could be built. It is the technology of the {{ n.m_pulsed_record }} T record: copper-alloy coils, heavily reinforced, cooled in liquid nitrogen and driven in pulses. For this press it needs about {{ n.m_resistive_turns }} turns, {{ n.m_resistive_kv }} kV and {{ n.m_resistive_ma }} MA from a {{ n.m_resistive_bank }} MJ bank. The record magnet has a {{ n.m_pulsed_bore }} mm bore and fires its inner coil from a {{ n.e_record_bank }} MJ bank; this one asks for the same field over {{ n.e_area_ratio }} times the area, from a bank {{ n.e_bank_ratio }} times the size. That is an extrapolation and a large one, with no law of physics in the way.

**The hybrid** looks like a way to keep a superconductor in the design: a steady superconducting outer coil, with a copper insert supplying the pulse. It saves some bank energy and solves nothing. The pulsed insert's flux threads the outer coil, and for an insert this thick it is {{ n.m_hybrid_flux }} times the outer coil's own flux. A superconducting loop holds its flux constant, so the pulse would drive the outer coil's current through zero and out the other side, {{ n.m_hybrid_dip }} T of swing in milliseconds.

**A single-turn coil** reaches these fields easily, for a few microseconds, and is blown apart every time. At this bore it would take {{ n.m_single_ma }} MA at {{ n.m_single_kv }} kV. **An industrial forming coil** is real and reusable and stops at 50 T, which is {{ n.m_emf_pct_hi }} % of the pressure asked for.

## The conflict that decides it

The table above is kind to the machine, because a solid billet is a patient workpiece. Most of what industry forms with magnetic pulses is thin: tubes, sheet, flyers. A 2 mm copper wall needs the field up in {{ n.m_rise_tube }} µs or the field leaks through it.

{% include arch-table.html rows=site.data.architectures.tube_4 %}

At that speed the resistive magnet needs {{ n.m_tube_kv }} kV and {{ n.m_tube_ma }} MA through a single turn, which no bank delivers and no solid conductor survives. The fastest rise within demonstrated practice is {{ n.m_tube_relaxed_rise }} ms, and at that rise the wall passes {{ n.m_tube_relaxed_pct }} % of the pressure. The workpiece wants microseconds and the power supply wants milliseconds. [Chapter 3](power.md) shows where that limit comes from.

## Check it yourself

```console
$ magform press --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2
$ magform architectures --pressure 4000 --bore 0.1 --length 0.5 --winding 0.2 --wall 0.002
```

```python
from magform import design_press, reference_spec

design = design_press(reference_spec(4e9))
design.status                      # infeasible: the concept magnet
design.best_architecture           # 'pulsed_resistive'
design.structure.required_composite_modulus / 1e9    # 1373 GPa
```
