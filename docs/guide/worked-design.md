---
title: A worked press design
kicker: Guide
doc: MF-G3
description: One design carried from the pressure on the workpiece to the power supply, in seven short steps. The output under each step is produced by running the code above it.
models: [press, pulse, conductor, fieldmap]
---
{% assign n = site.data.n %}
{% assign w = site.data.worked %}

This page designs the press the concept note asks for, at the top of its window: 4 GPa on a billet in a 100 mm bore. It uses one function, `design_press`, which calls the rest of the package in the right order. The code blocks build on one another, and the text under each is what that code prints.

## 1. State the requirement

A press is specified by the pressure it must deliver and the coil that surrounds the workpiece. The coil here is 100 mm in bore, half a metre long, with 200 mm of winding.

```python
{{ w.spec.code }}
```

```text
{{ w.spec.output }}
```

The pressure fixes the field, and nothing else does. The winding has to carry {{ n.m_j_hi }} A/mm² averaged over its whole cross-section, and the field tries to burst it with a hoop stress close to the magnetic pressure itself. The energy is that of about {{ n.h_energy_tnt }} kg of TNT.

## 2. Ask what the workpiece needs

A field presses on metal only while it is kept out of it. That sets how fast the field must rise, and the model takes it from the workpiece: a skin depth one third of the wall.

```python
{{ w.workpiece.code }}
```

```text
{{ w.workpiece.output }}
```

A solid 80 mm copper billet tolerates a slow rise, about ten milliseconds. The pressure on its flank is a little above nominal, because the billet squeezes the flux into the gap, and several times nominal at its corners. And at 100 T the surface of copper melts. That last line is a property of the field, not of this design.

## 3. Grade the note's magnet

The concept note specifies a superconducting winding reinforced with carbon nanosheets. The design carries three separate assessments of it.

```python
{{ w.magnet.code }}
```

```text
{{ w.magnet.output }}
```

Each of the three is enough on its own. The current density is twice what the tape can carry at this field. No reinforcement, real or hypothetical, keeps the tape within its strain limit. And every pulse deposits more heat than the winding can absorb, so it stops superconducting on every shot. [Chapter 2](../story/machine.md) takes these one at a time.

## 4. Compare the alternatives

The same press can be built in other ways. `design_press` grades six.

```python
{{ w.architectures.code }}
```

```text
{{ w.architectures.output }}
```

The resistive pulsed magnet is the technology that holds the world record. It is rated an extrapolation because the record was set in a {{ n.m_pulsed_bore }} mm bore and this press asks for the same field over {{ n.e_area_ratio }} times the area.

## 5. Size the power supply

For the best architecture the design includes a capacitor-bank network.

```python
{{ w.network.code }}
```

```text
{{ w.network.output }}
```

The bank is twice the size of the largest that drives a magnet today. The last line but one is a warning in disguise: the coil swallows its own field energy as heat and ends the shot far above the 400 K a reinforced coil tolerates, so the energy has to be taken out some other way. [Chapter 3](../story/power.md) covers both.

## 6. Change the workpiece

Now swap the billet for a tube with a 2 mm wall. Nothing about the coil changes.

```python
{{ w.thin.code }}
```

```text
{{ w.thin.output }}
```

The thin wall needs the field to rise four hundred times faster. The energy is the same, so the power is four hundred times higher, and the resistive magnet that was the best choice a moment ago is now infeasible. What is left is a single-turn coil that destroys itself, or an industrial forming coil at half the field. The workpiece, not the magnet, decided that.

## 7. Keep the result

Every part of the design is a dataclass that can print itself or turn into JSON.

```python
{{ w.report.code }}
```

```text
{{ w.report.output }}
```

The same design from the command line is `magform press --pressure 4000`, and with `--json` it prints the whole structure for another program to read.
