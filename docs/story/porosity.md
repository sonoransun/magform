---
title: Zero porosity
kicker: Chapter 8 · Case 5
doc: MF-08
description: The note would squeeze the pores out of turbine alloys with one cold magnetic pulse. Closing a pore and healing it are different jobs, and the pulse does neither well. The job is already done, hot and slowly, at a fraction of the pressure.
pitch: Every internal pore in a turbine alloy squeezed shut.
brief: A cold pulse does not close small pores and cannot heal them. Hot isostatic pressing does both at a fraction of the pressure.
models: [porosity, eddy, plasticity]
verdict: infeasible
claims: [C23, C24, C25]
---
{% assign n = site.data.n %}

Castings contain pores. Metal shrinks as it freezes, and the last pockets of liquid leave small voids behind, typically half a percent of the volume. In a turbine blade each one is a place for a fatigue crack to start. The note proposes to "completely close all internal porosities" with a 4 GPa magnetic press.

It starts from a true statement. Titanium aluminide and oxide-dispersion-strengthened alloys really are hard to forge. Titanium aluminide stretches {{ n.z_ductility }} % before it breaks and stays brittle up to about {{ n.z_dbtt_c }} °C. Then the argument takes a wrong turn, because it treats a pore as something pressure can simply flatten.

## A small pore is hard to crush

A pore is surrounded by solid metal, and that metal carries the load around it like an arch. To collapse the pore, the pressure outside has to make the whole surrounding shell yield. The smaller the pore fraction, the thicker the shell relative to the hole, and the more pressure it takes.

{% include figure.html id="pore_collapse" caption="<b>Pressure to start collapsing a pore.</b> The curves rise to the left: the last traces of porosity are the hardest to remove. At pulse rates the alloy is stronger and the whole curve moves up, out of the note's window." %}

For half a percent of porosity the threshold is {{ n.z_factor }} times the metal's strength. Loaded slowly, titanium aluminide yields at {{ n.z_strength_slow }} GPa, and its pores begin to close at {{ n.z_threshold_slow }} GPa. But a magnetic pulse is not slow. At pulse rates the alloy's strength is {{ n.z_strength_fast }} GPa and the threshold is {{ n.z_threshold_fast }} GPa, above the top of the note's window. The pores do not start to move.

<aside class="aside">
<p class="label">To close them</p>
<span class="big">{{ n.z_close_fast }} GPa</span>
<p>is the pressure that takes half a percent of porosity down to a hundredth of a percent at pulse rates. Loaded slowly it would be {{ n.z_close_slow }} GPa.</p>
</aside>

The band between the slow and fast figures is the honest answer: at 4 GPa the pores close if the alloy behaves as it does in a slow test, and do not if it behaves as it does in a fast one. At 1.6 GPa they stay open either way. The other turbine alloys are no easier. At 4 GPa and pulse rates Ti-6Al-4V keeps {{ n.z_ti64_left }} % of its 1 % and would need {{ n.z_ti64_need }} GPa; Inconel 718 keeps all of it and would need {{ n.z_in718_need }} GPa.

## Closed is not healed

Suppose a pore does collapse. It takes {{ n.z_collapse_ns }} ns; time is not the obstacle. What is left is two metal faces pressed together. Whether they are *joined* is a separate question, and the answer comes from diffusion: atoms have to cross the interface to make it disappear.

{% include figure.html id="bond_time" caption="<b>Time to bond the faces of a closed pore.</b> At pressing temperature it takes about a second, and the hold lasts hours. At room temperature it does not happen at all." %}

At the temperature of a hot press the faces bond in {{ n.z_hip_bond_s }} s. At room temperature the model's figure is {{ n.z_bond_time }} s, which is a way of saying never. A pore closed cold is left as a flat, unbonded flaw: a crack. For fatigue that is worse than the round pore it replaced.

The metal around the pore fares badly too. To fold a pore shut, its wall has to strain by {{ n.z_wall_strain }}, in an alloy that breaks at {{ n.z_ductility }} %. Pressure helps a brittle metal deform, but the wall of a pore is a free surface where the pressure is zero.

## Reaching the part at all

Two more problems are specific to doing this with a magnet.

Titanium aluminide is a poor conductor, {{ n.z_resistivity }} µΩ·m. At 10 kHz the field soaks {{ n.z_skin_mm }} mm into it. A blade section 2 mm thick is thinner than that, so part of the field passes straight through and the section feels {{ n.z_transfer }} % of the nominal pressure.

And from [chapter 4](forging.md): on a shaped part magnetic pressure is highest on edges and lowest in hollows. A turbine blade is all edges and hollows. An uneven squeeze does not close pores evenly. It bends the blade.

## How it is done today

Cast blades are already freed of porosity, routinely, by hot isostatic pressing. The part sits in argon at {{ n.z_hip_mpa }} MPa and {{ n.z_hip_c }} °C for {{ n.z_hip_h }} hours.

<ul class="stats wide">
  <li class="stat"><div class="stat__label">Pressure</div><div class="stat__value">{{ n.z_hip_mpa }}<span class="stat__unit">MPa</span></div><div class="stat__note">a twenty-third of the note's 4 GPa</div></li>
  <li class="stat"><div class="stat__label">Temperature</div><div class="stat__value">{{ n.z_hip_c }}<span class="stat__unit">°C</span></div><div class="stat__note">the alloy is soft and atoms are mobile</div></li>
  <li class="stat"><div class="stat__label">Hold</div><div class="stat__value">{{ n.z_hip_h }}<span class="stat__unit">h</span></div><div class="stat__note">against {{ n.z_hip_bond_s }} s needed to bond</div></li>
</ul>

Every difficulty above is answered by that recipe. Heat lowers the strength until a modest pressure collapses the pores. Heat and time bond the faces. And the pressure comes from a gas, which pushes equally on every surface of any shape, something a field cannot do. The model rates the standard cycle {{ n.z_hip_status }}.

So the benefit the note claims for turbine parts, longer life from removing porosity, is real and is already collected. The package does not compute fatigue life or service temperature, and the ledger marks that claim as not modelled. It can say that heat tolerance belongs to the alloy, not to its pores.

## What a pulse does compact

One related job suits a pressure pulse: pressing loose powder.

{% include figure.html id="powder_density" caption="<b>Density of a powder after one pulse.</b> Iron powder is fully dense before the note's window begins. The compact still has to be sintered to bond it." %}

Powder particles touch at points and yield easily there. Iron powder reaches {{ n.z_fe_density }} % of solid density at 1.6 GPa, and magnetic compaction of powders at about that pressure is a demonstrated process. Titanium aluminide powder reaches {{ n.z_tial_density }} % at 4 GPa. In both cases the particles are packed, not joined, and the compact goes on to a furnace. That is a useful niche, and it is a long way from a finished blade with no pores.

## Check it yourself

```console
$ magform pores gamma_tial --pressure 4000 --porosity 0.005
$ magform pores gamma_tial --pressure 172 --temperature 1533 --pulse 14400
$ magform pores gamma_tial --pressure 4000 --section 0.002 --frequency 10000
```

```python
from magform import porosity

cold = porosity.assess_pore_closure("gamma_tial", 4e9, initial_porosity=0.005)
cold.closure_status, cold.healing_status          # extrapolation, infeasible
cold.collapse_pressure_dynamic / 1e9              # 4.95 GPa

porosity.hip_cycle("gamma_tial")                  # 172 MPa, 1533 K, 4 h
porosity.diffusion_bonding_time("gamma_tial", 1533.15)    # 1.7 s
```
