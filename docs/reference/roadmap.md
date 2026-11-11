---
title: Project roadmap
kicker: Reference
doc: MF-R5
description: What version 0.2 of the package covers, what it leaves out on purpose, and what would make its verdicts sharper.
---
{% assign n = site.data.n %}

This page is about the software. For what the numbers suggest building, see [verdict and roadmap](../story/verdict.md).

## What version {{ site.data.meta.version }} covers

- **The press**: thick-solenoid fields on and off the axis, the pressure distribution on a workpiece that excludes the field, REBCO conductor limits, composite reinforcement, pulsed-operation losses, capacitor-bank and homopolar supplies, and six architectures compared on one specification.
- **The workpiece**: field diffusion and skin heating, tube compression, rate-dependent flow stress across a registry of {{ n.ref_material_count }} materials, grain and dislocation structure after a pulse.
- **The applications**: impact welding windows, shock and static synthesis routes, pore collapse and healing, powder compaction, pressure-vessel walls and launchers.
- **The ledger**: {{ n.l_total }} statements from the concept note, each tied to the functions that grade it.
- **The magnetar mode**: a thought experiment with an assumed source, kept apart from the graded results.
- **This site**: {{ site.data.meta.figures }} charts and {{ site.data.meta.numbers }} quoted numbers, all generated.

## Limits that are deliberate

These are engineering estimates, built from closed-form relations and one-dimensional models with stated ranges. They are meant to say which side of a limit a design falls on, and by roughly how much. They are not a substitute for the following, none of which is attempted.

- **Finite-element analysis** of a real coil: turn-by-turn stresses, layer transitions, leads, joints.
- **Coupled simulation** of circuit, field diffusion, heating and motion together. The tube model couples them in one dimension only.
- **Fracture, fatigue and creep.** Claims about toughness and service life are marked as not modelled.
- **Quench propagation** in a superconducting winding. The model stops at whether a pulse exceeds the thermal margin.
- **Cost, schedule and safety engineering.** A status of "extrapolation" says no physical barrier was found, and nothing about price.

## What would sharpen the verdicts

In rough order of how much each would change a result:

1. **Measured data where the model extrapolates.** Critical current of REBCO tape above {{ n.m_je_measured }} T. Flow stress of tungsten and molybdenum at forming rates between room temperature and 500 °C. Hugoniot data for porous graphite, which the shock-synthesis verdict leans on.
2. **A two-dimensional workpiece model**, so that the corner pressures found by the field solver can be followed into deformation instead of stopping at a ratio.
3. **A coupled coil model**: conductor heating with field diffusion in the winding, to replace the adiabatic estimate that decides how much energy must be dumped.
4. **A workpiece-in-circuit model.** A workpiece that excludes the field lowers the coil's inductance and the energy the bank must supply. The present numbers leave it out and so overstate the bank by a modest amount.
5. **Uncertainty propagation.** Material records carry a confidence level. Carrying it through to a band on each margin would say which verdicts are robust and which are close calls.

## Confidence levels

Each material record and benchmark is tagged, and the tag is shown wherever the value is listed.

| Level | Meaning |
|---|---|
| `handbook` | A handbook, datasheet or primary measurement. |
| `representative` | Typical of the literature; a specific product or condition may differ. |
| `reported` | Announced by the institution concerned and not yet peer reviewed. |
| `estimated` | Derived, fitted or judged for this package; treat as order of magnitude. |

## Contributing a correction

The quickest way to improve the package is to replace an estimated value with a measured one. Change the record in `magform/materials.py` or `magform/benchmarks.py`, cite the source in the same record, then run:

```console
$ python -m pytest
$ python tools/build_docs.py
```

If a verdict in the ledger changes, the test that pins the verdict counts will say so, and that is the point of it.
