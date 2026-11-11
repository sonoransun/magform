"""Design the reference press at the top of the concept note's window and print what decides it.

    python examples/reference_press.py
"""

from dataclasses import replace

from magform.press import design_press, reference_spec

design = design_press(reference_spec(pressure=4e9))

print(f"{design.spec.pressure / 1e9:.1f} GPa needs {design.field:.2f} T; the field stores "
      f"{design.stored_energy / 1e6:.1f} MJ and loads the winding to {design.peak_hoop_stress / 1e9:.2f} GPa")
print(f"the workpiece sets a rise time of {design.rise_time * 1e3:.1f} ms\n")

print(f"the note's superconducting magnet: {design.status.label}")
for part, result in (("conductor", design.conductor), ("structure", design.structure),
                     ("pulsed operation", design.pulsed_operation)):
    print(f"  {part:17s} {result.status.label}")

print("\nsix ways of building it:")
for option in design.architectures:
    print(f"  {option.name:45s} {option.status.label:17s} limited by {option.limiting}")
print(f"best: {design.best_architecture} ({design.best_status.label})")

tube = design_press(replace(design.spec, workpiece_wall=0.002))
print(f"\nwith a 2 mm tube instead of a billet the rise time is {tube.rise_time * 1e6:.0f} microseconds "
      f"and the best option is {tube.best_architecture} ({tube.best_status.label})")
