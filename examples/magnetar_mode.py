"""The thought experiment: a magnetar-strength field on demand, at a source in space.

The source is assumed, not assessed. Everything printed here follows from that premise.

    python examples/magnetar_mode.py
"""

from magform import magnetar
from magform.magnetics import field_for_pressure

FIELD, RADIUS = 1e10, 0.01          # tesla at the surface of the source; metres

source = magnetar.MagnetarSource(FIELD, radius=RADIUS)
print(f"a {RADIUS * 100:.0f} cm source at {FIELD:.0e} T holds {source.megatons():,.0f} megatons "
      f"and disperses in {source.confinement_time() * 1e12:.0f} ps")

print("\nwhat an iron workpiece becomes as the field rises:")
for threshold in sorted(magnetar.regime_thresholds("iron_armco"), key=lambda t: t.field):
    print(f"  {threshold.field:10.3g} T  {threshold.label}")

window = [magnetar.standoff_radius(FIELD, RADIUS, field_for_pressure(p)) for p in (4.0e9, 1.6e9)]
print(f"\nthe concept note's 4 to 1.6 GPa is found {window[0]:.1f} to {window[1]:.1f} m from the source")

assessment = magnetar.assess_magnetar_mode(FIELD, RADIUS)
print(f"status of the source: {assessment.status.value}")
for name, outcome in assessment.processes.items():
    print(f"  {name:20s} {outcome}")
