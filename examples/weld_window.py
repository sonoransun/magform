"""Sweep the drive pressure for an aluminium-to-steel pulse weld and show where the window is.

    python examples/weld_window.py
"""

from magform import welding
from magform.magnetics import field_for_pressure

FLYER, TARGET, THICKNESS, GAP = "aluminum_6061", "steel_1018", 1e-3, 2e-3

window = welding.drive_pressure_window(FLYER, TARGET, THICKNESS, GAP)
print(f"drive window: {window.pressure_min / 1e6:.0f} to {window.pressure_max / 1e6:.0f} MPa "
      f"({window.field_min:.0f} to {window.field_max:.0f} T)\n")

print(f"{'drive':>10s} {'field':>8s} {'impact':>10s} {'at the joint':>13s}  verdict")
for pressure in (0.05e9, 0.1e9, 0.2e9, 0.4e9, 1.6e9, 4.0e9):
    weld = welding.assess_weld(FLYER, TARGET, THICKNESS, GAP, pressure)
    print(f"{pressure / 1e6:8.0f} MPa {field_for_pressure(pressure):6.1f} T {weld.impact_velocity:8.0f} m/s "
          f"{weld.impact_pressure / 1e9:9.1f} GPa  {weld.status.label}")
