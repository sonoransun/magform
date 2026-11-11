"""Physical constants and unit helpers (CODATA 2018 values, SI units)."""

from __future__ import annotations

import math

# --- fundamental constants (CODATA 2018) ------------------------------------
MU0: float = 1.25663706212e-6
"""Vacuum permeability, H/m."""

EPS0: float = 8.8541878128e-12
"""Vacuum permittivity, F/m."""

C_LIGHT: float = 299_792_458.0
"""Speed of light in vacuum, m/s."""

K_B: float = 1.380649e-23
"""Boltzmann constant, J/K."""

R_GAS: float = 8.314462618
"""Molar gas constant, J/(mol K)."""

N_A: float = 6.02214076e23
"""Avogadro constant, 1/mol."""

E_CHARGE: float = 1.602176634e-19
"""Elementary charge, C."""

G_ACCEL: float = 9.80665
"""Standard gravity, m/s^2."""

# --- reference conditions ---------------------------------------------------
T_ROOM: float = 293.15
"""Reference room temperature, K (20 C)."""

T_LIQUID_HELIUM: float = 4.2
"""Boiling point of helium at 1 atm, K."""

T_LIQUID_NITROGEN: float = 77.0
"""Boiling point of nitrogen at 1 atm, K."""

P_ATM: float = 101_325.0
"""Standard atmosphere, Pa."""

ZERO_CELSIUS: float = 273.15
"""0 C in kelvin."""


# --- unit helpers -----------------------------------------------------------
def kelvin_from_celsius(t_c: float) -> float:
    """Convert a Celsius temperature to kelvin."""
    return t_c + ZERO_CELSIUS


def celsius_from_kelvin(t_k: float) -> float:
    """Convert a kelvin temperature to Celsius."""
    return t_k - ZERO_CELSIUS


def mpa(value: float) -> float:
    """Megapascals to pascals."""
    return value * 1e6


def gpa(value: float) -> float:
    """Gigapascals to pascals."""
    return value * 1e9


def to_mpa(value_pa: float) -> float:
    """Pascals to megapascals."""
    return value_pa / 1e6


def to_gpa(value_pa: float) -> float:
    """Pascals to gigapascals."""
    return value_pa / 1e9


def deg(value_rad: float) -> float:
    """Radians to degrees."""
    return math.degrees(value_rad)


def rad(value_deg: float) -> float:
    """Degrees to radians."""
    return math.radians(value_deg)


def a_per_mm2(value_a_per_m2: float) -> float:
    """A/m^2 to A/mm^2 (the usual engineering current-density unit)."""
    return value_a_per_m2 / 1e6


def from_a_per_mm2(value: float) -> float:
    """A/mm^2 to A/m^2."""
    return value * 1e6


__all__ = [
    "MU0", "EPS0", "C_LIGHT", "K_B", "R_GAS", "N_A", "E_CHARGE", "G_ACCEL",
    "T_ROOM", "T_LIQUID_HELIUM", "T_LIQUID_NITROGEN", "P_ATM", "ZERO_CELSIUS",
    "kelvin_from_celsius", "celsius_from_kelvin", "mpa", "gpa", "to_mpa", "to_gpa",
    "deg", "rad", "a_per_mm2", "from_a_per_mm2",
]
