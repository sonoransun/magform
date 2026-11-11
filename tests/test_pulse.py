"""Tests for magform.pulse: RLC discharge, capacitor-bank sizing, conductor heating, homopolar
generators, source comparison and the pulse-network assessment."""

from __future__ import annotations

import json
import math

import pytest

from magform import eddy, materials
from magform import pulse as P
from magform._numerics import linspace, trapz
from magform._report import Assessment, ResultMixin
from magform.benchmarks import (
    ENERGY_MAGNET_DRIVER_SOTA,
    PEAK_POWER_SOTA,
    PULSED_CURRENT_SOTA,
    PULSED_NONDESTRUCTIVE_RECORD,
    PULSED_VOLTAGE_SOTA,
)
from magform.constants import T_ROOM
from magform.errors import InputError, UnknownMaterialError
from magform.magnetics import Solenoid, field_for_pressure
from magform.status import Status, worst

PRESS = Solenoid(0.05, 0.25, 0.5)        # reference press: 100 mm bore, 0.2 m build, 0.5 m long
INSERT = Solenoid(0.0075, 0.11, 0.15)    # LANL-like 15 mm bore insert
B4 = field_for_pressure(4e9)             # 100.27 T
B16 = field_for_pressure(1.6e9)          # 63.41 T
SOTA, EXTRA, ASPIR, INFEAS = (Status.STATE_OF_THE_ART, Status.EXTRAPOLATION, Status.ASPIRATIONAL,
                              Status.INFEASIBLE)


def _statuses(a: P.PulseNetworkAssessment) -> tuple[Status, ...]:
    return (a.energy_status, a.voltage_status, a.current_status, a.power_status, a.thermal_status)


def _analytic_temperature(mat: materials.Material, t0: float, action: float) -> float:
    """Closed form of dT/dA = rho_e(T)/(rho_m c_p) for the registry's linear resistivity law."""
    a, rho0 = mat.resistivity_temp_coeff, mat.resistivity
    k = a * rho0 / (mat.density * mat.specific_heat)
    return T_ROOM + ((1.0 + a * (t0 - T_ROOM)) * math.exp(k * action) - 1.0) / a


# ------------------------------------------------------------------ quarter-period law
def test_quarter_period_law_and_its_inverses():
    e, t = 99.5e6, 25e-6
    assert P.angular_frequency_for_rise(t) == pytest.approx(math.pi / (2.0 * t), rel=1e-15)
    assert P.voltage_current_product(e, t) == pytest.approx(math.pi * e / t, rel=1e-15)
    assert P.voltage_current_product(e, t) == pytest.approx(1.25e13, rel=1e-2)      # 267 kV x 46.7 MA
    assert P.discharge_peak_power(e, t) == pytest.approx(0.5 * P.voltage_current_product(e, t), rel=1e-15)
    assert P.discharge_peak_power(e, t) == pytest.approx(6.25e12, rel=1e-2)         # terawatt class
    assert P.discharge_peak_power(e, 1e-3) == pytest.approx(1.563e11, rel=1e-3)
    assert P.discharge_peak_power(e, 1e-2) == pytest.approx(1.563e10, rel=1e-3)
    assert P.rise_time_for_product(e, 267e3, 46.75e6) == pytest.approx(t, rel=1e-2)
    assert P.rise_time_for_product(e, 25e3, 46.75e6) == pytest.approx(2.67e-4, rel=1e-2)   # 25 kV, one turn
    v, i = 25e3, 46.75e6
    assert P.voltage_current_product(e, P.rise_time_for_product(e, v, i)) == pytest.approx(v * i, rel=1e-12)
    assert P.voltage_current_product(0.0, t) == 0.0 and P.discharge_peak_power(0.0, t) == 0.0
    powers = [P.discharge_peak_power(e, x) for x in (1e-5, 1e-4, 1e-3, 1e-2)]
    assert all(b < a for a, b in zip(powers, powers[1:]))


def test_turns_only_trade_current_against_voltage():
    e = PRESS.stored_energy(B4)
    t = 1e-3
    product = P.voltage_current_product(e, t)
    base = None
    for n in (1, 3, 10, 100):
        coil = P.coil_electrical(PRESS, n)
        bank = P.size_capacitor_bank(coil.inductance, PRESS.current_for_field(B4, n), t)   # lossless
        assert bank.charge_voltage * bank.peak_current == pytest.approx(product, rel=1e-9)
        assert bank.bank_energy == pytest.approx(e, rel=1e-9)
        assert bank.peak_power == pytest.approx(P.discharge_peak_power(e, t), rel=1e-6)
        if base is None:
            base = bank
        assert bank.charge_voltage == pytest.approx(n * base.charge_voltage, rel=1e-9)
        assert bank.peak_current == pytest.approx(base.peak_current / n, rel=1e-9)
        assert bank.capacitance == pytest.approx(base.capacitance / n ** 2, rel=1e-9)


def test_inductive_store_passes_at_most_a_quarter_of_its_energy():
    assert P.inductive_transfer_efficiency(1e-6, 1e-6) == pytest.approx(0.25, rel=1e-15)
    assert P.INDUCTIVE_TRANSFER_LIMIT == 0.25
    for ratio in (0.01, 0.3, 3.0, 100.0):
        eta = P.inductive_transfer_efficiency(1e-6, ratio * 1e-6)
        assert 0.0 < eta < 0.25
        assert eta == pytest.approx(ratio / (1.0 + ratio) ** 2, rel=1e-12)
        assert eta == pytest.approx(P.inductive_transfer_efficiency(ratio * 1e-6, 1e-6), rel=1e-12)   # symmetric
        # flux conservation: load + store + switch shares add up to the initial energy
        store_left = 1.0 / (1.0 + ratio) ** 2
        switch = ratio / (1.0 + ratio)
        assert eta + store_left + switch == pytest.approx(1.0, rel=1e-12)


# ------------------------------------------------------------------ series RLC
def test_rlc_lossless_closed_form():
    c, v, ind = 1e-3, 1e4, 1e-6
    d = P.rlc_discharge(c, v, ind)
    w0 = 1.0 / math.sqrt(ind * c)
    assert d.regime == "underdamped" and d.damping_ratio == 0.0 and d.decay_rate == 0.0
    assert d.natural_frequency == pytest.approx(w0) and d.damped_frequency == pytest.approx(w0)
    assert d.time_to_peak == pytest.approx(0.5 * math.pi / w0, rel=1e-12)
    assert d.peak_current == pytest.approx(v * math.sqrt(c / ind), rel=1e-12)
    assert d.peak_current == pytest.approx(316.2e3, rel=1e-3) and d.time_to_peak == pytest.approx(49.67e-6, rel=1e-3)
    assert d.stored_energy == pytest.approx(5e4) and d.magnetic_energy_at_peak == pytest.approx(5e4)
    assert d.energy_dissipated_to_peak == 0.0 and d.capacitor_voltage_at_peak == 0.0
    assert d.transfer_efficiency == pytest.approx(1.0) and d.peak_di_dt == pytest.approx(v / ind)
    assert d.peak_power == pytest.approx(d.stored_energy * w0, rel=1e-6)            # E omega0 = V0 I_pk / 2
    assert d.peak_power == pytest.approx(P.discharge_peak_power(d.stored_energy, d.time_to_peak), rel=1e-6)
    assert d.voltage_reversal == pytest.approx(1.0) and math.isinf(d.crowbar_time_constant)
    i = P.rlc_current(c, v, ind)
    u = P.rlc_capacitor_voltage(c, v, ind)
    for x in (0.1, 0.5, 1.0, 1.7, 3.2):
        t = x * d.time_to_peak
        assert i(t) == pytest.approx(d.peak_current * math.sin(w0 * t), rel=1e-12, abs=1e-6)
        assert u(t) == pytest.approx(v * math.cos(w0 * t), rel=1e-9, abs=1e-6)
    assert i(0.0) == 0.0 and i(-1.0) == 0.0 and u(-1.0) == v and u(0.0) == v
    zero = P.rlc_discharge(c, 0.0, ind)
    assert zero.peak_current == 0.0 and zero.peak_power == 0.0 and zero.transfer_efficiency == 0.0


@pytest.mark.parametrize("ratio, regime", [(0.3, "underdamped"), (1.0, "critical"), (3.0, "overdamped")])
def test_rlc_energy_is_conserved_in_every_regime(ratio, regime):
    c, v, ind = 2e-3, 2e4, 5e-7
    r = ratio * 2.0 * math.sqrt(ind / c)
    d = P.rlc_discharge(c, v, ind, r)
    assert d.regime == regime and d.damping_ratio == pytest.approx(ratio, rel=1e-9)
    i = P.rlc_current(c, v, ind, r)
    u = P.rlc_capacitor_voltage(c, v, ind, r)
    e0 = 0.5 * c * v * v
    assert d.stored_energy == pytest.approx(e0)
    for t_end in (0.5 * d.time_to_peak, d.time_to_peak, 3.0 * d.time_to_peak):
        ts = linspace(0.0, t_end, 4001)
        joule = trapz([r * i(t) ** 2 for t in ts], ts)
        assert 0.5 * c * u(t_end) ** 2 + 0.5 * ind * i(t_end) ** 2 + joule == pytest.approx(e0, rel=1e-5)
    ts = linspace(0.0, d.time_to_peak, 4001)
    assert d.energy_dissipated_to_peak == pytest.approx(trapz([r * i(t) ** 2 for t in ts], ts), rel=1e-5)
    # at the current peak L dI/dt = 0, so the capacitor holds exactly the resistive drop
    assert d.capacitor_voltage_at_peak == pytest.approx(r * d.peak_current, rel=1e-12)
    assert u(d.time_to_peak) == pytest.approx(r * d.peak_current, rel=1e-9)
    assert i(0.999 * d.time_to_peak) < d.peak_current and i(1.001 * d.time_to_peak) < d.peak_current
    assert i(d.time_to_peak) == pytest.approx(d.peak_current, rel=1e-12)
    # I_pk = V0 sqrt(C/L) exp(-alpha t_pk) and E_field/E_bank = exp(-R t_pk/L) in every regime
    assert d.peak_current == pytest.approx(v * math.sqrt(c / ind) * math.exp(-d.decay_rate * d.time_to_peak), rel=1e-12)
    assert d.transfer_efficiency == pytest.approx(math.exp(-r * d.time_to_peak / ind), rel=1e-12)
    assert d.stored_energy == pytest.approx(d.magnetic_energy_at_peak + d.energy_dissipated_to_peak
                                            + 0.5 * c * d.capacitor_voltage_at_peak ** 2, rel=1e-12)
    sampled = max(u(t) * i(t) for t in ts)
    assert d.peak_power == pytest.approx(sampled, rel=1e-5) and d.peak_power >= sampled * (1.0 - 1e-12)
    assert d.peak_di_dt == pytest.approx(v / ind)
    assert (i(1e-9 * d.time_to_peak) - i(0.0)) / (1e-9 * d.time_to_peak) == pytest.approx(v / ind, rel=1e-6)


def test_rlc_regime_boundaries_and_limits():
    c, v, ind = 1e-3, 1e4, 1e-6
    r_crit = 2.0 * math.sqrt(ind / c)
    crit = P.rlc_discharge(c, v, ind, r_crit)
    assert crit.regime == "critical" and crit.damping_ratio == pytest.approx(1.0, rel=1e-12)
    assert crit.time_to_peak == pytest.approx(2.0 * ind / r_crit, rel=1e-12)
    assert crit.peak_current == pytest.approx(2.0 * v / (math.e * r_crit), rel=1e-12)
    assert crit.damped_frequency == 0.0 and crit.voltage_reversal == 0.0
    below = P.rlc_discharge(c, v, ind, 0.999 * r_crit)
    above = P.rlc_discharge(c, v, ind, 1.001 * r_crit)
    assert (below.regime, above.regime) == ("underdamped", "overdamped")
    assert below.peak_current > crit.peak_current > above.peak_current              # continuous and monotone
    assert below.peak_current == pytest.approx(crit.peak_current, rel=2e-3)
    assert above.peak_current == pytest.approx(crit.peak_current, rel=2e-3)
    assert below.time_to_peak == pytest.approx(crit.time_to_peak, rel=2e-3)
    assert above.time_to_peak == pytest.approx(crit.time_to_peak, rel=2e-3)
    assert above.voltage_reversal == 0.0 and above.damped_frequency == 0.0
    # heavy damping: the current creeps to V/R; weak damping: reversal exp(-pi alpha/omega_d)
    heavy = P.rlc_discharge(c, v, ind, 100.0 * r_crit)
    assert heavy.peak_current == pytest.approx(v / (100.0 * r_crit), rel=1e-3)
    assert heavy.peak_current < v / (100.0 * r_crit)
    under = P.rlc_discharge(c, v, ind, 0.2 * r_crit)
    assert under.voltage_reversal == pytest.approx(math.exp(-math.pi * 0.2 / math.sqrt(1.0 - 0.04)), rel=1e-12)
    assert under.damped_frequency == pytest.approx(under.natural_frequency * math.sqrt(0.96), rel=1e-12)
    assert under.time_to_peak == pytest.approx(math.atan(math.sqrt(0.96) / 0.2) / under.damped_frequency, rel=1e-12)
    runs = [P.rlc_discharge(c, v, ind, x * r_crit) for x in (0.0, 0.1, 0.5, 1.0, 2.0, 10.0)]
    for a, b in zip(runs, runs[1:]):
        assert b.peak_current < a.peak_current and b.time_to_peak < a.time_to_peak
        assert b.transfer_efficiency < a.transfer_efficiency and b.voltage_reversal <= a.voltage_reversal


def test_crowbar_freezes_the_current_into_an_l_over_r_decay():
    c, v, ind, r = 1e-3, 1e4, 1e-6, 0.01
    d = P.rlc_discharge(c, v, ind, r)
    free = P.rlc_current(c, v, ind, r)
    crow = P.rlc_current(c, v, ind, r, crowbar=True)
    for x in (0.0, 0.2, 0.7, 1.0):
        assert crow(x * d.time_to_peak) == free(x * d.time_to_peak)
    tau = ind / r
    assert d.crowbar_time_constant == pytest.approx(tau)
    for dt in (0.5 * tau, tau, 3.0 * tau):
        assert crow(d.time_to_peak + dt) == pytest.approx(d.peak_current * math.exp(-dt / tau), rel=1e-12)
    assert free(d.time_to_peak + 3.0 * tau) < 0.0 < crow(d.time_to_peak + 3.0 * tau)   # ringing vs unipolar
    d0 = P.rlc_discharge(c, v, ind)
    lossless = P.rlc_current(c, v, ind, 0.0, crowbar=True)
    assert lossless(10.0 * d0.time_to_peak) == pytest.approx(d0.peak_current, rel=1e-12)


@pytest.mark.parametrize("x", [0.0, 1e-6, 0.05, 0.5, 0.999999, 1.0, 1.000001, 2.0, 10.0, 50.0])
def test_capacitance_for_rise_time_round_trip(x):
    ind, t = 1e-6, 1e-4
    r = x * 2.0 * ind / t                                    # x = alpha t_rise
    cap = P.capacitance_for_rise_time(ind, t, r)
    d = P.rlc_discharge(cap, 1.0, ind, r)
    assert d.time_to_peak == pytest.approx(t, rel=1e-9)
    v0 = P.voltage_for_peak_current(cap, ind, 1e5, r)
    assert P.rlc_discharge(cap, v0, ind, r).peak_current == pytest.approx(1e5, rel=1e-9)
    assert v0 == pytest.approx(1e5 * math.sqrt(ind / cap) * math.exp(x), rel=1e-9)   # I sqrt(L/C) exp(alpha t)
    assert 0.5 * cap * v0 * v0 == pytest.approx(0.5 * ind * 1e10 * math.exp(2.0 * x), rel=1e-9)


def test_capacitance_for_rise_time_limits_and_errors():
    ind, t = 1e-6, 1e-4
    assert P.capacitance_for_rise_time(ind, t) == pytest.approx((2.0 * t / math.pi) ** 2 / ind, rel=1e-12)
    r_crit = 2.0 * ind / t
    assert P.capacitance_for_rise_time(ind, t, r_crit) == pytest.approx(4.0 * ind / r_crit ** 2, rel=1e-9)
    caps = [P.capacitance_for_rise_time(ind, t, x * r_crit) for x in (0.0, 0.2, 0.9, 1.0, 1.1, 3.0)]
    assert all(b > a for a, b in zip(caps, caps[1:]))        # more resistance needs a bigger bank
    with pytest.raises(InputError):
        P.capacitance_for_rise_time(ind, t, 800.0 * r_crit)  # would need exp(1600) farads
    assert P.voltage_for_peak_current(1e-3, 1e-6, 0.0) == 0.0


# ------------------------------------------------------------------ coil electrical parameters
def test_coil_electrical_reference_press():
    cuag = materials.get("cuag_conductor")
    rho = cuag.resistivity_at(77.0)
    c1 = P.coil_electrical(PRESS, 1)
    assert c1.inductance == pytest.approx(PRESS.inductance_from_energy(1.0), rel=1e-12)
    assert c1.inductance == pytest.approx(91.04e-9, rel=1e-3)
    assert c1.inductance_wheeler == pytest.approx(PRESS.inductance(1.0), rel=1e-12)
    assert c1.conductor == "cuag_conductor" and c1.temperature == 77.0 and c1.fill_factor == 0.5
    assert c1.resistivity == pytest.approx(rho, rel=1e-12)
    assert c1.resistance == pytest.approx(rho * 2.0 * math.pi * 0.15 / (0.5 * 0.5 * 0.2), rel=1e-12)
    assert c1.resistance == pytest.approx(2.147e-7, rel=1e-3)                        # 0.215 micro-ohm
    assert c1.time_constant == pytest.approx(c1.inductance / c1.resistance) and c1.time_constant == pytest.approx(0.4241, rel=1e-3)
    assert c1.current_per_tesla * B4 == pytest.approx(PRESS.current_for_field(B4, 1.0), rel=1e-12)
    assert c1.conductor_area == pytest.approx(0.05) and c1.conductor_side == pytest.approx(math.sqrt(0.05))
    assert c1.mean_turn_length == pytest.approx(2.0 * math.pi * 0.15) and c1.layers == 1
    assert c1.conductor_volume == pytest.approx(0.5 * PRESS.winding_volume)
    assert c1.conductor_mass == pytest.approx(0.5 * PRESS.winding_volume * cuag.density)
    assert c1.conductor_mass == pytest.approx(437.5, rel=1e-3)
    # the lumped resistance reproduces the Joule power of the uniform-J winding
    j_c = PRESS.current_density_for_field(B4) / 0.5
    i = PRESS.current_for_field(B4, 1.0)
    assert i * i * c1.resistance == pytest.approx(rho * j_c ** 2 * c1.conductor_volume, rel=1e-12)
    assert 0.5 * c1.inductance * i * i == pytest.approx(PRESS.stored_energy(B4), rel=1e-12)


def test_coil_electrical_scaling():
    c1 = P.coil_electrical(PRESS, 1)
    for n in (2, 37, 500, 2.5):
        cn = P.coil_electrical(PRESS, n)
        assert cn.inductance == pytest.approx(n * n * c1.inductance, rel=1e-12)
        assert cn.resistance == pytest.approx(n * n * c1.resistance, rel=1e-12)
        assert cn.time_constant == pytest.approx(c1.time_constant, rel=1e-12)       # L/R does not depend on N
        assert cn.conductor_area == pytest.approx(c1.conductor_area / n, rel=1e-12)
        assert cn.conductor_mass == pytest.approx(c1.conductor_mass, rel=1e-12)
        assert cn.current_per_tesla == pytest.approx(c1.current_per_tesla / n, rel=1e-12)
    assert P.coil_electrical(PRESS, 37).layers == 4 and P.coil_electrical(PRESS, 100).layers == 6
    assert P.coil_electrical(INSERT, 134).layers == 10
    cuag = materials.get("cuag_conductor")
    warm = P.coil_electrical(PRESS, 1, temperature=T_ROOM)
    assert warm.resistance / c1.resistance == pytest.approx(cuag.resistivity / cuag.resistivity_at(77.0), rel=1e-12)
    thin = P.coil_electrical(PRESS, 1, fill_factor=0.25)
    assert thin.resistance == pytest.approx(2.0 * c1.resistance) and thin.conductor_mass == pytest.approx(0.5 * c1.conductor_mass)
    cu = P.coil_electrical(PRESS, 1, conductor="copper_ofhc")
    assert cu.conductor == "copper_ofhc" and cu.resistance < c1.resistance and cu.inductance == c1.inductance
    assert P.coil_electrical(PRESS, 1, conductor=cuag) == c1


def test_turns_for_voltage_reference_press():
    assert [P.turns_for_voltage(PRESS, B4, t) for t in (25e-6, 1e-3, 1e-2, 0.1)] == [1, 3, 37, 348]
    assert [P.turns_for_voltage(PRESS, B16, t) for t in (25e-6, 1e-3, 1e-2)] == [1, 5, 58]
    for t in (1e-3, 1e-2, 0.1):
        n = P.turns_for_voltage(PRESS, B4, t)
        assert P.assess_pulse_network(PRESS, B4, t, turns=n).charge_voltage <= PULSED_VOLTAGE_SOTA
        assert P.assess_pulse_network(PRESS, B4, t, turns=n + 1).charge_voltage > PULSED_VOLTAGE_SOTA
    # a higher ceiling buys proportionally more turns; stray inductance costs some
    assert P.turns_for_voltage(PRESS, B4, 1e-2, max_voltage=50e3) == 74
    with_stray = P.turns_for_voltage(PRESS, B4, 1e-2, stray_inductance=20e-6)
    assert with_stray == 29
    assert P.assess_pulse_network(PRESS, B4, 1e-2, stray_inductance=20e-6).charge_voltage <= PULSED_VOLTAGE_SOTA
    assert P.turns_for_voltage(PRESS, B4, 1e-2, conductor="copper_ofhc") >= 37   # less resistance, less voltage


# ------------------------------------------------------------------ capacitor bank
def test_size_capacitor_bank_lossless_closed_forms():
    ind, i, t = 2e-6, 1e6, 1e-3
    e = 0.5 * ind * i * i                                    # 1 MJ
    b = P.size_capacitor_bank(ind, i, t)
    assert isinstance(b, Assessment)
    assert b.capacitance == pytest.approx((2.0 * t / math.pi) ** 2 / ind, rel=1e-12)
    assert b.charge_voltage == pytest.approx(i * math.sqrt(ind / b.capacitance), rel=1e-12)
    assert b.charge_voltage == pytest.approx(math.pi * e / (i * t), rel=1e-12)       # V0 I = pi E / t
    assert b.bank_energy == pytest.approx(e, rel=1e-12) and b.magnetic_energy == pytest.approx(e, rel=1e-12)
    assert b.transfer_efficiency == pytest.approx(1.0) and b.energy_dissipated_to_peak == 0.0
    assert b.peak_power == pytest.approx(math.pi * e / (2.0 * t), rel=1e-6)
    assert b.peak_di_dt == pytest.approx(b.charge_voltage / ind)
    assert b.action_to_peak == pytest.approx(0.5 * i * i * t, rel=1e-6)              # quarter sine
    assert math.isinf(b.action_crowbar) and math.isinf(b.crowbar_time_constant)
    assert b.charge_to_peak == pytest.approx(2.0 * i * t / math.pi, rel=1e-9)        # C V0 = integral of I dt
    assert b.regime == "underdamped" and b.damping_ratio == 0.0 and b.voltage_reversal == pytest.approx(1.0)
    assert b.switch_count == 10 and b.switch_current == P.SWITCH_CURRENT_RATING
    assert P.size_capacitor_bank(ind, i, t, switch_current=3e5).switch_count == 4
    assert b.bank_mass == pytest.approx(e / P.CAPACITOR_SPECIFIC_ENERGY)
    assert b.bank_volume == pytest.approx(e / P.CAPACITOR_ENERGY_DENSITY)
    lo, hi = b.bank_mass_range
    assert lo == pytest.approx(e / 2.5e3) and hi == pytest.approx(e / 0.2e3) and lo < b.bank_mass < hi
    lo_v, hi_v = b.bank_volume_range
    assert lo_v < b.bank_volume < hi_v
    assert P.size_capacitor_bank(ind, i, t, specific_energy=500.0).bank_mass == pytest.approx(2.0 * b.bank_mass)
    assert (b.energy_status, b.voltage_status, b.current_status, b.power_status) == (SOTA, SOTA, SOTA, SOTA)
    assert b.status is SOTA and b.warnings == ()
    assert b.margin == pytest.approx(min(ENERGY_MAGNET_DRIVER_SOTA / e, PULSED_VOLTAGE_SOTA / b.charge_voltage,
                                         PULSED_CURRENT_SOTA / i, PEAK_POWER_SOTA / b.peak_power))
    assert b.margin == pytest.approx(5.0)                    # the 5 MA current limit binds first


def test_size_capacitor_bank_with_resistance_and_stray_inductance():
    ind, i, t = 2e-6, 1e6, 1e-3
    ideal = P.size_capacitor_bank(ind, i, t)
    r = 0.4 * 2.0 * ind / t                                  # alpha t_rise = 0.4
    lossy = P.size_capacitor_bank(ind, i, t, resistance=r)
    assert lossy.regime == "underdamped" and lossy.rise_time == t and lossy.resistance == r
    assert lossy.bank_energy > ideal.bank_energy and lossy.charge_voltage > 0.0
    assert lossy.magnetic_energy == pytest.approx(ideal.magnetic_energy)
    # E_bank = E_field exp(R t_rise / L), whatever the damping regime
    assert lossy.bank_energy == pytest.approx(ideal.bank_energy * math.exp(0.8), rel=1e-9)
    assert lossy.transfer_efficiency == pytest.approx(math.exp(-0.8), rel=1e-9)
    assert lossy.bank_energy == pytest.approx(lossy.magnetic_energy + lossy.energy_dissipated_to_peak
                                              + 0.5 * lossy.capacitance * (r * i) ** 2, rel=1e-9)
    assert lossy.energy_dissipated_to_peak == pytest.approx(r * lossy.action_to_peak, rel=1e-5)
    assert lossy.action_crowbar == pytest.approx(lossy.action_to_peak + i * i * ind / (2.0 * r), rel=1e-12)
    assert lossy.crowbar_time_constant == pytest.approx(ind / r)
    assert lossy.charge_to_peak == pytest.approx(lossy.capacitance * (lossy.charge_voltage - r * i), rel=1e-12)
    stray = P.size_capacitor_bank(ind, i, t, stray_inductance=ind)
    assert stray.transfer_efficiency == pytest.approx(0.5, rel=1e-9)
    assert stray.bank_energy == pytest.approx(2.0 * ideal.bank_energy, rel=1e-9)
    assert stray.charge_voltage == pytest.approx(2.0 * ideal.charge_voltage, rel=1e-9)
    assert stray.magnetic_energy == pytest.approx(ideal.magnetic_energy) and stray.stray_inductance == ind
    assert any("stray inductance" in n for n in stray.notes)
    over = P.size_capacitor_bank(ind, i, t, resistance=5.0 * 2.0 * ind / t)
    assert over.regime == "overdamped" and over.transfer_efficiency == pytest.approx(math.exp(-10.0), rel=1e-6)
    assert any("rectifier" in w for w in over.warnings) and over.voltage_reversal == 0.0
    crit = P.size_capacitor_bank(ind, i, t, resistance=2.0 * ind / t)
    assert crit.regime == "critical" and crit.transfer_efficiency == pytest.approx(math.exp(-2.0), rel=1e-6)


def test_size_capacitor_bank_statuses_follow_the_benchmarks():
    big = P.size_capacitor_bank(9.104e-8, 46.75e6, 25e-6)   # the single-turn 100 T press in 25 us
    assert (big.energy_status, big.voltage_status, big.current_status, big.power_status) == (EXTRA, ASPIR, ASPIR, EXTRA)
    assert big.status is ASPIR and big.margin == pytest.approx(PULSED_VOLTAGE_SOTA / big.charge_voltage)
    assert big.margin < 0.1 and len(big.warnings) == 4
    assert big.switch_count == 468 and big.bank_mass == pytest.approx(99.5e3, rel=1e-2)   # ~100 t of capacitors
    assert any("267 kV" in n and "46.8 MA" in n for n in big.notes)


# ------------------------------------------------------------------ conductor heating
def test_conductor_heating_half_sine_matches_the_action_integral():
    cuag = materials.get("cuag_conductor")
    h = P.conductor_heating(PRESS, B4, 1e-2, "half_sine")
    j = PRESS.current_density_for_field(B4) / 0.5
    assert h.current_density == pytest.approx(j, rel=1e-12) and j == pytest.approx(9.35e8, rel=1e-3)
    assert h.action == pytest.approx(j * j * 1e-2, rel=1e-6)                         # J^2 t_rise over rise + fall
    assert h.final_temperature == pytest.approx(_analytic_temperature(cuag, 77.0, h.action), rel=1e-5)
    assert h.temperature_rise == pytest.approx(33.6, rel=1e-2)
    assert h.rise_phase_temperature_rise == pytest.approx(16.04, rel=1e-2)
    assert h.stored_energy_temperature_rise == 0.0 and h.waveform == "half_sine"
    assert h.within_limit and not h.melted and h.initial_temperature == 77.0 and h.temperature_limit == 400.0
    # the longest pulse is the action to the 400 K limit divided by J^2
    g = eddy.specific_action_to_temperature(cuag, 77.0, 400.0)
    assert h.max_rise_time == pytest.approx(g / j ** 2, rel=1e-12)
    assert h.max_rise_time == pytest.approx(0.0589, rel=1e-2)                        # ~59 ms at 100 T
    at_limit = P.conductor_heating(PRESS, B4, h.max_rise_time, "half_sine")
    assert at_limit.final_temperature == pytest.approx(400.0, rel=2e-3)
    assert P.conductor_heating(PRESS, B4, 2.0 * h.max_rise_time, "half_sine").within_limit is False
    rises = [P.conductor_heating(PRESS, B4, t, "half_sine").temperature_rise for t in (1e-4, 1e-3, 1e-2, 5e-2)]
    assert all(b > a for a, b in zip(rises, rises[1:]))
    long = P.conductor_heating(PRESS, B4, 0.2, "half_sine")
    assert long.melted and long.final_temperature == cuag.melting_point and not long.within_limit
    # a lower field or a larger conductor fraction allows a longer pulse
    assert P.conductor_heating(PRESS, B16, 1e-2, "half_sine").max_rise_time == pytest.approx(
        h.max_rise_time * (B4 / B16) ** 2, rel=1e-9)
    assert P.conductor_heating(PRESS, B4, 1e-2, "half_sine", fill_factor=1.0).max_rise_time == pytest.approx(
        4.0 * h.max_rise_time, rel=1e-9)


def test_conductor_heating_crowbar_coil_absorbs_its_own_field_energy():
    cuag = materials.get("cuag_conductor")
    e = PRESS.stored_energy(B4)
    h = P.conductor_heating(PRESS, B4, 1e-2)                 # crowbar, no external dump
    self_rise = e / (h.conductor_mass * cuag.specific_heat)
    assert h.waveform == "crowbar" and h.dump_fraction == 0.0
    assert h.magnetic_energy == pytest.approx(e) and h.conductor_mass == pytest.approx(437.5, rel=1e-3)
    assert h.self_energy_temperature_rise == pytest.approx(self_rise, rel=1e-12)
    assert self_rise == pytest.approx(649.6, rel=1e-3)                               # the coil cannot swallow 99.5 MJ
    assert h.stored_energy_temperature_rise == pytest.approx(self_rise, rel=1e-12)
    assert h.temperature_rise == pytest.approx(h.rise_phase_temperature_rise + self_rise, rel=1e-12)
    assert h.temperature_rise == pytest.approx(665.7, rel=1e-3)
    assert h.final_temperature == pytest.approx(742.7, rel=1e-3) and not h.within_limit and not h.melted
    assert h.action == pytest.approx(0.5 * h.current_density ** 2 * 1e-2, rel=1e-6)  # quarter-sine rise only
    assert P.conductor_heating(PRESS, B4, 25e-6).self_energy_temperature_rise == pytest.approx(self_rise)
    dumped = P.conductor_heating(PRESS, B4, 1e-2, dump_fraction=0.6)
    assert dumped.stored_energy_temperature_rise == pytest.approx(0.4 * self_rise, rel=1e-12)
    assert dumped.temperature_rise == pytest.approx(275.9, rel=1e-3) and dumped.within_limit
    full_dump = P.conductor_heating(PRESS, B4, 1e-2, dump_fraction=1.0)
    assert full_dump.temperature_rise == pytest.approx(h.rise_phase_temperature_rise, rel=1e-12)
    # at 63 T the same winding ends a full-energy pulse near room temperature
    low = P.conductor_heating(PRESS, B16, 1e-2)
    assert low.within_limit and low.final_temperature == pytest.approx(343.1, rel=1e-2)
    assert low.self_energy_temperature_rise == pytest.approx(self_rise * (B16 / B4) ** 2, rel=1e-9)
    # melting caps the temperature
    hot = P.conductor_heating(PRESS, 1.5 * B4, 1e-2)
    assert hot.melted and hot.final_temperature == cuag.melting_point
    idle = P.conductor_heating(PRESS, 0.0, 1e-2)
    assert idle.temperature_rise == 0.0 and math.isinf(idle.max_rise_time) and idle.within_limit
    # the copper record gives a similar answer
    cu = P.conductor_heating(PRESS, B4, 1e-2, conductor="copper_ofhc")
    assert cu.conductor == "copper_ofhc" and cu.temperature_rise == pytest.approx(615.5, rel=1e-2)


# ------------------------------------------------------------------ homopolar generator
def test_homopolar_generator_is_a_kilofarad_capacitor():
    e = PRESS.stored_energy(B4)
    g = P.homopolar_for_energy(e)
    assert g.stored_energy() == pytest.approx(e, rel=1e-12)
    assert g.mass() == pytest.approx(4.0 * e / 200.0 ** 2, rel=1e-12) and g.mass() == pytest.approx(9948.0, rel=1e-3)
    assert g.rotor_radius == pytest.approx(0.5865, rel=1e-3) and g.rotor_length == pytest.approx(2.0 * g.rotor_radius)
    assert g.voltage() == pytest.approx(0.5 * 2.0 * 200.0 * g.rotor_radius, rel=1e-12)
    assert g.voltage() == pytest.approx(117.3, rel=1e-3)                             # ~100 V whatever the size
    assert g.equivalent_capacitance() == pytest.approx(2.0 * e / g.voltage() ** 2, rel=1e-12)
    assert g.equivalent_capacitance() == pytest.approx(14463.0, rel=1e-3)            # 14.5 kF
    assert g.voltage_constant() * g.angular_speed() == pytest.approx(g.voltage(), rel=1e-12)
    assert g.moment_of_inertia() == pytest.approx(0.5 * g.mass() * g.rotor_radius ** 2, rel=1e-12)
    assert g.rotor_stress() == pytest.approx(3.3 / 8.0 * 7850.0 * 200.0 ** 2, rel=1e-12)   # 130 MPa
    one = P.coil_electrical(PRESS, 1)
    t_q = g.quarter_period(one.inductance)
    assert t_q == pytest.approx(0.5 * math.pi * math.sqrt(one.inductance * g.equivalent_capacitance()), rel=1e-12)
    assert t_q == pytest.approx(0.0570, rel=1e-3)                                    # 57 ms into one turn
    assert t_q / 25e-6 > 2000.0                                                      # vs a 25 us forming pulse
    # all of the rotor energy in the coil is exactly the ampere-turns of the field
    assert g.peak_current(one.inductance) == pytest.approx(PRESS.current_for_field(B4, 1.0), rel=1e-9)
    five = P.coil_electrical(PRESS, 5)
    assert g.quarter_period(five.inductance) == pytest.approx(5.0 * t_q, rel=1e-12)
    assert g.peak_current(five.inductance) == pytest.approx(9.35e6, rel=1e-3)
    d = g.discharge(one.inductance, one.resistance)
    assert d.regime == "underdamped" and d.voltage == pytest.approx(g.voltage())
    assert d.time_to_peak == pytest.approx(0.0555, rel=1e-2) and d.time_to_peak < t_q
    assert d.transfer_efficiency == pytest.approx(math.exp(-d.time_to_peak / one.time_constant), rel=1e-9)
    assert d.peak_current < g.peak_current(one.inductance)


def test_homopolar_generator_geometry_and_limits():
    disk = P.HomopolarGenerator(0.5, 0.3)
    assert disk.tip_speed == P.HPG_TIP_SPEED == 200.0 and disk.field == P.HPG_FLUX_DENSITY == 2.0
    assert disk.stored_energy() == pytest.approx(0.25 * disk.mass() * 200.0 ** 2, rel=1e-12)   # 10 kJ/kg
    assert disk.voltage() == pytest.approx(100.0, rel=1e-12)                         # 2 T, 200 m/s, 0.5 m
    assert disk.angular_speed() == pytest.approx(400.0)
    ring = P.HomopolarGenerator(0.5, 0.3, inner_radius=0.25)
    assert ring.mass() == pytest.approx(0.75 * disk.mass(), rel=1e-12)
    assert ring.moment_of_inertia() == pytest.approx(0.5 * ring.mass() * (0.25 + 0.0625), rel=1e-12)
    assert ring.voltage() == pytest.approx(0.75 * disk.voltage(), rel=1e-12)
    assert ring.equivalent_capacitance() == pytest.approx(2.0 * ring.stored_energy() / ring.voltage() ** 2, rel=1e-12)
    pin = P.HomopolarGenerator(0.5, 0.3, inner_radius=1e-6)
    assert pin.rotor_stress() == pytest.approx(2.0 * disk.rotor_stress(), rel=1e-6)  # a pin hole doubles it
    assert ring.rotor_stress() > pin.rotor_stress()
    fast = P.HomopolarGenerator(0.5, 0.3, tip_speed=400.0)
    assert fast.stored_energy() == pytest.approx(4.0 * disk.stored_energy()) and fast.voltage() == pytest.approx(200.0)
    assert fast.equivalent_capacitance() == pytest.approx(disk.equivalent_capacitance(), rel=1e-12)   # J/k^2
    strong = P.HomopolarGenerator(0.5, 0.3, field=4.0)
    assert strong.equivalent_capacitance() == pytest.approx(0.25 * disk.equivalent_capacitance(), rel=1e-12)
    v_lim = P.rotor_tip_speed_limit(500e6)
    assert v_lim == pytest.approx(393.0, rel=1e-3)
    assert P.HomopolarGenerator(0.5, 0.3, tip_speed=v_lim).rotor_stress() == pytest.approx(500e6, rel=1e-12)
    assert P.rotor_tip_speed_limit(500e6, density=2000.0) > v_lim
    lossy = P.HomopolarGenerator(0.5, 0.3, internal_resistance=1e-5)
    assert lossy.discharge(1e-6).resistance == 1e-5 and lossy.discharge(1e-6, 2e-5).resistance == pytest.approx(3e-5)
    slender = P.homopolar_for_energy(1e8, aspect_ratio=3.0)
    assert slender.stored_energy() == pytest.approx(1e8) and slender.rotor_length == pytest.approx(6.0 * slender.rotor_radius)
    assert slender.voltage() < P.homopolar_for_energy(1e8).voltage()


# ------------------------------------------------------------------ source comparison
def test_source_comparison_for_the_reference_press():
    e = 99.5e6
    fast = {o.key: o for o in P.source_comparison(e, 25e-6)}
    assert list(fast) == ["capacitor_bank", "motor_generator", "homopolar_generator", "flux_compression_generator"]
    assert fast["capacitor_bank"].status is EXTRA and fast["capacitor_bank"].fits and fast["capacitor_bank"].reusable
    assert fast["capacitor_bank"].fastest_rise_time == pytest.approx(0.5 * math.pi * e / PEAK_POWER_SOTA)
    assert fast["motor_generator"].status is INFEAS and not fast["motor_generator"].fits
    hpg = fast["homopolar_generator"]
    assert hpg.status is INFEAS and not hpg.fits and "25%" in hpg.reason
    assert hpg.fastest_rise_time == pytest.approx(0.0261, rel=1e-2)                  # a thousand times too slow
    assert hpg.fastest_rise_time == pytest.approx(
        0.5 * math.pi * math.sqrt(P.HPG_INTERNAL_INDUCTANCE * 2.0 * e / P.HPG_VOLTAGE_CLASS ** 2), rel=1e-12)
    fcg = fast["flux_compression_generator"]
    assert fcg.status is EXTRA and fcg.fits and not fcg.reusable and "consumed" in fcg.reason
    assert P.source_comparison(10e6, 25e-6)[3].status is SOTA                         # tens of MJ: single-shot practice
    assert all(o.peak_power == pytest.approx(6.25e12, rel=1e-2) for o in fast.values())
    mid = {o.key: o for o in P.source_comparison(e, 1e-2)}
    assert mid["motor_generator"].status is ASPIR and mid["motor_generator"].fits    # 15.6 GW vs 1.4 GW
    assert mid["homopolar_generator"].status is INFEAS and mid["flux_compression_generator"].status is INFEAS
    slow = {o.key: o for o in P.source_comparison(e, 0.5)}
    assert slow["motor_generator"].status is SOTA                                    # the LANL outsert regime
    assert slow["homopolar_generator"].status is EXTRA                               # 60 MJ < E < 500 MJ
    assert slow["capacitor_bank"].status is EXTRA and slow["flux_compression_generator"].status is INFEAS
    small = {o.key: o for o in P.source_comparison(30e6, 0.5)}
    assert small["homopolar_generator"].status is SOTA and small["capacitor_bank"].status is SOTA
    huge = {o.key: o for o in P.source_comparison(5e9, 0.5)}
    assert huge["capacitor_bank"].status is ASPIR and huge["homopolar_generator"].status is ASPIR
    assert huge["motor_generator"].status is ASPIR
    for group in (fast, mid, slow, small, huge):
        for o in group.values():
            assert o.fits == (o.status is not INFEAS) and len(o.reason) > 40 and len(o.reference) > 20
            assert o.reusable == (o.key != "flux_compression_generator")


# ------------------------------------------------------------------ network assessment: headline results
def test_reference_press_at_4_gpa_headline_numbers():
    e = PRESS.stored_energy(B4)
    assert B4 == pytest.approx(100.27, rel=1e-4) and e == pytest.approx(99.48e6, rel=1e-3)   # ~0.1 GJ
    fast = P.assess_pulse_network(PRESS, B4, 25e-6)
    assert fast.magnetic_energy == pytest.approx(e, rel=1e-12)
    assert fast.turns == 1 and fast.voltage_limited
    assert fast.charge_voltage == pytest.approx(267.4e3, rel=1e-3)
    assert fast.peak_current == pytest.approx(46.75e6, rel=1e-3) and fast.ampere_turns == pytest.approx(fast.peak_current)
    assert fast.peak_power == pytest.approx(6.25e12, rel=1e-3)                       # terawatt class
    assert fast.charge_voltage * fast.peak_current == pytest.approx(math.pi * e / 25e-6, rel=1e-3)
    assert _statuses(fast) == (EXTRA, ASPIR, ASPIR, EXTRA, INFEAS) and fast.status is INFEAS
    assert fast.margin == pytest.approx(PULSED_VOLTAGE_SOTA / fast.charge_voltage) and fast.margin < 0.1
    assert fast.switch_count == 468 and fast.bank_mass == pytest.approx(99.5e3, rel=1e-3)
    assert any("even a single turn" in w for w in fast.warnings)

    ms = P.assess_pulse_network(PRESS, B4, 1e-3)
    assert ms.turns == 3 and not ms.voltage_limited
    assert ms.charge_voltage == pytest.approx(20.07e3, rel=1e-3)
    assert ms.peak_current == pytest.approx(15.58e6, rel=1e-3)
    assert ms.peak_power == pytest.approx(1.565e11, rel=1e-3)
    assert _statuses(ms) == (EXTRA, SOTA, EXTRA, SOTA, INFEAS) and ms.status is INFEAS

    slow = P.assess_pulse_network(PRESS, B4, 1e-2)
    assert slow.turns == 37 and slow.layers == 4 and not slow.voltage_limited
    assert slow.charge_voltage == pytest.approx(24.91e3, rel=1e-3)
    assert slow.peak_current == pytest.approx(1.2635e6, rel=1e-3)
    assert slow.peak_power == pytest.approx(1.5855e10, rel=1e-3)
    assert slow.bank_energy == pytest.approx(101.86e6, rel=1e-3)                     # twice the HZDR bank
    assert _statuses(slow) == (EXTRA, SOTA, SOTA, SOTA, EXTRA) and slow.status is EXTRA
    assert slow.margin == pytest.approx(ENERGY_MAGNET_DRIVER_SOTA / slow.bank_energy)
    assert slow.margin == pytest.approx(0.491, rel=1e-2)
    assert slow.bank_mass == pytest.approx(101.86e3, rel=1e-3) and slow.switch_count == 13
    # the field energy does not depend on the rise time; the faster, the worse
    assert fast.magnetic_energy == ms.magnetic_energy == slow.magnetic_energy
    assert fast.peak_power > ms.peak_power > slow.peak_power
    assert fast.status.rank >= ms.status.rank >= slow.status.rank


def test_reference_press_at_1p6_gpa_headline_numbers():
    e = PRESS.stored_energy(B16)
    assert B16 == pytest.approx(63.41, rel=1e-4) and e == pytest.approx(39.79e6, rel=1e-3)
    fast = P.assess_pulse_network(PRESS, B16, 25e-6)
    assert fast.turns == 1 and fast.voltage_limited
    assert fast.charge_voltage == pytest.approx(169.1e3, rel=1e-3)
    assert fast.peak_current == pytest.approx(29.57e6, rel=1e-3)
    assert fast.peak_power == pytest.approx(2.50e12, rel=1e-3)
    assert _statuses(fast) == (SOTA, ASPIR, ASPIR, EXTRA, EXTRA) and fast.status is ASPIR
    assert fast.skin_limited and not fast.surface_melts
    assert fast.surface_temperature_rise == pytest.approx(492.3, rel=1e-3)           # hot, not molten, at 63 T

    ms = P.assess_pulse_network(PRESS, B16, 1e-3)
    assert ms.turns == 5 and ms.charge_voltage == pytest.approx(21.16e3, rel=1e-3)
    assert ms.peak_current == pytest.approx(5.913e6, rel=1e-3) and ms.peak_power == pytest.approx(6.26e10, rel=1e-3)
    assert _statuses(ms) == (SOTA, SOTA, EXTRA, SOTA, EXTRA) and ms.status is EXTRA

    slow = P.assess_pulse_network(PRESS, B16, 1e-2)
    assert slow.turns == 58 and slow.charge_voltage == pytest.approx(24.70e3, rel=1e-3)
    assert slow.peak_current == pytest.approx(5.098e5, rel=1e-3) and slow.peak_power == pytest.approx(6.342e9, rel=1e-3)
    assert slow.bank_energy == pytest.approx(40.74e6, rel=1e-3)
    assert _statuses(slow) == (SOTA, SOTA, SOTA, SOTA, SOTA) and slow.status is SOTA
    assert slow.margin == pytest.approx(1.012, rel=1e-2) and slow.margin >= 1.0       # an HZDR-class bank
    assert slow.warnings == () or all("skin-limited" in w for w in slow.warnings)
    # energy scales as B^2 and the ampere-turns as B
    high = P.assess_pulse_network(PRESS, B4, 1e-2)
    assert slow.magnetic_energy / high.magnetic_energy == pytest.approx(0.4, rel=1e-9)
    assert slow.ampere_turns / high.ampere_turns == pytest.approx(math.sqrt(0.4), rel=1e-9)


def test_lanl_like_insert_is_electrically_state_of_the_art():
    a = P.assess_pulse_network(INSERT, PULSED_NONDESTRUCTIVE_RECORD, 5e-3)
    assert a.bore_diameter == pytest.approx(0.015) and a.magnetic_energy == pytest.approx(4.415e6, rel=1e-3)
    assert PRESS.stored_energy(B4) / a.magnetic_energy == pytest.approx(22.5, rel=1e-2)   # energy follows the bore
    assert a.turns == 134 and a.layers == 10
    assert a.peak_current == pytest.approx(113.96e3, rel=1e-3)
    assert a.charge_voltage == pytest.approx(24.87e3, rel=1e-3)
    assert a.peak_power == pytest.approx(1.45e9, rel=1e-2) and a.bank_energy == pytest.approx(4.746e6, rel=1e-3)
    assert _statuses(a)[:4] == (SOTA, SOTA, SOTA, SOTA) and a.margin >= 1.0
    assert not a.skin_limited and a.surface_temperature_rise == 0.0
    # a stand-alone 100 T mono-coil cannot swallow its own field energy below 400 K
    assert a.thermal_status is EXTRA and a.status is EXTRA
    assert a.conductor_temperature_rise == pytest.approx(517.1, rel=1e-3)
    assert a.half_sine_temperature_rise == pytest.approx(85.5, rel=1e-2)
    recovered = P.assess_pulse_network(INSERT, PULSED_NONDESTRUCTIVE_RECORD, 5e-3, waveform="half_sine")
    assert recovered.thermal_status is SOTA and recovered.status is SOTA


# ------------------------------------------------------------------ network assessment: mechanisms
def test_skin_effect_and_transposition():
    cuag = materials.get("cuag_conductor")
    solid = P.assess_pulse_network(PRESS, B4, 1e-3)
    assert solid.skin_limited and solid.layers == 1 and solid.layer_field_step == pytest.approx(B4)
    assert solid.skin_depth == pytest.approx(eddy.skin_depth(cuag.resistivity_at(77.0), 250.0), rel=1e-12)
    assert solid.conductor_side > P.SKIN_LIMIT_RATIO * solid.skin_depth
    assert solid.surface_temperature_rise == pytest.approx(eddy.skin_temperature_rise(cuag, B4), rel=1e-12)
    assert solid.surface_temperature_rise == pytest.approx(1231.0, rel=1e-3)
    assert eddy.skin_melting_field(cuag, 77.0) == pytest.approx(89.24, rel=1e-3) and B4 > 89.24
    assert solid.surface_melts and solid.thermal_status is INFEAS
    assert any("single-turn-coil regime" in w for w in solid.warnings)
    litz = P.assess_pulse_network(PRESS, B4, 1e-3, transposed=True)
    assert litz.transposed and not litz.skin_limited and litz.surface_temperature_rise == 0.0
    assert not litz.surface_melts and litz.thermal_status is EXTRA and litz.status is EXTRA
    # many layers divide the field step across one conductor but multiply the voltage
    many = P.assess_pulse_network(PRESS, B4, 1e-3, turns=100)
    assert many.layers == 6 and many.layer_field_step == pytest.approx(B4 / 6.0)
    assert many.surface_temperature_rise == pytest.approx(solid.surface_temperature_rise / 36.0, rel=1e-9)
    assert not many.surface_melts and many.charge_voltage == pytest.approx(669e3, rel=1e-3)
    assert many.voltage_status is ASPIR and many.voltage_limited and many.status is ASPIR
    assert any("100 turns need" in w and "allows 3 turn" in w for w in many.warnings)
    # pure copper from 77 K melts only above 105 T: hot but solid at 100 T
    cu = P.assess_pulse_network(PRESS, B4, 1e-3, conductor="copper_ofhc")
    assert cu.conductor == "copper_ofhc" and cu.skin_limited and not cu.surface_melts
    assert cu.thermal_status is EXTRA and cu.surface_temperature_rise == pytest.approx(1159.6, rel=1e-3)
    # a slow, many-turn winding is still skin-limited but only warm at its surface
    slow = P.assess_pulse_network(PRESS, B4, 1e-2)
    assert slow.skin_limited and slow.surface_temperature_rise == pytest.approx(76.9, rel=1e-2)
    assert any("transposed=True" in w for w in slow.warnings)


def test_waveform_dump_and_explicit_turns():
    crow = P.assess_pulse_network(PRESS, B4, 1e-2)
    half = P.assess_pulse_network(PRESS, B4, 1e-2, waveform="half_sine")
    assert crow.waveform == "crowbar" and half.waveform == "half_sine"
    assert crow.conductor_temperature_rise == pytest.approx(665.7, rel=1e-3)          # the 10 ms pulse, no dump
    assert crow.conductor_final_temperature == pytest.approx(742.7, rel=1e-3)
    assert crow.half_sine_temperature_rise == pytest.approx(half.conductor_temperature_rise, rel=1e-12)
    assert half.conductor_temperature_rise == pytest.approx(33.6, rel=1e-2)           # energy returned to the bank
    assert crow.self_energy_temperature_rise == pytest.approx(649.6, rel=1e-3)
    assert crow.max_rise_time_thermal == pytest.approx(0.0589, rel=1e-2)
    assert crow.conductor_current_density == pytest.approx(9.35e8, rel=1e-3)
    assert any("cannot absorb its own field energy" in w for w in crow.warnings)
    assert crow.thermal_status is EXTRA and half.thermal_status is SOTA and half.status is EXTRA   # 102 MJ bank
    dumped = P.assess_pulse_network(PRESS, B4, 1e-2, dump_fraction=0.6)
    assert dumped.thermal_status is SOTA and dumped.conductor_temperature_rise == pytest.approx(275.9, rel=1e-3)
    assert dumped.charge_voltage == crow.charge_voltage and dumped.dump_fraction == 0.6
    # explicit turns override the voltage rule
    ten = P.assess_pulse_network(PRESS, B4, 1e-2, turns=10)
    assert ten.turns == 10 and ten.peak_current == pytest.approx(4.675e6, rel=1e-3) and not ten.voltage_limited
    assert ten.charge_voltage == pytest.approx(crow.charge_voltage * 10.0 / 37.0, rel=1e-9)
    assert ten.bank_energy == pytest.approx(crow.bank_energy, rel=1e-9)
    # slow pulses melt this compact winding (action limit ~59 ms); the right source is then a motor-generator
    slow = P.assess_pulse_network(PRESS, B4, 0.2)
    assert slow.thermal_status is INFEAS and slow.status is INFEAS and slow.recommended_source == "motor_generator"
    assert any("melting point" in w for w in slow.warnings)
    assert slow.transfer_efficiency == pytest.approx(math.exp(-0.2 / slow.time_constant), rel=1e-6)
    # a stricter temperature limit or a warm start changes the thermal grade only
    strict = P.assess_pulse_network(PRESS, B16, 1e-2, temperature_limit=300.0)
    assert strict.thermal_status is EXTRA and strict.charge_voltage == P.assess_pulse_network(PRESS, B16, 1e-2).charge_voltage


def test_assessment_bookkeeping_and_scaling():
    a = P.assess_pulse_network(PRESS, B4, 1e-2)
    assert isinstance(a, Assessment)
    assert a.status is worst(_statuses(a))
    assert a.bore_diameter == pytest.approx(0.1) and a.winding_thickness == pytest.approx(0.2) and a.coil_length == 0.5
    assert a.field == B4 and a.rise_time == 1e-2 and a.max_voltage == PULSED_VOLTAGE_SOTA
    assert a.conductor == "cuag_conductor" and a.fill_factor == 0.5 and a.initial_temperature == 77.0
    assert a.ampere_turns == pytest.approx(a.turns * a.peak_current, rel=1e-12)
    assert a.ampere_turns == pytest.approx(PRESS.current_for_field(B4, 1.0), rel=1e-12)
    assert a.inductance == pytest.approx(a.turns ** 2 * PRESS.inductance_from_energy(1.0), rel=1e-12)
    assert a.inductance_wheeler == pytest.approx(PRESS.inductance(float(a.turns)), rel=1e-12)
    assert 0.5 * a.inductance * a.peak_current ** 2 == pytest.approx(a.magnetic_energy, rel=1e-12)
    assert a.bank_energy == pytest.approx(a.magnetic_energy * math.exp(a.rise_time / a.time_constant), rel=1e-9)
    assert a.transfer_efficiency == pytest.approx(a.magnetic_energy / a.bank_energy, rel=1e-12)
    assert 0.5 * a.capacitance * a.charge_voltage ** 2 == pytest.approx(a.bank_energy, rel=1e-12)
    assert a.ramp_voltage == pytest.approx(2.0 * a.magnetic_energy / (a.peak_current * a.rise_time), rel=1e-12)
    assert a.ramp_voltage < a.charge_voltage                                          # linear ramp: 2/pi of it
    assert a.minimum_rise_time == pytest.approx(math.pi * a.magnetic_energy / (25e3 * a.ampere_turns), rel=1e-12)
    assert a.minimum_rise_time == pytest.approx(2.674e-4, rel=1e-3)                   # 25 kV cannot beat 0.27 ms
    assert a.homopolar_rise_time == pytest.approx(0.0570, rel=1e-3) and a.homopolar_rise_time > 5.0 * a.rise_time
    assert a.damping_regime == "underdamped" and 0.0 < a.damping_ratio < 0.01 and 0.9 < a.voltage_reversal < 1.0
    assert a.recommended_source == "capacitor_bank" and len(a.sources) == 4
    assert a.sources == P.source_comparison(a.bank_energy, a.rise_time)
    assert a.bank_volume == pytest.approx(a.bank_energy / P.CAPACITOR_ENERGY_DENSITY)
    assert a.switch_count == math.ceil(a.peak_current / P.SWITCH_CURRENT_RATING)
    assert a.peak_di_dt == pytest.approx(a.charge_voltage / a.inductance, rel=1e-12)
    # same winding, slower pulse: same energy and current, less voltage and power
    b = P.assess_pulse_network(PRESS, B4, 2e-2, turns=a.turns)
    assert b.magnetic_energy == a.magnetic_energy and b.peak_current == a.peak_current
    assert b.charge_voltage < a.charge_voltage and b.peak_power < a.peak_power and b.capacitance > a.capacitance
    # half the field: a quarter of the energy, half the current
    c = P.assess_pulse_network(PRESS, 0.5 * B4, 1e-2, turns=a.turns)
    assert c.magnetic_energy == pytest.approx(0.25 * a.magnetic_energy, rel=1e-12)
    assert c.peak_current == pytest.approx(0.5 * a.peak_current, rel=1e-12)
    assert c.charge_voltage == pytest.approx(0.5 * a.charge_voltage, rel=1e-9)
    # bank and bus inductance are costly for a single-turn coil
    s = P.assess_pulse_network(PRESS, B4, 25e-6, stray_inductance=50e-9)
    total = s.inductance + 50e-9
    assert s.turns == 1 and s.stray_inductance == 50e-9
    assert s.transfer_efficiency == pytest.approx(s.inductance / total * math.exp(-25e-6 * s.resistance / total), rel=1e-6)
    assert s.charge_voltage == pytest.approx(414.3e3, rel=1e-3) and s.bank_energy == pytest.approx(154.1e6, rel=1e-3)
    # a higher voltage ceiling buys more turns and less current
    hv = P.assess_pulse_network(PRESS, B4, 1e-2, max_voltage=50e3)
    assert hv.turns == 74 and hv.peak_current == pytest.approx(0.5 * a.peak_current, rel=1e-9)
    assert hv.voltage_status is EXTRA and hv.max_voltage == 50e3 and not hv.voltage_limited


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("call", [
    lambda: P.angular_frequency_for_rise(0.0),
    lambda: P.voltage_current_product(-1.0, 1e-3),
    lambda: P.discharge_peak_power(1e6, -1e-3),
    lambda: P.rise_time_for_product(1e6, 0.0, 1e6),
    lambda: P.rise_time_for_product(1e6, 1e3, float("inf")),
    lambda: P.inductive_transfer_efficiency(0.0, 1e-6),
    lambda: P.rlc_discharge(0.0, 1e4, 1e-6),
    lambda: P.rlc_discharge(1e-3, -1.0, 1e-6),
    lambda: P.rlc_discharge(1e-3, 1e4, 0.0),
    lambda: P.rlc_discharge(1e-3, 1e4, 1e-6, -0.1),
    lambda: P.rlc_current(1e-3, float("nan"), 1e-6),
    lambda: P.rlc_capacitor_voltage(1e-3, 1e4, float("inf")),
    lambda: P.capacitance_for_rise_time(1e-6, 0.0),
    lambda: P.capacitance_for_rise_time(0.0, 1e-4),
    lambda: P.capacitance_for_rise_time(1e-6, 1e-4, -1.0),
    lambda: P.voltage_for_peak_current(1e-3, 1e-6, -1.0),
    lambda: P.coil_electrical(PRESS, 0),
    lambda: P.coil_electrical(PRESS, 10, fill_factor=0.0),
    lambda: P.coil_electrical(PRESS, 10, fill_factor=1.5),
    lambda: P.coil_electrical(PRESS, 10, temperature=0.0),
    lambda: P.coil_electrical(PRESS, 10, temperature=2000.0),            # above the Cu-Ag eutectic
    lambda: P.coil_electrical("not a solenoid", 10),
    lambda: P.turns_for_voltage(PRESS, 0.0, 1e-3),
    lambda: P.turns_for_voltage(PRESS, B4, 1e-3, max_voltage=0.0),
    lambda: P.turns_for_voltage(PRESS, B4, 1e-3, stray_inductance=-1e-9),
    lambda: P.size_capacitor_bank(1e-6, 0.0, 1e-3),
    lambda: P.size_capacitor_bank(0.0, 1e6, 1e-3),
    lambda: P.size_capacitor_bank(1e-6, 1e6, 1e-3, stray_inductance=-1e-9),
    lambda: P.size_capacitor_bank(1e-6, 1e6, 1e-3, switch_current=0.0),
    lambda: P.size_capacitor_bank(1e-6, 1e6, 1e-3, specific_energy=0.0),
    lambda: P.conductor_heating(PRESS, -1.0, 1e-3),
    lambda: P.conductor_heating(PRESS, B4, 0.0),
    lambda: P.conductor_heating(PRESS, B4, 1e-3, waveform="square"),
    lambda: P.conductor_heating(PRESS, B4, 1e-3, dump_fraction=1.5),
    lambda: P.conductor_heating(PRESS, B4, 1e-3, temperature_limit=50.0),
    lambda: P.conductor_heating(PRESS, B4, 1e-3, fill_factor=0.0),
    lambda: P.HomopolarGenerator(0.0, 1.0),
    lambda: P.HomopolarGenerator(0.5, 0.0),
    lambda: P.HomopolarGenerator(0.5, 1.0, inner_radius=0.5),
    lambda: P.HomopolarGenerator(0.5, 1.0, tip_speed=-1.0),
    lambda: P.HomopolarGenerator(0.5, 1.0, field=0.0),
    lambda: P.HomopolarGenerator(0.5, 1.0).quarter_period(0.0),
    lambda: P.HomopolarGenerator(0.5, 1.0).discharge(1e-6, -1.0),
    lambda: P.homopolar_for_energy(0.0),
    lambda: P.homopolar_for_energy(1e8, aspect_ratio=0.0),
    lambda: P.rotor_tip_speed_limit(0.0),
    lambda: P.source_comparison(0.0, 1e-3),
    lambda: P.source_comparison(1e6, 0.0),
    lambda: P.assess_pulse_network(PRESS, 0.0, 1e-3),
    lambda: P.assess_pulse_network(PRESS, B4, 0.0),
    lambda: P.assess_pulse_network(PRESS, B4, 1e-3, turns=0),
    lambda: P.assess_pulse_network(PRESS, B4, 1e-3, turns=2.5),
    lambda: P.assess_pulse_network(PRESS, B4, 1e-3, turns=True),
    lambda: P.assess_pulse_network(PRESS, B4, 1e-3, transposed="yes"),
    lambda: P.assess_pulse_network(PRESS, B4, 1e-3, max_voltage=-1.0),
    lambda: P.assess_pulse_network(PRESS, B4, 1e-3, waveform="ringing"),
    lambda: P.assess_pulse_network(PRESS, B4, 1e-3, initial_temperature=1500.0),
    lambda: P.assess_pulse_network(PRESS, B4, 1e-3, temperature_limit=77.0),
    lambda: P.assess_pulse_network((0.05, 0.25, 0.5), B4, 1e-3),
    lambda: P.assess_pulse_network(PRESS, "strong", 1e-3),
])
def test_invalid_inputs_raise_input_error(call):
    with pytest.raises(InputError):
        call()


def test_unknown_conductor_and_waveform_aliases():
    with pytest.raises(UnknownMaterialError):
        P.coil_electrical(PRESS, 10, conductor="unobtainium")
    with pytest.raises(UnknownMaterialError):
        P.assess_pulse_network(PRESS, B4, 1e-3, conductor="unobtainium")
    assert P.conductor_heating(PRESS, B4, 1e-2, " Half_Sine ").waveform == "half_sine"
    by_alias = P.assess_pulse_network(PRESS, B4, 1e-2, conductor="cuag")
    assert by_alias.json() == P.assess_pulse_network(PRESS, B4, 1e-2).json()         # deterministic


# ------------------------------------------------------------------ serialisation / metadata
def test_results_round_trip_json_and_render_text():
    a = P.assess_pulse_network(PRESS, B4, 25e-6)
    results = (
        P.rlc_discharge(1e-3, 1e4, 1e-6), P.coil_electrical(PRESS, 37), P.size_capacitor_bank(1e-6, 1e6, 1e-3),
        P.conductor_heating(PRESS, B4, 1e-2), P.homopolar_for_energy(1e8), a.sources[0], a,
    )
    for r in results:
        assert isinstance(r, ResultMixin)
        data = json.loads(r.json())
        assert isinstance(data, dict) and data
        assert r.text().startswith(type(r).__name__)
    data = json.loads(a.json())
    assert data["status"] == "infeasible" and data["thermal_status"] == "infeasible" and data["turns"] == 1
    assert data["voltage_status"] == "aspirational" and data["surface_melts"] is True
    assert data["sources"][3]["key"] == "flux_compression_generator" and data["sources"][3]["reusable"] is False
    assert data["charge_voltage"] == pytest.approx(a.charge_voltage)
    lossless = json.loads(P.rlc_discharge(1e-3, 1e4, 1e-6).json())
    assert lossless["crowbar_time_constant"] is None                                 # inf -> null keeps JSON strict
    bank = json.loads(P.size_capacitor_bank(1e-6, 1e6, 1e-3).json())
    assert bank["action_crowbar"] is None and len(bank["bank_mass_range"]) == 2
    txt = a.text()
    assert "charge_voltage [V] = 2.674e+05" in txt and "peak_power [W]" in txt and "sources:" in txt
    assert "quarter-period law" in txt and "homopolar generator" in txt
    with pytest.raises(Exception):
        a.turns = 2  # frozen


def test_module_metadata():
    assert set(P.__all__) <= set(dir(P))
    assert isinstance(P.REFERENCES, dict) and len(P.REFERENCES) >= 8
    assert all(isinstance(k, str) and isinstance(v, str) and len(v) > 20 for k, v in P.REFERENCES.items())
    for name in P.__all__:
        obj = getattr(P, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 80 and "Reference" in obj.__doc__, name
    for method in ("mass", "moment_of_inertia", "angular_speed", "stored_energy", "rotor_stress",
                   "voltage_constant", "voltage", "equivalent_capacitance", "quarter_period", "peak_current",
                   "discharge"):
        assert "Reference" in getattr(P.HomopolarGenerator, method).__doc__, method
    assert P.REGIMES == ("underdamped", "critical", "overdamped") and P.WAVEFORMS == ("crowbar", "half_sine")
    assert P.DEFAULT_MAX_VOLTAGE == PULSED_VOLTAGE_SOTA == 25e3
    assert P.PRECOOL_TEMPERATURE == 77.0 and P.COIL_TEMPERATURE_LIMIT == 400.0 and P.DEFAULT_FILL_FACTOR == 0.5
    lo, hi = P.CAPACITOR_SPECIFIC_ENERGY_RANGE
    assert lo < P.CAPACITOR_SPECIFIC_ENERGY < hi
    lo, hi = P.CAPACITOR_ENERGY_DENSITY_RANGE
    assert lo < P.CAPACITOR_ENERGY_DENSITY < hi
    assert P.HPG_ENERGY_SOTA == 60e6 and P.HPG_ENERGY_RECORD == 500e6                 # from benchmarks.ENERGY_LADDER
    assert P.MOTOR_GENERATOR_ENERGY == pytest.approx(0.6e9) and P.MOTOR_GENERATOR_POWER == 1.4e9
    assert P.FCG_MIN_RISE_TIME < P.FCG_MAX_RISE_TIME < P.MOTOR_GENERATOR_MIN_RISE_TIME
