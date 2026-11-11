"""Tests for magform.eddy: skin effect, field diffusion, skin heating and tube compression.

Every numeric check quoted in the module's docstrings is pinned here, and the
solvers are compared with closed-form results: the exponential shell response,
the erfc/E1 solution of a field step, the Duhamel surface heating of a
half-sine pulse, flux conservation and the work-energy balance of the tube.
"""

from __future__ import annotations

import json
import math

import pytest

from magform import eddy, magnetics, materials, plasticity
from magform._numerics import bisect, trapz
from magform.benchmarks import EMF_CONVENTIONAL_FIELD
from magform.constants import MU0, T_ROOM
from magform.errors import InputError, UnknownMaterialError

RHO_CU = 1.71e-8                      # ohm m, copper_ofhc at 293.15 K
R_OUT, WALL = 0.025, 0.002            # reference tube: 50 mm OD, 2 mm wall
R_MID = R_OUT - 0.5 * WALL
FREQ = eddy.pulse_frequency_recommendation("copper_ofhc", WALL).frequency   # skin depth = wall/3
HALF = 0.5 / FREQ                     # half-sine drive of the same quarter period (51.3 us)


def _const_cu() -> materials.Material:
    """Copper with a temperature-independent resistivity (linear diffusion, analytic solutions)."""
    return materials.get("copper_ofhc").with_changes(key="copper_const_rho", resistivity_temp_coeff=0.0)


def _tube(material="copper_ofhc", **kw) -> eddy.TubeCompression:
    return eddy.TubeCompression(material, R_OUT, WALL, **kw)


def _pulse(field: float):
    return eddy.half_sine_field(field, HALF)


def _exp_integral(x: float) -> float:
    """E1(x) from its convergent power series (0 < x <~ 10)."""
    total = -0.5772156649015329 - math.log(x)
    term = 1.0
    for k in range(1, 80):
        term *= -x / k
        total -= term / k
    return total


def _simpson(f, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return total * h / 3.0


def _duhamel_kappa(s_end: float) -> float:
    """Exact surface heating of a half-sine pulse on a constant-property half-space.

    q_s / (B^2/2 mu0) = 2 pi int_0^s G^2 ds',  G(s) = int_0^min(s,1) cos(pi x)/sqrt(s - x) dx,
    with s = t / T_half (Duhamel superposition of the step solution J_s = B/(mu0 sqrt(pi D t))).
    """
    def g(s: float) -> float:
        if s <= 0.0:
            return 0.0
        lo, hi = math.sqrt(max(s - 1.0, 0.0)), math.sqrt(s)
        return 2.0 * _simpson(lambda u: math.cos(math.pi * (s - u * u)), lo, hi, 60)

    total = _simpson(lambda s: g(s) ** 2, 0.0, min(s_end, 1.0), 200)
    if s_end > 1.0:  # G has a square-root kink at s = 1: integrate in w = sqrt(s - 1)
        total += _simpson(lambda w: g(1.0 + w * w) ** 2 * 2.0 * w, 0.0, math.sqrt(s_end - 1.0), 200)
    return 2.0 * math.pi * total


# ---------------------------------------------------------------- constants / metadata
def test_module_metadata():
    assert set(eddy.__all__) <= set(dir(eddy))
    assert len(eddy.REFERENCES) >= 10
    assert all(isinstance(k, str) and isinstance(v, str) and len(v) > 40 for k, v in eddy.REFERENCES.items())
    for name in eddy.__all__:
        obj = getattr(eddy, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 40, name
            if not isinstance(obj, type):
                assert "Reference" in obj.__doc__, name     # every model function cites its source
    assert eddy.TubeCompression.simulate.__doc__ and "Reference" in eddy.TubeCompression.simulate.__doc__
    assert eddy.KAPPA_HALF_SINE == 1.0
    assert eddy.KAPPA_STEP == pytest.approx(math.sqrt(2.0) / (2.0 - math.sqrt(2.0)), rel=1e-2)   # 2.414
    assert eddy.STRAIN_RATE_FLOOR == 1e-3
    assert (eddy.DEFAULT_TUBE_STEPS, eddy.ARREST_STEPS, eddy.HISTORY_MAX_POINTS) == (20_000, 50, 2000)
    assert eddy.DEFAULT_DIFFUSION_STEPS == 2000


# ---------------------------------------------------------------- skin depth, time constants, regimes
def test_skin_depth_spec_check_and_inverse():
    delta = eddy.skin_depth(1.68e-8, 1e4)
    assert delta == pytest.approx(0.652e-3, rel=1e-3)                       # docstring check
    assert delta == pytest.approx(math.sqrt(1.68e-8 / (math.pi * 1e4 * MU0)), rel=1e-12)
    assert eddy.skin_depth(1.68e-8, 1e4, 100.0) == pytest.approx(delta / 10.0, rel=1e-12)
    assert eddy.skin_depth(4 * 1.68e-8, 1e4) == pytest.approx(2.0 * delta, rel=1e-12)
    assert eddy.skin_depth(1.68e-8, 4e4) == pytest.approx(0.5 * delta, rel=1e-12)
    assert eddy.frequency_for_skin_depth(1.68e-8, delta) == pytest.approx(1e4, rel=1e-12)
    for rho, depth in ((RHO_CU, 1e-3), (7e-7, 3e-4), (5.28e-8, 2e-3)):
        assert eddy.skin_depth(rho, eddy.frequency_for_skin_depth(rho, depth)) == pytest.approx(depth, rel=1e-12)


def test_skin_depth_for_material_follows_the_resistivity(cu):
    room = eddy.skin_depth_for_material(cu, 1e4)
    assert room == eddy.skin_depth(cu.resistivity, 1e4) == pytest.approx(0.658e-3, rel=1e-3)
    cold = eddy.skin_depth_for_material("copper_ofhc", 1e4, 77.0)
    hot = eddy.skin_depth_for_material("Cu", 1e4, 600.0)
    assert cold < room < hot
    assert cold == pytest.approx(0.2608e-3, rel=1e-3)
    assert hot / room == pytest.approx(math.sqrt(cu.resistivity_at(600.0) / cu.resistivity), rel=1e-12)


def test_shell_time_constant_and_diffusion_time():
    tau = eddy.shell_time_constant(RHO_CU, 0.02, 0.002)
    tau_d = eddy.diffusion_time(RHO_CU, 0.002)
    assert tau == pytest.approx(MU0 * 0.02 * 0.002 / (2.0 * RHO_CU), rel=1e-12) == pytest.approx(1.4698e-3, rel=1e-4)
    assert tau_d == pytest.approx(MU0 * 0.002 ** 2 / RHO_CU, rel=1e-12) == pytest.approx(2.9395e-4, rel=1e-4)
    assert tau / tau_d == pytest.approx(0.02 / (2.0 * 0.002), rel=1e-12)    # a/(2h), as documented
    # a wall three skin depths thick has tau_d = 9 mu0 delta^2/rho = 9/(pi f)
    assert tau_d == pytest.approx(9.0 / (math.pi * FREQ), rel=1e-9)


def test_shell_regime_and_effective_current_depth():
    delta = eddy.skin_depth(RHO_CU, 1e4)
    assert eddy.shell_regime(RHO_CU, 0.9 * delta, 1e4) == "thin"
    assert eddy.shell_regime(RHO_CU, 1.1 * delta, 1e4) == "thick"
    assert eddy.shell_regime(RHO_CU, 0.002, 1e4) == "thick" and eddy.shell_regime(RHO_CU, 0.0002, 1e4) == "thin"
    assert eddy.shell_regime(RHO_CU, 0.002, 100.0) == "thin"               # slow pulse: field leaks through
    assert eddy.effective_current_depth(2e-3, 0.5e-3) == 0.5e-3
    assert eddy.effective_current_depth(2e-4, 0.5e-3) == 2e-4


@pytest.mark.parametrize("field, pressure", [(63.41, 1.6e9), (70.90, 2.0e9), (100.27, 4.0e9)])
def test_pressure_on_shell_spec_checks(field, pressure):
    assert eddy.pressure_on_shell(field, 0.0) == pytest.approx(pressure, rel=1e-3)
    assert eddy.pressure_on_shell(-field, 0.0) == eddy.pressure_on_shell(field, 0.0)
    assert eddy.pressure_on_shell(field, 0.0) == pytest.approx(magnetics.magnetic_pressure(field), rel=1e-12)


def test_pressure_on_shell_static_field_and_emf_window():
    assert eddy.pressure_on_shell(50.0, 50.0) == 0.0                         # fully diffused: no force
    assert eddy.pressure_on_shell(0.0, 30.0) == -eddy.pressure_on_shell(30.0, 0.0)   # trapped flux pushes outward
    assert eddy.pressure_on_shell(40.0, 30.0) == pytest.approx(
        magnetics.pressure_with_background(30.0, 10.0), rel=1e-12)
    lo, hi = (eddy.pressure_on_shell(b, 0.0) for b in EMF_CONVENTIONAL_FIELD)
    assert lo == pytest.approx(0.159e9, rel=1e-2) and hi == pytest.approx(0.995e9, rel=1e-2)   # 0.16-1 GPa
    # the README's 1.6-4 GPa window is 1.6-25x the industrial range (module docstring)
    assert eddy.pressure_on_shell(63.41, 0.0) / hi == pytest.approx(1.61, rel=1e-2)
    assert eddy.pressure_on_shell(100.27, 0.0) / lo == pytest.approx(25.1, rel=1e-2)


def test_sheet_shielding_factor_and_transmitted_pressure():
    d = 1e-3
    assert eddy.sheet_shielding_factor(0.0, d) == 0.0
    assert eddy.sheet_shielding_factor(d, d) == pytest.approx(1.0 - math.exp(-2.0), rel=1e-12)
    assert eddy.sheet_shielding_factor(3 * d, d) == pytest.approx(1.0 - math.exp(-6.0), rel=1e-12)   # 0.9975
    assert eddy.sheet_shielding_factor(1.0, d) == 1.0
    factors = [eddy.sheet_shielding_factor(k * d, d) for k in (0.1, 0.5, 1.0, 2.0, 5.0)]
    assert all(b > a for a, b in zip(factors, factors[1:])) and all(0.0 < f < 1.0 for f in factors)
    assert eddy.sheet_shielding_factor(1e-6 * d, d) == pytest.approx(2e-6, rel=1e-5)    # thin limit 2h/delta
    assert eddy.transmitted_pressure(4e9, 2e-3, 2e-3 / 3) == pytest.approx(3.990e9, rel=1e-3)
    assert eddy.transmitted_pressure(4e9, 0.0, d) == 0.0
    assert eddy.transmitted_pressure(4e9, d, d) == pytest.approx(4e9 * (1.0 - math.exp(-2.0)), rel=1e-12)


# ---------------------------------------------------------------- skin heating
def test_surface_heating_energy_density():
    q = eddy.surface_heating_energy_density(100.0)
    assert q == pytest.approx(100.0 ** 2 / (2.0 * MU0), rel=1e-12) == pytest.approx(3.979e9, rel=1e-3)
    assert q == pytest.approx(magnetics.energy_density(100.0), rel=1e-12)   # kappa = 1: the magnetic pressure
    assert eddy.surface_heating_energy_density(-100.0) == q
    assert eddy.surface_heating_energy_density(100.0, eddy.KAPPA_STEP) == pytest.approx(2.4 * q, rel=1e-12)
    assert eddy.surface_heating_energy_density(50.0) == pytest.approx(q / 4.0, rel=1e-12)


def test_skin_temperature_rise_spec_checks(cu, al):
    d_cu = eddy.skin_temperature_rise(cu, 100.0)
    assert d_cu == pytest.approx(1153.4, rel=1e-3) and d_cu == pytest.approx(1160.0, rel=1e-2)   # docstring: ~1160 K
    assert 1100.0 < d_cu < 1250.0                                         # Herlach (1986): ~1200 K per (100 T)^2
    d_al = eddy.skin_temperature_rise(al, 63.4)
    assert d_al == pytest.approx(661.1, rel=1e-3) and d_al == pytest.approx(660.0, rel=1e-2)     # docstring: ~660 K
    # dT = kappa P_mag / (rho c): independent of resistivity and of the pulse length
    assert d_cu == pytest.approx(magnetics.magnetic_pressure(100.0) / cu.volumetric_heat_capacity, rel=1e-12)
    assert eddy.skin_temperature_rise(cu, 100.0, 2.4) == pytest.approx(2.4 * d_cu, rel=1e-12)
    assert eddy.skin_temperature_rise(cu, 50.0) == pytest.approx(d_cu / 4.0, rel=1e-12)
    assert eddy.skin_temperature_rise("copper_ofhc", 0.0) == 0.0


def test_skin_melting_field_spec_checks(cu, al):
    assert eddy.skin_melting_field(cu) == pytest.approx(95.99, rel=1e-3)                    # docstring: ~96 T
    assert eddy.skin_melting_field(al) == pytest.approx(58.45, rel=1e-3)                    # ~58 T (solidus 855 K)
    assert eddy.skin_melting_field("aluminum_1100") == pytest.approx(61.93, rel=1e-3)       # ~62 T
    for m in (cu, al, materials.get("tungsten")):
        b_melt = eddy.skin_melting_field(m)
        assert T_ROOM + eddy.skin_temperature_rise(m, b_melt) == pytest.approx(m.melting_point, rel=1e-12)
    # complete melting needs the latent heat as well
    full = eddy.skin_melting_field(cu, include_latent=True)
    heat = cu.density * (cu.specific_heat * (cu.melting_point - T_ROOM) + cu.latent_heat_fusion)
    assert full == pytest.approx(math.sqrt(2.0 * MU0 * heat), rel=1e-12) == pytest.approx(117.99, rel=1e-3)
    assert full > eddy.skin_melting_field(cu)
    # B_melt ~ kappa^-1/2; a colder start raises it
    assert eddy.skin_melting_field(cu, kappa=2.4) == pytest.approx(95.99 / math.sqrt(2.4), rel=1e-3)
    assert eddy.skin_melting_field(cu, initial_temperature=77.0) == pytest.approx(105.30, rel=1e-3)


def test_skin_melting_fields_place_the_mhd_label():
    """Module docstring: 'MHD' is literally true only where the skin melts (kappa = 1)."""
    fields = {k: eddy.skin_melting_field(k) for k in
              ("aluminum_6061", "copper_ofhc", "ti6al4v", "titanium_cp2", "steel_1018", "tungsten")}
    assert fields["aluminum_6061"] == pytest.approx(58.4, rel=1e-2)
    assert fields["copper_ofhc"] == pytest.approx(96.0, rel=1e-2)
    assert fields["ti6al4v"] == pytest.approx(96.3, rel=1e-2)
    assert fields["titanium_cp2"] == pytest.approx(98.8, rel=1e-2)
    assert fields["steel_1018"] == pytest.approx(117.2, rel=1e-2)
    assert fields["tungsten"] == pytest.approx(148.5, rel=1e-2)
    # the README's 63-100 T window: an aluminium skin melts at its lower end, a copper skin at its upper end
    assert fields["aluminum_6061"] < 63.41 < fields["copper_ofhc"] < 100.27 < fields["steel_1018"]


# ---------------------------------------------------------------- lumped interior field
def test_interior_field_linear_ramp_is_exact():
    tau, rate = 1e-3, 1e4
    t, b_in = eddy.interior_field(lambda x: rate * x, tau, 5e-3, 50)
    assert len(t) == len(b_in) == 51 and t[0] == 0.0 and t[-1] == 5e-3
    for ti, bi in zip(t, b_in):   # dB_in/dt = (k t - B_in)/tau  ->  B_in = k (t - tau (1 - exp(-t/tau)))
        assert bi == pytest.approx(rate * (ti - tau * (1.0 - math.exp(-ti / tau))), abs=1e-11)
    # late-time lag of a ramp is k tau
    assert rate * t[-1] - b_in[-1] == pytest.approx(rate * tau * (1.0 - math.exp(-5.0)), rel=1e-10)


def test_interior_field_sine_drive_converges_second_order():
    tau, amp, omega = 1e-3, 50.0, 2.0 * math.pi * 1e3

    def exact(x: float) -> float:
        wt = omega * tau
        return amp / (1.0 + wt * wt) * (math.sin(omega * x) - wt * math.cos(omega * x) + wt * math.exp(-x / tau))

    def max_error(n: int) -> float:
        t, b_in = eddy.interior_field(lambda x: amp * math.sin(omega * x), tau, 3e-3, n)
        return max(abs(b - exact(x)) for x, b in zip(t, b_in))

    e200, e400, e800 = max_error(200), max_error(400), max_error(800)
    assert e200 / e400 == pytest.approx(4.0, rel=5e-2) and e400 / e800 == pytest.approx(4.0, rel=5e-2)
    assert e800 < 1e-3
    # steady state: amplitude attenuated by 1/sqrt(1 + (omega tau)^2) = 0.157
    t, b_in = eddy.interior_field(lambda x: amp * math.sin(omega * x), tau, 20e-3, 4000)
    late = [abs(b) for x, b in zip(t, b_in) if x > 15e-3]
    assert max(late) == pytest.approx(amp / math.sqrt(1.0 + (omega * tau) ** 2), rel=2e-3)


def test_interior_field_starts_in_equilibrium_and_limits():
    # a field that was steady before t = 0 has fully diffused in: no shielding, no pressure
    t, b_in = eddy.interior_field(lambda x: 30.0, 1e-3, 5e-3, 10)
    assert b_in == [30.0] * 11
    assert all(eddy.pressure_on_shell(30.0, b) == 0.0 for b in b_in)
    # a jump on top of it relaxes with the shell time constant
    t, b_in = eddy.interior_field(lambda x: 30.0 + (20.0 if x > 0.0 else 0.0), 1e-3, 10e-3, 1000)
    assert b_in[0] == 30.0 and b_in[-1] == pytest.approx(50.0, abs=2e-3)
    assert b_in[100] == pytest.approx(30.0 + 20.0 * (1.0 - math.exp(-1.0)), rel=2e-3)   # t = tau
    assert all(b2 >= b1 for b1, b2 in zip(b_in, b_in[1:]))
    # tau -> infinity: perfect shield; tau -> 0: transparent wall
    _, shielded = eddy.interior_field(lambda x: 1e4 * x, 1e6, 1e-3, 10)
    assert max(abs(b) for b in shielded) < 1e-8
    t, leaky = eddy.interior_field(lambda x: 1e4 * x, 1e-12, 1e-3, 10)
    assert all(b == pytest.approx(1e4 * x, abs=1e-7) for x, b in zip(t, leaky))


# ---------------------------------------------------------------- action integral, adiabatic Joule heating
def test_action_integral():
    assert eddy.action_integral([2.0, 2.0], [0.0, 3.0]) == pytest.approx(12.0)
    t = [1e-6 * i for i in range(1001)]
    current = [1e6 * math.sin(math.pi * x / 1e-3) for x in t]
    assert eddy.action_integral(current, t) == pytest.approx(1e12 * 1e-3 / 2.0, rel=1e-9)   # I^2 T/2
    assert eddy.action_integral([5.0], [0.0]) == 0.0
    assert eddy.action_integral([-3.0, -3.0], [0.0, 1.0]) == pytest.approx(9.0)


def test_specific_action_spec_check_and_closed_form(cu):
    action = eddy.specific_action_to_temperature(cu, 293.0, 1356.0)
    assert action == pytest.approx(8.5e16, rel=1e-2) and 8e16 < action < 1.2e17      # docstring check
    # rho_e linear in T  ->  A = rho c /(rho_0 alpha) ln[(1 + alpha (T2 - T0)) / (1 + alpha (T1 - T0))]
    alpha, rho_c = cu.resistivity_temp_coeff, cu.density * cu.specific_heat

    def closed(t1: float, t2: float) -> float:
        ratio = (1.0 + alpha * (t2 - T_ROOM)) / (1.0 + alpha * (t1 - T_ROOM))
        return rho_c / (cu.resistivity * alpha) * math.log(ratio)

    assert action == pytest.approx(closed(293.0, 1356.0), rel=1e-4)
    assert eddy.specific_action_to_temperature(cu, T_ROOM, 600.0) == pytest.approx(closed(T_ROOM, 600.0), rel=1e-5)
    assert eddy.specific_action_to_temperature(cu, 400.0, 400.0) == 0.0
    # additive in temperature, and a cold start (lower resistivity) allows more action per kelvin
    whole = eddy.specific_action_to_temperature(cu, 300.0, 900.0, n=4000)
    parts = (eddy.specific_action_to_temperature(cu, 300.0, 500.0, n=4000)
             + eddy.specific_action_to_temperature(cu, 500.0, 900.0, n=4000))
    assert whole == pytest.approx(parts, rel=1e-6)
    assert eddy.specific_action_to_temperature(cu, 77.0, 177.0) > eddy.specific_action_to_temperature(cu, 293.0, 393.0)


def test_adiabatic_joule_heating_is_consistent_with_the_specific_action(cu):
    j = 5e9                                                      # A/m^2, constant
    action = eddy.specific_action_to_temperature(cu, T_ROOM, 600.0)
    res = eddy.adiabatic_joule_heating([j, j], [0.0, action / j ** 2], cu)
    assert res.final_temperature == pytest.approx(600.0, abs=0.01)
    assert res.delta_t == pytest.approx(res.final_temperature - res.initial_temperature)
    assert res.initial_temperature == T_ROOM
    assert res.action == pytest.approx(action, rel=1e-12) and not res.melted and res.time_to_melt is None
    # closed form T(A) = T0 + (exp(alpha rho_0 A/(rho c)) - 1)/alpha
    alpha, rho_c = cu.resistivity_temp_coeff, cu.density * cu.specific_heat
    exact = T_ROOM + math.expm1(alpha * cu.resistivity * action / rho_c) / alpha
    assert res.final_temperature == pytest.approx(exact, rel=1e-9)
    # only the action matters: a coarse and a fine sampling of the same half-sine heat identically
    finals = []
    for n in (11, 101, 1001):
        t = [1e-3 * i / (n - 1) for i in range(n)]
        r = eddy.adiabatic_joule_heating([4e9 * math.sin(math.pi * x / 1e-3) for x in t], t, "copper_ofhc")
        assert r.action == pytest.approx(4e9 ** 2 * 1e-3 / 2.0, rel=1e-9)
        finals.append(r.final_temperature)
    assert finals[0] == pytest.approx(336.04, rel=1e-4) and max(finals) - min(finals) < 1e-6
    idle = eddy.adiabatic_joule_heating([0.0, 0.0], [0.0, 1.0], cu)
    assert idle.delta_t == 0.0 and idle.action == 0.0 and idle.final_temperature == T_ROOM
    assert eddy.adiabatic_joule_heating([1e9], [0.0], cu).delta_t == 0.0
    assert eddy.adiabatic_joule_heating([j, j], [0.0, 1e-4], cu, initial_temperature=77.0).initial_temperature == 77.0


def test_adiabatic_joule_heating_detects_melting(cu):
    alpha, rho_c = cu.resistivity_temp_coeff, cu.density * cu.specific_heat
    a_melt = rho_c / (cu.resistivity * alpha) * math.log(1.0 + alpha * (cu.melting_point - T_ROOM))
    assert eddy.specific_action_to_temperature(cu, T_ROOM, cu.melting_point) == pytest.approx(a_melt, rel=1e-4)
    j = 5e9
    t_melt = a_melt / j ** 2
    res = eddy.adiabatic_joule_heating([j, j], [0.0, 2.0 * t_melt], cu)
    assert res.melted is True and res.final_temperature == cu.melting_point
    assert res.time_to_melt == pytest.approx(t_melt, rel=1e-4)
    assert res.action == pytest.approx(j * j * 2.0 * t_melt, rel=1e-12)       # always the full integral
    just_short = eddy.adiabatic_joule_heating([j, j], [0.0, 0.99 * t_melt], cu)
    assert just_short.melted is False and just_short.final_temperature < cu.melting_point
    # half-sine: J0^2 (t/2 - T sin(2 pi t/T)/(4 pi)) = A_melt
    j0, period = 1.5e10, 1e-3
    t = [period * i / 1000 for i in range(1001)]
    res = eddy.adiabatic_joule_heating([j0 * math.sin(math.pi * x / period) for x in t], t, cu)

    def action_excess(x: float) -> float:
        return j0 ** 2 * (x / 2.0 - period * math.sin(2.0 * math.pi * x / period) / (4.0 * math.pi)) - a_melt

    exact = bisect(action_excess, 0.0, period)
    assert res.melted and res.time_to_melt == pytest.approx(exact, rel=1e-4)
    assert exact == pytest.approx(0.634e-3, rel=1e-3)


# ---------------------------------------------------------------- 1-D magnetic diffusion into a slab
def test_slab_step_penetration_depth_matches_the_erfc_profile():
    cu0 = _const_cu()
    t_end = 1e-4
    scale = math.sqrt(2.0 * cu0.resistivity * t_end / MU0)
    res = eddy.slab_field_diffusion(cu0, lambda t: 10.0, 8e-3, t_end)
    assert res.penetration_depth == pytest.approx(0.90 * scale, rel=2e-3)          # docstring: 0.90 sqrt(2 rho t/mu0)
    # exact coefficient: erfc(x) = 1/e at x = 0.63672, i.e. 2 x / sqrt(2) = 0.90045
    x = bisect(lambda u: math.erfc(u) - math.exp(-1.0), 0.0, 2.0)
    assert res.penetration_depth == pytest.approx(math.sqrt(2.0) * x * scale, rel=5e-4)
    # diffusive scaling: four times the time, twice the depth
    later = eddy.slab_field_diffusion(cu0, lambda t: 10.0, 16e-3, 4.0 * t_end)
    assert later.penetration_depth == pytest.approx(2.0 * res.penetration_depth, rel=2e-3)
    assert (res.n_nodes, res.n_steps) == (60, eddy.DEFAULT_DIFFUSION_STEPS)
    assert res.material == "copper_const_rho" and res.thickness == 8e-3 and res.t_end == t_end


def test_slab_step_heating_profile_matches_the_exponential_integral():
    """dT(z) = B^2 E1(z^2/(2 D t)) / (mu0 pi rho c) for a field step (slab_field_diffusion docstring)."""
    cu0 = _const_cu()
    b0, t_end = 10.0, 1e-4
    d = cu0.resistivity / MU0
    scale = math.sqrt(2.0 * d * t_end)
    rho_c = cu0.density * cu0.specific_heat
    res = eddy.slab_field_diffusion(cu0, lambda t: b0, 6e-3, t_end, n_nodes=61)
    assert _exp_integral(1.0) == pytest.approx(0.2193839, rel=1e-6)
    checked = 0
    for z, temp in zip(res.z[1:], res.final_temperature_profile[1:]):
        if 0.3 * scale <= z <= 2.0 * scale:
            exact = b0 ** 2 * _exp_integral(z * z / (2.0 * d * t_end)) / (MU0 * math.pi * rho_c)
            assert temp - T_ROOM == pytest.approx(exact, rel=1e-2)
            checked += 1
    assert checked >= 20
    # the profile falls monotonically from the surface, which is the hottest point
    profile = res.final_temperature_profile
    assert all(b <= a for a, b in zip(profile, profile[1:]))
    assert res.surface_peak_temperature == profile[0] == res.max_temperature


def test_slab_step_joule_energy_bookkeeping_and_first_order_convergence():
    """Joule energy of a step = (B^2/mu0) sqrt(2 D t/pi) = 2.41 x the stored field energy (KAPPA_STEP)."""
    cu0 = _const_cu()
    b0, t_end, h = 10.0, 1e-4, 6e-3
    d = cu0.resistivity / MU0
    rho_c = cu0.density * cu0.specific_heat
    exact = b0 ** 2 / MU0 * math.sqrt(2.0 * d * t_end / math.pi)
    stored = b0 ** 2 / (2.0 * MU0) * 2.0 * math.sqrt(d * t_end) * (2.0 - math.sqrt(2.0)) / math.sqrt(math.pi)
    assert exact / stored == pytest.approx(math.sqrt(2.0) / (2.0 - math.sqrt(2.0)), rel=1e-12)
    energy = {}
    for nn in (31, 61, 121):
        res = eddy.slab_field_diffusion(cu0, lambda t: b0, h, t_end, n_nodes=nn)
        energy[nn] = res.joule_energy_per_area
        # the heat in the cells is exactly the Joule energy
        dz = h / (nn - 1)
        heat = sum(temp - T_ROOM for temp in res.final_temperature_profile[1:]) * rho_c * dz
        assert heat == pytest.approx(res.joule_energy_per_area, rel=1e-9)
        assert len(res.z) == len(res.final_temperature_profile) == nn
        assert res.z[0] == 0.0 and res.z[1] == pytest.approx(0.5 * dz) and res.z[-1] == pytest.approx(h - 0.5 * dz)
    # the initial discontinuity costs O(dz): the deficit halves with the cell size ...
    deficits = [1.0 - energy[nn] / exact for nn in (31, 61, 121)]
    assert deficits[0] == pytest.approx(0.053, rel=5e-2)
    assert deficits[0] / deficits[1] == pytest.approx(2.0, rel=3e-2)
    assert deficits[1] / deficits[2] == pytest.approx(2.0, rel=3e-2)
    # ... and Richardson extrapolation recovers the analytic energy
    assert 2.0 * energy[121] - energy[61] == pytest.approx(exact, rel=1e-3)


def test_slab_half_sine_surface_heating_matches_the_duhamel_solution():
    """Constant properties: kappa = 1.64 at the field peak, 2.19 at the end of the pulse, 2.42-2.43 after it."""
    exact = {s: _duhamel_kappa(s) for s in (0.5, 1.0, 3.0)}
    assert exact[0.5] == pytest.approx(1.638, rel=1e-3)
    assert exact[1.0] == pytest.approx(2.186, rel=1e-3)
    assert exact[3.0] == pytest.approx(2.425, rel=1e-3)
    assert _duhamel_kappa(40.0) == pytest.approx(2.4306, rel=1e-3)          # asymptote
    cu0 = _const_cu()
    rho_c = cu0.density * cu0.specific_heat
    field, period = 20.0, 1e-4
    delta = eddy.skin_depth(cu0.resistivity, 1.0 / (2.0 * period))
    p_mag = field ** 2 / (2.0 * MU0)
    kappa = {}
    for s in (0.5, 1.0, 3.0):
        res = eddy.slab_field_diffusion(cu0, eddy.half_sine_field(field, period), 6.0 * delta, s * period)
        kappa[s] = (res.surface_peak_temperature - T_ROOM) * rho_c / p_mag
        assert kappa[s] == pytest.approx(exact[s], rel=3e-3)
        assert not res.melted and res.melt_depth == 0.0
    # the numbers quoted in the docstrings of slab_field_diffusion and surface_heating_energy_density
    assert kappa[0.5] == pytest.approx(1.64, rel=5e-3)
    assert kappa[1.0] == pytest.approx(2.19, rel=5e-3)
    assert kappa[3.0] == pytest.approx(2.42, rel=5e-3)
    assert eddy.KAPPA_HALF_SINE < kappa[0.5] < kappa[1.0] < kappa[3.0] < eddy.KAPPA_STEP + 0.05
    # so skin_temperature_rise (kappa = 1) is the lower estimate
    assert res.surface_peak_temperature - T_ROOM > eddy.skin_temperature_rise(cu0, field)


def test_slab_copper_at_100_tesla_melts_whatever_the_pulse_length(cu):
    rho_c = cu.density * cu.specific_heat
    p_mag = 100.0 ** 2 / (2.0 * MU0)
    results = {}
    for period in (1e-4, 1e-3):
        delta = eddy.skin_depth(cu.resistivity, 1.0 / (2.0 * period))
        results[period] = eddy.slab_field_diffusion(cu, eddy.half_sine_field(100.0, period), 10.0 * delta, period)
    short, long_ = results[1e-4], results[1e-3]
    assert short.melted and long_.melted
    assert short.surface_peak_temperature == pytest.approx(2200.0, rel=1e-2)     # by the end of the pulse
    # surface heating is independent of the pulse length; the molten depth grows as sqrt(time)
    assert long_.surface_peak_temperature == pytest.approx(short.surface_peak_temperature, rel=1e-3)
    assert short.melt_depth == pytest.approx(0.72e-3, rel=2e-2)
    assert long_.melt_depth / short.melt_depth == pytest.approx(math.sqrt(10.0), rel=1e-2)
    assert 0.0 < short.melt_depth < short.penetration_depth < short.thickness
    # the rising resistivity of the hot skin lowers the effective kappa from 2.19 to ~1.65 at the end of
    # the pulse and from 2.43 to ~1.8 after it (surface_heating_energy_density docstring)
    assert (short.surface_peak_temperature - T_ROOM) * rho_c / p_mag == pytest.approx(1.65, rel=1e-2)
    delta = eddy.skin_depth(cu.resistivity, 1.0 / 2e-4)
    after = eddy.slab_field_diffusion(cu, eddy.half_sine_field(100.0, 1e-4), 10.0 * delta, 3e-4)
    assert (after.surface_peak_temperature - T_ROOM) * rho_c / p_mag == pytest.approx(1.8, rel=3e-2)
    # at 63.41 T (1.6 GPa) the same solver leaves the copper surface solid, at 30 T barely warm
    mid = eddy.slab_field_diffusion(cu, eddy.half_sine_field(63.41, 1e-4), 10.0 * delta, 1e-4)
    low = eddy.slab_field_diffusion(cu, eddy.half_sine_field(30.0, 1e-4), 10.0 * delta, 1e-4)
    assert not mid.melted and mid.surface_peak_temperature == pytest.approx(1115.0, rel=1e-2)
    assert not low.melted and low.surface_peak_temperature == pytest.approx(500.0, rel=1e-2)


def test_slab_converges_with_nodes_and_steps(cu):
    period = 1e-4
    delta = eddy.skin_depth(cu.resistivity, 1.0 / (2.0 * period))
    drive = eddy.half_sine_field(100.0, period)
    coarse = eddy.slab_field_diffusion(cu, drive, 10.0 * delta, 0.5 * period, n_nodes=30, n_steps=500)
    default = eddy.slab_field_diffusion(cu, drive, 10.0 * delta, 0.5 * period)
    fine = eddy.slab_field_diffusion(cu, drive, 10.0 * delta, 0.5 * period, n_nodes=120, n_steps=4000)
    assert (coarse.n_nodes, coarse.n_steps, fine.n_nodes, fine.n_steps) == (30, 500, 120, 4000)
    assert default.surface_peak_temperature == pytest.approx(fine.surface_peak_temperature, rel=1e-3)
    assert coarse.surface_peak_temperature == pytest.approx(fine.surface_peak_temperature, rel=1e-2)
    assert abs(default.surface_peak_temperature - fine.surface_peak_temperature) < abs(
        coarse.surface_peak_temperature - fine.surface_peak_temperature)
    assert default.joule_energy_per_area == pytest.approx(fine.joule_energy_per_area, rel=5e-3)
    assert default.penetration_depth == pytest.approx(fine.penetration_depth, rel=5e-3)
    assert default.melt_depth == pytest.approx(fine.melt_depth, rel=5e-3)


def test_slab_without_field_stays_cold(cu):
    res = eddy.slab_field_diffusion(cu, lambda t: 0.0, 1e-3, 1e-4, n_nodes=10, n_steps=10)
    assert res.surface_peak_temperature == res.max_temperature == T_ROOM
    assert res.melted is False and res.melt_depth == 0.0 and res.penetration_depth == 0.0
    assert res.joule_energy_per_area == 0.0 and set(res.final_temperature_profile) == {T_ROOM}
    warm = eddy.slab_field_diffusion(cu, lambda t: 0.0, 1e-3, 1e-4, n_nodes=3, n_steps=1, initial_temperature=400.0)
    assert warm.max_temperature == 400.0 and warm.z == pytest.approx((0.0, 0.25e-3, 0.75e-3))


# ---------------------------------------------------------------- drive waveforms
def test_damped_sine_field_is_normalised_to_its_first_peak():
    freq, tau = 1e4, 1e-4
    omega = 2.0 * math.pi * freq
    drive = eddy.damped_sine_field(50.0, freq, tau)
    t_peak = math.atan(omega * tau) / omega
    assert drive(t_peak) == pytest.approx(50.0, rel=1e-12)
    samples = [drive(i * 1e-8) for i in range(40001)]                       # four periods
    assert max(samples) == pytest.approx(50.0, rel=1e-6) and max(samples) <= 50.0 * (1.0 + 1e-12)
    assert samples.index(max(samples)) * 1e-8 == pytest.approx(t_peak, abs=2e-8)
    # successive extrema decay by exp(-pi/(omega tau))
    assert min(samples) / max(samples) == pytest.approx(-math.exp(-math.pi / (omega * tau)), rel=1e-6)
    assert drive(0.0) == 0.0 and drive(-1e-6) == 0.0
    # heavier damping moves the peak earlier but keeps its height
    heavy = eddy.damped_sine_field(50.0, freq, 1e-6)
    assert heavy(math.atan(omega * 1e-6) / omega) == pytest.approx(50.0, rel=1e-12)
    assert math.atan(omega * 1e-6) / omega < t_peak < 0.25 / freq
    # undamped limit
    ideal = eddy.damped_sine_field(50.0, freq)
    assert ideal(0.25 / freq) == pytest.approx(50.0, rel=1e-12)
    assert ideal(0.75 / freq) == pytest.approx(-50.0, rel=1e-12)
    assert ideal(-1e-6) == 0.0
    assert eddy.damped_sine_field(50.0, freq, 1e3)(0.25 / freq) == pytest.approx(50.0, rel=1e-6)
    assert eddy.damped_sine_field(0.0, freq, tau)(1e-5) == 0.0


def test_half_sine_field():
    drive = eddy.half_sine_field(50.0, 1e-4)
    assert drive(0.0) == 0.0 and drive(0.5e-4) == pytest.approx(50.0, rel=1e-12)
    assert drive(0.25e-4) == pytest.approx(50.0 * math.sin(math.pi / 4.0), rel=1e-12)
    assert drive(1e-4) == pytest.approx(0.0, abs=1e-12)
    assert drive(1.0001e-4) == 0.0 and drive(-1e-9) == 0.0 and drive(1.0) == 0.0
    assert all(drive(1e-6 * i) >= 0.0 for i in range(101))                  # unipolar


# ---------------------------------------------------------------- pulse-frequency recommendation
def test_pulse_frequency_recommendation_spec_check(cu):
    rec = eddy.pulse_frequency_recommendation(cu, 2e-3)
    assert rec.material == "copper_ofhc" and rec.wall == 2e-3 and rec.depth_ratio == 3.0
    assert rec.frequency == pytest.approx(9.7e3, rel=1e-2)                  # docstring: ~9.7 kHz
    assert rec.quarter_period == pytest.approx(26e-6, rel=2e-2)             # ~26 us
    assert rec.quarter_period == pytest.approx(0.25 / rec.frequency, rel=1e-12)
    assert rec.skin_depth == pytest.approx(2e-3 / 3.0, rel=1e-12)
    assert eddy.skin_depth(cu.resistivity, rec.frequency) == pytest.approx(rec.skin_depth, rel=1e-12)
    assert rec.diffusion_time == pytest.approx(eddy.diffusion_time(cu.resistivity, 2e-3), rel=1e-12)
    assert rec.regime == "thick" and "thick regime" in rec.note and "0.998" in rec.note
    # f ~ rho / h^2: colder copper needs a lower frequency, a thinner wall a higher one
    assert eddy.pulse_frequency_recommendation(cu, 2e-3, temperature=77.0).frequency < rec.frequency
    assert eddy.pulse_frequency_recommendation(cu, 1e-3).frequency == pytest.approx(4.0 * rec.frequency, rel=1e-12)
    # poor conductors need impractically fast pulses for the same shielding (hence driver sheets)
    assert eddy.pulse_frequency_recommendation("ti6al4v", 2e-3).frequency > 100.0 * rec.frequency
    data = json.loads(rec.json())
    assert data["regime"] == "thick" and data["frequency"] == pytest.approx(rec.frequency)
    assert "frequency [Hz]" in rec.text() and "quarter_period [s]" in rec.text()


def test_pulse_frequency_recommendation_thin_branch_and_boundary(cu):
    thin = eddy.pulse_frequency_recommendation(cu, 2e-3, depth_ratio=0.5)
    assert thin.regime == "thin" and thin.skin_depth == pytest.approx(4e-3)
    assert "thin regime" in thin.note and "0.632" in thin.note               # 1 - exp(-1) of the pressure
    assert thin.frequency == pytest.approx(eddy.pulse_frequency_recommendation(cu, 2e-3).frequency / 36.0, rel=1e-12)
    # h == delta is thin by definition (h <= delta), whatever the round-off of the frequency round trip
    edge = eddy.pulse_frequency_recommendation(cu, 2e-3, depth_ratio=1.0)
    assert edge.regime == "thin" and "thin regime" in edge.note and "0.865" in edge.note
    assert eddy.pulse_frequency_recommendation(cu, 2e-3, depth_ratio=1.01).regime == "thick"
    for ratio in (0.3, 0.99, 1.0, 1.01, 3.0, 10.0):
        rec = eddy.pulse_frequency_recommendation(cu, 2e-3, depth_ratio=ratio)
        assert rec.regime == ("thin" if ratio <= 1.0 else "thick")


# ---------------------------------------------------------------- tube compression: set-up
def test_tube_geometry_and_validation():
    tube = _tube()
    assert tube.inner_radius == pytest.approx(0.023) and tube.mean_radius == pytest.approx(R_MID)
    assert json.loads(tube.json())["material"] == "copper_ofhc"
    assert json.loads(_tube(materials.get("tungsten")).json())["material"] == "tungsten"
    base = dict(material="copper_ofhc", outer_radius=R_OUT, wall=WALL)
    bad = [dict(outer_radius=0.0), dict(outer_radius=float("nan")), dict(wall=0.0), dict(wall=0.03),
           dict(wall=R_OUT), dict(length=0.0), dict(mandrel_radius=0.023), dict(mandrel_radius=-1e-3),
           dict(temperature=0.0), dict(temperature=1400.0), dict(end_condition="plane_stress"),
           dict(driver_thickness=1e-3), dict(driver_material="copper_ofhc", driver_thickness=-1e-3),
           dict(taylor_quinney=1.5), dict(taylor_quinney=-0.1)]
    for kw in bad:
        with pytest.raises(InputError):
            eddy.TubeCompression(**{**base, **kw})
    with pytest.raises(InputError):   # the driver would already be molten
        eddy.TubeCompression("tungsten", R_OUT, WALL, temperature=1000.0, driver_material="aluminum_1100",
                             driver_thickness=1e-3)
    with pytest.raises(UnknownMaterialError):
        _tube("unobtainium")
    with pytest.raises(UnknownMaterialError):
        _tube(driver_material="unobtainium")
    # valid edge cases: a solid mandrel of zero radius, a driver material without thickness
    assert _tube(mandrel_radius=0.0).mandrel_radius == 0.0
    assert _tube(driver_material="aluminum_1100").driver_thickness == 0.0


def test_tube_simulate_validation():
    tube = _tube()
    drive = _pulse(12.0)
    for args, kw in (
        (("not callable", 1e-4), {}),
        ((drive, 0.0), {}),
        ((drive, -1e-4), {}),
        ((drive, 1e-4), dict(dt=0.0)),
        ((drive, 1e-4), dict(dt=2e-4)),                 # dt > t_end
        ((drive, 1e-4), dict(dt=1e-11)),                # 1e7 steps > max_steps
        ((drive, 1e-4), dict(max_steps=100)),           # default 20000 steps > 100
        ((drive, 1e-4), dict(max_steps=0)),
        ((drive, 1e-4), dict(flow_stress_fn=5.0)),
        ((drive, 1e-4), dict(flow_stress_fn=lambda e, r, t: -1.0)),
        ((drive, 1e-4), dict(flow_stress_fn=lambda e, r, t: float("nan"))),
        ((lambda t: float("nan"), 1e-4), {}),
        ((lambda t: float("inf"), 1e-4), {}),
    ):
        with pytest.raises(InputError):
            tube.simulate(*args, **kw)
    with pytest.raises(InputError, match="max_steps"):
        tube.simulate(drive, 1e-4, dt=1e-11)
    # the guard is on the step count, not on dt itself
    assert tube.simulate(drive, 4.0 * HALF, dt=4.0 * HALF / 500, max_steps=500).summary.final_strain > 0.0


# ---------------------------------------------------------------- tube compression: physics
def test_static_field_exerts_no_pressure_and_no_heating():
    """A steady field has diffused in: no eddy current, no force, no heat - at any field strength."""
    for field in (30.0, 100.0):
        res = _tube().simulate(lambda t: field, 2e-4, dt=2e-4 / 2000)
        s, h = res.summary, res.history
        assert s.final_strain == 0.0 and s.final_radius == R_MID and s.final_wall == WALL
        assert s.peak_velocity == 0.0 and s.peak_pressure == 0.0 and s.transfer_efficiency == 0.0
        assert s.plastic_work_per_length == 0.0 and s.final_temperature == T_ROOM
        assert s.bulk_joule_temperature_rise == 0.0 and s.plastic_temperature_rise == 0.0
        assert s.peak_field == field and set(h.b_in) == {field} and set(h.p_net) == {0.0}
        assert s.stopped_by == "t_end" and s.time_to_stop == 0.0
        assert any("no plastic deformation" in w for w in s.warnings)
        # no field change, no skin current: zero skin heating even at 100 T, where a pulse would melt copper
        assert s.skin_temperature_rise_estimate == 0.0 and s.regime == "thin"
        assert any("static drive" in w for w in s.warnings)
        assert not any("melting point" in w for w in s.warnings)


def test_no_motion_below_the_yield_pressure(cu):
    sigma0 = plasticity.flow_stress(cu, 0.0, eddy.STRAIN_RATE_FLOOR, T_ROOM)
    p_yield = 2.0 / math.sqrt(3.0) * sigma0 * WALL / R_MID                  # k sigma h / r
    b_yield = math.sqrt(2.0 * MU0 * p_yield)
    assert b_yield == pytest.approx(4.24, rel=1e-2)
    below = _tube().simulate(_pulse(0.97 * b_yield), 4.0 * HALF).summary
    assert below.final_strain == 0.0 and below.peak_velocity == 0.0 and below.plastic_work_per_length == 0.0
    assert below.stopped_by == "arrested" and below.time_to_stop == 0.0
    assert any("never exceeded the yield pressure" in w for w in below.warnings)
    assert below.peak_pressure == pytest.approx(0.97 ** 2 * p_yield, rel=1e-3)
    assert below.bulk_joule_temperature_rise > 0.0                          # eddy currents still heat the wall
    above = _tube().simulate(_pulse(1.5 * b_yield), 4.0 * HALF).summary
    assert above.final_strain > 0.0 and above.peak_velocity > 0.0
    assert not any("never exceeded" in w for w in above.warnings)


def test_strain_grows_monotonically_with_the_peak_field():
    fields = (5.0, 8.0, 12.0, 16.0, 20.0, 25.0)
    runs = [_tube().simulate(_pulse(b), 4.0 * HALF).summary for b in fields]
    for attr in ("final_strain", "peak_velocity", "peak_strain_rate", "peak_pressure", "plastic_work_per_length",
                 "final_temperature", "final_wall"):
        values = [getattr(s, attr) for s in runs]
        assert all(b > a for a, b in zip(values, values[1:])), attr
    radii = [s.final_radius for s in runs]
    assert all(b < a for a, b in zip(radii, radii[1:]))
    assert all(s.stopped_by == "arrested" for s in runs)
    for b, s in zip(fields, runs):
        assert s.peak_field == pytest.approx(b, rel=1e-6)
        assert s.peak_pressure == pytest.approx(magnetics.magnetic_pressure(b), rel=1e-3)   # well shielded
        assert 0.0 < s.transfer_efficiency <= 1.0


def test_tube_reference_cases_copper_50mm():
    """Pinned outcomes for the 50 mm OD x 2 mm copper tube (half-sine drive, 51 us long)."""
    assert FREQ == pytest.approx(9746.0, rel=1e-3) and HALF == pytest.approx(51.3e-6, rel=1e-3)
    s12 = _tube().simulate(_pulse(12.0), 4.0 * HALF).summary
    assert s12.final_strain == pytest.approx(0.0781, rel=1e-2) and s12.peak_velocity == pytest.approx(44.3, rel=1e-2)
    s20 = _tube().simulate(_pulse(20.0), 4.0 * HALF).summary
    assert s20.final_strain == pytest.approx(0.530, rel=1e-2)
    assert s20.peak_velocity == pytest.approx(166.0, rel=1e-2)
    assert s20.peak_strain_rate == pytest.approx(9.28e3, rel=1e-2)
    assert s20.peak_pressure == pytest.approx(0.159e9, rel=1e-2)
    assert s20.bulk_joule_temperature_rise == pytest.approx(7.45, rel=2e-2)
    assert s20.plastic_temperature_rise == pytest.approx(45.1, rel=2e-2)
    assert s20.skin_temperature_rise_estimate == pytest.approx(46.1, rel=1e-2)
    assert s20.regime == "thick" and s20.stopped_by == "arrested"
    assert s20.time_to_stop == pytest.approx(105e-6, rel=2e-2)
    # 30 T (0.36 GPa, industrial EMF) already collapses the tube completely
    s30 = _tube().simulate(_pulse(30.0), 4.0 * HALF).summary
    assert s30.stopped_by == "closed" and s30.peak_velocity == pytest.approx(391.5, rel=1e-2)
    assert s30.peak_strain_rate == pytest.approx(4.17e4, rel=2e-2)
    # the README's window: 1.6 GPa and 4 GPa close it in 35 us and 26 us at 1.3 and 1.9 km/s
    s63 = _tube().simulate(_pulse(63.41), 4.0 * HALF).summary
    s100 = _tube().simulate(_pulse(100.27), 4.0 * HALF).summary
    assert s63.stopped_by == s100.stopped_by == "closed"
    assert s63.peak_pressure == pytest.approx(1.6e9, rel=1e-2) and s100.peak_pressure == pytest.approx(4.0e9, rel=2e-2)
    assert s63.peak_velocity == pytest.approx(1260.0, rel=1e-2)
    assert s100.peak_velocity == pytest.approx(1942.0, rel=1e-2)
    assert s63.peak_strain_rate == pytest.approx(2.97e5, rel=2e-2)
    assert s100.peak_strain_rate == pytest.approx(4.56e5, rel=2e-2)
    assert s63.time_to_stop == pytest.approx(35.4e-6, rel=1e-2)
    assert s100.time_to_stop == pytest.approx(26.3e-6, rel=1e-2)
    assert s63.skin_temperature_rise_estimate == pytest.approx(464.0, rel=1e-2)
    assert s100.skin_temperature_rise_estimate == pytest.approx(1160.0, rel=1e-2)
    assert not any("melting point" in w for w in s63.warnings)
    assert any("outer skin of copper_ofhc reaches its melting point" in w for w in s100.warnings)
    # every closed plane-strain tube ends at the same solid radius sqrt(h0 r0/2) and strain (2/sqrt3) ln(r0/r)
    r_closed = math.sqrt(WALL * R_MID / 2.0)
    for s in (s30, s63, s100):
        assert s.final_radius == pytest.approx(r_closed, rel=1e-9)
        assert s.final_inner_radius == pytest.approx(0.0, abs=1e-12)
        assert s.final_strain == pytest.approx(2.0 / math.sqrt(3.0) * math.log(R_MID / r_closed), rel=1e-9)
        assert s.final_strain == pytest.approx(1.835, rel=1e-3)
        assert any("bore closed completely" in w for w in s.warnings)


def test_mandrel_stops_the_bore():
    res = _tube(mandrel_radius=0.015).simulate(_pulse(30.0), 4.0 * HALF)
    s = res.summary
    assert s.stopped_by == "mandrel" and any("reached the mandrel" in w for w in s.warnings)
    assert s.final_inner_radius == pytest.approx(0.015, abs=1e-12)
    assert s.final_radius * s.final_wall == pytest.approx(R_MID * WALL, rel=1e-12)     # plane strain: area conserved
    assert s.final_strain == pytest.approx(0.4356, rel=1e-3)
    assert s.time_to_stop == pytest.approx(res.history.t[-1]) and s.time_to_stop == pytest.approx(43.1e-6, rel=1e-2)
    assert res.history.velocity[-1] == 0.0                                  # the wall is brought to rest
    # a mandrel the tube never reaches changes nothing
    free = _tube().simulate(_pulse(12.0), 4.0 * HALF).summary
    held = _tube(mandrel_radius=0.015).simulate(_pulse(12.0), 4.0 * HALF).summary
    assert held.stopped_by == "arrested" and held.final_strain == free.final_strain
    assert held.final_inner_radius > 0.015


def test_plane_strain_versus_free_ends():
    plane = _tube().simulate(_pulse(16.0), 4.0 * HALF).summary
    free = _tube(end_condition="free_ends").simulate(_pulse(16.0), 4.0 * HALF).summary
    # wall-thickness laws h = h0 r0/r (plane strain) and h0 sqrt(r0/r) (free ends)
    assert plane.final_wall == pytest.approx(WALL * R_MID / plane.final_radius, rel=1e-12)
    assert free.final_wall == pytest.approx(WALL * math.sqrt(R_MID / free.final_radius), rel=1e-12)
    # equivalent strain (2/sqrt3) ln(r0/r) versus ln(r0/r)
    assert plane.final_strain == pytest.approx(2.0 / math.sqrt(3.0) * math.log(R_MID / plane.final_radius), rel=1e-12)
    assert free.final_strain == pytest.approx(math.log(R_MID / free.final_radius), rel=1e-12)
    # the uniaxial (free-ends) tube yields at a lower pressure and moves further in
    assert free.final_radius < plane.final_radius and free.final_wall < plane.final_wall
    assert plane.final_strain == pytest.approx(0.239, rel=1e-2) and free.final_strain == pytest.approx(0.254, rel=1e-2)
    # closure of a free-ends tube: bore r - h0 sqrt(r0/r)/2 = 0  ->  r = (h0 sqrt(r0)/2)^(2/3)
    closed = _tube(end_condition="free_ends").simulate(_pulse(40.0), 4.0 * HALF).summary
    assert closed.stopped_by == "closed"
    assert closed.final_radius == pytest.approx((WALL * math.sqrt(R_MID) / 2.0) ** (2.0 / 3.0), rel=1e-9)
    assert closed.final_inner_radius == pytest.approx(0.0, abs=1e-12)
    stopped = _tube(end_condition="free_ends", mandrel_radius=0.012).simulate(_pulse(30.0), 4.0 * HALF).summary
    assert stopped.stopped_by == "mandrel" and stopped.final_inner_radius == pytest.approx(0.012, abs=1e-11)


def test_work_of_the_magnetic_pressure_equals_the_plastic_work():
    """Plane strain, tube at rest at the end: int p_net 2 pi r (-v) dt = int sigma d(eps) dV."""
    for field in (16.0, 20.0, 25.0):
        res = _tube().simulate(_pulse(field), 4.0 * HALF)
        s, h = res.summary, res.history
        assert s.stopped_by == "arrested" and h.velocity[-1] == 0.0
        power = [p * 2.0 * math.pi * r * (-v) for p, r, v in zip(h.p_net, h.radius, h.velocity)]
        assert trapz(power, h.t) == pytest.approx(s.plastic_work_per_length, rel=2e-3)
        assert s.final_temperature == pytest.approx(
            T_ROOM + s.bulk_joule_temperature_rise + s.plastic_temperature_rise, rel=1e-12)
        assert s.peak_velocity == pytest.approx(max(-v for v in h.velocity), rel=1e-4)
        assert s.peak_strain_rate == pytest.approx(max(h.strain_rate), rel=1e-4)
        assert s.peak_pressure == pytest.approx(max(h.p_net), rel=1e-4)
        assert s.transfer_efficiency == pytest.approx(
            s.peak_pressure / magnetics.magnetic_pressure(s.peak_field), rel=1e-12)
        assert s.skin_temperature_rise_estimate == pytest.approx(
            eddy.skin_temperature_rise("copper_ofhc", s.peak_field), rel=1e-12)


def test_custom_flow_stress_function(cu):
    sigma = 200e6
    res = _tube().simulate(_pulse(20.0), 4.0 * HALF, flow_stress_fn=lambda eps, rate, temp: sigma)
    s = res.summary
    volume = 2.0 * math.pi * R_MID * WALL                                  # per metre of tube
    assert s.plastic_work_per_length == pytest.approx(sigma * s.final_strain * volume, rel=1e-9)
    assert s.plastic_temperature_rise == pytest.approx(
        0.9 * sigma * s.final_strain / cu.volumetric_heat_capacity, rel=1e-9)
    assert set(res.history.flow_stress) == {sigma} and s.final_strain == pytest.approx(0.844, rel=1e-2)
    # no heat from plastic work with a zero Taylor-Quinney coefficient
    cold = _tube(taylor_quinney=0.0).simulate(_pulse(20.0), 4.0 * HALF, flow_stress_fn=lambda eps, rate, temp: sigma)
    assert cold.summary.plastic_temperature_rise == 0.0 and cold.summary.plastic_work_per_length > 0.0
    # the default is plasticity.flow_stress of the tube material, called with the rate floored at 1e-3 1/s
    calls = []

    def recorded(eps: float, rate: float, temp: float) -> float:
        calls.append((eps, rate, temp))
        return plasticity.flow_stress(cu, eps, rate, temp)

    explicit = _tube().simulate(_pulse(20.0), 4.0 * HALF, flow_stress_fn=recorded)
    assert explicit.json() == _tube().simulate(_pulse(20.0), 4.0 * HALF).json()
    assert calls[0] == (0.0, eddy.STRAIN_RATE_FLOOR, T_ROOM)
    assert min(c[1] for c in calls) == eddy.STRAIN_RATE_FLOOR and max(c[1] for c in calls) > 5e3
    assert all(c[0] >= 0.0 and c[2] >= T_ROOM for c in calls)
    # a stronger material deforms less
    strong = _tube().simulate(_pulse(20.0), 4.0 * HALF, flow_stress_fn=lambda eps, rate, temp: 2.0 * sigma)
    assert strong.summary.final_strain < s.final_strain


def test_flux_compression_conserves_the_trapped_flux(cu):
    """Perfectly conducting shell with a seed field inside: B_in r^2 = const (Fowler et al. 1960)."""
    shell = cu.with_changes(key="copper_perfect", resistivity=1e-14)
    period = 1e-4

    def drive(t: float) -> float:
        return 5.0 + (25.0 * math.sin(math.pi * t / period) if 0.0 <= t <= period else 0.0)

    res = _tube(shell).simulate(drive, 2.0 * period)
    s, h = res.summary, res.history
    flux = [b * r * r for b, r in zip(h.b_in, h.radius)]
    assert max(flux) / flux[0] - 1.0 < 1e-6 and min(flux) / flux[0] - 1.0 > -1e-6
    assert h.b_in[0] == 5.0 and h.b_in[-1] == pytest.approx(5.0 * (R_MID / s.final_radius) ** 2, rel=1e-6)
    assert max(h.b_in) > 60.0 > max(h.b_out)                                # seed amplified more than tenfold
    # the compressed flux stops the wall before the bore closes, then pushes outward
    assert s.stopped_by == "arrested" and 0.0 < s.final_inner_radius < 0.002
    assert min(h.p_net) < 0.0 and any("trapped flux pushes outward" in w for w in s.warnings)
    assert s.bulk_joule_temperature_rise == pytest.approx(0.0, abs=1e-3)    # no resistance, no Joule heat
    assert s.peak_pressure < magnetics.magnetic_pressure(30.0)
    assert s.transfer_efficiency == pytest.approx(s.peak_pressure / magnetics.magnetic_pressure(30.0), rel=1e-9)


def test_dc_background_biases_a_pulse_but_is_flux_compressed():
    """Hybrid drive: 30 T steady + 33.41 T pulse reaches 63.41 T (1.6 GPa on an empty bore)."""
    def biased(t: float) -> float:
        return 30.0 + (33.41 * math.sin(math.pi * t / HALF) if 0.0 <= t <= HALF else 0.0)

    res = _tube().simulate(biased, 4.0 * HALF)
    s, h = res.summary, res.history
    assert s.peak_field == pytest.approx(63.41, rel=1e-6) and h.b_in[0] == 30.0 and h.p_net[0] == 0.0
    # the net pressure can never exceed the static-shell value ((B_dc + dB)^2 - B_dc^2)/(2 mu0) = 1.24 GPa ...
    ceiling = magnetics.pressure_with_background(30.0, 33.41)
    assert ceiling == pytest.approx(1.242e9, rel=1e-3)
    assert s.peak_pressure < ceiling < magnetics.magnetic_pressure(63.41)
    # ... and the collapsing tube compresses the trapped background, which cuts it further
    assert s.peak_pressure == pytest.approx(0.83e9, rel=2e-2)
    assert max(h.b_in) == pytest.approx(30.0 * (R_MID / s.final_radius) ** 2, rel=5e-2) and max(h.b_in) > 100.0
    assert s.transfer_efficiency == pytest.approx(0.52, rel=2e-2)
    unbiased = _tube().simulate(_pulse(63.41), 4.0 * HALF).summary
    assert s.final_strain < unbiased.final_strain and s.final_inner_radius > 0.0
    assert any("trapped flux pushes outward" in w for w in s.warnings)
    # skin heating follows the 33.41 T swing, not the 63.41 T total field
    assert s.skin_temperature_rise_estimate == pytest.approx(eddy.skin_temperature_rise("copper_ofhc", 33.41), rel=1e-6)
    assert s.skin_temperature_rise_estimate == pytest.approx(128.7, rel=1e-2)
    assert s.skin_temperature_rise_estimate < 0.3 * unbiased.skin_temperature_rise_estimate
    assert s.regime == "thick"


def test_driver_sheet_shields_a_poor_conductor():
    bare = _tube("ti6al4v").simulate(_pulse(30.0), 6.0 * HALF).summary
    driven = _tube("ti6al4v", driver_material="copper_ofhc", driver_thickness=1e-3).simulate(
        _pulse(30.0), 6.0 * HALF).summary
    # bare Ti-6Al-4V: the field leaks through the 2 mm wall (thin regime) and heats it by hundreds of kelvin
    assert bare.regime == "thin" and any("thin wall (h <= delta)" in w for w in bare.warnings)
    assert bare.transfer_efficiency == pytest.approx(0.674, rel=2e-2)
    assert bare.bulk_joule_temperature_rise == pytest.approx(547.0, rel=2e-2)
    assert any("trapped flux pushes outward" in w for w in bare.warnings)
    # 1 mm of copper carries the current instead
    assert driven.regime == "thick" and driven.transfer_efficiency > 0.999
    assert driven.bulk_joule_temperature_rise < 0.01 * bare.bulk_joule_temperature_rise
    assert driven.peak_pressure > 1.4 * bare.peak_pressure and driven.final_strain > bare.final_strain
    assert any("refers to the driver sheet (copper_ofhc)" in w for w in driven.warnings)
    assert driven.skin_temperature_rise_estimate == pytest.approx(
        eddy.skin_temperature_rise("copper_ofhc", 30.0), rel=1e-6)
    assert bare.skin_temperature_rise_estimate == pytest.approx(
        eddy.skin_temperature_rise("ti6al4v", 30.0), rel=1e-6)
    assert not any("driver sheet" in w for w in bare.warnings)


def test_driver_sheet_adds_mass():
    """Strength switched off: the early velocity scales with 1/(rho h + rho_d h_d)."""
    def no_strength(eps: float, rate: float, temp: float) -> float:
        return 0.0

    drive = _pulse(20.0)
    bare = _tube().simulate(drive, 4.0 * HALF, flow_stress_fn=no_strength)
    twin = _tube(driver_material="copper_ofhc", driver_thickness=WALL).simulate(
        drive, 4.0 * HALF, flow_stress_fn=no_strength)
    light = _tube(driver_material="aluminum_1100", driver_thickness=WALL).simulate(
        drive, 4.0 * HALF, flow_stress_fn=no_strength)
    rho_cu, rho_al = materials.get("copper_ofhc").density, materials.get("aluminum_1100").density
    for i in (20, 50, 100):                                                 # 2-11 us, displacement < 0.2 %
        assert bare.history.t[i] == twin.history.t[i] == light.history.t[i]
        assert bare.history.velocity[i] < 0.0
        assert twin.history.velocity[i] / bare.history.velocity[i] == pytest.approx(0.5, rel=2e-3)
        assert light.history.velocity[i] / bare.history.velocity[i] == pytest.approx(
            rho_cu / (rho_cu + rho_al), rel=2e-3)
    assert twin.summary.time_to_stop > bare.summary.time_to_stop           # the heavier stack closes later
    assert bare.summary.plastic_work_per_length == 0.0 and bare.summary.plastic_temperature_rise == 0.0


def test_bulk_melting_stops_the_run():
    ti = materials.get("ti6al4v")
    s = _tube(ti).simulate(_pulse(63.41), 6.0 * HALF).summary
    assert s.stopped_by == "melted" and any("reached its melting point" in w for w in s.warnings)
    assert s.final_temperature == pytest.approx(ti.melting_point, rel=1e-3) and s.final_temperature >= ti.melting_point
    assert s.bulk_joule_temperature_rise > 1400.0 and s.time_to_stop < HALF
    assert s.regime == "thin" and s.final_inner_radius > 0.0
    # a thin aluminium driver in front of steel melts first (lumped bulk) and is reported, without stopping the run
    al = _tube("steel_304", driver_material="aluminum_1100", driver_thickness=2e-4).simulate(
        _pulse(63.41), 4.0 * HALF).summary
    assert any("driver sheet (aluminum_1100) reaches its melting point" in w for w in al.warnings)
    assert any("outer skin of aluminum_1100 reaches its melting point" in w for w in al.warnings)
    assert al.stopped_by != "melted"


def test_tube_warnings_short_tube_and_unfinished_run():
    short = _tube(length=0.02).simulate(_pulse(12.0), 4.0 * HALF).summary
    assert any("short tube" in w and "2.08x" in w for w in short.warnings)   # (l + 0.9 r)/l
    long_ = _tube(length=1.0).simulate(_pulse(12.0), 4.0 * HALF).summary
    assert not any("short tube" in w for w in long_.warnings)
    assert short.final_strain == long_.final_strain                         # the length only triggers the warning
    assert any("thick wall (sum h/delta = 3 at" in w for w in long_.warnings)
    # cutting the run at the field peak leaves the wall moving
    cut = _tube().simulate(_pulse(20.0), 0.5 * HALF, dt=0.5 * HALF / 2000)
    assert cut.summary.stopped_by == "t_end" and math.isinf(cut.summary.time_to_stop)
    assert any("still moving at t_end" in w for w in cut.summary.warnings)
    assert cut.history.velocity[-1] < 0.0
    assert cut.summary.final_strain < _tube().simulate(_pulse(20.0), 4.0 * HALF).summary.final_strain
    assert json.loads(cut.json())["summary"]["time_to_stop"] is None        # inf -> null keeps the JSON strict


def test_cold_start_and_damped_drive():
    # 77 K copper conducts ~6.6x better: more shielding (h/delta = 7.6), less Joule heat, same mechanics
    warm = _tube().simulate(_pulse(20.0), 4.0 * HALF).summary
    cold = _tube(temperature=77.0).simulate(_pulse(20.0), 4.0 * HALF).summary
    assert cold.bulk_joule_temperature_rise < 0.25 * warm.bulk_joule_temperature_rise
    assert cold.transfer_efficiency >= warm.transfer_efficiency
    assert any("sum h/delta = 7.6" in w for w in cold.warnings)
    assert cold.final_temperature == pytest.approx(
        77.0 + cold.bulk_joule_temperature_rise + cold.plastic_temperature_rise)
    # a ringing (damped-sine) discharge: the first half cycle does the work, later ones add heat only
    ring = _tube().simulate(eddy.damped_sine_field(14.0, FREQ, 2e-4), 6e-4)
    assert ring.summary.stopped_by == "arrested" and ring.summary.peak_field == pytest.approx(14.0, rel=1e-4)
    assert ring.summary.final_strain == pytest.approx(0.218, rel=2e-2)
    assert ring.summary.time_to_stop < 6e-4 and ring.history.t[-1] < 6e-4   # stopped once the drive cannot yield it


# ---------------------------------------------------------------- tube compression: output records
def test_tube_history_is_bounded_and_monotone():
    res = _tube().simulate(_pulse(20.0), 4.0 * HALF)
    h = res.history
    columns = (h.t, h.radius, h.velocity, h.strain, h.strain_rate, h.p_net, h.b_out, h.b_in, h.temperature,
               h.flow_stress)
    assert len({len(c) for c in columns}) == 1 and 2 <= len(h.t) <= eddy.HISTORY_MAX_POINTS
    assert h.t[0] == 0.0 and all(b > a for a, b in zip(h.t, h.t[1:]))
    assert h.radius[0] == R_MID and all(b <= a for a, b in zip(h.radius, h.radius[1:]))   # no elastic rebound
    assert max(h.velocity) == 0.0 and h.velocity[0] == 0.0
    assert h.strain[0] == 0.0 and all(b >= a for a, b in zip(h.strain, h.strain[1:]))
    assert h.temperature[0] == T_ROOM and all(b >= a for a, b in zip(h.temperature, h.temperature[1:]))
    assert h.strain[-1] == pytest.approx(res.summary.final_strain) and h.radius[-1] == res.summary.final_radius
    assert h.temperature[-1] == pytest.approx(res.summary.final_temperature)
    assert all(p == pytest.approx((bo * bo - bi * bi) / (2.0 * MU0), rel=1e-9, abs=1.0)
               for p, bo, bi in zip(h.p_net, h.b_out, h.b_in))
    assert all(s >= 0.0 for s in h.strain_rate) and all(s > 0.0 for s in h.flow_stress)
    # a run that lasts to t_end at the default 20000 steps is thinned to <= 2000 samples
    full = _tube().simulate(lambda t: 30.0, 2e-4)
    assert 1500 < len(full.history.t) <= eddy.HISTORY_MAX_POINTS and full.history.t[-1] == pytest.approx(2e-4)
    # with few steps every step is kept
    few = _tube().simulate(_pulse(20.0), 0.5 * HALF, dt=0.5 * HALF / 100)
    assert len(few.history.t) == 101 and few.history.t[-1] == pytest.approx(0.5 * HALF)


def test_run_ends_once_the_tube_is_arrested():
    t_end = 4.0 * HALF
    dt = t_end / eddy.DEFAULT_TUBE_STEPS
    for field in (5.0, 8.0, 12.0, 20.0, 25.0):
        res = _tube().simulate(_pulse(field), t_end)
        s = res.summary
        assert s.stopped_by == "arrested" and 0.0 < s.time_to_stop < res.history.t[-1] < t_end
        # declared at rest ARREST_STEPS consecutive steps after the wall last moved (the stopping step counts)
        waited = (res.history.t[-1] - s.time_to_stop) / dt
        assert eddy.ARREST_STEPS - 1.5 < waited < eddy.ARREST_STEPS + 0.5
    # at 20 T the wall coasts on its inertia for another half period after the 51 us pulse has ended
    assert _tube().simulate(_pulse(20.0), t_end).summary.time_to_stop == pytest.approx(2.05 * HALF, rel=1e-2)


def test_tube_step_size_convergence():
    coarse = _tube().simulate(_pulse(20.0), 4.0 * HALF, dt=4.0 * HALF / 1000).summary
    default = _tube().simulate(_pulse(20.0), 4.0 * HALF).summary
    fine = _tube().simulate(_pulse(20.0), 4.0 * HALF, dt=4.0 * HALF / 80_000).summary
    for attr in ("final_strain", "peak_velocity", "final_temperature", "plastic_work_per_length"):
        assert getattr(default, attr) == pytest.approx(getattr(fine, attr), rel=1e-3), attr
        assert getattr(coarse, attr) == pytest.approx(getattr(fine, attr), rel=2e-2), attr


def test_tube_results_round_trip_through_json_and_render_as_text():
    res = _tube().simulate(_pulse(12.0), 4.0 * HALF)
    data = json.loads(res.json())
    assert set(data) == {"material", "summary", "history"} and data["material"] == "copper_ofhc"
    assert data["summary"]["stopped_by"] == "arrested" and data["summary"]["regime"] == "thick"
    assert data["summary"]["final_strain"] == pytest.approx(res.summary.final_strain)
    assert isinstance(data["summary"]["warnings"], list) and data["summary"]["warnings"]
    assert len(data["history"]["t"]) == len(res.history.t) and data["history"]["b_in"][0] == 0.0
    assert res.as_dict()["summary"]["peak_field"] == pytest.approx(12.0, rel=1e-6)
    text = res.text()
    assert text.startswith("TubeResult") and "material = copper_ofhc" in text
    for fragment in ("summary:", "history:", "final_radius [m] =", "peak_velocity [m/s] =", "peak_strain_rate [1/s] =",
                     "skin_temperature_rise_estimate [K] =", "stopped_by = arrested", "warnings:", "- thick wall"):
        assert fragment in text, fragment
    assert f"t [s] = [{len(res.history.t)} values]" in text                # long series are summarised
    with pytest.raises(Exception):
        res.summary.final_strain = 0.0   # frozen


def test_other_result_types_serialise():
    joule = eddy.adiabatic_joule_heating([5e9, 5e9], [0.0, 1e-3], "copper_ofhc")
    assert isinstance(joule, eddy.JouleResult)
    data = json.loads(joule.json())
    assert data["material"] == "copper_ofhc" and data["melted"] is False and data["time_to_melt"] is None
    assert "action [A^2 s/m^4]" in joule.text() and "delta_t [K]" in joule.text()
    slab = eddy.slab_field_diffusion("copper_ofhc", eddy.half_sine_field(30.0, 1e-4), 5e-3, 1e-4, n_nodes=20,
                                     n_steps=200)
    assert isinstance(slab, eddy.DiffusionResult)
    data = json.loads(slab.json())
    assert data["n_nodes"] == 20 and len(data["z"]) == len(data["final_temperature_profile"]) == 20
    assert "surface_peak_temperature [K]" in slab.text() and "joule_energy_per_area [J/m^2]" in slab.text()
    assert isinstance(eddy.pulse_frequency_recommendation("copper_ofhc", 2e-3), eddy.PulseRecommendation)
    tube = _tube().simulate(_pulse(12.0), 4.0 * HALF, dt=4.0 * HALF / 2000)
    assert isinstance(tube, eddy.TubeResult)
    assert isinstance(tube.summary, eddy.TubeSummary) and isinstance(tube.history, eddy.TubeHistory)


# ---------------------------------------------------------------- invalid inputs
@pytest.mark.parametrize("call", [
    lambda: eddy.skin_depth(0.0, 1e4),
    lambda: eddy.skin_depth(1.7e-8, 0.0),
    lambda: eddy.skin_depth(1.7e-8, 1e4, 0.0),
    lambda: eddy.skin_depth(float("nan"), 1e4),
    lambda: eddy.skin_depth("copper", 1e4),
    lambda: eddy.skin_depth_for_material("copper_ofhc", 1e4, 0.0),
    lambda: eddy.skin_depth_for_material("copper_ofhc", -1.0),
    lambda: eddy.frequency_for_skin_depth(1.7e-8, 0.0),
    lambda: eddy.frequency_for_skin_depth(-1.0, 1e-3),
    lambda: eddy.shell_time_constant(1.7e-8, 0.0, 1e-3),
    lambda: eddy.shell_time_constant(1.7e-8, 0.02, -1e-3),
    lambda: eddy.shell_time_constant(0.0, 0.02, 1e-3),
    lambda: eddy.diffusion_time(1.7e-8, 0.0),
    lambda: eddy.diffusion_time(0.0, 1e-3),
    lambda: eddy.shell_regime(1.7e-8, 0.0, 1e4),
    lambda: eddy.shell_regime(1.7e-8, 1e-3, 0.0),
    lambda: eddy.pressure_on_shell(float("inf"), 0.0),
    lambda: eddy.pressure_on_shell(10.0, float("nan")),
    lambda: eddy.sheet_shielding_factor(-1e-3, 1e-3),
    lambda: eddy.sheet_shielding_factor(1e-3, 0.0),
    lambda: eddy.transmitted_pressure(-1.0, 1e-3, 1e-3),
    lambda: eddy.effective_current_depth(0.0, 1e-3),
    lambda: eddy.effective_current_depth(1e-3, 0.0),
    lambda: eddy.surface_heating_energy_density(float("nan")),
    lambda: eddy.surface_heating_energy_density(100.0, 0.0),
    lambda: eddy.skin_temperature_rise("copper_ofhc", 100.0, -1.0),
    lambda: eddy.skin_melting_field("copper_ofhc", initial_temperature=1356.0),
    lambda: eddy.skin_melting_field("copper_ofhc", initial_temperature=0.0),
    lambda: eddy.skin_melting_field("copper_ofhc", kappa=0.0),
    lambda: eddy.skin_melting_field("hastelloy_c276", include_latent=True),   # no latent heat recorded
    lambda: eddy.interior_field(5.0, 1e-3, 1e-3, 10),
    lambda: eddy.interior_field(lambda t: t, 0.0, 1e-3, 10),
    lambda: eddy.interior_field(lambda t: t, 1e-3, 0.0, 10),
    lambda: eddy.interior_field(lambda t: t, 1e-3, 1e-3, 0),
    lambda: eddy.interior_field(lambda t: t, 1e-3, 1e-3, 2.5),
    lambda: eddy.interior_field(lambda t: t, 1e-3, 1e-3, True),
    lambda: eddy.interior_field(lambda t: float("nan"), 1e-3, 1e-3, 10),
    lambda: eddy.action_integral([1.0, 2.0], [0.0]),
    lambda: eddy.action_integral([], []),
    lambda: eddy.action_integral([1.0, 1.0], [1.0, 0.0]),
    lambda: eddy.action_integral([1.0, float("nan")], [0.0, 1.0]),
    lambda: eddy.adiabatic_joule_heating([1e9, 1e9], [0.0, 1.0, 2.0], "copper_ofhc"),
    lambda: eddy.adiabatic_joule_heating([1e9, 1e9], [1.0, 0.0], "copper_ofhc"),
    lambda: eddy.adiabatic_joule_heating([1e9, 1e9], [0.0, 1.0], "copper_ofhc", initial_temperature=2000.0),
    lambda: eddy.adiabatic_joule_heating([1e9, 1e9], [0.0, 1.0], "copper_ofhc", initial_temperature=0.0),
    lambda: eddy.specific_action_to_temperature("copper_ofhc", 500.0, 400.0),
    lambda: eddy.specific_action_to_temperature("copper_ofhc", 0.0, 400.0),
    lambda: eddy.specific_action_to_temperature("copper_ofhc", 300.0, 400.0, n=0),
    lambda: eddy.slab_field_diffusion("copper_ofhc", 10.0, 1e-3, 1e-4),
    lambda: eddy.slab_field_diffusion("copper_ofhc", lambda t: 10.0, 0.0, 1e-4),
    lambda: eddy.slab_field_diffusion("copper_ofhc", lambda t: 10.0, 1e-3, 0.0),
    lambda: eddy.slab_field_diffusion("copper_ofhc", lambda t: 10.0, 1e-3, 1e-4, n_nodes=2),
    lambda: eddy.slab_field_diffusion("copper_ofhc", lambda t: 10.0, 1e-3, 1e-4, n_steps=0),
    lambda: eddy.slab_field_diffusion("copper_ofhc", lambda t: 10.0, 1e-3, 1e-4, n_nodes=10.0),
    lambda: eddy.slab_field_diffusion("copper_ofhc", lambda t: 10.0, 1e-3, 1e-4, initial_temperature=1356.0),
    lambda: eddy.slab_field_diffusion("copper_ofhc", lambda t: float("inf"), 1e-3, 1e-4, n_nodes=5, n_steps=5),
    lambda: eddy.damped_sine_field(-1.0, 1e4),
    lambda: eddy.damped_sine_field(50.0, 0.0),
    lambda: eddy.damped_sine_field(50.0, 1e4, 0.0),
    lambda: eddy.damped_sine_field(50.0, 1e4, -1e-4),
    lambda: eddy.half_sine_field(-1.0, 1e-4),
    lambda: eddy.half_sine_field(50.0, 0.0),
    lambda: eddy.pulse_frequency_recommendation("copper_ofhc", 0.0),
    lambda: eddy.pulse_frequency_recommendation("copper_ofhc", 2e-3, depth_ratio=0.0),
    lambda: eddy.pulse_frequency_recommendation("copper_ofhc", 2e-3, temperature=0.0),
])
def test_invalid_inputs_raise_input_error(call):
    with pytest.raises(InputError):
        call()


def test_unknown_materials_raise():
    for call in (
        lambda: eddy.skin_depth_for_material("unobtainium", 1e4),
        lambda: eddy.skin_temperature_rise("unobtainium", 50.0),
        lambda: eddy.skin_melting_field("unobtainium"),
        lambda: eddy.adiabatic_joule_heating([1e9, 1e9], [0.0, 1.0], "unobtainium"),
        lambda: eddy.specific_action_to_temperature("unobtainium", 300.0, 400.0),
        lambda: eddy.slab_field_diffusion("unobtainium", lambda t: 10.0, 1e-3, 1e-4),
        lambda: eddy.pulse_frequency_recommendation("unobtainium", 2e-3),
    ):
        with pytest.raises(UnknownMaterialError):
            call()
