import math

import pytest

from magform import constants
from magform._numerics import (
    argmax, bisect, brent, clamp, cumtrapz, find_bracket, interp1, linspace,
    logspace, newton, rk4, solve_quadratic, trapz,
)


def test_constants_codata():
    assert constants.MU0 == pytest.approx(4e-7 * math.pi, rel=1e-9)
    assert constants.kelvin_from_celsius(20.0) == pytest.approx(293.15)
    assert constants.gpa(4.0) == 4e9 and constants.to_mpa(1.6e9) == 1600.0
    assert constants.from_a_per_mm2(constants.a_per_mm2(3.98e8)) == pytest.approx(3.98e8)


def test_bisect_and_brent_agree_on_cosine_root():
    r1 = bisect(math.cos, 1.0, 2.0)
    r2 = brent(math.cos, 1.0, 2.0)
    assert r1 == pytest.approx(math.pi / 2, abs=1e-9)
    assert r2 == pytest.approx(math.pi / 2, abs=1e-12)


def test_brent_rejects_unbracketed():
    with pytest.raises(ValueError):
        brent(lambda x: x * x + 1.0, 0.0, 1.0)


def test_newton_with_and_without_derivative():
    f = lambda x: x * x - 2.0
    assert newton(f, 1.0, lambda x: 2 * x) == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert newton(f, 1.0) == pytest.approx(math.sqrt(2.0), rel=1e-9)


def test_find_bracket_then_root():
    f = lambda x: math.exp(x) - 50.0
    a, b = find_bracket(f, 0.0, 1.0)
    assert f(a) * f(b) <= 0
    assert brent(f, a, b) == pytest.approx(math.log(50.0), rel=1e-10)


def test_solve_quadratic_cases():
    assert solve_quadratic(1, -3, 2) == pytest.approx([1.0, 2.0])
    assert solve_quadratic(1, 0, 1) == []
    assert solve_quadratic(0, 2, -4) == pytest.approx([2.0])
    # ill-conditioned case: stable formula keeps the small root accurate
    roots = solve_quadratic(1.0, -1e8, 1.0)
    assert roots[0] == pytest.approx(1e-8, rel=1e-6)


def test_rk4_exponential_decay_and_stop():
    sol = rk4(lambda t, y: [-y[0]], [1.0], 0.0, 2.0, 200)
    t, y = sol[-1]
    assert t == pytest.approx(2.0)
    assert y[0] == pytest.approx(math.exp(-2.0), rel=1e-7)
    sol2 = rk4(lambda t, y: [-y[0]], [1.0], 0.0, 10.0, 1000, stop=lambda t, y: y[0] < 0.5)
    assert sol2[-1][0] == pytest.approx(math.log(2.0), abs=0.02)


def test_grids_quadrature_interp():
    xs = linspace(0.0, 1.0, 101)
    assert len(xs) == 101 and xs[0] == 0.0 and xs[-1] == 1.0
    assert logspace(0, 2, 3) == pytest.approx([1.0, 10.0, 100.0])
    ys = [x * x for x in xs]
    assert trapz(ys, xs) == pytest.approx(1.0 / 3.0, rel=1e-3)
    assert cumtrapz(ys, xs)[-1] == pytest.approx(trapz(ys, xs))
    assert interp1(0.25, [0.0, 1.0], [0.0, 2.0]) == pytest.approx(0.5)
    assert interp1(-1.0, [0.0, 1.0], [0.0, 2.0]) == 0.0  # clamped
    assert clamp(5.0, 0.0, 1.0) == 1.0 and clamp(-1.0, 0.0, 1.0) == 0.0
    assert argmax([1.0, 5.0, 3.0]) == 1


def test_solve_linear_and_gauss_legendre():
    from magform._numerics import gauss_legendre, solve_linear
    from magform.errors import ConvergenceError, InputError

    a = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
    b = [8.0, -11.0, -3.0]
    assert solve_linear(a, b) == pytest.approx([2.0, 3.0, -1.0])
    assert a[0] == [2.0, 1.0, -1.0] and b == [8.0, -11.0, -3.0]          # inputs untouched
    assert solve_linear([[0.0, 1.0], [1.0, 0.0]], [3.0, 4.0]) == pytest.approx([4.0, 3.0])  # needs pivoting
    with pytest.raises(ConvergenceError):
        solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
    with pytest.raises(InputError):
        solve_linear([[1.0, 2.0]], [1.0])
    for n in (1, 2, 5, 16):
        xs, ws = gauss_legendre(n)
        assert len(xs) == n and sum(ws) == pytest.approx(2.0, rel=1e-13)
        assert list(xs) == sorted(xs) and all(-1.0 < x < 1.0 for x in xs)
        # exact for polynomials up to degree 2n - 1
        degree = 2 * n - 2
        assert sum(w * x ** degree for x, w in zip(xs, ws)) == pytest.approx(2.0 / (degree + 1), rel=1e-12)
    assert gauss_legendre(5) is gauss_legendre(5)                            # cached
    with pytest.raises(InputError):
        gauss_legendre(0)
