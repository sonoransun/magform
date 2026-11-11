"""Small, dependency-free numerical helpers shared by the physics modules.

Everything here works on plain Python floats and lists so the core package has
no third-party requirements.  The routines are deliberately simple and robust
rather than fast: the models in this package are low-dimensional.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from .errors import ConvergenceError, InputError

_EPS = 2.220446049250313e-16

Vector = list[float]
Derivative = Callable[[float, Vector], Vector]


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------
def rk4_step(f: Derivative, t: float, y: Vector, h: float) -> Vector:
    """One classical fourth-order Runge-Kutta step for ``dy/dt = f(t, y)``."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, [yi + 0.5 * h * ki for yi, ki in zip(y, k1)])
    k3 = f(t + 0.5 * h, [yi + 0.5 * h * ki for yi, ki in zip(y, k2)])
    k4 = f(t + h, [yi + h * ki for yi, ki in zip(y, k3)])
    return [
        yi + (h / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for yi, a, b, c, d in zip(y, k1, k2, k3, k4)
    ]


def rk4(
    f: Derivative,
    y0: Sequence[float],
    t0: float,
    t1: float,
    n_steps: int,
    stop: Callable[[float, Vector], bool] | None = None,
) -> list[tuple[float, Vector]]:
    """Integrate ``dy/dt = f(t, y)`` from ``t0`` to ``t1`` with fixed-step RK4.

    Returns a list of ``(t, y)`` samples including the initial state.  If
    ``stop(t, y)`` returns True after a step, integration ends early (the
    terminating sample is included).
    """
    if n_steps < 1:
        raise InputError("n_steps must be >= 1")
    if t1 <= t0:
        raise InputError("t1 must be greater than t0")
    h = (t1 - t0) / n_steps
    t = t0
    y = list(map(float, y0))
    out: list[tuple[float, Vector]] = [(t, list(y))]
    for _ in range(n_steps):
        y = rk4_step(f, t, y, h)
        t += h
        out.append((t, list(y)))
        if stop is not None and stop(t, y):
            break
    return out


# ---------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------
def bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Bisection on ``[a, b]``; ``f(a)`` and ``f(b)`` must differ in sign."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise InputError("bisect: root is not bracketed by [a, b]")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0.0 or 0.5 * abs(b - a) < tol * max(1.0, abs(m)):
            return m
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Brent's method (Van Wijngaarden-Dekker-Brent) for a bracketed root.

    Combines bisection, secant and inverse quadratic interpolation; the root
    must be bracketed by ``[a, b]``.
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise InputError("brent: root is not bracketed by [a, b]")
    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * _EPS * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            min1 = 3.0 * xm * q - abs(tol1 * q)
            min2 = abs(e * q)
            if 2.0 * p < min(min1, min2):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a, fa = b, fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0.0 else -tol1
        fb = f(b)
    raise ConvergenceError("brent: no convergence within max_iter")


def newton(
    f: Callable[[float], float],
    x0: float,
    dfdx: Callable[[float], float] | None = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Newton's method; falls back to a secant step if no derivative is given."""
    x = float(x0)
    x_prev = x * (1.0 + 1e-4) + 1e-8
    f_prev = f(x_prev)
    for _ in range(max_iter):
        fx = f(x)
        if fx == 0.0:
            return x
        if dfdx is not None:
            slope = dfdx(x)
        else:
            denom = x - x_prev
            slope = (fx - f_prev) / denom if denom != 0.0 else 0.0
        if slope == 0.0 or not math.isfinite(slope):
            raise ConvergenceError("newton: zero or non-finite slope")
        x_new = x - fx / slope
        x_prev, f_prev = x, fx
        if abs(x_new - x) <= tol * max(1.0, abs(x_new)):
            return x_new
        x = x_new
    raise ConvergenceError("newton: no convergence within max_iter")


def find_bracket(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    factor: float = 1.6,
    max_iter: int = 60,
) -> tuple[float, float]:
    """Expand ``[x0, x1]`` geometrically until ``f`` changes sign across it."""
    if x0 == x1:
        raise InputError("find_bracket: initial points must differ")
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if f0 * f1 <= 0.0:
            return (x0, x1) if x0 < x1 else (x1, x0)
        if abs(f0) < abs(f1):
            x0 += factor * (x0 - x1)
            f0 = f(x0)
        else:
            x1 += factor * (x1 - x0)
            f1 = f(x1)
    raise ConvergenceError("find_bracket: could not bracket a root")


def solve_quadratic(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a x^2 + b x + c = 0`` in ascending order (numerically stable)."""
    if a == 0.0:
        if b == 0.0:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    q = -0.5 * (b + math.copysign(sq, b)) if b != 0.0 else -0.5 * sq
    if q == 0.0:
        return [0.0, 0.0]
    r1 = q / a
    r2 = c / q
    return sorted([r1, r2])


# ---------------------------------------------------------------------------
# Grids, quadrature, interpolation
# ---------------------------------------------------------------------------
def linspace(a: float, b: float, n: int) -> list[float]:
    """``n`` evenly spaced samples from ``a`` to ``b`` inclusive."""
    if n < 1:
        raise InputError("n must be >= 1")
    if n == 1:
        return [float(a)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def logspace(exp_a: float, exp_b: float, n: int, base: float = 10.0) -> list[float]:
    """``n`` samples log-spaced from ``base**exp_a`` to ``base**exp_b``."""
    return [base ** e for e in linspace(exp_a, exp_b, n)]


def trapz(y: Sequence[float], x: Sequence[float]) -> float:
    """Trapezoidal integral of samples ``y`` over abscissae ``x``."""
    if len(y) != len(x):
        raise InputError("trapz: y and x must have the same length")
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def cumtrapz(y: Sequence[float], x: Sequence[float]) -> list[float]:
    """Cumulative trapezoidal integral; same length as the inputs, starts at 0."""
    if len(y) != len(x):
        raise InputError("cumtrapz: y and x must have the same length")
    out = [0.0]
    for i in range(1, len(x)):
        out.append(out[-1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]))
    return out


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp ``x`` to the closed interval ``[lo, hi]``."""
    if lo > hi:
        raise InputError("clamp: lo must be <= hi")
    return lo if x < lo else hi if x > hi else x


def interp1(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Piecewise-linear interpolation with end clamping; ``xs`` must be increasing."""
    if len(xs) != len(ys) or len(xs) == 0:
        raise InputError("interp1: xs and ys must be non-empty and equal length")
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    frac = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + frac * (ys[hi] - ys[lo])


def argmax(values: Sequence[float]) -> int:
    """Index of the maximum value."""
    if not values:
        raise InputError("argmax of empty sequence")
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def solve_tridiagonal(
    lower: Sequence[float],
    diag: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> list[float]:
    """Thomas algorithm for a tridiagonal system.

    ``lower[i]`` multiplies x[i-1] in row i (``lower[0]`` ignored),
    ``upper[i]`` multiplies x[i+1] in row i (``upper[-1]`` ignored).
    Intended for diagonally dominant systems (implicit diffusion steps).
    """
    n = len(diag)
    if not (len(lower) == len(upper) == len(rhs) == n) or n == 0:
        raise InputError("solve_tridiagonal: all inputs must have the same non-zero length")
    c = [0.0] * n
    d = [0.0] * n
    denom = diag[0]
    if denom == 0.0:
        raise ConvergenceError("solve_tridiagonal: zero pivot")
    c[0] = upper[0] / denom if n > 1 else 0.0
    d[0] = rhs[0] / denom
    for i in range(1, n):
        denom = diag[i] - lower[i] * c[i - 1]
        if denom == 0.0:
            raise ConvergenceError("solve_tridiagonal: zero pivot")
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    x = [0.0] * n
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


bracket = find_bracket
"""Alias of :func:`find_bracket`."""


def solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve the dense system ``A x = b`` by Gaussian elimination with partial pivoting.

    ``matrix`` is a square list of rows; neither input is modified.  Intended for
    the small (n <~ 300) boundary-element systems of :mod:`magform.fieldmap`.
    Raises :class:`ConvergenceError` for a singular matrix.
    """
    n = len(rhs)
    if n == 0 or len(matrix) != n or any(len(row) != n for row in matrix):
        raise InputError("solve_linear: matrix must be square and match the right-hand side")
    a = [list(map(float, row)) + [float(b)] for row, b in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(a[i][col]))
        if a[pivot][col] == 0.0 or not math.isfinite(a[pivot][col]):
            raise ConvergenceError("solve_linear: singular matrix")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        row_c = a[col]
        inv = 1.0 / row_c[col]
        for i in range(col + 1, n):
            row_i = a[i]
            factor = row_i[col] * inv
            if factor != 0.0:
                for j in range(col + 1, n + 1):
                    row_i[j] -= factor * row_c[j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row_i = a[i]
        total = row_i[n]
        for j in range(i + 1, n):
            total -= row_i[j] * x[j]
        x[i] = total / row_i[i]
    return x


_GAUSS_LEGENDRE_CACHE: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}


def gauss_legendre(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes and weights of the ``n``-point Gauss-Legendre rule on ``[-1, 1]`` (cached).

    Exact for polynomials of degree ``2n - 1``; nodes are found by Newton iteration
    on the Legendre recurrence and returned in ascending order.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise InputError("gauss_legendre: n must be an integer >= 1")
    cached = _GAUSS_LEGENDRE_CACHE.get(n)
    if cached is not None:
        return cached
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        dp = 1.0
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0) if n > 1 else 1.0
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    order = sorted(range(n), key=lambda i: nodes[i])
    result = (tuple(nodes[i] for i in order), tuple(weights[i] for i in order))
    _GAUSS_LEGENDRE_CACHE[n] = result
    return result


__all__ = [
    "rk4", "rk4_step", "bisect", "brent", "newton", "find_bracket", "bracket", "solve_quadratic",
    "solve_tridiagonal", "solve_linear", "gauss_legendre", "linspace", "logspace", "trapz", "cumtrapz",
    "clamp", "interp1", "argmax",
]
