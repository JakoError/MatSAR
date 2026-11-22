"""Exact NumPy reference for each MathematicalFunction tag.
MatSAR dim_idx=k <-> numpy axis=k, reduced axis kept as length 1 (keepdims).
A bare call defaults dim_idx=0; global only when the caller passes `none` (axis=None here).
Global cumulative ops are restricted to 1-D in the pilot (C-order == F-order there)."""
import numpy as np


def _wrap1(v):
    return np.asarray([v], dtype=np.float64)   # global scalar -> shape (1,)


ORACLE = {
    "sum":     lambda a, axis: (_wrap1(a.sum())  if axis is None else a.sum(axis=axis, keepdims=True)),
    "prod":    lambda a, axis: (_wrap1(a.prod()) if axis is None else a.prod(axis=axis, keepdims=True)),
    "cumsum":  lambda a, axis: (np.cumsum(a)     if axis is None else np.cumsum(a, axis=axis)),
    "cumprod": lambda a, axis: (np.cumprod(a)    if axis is None else np.cumprod(a, axis=axis)),
    # mean/var use MatSAR's 1-based `dim`; the harness stores the equivalent 0-based
    # axis (matching sum) and the replay maps it back with +1 / <=0-for-global.
    # var default is population variance (ddof=0) == MatSAR's default w=false.
    "mean":    lambda a, axis: (_wrap1(a.mean()) if axis is None else a.mean(axis=axis, keepdims=True)),
    "var":     lambda a, axis: (_wrap1(a.var())  if axis is None else a.var(axis=axis, keepdims=True)),
}


# ---- Trigonometry (elementwise) ----------------------------------------------------
# References come from mpmath at 40 digits, rounded once to double: correctly rounded and
# platform-independent. (NumPy's transcendental functions depend on the CPU's SIMD path and
# the platform libm, so their last bit can differ between machines, which would break
# the byte-exact CI regen guard.) NumPy is still used for the seeded random inputs.
# MatSAR follows MATLAB for the reciprocal family: cot = 1/tan, acot(x) = atan(1/x),
# asec(x) = acos(1/x), acsc(x) = asin(1/x), ...; the inner 1/x is a double division, as in
# MatSAR. Real-valued: an out-of-domain input (complex result) is NaN, as in std:: and NumPy.
import math
import mpmath as mp

mp.mp.dps = 40


def _to_double(v):
    if isinstance(v, mp.mpc):
        return float("nan") if v.imag != 0 else float(v.real)
    return float(v)


def _recip(t, x):
    """1/t with IEEE semantics at t == 0 (only reached for x == +-0)."""
    return math.copysign(math.inf, x) if t == 0 else 1 / t


def _elementwise(f):
    return np.vectorize(lambda x: _to_double(f(mp.mpf(float(x)), float(x))), otypes=[np.float64])


def _of_recip(f):   # f(1/x), with 1/x rounded to double first
    return _elementwise(lambda v, x: f(mp.mpf(1.0 / x)) if x != 0 else f(mp.mpf(math.copysign(math.inf, x))))


TRIG_ORACLE = {
    "sin": _elementwise(lambda v, x: mp.sin(v)), "sinh": _elementwise(lambda v, x: mp.sinh(v)),
    "asin": _elementwise(lambda v, x: mp.asin(v)), "asinh": _elementwise(lambda v, x: mp.asinh(v)),
    "cos": _elementwise(lambda v, x: mp.cos(v)), "cosh": _elementwise(lambda v, x: mp.cosh(v)),
    "acos": _elementwise(lambda v, x: mp.acos(v)), "acosh": _elementwise(lambda v, x: mp.acosh(v)),
    "tan": _elementwise(lambda v, x: mp.tan(v)), "tanh": _elementwise(lambda v, x: mp.tanh(v)),
    "atan": _elementwise(lambda v, x: mp.atan(v)), "atanh": _elementwise(lambda v, x: mp.atanh(v)),
    "cot": _elementwise(lambda v, x: _recip(mp.tan(v), x)), "coth": _elementwise(lambda v, x: _recip(mp.tanh(v), x)),
    "sec": _elementwise(lambda v, x: _recip(mp.cos(v), x)), "sech": _elementwise(lambda v, x: _recip(mp.cosh(v), x)),
    "csc": _elementwise(lambda v, x: _recip(mp.sin(v), x)), "csch": _elementwise(lambda v, x: _recip(mp.sinh(v), x)),
    "acot": _of_recip(mp.atan), "acoth": _of_recip(mp.atanh),
    "asec": _of_recip(mp.acos), "asech": _of_recip(mp.acosh),
    "acsc": _of_recip(mp.asin), "acsch": _of_recip(mp.asinh),
    "angle": _elementwise(lambda v, x: mp.pi if x < 0 else 0),   # real input: pi for x < 0, else 0
}

# Binary: atan2(Y, X); pol2cart(theta, rho) -> x = rho*cos(theta), y = rho*sin(theta).
_binary = lambda f: np.vectorize(lambda a, b: _to_double(f(mp.mpf(float(a)), mp.mpf(float(b)))), otypes=[np.float64])
TRIG_BINARY_ORACLE = {
    "atan2": _binary(mp.atan2),
    "pol2cart_x": _binary(lambda theta, rho: rho * mp.cos(theta)),
    "pol2cart_y": _binary(lambda theta, rho: rho * mp.sin(theta)),
}
