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


# ---- MatrixOperation ---------------------------------------------------------------
# Rounding and integer functions are computed exactly in Python (no floating-point
# reference error at all); exp/log/gamma come from mpmath like the trig references.
# MATLAB semantics: round() rounds half away from zero (not NumPy's half-to-even), and
# round(x, n) rounds the exact value of x to n decimal digits (n < 0: to tens, hundreds, ...).
import decimal
from fractions import Fraction


def _finite(f):   # f on finite x; NaN/Inf pass through unchanged (as in C and MATLAB)
    return lambda x: float(f(x)) if math.isfinite(x) else x


def _round_digits(x, n):
    if not math.isfinite(x):
        return x
    with decimal.localcontext() as ctx:
        ctx.prec = 400
        q = decimal.Decimal(1).scaleb(-n)                     # 10^-n
        r = float(decimal.Decimal(x).quantize(q, rounding=decimal.ROUND_HALF_UP))
    return math.copysign(r, x) if r == 0 else r


def _exact(f):
    return np.vectorize(_finite(f), otypes=[np.float64])


def _gamma(v, x):
    return math.copysign(math.inf, x) if x == 0 else mp.gamma(v)   # C tgamma: pole at +-0


def _gammaln(v, x):
    return math.inf if x == 0 else mp.loggamma(v)


MATOP_ORACLE = {
    "abs": np.vectorize(math.fabs, otypes=[np.float64]), "ceil": _exact(math.ceil), "floor": _exact(math.floor),
    "fix": _exact(math.trunc), "round": _exact(lambda x: _round_digits(x, 0)),
    "sqrt": _elementwise(lambda v, x: mp.sqrt(v)),
    "exp": _elementwise(lambda v, x: mp.exp(v)), "exp2": _elementwise(lambda v, x: mp.mpf(2) ** v),
    "log": _elementwise(lambda v, x: mp.log(v)), "ln": _elementwise(lambda v, x: mp.log(v)),
    "log2": _elementwise(lambda v, x: mp.log(v, 2)), "log10": _elementwise(lambda v, x: mp.log10(v)),
    "gamma": _elementwise(_gamma), "gammaln": _elementwise(_gammaln),
}


def round_digits(a, n):
    return np.vectorize(lambda x: _round_digits(float(x), n), otypes=[np.float64])(a)


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _factorial(x):
    if x != math.floor(x) or x < 0:
        return math.nan
    try:
        return float(math.factorial(int(x)))
    except OverflowError:
        return math.inf


def _nchoosek(n, k):   # MatSAR: NaN for a non-integer, negative, or k > n argument
    if n != math.floor(n) or k != math.floor(k) or n < 0 or k < 0 or k > n:
        return math.nan
    try:
        return float(math.comb(int(n), int(k)))
    except OverflowError:
        return math.inf


_int2 = lambda f: np.vectorize(lambda a, b: float(f(int(a), int(b))), otypes=[np.float64])
MATOP_INT_ORACLE = {
    "isPrime": np.vectorize(lambda x: float(_is_prime(int(x))), otypes=[np.float64]),
    "factorial": np.vectorize(lambda x: _factorial(float(x)), otypes=[np.float64]),
}
MATOP_BINARY_ORACLE = {
    "gcd": _int2(math.gcd),                                        # always >= 0
    "numden_num": _int2(lambda a, b: Fraction(a, b).numerator),    # sign on the numerator
    "numden_den": _int2(lambda a, b: Fraction(a, b).denominator),
    "nchoosek": np.vectorize(lambda a, b: _nchoosek(float(a), float(b)), otypes=[np.float64]),
}


# Matrix functions (MATLAB expm / logm / mpower: functions of the matrix, not of each
# element), evaluated by mpmath at 40 digits and rounded once to double.
def _to_mp(a):
    return mp.matrix(np.asarray(a, dtype=np.float64).tolist())


def _from_mp(M):
    out = np.empty((M.rows, M.cols))
    for i in range(M.rows):
        for j in range(M.cols):
            v = M[i, j]
            if isinstance(v, mp.mpc):
                assert abs(v.imag) <= mp.mpf(10) ** -30 * max(1, abs(v)), "complex result"
                v = v.real
            out[i, j] = float(v)
    return out


MATRIX_ORACLE = {
    "expm": lambda a: _from_mp(mp.expm(_to_mp(a))),
    "logm": lambda a: _from_mp(mp.logm(_to_mp(a))),
}


def mpower(a, p):            # A^p for a scalar p (integer: repeated products; else principal power)
    A = _to_mp(a)
    if p == int(p):
        return _from_mp(A ** int(p))
    return _from_mp(mp.powm(A, mp.mpf(p)))


def mpower_base(s, a):       # s^A = expm(log(s) * A), s > 0
    return _from_mp(mp.expm(mp.log(mp.mpf(s)) * _to_mp(a)))


# ---- Probability -------------------------------------------------------------------
# Reductions: exact NumPy (max/min omit NaN as MATLAB does; an all-NaN slice is NaN).
# Densities: mpmath at 40 digits from the double inputs, rounded once. Conventions (MATLAB /
# SciPy): a NaN x gives NaN; an invalid parameter gives NaN; x outside the support gives 0
# (for the discrete ones also a non-integer x).
import warnings


def _keep(f):
    def g(a, axis):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN slice
            if axis is None:
                return np.asarray([f(a, axis=None)], dtype=np.float64)
            return f(a, axis=axis, keepdims=True)
    return g


PROB_REDUCE_ORACLE = {
    "max": _keep(np.nanmax), "min": _keep(np.nanmin), "median": _keep(np.median),
    "var_sample": _keep(lambda a, axis=None, **kw: np.var(a, axis=axis, ddof=1, **kw)),
}

_NAN, _INF = mp.nan, mp.inf


def _is_int(v):
    return v == mp.floor(v)


def _normpdf(x, mu, s):
    if s <= 0: return _NAN
    if mp.isinf(x): return mp.mpf(0)
    z = (x - mu) / s
    return mp.exp(-z * z / 2) / (s * mp.sqrt(2 * mp.pi))


def _binopdf(x, n, p):
    if n < 0 or not _is_int(n) or p < 0 or p > 1: return _NAN
    if not mp.isfinite(x) or not _is_int(x) or x < 0 or x > n: return mp.mpf(0)
    return mp.binomial(n, x) * mp.power(p, x) * mp.power(1 - p, n - x)


def _poisspdf(x, lam):
    if lam < 0: return _NAN
    if not mp.isfinite(x) or not _is_int(x) or x < 0: return mp.mpf(0)
    if lam == 0: return mp.mpf(1) if x == 0 else mp.mpf(0)
    return mp.exp(-lam + x * mp.log(lam) - mp.loggamma(x + 1))


def _chi2pdf(x, nu):
    if nu <= 0: return _NAN
    if x < 0 or mp.isinf(x): return mp.mpf(0)
    if x == 0: return _INF if nu < 2 else (mp.mpf(1) / 2 if nu == 2 else mp.mpf(0))
    k = nu / 2
    return mp.exp((k - 1) * mp.log(x) - x / 2 - k * mp.log(2) - mp.loggamma(k))


def _exppdf(x, mu):
    if mu <= 0: return _NAN
    if x < 0 or mp.isinf(x): return mp.mpf(0)
    return mp.exp(-x / mu) / mu


def _raylpdf(x, b):
    if b <= 0: return _NAN
    if x < 0 or mp.isinf(x): return mp.mpf(0)
    return x / (b * b) * mp.exp(-x * x / (2 * b * b))


def _weibpdf(x, lam, k):
    if lam <= 0 or k <= 0: return _NAN
    if x < 0 or mp.isinf(x): return mp.mpf(0)
    if x == 0: return _INF if k < 1 else (1 / lam if k == 1 else mp.mpf(0))
    t = x / lam
    return k / lam * mp.power(t, k - 1) * mp.exp(-mp.power(t, k))


def _unifpdf(x, a, b):
    if a >= b: return _NAN
    return 1 / (b - a) if a <= x <= b else mp.mpf(0)


def _pdf(f):
    """f(x, *params) on doubles -> np.float64 array; NaN x -> NaN."""
    def g(xs, params):
        ps = [mp.mpf(float(v)) for v in params]
        out = []
        for x in np.asarray(xs, dtype=np.float64).ravel(order="C"):
            if math.isnan(x):
                out.append(math.nan)
                continue
            v = f(mp.mpf(float(x)), *ps)
            out.append(math.nan if mp.isnan(v) else float(v))
        return np.asarray(out, dtype=np.float64).reshape(np.shape(xs))
    return g


PROB_PDF_ORACLE = {
    "normpdf": _pdf(_normpdf), "binopdf": _pdf(_binopdf), "poisspdf": _pdf(_poisspdf),
    "chi2pdf": _pdf(_chi2pdf), "exppdf": _pdf(_exppdf), "raylpdf": _pdf(_raylpdf),
    "weibpdf": _pdf(_weibpdf), "unifpdf": _pdf(_unifpdf),
}
