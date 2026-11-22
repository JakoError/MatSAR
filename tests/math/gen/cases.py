"""Case spec. Curated cases are filled with arange(1..n) (deterministic, readable);
random cases are drawn with numpy.random.default_rng(seed)."""
AXIS_NONE = None

CASES = {
    "sum": {
        "curated": [dict(shape=(2, 3), axis=0), dict(shape=(2, 3), axis=1),
                    dict(shape=(4,), axis=0), dict(shape=(4,), axis=AXIS_NONE), dict(shape=(1,), axis=0)],
        "random": dict(count=40, shapes=[(4,), (2, 3), (5, 4)], axes=[0, 1, AXIS_NONE],
                       lo=-10.0, hi=10.0, seed=1001),
        "rtol": 1e-12, "atol": 1e-13,
    },
    "prod": {
        "curated": [dict(shape=(2, 3), axis=0), dict(shape=(2, 3), axis=1), dict(shape=(4,), axis=AXIS_NONE)],
        "random": dict(count=30, shapes=[(4,), (2, 3)], axes=[0, 1, AXIS_NONE],
                       lo=-2.0, hi=2.0, seed=1002),
        "rtol": 1e-12, "atol": 1e-13,
    },
    "cumsum": {
        "curated": [dict(shape=(2, 3), axis=0), dict(shape=(2, 3), axis=1),
                    dict(shape=(4,), axis=0), dict(shape=(4,), axis=AXIS_NONE)],
        "random": dict(count=30, shapes=[(4,), (2, 3), (5, 4)], axes=[0, 1],
                       lo=-10.0, hi=10.0, seed=1003),
        "rtol": 1e-12, "atol": 1e-13,
    },
    "cumprod": {
        "curated": [dict(shape=(2, 3), axis=0), dict(shape=(4,), axis=AXIS_NONE)],
        "random": dict(count=20, shapes=[(4,), (2, 3)], axes=[0, 1],
                       lo=-1.5, hi=1.5, seed=1004),
        "rtol": 1e-12, "atol": 1e-13,
    },
    # mean/var are 1-based `dim` in MatSAR; the stored axis below is the 0-based
    # equivalent (matching sum), and the C++ replay maps it back (axis+1; dim<=0=global).
    # The non-square (2,3) axis-0 AND axis-1 plus the 1-D (4,) AXIS_NONE cases are exactly
    # the inputs that expose the off-by-one / global-vs-axis reduce bug.
    "mean": {
        "curated": [dict(shape=(2, 3), axis=0), dict(shape=(2, 3), axis=1),
                    dict(shape=(4,), axis=0), dict(shape=(4,), axis=AXIS_NONE),
                    dict(shape=(2, 3), axis=AXIS_NONE)],
        "random": dict(count=30, shapes=[(4,), (2, 3), (5, 4)], axes=[0, 1, AXIS_NONE],
                       lo=-10.0, hi=10.0, seed=2001),
        "rtol": 1e-12, "atol": 1e-13,
    },
    "var": {
        "curated": [dict(shape=(2, 3), axis=0), dict(shape=(2, 3), axis=1),
                    dict(shape=(4,), axis=0), dict(shape=(4,), axis=AXIS_NONE),
                    dict(shape=(2, 3), axis=AXIS_NONE)],
        "random": dict(count=30, shapes=[(4,), (2, 3), (5, 4)], axes=[0, 1, AXIS_NONE],
                       lo=-10.0, hi=10.0, seed=2002),
        "rtol": 1e-12, "atol": 1e-13,
    },
}

INT_SUBSET = {
    "sum":  [dict(shape=(2, 3), axis=0), dict(shape=(4,), axis=AXIS_NONE)],
    "prod": [dict(shape=(3,), axis=AXIS_NONE)],
}


# ---- Trigonometry (elementwise) -------------------------------------------------
# points: exact edge/special values (expected computed by NumPy, incl. NaN/Inf).
# random: |x| ~ U(lo, hi) (times a random sign when signed), restricted to the domain.
_T = dict(rtol=1e-12, atol=1e-14)
TRIG_SHAPES = [(4,), (2, 3), (3, 2)]
TRIG_CASES = {
    "sin":   dict(points=[0, 0.5, -0.5, 1.5707963267948966, 3.141592653589793, -2.5], random=dict(lo=0, hi=10, signed=True), **_T),
    "cos":   dict(points=[0, 0.5, -0.5, 3.141592653589793, -2.5, 7], random=dict(lo=0, hi=10, signed=True), **_T),
    "tan":   dict(points=[0, 0.7853981633974483, -0.7853981633974483, 1.2, -1.2], random=dict(lo=0, hi=1.5, signed=True), **_T),
    "sinh":  dict(points=[0, 1, -1, 3.5], random=dict(lo=0, hi=5, signed=True), **_T),
    "cosh":  dict(points=[0, 1, -1, 3.5], random=dict(lo=0, hi=5, signed=True), **_T),
    "tanh":  dict(points=[0, 1, -1, 20, -20], random=dict(lo=0, hi=5, signed=True), **_T),
    "asin":  dict(points=[0, 1, -1, 0.5, -0.5, 2], random=dict(lo=0, hi=1, signed=True), **_T),
    "acos":  dict(points=[0, 1, -1, 0.5, -0.5, 2], random=dict(lo=0, hi=1, signed=True), **_T),
    "atan":  dict(points=[0, 1, -1, 1e6, -1e6], random=dict(lo=0, hi=50, signed=True), **_T),
    "asinh": dict(points=[0, 1, -1, 100], random=dict(lo=0, hi=50, signed=True), **_T),
    "acosh": dict(points=[1, 2, 10, 0.5], random=dict(lo=1, hi=20), **_T),
    "atanh": dict(points=[0, 0.5, -0.5, 1, -1, 2], random=dict(lo=0, hi=0.99, signed=True), **_T),
    "cot":   dict(points=[0.7853981633974483, -0.7853981633974483, 1, 2.5, 0], random=dict(lo=0.1, hi=3.0, signed=True), **_T),
    "coth":  dict(points=[1, -1, 0.5, 0], random=dict(lo=0.1, hi=5, signed=True), **_T),
    "acot":  dict(points=[1, -1, 0.5, 100, -100], random=dict(lo=0.05, hi=50, signed=True), **_T),
    "acoth": dict(points=[2, -2, 1.5, 100, 1], random=dict(lo=1.05, hi=50, signed=True), **_T),
    "sec":   dict(points=[0, 1, -1, 3.141592653589793], random=dict(lo=0, hi=1.4, signed=True), **_T),
    "sech":  dict(points=[0, 1, -1, 5], random=dict(lo=0, hi=10, signed=True), **_T),
    "asec":  dict(points=[1, -1, 2, -2, 100, 0.5], random=dict(lo=1, hi=50, signed=True), **_T),
    "asech": dict(points=[1, 0.5, 0.1, 2], random=dict(lo=0.02, hi=1), **_T),
    "csc":   dict(points=[1.5707963267948966, -1.5707963267948966, 1, 0.5, 0], random=dict(lo=0.1, hi=3.0, signed=True), **_T),
    "csch":  dict(points=[1, -1, 0.5, 5, 0], random=dict(lo=0.05, hi=10, signed=True), **_T),
    "acsc":  dict(points=[1, -1, 2, -2, 100, 0.5], random=dict(lo=1, hi=50, signed=True), **_T),
    "acsch": dict(points=[1, -1, 0.5, 100], random=dict(lo=0.05, hi=50, signed=True), **_T),
    "angle": dict(points=[1, 0, -1, 2.5, -2.5], random=dict(lo=0.01, hi=10, signed=True), **_T),
}
TRIG_RANDOM_COUNT = 12

# Integer inputs (Matrix<int64>): results are floating point, same values as for doubles.
TRIG_INT_POINTS = {
    "sin": [-3, -1, 0, 1, 2, 7], "cos": [-3, -1, 0, 1, 2, 7], "atan": [-5, -1, 0, 1, 3, 40],
    "asinh": [-5, 0, 1, 9], "tanh": [-2, 0, 1, 3], "acosh": [1, 2, 5, 30],
}

# Binary cases: (shape_a, shape_b) pairs, same-rank broadcasting only.
TRIG_BINARY_SHAPES = [((4,), (4,)), ((2, 3), (2, 3)), ((2, 3), (1, 3)), ((2, 3), (2, 1)), ((2, 3), (1, 1))]
TRIG_BINARY_CASES = {
    # atan2(Y, X): quadrant edges (no -0.0, whose sign NumPy and MatSAR may treat differently)
    "atan2": dict(points=([0, 1, -1, 1, -1, 0, 2], [1, 0, 0, -1, -1, -1, 2]),
                  random=dict(lo_a=-5, hi_a=5, lo_b=-5, hi_b=5, seed=3101), **_T),
    "pol2cart_x": dict(points=([0, 1.5707963267948966, 3.141592653589793, 0.5], [1, 2, 3, 0]),
                       random=dict(lo_a=-6.3, hi_a=6.3, lo_b=0, hi_b=10, seed=3102), **_T),
    "pol2cart_y": dict(points=([0, 1.5707963267948966, 3.141592653589793, 0.5], [1, 2, 3, 0]),
                       random=dict(lo_a=-6.3, hi_a=6.3, lo_b=0, hi_b=10, seed=3102), **_T),
}


# ---- MatrixOperation ---------------------------------------------------------------
# Elementwise: like Trigonometry (points + seeded random). Tolerances: exact for the
# rounding/abs/sqrt family (IEEE-exact operations); a few ulp for libm exp/log; gamma and
# gammaln are looser (libm tgamma/lgamma are only accurate to several ulp).
_EXACT = dict(rtol=0.0, atol=0.0)
_LIBM = dict(rtol=1e-15, atol=0.0)
_GAMMA = dict(rtol=1e-13, atol=1e-15)
inf, nan = float("inf"), float("nan")
MATOP_SHAPES = [(4,), (2, 3), (3, 2)]
MATOP_RANDOM_COUNT = 8
_ROUNDING_POINTS = [0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, -0.49999999999999994,
                    2.0000000000000004, -2.7, 2.7, 4503599627370495.5, 1e16, inf, -inf, nan]
MATOP_CASES = {
    "abs":   dict(points=[0, 1.5, -1.5, -7, -1e300, 5e-324, -inf, nan], random=dict(lo=0, hi=100, signed=True), **_EXACT),
    "ceil":  dict(points=_ROUNDING_POINTS, random=dict(lo=0, hi=100, signed=True), **_EXACT),
    "floor": dict(points=_ROUNDING_POINTS, random=dict(lo=0, hi=100, signed=True), **_EXACT),
    "fix":   dict(points=_ROUNDING_POINTS, random=dict(lo=0, hi=100, signed=True), **_EXACT),
    "round": dict(points=_ROUNDING_POINTS, random=dict(lo=0, hi=100, signed=True), **_EXACT),
    "sqrt":  dict(points=[0, 1, 2, 4, 1e-300, 1e300, -1, inf, nan], random=dict(lo=0, hi=1000), **_EXACT),
    "exp":   dict(points=[0, 1, -1, 1e-10, 709.7, 710, -700, -800, inf, -inf, nan], random=dict(lo=0, hi=50, signed=True), **_LIBM),
    "exp2":  dict(points=[0, 1, -1, 10, 0.5, 3.3, -20, 1023, 1024, -inf, nan], random=dict(lo=0, hi=60, signed=True), **_LIBM),
    "log":   dict(points=[1, 2.718281828459045, 0.5, 1e-300, 1e300, 0, -1, inf, nan], random=dict(lo=-3, hi=3, log10=True), **_LIBM),
    "ln":    dict(points=[1, 10, 0.25, 0, -2], random=dict(lo=-3, hi=3, log10=True), **_LIBM),
    "log2":  dict(points=[1, 2, 8, 1024, 0.5, 3, 1e-300, 0, -1, inf], random=dict(lo=-3, hi=3, log10=True), **_LIBM),
    "log10": dict(points=[1, 10, 100, 1e-5, 2, 1e300, 0, -1, inf], random=dict(lo=-3, hi=3, log10=True), **_LIBM),
    "gamma": dict(points=[1, 2, 3, 5, 0.5, -0.5, -1.5, 10, 20, 170.5, 172, 0, 1e-10], random=dict(lo=0.05, hi=40), **_GAMMA),
    "gammaln": dict(points=[1, 2, 0.5, 3, 10, 100, 1e5, 1e-10, 0], random=dict(lo=0.05, hi=1000), **_GAMMA),
}
# Matrix<int64> input. abs keeps the integer type; the others return floating point.
MATOP_INT_POINTS = {
    "abs": [-5, 0, 7, -123456789], "ceil": [-7, 0, 7], "floor": [-7, 0, 7], "fix": [-7, 0, 7],
    "round": [-7, 0, 7], "sqrt": [0, 1, 2, 9, 16, 1000000], "log": [1, 2, 10, 1000],
    "exp": [0, 1, -3, 20], "gamma": [1, 2, 5, 10, 20],
}

# round(x, n): n digits after the point (n < 0: before it). Each case uses one n.
ROUND_N_POINTS = {
    1: [0.3, 0.1 + 0.2, 1.25, -1.25, 0.05, 2.675, 1e15 + 0.125, 1e-20, -0.04, inf, nan],
    2: [2.675, 1.005, 0.125, -0.125, 123.456, 9.995],
    3: [1.0005, 2.0004, -3.14159, 5e-4],
    -1: [15, -15, 14.9, 25, 1234.5],
    -2: [1250, -1250, 1249, 1234.5678, 50, -49],
}
ROUND_N_RANDOM = dict(ns=[1, 2, 3, -1, -2], lo=-1000, hi=1000, seed=3201)
# Matrix<int64> input: before the fix, an integer input was truncated before rounding.
ROUND_N_INT_POINTS = {-1: [15, -15, 14, 25, 0], -2: [1250, -1250, 1249, 1251, 50, -50, 999999], 2: [7, -7]}

MATOP_INT_CASES = {
    "isPrime": list(range(-3, 60)) + [7919, 7921, 104729, 1000003, 999999937, 2147483647, 4294967297,
                                      999999999989],
    "factorial": [0, 1, 2, 3, 5, 10, 18, 20, 22],
}
# Factorial of a float64 integer-valued input; above 22! the product loop rounds, so a
# small relative tolerance (171! overflows to Inf).
FACTORIAL_LARGE = dict(points=[23, 25, 50, 100, 150, 170, 171], rtol=1e-14, atol=0.0)

# Binary integer functions: points (a, b) + seeded random with same-rank broadcasting.
MATOP_BINARY_SHAPES = [((4,), (4,)), ((2, 3), (2, 3)), ((2, 3), (1, 3)), ((2, 3), (2, 1)), ((2, 3), (1, 1))]
MATOP_BINARY_CASES = {
    "gcd": dict(points=([12, 4, -4, 4, -4, 0, 0, 7, 17, 270, -270, 1071, -5],
                        [18, -6, 6, 0, 0, 5, 0, 13, 51, 192, -192, 462, -5]),
                random=dict(lo=-1000, hi=1000, nonzero_b=False, seed=3301), **_EXACT),
    "numden": dict(points=([1, 2, -2, 2, -2, 0, 0, 6, 100, -100, 7, 5],
                           [2, 4, 4, -4, -4, 5, -5, 9, 75, -75, 1, 1]),
                   random=dict(lo=-200, hi=200, nonzero_b=True, seed=3302), **_EXACT),
}
NCHOOSEK_POINTS = {
    # exactly representable results; NaN for k > n, negative or non-integer arguments
    "exact": dict(points=([5, 10, 0, 5, 5, 20, 50, 2000, 3, -1, 2.5, 5],
                          [2, 3, 0, 0, 5, 10, 25, 1, 5, 2, 1, -1]), rtol=0.0, atol=0.0),
    # large n: results that need rounding (or a product that overflows a naive loop)
    "large": dict(points=([60, 1000, 2000, 1030, 200, 1500],
                          [30, 500, 1999, 1000, 100, 1497]), rtol=1e-13, atol=0.0),
}

# Matrix functions. Normwise tolerance: |got - want| <= tol * (|want| + max|want|).
EXPM_CURATED = {
    "zeros3": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "diag": [[1, 0, 0], [0, 2, 0], [0, 0, -3]],
    "nilpotent": [[0, 1], [0, 0]],
    "jordan": [[1, 1], [0, 1]],
    "rotation": [[0, -1.0471975511965976], [1.0471975511965976, 0]],
    "scalar": [[2.5]],
    "moler_van_loan": [[-49, 24], [-64, 31]],
    "upper3": [[1, 2, 3], [0, 4, 5], [0, 0, 6]],
    # just below each Pade degree threshold theta_m (1-norm == spectral radius: the largest
    # truncation error a degree must absorb), and 2 * theta_13 (one squaring)
    "edge_pade3": [[0.0149, 0], [0, -0.0149]],
    "edge_pade5": [[0.2539, 0], [0, -0.2539]],
    "edge_pade7": [[0.9504, 0], [0, -0.9504]],
    "edge_pade9": [[2.0978, 0], [0, -2.0978]],
    "edge_pade13": [[5.3719, 0], [0, -5.3719]],
    "edge_scaled": [[10.7438, 0], [0, -10.7438]],
}
EXPM_RANDOM = dict(norms=[0.01, 0.1, 0.5, 0.9, 2, 5, 10, 30], sizes=[2, 3, 4, 5], seed=3401)
LOGM_CURATED = {
    "identity": [[1, 0], [0, 1]],
    "diag": [[2.718281828459045, 0, 0], [0, 7.38905609893065, 0], [0, 0, 1]],
    "jordan": [[1, 1], [0, 1]],
    "complex_pair": [[1, -1], [1, 1]],          # eigenvalues 1 +- i: the logarithm is real
    "scalar": [[10]],
    "spd": [[4, 1], [1, 3]],
    "upper": [[2, 1], [0, 3]],
    "wide_spectrum": [[1000, 1], [0, 0.001]],
}
LOGM_RANDOM = dict(norms=[0.1, 1, 2], sizes=[2, 3, 4], spd_sizes=[2, 3, 5], seed=3402)
MATFUN_TOL = {"expm": 1e-13, "logm": 1e-13, "mpower_int": 1e-13, "mpower_frac": 1e-13, "mpower_base": 1e-13}
MPOWER_P = [0, 1, 2, 3, 7, -1, -2, 0.5, 1.5, -0.5, 1 / 3]
MPOWER_BASES = [2, 0.5, 10, 2.718281828459045]
MPOWER_RANDOM = dict(sizes=[2, 3, 4], seed=3403)
