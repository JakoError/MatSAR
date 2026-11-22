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
