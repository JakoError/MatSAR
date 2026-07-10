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
