"""Generate tests/math/golden/golden_mathfunc.hpp from cases.py via oracle.py (deterministic)."""
import os
import zlib
import numpy as np
from cases import CASES, INT_SUBSET, AXIS_NONE
from cases import (TRIG_CASES, TRIG_SHAPES, TRIG_RANDOM_COUNT, TRIG_INT_POINTS,
                   TRIG_BINARY_SHAPES, TRIG_BINARY_CASES)
from oracle import ORACLE, TRIG_ORACLE, TRIG_BINARY_ORACLE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "golden", "golden_mathfunc.hpp")
OUT_TRIG = os.path.join(HERE, "..", "golden", "golden_trig.hpp")
TWO53 = 2.0 ** 53


def fmt(x):
    x = float(x)
    if np.isnan(x):     return "std::numeric_limits<double>::quiet_NaN()"
    if np.isposinf(x):  return "INFINITY"
    if np.isneginf(x):  return "-INFINITY"
    return "%.17g" % x


def flat_lit(a):
    vals = np.asarray(a, dtype=np.float64).ravel(order="C")
    for v in vals:
        if np.isfinite(v):
            assert abs(v) < TWO53, "golden magnitude %.3g exceeds 2^53 (not double-exact)" % v
    return "{" + ",".join(fmt(v) for v in vals) + "}"


def shape_lit(shape):
    return "{" + ",".join(str(int(d)) for d in shape) + "}"


def arange_input(shape):
    n = int(np.prod(shape))
    return np.arange(1, n + 1, dtype=np.float64).reshape(shape)


def case_lit(name, a, axis, expected, rtol, atol):
    ax = -1 if axis is None else int(axis)
    return ("{\"%s\",%s,%s,{},{},%d,%s,%s,%s,%s}"
            % (name, flat_lit(a), shape_lit(a.shape), ax,
               flat_lit(expected), shape_lit(expected.shape), fmt(rtol), fmt(atol)))


def case_lit2(name, a, b, expected, rtol, atol):
    return ("{\"%s\",%s,%s,%s,%s,-1,%s,%s,%s,%s}"
            % (name, flat_lit(a), shape_lit(a.shape), flat_lit(b), shape_lit(b.shape),
               flat_lit(expected), shape_lit(expected.shape), fmt(rtol), fmt(atol)))


def emit_array(cpp_name, cases):
    body = ",\n    ".join(cases)
    return "inline const std::vector<GoldenCase> %s = {\n    %s\n};\n" % (cpp_name, body)


def build_cases(tag, spec, force_int_tol=False):
    out = []
    rtol = 0.0 if force_int_tol else spec["rtol"]
    atol = 0.0 if force_int_tol else spec["atol"]
    curated = spec if isinstance(spec, list) else spec["curated"]
    for i, c in enumerate(curated):
        a = arange_input(c["shape"])
        exp = np.asarray(ORACLE[tag](a, c["axis"]), dtype=np.float64)
        axs = "none" if c["axis"] is None else str(c["axis"])
        out.append(case_lit("%s/cur%d/%s/ax%s" % (tag, i, "x".join(map(str, c["shape"])), axs), a, c["axis"], exp, rtol, atol))
    if isinstance(spec, dict) and "random" in spec:
        r = spec["random"]
        rng = np.random.default_rng(r["seed"])
        for i in range(r["count"]):
            shape = r["shapes"][rng.integers(len(r["shapes"]))]
            valid_axes = [ax for ax in r["axes"] if ax is None or ax < len(shape)]
            axis = valid_axes[int(rng.integers(len(valid_axes)))]
            a = rng.uniform(r["lo"], r["hi"], size=shape).astype(np.float64)
            exp = np.asarray(ORACLE[tag](a, axis), dtype=np.float64)
            axs = "none" if axis is None else str(axis)
            out.append(case_lit("%s/rnd%03d/ax%s" % (tag, i, axs), a, axis, exp, rtol, atol))
    return out


def tag_seed(tag):
    return 3000 + zlib.crc32(tag.encode()) % 1000   # stable per tag, independent of dict order


def build_trig_unary(tag, spec):
    fn = TRIG_ORACLE[tag]
    out = []
    with np.errstate(all="ignore"):
        pts = np.asarray(spec["points"], dtype=np.float64)
        out.append(case_lit("%s/points" % tag, pts, None, fn(pts), spec["rtol"], spec["atol"]))
        r = spec["random"]
        rng = np.random.default_rng(tag_seed(tag))
        for i in range(TRIG_RANDOM_COUNT):
            shape = TRIG_SHAPES[int(rng.integers(len(TRIG_SHAPES)))]
            a = rng.uniform(r["lo"], r["hi"], size=shape)
            if r.get("signed"):
                a = a * rng.choice([-1.0, 1.0], size=shape)
            out.append(case_lit("%s/rnd%03d/%s" % (tag, i, "x".join(map(str, shape))),
                                a, None, fn(a), spec["rtol"], spec["atol"]))
    return out


def build_trig_int(tag, points):
    a = np.asarray(points, dtype=np.float64)
    spec = TRIG_CASES[tag]
    return [case_lit("%s/int" % tag, a, None, TRIG_ORACLE[tag](a), spec["rtol"], spec["atol"])]


def build_trig_binary(tag, spec):
    fn = TRIG_BINARY_ORACLE[tag]
    pa, pb = (np.asarray(v, dtype=np.float64) for v in spec["points"])
    out = [case_lit2("%s/points" % tag, pa, pb, fn(pa, pb), spec["rtol"], spec["atol"])]
    r = spec["random"]
    rng = np.random.default_rng(r["seed"])
    for i, (sa, sb) in enumerate(TRIG_BINARY_SHAPES):
        a = rng.uniform(r["lo_a"], r["hi_a"], size=sa)
        b = rng.uniform(r["lo_b"], r["hi_b"], size=sb)
        out.append(case_lit2("%s/rnd%03d/%s_%s" % (tag, i, "x".join(map(str, sa)), "x".join(map(str, sb))),
                             a, b, fn(a, b), spec["rtol"], spec["atol"]))
    return out


def write_header(path, guard, arrays):
    parts = ["#ifndef %s" % guard, "#define %s" % guard,
             "// GENERATED by tests/math/gen/generate.py -- do not edit by hand.",
             "#include \"golden.hpp\"", "", "namespace matsar_test {", ""]
    parts += arrays
    parts += ["}  // namespace matsar_test", "#endif", ""]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(parts))
    with open(path + ".provenance", "w", newline="\n") as f:
        f.write("numpy=%s\ngenerator=generate.py@v1\n" % np.__version__)
    print("wrote", path)


def main_trig():
    arrays = [emit_array("%s_golden" % t, build_trig_unary(t, s)) for t, s in TRIG_CASES.items()]
    arrays += [emit_array("%s_golden_int" % t, build_trig_int(t, p)) for t, p in TRIG_INT_POINTS.items()]
    arrays += [emit_array("%s_golden" % t, build_trig_binary(t, s)) for t, s in TRIG_BINARY_CASES.items()]
    write_header(OUT_TRIG, "MATSAR_GOLDEN_TRIG_HPP", arrays)


def main():
    parts = ["#ifndef MATSAR_GOLDEN_MATHFUNC_HPP", "#define MATSAR_GOLDEN_MATHFUNC_HPP",
             "// GENERATED by tests/math/gen/generate.py -- do not edit by hand.",
             "#include \"golden.hpp\"", "", "namespace matsar_test {", ""]
    for tag, spec in CASES.items():
        parts.append(emit_array("%s_golden" % tag, build_cases(tag, spec)))
    for tag, sub in INT_SUBSET.items():
        parts.append(emit_array("%s_golden_int" % tag, build_cases(tag, sub, force_int_tol=True)))
    parts += ["}  // namespace matsar_test", "#endif", ""]
    text = "\n".join(parts)
    with open(OUT, "w", newline="\n") as f:
        f.write(text)
    with open(OUT + ".provenance", "w", newline="\n") as f:
        f.write("numpy=%s\ngenerator=generate.py@v1\n" % np.__version__)
    print("wrote", OUT)
    main_trig()


if __name__ == "__main__":
    main()
