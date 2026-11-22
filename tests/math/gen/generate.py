"""Generate tests/math/golden/golden_mathfunc.hpp from cases.py via oracle.py (deterministic)."""
import os
import zlib
import numpy as np
from cases import CASES, INT_SUBSET, AXIS_NONE
from cases import (TRIG_CASES, TRIG_SHAPES, TRIG_RANDOM_COUNT, TRIG_INT_POINTS,
                   TRIG_BINARY_SHAPES, TRIG_BINARY_CASES)
from oracle import ORACLE, TRIG_ORACLE, TRIG_BINARY_ORACLE
import cases as C
import oracle as O

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "golden", "golden_mathfunc.hpp")
OUT_TRIG = os.path.join(HERE, "..", "golden", "golden_trig.hpp")
OUT_MATOP = os.path.join(HERE, "..", "golden", "golden_matop.hpp")
TWO53 = 2.0 ** 53


def fmt(x):
    x = float(x)
    if np.isnan(x):     return "std::numeric_limits<double>::quiet_NaN()"
    if np.isposinf(x):  return "INFINITY"
    if np.isneginf(x):  return "-INFINITY"
    return "%.17g" % x


def flat_lit(a, check53=True):
    vals = np.asarray(a, dtype=np.float64).ravel(order="C")
    for v in vals:
        if check53 and np.isfinite(v):
            assert abs(v) < TWO53, "golden magnitude %.3g exceeds 2^53 (not double-exact)" % v
    return "{" + ",".join(fmt(v) for v in vals) + "}"


def shape_lit(shape):
    return "{" + ",".join(str(int(d)) for d in shape) + "}"


def arange_input(shape):
    n = int(np.prod(shape))
    return np.arange(1, n + 1, dtype=np.float64).reshape(shape)


def case_lit(name, a, axis, expected, rtol, atol, check53=True):
    ax = -1 if axis is None else int(axis)
    return ("{\"%s\",%s,%s,{},{},%d,%s,%s,%s,%s}"
            % (name, flat_lit(a, check53), shape_lit(a.shape), ax,
               flat_lit(expected, check53), shape_lit(expected.shape), fmt(rtol), fmt(atol)))


def case_lit2(name, a, b, expected, rtol, atol, check53=True):
    return ("{\"%s\",%s,%s,%s,%s,-1,%s,%s,%s,%s}"
            % (name, flat_lit(a, check53), shape_lit(a.shape), flat_lit(b, check53), shape_lit(b.shape),
               flat_lit(expected, check53), shape_lit(expected.shape), fmt(rtol), fmt(atol)))


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


# ---- MatrixOperation ----------------------------------------------------------------
# Magnitudes here are not limited to 2^53 (factorial, exp, ...): %.17g round-trips any double.
def _arr(v):
    return np.asarray(v, dtype=np.float64)


def _lit1(name, a, out, rtol, atol):
    return case_lit(name, _arr(a), None, _arr(out), rtol, atol, check53=False)


def _lit2(name, a, b, out, rtol, atol):
    return case_lit2(name, _arr(a), _arr(b), _arr(out), rtol, atol, check53=False)


def _shape_tag(*shapes):
    return "_".join("x".join(map(str, s)) for s in shapes)


def build_matop_unary(tag, spec):
    fn, out = O.MATOP_ORACLE[tag], []
    pts = _arr(spec["points"])
    out.append(_lit1("%s/points" % tag, pts, fn(pts), spec["rtol"], spec["atol"]))
    r = spec["random"]
    rng = np.random.default_rng(tag_seed("matop/" + tag))
    for i in range(C.MATOP_RANDOM_COUNT):
        shape = C.MATOP_SHAPES[int(rng.integers(len(C.MATOP_SHAPES)))]
        a = rng.uniform(r["lo"], r["hi"], size=shape)
        if r.get("log10"):
            a = 10.0 ** a
        if r.get("signed"):
            a = a * rng.choice([-1.0, 1.0], size=shape)
        out.append(_lit1("%s/rnd%03d/%s" % (tag, i, _shape_tag(shape)), a, fn(a), spec["rtol"], spec["atol"]))
    return out


def build_round_n():
    out = []
    for n, pts in C.ROUND_N_POINTS.items():
        out.append(_lit2("round_n/points/n=%d" % n, pts, [n], O.round_digits(_arr(pts), n), 0.0, 0.0))
    r = C.ROUND_N_RANDOM
    rng = np.random.default_rng(r["seed"])
    for n in r["ns"]:
        a = rng.uniform(r["lo"], r["hi"], size=(2, 3))
        out.append(_lit2("round_n/rnd/n=%d" % n, a, [n], O.round_digits(a, n), 0.0, 0.0))
    return out


def build_round_n_int():
    return [_lit2("round_n/int/n=%d" % n, pts, [n], O.round_digits(_arr(pts), n), 0.0, 0.0)
            for n, pts in C.ROUND_N_INT_POINTS.items()]


def build_matop_binary(spec, oracle_tags):
    """One golden array per oracle tag (numden has two outputs), all on the same inputs."""
    pa, pb = (_arr(v) for v in spec["points"])
    inputs = [("points", pa, pb)]
    r = spec["random"]
    rng = np.random.default_rng(r["seed"])
    for i, (sa, sb) in enumerate(C.MATOP_BINARY_SHAPES):
        a = rng.integers(r["lo"], r["hi"] + 1, size=sa).astype(np.float64)
        b = rng.integers(r["lo"], r["hi"] + 1, size=sb).astype(np.float64)
        if r["nonzero_b"]:
            b[b == 0] = 1
        inputs.append(("rnd%03d/%s" % (i, _shape_tag(sa, sb)), a, b))
    return {t: [_lit2("%s/%s" % (t, nm), a, b, O.MATOP_BINARY_ORACLE[t](a, b), spec["rtol"], spec["atol"])
                for nm, a, b in inputs] for t in oracle_tags}


def build_nchoosek():
    out = []
    for nm, spec in C.NCHOOSEK_POINTS.items():
        a, b = (_arr(v) for v in spec["points"])
        out.append(_lit2("nchoosek/%s" % nm, a, b, O.MATOP_BINARY_ORACLE["nchoosek"](a, b),
                         spec["rtol"], spec["atol"]))
    return out


def _matfun_lit(name, a, b, expected, tol):
    """Normwise tolerance: |got - want| <= tol * |want| + tol * max|want|."""
    expected = _arr(expected)
    atol = tol * float(np.max(np.abs(expected))) if expected.size else 0.0
    if b is None:
        return _lit1(name, a, expected, tol, atol)
    return _lit2(name, a, b, expected, tol, atol)


def _scaled_normal(rng, n, norm):
    a = rng.standard_normal((n, n))
    return a * (norm / np.abs(a).sum(axis=0).max())


def build_expm():
    tol, fn = C.MATFUN_TOL["expm"], O.MATRIX_ORACLE["expm"]
    out = [_matfun_lit("expm/%s" % nm, _arr(a), None, fn(_arr(a)), tol) for nm, a in C.EXPM_CURATED.items()]
    r = C.EXPM_RANDOM
    rng = np.random.default_rng(r["seed"])
    for norm in r["norms"]:
        n = r["sizes"][int(rng.integers(len(r["sizes"])))]
        a = _scaled_normal(rng, n, norm)
        out.append(_matfun_lit("expm/rnd/norm=%g/%dx%d" % (norm, n, n), a, None, fn(a), tol))
    return out


def build_logm():
    tol, fn = C.MATFUN_TOL["logm"], O.MATRIX_ORACLE["logm"]
    out = [_matfun_lit("logm/%s" % nm, _arr(a), None, fn(_arr(a)), tol) for nm, a in C.LOGM_CURATED.items()]
    r = C.LOGM_RANDOM
    rng = np.random.default_rng(r["seed"])
    for norm in r["norms"]:                     # A = expm(X): eigenvalues of X well inside |Im| < pi
        for n in r["sizes"]:
            a = O.MATRIX_ORACLE["expm"](_scaled_normal(rng, n, norm))
            out.append(_matfun_lit("logm/rnd_expm/norm=%g/%dx%d" % (norm, n, n), a, None, fn(a), tol))
    for n in r["spd_sizes"]:                    # symmetric positive definite, spectrum 1e-2 .. 1e2
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        a = (q * 10.0 ** rng.uniform(-2, 2, size=n)) @ q.T
        out.append(_matfun_lit("logm/rnd_spd/%dx%d" % (n, n), a, None, fn(a), tol))
    return out


def build_mpower():
    out = []
    curated = [("fibonacci", [[1, 1], [1, 0]], 10), ("upper", [[2, 1], [0, 3]], -3),
               ("nilpotent", [[0, 1], [0, 0]], 2), ("singular_p0", [[1, 2], [2, 4]], 0),
               ("rotation_quarter", [[0, -1], [1, 0]], 0.5), ("spd_sqrt", [[4, 1], [1, 3]], 0.5)]
    for nm, a, p in curated:
        tol = C.MATFUN_TOL["mpower_int" if p == int(p) else "mpower_frac"]
        out.append(_matfun_lit("mpower/%s" % nm, _arr(a), [p], O.mpower(_arr(a), p), tol))
    r = C.MPOWER_RANDOM
    rng = np.random.default_rng(r["seed"])
    for k, p in enumerate(C.MPOWER_P):          # A = I + B, Gershgorin: eigenvalues have Re >= 0.2
        n = r["sizes"][k % len(r["sizes"])]
        b = rng.standard_normal((n, n))
        a = np.eye(n) + b * (0.8 / np.abs(b).sum(axis=1).max())
        tol = C.MATFUN_TOL["mpower_int" if p == int(p) else "mpower_frac"]
        out.append(_matfun_lit("mpower/rnd/p=%.6g/%dx%d" % (p, n, n), a, [p], O.mpower(a, p), tol))
    return out


def build_mpower_base():
    out, tol = [], C.MATFUN_TOL["mpower_base"]
    rng = np.random.default_rng(C.MPOWER_RANDOM["seed"] + 1)
    for k, s in enumerate(C.MPOWER_BASES):
        n = C.MPOWER_RANDOM["sizes"][k % len(C.MPOWER_RANDOM["sizes"])]
        a = _scaled_normal(rng, n, 1.5)
        out.append(_matfun_lit("mpower_base/s=%.6g/%dx%d" % (s, n, n), [s], a, O.mpower_base(s, a), tol))
    return out


def main_matop():
    with np.errstate(all="ignore"):
        arrays = [emit_array("%s_golden" % t, build_matop_unary(t, s)) for t, s in C.MATOP_CASES.items()]
        for t, pts in C.MATOP_INT_POINTS.items():
            sp = C.MATOP_CASES[t]
            arrays.append(emit_array("%s_golden_int" % t, [_lit1("%s/int" % t, pts, O.MATOP_ORACLE[t](_arr(pts)),
                                                                 sp["rtol"], sp["atol"])]))
        arrays.append(emit_array("round_n_golden", build_round_n()))
        arrays.append(emit_array("round_n_golden_int", build_round_n_int()))
        for t, pts in C.MATOP_INT_CASES.items():
            arrays.append(emit_array("%s_golden_int" % t,
                                     [_lit1("%s/int" % t, pts, O.MATOP_INT_ORACLE[t](_arr(pts)), 0.0, 0.0)]))
        fl = C.FACTORIAL_LARGE
        arrays.append(emit_array("factorial_golden", [_lit1("factorial/large", fl["points"],
                                                            O.MATOP_INT_ORACLE["factorial"](_arr(fl["points"])),
                                                            fl["rtol"], fl["atol"])]))
        binary = dict(build_matop_binary(C.MATOP_BINARY_CASES["gcd"], ["gcd"]))
        binary.update(build_matop_binary(C.MATOP_BINARY_CASES["numden"], ["numden_num", "numden_den"]))
        arrays += [emit_array("%s_golden" % t, cs) for t, cs in binary.items()]
        arrays.append(emit_array("nchoosek_golden", build_nchoosek()))
        arrays.append(emit_array("expm_golden", build_expm()))
        arrays.append(emit_array("logm_golden", build_logm()))
        arrays.append(emit_array("mpower_golden", build_mpower()))
        arrays.append(emit_array("mpower_base_golden", build_mpower_base()))
    write_header(OUT_MATOP, "MATSAR_GOLDEN_MATOP_HPP", arrays)


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
    main_matop()


if __name__ == "__main__":
    main()
