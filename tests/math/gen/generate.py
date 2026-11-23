"""Generate tests/math/golden/golden_mathfunc.hpp from cases.py via oracle.py (deterministic)."""
import os
import zlib
import numpy as np
import mpmath as mp
from fractions import Fraction
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
OUT_PROB = os.path.join(HERE, "..", "golden", "golden_prob.hpp")
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


# ---- Probability --------------------------------------------------------------------
def _reduce_axes(shape):
    return [None] + list(range(len(shape)))


def build_prob_reduce(tag, integer=False):
    fn = O.PROB_REDUCE_ORACLE[tag]
    rtol, atol = C.PROB_REDUCE_TOL[tag]
    out = []
    if not integer and tag in ("max", "min"):
        for nm, rows, axis in C.PROB_REDUCE_CURATED:
            a = _arr(rows)
            out.append(case_lit("%s/%s" % (tag, nm), a, axis, fn(a, axis), rtol, atol, check53=False))
    r = C.PROB_REDUCE_RANDOM
    rng = np.random.default_rng(tag_seed("prob/" + tag + ("/int" if integer else "")))
    for i in range(r["count"]):
        shape = C.PROB_REDUCE_SHAPES[i % len(C.PROB_REDUCE_SHAPES)]
        if integer:
            a = rng.integers(r["lo"], r["hi"] + 1, size=shape).astype(np.float64)
        else:
            a = rng.uniform(r["lo"], r["hi"], size=shape)
        for axis in _reduce_axes(shape):
            out.append(case_lit("%s/rnd%03d/%s/ax%s" % (tag, i, _shape_tag(shape), "g" if axis is None else axis),
                                a, axis, fn(a, axis), rtol, atol, check53=False))
    return out


def _pdf_random_x(tag, params, rng, n, span):
    if tag == "normpdf":
        mu, s = params
        return mu + s * rng.uniform(-span, span, size=n)
    if tag == "binopdf":
        return rng.integers(0, int(params[0]) + 1, size=n).astype(np.float64)
    if tag == "poisspdf":
        return rng.integers(0, int(3 * params[0]) + 6, size=n).astype(np.float64)
    if tag == "chi2pdf":
        return rng.uniform(0, params[0] * span + 5, size=n)
    if tag in ("exppdf", "raylpdf"):
        return rng.uniform(0, params[0] * span, size=n)
    if tag == "weibpdf":
        return rng.uniform(0, params[0] * span, size=n)
    if tag == "unifpdf":
        a, b = params
        return rng.uniform(a - span * (b - a), b + span * (b - a), size=n)
    raise KeyError(tag)


def _pdf_rtol(spec, out):
    """A density is exp(y): y's rounding error |y| * eps becomes relative error in exp(y), so the
    tolerance grows with |ln pdf| (deep tails, e.g. e^-500, are only good to ~500 eps)."""
    if spec["rtol"] == 0:
        return 0.0
    pos = out[np.isfinite(out) & (out > 0)]
    return spec["rtol"] + C.PROB_PDF_LOG_RTOL * (float(np.max(np.abs(np.log(pos)))) if pos.size else 0.0)


def build_prob_pdf(tag, spec):
    fn = O.PROB_PDF_ORACLE[tag]
    rng = np.random.default_rng(tag_seed("prob/" + tag))
    rr = spec["random"]
    out = []
    for k, (params, pts) in enumerate(spec["cases"]):
        pts = list(pts) + (C._X_EDGE if k == 0 else [])
        x = _arr(pts)
        pname = ",".join("%g" % v for v in params)
        y = fn(x, params)
        out.append(_lit2("%s/p=%s/points" % (tag, pname), x, list(params), y, _pdf_rtol(spec, y), spec["atol"]))
        if np.isnan(fn(_arr([0.5]), params)).all():   # invalid parameters: NaN everywhere
            continue
        xs = _pdf_random_x(tag, params, rng, C.PROB_PDF_RANDOM["count"], rr.get("span", 1))
        shape = (2, 3) if xs.size == 6 else xs.shape
        xs = xs.reshape(shape)
        y = fn(xs, params)
        out.append(_lit2("%s/p=%s/rnd" % (tag, pname), xs, list(params), y, _pdf_rtol(spec, y), spec["atol"]))
    return out


def main_prob():
    with np.errstate(all="ignore"):
        arrays = [emit_array("%s_golden" % t, build_prob_reduce(t)) for t in O.PROB_REDUCE_ORACLE]
        arrays += [emit_array("%s_golden_int" % t, build_prob_reduce(t, integer=True)) for t in ("max", "min", "median")]
        arrays += [emit_array("%s_golden" % t, build_prob_pdf(t, sp)) for t, sp in C.PROB_PDF_CASES.items()]
    write_header(OUT_PROB, "MATSAR_GOLDEN_PROB_HPP", arrays)


# ---- LinearAlgebra ------------------------------------------------------------------
OUT_LINALG = os.path.join(HERE, "..", "golden", "golden_linalg.hpp")
OUT_LAPACK = os.path.join(HERE, "..", "golden", "golden_lapack.hpp")
EPS = 2.0 ** -52


def case_lit_axis2(name, a, b, axis, expected, rtol, atol):
    """Two inputs and an integer side channel in `axis` (ppval: the number of breaks)."""
    return ("{\"%s\",%s,%s,%s,%s,%d,%s,%s,%s,%s}"
            % (name, flat_lit(a, False), shape_lit(a.shape), flat_lit(b, False), shape_lit(b.shape), int(axis),
               flat_lit(expected, False), shape_lit(expected.shape), fmt(rtol), fmt(atol)))


def build_linalg_mean(tag):
    fn, tol = O.LINALG_MEAN_ORACLE[tag], C.LINALG_MEAN_TOL
    out = []
    for nm, rows, axis in C.LINALG_MEAN_CURATED[tag]:
        a = _arr(rows)
        out.append(case_lit("%s/%s" % (tag, nm), a, axis, fn(a, axis), tol["rtol"], tol["atol"], check53=False))
    r = C.LINALG_MEAN_RANDOM
    rng = np.random.default_rng(tag_seed("linalg/" + tag))
    for i in range(r["count"]):
        shape = C.LINALG_MEAN_SHAPES[i % len(C.LINALG_MEAN_SHAPES)]
        a = 10.0 ** rng.uniform(r["lo"], r["hi"], size=shape)
        for axis in _reduce_axes(shape):
            out.append(case_lit("%s/rnd%03d/%s/ax%s" % (tag, i, _shape_tag(shape), "g" if axis is None else axis),
                                a, axis, fn(a, axis), tol["rtol"], tol["atol"], check53=False))
    return out


def build_kron():
    r = C.LINALG_KRON_RANDOM
    rng = np.random.default_rng(r["seed"])
    out = []
    for sa, sb in C.LINALG_KRON_SHAPES:
        a = rng.integers(r["lo"], r["hi"] + 1, size=sa).astype(np.float64)
        b = rng.integers(r["lo"], r["hi"] + 1, size=sb).astype(np.float64)
        out.append(_lit2("kron/%s" % _shape_tag(sa, sb), a, b, np.kron(a, b), 0.0, 0.0))
    return out


def _random_poly(rng, r):
    return [float(v) for v in rng.integers(r["lo"], r["hi"] + 1, size=int(rng.integers(1, r["max_degree"] + 2)))]


def build_polyder():
    out = [_lit1("polyder/cur%d" % i, p, O.polyder(p), 0.0, 0.0) for i, p in enumerate(C.POLYDER_POINTS)]
    r = C.POLY_RANDOM
    rng = np.random.default_rng(r["seed"])
    for i in range(r["count"]):
        p = _random_poly(rng, r)
        out.append(_lit1("polyder/rnd%03d" % i, p, O.polyder(p), 0.0, 0.0))
    return out


def build_polyder2():
    out = [_lit2("polyder2/cur%d" % i, a, b, O.polyder2(a, b), 0.0, 0.0) for i, (a, b) in enumerate(C.POLYDER2_POINTS)]
    r = C.POLY_RANDOM
    rng = np.random.default_rng(r["seed"] + 1)
    for i in range(r["count"]):
        a, b = _random_poly(rng, r), _random_poly(rng, r)
        out.append(_lit2("polyder2/rnd%03d" % i, a, b, O.polyder2(a, b), 0.0, 0.0))
    return out


def build_polyeval():
    out = [_lit2("polyeval/cur%d" % i, c, x, O.polyeval(c, x), 0.0, 0.0) for i, (c, x) in enumerate(C.POLYEVAL_POINTS)]
    r = C.POLYEVAL_RANDOM
    rng = np.random.default_rng(r["seed"])
    for i in range(r["count"]):
        c = list(rng.uniform(0, 1, size=int(rng.integers(1, r["max_degree"] + 2))))
        x = list(rng.uniform(0, 3, size=5))
        out.append(_lit2("polyeval/rnd%03d" % i, c, x, O.polyeval(c, x), 1e-14, 0.0))
    return out


def _ppval_lit(name, breaks, coefs, xs):
    want = O.ppval(breaks, coefs, xs)
    bound = 0.0   # rounding scale: sum |c_i| |t|^i at the worst query point
    for x in xs:
        if not np.isnan(x):
            for i, b in enumerate(breaks[:-1]):
                t = abs(float(x) - float(b))
                bound = max(bound, sum(abs(c) * t ** (len(coefs[i]) - 1 - k) for k, c in enumerate(coefs[i])))
    return case_lit_axis2(name, _arr(coefs), _arr(list(breaks) + list(xs)), len(breaks), _arr(want),
                          1e-14, 1e-14 * bound)


def build_ppval():
    out = [_ppval_lit("ppval/%s" % nm, *spec) for nm, spec in C.PPVAL_CASES.items()]
    r = C.PPVAL_RANDOM
    rng = np.random.default_rng(r["seed"])
    for i in range(r["count"]):
        breaks = sorted(set(np.round(rng.uniform(-5, 5, size=r["pieces"] + 1), 3).tolist()))
        coefs = rng.uniform(-2, 2, size=(len(breaks) - 1, r["order"])).tolist()
        xs = sorted(rng.uniform(breaks[0] - 1, breaks[-1] + 1, size=7).tolist()) + breaks[1:-1]
        out.append(_ppval_lit("ppval/rnd%03d" % i, breaks, coefs, xs))
    return out


def build_expint_real():
    out = [_lit1("expint/points", C.EXPINT_REAL_POINTS, O.expint_real(C.EXPINT_REAL_POINTS), C.EXPINT_TOL, 0.0)]
    r = C.EXPINT_REAL_RANDOM
    rng = np.random.default_rng(r["seed"])
    xs = (10.0 ** rng.uniform(r["lo"], r["hi"], size=r["count"])).reshape(3, 4)
    out.append(_lit1("expint/rnd", xs, np.asarray(O.expint_real(xs.ravel())).reshape(xs.shape), C.EXPINT_TOL, 0.0))
    return out


def build_expint_complex():
    r = C.EXPINT_COMPLEX_RANDOM
    rng = np.random.default_rng(r["seed"])
    rho = r["radius"] * np.sqrt(rng.uniform(0, 1, size=r["count"]))
    theta = rng.uniform(-np.pi, np.pi, size=r["count"])
    pts = list(C.EXPINT_COMPLEX_POINTS) + list(zip(rho * np.cos(theta), rho * np.sin(theta)))
    out = []
    for re, im in pts:
        wr, wi = O.expint_complex(re, im)
        atol = C.EXPINT_TOL * float(np.hypot(wr, wi))
        out.append(_lit2("expint_c/z=%.17g%+.17gi" % (re, im), [re], [im], [wr, wi], 0.0, atol))
    return out


def _spdiags_b(b):
    return b if b.size else np.zeros((0,))


def build_spdiags():
    mats = [(nm, _arr(a)) for nm, a in C.SPDIAGS_CURATED.items()]
    r = C.SPDIAGS_RANDOM
    rng = np.random.default_rng(r["seed"])
    for shape in r["shapes"]:
        a = rng.integers(1, 10, size=shape).astype(np.float64) * (rng.uniform(size=shape) < r["density"])
        mats.append(("rnd/%s" % _shape_tag(shape), a))
    b_cases, d_cases = [], []
    for nm, a in mats:
        b, d = O.spdiags_all(a)
        b_cases.append(_lit1("spdiags/%s" % nm, a, _spdiags_b(b), 0.0, 0.0))
        d_cases.append(_lit1("spdiags/%s" % nm, a, d, 0.0, 0.0))
    ext = [_lit2("spdiags_ext/%s" % nm, _arr(C.SPDIAGS_CURATED[nm]), ds,
                 _spdiags_b(O.spdiags_extract(C.SPDIAGS_CURATED[nm], ds)), 0.0, 0.0)
           for nm, ds in C.SPDIAGS_EXTRACT.items()]
    return b_cases, d_cases, ext


def main_linalg():
    with np.errstate(all="ignore"):
        arrays = [emit_array("%s_golden" % t, build_linalg_mean(t)) for t in O.LINALG_MEAN_ORACLE]
        arrays.append(emit_array("kron_golden", build_kron()))
        arrays.append(emit_array("polyder_golden", build_polyder()))
        arrays.append(emit_array("polyder2_golden", build_polyder2()))
        arrays.append(emit_array("polyeval_golden", build_polyeval()))
        arrays.append(emit_array("ppval_golden", build_ppval()))
        arrays.append(emit_array("expint_golden", build_expint_real()))
        arrays.append(emit_array("expint_golden_c", build_expint_complex()))
        b, d, ext = build_spdiags()
        arrays += [emit_array("spdiags_B_golden", b), emit_array("spdiags_d_golden", d),
                   emit_array("spdiags_ext_golden", ext)]
    write_header(OUT_LINALG, "MATSAR_GOLDEN_LINALG_HPP", arrays)


# LAPACK-backed: tolerance from the condition number (forward error ~ cond * eps).
def _cond(A, cplx):
    s = O.svd_values(A, cplx)
    nz = [v for v in s if v > mp.mpf(10) ** -30 * s[0]] if s and s[0] != 0 else []
    return float(nz[0] / nz[-1]) if nz else 1.0


def _lapack_tol(cond):
    return max(C.LAPACK_TOL_FLOOR, C.LAPACK_TOL_COND * EPS * cond)


def _rand_matrix(rng, shape, cplx):
    """A random normal matrix with cond < LAPACK_COND_MAX (redrawn otherwise)."""
    while True:
        re = rng.standard_normal(shape)
        im = rng.standard_normal(shape) if cplx else None
        if _cond(O.mp_matrix(re, im), cplx) < C.LAPACK_COND_MAX:
            return re, im


def _lit_c(name, re, im, out, rtol, atol):
    """Complex input: real parts in `in`, imaginary parts in `in2`."""
    return _lit2(name, re, im, out, rtol, atol)


def _cplx_out(re, im):
    """Complex values as real rows: a vector of n -> (2, n) [re; im], an m x n matrix -> (2m, n)."""
    return np.vstack([np.atleast_2d(np.asarray(re, dtype=np.float64)), np.atleast_2d(np.asarray(im, dtype=np.float64))])


def _lapack_inputs(seed_tag, square=True, rect=True, curated=True, cplx=False):
    """(name, re, im) inputs shared by the LAPACK builders."""
    rng = np.random.default_rng(tag_seed("lapack/" + seed_tag + ("/c" if cplx else "")))
    shapes = ([(n, n) for n in C.LAPACK_SQUARE_SIZES] if square else []) + (C.LAPACK_RECT_SHAPES if rect else [])
    out = []
    if curated and not cplx:
        out += [(nm, _arr(a), None) for nm, a in C.LAPACK_CURATED_SQUARE.items()]
    for shape in shapes:
        re, im = _rand_matrix(rng, shape, cplx)
        out.append(("rnd/%s" % _shape_tag(shape), re, im))
    return out


def build_svd(cplx):
    out = []
    for nm, re, im in _lapack_inputs("svd", cplx=cplx):
        s = O.svd_values(O.mp_matrix(re, im), cplx)
        want = _arr([float(v) for v in s])
        tol = C.LAPACK_TOL_FLOOR
        atol = tol * float(want.max(initial=0))
        out.append(_lit_c("svd_c/%s" % nm, re, im, want, tol, atol) if cplx else _lit1("svd/%s" % nm, re, want, tol, atol))
    return out


def _exact_rank(a):
    m = [[Fraction(int(v)) for v in row] for row in a]
    rank, rows, cols = 0, len(m), len(m[0])
    for c in range(cols):
        p = next((r for r in range(rank, rows) if m[r][c] != 0), None)
        if p is None:
            continue
        m[rank], m[p] = m[p], m[rank]
        for r in range(rows):
            if r != rank and m[r][c] != 0:
                f = m[r][c] / m[rank][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[rank])]
        rank += 1
    return rank


def build_rank():
    rng = np.random.default_rng(C.LAPACK_RANDOM["seed"])
    out = []
    for (m, n), r in C.RANK_CASES:
        while True:
            a = (rng.integers(-3, 4, size=(m, r)) @ rng.integers(-3, 4, size=(r, n))).astype(np.float64) \
                if r else np.zeros((m, n))
            if _exact_rank(a) == r:
                break
        out.append(_lit1("rank/%s/r=%d" % (_shape_tag((m, n)), r), a, [r], 0.0, 0.0))
    tol_cases = [_lit2("rank_tol/%d" % i, _arr(a), [t], [r], 0.0, 0.0) for i, (a, t, r) in enumerate(C.RANK_TOL_CASES)]
    return out, tol_cases


def build_det(cplx):
    out = []
    for nm, re, im in _lapack_inputs("det", rect=False, cplx=cplx):
        A = O.mp_matrix(re, im)
        d = mp.det(A)
        tol = _lapack_tol(_cond(A, cplx) * A.rows)
        if cplx:
            out.append(_lit_c("det_c/%s" % nm, re, im, [float(mp.re(d)), float(mp.im(d))], 0.0, tol * float(abs(d))))
        else:
            out.append(_lit1("det/%s" % nm, re, [float(d)], tol, 0.0))
    return out


def build_inv(cplx):
    out = []
    for nm, re, im in _lapack_inputs("inv", rect=False, cplx=cplx):
        A = O.mp_matrix(re, im)
        if mp.det(A) == 0:
            continue
        X = mp.inverse(A)
        tol = _lapack_tol(_cond(A, cplx))
        if cplx:
            xr, xi = O.mp_to_np(X, cplx=True)
            want = _cplx_out(xr, xi)
            out.append(_lit_c("inv_c/%s" % nm, re, im, want, 0.0, tol * float(np.abs(xr + 1j * xi).max())))
        else:
            want = O.mp_to_np(X)
            out.append(_lit1("inv/%s" % nm, re, want, tol, tol * float(np.abs(want).max())))
    return out


def build_pinv(cplx):
    rng = np.random.default_rng(tag_seed("lapack/pinv/rankdef"))
    inputs = _lapack_inputs("pinv", curated=False, cplx=cplx)
    if not cplx:
        for (m, n), r in [((3, 3), 2), ((4, 6), 2), ((6, 4), 3), ((2, 2), 1)]:
            a = (rng.integers(-3, 4, size=(m, r)) @ rng.integers(-3, 4, size=(r, n))).astype(np.float64)
            assert _exact_rank(a) == r
            inputs.append(("rankdef/%s/r=%d" % (_shape_tag((m, n)), r), a, None))
        inputs += [("singular", _arr([[1, 2], [2, 4]]), None), ("vector", _arr([1, 2, 3]), None)]
    out = []
    for nm, re, im in inputs:
        A = O.mp_matrix(re.reshape(-1, 1) if re.ndim == 1 else re, None if im is None else im)
        X = O.pinv(A, cplx)
        tol = _lapack_tol(_cond(A, cplx))
        if cplx:
            xr, xi = O.mp_to_np(X, cplx=True)
            out.append(_lit_c("pinv_c/%s" % nm, re, im, _cplx_out(xr, xi), 0.0, tol * float(np.abs(xr + 1j * xi).max())))
        else:
            want = O.mp_to_np(X)
            out.append(_lit1("pinv/%s" % nm, re, want, tol, tol * float(np.abs(want).max())))
    return out


def build_lu():
    """lu (MATLAB [L, U] = lu(A): L = P' * L) and lup ([L, U, P] = lu(A): P * A = L * U)."""
    inputs = _lapack_inputs("lu") + [("tall_tie", _arr([[1, 2], [3, 4], [5, 6], [7, 9]]), None),
                                     ("wide", _arr([[1, 2, 3, 4], [5, 6, 7, 9]]), None),
                                     ("column", _arr([[1], [-3], [2]]), None), ("row", _arr([[2, -1, 4]]), None)]
    arrs = {k: [] for k in ("lu_L", "lu_U", "lup_L", "lup_U", "lup_P")}
    for nm, re, _ in inputs:
        A = O.mp_matrix(re)
        L, U, perm, gap = O.lu_partial_pivot(A)
        if gap < 1e-6:            # two near-equal pivot candidates: rounding could pick either
            continue
        P = np.zeros((A.rows, A.rows))
        for i, p in enumerate(perm):
            P[i, p] = 1
        Ln, Un = O.mp_to_np(L), O.mp_to_np(U)
        tol = _lapack_tol(_cond(A, False))
        lit = lambda tag, w: _lit1("%s/%s" % (tag, nm), re, w, tol, tol * float(np.abs(w).max(initial=1)))
        arrs["lu_L"].append(lit("lu_L", P.T @ Ln))
        arrs["lu_U"].append(lit("lu_U", Un))
        arrs["lup_L"].append(lit("lup_L", Ln))
        arrs["lup_U"].append(lit("lup_U", Un))
        arrs["lup_P"].append(_lit1("lup_P/%s" % nm, re, P, 0.0, 0.0))
    return arrs


def _eig_tol(A):
    """eigenvalue error ~ eps * |A|_2 * condeig: normwise atol from the worst condition number."""
    smax = float(O.svd_values(A, any(isinstance(x, mp.mpc) for x in A))[0])
    worst = max(float(s) for _, s in O.condeig(A))
    return max(C.LAPACK_TOL_FLOOR, C.LAPACK_TOL_COND * EPS * worst) * max(smax, 1e-300)


def _eig_separated(vals):
    """Sorting by (re, im) must not depend on rounding: distinct real parts differ clearly
    (a conjugate pair shares its real part exactly and is ordered by the imaginary part)."""
    for a, b in zip(vals, vals[1:]):
        dr = abs(float(mp.re(b) - mp.re(a)))
        pair = abs(float(mp.re(b) - mp.re(a))) < 1e-30 and abs(float(mp.im(a) + mp.im(b))) < 1e-30
        if dr < 1e-6 and not pair:
            return False
    return True


def build_eig(cplx):
    rng = np.random.default_rng(tag_seed("lapack/eig/sym"))
    inputs = _lapack_inputs("eig", rect=False, curated=False, cplx=cplx)
    if not cplx:
        inputs = [(nm, _arr(a), None) for nm, a in C.EIG_CURATED.items()] + inputs
        for n in (3, 5):
            b = rng.standard_normal((n, n))
            inputs.append(("sym/%dx%d" % (n, n), b + b.T, None))
    out = []
    for nm, re, im in inputs:
        A = O.mp_matrix(re, im)
        vals = O.eig_values(A)
        if not _eig_separated(vals):
            continue
        want = _cplx_out([float(mp.re(v)) for v in vals], [float(mp.im(v)) for v in vals])
        atol = _eig_tol(A)
        out.append(_lit_c("eig_c/%s" % nm, re, im, want, 0.0, atol) if cplx else _lit1("eig/%s" % nm, re, want, 0.0, atol))
    return out


def build_condeig():
    out = []
    inputs = [(nm, _arr(a), None) for nm, a in C.EIG_CURATED.items() if nm != "symmetric"] + \
        _lapack_inputs("condeig", rect=False, curated=False)
    for nm, re, _ in inputs:
        A = O.mp_matrix(re)
        pairs = O.condeig(A)
        if not _eig_separated([lam for lam, _ in pairs]) or A.rows == 1:
            continue
        out.append(_lit1("condeig/%s" % nm, re, [float(s) for _, s in pairs], C.CONDEIG_RTOL, 0.0))
    return out


def build_polyvalm():
    return [_lit2("polyvalm/cur%d" % i, p, x, O.polyvalm(p, x), 0.0, 0.0) for i, (p, x) in enumerate(C.POLYVALM_CASES)]


def main_lapack():
    with np.errstate(all="ignore"):
        arrays = [emit_array("svd_golden", build_svd(False)), emit_array("svd_golden_c", build_svd(True))]
        rank, rank_tol = build_rank()
        arrays += [emit_array("rank_golden", rank), emit_array("rank_tol_golden", rank_tol)]
        arrays += [emit_array("det_golden", build_det(False)), emit_array("det_golden_c", build_det(True))]
        arrays += [emit_array("inv_golden", build_inv(False)), emit_array("inv_golden_c", build_inv(True))]
        arrays += [emit_array("pinv_golden", build_pinv(False)), emit_array("pinv_golden_c", build_pinv(True))]
        arrays += [emit_array("%s_golden" % k, v) for k, v in build_lu().items()]
        arrays += [emit_array("eig_golden", build_eig(False)), emit_array("eig_golden_c", build_eig(True))]
        arrays.append(emit_array("condeig_golden", build_condeig()))
        arrays.append(emit_array("polyvalm_golden", build_polyvalm()))
    write_header(OUT_LAPACK, "MATSAR_GOLDEN_LAPACK_HPP", arrays)


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
    main_prob()
    main_linalg()
    main_lapack()


if __name__ == "__main__":
    main()
