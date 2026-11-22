# MatSAR math-module test harness

NumPy-golden + property-based verification for the experimental `math/` modules, piloted on
`MathematicalFunction` (`sum` / `prod` / `cumsum` / `cumprod`).
Design spec: `docs/superpowers/specs/2026-06-05-math-test-harness-design.md`.

## Step-by-step: "I wrote a function — how do I test it?"

**First decide:** is the function already covered? `sum` / `prod` / `cumsum` / `cumprod` are.
- **You edited an existing one** (bugfix / refactor): skip to **Step 5 (build & run)**. The existing golden
  cases + properties already check it.
- **It's a brand-new function** (e.g. `mean`, `var`, `median`): do all steps below. The worked example adds `mean`.

**Step 1 — tell the harness what "correct" is (the NumPy oracle).**
In `gen/oracle.py`, add your function's NumPy equivalent. Match MatSAR's shape convention
(`keepdims=True` for an axis; shape `(1,)` for global — same as `sum`):
```python
ORACLE = {
    ...
    "mean": lambda a, axis: (_wrap1(a.mean()) if axis is None else a.mean(axis=axis, keepdims=True)),
}
```

**Step 2 — list the inputs to test.** In `gen/cases.py`, add a block (curated edge cases + a seeded random batch):
```python
"mean": {
    "curated": [dict(shape=(2,3), axis=0), dict(shape=(2,3), axis=1), dict(shape=(4,), axis=AXIS_NONE)],
    "random":  dict(count=30, shapes=[(4,),(2,3),(5,4)], axes=[0,1,AXIS_NONE], lo=-10.0, hi=10.0, seed=2001),
    "rtol": 1e-12, "atol": 1e-13,
},
```

**Step 3 — generate the golden answers** (NumPy computes them; you never type expected values):
```bash
python tests/math/gen/generate.py
```
This writes a `mean_golden` array into `golden/golden_mathfunc.hpp`.

**Step 4 — wire the check** (one ~10-line block in `test_mathfunc.cpp`):
```cpp
TEST_CASE("golden · mean") { REPLAY_REDUCE(mean, mean_golden) }
```
`REPLAY_REDUCE` already handles axis-vs-global. For a different signature (2 inputs, no axis, …), copy the
macro body and adjust the `Math::...` call.

**Step 5 — build & run** (from the repo root):
```bash
export PATH=/d/msys64/mingw64/bin:$PATH            # MinGW; omit on WSL
g++ -std=c++17 -D_GNU_SOURCE -DMATSAR_ENABLE_EXPERIMENTAL -O0 \
    -I include -I tests -I tests/math/include -I tests/math/golden \
    -o /tmp/tmf tests/math/test_mathfunc.cpp && /tmp/tmf
```

**Step 6 — read the result:**
- `Status: SUCCESS!` → your function matches NumPy on every case and the invariants hold. Done.
- `FAIL  golden · mean   case: mean/cur1/2x3/ax1   i=1   got=… want=…` → your output differs from NumPy at
  that exact input and index. Either a **bug** (fix the function, rerun) or an **intended change** (then update
  `cases.py`/`oracle.py`, regenerate, and commit the new golden header).

**Step 7 — commit.** If you touched `cases.py`, regenerate and commit `golden/golden_mathfunc.hpp` too — CI's
`golden-regen-guard` fails otherwise.

## Layout

| Path | What |
|---|---|
| `include/approx.hpp` | `approx_eq(got, want, rtol, atol)` — rel+abs, NaN/Inf-aware compare |
| `include/golden.hpp` | `GoldenCase`, `build<T>()`, `check_matrix()`, `brute_sum/brute_prod`, `random_matrix` |
| `include/rng.hpp` | seeded `Rng` + `random_shape` |
| `gen/cases.py` | **the case spec you edit** (curated + seeded-random per function) |
| `gen/oracle.py` | tag → the NumPy reference call |
| `gen/generate.py` | emits the golden header (deterministic) |
| `gen/requirements.txt` | pinned `numpy==2.1.1` + `mpmath==1.3.0` (Trigonometry / MatrixOperation references: correctly rounded, platform-independent) |
| `golden/golden_mathfunc.hpp` | **GENERATED + committed** golden vectors — do not hand-edit |
| `golden/golden_trig.hpp` | **GENERATED + committed** Trigonometry goldens (domain-aware inputs, int64 subset, binary/broadcast) |
| `golden/golden_matop.hpp` | **GENERATED + committed** MatrixOperation goldens: exact Python references (rounding, gcd, numden, factorial, nchoosek), mpmath for exp/log/gamma and for `expm` / `logm` / `mpower` |
| `test_mathfunc.cpp` | doctest: Layer 1 (golden replay) + Layer 2 (properties) |
| `test_trig.cpp` | doctest: golden replay for every `Trigonometry.hpp` function (`test_mathtrig`) |
| `test_matop.cpp` | doctest: golden replay + hand-derived cases for `MatrixOperation.hpp` (`test_mathmatop`) |
| `death_matop.cpp` | MatrixOperation error cases, one per `argv[1]` (CTest `death_matop_<case>` matches the ValueError text) |
| `selftest_helpers.cpp` | self-test for the harness's own helpers |

## Run it

**Via CMake/CTest** (opt-in; the real, CI-covered path):
```bash
cmake -G Ninja -B build-exp -DMATSAR_BUILD_EXPERIMENTAL_TESTS=ON -DMATSAR_WITH_BLAS=OFF
cmake --build build-exp
ctest --test-dir build-exp --output-on-failure
```

**Fast standalone loop** (no CMake), from the repo root:
```bash
export PATH=/d/msys64/mingw64/bin:$PATH            # MinGW; omit on Linux/WSL
g++ -std=c++17 -D_GNU_SOURCE -DMATSAR_ENABLE_EXPERIMENTAL -O0 \
    -I include -I tests -I tests/math/include -I tests/math/golden \
    -o /tmp/tmf tests/math/test_mathfunc.cpp && /tmp/tmf
```
Filter inside the doctest binary: `/tmp/tmf -tc="golden · sum"`; list cases: `/tmp/tmf --list-test-cases`.

## The daily dev loop

**Developing / fixing a function.** Edit the library (`include/MatSAR/math/...`) → rebuild → rerun.
Layer 1 tells you if outputs match NumPy; Layer 2 catches invariant breaks. A failure prints the
case name (e.g. `sum/cur0/2x3/ax0`) and the `i,j` index.

**Adding coverage** (no C++ change). Edit `gen/cases.py` — add a curated `dict(shape=…, axis=…)` or bump a
`random` `count` — then regenerate; NumPy recomputes the expected values:
```bash
python tests/math/gen/generate.py
```
Then rebuild + rerun.

**Intentional behavior change.** If you *mean* to change a function's output, update `cases.py`/`oracle.py`,
regenerate (the golden truth updates from NumPy), and commit the regenerated header.

## Add a new function (same module)
1. `gen/oracle.py`: add `"myfn": lambda a, axis: <numpy call>`.
2. `gen/cases.py`: add a `"myfn"` entry (curated + random; set `rtol`/`atol`).
3. `python tests/math/gen/generate.py` → a new `myfn_golden` array appears in the header.
4. `test_mathfunc.cpp`: add a ~10-line replay block (`REPLAY_REDUCE(myfn, myfn_golden)` or a hand loop),
   plus any invariant in the property `TEST_CASE`.

## Add a new module (e.g. Trigonometry)
Repeat the pilot's five pieces: an `oracle.py` entry, a `cases.py` entry, a `golden_<module>.hpp`
(extend `generate.py`), a `test_<module>.cpp`, and a `test_<module>` target in `tests/CMakeLists.txt`.
Round-trip properties (e.g. `asin(sin x) ≈ x`) are natural Layer-2 checks for elementwise functions.

## Regenerating & the CI guard
The golden header is committed. **If you edit `cases.py` you must regenerate and commit the header** —
CI's `golden-regen-guard` job fails when `generate.py`'s output differs from what's committed
(byte-stable, pinned numpy). Python is needed only to regenerate, never to build or run the tests.

## Numerical notes
Tolerances live per-function in `cases.py` (`rtol`/`atol`); integer cases use exact `0/0`. `approx_eq`
treats `NaN` as matching `NaN` (domain-error cases) and requires sign-correct `Inf`. Global `cumsum`/`cumprod`
are 1-D only in the pilot (MatSAR's global cumulative scan is column-major storage order).
