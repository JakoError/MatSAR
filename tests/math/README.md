# MatSAR math-module test harness

NumPy-golden + property-based verification for the experimental `math/` modules, piloted on
`MathematicalFunction` (`sum` / `prod` / `cumsum` / `cumprod`).
Design spec: `docs/superpowers/specs/2026-06-05-math-test-harness-design.md`.

## Layout

| Path | What |
|---|---|
| `include/approx.hpp` | `approx_eq(got, want, rtol, atol)` — rel+abs, NaN/Inf-aware compare |
| `include/golden.hpp` | `GoldenCase`, `build<T>()`, `check_matrix()`, `brute_sum/brute_prod`, `random_matrix` |
| `include/rng.hpp` | seeded `Rng` + `random_shape` |
| `gen/cases.py` | **the case spec you edit** (curated + seeded-random per function) |
| `gen/oracle.py` | tag → the NumPy reference call |
| `gen/generate.py` | emits the golden header (deterministic) |
| `gen/requirements.txt` | pinned `numpy==2.1.1` |
| `golden/golden_mathfunc.hpp` | **GENERATED + committed** golden vectors — do not hand-edit |
| `test_mathfunc.cpp` | doctest: Layer 1 (golden replay) + Layer 2 (properties) |
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
