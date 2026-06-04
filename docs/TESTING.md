# Building & Testing MatSAR

MatSAR is a header-only C++17 library. Tests use [doctest] (vendored at
`tests/doctest.h`) and are run through **CTest**.

## TL;DR

From the repository root, with a compiler + CMake + a generator on PATH:

```bash
cmake -G Ninja -B build                          # 1. configure
cmake --build build                              # 2. compile the tests
ctest --test-dir build --output-on-failure       # 3. run the tests
```

Expected: `100% tests passed, 0 tests failed out of 10` (or `9` with BLAS off).

---

## Prerequisites by platform

You need a C++17 compiler, CMake ≥ 3.22, a build tool (Ninja or Make or MSBuild),
and — for the BLAS-backed matrix product (`mmul`/`%`) — OpenBLAS.

### Windows · MSYS2 / MinGW-w64

Open the **MinGW-w64 64-bit shell** (`mingw64.exe` in your MSYS2 install — the
default location is `C:\msys64`) and install once:

```bash
pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake \
                   mingw-w64-x86_64-ninja mingw-w64-x86_64-pkgconf \
                   mingw-w64-x86_64-openblas
```

> `openblas` + `pkgconf` are only needed for `mmul`/`%`. Without them the build
> falls back to the bundled `bin/libopenblas.dll`; or build with `-DMATSAR_WITH_BLAS=OFF`.

### Windows · MSVC (Visual Studio 2022/2026)

Use a **Developer PowerShell for VS** (so `cl`, `cmake`, `ctest` are on PATH), or
the CMake bundled with VS. OpenBLAS via [vcpkg]:

```powershell
vcpkg install openblas:x64-windows
```

### Linux / WSL (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y g++ cmake ninja-build pkg-config libopenblas-dev
```

---

## Build & test

### MinGW (Ninja)

```bash
cmake -G Ninja -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### MSVC (Visual Studio generator, multi-config)

```powershell
# BLAS via the bundled lib (no vcpkg needed):
cmake -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure

# …or OpenBLAS via vcpkg:
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_INSTALLATION_ROOT\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

### Linux / WSL (Ninja)

Use a build directory **separate** from any Windows one (see the note below):

```bash
cmake -G Ninja -B build-wsl
cmake --build build-wsl
ctest --test-dir build-wsl --output-on-failure
```

> **Sharing a checkout between Windows and WSL/MSVC/MinGW:** a build directory is
> **not** portable across them — the CMake cache stores absolute paths (`C:\…` vs
> `/mnt/c/…`) and a specific compiler. Reusing Windows' `build/` from WSL fails with
> *"The source … does not match the source … used to generate cache."* Use a distinct
> directory per environment: `build/` (MinGW), `build-wsl/` (WSL), `build-msvc/` (MSVC),
> etc. All `build*/` dirs are git-ignored.
> Tip: building under a Windows drive (e.g. `/mnt/c`) from WSL is slow (9p filesystem); for speed, build inside
> the WSL filesystem, e.g. `-B ~/matsar-build`.

---

## CMake options

| Option | Default | Effect |
|---|---|---|
| `MATSAR_WITH_BLAS` | `ON` | Enable OpenBLAS-backed `mmul()` / `operator%`. `OFF` builds the core with **no** OpenBLAS dependency (and skips `test_mmul`). |
| `MATSAR_ENABLE_EXPERIMENTAL` | `OFF` | Include unfinished modules (advanced LinearAlgebra, Probability, …). Not expected to build yet. |
| `MATSAR_LEGACY_COLUMN_INITIALIZER_LIST` | `OFF` | Temporary migration aid: makes `{{...}}` column-wise again. Prefer `fromRows`/`fromColumns`. |
| `MATSAR_BUILD_TESTS` | `ON` (top-level) | Build the test suite. |
| `MATSAR_BUILD_EXAMPLES` | `ON` (top-level) | Build the runnable programs in `examples/`. |

Pass options at configure time, e.g. core-only without BLAS:

```bash
cmake -G Ninja -B build-noblas -DMATSAR_WITH_BLAS=OFF
cmake --build build-noblas
ctest --test-dir build-noblas --output-on-failure
```

The configure step prints which OpenBLAS it found, e.g.
`-- MatSAR BLAS: pkg-config openblas (0.3.33)` (or `bundled bin/libopenblas.dll`).

---

## Running specific tests

```bash
ctest --test-dir build -N                        # list all tests
ctest --test-dir build -R test_arithmetic -V     # run one test, verbose
ctest --test-dir build -R "mmul|comparison"      # run tests matching a regex
```

Each test is also a standalone doctest executable with its own CLI:

```bash
./build/tests/test_arithmetic                    # run it directly
./build/tests/test_arithmetic --list-test-cases  # list cases in this binary
./build/tests/test_arithmetic -tc="operator**"   # run cases matching a pattern
./build/tests/test_arithmetic -s                 # also report successful asserts
```

On Windows the binaries are at `build\tests\<name>.exe` (MinGW/Ninja) or
`build\tests\Debug\<name>.exe` (MSVC). The OpenBLAS DLL is copied next to the
exe automatically for BLAS-enabled builds.

---

## What the tests cover

| Test | Checks |
|---|---|
| `test_smoke` | doctest/CTest harness works |
| `test_core` | container construction, indexing, `eye`/creation |
| `test_public_header` | `<MatSAR.hpp>` compiles (experimental quarantined) |
| `test_arithmetic` | element-wise `*` (NumPy-style) + scalar + broadcasting |
| `test_comparison` | `< <= > >= == !=` incl. the broadcast branch |
| `test_build_config` | core compiles/runs with `MATSAR_WITH_BLAS=OFF` |
| `death_invalid_shape` | incompatible shapes are reported (process exits) |
| `test_mmul` | OpenBLAS-backed `mmul()` / `%` (BLAS builds only) |
| `test_print` | `toString` / `toRowColString` output (name, dtype, shape, values) |
| `test_construct` | row-wise default `{{...}}` + `vector<vector>` ctor, `fromRows`/`fromColumns` (list & vector), 1-D, legacy flag |

> **Note on layout (row-wise construction):** nested initializer lists **and**
> nested `vector<vector<T>>` are **row-wise** by default — `{{1,2},{3,4}}` has rows
> `[1,2]`,`[3,4]`, so `m(0,1) == 2`. Build explicitly with `Matrix<T>::fromRows(...)`
> (same as the default) or `Matrix<T>::fromColumns(...)` (column-wise); both accept an
> initializer list or a `vector<vector<T>>`. `-DMATSAR_LEGACY_COLUMN_INITIALIZER_LIST=ON`
> is a temporary migration aid that reverts both default nested constructors to
> column-wise. Matrix product is `mmul()` / `operator%`; `*` is element-wise.
>
> **Storage boundary (still column-major):** only *construction*, `operator()`, and the
> *display* functions (`toString`/`toRowColString`) are row-wise. Internal storage stays
> column-major, so these remain **column-major / storage-order** and do **not** match the
> row-wise reading:
> - **iteration** (`begin()`/`end()`, range-`for`) and **`getData()`** yield column-major order;
> - **`reshape`** reinterprets the column-major buffer (not a NumPy row-major reshape);
> - broadcast alignment and `newAxis` (1-D → `{n,1}`) are column-major-flavored.
>
> Full row-wise semantics here would require row-major internals (a larger, separate change).

[doctest]: https://github.com/doctest/doctest
[vcpkg]: https://github.com/microsoft/vcpkg
