# MatSAR

[![CI](https://github.com/JakoError/MatSAR/actions/workflows/ci.yml/badge.svg)](https://github.com/JakoError/MatSAR/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![header-only](https://img.shields.io/badge/header--only-yes-brightgreen.svg)

**MatSAR** is a header-only C++17 matrix library with a NumPy/MATLAB-like feel:
N-dimensional containers, row-wise literal construction, element-wise arithmetic
with broadcasting, comparisons, and a BLAS-backed matrix product. The goal is an
expressive, single-`#include` numerical container that stays close to the math
you write on paper while compiling on MSVC, MinGW-w64, and Linux GCC/Clang.

> **Project status — early (`0.x`).** The stable core documented here is tested
> across platforms, but this is a pre-1.0 release: the public API may change
> between minor versions. Unfinished modules are quarantined behind an opt-in
> flag and are not part of the supported surface yet.

## Features

- **N-dimensional `matsar::Matrix<T>`** — header-only, templated on the element type.
- **Row-wise literal construction** — `{{1,2,3},{4,5,6}}` reads as rows (NumPy/MATLAB
  intuition), with explicit `Matrix<T>::fromRows(...)` / `fromColumns(...)` builders.
- **Element-wise arithmetic** (`+ - * /`) with **NumPy-style broadcasting** and scalar operands.
- **Element-wise comparisons** (`< <= > >= == !=`), also broadcasting, yielding a boolean matrix.
- **BLAS-backed matrix product** — `mmul(a, b)` and its alias `a % b` (OpenBLAS `*gemm`).
- **Readable printing** — `toString(...)` (braced view) and `toRowColString(...)` (grid view).
- **Creation helpers** — `zeros`, `ones`, `eye`, `diag`, `arange`, and friends.

> Between two matrices, `*` is **element-wise** (like NumPy), *not* a matrix
> product. Use `mmul(a, b)` or `a % b` for the matrix product.

## Quick example

```cpp
#include <iostream>
#include <MatSAR/MatSAR.hpp>

using namespace matsar;

int main() {
    Matrix<double> a = {{1, 2, 3},
                        {4, 5, 6}};        // 2x3, read row-wise

    std::cout << (a + 10).toRowColString("a + 10");   // scalar broadcast
    std::cout << (a * a).toRowColString("a * a");      // element-wise square

    auto row = Matrix<double>::fromRows({{100, 200, 300}});  // shape {1,3}
    std::cout << (a + row).toRowColString("a + row");   // broadcast over rows

    auto c = mmul(a, a.T());   // (2x3) * (3x2) -> 2x2 matrix product
    std::cout << c.toRowColString("a * a^T");
}
```

More, runnable, in [`examples/`](examples/) (`basic`, `broadcasting`, `matmul`)
and a guided tour in [`docs/usage.md`](docs/usage.md).

## Requirements

- A **C++17** compiler: MSVC (VS 2019+), MinGW-w64 GCC, or Linux GCC/Clang.
- **CMake ≥ 3.22**.
- **OpenBLAS** — only for the matrix product (`mmul` / `%`). It is optional:
  build with `-DMATSAR_WITH_BLAS=OFF` to drop the dependency and the rest of the
  library still works.

## Using MatSAR in your project

MatSAR ships an `INTERFACE` CMake target, `MatSAR::matsar`, and installs a
package config, so both `FetchContent` and `find_package` work.

### FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(MatSAR
    GIT_REPOSITORY https://github.com/JakoError/MatSAR.git
    GIT_TAG        v0.1.0)
FetchContent_MakeAvailable(MatSAR)

target_link_libraries(my_app PRIVATE MatSAR::matsar)
```

### find_package (installed / packaged)

```cmake
find_package(MatSAR 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE MatSAR::matsar)
```

Then, in your code:

```cpp
#include <MatSAR/MatSAR.hpp>   // or <MatSAR/Matrix.hpp>, <MatSAR/MatrixCreation.hpp>
```

When `MATSAR_WITH_BLAS` is on (the default), `find_package(MatSAR)` resolves
OpenBLAS for you through the same discovery logic the library builds with
(pkg-config → CMake `FindBLAS` → an OpenBLAS CMake package). If you need the
matrix product, make sure OpenBLAS is installed (a system package, or vcpkg's
`openblas` on Windows); otherwise configure consumers against a MatSAR built
with `-DMATSAR_WITH_BLAS=OFF`.

## Build options

| Option | Default | Effect |
|---|---|---|
| `MATSAR_WITH_BLAS` | `ON` | OpenBLAS-backed `mmul()` / `operator%`. `OFF` drops the OpenBLAS dependency. |
| `MATSAR_BUILD_TESTS` | `ON` when top-level | Build the test suite. |
| `MATSAR_BUILD_EXAMPLES` | `ON` when top-level | Build the programs in `examples/`. |
| `MATSAR_LEGACY_COLUMN_INITIALIZER_LIST` | `OFF` | Temporary migration aid: makes nested `{{...}}` column-wise again. Prefer `fromRows`/`fromColumns`. |
| `MATSAR_ENABLE_EXPERIMENTAL` | `OFF` | Opt into unfinished modules (not part of the supported API yet). |

## Building & testing this repository

```bash
cmake -G Ninja -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Full platform-by-platform instructions (MSVC/vcpkg, MinGW/MSYS2, Linux/WSL) and
the option reference are in [`docs/TESTING.md`](docs/TESTING.md).

## Documentation

- [`docs/usage.md`](docs/usage.md) — constructing matrices, arithmetic/broadcasting,
  comparisons, the matrix product, printing, and the column-major storage caveat.
- [`docs/TESTING.md`](docs/TESTING.md) — building & running the tests on each platform.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to build, the project conventions, and the PR flow.
- [`CHANGELOG.md`](CHANGELOG.md) — notable changes per release.

## Author

MatSAR is developed and maintained by **Zhexian Zhou**.

- GitHub: [@JakoError](https://github.com/JakoError)
- ORCID: [0009-0004-8278-1230](https://orcid.org/0009-0004-8278-1230)

## Citation

If you use MatSAR in research, publications, academic projects, or software that
should acknowledge dependencies, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

```text
Zhou, Zhexian. MatSAR: A header-only C++17 matrix library. Version 0.1.0.
```

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
