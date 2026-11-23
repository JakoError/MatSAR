# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version
is `0.x`, the public API may change between minor releases.

## [Unreleased]

### Fixed
- Nested initializer lists: each pair of braces is now one dimension, so rows of one
  element (`{{1.0}, {2.0}}`, a 2x1 column) and `{{x}}` (1x1) compile; they used to be
  ambiguous with the 1-D form. An element and a list side by side is a `ValueError`.
- A ragged nested initializer list or `vector<vector<T>>` (rows of different lengths) is a
  `ValueError`; it used to be copied past the end of the matrix buffer. An empty nested
  vector gives an empty matrix instead of dereferencing an empty vector.
- `Matrix<T>(shape, 0)` fills with zeros; the literal 0 used to be ambiguous with the
  buffer constructor `Matrix(shape, T *)`.

## [0.1.0] - 2026-06-03

Initial public release of the stable MatSAR core.

### Added
- Header-only, N-dimensional `matsar::Matrix<T>` container with column-major storage.
- Row-wise nested initializer-list construction by default, plus explicit
  `Matrix<T>::fromRows(...)` / `Matrix<T>::fromColumns(...)` (accepting an initializer
  list or a `std::vector<std::vector<T>>`).
- Element-wise arithmetic (`+ - * /`) with NumPy-style broadcasting and scalar operands.
- Element-wise comparisons (`< <= > >= == !=`) with broadcasting.
- OpenBLAS-backed matrix product `mmul(a, b)` and its alias `operator%`.
- Printing via `toString(...)` / `toRowColString(...)` (logical row-major).
- Creation helpers (`eye`, `ones`, ...).
- CMake packaging: `find_package(MatSAR)` and `FetchContent` support, target
  `MatSAR::matsar`; OpenBLAS discovered via the bundled `MatSARBlas` module
  (pkg-config → FindBLAS → OpenBLAS config). Build option `MATSAR_WITH_BLAS=OFF`
  drops the OpenBLAS dependency.
- `MATSAR_LEGACY_COLUMN_INITIALIZER_LIST` migration flag (default OFF) to temporarily
  restore legacy column-wise nested construction.
- Runnable examples under `examples/` (construction/printing, broadcasting, matrix
  product), built when `MATSAR_BUILD_EXAMPLES` is on (default ON when top-level).
- Documentation: `README.md`, a usage guide (`docs/usage.md`), build/test guide
  (`docs/TESTING.md`), and `CONTRIBUTING.md`; project tooling (`.clang-format`,
  GitHub issue/PR templates).

### Notes
- This is an early `0.x` release: the API may change before `1.0`.
- Only *construction*, `operator()`, and the *display* functions are row-wise; internal
  storage (iteration, `getData()`, `reshape`) remains column-major. See `docs/usage.md`.

[Unreleased]: https://github.com/JakoError/MatSAR/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JakoError/MatSAR/releases/tag/v0.1.0
