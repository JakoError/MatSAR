# Contributing to MatSAR

Thanks for your interest in MatSAR! This guide covers how to build, the project
conventions, and how to get a change merged. By participating you agree to the
[Apache-2.0 license](LICENSE) terms for your contributions (see
[Developer Certificate of Origin](#developer-certificate-of-origin) below).

## Getting set up

MatSAR is a header-only C++17 library; you need a C++17 compiler, CMake ≥ 3.22,
a generator (Ninja/Make/MSBuild), and — for the matrix product (`mmul`/`%`) —
OpenBLAS. Platform-by-platform setup (MSVC/vcpkg, MinGW/MSYS2, Linux/WSL) is in
[`docs/TESTING.md`](docs/TESTING.md).

```bash
cmake -G Ninja -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Please also build the no-BLAS configuration, since it is a supported path:

```bash
cmake -G Ninja -B build-noblas -DMATSAR_WITH_BLAS=OFF
cmake --build build-noblas
ctest --test-dir build-noblas --output-on-failure
```

## Project conventions

These keep the library portable and predictable. New code should follow them.

- **Row-wise construction.** Nested initializer lists are row-wise by default;
  the internal storage is column-major. Understand the boundary before touching
  construction, indexing, iteration, or `reshape` — it is documented in
  [`docs/usage.md`](docs/usage.md#7-printing-and-the-column-major-storage-caveat).
- **camelCase / PascalCase member names.** Methods and free functions use
  camelCase or PascalCase (e.g. `getShape`, `toRowColString`, `fromRows`), *not*
  `snake_case`. This is deliberate: it avoids colliding with `std::`-style names
  and ADL surprises for a library that leans on `using namespace matsar;`.
- **Strict C++17, no compiler extensions.** The build sets
  `CMAKE_CXX_EXTENSIONS OFF`. Don't rely on GNU-only constructs; portability
  shims belong in [`include/MatSAR/util/Compat.hpp`](include/MatSAR/util/Compat.hpp).
  Code must compile on MSVC, MinGW-w64 GCC, and Linux GCC/Clang.
- **Core uses only CBLAS.** The stable core includes `<cblas.h>` (never
  `<lapacke.h>`) so it builds against any OpenBLAS, including distros that ship
  LAPACKE separately. LAPACKE-dependent code is experimental-only.
- **Experimental work stays quarantined.** Unfinished modules live behind
  `MATSAR_ENABLE_EXPERIMENTAL` (off by default) and are not part of the
  supported API. New experimental code stays opt-in until it is stabilized
  *with its own tests*; it must not break the default (experimental-off) build.

## Tests

Tests use [doctest] (vendored at `tests/doctest.h`) and run through CTest. Each
`tests/*.cpp` is one CTest case; see [`docs/TESTING.md`](docs/TESTING.md) for the
layout and how to run a single test.

- **Any behavior change needs a test.** Add or update a case that fails before
  your change and passes after.
- Tests that depend on a fixed layout should use the explicit `fromRows` /
  `fromColumns` builders so they pass under both the default and the
  `MATSAR_LEGACY_COLUMN_INITIALIZER_LIST` flag.
- Keep the default (experimental-off) build and both BLAS on/off configurations
  green.

## Code style

The repository ships a [`.clang-format`](.clang-format) and a
[`.clang-tidy`](.clang-tidy). The style is intentionally light-touch (it does not
reflow your line breaks); a CI check enforces it. clang-format's output varies by
version, so use the **same pinned version CI uses** to avoid spurious diffs:

```bash
pip install clang-format==22.1.4   # the version CI checks against
clang-format -i path/to/changed_file.hpp
```

Match the style of the surrounding code; keep diffs focused — avoid reformatting
unrelated lines. Hand-laid-out blocks (e.g. the code-generation macros) are fenced
with `// clang-format off` / `// clang-format on`; leave those fences in place.

## Submitting changes

1. **Branch** off `main` (e.g. `git switch -c my-feature`). Don't commit to `main`.
2. Make focused commits with clear messages. Sign each one off (see below).
3. Ensure the suite is green (BLAS on **and** off) and your code is formatted.
4. Open a **pull request** against `main` describing the change and the
   motivation. CI must pass before merge; PRs are merged as merge commits.

### Developer Certificate of Origin

Contributions are accepted under the [DCO](https://developercertificate.org/):
by signing off, you certify you wrote the patch or otherwise have the right to
submit it under the project's license. Add a sign-off line to every commit:

```
Signed-off-by: Your Name <your.email@example.com>
```

`git commit -s` adds it for you (configure `user.name` / `user.email` first).

> A Contributor License Agreement (CLA) may be required for substantial
> contributions in the future. There is no CLA bot today — the DCO sign-off is
> all that is needed for now. Questions: <runningboymy@gmail.com>.

[doctest]: https://github.com/doctest/doctest
