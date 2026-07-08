# MatSAR `math/` Module Test Census — 2026-06-04

Autonomous compile- and correctness-test pass over the experimental `math/` tree
(plus its keystone `UniversalFunctions.hpp`). Goal: **identify issues one module at
a time**, with minimal/no changes to the library. No library files were modified to
produce this report — all blockers were worked around with compiler flags only (see
*Methodology*), and findings are cataloged for a later fix pass.

Toolchain: MinGW-w64 GCC 16.1.0, system OpenBLAS 0.3.33 (pkg-config), lapacke.h present.

---

## TL;DR scoreboard

| Module | Compiles under real build (`-std=c++17`)? | Algorithm verified correct? | Headline issue(s) |
|---|---|---|---|
| **UniversalFunctions** (keystone) | ❌ | n/a (infra) | `typeof`; **`apply_function` default-arg bug**; `DimIter` non-assignable |
| **Trigonometry** | ❌ (include path + UF) | ✅ **40/40** (scalar kernels + `angle`) | matrix wrappers + `pol2cart` blocked by `apply_function` |
| **MathematicalFunction** | ❌ (UF only) | ✅ **10/10** (sum/prod/cumsum/cumprod, 1-D) | none of its own found |
| **DigitalSignal** | ❌ (`typeof` + UF) | ✅ **21/21** (fft/ifft/conv, incl. N=3) | **leftover `std::cout` debug prints** in FFT |
| **LogicFunctions** | ❌ (UF only) | ❌ **4/8 — real bug** | **`all`/`any` reduce only the first element** |
| **SortSearchCount** | ❌ **won't compile** | ⛔ untested | `std::sort` needs assignable iterator; `DimIter` is not |
| **LinearAlgebra** | ❌ (16 errors w/ BLAS) | ⛔ untested | undeclared **`boost`** dep; ambiguous `lu<complex>` |
| **MatrixOperation** | ❌ | ⛔ untested | `apply_function` positional-caller conflict; inherits LinAlg |
| **Probability** | ❌ (21 errors) | ⛔ untested | dup `mean`/`median`; missing `sort` include; inherits LinAlg |

**Good news:** every module that could be exercised has **correct math** except
LogicFunctions. The thing blocking nearly everything is a small set of **cross-cutting
mechanical bugs**, not bad algorithms.

---

## Cross-cutting issues (fix these first — they unblock most modules)

### X1. `typeof` instead of `decltype` — *breaks the real `-std=c++17` build*
The project builds with `CMAKE_CXX_EXTENSIONS OFF` → `-std=c++17`, where `typeof`
(a GNU extension) does not exist. 10 active occurrences:
- `UniversalFunctions.hpp`: 100, 350, 367, 527, 564
- `math/DigitalSignal.hpp`: 94, 304, 442
- `math/LinearAlgebra.hpp`: 231, 287

**Fix:** replace `typeof(expr)` → `decltype(expr)` (semantics identical here).

### X2. Broken `#include` paths (post-restructure regression)
| File:line | Has | Should be |
|---|---|---|
| `UniversalFunctions.hpp:12` | `"../Matrix.hpp"` | `"Matrix.hpp"` |
| `SparseMatrix.hpp:11` | `"../MatSARnew/Matrix.hpp"` | `"Matrix.hpp"` |
| `math/Trigonometry.hpp:19` | `"UniversalFunctions.hpp"` | `"../UniversalFunctions.hpp"` |

`UniversalFunctions.hpp` is the keystone, so its bad path (X2) alone makes **every**
math module fail to find `Matrix.hpp`.

### X3. `apply_function` default template argument is ill-formed — *breaks every elementwise op*
`UniversalFunctions.hpp:190` (and the 2-matrix overload at `:213`):
```cpp
template<typename T, typename TargetType=decltype(fun(std::declval<T>())), typename F>
auto apply_function(const Matrix<T> &m, F fun) -> Matrix<TargetType> { ... }
```
`TargetType`'s default references **`fun`** (a function *parameter*, not in scope) and
its type `F` is declared *after* `TargetType`. Only checked on instantiation, so the
syntax-only census missed it; it fires the moment any `apply_function(m, fn)` is called
(e.g. `Math::sin(matrix)`), with *"'fun' was not declared in this scope"*.

**Caveat for the fix:** two caller conventions exist, so this isn't a pure reorder:
- Trigonometry calls `apply_function(m, Sin<T>)` → relies on the `TargetType` *default*.
- MatrixOperation calls `apply_function<dType, TargetType>(m, fn)` (`:184`) → passes
  `T, TargetType` *positionally*, which forbids putting `F` before `TargetType`.
A correct fix must satisfy both (e.g. an SFINAE/`std::invoke_result_t` helper, or
overloads), not just swap parameter order.

**Verified fix (compiles under real `-std=c++17`, 9/9 value checks pass — both caller
conventions, incl. type-changing `double→int`):**
```cpp
template<typename T, typename TargetType = void, typename F>   // F trailing + deduced
auto apply_function(const Matrix<T> &m, F fun)
    -> Matrix<std::conditional_t<std::is_void_v<TargetType>,
                                 std::invoke_result_t<F&, const T&>, TargetType>> {
    using Out = std::conditional_t<std::is_void_v<TargetType>,
                                   std::invoke_result_t<F&, const T&>, TargetType>;
    Matrix<Out> r(m.getShape(), Out(0), m.getData());
    auto it = m.begin();
    for (auto &o : r) { o = fun(*it); ++it; }
    return r;
}
```
`void` sentinel → derive target from `invoke_result`; explicit `<T,Target>` still works
because `F` is a *trailing deduced* parameter. Apply the same shape to the 2-matrix
overload (`std::invoke_result_t<F&, const T1&, const T2&>`). Verified in
`_mathtest/applyfn_fix_probe.cpp`.

### X4. `DimIter` is not copy-assignable — *breaks `std::sort` and any mutating-iterator algorithm*
`UniversalFunctions.hpp:62` gives `DimIter<T>` a `const msize_type length` (and other
`const`) member, which deletes `operator=`. `std::sort` requires assignable iterators →
`SortSearchCount::sort` fails to compile. **Fix:** drop `const` on the data members (or
give `DimIter` a hand-written `operator=`).

---

## Per-module detail

### UniversalFunctions.hpp (keystone infrastructure)
Issues X1, X2, X3, X4 all live here. Nothing else can be trusted until this file is
clean. After X1–X4, re-test the `apply_function`/`apply_tfunction` families directly.

### Trigonometry.hpp — math is correct ✅
`trig_probe` → **40/40**. Verified: `Sin/Cos/Tan`, all hyperbolics, all inverses, all six
reciprocals (`Cot/Sec/Csc` + inverses, formula-correct), `Atan2`, and `angle()` (real
inputs: negative→π, else 0). Out-of-domain reals return **NaN** (matches the agreed
NumPy-style decision). `angle()` works because it iterates manually.
- **Blocked, untested:** the matrix wrappers `sin(Matrix)…` and `pol2cart` (they route
  through the broken `apply_function`, X3). Expected fine once X3 lands.
- Minor: `pol2cart(Matrix&, Matrix&)` takes non-const refs → can't accept temporaries.

### DigitalSignal.hpp — math is correct ✅, but prints debug noise
`dsp_probe` → **21/21**: `fft([1,2,3,4]) = [10, -2+2i, -2, -2-2i]`, impulse/DC, an 8-pt
`ifft(fft(x))==x` round trip, and **non-power-of-2 N=3**.
- **Bug:** active `std::cout` debug output in the FFT radix logic —
  `DigitalSignal.hpp:129, 132, 136, 138` (the commented-out twins are at 111–120).
  Pollutes stdout and slows the transform. **Fix:** delete lines 129–138's prints.

### MathematicalFunction.hpp — math is correct ✅
`reduce_probe` → **10/10** for `sum/prod/cumsum/cumprod` on a 1-D vector. (Multi-axis
orientation not yet tested — needs layout-aware cases.)

### LogicFunctions.hpp — REAL CORRECTNESS BUG ❌
`logic_probe` → **4/8**. `all([1,0,1])`→true (want false); `any([0,0,1])`→false (want
true); fails identically for per-dim *and* global paths.
- **Root cause:** the four reduction functors `ALL_BOOL/ANY_BOOL/ALL_NUM/ANY_NUM`
  (`LogicFunctions.hpp:27–92`) **never advance the source iterator** — the loop reads
  `*src_iter` `n` times without `++src_iter`, so only element 0 is inspected:
  ```cpp
  for (msize_type i = 0; i < n; ++i) {
      if (*src_iter == 0) { *dst_iter = false; break; }
      // MISSING: ++src_iter;
  }
  ```
  **Fix:** add `++src_iter;` at the end of each loop body (4 functors). The
  commented-out reference loops above each functor were correct.

### SortSearchCount.hpp — won't compile ⛔
`std::sort` instantiation fails: `DimIter` non-assignable (issue **X4**). Fix X4, then
re-test `sort` (asc/desc, multi-dim).

### LinearAlgebra.hpp — won't compile (16 errors with BLAS+LAPACKE) ⛔
With `-DMATSAR_WITH_BLAS` and lapacke.h available, it drops from 188→16 errors:
- **Undeclared `boost`**: `Expint` uses `boost::math::expint` at lines **1016, 1020,
  1032**, but Boost is neither included nor a project dependency. **Decide:** add a Boost
  dependency, or reimplement/remove `Expint`.
- **Ambiguous `lu<complex>` overloads** at **526, 530** (`lu<complex64>` / `lu<complex128>`).
- Plus X1 (`typeof` 231/287) and the previously-noted brace-init return latent bug.
- Structural: `lapack_int`/`<lapacke.h>` are under `#ifdef MATSAR_WITH_BLAS` (36–55), but
  the code using `lapack_int` (89+) is **not** guarded → no-BLAS builds can't compile this
  module at all. Either guard the whole module or provide a no-BLAS fallback.

### MatrixOperation.hpp — won't compile ⛔
Inherits all LinearAlgebra errors (it includes it), plus its own `apply_function`
positional-caller failure at `:184` (see X3 caveat). Re-test after X3 + LinearAlgebra.

### Probability.hpp — won't compile (21 errors) ⛔
Inherits LinearAlgebra's 16, plus its own:
- **`'sort' is not a member of 'matsar::Math'`** at `:150` (median needs SortSearchCount,
  which Probability doesn't include — and `sort` is itself broken by X4).
- **Redefinition of `median`** (`:190`) and **`mean`** (`:214`) — duplicate templates.
- **`factorial<float64>(float64&)` no matching call** at `:822`.
- Note: it is also doubly-quarantined (commented out in `MatSAR.hpp:34`). RNG-based
  functions will need a statistical (seeded) test strategy, separate from these.

---

## Suggested fix order (dependency-aware)

1. **X2** (include paths) + **X1** (`typeof`→`decltype`) — unblocks compilation everywhere.
2. **X3** (`apply_function`) — unblocks all elementwise ops (Trig wrappers, MatrixOp, …).
3. **X4** (`DimIter` assignable) — unblocks `sort` and mutating algorithms.
4. **LogicFunctions** `++src_iter` — fixes the one confirmed wrong algorithm.
5. **DigitalSignal** — delete debug prints (129–138).
6. **LinearAlgebra** — resolve Boost (`Expint`) + ambiguous `lu`; then MatrixOperation,
   then Probability (dedupe mean/median, add sort include, fix factorial).

After 1–3, the verified-correct modules (Trigonometry, MathematicalFunction,
DigitalSignal) should compile *and* pass through their public matrix APIs, ready to be
wired into the doctest suite (`MATSAR_BUILD_EXPERIMENTAL_TESTS` scaffold from the
brainstorm).

---

## Methodology / caveats

- **Strict census** (`-std=c++17 -I include` + OpenBLAS cflags) = the honest "does a
  consumer/CI build see this" test. All modules fail here today (X1+X2).
- **Probe mode** (`-std=gnu++17` + extra `-I include/MatSAR -I include/MatSAR/util`) was
  used *only* to see past the known mechanical blockers (X1 via GNU mode, X2 via search
  paths) so the actual algorithms could run. **These are not fixes** — the real build
  still needs X1/X2 resolved. `-std=gnu++17` masks `typeof`; `-DMATSAR_WITH_BLAS` was
  added for the LinAlg/MatrixOp/Probability pass.
- Correctness was checked by direct value comparison vs `std::`/hand-computed/NumPy
  references. For Trigonometry the **scalar kernels** were tested directly (bypassing the
  broken `apply_function`), which validates the formulas but not the matrix-wrapping layer.
- Scratch probes live in `E:\Code\C++\MatSAR\_mathtest\` (`trig_probe.cpp`, `dsp_probe.cpp`,
  `reduce_probe.cpp`, `sort_probe.cpp`, `logic_probe.cpp`) — outside the repo, nothing
  committed.
- Not yet covered: multi-axis/ND reductions, complex-input trig, `conv` modes other than
  full, and all of LinAlg/MatrixOp/Probability runtime behavior (blocked on compilation).
