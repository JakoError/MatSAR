# Using MatSAR

A guided tour of the stable core. Every snippet assumes:

```cpp
#include <MatSAR/MatSAR.hpp>   // umbrella: Matrix + MatrixCreation
using namespace matsar;
```

You can also include the pieces directly: `<MatSAR/Matrix.hpp>` (the container,
arithmetic, comparisons, the matrix product) and `<MatSAR/MatrixCreation.hpp>`
(`zeros`, `ones`, `eye`, `diag`, `arange`, …).

Runnable versions of most of this live in [`../examples/`](../examples/).

---

## 1. Constructing matrices

### Nested initializer lists are row-wise

A 2-D nested initializer list is read **row by row** — the NumPy/MATLAB-literal
intuition:

```cpp
Matrix<int64> a = {{1, 2, 3},
                   {4, 5, 6}};   // 2 rows, 3 columns
// a is 2x3:  a(0,1) == 2,  a(1,0) == 4
```

A **flat** list is a 1-D vector of shape `{N}`:

```cpp
Matrix<int64> v = {10, 20, 30};   // shape {3};  v(2) == 30
```

### Explicit row/column builders

When you want the layout to be unambiguous (and independent of the legacy flag
below), use the static builders. They accept either an initializer list or a
`std::vector<std::vector<T>>`:

```cpp
auto r = Matrix<int64>::fromRows({{1, 2, 3}, {4, 5, 6}});      // 2x3 (== the default)
auto c = Matrix<int64>::fromColumns({{1, 2, 3}, {4, 5, 6}});   // 3x2 (each list is a column)
// fromRows(x) == fromColumns(x).T()
```

> **Legacy flag.** `-DMATSAR_LEGACY_COLUMN_INITIALIZER_LIST=ON` makes the *default*
> nested `{{...}}` (and the `vector<vector<T>>` constructor) column-wise again.
> It is a temporary migration aid — prefer `fromRows` / `fromColumns` in code
> that must be unambiguous.

### Creation helpers

```cpp
zeros<float64>({2, 3});   // 2x3 of 0
ones<float64>({2, 2});    // 2x2 of 1
eye<float64>(3);          // 3x3 identity
eye<float64>(3, 4, 1);    // 3x4, ones on the first super-diagonal
diag(Matrix<int64>{1, 2, 3});   // 1-D -> 3x3 diagonal matrix (2-D -> extracts the diagonal)
arange<int64>(0, 10, 2);  // {0, 2, 4, 6, 8}
```

There are also structural helpers: `flip`, `fliplr`, `flipud`, `rot90`, `tril`,
`triu`.

---

## 2. Indexing

`operator()` takes the two-dimensional matrix as the smallest unit. Indices may
be **negative** (counting from the end, like Python):

```cpp
auto a = Matrix<int64>::fromRows({{1, 2, 3}, {4, 5, 6}});
a(0, 1);    // 2     -> row 0, column 1
a(-1, -1);  // 6     -> last row, last column
```

For higher-dimensional matrices, `a(n, i, j)` selects element `(i, j)` of the
`n`-th 2-D page. A single argument `a(offset)` is a raw linear index into the
underlying buffer (see the storage note in §7).

The non-`const` overloads return a reference, so you can assign:

```cpp
a(0, 0) = 42;
```

---

## 3. Shape and structure

```cpp
a.getShape();   // std::vector<int64>, e.g. {2, 3}
a.getDims();    // number of dimensions, e.g. 2
a.getSize();    // total element count, e.g. 6
a.isEmpty(); a.isVec(); a.is2D(); a.isSquare();

a.T();              // transpose (swaps the first two axes)
a.reshape({3, 2});  // new shape with the same element count (one axis may be -1 to infer)
a.squeeze();        // drop size-1 dimensions
```

`reshape` reinterprets the existing buffer (see §7) — it does not copy in a
NumPy row-major sense.

---

## 4. Arithmetic and broadcasting

`+ - * /` are **element-wise**, with scalar operands allowed on either side:

```cpp
auto a = Matrix<float64>::fromRows({{1, 2, 3}, {4, 5, 6}});

a + 10;   // add 10 to every element
2 * a;    // scale every element
a * a;    // element-wise square (NOT a matrix product)
a / a;    // element-wise divide
```

> **`*` is element-wise.** Between two matrices, `a * b` multiplies element by
> element (NumPy semantics). For the linear-algebra matrix product use
> `mmul(a, b)` or `a % b` (§6). `mul(a, b)` is a named alias for the
> element-wise `*`.

**Broadcasting** stretches a size-1 dimension to match the other operand, so a
single row (or column, or scalar-shaped matrix) applies across the larger one:

```cpp
auto row = Matrix<float64>::fromRows({{100, 200, 300}});  // shape {1, 3}
a + row;
//  101  202  303
//  104  205  306
```

---

## 5. Comparisons

`< <= > >= == !=` are element-wise and broadcasting, and produce a boolean
matrix:

```cpp
a > 3;
//  0  0  0
//  1  1  1
```

(Complex operands compare by magnitude.)

---

## 6. The matrix product

`mmul(a, b)` is the BLAS-backed matrix product; `a % b` is the same thing
spelled as an operator. Both are available only in BLAS-enabled builds
(`MATSAR_WITH_BLAS`, on by default).

```cpp
auto a = Matrix<float64>::fromRows({{1, 2, 3}, {4, 5, 6}});       // 2x3
auto b = Matrix<float64>::fromRows({{7, 8}, {9, 10}, {11, 12}});  // 3x2

mmul(a, b);   // 2x2:  { {58, 64}, {139, 154} }
a % b;        // identical
```

`dot(A, B)` covers the common dot-product cases (vector·vector, vector·matrix,
matrix·matrix).

---

## 7. Printing, and the column-major storage caveat

Two views are available:

```cpp
std::cout << a.toString("a");        // braced, data-structure view (good for N-D)
std::cout << a.toRowColString("a");  // classic row/column grid
```

Both print in **logical row-major order**, matching `operator()` — what you see
is what `a(i, j)` returns.

### Storage is column-major

Internally MatSAR stores data **column-major**. Construction, `operator()`, and
the two `toString` views are row-wise, but anything that exposes the raw buffer
follows **storage order**, which does *not* match the row-wise reading. Concretely:

```cpp
auto m = Matrix<int64>::fromRows({{1, 2, 3},
                                  {4, 5, 6}});   // logically rows [1,2,3],[4,5,6]
m(0, 1);   // 2   (row-wise, as expected)

// Iteration (begin()/end(), range-for) and getData() yield COLUMN-major order:
for (auto x : m) std::cout << x << ' ';   // prints: 1 4 2 5 3 6

// reshape() reinterprets that column-major buffer (not a NumPy row-major reshape):
auto r = m.reshape({3, 2});
r(0, 1);   // 5   (from the column-major buffer 1 4 2 5 3 6), NOT 2
```

So:

- **`operator()`, construction, `toString`/`toRowColString`** → row-wise (logical).
- **`begin()`/`end()`, range-`for`, `getData()`, `reshape`**, broadcast alignment,
  and `newAxis` (1-D → `{n,1}`) → **column-major / storage order**.

Keep this boundary in mind when you iterate elements or hand the buffer to other
code. Full row-wise semantics everywhere (row-major internals) would be a larger,
separate change. The same note, with more detail, is in
[`TESTING.md`](TESTING.md#what-the-tests-cover).
