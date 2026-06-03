---
name: Bug report
about: Report incorrect behavior, a crash, or a build failure
title: "[Bug] "
labels: bug
---

## Description

A clear description of what's wrong.

## Minimal reproducer

The smallest snippet that shows the problem (please make it compilable):

```cpp
#include <MatSAR/MatSAR.hpp>
using namespace matsar;

int main() {
    // ...
}
```

## Expected vs. actual

- **Expected:** what you thought would happen.
- **Actual:** what happened (paste the exact output, error, or stack trace).

## Environment

- OS:                      <!-- e.g. Windows 11 / Ubuntu 22.04 -->
- Compiler & version:      <!-- e.g. MSVC 19.4x / GCC 13 / Clang 17 / MinGW-w64 GCC 16 -->
- CMake version:
- `MATSAR_WITH_BLAS`:      <!-- ON or OFF -->  (and OpenBLAS version if ON)
- MatSAR version / commit:

## Additional context

Anything else that might help (related to broadcasting, the row-wise / column-major
storage boundary, BLAS discovery, etc.).
