---
name: Feature request
about: Suggest a new capability or API for MatSAR
title: "[Feature] "
labels: enhancement
---

## Problem / motivation

What are you trying to do that MatSAR doesn't support today? If it's modeled on
NumPy/MATLAB behavior, link or describe the reference semantics.

## Proposed API

How might it look from the caller's side?

```cpp
// e.g. auto r = matsar::someNewThing(a, b);
```

## Alternatives considered

Other ways to achieve this, and why they fall short.

## Additional context

Notes on element type support, broadcasting, the column-major storage boundary,
whether it needs BLAS/LAPACK, or whether it belongs in the experimental modules.
