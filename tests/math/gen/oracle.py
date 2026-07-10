"""Exact NumPy reference for each MathematicalFunction tag.
MatSAR dim_idx=k <-> numpy axis=k, reduced axis kept as length 1 (keepdims).
A bare call defaults dim_idx=0; global only when the caller passes `none` (axis=None here).
Global cumulative ops are restricted to 1-D in the pilot (C-order == F-order there)."""
import numpy as np


def _wrap1(v):
    return np.asarray([v], dtype=np.float64)   # global scalar -> shape (1,)


ORACLE = {
    "sum":     lambda a, axis: (_wrap1(a.sum())  if axis is None else a.sum(axis=axis, keepdims=True)),
    "prod":    lambda a, axis: (_wrap1(a.prod()) if axis is None else a.prod(axis=axis, keepdims=True)),
    "cumsum":  lambda a, axis: (np.cumsum(a)     if axis is None else np.cumsum(a, axis=axis)),
    "cumprod": lambda a, axis: (np.cumprod(a)    if axis is None else np.cumprod(a, axis=axis)),
    # mean/var use MatSAR's 1-based `dim`; the harness stores the equivalent 0-based
    # axis (matching sum) and the replay maps it back with +1 / <=0-for-global.
    # var default is population variance (ddof=0) == MatSAR's default w=false.
    "mean":    lambda a, axis: (_wrap1(a.mean()) if axis is None else a.mean(axis=axis, keepdims=True)),
    "var":     lambda a, axis: (_wrap1(a.var())  if axis is None else a.var(axis=axis, keepdims=True)),
}
