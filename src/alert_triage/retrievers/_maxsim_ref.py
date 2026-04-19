from __future__ import annotations

import numpy as np


def _validate_pair(query: np.ndarray, doc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float32)
    d = np.asarray(doc, dtype=np.float32)
    if q.ndim != 2 or d.ndim != 2:
        raise ValueError(f"expected 2D matrices, got {q.shape=} {d.shape=}")
    if q.shape[1] != d.shape[1]:
        raise ValueError(f"dimension mismatch: {q.shape[1]} != {d.shape[1]}")
    return q, d


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def maxsim_cosine(query: np.ndarray, doc: np.ndarray) -> float:
    """Reference late-interaction cosine maxsim scorer."""
    q, d = _validate_pair(query, doc)
    if q.shape[0] == 0 or d.shape[0] == 0:
        return 0.0
    qn = _normalize_rows(q)
    dn = _normalize_rows(d)
    sims = qn @ dn.T
    return float(np.max(sims, axis=1).sum())
