from __future__ import annotations

import numpy as np


BITS = 48
STRIDE = 8
MASK48 = (1 << BITS) - 1


def _validate_binary_tokens(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape={arr.shape}")
    if arr.shape[1] != STRIDE:
        raise ValueError(f"expected stride={STRIDE}, got shape={arr.shape}")
    return arr


def packed_rows_to_u64(x: np.ndarray) -> list[int]:
    rows = _validate_binary_tokens(x)
    return [int.from_bytes(row.tobytes(), "little") & MASK48 for row in rows]


def hamming_distance_u64(lhs: int, rhs: int) -> int:
    return ((lhs ^ rhs) & MASK48).bit_count()


def maxsim_hamming(query: np.ndarray, doc: np.ndarray, *, bits: int = BITS) -> int:
    if bits != BITS:
        raise ValueError(f"phase-0 reference only supports bits={BITS}")
    q_rows = packed_rows_to_u64(query)
    d_rows = packed_rows_to_u64(doc)
    if not q_rows or not d_rows:
        return 0
    score = 0
    for q in q_rows:
        min_h = min(hamming_distance_u64(q, d) for d in d_rows)
        score += bits - min_h
    return score
