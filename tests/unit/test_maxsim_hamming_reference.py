from __future__ import annotations

import numpy as np

from alert_triage.retrievers._maxsim_hamming_ref import maxsim_hamming


def _pack(bits: np.ndarray) -> np.ndarray:
    packed = np.packbits(bits.astype(np.uint8), axis=-1, bitorder="little")
    out = np.zeros((bits.shape[0], 8), dtype=np.uint8)
    out[:, : packed.shape[1]] = packed
    return out


def _bruteforce_bool(query_bits: np.ndarray, doc_bits: np.ndarray) -> int:
    if query_bits.shape[0] == 0 or doc_bits.shape[0] == 0:
        return 0
    total = 0
    for q in query_bits:
        distances = np.sum(np.logical_xor(q, doc_bits), axis=1)
        total += 48 - int(np.min(distances))
    return total


def test_maxsim_hamming_matches_hand_computation() -> None:
    query = np.zeros((2, 8), dtype=np.uint8)
    doc = np.zeros((2, 8), dtype=np.uint8)
    query[1, 0] = 3
    doc[0, 0] = 1
    doc[1, 0] = 3

    assert maxsim_hamming(query, doc) == 95


def test_maxsim_hamming_matches_bruteforce_property(rng: np.random.Generator) -> None:
    for _ in range(25):
        query_bits = rng.integers(0, 2, size=(rng.integers(1, 5), 48), dtype=np.uint8)
        doc_bits = rng.integers(0, 2, size=(rng.integers(1, 6), 48), dtype=np.uint8)
        query = _pack(query_bits)
        doc = _pack(doc_bits)
        assert maxsim_hamming(query, doc) == _bruteforce_bool(query_bits, doc_bits)
