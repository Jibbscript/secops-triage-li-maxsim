from __future__ import annotations

import numpy as np
import pytest

from alert_triage.retrievers._maxsim_ref import maxsim_cosine


def test_maxsim_cosine_matches_hand_computation() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    doc = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    score = maxsim_cosine(query, doc)

    assert score == pytest.approx(float(np.float32(1.0 + 2**-0.5)))


def test_maxsim_cosine_returns_zero_for_empty_doc() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    doc = np.zeros((0, 2), dtype=np.float32)

    assert maxsim_cosine(query, doc) == 0.0
