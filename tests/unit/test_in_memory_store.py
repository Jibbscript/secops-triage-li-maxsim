from __future__ import annotations

import numpy as np
import pytest

from alert_triage.encoders.base import EncodedTokens
from alert_triage.storage.in_memory import InMemoryTokenVectorStore


def test_in_memory_store_upserts_and_fetches_fp16_and_binary() -> None:
    store = InMemoryTokenVectorStore()
    store.upsert(
        ["a", "b"],
        [
            EncodedTokens(
                fp16=np.asarray([[1.0, 0.0]], dtype=np.float32),
                binary=np.asarray([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8),
            ),
            EncodedTokens(fp16=np.asarray([[0.0, 1.0]], dtype=np.float32)),
        ],
    )

    fp16_docs = store.fetch_fp16(["b", "a", "missing"])
    bin_docs = store.fetch_bin(["a", "b"])

    assert list(fp16_docs) == ["b", "a"]
    assert set(bin_docs) == {"a"}
    assert store.size() == 2


def test_in_memory_store_rejects_mismatched_lengths() -> None:
    store = InMemoryTokenVectorStore()

    with pytest.raises(ValueError, match="matching lengths"):
        store.upsert(["a"], [])
