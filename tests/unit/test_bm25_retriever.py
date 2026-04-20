from __future__ import annotations

import numpy as np
import pytest

from alert_triage.encoders.base import EncodedTokens
from alert_triage.retrievers.base import QueryBundle
from alert_triage.retrievers.bm25 import BM25Retriever


def _docs() -> list[EncodedTokens]:
    return [
        EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)),
        EncodedTokens(fp16=np.asarray([[0.0, 1.0]], dtype=np.float32)),
        EncodedTokens(fp16=np.asarray([[1.0, 1.0]], dtype=np.float32)),
    ]


def test_bm25_requires_query_text() -> None:
    retriever = BM25Retriever()
    retriever.index(
        ["a"],
        [EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32))],
        texts=["phishing alert"],
    )

    with pytest.raises(ValueError, match="query.query_text is required for BM25Retriever"):
        retriever.search(QueryBundle(query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)))


def test_bm25_ranks_matching_documents_by_lexical_score() -> None:
    retriever = BM25Retriever()
    retriever.index(
        ["alert-1", "alert-2", "alert-3"],
        _docs(),
        texts=[
            "phishing credential reset playbook",
            "ransomware beacon containment guide",
            "phishing malware mixed triage note",
        ],
    )

    hits = retriever.search(QueryBundle(query_text="phishing credential reset"), k=3)

    assert [hit.alert_id for hit in hits] == ["alert-1", "alert-3"]
    assert all(hit.stage == "bm25" for hit in hits)


def test_bm25_index_requires_texts() -> None:
    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="texts are required for BM25Retriever.index"):
        retriever.index(["alert-1"], _docs()[:1], texts=None)
