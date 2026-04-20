from __future__ import annotations

import numpy as np
import pytest

from alert_triage.encoders.base import EncodedTokens
from alert_triage.retrievers.base import QueryBundle
from alert_triage.retrievers.bm25 import BM25Retriever
from alert_triage.retrievers.hybrid_rerank import HybridBM25ThenFP16RerankRetriever
from alert_triage.storage.in_memory import InMemoryTokenVectorStore


def _encoded_docs() -> list[EncodedTokens]:
    return [
        EncodedTokens(fp16=np.asarray([[0.7, 0.7]], dtype=np.float32)),
        EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)),
        EncodedTokens(fp16=np.asarray([[0.0, 1.0]], dtype=np.float32)),
    ]


def test_hybrid_rerank_requires_query_text_and_fp16() -> None:
    bm25 = BM25Retriever()
    docs = _encoded_docs()
    texts = ["phishing credential reset", "phishing reset playbook", "ransomware beacon"]
    bm25.index(["a", "b", "c"], docs, texts=texts)
    store = InMemoryTokenVectorStore()
    store.upsert(["a", "b", "c"], docs)
    retriever = HybridBM25ThenFP16RerankRetriever(candidate_retriever=bm25, vector_store=store)

    with pytest.raises(ValueError, match="query.query_text is required"):
        retriever.search(QueryBundle(query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)))
    with pytest.raises(ValueError, match="query.query_fp16 is required"):
        retriever.search(QueryBundle(query_text="phishing reset"))


def test_hybrid_rerank_rescores_bm25_candidates_with_fp16() -> None:
    docs = _encoded_docs()
    bm25 = BM25Retriever()
    bm25.index(
        ["doc-a", "doc-b", "doc-c"],
        docs,
        texts=[
            "phishing credential reset guide",
            "phishing reset escalation",
            "ransomware beacon containment",
        ],
    )
    store = InMemoryTokenVectorStore()
    store.upsert(["doc-a", "doc-b", "doc-c"], docs)
    retriever = HybridBM25ThenFP16RerankRetriever(
        candidate_retriever=bm25,
        vector_store=store,
        prefilter_top_n=2,
    )

    hits, timings = retriever.search_with_timings(
        QueryBundle(
            query_text="phishing reset",
            query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32),
        ),
        k=2,
    )

    assert [hit.alert_id for hit in hits] == ["doc-b", "doc-a"]
    assert all(hit.stage == "hybrid" for hit in hits)
    assert hits[0].debug["bm25_score"] > 0.0
    assert timings["rerank_ms"] >= 0.0
