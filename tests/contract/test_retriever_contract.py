from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from alert_triage.encoders.base import EncodedTokens
from alert_triage.retrievers.base import Candidate, QueryBundle
from alert_triage.retrievers.bm25 import BM25Retriever
from alert_triage.retrievers.binary_rerank import BinaryThenFP16RerankRetriever
from alert_triage.retrievers.hybrid_rerank import HybridBM25ThenFP16RerankRetriever
from alert_triage.retrievers.lance_mv import LanceMVRetriever
from alert_triage.storage.in_memory import InMemoryTokenVectorStore


@dataclass
class FakeCandidateRetriever:
    candidates: list[Candidate]

    id: str = "fake-coarse"

    def index(self, alert_ids, docs, texts=None) -> None:  # pragma: no cover - protocol stub
        return None

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        return self.candidates[:k]

    def size(self) -> int:
        return len(self.candidates)


@dataclass
class FakeVectorStore:
    docs: dict[str, np.ndarray]

    def fetch_fp16(self, ids):
        return {doc_id: self.docs[doc_id] for doc_id in ids if doc_id in self.docs}

    def fetch_bin(self, ids):
        return {}


def test_binary_rerank_requires_both_query_representations() -> None:
    retriever = BinaryThenFP16RerankRetriever(
        candidate_retriever=FakeCandidateRetriever([]),
        vector_store=FakeVectorStore({}),
    )

    with pytest.raises(ValueError, match="query.query_bin is required"):
        retriever.search(QueryBundle(query_fp16=np.ones((1, 2), dtype=np.float32)))
    with pytest.raises(ValueError, match="query.query_fp16 is required"):
        retriever.search(QueryBundle(query_bin=np.zeros((1, 8), dtype=np.uint8)))


def test_binary_rerank_sorts_by_fp16_score_and_preserves_binary_debug() -> None:
    coarse = FakeCandidateRetriever(
        [
            Candidate(alert_id="a", score=9.0, stage="binary_hamming"),
            Candidate(alert_id="b", score=8.0, stage="binary_hamming"),
            Candidate(alert_id="missing", score=7.0, stage="binary_hamming"),
        ]
    )
    store = FakeVectorStore(
        {
            "a": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "b": np.asarray([[1.0, 0.0]], dtype=np.float32),
        }
    )
    retriever = BinaryThenFP16RerankRetriever(candidate_retriever=coarse, vector_store=store)
    query = QueryBundle(
        query_bin=np.zeros((1, 8), dtype=np.uint8),
        query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    hits = retriever.search(query, k=2)

    assert [hit.alert_id for hit in hits] == ["b", "a"]
    assert [hit.stage for hit in hits] == ["fp16_maxsim", "fp16_maxsim"]
    assert hits[0].debug["binary_score"] == 8.0
    assert hits[1].debug["binary_score"] == 9.0


def test_binary_rerank_size_delegates_to_candidate_retriever() -> None:
    coarse = FakeCandidateRetriever([Candidate(alert_id="a", score=1.0, stage="binary_hamming")])
    retriever = BinaryThenFP16RerankRetriever(
        candidate_retriever=coarse,
        vector_store=FakeVectorStore({}),
    )

    assert retriever.size() == 1


def test_bm25_requires_query_text() -> None:
    retriever = BM25Retriever()
    retriever.index(
        ["doc-a"],
        [EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32))],
        texts=["phishing alert"],
    )

    with pytest.raises(ValueError, match="query.query_text is required for BM25Retriever"):
        retriever.search(QueryBundle(query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)))


def test_lance_mv_requires_fp16_query(tmp_path: Path) -> None:
    retriever = LanceMVRetriever(dataset_path=tmp_path / "fixture.lance")
    retriever.index(
        ["doc-a"],
        [EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32))],
        texts=["phishing alert"],
    )

    with pytest.raises(ValueError, match="query.query_fp16 is required for LanceMVRetriever"):
        retriever.search(QueryBundle(query_text="phishing"))


def test_hybrid_requires_query_text_and_fp16() -> None:
    docs = [
        EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)),
        EncodedTokens(fp16=np.asarray([[0.0, 1.0]], dtype=np.float32)),
    ]
    bm25 = BM25Retriever()
    bm25.index(["doc-a", "doc-b"], docs, texts=["phishing reset", "ransomware beacon"])
    store = InMemoryTokenVectorStore()
    store.upsert(["doc-a", "doc-b"], docs)
    retriever = HybridBM25ThenFP16RerankRetriever(candidate_retriever=bm25, vector_store=store)

    with pytest.raises(ValueError, match="query.query_text is required"):
        retriever.search(QueryBundle(query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)))
    with pytest.raises(ValueError, match="query.query_fp16 is required"):
        retriever.search(QueryBundle(query_text="phishing reset"))
