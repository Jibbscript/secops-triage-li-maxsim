from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from alert_triage.retrievers.base import Candidate, FilterClause, QueryBundle, QueryFilter
from alert_triage.retrievers.binary_rerank import BinaryThenFP16RerankRetriever
from alert_triage.retrievers.hamming_udf import HammingUDFRetriever


def test_query_bundle_rejects_raw_filter_expr_strings() -> None:
    with pytest.raises(ValueError, match="Raw SQL filter strings are not supported"):
        QueryBundle(filter_expr="severity = 'high'")


def test_hamming_udf_rejects_structured_filters_before_sql_execution() -> None:
    retriever = HammingUDFRetriever(ctx=object())
    query = QueryBundle(
        query_bin=np.zeros((1, 8), dtype=np.uint8),
        filter=QueryFilter(clauses=(FilterClause(field="tenant", op="eq", value="a"),)),
    )

    with pytest.raises(NotImplementedError, match="Structured filters are not supported"):
        retriever.search(query)


@dataclass
class FakeCandidateRetriever:
    id: str = "fake-coarse"

    def index(self, alert_ids, docs, texts=None) -> None:
        return None

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        return [Candidate(alert_id="a", score=1.0, stage="binary_hamming")]

    def size(self) -> int:
        return 1


@dataclass
class FakeVectorStore:
    def fetch_fp16(self, ids):
        return {"a": np.asarray([[1.0, 0.0]], dtype=np.float32)}

    def fetch_bin(self, ids):
        return {}


def test_binary_rerank_rejects_structured_filters_before_candidate_fetch() -> None:
    retriever = BinaryThenFP16RerankRetriever(
        candidate_retriever=FakeCandidateRetriever(),
        vector_store=FakeVectorStore(),
    )
    query = QueryBundle(
        query_bin=np.zeros((1, 8), dtype=np.uint8),
        query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32),
        filter=QueryFilter(clauses=(FilterClause(field="tenant", op="eq", value="a"),)),
    )

    with pytest.raises(NotImplementedError, match="Structured filters are not supported"):
        retriever.search(query)
