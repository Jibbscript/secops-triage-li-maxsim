from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pytest
from datafusion import SessionContext
from hamming_maxsim_py import register

from alert_triage.retrievers.base import Candidate, FilterClause, QueryBundle, QueryFilter
from alert_triage.retrievers.binary_rerank import BinaryThenFP16RerankRetriever
from alert_triage.retrievers.hamming_udf import HammingUDFRetriever
from alert_triage.retrievers.hamming_udf_runtime import binary_tokens_to_pylist


def test_query_bundle_rejects_raw_filter_expr_strings() -> None:
    with pytest.raises(ValueError, match="Raw SQL filter strings are not supported"):
        QueryBundle(filter_expr="severity = 'high'")


def test_query_bundle_accepts_typed_filter_shell() -> None:
    query = QueryBundle(
        query_bin=np.zeros((1, 8), dtype=np.uint8),
        filter=QueryFilter(clauses=(FilterClause(field="tenant", op="eq", value="a"),)),
    )

    assert query.filter is not None
    assert query.filter.clauses[0].field == "tenant"


@dataclass
class FakeCandidateRetriever:
    id: str = "fake-coarse"
    last_query: QueryBundle | None = None

    def index(self, alert_ids, docs, texts=None) -> None:
        return None

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        self.last_query = query
        return [Candidate(alert_id="a", score=1.0, stage="binary_hamming")]

    def size(self) -> int:
        return 1


@dataclass
class FakeVectorStore:
    def fetch_fp16(self, ids):
        return {"a": np.asarray([[1.0, 0.0]], dtype=np.float32)}

    def fetch_bin(self, ids):
        return {}


def test_binary_rerank_forwards_structured_filters_to_candidate_retriever() -> None:
    coarse = FakeCandidateRetriever()
    retriever = BinaryThenFP16RerankRetriever(candidate_retriever=coarse, vector_store=FakeVectorStore())
    query = QueryBundle(
        query_bin=np.zeros((1, 8), dtype=np.uint8),
        query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32),
        filter=QueryFilter(clauses=(FilterClause(field="tenant", op="eq", value="a"),)),
    )

    hits = retriever.search(query)

    assert [hit.alert_id for hit in hits] == ["a"]
    assert coarse.last_query is query


def test_hamming_udf_rejects_empty_tuple_for_in_filters() -> None:
    ctx = SessionContext()
    register(ctx)
    ctx.from_arrow(
        pa.table(
            {
                "_rowid": ["0"],
                "tenant": ["alpha"],
                "mv_bin": [binary_tokens_to_pylist(np.zeros((1, 8), dtype=np.uint8))],
            }
        ),
        "alerts_mv",
    )
    retriever = HammingUDFRetriever(ctx=ctx)

    with pytest.raises(ValueError, match="in filters require a non-empty tuple"):
        retriever.search(
            QueryBundle(
                query_bin=np.zeros((1, 8), dtype=np.uint8),
                filter=QueryFilter(clauses=(FilterClause(field="tenant", op="in", value=()),)),
            )
        )


def test_hamming_udf_rejects_scalar_not_in_filters() -> None:
    ctx = SessionContext()
    register(ctx)
    ctx.from_arrow(
        pa.table(
            {
                "_rowid": ["0"],
                "tenant": ["alpha"],
                "mv_bin": [binary_tokens_to_pylist(np.zeros((1, 8), dtype=np.uint8))],
            }
        ),
        "alerts_mv",
    )
    retriever = HammingUDFRetriever(ctx=ctx)

    with pytest.raises(ValueError, match="not_in filters require a non-empty tuple"):
        retriever.search(
            QueryBundle(
                query_bin=np.zeros((1, 8), dtype=np.uint8),
                filter=QueryFilter(clauses=(FilterClause(field="tenant", op="not_in", value="alpha"),)),
            )
        )


def test_hamming_udf_rejects_tuple_values_for_eq_filters() -> None:
    ctx = SessionContext()
    register(ctx)
    ctx.from_arrow(
        pa.table(
            {
                "_rowid": ["0"],
                "tenant": ["alpha"],
                "mv_bin": [binary_tokens_to_pylist(np.zeros((1, 8), dtype=np.uint8))],
            }
        ),
        "alerts_mv",
    )
    retriever = HammingUDFRetriever(ctx=ctx)

    with pytest.raises(ValueError, match="eq filters do not accept tuple values"):
        retriever.search(
            QueryBundle(
                query_bin=np.zeros((1, 8), dtype=np.uint8),
                filter=QueryFilter(
                    clauses=(FilterClause(field="tenant", op="eq", value=("alpha", "beta")),)
                ),
            )
        )
