from __future__ import annotations

import numpy as np
import pytest

from alert_triage.encoders.base import EncodedTokens
from alert_triage.retrievers.base import FilterClause, QueryBundle, QueryFilter
from alert_triage.retrievers.fp16_ref import FP16ReferenceRetriever


def test_fp16_reference_retriever_ranks_best_match_first() -> None:
    retriever = FP16ReferenceRetriever()
    retriever.index(
        ["a", "b"],
        [
            EncodedTokens(fp16=np.asarray([[0.0, 1.0]], dtype=np.float32)),
            EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)),
        ],
    )

    hits = retriever.search(
        QueryBundle(query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)),
        k=2,
    )

    assert [hit.alert_id for hit in hits] == ["b", "a"]
    assert all(hit.stage == "fp16_maxsim" for hit in hits)


def test_fp16_reference_retriever_requires_query_fp16() -> None:
    retriever = FP16ReferenceRetriever()

    with pytest.raises(ValueError, match="query.query_fp16 is required"):
        retriever.search(QueryBundle())


def test_fp16_reference_retriever_rejects_structured_filters_in_phase1() -> None:
    retriever = FP16ReferenceRetriever()
    retriever.index(["a"], [EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32))])
    query = QueryBundle(
        query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32),
        filter=QueryFilter(clauses=(FilterClause(field="severity", op="eq", value="high"),)),
    )

    with pytest.raises(NotImplementedError, match="Structured filters are not supported"):
        retriever.search(query)
