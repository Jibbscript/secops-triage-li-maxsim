from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from datafusion import SessionContext
from hamming_maxsim_py import register

from alert_triage.retrievers.base import FilterClause, QueryBundle, QueryFilter
from alert_triage.retrievers.hamming_udf import HammingUDFRetriever
from alert_triage.retrievers.hamming_udf_runtime import binary_tokens_to_pylist


def _binary_row(*values: int) -> np.ndarray:
    out = np.zeros((len(values), 8), dtype=np.uint8)
    for index, value in enumerate(values):
        out[index] = np.frombuffer(int(value).to_bytes(8, "little"), dtype=np.uint8)
    return out


def _build_context() -> tuple[SessionContext, dict[str, str]]:
    ctx = SessionContext()
    register(ctx)
    rowids = ["0", "1", "2"]
    doc_ids = ["doc-a", "doc-b", "doc-c"]
    table = pa.table(
        {
            "_rowid": pa.array(rowids, type=pa.string()),
            "tenant": pa.array(["alpha", "beta", "alpha"], type=pa.string()),
            "severity": pa.array([1, 2, 3], type=pa.int64()),
            "mv_bin": pa.array(
                [
                    binary_tokens_to_pylist(_binary_row(3)),
                    binary_tokens_to_pylist(_binary_row(3)),
                    binary_tokens_to_pylist(_binary_row(0)),
                ]
            ),
        }
    )
    ctx.from_arrow(table, "alerts_mv")
    return ctx, dict(zip(rowids, doc_ids, strict=True))


def test_hamming_udf_retriever_returns_binary_candidates_in_score_then_rowid_order() -> None:
    ctx, rowid_to_alert_id = _build_context()
    retriever = HammingUDFRetriever(ctx=ctx, rowid_to_alert_id=rowid_to_alert_id.get)

    hits = retriever.search(QueryBundle(query_bin=_binary_row(3)), k=2)

    assert [hit.alert_id for hit in hits] == ["doc-a", "doc-b"]
    assert [hit.score for hit in hits] == [48.0, 48.0]
    assert all(hit.stage == "binary_hamming" for hit in hits)


def test_hamming_udf_retriever_lowers_structured_filters() -> None:
    ctx, rowid_to_alert_id = _build_context()
    retriever = HammingUDFRetriever(ctx=ctx, rowid_to_alert_id=rowid_to_alert_id.get)

    hits = retriever.search(
        QueryBundle(
            query_bin=_binary_row(3),
            filter=QueryFilter(
                clauses=(
                    FilterClause(field="tenant", op="eq", value="alpha"),
                    FilterClause(field="severity", op="gte", value=2),
                )
            ),
        ),
        k=5,
    )

    assert [hit.alert_id for hit in hits] == ["doc-c"]


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        (FilterClause(field="tenant", op="eq", value="alpha"), ["doc-a", "doc-c"]),
        (FilterClause(field="tenant", op="ne", value="alpha"), ["doc-b"]),
        (FilterClause(field="severity", op="lt", value=2), ["doc-a"]),
        (FilterClause(field="severity", op="lte", value=2), ["doc-a", "doc-b"]),
        (FilterClause(field="severity", op="gt", value=2), ["doc-c"]),
        (FilterClause(field="severity", op="gte", value=2), ["doc-b", "doc-c"]),
        (FilterClause(field="tenant", op="in", value=("beta",)), ["doc-b"]),
        (FilterClause(field="tenant", op="not_in", value=("beta",)), ["doc-a", "doc-c"]),
    ],
)
def test_hamming_udf_retriever_supports_phase2_filter_allowlist(
    clause: FilterClause, expected: list[str]
) -> None:
    ctx, rowid_to_alert_id = _build_context()
    retriever = HammingUDFRetriever(ctx=ctx, rowid_to_alert_id=rowid_to_alert_id.get)

    hits = retriever.search(
        QueryBundle(query_bin=_binary_row(3), filter=QueryFilter(clauses=(clause,))),
        k=5,
    )

    assert [hit.alert_id for hit in hits] == expected
