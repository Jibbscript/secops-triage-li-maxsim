from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datafusion import col, lit

from .base import Candidate, QueryBundle, lower_structured_filter
from .hamming_udf_runtime import binary_tokens_to_scalar, build_hamming_maxsim_udf


@dataclass
class HammingUDFRetriever:
    """Binary Hamming candidate generator backed by a DataFusion UDF.

    Parameters
    ----------
    ctx:
        Shared datafusion SessionContext that already has `hamming_maxsim` registered.
    table_name:
        SQL-visible table name for the Lance-backed corpus.
    rowid_to_alert_id:
        Optional mapping layer when `_rowid` is not the external alert identifier.
    """

    ctx: Any
    table_name: str = "alerts_mv"
    rowid_to_alert_id: Any | None = None
    id: str = "hamming-udf-bin"

    def index(self, alert_ids, docs, texts=None) -> None:
        # Index lifecycle is owned by the Lance dataset. Kept for protocol compatibility.
        return None

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        if query.query_bin is None:
            raise ValueError("query.query_bin is required for HammingUDFRetriever")
        if k < 1:
            raise ValueError("k must be >= 1")
        query_scalar = binary_tokens_to_scalar(query.query_bin)
        df = self.ctx.table(self.table_name)
        filter_expr = lower_structured_filter(query.filter)
        if filter_expr is not None:
            df = df.filter(filter_expr)

        rows = (
            df.with_column("__query_bin", lit(query_scalar))
            .select(
                col("_rowid"),
                build_hamming_maxsim_udf()(col("__query_bin"), col("mv_bin")).alias("score"),
            )
            .sort(col("score").sort(ascending=False, nulls_first=False), col("_rowid").sort())
            .limit(k)
            .to_pylist()
        )

        out: list[Candidate] = []
        for row in rows:
            row_id = str(row["_rowid"])
            alert_id = (
                self.rowid_to_alert_id(row_id) if callable(self.rowid_to_alert_id) else row_id
            )
            out.append(
                Candidate(
                    alert_id=alert_id,
                    score=float(row["score"]),
                    stage="binary_hamming",
                    debug={},
                )
            )
        return out

    def size(self) -> int:
        return int(self.ctx.table(self.table_name).count())
