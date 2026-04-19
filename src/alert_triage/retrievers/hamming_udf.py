from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alert_triage.retrievers.base import Candidate, QueryBundle, Retriever


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

        sql = f"""
        select _rowid, hamming_maxsim(?, mv_bin) as score
        from {self.table_name}
        {f"where {query.filter_expr}" if query.filter_expr else ""}
        order by score desc
        limit {k}
        """

        table = self.ctx.sql(sql, params=[query.query_bin]).to_arrow_table()
        rows = table.to_pylist()

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
        table = self.ctx.sql(f"select count(*) as n from {self.table_name}").to_arrow_table()
        rows = table.to_pylist()
        return int(rows[0]["n"])
