from __future__ import annotations

import pyarrow as pa
from datafusion import SessionContext

from hamming_maxsim_py import register


def main() -> None:
    ctx = SessionContext()
    token_type = pa.list_(pa.uint8(), 8)
    mv_type = pa.list_(token_type)
    ctx.from_arrow(
        pa.table(
            {
                "_rowid": pa.array(["0", "1"], type=pa.string()),
                "mv_bin": pa.array(
                    [
                        [[3, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0]],
                    ],
                    type=mv_type,
                ),
            }
        ),
        "alerts_mv",
    )
    ctx.from_arrow(
        pa.table(
            {
                "query_bin": pa.array(
                    [[[3, 0, 0, 0, 0, 0, 0, 0]]],
                    type=mv_type,
                )
            }
        ),
        "query_mv",
    )
    register(ctx)
    rows = ctx.sql(
        """
        select a._rowid, hamming_maxsim(q.query_bin, a.mv_bin) as score
        from alerts_mv a
        cross join query_mv q
        order by score desc, a._rowid asc
        limit 2
        """
    ).to_pylist()
    assert rows == [{"_rowid": "0", "score": 48.0}, {"_rowid": "1", "score": 46.0}]


if __name__ == "__main__":
    main()
