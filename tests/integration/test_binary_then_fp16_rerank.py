from __future__ import annotations

import numpy as np
import pyarrow as pa
from datafusion import SessionContext
from hamming_maxsim_py import register

from alert_triage.encoders.base import EncodedTokens
from alert_triage.retrievers.base import QueryBundle
from alert_triage.retrievers.binary_rerank import BinaryThenFP16RerankRetriever
from alert_triage.retrievers.hamming_udf import HammingUDFRetriever
from alert_triage.retrievers.hamming_udf_runtime import binary_tokens_to_pylist
from alert_triage.storage.in_memory import InMemoryTokenVectorStore


def _binary_row(*values: int) -> np.ndarray:
    out = np.zeros((len(values), 8), dtype=np.uint8)
    for index, value in enumerate(values):
        out[index] = np.frombuffer(int(value).to_bytes(8, "little"), dtype=np.uint8)
    return out


def test_binary_then_fp16_rerank_uses_live_binary_candidates_and_deterministic_rerank() -> None:
    ctx = SessionContext()
    register(ctx)
    rowids = ["0", "1", "2"]
    doc_ids = ["doc-a", "doc-b", "doc-c"]
    ctx.from_arrow(
        pa.table(
            {
                "_rowid": pa.array(rowids, type=pa.string()),
                "mv_bin": pa.array(
                    [
                        binary_tokens_to_pylist(_binary_row(3)),
                        binary_tokens_to_pylist(_binary_row(3)),
                        binary_tokens_to_pylist(_binary_row(0)),
                    ]
                ),
            }
        ),
        "alerts_mv",
    )

    vector_store = InMemoryTokenVectorStore()
    vector_store.upsert(
        doc_ids,
        [
            EncodedTokens(
                fp16=np.asarray([[0.7, 0.7]], dtype=np.float32),
                binary=_binary_row(3),
            ),
            EncodedTokens(
                fp16=np.asarray([[1.0, 0.0]], dtype=np.float32),
                binary=_binary_row(3),
            ),
            EncodedTokens(
                fp16=np.asarray([[0.0, 1.0]], dtype=np.float32),
                binary=_binary_row(0),
            ),
        ],
    )

    retriever = BinaryThenFP16RerankRetriever(
        candidate_retriever=HammingUDFRetriever(
            ctx=ctx,
            rowid_to_alert_id=dict(zip(rowids, doc_ids, strict=True)).get,
        ),
        vector_store=vector_store,
        prefilter_top_n=2,
    )

    hits = retriever.search(
        QueryBundle(
            query_bin=_binary_row(3),
            query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32),
        ),
        k=2,
    )

    assert [hit.alert_id for hit in hits] == ["doc-b", "doc-a"]
    assert hits[0].score > hits[1].score
