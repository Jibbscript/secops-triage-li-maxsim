from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyarrow as pa
from datafusion import SessionContext, udf


LOW48_MASK = (1 << 48) - 1
TOKEN_WIDTH_BYTES = 8
TOKEN_TYPE = pa.list_(pa.uint8(), TOKEN_WIDTH_BYTES)
MULTIVECTOR_TYPE = pa.list_(TOKEN_TYPE)


@lru_cache(maxsize=1)
def _rust_maxsim():
    from hamming_maxsim_py import maxsim

    return maxsim


def _row_to_u64_tokens(row: list[list[int]] | None) -> list[int] | None:
    if row is None:
        return None
    return [int.from_bytes(bytes(token), "little") & LOW48_MASK for token in row]


def hamming_maxsim_batch(query: pa.Array, doc: pa.Array) -> pa.Array:
    scores: list[float | None] = []
    maxsim = _rust_maxsim()
    for query_row, doc_row in zip(query.to_pylist(), doc.to_pylist(), strict=True):
        query_tokens = _row_to_u64_tokens(query_row)
        doc_tokens = _row_to_u64_tokens(doc_row)
        if query_tokens is None or doc_tokens is None:
            scores.append(None)
            continue
        scores.append(float(maxsim(query_tokens, doc_tokens)))
    return pa.array(scores, type=pa.float32())


@lru_cache(maxsize=1)
def build_hamming_maxsim_udf():
    return udf(
        hamming_maxsim_batch,
        [MULTIVECTOR_TYPE, MULTIVECTOR_TYPE],
        pa.float32(),
        "immutable",
        "hamming_maxsim",
    )


def register_hamming_udf(ctx: SessionContext) -> None:
    ctx.register_udf(build_hamming_maxsim_udf())


def binary_tokens_to_pylist(tokens: np.ndarray) -> list[list[int]]:
    arr = np.asarray(tokens, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != TOKEN_WIDTH_BYTES:
        raise ValueError(
            f"expected binary multivector shape=(tokens,{TOKEN_WIDTH_BYTES}), got {arr.shape}"
        )
    return arr.tolist()


def binary_tokens_to_scalar(tokens: np.ndarray) -> pa.Scalar:
    return pa.scalar(binary_tokens_to_pylist(tokens), type=MULTIVECTOR_TYPE)
