from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from datafusion import SessionConfig, SessionContext
from hamming_maxsim_py import maxsim, register

from alert_triage.encoders.base import BinaryCalibratorModel, EncodedTokens
from alert_triage.evals.tier1_phase1 import _latency_percentile, _load_corpus, _load_queries
from alert_triage.retrievers.base import QueryBundle
from alert_triage.retrievers.binary_rerank import BinaryThenFP16RerankRetriever
from alert_triage.retrievers.fp16_ref import FP16ReferenceRetriever
from alert_triage.retrievers.hamming_udf import HammingUDFRetriever
from alert_triage.retrievers.hamming_udf_runtime import binary_tokens_to_pylist
from alert_triage.storage.in_memory import InMemoryTokenVectorStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase-2 binary integration harness.")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--cache-state", default="cold")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rerank-depth", type=int, default=5)
    return parser.parse_args()


def _binary_scalar_to_u64_tokens(tokens: np.ndarray) -> list[int]:
    return [int.from_bytes(bytes(row.tolist()), "little") & ((1 << 48) - 1) for row in tokens]


def _pad_to_binary_dim(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D token matrix, got shape={arr.shape}")
    if arr.shape[1] > 48:
        raise ValueError(f"phase-2 harness supports up to 48 dims, got shape={arr.shape}")
    if arr.shape[1] == 48:
        return arr
    padded = np.zeros((arr.shape[0], 48), dtype=np.float32)
    padded[:, : arr.shape[1]] = arr
    return padded


def _build_binary_docs(fp16_docs: list[np.ndarray]) -> tuple[BinaryCalibratorModel, list[np.ndarray]]:
    padded_docs = [_pad_to_binary_dim(doc) for doc in fp16_docs]
    train = np.vstack(padded_docs).astype(np.float32, copy=False)
    calibrator = BinaryCalibratorModel.fit(train)
    return calibrator, [calibrator.apply(doc) for doc in padded_docs]


def _build_context(
    doc_ids: list[str], binary_docs: list[np.ndarray], *, threads: int
) -> tuple[SessionContext, dict[str, str]]:
    config = SessionConfig().with_target_partitions(threads)
    ctx = SessionContext(config)
    register(ctx)
    rowids = [str(index) for index in range(len(doc_ids))]
    rowid_to_alert_id = dict(zip(rowids, doc_ids, strict=True))
    table = pa.table(
        {
            "_rowid": pa.array(rowids, type=pa.string()),
            "mv_bin": pa.array([binary_tokens_to_pylist(doc) for doc in binary_docs]),
        }
    )
    ctx.from_arrow(table, "alerts_mv")
    return ctx, rowid_to_alert_id


def _build_retrievers(
    corpus_ids: list[str],
    fp16_docs: list[np.ndarray],
    binary_docs: list[np.ndarray],
    *,
    rerank_depth: int,
    threads: int,
) -> tuple[FP16ReferenceRetriever, HammingUDFRetriever, BinaryThenFP16RerankRetriever]:
    ctx, rowid_to_alert_id = _build_context(corpus_ids, binary_docs, threads=threads)
    binary_retriever = HammingUDFRetriever(ctx=ctx, rowid_to_alert_id=rowid_to_alert_id.get)

    vector_store = InMemoryTokenVectorStore()
    vector_store.upsert(
        corpus_ids,
        [
            EncodedTokens(fp16=doc, binary=doc_bin)
            for doc, doc_bin in zip(fp16_docs, binary_docs, strict=True)
        ],
    )
    staged_retriever = BinaryThenFP16RerankRetriever(
        candidate_retriever=binary_retriever,
        vector_store=vector_store,
        prefilter_top_n=rerank_depth,
    )
    fp16_retriever = FP16ReferenceRetriever(vector_store=vector_store)
    return fp16_retriever, binary_retriever, staged_retriever


def run_phase2(
    fixture_dir: Path,
    out_json: Path,
    *,
    cache_state: str,
    threads: int,
    rerank_depth: int,
) -> dict[str, object]:
    if cache_state not in {"cold", "warm"}:
        raise ValueError("cache_state must be 'cold' or 'warm'")
    if threads < 1:
        raise ValueError("threads must be >= 1")

    corpus_ids, docs = _load_corpus(fixture_dir / "corpus_fp16.npz")
    query_ids, queries = _load_queries(fixture_dir / "queries_fp16.npz")
    fp16_docs = [doc.fp16.astype(np.float32, copy=False) for doc in docs]
    calibrator, binary_docs = _build_binary_docs(fp16_docs)
    query_bins = [calibrator.apply(_pad_to_binary_dim(query)) for query in queries]

    fp16_latencies: list[float] = []
    binary_latencies: list[float] = []
    kernel_latencies: list[float] = []
    coarse_counts: list[int] = []

    warm_retrievers = None
    if cache_state == "warm":
        warm_retrievers = _build_retrievers(
            corpus_ids,
            fp16_docs,
            binary_docs,
            rerank_depth=rerank_depth,
            threads=threads,
        )

    for query_fp16, query_bin in zip(queries, query_bins, strict=True):
        query = QueryBundle(query_bin=query_bin, query_fp16=query_fp16)
        if cache_state == "cold":
            fp16_retriever, binary_retriever, staged_retriever = _build_retrievers(
                corpus_ids,
                fp16_docs,
                binary_docs,
                rerank_depth=rerank_depth,
                threads=threads,
            )
        else:
            assert warm_retrievers is not None
            fp16_retriever, binary_retriever, staged_retriever = warm_retrievers
            fp16_retriever.search(QueryBundle(query_fp16=query_fp16), k=10)
            binary_retriever.search(query, k=rerank_depth)
            staged_retriever.search(query, k=10)

        started = time.perf_counter()
        fp16_retriever.search(QueryBundle(query_fp16=query_fp16), k=10)
        fp16_latencies.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        coarse = binary_retriever.search(query, k=rerank_depth)
        binary_latencies.append((time.perf_counter() - started) * 1000.0)
        coarse_counts.append(len(coarse))

        started = time.perf_counter()
        query_u64 = _binary_scalar_to_u64_tokens(query_bin)
        for doc_bin in binary_docs:
            maxsim(query_u64, _binary_scalar_to_u64_tokens(doc_bin))
        kernel_latencies.append((time.perf_counter() - started) * 1000.0)

        staged_retriever.search(query, k=10)

    ffi_overhead_ms = float(max(np.mean(binary_latencies) - np.mean(kernel_latencies), 0.0))
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path.cwd()).strip()
    )
    command = " ".join(
        [
            "python",
            "-m",
            "alert_triage.evals.tier1_phase2",
            "--fixture-dir",
            str(fixture_dir),
            "--out-json",
            str(out_json),
            "--cache-state",
            cache_state,
            "--threads",
            str(threads),
            "--rerank-depth",
            str(rerank_depth),
        ]
    )

    report = {
        "fixture_id": fixture_dir.name,
        "corpus_size": len(corpus_ids),
        "query_count": len(query_ids),
        "cache_state": cache_state,
        "thread_count": threads,
        "fp16_p50_ms": _latency_percentile(fp16_latencies, 50),
        "fp16_p95_ms": _latency_percentile(fp16_latencies, 95),
        "binary_p50_ms": _latency_percentile(binary_latencies, 50),
        "binary_p95_ms": _latency_percentile(binary_latencies, 95),
        "kernel_ms": float(np.mean(kernel_latencies) if kernel_latencies else 0.0),
        "ffi_overhead_ms": ffi_overhead_ms,
        "binary_candidate_count": int(np.mean(coarse_counts) if coarse_counts else 0),
        "rerank_depth": rerank_depth,
        "command": command,
        "git_commit": git_commit,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    args = _parse_args()
    run_phase2(
        args.fixture_dir,
        args.out_json,
        cache_state=args.cache_state,
        threads=args.threads,
        rerank_depth=args.rerank_depth,
    )


if __name__ == "__main__":
    main()
