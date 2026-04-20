from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from alert_triage.encoders.base import EncodedTokens
from alert_triage.evals.tier1_phase1 import (
    _latency_percentile,
    _load_corpus,
    _load_qrels,
    _load_queries,
    _ndcg_at_k,
    _recall_at_k,
    _reciprocal_rank,
)
from alert_triage.evals.tier1_phase2 import _build_binary_docs, _build_context, _pad_to_binary_dim
from alert_triage.retrievers.base import QueryBundle
from alert_triage.retrievers.binary_rerank import BinaryThenFP16RerankRetriever
from alert_triage.retrievers.bm25 import BM25Retriever
from alert_triage.retrievers.hamming_udf import HammingUDFRetriever
from alert_triage.retrievers.hybrid_rerank import HybridBM25ThenFP16RerankRetriever
from alert_triage.retrievers.lance_mv import LanceMVRetriever
from alert_triage.storage.in_memory import InMemoryTokenVectorStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase-3 retrieval comparison harness.")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-latency", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rerank-depth", type=int, default=5)
    return parser.parse_args()


def _load_text_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    return {str(key): str(value) for key, value in payload.items()}


def _build_retrievers(
    fixture_dir: Path,
    corpus_ids: list[str],
    docs: list[EncodedTokens],
    texts: list[str],
    binary_docs: list[np.ndarray],
    *,
    threads: int,
    rerank_depth: int,
):
    vector_store = InMemoryTokenVectorStore()
    vector_store.upsert(
        corpus_ids,
        [
            EncodedTokens(fp16=doc.require_fp16(), binary=doc_bin)
            for doc, doc_bin in zip(docs, binary_docs, strict=True)
        ],
    )

    bm25 = BM25Retriever()
    bm25.index(corpus_ids, docs, texts=texts)

    ctx, rowid_to_alert_id = _build_context(corpus_ids, binary_docs, threads=threads)
    binary = HammingUDFRetriever(ctx=ctx, rowid_to_alert_id=rowid_to_alert_id.get)
    binary_rerank = BinaryThenFP16RerankRetriever(
        candidate_retriever=binary,
        vector_store=vector_store,
        prefilter_top_n=rerank_depth,
    )
    hybrid = HybridBM25ThenFP16RerankRetriever(
        candidate_retriever=bm25,
        vector_store=vector_store,
        prefilter_top_n=rerank_depth,
    )

    tmpdir = tempfile.TemporaryDirectory(prefix="phase3-lance-", dir=fixture_dir)
    lance = LanceMVRetriever(dataset_path=Path(tmpdir.name) / "corpus_fp16.lance")
    lance.index(corpus_ids, docs, texts=texts)
    return {"bm25": bm25, "lance-mv": lance, "hamming-udf-bin": binary, "binary-then-fp16-rerank": binary_rerank, "hybrid-bm25-then-fp16-rerank": hybrid}, tmpdir


def run_phase3(
    fixture_dir: Path,
    out_json: Path,
    out_latency: Path,
    *,
    threads: int,
    rerank_depth: int,
) -> dict[str, object]:
    if threads < 1:
        raise ValueError("threads must be >= 1")

    corpus_ids, docs = _load_corpus(fixture_dir / "corpus_fp16.npz")
    query_ids, queries = _load_queries(fixture_dir / "queries_fp16.npz")
    qrels = _load_qrels(fixture_dir / "qrels.json")
    corpus_text_map = _load_text_map(fixture_dir / "corpus_texts.json")
    query_text_map = _load_text_map(fixture_dir / "query_texts.json")
    corpus_texts = [corpus_text_map[doc_id] for doc_id in corpus_ids]
    query_texts = [query_text_map[query_id] for query_id in query_ids]

    fp16_docs = [doc.require_fp16().astype(np.float32, copy=False) for doc in docs]
    calibrator, binary_docs = _build_binary_docs(fp16_docs)
    query_bins = [
        calibrator.apply(_pad_to_binary_dim(query.astype(np.float32, copy=False))) for query in queries
    ]
    retrievers, lance_tmpdir = _build_retrievers(
        fixture_dir,
        corpus_ids,
        docs,
        corpus_texts,
        binary_docs,
        threads=threads,
        rerank_depth=rerank_depth,
    )

    retrievals: dict[str, dict[str, list[str]]] = {name: {} for name in retrievers}
    latencies: dict[str, list[float]] = {name: [] for name in retrievers}
    rerank_latencies: dict[str, list[float]] = {
        "binary-then-fp16-rerank": [],
        "hybrid-bm25-then-fp16-rerank": [],
    }
    latency_rows: list[dict[str, object]] = []

    try:
        for query_id, query_fp16, query_bin, query_text in zip(
            query_ids, queries, query_bins, query_texts, strict=True
        ):
            query_variants = {
                "bm25": QueryBundle(query_text=query_text),
                "lance-mv": QueryBundle(query_fp16=query_fp16),
                "hamming-udf-bin": QueryBundle(query_bin=query_bin),
                "binary-then-fp16-rerank": QueryBundle(query_bin=query_bin, query_fp16=query_fp16),
                "hybrid-bm25-then-fp16-rerank": QueryBundle(
                    query_text=query_text,
                    query_fp16=query_fp16,
                ),
            }

            for retriever_name, retriever in retrievers.items():
                query = query_variants[retriever_name]
                if hasattr(retriever, "search_with_timings"):
                    started = time.perf_counter()
                    hits, timings = retriever.search_with_timings(query, k=10)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    rerank_latencies[retriever_name].append(float(timings["rerank_ms"]))
                else:
                    started = time.perf_counter()
                    hits = retriever.search(query, k=10)
                    latency_ms = (time.perf_counter() - started) * 1000.0

                retrievals[retriever_name][query_id] = [hit.alert_id for hit in hits]
                latencies[retriever_name].append(latency_ms)
                latency_rows.append(
                    {
                        "retriever": retriever_name,
                        "query_id": query_id,
                        "k": 10,
                        "stage": hits[0].stage if hits else "none",
                        "latency_ms": latency_ms,
                    }
                )
    finally:
        lance_tmpdir.cleanup()

    metrics: dict[str, dict[str, float]] = {}
    for retriever_name, query_hits in retrievals.items():
        recalls = [_recall_at_k(qrels[query_id], query_hits[query_id], 10) for query_id in query_ids]
        ndcgs = [_ndcg_at_k(qrels[query_id], query_hits[query_id], 10) for query_id in query_ids]
        mrrs = [_reciprocal_rank(qrels[query_id], query_hits[query_id]) for query_id in query_ids]
        metrics[retriever_name] = {
            "recall@10": float(np.mean(recalls) if recalls else 0.0),
            "ndcg@10": float(np.mean(ndcgs) if ndcgs else 0.0),
            "mrr": float(np.mean(mrrs) if mrrs else 0.0),
            "p50_ms": _latency_percentile(latencies[retriever_name], 50),
            "p95_ms": _latency_percentile(latencies[retriever_name], 95),
        }
        if retriever_name in rerank_latencies:
            metrics[retriever_name]["rerank_ms"] = float(
                np.mean(rerank_latencies[retriever_name]) if rerank_latencies[retriever_name] else 0.0
            )

    report = {
        "experiment_id": "tier1-phase3",
        "retrievers": list(retrievers),
        "query_count": len(query_ids),
        "metrics": metrics,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    out_latency.parent.mkdir(parents=True, exist_ok=True)
    with out_latency.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["retriever", "query_id", "k", "stage", "latency_ms"],
        )
        writer.writeheader()
        writer.writerows(latency_rows)

    return report


def main() -> None:
    args = _parse_args()
    run_phase3(
        args.fixture_dir,
        args.out_json,
        args.out_latency,
        threads=args.threads,
        rerank_depth=args.rerank_depth,
    )


if __name__ == "__main__":
    main()
