from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from alert_triage.encoders.base import EncodedTokens
from alert_triage.retrievers.base import QueryBundle
from alert_triage.retrievers.fp16_ref import FP16ReferenceRetriever


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase-1 tier-1 fp16 fixture harness.")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-latency", type=Path, required=True)
    return parser.parse_args()


def _load_corpus(path: Path) -> tuple[list[str], list[EncodedTokens]]:
    with np.load(path, allow_pickle=False) as data:
        doc_ids = [str(doc_id) for doc_id in data["doc_ids"].tolist()]
        docs = [
            EncodedTokens(fp16=np.asarray(doc_tokens, dtype=np.float32))
            for doc_tokens in data["doc_tokens"]
        ]
    return doc_ids, docs


def _load_queries(path: Path) -> tuple[list[str], list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        query_ids = [str(query_id) for query_id in data["query_ids"].tolist()]
        queries = [np.asarray(query_tokens, dtype=np.float32) for query_tokens in data["query_tokens"]]
    return query_ids, queries


def _load_qrels(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    return {str(query_id): [str(doc_id) for doc_id in doc_ids] for query_id, doc_ids in payload.items()}


def _recall_at_k(expected: list[str], actual: list[str], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(expected).intersection(actual[:k])) / len(expected)


def _reciprocal_rank(expected: list[str], actual: list[str]) -> float:
    expected_set = set(expected)
    for rank, doc_id in enumerate(actual, start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(expected: list[str], actual: list[str], k: int) -> float:
    expected_set = set(expected)
    dcg = 0.0
    for rank, doc_id in enumerate(actual[:k], start=1):
        if doc_id in expected_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(expected_set), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


def _latency_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def run_phase1(fixture_dir: Path, out_json: Path, out_latency: Path) -> dict[str, object]:
    corpus_ids, docs = _load_corpus(fixture_dir / "corpus_fp16.npz")
    query_ids, queries = _load_queries(fixture_dir / "queries_fp16.npz")
    qrels = _load_qrels(fixture_dir / "qrels.json")

    retriever = FP16ReferenceRetriever()
    retriever.index(corpus_ids, docs)

    retrievals: dict[str, list[str]] = {}
    latency_rows: list[dict[str, object]] = []

    for query_id, query_tokens in zip(query_ids, queries, strict=True):
        started = time.perf_counter()
        hits = retriever.search(QueryBundle(query_fp16=query_tokens), k=10)
        latency_ms = (time.perf_counter() - started) * 1000.0
        retrievals[query_id] = [hit.alert_id for hit in hits]
        latency_rows.append(
            {
                "retriever": retriever.id,
                "query_id": query_id,
                "k": 10,
                "latency_ms": latency_ms,
            }
        )

    recalls = [_recall_at_k(qrels[query_id], retrievals[query_id], 10) for query_id in query_ids]
    ndcgs = [_ndcg_at_k(qrels[query_id], retrievals[query_id], 10) for query_id in query_ids]
    mrrs = [_reciprocal_rank(qrels[query_id], retrievals[query_id]) for query_id in query_ids]
    latencies = [float(row["latency_ms"]) for row in latency_rows]

    report = {
        "experiment_id": "tier1-phase1",
        "retriever": retriever.id,
        "query_count": len(query_ids),
        "metrics": {
            "recall@10": float(np.mean(recalls) if recalls else 0.0),
            "ndcg@10": float(np.mean(ndcgs) if ndcgs else 0.0),
            "mrr": float(np.mean(mrrs) if mrrs else 0.0),
            "p50_ms": _latency_percentile(latencies, 50),
            "p95_ms": _latency_percentile(latencies, 95),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    out_latency.parent.mkdir(parents=True, exist_ok=True)
    with out_latency.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["retriever", "query_id", "k", "latency_ms"])
        writer.writeheader()
        writer.writerows(latency_rows)

    return report


def main() -> None:
    args = _parse_args()
    run_phase1(args.fixture_dir, args.out_json, args.out_latency)


if __name__ == "__main__":
    main()
