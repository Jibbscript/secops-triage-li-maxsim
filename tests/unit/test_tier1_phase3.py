from __future__ import annotations

import csv
import json
from pathlib import Path

from alert_triage.evals.tier1_phase3 import run_phase3


def test_phase3_harness_writes_multi_retriever_outputs(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/phase1_tier1")
    out_json = tmp_path / "tier1_phase3.json"
    out_latency = tmp_path / "latency_phase3.csv"

    report = run_phase3(fixture_dir, out_json, out_latency, threads=1, rerank_depth=2)

    assert report["experiment_id"] == "tier1-phase3"
    assert set(report["retrievers"]) == {
        "bm25",
        "lance-mv",
        "hamming-udf-bin",
        "binary-then-fp16-rerank",
        "hybrid-bm25-then-fp16-rerank",
    }
    assert report["query_count"] == 2
    assert report["metrics"]["hybrid-bm25-then-fp16-rerank"]["rerank_ms"] >= 0.0
    assert report["metrics"]["binary-then-fp16-rerank"]["rerank_ms"] >= 0.0
    assert "rerank_ms" not in report["metrics"]["bm25"]
    assert "rerank_ms" not in report["metrics"]["lance-mv"]
    assert "rerank_ms" not in report["metrics"]["hamming-udf-bin"]

    payload = json.loads(out_json.read_text())
    assert set(payload) >= {"experiment_id", "retrievers", "query_count", "metrics"}
    assert set(payload["metrics"]["bm25"]) >= {"recall@10", "ndcg@10", "mrr", "p50_ms", "p95_ms"}

    with out_latency.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["retriever", "query_id", "k", "stage", "latency_ms"]
        rows = list(reader)

    assert len(rows) == 10
    assert {row["retriever"] for row in rows} == set(report["retrievers"])
