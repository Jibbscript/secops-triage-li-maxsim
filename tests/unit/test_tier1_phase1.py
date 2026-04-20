from __future__ import annotations

import csv
import json
from pathlib import Path

from alert_triage.evals.tier1_phase1 import run_phase1


def test_phase1_harness_writes_bundle_compatible_outputs(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/phase1_tier1")
    out_json = tmp_path / "tier1.json"
    out_latency = tmp_path / "latency.csv"

    report = run_phase1(fixture_dir, out_json, out_latency)

    assert report["experiment_id"] == "tier1-phase1"
    assert report["retriever"] == "fp16-ref"
    assert report["query_count"] == 2

    payload = json.loads(out_json.read_text())
    assert set(payload) >= {"experiment_id", "retriever", "query_count", "metrics"}
    assert set(payload["metrics"]) >= {"recall@10", "ndcg@10", "mrr", "p50_ms", "p95_ms"}
    assert payload["metrics"]["recall@10"] > 0.0

    with out_latency.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["retriever", "query_id", "k", "latency_ms"]
        rows = list(reader)

    assert len(rows) == 2
    assert {row["retriever"] for row in rows} == {"fp16-ref"}
