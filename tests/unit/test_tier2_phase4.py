from __future__ import annotations

import json
from pathlib import Path

from alert_triage.evals.tier2_phase4 import run_phase4_reasoning


def test_phase4_reasoning_harness_writes_report_and_trace(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/phase1_tier1")
    out_json = tmp_path / "tier2.json"
    out_trace = tmp_path / "tier2_trace.jsonl"

    report = run_phase4_reasoning(
        fixture_dir,
        out_json,
        out_trace,
        threads=1,
        rerank_depth=2,
        llm_model="local:deterministic-triager-v1",
        judge_model="local:deterministic-judge-v1",
    )

    assert report["experiment_id"] == "tier2-phase4"
    assert report["retriever"] == "binary-then-fp16-rerank"
    assert report["sample_count"] == 2
    assert report["terminal_output_kind"] == "tool_call"
    assert report["terminal_tool"] == "propose_investigation_step"
    assert report["orphan_llm_calls"] == 0
    assert report["orphan_tool_calls"] == 0
    assert set(report["metrics"]) == {
        "action_validity",
        "evidence_grounding",
        "specificity",
        "calibration",
    }

    payload = json.loads(out_json.read_text())
    assert set(payload) >= {
        "experiment_id",
        "retriever",
        "sample_count",
        "llm_model",
        "judge_model",
        "metrics",
        "terminal_output_kind",
        "terminal_tool",
        "orphan_llm_calls",
        "orphan_tool_calls",
        "samples",
    }

    lines = out_trace.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["terminal_output_kind"] == "tool_call"
    assert first["terminal_tool"] == "propose_investigation_step"
    assert len(first["audit_records"]) == 3
