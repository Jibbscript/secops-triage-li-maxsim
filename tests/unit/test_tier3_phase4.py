from __future__ import annotations

from pathlib import Path

from alert_triage.evals.tier2_phase4 import run_phase4_reasoning
from alert_triage.evals.tier3_phase4 import run_phase4_disposition


def test_phase4_disposition_harness_scores_predictions(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/phase1_tier1")
    tier2_json = tmp_path / "tier2.json"
    tier2_trace = tmp_path / "tier2_trace.jsonl"
    tier3_json = tmp_path / "tier3.json"

    run_phase4_reasoning(
        fixture_dir,
        tier2_json,
        tier2_trace,
        threads=1,
        rerank_depth=2,
        llm_model="local:deterministic-triager-v1",
        judge_model="local:deterministic-judge-v1",
    )
    report = run_phase4_disposition(fixture_dir, tier2_json, tier3_json)

    assert report["experiment_id"] == "tier3-phase4"
    assert report["sample_count"] == 2
    assert report["metrics"]["weighted_f1"] == 1.0
    assert report["metrics"]["cohens_kappa"] == 1.0
    assert report["per_label"]["credential_reset"]["support"] == 1
    assert report["confusion_matrix"]["contain_host"]["contain_host"] == 1
