from __future__ import annotations

import pytest

from alert_triage.triage import compute_disposition_metrics


def test_disposition_metrics_score_perfect_predictions() -> None:
    metrics = compute_disposition_metrics(
        {"q1": "credential_reset", "q2": "contain_host"},
        {"q1": "credential_reset", "q2": "contain_host"},
    )

    assert metrics["weighted_f1"] == 1.0
    assert metrics["cohens_kappa"] == 1.0
    assert metrics["per_label"]["credential_reset"]["support"] == 1
    assert metrics["confusion_matrix"]["contain_host"]["contain_host"] == 1


def test_disposition_metrics_reject_query_id_mismatch() -> None:
    with pytest.raises(ValueError, match="query-id mismatch"):
        compute_disposition_metrics(
            {"q1": "credential_reset", "q2": "contain_host"},
            {"q1": "credential_reset"},
        )
