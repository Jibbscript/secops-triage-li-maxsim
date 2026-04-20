from __future__ import annotations

from alert_triage.triage import EvidenceHit, RuleBasedJudge, RuleBasedTriageEngine
from alert_triage.triage.reasoning import ReasoningTarget, build_reasoning_sample


def test_rule_based_reasoning_emits_terminal_tool_call_and_audit_records() -> None:
    sample = build_reasoning_sample(
        query_id="query-1",
        query_text="phishing credential reset",
        evidence=(
            EvidenceHit(
                alert_id="alert-1",
                score=1.0,
                stage="fp16_maxsim",
                text="phishing credential reset playbook",
            ),
        ),
        target=ReasoningTarget(
            expected_action="reset_credentials_and_scope_phishing",
            expected_disposition="credential_reset",
            expected_evidence_ids=("alert-1",),
            rationale_keyword="phishing",
        ),
        triager=RuleBasedTriageEngine(),
        judge=RuleBasedJudge(),
    )

    assert sample.action == "reset_credentials_and_scope_phishing"
    assert sample.disposition == "credential_reset"
    assert sample.terminal_output_kind == "tool_call"
    assert sample.terminal_tool == "propose_investigation_step"
    assert [record.kind for record in sample.audit_records] == ["model_call", "model_call", "tool_call"]
    assert sample.audit_records[-1].name == "propose_investigation_step"
    assert sample.sample_metrics["action_validity"] == 1.0
    assert sample.sample_metrics["evidence_grounding"] == 1.0
