from __future__ import annotations

import sys
from pathlib import Path

import pytest

from alert_triage.triage import (
    EvidenceHit,
    InvestigationStepToolRuntime,
    MCPTerminalToolRuntime,
    ReplayJudge,
    ReplayTriageEngine,
    RuleBasedJudge,
    RuleBasedTriageEngine,
    load_replay_runtime_fixture,
)
from alert_triage.triage.reasoning import ReasoningTarget, build_reasoning_sample


def _mcp_server_command(mode: str = "ok") -> tuple[str, ...]:
    return (sys.executable, "tests/fixtures/mcp/fake_terminal_server.py", mode)


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
    assert [record.inputs.get("role") for record in sample.audit_records[:2]] == ["triager", "judge"]
    assert sample.sample_metrics["action_validity"] == 1.0
    assert sample.sample_metrics["evidence_grounding"] == 1.0


def test_replay_runtime_emits_fixture_backed_audit_records() -> None:
    fixture = load_replay_runtime_fixture(Path("tests/fixtures/phase1_tier1/replay_runtime.json"))
    sample = build_reasoning_sample(
        query_id="query-1",
        query_text="Investigate credential phishing activity around suspicious inbox forwarding.",
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
        triager=ReplayTriageEngine(fixture=fixture),
        judge=ReplayJudge(fixture=fixture),
        terminal_tool=InvestigationStepToolRuntime(),
    )

    assert sample.action == "reset_credentials_and_scope_phishing"
    assert sample.disposition == "credential_reset"
    assert [record.name for record in sample.audit_records] == [
        "replay:fixture-triager-v1",
        "replay:fixture-judge-v1",
        "propose_investigation_step",
    ]
    assert [record.inputs.get("role") for record in sample.audit_records[:2]] == ["triager", "judge"]
    assert sample.sample_metrics["specificity"] == 1.0


def test_replay_runtime_raises_for_missing_query_fixture() -> None:
    fixture = load_replay_runtime_fixture(Path("tests/fixtures/phase1_tier1/replay_runtime.json"))
    triager = ReplayTriageEngine(fixture=fixture)

    with pytest.raises(ValueError, match="missing replay decision fixture"):
        triager.decide(
            query_id="missing-query",
            query_text="unknown",
            evidence=(),
        )


def test_reasoning_can_emit_terminal_audit_via_mcp_runtime() -> None:
    runtime = MCPTerminalToolRuntime(server_command=_mcp_server_command())
    try:
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
            terminal_tool=runtime,
        )
    finally:
        runtime.close()

    assert sample.terminal_tool == "propose_investigation_step"
    assert sample.audit_records[-1].name == "propose_investigation_step"
    assert sample.audit_records[-1].outputs["structuredContent"] == {
        "accepted_action": "reset_credentials_and_scope_phishing",
        "accepted_disposition": "credential_reset",
    }
