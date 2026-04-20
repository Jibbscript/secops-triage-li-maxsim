from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict

from .audit import AuditRecord


class ReasoningTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_action: str
    expected_disposition: str
    expected_evidence_ids: tuple[str, ...]
    rationale_keyword: str


class EvidenceHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str
    score: float
    stage: str
    text: str


class TriageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    query_text: str
    action: str
    disposition: str
    confidence: float
    rationale: str
    cited_evidence_ids: tuple[str, ...]


class ReasoningSampleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    query_text: str
    retrieved_alert_ids: tuple[str, ...]
    evidence: tuple[EvidenceHit, ...]
    action: str
    disposition: str
    confidence: float
    rationale: str
    cited_evidence_ids: tuple[str, ...]
    terminal_output_kind: str = "tool_call"
    terminal_tool: str = "propose_investigation_step"
    sample_metrics: dict[str, float]
    audit_records: tuple[AuditRecord, ...]


class ReplayRuntimeSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: TriageDecision
    scores: dict[str, float]


class ReplayRuntimeFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_model: str
    judge_model: str
    samples: dict[str, ReplayRuntimeSample]


class TriageRuntime(Protocol):
    model_id: str

    def decide(
        self,
        *,
        query_id: str,
        query_text: str,
        evidence: Sequence[EvidenceHit],
    ) -> tuple[TriageDecision, AuditRecord]: ...


class JudgeRuntime(Protocol):
    model_id: str

    def evaluate(
        self,
        *,
        query_id: str,
        query_text: str,
        decision: TriageDecision,
        evidence: Sequence[EvidenceHit],
        target: "ReasoningTarget",
    ) -> tuple[dict[str, float], AuditRecord]: ...


class TerminalToolRuntime(Protocol):
    tool_name: str

    def emit(
        self,
        *,
        query_id: str,
        query_text: str,
        decision: TriageDecision,
    ) -> AuditRecord: ...


def load_reasoning_targets(path: Path) -> dict[str, ReasoningTarget]:
    payload = json.loads(path.read_text())
    return {
        str(query_id): ReasoningTarget.model_validate(target)
        for query_id, target in payload.items()
    }


def load_replay_runtime_fixture(path: Path) -> ReplayRuntimeFixture:
    return ReplayRuntimeFixture.model_validate(json.loads(path.read_text()))


def _replay_sample(fixture: ReplayRuntimeFixture, query_id: str, *, kind: str) -> ReplayRuntimeSample:
    sample = fixture.samples.get(query_id)
    if sample is None:
        raise ValueError(f"missing replay {kind} fixture for {query_id}")
    return sample


class RuleBasedTriageEngine:
    default_model_id = "local:deterministic-triager-v1"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or self.default_model_id

    def decide(self, *, query_id: str, query_text: str, evidence: Sequence[EvidenceHit]) -> tuple[TriageDecision, AuditRecord]:
        lowered_query = query_text.lower()
        evidence_text = " ".join(hit.text.lower() for hit in evidence)
        top_hit_ids = tuple(hit.alert_id for hit in evidence[:2])

        if "phishing" in lowered_query or "credential" in lowered_query:
            action = "reset_credentials_and_scope_phishing"
            disposition = "credential_reset"
            confidence = 0.93
            rationale = "Phishing credential evidence supports a scoped credential reset investigation."
        elif "ransomware" in lowered_query or "beacon" in lowered_query:
            action = "isolate_host_and_block_beacon"
            disposition = "contain_host"
            confidence = 0.96
            rationale = "Ransomware beacon evidence requires host containment before deeper investigation."
        elif "ransomware" in evidence_text or "beacon" in evidence_text:
            action = "isolate_host_and_block_beacon"
            disposition = "contain_host"
            confidence = 0.9
            rationale = "Retrieved ransomware beacon evidence points to containment as the next investigation step."
        elif "phishing" in evidence_text or "credential" in evidence_text:
            action = "reset_credentials_and_scope_phishing"
            disposition = "credential_reset"
            confidence = 0.88
            rationale = "Retrieved phishing credential evidence supports a scoped credential reset investigation."
        else:
            action = "collect_more_context"
            disposition = "needs_triage"
            confidence = 0.55
            rationale = "The retrieved evidence is not specific enough yet, so the next step is bounded investigation."

        cited_evidence_ids = top_hit_ids if top_hit_ids else tuple(hit.alert_id for hit in evidence[:1])
        payload = TriageDecision(
            query_id=query_id,
            query_text=query_text,
            action=action,
            disposition=disposition,
            confidence=confidence,
            rationale=rationale,
            cited_evidence_ids=cited_evidence_ids,
        )
        audit = AuditRecord(
            kind="model_call",
            name=self.model_id,
            inputs={
                "query_text": query_text,
                "retrieved_alert_ids": [hit.alert_id for hit in evidence],
            },
            outputs=payload.model_dump(mode="json"),
        )
        return payload, audit


class RuleBasedJudge:
    default_model_id = "local:deterministic-judge-v1"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or self.default_model_id

    def evaluate(
        self,
        *,
        query_id: str,
        query_text: str,
        decision: TriageDecision,
        evidence: Sequence[EvidenceHit],
        target: ReasoningTarget,
    ) -> tuple[dict[str, float], AuditRecord]:
        retrieved_ids = {hit.alert_id for hit in evidence}
        cited_ids = decision.cited_evidence_ids
        action_validity = 1.0 if decision.action == target.expected_action else 0.0
        grounded_hits = sum(alert_id in target.expected_evidence_ids for alert_id in cited_ids)
        evidence_grounding = grounded_hits / len(target.expected_evidence_ids) if target.expected_evidence_ids else 0.0
        if any(alert_id not in retrieved_ids for alert_id in cited_ids):
            evidence_grounding = 0.0

        rationale = decision.rationale.lower()
        specificity = 1.0 if target.rationale_keyword.lower() in rationale and len(cited_ids) > 0 else 0.0
        correctness = 1.0 if (
            decision.action == target.expected_action
            and decision.disposition == target.expected_disposition
        ) else 0.0
        calibration = max(0.0, 1.0 - abs(decision.confidence - correctness))
        scores = {
            "action_validity": action_validity,
            "evidence_grounding": evidence_grounding,
            "specificity": specificity,
            "calibration": calibration,
        }
        audit = AuditRecord(
            kind="model_call",
            name=self.model_id,
            inputs={
                "query_id": query_id,
                "query_text": query_text,
                "retrieved_alert_ids": [hit.alert_id for hit in evidence],
            },
            outputs=scores,
        )
        return scores, audit


class ReplayTriageEngine:
    default_model_id = "replay:fixture-triager-v1"

    def __init__(self, fixture: ReplayRuntimeFixture, model_id: str | None = None) -> None:
        self._fixture = fixture
        self.model_id = model_id or fixture.llm_model or self.default_model_id

    def decide(self, *, query_id: str, query_text: str, evidence: Sequence[EvidenceHit]) -> tuple[TriageDecision, AuditRecord]:
        sample = _replay_sample(self._fixture, query_id, kind="decision")
        payload = sample.decision
        audit = AuditRecord(
            kind="model_call",
            name=self.model_id,
            inputs={
                "query_text": query_text,
                "retrieved_alert_ids": [hit.alert_id for hit in evidence],
            },
            outputs=payload.model_dump(mode="json"),
        )
        return payload, audit


class ReplayJudge:
    default_model_id = "replay:fixture-judge-v1"

    def __init__(self, fixture: ReplayRuntimeFixture, model_id: str | None = None) -> None:
        self._fixture = fixture
        self.model_id = model_id or fixture.judge_model or self.default_model_id

    def evaluate(
        self,
        *,
        query_id: str,
        query_text: str,
        decision: TriageDecision,
        evidence: Sequence[EvidenceHit],
        target: ReasoningTarget,
    ) -> tuple[dict[str, float], AuditRecord]:
        sample = _replay_sample(self._fixture, query_id, kind="judge")
        scores = {metric: float(value) for metric, value in sample.scores.items()}
        audit = AuditRecord(
            kind="model_call",
            name=self.model_id,
            inputs={
                "query_id": query_id,
                "query_text": query_text,
                "retrieved_alert_ids": [hit.alert_id for hit in evidence],
            },
            outputs=scores,
        )
        return scores, audit


class InvestigationStepToolRuntime:
    tool_name = "propose_investigation_step"

    def emit(
        self,
        *,
        query_id: str,
        query_text: str,
        decision: TriageDecision,
    ) -> AuditRecord:
        return AuditRecord(
            kind="tool_call",
            name=self.tool_name,
            inputs={
                "query_id": query_id,
                "query_text": query_text,
            },
            outputs={
                "action": decision.action,
                "disposition": decision.disposition,
                "cited_evidence_ids": decision.cited_evidence_ids,
            },
        )


def build_reasoning_sample(
    *,
    query_id: str,
    query_text: str,
    evidence: Sequence[EvidenceHit],
    target: ReasoningTarget,
    triager: TriageRuntime,
    judge: JudgeRuntime,
    terminal_tool: TerminalToolRuntime | None = None,
) -> ReasoningSampleResult:
    terminal_tool = terminal_tool or InvestigationStepToolRuntime()
    decision, triager_audit = triager.decide(query_id=query_id, query_text=query_text, evidence=evidence)
    sample_metrics, judge_audit = judge.evaluate(
        query_id=query_id,
        query_text=query_text,
        decision=decision,
        evidence=evidence,
        target=target,
    )
    tool_audit = terminal_tool.emit(query_id=query_id, query_text=query_text, decision=decision)
    return ReasoningSampleResult(
        query_id=query_id,
        query_text=query_text,
        retrieved_alert_ids=tuple(hit.alert_id for hit in evidence),
        evidence=tuple(evidence),
        action=decision.action,
        disposition=decision.disposition,
        confidence=decision.confidence,
        rationale=decision.rationale,
        cited_evidence_ids=decision.cited_evidence_ids,
        sample_metrics=sample_metrics,
        audit_records=(triager_audit, judge_audit, tool_audit),
    )
