from __future__ import annotations

import json

import pytest

from alert_triage.triage import (
    AnthropicJudge,
    AnthropicTriageEngine,
    EvidenceHit,
    OpenAIResponsesJudge,
    OpenAIResponsesTriageEngine,
    ReasoningTarget,
    TriageDecision,
)


class FakeTransport:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self._responses:
            raise AssertionError("unexpected provider request")
        return self._responses.pop(0)


def _evidence() -> tuple[EvidenceHit, ...]:
    return (
        EvidenceHit(
            alert_id="alert-1",
            score=1.0,
            stage="fp16_maxsim",
            text="phishing credential reset playbook",
        ),
    )


def _decision() -> TriageDecision:
    return TriageDecision(
        query_id="query-1",
        query_text="Investigate credential phishing activity around suspicious inbox forwarding.",
        action="reset_credentials_and_scope_phishing",
        disposition="credential_reset",
        confidence=0.91,
        rationale="Phishing evidence supports a scoped credential reset.",
        cited_evidence_ids=("alert-1",),
    )


def _target() -> ReasoningTarget:
    return ReasoningTarget(
        expected_action="reset_credentials_and_scope_phishing",
        expected_disposition="credential_reset",
        expected_evidence_ids=("alert-1",),
        rationale_keyword="phishing",
    )


def test_openai_triage_adapter_normalizes_function_call_payload() -> None:
    transport = FakeTransport(
        [
            {
                "id": "resp_openai_triage",
                "output": [
                    {
                        "type": "function_call",
                        "name": "emit_triage_decision",
                        "arguments": json.dumps(_decision().model_dump(mode="json")),
                    }
                ],
            }
        ]
    )
    triager = OpenAIResponsesTriageEngine(
        model_id="openai:gpt-5",
        api_key="sk-test",
        transport=transport,
    )

    decision, audit = triager.decide(
        query_id="query-1",
        query_text="Investigate credential phishing activity around suspicious inbox forwarding.",
        evidence=_evidence(),
    )

    assert decision.action == "reset_credentials_and_scope_phishing"
    assert decision.cited_evidence_ids == ("alert-1",)
    assert audit.name == "openai:gpt-5"
    call = transport.calls[0]
    assert call["url"] == "https://api.openai.com/v1/responses"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["payload"]["model"] == "gpt-5"
    assert call["payload"]["tool_choice"] == {"type": "function", "name": "emit_triage_decision"}
    assert "alert-1" in call["payload"]["input"]


def test_openai_judge_adapter_normalizes_scores() -> None:
    transport = FakeTransport(
        [
            {
                "id": "resp_openai_judge",
                "output": [
                    {
                        "type": "function_call",
                        "name": "emit_reasoning_scores",
                        "arguments": json.dumps(
                            {
                                "action_validity": 1.0,
                                "evidence_grounding": 1.0,
                                "specificity": 1.0,
                                "calibration": 0.91,
                            }
                        ),
                    }
                ],
            }
        ]
    )
    judge = OpenAIResponsesJudge(
        model_id="openai:gpt-5",
        api_key="sk-test",
        transport=transport,
    )

    scores, audit = judge.evaluate(
        query_id="query-1",
        query_text=_decision().query_text,
        decision=_decision(),
        evidence=_evidence(),
        target=_target(),
    )

    assert scores == {
        "action_validity": 1.0,
        "evidence_grounding": 1.0,
        "specificity": 1.0,
        "calibration": 0.91,
    }
    assert audit.name == "openai:gpt-5"
    call = transport.calls[0]
    assert call["payload"]["model"] == "gpt-5"
    assert call["payload"]["tool_choice"] == {"type": "function", "name": "emit_reasoning_scores"}
    assert "expected_action" in call["payload"]["input"]


def test_anthropic_triage_adapter_normalizes_tool_use_payload() -> None:
    transport = FakeTransport(
        [
            {
                "id": "msg_anthropic_triage",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "emit_triage_decision",
                        "input": _decision().model_dump(mode="json"),
                    }
                ],
            }
        ]
    )
    triager = AnthropicTriageEngine(
        model_id="anthropic:claude-sonnet-4-5",
        api_key="anth-test",
        transport=transport,
    )

    decision, audit = triager.decide(
        query_id="query-1",
        query_text=_decision().query_text,
        evidence=_evidence(),
    )

    assert decision.disposition == "credential_reset"
    assert audit.name == "anthropic:claude-sonnet-4-5"
    call = transport.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == "anth-test"
    assert call["payload"]["model"] == "claude-sonnet-4-5"
    assert call["payload"]["tool_choice"] == {"type": "tool", "name": "emit_triage_decision"}


def test_anthropic_judge_adapter_normalizes_scores() -> None:
    transport = FakeTransport(
        [
            {
                "id": "msg_anthropic_judge",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "emit_reasoning_scores",
                        "input": {
                            "action_validity": 1.0,
                            "evidence_grounding": 1.0,
                            "specificity": 1.0,
                            "calibration": 0.91,
                        },
                    }
                ],
            }
        ]
    )
    judge = AnthropicJudge(
        model_id="anthropic:claude-sonnet-4-5",
        api_key="anth-test",
        transport=transport,
    )

    scores, audit = judge.evaluate(
        query_id="query-1",
        query_text=_decision().query_text,
        decision=_decision(),
        evidence=_evidence(),
        target=_target(),
    )

    assert scores["action_validity"] == 1.0
    assert scores["calibration"] == 0.91
    assert audit.name == "anthropic:claude-sonnet-4-5"
    assert transport.calls[0]["payload"]["tool_choice"] == {"type": "tool", "name": "emit_reasoning_scores"}


def test_provider_adapters_require_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIResponsesTriageEngine(model_id="openai:gpt-5")


def test_provider_adapters_reject_cross_provider_model_ids() -> None:
    with pytest.raises(ValueError, match="anthropic runtime requires anthropic:<model> ids"):
        AnthropicJudge(model_id="openai:gpt-5", api_key="anth-test")


def test_openai_adapter_rejects_missing_function_call() -> None:
    transport = FakeTransport([{"id": "resp_missing", "output": [{"type": "message"}]}])
    triager = OpenAIResponsesTriageEngine(
        model_id="openai:gpt-5",
        api_key="sk-test",
        transport=transport,
    )

    with pytest.raises(ValueError, match="missing function_call"):
        triager.decide(
            query_id="query-1",
            query_text=_decision().query_text,
            evidence=_evidence(),
        )


def test_anthropic_adapter_rejects_missing_tool_use() -> None:
    transport = FakeTransport([{"id": "msg_missing", "content": [{"type": "text", "text": "no tool"}]}])
    judge = AnthropicJudge(
        model_id="anthropic:claude-sonnet-4-5",
        api_key="anth-test",
        transport=transport,
    )

    with pytest.raises(ValueError, match="missing tool_use"):
        judge.evaluate(
            query_id="query-1",
            query_text=_decision().query_text,
            decision=_decision(),
            evidence=_evidence(),
            target=_target(),
        )
