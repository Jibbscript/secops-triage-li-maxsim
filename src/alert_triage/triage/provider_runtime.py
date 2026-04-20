from __future__ import annotations

import json
import os
from typing import Any, Protocol, Sequence
from urllib import error, request

from pydantic import BaseModel, ConfigDict

from .audit import AuditRecord
from .reasoning import EvidenceHit, JudgeRuntime, ReasoningTarget, TriageDecision, TriageRuntime


class ReasoningScorePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_validity: float
    evidence_grounding: float
    specificity: float
    calibration: float

    def as_metrics(self) -> dict[str, float]:
        return {
            "action_validity": float(self.action_validity),
            "evidence_grounding": float(self.evidence_grounding),
            "specificity": float(self.specificity),
            "calibration": float(self.calibration),
        }


class JSONTransport(Protocol):
    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, Any]: ...


class UrllibJSONTransport:
    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"provider request failed with HTTP {exc.code}: {message}") from exc
        except error.URLError as exc:
            raise ValueError(f"provider request failed: {exc.reason}") from exc


def _tool_schema(
    *,
    description: str,
    schema: dict[str, object],
    tool_name: str,
    provider: str,
) -> dict[str, object]:
    if provider == "openai":
        return {
            "type": "function",
            "name": tool_name,
            "description": description,
            "parameters": schema,
            "strict": True,
        }
    if provider == "anthropic":
        return {
            "name": tool_name,
            "description": description,
            "input_schema": schema,
        }
    raise ValueError(f"unsupported provider: {provider}")


def _decision_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "query_id": {"type": "string"},
            "query_text": {"type": "string"},
            "action": {"type": "string"},
            "disposition": {"type": "string"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
            "cited_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "query_id",
            "query_text",
            "action",
            "disposition",
            "confidence",
            "rationale",
            "cited_evidence_ids",
        ],
        "additionalProperties": False,
    }


def _score_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "action_validity": {"type": "number"},
            "evidence_grounding": {"type": "number"},
            "specificity": {"type": "number"},
            "calibration": {"type": "number"},
        },
        "required": [
            "action_validity",
            "evidence_grounding",
            "specificity",
            "calibration",
        ],
        "additionalProperties": False,
    }


def _json_prompt(title: str, payload: dict[str, object], instructions: str) -> str:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    return f"{title}\n\n{instructions}\n\nJSON_INPUT:\n{rendered}\n"


def _evidence_payload(evidence: Sequence[EvidenceHit]) -> list[dict[str, object]]:
    return [
        {
            "alert_id": hit.alert_id,
            "score": hit.score,
            "stage": hit.stage,
            "text": hit.text,
        }
        for hit in evidence
    ]


def _resolve_api_key(
    *,
    provider: str,
    api_key: str | None,
    api_key_env: str | None,
) -> str:
    if api_key:
        return api_key
    env_name = api_key_env or {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    raise ValueError(f"{provider} runtime requires API key env {env_name}")


def _normalize_model_id(provider: str, model_id: str) -> tuple[str, str]:
    if ":" not in model_id:
        return model_id, f"{provider}:{model_id}"
    prefix, raw_model = model_id.split(":", 1)
    if prefix != provider:
        raise ValueError(f"{provider} runtime requires {provider}:<model> ids, got {model_id}")
    return raw_model, model_id


def _join_base_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _coerce_mapping(payload: object, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


class _ProviderRuntimeBase:
    provider_name: str

    def __init__(
        self,
        *,
        provider_name: str,
        model_id: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base_url: str,
        timeout_seconds: float = 30.0,
        transport: JSONTransport | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.api_model, self.model_id = _normalize_model_id(provider_name, model_id)
        self.api_key = _resolve_api_key(provider=provider_name, api_key=api_key, api_key_env=api_key_env)
        self.api_base_url = api_base_url
        self.timeout_seconds = timeout_seconds
        self.transport = transport or UrllibJSONTransport()

    def _request(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, object],
        path: str,
    ) -> dict[str, Any]:
        return self.transport.post_json(
            url=_join_base_url(self.api_base_url, path),
            headers=headers,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )


class OpenAIResponsesTriageEngine(_ProviderRuntimeBase):
    default_model_id = "openai:gpt-5"
    _tool_name = "emit_triage_decision"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        transport: JSONTransport | None = None,
    ) -> None:
        super().__init__(
            provider_name="openai",
            model_id=model_id or self.default_model_id,
            api_key=api_key,
            api_key_env=api_key_env,
            api_base_url=api_base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )

    def decide(self, *, query_id: str, query_text: str, evidence: Sequence[EvidenceHit]) -> tuple[TriageDecision, AuditRecord]:
        payload = {
            "model": self.api_model,
            "store": False,
            "parallel_tool_calls": False,
            "input": _json_prompt(
                "Decide the next bounded alert-triage action.",
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "evidence": _evidence_payload(evidence),
                },
                (
                    "Call the emit_triage_decision function exactly once. "
                    "Ground the answer only in the provided evidence, cite only retrieved alert_ids, "
                    "and keep confidence between 0 and 1."
                ),
            ),
            "tool_choice": {"type": "function", "name": self._tool_name},
            "tools": [
                _tool_schema(
                    description="Emit the normalized triage decision payload.",
                    schema=_decision_schema(),
                    tool_name=self._tool_name,
                    provider="openai",
                )
            ],
        }
        response = self._request(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            path="/responses",
        )
        arguments = _openai_function_arguments(response, tool_name=self._tool_name)
        decision = TriageDecision.model_validate(arguments)
        audit = AuditRecord(
            kind="model_call",
            name=self.model_id,
            inputs={
                "provider": self.provider_name,
                "role": "triager",
                "query_text": query_text,
                "retrieved_alert_ids": [hit.alert_id for hit in evidence],
            },
            outputs={
                "response_id": response.get("id"),
                "decision": decision.model_dump(mode="json"),
            },
        )
        return decision, audit


class OpenAIResponsesJudge(_ProviderRuntimeBase):
    default_model_id = "openai:gpt-5"
    _tool_name = "emit_reasoning_scores"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        transport: JSONTransport | None = None,
    ) -> None:
        super().__init__(
            provider_name="openai",
            model_id=model_id or self.default_model_id,
            api_key=api_key,
            api_key_env=api_key_env,
            api_base_url=api_base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )

    def evaluate(
        self,
        *,
        query_id: str,
        query_text: str,
        decision: TriageDecision,
        evidence: Sequence[EvidenceHit],
        target: ReasoningTarget,
    ) -> tuple[dict[str, float], AuditRecord]:
        payload = {
            "model": self.api_model,
            "store": False,
            "parallel_tool_calls": False,
            "input": _json_prompt(
                "Score the triage decision against the expected target.",
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "decision": decision.model_dump(mode="json"),
                    "evidence": _evidence_payload(evidence),
                    "target": target.model_dump(mode="json"),
                },
                (
                    "Call the emit_reasoning_scores function exactly once. "
                    "Return numeric scores between 0 and 1 for action_validity, "
                    "evidence_grounding, specificity, and calibration."
                ),
            ),
            "tool_choice": {"type": "function", "name": self._tool_name},
            "tools": [
                _tool_schema(
                    description="Emit the normalized reasoning score payload.",
                    schema=_score_schema(),
                    tool_name=self._tool_name,
                    provider="openai",
                )
            ],
        }
        response = self._request(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            path="/responses",
        )
        arguments = _openai_function_arguments(response, tool_name=self._tool_name)
        scores = ReasoningScorePayload.model_validate(arguments).as_metrics()
        audit = AuditRecord(
            kind="model_call",
            name=self.model_id,
            inputs={
                "provider": self.provider_name,
                "role": "judge",
                "query_id": query_id,
                "query_text": query_text,
                "retrieved_alert_ids": [hit.alert_id for hit in evidence],
            },
            outputs={
                "response_id": response.get("id"),
                "scores": scores,
            },
        )
        return scores, audit


class AnthropicTriageEngine(_ProviderRuntimeBase):
    default_model_id = "anthropic:claude-sonnet-4-5"
    _tool_name = "emit_triage_decision"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base_url: str = "https://api.anthropic.com/v1",
        timeout_seconds: float = 30.0,
        transport: JSONTransport | None = None,
    ) -> None:
        super().__init__(
            provider_name="anthropic",
            model_id=model_id or self.default_model_id,
            api_key=api_key,
            api_key_env=api_key_env,
            api_base_url=api_base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )

    def decide(self, *, query_id: str, query_text: str, evidence: Sequence[EvidenceHit]) -> tuple[TriageDecision, AuditRecord]:
        payload = {
            "model": self.api_model,
            "max_tokens": 512,
            "system": (
                "You are scoring a bounded alert-triage evaluation harness. "
                "Use the tool exactly once and ground the answer only in the provided evidence."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": _json_prompt(
                        "Decide the next bounded alert-triage action.",
                        {
                            "query_id": query_id,
                            "query_text": query_text,
                            "evidence": _evidence_payload(evidence),
                        },
                        "Call the tool exactly once and cite only retrieved alert_ids.",
                    ),
                }
            ],
            "tool_choice": {"type": "tool", "name": self._tool_name},
            "tools": [
                _tool_schema(
                    description="Emit the normalized triage decision payload.",
                    schema=_decision_schema(),
                    tool_name=self._tool_name,
                    provider="anthropic",
                )
            ],
        }
        response = self._request(
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            payload=payload,
            path="/messages",
        )
        tool_input = _anthropic_tool_input(response, tool_name=self._tool_name)
        decision = TriageDecision.model_validate(tool_input)
        audit = AuditRecord(
            kind="model_call",
            name=self.model_id,
            inputs={
                "provider": self.provider_name,
                "role": "triager",
                "query_text": query_text,
                "retrieved_alert_ids": [hit.alert_id for hit in evidence],
            },
            outputs={
                "response_id": response.get("id"),
                "decision": decision.model_dump(mode="json"),
            },
        )
        return decision, audit


class AnthropicJudge(_ProviderRuntimeBase):
    default_model_id = "anthropic:claude-sonnet-4-5"
    _tool_name = "emit_reasoning_scores"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base_url: str = "https://api.anthropic.com/v1",
        timeout_seconds: float = 30.0,
        transport: JSONTransport | None = None,
    ) -> None:
        super().__init__(
            provider_name="anthropic",
            model_id=model_id or self.default_model_id,
            api_key=api_key,
            api_key_env=api_key_env,
            api_base_url=api_base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )

    def evaluate(
        self,
        *,
        query_id: str,
        query_text: str,
        decision: TriageDecision,
        evidence: Sequence[EvidenceHit],
        target: ReasoningTarget,
    ) -> tuple[dict[str, float], AuditRecord]:
        payload = {
            "model": self.api_model,
            "max_tokens": 512,
            "system": (
                "You are scoring a bounded alert-triage evaluation harness. "
                "Use the tool exactly once and return only normalized numeric scores."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": _json_prompt(
                        "Score the triage decision against the expected target.",
                        {
                            "query_id": query_id,
                            "query_text": query_text,
                            "decision": decision.model_dump(mode="json"),
                            "evidence": _evidence_payload(evidence),
                            "target": target.model_dump(mode="json"),
                        },
                        (
                            "Call the tool exactly once and return numeric scores between 0 and 1 "
                            "for action_validity, evidence_grounding, specificity, and calibration."
                        ),
                    ),
                }
            ],
            "tool_choice": {"type": "tool", "name": self._tool_name},
            "tools": [
                _tool_schema(
                    description="Emit the normalized reasoning score payload.",
                    schema=_score_schema(),
                    tool_name=self._tool_name,
                    provider="anthropic",
                )
            ],
        }
        response = self._request(
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            payload=payload,
            path="/messages",
        )
        tool_input = _anthropic_tool_input(response, tool_name=self._tool_name)
        scores = ReasoningScorePayload.model_validate(tool_input).as_metrics()
        audit = AuditRecord(
            kind="model_call",
            name=self.model_id,
            inputs={
                "provider": self.provider_name,
                "role": "judge",
                "query_id": query_id,
                "query_text": query_text,
                "retrieved_alert_ids": [hit.alert_id for hit in evidence],
            },
            outputs={
                "response_id": response.get("id"),
                "scores": scores,
            },
        )
        return scores, audit


def _openai_function_arguments(response: dict[str, Any], *, tool_name: str) -> dict[str, Any]:
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("openai response missing output list")
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call" or item.get("name") != tool_name:
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("openai function_call arguments missing string payload")
        parsed = json.loads(arguments)
        return _coerce_mapping(parsed, label="openai function_call arguments")
    raise ValueError(f"openai response missing function_call for {tool_name}")


def _anthropic_tool_input(response: dict[str, Any], *, tool_name: str) -> dict[str, Any]:
    content = response.get("content")
    if not isinstance(content, list):
        raise ValueError("anthropic response missing content list")
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_use" or item.get("name") != tool_name:
            continue
        return _coerce_mapping(item.get("input"), label="anthropic tool_use input")
    raise ValueError(f"anthropic response missing tool_use for {tool_name}")


def build_provider_runtimes(
    *,
    runtime: str,
    llm_model: str,
    judge_model: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    api_base_url: str | None = None,
    timeout_seconds: float = 30.0,
    transport: JSONTransport | None = None,
) -> tuple[TriageRuntime, JudgeRuntime]:
    if runtime == "openai":
        base_url = api_base_url or "https://api.openai.com/v1"
        return (
            OpenAIResponsesTriageEngine(
                model_id=llm_model,
                api_key=api_key,
                api_key_env=api_key_env,
                api_base_url=base_url,
                timeout_seconds=timeout_seconds,
                transport=transport,
            ),
            OpenAIResponsesJudge(
                model_id=judge_model,
                api_key=api_key,
                api_key_env=api_key_env,
                api_base_url=base_url,
                timeout_seconds=timeout_seconds,
                transport=transport,
            ),
        )
    if runtime == "anthropic":
        base_url = api_base_url or "https://api.anthropic.com/v1"
        return (
            AnthropicTriageEngine(
                model_id=llm_model,
                api_key=api_key,
                api_key_env=api_key_env,
                api_base_url=base_url,
                timeout_seconds=timeout_seconds,
                transport=transport,
            ),
            AnthropicJudge(
                model_id=judge_model,
                api_key=api_key,
                api_key_env=api_key_env,
                api_base_url=base_url,
                timeout_seconds=timeout_seconds,
                transport=transport,
            ),
        )
    raise ValueError(f"unsupported provider runtime: {runtime}")
