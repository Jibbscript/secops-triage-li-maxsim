from __future__ import annotations

import json
from pathlib import Path

import pytest

from alert_triage.evals.tier2_phase4 import _build_reasoning_runtimes, run_phase4_reasoning
from alert_triage.triage import AnthropicJudge, AnthropicTriageEngine, OpenAIResponsesJudge, OpenAIResponsesTriageEngine


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
        runtime="local",
        llm_model="local:deterministic-triager-v1",
        judge_model="local:deterministic-judge-v1",
    )

    assert report["experiment_id"] == "tier2-phase4"
    assert report["retriever"] == "binary-then-fp16-rerank"
    assert report["runtime"] == "local"
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
        "runtime",
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


def test_phase4_reasoning_harness_supports_replay_runtime(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures/phase1_tier1")
    out_json = tmp_path / "tier2-replay.json"
    out_trace = tmp_path / "tier2-replay-trace.jsonl"

    report = run_phase4_reasoning(
        fixture_dir,
        out_json,
        out_trace,
        threads=1,
        rerank_depth=2,
        runtime="replay",
        runtime_fixture=fixture_dir / "replay_runtime.json",
        llm_model="replay:fixture-triager-v1",
        judge_model="replay:fixture-judge-v1",
    )

    assert report["runtime"] == "replay"
    assert report["llm_model"] == "replay:fixture-triager-v1"
    assert report["judge_model"] == "replay:fixture-judge-v1"
    assert report["metrics"]["action_validity"] == 1.0

    payload = json.loads(out_json.read_text())
    assert payload["runtime"] == "replay"
    first = json.loads(out_trace.read_text().strip().splitlines()[0])
    assert [record["name"] for record in first["audit_records"]] == [
        "replay:fixture-triager-v1",
        "replay:fixture-judge-v1",
        "propose_investigation_step",
    ]


@pytest.mark.parametrize(
    ("runtime", "triager_type", "judge_type", "expected_model"),
    [
        ("openai", OpenAIResponsesTriageEngine, OpenAIResponsesJudge, "openai:gpt-5"),
        ("anthropic", AnthropicTriageEngine, AnthropicJudge, "anthropic:claude-sonnet-4-5"),
    ],
)
def test_build_reasoning_runtimes_supports_live_provider_runtimes(
    runtime: str,
    triager_type: type[object],
    judge_type: type[object],
    expected_model: str,
) -> None:
    triager, judge, terminal_tool = _build_reasoning_runtimes(
        runtime=runtime,
        llm_model="local:deterministic-triager-v1",
        judge_model="local:deterministic-judge-v1",
        runtime_fixture=None,
        api_key="provider-test-key",
        transport=FakeTransport([]),
    )

    assert isinstance(triager, triager_type)
    assert isinstance(judge, judge_type)
    assert triager.model_id == expected_model
    assert judge.model_id == expected_model
    assert terminal_tool.tool_name == "propose_investigation_step"


@pytest.mark.parametrize(
    ("runtime", "response_sequence", "expected_names"),
    [
        (
            "openai",
            [
                {
                    "id": "resp_openai_q1_triage",
                    "output": [
                        {
                            "type": "function_call",
                            "name": "emit_triage_decision",
                            "arguments": json.dumps(
                                {
                                    "query_id": "query-1",
                                    "query_text": "Investigate credential phishing activity around suspicious inbox forwarding.",
                                    "action": "reset_credentials_and_scope_phishing",
                                    "disposition": "credential_reset",
                                    "confidence": 0.91,
                                    "rationale": "Phishing evidence supports a scoped credential reset.",
                                    "cited_evidence_ids": ["alert-1"],
                                }
                            ),
                        }
                    ],
                },
                {
                    "id": "resp_openai_q1_judge",
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
                },
                {
                    "id": "resp_openai_q2_triage",
                    "output": [
                        {
                            "type": "function_call",
                            "name": "emit_triage_decision",
                            "arguments": json.dumps(
                                {
                                    "query_id": "query-2",
                                    "query_text": "Investigate ransomware beaconing and lateral movement from the finance subnet.",
                                    "action": "isolate_host_and_block_beacon",
                                    "disposition": "contain_host",
                                    "confidence": 0.94,
                                    "rationale": "Ransomware beacon evidence supports host isolation.",
                                    "cited_evidence_ids": ["alert-2"],
                                }
                            ),
                        }
                    ],
                },
                {
                    "id": "resp_openai_q2_judge",
                    "output": [
                        {
                            "type": "function_call",
                            "name": "emit_reasoning_scores",
                            "arguments": json.dumps(
                                {
                                    "action_validity": 1.0,
                                    "evidence_grounding": 1.0,
                                    "specificity": 1.0,
                                    "calibration": 0.94,
                                }
                            ),
                        }
                    ],
                },
            ],
            ["openai:gpt-5", "openai:gpt-5", "propose_investigation_step"],
        ),
        (
            "anthropic",
            [
                {
                    "id": "msg_anthropic_q1_triage",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "emit_triage_decision",
                            "input": {
                                "query_id": "query-1",
                                "query_text": "Investigate credential phishing activity around suspicious inbox forwarding.",
                                "action": "reset_credentials_and_scope_phishing",
                                "disposition": "credential_reset",
                                "confidence": 0.91,
                                "rationale": "Phishing evidence supports a scoped credential reset.",
                                "cited_evidence_ids": ["alert-1"],
                            },
                        }
                    ],
                },
                {
                    "id": "msg_anthropic_q1_judge",
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
                },
                {
                    "id": "msg_anthropic_q2_triage",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "emit_triage_decision",
                            "input": {
                                "query_id": "query-2",
                                "query_text": "Investigate ransomware beaconing and lateral movement from the finance subnet.",
                                "action": "isolate_host_and_block_beacon",
                                "disposition": "contain_host",
                                "confidence": 0.94,
                                "rationale": "Ransomware beacon evidence supports host isolation.",
                                "cited_evidence_ids": ["alert-2"],
                            },
                        }
                    ],
                },
                {
                    "id": "msg_anthropic_q2_judge",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "emit_reasoning_scores",
                            "input": {
                                "action_validity": 1.0,
                                "evidence_grounding": 1.0,
                                "specificity": 1.0,
                                "calibration": 0.94,
                            },
                        }
                    ],
                },
            ],
            ["anthropic:claude-sonnet-4-5", "anthropic:claude-sonnet-4-5", "propose_investigation_step"],
        ),
    ],
)
def test_phase4_reasoning_harness_supports_live_provider_runtimes(
    tmp_path: Path,
    runtime: str,
    response_sequence: list[dict[str, object]],
    expected_names: list[str],
) -> None:
    fixture_dir = Path("tests/fixtures/phase1_tier1")
    out_json = tmp_path / f"tier2-{runtime}.json"
    out_trace = tmp_path / f"tier2-{runtime}-trace.jsonl"
    transport = FakeTransport(response_sequence)

    report = run_phase4_reasoning(
        fixture_dir,
        out_json,
        out_trace,
        threads=1,
        rerank_depth=2,
        runtime=runtime,
        llm_model="local:deterministic-triager-v1",
        judge_model="local:deterministic-judge-v1",
        api_key="provider-test-key",
        transport=transport,
    )

    assert report["runtime"] == runtime
    assert report["sample_count"] == 2
    assert report["orphan_llm_calls"] == 0
    assert report["orphan_tool_calls"] == 0
    assert report["terminal_tool"] == "propose_investigation_step"
    first = json.loads(out_trace.read_text().strip().splitlines()[0])
    assert [record["name"] for record in first["audit_records"]] == expected_names
    assert [record["inputs"].get("role") for record in first["audit_records"][:2]] == ["triager", "judge"]
