from __future__ import annotations

import sys
from pathlib import Path

import pytest

from alert_triage.triage import MCPTerminalToolRuntime, StdioMCPClient, TriageDecision


def _server_command(mode: str = "ok") -> tuple[str, ...]:
    fixture = Path("tests/fixtures/mcp/fake_terminal_server.py")
    return (sys.executable, str(fixture), mode)


def _decision() -> TriageDecision:
    return TriageDecision(
        query_id="query-1",
        query_text="Investigate suspicious credential reset activity.",
        action="reset_credentials_and_scope_phishing",
        disposition="credential_reset",
        confidence=0.91,
        rationale="Credential phishing evidence supports a reset action.",
        cited_evidence_ids=("alert-1",),
    )


def test_stdio_mcp_client_initializes_lists_tools_and_calls_tool() -> None:
    client = StdioMCPClient(command=_server_command())
    try:
        tools = client.list_tools()
        assert tools[0]["name"] == "propose_investigation_step"

        result = client.call_tool(
            tool_name="propose_investigation_step",
            arguments={
                "query_id": "query-1",
                "action": "reset_credentials_and_scope_phishing",
            },
        )

        assert result["structuredContent"] == {
            "accepted_action": "reset_credentials_and_scope_phishing",
            "accepted_disposition": None,
        }
    finally:
        client.close()


def test_mcp_terminal_runtime_raises_when_tool_is_missing() -> None:
    runtime = MCPTerminalToolRuntime(server_command=_server_command("missing-tool"))
    try:
        with pytest.raises(ValueError, match="does not expose tool"):
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
    finally:
        runtime.close()


def test_mcp_terminal_runtime_raises_on_malformed_tool_result() -> None:
    runtime = MCPTerminalToolRuntime(server_command=_server_command("malformed-result"))
    try:
        with pytest.raises(ValueError, match="content must be a list"):
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
    finally:
        runtime.close()


def test_mcp_terminal_runtime_raises_on_error_tool_result() -> None:
    runtime = MCPTerminalToolRuntime(server_command=_server_command("tool-error"))
    try:
        with pytest.raises(ValueError, match="returned an error result"):
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
    finally:
        runtime.close()
