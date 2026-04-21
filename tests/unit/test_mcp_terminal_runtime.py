from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from alert_triage.triage import MCPError, MCPTerminalToolRuntime, RetryPolicy, StdioMCPClient, TriageDecision


def _server_command(mode: str = "ok", state_path: Path | None = None) -> tuple[str, ...]:
    fixture = Path("tests/fixtures/mcp/fake_terminal_server.py")
    command: tuple[str, ...] = (sys.executable, str(fixture), mode)
    if state_path is not None:
        command += (str(state_path),)
    return command


def _reference_server_command(mode: str = "ok") -> tuple[str, ...]:
    fixture = Path("tests/fixtures/mcp/fake_reference_server.py")
    return (sys.executable, str(fixture), mode)


def _read_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {key: int(value) for key, value in payload.items()}


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


def test_stdio_mcp_client_rejects_empty_command_with_configuration_code() -> None:
    with pytest.raises(MCPError, match="mcp server command is required") as excinfo:
        StdioMCPClient(command=())
    assert excinfo.value.code == "mcp_configuration_invalid"


def test_stdio_mcp_client_retries_transient_startup_timeout(tmp_path: Path) -> None:
    state_path = tmp_path / "startup-state.json"
    client = StdioMCPClient(
        command=_server_command("initialize-timeout-once", state_path),
        startup_timeout_seconds=0.05,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    try:
        tools = client.list_tools()
    finally:
        client.close()

    assert tools[0]["name"] == "propose_investigation_step"
    assert _read_state(state_path)["initialize"] == 2


def test_mcp_terminal_runtime_raises_when_tool_is_missing() -> None:
    runtime = MCPTerminalToolRuntime(server_command=_server_command("missing-tool"))
    try:
        with pytest.raises(MCPError, match="does not expose tool") as excinfo:
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
        assert excinfo.value.code == "mcp_tool_not_found"
    finally:
        runtime.close()


def test_mcp_terminal_runtime_raises_on_malformed_tool_result() -> None:
    runtime = MCPTerminalToolRuntime(server_command=_server_command("malformed-result"))
    try:
        with pytest.raises(MCPError, match="content must be a list") as excinfo:
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
        assert excinfo.value.code == "mcp_tool_result_invalid"
    finally:
        runtime.close()


def test_mcp_terminal_runtime_raises_on_error_tool_result() -> None:
    runtime = MCPTerminalToolRuntime(server_command=_server_command("tool-error"))
    try:
        with pytest.raises(MCPError, match="returned an error result") as excinfo:
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
        assert excinfo.value.code == "mcp_tool_result_invalid"
    finally:
        runtime.close()


def test_mcp_terminal_runtime_retries_transient_tool_timeout(tmp_path: Path) -> None:
    state_path = tmp_path / "call-state.json"
    runtime = MCPTerminalToolRuntime(
        server_command=_server_command("call-timeout-once", state_path),
        call_timeout_seconds=0.05,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    try:
        audit = runtime.emit(
            query_id="query-1",
            query_text="Investigate suspicious credential reset activity.",
            decision=_decision(),
        )
    finally:
        runtime.close()

    assert audit.outputs["structuredContent"]["accepted_action"] == _decision().action
    assert _read_state(state_path)["tools_call"] == 2


def test_mcp_terminal_runtime_supports_everything_echo_profile() -> None:
    runtime = MCPTerminalToolRuntime(
        server_command=_reference_server_command(),
        mcp_profile="everything_echo",
    )
    try:
        audit = runtime.emit(
            query_id="query-1",
            query_text="Investigate suspicious credential reset activity.",
            decision=_decision(),
        )
    finally:
        runtime.close()

    assert audit.name == "propose_investigation_step"
    assert audit.outputs["mcp_profile"] == "everything_echo"
    assert audit.outputs["mcp_tool_name"] == "echo"
    assert audit.outputs["structuredContent"] == {
        "accepted_action": "reset_credentials_and_scope_phishing",
        "accepted_disposition": "credential_reset",
    }
    assert audit.outputs["raw_mcp_result"]["structuredContent"]["echoed"]


def test_mcp_terminal_runtime_supports_prefixed_everything_echo_payload() -> None:
    runtime = MCPTerminalToolRuntime(
        server_command=_reference_server_command("prefixed-echo"),
        mcp_profile="everything_echo",
    )
    try:
        audit = runtime.emit(
            query_id="query-1",
            query_text="Investigate suspicious credential reset activity.",
            decision=_decision(),
        )
    finally:
        runtime.close()

    assert audit.outputs["structuredContent"] == {
        "accepted_action": "reset_credentials_and_scope_phishing",
        "accepted_disposition": "credential_reset",
    }


def test_mcp_terminal_runtime_rejects_invalid_profile() -> None:
    with pytest.raises(MCPError, match="unsupported mcp terminal tool profile") as excinfo:
        MCPTerminalToolRuntime(server_command=_server_command(), mcp_profile="unknown")
    assert excinfo.value.code == "mcp_profile_invalid"


def test_mcp_terminal_runtime_rejects_custom_tool_name_for_everything_echo_profile() -> None:
    with pytest.raises(MCPError, match="requires tool_name propose_investigation_step") as excinfo:
        MCPTerminalToolRuntime(
            server_command=_reference_server_command(),
            mcp_profile="everything_echo",
            tool_name="custom_step",
        )
    assert excinfo.value.code == "mcp_configuration_invalid"


def test_mcp_terminal_runtime_raises_when_echo_tool_is_missing() -> None:
    runtime = MCPTerminalToolRuntime(
        server_command=_reference_server_command("missing-tool"),
        mcp_profile="everything_echo",
    )
    try:
        with pytest.raises(MCPError, match="does not expose tool echo") as excinfo:
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
        assert excinfo.value.code == "mcp_tool_not_found"
    finally:
        runtime.close()


def test_mcp_terminal_runtime_does_not_retry_missing_tools(tmp_path: Path) -> None:
    state_path = tmp_path / "missing-tool-state.json"
    runtime = MCPTerminalToolRuntime(
        server_command=_server_command("missing-tool", state_path),
        retry_policy=RetryPolicy(max_attempts=3),
    )
    try:
        with pytest.raises(MCPError, match="does not expose tool") as excinfo:
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
        assert excinfo.value.code == "mcp_tool_not_found"
    finally:
        runtime.close()

    state = _read_state(state_path)
    assert state["initialize"] == 1
    assert state["tools_list"] == 1


def test_mcp_terminal_runtime_raises_on_non_json_echo_payload() -> None:
    runtime = MCPTerminalToolRuntime(
        server_command=_reference_server_command("bad-echo-payload"),
        mcp_profile="everything_echo",
    )
    try:
        with pytest.raises(MCPError, match="echoed payload must be valid JSON") as excinfo:
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
        assert excinfo.value.code == "mcp_tool_adapter_result_invalid"
    finally:
        runtime.close()


def test_mcp_terminal_runtime_raises_on_invalid_echo_payload_shape() -> None:
    runtime = MCPTerminalToolRuntime(
        server_command=_reference_server_command("invalid-echo-shape"),
        mcp_profile="everything_echo",
    )
    try:
        with pytest.raises(MCPError, match="must contain string action and disposition") as excinfo:
            runtime.emit(
                query_id="query-1",
                query_text="Investigate suspicious credential reset activity.",
                decision=_decision(),
            )
        assert excinfo.value.code == "mcp_tool_adapter_result_invalid"
    finally:
        runtime.close()


def test_stdio_mcp_client_raises_startup_code_on_invalid_initialize_response() -> None:
    fixture = Path("tests/fixtures/mcp/fake_reference_server.py")
    client = StdioMCPClient(command=(sys.executable, str(fixture), "invalid-protocol"))
    try:
        with pytest.raises(MCPError, match="missing protocolVersion") as excinfo:
            client.list_tools()
        assert excinfo.value.code == "mcp_startup_failed"
    finally:
        client.close()


def test_stdio_mcp_client_raises_tool_discovery_code_on_invalid_tool_entry() -> None:
    fixture = Path("tests/fixtures/mcp/fake_reference_server.py")
    client = StdioMCPClient(command=(sys.executable, str(fixture), "invalid-tool-entry"))
    try:
        with pytest.raises(MCPError, match="invalid tool entry") as excinfo:
            client.list_tools()
        assert excinfo.value.code == "mcp_tool_discovery_failed"
    finally:
        client.close()
