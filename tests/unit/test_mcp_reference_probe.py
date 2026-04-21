from __future__ import annotations

import sys
from pathlib import Path

import pytest

from alert_triage.evals.mcp_reference_probe import (
    DEFAULT_REFERENCE_COMMAND,
    DEFAULT_REFERENCE_VERSION,
    normalize_reference_command,
    run_reference_probe,
)
from alert_triage.triage import MCPError


def _server_command(mode: str = "ok") -> tuple[str, ...]:
    fixture = Path("tests/fixtures/mcp/fake_reference_server.py")
    return (sys.executable, str(fixture), mode)


def test_normalize_reference_command_returns_default_reference_server_command() -> None:
    assert normalize_reference_command() == DEFAULT_REFERENCE_COMMAND


def test_normalize_reference_command_rejects_empty_command() -> None:
    with pytest.raises(MCPError, match="probe command is required") as excinfo:
        normalize_reference_command(())
    assert excinfo.value.code == "mcp_configuration_invalid"


def test_run_reference_probe_returns_echo_result() -> None:
    payload = run_reference_probe(command=_server_command(), message="hello from probe")
    assert payload["command_source"] == "custom"
    assert "server_version" not in payload
    assert payload["tool_name"] == "echo"
    assert "echo" in payload["available_tools"]
    assert payload["result"]["structuredContent"] == {"echoed": "hello from probe"}
    assert payload["result"]["content"][0]["text"] == "hello from probe"


def test_run_reference_probe_reports_official_reference_metadata_for_default_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *, command: tuple[str, ...], startup_timeout_seconds: float, call_timeout_seconds: float) -> None:
            self.command = command

        def list_tools(self) -> tuple[dict[str, object], ...]:
            return ({"name": "echo"},)

        def call_tool(self, *, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
            return {"content": [{"type": "text", "text": f"Echo: {arguments['message']}"}]}

        def close(self) -> None:
            return None

    monkeypatch.setattr("alert_triage.evals.mcp_reference_probe.StdioMCPClient", FakeClient)

    payload = run_reference_probe(command=None)
    assert payload["command_source"] == "default_reference"
    assert payload["server_version"] == DEFAULT_REFERENCE_VERSION


def test_run_reference_probe_raises_tool_not_found_code() -> None:
    with pytest.raises(MCPError, match="does not expose tool echo") as excinfo:
        run_reference_probe(command=_server_command("missing-tool"))
    assert excinfo.value.code == "mcp_tool_not_found"


def test_run_reference_probe_raises_tool_result_invalid_code() -> None:
    with pytest.raises(MCPError, match="returned an error result") as excinfo:
        run_reference_probe(command=_server_command("tool-error"))
    assert excinfo.value.code == "mcp_tool_result_invalid"
