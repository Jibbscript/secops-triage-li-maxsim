from __future__ import annotations

import os
import shutil

import pytest

from alert_triage.evals.mcp_reference_probe import (
    DEFAULT_REFERENCE_COMMAND,
    DEFAULT_REFERENCE_TOOL_NAME,
    DEFAULT_REFERENCE_VERSION,
    run_reference_probe,
)


pytestmark = pytest.mark.external_mcp_live


def test_mcp_reference_probe_runs_against_official_everything_server() -> None:
    if os.getenv("RUN_EXTERNAL_MCP_LIVE") != "1":
        pytest.skip("set RUN_EXTERNAL_MCP_LIVE=1 to run live MCP reference probe")
    if shutil.which("npx") is None:
        pytest.skip("npx is required for the live MCP reference probe")

    payload = run_reference_probe(startup_timeout_seconds=20.0, call_timeout_seconds=20.0)

    assert payload["command"] == list(DEFAULT_REFERENCE_COMMAND)
    assert payload["server_version"] == DEFAULT_REFERENCE_VERSION
    assert DEFAULT_REFERENCE_TOOL_NAME in payload["available_tools"]
    assert isinstance(payload["result"]["content"], list)
    assert payload["result"]["content"]
