from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from alert_triage.evals.mcp_reference_probe import DEFAULT_REFERENCE_COMMAND
from alert_triage.evals.tier2_phase4 import run_phase4_reasoning


pytestmark = pytest.mark.external_mcp_live


def test_phase4_reasoning_runs_against_official_everything_server_via_echo_profile(tmp_path: Path) -> None:
    if os.getenv("RUN_EXTERNAL_MCP_LIVE") != "1":
        pytest.skip("set RUN_EXTERNAL_MCP_LIVE=1 to run live MCP tier-2 verification")
    if shutil.which("npx") is None:
        pytest.skip("npx is required for the live MCP tier-2 verification")

    fixture_dir = Path("tests/fixtures/phase1_tier1")
    out_json = tmp_path / "tier2-mcp-live.json"
    out_trace = tmp_path / "tier2-mcp-live-trace.jsonl"

    report = run_phase4_reasoning(
        fixture_dir,
        out_json,
        out_trace,
        threads=1,
        rerank_depth=2,
        runtime="local",
        terminal_runtime="mcp",
        mcp_server_command=DEFAULT_REFERENCE_COMMAND[0],
        mcp_server_args=DEFAULT_REFERENCE_COMMAND[1:],
        mcp_profile="everything_echo",
        mcp_startup_timeout_seconds=20.0,
        mcp_call_timeout_seconds=20.0,
        llm_model="local:deterministic-triager-v1",
        judge_model="local:deterministic-judge-v1",
    )

    assert report["terminal_tool"] == "propose_investigation_step"
    first = json.loads(out_trace.read_text().strip().splitlines()[0])
    assert first["audit_records"][-1]["name"] == "propose_investigation_step"
    assert first["audit_records"][-1]["outputs"]["mcp_profile"] == "everything_echo"
    assert first["audit_records"][-1]["outputs"]["mcp_tool_name"] == "echo"
    assert first["audit_records"][-1]["outputs"]["structuredContent"]["accepted_action"]
