from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from alert_triage.triage import MCPError, StdioMCPClient


DEFAULT_REFERENCE_PACKAGE = "@modelcontextprotocol/server-everything"
DEFAULT_REFERENCE_VERSION = "2025.8.4"
DEFAULT_REFERENCE_COMMAND = (
    "npx",
    "-y",
    f"{DEFAULT_REFERENCE_PACKAGE}@{DEFAULT_REFERENCE_VERSION}",
    "stdio",
)
DEFAULT_REFERENCE_TOOL_NAME = "echo"
DEFAULT_REFERENCE_MESSAGE = "phase8-reference-probe"


def normalize_reference_command(command: Sequence[str] | None = None) -> tuple[str, ...]:
    if command is None:
        return DEFAULT_REFERENCE_COMMAND
    normalized = tuple(part for part in command if part)
    if not normalized:
        raise MCPError("mcp reference probe command is required", code="mcp_configuration_invalid")
    return normalized


def run_reference_probe(
    *,
    command: Sequence[str] | None = None,
    tool_name: str = DEFAULT_REFERENCE_TOOL_NAME,
    message: str = DEFAULT_REFERENCE_MESSAGE,
    startup_timeout_seconds: float = 15.0,
    call_timeout_seconds: float = 15.0,
) -> dict[str, object]:
    normalized_command = normalize_reference_command(command)
    using_default_reference = command is None
    client = StdioMCPClient(
        command=normalized_command,
        startup_timeout_seconds=startup_timeout_seconds,
        call_timeout_seconds=call_timeout_seconds,
    )
    try:
        tools = client.list_tools()
        available_tools = tuple(tool["name"] for tool in tools)
        if tool_name not in available_tools:
            raise MCPError(
                f"mcp reference server does not expose tool {tool_name}",
                code="mcp_tool_not_found",
            )
        result = client.call_tool(tool_name=tool_name, arguments={"message": message})
        payload: dict[str, object] = {
            "command_source": "default_reference" if using_default_reference else "custom",
            "command": list(normalized_command),
            "tool_name": tool_name,
            "message": message,
            "available_tools": list(available_tools),
            "result": result,
        }
        if using_default_reference:
            payload["server_package"] = DEFAULT_REFERENCE_PACKAGE
            payload["server_version"] = DEFAULT_REFERENCE_VERSION
        return payload
    finally:
        client.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase-8 MCP reference probe.")
    parser.add_argument("--server-command")
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--tool-name", default=DEFAULT_REFERENCE_TOOL_NAME)
    parser.add_argument("--message", default=DEFAULT_REFERENCE_MESSAGE)
    parser.add_argument("--startup-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--call-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command: Sequence[str] | None
    if args.server_command:
        command = (args.server_command, *args.server_arg)
    else:
        command = None
    payload = run_reference_probe(
        command=command,
        tool_name=args.tool_name,
        message=args.message,
        startup_timeout_seconds=args.startup_timeout_seconds,
        call_timeout_seconds=args.call_timeout_seconds,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
