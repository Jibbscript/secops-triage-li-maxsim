from __future__ import annotations

import json
import sys


MODE = sys.argv[1] if len(sys.argv) > 1 else "ok"
TOOL_NAME = "propose_investigation_step"


def read_message() -> dict[str, object] | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8"))


def write_message(payload: dict[str, object]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    if message is None:
        break

    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-terminal-server", "version": "0.1.0"},
                },
            }
        )
        continue

    if method == "shutdown":
        write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {},
            }
        )
        continue

    if method == "exit":
        break

    if method == "notifications/initialized":
        continue

    if method == "tools/list":
        tools = [] if MODE == "missing-tool" else [{"name": TOOL_NAME}]
        write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": tools,
                },
            }
        )
        continue

    if method == "tools/call":
        params = message.get("params", {})
        arguments = params.get("arguments", {}) if isinstance(params, dict) else {}
        if MODE == "malformed-result":
            result = {"content": "not-a-list"}
        elif MODE == "tool-error":
            result = {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": "tool rejected request",
                    }
                ],
            }
        else:
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": f"planned:{arguments.get('action', 'unknown')}",
                    }
                ],
                "structuredContent": {
                    "accepted_action": arguments.get("action"),
                    "accepted_disposition": arguments.get("disposition"),
                },
            }
        write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        )
        continue

    write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unsupported method: {method}"},
        }
    )
