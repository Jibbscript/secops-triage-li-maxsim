from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
from collections.abc import Sequence
from typing import Any

from .audit import AuditRecord
from .reasoning import TriageDecision


_JSONRPC_VERSION = "2.0"
_MCP_PROTOCOL_VERSION = "2024-11-05"


class StdioMCPClient:
    def __init__(
        self,
        *,
        command: Sequence[str],
        startup_timeout_seconds: float = 5.0,
        call_timeout_seconds: float = 10.0,
    ) -> None:
        if not command:
            raise ValueError("mcp server command is required")
        self._command = tuple(command)
        self._startup_timeout_seconds = startup_timeout_seconds
        self._call_timeout_seconds = call_timeout_seconds
        self._process: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        self._buffer = bytearray()
        self._tools_cache: tuple[dict[str, object], ...] | None = None

    def close(self) -> None:
        process = self._process
        self._process = None
        self._tools_cache = None
        self._buffer.clear()
        if process is None:
            return
        self._graceful_shutdown()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

    def list_tools(self) -> tuple[dict[str, object], ...]:
        if self._tools_cache is not None:
            return self._tools_cache
        result = self._request("tools/list", {}, timeout_seconds=self._call_timeout_seconds)
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise ValueError("mcp tools/list result must contain a tools list")
        normalized = []
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                raise ValueError("mcp tools/list returned an invalid tool entry")
            normalized.append(tool)
        self._tools_cache = tuple(normalized)
        return self._tools_cache

    def call_tool(self, *, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        result = self._request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
            timeout_seconds=self._call_timeout_seconds,
        )
        if not isinstance(result, dict):
            raise ValueError("mcp tools/call result must be an object")
        if result.get("isError") is True:
            raise ValueError(f"mcp tool {tool_name} returned an error result")
        content = result.get("content")
        if content is not None and not isinstance(content, list):
            raise ValueError("mcp tools/call result content must be a list")
        return result

    def _ensure_started(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise ValueError(f"failed to launch mcp server {self._command[0]}: {exc.strerror}") from exc

        result = self._request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "alert-triage-li",
                    "version": "0.1.0",
                },
            },
            timeout_seconds=self._startup_timeout_seconds,
        )
        protocol_version = result.get("protocolVersion")
        if not isinstance(protocol_version, str):
            raise ValueError("mcp initialize result missing protocolVersion")
        self._notify("notifications/initialized", {})

    def _notify(self, method: str, params: dict[str, object]) -> None:
        self._write_message(
            {
                "jsonrpc": _JSONRPC_VERSION,
                "method": method,
                "params": params,
            }
        )

    def _request(self, method: str, params: dict[str, object], *, timeout_seconds: float) -> dict[str, Any]:
        self._ensure_started()
        request_id = self._next_id
        self._next_id += 1
        self._write_message(
            {
                "jsonrpc": _JSONRPC_VERSION,
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        response = self._read_message(timeout_seconds=timeout_seconds)
        if response.get("id") != request_id:
            raise ValueError(f"mcp response id mismatch for {method}")
        error_payload = response.get("error")
        if error_payload is not None:
            raise ValueError(f"mcp {method} failed: {error_payload}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"mcp {method} result must be an object")
        return result

    def _write_message(self, payload: dict[str, object]) -> None:
        process = self._require_process()
        stdin = process.stdin
        if stdin is None:
            raise ValueError("mcp server stdin is unavailable")
        body = json.dumps(payload).encode("utf-8")
        message = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        try:
            stdin.write(message)
            stdin.flush()
        except BrokenPipeError as exc:
            raise ValueError(self._format_error("mcp server closed stdin unexpectedly")) from exc

    def _read_message(self, *, timeout_seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        header_bytes = self._read_until(b"\r\n\r\n", deadline=deadline)
        header_blob = header_bytes[: -len(b"\r\n\r\n")]

        headers = {}
        for line in header_blob.decode("ascii").split("\r\n"):
            if not line:
                continue
            if ":" not in line:
                raise ValueError("mcp response header is malformed")
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        try:
            content_length = int(headers["content-length"])
        except (KeyError, ValueError) as exc:
            raise ValueError("mcp response missing valid Content-Length") from exc

        body = self._read_exact(content_length, deadline=deadline)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("mcp response body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("mcp response body must be a JSON object")
        return payload

    def _read_until(self, marker: bytes, *, deadline: float) -> bytes:
        while True:
            index = self._buffer.find(marker)
            if index >= 0:
                end = index + len(marker)
                chunk = bytes(self._buffer[:end])
                del self._buffer[:end]
                return chunk
            self._pump_stdout(deadline=deadline)

    def _read_exact(self, length: int, *, deadline: float) -> bytes:
        while len(self._buffer) < length:
            self._pump_stdout(deadline=deadline)
        chunk = bytes(self._buffer[:length])
        del self._buffer[:length]
        return chunk

    def _pump_stdout(self, *, deadline: float) -> None:
        process = self._require_process()
        stdout = process.stdout
        if stdout is None:
            raise ValueError("mcp server stdout is unavailable")

        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise ValueError("timed out waiting for mcp server response")

        fd = stdout.fileno()
        selector = selectors.DefaultSelector()
        try:
            selector.register(fd, selectors.EVENT_READ)
            events = selector.select(timeout)
        finally:
            selector.close()
        if not events:
            raise ValueError(self._format_error("timed out waiting for mcp server response"))

        chunk = os.read(fd, 4096)
        if not chunk:
            raise ValueError(self._format_error("mcp server closed stdout unexpectedly"))
        self._buffer.extend(chunk)

    def _require_process(self) -> subprocess.Popen[bytes]:
        process = self._process
        if process is None:
            raise ValueError("mcp server process is not running")
        return process

    def _graceful_shutdown(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            self._request("shutdown", {}, timeout_seconds=1.0)
        except ValueError:
            pass
        try:
            self._notify("exit", {})
        except ValueError:
            pass

    def _format_error(self, message: str) -> str:
        stderr_excerpt = self._read_available_stderr()
        if stderr_excerpt:
            return f"{message}; stderr: {stderr_excerpt}"
        return message

    def _read_available_stderr(self) -> str:
        process = self._process
        if process is None or process.stderr is None:
            return ""
        fd = process.stderr.fileno()
        chunks: list[bytes] = []
        selector = selectors.DefaultSelector()
        try:
            selector.register(fd, selectors.EVENT_READ)
            while selector.select(0):
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            selector.close()
        return b"".join(chunks).decode("utf-8", errors="replace").strip()


class MCPTerminalToolRuntime:
    def __init__(
        self,
        *,
        server_command: Sequence[str],
        tool_name: str = "propose_investigation_step",
        startup_timeout_seconds: float = 5.0,
        call_timeout_seconds: float = 10.0,
    ) -> None:
        self.tool_name = tool_name
        self._client = StdioMCPClient(
            command=server_command,
            startup_timeout_seconds=startup_timeout_seconds,
            call_timeout_seconds=call_timeout_seconds,
        )
        self._verified_tool = False

    def close(self) -> None:
        self._client.close()

    def emit(
        self,
        *,
        query_id: str,
        query_text: str,
        decision: TriageDecision,
    ) -> AuditRecord:
        if not self._verified_tool:
            available_tools = {tool["name"] for tool in self._client.list_tools()}
            if self.tool_name not in available_tools:
                raise ValueError(f"mcp server does not expose tool {self.tool_name}")
            self._verified_tool = True

        arguments = {
            "query_id": query_id,
            "query_text": query_text,
            "action": decision.action,
            "disposition": decision.disposition,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "cited_evidence_ids": list(decision.cited_evidence_ids),
        }
        result = self._client.call_tool(tool_name=self.tool_name, arguments=arguments)
        return AuditRecord(
            kind="tool_call",
            name=self.tool_name,
            inputs=arguments,
            outputs=result,
        )
