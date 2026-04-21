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
_MCP_PROTOCOL_VERSION = "2025-06-18"
_DEFAULT_TERMINAL_TOOL_NAME = "propose_investigation_step"
_DIRECT_MCP_PROFILE = "direct"
_EVERYTHING_ECHO_MCP_PROFILE = "everything_echo"
_EVERYTHING_ECHO_TOOL_NAME = "echo"


class MCPError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = {} if details is None else dict(details)


def _normalize_mcp_terminal_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if normalized in {_DIRECT_MCP_PROFILE, _EVERYTHING_ECHO_MCP_PROFILE}:
        return normalized
    raise MCPError(
        f"unsupported mcp terminal tool profile: {profile}",
        code="mcp_profile_invalid",
    )


class StdioMCPClient:
    def __init__(
        self,
        *,
        command: Sequence[str],
        startup_timeout_seconds: float = 5.0,
        call_timeout_seconds: float = 10.0,
    ) -> None:
        if not command:
            raise MCPError("mcp server command is required", code="mcp_configuration_invalid")
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
        result = self._request(
            "tools/list",
            {},
            timeout_seconds=self._call_timeout_seconds,
            error_code="mcp_tool_discovery_failed",
        )
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise MCPError(
                "mcp tools/list result must contain a tools list",
                code="mcp_tool_discovery_failed",
            )
        normalized = []
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                raise MCPError(
                    "mcp tools/list returned an invalid tool entry",
                    code="mcp_tool_discovery_failed",
                )
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
            error_code="mcp_tool_result_invalid",
        )
        if not isinstance(result, dict):
            raise MCPError("mcp tools/call result must be an object", code="mcp_tool_result_invalid")
        if result.get("isError") is True:
            raise MCPError(
                f"mcp tool {tool_name} returned an error result",
                code="mcp_tool_result_invalid",
            )
        content = result.get("content")
        if content is not None and not isinstance(content, list):
            raise MCPError(
                "mcp tools/call result content must be a list",
                code="mcp_tool_result_invalid",
            )
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
            raise MCPError(
                f"failed to launch mcp server {self._command[0]}: {exc.strerror}",
                code="mcp_startup_failed",
            ) from exc

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
            error_code="mcp_startup_failed",
        )
        protocol_version = result.get("protocolVersion")
        if not isinstance(protocol_version, str):
            raise MCPError("mcp initialize result missing protocolVersion", code="mcp_startup_failed")
        self._notify("notifications/initialized", {}, error_code="mcp_startup_failed")

    def _notify(self, method: str, params: dict[str, object], *, error_code: str) -> None:
        self._write_message(
            {
                "jsonrpc": _JSONRPC_VERSION,
                "method": method,
                "params": params,
            },
            error_code=error_code,
        )

    def _request(
        self,
        method: str,
        params: dict[str, object],
        *,
        timeout_seconds: float,
        error_code: str,
    ) -> dict[str, Any]:
        self._ensure_started()
        request_id = self._next_id
        self._next_id += 1
        self._write_message(
            {
                "jsonrpc": _JSONRPC_VERSION,
                "id": request_id,
                "method": method,
                "params": params,
            },
            error_code=error_code,
        )
        response = self._read_message(timeout_seconds=timeout_seconds, error_code=error_code)
        if response.get("id") != request_id:
            raise MCPError(f"mcp response id mismatch for {method}", code=error_code)
        error_payload = response.get("error")
        if error_payload is not None:
            raise MCPError(f"mcp {method} failed: {error_payload}", code=error_code)
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPError(f"mcp {method} result must be an object", code=error_code)
        return result

    def _write_message(self, payload: dict[str, object], *, error_code: str) -> None:
        process = self._require_process(error_code=error_code)
        stdin = process.stdin
        if stdin is None:
            raise MCPError("mcp server stdin is unavailable", code=error_code)
        message = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            stdin.write(message)
            stdin.flush()
        except BrokenPipeError as exc:
            raise MCPError(
                self._format_error("mcp server closed stdin unexpectedly"),
                code=error_code,
            ) from exc

    def _read_message(self, *, timeout_seconds: float, error_code: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        body = self._read_until(b"\n", deadline=deadline, error_code=error_code).rstrip(b"\n")
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise MCPError("mcp response body is not valid JSON", code=error_code) from exc
        if not isinstance(payload, dict):
            raise MCPError("mcp response body must be a JSON object", code=error_code)
        return payload

    def _read_until(self, marker: bytes, *, deadline: float, error_code: str) -> bytes:
        while True:
            index = self._buffer.find(marker)
            if index >= 0:
                end = index + len(marker)
                chunk = bytes(self._buffer[:end])
                del self._buffer[:end]
                return chunk
            self._pump_stdout(deadline=deadline, error_code=error_code)

    def _pump_stdout(self, *, deadline: float, error_code: str) -> None:
        process = self._require_process(error_code=error_code)
        stdout = process.stdout
        if stdout is None:
            raise MCPError("mcp server stdout is unavailable", code=error_code)

        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise MCPError("timed out waiting for mcp server response", code=error_code)

        fd = stdout.fileno()
        selector = selectors.DefaultSelector()
        try:
            selector.register(fd, selectors.EVENT_READ)
            events = selector.select(timeout)
        finally:
            selector.close()
        if not events:
            raise MCPError(
                self._format_error("timed out waiting for mcp server response"),
                code=error_code,
            )

        chunk = os.read(fd, 4096)
        if not chunk:
            raise MCPError(
                self._format_error("mcp server closed stdout unexpectedly"),
                code=error_code,
            )
        self._buffer.extend(chunk)

    def _require_process(self, *, error_code: str) -> subprocess.Popen[bytes]:
        process = self._process
        if process is None:
            raise MCPError("mcp server process is not running", code=error_code)
        return process

    def _graceful_shutdown(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            self._request("shutdown", {}, timeout_seconds=1.0, error_code="mcp_startup_failed")
        except MCPError:
            pass
        try:
            self._notify("exit", {}, error_code="mcp_startup_failed")
        except MCPError:
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
        tool_name: str = _DEFAULT_TERMINAL_TOOL_NAME,
        mcp_profile: str = _DIRECT_MCP_PROFILE,
        startup_timeout_seconds: float = 5.0,
        call_timeout_seconds: float = 10.0,
    ) -> None:
        self.mcp_profile = _normalize_mcp_terminal_profile(mcp_profile)
        if self.mcp_profile == _EVERYTHING_ECHO_MCP_PROFILE and tool_name != _DEFAULT_TERMINAL_TOOL_NAME:
            raise MCPError(
                "mcp everything_echo profile requires tool_name propose_investigation_step",
                code="mcp_configuration_invalid",
            )
        self.tool_name = (
            _DEFAULT_TERMINAL_TOOL_NAME
            if self.mcp_profile == _EVERYTHING_ECHO_MCP_PROFILE
            else tool_name
        )
        self._mcp_tool_name = (
            self.tool_name
            if self.mcp_profile == _DIRECT_MCP_PROFILE
            else _EVERYTHING_ECHO_TOOL_NAME
        )
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
        repo_arguments = {
            "query_id": query_id,
            "query_text": query_text,
            "action": decision.action,
            "disposition": decision.disposition,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "cited_evidence_ids": list(decision.cited_evidence_ids),
        }
        if not self._verified_tool:
            available_tools = {tool["name"] for tool in self._client.list_tools()}
            if self._mcp_tool_name not in available_tools:
                raise MCPError(
                    f"mcp server does not expose tool {self._mcp_tool_name}",
                    code="mcp_tool_not_found",
                )
            self._verified_tool = True

        mcp_arguments = self._build_mcp_arguments(repo_arguments)
        raw_result = self._client.call_tool(tool_name=self._mcp_tool_name, arguments=mcp_arguments)
        return AuditRecord(
            kind="tool_call",
            name=self.tool_name,
            inputs=repo_arguments,
            outputs=self._normalize_tool_result(raw_result),
        )

    def _build_mcp_arguments(self, repo_arguments: dict[str, object]) -> dict[str, object]:
        if self.mcp_profile == _DIRECT_MCP_PROFILE:
            return repo_arguments
        return {
            "message": json.dumps(repo_arguments, sort_keys=True),
        }

    def _normalize_tool_result(self, raw_result: dict[str, object]) -> dict[str, object]:
        if self.mcp_profile == _DIRECT_MCP_PROFILE:
            return raw_result

        echoed_message = self._extract_echoed_message(raw_result)
        echoed_payload = self._parse_echoed_payload(echoed_message)
        action = echoed_payload.get("action")
        disposition = echoed_payload.get("disposition")
        if not isinstance(action, str) or not isinstance(disposition, str):
            raise MCPError(
                "mcp everything_echo profile echoed payload must contain string action and disposition",
                code="mcp_tool_adapter_result_invalid",
            )
        return {
            "mcp_profile": self.mcp_profile,
            "mcp_tool_name": self._mcp_tool_name,
            "raw_mcp_result": raw_result,
            "content": raw_result.get("content"),
            "structuredContent": {
                "accepted_action": action,
                "accepted_disposition": disposition,
            },
        }

    def _extract_echoed_message(self, raw_result: dict[str, object]) -> str:
        structured = raw_result.get("structuredContent")
        if isinstance(structured, dict):
            echoed = structured.get("echoed")
            if isinstance(echoed, str):
                return echoed

        content = raw_result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    return text

        raise MCPError(
            "mcp everything_echo profile requires echoed text output",
            code="mcp_tool_adapter_result_invalid",
        )

    def _parse_echoed_payload(self, echoed_message: str) -> dict[str, object]:
        candidates = [echoed_message]
        json_object_start = echoed_message.find("{")
        if json_object_start > 0:
            candidates.append(echoed_message[json_object_start:])

        payload: object | None = None
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

        if payload is None:
            raise MCPError(
                "mcp everything_echo profile echoed payload must be valid JSON",
                code="mcp_tool_adapter_result_invalid",
                details={"echoed_message": echoed_message},
            )
        if not isinstance(payload, dict):
            raise MCPError(
                "mcp everything_echo profile echoed payload must be an object",
                code="mcp_tool_adapter_result_invalid",
            )
        return payload
