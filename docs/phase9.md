# Phase-9 MCP Terminal Adapter Profile

Phase-9 advances the Phase-8 raw external MCP proof into a bounded terminal
adapter profile for the tier-2 reasoning harness.

## Included in this slice

- explicit MCP terminal profile selection:
  - `direct`
  - `everything_echo`
- preserved repo-owned terminal audit/report contract:
  - terminal audit record name remains `propose_investigation_step`
  - report `terminal_tool` remains `propose_investigation_step`
- normalized adapter metadata for the `everything_echo` profile:
  - `mcp_profile`
  - `mcp_tool_name`
  - `raw_mcp_result`
- fake-server regression coverage for direct and `everything_echo` paths
- opt-in live tier-2 verification against the official Everything reference
  server through the `everything_echo` profile

## Direct MCP harness command

```bash
./.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --terminal-runtime mcp \
  --mcp-server-command python3 \
  --mcp-server-arg tests/fixtures/mcp/fake_terminal_server.py \
  --out-json data/runs/reports/tier2_mcp_terminal.json \
  --out-trace data/runs/traces/tier2_mcp_terminal_trace.jsonl
```

## Everything echo profile command

```bash
./.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --terminal-runtime mcp \
  --mcp-profile everything_echo \
  --mcp-server-command npx \
  --mcp-server-arg -y \
  --mcp-server-arg @modelcontextprotocol/server-everything@2025.8.4 \
  --mcp-server-arg stdio \
  --out-json data/runs/reports/tier2_mcp_everything_echo.json \
  --out-trace data/runs/traces/tier2_mcp_everything_echo_trace.jsonl
```

## Opt-in live pytest lane

```bash
RUN_EXTERNAL_MCP_LIVE=1 ./.venv/bin/python -m pytest -m external_mcp_live
```

This lane now covers both:

- the Phase-8 standalone reference probe
- the Phase-9 tier-2 `everything_echo` adapter path

## Important claim boundary

Phase-9 proves a bounded external MCP adapter profile for the official
Everything server `echo` tool. It does not yet claim:

- arbitrary external terminal-tool compatibility
- arbitrary multi-tool or multi-server MCP orchestration
- MCP-backed triage or judge execution
- retry/backoff policy for provider or MCP transports

## Remaining later work

- Phase-10 shared retry/backoff policy
- later deliberate widening of MCP orchestration policy, if justified
