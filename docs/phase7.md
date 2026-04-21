# Phase-7 MCP Terminal Tool Runtime Slice

Phase-7 advances the Phase-6 live-provider tier-2 slice into executable
MCP-backed terminal-tool execution.

## Included terminal runtimes

- retained deterministic `local` terminal-tool runtime for regression stability
- added `mcp` terminal-tool runtime that speaks stdio MCP to one configured tool
- retained the existing local / replay / provider triage and judge runtime seams

## Important claim boundary

This slice adds MCP-backed execution for the final terminal tool, but it does
not claim:

- full production workflow orchestration across arbitrary MCP servers
- MCP-backed triage or judge model execution
- retries, backoff, or pooled MCP session management
- CI-backed verification against live third-party MCP servers

The deterministic local tool lane remains the default regression fixture, and
the existing report, trace, and terminality contract are preserved.

## Harness command

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

## Required MCP settings

- `--mcp-server-command` for the server executable
- repeated `--mcp-server-arg` values for server arguments
- optional `--mcp-tool-name` if the terminal tool name differs from the default
  `propose_investigation_step`

## Output shape

- `data/runs/reports/tier2.json`: deterministic local terminal-tool report
- `data/runs/reports/tier2_mcp_terminal.json`: MCP terminal-tool report
- both traces preserve the existing three-record sample shape:
  triager `model_call`, judge `model_call`, terminal `tool_call`

## Remaining later work

- broader production workflow integration around the MCP terminal lane
- provider retry / backoff strategies
- richer benchmark sweeps and non-fixture workflow validation
