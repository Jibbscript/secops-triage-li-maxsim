# Phase-8 MCP Compatibility Profile and Live Reference Verification

Phase-8 advances the Phase-7 MCP terminal-tool slice by proving the underlying
stdio MCP client against one official external reference target:
`@modelcontextprotocol/server-everything@2025.8.4`.

## Included in this slice

- stable MCP error classification through repo-owned typed exceptions with a
  `.code` field
- standalone reference probe entrypoint at
  `/Users/jbz/src/secops-triage-li-maxsim/src/alert_triage/evals/mcp_reference_probe.py`
- deterministic unit coverage for:
  - configuration failure
  - startup failure
  - tool discovery failure
  - tool-not-found failure
  - tool-result failure
- opt-in live integration verification against the official Everything
  reference server via the `echo` tool

## Manual reference probe

```bash
./.venv/bin/python -m alert_triage.evals.mcp_reference_probe \
  --out-json data/runs/reports/mcp_reference_probe.json
```

## Opt-in live pytest lane

```bash
RUN_EXTERNAL_MCP_LIVE=1 ./.venv/bin/python -m pytest -m external_mcp_live
```

## Important claim boundary

Phase-8 verifies only the live external stdio MCP client path against the
official Everything reference server. It does not yet claim:

- generic external terminal-tool compatibility for
  `propose_investigation_step`
- broader MCP orchestration across multiple tools or servers
- MCP-backed triage or judge execution
- retry/backoff policy for provider or MCP transports

## Remaining later work

- Phase-9 broader MCP orchestration
- Phase-10 shared retry/backoff policy
