# Phase-10 Shared Retry / Backoff Policy

Phase-10 advances the Phase-9 bounded provider + MCP runtime surface into a
shared transport resilience slice.

## Included in this slice

- one shared retry/backoff policy abstraction reused by:
  - provider HTTP transport requests
  - bounded stdio MCP startup, discovery, and terminal tool-call requests
- deterministic defaults:
  - single attempt
  - no implicit sleep
- explicit tier-2 harness retry flags:
  - `--retry-max-attempts`
  - `--retry-initial-backoff-seconds`
  - `--retry-max-backoff-seconds`
  - `--retry-backoff-multiplier`
- provider transport retry classification for transient HTTP/network failures
- MCP retry handling for transient timeout / transport failures in the bounded
  terminal-tool lane
- stateful fake-server regression coverage for one-time startup and tool-call
  timeouts

## Example retry-enabled harness command

```bash
./.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --runtime openai \
  --terminal-runtime mcp \
  --mcp-server-command python3 \
  --mcp-server-arg tests/fixtures/mcp/fake_terminal_server.py \
  --retry-max-attempts 2 \
  --retry-initial-backoff-seconds 0.05 \
  --out-json data/runs/reports/tier2_retry_demo.json \
  --out-trace data/runs/traces/tier2_retry_demo_trace.jsonl
```

## Important claim boundary

Phase-10 adds a bounded shared retry/backoff layer for the current provider
HTTP path and the repo-owned terminal-tool MCP lane. It does not yet claim:

- full production rate-limit strategy or circuit breaking
- arbitrary MCP tool-call retry safety beyond the bounded repo-owned terminal
  tool contract
- broader multi-tool or multi-server orchestration resilience
- CI-mandated live verification with retries enabled by default

## Remaining later work

- broader operational timeout budgeting across full workflows
- deliberate widening of MCP orchestration policy, if justified
