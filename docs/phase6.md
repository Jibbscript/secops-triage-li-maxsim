# Phase-6 Live Provider Tier-2 Runtime Adapter Slice

Phase-6 advances the Phase-5 runtime seam into executable live provider-backed
tier-2 reasoning adapters.

## Included live runtimes

- retained deterministic `local` triage and judge runtime for regression
  stability
- retained `replay` runtime for checked-in provider-shaped fixture flow
- live `openai` triage and judge adapters using the Responses API
- live `anthropic` triage and judge adapters using the Messages API
- preserved local terminal-tool runtime so the terminality boundary stays
  `propose_investigation_step`

## Important claim boundary

This slice adds live provider-backed tier-2 model execution, but it does not
claim:

- production MCP connector wiring
- provider SDK integration
- CI-backed live credential verification
- retries, backoff orchestration, or production rate-limit handling

The deterministic `local` lane remains the default regression fixture, and the
existing report, trace, and audit schema are preserved.

## Harness commands

```bash
./.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --runtime openai \
  --llm-model openai:gpt-5 \
  --judge-model openai:gpt-5 \
  --out-json data/runs/reports/tier2_openai.json \
  --out-trace data/runs/traces/tier2_openai_trace.jsonl

./.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --runtime anthropic \
  --llm-model anthropic:claude-sonnet-4-5 \
  --judge-model anthropic:claude-sonnet-4-5 \
  --out-json data/runs/reports/tier2_anthropic.json \
  --out-trace data/runs/traces/tier2_anthropic_trace.jsonl
```

Required environment:

- `OPENAI_API_KEY` for `runtime=openai`
- `ANTHROPIC_API_KEY` for `runtime=anthropic`

Optional runtime overrides:

- `--api-key-env` to read credentials from a non-default environment variable
- `--api-base-url` for provider-compatible proxy or test endpoints
- `--timeout-seconds` to change request timeout behavior

## Output shape

- `data/runs/reports/tier2.json`: deterministic local tier-2 report
- `data/runs/reports/tier2_replay.json`: replay-runtime tier-2 report
- `data/runs/reports/tier2_openai.json`: live OpenAI tier-2 report
- `data/runs/reports/tier2_anthropic.json`: live Anthropic tier-2 report
- all tier-2 traces preserve the existing three-record sample shape:
  triager `model_call`, judge `model_call`, terminal `tool_call`

## Remaining later work

- MCP-backed terminal-tool execution, landed in `/Users/jbz/src/secops-triage-li-maxsim/docs/phase7.md`
- provider retry/backoff and rate-limit strategy
- broader benchmark sweeps and production workflow integration
