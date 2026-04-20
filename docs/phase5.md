# Phase-5 Triage Runtime Adapter Seam Slice

Phase-5 makes the Phase-4 triage harness provider-ready without claiming live
provider or production MCP execution.

## Included runtime seams

- explicit triage, judge, and terminal-tool runtime interfaces
- retained deterministic local runtime for regression stability
- replay-backed runtime that loads checked-in fixtures and emits the same audit
  and terminality schema as the local lane

## Important claim boundary

This slice does not execute live network requests or production MCP tools. It
adds the runtime seam that later live-provider work can plug into while keeping
the deterministic local lane as the stable regression fixture.

## Harness commands

```bash
./.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --out-json data/runs/reports/tier2.json \
  --out-trace data/runs/traces/tier2_trace.jsonl

./.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --runtime replay \
  --runtime-fixture tests/fixtures/phase1_tier1/replay_runtime.json \
  --llm-model replay:fixture-triager-v1 \
  --judge-model replay:fixture-judge-v1 \
  --out-json data/runs/reports/tier2_replay.json \
  --out-trace data/runs/traces/tier2_replay_trace.jsonl
```

## Output shape

- `data/runs/reports/tier2.json`: deterministic local tier-2 report
- `data/runs/reports/tier2_replay.json`: replay-runtime tier-2 report
- `data/runs/traces/tier2_trace.jsonl`: local runtime trace
- `data/runs/traces/tier2_replay_trace.jsonl`: replay runtime trace

## Remaining later work

- live provider-backed model adapters if approved
- production MCP connector wiring
- broader benchmark sweeps and production workflow integration
