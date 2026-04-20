# Phase-4 Triage Evaluation Slice

Phase-4 adds the first executable reasoning, disposition, and auditability
layer on top of the completed retrieval surfaces.

## Included harnesses

- `tier2_phase4`: runs `binary-then-fp16-rerank`, emits deterministic
  investigation-step proposals, records audit traces, and scores:
  - `action_validity`
  - `evidence_grounding`
  - `specificity`
  - `calibration`
- `tier3_phase4`: scores predicted dispositions against checked-in analyst
  labels and emits:
  - `weighted_f1`
  - `cohens_kappa`

## Important claim boundary

This slice is executable and deterministic, but it does not claim live external
LLM or MCP execution. The current implementation uses a local rule-based triage
engine and judge so the report schema, trace schema, and terminality boundary
can be verified in-repo before any provider-specific wiring is attempted.

## Harness commands

```bash
./.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --out-json data/runs/reports/tier2.json \
  --out-trace data/runs/traces/tier2_trace.jsonl

./.venv/bin/python -m alert_triage.evals.tier3_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --tier2-json data/runs/reports/tier2.json \
  --out-json data/runs/reports/tier3.json
```

## Output shape

- `data/runs/reports/tier2.json`: aggregate reasoning metrics plus per-sample
  outputs, terminality, and orphan-audit counts
- `data/runs/traces/tier2_trace.jsonl`: one JSON line per sample with audit
  records and terminal output data
- `data/runs/reports/tier3.json`: aggregate disposition metrics plus per-label
  support and confusion data

## Remaining later work

- provider-ready runtime adapter seam for tier-2 execution, landed in Phase-5
- live provider-backed model adapters if approved
- richer analyst-label datasets beyond the checked-in fixture pack
- broader benchmark sweeps and production workflow integration
