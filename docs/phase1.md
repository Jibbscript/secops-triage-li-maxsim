# Phase-1 Baseline Slice

Phase-1 intentionally ships only the executable in-memory fp16 reference lane.
It does not claim the Lance-backed baseline, lexical retrieval, or live binary
candidate generation yet.

## Included artifacts

- `src/alert_triage/storage/in_memory.py`
- `src/alert_triage/retrievers/fp16_ref.py`
- `src/alert_triage/evals/tier1_phase1.py`
- `tests/fixtures/phase1_tier1/`

## Harness command

```bash
./.venv/bin/python -m alert_triage.evals.tier1_phase1 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --out-json data/runs/reports/tier1.json \
  --out-latency data/runs/reports/latency.csv
```

## Filter contract

`QueryBundle.filter_expr` is rejected at model construction time. Phase-1 keeps
only a typed filter placeholder at `QueryBundle.filter`; executable retrievers
explicitly reject non-empty structured filters until lowering exists.
