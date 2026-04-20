# Phase-3 Retrieval Surface Slice

Phase-3 widens the executable retrieval set beyond the Phase-2 binary lane.

## Included retrieval surfaces

- `bm25`: small deterministic lexical retrieval over checked-in fixture text
- `lance-mv`: fp16 maxsim over a file-backed Arrow adapter that preserves the
  future Lance corpus contract
- `hamming-udf-bin`: DataFusion-backed binary candidate generation
- `binary-then-fp16-rerank`: binary candidate generation followed by fp16 rerank
- `hybrid-bm25-then-fp16-rerank`: lexical candidate generation followed by fp16 rerank

## Important claim boundary

`lance-mv` is executable in this slice, but it is not the upstream Lance package
or storage engine. The current implementation uses an Arrow-backed adapter at
`src/alert_triage/storage/lance_adapter.py` so the retrieval contract, harness,
and docs can stay stable until a real Lance dependency is approved.

## Harness command

```bash
./.venv/bin/python -m alert_triage.evals.tier1_phase3 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --out-json data/runs/reports/tier1_phase3.json \
  --out-latency data/runs/reports/latency_phase3.csv
```

## Output shape

- `data/runs/reports/tier1_phase3.json`: per-retriever quality and latency metrics
- `data/runs/reports/latency_phase3.csv`: per-query latency rows with the executed stage

## Remaining later-phase work

- real Lance dependency / backend adoption if approved
- tier-2 reasoning and tier-3 disposition harnesses
- agent auditability and terminality gates from `eval_bundle.yaml`
