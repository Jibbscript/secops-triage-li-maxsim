# alert-triage-li phase-1 baseline slice

This repository currently implements the phase-1 slice of the merged
`alert-triage-li` + `binary hamming maxsim` design:

- Python encoder and retriever contracts
- binary calibrator persistence helpers
- fp16 and binary reference scorers
- in-memory fp16 reference retriever and token-vector store
- phase-1 tier-1 evaluation harness over a tiny checked-in fixture pack
- staged binary candidate generation + fp16 rerank contract
- Rust kernel, DataFusion UDF, and PyO3 shim crates
- phase-1 unit and contract tests for the exercised baseline slice

What is intentionally not here yet:

- ingestion connectors
- production storage backends such as Lance
- the full evaluation harness from `eval_bundle.yaml`
- agent orchestration and MCP wiring
- phase-2 binary `register(ctx)` bridge wiring
- benchmark sweep infrastructure beyond the Rust workspace scaffold

## Local verification

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pytest
.venv/bin/python -m alert_triage.evals.tier1_phase1 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --out-json data/runs/reports/tier1.json \
  --out-latency data/runs/reports/latency.csv
cargo test --workspace --manifest-path rust/Cargo.toml
```

## Phase-1 status

- The executable retrieval claim is limited to the in-memory fp16 reference
  lane. The repo does not yet claim `lance-mv`, `bm25`, or full eval-bundle
  coverage.
- Raw SQL filter strings are rejected at the shared `QueryBundle` boundary.
  Phase-1 exposes only a typed filter placeholder so later work can add
  lowering without redesigning the API.
- The PyO3 `register()` bridge remains the version-sensitive seam between the
  Python `datafusion` package and the Rust `datafusion-ffi` handle model. The
  phase-1 bridge lane records proof or negative proof only; it is not yet a
  product-ready retrieval path.

See `/Users/jbz/src/secops-triage-li-maxsim/docs/phase1.md` for the phase-1
fixture and harness details.
