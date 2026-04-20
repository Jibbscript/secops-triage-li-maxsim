# alert-triage-li phase-2 binary integration slice

This repository currently implements the phase-2 slice of the merged
`alert-triage-li` + `binary hamming maxsim` design:

- Python encoder and retriever contracts
- binary calibrator persistence helpers
- fp16 and binary reference scorers
- in-memory fp16 reference retriever and token-vector store
- phase-1 tier-1 evaluation harness over a tiny checked-in fixture pack
- phase-2 `register(ctx)` bridge wiring into a Python-owned DataFusion `SessionContext`
- DataFusion-backed binary candidate generation plus typed filter lowering
- staged binary candidate generation + fp16 rerank integration
- phase-2 paired fp16-vs-binary performance harness
- Rust kernel, DataFusion UDF, and PyO3 shim crates
- phase-1/phase-2 unit, contract, and integration tests for the exercised slice

What is intentionally not here yet:

- ingestion connectors
- production storage backends such as Lance
- the full evaluation harness from `eval_bundle.yaml`
- agent orchestration and MCP wiring
- benchmark sweep infrastructure beyond the minimum real phase-2 workload

## Local verification

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .[dev]
.venv/bin/python -m pytest
.venv/bin/python -m pip install -e 'rust/crates/hamming_maxsim_py[dev]'
.venv/bin/python -m maturin develop --manifest-path rust/crates/hamming_maxsim_py/Cargo.toml
.venv/bin/python -m alert_triage.evals.tier1_phase1 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --out-json data/runs/reports/tier1.json \
  --out-latency data/runs/reports/latency.csv
.venv/bin/python -m alert_triage.evals.tier1_phase2 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --cache-state warm \
  --threads 1 \
  --rerank-depth 5 \
  --out-json data/runs/reports/tier1_phase2_perf.json
cargo test --workspace --manifest-path rust/Cargo.toml
cargo bench -p hamming_maxsim_kernel --manifest-path rust/Cargo.toml --bench kernel_bench
```

## Phase-2 status

- The executable retrieval claim now includes the DataFusion-backed binary
  candidate lane and the staged `binary-then-fp16-rerank` path on a checked-in
  fixture pack.
- Raw SQL filter strings are still rejected at the shared `QueryBundle`
  boundary, and phase-2 lowers the typed filter shell into the binary retrieval
  path for the allowlisted operators exercised by the tests.
- The repo still does not claim `lance-mv`, `bm25`, hybrid retrieval-surface
  completeness, or full eval-bundle coverage. Those remain later-phase work.

See `/Users/jbz/src/secops-triage-li-maxsim/docs/phase1.md` for the phase-1
fixture details, and `data/runs/reports/tier1_phase2_perf.json` for the minimum
phase-2 paired performance artifact.
