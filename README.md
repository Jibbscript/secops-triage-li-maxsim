# alert-triage-li phase-3 retrieval surface slice

This repository currently implements the phase-3 slice of the merged
`alert-triage-li` + `binary hamming maxsim` design:

- Python encoder and retriever contracts
- binary calibrator persistence helpers
- fp16, lexical, and binary reference scorers
- in-memory fp16 reference retriever and token-vector store
- checked-in fixture text pack for lexical and hybrid retrieval
- phase-1 tier-1 evaluation harness over a tiny checked-in fixture pack
- phase-2 `register(ctx)` bridge wiring into a Python-owned DataFusion `SessionContext`
- DataFusion-backed binary candidate generation plus typed filter lowering
- staged binary candidate generation + fp16 rerank integration
- lexical `bm25` retrieval surface
- file-backed `lance-mv` adapter surface for fp16 maxsim over a Lance-shaped corpus contract
- staged `hybrid-bm25-then-fp16-rerank` integration
- phase-3 retrieval comparison harness spanning lexical, fp16, binary, and staged lanes
- phase-2 paired fp16-vs-binary performance harness
- Rust kernel, DataFusion UDF, and PyO3 shim crates
- phase-1/phase-2/phase-3 unit, contract, and integration tests for the exercised slice

What is intentionally not here yet:

- ingestion connectors
- the full evaluation harness from `eval_bundle.yaml`
- the upstream Lance package / storage engine; the current `lance-mv` surface is an
  executable Arrow-backed adapter boundary, not a production Lance dependency
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
.venv/bin/python -m alert_triage.evals.tier1_phase3 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --out-json data/runs/reports/tier1_phase3.json \
  --out-latency data/runs/reports/latency_phase3.csv
cargo test --workspace --manifest-path rust/Cargo.toml
cargo bench -p hamming_maxsim_kernel --manifest-path rust/Cargo.toml --bench kernel_bench
```

## Phase-3 status

- The executable retrieval claim now includes `bm25`, `lance-mv`,
  `hamming-udf-bin`, `binary-then-fp16-rerank`, and
  `hybrid-bm25-then-fp16-rerank` on a checked-in fixture pack.
- `lance-mv` is currently a file-backed Arrow adapter that preserves the future
  Lance corpus contract without claiming the upstream Lance package is installed
  in this repo.
- Raw SQL filter strings are still rejected at the shared `QueryBundle`
  boundary, and phase-2 lowers the typed filter shell into the binary retrieval
  path for the allowlisted operators exercised by the tests.
- The repo still does not claim tier-2 reasoning, tier-3 disposition, audit
  wiring, or full eval-bundle coverage. Those remain phase-4 work.

See `/Users/jbz/src/secops-triage-li-maxsim/docs/phase1.md` for the phase-1
fixture details, `/Users/jbz/src/secops-triage-li-maxsim/docs/phase3.md` for the
Phase-3 retrieval-surface notes, `data/runs/reports/tier1_phase2_perf.json` for
the minimum phase-2 paired performance artifact, and
`data/runs/reports/tier1_phase3.json` for the phase-3 comparison artifact.
