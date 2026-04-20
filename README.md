# alert-triage-li phase-4 triage evaluation slice

This repository currently implements the phase-4 slice of the merged
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
- deterministic phase-4 tier-2 reasoning harness over `binary-then-fp16-rerank`
- deterministic phase-4 tier-3 disposition scoring harness
- audit and terminality trace artifacts for the local triage lane
- phase-2 paired fp16-vs-binary performance harness
- Rust kernel, DataFusion UDF, and PyO3 shim crates
- phase-1 through phase-4 unit, contract, and integration tests for the exercised slice

What is intentionally not here yet:

- ingestion connectors
- the upstream Lance package / storage engine; the current `lance-mv` surface is an
  executable Arrow-backed adapter boundary, not a production Lance dependency
- live provider-backed LLM execution or production MCP wiring; the current
  phase-4 slice is a deterministic local harness that preserves the report and
  audit boundaries without claiming external service integration
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
.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --out-json data/runs/reports/tier2.json \
  --out-trace data/runs/traces/tier2_trace.jsonl
.venv/bin/python -m alert_triage.evals.tier3_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --tier2-json data/runs/reports/tier2.json \
  --out-json data/runs/reports/tier3.json
cargo test --workspace --manifest-path rust/Cargo.toml
cargo bench -p hamming_maxsim_kernel --manifest-path rust/Cargo.toml --bench kernel_bench
```

## Phase-4 status

- The executable retrieval claim now includes `bm25`, `lance-mv`,
  `hamming-udf-bin`, `binary-then-fp16-rerank`, and
  `hybrid-bm25-then-fp16-rerank` on a checked-in fixture pack.
- `lance-mv` is currently a file-backed Arrow adapter that preserves the future
  Lance corpus contract without claiming the upstream Lance package is installed
  in this repo.
- Raw SQL filter strings are still rejected at the shared `QueryBundle`
  boundary, and phase-2 lowers the typed filter shell into the binary retrieval
  path for the allowlisted operators exercised by the tests.
- The repo now ships a deterministic local tier-2/tier-3 harness over the
  checked-in fixture pack. It emits reasoning metrics, disposition metrics, and
  audit/terminality traces without claiming live provider-backed LLM execution.

See `/Users/jbz/src/secops-triage-li-maxsim/docs/phase1.md` for the phase-1
fixture details, `/Users/jbz/src/secops-triage-li-maxsim/docs/phase3.md` for the
Phase-3 retrieval-surface notes, `/Users/jbz/src/secops-triage-li-maxsim/docs/phase4.md`
for the phase-4 triage harness notes, `data/runs/reports/tier1_phase2_perf.json`
for the minimum phase-2 paired performance artifact,
`data/runs/reports/tier1_phase3.json` for the phase-3 comparison artifact,
`data/runs/reports/tier2.json` for the phase-4 reasoning artifact, and
`data/runs/reports/tier3.json` for the phase-4 disposition artifact.
