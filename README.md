# alert-triage-li phase-7 MCP terminal tool slice

This repository currently implements the phase-7 slice of the merged
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
- phase-5 provider-ready triage, judge, and terminal-tool runtime seams
- phase-5 replay-backed tier-2 runtime exercising external-runtime-shaped payload flow
- phase-6 live OpenAI Responses and Anthropic Messages tier-2 runtime adapters
- phase-7 stdio MCP-backed terminal-tool runtime behind the existing reasoning seam
- phase-2 paired fp16-vs-binary performance harness
- Rust kernel, DataFusion UDF, and PyO3 shim crates
- phase-1 through phase-7 unit, contract, and integration tests for the exercised slice,
  with Phase-7 coverage focused on unit-level seam and harness regression tests

What is intentionally not here yet:

- ingestion connectors
- the upstream Lance package / storage engine; the current `lance-mv` surface is an
  executable Arrow-backed adapter boundary, not a production Lance dependency
- full production workflow integration; Phase-7 adds MCP-backed terminal-tool
  execution, but it still does not claim arbitrary MCP orchestration or full
  connector rollout
- provider SDK integration, retry/backoff orchestration, or CI-backed live
  credential verification
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
.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --runtime replay \
  --runtime-fixture tests/fixtures/phase1_tier1/replay_runtime.json \
  --llm-model replay:fixture-triager-v1 \
  --judge-model replay:fixture-judge-v1 \
  --out-json data/runs/reports/tier2_replay.json \
  --out-trace data/runs/traces/tier2_replay_trace.jsonl
.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --runtime openai \
  --llm-model openai:gpt-5 \
  --judge-model openai:gpt-5 \
  --out-json data/runs/reports/tier2_openai.json \
  --out-trace data/runs/traces/tier2_openai_trace.jsonl
.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --runtime anthropic \
  --llm-model anthropic:claude-sonnet-4-5 \
  --judge-model anthropic:claude-sonnet-4-5 \
  --out-json data/runs/reports/tier2_anthropic.json \
  --out-trace data/runs/traces/tier2_anthropic_trace.jsonl
.venv/bin/python -m alert_triage.evals.tier2_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --threads 1 \
  --rerank-depth 2 \
  --terminal-runtime mcp \
  --mcp-server-command python3 \
  --mcp-server-arg tests/fixtures/mcp/fake_terminal_server.py \
  --out-json data/runs/reports/tier2_mcp_terminal.json \
  --out-trace data/runs/traces/tier2_mcp_terminal_trace.jsonl
.venv/bin/python -m alert_triage.evals.tier3_phase4 \
  --fixture-dir tests/fixtures/phase1_tier1 \
  --tier2-json data/runs/reports/tier2.json \
  --out-json data/runs/reports/tier3.json
cargo test --workspace --manifest-path rust/Cargo.toml
cargo bench -p hamming_maxsim_kernel --manifest-path rust/Cargo.toml --bench kernel_bench
```

## Phase-7 status

- The executable retrieval claim now includes `bm25`, `lance-mv`,
  `hamming-udf-bin`, `binary-then-fp16-rerank`, and
  `hybrid-bm25-then-fp16-rerank` on a checked-in fixture pack.
- `lance-mv` is currently a file-backed Arrow adapter that preserves the future
  Lance corpus contract without claiming the upstream Lance package is installed
  in this repo.
- Raw SQL filter strings are still rejected at the shared `QueryBundle`
  boundary, and phase-2 lowers the typed filter shell into the binary retrieval
  path for the allowlisted operators exercised by the tests.
- The repo now ships runtime-selected tier-2 reasoning over the checked-in
  fixture pack. The default `local` runtime preserves the deterministic Phase-4
  behavior, the `replay` runtime exercises provider-shaped payload flow, and
  the `openai` / `anthropic` runtimes execute live provider-backed triage and
  judge calls while preserving the same report, audit, and terminality
  contract.
- The final terminal tool can now remain local or execute through a configured
  stdio MCP server while preserving the same three-record audit sequence and
  terminality contract.
- The repo still does not claim arbitrary MCP orchestration, provider retry /
  backoff, or CI-backed live credential verification in this slice.

See `/Users/jbz/src/secops-triage-li-maxsim/docs/phase1.md` for the phase-1
fixture details, `/Users/jbz/src/secops-triage-li-maxsim/docs/phase3.md` for the
Phase-3 retrieval-surface notes, `/Users/jbz/src/secops-triage-li-maxsim/docs/phase4.md`
for the phase-4 triage harness notes, `/Users/jbz/src/secops-triage-li-maxsim/docs/phase5.md`
for the phase-5 runtime seam notes, `/Users/jbz/src/secops-triage-li-maxsim/docs/phase6.md`
for the phase-6 live-provider adapter notes, and `/Users/jbz/src/secops-triage-li-maxsim/docs/phase7.md`
for the phase-7 MCP terminal-runtime notes. See `data/runs/reports/tier1_phase2_perf.json`
for the minimum phase-2 paired performance artifact,
`data/runs/reports/tier1_phase3.json` for the phase-3 comparison artifact,
`data/runs/reports/tier2.json` for the phase-4 local reasoning artifact,
`data/runs/reports/tier2_replay.json` for the phase-5 replay reasoning artifact, and
`data/runs/reports/tier3.json` for the phase-4 disposition artifact.
