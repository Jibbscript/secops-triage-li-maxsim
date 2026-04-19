# alert-triage-li phase-0 scaffold

This repository currently implements the phase-0 slice of the merged
`alert-triage-li` + `binary hamming maxsim` design:

- Python encoder and retriever contracts
- binary calibrator persistence helpers
- fp16 and binary reference scorers
- staged binary candidate generation + fp16 rerank contract
- Rust kernel, DataFusion UDF, and PyO3 shim crates
- phase-0 unit and contract tests

What is intentionally not here yet:

- ingestion connectors
- storage backends
- full evaluation harness
- agent orchestration and MCP wiring
- benchmark sweep infrastructure beyond the Rust workspace scaffold

## Local verification

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pytest
cargo test --workspace --manifest-path rust/Cargo.toml
```

## Notes

The PyO3 `register()` bridge remains the version-sensitive seam between the
Python `datafusion` package and the Rust `datafusion-ffi` handle model. Phase-0
proves the UDF contract in Rust directly and keeps the Python bridge explicit
instead of silently claiming end-to-end registration is already done.
