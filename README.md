# generated artifacts

this bundle contains first-pass copy-paste-ready code artifacts for the merged
`alert-triage-li` + `binary hamming maxsim` design.

included:
- eval_bundle.yaml
- revised python protocol files
- binary calibrator helper
- binary candidate-gen + rerank retrievers
- rust cargo workspace
- kernel crate
- datafusion udf crate
- pyo3 shim crate

note:
the pyo3 `register()` bridge is intentionally left as the one version-sensitive stub,
because the exact extraction path for a Python-owned `SessionContext` depends on
the aligned versions of `datafusion`, `datafusion-ffi`, and the Rust bindings.
