# ADR-0001: Phase-1 Bridge Negative Proof

## Status

Accepted for phase-1 gating.

## Context

The phase-1 plan requires a dedicated bridge-proof lane for the
`rust/crates/hamming_maxsim_py` crate pinned to Python `datafusion==48.0.0`
against the workspace Rust `datafusion = 48` and `datafusion-ffi = 48`
dependency line.

Phase-1 does not claim a working `register(ctx)` bridge. It must record either
an executable smoke proof or a failing repro that blocks phase-2 until the
compatibility tuple is resolved.

## Decision

Run a dedicated smoke workflow in the bridge crate environment:

```bash
python -m pip install -e .[dev]
maturin develop
python tests/register_smoke.py
```

If the command succeeds, phase-2 may proceed from a proof-backed baseline. If it
fails, record the command, failure surface, and blocker here as a negative proof.

## Phase-1 Outcome

Negative proof recorded on 2026-04-19.

Successful repro command path:

```bash
python3 -m venv .venv-bridge-fast
./.venv-bridge-fast/bin/pip install 'datafusion==48.0.0' 'maturin>=1.7,<2.0' 'pytest>=8,<9'
VIRTUAL_ENV="$PWD/.venv-bridge-fast" PATH="$PWD/.venv-bridge-fast/bin:$PATH" maturin develop
VIRTUAL_ENV="$PWD/.venv-bridge-fast" PATH="$PWD/.venv-bridge-fast/bin:$PATH" python tests/register_smoke.py
```

Observed result:

```text
NotImplementedError: SessionContext registration is not implemented in phase-0.
The Rust UDF contract is verified directly in the hamming_maxsim_udf crate;
wire this bridge only after pinning matching Python datafusion and
datafusion-ffi versions.
```

Interpretation:

- The Python `datafusion==48.0.0` package and the Rust `datafusion = 48` build
  can coexist in a dedicated bridge environment.
- The remaining blocker is the unimplemented `register(ctx)` bridge in
  `rust/crates/hamming_maxsim_py/src/register.rs`, so phase-2 still requires the
  actual bridge implementation and compatibility validation before the binary
  lane can be claimed end to end.

## Directive

Do not implement phase-2 binary retrieval wiring until this proof lane moves
from negative proof to a confirmed compatible tuple.
