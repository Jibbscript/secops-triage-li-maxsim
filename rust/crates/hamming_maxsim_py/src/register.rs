use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

/// Register the Rust-side DataFusion UDF into a Python-owned SessionContext.
///
/// Note:
/// this is intentionally a first-pass skeleton. The exact extraction of the underlying
/// SessionContext depends on the version alignment between:
/// - the python `datafusion` package
/// - `datafusion-ffi`
/// - this crate
///
/// Pin all three together before expecting this to compile untouched.
pub fn register(_py: Python<'_>, _ctx: PyObject) -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "SessionContext registration is not implemented in phase-0. \
        The Rust UDF contract is verified directly in the hamming_maxsim_udf crate; \
        wire this bridge only after pinning matching Python datafusion and datafusion-ffi versions.",
    ))
}
