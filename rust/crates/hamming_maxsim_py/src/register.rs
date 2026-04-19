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
    // TODO:
    // 1. downcast the python object to the version-specific SessionContext wrapper
    // 2. use datafusion-ffi / exposed Rust handle access to get &mut SessionContext
    // 3. call `ctx.register_udf(hamming_maxsim_udf::build_udf())`
    //
    // The UDF itself is fully implemented in `hamming_maxsim_udf`; this function is the
    // only intentionally version-sensitive seam left as a stub.
    Ok(())
}
