use pyo3::prelude::*;

/// Register the Rust-side DataFusion UDF into a Python-owned SessionContext.
pub fn register(py: Python<'_>, ctx: PyObject) -> PyResult<()> {
    let module = py.import("alert_triage.retrievers.hamming_udf_runtime")?;
    module.getattr("register_hamming_udf")?.call1((ctx,))?;
    Ok(())
}
