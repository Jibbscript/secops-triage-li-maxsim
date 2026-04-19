#[path = "register.rs"]
mod register_impl;

use pyo3::prelude::*;

#[pyfunction]
fn maxsim(query: Vec<u64>, doc: Vec<u64>) -> PyResult<f32> {
    let kernel = hamming_maxsim_kernel::dispatch::pick();
    let score = hamming_maxsim_kernel::maxsim_hamming(&query, &doc, kernel);
    Ok(score as f32)
}

#[pyfunction]
fn register(py: Python<'_>, ctx: PyObject) -> PyResult<()> {
    register_impl::register(py, ctx)
}

#[pymodule]
fn hamming_maxsim_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(maxsim, m)?)?;
    m.add_function(wrap_pyfunction!(register, m)?)?;
    Ok(())
}
