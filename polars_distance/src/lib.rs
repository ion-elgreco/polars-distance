mod array;
mod expressions;
mod list;
mod other_dist;
mod string;
use pyo3::types::PyModule;
use pyo3::{pymodule, Bound, PyResult, Python};
use pyo3_polars::PolarsAllocator;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
