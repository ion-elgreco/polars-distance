use polars::prelude::arity::try_binary_elementwise;
use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray};

fn collect_into_vecf64(arr: Box<dyn Array>) -> Vec<f64> {
    arr.as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .unwrap()
        .values_iter()
        .map(|v| *v)
        .collect::<Vec<_>>()
}

pub fn distance_calc_float_inp(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
    f: fn(&[f64], &[f64]) -> f64,
) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_float(),
        ComputeError: "inner data types must be float"
    );

    try_binary_elementwise(a, b, |a: Option<Box<dyn Array>>, b| match (a, b) {
        (Some(a), Some(b)) => {
            if a.null_count() > 0 || b.null_count() > 0 {
                polars_bail!(ComputeError: "array cannot contain nulls")
            } else {
                let a = &collect_into_vecf64(a);
                let b = &collect_into_vecf64(b);
                Ok(Some(f(a, b)))
            }
        }
        _ => Ok(None),
    })
}

pub fn euclidean_dist(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_float(),
        ComputeError: "inner data types must be float"
    );

    try_binary_elementwise(a, b, |a: Option<Box<dyn Array>>, b| match (a, b) {
        (Some(a), Some(b)) => {
            if a.null_count() > 0 || b.null_count() > 0 {
                polars_bail!(ComputeError: "array cannot contain nulls")
            } else {
                let a = a
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .unwrap()
                    .values_iter();
                let b = b
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .unwrap()
                    .values_iter();
                Ok(Some(
                    a.zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt(),
                ))
            }
        }
        _ => Ok(None),
    })
}

pub fn cosine_dist(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_float(),
        ComputeError: "inner data types must be float"
    );

    try_binary_elementwise(a, b, |a: Option<Box<dyn Array>>, b| match (a, b) {
        (Some(a), Some(b)) => {
            if a.null_count() > 0 || b.null_count() > 0 {
                polars_bail!(ComputeError: "array cannot contain nulls")
            } else {
                let a = a
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .unwrap()
                    .values_iter();
                let b = b
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .unwrap()
                    .values_iter();

                let dot_prod: f64 = a.clone().zip(b.clone()).map(|(x, y)| x * y).sum();
                let mag1: f64 = a.map(|x| x.powi(2)).sum::<f64>().sqrt();
                let mag2: f64 = b.map(|y| y.powi(2)).sum::<f64>().sqrt();

                let res = if mag1 == 0.0 || mag2 == 0.0 {
                    0.0
                } else {
                    1.0 - (dot_prod / (mag1 * mag2))
                };
                Ok(Some(res))
            }
        }
        _ => Ok(None),
    })
}
