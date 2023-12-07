use distances::strings::{hamming, levenshtein};
use distances::vectors::{canberra, chebyshev};
use polars::prelude::arity::try_binary_elementwise;
use polars::prelude::*;
use polars_arrow::array::Array;
use polars_core::utils::arrow::array::PrimitiveArray;
use pyo3_polars::derive::polars_expr;

fn hamming_distance_string(x: &str, y: &str) -> u32 {
    hamming(x, y)
}

fn levenshtein_distance_string(x: &str, y: &str) -> u32 {
    levenshtein(x, y)
}

fn collect_into_vecf64(arr: Box<dyn Array>) -> Vec<f64> {
    arr.as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .unwrap()
        .values_iter()
        .map(|v| *v)
        .collect::<Vec<_>>()
}

fn distance_calc_float_inp(
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

fn euclidean_dist(
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

fn cosine_dist(
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

#[polars_expr(output_type=UInt32)]
fn hamming_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::Utf8 || inputs[1].dtype() != &DataType::Utf8 {
        polars_bail!(InvalidOperation: "String hamming distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].utf8()?;
    let y = inputs[1].utf8()?;

    let out: UInt32Chunked = arity::binary_elementwise_values(x, y, hamming_distance_string);
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt32)]
fn levenshtein_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::Utf8 || inputs[1].dtype() != &DataType::Utf8 {
        polars_bail!(InvalidOperation: "");
    }
    let x = inputs[0].utf8()?;
    let y = inputs[1].utf8()?;

    let out: UInt32Chunked = arity::binary_elementwise_values(x, y, levenshtein_distance_string);
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn euclidean_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<FixedSizeListType> = inputs[0].array()?;
    let y: &ChunkedArray<FixedSizeListType> = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same. 
                `{}` width: {}, 
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    euclidean_dist(x, y).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn cosine_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<FixedSizeListType> = inputs[0].array()?;
    let y: &ChunkedArray<FixedSizeListType> = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same. 
                `{}` width: {}, 
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    cosine_dist(x, y).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn chebyshev_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<FixedSizeListType> = inputs[0].array()?;
    let y: &ChunkedArray<FixedSizeListType> = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    distance_calc_float_inp(x, y, chebyshev).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn canberra_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<FixedSizeListType> = inputs[0].array()?;
    let y: &ChunkedArray<FixedSizeListType> = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    distance_calc_float_inp(x, y, canberra).map(|ca| ca.into_series())
}
