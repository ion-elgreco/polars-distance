use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use distances::strings::{hamming, levenshtein};
use distances::simd::euclidean_f64;
// use distances::vectors::{euclidean};

use pyo3_polars::export::polars_core::with_match_physical_float_polars_type;


fn hamming_distance_string(x: &str, y: &str) -> u32 {
    hamming(x, y)
}

fn levenshtein_distance_string(x: &str, y: &str) -> u32 {
    levenshtein(x, y)
}

fn euclidean_vector_simd(a: &ListChunked, b: &ListChunked) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_float(),
        ComputeError: "inner data types must be float"
    );

    Ok(with_match_physical_float_polars_type!(a.inner_dtype(), |$T| {
    polars::prelude::arity::binary_elementwise(a, b, |a, b| {
        match (a, b) {
            (Some(a), Some(b)) => {
                let a = a.as_any().downcast_ref::<Vec<f64>>().unwrap();
                let b = b.as_any().downcast_ref::<Vec<f64>>().unwrap();
                Some(euclidean_f64(a, b))
            },
            _ => None
        }
    })
    }))
}

#[polars_expr(output_type=UInt32)]
fn hamming_string(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::Utf8 || inputs[1].dtype() != &DataType::Utf8 {
        polars_bail!(InvalidOperation: "String hamming distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].utf8()?;
    let y = inputs[1].utf8()?;

    let out: UInt32Chunked =
        arity::binary_elementwise_values(x, y, hamming_distance_string);
    Ok(out.into_series())
}


#[polars_expr(output_type=UInt32)]
fn levenshtein_string(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::Utf8 || inputs[1].dtype() != &DataType::Utf8 {
        polars_bail!(InvalidOperation: "");
    }
    let x = inputs[0].utf8()?;
    let y = inputs[1].utf8()?;

    let out: UInt32Chunked =
        arity::binary_elementwise_values(x, y, levenshtein_distance_string);
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn euclidean_accelerated(inputs: &[Series]) -> PolarsResult<Series> {
    // if inputs[0].dtype() != &DataType::List || inputs[1].dtype() != &DataType::List {
    //     polars_bail!(InvalidOperation: "");
    // }
    let x = inputs[0].list()?;
    let y = inputs[1].list()?;
    
    euclidean_vector_simd(x, y).map(|ca| ca.into_series())
}