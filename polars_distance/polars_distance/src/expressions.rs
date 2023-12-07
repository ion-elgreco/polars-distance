use distances::strings::{hamming, levenshtein};
use polars::prelude::{*};
use polars::prelude::arity::{binary_elementwise, try_binary_elementwise};
use distances::simd::euclidean_f64;
use pyo3_polars::derive::polars_expr;

fn hamming_distance_string(x: &str, y: &str) -> u32 {
    hamming(x, y)
}

fn levenshtein_distance_string(x: &str, y: &str) -> u32 {
    levenshtein(x, y)
}


fn euclidean_vector_simd(a: &ChunkedArray<FixedSizeListType>, b: &ChunkedArray<FixedSizeListType>) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_numeric(),
        ComputeError: "inner data types must be numeric"
    );

    try_binary_elementwise(a, b, |a, b| {
        match (a, b) {
            (Some(a), Some(b)) => {
                // Ok(Some(1.0))
                dbg!(&a);
                dbg!(&b);
                let a = a.as_any().downcast_ref::<Vec<_>>();
                let b = b.as_any().downcast_ref::<Vec<_>>();
                dbg!(&a);
                dbg!(&b);
                match (a, b) {
                    (Some(a), Some(b)) => {
                        Ok(Some(euclidean_f64(a, b)))
                    }
                    _ => Ok(None)
                }
            },
            _ => Ok(None)
        }
    })
}

#[polars_expr(output_type=UInt32)]
fn hamming_string(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::Utf8 || inputs[1].dtype() != &DataType::Utf8 {
        polars_bail!(InvalidOperation: "String hamming distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].utf8()?;
    let y = inputs[1].utf8()?;

    let out: UInt32Chunked = arity::binary_elementwise_values(x, y, hamming_distance_string);
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt32)]
fn levenshtein_string(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::Utf8 || inputs[1].dtype() != &DataType::Utf8 {
        polars_bail!(InvalidOperation: "");
    }
    let x = inputs[0].utf8()?;
    let y = inputs[1].utf8()?;

    let out: UInt32Chunked = arity::binary_elementwise_values(x, y, levenshtein_distance_string);
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn euclidean_accelerated(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<FixedSizeListType> = inputs[0].array()?;
    let y: &ChunkedArray<FixedSizeListType> = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same. 
                `{}` width: {}, 
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    euclidean_vector_simd(x, y).map(|ca| ca.into_series())
}
