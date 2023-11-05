use distances::strings::{hamming, levenshtein};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn hamming_distance_string(x: &str, y: &str) -> u32 {
    hamming(x, y)
}

fn levenshtein_distance_string(x: &str, y: &str) -> u32 {
    levenshtein(x, y)
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
