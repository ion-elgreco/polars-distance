use crate::array::{cosine_dist, distance_calc_float_inp, euclidean_dist};
use crate::list::jaccard_index;
use crate::string::{hamming_distance_string, levenshtein_distance_string};
use distances::vectors::{canberra, chebyshev};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

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
        polars_bail!(InvalidOperation: "String levenshtein distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].utf8()?;
    let y = inputs[1].utf8()?;

    let out: UInt32Chunked = arity::binary_elementwise_values(x, y, levenshtein_distance_string);
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn euclidean_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

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
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

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
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

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
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    distance_calc_float_inp(x, y, canberra).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn jaccard_index_list(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<ListType> = inputs[0].list()?;
    let y: &ChunkedArray<ListType> = inputs[1].list()?;
    jaccard_index(x, y).map(|ca| ca.into_series())
}
