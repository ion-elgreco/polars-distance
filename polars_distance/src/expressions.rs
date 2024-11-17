use crate::array::{
    cosine_dist, distance_calc_numeric_inp, distance_calc_uint_inp, euclidean_dist, minkowski_dist,
};
use crate::list::{
    cosine_set_distance, jaccard_index, overlap_coef, sorensen_index, tversky_index,
};
use crate::other_dist::haversine_dist;
use crate::string::{
    dam_levenshtein_dist, dam_levenshtein_normalized_dist, gestalt_ratio, hamming_dist,
    hamming_normalized_dist, indel_dist, indel_normalized_dist, jaro_dist, jaro_winkler_dist,
    lcs_seq_dist, lcs_seq_normalized_dist, levenshtein_dist, levenshtein_normalized_dist, osa_dist,
    osa_normalized_dist, postfix_dist, postfix_normalized_dist, prefix_dist,
    prefix_normalized_dist,
};
use distances::vectors::{bray_curtis, canberra, chebyshev, l3_norm, l4_norm, manhattan};
use polars::prelude::*;
use polars_arrow::array::new_null_array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct TverskyIndexKwargs {
    alpha: f64,
    beta: f64,
}

#[derive(Deserialize)]
struct MinkowskiKwargs {
    p: i32,
}

#[derive(Deserialize)]
struct HaversineKwargs {
    unit: String,
}

fn elementwise_str_u32(
    x: &ChunkedArray<StringType>,
    y: &ChunkedArray<StringType>,
    f: fn(&str, &str) -> u32,
) -> UInt32Chunked {
    let (x, y) = if x.len() < y.len() { (y, x) } else { (x, y) };
    match y.len() {
        1 => match unsafe { y.get_unchecked(0) } {
            Some(y_value) => arity::unary_elementwise(x, |x| x.map(|x| f(x, y_value))),
            None => unsafe {
                ChunkedArray::from_chunks(
                    x.name().clone(),
                    vec![new_null_array(ArrowDataType::UInt32, x.len())],
                )
            },
        },
        _ => arity::binary_elementwise_values(x, y, f),
    }
}

fn elementwise_str_f64(
    x: &ChunkedArray<StringType>,
    y: &ChunkedArray<StringType>,
    f: fn(&str, &str) -> f64,
) -> Float64Chunked {
    let (x, y) = if x.len() < y.len() { (y, x) } else { (x, y) };
    match y.len() {
        1 => match unsafe { y.get_unchecked(0) } {
            Some(y_value) => arity::unary_elementwise(x, |x| x.map(|x| f(x, y_value))),
            None => unsafe {
                ChunkedArray::from_chunks(
                    x.name().clone(),
                    vec![new_null_array(ArrowDataType::Float64, x.len())],
                )
            },
        },
        _ => arity::binary_elementwise_values(x, y, f),
    }
}

// STR EXPRESSIONS
#[polars_expr(output_type=UInt32)]
fn hamming_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String hamming distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_u32(x, y, hamming_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn hamming_normalized_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String hamming distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    // If one side is a literal it will be shorter but is moved to RHS so we can use unsafe access
    Ok(elementwise_str_f64(x, y, hamming_normalized_dist).into_series())
}

#[polars_expr(output_type=UInt32)]
fn levenshtein_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String levenshtein distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_u32(x, y, levenshtein_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn levenshtein_normalized_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String levenshtein distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, levenshtein_normalized_dist).into_series())
}

#[polars_expr(output_type=UInt32)]
fn damerau_levenshtein_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String damerau levenshtein distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_u32(x, y, dam_levenshtein_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn damerau_levenshtein_normalized_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String damerau levenshtein distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, dam_levenshtein_normalized_dist).into_series())
}

#[polars_expr(output_type=UInt32)]
fn indel_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String indel distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_u32(x, y, indel_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn indel_normalized_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String indel distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, indel_normalized_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn jaro_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String jaro distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, jaro_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn jaro_winkler_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String jaro winkler distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, jaro_winkler_dist).into_series())
}

#[polars_expr(output_type=UInt32)]
fn lcs_seq_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String longest common subsequence distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_u32(x, y, lcs_seq_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn lcs_seq_normalized_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String longest common subsequence distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, lcs_seq_normalized_dist).into_series())
}

#[polars_expr(output_type=UInt32)]
fn osa_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String osa distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_u32(x, y, osa_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn osa_normalized_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String osa distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, osa_normalized_dist).into_series())
}

#[polars_expr(output_type=UInt32)]
fn postfix_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String postfix distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_u32(x, y, postfix_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn postfix_normalized_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String postfix distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, postfix_normalized_dist).into_series())
}

#[polars_expr(output_type=UInt32)]
fn prefix_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String prefix distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_u32(x, y, prefix_dist).into_series())
}

#[polars_expr(output_type=Float64)]
fn prefix_normalized_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "String prefix distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, prefix_normalized_dist).into_series())
}

// ARRAY EXPRESSIONS
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
fn minkowski_arr(inputs: &[Series], kwargs: MinkowskiKwargs) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    minkowski_dist(x, y, kwargs.p).map(|ca| ca.into_series())
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
    distance_calc_numeric_inp(x, y, chebyshev).map(|ca| ca.into_series())
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
    distance_calc_numeric_inp(x, y, canberra).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn manhatten_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }

    distance_calc_numeric_inp(x, y, manhattan).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn l3_norm_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }

    distance_calc_numeric_inp(x, y, l3_norm).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn l4_norm_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }

    distance_calc_numeric_inp(x, y, l4_norm).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn bray_curtis_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }

    distance_calc_uint_inp(x, y, bray_curtis).map(|ca| ca.into_series())
}

// SET (list) EXPRESSIONS
#[polars_expr(output_type=Float64)]
fn jaccard_index_list(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<ListType> = inputs[0].list()?;
    let y: &ChunkedArray<ListType> = inputs[1].list()?;
    jaccard_index(x, y).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn sorensen_index_list(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<ListType> = inputs[0].list()?;
    let y: &ChunkedArray<ListType> = inputs[1].list()?;
    sorensen_index(x, y).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn overlap_coef_list(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<ListType> = inputs[0].list()?;
    let y: &ChunkedArray<ListType> = inputs[1].list()?;
    overlap_coef(x, y).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn cosine_list(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ChunkedArray<ListType> = inputs[0].list()?;
    let y: &ChunkedArray<ListType> = inputs[1].list()?;
    cosine_set_distance(x, y).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn tversky_index_list(inputs: &[Series], kwargs: TverskyIndexKwargs) -> PolarsResult<Series> {
    let x: &ChunkedArray<ListType> = inputs[0].list()?;
    let y: &ChunkedArray<ListType> = inputs[1].list()?;
    tversky_index(x, y, kwargs.alpha, kwargs.beta).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn haversine_struct(inputs: &[Series], kwargs: HaversineKwargs) -> PolarsResult<Series> {
    let ca_x: &StructChunked = inputs[0].struct_()?;
    let ca_y: &StructChunked = inputs[1].struct_()?;

    let x_lat = ca_x.field_by_name("latitude")?;
    let x_long = ca_x.field_by_name("longitude")?;

    let y_lat = ca_y.field_by_name("latitude")?;
    let y_long = ca_y.field_by_name("longitude")?;

    polars_ensure!(
        x_lat.dtype() == x_long.dtype() && x_lat.dtype().is_float(),
        ComputeError: "x data types should match"
    );

    polars_ensure!(
        y_lat.dtype() == y_long.dtype() && y_lat.dtype().is_float(),
        ComputeError: "y data types should match"
    );

    polars_ensure!(
        x_lat.dtype() == y_lat.dtype(),
        ComputeError: "x and y data types should match"
    );

    Ok(match *x_lat.dtype() {
        DataType::Float32 => {
            let x_lat = x_lat.f32().unwrap();
            let x_long = x_long.f32().unwrap();
            let y_lat = y_lat.f32().unwrap();
            let y_long = y_long.f32().unwrap();
            haversine_dist(
                x_lat,
                x_long,
                y_lat,
                y_long,
                kwargs.unit,
                ArrowDataType::Float32,
            )?
            .into_series()
        }
        DataType::Float64 => {
            let x_lat = x_lat.f64().unwrap();
            let x_long = x_long.f64().unwrap();
            let y_lat = y_lat.f64().unwrap();
            let y_long = y_long.f64().unwrap();
            haversine_dist(
                x_lat,
                x_long,
                y_lat,
                y_long,
                kwargs.unit,
                ArrowDataType::Float64,
            )?
            .into_series()
        }
        _ => unimplemented!(),
    })
}

#[polars_expr(output_type=Float64)]
fn gestalt_ratio_str(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != &DataType::String || inputs[1].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "Gestalt ratio distance works only on Utf8 types. Please cast to Utf8 first.");
    }
    let x = inputs[0].str()?;
    let y = inputs[1].str()?;

    Ok(elementwise_str_f64(x, y, gestalt_ratio).into_series())
}
