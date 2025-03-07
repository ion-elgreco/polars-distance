use crate::array::{
    bray_curtis_dist, canberra_dist, chebyshev_dist, cosine_dist, euclidean_dist, l3_norm_dist, l4_norm_dist, manhattan_dist, minkowski_dist,
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

// General helper for all distance metrics
fn infer_distance_arr_output(input_fields: &[Field], metric_name: &str) -> PolarsResult<Field> {
    // We expect two input Fields for a binary expression (the two Series).
    // The first input_fields[0] must be an Array(F32 or F64, width).
    // Similarly input_fields[1].dtype should match.

    if input_fields.len() != 2 {
        polars_bail!(ShapeMismatch: "{}_arr expects 2 inputs, got {}", metric_name, input_fields.len());
    }

    // Get the type of first input
    let first_type = match &input_fields[0].dtype {
        // If the input is an Array with an inner dtype (like Float32 or Float64)
        DataType::Array(inner, _width) => &**inner,
        dt => {
            polars_bail!(
                ComputeError:
                "{}_arr input must be an Array, got {}",
                metric_name,
                dt
            );
        }
    };
    
    // Get the type of second input
    let second_type = match &input_fields[1].dtype {
        DataType::Array(inner, _width) => &**inner,
        _ => first_type, // Default to first type if second isn't an array
    };

    // If either input is Float64, output is Float64, otherwise Float32
    match (first_type, second_type) {
        (DataType::Float32, DataType::Float32) => {
            Ok(Field::new(metric_name.into(), DataType::Float32))
        }
        (DataType::Float64, _) | (_, DataType::Float64) => {
            Ok(Field::new(metric_name.into(), DataType::Float64))
        }
        (DataType::UInt64, _) | (_, DataType::UInt64) => {
            Ok(Field::new(metric_name.into(), DataType::Float64))
        }
        (DataType::Float32, _) | (_, DataType::Float32) => {
            Ok(Field::new(metric_name.into(), DataType::Float32))
        }
        _ => {
            polars_bail!(
                ComputeError:
                "{} distance not supported for inner types",
                metric_name
            );
        }
    }
}

fn infer_euclidean_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "euclidean")
}

fn infer_cosine_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "cosine")
}

fn infer_chebyshev_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "chebyshev")
}

fn infer_canberra_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "canberra")
}

fn infer_manhatten_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "manhatten")
}

fn infer_l3_norm_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "l3_norm")
}

fn infer_l4_norm_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "l4_norm")
}

fn infer_bray_curtis_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "bray_curtis")
}

fn infer_minkowski_arr_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    infer_distance_arr_output(input_fields, "minkowski")
}

// Function to determine the output data type for distance calculations
pub fn determine_distance_output_type(
    x_dtype: &DataType,
    y_dtype: &DataType,
    distance_name: &str,
) -> PolarsResult<DataType> {
    // If any input is Float64 or UInt64, output is Float64
    if matches!(x_dtype, DataType::Float64 | DataType::UInt64) || 
       matches!(y_dtype, DataType::Float64 | DataType::UInt64) {
        return Ok(DataType::Float64);
    }
    // Both are Float32, output is Float32
    if matches!(x_dtype, DataType::Float32) && matches!(y_dtype, DataType::Float32) {
        return Ok(DataType::Float32);
    }
    polars_bail!(
        ComputeError:
        "{} distance not supported for inner dtype: {:?} and {:?}",
        distance_name,
        x_dtype,
        y_dtype
    )
}

fn compute_array_distance<F, G>(
    x: &ArrayChunked, 
    y: &ArrayChunked, 
    distance_name: &str,
    f32_impl: F,
    f64_impl: G
) -> PolarsResult<Series> 
where 
    F: FnOnce(&ArrayChunked, &ArrayChunked) -> PolarsResult<Series>,
    G: FnOnce(&ArrayChunked, &ArrayChunked) -> PolarsResult<Series>
{
    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `x` width: {},
                `y` width: {}", x.width(), y.width());
    }

    // Determine the output type
    let output_type = determine_distance_output_type(&x.inner_dtype(), &y.inner_dtype(), distance_name)?;
    
    match output_type {
        DataType::Float32 => {
            // Both must be Float32 already (from our type determination logic)
            f32_impl(x, y)
        },
        DataType::Float64 => {
            // Create variables to hold the owned data if we need to cast
            let x_cast_result = if !matches!(x.inner_dtype(), DataType::Float64) {
                Some(x.cast(&DataType::Array(Box::new(DataType::Float64), x.width()))?)
            } else {
                None
            };
            
            let y_cast_result = if !matches!(y.inner_dtype(), DataType::Float64) {
                Some(y.cast(&DataType::Array(Box::new(DataType::Float64), y.width()))?)
            } else {
                None
            };
            
            // Now get the references, from either the cast result or the original
            let x_f64 = if let Some(cast_x) = &x_cast_result {
                cast_x.array()?
            } else {
                x
            };
            
            let y_f64 = if let Some(cast_y) = &y_cast_result {
                cast_y.array()?
            } else {
                y
            };
            
            f64_impl(x_f64, y_f64)
        },
        _ => unreachable!("Output type should only be Float32 or Float64")
    }
}

// ARRAY EXPRESSIONS
#[polars_expr(output_type_func=infer_euclidean_arr_dtype)]
fn euclidean_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;


    // Use our reusable function to compute the distance
    compute_array_distance(
        x, 
        y, 
        "euclidean",
        |x_f32, y_f32| euclidean_dist::<Float32Type>(x_f32, y_f32).map(|ca| ca.into_series()),
        |x_f64, y_f64| euclidean_dist::<Float64Type>(x_f64, y_f64).map(|ca| ca.into_series())
    )
}

#[polars_expr(output_type_func=infer_cosine_arr_dtype)]
fn cosine_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;


    // Use our reusable function to compute the distance
    compute_array_distance(
        x, 
        y, 
        "cosine",
        |x_f32, y_f32| cosine_dist::<Float32Type>(x_f32, y_f32).map(|ca| ca.into_series()),
        |x_f64, y_f64| cosine_dist::<Float64Type>(x_f64, y_f64).map(|ca| ca.into_series())
    )
}

#[polars_expr(output_type_func=infer_minkowski_arr_dtype)]
fn minkowski_arr(inputs: &[Series], kwargs: MinkowskiKwargs) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    
    let p = kwargs.p;
    compute_array_distance(
        x, 
        y, 
        "minkowski",
        |x_f32, y_f32| minkowski_dist::<Float32Type>(x_f32, y_f32, p).map(|ca| ca.into_series()),
        |x_f64, y_f64| minkowski_dist::<Float64Type>(x_f64, y_f64, p).map(|ca| ca.into_series())
    )
}

#[polars_expr(output_type_func=infer_chebyshev_arr_dtype)]
fn chebyshev_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    
    compute_array_distance(
        x, 
        y, 
        "chebyshev",
        |x_f32, y_f32| chebyshev_dist::<Float32Type>(x_f32, y_f32).map(|ca| ca.into_series()),
        |x_f64, y_f64| chebyshev_dist::<Float64Type>(x_f64, y_f64).map(|ca| ca.into_series())
    )
}

#[polars_expr(output_type_func=infer_canberra_arr_dtype)]
fn canberra_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }
    
    compute_array_distance(
        x, 
        y, 
        "canberra",
        |x_f32, y_f32| canberra_dist::<Float32Type>(x_f32, y_f32).map(|ca| ca.into_series()),
        |x_f64, y_f64| canberra_dist::<Float64Type>(x_f64, y_f64).map(|ca| ca.into_series())
    )
}

#[polars_expr(output_type_func=infer_manhatten_arr_dtype)]
fn manhatten_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }

    compute_array_distance(
        x, 
        y, 
        "manhatten",
        |x_f32, y_f32| manhattan_dist::<Float32Type>(x_f32, y_f32).map(|ca| ca.into_series()),
        |x_f64, y_f64| manhattan_dist::<Float64Type>(x_f64, y_f64).map(|ca| ca.into_series())
    )
}

#[polars_expr(output_type_func=infer_l3_norm_arr_dtype)]
fn l3_norm_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }

    compute_array_distance(
        x, 
        y, 
        "l3_norm",
        |x_f32, y_f32| l3_norm_dist::<Float32Type>(x_f32, y_f32).map(|ca| ca.into_series()),
        |x_f64, y_f64| l3_norm_dist::<Float64Type>(x_f64, y_f64).map(|ca| ca.into_series())
    )
}

#[polars_expr(output_type_func=infer_l4_norm_arr_dtype)]
fn l4_norm_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }

    compute_array_distance(
        x, 
        y, 
        "l4_norm",
        |x_f32, y_f32| l4_norm_dist::<Float32Type>(x_f32, y_f32).map(|ca| ca.into_series()),
        |x_f64, y_f64| l4_norm_dist::<Float64Type>(x_f64, y_f64).map(|ca| ca.into_series())
    )
}

#[polars_expr(output_type_func=infer_bray_curtis_arr_dtype)]
fn bray_curtis_arr(inputs: &[Series]) -> PolarsResult<Series> {
    let x: &ArrayChunked = inputs[0].array()?;
    let y: &ArrayChunked = inputs[1].array()?;

    if x.width() != y.width() {
        polars_bail!(InvalidOperation:
            "The dimensions of each array are not the same.
                `{}` width: {},
                `{}` width: {}", inputs[0].name(), x.width(), inputs[1].name(), y.width());
    }

    compute_array_distance(
        x, 
        y, 
        "bray_curtis",
        |x_f32, y_f32| bray_curtis_dist::<Float32Type>(x_f32, y_f32).map(|ca| ca.into_series()),
        |x_f64, y_f64| bray_curtis_dist::<Float64Type>(x_f64, y_f64).map(|ca| ca.into_series())
    )
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
