use distances::vectors::minkowski;
use polars::prelude::arity::{try_binary_elementwise, try_unary_elementwise};
use polars::prelude::*;
use polars_arrow::array::{new_null_array, Array, PrimitiveArray};

fn collect_into_vecf64(arr: Box<dyn Array>) -> Vec<f64> {
    arr.as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .unwrap()
        .values_iter()
        .copied()
        .collect::<Vec<_>>()
}

fn collect_into_uint64(arr: Box<dyn Array>) -> Vec<u64> {
    arr.as_any()
        .downcast_ref::<PrimitiveArray<_>>()
        .unwrap()
        .values_iter()
        .copied()
        .collect::<Vec<_>>()
}

pub fn distance_calc_numeric_inp(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
    f: fn(&[f64], &[f64]) -> f64,
) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_numeric(),
        ComputeError: "inner data types must be numeric"
    );

    let s1 = a.cast(&DataType::Array(Box::new(DataType::Float64), a.width()))?;
    let s2 = b.cast(&DataType::Array(Box::new(DataType::Float64), a.width()))?;

    let a: &ArrayChunked = s1.array()?;
    let b: &ArrayChunked = s2.array()?;

    // If one side is a literal it will be shorter but is moved to RHS so we can use unsafe access
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    match b.len() {
        1 => match unsafe { b.get_unchecked(0) } {
            Some(b_value) => {
                if b_value.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                }
                try_unary_elementwise(a, |a| match a {
                    Some(a) => {
                        if a.null_count() > 0 {
                            polars_bail!(ComputeError: "array cannot contain nulls")
                        }
                        let a = &collect_into_vecf64(a);
                        let b = &collect_into_vecf64(b_value.clone());
                        Ok(Some(f(a, b)))
                    }
                    _ => Ok(None),
                })
            }
            None => unsafe {
                Ok(ChunkedArray::from_chunks(
                    a.name().clone(),
                    vec![new_null_array(ArrowDataType::Float64, a.len())],
                ))
            },
        },
        _ => try_binary_elementwise(a, b, |a, b| match (a, b) {
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
        }),
    }
}

pub fn distance_calc_uint_inp(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
    f: fn(&[u64], &[u64]) -> f64,
) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_unsigned_integer(),
        ComputeError: "inner data types must be unsigned integer"
    );

    let s1 = a.cast(&DataType::Array(Box::new(DataType::UInt64), a.width()))?;
    let s2 = b.cast(&DataType::Array(Box::new(DataType::UInt64), a.width()))?;

    let a: &ArrayChunked = s1.array()?;
    let b: &ArrayChunked = s2.array()?;

    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    match b.len() {
        1 => match unsafe { b.get_unchecked(0) } {
            Some(b_value) => {
                if b_value.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                }
                try_unary_elementwise(a, |a| match a {
                    Some(a) => {
                        if a.null_count() > 0 {
                            polars_bail!(ComputeError: "array cannot contain nulls")
                        }
                        let a = &collect_into_uint64(a);
                        let b = &collect_into_uint64(b_value.clone());
                        Ok(Some(f(a, b)))
                    }
                    _ => Ok(None),
                })
            }
            None => unsafe {
                Ok(ChunkedArray::from_chunks(
                    a.name().clone(),
                    vec![new_null_array(ArrowDataType::Float64, a.len())],
                ))
            },
        },
        _ => try_binary_elementwise(a, b, |a, b| match (a, b) {
            (Some(a), Some(b)) => {
                if a.null_count() > 0 || b.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                } else {
                    let a = &collect_into_uint64(a);
                    let b = &collect_into_uint64(b);
                    Ok(Some(f(a, b)))
                }
            }
            _ => Ok(None),
        }),
    }
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
        a.inner_dtype().is_numeric(),
        ComputeError: "inner data types must be numeric"
    );

    let s1 = a.cast(&DataType::Array(Box::new(DataType::Float64), a.width()))?;
    let s2 = b.cast(&DataType::Array(Box::new(DataType::Float64), a.width()))?;

    let a: &ArrayChunked = s1.array()?;
    let b: &ArrayChunked = s2.array()?;

    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    match b.len() {
        1 => match unsafe { b.get_unchecked(0) } {
            Some(b_value) => {
                if b_value.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                }
                try_unary_elementwise(a, |a| match a {
                    Some(a) => {
                        if a.null_count() > 0 {
                            polars_bail!(ComputeError: "array cannot contain nulls")
                        }
                        let a = a
                            .as_any()
                            .downcast_ref::<PrimitiveArray<f64>>()
                            .unwrap()
                            .values_iter();
                        let b = b_value
                            .as_any()
                            .downcast_ref::<PrimitiveArray<f64>>()
                            .unwrap()
                            .values_iter();
                        Ok(Some(
                            a.zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt(),
                        ))
                    }
                    _ => Ok(None),
                })
            }
            None => unsafe {
                Ok(ChunkedArray::from_chunks(
                    a.name().clone(),
                    vec![new_null_array(ArrowDataType::Float64, a.len())],
                ))
            },
        },
        _ => try_binary_elementwise(a, b, |a, b| match (a, b) {
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
        }),
    }
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
        a.inner_dtype().is_numeric(),
        ComputeError: "inner data types must be numeric"
    );

    let s1 = a.cast(&DataType::Array(Box::new(DataType::Float64), a.width()))?;
    let s2 = b.cast(&DataType::Array(Box::new(DataType::Float64), a.width()))?;

    let a: &ArrayChunked = s1.array()?;
    let b: &ArrayChunked = s2.array()?;

    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    match b.len() {
        1 => match unsafe { b.get_unchecked(0) } {
            Some(b_value) => {
                if b_value.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                }
                try_unary_elementwise(a, |a| match a {
                    Some(a) => {
                        if a.null_count() > 0 {
                            polars_bail!(ComputeError: "array cannot contain nulls")
                        }
                        let a = a
                            .as_any()
                            .downcast_ref::<PrimitiveArray<f64>>()
                            .unwrap()
                            .values_iter();
                        let b = b_value
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
                    _ => Ok(None),
                })
            }
            None => unsafe {
                Ok(ChunkedArray::from_chunks(
                    a.name().clone(),
                    vec![new_null_array(ArrowDataType::Float64, a.len())],
                ))
            },
        },
        _ => try_binary_elementwise(a, b, |a, b| match (a, b) {
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
        }),
    }
}

pub fn minkowski_dist(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
    p: i32,
) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_numeric(),
        ComputeError: "inner data types must be numeric"
    );

    let s1 = a.cast(&DataType::Array(Box::new(DataType::Float64), a.width()))?;
    let s2 = b.cast(&DataType::Array(Box::new(DataType::Float64), a.width()))?;

    let a: &ArrayChunked = s1.array()?;
    let b: &ArrayChunked = s2.array()?;

    // If one side is a literal it will be shorter but is moved to RHS so we can use unsafe access
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    match b.len() {
        1 => match unsafe { b.get_unchecked(0) } {
            Some(b_value) => {
                if b_value.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                }
                try_unary_elementwise(a, |a| match a {
                    Some(a) => {
                        if a.null_count() > 0 {
                            polars_bail!(ComputeError: "array cannot contain nulls")
                        }
                        let a = &collect_into_vecf64(a);
                        let b = &collect_into_vecf64(b_value.clone());
                        let metric = minkowski(p);
                        Ok(Some(metric(a, b)))
                    }
                    _ => Ok(None),
                })
            }
            None => unsafe {
                Ok(ChunkedArray::from_chunks(
                    a.name().clone(),
                    vec![new_null_array(ArrowDataType::Float64, a.len())],
                ))
            },
        },
        _ => try_binary_elementwise(a, b, |a, b| match (a, b) {
            (Some(a), Some(b)) => {
                if a.null_count() > 0 || b.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                } else {
                    let a = &collect_into_vecf64(a);
                    let b = &collect_into_vecf64(b);
                    let metric = minkowski(p);
                    Ok(Some(metric(a, b)))
                }
            }
            _ => Ok(None),
        }),
    }
}
