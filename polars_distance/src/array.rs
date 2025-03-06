use distances::vectors::minkowski;
use polars::prelude::arity::{try_binary_elementwise, try_unary_elementwise};
use polars::prelude::*;
use polars_arrow::array::{new_null_array, Array, PrimitiveArray};
use polars::export::num::Float;


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

pub fn euclidean_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_numeric(),
        ComputeError: "inner data types must be numeric"
    );

    // Cast to the target float type
    let s1 = a.cast(&DataType::Array(Box::new(T::get_dtype()), a.width()))?;
    let s2 = b.cast(&DataType::Array(Box::new(T::get_dtype()), a.width()))?;

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
                            .downcast_ref::<PrimitiveArray<T::Native>>()
                            .unwrap()
                            .values_iter();
                        let b = b_value
                            .as_any()
                            .downcast_ref::<PrimitiveArray<T::Native>>()
                            .unwrap()
                            .values_iter();
                        Ok(Some(
                            a.zip(b).map(|(x, y)| (*x - *y).powi(2)).sum::<T::Native>().sqrt(),
                        ))
                    }
                    _ => Ok(None),
                })
            }
            None => unsafe {
                // Use T's data type to create a null array of appropriate type
                let arrow_data_type = match T::get_dtype() {
                    DataType::Float32 => ArrowDataType::Float32,
                    DataType::Float64 => ArrowDataType::Float64,
                    _ => unreachable!("T must be Float32Type or Float64Type"),
                };

                Ok(ChunkedArray::from_chunks(
                    a.name().clone(),
                    vec![new_null_array(arrow_data_type, a.len())],
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
                        .downcast_ref::<PrimitiveArray<T::Native>>()
                        .unwrap()
                        .values_iter();
                    let b = b
                        .as_any()
                        .downcast_ref::<PrimitiveArray<T::Native>>()
                        .unwrap()
                        .values_iter();
                    Ok(Some(
                        a.zip(b).map(|(x, y)| (*x - *y).powi(2)).sum::<T::Native>().sqrt(),
                    ))
                }
            }
            _ => Ok(None),
        }),
    }
}

pub fn cosine_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_numeric(),
        ComputeError: "inner data types must be numeric"
    );

    // Cast to the target float type
    let s1 = a.cast(&DataType::Array(Box::new(T::get_dtype()), a.width()))?;
    let s2 = b.cast(&DataType::Array(Box::new(T::get_dtype()), a.width()))?;

    let a: &ArrayChunked = s1.array()?;
    let b: &ArrayChunked = s2.array()?;

    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    match b.len() {
        1 => match unsafe { b.get_unchecked(0) } {
            Some(b_value) => {
                if b_value.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                }
                arity::try_unary_elementwise(a, |a| match a {
                    Some(a) => {
                        if a.null_count() > 0 {
                            polars_bail!(ComputeError: "array cannot contain nulls")
                        }
                        let a = a
                            .as_any()
                            .downcast_ref::<PrimitiveArray<T::Native>>()
                            .unwrap()
                            .values_iter();
                        let b = b_value
                            .as_any()
                            .downcast_ref::<PrimitiveArray<T::Native>>()
                            .unwrap()
                            .values_iter();

                        let dot_prod = a.clone().zip(b.clone()).map(|(x, y)| *x * *y).sum();
                        let mag1 = a.map(|x| x.powi(2)).sum::<T::Native>().sqrt();
                        let mag2 = b.map(|y| y.powi(2)).sum::<T::Native>().sqrt();

                        let zero = T::Native::zero();
                        let one = T::Native::one();
                        
                        let res = if mag1 == zero || mag2 == zero {
                            zero
                        } else {
                            one - (dot_prod / (mag1 * mag2))
                        };
                        Ok(Some(res))
                    }
                    _ => Ok(None),
                })
            }
            None => unsafe {
                // Use T's data type to create a null array of appropriate type
                let arrow_data_type = match T::get_dtype() {
                    DataType::Float32 => ArrowDataType::Float32,
                    DataType::Float64 => ArrowDataType::Float64,
                    _ => unreachable!("T must be Float32Type or Float64Type"),
                };

                Ok(ChunkedArray::from_chunks(
                    a.name().clone(),
                    vec![new_null_array(arrow_data_type, a.len())],
                ))
            },
        },
        _ => arity::try_binary_elementwise(a, b, |a, b| match (a, b) {
            (Some(a), Some(b)) => {
                if a.null_count() > 0 || b.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                } else {
                    let a = a
                        .as_any()
                        .downcast_ref::<PrimitiveArray<T::Native>>()
                        .unwrap()
                        .values_iter();
                    let b = b
                        .as_any()
                        .downcast_ref::<PrimitiveArray<T::Native>>()
                        .unwrap()
                        .values_iter();

                    let dot_prod = a.clone().zip(b.clone()).map(|(x, y)| *x * *y).sum();
                    let mag1 = a.map(|x| x.powi(2)).sum::<T::Native>().sqrt();
                    let mag2 = b.map(|y| y.powi(2)).sum::<T::Native>().sqrt();

                    let zero = T::Native::zero();
                    let one = T::Native::one();
                    
                    let res = if mag1 == zero || mag2 == zero {
                        zero
                    } else {
                        one - (dot_prod / (mag1 * mag2))
                    };
                    Ok(Some(res))
                }
            }
            _ => Ok(None),
        }),
    }
}

pub fn minkowski_dist_generic<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
    p: i32,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    let p_float = T::Native::from(p).unwrap();
    let inv_p = T::Native::one() / p_float;
    
    vector_distance_calc::<T, _>(a, b, move |a, b| {
        let sum: T::Native = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs().powf(p_float))
            .sum();
        sum.powf(inv_p)
    })
}

pub fn vector_distance_calc<T, F>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
    distance_fn: F,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
    F: Fn(&[T::Native], &[T::Native]) -> T::Native,
{
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_numeric(),
        ComputeError: "inner data types must be numeric"
    );

    // Cast to the target float type
    let s1 = a.cast(&DataType::Array(Box::new(T::get_dtype()), a.width()))?;
    let s2 = b.cast(&DataType::Array(Box::new(T::get_dtype()), a.width()))?;

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
                arity::try_unary_elementwise(a, |a| match a {
                    Some(a) => {
                        if a.null_count() > 0 {
                            polars_bail!(ComputeError: "array cannot contain nulls")
                        }
                        let a = a
                            .as_any()
                            .downcast_ref::<PrimitiveArray<T::Native>>()
                            .unwrap()
                            .values
                            .to_vec();
                        let b = b_value
                            .as_any()
                            .downcast_ref::<PrimitiveArray<T::Native>>()
                            .unwrap()
                            .values
                            .to_vec();
                        Ok(Some(distance_fn(&a, &b)))
                    }
                    _ => Ok(None),
                })
            }
            None => unsafe {
                // Use T's data type to create a null array of appropriate type
                let arrow_data_type = match T::get_dtype() {
                    DataType::Float32 => ArrowDataType::Float32,
                    DataType::Float64 => ArrowDataType::Float64,
                    _ => unreachable!("T must be Float32Type or Float64Type"),
                };

                Ok(ChunkedArray::from_chunks(
                    a.name().clone(),
                    vec![new_null_array(arrow_data_type, a.len())],
                ))
            },
        },
        _ => arity::try_binary_elementwise(a, b, |a, b| match (a, b) {
            (Some(a), Some(b)) => {
                if a.null_count() > 0 || b.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                } else {
                    let a = a
                        .as_any()
                        .downcast_ref::<PrimitiveArray<T::Native>>()
                        .unwrap()
                        .values
                        .to_vec();
                    let b = b
                        .as_any()
                        .downcast_ref::<PrimitiveArray<T::Native>>()
                        .unwrap()
                        .values
                        .to_vec();
                    Ok(Some(distance_fn(&a, &b)))
                }
            }
            _ => Ok(None),
        }),
    }
}

pub fn chebyshev_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    vector_distance_calc::<T, _>(a, b, |a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs())
            .fold(T::Native::zero(), |max, val| if val > max { val } else { max })
    })
}

pub fn canberra_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    vector_distance_calc::<T, _>(a, b, |a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let abs_diff = (*x - *y).abs();
                let abs_sum = x.abs() + y.abs();
                if abs_sum > T::Native::zero() {
                    abs_diff / abs_sum
                } else {
                    T::Native::zero()
                }
            })
            .sum()
    })
}

pub fn manhattan_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    vector_distance_calc::<T, _>(a, b, |a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs())
            .sum()
    })
}

pub fn bray_curtis_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    vector_distance_calc::<T, _>(a, b, |a, b| {
        let sum_abs_diff: T::Native = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs())
            .sum();
        let sum_abs_sum: T::Native = a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.abs() + y.abs())
            .sum();
        
        if sum_abs_sum > T::Native::zero() {
            sum_abs_diff / sum_abs_sum
        } else {
            T::Native::zero()
        }
    })
}

pub fn l3_norm_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    vector_distance_calc::<T, _>(a, b, |a, b| {
        let sum: T::Native = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs().powi(3))
            .sum();
        sum.powf(T::Native::one() / T::Native::from(3.0).unwrap())
    })
}

pub fn l4_norm_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native>,
{
    vector_distance_calc::<T, _>(a, b, |a, b| {
        let sum: T::Native = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs().powi(4))
            .sum();
        sum.powf(T::Native::one() / T::Native::from(4.0).unwrap())
    })
}
