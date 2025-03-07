use distances::vectors::{bray_curtis, canberra, chebyshev, l3_norm, l4_norm, manhattan, minkowski};
use polars::prelude::arity::{try_binary_elementwise, try_unary_elementwise};
use polars::prelude::*;
use polars_arrow::array::{new_null_array, PrimitiveArray};
use num_traits::{Zero, One, Float, FromPrimitive};

pub fn vector_distance_calc<T, F>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
    distance_fn: F,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
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
                try_unary_elementwise(a, |a| match a {
                    Some(a) => {
                        if a.null_count() > 0 {
                            polars_bail!(ComputeError: "array cannot contain nulls")
                        }
                        let a = a
                            .as_any()
                            .downcast_ref::<PrimitiveArray<T::Native>>()
                            .unwrap()
                            .values()
                            .to_vec();
                        let b = b_value
                            .as_any()
                            .downcast_ref::<PrimitiveArray<T::Native>>()
                            .unwrap()
                            .values()
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
        _ => try_binary_elementwise(a, b, |a, b| match (a, b) {
            (Some(a), Some(b)) => {
                if a.null_count() > 0 || b.null_count() > 0 {
                    polars_bail!(ComputeError: "array cannot contain nulls")
                } else {
                    let a = a
                        .as_any()
                        .downcast_ref::<PrimitiveArray<T::Native>>()
                        .unwrap()
                        .values()
                        .to_vec();
                    let b = b
                        .as_any()
                        .downcast_ref::<PrimitiveArray<T::Native>>()
                        .unwrap()
                        .values()
                        .to_vec();
                    Ok(Some(distance_fn(&a, &b)))
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
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
{
    vector_distance_calc::<T, _>(a, b, |a_slice, b_slice| {
        a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<T::Native>()
            .sqrt()
    })
}

pub fn cosine_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
{
    vector_distance_calc::<T, _>(a, b, |a_slice, b_slice| {
        let dot_prod = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(x, y)| *x * *y)
            .sum::<T::Native>();
        let mag1 = a_slice
            .iter()
            .map(|x| x.powi(2))
            .sum::<T::Native>()
            .sqrt();
        let mag2 = b_slice
            .iter()
            .map(|y| y.powi(2))
            .sum::<T::Native>()
            .sqrt();

        if mag1.is_zero() || mag2.is_zero() {
            T::Native::zero()
        } else {
            T::Native::one() - (dot_prod / (mag1 * mag2))
        }
    })
}

pub fn minkowski_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
    p: i32,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
    <T as PolarsNumericType>::Native: distances::number::Float,
{
    let metric = minkowski(p);
    vector_distance_calc::<T, _>(a, b, metric)
}

pub fn chebyshev_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
    <T as polars::prelude::PolarsNumericType>::Native: distances::Number,
{
    vector_distance_calc::<T, _>(a, b, chebyshev)
}

pub fn canberra_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
    <T as polars::prelude::PolarsNumericType>::Native: distances::number::Float,
{
    vector_distance_calc::<T, _>(a, b, canberra)
}

pub fn manhattan_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
    <T as polars::prelude::PolarsNumericType>::Native: distances::Number,
{
    vector_distance_calc::<T, _>(a, b, manhattan)
}

pub fn bray_curtis_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
    <T as polars::prelude::PolarsNumericType>::Native: distances::number::Float,
{
    vector_distance_calc::<T, _>(a, b, bray_curtis)
}

pub fn l3_norm_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
    <T as polars::prelude::PolarsNumericType>::Native: distances::number::Float,
{
    vector_distance_calc::<T, _>(a, b, l3_norm)
}

pub fn l4_norm_dist<T>(
    a: &ChunkedArray<FixedSizeListType>,
    b: &ChunkedArray<FixedSizeListType>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float + std::ops::Sub<Output = T::Native> + FromPrimitive + Zero + One,
    <T as polars::prelude::PolarsNumericType>::Native: distances::number::Float,
{
    vector_distance_calc::<T, _>(a, b, l4_norm)
}
