use core::hash::Hash;
use polars::prelude::arity::binary_elementwise;
use polars::prelude::*;
use polars_arrow::array::{PrimitiveArray, Utf8Array};
use polars_arrow::types::NativeType;
use polars_core::with_match_physical_integer_type;

fn jacc_int_array<T: NativeType + Hash + Eq>(a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    len_intersect as f64 / (s1.len() + s2.len() - len_intersect) as f64
}

fn jacc_str_array(a: &Utf8Array<i64>, b: &Utf8Array<i64>) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    len_intersect as f64 / (s1.len() + s2.len() - len_intersect) as f64
}

fn sorensen_int_array<T: NativeType + Hash + Eq>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    (2 * len_intersect) as f64 / (s1.len() + s2.len()) as f64
}

fn sorensen_str_array(a: &Utf8Array<i64>, b: &Utf8Array<i64>) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    (2 * len_intersect) as f64 / (s1.len() + s2.len()) as f64
}

fn overlap_int_array<T: NativeType + Hash + Eq>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    len_intersect as f64 / std::cmp::min(s1.len(), s2.len()) as f64
}

fn overlap_str_array(a: &Utf8Array<i64>, b: &Utf8Array<i64>) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    len_intersect as f64 / std::cmp::min(s1.len(), s2.len()) as f64
}

pub fn jaccard_index(a: &ListChunked, b: &ListChunked) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );

    if a.inner_dtype().is_integer() {
        Ok(with_match_physical_integer_type!(a.inner_dtype(), |$T| {
            binary_elementwise(a, b, |a, b| {
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let a = a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                        let b = b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                        Some(jacc_int_array(a, b))
                    },
                    _ => None
                }
                })
            }
        ))
    } else {
        match a.inner_dtype() {
            DataType::Utf8 => {
                Ok(binary_elementwise(a, b, |a, b| {
                    match (a, b) {
                        (Some(a), Some(b)) => {
                            let a = a.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
                            let b = b.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
                            Some(jacc_str_array(a, b))
                        },
                        _ => None
                    }
                    }))
                },
            // DataType::Categorical(_) =>  {
            //     let a = a.cast(&DataType::List(Box::new(DataType::Utf8)))?;
            //     let b = b.cast(&DataType::List(Box::new(DataType::Utf8)))?;
            //     Ok(binary_elementwise(a.list()?, b.list()?, |a, b| {
            //         match (a, b) {
            //             (Some(a), Some(b)) => {
            //                 let a = a.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            //                 let b = b.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            //                 Some(5.0)
            //             },
            //             _ => None
            //         }
            //         }))
            //     },
            _ => Err(PolarsError::ComputeError(
                format!("jaccard index only works on inner dtype Utf8 or integer. Use of {} is not supported", a.inner_dtype()).into(),
            ))
        }
    }
}

pub fn sorensen_index(a: &ListChunked, b: &ListChunked) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );

    if a.inner_dtype().is_integer() {
        Ok(with_match_physical_integer_type!(a.inner_dtype(), |$T| {
            binary_elementwise(a, b, |a, b| {
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let a = a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                        let b = b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                        Some(sorensen_int_array(a, b))
                    },
                    _ => None
                }
                })
            }
        ))
    } else {
        match a.inner_dtype() {
            DataType::Utf8 => {
                Ok(binary_elementwise(a, b, |a, b| {
                    match (a, b) {
                        (Some(a), Some(b)) => {
                            let a = a.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
                            let b = b.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
                            Some(sorensen_str_array(a, b))
                        },
                        _ => None
                    }
                    }))
                },
            _ => Err(PolarsError::ComputeError(
                format!("sorensen index only works on inner dtype Utf8 or integer. Use of {} is not supported", a.inner_dtype()).into(),
            ))
        }
    }
}

pub fn overlap_coef(a: &ListChunked, b: &ListChunked) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );

    if a.inner_dtype().is_integer() {
        Ok(with_match_physical_integer_type!(a.inner_dtype(), |$T| {
            binary_elementwise(a, b, |a, b| {
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let a = a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                        let b = b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                        Some(overlap_int_array(a, b))
                    },
                    _ => None
                }
                })
            }
        ))
    } else {
        match a.inner_dtype() {
            DataType::Utf8 => {
                Ok(binary_elementwise(a, b, |a, b| {
                    match (a, b) {
                        (Some(a), Some(b)) => {
                            let a = a.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
                            let b = b.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
                            Some(overlap_str_array(a, b))
                        },
                        _ => None
                    }
                    }))
                },
            _ => Err(PolarsError::ComputeError(
                format!("overlap coefficient only works on inner dtype Utf8 or integer. Use of {} is not supported", a.inner_dtype()).into(),
            ))
        }
    }
}
