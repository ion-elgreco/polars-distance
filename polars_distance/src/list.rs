use core::hash::Hash;
use polars::prelude::arity::binary_elementwise;
use polars::prelude::*;
use polars_arrow::array::{PrimitiveArray, Utf8ViewArray};
use polars_arrow::types::NativeType;
use polars_core::with_match_physical_integer_type;

fn jacc_int_array<T: NativeType + Hash + Eq>(a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    len_intersect as f64 / (s1.len() + s2.len() - len_intersect) as f64
}

fn jacc_str_array(a: &Utf8ViewArray, b: &Utf8ViewArray) -> f64 {
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

fn sorensen_str_array(a: &Utf8ViewArray, b: &Utf8ViewArray) -> f64 {
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

fn overlap_str_array(a: &Utf8ViewArray, b: &Utf8ViewArray) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    len_intersect as f64 / std::cmp::min(s1.len(), s2.len()) as f64
}

fn cosine_int_array<T: NativeType + Hash + Eq>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    len_intersect as f64 / (s1.len() as f64).sqrt() * (s2.len() as f64).sqrt()
}

fn cosine_str_array(a: &Utf8ViewArray, b: &Utf8ViewArray) -> f64 {
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();
    let len_intersect = s1.intersection(&s2).count();

    len_intersect as f64 / (s1.len() as f64).sqrt() * (s2.len() as f64).sqrt()
}

pub fn elementwise_int_inp<T: NativeType + Hash + Eq>(
    a: &ListChunked,
    b: &ListChunked,
    f: fn(&PrimitiveArray<T>, &PrimitiveArray<T>) -> f64,
) -> PolarsResult<Float64Chunked> {
    Ok(binary_elementwise(a, b, |a, b| match (a, b) {
        (Some(a), Some(b)) => {
            let a = a.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
            let b = b.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
            Some(f(a, b))
        }
        _ => None,
    }))
}

pub fn elementwise_string_inp(
    a: &ListChunked,
    b: &ListChunked,
    f: fn(&Utf8ViewArray, &Utf8ViewArray) -> f64,
) -> PolarsResult<Float64Chunked> {
    Ok(binary_elementwise(a, b, |a, b| match (a, b) {
        (Some(a), Some(b)) => {
            let a = a.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            let b = b.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            Some(f(a, b))
        }
        _ => None,
    }))
}

pub fn jaccard_index(a: &ListChunked, b: &ListChunked) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );

    if a.inner_dtype().is_integer() {
        with_match_physical_integer_type!(a.inner_dtype(), |$T| {elementwise_int_inp(a, b, jacc_int_array::<$T>)})
    } else {
        match a.inner_dtype() {
            DataType::String => elementwise_string_inp(a,b, jacc_str_array),
            _ => Err(PolarsError::ComputeError(
                format!("jaccard index only works on inner dtype Utf8 and integer. Use of {} is not supported", a.inner_dtype()).into(),
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
        with_match_physical_integer_type!(a.inner_dtype(), |$T| {elementwise_int_inp(a, b, sorensen_int_array::<$T>)})
    } else {
        match a.inner_dtype() {
            DataType::String => elementwise_string_inp(a,b, sorensen_str_array),
            _ => Err(PolarsError::ComputeError(
                format!("sorensen index only works on inner dtype Utf8 and integer. Use of {} is not supported", a.inner_dtype()).into(),
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
        with_match_physical_integer_type!(a.inner_dtype(), |$T| {elementwise_int_inp(a, b, overlap_int_array::<$T>)})
    } else {
        match a.inner_dtype() {
            DataType::String => elementwise_string_inp(a,b, overlap_str_array),
            _ => Err(PolarsError::ComputeError(
                format!("overlap coefficient only works on inner dtype Utf8 and integer. Use of {} is not supported", a.inner_dtype()).into(),
            ))
        }
    }
}

pub fn cosine_set_distance(a: &ListChunked, b: &ListChunked) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );

    if a.inner_dtype().is_integer() {
        with_match_physical_integer_type!(a.inner_dtype(), |$T| {elementwise_int_inp(a, b, cosine_int_array::<$T>)})
    } else {
        match a.inner_dtype() {
            DataType::String => elementwise_string_inp(a,b, cosine_str_array),
            _ => Err(PolarsError::ComputeError(
                format!("cosine set distance only works on inner dtype Utf8 and integer. Use of {} is not supported", a.inner_dtype()).into(),
            ))
        }
    }
}

pub fn tversky_index(
    a: &ListChunked,
    b: &ListChunked,
    alpha: f64,
    beta: f64,
) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );

    if a.inner_dtype().is_integer() {
        with_match_physical_integer_type!(a.inner_dtype(), |$T| {
            Ok(binary_elementwise(a, b, |a, b| match (a, b) {
                (Some(a), Some(b)) => {
                    let a = a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                    let b = b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                    let s1 = a.into_iter().collect::<PlHashSet<_>>();
                    let s2 = b.into_iter().collect::<PlHashSet<_>>();
                    let len_intersect = s1.intersection(&s2).count() as f64;
                    let len_diff1 = s1.difference(&s2).count();
                    let len_diff2 = s2.difference(&s1).count();

                    Some(len_intersect / (len_intersect +  (alpha * len_diff1 as f64) + (beta * len_diff2 as f64)))
                }
                _ => None,
            }))

        })
    } else {
        match a.inner_dtype() {
            DataType::String => {
                Ok(binary_elementwise(a, b, |a, b| match (a, b) {
                    (Some(a), Some(b)) => {
                        let a = a.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                        let b = b.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                        let s1 = a.into_iter().collect::<PlHashSet<_>>();
                        let s2 = b.into_iter().collect::<PlHashSet<_>>();
                        let len_intersect = s1.intersection(&s2).count() as f64;
                        let len_diff1 = s1.difference(&s2).count();
                        let len_diff2 = s2.difference(&s1).count();
                        Some(len_intersect / (len_intersect +  (alpha * len_diff1 as f64) + (beta * len_diff2 as f64)))
                    }
                    _ => None,
                }))
            },
            _ => Err(PolarsError::ComputeError(
                format!("tversky index distance only works on inner dtype Utf8 and integer. Use of {} is not supported", a.inner_dtype()).into(),
            ))
        }
    }
}
