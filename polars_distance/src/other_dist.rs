use arity::binary_elementwise;
use polars::export::num::Float;
use polars::prelude::*;
use polars_arrow::array::new_null_array;

fn haversine_elementwise<T: Float>(x_lat: T, x_long: T, y_lat: T, y_long: T, radius: f64) -> T {
    let radius = T::from(radius).unwrap();
    let two = T::from(2.0).unwrap();
    let one = T::one();

    let d_lat = (y_lat - x_lat).to_radians();
    let d_lon = (y_long - x_long).to_radians();
    let lat1 = (x_lat).to_radians();
    let lat2 = (y_lat).to_radians();

    let a = ((d_lat / two).sin()) * ((d_lat / two).sin())
        + ((d_lon / two).sin()) * ((d_lon / two).sin()) * (lat1.cos()) * (lat2.cos());
    let c = two * ((a.sqrt()).atan2((one - a).sqrt()));
    radius * c
}

pub fn haversine_dist<T>(
    x_lat: &ChunkedArray<T>,
    x_long: &ChunkedArray<T>,
    y_lat: &ChunkedArray<T>,
    y_long: &ChunkedArray<T>,
    unit: String,
    arrow_dtype: ArrowDataType,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let radius = match unit.to_ascii_lowercase().as_str() {
        "km" => 6371.0,
        "miles" => 3960.0,
        _ => {
            polars_bail!(InvalidOperation: "Incorrect unit passed to haversine distance. Only 'km' or 'miles' are supported.")
        }
    };

    let (x_lat, x_long, y_lat, y_long) = if x_lat.len() < y_lat.len() {
        (y_lat, y_long, x_lat, x_long)
    } else {
        (x_lat, x_long, y_lat, y_long)
    };

    let out: ChunkedArray<T> = match y_lat.len() {
        1 => match unsafe { (y_lat.get_unchecked(0), y_long.get_unchecked(0)) } {
            (Some(y_lat_value), Some(y_long_value)) => {
                binary_elementwise(x_lat, x_long, |a, b| match (a, b) {
                    (Some(x_lat), Some(x_long)) => Some(haversine_elementwise(
                        x_lat,
                        x_long,
                        y_lat_value,
                        y_long_value,
                        radius,
                    )),
                    _ => None,
                })
            }
            (_, _) => unsafe {
                ChunkedArray::from_chunks(
                    x_lat.name().clone(),
                    vec![new_null_array(arrow_dtype, x_lat.len())],
                )
            },
        },
        _ => x_lat
            .into_iter()
            .zip(x_long.into_iter())
            .zip(y_lat.into_iter())
            .zip(y_long.into_iter())
            .map(|(((x_lat, x_long), y_lat), y_long)| {
                let x_lat = x_lat?;
                let x_long = x_long?;
                let y_lat = y_lat?;
                let y_long = y_long?;
                Some(haversine_elementwise(x_lat, x_long, y_lat, y_long, radius))
            })
            .collect(),
    };
    Ok(out.with_name("haversine".into()))
}
