import polars as pl
import polars_distance as pld
import pytest
from polars.testing import assert_frame_equal


@pytest.fixture()
def data():
    return pl.DataFrame(
        {
            "arr": [[1.0, 2.0, 3.0, 4.0]],
            "arr2": [[10.0, 8.0, 5.0, 3.0]],
            "arr3": [[1.0, 2.0, 3.0, 4.0]],
            "arr4": [[10.0, 8.0, 5.0, 3.0]],
            "str_l": ["hello world"],
            "str_r": ["hela wrld"],
        },
        schema={
            "arr": pl.Array(inner=pl.Float64, shape=4),
            "arr2": pl.Array(inner=pl.Float64, shape=4),
            "arr3": pl.Array(inner=pl.Float32, shape=4),
            "arr4": pl.Array(inner=pl.Float32, shape=4),
            "str_l": pl.Utf8,
            "str_r": pl.Utf8,
        },
    )


@pytest.fixture()
def data_sets():
    return pl.DataFrame(
        {
            "x_str": [["1"]],
            "y_str": [["1", "2", "3"]],
            "x_int": [[1]],
            "y_int": [[1, 2, 3]],
        },
    )


def test_cosine(data):
    result = data.select(
        pld.col("arr").dist_arr.cosine("arr2").alias("dist_cosine"),
        pld.col("arr3").dist_arr.cosine("arr4").alias("dist_cosine_f32"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_cosine", [0.31232593265732134], dtype=pl.Float64),
            pl.Series("dist_cosine_f32", [0.31232593265732134], dtype=pl.Float32),
        ]
    )

    assert_frame_equal(result, expected)


def test_chebyshev(data):
    result = data.select(
        pld.col("arr").dist_arr.chebyshev("arr2").alias("dist_chebyshev"),
        pld.col("arr3").dist_arr.chebyshev("arr4").alias("dist_chebyshev_f32"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_chebyshev", [9.0], dtype=pl.Float64),
            pl.Series("dist_chebyshev_f32", [9.0], dtype=pl.Float32),
        ]
    )

    assert_frame_equal(result, expected)


def test_canberra(data):
    result = data.select(
        pld.col("arr").dist_arr.canberra("arr2").alias("dist_canberra"),
        pld.col("arr3").dist_arr.canberra("arr4").alias("dist_canberra_f32"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_canberra", [1.811038961038961], dtype=pl.Float64),
            pl.Series("dist_canberra_f32", [1.811038961038961], dtype=pl.Float32),
        ]
    )

    assert_frame_equal(result, expected)


def test_bray_curtis(data):
    result = data.select(
        pld.col("arr")
        .cast(pl.Array(pl.UInt64, 4))
        .dist_arr.bray_curtis(pl.col("arr2").cast(pl.Array(pl.UInt64, 4)))
        .alias("dist_bray"),
        pld.col("arr3")
        .cast(pl.Array(pl.UInt64, 4))
        .dist_arr.bray_curtis(pl.col("arr4").cast(pl.Array(pl.UInt64, 4)))
        .alias("dist_bray_f32"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_bray", [0.5], dtype=pl.Float64),
            pl.Series("dist_bray_f32", [0.5], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result, expected)


def test_manhatten(data):
    result = data.select(
        pld.col("arr").dist_arr.manhatten("arr2").alias("dist_manhatten"),
        pld.col("arr3").dist_arr.manhatten("arr4").alias("dist_manhatten_f32"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_manhatten", [18.0], dtype=pl.Float64),
            pl.Series("dist_manhatten_f32", [18.0], dtype=pl.Float32),
        ]
    )

    assert_frame_equal(result, expected)


def test_euclidean(data):
    result = data.select(
        pld.col("arr").dist_arr.euclidean("arr2").alias("dist_euclidean"),
        pld.col("arr3").dist_arr.euclidean("arr4").alias("dist_euclidean_f32"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_euclidean", [11.045361017187261], dtype=pl.Float64),
            pl.Series("dist_euclidean_f32", [11.045361017187261], dtype=pl.Float32),
        ]
    )

    assert_frame_equal(result, expected)


def test_float32_dtype_mixing(data):
    """Mixing types should follow Polars casting rules (higher precision wins)."""
    result_mixed = data.select(
        pl.col("arr2").dist_arr.euclidean("arr3").alias("dist_mixed"),
        pl.col("arr3").dist_arr.euclidean("arr2").alias("dist_mixed_rev"),
    )

    assert result_mixed["dist_mixed"].dtype == pl.Float64
    assert result_mixed["dist_mixed_rev"].dtype == pl.Float64


def test_hamming_str(data):
    result = data.select(
        pld.col("str_l").dist_str.hamming("str_r").alias("dist_hamming"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_hamming", [8], dtype=pl.UInt32),
        ]
    )

    assert_frame_equal(result, expected)


def test_levenshtein(data):
    result = data.select(
        pld.col("str_l").dist_str.levenshtein("str_r").alias("dist_levenshtein")
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_levenshtein", [3], dtype=pl.UInt32),
        ]
    )

    assert_frame_equal(result, expected)


def test_jaccard_index(data_sets):
    result = data_sets.select(
        pld.col("x_str").dist_list.jaccard_index("y_str").alias("jaccard_index")
    )

    result_int = data_sets.select(
        pld.col("x_int").dist_list.jaccard_index("y_int").alias("jaccard_index")
    )

    expected = pl.DataFrame(
        [
            pl.Series("jaccard_index", [0.3333333333333333], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result, expected)
    assert_frame_equal(result_int, expected)


def test_sorensen_index(data_sets):
    result = data_sets.select(
        pld.col("x_str").dist_list.sorensen_index("y_str").alias("sorensen_index")
    )

    result_int = data_sets.select(
        pld.col("x_int").dist_list.sorensen_index("y_int").alias("sorensen_index")
    )

    expected = pl.DataFrame(
        [
            pl.Series("sorensen_index", [0.5], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result, expected)
    assert_frame_equal(result_int, expected)


def test_overlap_coef(data_sets):
    result = data_sets.select(
        pld.col("x_str").dist_list.overlap_coef("y_str").alias("overlap")
    )

    result_int = data_sets.select(
        pld.col("x_int").dist_list.overlap_coef("y_int").alias("overlap")
    )

    expected = pl.DataFrame(
        [
            pl.Series("overlap", [1.0], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result, expected)
    assert_frame_equal(result_int, expected)


def test_cosine_set_distance(data_sets):
    result = data_sets.select(
        pld.col("x_str").dist_list.cosine("y_str").alias("cosine_set")
    )

    result_int = data_sets.select(
        pld.col("x_int").dist_list.cosine("y_int").alias("cosine_set")
    )

    expected = pl.DataFrame(
        [
            pl.Series("cosine_set", [1.7320508075688772], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result, expected)
    assert_frame_equal(result_int, expected)


def test_tversky_set_distance(data_sets):
    result = data_sets.select(
        pld.col("x_str")
        .dist_list.tversky_index("y_str", alpha=1, beta=1)
        .alias("tversky")
    )

    result_int = data_sets.select(
        pld.col("x_int")
        .dist_list.tversky_index("y_int", alpha=1, beta=1)
        .alias("tversky")
    )

    expected = pl.DataFrame(
        [
            pl.Series("tversky", [0.3333333333333333], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result, expected)
    assert_frame_equal(result_int, expected)


@pytest.mark.parametrize(
    "unit,value", [("km", 0.5491557912038084), ("miles", 0.341336828310639)]
)
def test_haversine(unit, value):
    df = pl.DataFrame(
        {
            "x": [{"latitude": 38.898556, "longitude": -77.037852}],
            "y": [{"latitude": 38.897147, "longitude": -77.043934}],
        }
    )
    print(unit)
    result = df.select(pld.col("x").dist.haversine("y", unit=unit).alias("haversine"))
    expected = pl.DataFrame(
        [
            pl.Series("haversine", [value], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result, expected)


def test_gestalt(data):
    result = data.select(
        pld.col("str_l").dist_str.gestalt_ratio(pld.col("str_r")).alias("dist_gestalt")
    )

    expected = pl.DataFrame(
        [
            pl.Series("dist_gestalt", [0.8], dtype=pl.Float64),
        ]
    )

    assert_frame_equal(result, expected)


def test_haversine_null():
    df = pl.DataFrame(
        {
            "x": [
                {"latitude": 38.898556, "longitude": -77.037852},
                {"latitude": 38.898556, "longitude": -77.037852},
            ],
        }
    )
    result = df.select(
        pld.col("x")
        .dist.haversine(
            pl.lit(
                {"latitude": None, "longitude": None},
                dtype=pl.Struct({"latitude": pl.Float64, "longitude": pl.Float64}),
            ),
            unit="km",
        )
        .alias("haversine")
    )

    expected = pl.DataFrame(
        [
            pl.Series("haversine", [None, None], dtype=pl.Float64),
        ]
    )
    assert_frame_equal(result, expected)


def test_tversky_null():
    df_test = pl.DataFrame(
        {
            "a1": [[1, 2], [1, 2, 3]],
        }
    )
    result = df_test.select(
        d=pld.col("a1").dist_list.tversky_index(
            pl.lit(None, dtype=pl.List(pl.Int64)), 0, 1
        )
    )
    expected = pl.DataFrame(
        [
            pl.Series("d", [None, None], dtype=pl.Float64),
        ]
    )
    assert_frame_equal(result, expected)


def test_broadcast():
    df = pl.DataFrame(
        {
            "x": [
                {"latitude": 38.898556, "longitude": -77.037852},
                {"latitude": 12.1, "longitude": -55.1},
            ],
        }
    )
    result = df.select(
        pld.col("x")
        .dist.haversine(
            pl.lit(
                {"latitude": 52.3, "longitude": 12.45},
                dtype=pl.Struct({"latitude": pl.Float64, "longitude": pl.Float64}),
            ),
            unit="km",
        )
        .alias("haversine")
    )
    expected = pl.DataFrame(
        [
            pl.Series(
                "haversine", [6663.617141243698, 7426.0664022744795], dtype=pl.Float64
            ),
        ]
    )
    assert_frame_equal(result, expected)

    df = pl.DataFrame(
        {
            "a1": ["test1", "hello", "test1", "hello", "test1", "hello"],
        }
    )
    result = df.select(d=pld.col("a1").dist_str.levenshtein(pl.lit("testaa")))
    expected = pl.DataFrame(
        [
            pl.Series("d", [2, 5, 2, 5, 2, 5], dtype=pl.UInt32),
        ]
    )
    assert_frame_equal(result, expected)
