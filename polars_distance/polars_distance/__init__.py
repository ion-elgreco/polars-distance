from __future__ import annotations

from typing import Iterable, Literal, Protocol, cast

import polars as pl
from pathlib import Path
from polars.plugins import register_plugin_function
from ._internal import __version__ as __version__

# Import typing - this will eventually move to polars.typing, but currently we use the internal package and fallback to the old package for older versions of Polars.
try:
    from polars._typing import IntoExpr, PolarsDataType
except ImportError:
    from polars.type_aliases import IntoExpr, PolarsDataType  # type: ignore[no-redef]


@pl.api.register_expr_namespace("dist")
class DistancePairWise:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def haversine(
        self, other: IntoExpr, unit: Literal["km", "miles"] = "km"
    ) -> pl.Expr:
        """Returns haversine distance between two structs with the keys latitude, longitude.

        Example:
            ```python
            df = pl.DataFrame(
                    {
                        "x": [{"latitude": 38.898556, "longitude": -77.037852}],
                        "y": [{"latitude": 38.897147, "longitude": -77.043934}],
                    }
                )
            df.select(pld.col('x').dist.haversine('y', 'km').alias('haversine'))

            shape: (1, 1)
            ┌───────────┐
            │ haversine │
            │ ---       │
            │ f64       │
            ╞═══════════╡
            │ 0.549156  │
            └───────────┘
            ```
        """
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            kwargs={"unit": unit},
            function_name="haversine_struct",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("dist_arr")
class DistancePairWiseArray:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def euclidean(self, other: IntoExpr) -> pl.Expr:
        """Returns euclidean distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="euclidean_arr",
            is_elementwise=True,
        )

    def cosine(self, other: IntoExpr) -> pl.Expr:
        """Returns cosine distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="cosine_arr",
            is_elementwise=True,
        )

    def chebyshev(self, other: IntoExpr) -> pl.Expr:
        """Returns chebyshev distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="chebyshev_arr",
            is_elementwise=True,
        )

    def canberra(self, other: IntoExpr) -> pl.Expr:
        """Returns canberra distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="canberra_arr",
            is_elementwise=True,
        )

    def bray_curtis(self, other: IntoExpr) -> pl.Expr:
        """Returns bray_curtis distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="bray_curtis_arr",
            is_elementwise=True,
        )

    def manhatten(self, other: IntoExpr) -> pl.Expr:
        """Returns manhatten distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="manhatten_arr",
            is_elementwise=True,
        )

    def minkowski(self, other: IntoExpr, p: int) -> pl.Expr:
        """Returns minkowski distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            kwargs={"p": p},
            function_name="minkowski_arr",
            is_elementwise=True,
        )

    def l3_norm(self, other: IntoExpr) -> pl.Expr:
        """Returns l3_norm distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="l3_norm_arr",
            is_elementwise=True,
        )

    def l4_norm(self, other: IntoExpr) -> pl.Expr:
        """Returns l4_norm distance between two vectors"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="l4_norm_arr",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("dist_str")
class DistancePairWiseString:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def hamming(self, other: IntoExpr, normalized: bool = False) -> pl.Expr:
        """Returns hamming distance between two expressions.

        The length of the shortest string is padded to the length of longest string.
        """
        if normalized:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="hamming_normalized_str",
                is_elementwise=True,
            )
        else:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="hamming_str",
                is_elementwise=True,
            )

    def levenshtein(self, other: IntoExpr, normalized: bool = False) -> pl.Expr:
        """Returns levenshtein distance between two expressions"""
        if normalized:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="levenshtein_normalized_str",
                is_elementwise=True,
            )
        else:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="levenshtein_str",
                is_elementwise=True,
            )

    def damerau_levenshtein(self, other: IntoExpr, normalized: bool = False) -> pl.Expr:
        """Returns damerau levenshtein distance between two expressions"""
        if normalized:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="damerau_levenshtein_normalized_str",
                is_elementwise=True,
            )
        else:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="damerau_levenshtein_str",
                is_elementwise=True,
            )

    def indel(self, other: IntoExpr, normalized: bool = False) -> pl.Expr:
        """Returns indel distance between two expressions"""
        if normalized:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="indel_normalized_str",
                is_elementwise=True,
            )
        else:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="indel_str",
                is_elementwise=True,
            )

    def jaro(self, other: IntoExpr) -> pl.Expr:
        """Returns jaro distance between two expressions. Which is normalized by default."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="jaro_str",
            is_elementwise=True,
        )

    def jaro_winkler(self, other: IntoExpr) -> pl.Expr:
        """Returns jaro_winkler distance between two expressions. Which is normalized by default."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="jaro_winkler_str",
            is_elementwise=True,
        )

    def lcs_seq(self, other: IntoExpr, normalized: bool = False) -> pl.Expr:
        """Returns lcs_seq distance between two expressions"""
        if normalized:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="lcs_seq_normalized_str",
                is_elementwise=True,
            )

        else:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="lcs_seq_str",
                is_elementwise=True,
            )

    def osa(self, other: IntoExpr, normalized: bool = False) -> pl.Expr:
        """Returns osa distance between two expressions"""
        if normalized:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="osa_normalized_str",
                is_elementwise=True,
            )

        else:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="osa_str",
                is_elementwise=True,
            )

    def postfix(self, other: IntoExpr, normalized: bool = False) -> pl.Expr:
        """Returns postfix distance between two expressions"""
        if normalized:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="postfix_normalized_str",
                is_elementwise=True,
            )

        else:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="postfix_str",
                is_elementwise=True,
            )

    def prefix(self, other: IntoExpr, normalized: bool = False) -> pl.Expr:
        """Returns prefix distance between two expressions"""
        if normalized:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="prefix_normalized_str",
                is_elementwise=True,
            )

        else:
            return register_plugin_function(
                plugin_path=Path(__file__).parent,
                args=[self._expr, other],
                function_name="prefix_str",
                is_elementwise=True,
            )

    def gestalt_ratio(self, other: IntoExpr) -> pl.Expr:
        """Returns gestalt ratio between two expressions"""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="gestalt_ratio_str",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("dist_list")
class DistancePairWiseList:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def jaccard_index(self, other: IntoExpr) -> pl.Expr:
        """Returns jaccard index between two lists. Each list is converted to a set."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="jaccard_index_list",
            is_elementwise=True,
        )

    def tversky_index(self, other: IntoExpr, alpha: float, beta: float) -> pl.Expr:
        """Returns tversky index between two lists. Each list is converted to a set."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            kwargs={"alpha": alpha, "beta": beta},
            function_name="tversky_index_list",
            is_elementwise=True,
        )

    def sorensen_index(self, other: IntoExpr) -> pl.Expr:
        """Returns sorensen index between two lists. Each list is converted to a set."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="sorensen_index_list",
            is_elementwise=True,
        )

    def overlap_coef(self, other: IntoExpr) -> pl.Expr:
        """Returns overlap coef between two lists. Each list is converted to a set."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="overlap_coef_list",
            is_elementwise=True,
        )

    def cosine(self, other: IntoExpr) -> pl.Expr:
        """Returns cosine distance between two lists. Each list is converted to a set."""
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            args=[self._expr, other],
            function_name="cosine_list",
            is_elementwise=True,
        )


class DExpr(pl.Expr):
    @property
    def dist(self) -> DistancePairWise:
        return DistancePairWise(self)

    @property
    def dist_arr(self) -> DistancePairWiseArray:
        return DistancePairWiseArray(self)

    @property
    def dist_str(self) -> DistancePairWiseString:
        return DistancePairWiseString(self)

    @property
    def dist_list(self) -> DistancePairWiseList:
        return DistancePairWiseList(self)


class DistColumn(Protocol):
    def __call__(
        self,
        name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
        *more_names: str | PolarsDataType,
    ) -> DExpr: ...

    def __getattr__(self, name: str) -> pl.Expr: ...

    @property
    def dist(self) -> DistancePairWise: ...

    @property
    def dist_arr(self) -> DistancePairWiseArray: ...

    @property
    def dist_str(self) -> DistancePairWiseString: ...

    @property
    def dist_list(self) -> DistancePairWiseList: ...


col = cast(DistColumn, pl.col)


__all__ = ["col"]
