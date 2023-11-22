import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from typing import Protocol, Iterable, cast
from polars.type_aliases import PolarsDataType, IntoExpr

lib = _get_shared_lib_location(__file__)

__version__ = "0.1.1"


@pl.api.register_expr_namespace("pdist")
class DistancePairWise:
    def __init__(self, expr: pl.Expr):
        self._expr = expr


@pl.api.register_expr_namespace("pdist_str")
class DistancePairWiseString:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def hamming(self, other: IntoExpr) -> pl.Expr:
        """Returns hamming distance between two expressions"""
        return self._expr.register_plugin(
            lib=lib,
            args=[other],
            symbol="hamming_string",
            is_elementwise=True,
        )

    def levenshtein(self, other: IntoExpr) -> pl.Expr:
        """Returns levenshtein distance between two expressions"""
        return self._expr.register_plugin(
            lib=lib,
            args=[other],
            symbol="levenshtein_string",
            is_elementwise=True,
        )


class DExpr(pl.Expr):
    @property
    def pdist(self) -> DistancePairWise:
        return DistancePairWise(self)

    @property
    def pdist_str(self) -> DistancePairWiseString:
        return DistancePairWiseString(self)


class DistColumn(Protocol):
    def __call__(
        self,
        name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
        *more_names: str | PolarsDataType,
    ) -> DExpr:
        ...

    def __getattr__(self, name: str) -> pl.Expr:
        ...

    @property
    def pdist(self) -> DistancePairWise:
        ...

    @property
    def pdist_str(self) -> DistancePairWiseString:
        ...


col = cast(DistColumn, pl.col)


__all__ = ["col"]
