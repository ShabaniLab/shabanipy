"""Convenience functions for converting lmfit output to pandas DataFrame."""

from typing import Any

from lmfit import Parameters
from pandas import DataFrame


def to_dataframe(obj: Any) -> DataFrame:
    """Convert an lmfit object to a pandas DataFrame."""
    return dispatch[type(obj)](obj)


def _params_to_df(params: Parameters) -> DataFrame:
    """Convert lmfit.Parameters to pandas.DataFrame."""
    return DataFrame(
        [
            (p.name, p.vary, p.value, p.stderr, p.min, p.max, p.brute_step, p.expr)
            for p in params.values()
        ],
        columns=(
            "name",
            "vary",
            "value",
            "stderr",
            "min",
            "max",
            "brute step",
            "expr",
        ),
    )


dispatch = {Parameters: _params_to_df}
