"""Convenience functions for converting lmfit output to pandas DataFrame."""

from typing import Any, OrderedDict, Tuple

from lmfit import Parameters
from lmfit.model import ModelResult
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


def _conf_intervals_to_df(cis: OrderedDict) -> DataFrame:
    """Convert lmfit confidence intervals to pandas.DataFrame."""
    ncis = len(list(cis.values())[0]) // 2  # e.g. [μ - σ, μ, μ + σ]
    return DataFrame(
        ((name, *[climit for _, climit in climits]) for name, climits in cis.items()),
        columns=(
            "name",
            *[
                f"{sign}{clevel:.3f}"
                for sign, (clevel, _) in zip(
                    ("-",) * ncis + ("",) + ("+",) * ncis, list(cis.values())[0]
                )
            ],
        ),
    )


def _modelresult_to_df(mr: ModelResult) -> Tuple[DataFrame, ...]:
    """Convert lmfit ModelResult to pandas.DataFrame(s)."""
    raise NotImplementedError()


dispatch = {Parameters: _params_to_df, ModelResult: _modelresult_to_df}
