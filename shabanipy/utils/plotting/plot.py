"""General plotting routines.

These functions only handle creation of the matplotlib objects and plotting of the data.
All plot formatting/styling should be left to the package consumer (ideally controlled
by an mplstyle sheet).
"""
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from shabanipy.labber import LabberData, get_data_dir

from .utils import stamp as sp_stamp


def plot(
    *plot_args,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    stamp: Optional[str] = None,
    **plot_kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 1-dimensional data y(x).

    Parameters
    ----------
    *plot_args
        Positional arguments to pass to matplotlib's plot function.
        E.g. the x, y data arrays and format string.

    Optional parameters
    -------------------
    xlabel, ylabel
        The axes labels.
    title
        The plot title.
    ax
        The axes in which to plot.
        If None, a new Figure and AxesSubplot will be created.
    stamp
        A small text label to put on the plot.
    **plot_kwargs
        Keyword arguments to pass to matplotlib's plot function.

    Returns
    -------
    The figure and axes where the data was plotted.
    """
    fig, ax = _fig_ax(ax)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if stamp is not None:
        sp_stamp(ax, stamp)

    lines = ax.plot(*plot_args, **plot_kwargs)
    if "label" in plot_kwargs:
        ax.legend()
    return fig, ax


# TODO support line cuts
def plot2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    stamp: Optional[str] = None,
    extend_min: Optional[bool] = None,
    **pcm_kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2-dimensional data z(x, y) in a color plot with a colorbar.

    Parameters
    ----------
    x, y, z
        The data z as a function of variables x and y.
        Compatible shapes are determined by matplotlib's pcolormesh.

    Optional parameters
    -------------------
    xlabel, ylabel, zlabel
        The axes labels. The zlabel will be applied to the colorbar.
    title
        The plot title.
    ax
        The axes in which to plot.
        If None, a new Figure and AxesSubplot will be created.
    stamp
        A small text label to put on the plot.
    extend_min
        Option to override automatic colorbar extensions (i.e. arrows) when colorbar
        does not span the full range of data.
    **pcm_kwargs
        Keyword arguments to pass to matplotlib's pcolormesh.

    Returns
    -------
    The figure and axes where the data was plotted.
    """
    fig, ax = _fig_ax(ax)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if stamp is not None:
        sp_stamp(ax, stamp)

    pcm_kwargs.setdefault("shading", "auto")
    mesh = ax.pcolormesh(x, y, z, **pcm_kwargs)

    if extend_min is None:
        extend_min = "vmin" in pcm_kwargs and pcm_kwargs["vmin"] > np.min(z)
    extend_max = "vmax" in pcm_kwargs and pcm_kwargs["vmax"] < np.max(z)
    extend = (
        "both"
        if extend_min and extend_max
        else "min"
        if extend_min
        else "max"
        if extend_max
        else "neither"
    )
    cb = fig.colorbar(mesh, ax=ax, extend=extend, label=zlabel)
    return fig, ax


def plot_labberdata(
    path: Union[str, Path],
    x: str,
    y: str,
    z: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    transform: Optional[Callable] = None,
    filters: Optional[Dict[str, float]] = None,
    xlim: Optional[Tuple[float]] = None,
    ylim: Optional[Tuple[float]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot data from a Labber .hdf5 log file.

    Parameters
    ----------
    path
        The path to data file. If not absolute, it should be relative to the output of
        `shabanipy.labber.get_data_dir`.
    x, y, z
        Channel names or vector channel `x_name`s specifying which data to plot.
    xlabel, ylabel, zlabel
        Axes labels.  If None, the names from `x`, `y`, and `z` will be used.
    transform
        Function with the signature `Tuple[np.ndarray] -> Tuple[np.ndarray]`,
        i.e. (x, y, z) -> (x_transformed, y_transformed, z_transformed) used to
        transform the data.
    filters
        Dictionary of {"channel": value} pairs used to select 2d slices of n-dimensional
        data.  Passed to LabberData.get_data().
    xlim, ylim
        x- and y-axis limits, in the form (min, max), referring to the transformed data
        if `transform` is given.  If either min or max is None, the limit is left
        unchanged.
    title
        Plot title.
    ax
        The matplotlib Axes in which to plot.
    **kwargs
        Additional keyword arguments passed to the plotting function
        (`shabanipy.utils.plotting.plot2d` in the case of 2d color plots).

    Returns
    -------
    The figure and axes where the data were plotted.
    """
    path = Path(path)
    if not path.is_absolute():
        path = get_data_dir() / path

    # get the data
    data = []
    with LabberData(path) as f:
        for name in (x, y, z):
            try:
                data.append(f.get_data(name, filters=filters))
            except ValueError:
                # TODO refactor LabberData.get_data to normalize data access
                for log in (l for l in f.logs if l.x_name == name):
                    vdata, _ = f.get_data(log.name, get_x=True, filters=filters)
                    data.append(vdata)
                    break

    # normalize shapes against potentially vectorial data and handle interrupted scans
    # TODO refactor into LabberData.get_data
    dims = [d.ndim for d in data]
    max_dim = max(dims)
    min_length = min([d.shape[0] for d in data])
    for i, d in enumerate(data):
        if d.ndim < max_dim:
            data[i] = np.expand_dims(
                d[:min_length], axis=tuple(-np.arange(1, 1 + max_dim - d.ndim))
            )
    data = np.broadcast_arrays(*data)

    # apply transformations
    if transform is not None:
        data = transform(*data)

    # plot
    fig, ax = plot2d(
        *data,
        # TODO: automatically add units in the default case
        xlabel=xlabel if xlabel is not None else x,
        ylabel=ylabel if ylabel is not None else y,
        zlabel=zlabel if zlabel is not None else z,
        title=title,
        **kwargs
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return fig, ax


def _fig_ax(ax=None):
    if ax is None:
        return plt.subplots()
    else:
        return ax.get_figure(), ax
