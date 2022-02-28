"""General plotting routines.

These functions only handle creation of the matplotlib objects and plotting of the data.
All plot formatting/styling should be left to the package consumer (ideally controlled
by an mplstyle sheet).
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from shabanipy.labber import LabberData, get_data_dir

from .utils import stamp as sp_stamp


def plot(
    x: np.ndarray,
    y: np.ndarray,
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
    x, y
        The data y as a function of the variable x.
        Compatible shapes are determined by matplotlib's plot function.

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

    lines = ax.plot(x, y, **plot_kwargs)
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
    xtransform: Optional[Callable] = None,
    ytransform: Optional[Callable] = None,
    ztransform: Optional[Callable] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    style: Union[str, Dict, Path, List] = "default",
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
    xtransform, ytransform, ztransform
        Functions with signature `np.ndarray -> np.ndarray` used to transform the data.
    title
        Plot title.
    ax
        The matplotlib Axes in which to plot.
    style
        Matplotlib style specifications passed to `matplotlib.pyplot.style.use`.
    **kwargs
        Additional keyword arguments passed to the plotting function
        (`shabanipy.plotting.plot2d` in the case of 2d color plots).

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
                data.append(f.get_data(name))
            except ValueError:
                # TODO refactor LabberData.get_data to normalize data access
                for log in (l for l in f.logs if l.x_name == name):
                    vdata, _ = f.get_data(log.name, get_x=True)
                    data.append(vdata)
                    break

    # normalize shapes against potentially vectorial data
    dims = [d.ndim for d in data]
    max_dim = max(dims)
    for i, d in enumerate(data):
        if d.ndim < max_dim:
            data[i] = np.expand_dims(d, axis=tuple(-np.arange(1, 1 + max_dim - d.ndim)))
    data = np.broadcast_arrays(*data)

    # apply transformations
    for i, (d, transform) in enumerate(zip(data, (xtransform, ytransform, ztransform))):
        if transform is not None:
            data[i] = transform(data[i])

    # plot
    plt.style.use(style)
    return plot2d(
        *data,
        # TODO: automatically add units in the default case
        xlabel=xlabel if xlabel is not None else x,
        ylabel=ylabel if ylabel is not None else y,
        zlabel=zlabel if zlabel is not None else z,
        title=title,
        **kwargs
    )


def _fig_ax(ax=None):
    if ax is None:
        return plt.subplots()
    else:
        return ax.get_figure(), ax
