"""General plotting routines.

These functions only handle creation of the matplotlib objects and plotting of the data.
All plot formatting/styling should be left to the package consumer (ideally controlled
by an mplstyle sheet).
"""
import numpy as np
from matplotlib.pyplot import subplots

from .utils import stamp as sp_stamp


def plot(
    x, y, xlabel=None, ylabel=None, title=None, ax=None, stamp=None, **plot_kwargs
):
    """Plot 1-dimensional data y(x).

    Parameters
    ----------
    x, y : np.ndarray
        The data y as a function of the variable x.
        Compatible shapes are determined by matplotlib's plot function.

    Optional parameters
    -------------------
    xlabel, ylabel : str
        The axes labels.
    title : str
        The plot title.
    ax : matplotlib Axes or AxesSubplot
        The axes in which to plot.
        If None, a new Figure and AxesSubplot will be created.
    stamp : str
        A small text label to put on the plot.
    **plot_kwargs : dict
        Keyword arguments to pass to matplotlib's plot function.

    Returns
    -------
    (Figure, Axes or AxesSubplot)
        The figure and axes where the data was plotted.
    """
    fig, ax = _fig_ax(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if stamp is not None:
        sp_stamp(ax, stamp)

    lines = ax.plot(x, y, **plot_kwargs)
    if "label" in plot_kwargs:
        ax.legend()
    return fig, ax


# TODO support line cuts
def plot2d(
    x,
    y,
    z,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    title=None,
    ax=None,
    stamp=None,
    **pcm_kwargs
):
    """Plot 2-dimensional data z(x, y) in a color plot with a colorbar.

    Parameters
    ----------
    x, y, z : np.ndarray
        The data z as a function of variables x and y.
        Compatible shapes are determined by matplotlib's pcolormesh.

    Optional parameters
    -------------------
    xlabel, ylabel, zlabel : str
        The axes labels. The zlabel will be applied to the colorbar.
    title : str
        The plot title.
    ax : matplotlib Axes or AxesSubplot
        The axes in which to plot.
        If None, a new Figure and AxesSubplot will be created.
    stamp : str
        A small text label to put on the plot.
    **pcm_kwargs : dict
        Keyword arguments to pass to matplotlib's pcolormesh.

    Returns
    -------
    (Figure, Axes or AxesSubplot)
        The figure and axes where the data was plotted.
    """
    fig, ax = _fig_ax(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if stamp is not None:
        sp_stamp(ax, stamp)

    pcm_kwargs.setdefault("shading", "auto")
    mesh = ax.pcolormesh(x, y, z, **pcm_kwargs)

    extend_min = "vmin" in pcm_kwargs and pcm_kwargs["vmin"] > np.min(z)
    extend_max = "vmax" in pcm_kwargs and pcm_kwargs["vmax"] < np.max(z)
    extend = (
        "both" if extend_min and extend_max
        else "min" if extend_min
        else "max" if extend_max
        else "neither"
    )
    cb = fig.colorbar(mesh, ax=ax, extend=extend, label=zlabel)
    return fig, ax


def _fig_ax(ax=None):
    if ax is None:
        return subplots()
    else:
        return ax.get_figure(), ax
