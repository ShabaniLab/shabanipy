"""General plotting utilities."""
from matplotlib import rcParams


def format_phase(value, tick_number):
    """The value are expected in unit of π

    """
    if value == 0:
        return '0'
    elif value == 1.0:
        return 'π'
    elif value == -1.0:
        return '-π'
    else:
        return f'{value:g}π'


def stamp(ax, text):
    """Stamp the plot with an ID."""
    ax.text(
        1,
        1,
        text,
        size=0.4 * rcParams["font.size"],
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
