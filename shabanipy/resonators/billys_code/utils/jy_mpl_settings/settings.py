from cycler import cycler
from jy_mpl_settings.colors import line_colors

jy_mpl_rc = {
    'axes.prop_cycle': cycler('color', line_colors),
    'image.cmap': 'jy_orange',
    'legend.fontsize': 16,
    'font.size': 16,
    'axes.labelsize': 20,
    'lines.markersize': 10,
    'lines.linewidth': 2,
    'axes.linewidth': 2,
    'xtick.labelsize': 16,
    'xtick.major.size': 4,
    'xtick.major.width': 2,
    'xtick.minor.size': 1,
    'xtick.minor.width': 1,
    'ytick.labelsize': 16,
    'ytick.major.size': 4,
    'ytick.major.width': 2,
    'ytick.minor.size': 1,
    'ytick.minor.width': 1,
}

