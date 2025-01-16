from collections import defaultdict

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap


def register():
    cpts = [  # RGB
        (4, 88, 147),  # tab:blue
        (150, 150, 150),  # ~grey
        (180, 12, 13),  # tab:red
    ]
    cpts = [(r / 255, g / 255, b / 255) for r, g, b in cpts]
    cdict = defaultdict(list)
    for i, c in enumerate(cpts):
        cdict["red"].append((i / (len(cpts) - 1), c[0], c[0]))
        cdict["green"].append((i / (len(cpts) - 1), c[1], c[1]))
        cdict["blue"].append((i / (len(cpts) - 1), c[2], c[2]))
    cmap = LinearSegmentedColormap("cooldarkwarm", cdict)
    colormaps.register(cmap=cmap)

    cpts = list(reversed(cpts))
    cdict = defaultdict(list)
    for i, c in enumerate(cpts):
        cdict["red"].append((i / (len(cpts) - 1), c[0], c[0]))
        cdict["green"].append((i / (len(cpts) - 1), c[1], c[1]))
        cdict["blue"].append((i / (len(cpts) - 1), c[2], c[2]))
    cmap = LinearSegmentedColormap("cooldarkwarm_r", cdict)
    colormaps.register(cmap=cmap)
