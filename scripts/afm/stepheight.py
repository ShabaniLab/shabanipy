"""Extract the height of a step measured by NMI AFM.

The input to this script is a histogram with two distinguishable peaks exported from
NMI's data analysis software.
"""

import argparse

import numpy as np
from lmfit.models import GaussianModel
from matplotlib import pyplot as plt
from pandas import read_csv

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "path", help="path to the .txt histogram data file exported from NMI AFM software"
)
args = parser.parse_args()
names = ("height (nm)", "pixel count")
df = read_csv(args.path, skiprows=4, sep="\s+", names=names)

plt.ion()
plt.style.use("fullscreen13")
plt.bar(*names, data=df)
plt.plot([], [])  # dummy plot to advance color cycle
plt.xlabel(names[0])
plt.ylabel(names[1])

# could be automated with scipy.signal.find_peaks
sep = float(input("Separate peaks at (nm): "))
plt.axvline(sep, color="k")


def fit_gaussian(x, y):
    model = GaussianModel()
    result = model.fit(y, x=x, center=x[np.argmax(y)])
    (line,) = plt.plot(x, result.eval())
    _, mean, std, _, height = result.params.valuesdict().values()
    plt.text(mean, height, f"{mean:.1f} $\pm$ {std:.1f} nm", color=line.get_color())
    return mean, std


x, y = df[names[0]].to_numpy(), df[names[1]].to_numpy()
mean1, std1 = fit_gaussian(x[x < sep], y[x < sep])
mean2, std2 = fit_gaussian(x[x > sep], y[x > sep])
plt.title(
    f"step height = {mean2 - mean1:.1f} $\pm$ {np.sqrt(std1**2 + std2**2):.1f} nm"
)

plt.ioff()
plt.savefig("stepheight.png")
plt.show()
