"""Fit the diode effect on the switching current vs. in-plane field.

This script fits Ic+(B||) and Ic-(B||) according to Eq. (1) of
https://arxiv.org/abs/2303.01902v2.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "datapath", help="path to .csv file containing positive & negative switching current vs. field"
)
args = parser.parse_args()

df = read_csv(args.datapath)

# Define the equation
def equation(By, Bs, b, c, I0, Ic):
    "used for ic+"
    return -(np.abs(Ic) / I0) + (1 - b * (1 + c * np.sign(By - Bs)) * (By - Bs) ** 2)

def equation2(By, Bs, b, c, I0, Ic):
    "used for ic-"
    return -(np.abs(Ic) / I0) + (1 - b * (1 - c * np.sign(By + Bs)) * (By + Bs) ** 2)

# Extract Ic and By from the DataFrame
Ic = np.abs(df['ic-'].values)
Icp = np.abs(df['ic+'].values)
By = df['b_inplane'].values

#restrict to +/- 100mT:

Ic_cut = Ic[np.where(By==-0.1)[0][0]:]
Ic_cut = Ic_cut[:np.where(By==0.1)[0][0]-len(By)+1]

Icp_cut = Icp[np.where(By==-0.1)[0][0]:]
Icp_cut = Icp_cut[:np.where(By==0.1)[0][0]-len(By)+1]


By_cut = By[np.where(By==-0.1)[0][0]:]
By_cut = By_cut[:np.where(By==0.1)[0][0]-len(By)+1]

#define I0+/-:
I0 = np.abs(df['ic-'].min())
I0p = df['ic+'].max()

# Concatenate the data for both curves
Ic_concat = np.concatenate((Icp_cut, Ic_cut))
By_concat = np.concatenate((By_cut, By_cut))

# Define the value of I0
I0_concat = np.max([I0, I0p])

# Define the combined equation
def combined_equation(By, Bs, b, c, I0, Ic, Icp):
    equation1 = -(np.abs(Ic) / I0) + (1 - b * (1 - c * np.sign(By + Bs)) * (By + Bs) ** 2)
    equation2 = -(np.abs(Icp) / I0p) + (1 - b * (1 + c * np.sign(By - Bs)) * (By - Bs) ** 2)
    return np.where(By >= 0, equation1, equation2)

initial_guess = [0.001, 25, -0.1]

# Define the bounds for the fitting parameters
bounds = ([-0.1, 10, -0.1], [0.1, 30, 0.1])

# Perform curve fitting
popt, pcov = curve_fit(
    lambda By_concat, Bs, b, c: combined_equation(By_concat, Bs, b, c, I0_concat, Ic_concat, Ic_concat),
    By_concat, Ic_concat,
    p0=initial_guess,
    bounds=bounds,
    maxfev=10000
)

# Retrieve the fitted parameters
Bs_fit, b_fit, c_fit = popt

print("Fitted parameters:")
print("Bs:", Bs_fit)
print("b:", b_fit)
print("c:", c_fit)


# Generate points for the fitted curve
By_fit = np.linspace(-0.1, 0.1, 100)
Ic_fit = equation2(By_fit, Bs_fit, b_fit, c_fit, I0, 0)  # Use 0 as the reference Ic value
Icp_fit = equation(By_fit, Bs_fit, b_fit, c_fit, I0p, 0)  # Use 0 as the reference Ic value

# Plot the original data points and the fitted curve
plt.scatter(By_cut, Icp_cut/I0p, label='Original Data Ic+')
plt.plot(By_fit, Icp_fit, 'g-', label='Fitted Curve Ic+')

plt.scatter(By_cut, Ic_cut/I0, label='Original Data Ic-')
plt.plot(By_fit, Ic_fit, 'r-', label='Fitted Curve Ic-')

plt.xlabel('By')
plt.ylabel('|Ic|/I0')
plt.title('Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()
