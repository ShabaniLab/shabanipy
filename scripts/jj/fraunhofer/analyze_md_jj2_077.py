"""Reconstruct current density distribution from wide-scan Fraunhofer.

Device ID: JS311_2HB-2JJ-5MGJJ-MD-001_JJ2.
Scan ID: JS311-BHENL001-2JJ-2HB-5MGJJ-JJ2-077.
Fridge: vector9
"""

import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as cs

from shabanipy.jj.fraunhofer.deterministic_reconstruction import (
    extract_current_distribution,
)
from shabanipy.jj.utils import extract_switching_current
from shabanipy.utils.labber_io import LabberData

LABBER_DATA_DIR = os.environ["LABBER_DATA_DIR"]
DATA_FILE_PATH = (
    Path(LABBER_DATA_DIR)
    / "2020/12/Data_1206/JS311-BHENL001-2JJ-2HB-5MGJJ-JJ2-077.hdf5"
)

# channel names
CH_MAGNET = "Magnet Source - Source current"
CH_DMM = "VITracer - VI curve"  # gives voltage drop across junction

# coil current to B-field conversion factor (for x-axis vector magnet)
CURR_TO_FIELD = 1 / 30

# constants
PHI0 = cs.h / (2 * cs.e)  # magnetic flux quantum
JJ_WIDTH = 4e-6
JJ_LENGTH = 1200e-9  # effective length (>> nominal length)
FIELD_TO_WAVENUM = 2 * np.pi * JJ_LENGTH / PHI0  # B-field to beta wavenumber

with LabberData(DATA_FILE_PATH) as f:
    field = f.get_data(CH_MAGNET) * CURR_TO_FIELD
    # Bias current from the custom Labber driver VICurveTracer isn't available via
    # LabberData methods.
    # NOTE: The use of np.unique assumes the bias values are identical for each sweep.
    # This is true for the current datafile but may not hold in general.
    bias = np.unique(f._file["/Traces/VITracer - VI curve"][:, 1, :])
    volt = f.get_data(CH_DMM)

# Plot the raw data to be corrected for field jumps.  Transposing makes manual cropping
# easier by displaying the data in imshow with field index on the horizontal axis.
# Taking differential helps to increase contrast between the superconducting and normal
# regions.
dV = np.transpose(np.diff(volt, axis=1))
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 5))
ax.set_title("raw data")
im = ax.imshow(dV, origin="lower", vmin=0, vmax=1e-4)
crop_l, crop_r = 300, 1305
ax.set_xlim(crop_l, crop_r)
# plt.show()
plt.close(fig)

# Correct the field jumps and plot
cut1_r, cut1_l = 527, 544
cut2_r, cut2_l = 558, 573
cut3_r, cut3_l = 1166, 1195
crop = np.r_[crop_l:cut1_r, cut1_l:cut2_r, cut2_l:cut3_r, cut3_l:crop_r]
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 5))
ax.set_title("corrected")
im = ax.imshow(dV[:, crop], origin="lower", vmin=0, vmax=1e-4)
# plt.show()
plt.close(fig)

# Save cropped data
dV = dV[:, crop]
volt = volt[crop, :]
# discard field values from the ends
field = field[crop_l + (cut1_l - cut1_r) + (cut2_l - cut2_r) : crop_r - cut3_l + cut3_r]

# Field and bias scans have evenly spaced steps
dfield = np.diff(field)[0]
dbias = np.diff(bias)[0]

# Save a corrected dV/dI plot
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 5))
ax.set_xlabel(r"$B_\perp$ (mT)")
ax.set_ylabel(r"$I_\mathrm{bias}$ (μA)")
im = ax.imshow(
    1e-3 * dV / np.diff(bias[:, np.newaxis], axis=0),
    origin="lower",
    vmin=0,
    vmax=15,
    extent=[
        (field.min() - dfield) * 1e3,
        (field.max() + dfield) * 1e3,
        # recall dV is diffed along bias and shrinks by 1 element
        (bias.min() + dbias) * 1e6,
        (bias.max() - dbias) * 1e6,
    ],
    aspect=3 / 2,
)
cb = fig.colorbar(im, label=r"$\Delta V / \Delta I$ (kΩ)")
plt.show()
# fig.savefig('077_wide-fraunhofer-dVdI_corrected.pdf')

# extract_switching_current needs arrays of the same shape: use tile
ic = extract_switching_current(
    np.tile(bias, volt.shape[:-1] + (1,)), volt, threshold=2.32e-3
)

# Plot critical current
fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 5))
ax.set_xlabel(r"$B_\perp$ (mT)")
ax.set_ylabel(r"$I_c$ (μA)")
ax.plot(field * 1e3, ic * 1e6)
plt.show()
# fig.savefig('077_wide-fraunhofer-Ic_corrected.pdf')

# Center and symmetrize fraunhofer (optional; this fraunhofer is already highly
# symmetric and centered).
# field = field - find_fraunhofer_center(field, ic)
# field, ic = symmetrize_fraunhofer(field, ic)

# Reconstruct supercurrent distribution and plot
x, jx = extract_current_distribution(field, ic, FIELD_TO_WAVENUM, JJ_WIDTH, len(field))
fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlabel(r"$x$ (μm)")
ax.set_ylabel(r"$J(x)$ (μA/μm)")
ax.plot(x * 1e6, jx)
plt.show()
# fig.savefig('077_hi-res-current-density.pdf')
