# Import necessary libraries and modules
import sys
sys.path.append('/Users/billystrickland/Documents/code/resonators')
from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.colors import line_colors
import matplotlib.pyplot as plt
from shabanipy.shabanipy.labber import LabberData
from shabanipy.resonators.notch_geometry import fit_complex
import numpy as np
import os
import cmath
from scipy.optimize import curve_fit
import utils.resonator_functions as rf

# Define utility functions
def to_db(x):
    return 20 * np.log10(x)

def func_deltafoverf(T, a, b):
    return -a / (2 - 2 * np.power(np.multiply(T, 1 / b), 4)) + a / 2

def alpha_k(fmeas, f0):
    return 1 - (fmeas / f0) ** 2

# Define root directory and sample info
root = '/Users/billystrickland/Library/CloudStorage/GoogleDrive-wms269@nyu.edu/.shortcut-targets-by-id/1p4A2foj_vBr4k6wrGEIpURdVkG-qAFgb/nyu-quantum-engineering-lab/labber/data-backups/qubitfridge/Data/'
year, month, day = '2022', '07', '20'
sample, file_num, res_index = 'JS626-4TR-Noconst-1-BSBHE-001', '187', 1
FILE = f"{root}/{year}/{month}/Data_{month}{day}/{sample}-{file_num}.hdf5"
P_CH, S21_CH = ['VNA - Output power', 'VNA - S21']
att, fridge_att = 0, 56

# Initialize variables for data storage
power, freq, data = None, None, None

# Read the data file and process the data
with LabberData(FILE) as f:
    _p = f.get_data(P_CH) - att - fridge_att  # Adjust power by attenuation and fridge attenuation
    _f, _d = f.get_data(S21_CH, get_x=True)  # Extract frequency and S21 data
    _p, _f, _d = _p[::-1], _f[::-1], _d[::-1]  # Reverse data to align with expected order
    if power is None:
        power, freq, data = _p, _f, _d
    else:
        power = np.append(power, _p, axis=0)  # Append power data
        freq = np.append(freq, _f, axis=0)  # Append frequency data
        data = np.append(data, _d, axis=0)  # Append S21 data

# Select the appropriate resonance data for the specified index
freq1, data1, power1 = freq[:, res_index], data[:, res_index], power[:, res_index]

# Fit the complex resonance data
results1, fdata = fit_complex(freq1[0], data1[0], powers=power1[0], gui_fit=True,
                               return_fit_data=True, delay_range=(-.001, +0.001), save_gui_fits=False,
                               save_gui_fits_filetype='.eps')

# Process the results from the CSV data
root = '/Users/billystrickland/Documents/code/resonators/data/'
sample = 'JS626-4TR-Noconst-1-BSBHE-001'
err_thresh = 10000
ID = '155res_i19_vg0V'
FILES = [root + sample + '/results/temperature/' + ID + '.csv']
results = rf.proc_csv(FILES)
photon, temp, qi_diacorr, qi_diacorr_err, qc, qc_err, ql, ql_err, freq, freq_err = rf.get_results(results, err_thresh)

# Calculate the frequency shift ratio
deltafoverf = [(r - freq[-1]) / freq[-1] for i, r in enumerate(freq)]

# Set the save path for the results
savepath = root + sample + '/results/'

# Define constants for frequency measurements
fmeas = 6.202848
f0 = 6.490670368783
alpha = 1 - (fmeas / f0) ** 2

# Fit the deltaf/f data
popt, pcov = curve_fit(func_deltafoverf, temp, deltafoverf,
                       bounds=([1 * 10 ** (-13), 1], [alpha, 10]),
                       maxfev=5000000)

# Extract fitted parameters and uncertainties
a, b = popt
perr = np.sqrt(np.diag(pcov))

# Generate the fitted curve
y_fit_1 = func_deltafoverf(temp, a, b)

# Plot the results
with plt.rc_context(jy_mpl_rc):
    fig = plt.figure(figsize=(10, 6))

    # Create subplots
    sub1 = plt.subplot(221)  # First subplot in a 2x2 grid
    plt.tick_params('x', labelbottom=False)
    sub2 = plt.subplot(223, sharex=sub1)  # Second subplot in a 2x2 grid, sharing x-axis
    sub3 = plt.subplot(2, 2, (2, 4))  # Third subplot spanning two cells

    # Plot S21 data in sub1
    sub1.plot(freq1[0] * 1e-9, to_db(abs(data1[0])) + 14, '.')
    sub1.set_ylabel('$|S_{21}|$ (dB)')
    sub1.annotate('R3, $T$ = 50 mK', xy=(6.15, -17), fontsize=14)
    sub1.annotate('$f_r$ = ' + str(round(results1[0][5] * 1e-9, 3)) + ' GHz', xy=(6.205, -41), fontsize=14)

    # Unwrap and plot the phase data in sub2
    new_data = [np.unwrap([cmath.phase(j) + 0.085 * its - 0.2 for its, j in enumerate(trace)]) for trace in data1]
    sub2.plot(freq1[0] * 1e-9, new_data[0], '.')
    sub2.set_xlabel('$f$ (GHz)')
    sub2.set_ylabel('Phase($S_{21}$) (rad)')

    # Plot deltaf/f vs temperature in sub3
    sub3.plot(temp, deltafoverf, 'D', label='Data')  # Plot the raw data
    sub3.plot(temp, y_fit_1, color=line_colors[1], label='Fit')  # Plot the fitted curve
    sub3.set_xlabel('$T$ (K)')
    sub3.set_ylabel('$\Delta f_r / f_r(0)$')

    # Format the layout
    fig.tight_layout()
    sub3.annotate(r'$ \alpha_K$ = ' + str(round(popt[0], 3)), xy=(0.1, -0.03))
    sub3.annotate('$T_C$ = ' + str(round(popt[1], 3)) + ' ± ' + str(round(perr[1], 3)) + ' K', xy=(0.1, -0.0325))

    # Add the legend
    plt.legend(bbox_to_anchor=(0.5, 0.4), frameon=False)

    # Save the figure as both EPS and PNG formats
    fig.savefig(root + sample + '/results/phase_s21_' + ID + '.eps', transparent=True)
    fig.savefig(root + sample + '/results/phase_' + ID + '.png')

    # Show the plot
    plt.show()

