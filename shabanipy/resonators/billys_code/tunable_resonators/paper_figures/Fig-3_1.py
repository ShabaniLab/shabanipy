import sys
# Adds a specific directory to the Python module search path (used for custom modules)
sys.path.append('/Users/billystrickland/Documents/code/resonators')

# Importing custom plotting settings and color configurations from the shabanipy package
from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.colors import line_colors
##from shabanipy.shabanipy.plotting import jy_pink

import csv
import matplotlib.pyplot as plt
# Import necessary functions for Labber data extraction and processing
from shabanipy.labber import LabberData
from shabanipy.resonators.notch_geometry import fit_complex, notch_from_results
import numpy as np
import os
from matplotlib.pyplot import cm
from matplotlib import colors
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit
# Import custom CSV processing utility functions
from utils.proc_csv import proc_csv

################################## CHANGE HERE #####################################################
# Define file paths and important parameters for processing
root = '/Users/billystrickland/Documents/code/resonators/data/'
sample = 'JS626-4TR-Noconst-1-BSBHE-001'  # Sample identifier
file_num = '178'  # Data file number
FILES = [root+sample+'/'+str(sample)+'-'+file_num+'.hdf5']  # Labber data file path
att = [40]  # Attenuation value (in dB)

fridge_att = 56  # Fridge attenuation
res_index = 0  # Resonator index to extract data for
skip = 1  # Skipping factor for data processing
ID = file_num+str(skip)+'new'  # ID for saving results

gui = True  # Flag for GUI-based fitting (unused here, can be activated)
err = True  # Error flag for checking results (not used directly in the code)

################################## CHANGE HERE #####################################################

# Define the channels for Labber data: Gate voltage, S21 parameter, and output power
VG_CH, S21_CH, P_CH = ['Gate - Source voltage', 'VNA - S21', 'VNA - Output power']

# Initialize variables to store processed data
power = None
freq  = None
vg = None
data  = None

# Iterate through each file in FILES and read the data
for i, FILE in enumerate(FILES[:3]):  # Loop through the first 3 files (if any)
    with LabberData(FILE) as f:  # Open the Labber data file
        _p = [-0]  # Placeholder for power data, typically could be zero if not used
        _f, _d = f.get_data(S21_CH, get_x=True)  # Get frequency and S21 data
        _v = f.get_data(VG_CH)  # Get gate voltage data
        _p = _p[::-1]  # Reverse the power data (optional based on experimental setup)
        _f = _f[::-1]  # Reverse the frequency data
        _d = _d[::-1]  # Reverse the S21 data
        _v = _v[::-1]  # Reverse the gate voltage data
        
        # Append data from current file to overall arrays (or initialize arrays if first file)
        if power is None:
            power = _p
            freq, data = _f, _d
            vg = _v
        else:
            power = np.append(power, _p, axis=0)
            freq = np.append(freq, _f, axis=0)
            data = np.append(data, _d, axis=0)
            vg = np.append(vg, _v, axis=0)

# Define directory to save processed results
newpath = root+sample+'/results'
if not os.path.exists(newpath):  # Check if the directory exists; create it if not
    os.makedirs(newpath)

# Create new arrays with sliced data for the first 22 data points (can be adjusted as needed)
new_data, new_freq, new_vg = [], [], []
for i in range(len(freq)):
    new_freq.append(freq[i][0:22].flatten())  # Slice and flatten frequency data
    new_data.append(data[i][0:22].flatten())  # Slice and flatten S21 data
    new_vg.append(vg[i][0].flatten())  # Flatten gate voltage data

# Convert the processed data into NumPy arrays for further processing
freq = np.array(new_freq)
data = np.array(new_data)
vg = np.array([r[0] for r in new_vg])  # Extract gate voltage from the nested lists

# Define limits for plotting the gate voltage and frequency
vglim, freqlim, datalim = [vg[-1], vg[0]], [freq[0][0], freq[0][-1]], [.5, 1]

# Set color limits and figure size for the plot
clims = 1
figsize = (12,5)
savepath = newpath+'/'+ID

root = '/Users/billystrickland/Documents/code/resonators/data/'
sample = 'JS626-4TR-Noconst-1-BSBHE-001'

# Define IDs for additional CSV results to process and analyze
ID1, ID2, ID3 = '1781new_tunableres', '1781new_fixedres', '2tuncav-noconst_surfimpe'

FILES = [[root+sample+'/results/csvs/'+ID1+'.csv'],
         [root+sample+'/results/csvs/'+ID2+'.csv'],
         [root+sample+'/results/csvs/'+ID3+'.csv']]
# Process the CSV files using the proc_csv function
results = [proc_csv(file) for file in FILES]

# Extract specific data (e.g., LJS and frequency values) from the processed results
ljs1 = [r[0] for i, r in enumerate(results[2]) if i < 21]
ljs2 = [r[0] for i, r in enumerate(results[2]) if i > 20]
f1 = [r[2] for i, r in enumerate(results[2]) if i < 21]
f2 = [r[1] for i, r in enumerate(results[2]) if i > 20]

# Concatenate the extracted data into one array for LJS and frequencies
ljs = np.concatenate((ljs1, ljs2))
fs = np.concatenate((f1, f2))

# Apply error threshold and extract relevant values for VG, frequency, and error
err_thresh = 100
err = [r[14] for r in results[0]]  # Extract errors from results
vg1 = [r[-2] for i, r in enumerate(results[0]) if err[i] < err_thresh]  # Gate voltage
freq1 = [r[5] for i, r in enumerate(results[0]) if err[i] < err_thresh]  # Frequency
freq_err1 = [r[11] for i, r in enumerate(results[0]) if err[i] < err_thresh]  # Frequency error

# Extract second set of results with similar structure
err = [r[14] for r in results[1]]
vg2 = [r[-2] for i, r in enumerate(results[1]) if err[i] < err_thresh]
freq2 = [r[5] for i, r in enumerate(results[1]) if err[i] < err_thresh]
freq_err2 = [r[11] for i, r in enumerate(results[1]) if err[i] < err_thresh]

# Concatenate gate voltage and frequency data from both sets
vg6 = vg1[0:125]
vg7 = vg2[125::]
freq6 = freq1[0:125]
freq7 = freq2[125::]
vg3 = np.concatenate((vg6[::5], vg7))  # Combine and downsample the gate voltage data
fs = np.concatenate((freq6[::5], freq7))  # Combine and downsample the frequency data

# Create the plot using matplotlib with a custom color map and labels
with plt.rc_context(jy_mpl_rc):
    fig, ax = plt.subplots(1,1, figsize=(14, 6))
    # Plot the absolute value of S21 against frequency and gate voltage
    img = ax.imshow(abs(data), aspect='auto', extent=[freqlim[0]*1e-9, freqlim[1]*1e-9, vglim[0], vglim[1]], cmap='viridis')
    ax.set_xlabel('$f$ (GHz)')  # X-axis label: Frequency in GHz
    ax.set_ylabel('$V_G$ (V)')  # Y-axis label: Gate voltage in V
    cbar = fig.colorbar(img)  # Colorbar for the heatmap
    cbar.set_label('$|S_{21}|$ (arb. units)')  # Colorbar label
    if clims == 1:  # Set custom color limits if needed
        cbar.mappable.set_clim(datalim[0], datalim[1])
    # Plot frequency data points as black markers on the color plot
    plt.plot(np.array(fs)*1e-9, vg3, marker='.', markersize=5, linestyle='None', color='black', label='$f_r$')
    ax.annotate('TR2', xy=(5.8, -7.5))  # Annotate specific features on the plot
    ax.annotate('TR1', xy=(5.15, -7.5))  # Annotate specific features on the plot
    fig.tight_layout()  # Tight layout for better spacing
    # Save the plot in multiple formats (EPS and PNG)
    fig.savefig(savepath+'_colorplot.eps', transparent=True, bbox_inches='tight')
    fig.savefig(savepath+'_colorplot.png', bbox_inches='tight')
    plt.show()  # Display the plot
