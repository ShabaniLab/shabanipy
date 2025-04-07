import sys
# Adds a specific directory to the Python module search path (used for custom modules)
sys.path.append('/Users/billystrickland/Documents/code/resonators')

# Importing custom plotting settings and color configurations from the shabanipy package
from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.colors import line_colors
import matplotlib.pyplot as plt
# Import necessary functions for Labber data extraction and processing
from shabanipy.shabanipy.labber import LabberData
import numpy as np
import math
##from make_plots import make_triple_plot
##from make_plots import make_deltaf_plot
# Import custom CSV processing utility functions
from utils.proc_csv import proc_csv
from scipy.interpolate import interp1d

# Define file paths for the CSV files
root = '/Users/billystrickland/Documents/code/resonators/data/'
sample = 'JS626-4TR-Noconst-1-BSBHE-001'
res_index = '0'

# Define IDs for the three CSV result files
ID1 = '1781new_tunableres'
ID2 = '1781new_fixedres'
ID3 = 'Project31_TR2'

# List of file paths to the CSV result files
FILES = [[root+sample+'/results/csvs/'+ID1+'.csv'],
         [root+sample+'/results/csvs/'+ID2+'.csv'],
         [root+sample+'/results/csvs/'+ID3+'.csv']]

# Process each CSV file using the proc_csv function
results = []
for i in range(len(FILES)):
    results1 = proc_csv(FILES[i])
    results.append(results1)

# Extract LJS and frequency data for the third result (Project31_TR2)
ljs1 = [r[0]*1e3 for r in results[2][:47]]  # First 47 data points
ljs2 = [r[0]*1e3 for r in results[2][47:]]  # Remaining data points
f1 = [r[2] for r in results[2][:47]]  # Frequencies for first part
f2 = [r[1] for r in results[2][47:]]  # Frequencies for second part

# Combine and sort the data for LJS and frequencies
ljs, fs = zip(*sorted(zip(ljs1+ljs2, f1+f2), reverse=True))

# Create the first plot showing LJS vs f_r (resonant frequency)
with plt.rc_context(jy_mpl_rc):
    fig, ax = plt.subplots(1, 1, figsize=(5.1,4), constrained_layout=True)
    ax.set_xlabel('$L_J$ (nH)')  # Label for the x-axis
    ax.set_ylabel('$f_r$ (GHz)')  # Label for the y-axis
    # Plot the data points with LJS (nH) vs resonant frequency (GHz)
    plt.plot(np.array(ljs)*1e-3, np.array(fs)*1e-9, marker = '.', linestyle='None', label='TR2')
    plt.legend()  # Add legend
    plt.show()

# Interpolate the frequency data using cubic interpolation
f2 = interp1d(fs, ljs, kind='cubic')

# Error threshold for filtering data based on error value
err_thresh = 100

# Function to extract voltage, frequency, and frequency error values from results
def get_values(results):
    err = [r[14] for r in results]  # Extract errors
    vg = [r[-2] for i, r in enumerate(results) if err[i] < err_thresh]  # Extract gate voltage where error is below threshold
    freq = [r[5] for i, r in enumerate(results) if err[i] < err_thresh]  # Extract frequency
    freq_err = [r[11] for i, r in enumerate(results) if err[i] < err_thresh]  # Extract frequency error
    return vg, freq, freq_err

# Extract data from two sets of results (first and second sets)
vg1, freq1, freq_err1 = get_values(results[0])
vg2, freq2, freq_err2 = get_values(results[1])

# Combine the gate voltage and frequency data from both sets
vg = np.concatenate((vg1[0:125], vg2[125::]))
fs = np.concatenate((freq1[0:125], freq2[125::]))
fs_err = np.concatenate((freq1[0:125], freq2[125::]))

# Calculate the detuning (difference between frequencies)
delta = abs(np.array(freq1) - np.array(freq2))

# Create the second plot showing detuning vs gate voltage
with plt.rc_context(jy_mpl_rc):
    fig, ax = plt.subplots(1, 1, figsize=(5.1,3), constrained_layout=True)
    plt.plot(vg1, np.array(delta)*1e-9, marker = '.', color=line_colors[2])  # Plot detuning in GHz vs gate voltage
    ax.set_yticks([0.0, 0.5, 1.0])  # Set y-axis ticks
    ax.set_xlabel('$V_G$ (V)')  # Label for the x-axis
    ax.set_ylabel('$\Delta$ (GHz)')  # Label for the y-axis
    savepath = root+sample+'/results/'+ID1
    # Save the plot as EPS and PNG files
    fig.savefig(savepath+'_detuning.eps', transparent=True, bbox_inches='tight')
    fig.savefig(savepath+'_detuning.png', bbox_inches='tight')
    plt.show()

# Create the third plot showing LJS vs gate voltage, along with IC (critical current)
fig, ax = plt.subplots(1, 1, figsize=(6,3), constrained_layout=True)
ax.set_xlabel('$V_G$ (V)')  # Label for the x-axis
ax.set_ylabel('$L_J$ (nH)')  # Label for the y-axis

# Annotate arrows on the plot to highlight specific regions (example with custom coordinates)
ax.annotate('', xy=(0.35, 0.3),  xycoords='axes fraction',
            xytext=(0.65, 0.3), textcoords='axes fraction',
            arrowprops=dict(facecolor=line_colors[1], ec = line_colors[1], shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

ax.annotate('', xy=(0.9, 0.75),  xycoords='axes fraction',
            xytext=(0.6, 0.75), textcoords='axes fraction',
            arrowprops=dict(facecolor=line_colors[0], ec = line_colors[0], shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

# Sort and plot LJS vs gate voltage
vg, fs = zip(*sorted(zip(vg, fs), reverse=True))
plt.plot(vg, f2(fs)*1e-3, marker = '.', color=line_colors[1])

# Create a secondary y-axis to plot IC (critical current)
ax2 = ax.twinx()
mfq = 2.068e-15  # Magnetic flux quantum (in units of Tesla*meter^2)
ics = mfq/(2*math.pi*f2(fs)*1e-12)*1e6  # Calculate critical current (IC) in µA
ax2.plot(vg, ics, marker = '.', color=line_colors[0])  # Plot IC vs gate voltage

# Set label for the secondary y-axis (critical current)
ax2.set_ylabel('$I_C$ (µA)')

# Save the plot as EPS and PNG files
fig.savefig(savepath+'_ics_interp.eps', transparent=True, bbox_inches='tight')
fig.savefig(savepath+'_ics_interp.png', bbox_inches='tight')
plt.show()
