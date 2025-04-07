import sys
# Adds a specific directory to the Python module search path (used for custom modules)
sys.path.append('/Users/billystrickland/Documents/code/resonators')

# Import custom plotting settings, helper functions, and color configurations from the shabanipy package
from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.helper import *
from shabanipy.jy_mpl_settings.colors import line_colors
import matplotlib.pyplot as plt
# Import necessary functions for Labber data extraction
from shabanipy.labber import LabberData
# Import specific functions for resonator notch geometry fitting
from shabanipy.resonators.notch_geometry import fit_complex, notch_from_results
import numpy as np
import os
# Import colormap from matplotlib
from matplotlib.pyplot import cm
# Import numpy's diff function for calculating numerical differences
from numpy import diff

# Function to convert raw values to decibels (dB), with an optional reference value
def to_db(value_raw, reference=1.):
    return 10*np.log10(value_raw/reference)

# Define the root directory and the specific data file paths
root = '/Users/billystrickland/Library/CloudStorage/GoogleDrive-wms269@nyu.edu/.shortcut-targets-by-id/1p4A2foj_vBr4k6wrGEIpURdVkG-qAFgb/nyu-quantum-engineering-lab/labber/data-backups/qubitfridge/Data/'
date = '2022/07/Data_0701'
sample = 'JS626-4TR-Noconst-1-BSBHE-001'
file_num = '124'
FILES = [root+date+'/'+str(sample)+'-'+file_num+'.hdf5']

# Set attenuation values and the index of the resonator
att = [0]  # Attenuation (dB)
fridge_att = 56  # Fridge attenuation (dB)
res_index = 3  # Index for selecting a specific resonator
ID = file_num  # Identifier for the current file
gui = False  # Flag for GUI usage (not used in this code)
gate = [-4]  # Gate voltage values (in V)
# Define the channels for Gate, S21, and Output power
VG_CH, S21_CH, P_CH = ['Gate - Source voltage', 'VNA - S21', 'VNA - Output power']
power = None  # Initialize power variable
freq  = None  # Initialize frequency variable
data  = None  # Initialize data variable

# Loop over the files (only the first file is processed in this case)
for i, FILE in enumerate(FILES[:3]):
    with LabberData(FILE) as f:
        # Read data from the file for power, gate voltage, frequency, and S21
        _p = f.get_data(P_CH) - att - fridge_att  # Adjust power with attenuation
        _v = f.get_data(VG_CH)  # Gate voltage data
        _f, _d = f.get_data(S21_CH, get_x=True)  # S21 data (frequency and magnitude)
        
        # Reverse the data for proper plotting (from high to low frequency)
        _p = _p[::-1]
        _f = _f[::-1]
        _d = _d[::-1]
        _v = _v[::-1]
        
        # If power is not yet defined, initialize it and the other variables
        if power is None:
            power = _p
            gate = _v
            freq, data = _f, _d
        else:
            # Otherwise, append the new data to the existing arrays
            gate = np.append(gate, _v, axis=0)
            power = np.append(power, _p, axis=0)
            freq = np.append(freq, _f, axis=0)
            data = np.append(data, _d, axis=0)


root = '/Users/billystrickland/Documents/code/resonators/data/'

# Define the frequency and power limits for the plot
freqlim = [freq[res_index][0][0]*.000000001, freq[res_index][0][-1]*.000000001]
plim = [-76, -56]

# Extract the frequency and data for the specific resonator index
x = freq[res_index][i+1]*1e-9
y = 1-abs(data[res_index][i+1])
dy = diff(y)/diff(x)  # Calculate the derivative of the data with respect to frequency

# Print power and maximum derivative information
print(power[0])
print(np.argmax(dy), max(dy))

results = []  # Initialize results list
maxes = []  # Initialize list for storing the maximum values

# Start plotting with custom settings defined in jy_mpl_rc
with plt.rc_context(jy_mpl_rc):
    fig, ax = plt.subplots(1, 1, figsize=(5, 12))  # Create a figure with specific size
    cholor = cm.viridis(np.linspace(0, 1, 200))  # Generate a colormap (viridis)
    color = iter(cm.viridis(np.linspace(0, 1, 200)))  # Create an iterator for the colors
    
    # Loop through 200 points and plot each one with varying colors
    for i in range(200):
        c = next(color)  # Get the next color from the colormap
        x = freq[res_index][i+1]*1e-9  # Convert frequency to GHz
        y = 1-abs(data[res_index][i+1])  # Calculate the absolute S21 magnitude
        dy = diff(y)/diff(x)  # Compute the derivative
        mx = np.argmax(dy)  # Find the index of the maximum derivative
        maxes.append(freq[res_index][i+1][mx]*1e-9)  # Store the frequency corresponding to the max derivative
        
        # Plot the data with the corresponding color
        if i%5==0:
            ax.plot(freq[res_index][i+1][200:800]*1e-9, 1-abs(data[res_index][i+1][200:800])-.0005*i, color=c)

    # Save the results to a CSV file
    np.savetxt(root+sample+'/results/'+ID+'_'+str(res_index)+'_maxes.csv', results, delimiter=',')
    
    # Set axis labels and title
    ax.set_xlabel('$f$ (GHz)')
    ax.set_ylabel('$1-|S_{21}|$ (a.u.)')
    ax.set_title('TR2, $V_G = -10$ V')
    
    # Annotate the plot with power information
    ax.annotate('$P$ = -76 dBm', xycoords='figure fraction', xy=(.65, .1), color=cholor[-1])
    ax.annotate('$P$ = -56 dBm', xycoords='figure fraction', xy=(.22, .91), color=cholor[0])
    ax.set_ylim(0.868, 0.98)  # Set the y-axis limits
    
    # Save the plot as an EPS file
    fig.savefig(root+sample+'/'+file_num+'_bif_lines'+str(res_index)+'.eps', transparent=True, bbox_inches='tight')
    fig.tight_layout()  # Adjust the layout for tight fitting

    # Create another plot displaying the S21 data as an image (heatmap)
    cmap = cm.viridis  # Use the viridis colormap
    fig, ax = plt.subplots(1, 1, figsize=(7.3 ,5))  # Create a new figure
    img = ax.imshow(abs(data[res_index]), aspect='auto',
                    extent=[freqlim[0], freqlim[1], plim[0], plim[1]],
                    cmap=cmap)  # Display the S21 data as a heatmap
    
    # Set axis labels based on resonator index
    if res_index == 3:
        ax.set_xlabel('$f$ (GHz)', color='w')  # White text for certain resonator
    if res_index == 2:
        ax.set_xlabel('$f$ (GHz)')  # Default label
    
    ax.set_ylabel('Power (dBm)')
    
    # Add a colorbar to the plot
    cbar = plt.colorbar(img)
    cbar.set_label('$|S_{21}|$ (a.u.)')  # Label for colorbar
    ax.axes.get_yaxis().set_visible(False)  # Hide the y-axis ticks
    cbar.mappable.set_clim(0.014, 0.032)  # Set the limits for color scaling
    ax.set_title('$V_G = $'+str(gate[res_index][0])+' V')  # Title with gate voltage value
    
    # Plot the maximum points on top of the heatmap
    ax.plot(maxes[::10], power[0][1::][::10], marker='*', markersize=4, linestyle='None', color='k')
    
    # Tighten the layout and display the plot
    fig.tight_layout()
    plt.show()
    
    # Save the second plot as an EPS file
    fig.savefig(root+sample+'/'+file_num+'_bif_'+str(res_index)+'.eps', transparent=True, bbox_inches='tight')

