# --- Setup system path to include a local directory for module access, should be where your shabanipy is ---
import sys
home_dir = '/Users/billystrickland/Documents/code/resonators/shabanipy'
sys.path.append(home_dir)  # Add custom directory to Python path to import local modules

# --- Import required libraries ---
import os
import numpy as np
from shabanipy.labber import LabberData  # Custom module to handle Labber data files
from shabanipy.resonators.notch_geometry import fit_complex  # Function for complex S21 fitting

# --- Define the root path to the Labber data files ---
root = '/Users/billystrickland/Library/CloudStorage/GoogleDrive-wms269@nyu.edu/.shortcut-targets-by-id/1p4A2foj_vBr4k6wrGEIpURdVkG-qAFgb/nyu-quantum-engineering-lab/labber/data-backups/qubitfridge/Data/'

# --- Metadata for data selection ---
year = '2025'
month = '04'
day = '04'
sample = 'CandleQubits-WMSAB-CD06'
file_num = '008'

# --- Construct full path to the HDF5 measurement file ---
FILE = root+year+'/'+month+'/'+'Data_'+month+day+'/'+sample+'-'+file_num+'.hdf5'

# --- Define a path to save processed data ---
savepath = home_dir+'/data/'+sample

# --- Define Labber channel names for power and S21 (transmission magnitude) ---
P_CH, S21_CH = ['Digital Attenuator - Attenuation', 'VNA - S21']

# --- Load measurement data using the LabberData context manager ---
with LabberData(FILE) as f:
    freq, data = f.get_data(S21_CH, get_x=True)  # Get frequency and S21 complex data
    power = -f.get_data(P_CH) - 76               # Calculate input power accounting for fridge attenuation

# --- Create directory to save fitting results ---
newpath = os.path.join(savepath, 'results', 'fits', file_num)
os.makedirs(newpath, exist_ok=True)  # Create path if it doesn't already exist

# --- (Optional) Print shapes of the datasets for debugging ---
##print(np.shape(freq))
##print(np.shape(data))
##print(np.shape(power))

# --- Fit the measured S21 data using a complex resonator model ---
results, fdata = fit_complex(
    freq,
    data,
    powers=power,
    gui_fit=True,                      # Enable GUI for interactive fitting
    return_fit_data=True,             # Return the fitted data for later use
    delay_range=(-.2, +.2),           # Set allowable cable delay range during fitting
    save_gui_fits=True,               # Save the GUI-generated fits
    save_gui_fits_path=newpath,       # Path to save those fits
    save_gui_fits_filetype='.eps'     # File format for saved fit plots
)

# --- Define CSV column names for the saved fitting results ---
data_columns = 'Qi_dia_corr, Qi_no_corr, absQc, Qc_dia_corr, Ql, fr, theta0, phi0, phi0_err, Ql_err, absQc_err, fr_err, chi_square, Qi_no_corr_err, Qi_dia_corr_err, prefactor_a, prefactor_alpha, baseline_slope, baseline_intercept, Power, Photon'

# --- Save the fit results to a CSV file ---
np.savetxt(f'{savepath}/results/{file_num}.csv', results, delimiter=',', header=data_columns)
