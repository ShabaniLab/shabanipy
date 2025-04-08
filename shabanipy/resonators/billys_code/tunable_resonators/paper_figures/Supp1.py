# Importing required libraries and functions
import sys
sys.path.append('/Users/billystrickland/Documents/code/resonators')

# Importing specific settings and functions for plotting and data handling
from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.helper import *
from shabanipy.jy_mpl_settings.colors import line_colors

import csv
import matplotlib.pyplot as plt
from shabanipy.shabanipy.labber import LabberData
import numpy as np
import os
import pandas as pd
from utils.proc_csv import proc_csv
import utils.resonator_functions as rf

# Setup for the experiment
################################## CHANGE HERE #####################################################
# Root directory for the data
root = '/Users/billystrickland/Documents/code/resonators/data/'

# Identifier for the data set
ID = '187'

# Indices for resonators
res_index = [0, 1]

# Error threshold for data processing
err_thresh = 1000

# Sample name and file directory
sample = 'JS626-4TR-Noconst-1-BSBHE-001'
file = root + sample + '/results/csvs/'

# Labels for the plots
labels = ['TR', 'Bare']

# Create a plot with specified size and settings
with plt.rc_context(jy_mpl_rc):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    
    # Loop over the resonator indices (res_index) for each data set
    for i in range(2):
        # Files corresponding to each resonator's data
        FILES = [
            file + '187res_' + str(res_index[i]) + '.csv',
            file + '189res_' + str(res_index[i]) + '.csv',
            file + '191res_' + str(res_index[i]) + '.csv',
        ]
        
        # Process the CSV files and extract relevant results
        results = rf.proc_csv(FILES)            
        photon, power, qi_diacorr, qi_diacorr_err, qc, qc_err, ql, ql_err, freq, freq_err = rf.get_results(results, err_thresh)
        
        # Plot the data with error bars
        ax.errorbar(photon, qi_diacorr, yerr=qi_diacorr_err,
                    marker='.', linestyle='None', label=labels[i])
        
        # Labeling the axes and setting the scales
        ax.set_ylabel('$Q_{int}$')  # Y-axis label
        ax.set_xscale('log')        # X-axis scale (logarithmic)
        ax.set_xlabel('$<n_{photon}>$')  # X-axis label
        ax.set_yscale('log')        # Y-axis scale (logarithmic)
        
        # Adjust layout for tight fitting
        fig.tight_layout()
        
        # Print the Q_int value and error at a specific photon value
        print('qint of ' + labels[i] + ' at ' + str(photon[-7]) + ' photons is ' + str(qi_diacorr[-7]) + '±' + str(qi_diacorr_err[-7]))
    
    # Save the figure in multiple formats (EPS and PNG)
    fig.savefig(root + sample + '/results/q_' + ID + '.eps', transparent=True)
    fig.savefig(root + sample + '/results/q_' + ID + '.png')
    
    # Show the legend and display the plot
    plt.legend()
    plt.show()
