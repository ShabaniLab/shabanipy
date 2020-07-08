import os
import re
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from shabanipy.cavities.notch_geometry import fit_complex
from shabanipy.cavities.utils import to_dB, from_dB

EXTERNAL_PATH = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper/'
IBM_FOLDER = 'IBM_Data'
DATA_FOLDER = "nyuresonatordataat18mk"

freqs = [0, 4317, 4383, 4498, 4640, 4976, 5217]

try:
    with open(os.path.join(EXTERNAL_PATH,IBM_FOLDER,'pickle_jar','ibm_fit_results.pickle'),'rb') as f:
        loaded_results = pickle.load(f)
except FileNotFoundError:
    loaded_results = {}
except EOFError:
    print('EOF on pickle')
    loaded_results = {}

for freq in data:
    for field in ['power','complex','frequency']:
        data[freq][field] = data[freq][field][exclude[freq]]

for freq in freqs:
    traces = data[freq]
    if freq in skip:
        if freq in loaded_results:
            results[freq] = loaded_results[freq]
        continue
    frequency = traces['frequency']
    cdata = traces['complex']
    if use_saved_results[freq]:
        results[freq] = loaded_results[freq]
    else:
        try:
            results[freq] = fit_complex(
                frequency,
                cdata,
                gui_fit = True,
                save_gui_fits = True,
                save_gui_fits_path = os.path.join(EXTERNAL_PATH,IBM_FOLDER,"fit_figures"),
                save_gui_fits_prefix = f'{freq}',
                save_gui_fits_suffix = [f'{p}dB' for p in traces['power']],
                save_gui_fits_title = [f'{p}dB' for p in traces['power']],
            )
        except ZeroDivisionError:
            print(f'Error on {freq}')
            continue

Q_fig,Q_axs = plt.subplots(1,2,figsize=(16,8))

markersize = 20
labelsize = 20
ticklabelsize = 16

for freq in freqs:
    if freq in skip:
        continue
    traces = data[freq]
    labels = [
        'Qi_dia_corr',      # 0
        'Qi_no_corr',       # 1
        'absQc',            # 2
        'Qc_dia_corr',      # 3
        'Ql',               # 4
        'fr',               # 5
        'theta0',           # 6
        'phi0',             # 7
        'phi0_err',         # 8
        'Ql_err',           # 9
        'absQc_err',        # 10
        'fr_err',           # 11
        'chi_square',       # 12
        'Qi_no_corr_err',   # 13
        'Qi_dia_corr_err',  # 14
    ]
    
    plot_individual = False
    if plot_individual:
        plot_quantities = [
            ( 0, (0,0)),
            ( 1, (1,0)),
            ( 2, (0,1)),
            ( 3, (1,1)),
            ( 4, (0,2)),
            (12, (1,2)),
        ]
        
        fig,axs = plt.subplots(2,3)
        for index,plot_coords in plot_quantities:
            r,c = plot_coords
            x = traces['power']
            y = results[freq][:,index]
            avg = np.average(y)
            
            axs[plot_coords].set_title(labels[index])
            axs[plot_coords].plot(x,y,'.')

        fig.tight_layout()
    
    x = traces['power']
    y = results[freq][:,0]
    
    Q_axs[0].plot(x, y, '.', label=f'{freq/1000:.3f} GHz', ms=markersize)
    Q_axs[0].set_title('Internal Quality Factor', fontsize=labelsize)
    Q_axs[0].set_xlabel('Power [dB]', fontsize=labelsize)
    Q_axs[0].set_ylabel('$Q_{int}$', fontsize=labelsize)
    Q_axs[0].legend(loc='best')
    Q_axs[0].tick_params(axis='both', labelsize=ticklabelsize)
    
    x = traces['power']
    y = results[freq][:,3]
    
    Q_axs[1].plot(x, y, '.', label=f'{freq/1000:.3f} GHz', ms=markersize)
    Q_axs[1].set_title('External Quality Factor', fontsize=labelsize)
    Q_axs[1].set_xlabel('Power [dB]', fontsize=labelsize)
    Q_axs[1].set_ylabel('$Q_{ext}$', fontsize=labelsize)
    Q_axs[1].legend(loc='best')
    Q_axs[1].tick_params(axis='both', labelsize=ticklabelsize)
    
Q_fig.tight_layout()
Q_fig.savefig(os.path.join(EXTERNAL_PATH,IBM_FOLDER,"figures",f"All_q_factor.png"))

#with open(os.path.join(EXTERNAL_PATH,IBM_FOLDER,'pickle_jar','ibm_fit_results.pickle'),'wb') as f:
#    pickle.dump(results,f)

plt.show()

