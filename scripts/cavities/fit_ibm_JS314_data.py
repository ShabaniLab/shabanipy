import os
import re
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from shabanipy.cavities.notch_geometry import fit_complex
from shabanipy.cavities.utils import to_dB, from_dB


def load_csv(filepath):
    f,m,p = [],[],[]
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            _,_f,_m,_p = map(float,row)
            f.append(_f)
            m.append(_m)
            p.append(_p)
    return map(np.array,(f,m,p))

EXTERNAL_PATH = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper/'
IBM_FOLDER = 'IBM_Data'
DATA_FOLDER = "nyuresonatordataat18mk"

data_path = os.path.join(EXTERNAL_PATH,IBM_FOLDER,DATA_FOLDER)

data = {}
for root, directory, files in os.walk(data_path):
    for f in files:
        if f.endswith('.csv'):
            if f.startswith('pat_resonator_averaging'):
                sample_info = re.findall(r'pat_resonator_averaging_on_(.*?).csv',f)
                freq = 0
                power = int(sample_info[0])
            else:
                sample_info = re.findall(r'pat_resonator_(.*?)MHz_averaging_on_(.*?).csv',f)
                freq, power = map(int,sample_info[0])
            f,m,p = load_csv(os.path.join(data_path,f))
            p = np.deg2rad(p)
            m = from_dB(m)
            if freq not in data:
                data[freq] = {
                    'power' : [],
                    'frequency' : [],
                    'complex' : []
                }
            data[freq]['power'].append(power)
            data[freq]['frequency'].append(f)
            data[freq]['complex'].append(m*np.exp(1j*p))
    for k,v in data.items():
        for kk,vv in v.items():
            data[k][kk] = np.array(vv)


results = {}
freqs = [0, 4317, 4383, 4498, 4640, 4976, 5217]
skip  = [0]

exclude = [
    [], # 0 
    [0, 1, 2, 5, 7], # 4317
    [0, 1, 2, 3, 4], # 4383
    [0, 1, 4, 7], # 4498
    [0, 1, 2], # 4640
    [2, 3], # 4976
    [], # 5217
]
exclude = {
    freq:np.array([i not in ex for i in range(len(data[freq]['power']))])
    for freq,ex in zip(freqs,exclude)
}

use_saved_results = [
    False, # 0 
    True,  # 4317
    True,  # 4383
    True,  # 4498
    True,  # 4640
    True,  # 4976
    False, # 5217
]
use_saved_results = {
    freq:flag for freq,flag in zip(freqs,use_saved_results)
}
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

