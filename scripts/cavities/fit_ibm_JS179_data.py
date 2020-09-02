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

EXTERNAL_PATH = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper'
IBM_FOLDER = '20200710_NYU_XLD8B'

resonance_info = [
#    ('5.63GHz', [], False),
#    ('5.89GHz', [], True),
#    ('6.104GHz', [], True),
    ('7.933GHz', [], False),
]

freqs = []
skip  = []

resonance_folders = [i[0] for i in resonance_info]
exclude = [i[1] for i in resonance_info]
use_saved_results = [i[2] for i in resonance_info]

data = {}
for resonance_folder in resonance_folders:
    data_path = os.path.join(
        EXTERNAL_PATH,
        IBM_FOLDER,
        resonance_folder
    )
    
    for root, directory, files in os.walk(data_path):
        for f in files:
            if f.endswith('.csv'):
                sample_info = re.findall(r'(.*?)GHz_averaging_on_(.*?)dB_4001pts.csv',f)
                print(f,sample_info)
                freq, power = map(float,sample_info[0])
                freq = int(freq*1000)
                f,m,p = load_csv(os.path.join(data_path,f))
                p = np.deg2rad(p)
                m = from_dB(m)
                if freq not in data:
                    data[freq] = {
                        'power' : [],
                        'frequency' : [],
                        'complex' : []
                    }
                    freqs.append(freq)
                data[freq]['power'].append(power)
                data[freq]['frequency'].append(f)
                data[freq]['complex'].append(m*np.exp(1j*p))
        for k,v in data.items():
            for kk,vv in v.items():
                data[k][kk] = np.array(vv)

use_saved_results = {
    freq:flag for freq,flag in zip(freqs,use_saved_results)
}
exclude = {
    freq:np.array([i not in ex for i in range(len(data[freq]['power']))])
    for freq,ex in zip(freqs,exclude)
}

results = {}
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
    power = traces['power']
    if use_saved_results[freq]:
        results[freq] = loaded_results[freq]
    else:
        try:
            results[freq] = fit_complex(
                frequency,
                cdata,
                powers = power,
                gui_fit = True,
                save_gui_fits = True,
                save_gui_fits_path = os.path.join(
                    EXTERNAL_PATH,
                    IBM_FOLDER,
                    "fit_figures"
                ),
                save_gui_fits_prefix = f'{freq}',
                save_gui_fits_suffix = [f'{p}dB' for p in traces['power']],
                save_gui_fits_title = [f'{p}dB' for p in traces['power']],
            )
        except ZeroDivisionError:
            print(f'Error on {freq}')
            continue

with open(os.path.join(EXTERNAL_PATH,IBM_FOLDER,'pickle_jar','ibm_fit_results.pickle'),'wb') as f:
    pickle.dump(results,f)


markersize = 20
labelsize = 20
ticklabelsize = 16

Q_fig,Q_axs = plt.subplots(1,2,figsize=(16,8))

Qlims = [1,1e6]

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
    Q_axs[0].set_ylim(Qlims)
    Q_axs[0].set_yscale('log')
    Q_axs[0].tick_params(axis='both', labelsize=ticklabelsize)
    
    x = traces['power']
    y = results[freq][:,3]
    
    Q_axs[1].plot(x, y, '.', label=f'{freq/1000:.3f} GHz', ms=markersize)
    Q_axs[1].set_title('External Quality Factor', fontsize=labelsize)
    Q_axs[1].set_xlabel('Power [dB]', fontsize=labelsize)
    Q_axs[1].set_ylabel('$Q_{ext}$', fontsize=labelsize)
    Q_axs[1].legend(loc='best')
    Q_axs[1].set_ylim(Qlims)
    Q_axs[1].set_yscale('log')
    Q_axs[1].tick_params(axis='both', labelsize=ticklabelsize)
    
Q_fig.tight_layout()
Q_fig.savefig(
    os.path.join(
        EXTERNAL_PATH,
        IBM_FOLDER,
        "figures",
        f"All_q_factor_correction.png"
    )
)


for freq in freqs:
    Q_fig,Q_axs = plt.subplots(1,1,figsize=(8,8))
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
    
    x = traces['power']
    y = results[freq][:,0]
    
    Q_axs.plot(x, y, '.', label=labels[0], ms=markersize)
    
    y = results[freq][:,3]
    Q_axs.plot(x, y, '.', label=labels[3], ms=markersize)
    
    y = results[freq][:,4]
    Q_axs.plot(x, y, '.', label=labels[4], ms=markersize)
    
    Q_axs.set_title('Quality Factor (Correction)', fontsize=labelsize)
    Q_axs.set_xlabel('Power [dB]', fontsize=labelsize)
    Q_axs.set_ylabel('Q', fontsize=labelsize)
    Q_axs.legend(loc='best')
    Q_axs.set_ylim(Qlims)
    Q_axs.set_yscale('log')
    Q_axs.tick_params(axis='both', labelsize=ticklabelsize)
    
    
    Q_fig.tight_layout()
    Q_fig.savefig(
        os.path.join(
            EXTERNAL_PATH,
            IBM_FOLDER,
            "figures",
            f"{freq}_AllQsOnePlot_corr.png"
        )
    )

for freq in freqs:
    if freq in skip:
        continue
    Q_fig,Q_axs = plt.subplots(1,1,figsize=(8,8))
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
    
    x = traces['power']
    y = results[freq][:,1]
    
    Q_axs.plot(x, y, '.', label=labels[1], ms=markersize)
    
    y = results[freq][:,2]
    Q_axs.plot(x, y, '.', label=labels[2], ms=markersize)
    
    y = results[freq][:,4]
    Q_axs.plot(x, y, '.', label=labels[4], ms=markersize)
    
    
    Q_axs.set_title('Quality Factor (No Correction)', fontsize=labelsize)
    Q_axs.set_xlabel('Power [dB]', fontsize=labelsize)
    Q_axs.set_ylabel('Q', fontsize=labelsize)
    Q_axs.legend(loc='best')
    Q_axs.set_ylim(Qlims)
    Q_axs.set_yscale('log')
    Q_axs.tick_params(axis='both', labelsize=ticklabelsize)
    
    
    Q_fig.tight_layout()
    Q_fig.savefig(
        os.path.join(
            EXTERNAL_PATH,
            IBM_FOLDER,
            "figures",
            f"{freq}_AllQsOnePlot_no_corr.png"
        )
    )

plt.show()

