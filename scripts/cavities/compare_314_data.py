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

data_path = os.path.join(EXTERNAL_PATH,IBM_FOLDER,DATA_FOLDER)

with open(os.path.join(EXTERNAL_PATH,IBM_FOLDER,'pickle_jar','ibm_fit_results.pickle'),'rb') as f:
    ibm_results = pickle.load(f)

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

freqs = [0, 4317, 4383, 4498, 4640, 4976, 5217]
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

for freq in data:
    for field in ['power','complex','frequency']:
        data[freq][field] = data[freq][field][exclude[freq]]

DATAPATH = os.path.join(EXTERNAL_PATH,'data')
CSV_PATH = os.path.join(EXTERNAL_PATH,'fits')

sample_info = {
    'JS314' : {
        'name'    : 'Lattice Matched (Short)',
        'indices' : [0, 1, 2, 3, 4, 5, 6],
        'files'   : ['JS314_CD1_att20_004', 'JS314_CD1_att40_006', 'JS314_CD1_att60_007'],
        'att'     : [-20, -40, -60]
    },
}

res_freq_array = []

samples = {sample : {} for sample in sample_info}

j = 0
for sample in sample_info:
    j = j+1
    for res_index in sample_info[sample]['indices']:
        if res_index not in samples[sample]:
            samples[sample][res_index] = {
                'power'  : [],
                'qi'     : [],
                'qi_err' : [],
                'ql'     : [],
                'ql_err' : [],
                'qc'     : [],
                'chi_square' : [],
                'photon_num' : [],
                'res_freq'   : [],
            }
        for filename in sample_info[sample]['files']:
            res_result_path = os.path.join(CSV_PATH,sample,filename + '-' + str(res_index) + '.csv')
            if os.path.exists(res_result_path):
                with open(res_result_path, mode='r') as f:
                    fieldnames = ['res_index', 'res_freq', 'attenuation', 'power', 'total_power','photon_num','qi','qi_err','qc','ql','ql_err','chi_square']
                    csv_reader = csv.DictReader(f)
                    next(csv_reader, None)  # skip the headers
                    for row in csv_reader:
                        samples[sample][res_index]['power'].append(float(row['total_power']))
                        samples[sample][res_index]['qi'].append(float(row['qi']))
                        samples[sample][res_index]['qi_err'].append(float(row['qi_err']))
                        samples[sample][res_index]['qc'].append(float(row['qc']))
                        samples[sample][res_index]['ql'].append(float(row['ql']))
                        samples[sample][res_index]['ql_err'].append(float(row['ql_err']))
                        samples[sample][res_index]['photon_num'].append(float(row['photon_num']))
                        samples[sample][res_index]['chi_square'].append(float(row['chi_square']))
                    samples[sample][res_index]['res_freq'] = float(row['res_freq'])

markersize = 20
labelsize = 20
ticklabelsize = 16

for res_index in samples['JS314']:
    print(samples['JS314'][res_index]['res_freq'])

data_pairs = [
    (1,0),
    (2,1),
    (3,2),
    (4,3),
]

for a,b in data_pairs:
    fig,axs = plt.subplots(1,2,figsize=(16,8))
    
    ibm_fr = list(ibm_results.keys())[a]
    our_fr = samples['JS314'][b]['res_freq']
   
    x = data[ibm_fr]['power']
    y = ibm_results[ibm_fr][:,0]
    
    axs[0].plot(x, y, '.', label='IBM $f_r = $' + f'{ibm_fr/1e3:.3f} GHz', ms=markersize)
    
    x = np.array(samples['JS314'][b]['power']) + 60
    y = np.array(samples['JS314'][b]['qi'])
    
    mask = np.where(np.abs(y) < 1e6)
    x = x[mask]
    y = y[mask]
    
    axs[0].plot(x, y, '.', label='JS $f_r = $' + f'{our_fr/1e9:.3f} GHz', ms=markersize)
    
    axs[0].set_title('Internal Quality Factor', fontsize=labelsize)
    axs[0].set_xlabel('Power [dB]', fontsize=labelsize)
    axs[0].set_ylabel('$Q_{int}$', fontsize=labelsize)
    axs[0].legend(loc='best')
    axs[0].tick_params(axis='both', labelsize=ticklabelsize)
    
    x = data[ibm_fr]['power']
    y = ibm_results[ibm_fr][:,3]
    
    axs[1].plot(x, y, '.', label='IBM $f_r = $' + f'{ibm_fr/1e3:.3f} GHz', ms=markersize)
    
    x = np.array(samples['JS314'][b]['power']) + 60
    y = np.array(samples['JS314'][b]['qc'])
    
    mask = np.where(np.abs(y) < 1e6)
    x = x[mask]
    y = y[mask]
    
    axs[1].plot(x, y, '.', label='JS $f_r = $' + f'{our_fr/1e9:.3f} GHz', ms=markersize)

    
    axs[1].set_title('External Quality Factor', fontsize=labelsize)
    axs[1].set_xlabel('Power [dB]', fontsize=labelsize)
    axs[1].set_ylabel('$Q_{ext}$', fontsize=labelsize)
    axs[1].legend(loc='best')
    axs[1].tick_params(axis='both', labelsize=ticklabelsize)

    fig.tight_layout()

    fig.savefig(os.path.join(EXTERNAL_PATH,IBM_FOLDER,"figures",f"compare_IBM{ibm_fr}_JS{our_fr//1e6}.png"))
plt.show()

