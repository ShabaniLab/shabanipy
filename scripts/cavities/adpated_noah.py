from resonator_tools import circuit
import numpy as np
import csv
from os import path as pth
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import h5py
from scipy.signal import savgol_filter
from shabanipy.utils.labber_io import LabberData
import matplotlib.pyplot as plt
from shabanipy.cavities.utils import (extract_baseline,
                                      estimate_central_frequency,
                                      estimate_width)

from shabanipy.cavities.notch_geometry import fit_complex



FREQ_COLUMN = 0
POWER_COLUMN = 1
PLOT_SUBTRACTED_BASELINE = False
PLOT_QINT_VS_POWER = True
PLOT_QC_VS_POWER = True
PLOT_PHOTON_VS_QC = True
PLOT_PHOTON_VS_QINT = True
GUI_FITTING = True

FIELDNAMES = [
    'res_index',
    'res_freq',
    'attenuation',
    'power',
    'total_power',
    'photon_num',
    'qi',
    'qi_err',
    'qc',
    'ql',
    'ql_err',
    'chi_square'
]

def calculate_total_power(freq,power):
    totalPower = (-(1/800000000)*freq - (215/4)) + power + ATTENUATION_ON_VNA
    return totalPower

# uncommentresonance parameters line for 0th resonance, 1st resonance, and so on.
RESONANCE_PARAMETERS = {
    0: ('min', 500, 1e13),
    #1: ('min', 500, 1e13),
    #2: ('min', 500, 1e13),
    #3: ('min', 500, 1e13),
    #4: ('min', 1000, 1e13),
    #5: ('min', 400, 1e13),
    #6: ('min', 500, 1e13),
    #7: ('min', 500, 1e13),
    #8: ('min', 500, 1e13),
    #9: ('min', 1000, 1e13),
    #10: ('min', 400, 1e13),
}
#put the path of all files you want to study and then put their associated attenuation below

SAMPLE = 'JS200'
BASEPATH = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper'

data_path = pth.join(BASEPATH,'data')
csv_directory = pth.join(BASEPATH,'fits',SAMPLE)
IMAGE_DIR = pth.join(BASEPATH,'images',SAMPLE)

filename = ['JS200_JY001_008']
#filename = ['JS314_CD1_att20_004', 'JS314_CD1_att40_006', 'JS314_CD1_att60_007']

# Strip off '.hdf5' in case file extensions are left on
filename = [i.rstrip('.hdf5') for i in filename] 

attenuation_on_vna = [0]

res_freq_array = []

powerList = {}
qList = {}
qlList = {}
qcList = {}
photonList = {}
qListErr = {}
qlListErr = {}
chiSquare = {}

csv_column_names = [
    'res_index',
    'res_freq',
    'attenuation',
    'power',
    'total_power',
    'photon_num',
    'qi',
    'qi_err',
    'qc',
    'ql',
    'ql_err',
    'chi_square',
]

j = 0
for FILENAME in filename:
    PATH = pth.join(data_path, FILENAME + '.hdf5')
    ATTENUATION_ON_VNA = attenuation_on_vna[j]
    j = j+1
    
    print(PATH)
    with LabberData(PATH) as data:
        shape = data.compute_shape((FREQ_COLUMN, POWER_COLUMN))
        shape = [shape[1], shape[0]] 
        powers_rev = np.unique(data.get_data(POWER_COLUMN))
        powers = np.flip(powers_rev)

    with h5py.File(PATH) as f:
        freq  = f['Traces']['VNA - S21'][:, 2].reshape([-1] + shape)
        real  = f['Traces']['VNA - S21'][:, 0].reshape([-1] + shape)
        imag  = f['Traces']['VNA - S21'][:, 1].reshape([-1] + shape)
        amp = np.abs(real + 1j*imag)
        freq  = f['Traces']['VNA - S21'][:, 2].reshape([-1] + shape)
        phase = np.arctan2(imag, real)

    for res_index, res_params in RESONANCE_PARAMETERS.items():
        res_result_path = pth.join(csv_directory,FILENAME + '-' + str(res_index) + '.csv')
        if False and pth.exists(res_result_path):
            with open(res_result_path, mode='a+') as data_base:
                data_base.seek(0)
                csv_reader = csv.DictReader(data_base)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    if res_index not in powerList:
                        powerList[res_index] = []
                        qList[res_index] = []
                        qListErr[res_index] = []
                        qcList[res_index] = []
                        qlList[res_index] = []
                        qlListErr[res_index] = []
                        photonList[res_index] = []
                        chiSquare[res_index] = []
                    powerList[res_index].append(float(row['total_power']))
                    qList[res_index].append(float(row['qi']))
                    qListErr[res_index].append(float(row['qi_err']))
                    qcList[res_index].append(float(row['qc']))
                    qlList[res_index].append(float(row['ql']))
                    qlListErr[res_index].append(float(row['ql_err']))
                    photonList[res_index].append(float(row['photon_num']))
                    chiSquare[res_index].append(float(row['chi_square']))
                    line_count += 1
                res_freq_array.append(float(row['res_freq']))
        else:
            #create a different csv file for each resonance within each .hdf5
            with open(res_result_path, mode='a+') as data_base:
                data_base.seek(0)
                
                writer = csv.DictWriter(data_base, fieldnames=FIELDNAMES)
                writer.writeheader()

                kind, e_delay, base_smooth = res_params
                for p_index, power in enumerate(powers):
                    f = freq[:, res_index, p_index]
                    i = imag[:, res_index, p_index]
                    r = real[:, res_index, p_index]
                    z = r + 1j*i
                    a = np.absolute(z)
                    
                    fc = estimate_central_frequency(f, a, kind)
                    
                    result = fit_complex(
                        f,
                        z,
                        powers=calculate_total_power(fc,power),
                        gui_fit=GUI_FITTING
                    )
                    
                    if res_index not in powerList:
                        powerList[res_index] = []
                        qList[res_index] = []
                        qListErr[res_index] = []
                        qcList[res_index] = []
                        qlList[res_index] = []
                        qlListErr[res_index] = []
                        photonList[res_index] = []
                        chiSquare[res_index] = []
                    
                    print(
                        "at power " + str(power) + " and " +
                        str(ATTENUATION_ON_VNA) + " attenuation:"
                    )
                    powerList[res_index].append(
                        float(calculate_total_power(fc,power))
                    )
                    fit = pd.DataFrame([result]).applymap(
                        lambda x: "{0:.2e}".format(x)
                    )
                    display(fit)
                    qcList[res_index].append(float(fit.iat[0,0]))
                    qList[res_index].append(float(fit.iat[0,1]))
                    qListErr[res_index].append(float(fit.iat[0,2]))
                    qlList[res_index].append(float(fit.iat[0,5]))
                    qlListErr[res_index].append(float(fit.iat[0,6]))
                    chiSquare[res_index].append(float(fit.iat[0,9]))
                    photonList[res_index].append(float(fit.iat))
                    
                    print("There are " + str(photonNum) + " photons in the cavity")
                    singlePhotonLim = port1.get_single_photon_limit()
                    print("Power for single photon limit: " + str(singlePhotonLim))
                    writer.writerow({
                            'res_index': str(res_index), 
                            'res_freq':str(fc),
                            'attenuation' : str(ATTENUATION_ON_VNA),
                            'power' : str(power),
                            'total_power': str(calculate_total_power(fc,power)),
                            'photon_num': str(photonNum),
                            'qi':str(fit.iat[0,1]),
                            'qi_err': str(fit.iat[0,2]),
                            'qc': str(fit.iat[0,0]),
                            'ql': str(fit.iat[0,5]),
                            'ql_err': str(fit.iat[0,6]),
                            'chi_square': str(fit.iat[0,9])
                    })
                res_freq_array.append(fc)
                data_base.close()

markersize   = 15
labelsize    = 16
tickfontsize = 12
legendsize   = 16

bottom = .12
top    = .93
right  = .98
left   = .15

plot_info = [
    [PLOT_QINT_VS_POWER, "Power Vs. Q$_i$", "linear", "log", "Power [dB]",
        "Internal Quality Factor", powerList, qList, chiSquare, '_Qi'],
    [PLOT_PHOTON_VS_QINT, "Photon Number Vs. Q$_i$", "log", "log", "Photon Number",
        "Internal Quality Factor", photonList, qList, chiSquare, '_Qi_photon'],
    [PLOT_QC_VS_POWER, "Power Vs. Q$_C$", "linear", "log", "Power [dB]",
        "Coupled Quality Factor", powerList, qcList, chiSquare, '_Qc'],
    [PLOT_PHOTON_VS_QC, "Photon Number Vs. Q$_C$", "log", "log", "Photon Number",
        "Coupled Quality Factor", photonList, qcList, chiSquare, '_Qc_photon'],
]

def moving_average(values,index,pts):
    if pts%2 == 0:
        step = pts//2
    else:
        step = (pts-1)//2
    L = max(0, index - step)
    R = max(len(values)-1, index + step)
    return np.average(values[L:R])

# Work in progress

def filter_points(xdata,ydata,chi2,box_average_pts=3,filter_multiple=10):
    return xdata,ydata
    filtered_x,filtered_y = [],[]
    for i in range(len(xdata)):
        avg_q = moving_average(ydata,i,box_average_pts)
        if avg_q*filter_multiple > ydata[i] > avg_q/filter_multiple:
            filtered_x.append(xdata[i])
            filtered_y.append(ydata[i])
    return filtered_x,filtered_y
        
chi2_cutoff = .1

for (plot_check, title, xscale, yscale, xlabel, ylabel, xdata, ydata, chi2,
        filename_suffix) in plot_info:
    if plot_check:
        i = 0
        f,ax = plt.subplots(1,1)
        for res_index, res_params in RESONANCE_PARAMETERS.items():
            lists = list(zip(xdata[res_index],ydata[res_index],chi2[res_index]))
            res = sorted(lists, key = lambda x: x[0])
            new_x, new_y = filter_points(*zip(*res),box_average_pts=5)
            label_string = '$f_r$ = ' + f'{res_freq_array[i]/1e9:.3f} GHz'
            ax.plot(new_x,new_y,".",label=label_string,markersize=markersize)
            i += 1
        ax.set_title(title,fontsize=labelsize)
        ax.set_xlabel(xlabel,fontsize=labelsize)
        ax.set_ylabel(ylabel,fontsize=labelsize)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        
        for _ax in [ax.xaxis,ax.yaxis]:
            for tick in _ax.get_major_ticks():
                tick.label.set_fontsize(tickfontsize)
        
        ax.legend(fontsize=legendsize)
        plt.subplots_adjust(
            bottom=bottom,
            right=right,
            top=top,
            left=left
        )
        f.savefig(pth.join(IMAGE_DIR,FILENAME + filename_suffix+'.png'))

plt.show()
