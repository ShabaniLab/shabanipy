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
                                      estimate_peak_amplitude,
                                      estimate_width,
                                      estimate_time_delay,
                                      correct_for_time_delay)


FREQ_COLUMN = 0
POWER_COLUMN = 1
SUBTRACT_BASELINE = True
PLOT_SUBTRACTED_BASELINE = False
PLOT_QINT_VS_POWER = True
PLOT_QC_VS_POWER = True
PLOT_PHOTON_VS_QC = True
PLOT_PHOTON_VS_QINT = True
GUI_FITTING = True


def calculate_total_power(freq,power):
    totalPower = (-(1/800000000)*freq - (215/4)) + power + ATTENUATION_ON_VNA
    return totalPower

#uncommentresonance parameters line for 0th resonance, 1st resonance, and so on.
RESONANCE_PARAMETERS = {
    0: ('min', 500, 1e13),
    1: ('min', 500, 1e13),
    2: ('min', 500, 1e13),
    3: ('min', 500, 1e13),
    4: ('min', 1000, 1e13),
    #5: ('min', 400, 1e13),
    #6: ('min', 500, 1e13),
    #7: ('min', 500, 1e13),
    #8: ('min', 500, 1e13),
    #9: ('min', 1000, 1e13),
    #10: ('min', 400, 1e13),
}
#put the path of all files you want to study and then put their associated attenuation below
path = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper/data/'
csv_directory = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper/fits/JS314/'

IMAGE_DIR = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper/images/JS314/'

filename = ['JS314_CD1_att20_004', 'JS314_CD1_att40_006', 'JS314_CD1_att60_007']
attenuation_on_vna = [-20,-40,-60]

res_freq_array = []

powerList = {}
qList = {}
qlList = {}
qcList = {}
photonList = {}
qListErr = {}
qlListErr = {}
chiSquare = {}

j = 0
for FILENAME in filename:
    PATH = path + FILENAME + '.hdf5'
    ATTENUATION_ON_VNA = attenuation_on_vna[j]
    j = j+1
    
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
        if pth.exists(csv_directory + FILENAME + '-' + str(res_index) + '.csv'):
            with open(csv_directory + FILENAME + '-' + str(res_index) + '.csv', mode='a+') as data_base:
                data_base.seek(0)
                fieldnames = ['res_index', 'res_freq', 'attenuation', 'power', 'total_power','photon_num','qi','qi_err','qc','ql','ql_err','chi_square']
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
            with open(csv_directory + FILENAME + '-' + str(res_index) + '.csv', mode='a+') as data_base:
                data_base.seek(0)
                fieldnames = ['res_index', 'res_freq', 'attenuation', 'power', 'total_power','photon_num','qi','qi_err','qc','ql','ql_err','chi_square']
                writer = csv.DictWriter(data_base, fieldnames=fieldnames)
                writer.writeheader()

                kind, e_delay, base_smooth = res_params
                for p_index, power in enumerate(powers):
                    f   = freq[:, res_index, p_index]
                    a   = amp[:, res_index, p_index]
                    i   = imag[:, res_index, p_index]
                    r   = real[:, res_index, p_index]
                    phi = phase[:, res_index, p_index]
                    phi = np.unwrap(phi)
                    fc  = estimate_central_frequency(f, a, kind)
                    
                    width = estimate_width(f, a, kind)
                    
                    indexes = slice(np.argmin(np.abs(f - fc + 10*width)),
                                    np.argmin(np.abs(f - fc - 10*width)))
                    f = f[indexes]
                    a = a[indexes]
                    phi = phi[indexes]
                    phi = np.unwrap(phi)
                    i = i[indexes]
                    r = r[indexes]

                    if SUBTRACT_BASELINE:
                        base = extract_baseline(a,
                                                0.8 if kind == 'min' else 0.2,
                                                base_smooth, plot=False)
                        a_init = a
                        a /= (base/base[0])
                        a_aft = a
                        mid = base[base.size//2]
                        base_init = base
                        base = base - mid
                        a = a + base
                        full = a*np.exp(1j*phi)
                        r = np.real(full)
                        i = np.imag(full)
                        if PLOT_SUBTRACTED_BASELINE == True:
                            fig, axes = plt.subplots(1, 2, sharex=True)
                            axes[0].plot(f, a_init, '+')
                            axes[0].plot(f, base_init)
                            axes[1].plot(f, a_init, '+')
                            axes[1].plot(f, np.absolute(full))
                            plt.show()
                    
                    names = ['freq','real','imag','mag','phase']
                    data = np.array([f,r,i,a,phi]).T
                    df = pd.DataFrame(data = data, columns = names)
                    #display(df.head())
                    
                    port1 = circuit.notch_port(f_data=df["freq"].values,
                                            z_data_raw=(df["real"].values + 1j*df["imag"].values))
                    
                    if GUI_FITTING:
                        port1.GUIfit()
                        port1.plotall()
                    else:
                        port1.autofit()
                    
                    if res_index not in powerList:
                        powerList[res_index] = []
                        qList[res_index] = []
                        qListErr[res_index] = []
                        qcList[res_index] = []
                        qlList[res_index] = []
                        qlListErr[res_index] = []
                        photonList[res_index] = []
                        chiSquare[res_index] = []
                    
                    print("at power " + str(power) + " and " + str(ATTENUATION_ON_VNA) + " attenuation:")
                    powerList[res_index].append(float(calculate_total_power(fc,power)))
                    fit = pd.DataFrame([port1.fitresults]).applymap(lambda x: "{0:.2e}".format(x))
                    display(fit)
                    qList[res_index].append(float(fit.iat[0,1]))
                    qListErr[res_index].append(float(fit.iat[0,2]))
                    qcList[res_index].append(float(fit.iat[0,0]))
                    qlList[res_index].append(float(fit.iat[0,5]))
                    qlListErr[res_index].append(float(fit.iat[0,6]))
                    chiSquare[res_index].append(float(fit.iat[0,9]))
                    
                    photonNum = port1.get_photons_in_resonator(calculate_total_power(fc,power),'dBm')
                    photonList[res_index].append(float(photonNum))
                    print("There are " + str(photonNum) + " photons in the cavity")
                    singlePhotonLim = port1.get_single_photon_limit()
                    print("Power for single photon limit: " + str(singlePhotonLim))
                    writer.writerow({'res_index': str(res_index), 'res_freq':str(fc), 'attenuation' : str(ATTENUATION_ON_VNA), 'power' : str(power), 'total_power': str(calculate_total_power(fc,power)),
                    'photon_num': str(photonNum),'qi':str(fit.iat[0,1]),'qi_err': str(fit.iat[0,2]),'qc': str(fit.iat[0,0]), 'ql': str(fit.iat[0,5]), 'ql_err': str(fit.iat[0,6]), 'chi_square': str(fit.iat[0,9])})

                res_freq_array.append(fc)
                data_base.close()

markersize   = 20
labelsize    = 16
tickfontsize = 12
legendsize   = 16

bottom = .12
top    = .93
right  = .98
left   = .15


plot_info = [
    [PLOT_QINT_VS_POWER, "Power Vs. Q$_i$", "linear", "log", "Power [dB]",
        "Internal Quality Factor", powerList, qList, '_Qi'],
    [PLOT_PHOTON_VS_QINT, "Photon Number Vs. Q$_i$", "log", "log", "Photon Number",
        "Internal Quality Factor", photonList, qList, '_Qi_photon'],
    [PLOT_QC_VS_POWER, "Power Vs. Q$_C$", "linear", "log", "Power [dB]",
        "Coupled Quality Factor", powerList, qcList, '_Qc'],
    [PLOT_PHOTON_VS_QC, "Photon Number Vs. Q$_C$", "log", "log", "Photon Number",
        "Coupled Quality Factor", photonList, qcList, '_Qc_photon'],
]

for plot_check, title, xscale, yscale, xlabel, ylabel, xdata, ydata, filename_suffix in plot_info:
    if plot_check:
        i = 0
        f,ax = plt.subplots(1,1)
        for res_index, res_params in RESONANCE_PARAMETERS.items():
            lists = list(zip(xdata[res_index],ydata[res_index]))
            res = sorted(lists, key = lambda x: x[0]) 
            new_x, new_y = zip(*res)
            label_string = f'{res_freq_array[i]/1e9:.3f} GHz'
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

