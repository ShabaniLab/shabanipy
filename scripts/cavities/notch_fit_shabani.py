from resonator_tools import circuit
import numpy as np
import csv
from os import path as pth
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


FREQ_COLUMN = 1
POWER_COLUMN = 0
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
#0: ('min', 500, 1e13),
#1: ('min', 500, 1e13),
2: ('min', 500, 1e13),
3: ('min', 500, 1e13),
#4: ('min', 1000, 1e13),
#5: ('min', 400, 1e13),
#6: ('min', 500, 1e13),
#7: ('min', 500, 1e13),
#8: ('min', 500, 1e13),
#9: ('min', 1000, 1e13),
#10: ('min', 400, 1e13),
    
    
    }
#put the path of all files you want to study and then put their associated attenuation below
path = '/Users/goss/Desktop/Shabani/data/short_stack/'
filename = ['JS314_CD1_att20_004', 'JS314_CD1_att40_006', 'JS314_CD1_att60_007'] #do not put file extension in filename
attenuation_on_vna = [-20,-40,-60]


res_freq_array = []

powerList0 = []
qList0 = []
qcList0= []
photonList0 = []

powerList1 = []
qList1 = []
qcList1= []
photonList1 = []

powerList2 = []
qList2 = []
qcList2= []
photonList2 = []

powerList3 = []
qList3 = []
qcList3= []
photonList3 = []

powerList4 = []
qList4 = []
qcList4= []
photonList4 = []

powerList5 = []
qList5 = []
qcList5= []
photonList5 = []

powerList = [powerList0,powerList1,powerList2, powerList3, powerList4, powerList5]
qList = [qList0,qList1,qList2,qList3, qcList4, qcList5]
qcList = [qcList0,qcList1,qcList2,qcList3,qcList4, qcList5]
photonList = [photonList0,photonList1,photonList2,photonList3, photonList4, photonList5]

j = 0
for FILENAME in filename:
    PATH = path + FILENAME + '.hdf5'
    print(PATH)
    ATTENUATION_ON_VNA = attenuation_on_vna[j]
    print(ATTENUATION_ON_VNA)
    j = j+1
    with LabberData(PATH) as data:
        print(data.list_channels())
        print(list(data._file['Traces']))
        shape = data.compute_shape((FREQ_COLUMN, POWER_COLUMN))
        shape = [shape[1], shape[0]] 
        powers = np.unique(data.get_data(POWER_COLUMN))
        print(powers)

    with h5py.File(PATH) as f:
        freq  = f['Traces']['VNA - S21'][:, 2].reshape([-1] + shape)
        real  = f['Traces']['VNA - S21'][:, 0].reshape([-1] + shape)
        imag  = f['Traces']['VNA - S21'][:, 1].reshape([-1] + shape)
        amp = np.abs(real + 1j*imag)
        freq  = f['Traces']['VNA - S21'][:, 2].reshape([-1] + shape)
        phase = np.arctan2(imag, real)

    for res_index, res_params in RESONANCE_PARAMETERS.items():
        
        if pth.exists(path + FILENAME + '-' + str(res_index) + '.csv'):
            with open(path + FILENAME + '-' + str(res_index) + '.csv', mode='a+') as data_base:
                data_base.seek(0)
                fieldnames = ['res_index', 'res_freq', 'attenuation', 'power', 'total_power','photon_num','qi','qc']
                csv_reader = csv.DictReader(data_base)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    powerList[res_index].append(float(row['total_power']))
                    qList[res_index].append(float(row['qi']))
                    qcList[res_index].append(float(row['qc']))
                    photonList[res_index].append(float(row['photon_num']))
                    res_freq_array.append(float(row['res_freq']))
                    line_count += 1
        else:
            #create a different csv file for each resonance within each .hdf5
            with open(path + FILENAME + '-' + str(res_index) + '.csv', mode='a+') as data_base:
                data_base.seek(0)
                fieldnames = ['res_index', 'res_freq', 'attenuation', 'power', 'total_power','photon_num','qi','qc']
                writer = csv.DictWriter(data_base, fieldnames=fieldnames)
                writer.writeheader()

                kind, e_delay, base_smooth = res_params
                for p_index, power in enumerate(powers):
                    
                    
                    f = freq[:, res_index, p_index]
                    a = amp[:, res_index, p_index]
                    phi = phase[:, res_index, p_index]
                    i = imag[:, res_index, p_index]
                    r = real[:, res_index, p_index]
                    phi = np.unwrap(phi)
                    
                    fc = estimate_central_frequency(f, a, kind)
                    
                    width = estimate_width(f, a, kind)
                    
                    indexes = slice(np.argmin(np.abs(f - fc + 10*width)),
                                    np.argmin(np.abs(f - fc - 10*width)))
                    f = f[indexes]
                    a = a[indexes]
                    phi = phi[indexes]
                    phi = np.unwrap(phi)
                    i = i[indexes]
                    r = r[indexes]

                    if SUBTRACT_BASELINE == True:
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
                    
                    if GUI_FITTING == True:
                        port1.GUIfit()
                        port1.plotall()
                    else:
                        port1.autofit()
                    
                    print("at power " + str(power) + " and " + str(ATTENUATION_ON_VNA) + " attenuation:")
                    powerList[res_index].append(float(calculate_total_power(fc,power)))
                    fit = pd.DataFrame([port1.fitresults]).applymap(lambda x: "{0:.2e}".format(x))
                    display(fit)
                    qList[res_index].append(float(fit.iat[0,1]))
                    qcList[res_index].append(float(fit.iat[0,0]))
                    
                    photonNum = port1.get_photons_in_resonator(calculate_total_power(fc,power),'dBm')
                    photonList[res_index].append(float(photonNum))
                    print("There are " + str(photonNum) + " photons in the cavity")
                    singlePhotonLim = port1.get_single_photon_limit()
                    print("Power for single photon limit: " + str(singlePhotonLim))
                    writer.writerow({'res_index': str(res_index), 'res_freq':str(fc), 'attenuation' : str(ATTENUATION_ON_VNA), 'power' : str(power), 'total_power': str(calculate_total_power(fc,power)),
                    'photon_num': str(photonNum),'qi':str(fit.iat[0,1]),'qc': str(fit.iat[0,0])})

                res_freq_array.append(fc)
                data_base.close()

if PLOT_QINT_VS_POWER == True:

    i = 0
    for res_index, res_params in RESONANCE_PARAMETERS.items():
        lists = list(zip(powerList[res_index],qList[res_index]))
        res = sorted(lists, key = lambda x: x[0]) 
        new_x, new_y = zip(*res)
        label_string = str('{:.3e}'.format(float(res_freq_array[i]))) +"Hz"
        plt.plot(new_x,new_y,".", label = label_string)
        i = i+1
    plt.title("Power Vs. Qi ")
    plt.xlabel("power [dB]")
    plt.ylabel("internal quality factor")
    plt.legend()
    plt.show()
    
if PLOT_PHOTON_VS_QINT == True:
    i = 0
    for res_index, res_params in RESONANCE_PARAMETERS.items():
        lists = list(zip(photonList[res_index],qList[res_index]))
        res = sorted(lists, key = lambda x: x[0]) 
        new_x, new_y = zip(*res)
        label_string = str('{:.3e}'.format(float(res_freq_array[i]))) +"Hz"
        plt.loglog(new_x,new_y,".", label = label_string)
        i = i+1
    plt.title("Photon Number Vs. Qi")
    plt.xlabel("photon number")
    plt.ylabel("internal quality factor")
    plt.legend()
    plt.show()

if PLOT_QC_VS_POWER == True:
    i = 0
    for res_index, res_params in RESONANCE_PARAMETERS.items():
        lists = list(zip(powerList[res_index],qcList[res_index]))
        res = sorted(lists, key = lambda x: x[0]) 
        new_x, new_y = zip(*res)
        label_string = str('{:.3e}'.format(float(res_freq_array[i]))) +"Hz"
        plt.plot(new_x,new_y,".", label = label_string)
        i = i+1
    plt.title("Power Vs. Qc ")
    plt.xlabel("power [dB]")
    plt.ylabel("coupled quality factor")
    plt.legend()
    plt.show()

    
    
if PLOT_PHOTON_VS_QC == True:
    i = 0
    for res_index, res_params in RESONANCE_PARAMETERS.items():
        lists = list(zip(photonList[res_index],qcList[res_index]))
        res = sorted(lists, key = lambda x: x[0]) 
        new_x, new_y = zip(*res)
        label_string = str('{:.3e}'.format(float(res_freq_array[i]))) +"Hz"
        plt.loglog(new_x,new_y,".", label = label_string)
        i = i+1
    plt.title("Photon Number Vs. Qc")
    plt.xlabel("photon number")
    plt.ylabel("coupled quality factor")
    plt.legend()
    plt.show()