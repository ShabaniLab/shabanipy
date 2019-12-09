from resonator_tools import circuit
import numpy as np
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

PATH = '/Users/goss/Desktop/Shabani/data/IBM_Resonators/IBM_resonators_JY002_006.hdf5'
FREQ_COLUMN = 0
POWER_COLUMN = 1
ATTENUATION_ON_VNA = 0
PLOT_Q_VS_POWER = True
PLOT_PHOTON_VS_Q = True
GET_PHOTON_NUMBER = True
GET_SINGLE_PHOTON_LIMIT = True


def calculate_total_power(freq,power):
    totalPower = (-(1/800000000)*freq - (215/4)) + power + ATTENUATION_ON_VNA
    return totalPower

#uncommentresonance parameters line for 0th resonance, 1st resonance, and so on.
RESONANCE_PARAMETERS = {
#0: ('min', 51, 500, 1e13),
#1: ('min', 51, 500, 1e13),
#2: ('min', 51, 500, 1e13),
3: ('min', 51, 500, 1e13),
#4: ('max', 51, 1000, 0),
# 5: ('max', 51, 400, 1e13),
    }

with LabberData(PATH) as data:
    shape = data.compute_shape((FREQ_COLUMN, POWER_COLUMN))
    shape = [shape[1], shape[0]]
    powers = np.unique(data.get_data(POWER_COLUMN))

with h5py.File(PATH) as f:
    freq  = f['Traces']['VNA - S21'][:, 2].reshape([-1] + shape)
    real  = f['Traces']['VNA - S21'][:, 0].reshape([-1] + shape)
    imag  = f['Traces']['VNA - S21'][:, 1].reshape([-1] + shape)
    amp = np.abs(real + 1j*imag)
    phase = np.arctan2(imag, real)
    
for res_index, res_params in RESONANCE_PARAMETERS.items():
    
    powerList = []
    qList = []
    photonList = []
    kind, sav_f, e_delay, base_smooth = res_params
    for p_index, power in enumerate(powers):
        
        f = freq[:, p_index, res_index]
        a = amp[:, p_index, res_index]
        phi = phase[:, p_index, res_index]
        i = imag[:, p_index, res_index]
        r = real[:, p_index, res_index]
        phi = np.unwrap(phi)
        #slope = estimate_time_delay(f, phi,
        #                          e_delay, False)
        #phi = correct_for_time_delay(f, phi, slope)
        
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

        base = extract_baseline(a,
                                    0.8 if kind == 'min' else 0.2,
                                    base_smooth, plot=False)
        #New background subtraction code snippet
        #a /= (base/base[0])
        #mid = base[base.size//2]
        #base = base - mid
        #a = a + base
        #full = a*np.exp(1j*phi)
        #r = np.real(full)
        #i = np.imag(full)
        
        names = ['freq','real','imag','mag','phase']
        data = np.array([f,r,i,a,phi]).T
        df = pd.DataFrame(data = data, columns = names)
        #display(df.head())
        
        port1 = circuit.notch_port(f_data=df["freq"].values,
                                z_data_raw=(df["real"].values + 1j*df["imag"].values))
        
        port1.GUIfit()
        
        port1.plotall()
        
        print("at power " + str(power) + " and " + str(ATTENUATION_ON_VNA) + " attenuation:")
        powerList.append(float(power))
        fit = pd.DataFrame([port1.fitresults]).applymap(lambda x: "{0:.2e}".format(x))
        display(fit)
        qList.append(float(fit.iat[0,1]))
        
        if GET_PHOTON_NUMBER == True:
            photonNum = port1.get_photons_in_resonator(calculate_total_power(fc,power),'dBm')
            photonList.append(float(photonNum))
            print("There are " + str(photonNum) + " photons in the cavity")
        if GET_SINGLE_PHOTON_LIMIT == True:
            singlePhotonLim = port1.get_single_photon_limit()
            print("Power for single photon limit: " + str(singlePhotonLim))

    if PLOT_Q_VS_POWER == True:
        plt.plot(powerList,qList)
        plt.title("Power Vs. Qi for resonance at " +str(fc) +"Hz")
        plt.xlabel("power [dB]")
        plt.ylabel("internal quality factor")
        plt.show()
    
    if PLOT_PHOTON_VS_Q == True:
        plt.semilogx(photonList,qList)
        plt.title("Photon Number Vs. Qi for resonance at " + str(fc) +"Hz")
        plt.xlabel("photon number")
        plt.ylabel("internal quality factor")
        plt.show()
