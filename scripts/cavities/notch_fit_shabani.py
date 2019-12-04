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
PLOT_Q_VS_POWER = True

#uncommentresonance parameters line for 0th resonance, 1st resonance, and so on.
RESONANCE_PARAMETERS = {
0: ('min', 51, 500, 1e13, 'amplitude', 'nelder'),
#1: ('min', 51, 500, 1e13, 'amplitude', 'nelder'),
#2: ('min', 51, 500, 1e13, 'amplitude', 'nelder'),
#3: ('min', 51, 500, 1e13, 'amplitude', 'nelder'),
    #4: ('max', 51, 1000, 0, 'amplitude', 'leastsq'),
    # 5: ('max', 51, 400, 1e13, 'amplitude', 'leastsq'),
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
    kind, sav_f, e_delay, base_smooth, method, fit_method = res_params
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

        print("at power ")
        print(power)
        powerList.append(float(power))
        fit = pd.DataFrame([port1.fitresults]).applymap(lambda x: "{0:.2e}".format(x))
        display(fit)
        qList.append(float(fit.iat[0,1]))
        
    if PLOT_Q_VS_POWER == True:
        plt.plot(powerList,qList)
        plt.title("Power Vs. Qi")
        plt.show()
    
