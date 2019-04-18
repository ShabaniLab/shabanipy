import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel
from resonator_tools.circuit import notch_port

FOLDER = '/Users/mdartiailh/Labber/Data/2018/11/Data_1106'
FILE = 'JS179C_MD001_003.hdf5'

RANGE = slice(5002,7503)

with h5py.File(os.path.join(FOLDER, FILE)) as f:
    port1 = notch_port(f['Traces']['VNA - S21'][RANGE, 2, 0],
                      f['Traces']['VNA - S21'][RANGE, 0, 0] +
                      1j*f['Traces']['VNA - S21'][RANGE, 1, 0])

lam = 1000e6#smoothness
p = 0.98  #asymmetry
fitted_baseline = port1.fit_baseline_amp(port1.z_data_raw,lam,p,niter=10)
plt.plot(np.absolute(port1.z_data_raw))
plt.plot(fitted_baseline)
plt.show()

port1 = notch_port(port1.f_data, port1.z_data_raw/fitted_baseline/0.99)
# model = LorentzianModel()
# data = 1 - np.abs(port1.


port1.autofit()
print("Fit results:", port1.fitresults)
port1.plotall()
print("single photon limit:", port1.get_single_photon_limit(diacorr=True), "dBm")
print("photons in reso for input -140dBm:", port1.get_photons_in_resonator(-140,unit='dBm',diacorr=True), "photons")
print("done")
