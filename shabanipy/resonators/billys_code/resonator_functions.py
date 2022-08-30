import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
from shabanipy.resonators.notch_geometry import fit_complex, notch_from_results
from shabanipy.labber import LabberData
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from shabanipy.jy_mpl_settings.settings import jy_mpl_rc
from shabanipy.jy_mpl_settings.colors import line_colors
from shabanipy.jy_mpl_settings.helper import *
from quasiparticleFunctions import expectedShift
from matplotlib.pyplot import cm
import scipy
from scipy import signal

def proc_csv(FILES):
    results = []
    for f in FILES:
        with open(f, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                results.append(list(map(float,row)))
    return results

def get_results(results, err_thresh):
    err = [r[14] for i, r in enumerate(results)]        ##### thresholding against r[14] (qi_diacorr_err)
    photon = [r[-1] for i, r in enumerate(results) if err[i]<err_thresh]
    power = [r[-2] for i, r in enumerate(results) if err[i]<err_thresh]
    qi_diacorr = [r[0] for i, r in enumerate(results) if err[i]<err_thresh]
    qi_diacorr_err = [r[14] for i, r in enumerate(results) if err[i]<err_thresh]
    qi = [r[1] for i, r in enumerate(results) if err[i]<err_thresh]
    qi_err = [r[13] for i, r in enumerate(results) if err[i]<err_thresh]
    qc = [r[3] for i, r in enumerate(results) if err[i]<err_thresh]
    qc_err = [r[10] for i, r in enumerate(results) if err[i]<err_thresh]
    ql = [r[4] for i, r in enumerate(results) if err[i]<err_thresh]
    ql_err = [r[9] for i, r in enumerate(results) if err[i]<err_thresh]
    freq = [r[5] for i, r in enumerate(results) if err[i]<err_thresh]
    freq_err = [r[11] for i, r in enumerate(results) if err[i]<err_thresh]
    return photon, power, qi_diacorr, qi_diacorr_err, qc, qc_err, ql, ql_err, freq, freq_err

def f_to_l(freq, capacitance):
    return np.array([((2*math.pi*x)**2*capacitance)**(-1) for x in freq])

def f_to_lj(freq, capacitance, lest):
    return np.array([((2*math.pi*x)**2*capacitance)**(-1)-lest for x in freq])

def alpha(freq_geo, freq_meas):
    return 1-(freq_meas/freq_geo)**2

def lj_to_ic(lj):
    mfq = 2.068*10**(-15)
    return [mfq/(2*math.pi*lj) for r in lj]


