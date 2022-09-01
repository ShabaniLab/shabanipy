# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Tools to analyse cavity in the notch geometry.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

def FluxFunc_ko1(fl,q0):
    return (1+q0*np.sin(np.pi*fl/2)*np.arctanh(np.sin(np.pi*fl/2))/(1-np.sin(np.pi*fl/2)*np.arctanh(np.sin(np.pi*fl/2))))**(-0.5)

def multiFluxFunc_ko1(fl,w0,q0,fl0,fs):
    return w0*(1+q0*np.sin(np.pi*(fl-fl0)/fs/2)*np.arctanh(np.sin(np.pi*(fl-fl0)/fs/2))/(1-np.sin(np.pi*(fl-fl0)/fs/2)*np.arctanh(np.sin(np.pi*(fl-fl0)/fs/2))))**(-0.5)
    
def multiFluxFunc_general(fl,w0,q0,fl0,fs):
    return 

def qpp_expectedShift(q0, f0, ls, phi, delta=170e-6):

    '''Returns the maximum expected shift (tau = 1) due to single quasiparticle

        trapping



    returns df in linear frequency

    -------------------------------

    q0:     participation ratio

    f0:     zero qp resonance at the given flux [in Hz]

    ls:     the Josephson current of nanosquid (Lj/2)

    phi:    reduced flux -- where you're monitoring

    delta:  superconducting gap in eV

    '''

    phi0 = hbar/2/e

    return (-q0 * f0 * ls * delta * e *

            (np.cos(pi * phi) + (np.sin(pi * phi / 2)**4)) / 8 /

            (phi0**2) / (1 - (np.sin(pi * phi / 2)**2)))





def f_n_phi(phi, n, L=1.82897e-9, C=0.739929e-12, Ls=20.8475e-12,

            Delta=2.72370016e-23):

    '''Returns the expected frequency with n trapped quasiparticles



    returns f(phi,n) in linear frequency

    -------------------------------

    phi:    reduced flux -- where you're monitoring

    n:      number of trapped quasiparticles

    L:      linear inductance

    C:      linear capacitance

    Ls:     squid inductance at 0 flux

    Delta:  superconducting gap in J

    '''

    de = np.pi*phi

    Lsphi = Ls/(1-np.sin(de/2)*np.arctanh(np.sin(de/2)))

    q = Lsphi/(L+Lsphi)

    rphi0 = (2.06783383*1e-15)/(2*np.pi)

    f0 = 1/(2*np.pi*np.sqrt((L+Lsphi)*C))

    alpha = Delta/(2*(rphi0**2))

    # L1 = alpha*(np.cos(de)+np.sin(de/2)**4)/((1-np.sin(de/2)**2)**1.5)

    L1 = alpha*np.cos(de/2)

    return f0 - (q*f0*Lsphi*n*L1/2)
