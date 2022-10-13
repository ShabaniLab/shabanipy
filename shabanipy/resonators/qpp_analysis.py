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
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from shabanipy.utils.plotting import jy_pink
jy_pink.register()
plt.style.use('bold')
plt.style.use('presentation')

def FluxFunc_ko1(fl,q0):
    return (1+q0*np.sin(np.pi*fl/2)*np.arctanh(np.sin(np.pi*fl/2))/(1-np.sin(np.pi*fl/2)*np.arctanh(np.sin(np.pi*fl/2))))**(-0.5)

def FluxFunc_general(fl,q0,t):
    return (1+q0*((t*np.cos(np.pi*fl)-t+2)**1.5/(2**0.5*(t/4*(np.cos(2*np.pi*fl)+3)-(t-2)*np.cos(np.pi*fl))) - 1))**(-0.5)

def multiFluxFunc_ko1(fl,w0,q0,fl0,fs):
    return w0*(1+q0*np.sin(np.pi*(fl-fl0)/fs/2)*np.arctanh(np.sin(np.pi*(fl-fl0)/fs/2))/(1-np.sin(np.pi*(fl-fl0)/fs/2)*np.arctanh(np.sin(np.pi*(fl-fl0)/fs/2))))**(-0.5)
    
def multiFluxFunc_general(fl,w0,q0,fl0,fs, t):
    return w0*(1+q0*((t*np.cos(np.pi*(fl-fl0)/fs)-t+2)**(1.5)/(2**(0.5)*(t/4*(np.cos(2*np.pi*(fl-fl0)/fs)+3)-(t-2)*np.cos(np.pi*(fl-fl0)/fs))) - 1))**(-0.5)

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

def plot_clearing_tone(
    tone_freq:np.ndarray,
    freq:np.ndarray,
    s21:np.ndarray,
    cb_lims: Optional[np.ndarray] = None,
    fig_size: Optional[np.ndarray] = None,
)-> None:

    fig, ax = plt.subplots(figsize = (20,10),constrained_layout=True, dpi = 50)
    m_ax = fig.gca()
    pm = m_ax.pcolormesh(tone_freq/1e9,
            freq/1e9,
            np.abs(s21),
            vmin = cb_lims[0] if cb_lims else  0,
            vmax = cb_lims[1] if cb_lims else  None,
            cmap = 'jy_pink', 
            shading = 'auto',
            linewidth=0,
            rasterized = True
            )
    cb = fig.colorbar(pm, ax = m_ax,pad = 0.02)
    cb.ax.set_xlabel(r'S21', labelpad = 10)
    m_ax.set_xlabel('Clearing tone frequency (GHz)', labelpad = 20)
    m_ax.set_ylabel('Freq (GHz)')
    # cb = fig.colorbar(hist[3], ax=ax)
