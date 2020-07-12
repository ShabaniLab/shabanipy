from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft
from scipy import constants as c

from shabanipy.jj.fraunhofer.generate_pattern import produce_fraunhofer_fast
from shabanipy.jj.fraunhofer.deterministic_reconstruction \
        import extract_theta, extract_current_distribution

# constants and junction dimensions
PHI0 = c.physical_constants['mag. flux quantum'][0]
a = 1e-6    # junction width
d = 100e-9  # junction length (really d + 2λ)


# tophat
jx = np.zeros(100)
jx[40:60] = 1  # A/m (20 points in junction)

# position step size
dx = a / len(jx[np.where(jx != 0)])
x = np.arange(len(jx))
x = (x - np.median(x))*dx  # center and scale

# fourier transform -> critical current
g = fft(jx)
g = np.conj(g)  # match sign convention of (3) vs. numpy.fft
ic = dx*np.abs(g)  # A

# field step size
dbeta = 2*np.pi / (len(jx)*dx)  # rad / m
beta = np.arange(len(ic))*dbeta
B = beta * PHI0 / (2*np.pi * d)  # T


###########

ic2 = produce_fraunhofer_fast(B, a, np.array([2*np.pi*d/PHI0]*len(jx)), jx, 1 +
        2**7)
theta = extract_theta(B, ic2)
x2, jx2 = extract_current_distribution(B, ic2, 2*np.pi*d/PHI0, a, 100)
