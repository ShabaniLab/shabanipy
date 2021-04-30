"""Study nonunivocal property of the current reconstruction.

H. H. Zappe (1975) gives examples of current distributions that yield the same
Fraunhofer.  Which current distribution (if either) does the current
reconstruction algorithm give back?

As noted by Zappe, the current distribution J(x) returned will be the one whose
Fourier transform is a minimum-phase function whose modulus is the Fraunhofer.
Also noted by Zappe, a decorrelation algorithm such as the one we use cannot
distinguish J(x) from J(-x), as illustrated below.

How might this change if J(x) is everywhere smooth and differentiable? E.g. use a sum of
gennorms instead of square waves.
"""

import numpy as np
from matplotlib import pyplot as plt

from shabanipy.jj.fraunhofer.dynesfulton import (
    critical_current_density,
)
from shabanipy.jj.fraunhofer.dynesfulton import fraunhofer

N = 1000
JJ_WIDTH = 2
x = np.linspace(-JJ_WIDTH // 2, JJ_WIDTH // 2, N)
jx1 = np.zeros(N)
jx2 = np.zeros(N)
jx1[: N // 3], jx1[N // 3 : 2 * N // 3], jx1[2 * N // 3 :] = 2, 5, 2
jx2[: N // 3], jx2[N // 3 : 2 * N // 3], jx2[2 * N // 3 :] = 1, 4, 4

x_padded = np.linspace(-JJ_WIDTH, JJ_WIDTH, 2 * N)
jx1_padded = np.concatenate((np.zeros(N // 2), jx1, np.zeros(N // 2)))
jx2_padded = np.concatenate((np.zeros(N // 2), jx2, np.zeros(N // 2)))
fig, ax = plt.subplots()
ax.plot(x_padded, jx1_padded, label="input 1")
ax.plot(x_padded, jx2_padded, label="input 2", linestyle="-.")
ax.legend()

B2BETA = 4
b = np.linspace(-10, 10, N)
g1 = fraunhofer(b, B2BETA, jx1, x, ret_fourier=True)
g2 = fraunhofer(b, B2BETA, jx2, x, ret_fourier=True)
ic1, ic2 = np.abs(g1), np.abs(g2)

fig, ax = plt.subplots()
ax.plot(b, ic1, label="from input 1")
ax.plot(b, ic2, label="from input 2", linestyle="-.")
ax.legend()

x1_ex, j1_ex = critical_current_density(b, ic1, B2BETA, JJ_WIDTH, N)
x2_ex, j2_ex = critical_current_density(b, ic2, B2BETA, JJ_WIDTH, N)

fig, ax = plt.subplots()
ax.plot(x1_ex, j1_ex, label="output 1")
ax.plot(x2_ex, j2_ex, label="output 2", linestyle="-.")
ax.legend()

plt.show()
