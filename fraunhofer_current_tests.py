import numpy as np
from matplotlib import pyplot as plt
from fraunhofer_current import fraunhofer, phase


def fraunhofer_test_tophat():
    #bfield = np.arange(-1, 1)  # TODO only step size needed? if at all?
    current_dist = np.zeros(60)
    current_dist[25:35] = 1
    fraun = fraunhofer(None, current_dist, 1, 1)
    # TODO: convert fraunhofer back to current dist. and assert equal
    # TODO: plotting optional?
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=[10,4])
    ax1.plot(current_dist, linewidth=0, marker='.')
    ax2.plot(fraun, linewidth=0.5, marker='.')
    # TODO: doesn't work
    hilb = phase(None, fraun)
    ax1.plot(hilb)
    return fraun, hilb
