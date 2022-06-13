"""Model the ratios of the junction and loop areas of a SQUID.

We want the critical current of the reference junction to be nearly constant over
several SQUID oscillations.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import physical_constants

PHI0 = physical_constants["mag. flux quantum"][0]

ref_area = 1e-6 * 1e-6  # reference JJ, effective length ~1μm due to Meissner focusing
loop_area = 7e-6 * 7e-6
print(f"{loop_area/ref_area=}")
squid_period = PHI0 / loop_area
print(f"squid_period={squid_period / 1e-6}μT")
print(f"jj_period={PHI0 / ref_area / 1e-3}mT")

bfield = 5 * squid_period
ref_ic = np.abs(np.sinc(bfield * ref_area / PHI0))
print(f"reference JJ: Ic(5 squid periods)/Ic(0) = {ref_ic}")
