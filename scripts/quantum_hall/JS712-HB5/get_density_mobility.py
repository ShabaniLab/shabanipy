import json
import numpy as np

from load_data import Loader
from shabanipy.quantum_hall.density import extract_density
from shabanipy.quantum_hall.mobility import extract_mobility

def get_density_mobility(z_field, rxy, rxx, ryy, geometric_factor, z_field_range=[]):
    if z_field_range == []:
        z_field_range = [z_field[0], z_field[-1]]
    density, _, _ = extract_density(z_field, rxy, [z_field_range[0], z_field_range[1]])
    mobility_xx, mobility_yy = extract_mobility(z_field, rxx, ryy, density, geometric_factor)
    return density, mobility_xx, mobility_yy
