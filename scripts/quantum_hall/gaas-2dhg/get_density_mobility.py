import json
import numpy as np
from shabanipy.quantum_hall.density import extract_density
from shabanipy.quantum_hall.mobility import extract_mobility

def get_density_mobility(z_field_xx, z_field_yy, z_field_xy, rxx, ryy, rxy, geometric_factor, z_field_range=[]):
    if z_field_range == []:
        z_field_range = [z_field[0], z_field[-1]]
    density, _, _ = extract_density(z_field_xy, rxy, [z_field_range[0], z_field_range[1]])
    mobility_xx, _ = extract_mobility(z_field_xx, rxx, ryy, density, geometric_factor)
    _, mobility_yy = extract_mobility(z_field_yy, rxx, ryy, density, geometric_factor)
    return density, mobility_xx, mobility_yy
