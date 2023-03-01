import json

from load_data import Loader
from shabanipy.quantum_hall.density import extract_density
from shabanipy.quantum_hall.mobility import extract_mobility

x_field, rxy = Loader.load_hall_resistance_d4()
density, density_std = extract_density(x_field, rxy, [x_field[0], x_field[-1]])
print(r"density = {0:.4f} (10e12 1/cm2)".format(density/1e16))
print(r"density_std = {0:.4f} (10e12 1/cm2)".format(density_std/1e16))

x_field, ryy, geometric_factor = Loader.load_longitudinal_resistance_d4()
mobility, _ = extract_mobility(x_field, ryy, ryy, density, geometric_factor)
print(r"mobility = {0:.0f} (cm2/Vs)".format(mobility*1e4))

data = {'density': density, 
    'density_std': density_std,
    'mobility': mobility}
with open("density_mobility_d4.txt", "w") as fp:
    json.dump(data, fp)
    print("saved file density_mobility_d4.txt")
