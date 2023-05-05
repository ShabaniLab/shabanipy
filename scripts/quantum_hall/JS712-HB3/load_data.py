import h5py
import numpy as np
from copy import deepcopy

class Loader():
    
    PATH_DATA_DIR = "data/"
    SAMPLE_NAME = "JS712-HB3"
    EXPERIMENTER = "-SMF-"
    PROBE_CURRENT = 1.0e-6
    GEOMETRIC_FACTOR = 0.75
    SCAN_RYY_RYX = "006"
    SCAN_DENSITY = "007"
    FORMAT = ".hdf5"

    @classmethod
    def load_hdf5(cls, scan):
        path = cls.PATH_DATA_DIR + cls.SAMPLE_NAME + cls.EXPERIMENTER + scan + cls.FORMAT
        print(f"loading {path}...", end =" ")
        f = h5py.File(path, 'r')
        data = deepcopy(np.array(f['Data']['Data']))
        f.close()
        print("done")
        return data

    @classmethod
    def load_ryy_ryx(cls, Z_FIELD_COLUMN=0, YY_VOLTAGE_COLUMN=6, YX_VOLTAGE_COLUMN=8):
        data = cls.load_hdf5(cls.SCAN_RYY_RYX)
        z_field = data[:, Z_FIELD_COLUMN, 0]
        ryy = data[:, YY_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        ryx = data[:, YX_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        if z_field[0] > z_field[-1]:
            z_field = z_field[::-1]
            ryy = ryy[::-1]
            ryx = ryx[::-1]
        print("Ryy and Ryx loaded")
        return z_field, ryy, ryx

    @classmethod
    def load_density(cls, Z_FIELD_COLUMN=0, GATE_COLUMN=1, YY_VOLTAGE_COLUMN=7, YX_VOLTAGE_COLUMN=9):
        data = cls.load_hdf5(cls.SCAN_DENSITY)
        z_field = data[:, Z_FIELD_COLUMN, :]
        gate = data[:, GATE_COLUMN, :]
        ryy = data[:, YY_VOLTAGE_COLUMN, :] / cls.PROBE_CURRENT
        ryx = data[:, YX_VOLTAGE_COLUMN, :] / cls.PROBE_CURRENT
        for i in range(np.shape(gate)[1]):    
            if z_field[0, i] > z_field[-1, i]:
                z_field[:, i] = z_field[::-1, i]
                ryy[:, i] = ryy[::-1, i]
                ryx[:, i] = ryx[::-1, i]
        # reverse gate order 
        z_field = z_field[:, ::-1]
        gate = gate[:, ::-1]
        ryy = ryy[:, ::-1]
        ryx = ryx[:, ::-1]
        # ignore the first 4 negative voltages as they have out of phase components
        z_field = z_field[:, 4:]
        gate = gate[:, 4:]
        ryy = ryy[:, 4:]
        ryx = ryx[:, 4:]
        print("Density data loaded")
        return z_field, gate, ryy, ryx
