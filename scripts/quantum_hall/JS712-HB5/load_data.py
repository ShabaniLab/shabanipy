import h5py
import numpy as np
from copy import deepcopy

class Loader():
    
    PATH_DATA_DIR = "data/"
    SAMPLE_NAME = "JS712-HB5"
    EXPERIMENTOR = "-SMF-"
    PROBE_CURRENT = 1.0e-6
    GEOMETRIC_FACTOR = 0.75
    SCAN_RXX_RYX = "003"
    SCAN_RYY_RXY = "004"
    SCAN_DENSITY = "016"
    FORMAT = ".hdf5"

    @classmethod
    def load_hdf5(cls, scan):
        path = cls.PATH_DATA_DIR + cls.SAMPLE_NAME + cls.EXPERIMENTOR + scan + cls.FORMAT
        print(f"loading {path}...", end =" ")
        f = h5py.File(path, 'r')
        data = deepcopy(np.array(f['Data']['Data']))
        f.close()
        print("done")
        return data

    @classmethod
    def load_rxx_ryx(cls, Z_FIELD_COLUMN=0, XX_VOLTAGE_COLUMN=6, YX_VOLTAGE_COLUMN=8):
        data = cls.load_hdf5(cls.SCAN_RXX_RYX)
        z_field = data[:, Z_FIELD_COLUMN, 0]
        rxx = data[:, XX_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        ryx = data[:, YX_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        if z_field[0] > z_field[-1]:
            z_field = z_field[::-1]
            rxx = rxx[::-1]
            ryx = ryx[::-1]
        print("Rxx and Ryx loaded")
        return z_field, rxx, ryx

    @classmethod
    def load_ryy_rxy(cls, Z_FIELD_COLUMN=0, YY_VOLTAGE_COLUMN=6, XY_VOLTAGE_COLUMN=8):
        data = cls.load_hdf5(cls.SCAN_RYY_RXY)
        z_field = data[:, Z_FIELD_COLUMN, 0]
        ryy = data[:, YY_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        rxy = data[:, XY_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        if z_field[0] > z_field[-1]:
            z_field = z_field[::-1]
            ryy = ryy[::-1]
            rxy = rxy[::-1]
        print("Ryy and Rxy loaded")
        return z_field, ryy, rxy
