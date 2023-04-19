import h5py
import numpy as np
from copy import deepcopy

class Loader():
    
    PATH_DATA_DIR = "data/"
    SAMPLE_NAME = "JS712-HB4"
    EXPERIMENTOR = "-SMF-"
    PROBE_CURRENT = 1.0e-6
    GEOMETRIC_FACTOR = 0.75
    SCAN_RXX = "003"
    SCAN_RYY = "004"
    SCAN_RYX = "005"
    SCAN_RXY = "006"
    SCAN_FAN = "013"
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
    def load_rxx(cls, Z_FIELD_COLUMN=0, XX_VOLTAGE_COLUMN=6):
        data = cls.load_hdf5(cls.SCAN_RXX)
        z_field = data[:, Z_FIELD_COLUMN, 0]
        r = data[:, XX_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        if z_field[0] > z_field[-1]:
            z_field = z_field[::-1]
            r = r[::-1]
        print("Rxx loaded")
        return z_field, r

    @classmethod
    def load_ryy(cls, Z_FIELD_COLUMN=0, YY_VOLTAGE_COLUMN=6):
        data = cls.load_hdf5(cls.SCAN_RYY)
        z_field = data[:, Z_FIELD_COLUMN, 0]
        r = data[:, YY_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        if z_field[0] > z_field[-1]:
            z_field = z_field[::-1]
            r = r[::-1]
        print("Ryy loaded")
        return z_field, r

    @classmethod
    def load_ryx(cls, Z_FIELD_COLUMN=0, YX_VOLTAGE_COLUMN=6):
        data = cls.load_hdf5(cls.SCAN_RYX)
        z_field = data[:, Z_FIELD_COLUMN, 0]
        r = data[:, YX_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        if z_field[0] > z_field[-1]:
            z_field = z_field[::-1]
            r = r[::-1]
        print("Ryx loaded")
        return z_field, r

    @classmethod
    def load_rxy(cls, Z_FIELD_COLUMN=0, XY_VOLTAGE_COLUMN=6):
        data = cls.load_hdf5(cls.SCAN_RXY)
        z_field = data[:, Z_FIELD_COLUMN, 0]
        r = data[:, XY_VOLTAGE_COLUMN, 0] / cls.PROBE_CURRENT
        if z_field[0] > z_field[-1]:
            z_field = z_field[::-1]
            r = r[::-1]
        print("Rxy loaded")
        return z_field, r

    @classmethod
    def load_fan(cls, Z_FIELD_COLUMN=0, GATE_COLUMN=1, XX_VOLTAGE_COLUMN=7, YX_VOLTAGE_COLUMN=9):
        data = cls.load_hdf5(cls.SCAN_FAN)
        z_field = data[:, Z_FIELD_COLUMN, :]
        gate = data[:, GATE_COLUMN, :]
        rxx = data[:, XX_VOLTAGE_COLUMN, :] / cls.PROBE_CURRENT
        ryx = data[:, YX_VOLTAGE_COLUMN, :] / cls.PROBE_CURRENT
        for i in range(np.shape(gate)[1]):    
            if z_field[0, i] > z_field[-1, i]:
                z_field[:, i] = z_field[::-1, i]
                rxx[:, i] = rxx[::-1, i]
                ryx[:, i] = ryx[::-1, i]
        # reverse gate order 
        z_field = z_field[:, ::-1]
        gate = gate[:, ::-1]
        rxx = rxx[:, ::-1]
        ryx = ryx[:, ::-1]
        print("Fan chart data loaded")
        return z_field, gate, rxx, ryx

    @classmethod
    def load_density(cls, Z_FIELD_COLUMN=0, GATE_COLUMN=1, XX_VOLTAGE_COLUMN=7, YX_VOLTAGE_COLUMN=9):
        data = cls.load_hdf5(cls.SCAN_DENSITY)
        z_field = data[:, Z_FIELD_COLUMN, :]
        gate = data[:, GATE_COLUMN, :]
        rxx = data[:, XX_VOLTAGE_COLUMN, :] / cls.PROBE_CURRENT
        ryx = data[:, YX_VOLTAGE_COLUMN, :] / cls.PROBE_CURRENT
        for i in range(np.shape(gate)[1]):    
            if z_field[0, i] > z_field[-1, i]:
                z_field[:, i] = z_field[::-1, i]
                rxx[:, i] = rxx[::-1, i]
                ryx[:, i] = ryx[::-1, i]
        # reverse gate order 
        z_field = z_field[:, ::-1]
        gate = gate[:, ::-1]
        rxx = rxx[:, ::-1]
        ryx = ryx[:, ::-1]
        print("Density data loaded")
        return z_field, gate, rxx, ryx
