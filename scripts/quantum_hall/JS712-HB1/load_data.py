import h5py
import numpy as np
from copy import deepcopy

class Loader():
    
    PATH_DATA_DIR = "data/"
    SAMPLE_NAME = "JS712-HB1"
    EXPERIMENTOR = "-SMF_"
    PROBE_CURRENT = 1.0e-6
    GEOMETRIC_FACTOR = 0.75
    SCAN_DENSITY = "009"
    SCAN_FAN = "013"
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
    def load_density(cls, Z_FIELD_COLUMN=0, GATE_COLUMN=1, YY_VOLTAGE_COLUMN=10, YX_VOLTAGE_COLUMN=8):
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
        if gate[0, 0] > gate[0, -1]:
            # reverse gate order 
            z_field = z_field[:, ::-1]
            gate = gate[:, ::-1]
            ryy = ryy[:, ::-1]
            ryx = ryx[:, ::-1]
        print("Density data loaded")
        return z_field, gate, ryy, ryx

    @classmethod
    def load_fan(cls, Z_FIELD_COLUMN=0, GATE_COLUMN=1, YY_VOLTAGE_COLUMN=10, YX_VOLTAGE_COLUMN=8):
        data = cls.load_hdf5(cls.SCAN_FAN)
        z_field = data[:, Z_FIELD_COLUMN, :]
        gate = data[:, GATE_COLUMN, :]
        ryy = data[:, YY_VOLTAGE_COLUMN, :] / cls.PROBE_CURRENT
        ryx = data[:, YX_VOLTAGE_COLUMN, :] / cls.PROBE_CURRENT
        for i in range(np.shape(gate)[1]):    
            if z_field[0, i] > z_field[-1, i]:
                z_field[:, i] = z_field[::-1, i]
                ryy[:, i] = ryy[::-1, i]
                ryx[:, i] = ryx[::-1, i]
        if gate[0, 0] > gate[0, -1]:
            # reverse gate order 
            z_field = z_field[:, ::-1]
            gate = gate[:, ::-1]
            ryy = ryy[:, ::-1]
            ryx = ryx[:, ::-1]
        print("Fan chart data loaded")
        return z_field, gate, ryy, ryx
