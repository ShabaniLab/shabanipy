import h5py
import numpy as np
from copy import deepcopy
from configparser import ConfigParser, ExtendedInterpolation

class Loader():

    config = ConfigParser(interpolation=ExtendedInterpolation())

    @classmethod
    def get_data(cls, config_path="config.ini", tag='VXX'):
        # read config file 
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        PATH = str(cls.config[tag]['PATH'])
        Z_FIELD_COLUMN = int(cls.config[tag]['Z_FIELD_COLUMN'])
        LOCKIN_COLUMN = int(cls.config[tag]['LOCKIN_COLUMN'])
        PROBE_CURRENT = float(cls.config[tag]['PROBE_CURRENT'])
        print("done")
        # read hdf5 file
        print(f"reading {PATH}...", end =" ")
        f = h5py.File(PATH, 'r')
        data = deepcopy(np.array(f['Data']['Data']))
        f.close()
        z_field = data[:, Z_FIELD_COLUMN]
        v = data[:, LOCKIN_COLUMN]
        r = v / PROBE_CURRENT
        # make sure z_field is in ascending order
        if z_field[0] > z_field[-1]:
            z_field = z_field[::-1]
            r = r[::-1]
        print("done")
        return z_field, r
    
    @classmethod
    def get_name(cls, config_path="config.ini"):
        return str(cls.config['GLOBAL']['NAME'])
