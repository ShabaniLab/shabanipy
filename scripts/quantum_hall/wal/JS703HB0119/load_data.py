from configparser import ConfigParser, ExtendedInterpolation
import h5py
import numpy as np

class Loader:

    config = ConfigParser(interpolation=ExtendedInterpolation())

    @classmethod
    def load_hall_resistance_d4(cls, config_path="config.ini", tag='HALL_D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        X_FIELD_COLUMN = int(cls.config[tag]['X_FIELD_COLUMN'])
        XY_VOLTAGE_COLUMN = int(cls.config[tag]['XY_VOLTAGE_COLUMN'])
        PROBE_CURRENT = float(cls.config['GLOBAL']['PROBE_CURRENT'])
        PATH = str(cls.config[tag]['PATH'])
        print("done")
        print(f"reading {PATH}...", end =" ")
        f = h5py.File(PATH, 'r')
        data = f['Log_2']['Data']['Data']
        x_field = data[:, X_FIELD_COLUMN]
        x_field = x_field[:, 0]
        rxy = data[:, XY_VOLTAGE_COLUMN] / PROBE_CURRENT
        rxy = rxy[:, 0]
        f.close()
        print("done")
        print("Hall resistance for device D4 loaded")
        return x_field, rxy

    @classmethod
    def load_longitudinal_resistance_d4(cls, config_path="config.ini", tag='LONGITUDINAL_D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        X_FIELD_COLUMN = int(cls.config[tag]['X_FIELD_COLUMN'])
        YY_VOLTAGE_COLUMN = int(cls.config[tag]['YY_VOLTAGE_COLUMN'])
        PROBE_CURRENT = float(cls.config['GLOBAL']['PROBE_CURRENT'])
        PATH = str(cls.config[tag]['PATH'])
        W = float(cls.config['GLOBAL']['HALL_BAR_WIDTH_MICRON'])
        L = float(cls.config['GLOBAL']['HALL_BAR_LENGTH_MICRON'])
        GEOMETRIC_FACTOR = W/L
        print("done")
        print(f"reading {PATH}...", end =" ")
        f = h5py.File(PATH, 'r')
        data = f['Data']['Data']
        x_field = data[:, X_FIELD_COLUMN]
        x_field = x_field[:, 0]
        ryy = data[:, YY_VOLTAGE_COLUMN] / PROBE_CURRENT
        ryy = ryy[:, 0]
        f.close()
        print("done")
        print("Longitudinal resistance for device D4 loaded")
        return x_field, ryy, GEOMETRIC_FACTOR

    @classmethod
    def load_depletion_sweep_d4(cls, config_path="config.ini", tag='DEPLETION_D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        DEPLETION_GATE_COLUMN = int(cls.config[tag]['DEPLETION_GATE_COLUMN'])
        YY_VOLTAGE_COLUMN = int(cls.config[tag]['YY_VOLTAGE_COLUMN'])
        PROBE_CURRENT = float(cls.config['GLOBAL']['PROBE_CURRENT'])
        PATH = str(cls.config[tag]['PATH'])
        print("done")
        print(f"reading {PATH}...", end =" ")
        f = h5py.File(PATH, 'r')
        data1 = f['Data']['Data']
        gate1 = data1[:, DEPLETION_GATE_COLUMN]
        vyy1 = data1[:, YY_VOLTAGE_COLUMN]
        data2 = f['Log_2']['Data']['Data']
        gate2 = data2[:, DEPLETION_GATE_COLUMN]
        vyy2 = data2[:, YY_VOLTAGE_COLUMN]
        data3 = f['Log_3']['Data']['Data']
        gate3 = data3[:, DEPLETION_GATE_COLUMN]
        vyy3 = data3[:, YY_VOLTAGE_COLUMN]
        data4 = f['Log_4']['Data']['Data']
        gate4 = data4[:, DEPLETION_GATE_COLUMN]
        vyy4 = data4[:, YY_VOLTAGE_COLUMN]
        gate = np.concatenate((gate1, gate2, gate3, gate4), axis=0)
        gate = gate[:, 0]
        vyy = np.concatenate((vyy1, vyy2, vyy3, vyy4), axis=0)
        vyy = vyy[:, 0]
        ryy = vyy / PROBE_CURRENT
        f.close()
        print("done")
        print("Depletion sweep for device D4 loaded")
        return gate, ryy

    @classmethod
    def load_wire_sweep_d4(cls, config_path="config.ini", tag='WIRE_D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        WIRE_GATE_COLUMN = int(cls.config[tag]['WIRE_GATE_COLUMN'])
        YY_VOLTAGE_COLUMN = int(cls.config[tag]['YY_VOLTAGE_COLUMN'])
        PROBE_CURRENT = float(cls.config['GLOBAL']['PROBE_CURRENT'])
        PATH = str(cls.config[tag]['PATH'])
        print("done")
        print(f"reading {PATH}...", end =" ")
        f = h5py.File(PATH, 'r')
        data = f['Data']['Data']
        gate = data[:, WIRE_GATE_COLUMN]
        vyy = data[:, YY_VOLTAGE_COLUMN]
        gate = gate[:, 0]
        vyy = vyy[:, 0]
        ryy = vyy / PROBE_CURRENT
        f.close()
        print("done")
        print("Wire sweep for device D4 loaded")
        return gate, ryy
    
    @classmethod
    def load_magnetoresistance_depletion_d4(cls, config_path="config.ini", tag='MAGNETORESISTANCE_DEPLETION_D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        X_FIELD_COLUMN = int(cls.config[tag]['X_FIELD_COLUMN'])
        DEPLETION_GATE_COLUMN = int(cls.config[tag]['DEPLETION_GATE_COLUMN'])
        YY_VOLTAGE_COLUMN = int(cls.config[tag]['YY_VOLTAGE_COLUMN'])
        PROBE_CURRENT = float(cls.config['GLOBAL']['PROBE_CURRENT'])
        W = float(cls.config['GLOBAL']['HALL_BAR_WIDTH_MICRON'])
        L = float(cls.config['GLOBAL']['HALL_BAR_LENGTH_MICRON'])
        GEOMETRIC_FACTOR = W/L
        PATH1 = str(cls.config[tag]['PATH1'])
        PATH2 = str(cls.config[tag]['PATH1'])
        print("done")
        print(f"reading {PATH1}...", end =" ")
        f = h5py.File(PATH1, 'r')
        data1 = f['Data']['Data']
        x_field1 = data1[:, X_FIELD_COLUMN, :]
        gate1 = data1[0, DEPLETION_GATE_COLUMN, :]
        vyy1 = data1[:, YY_VOLTAGE_COLUMN, :]
        f.close()
        print("done")
        print(f"reading {PATH2}...", end =" ")
        f = h5py.File(PATH1, 'r')
        data2 = f['Data']['Data']
        x_field2 = data2[:, X_FIELD_COLUMN, :]
        gate2 = data2[0, DEPLETION_GATE_COLUMN, :]
        vyy2 = data2[:, YY_VOLTAGE_COLUMN, :]
        f.close()
        print("done")
        x_field = np.concatenate((x_field1, x_field2), axis=1)
        gate = np.concatenate((gate1, gate2), axis=0)
        vyy = np.concatenate((vyy1, vyy2), axis=1)
        ryy = vyy / PROBE_CURRENT
        print("Magnetoresistance vs depletion gates for device D4 loaded")
        return x_field, gate, ryy, GEOMETRIC_FACTOR

    @classmethod
    def load_geometry_micron_d4(cls, config_path="config.ini", tag='D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        W = float(cls.config['GLOBAL']['HALL_BAR_WIDTH_MICRON'])
        L = float(cls.config['GLOBAL']['HALL_BAR_LENGTH_MICRON'])
        GEOMETRIC_FACTOR = W/L
        w = float(cls.config[tag]['WIRE_WIDTH_MICRON'])
        l = float(cls.config[tag]['WIRE_LENGTH_MICRON'])
        print("done")
        print("Geometry for device D4 loaded")
        return GEOMETRIC_FACTOR, w, l

    @classmethod
    def load_wire_z_field_d4(cls, config_path="config.ini", tag='WIRE_Z_MAGNET_D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        WIRE_GATE_COLUMN = int(cls.config[tag]['WIRE_GATE_COLUMN'])
        Z_FIELD_COLUMN = int(cls.config[tag]['Z_FIELD_COLUMN'])
        YY_VOLTAGE_COLUMN = int(cls.config[tag]['YY_VOLTAGE_COLUMN'])
        PROBE_CURRENT = float(cls.config['GLOBAL']['PROBE_CURRENT'])
        print("done")
        PATH = str(cls.config[tag]['PATH'])
        print("done")
        print(f"reading {PATH}...", end =" ")
        f = h5py.File(PATH, 'r')
        data = f['Data']['Data']
        gate = data[:, WIRE_GATE_COLUMN, :]
        z_field = data[0, Z_FIELD_COLUMN, :]
        vyy = data[:, YY_VOLTAGE_COLUMN, :]
        ryy = vyy / PROBE_CURRENT
        f.close()
        print("done")
        print("Wire gate vs Z magnetic field for device D4 loaded")
        return gate, z_field, ryy

    @classmethod
    def load_wire_y_field_d4(cls, config_path="config.ini", tag='WIRE_Y_MAGNET_D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        WIRE_GATE_COLUMN = int(cls.config[tag]['WIRE_GATE_COLUMN'])
        Y_FIELD_COLUMN = int(cls.config[tag]['Y_FIELD_COLUMN'])
        YY_VOLTAGE_COLUMN = int(cls.config[tag]['YY_VOLTAGE_COLUMN'])
        PROBE_CURRENT = float(cls.config['GLOBAL']['PROBE_CURRENT'])
        print("done")
        PATH = str(cls.config[tag]['PATH'])
        print("done")
        print(f"reading {PATH}...", end =" ")
        f = h5py.File(PATH, 'r')
        data = f['Data']['Data']
        gate = data[:, WIRE_GATE_COLUMN, :]
        y_field = data[0, Y_FIELD_COLUMN, :]
        vyy = data[:, YY_VOLTAGE_COLUMN, :]
        ryy = vyy / PROBE_CURRENT
        f.close()
        print("done")
        print("Wire gate vs Y magnetic field for device D4 loaded")
        return gate, y_field, ryy

    @classmethod
    def load_wire_x_field_d4(cls, config_path="config.ini", tag='WIRE_X_MAGNET_D4'):
        cls.config.read(config_path)
        print(f"reading {config_path}...", end =" ")
        WIRE_GATE_COLUMN = int(cls.config[tag]['WIRE_GATE_COLUMN'])
        X_FIELD_COLUMN = int(cls.config[tag]['X_FIELD_COLUMN'])
        YY_VOLTAGE_COLUMN = int(cls.config[tag]['YY_VOLTAGE_COLUMN'])
        PROBE_CURRENT = float(cls.config['GLOBAL']['PROBE_CURRENT'])
        print("done")
        PATH = str(cls.config[tag]['PATH'])
        print("done")
        print(f"reading {PATH}...", end =" ")
        f = h5py.File(PATH, 'r')
        data = f['Data']['Data']
        gate = data[:, WIRE_GATE_COLUMN, :]
        x_field = data[0, X_FIELD_COLUMN, :]
        vyy = data[:, YY_VOLTAGE_COLUMN, :]
        ryy = vyy / PROBE_CURRENT
        f.close()
        print("done")
        print("Wire gate vs X magnetic field for device D4 loaded")
        return gate, x_field, ryy
    
    
    # @classmethod
    # def load_yx_magnet_correction(cls, config_path="config.ini"):
    #     cls.config.read(config_path)
    #     print(f"reading {config_path}...", end =" ")
    #     X_FIELD_COLUMN = int(cls.config['YX_MAGNET_CORRECTION']['X_FIELD_COLUMN'])
    #     Y_FIELD_COLUMN = int(cls.config['YX_MAGNET_CORRECTION']['Y_FIELD_COLUMN'])
    #     YY_VOLTAGE_COLUMN = int(cls.config['YX_MAGNET_CORRECTION']['YY_VOLTAGE_COLUMN'])
    #     PROBE_CURRENT = float(cls.config['YX_MAGNET_CORRECTION']['PROBE_CURRENT'])
    #     PATH = str(cls.config['YX_MAGNET_CORRECTION']['PATH'])
    #     print("done")
    #     print(f"reading {PATH}...", end =" ")
    #     f = h5py.File(PATH, 'r')
    #     data = f['Data']['Data']
    #     x_field = data[:, X_FIELD_COLUMN, :]
    #     y_field = data[:, Y_FIELD_COLUMN, :]
    #     ryy = data[:, YY_VOLTAGE_COLUMN, :] / PROBE_CURRENT
    #     f.close()
    #     print("done")
    #     print("magnet correction loaded")
    #     return x_field, y_field, ryy

    # @classmethod
    # def load_magnetoresistance_for_depletion_gate(cls, config_path="config.ini"):
    #     cls.config.read(config_path)
    #     print(f"reading {config_path}...", end =" ")
    #     X_FIELD_COLUMN = int(cls.config['YX_MAGNET_CORRECTION']['X_FIELD_COLUMN'])
    #     Y_FIELD_COLUMN = int(cls.config['YX_MAGNET_CORRECTION']['Y_FIELD_COLUMN'])
    #     YY_VOLTAGE_COLUMN = int(cls.config['YX_MAGNET_CORRECTION']['YY_VOLTAGE_COLUMN'])
    #     PROBE_CURRENT = float(cls.config['YX_MAGNET_CORRECTION']['PROBE_CURRENT'])
    #     PATH = str(cls.config['YX_MAGNET_CORRECTION']['PATH'])
    #     print("done")
    #     print(f"reading {PATH}...", end =" ")
    #     f = h5py.File(PATH, 'r')
    #     data = f['Data']['Data']
    #     x_field = data[:, X_FIELD_COLUMN, :]
    #     y_field = data[:, Y_FIELD_COLUMN, :]
    #     ryy = data[:, YY_VOLTAGE_COLUMN, :] / PROBE_CURRENT
    #     f.close()
    #     print("done")
    #     return x_field, y_field, ryy
    
    # @classmethod
    # def create_sample(gate, field, rxx, rxy):
    #     # calculate transport
    #     density, density_stderr = extract_density(field.T, rxy.T, [-0.098, 0.098], plot_fit=False)
    #     mobility, mobility_stderr = extract_mobility(field.T, rxx.T, rxx.T, density, 
    #         GEOMETRIC_FACTORS[config['GLOBAL']['SAMPLE_GEOMETRY']])
    #     new_sample = Sample(SAMPLE_NAME, 
    #         probe_current, 
    #         GEOMETRIC_FACTOR, 
    #         EFFECTIVE_MASS, 
    #         gate, 
    #         density, 
    #         mobility)
    #     return new_sample
