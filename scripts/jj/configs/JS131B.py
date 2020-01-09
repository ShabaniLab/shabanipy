#: Common folder in which all data are stored
BASE_FOLDER = r'/Users/mdartiailh/Labber/Data/2019/08'

#: Name of the sample and associated parameters as a dict.
#: The currently expected keys are:
#: - path
#: - Tc (in K)
#: - gap size (in nm)
SAMPLES = {"JJ100": {"path": "Data_0815/JS131B_BM001_022.hdf5",
                     "Tc": 1.46, "gap size": 100},
            }

#: Path to the file in which to write the output
OUTPUT = "/Users/mdartiailh/Documents/PostDocNYU/DataAnalysis/JJ/JS131B/results.csv"

#: For all the following parameters one can use a dictionary with sample names
#: as keys can be used.

#: Name or index of the column containing the voltage bias data.
#: This should be a stepped channel ! use the applied voltage not the
#: measured current
BIAS_NAME = 0

#: Name or index of the column containing the voltage data
VOLTAGE_NAME = 2

#: Name or index of the column containing the gate value for scans with
#: gate traces (use None if absent).
GATE_NAME = 1

#: Should we correct the offset in voltage and if so on how many points to
#: average
CORRECT_VOLTAGE_OFFSET = 5

#: Conversion factor to apply to the current data (allow to convert from
#: applied voltage to current bias).
CURRENT_CONVERSION = 1e-6

#: Amplifier gain used to measure the voltage across the junction.
AMPLIFIER_GAIN = 1

#: Threshold to use to determine the critical current (in raw data units).
IC_VOLTAGE_THRESHOLD = 5e-6

#: Bias current at which we consider to be in the high bias regime and can fit
#: the resistance.
HIGH_BIAS_THRESHOLD = 18e-6