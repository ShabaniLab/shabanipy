# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Example of using data classification tools for Shapiro experiments.

"""
#: Path the raw data agglomerate
RAW_DATA = "/Users/mdartiailh/Documents/Coding/temp/test_06.hdf5"

#: Path into which to duplicate the raw data before starting preprocessing.
DUPLICATION_PATH = ""

#: Path into which to save the processed data output.
PROCESSING_PATH = ""

#: Directory in which to save the summary plots.
SUMMARY_DIRECTORY = ""

#: Path to a file in which to store teh log of teh aggregation
LOG_PATH = ""

import logging
import os

import h5py
import matplotlib
from shabanipy.jj.iv_analysis import analyse_vi_curve
from shabanipy.jj.shapiro.binning import bin_power_shapiro_steps
from shabanipy.jj.shapiro.plotting import (
    plot_differential_resistance_map,
    plot_shapiro_histogram,
    plot_step_weights,
)
from shabanipy.jj.shapiro.utils import correct_voltage_offset_per_power
from shabanipy.jj.utils import compute_resistance, correct_voltage_offset
from shabanipy.bulk.data_processing import (
    PreProcessingStep,
    ProcessCoordinator,
    ProcessingStep,
    SummarizingStep,
)

matplotlib.use("Agg")


# Parametrize the logging system
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(None)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
ch = logging.FileHandler(LOG_PATH)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# Avoid matplotlib debug messages
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

# Get large bias measurements and Shapiro with a minimum number of points in power
coor = ProcessCoordinator(
    archive_path=RAW_DATA,
    duplicate_path=DUPLICATION_PATH,
    processing_path=PROCESSING_PATH,
    summary_directory=SUMMARY_DIRECTORY,
    preprocessing_steps=[
        PreProcessingStep(
            name="bias_scaling",
            measurements=["IcRn", "Shapiro"],
            input_quantities=["Bias"],
            parameters={"scaling": 1e-6},
            routine=lambda bias, scaling, debug: bias * scaling,
            output_quantities=["ScaledBias"],
        ),
        PreProcessingStep(
            name="voltage_scaling",
            measurements=["IcRn", "Shapiro"],
            input_quantities=["Voltage"],
            parameters={"scaling": 1e-2},
            routine=lambda volt, scaling, debug: volt * scaling,
            output_quantities=["ScaledVoltage"],
        ),
        PreProcessingStep(
            name="voltage_offset_coarse",
            measurements=["IcRn", "Shapiro"],
            input_quantities=["ScaledBias", "ScaledVoltage"],
            parameters={"index": 0, "n_peak_width": 3},
            routine=correct_voltage_offset,
            output_quantities=["VoltageCoarseOffset"],
        ),
        PreProcessingStep(
            name="voltage_offset_fine",
            measurements=["Shapiro"],
            input_quantities=["RF Power", "ScaledBias", "VoltageCoarseOffset"],
            parameters={
                "frequency": "classifiers@2@Frequency",
                "n_peak_width": 3,
                "n_std_as_bin": 2,
            },
            routine=correct_voltage_offset_per_power,
            output_quantities=["VoltageFineOffset"],
        ),
    ],
    processing_steps=[
        ProcessingStep(
            name="ic_rn",
            input_origin="raw@IcRn",
            input_quantities=["ScaledBias", "VoltageCoarseOffset"],
            parameters={"ic_voltage_threshold": 5e-6, "high_bias_threshold": 29e-6},
            routine=analyse_vi_curve,
            tier="0",
            output_quantities=["Rn_c", "Rn_h", "Ic_c", "Ic_h", "Ie_c", "Ie_h"],
        ),
        # Copy the RF power over the processed quantity since it is needed for
        # summary
        ProcessingStep(
            name="copy_rf_power",
            input_origin="raw@Shapiro",
            input_quantities=["RF Power"],
            parameters={},
            routine=lambda rf, debug: rf,
            tier="0",
            output_quantities=["RF Power"],
        ),
        ProcessingStep(
            name="derivative",
            input_origin="raw@Shapiro",
            input_quantities=["ScaledBias", "VoltageFineOffset"],
            parameters={},
            routine=compute_resistance,
            tier="0",
            output_quantities=["DifferentialResistance_Bias", "DifferentialResistance"],
        ),
        ProcessingStep(
            name="shapiro_binning",
            input_origin="raw@Shapiro",
            input_quantities=["RF Power", "ScaledBias", "VoltageFineOffset"],
            parameters={"frequency": "classifiers@2@Frequency", "step_fraction": 0.1},
            routine=bin_power_shapiro_steps,
            tier="0",
            output_quantities=["VoltageBins", "HistogramCounts"],
        ),
    ],
    summary_steps=[
        SummarizingStep(
            name="plot_resistance",
            input_quantities=[
                "RF Power",
                "DifferentialResistance_Bias",
                "DifferentialResistance",
            ],
            parameters={},
            routine=plot_differential_resistance_map,
            tier="0",
            use_named_subdirectory=True,
        ),
        SummarizingStep(
            name="plot_histo",
            input_quantities=["RF Power", "VoltageBins", "HistogramCounts",],
            parameters={"mark_steps": [0, -1, -2], "mark_steps_limit": 0.2,},
            routine=plot_shapiro_histogram,
            tier="0",
            use_named_subdirectory=True,
        ),
        SummarizingStep(
            name="plot_weights",
            input_quantities=["RF Power", "VoltageBins", "HistogramCounts",],
            parameters={"steps": [0, -1, -2], "ic": "data@Ic_h", "rn": "data@Rn_h",},
            routine=plot_step_weights,
            tier="0",
            use_named_subdirectory=True,
        ),
    ],
)

coor.run_preprocess()

coor.run_process()

coor.run_summary()
