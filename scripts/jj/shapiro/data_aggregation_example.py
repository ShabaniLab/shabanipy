# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Example of using data classification tools for Shapiro experiments.

"""
#: List of path to inspect looking for raw data
FOLDERS = []

#: Name of the sample that must appear in the measurement name.
SAMPLE_NAME = ""

#: Path under which saving the consolidated data.
CONSOLIDATION_PATH = ""

#: Path to a file in which to store teh log of teh aggregation
LOG_PATH = ""

import logging
import os

import numpy as np
import h5py

from shabanipy.utils.data_classifying import (
    DataClassifier,
    FilenamePattern,
    LogPattern,
    MeasurementPattern,
    NamePattern,
    RampPattern,
    StepPattern,
    ValuePattern,
)

# Parametrize the logging system
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(None)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
ch = logging.FileHandler(LOG_PATH, mode="w")
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


# --- Configuation ---------------------------------------------------------------------

# Get large bias measurements and Shapiro with a minimum number of points in power

clf = DataClassifier(
    patterns=[
        MeasurementPattern(
            name="IcRn",
            filename_pattern=FilenamePattern(
                regex=f"^.+-(?P<sample>{SAMPLE_NAME})-.+$", use_in_classification=True
            ),
            steps=[
                StepPattern(
                    name="Bias",
                    name_pattern=0,
                    ramps=[RampPattern(span=ValuePattern(greater=50))],
                ),
                StepPattern(
                    name="Counter",
                    name_pattern=1,
                    ramps=[RampPattern(points=ValuePattern(value=2))],
                ),
                StepPattern(
                    name="Gate",
                    name_pattern=NamePattern(names=["Keithley2 - Source voltage"]),
                    use_in_classification=True,
                    classifier_level=1,
                ),
                StepPattern(
                    name="Field",
                    name_pattern=NamePattern(names=["Vector Magnet - Field X"]),
                    use_in_classification=True,
                    classifier_level=1,
                ),
            ],
            logs=[LogPattern(name="Voltage", pattern=0)],
        ),
        MeasurementPattern(
            name="Shapiro",
            filename_pattern=FilenamePattern(
                regex=f"^.+-(?P<sample>{SAMPLE_NAME})-.+$", use_in_classification=True
            ),
            steps=[
                StepPattern(name="Bias", name_pattern=0, ramps=[RampPattern()],),
                StepPattern(
                    name="RF ON",
                    name_pattern=NamePattern(names=["BNC - Output"]),
                    value=ValuePattern(value=1),
                ),
                StepPattern(
                    name="RF Power",
                    name_pattern=NamePattern(names=["BNC - Power"]),
                    ramps=[RampPattern(points=ValuePattern(greater=20))],
                ),
                StepPattern(
                    name="Gate",
                    name_pattern=NamePattern(names=["Keithley2 - Source voltage"]),
                    use_in_classification=True,
                    classifier_level=1,
                ),
                StepPattern(
                    name="Field",
                    name_pattern=NamePattern(names=["Vector Magnet - Field X"]),
                    use_in_classification=True,
                    classifier_level=1,
                ),
                StepPattern(
                    name="Frequency",
                    name_pattern=NamePattern(names=["BNC - Frequency"]),
                    use_in_classification=True,
                    classifier_level=2,
                ),
            ],
            logs=[LogPattern(name="Voltage", pattern=0)],
        ),
    ]
)

# --- Run ------------------------------------------------------------------------------

clf.identify_datasets(FOLDERS)

for n, datasets in clf._datasets.items():
    print(f"Measurement: {n}")
    for d in datasets:
        print(f"  - {d.rsplit(os.sep, 1)[-1]}")

clf.classify_datasets()
for n, ds in clf._classified_datasets.items():
    print(f"Measurement: {n}")
    for path, classifiers in ds.items():
        print(f"  - {d.rsplit(os.sep, 1)[-1]}")
        for level, values in classifiers.items():
            print(f"    - level {level}: {values}")

clf.consolidate_dataset(CONSOLIDATION_PATH)


def pretty_print(root: h5py.Group, increment=""):
    for k, v in root.items():
        if isinstance(v, h5py.Group):
            print(increment + f"- Group {k}:")
            print(increment + f"  {dict(v.attrs)}")
            pretty_print(v, increment + "  ")
        elif isinstance(v, h5py.Dataset):
            print(
                increment + f"- Dataset {k}: shape {v.shape}, dtype {v.dtype}, "
                f"min {np.min(v[...])}, max {np.min(v[...])}"
            )


with h5py.File(CONSOLIDATION_PATH, "r") as f:
    pretty_print(f)
