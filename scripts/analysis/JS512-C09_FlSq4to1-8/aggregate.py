"""Aggregate data for JS512-C09_FlSq4to1-8.

(Shapiro data exists but not included here.)

Wafer:      JS512
Piece:      C09
Layout:     FlSq4to1-8.gds
Devices:    FlSq4toX (where X in [1, 2, 3, 4, 5, 6, 7, 8])
Cooldowns:  WFS-01 (vector9), WFS02 (vector10)

Usage:
    $ python aggregate.py [MeasurementPattern name] [MeasurementPattern name] [...]
If no MeasurementPattern names are specified, all will run.
"""

import os
import re
import sys
from pathlib import Path

from shabanipy.bulk.data_classifying import (
    DataClassifier,
    FilenamePattern,
    InstrumentConfigPattern,
    LogPattern,
    MeasurementPattern,
    NamePattern,
    RampPattern,
    StepPattern,
    ValuePattern,
)
from shabanipy.labber import get_data_dir
from shabanipy.logging import configure_logging

configure_logging()

DATA_DIR = get_data_dir()
SAMPLE_ID = "JS512-C09_FlSq4to1-8"
SEARCH_PATHS = [
    (Path(DATA_DIR) / "vector9/2021/06").expanduser().absolute().as_posix(),
    (Path(DATA_DIR) / "vector9/2021/07").expanduser().absolute().as_posix(),
    (Path(DATA_DIR) / "vector10/2021/08").expanduser().absolute().as_posix(),
    (Path(DATA_DIR) / "vector10/2021/09").expanduser().absolute().as_posix(),
]
OUTPUT_FILEPATH = Path("data_aggregated.hdf5").as_posix()

filename_pattern = FilenamePattern(
    regex=f"{re.escape(SAMPLE_ID)}_(?P<device>[^\W_]+).*?(?P<scan>WFS-?0\d[-_]\d\d\d)\.hdf5$",
    use_in_classification=True,
    classifier_level={"device": 2, "scan": sys.maxsize},
)

# steps
yoko_step = StepPattern(
    name="dc bias (volts)",
    name_pattern=NamePattern(regex="^Yoko - Voltage$|^dc bias - Voltage"),
)
x_magnet_step = StepPattern(
    name="x magnet current",
    name_pattern=NamePattern(regex="^[Xx]( -)? [Mm]agnet( Source)? - Source current"),
)
y_magnet_step = StepPattern(
    name="y magnet field", name_pattern=NamePattern(regex=".* - Field Y$")
)
gate4um_step = StepPattern(
    name="4um gate voltage",
    name_pattern=NamePattern(regex="(4um gate|SM3) - Source voltage"),
)
gateXum_step = StepPattern(
    name="Xum gate voltage",
    name_pattern=NamePattern(regex="([1-35-9]um gate|SM2) - Source voltage"),
)
fluxline_current_step = StepPattern(
    name="fluxline current bias",
    name_pattern=NamePattern(regex="SM1 - Source current"),
)
fluxline_voltage_step = StepPattern(
    name="fluxline voltage", name_pattern=NamePattern(regex="SM1 - Source voltage")
)
temp_step = StepPattern(
    name="temperature setpoint", name_pattern=NamePattern(regex=".*target temperature$")
)

# instrument configs
y_magnet_config = InstrumentConfigPattern(
    name="[Vv]ector [Mm]agnet", quantity="Field Y"
)
gate4um_config = InstrumentConfigPattern(name="4um gate|SM3", quantity="Source voltage")
gateXum_config = InstrumentConfigPattern(
    name="[1-35-9]um gate|SM2", quantity="Source voltage"
)
temp_config = InstrumentConfigPattern(
    name="[Ff]ridge|Vector Fridge",
    quantity="Temperature closed loop - target temperature",
)

# fixed steps
y_magnet_fixed = y_magnet_step.copy_with(
    fallback=y_magnet_config, use_in_classification=True, classifier_level=3,
)
gate4um_fixed = gate4um_step.copy_with(
    fallback=gate4um_config, use_in_classification=True, classifier_level=4,
)
gateXum_fixed = gateXum_step.copy_with(
    fallback=gateXum_config, use_in_classification=True, classifier_level=4,
)
temp_fixed = temp_step.copy_with(
    fallback=temp_config, use_in_classification=True, classifier_level=3
)

# logs
dmm_vec_log = LogPattern(
    name="dmm", pattern=NamePattern(regex=".*- VI curve.*"), x_name="dc current bias",
)
dmm_log = LogPattern(name="dmm", pattern=NamePattern(regex=".*- SingleValue$"),)
lockin_vec_log = LogPattern(
    name="lock-in (impedance)",
    pattern=NamePattern(regex=".*- dR vs I curve"),
    x_name="dc current bias",
    is_required=False,
)
lockin_log = LogPattern(
    name="lock-in (volts)", pattern=NamePattern(regex=".* - Value$"), is_required=False,
)
gate4um_log = LogPattern(
    name="4um gate current", pattern=NamePattern(regex="(4um gate|SM3) - Current")
)
gateXum_log = LogPattern(
    name="Xum gate current",
    pattern=NamePattern(regex="([1-35-9]um gate|SM2) - Current"),
)
fluxline_log = LogPattern(
    name="fluxline current measured",
    pattern=NamePattern(regex="SM1 - Current"),
    is_required=False,
)
temp_log = LogPattern(
    name="mc temperature",
    pattern=NamePattern(regex=".*MC-RuOx-Temperature$"),
    is_required=False,
)

measurements = [
    MeasurementPattern(
        name="4um gate leakage",
        filename_pattern=filename_pattern,
        patterns=[
            gate4um_step.copy_with(
                ramps=[RampPattern(start=0, span=ValuePattern(greater=10))]
            ),
            gateXum_fixed,
            gate4um_log,
            gateXum_log,
            fluxline_log,
        ],
    ),
    MeasurementPattern(
        name="Xum gate leakage",
        filename_pattern=filename_pattern,
        patterns=[
            gateXum_step.copy_with(
                ramps=[RampPattern(start=0, span=ValuePattern(greater=10))]
            ),
            gate4um_fixed,
            gate4um_log,
            gateXum_log,
            fluxline_log,
        ],
    ),
    MeasurementPattern(
        name="4um gate vs bias",  # with curvetracer
        filename_pattern=filename_pattern,
        patterns=[
            y_magnet_fixed,
            gate4um_step.copy_with(ramps=[RampPattern(points=ValuePattern(greater=5))]),
            gateXum_fixed,
            temp_fixed,
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="4um gate vs bias without curvetracer",
        filename_pattern=filename_pattern,
        patterns=[
            yoko_step.copy_with(ramps=[RampPattern(points=ValuePattern(greater=10))]),
            y_magnet_fixed,
            gate4um_step.copy_with(ramps=[RampPattern(points=ValuePattern(greater=5))]),
            gateXum_fixed,
            temp_fixed,
            dmm_log,
            lockin_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="Xum gate vs bias",  # with curvetracer
        filename_pattern=filename_pattern,
        patterns=[
            y_magnet_fixed,
            gateXum_step.copy_with(ramps=[RampPattern(points=ValuePattern(greater=5))]),
            gate4um_fixed,
            temp_fixed,
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="Xum gate vs bias without curvetracer",
        filename_pattern=filename_pattern,
        patterns=[
            yoko_step.copy_with(ramps=[RampPattern(points=ValuePattern(greater=10))]),
            y_magnet_fixed,
            gateXum_step.copy_with(ramps=[RampPattern(points=ValuePattern(greater=5))]),
            gate4um_fixed,
            temp_fixed,
            dmm_log,
            lockin_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="fraunhofer",
        filename_pattern=filename_pattern,
        patterns=[
            x_magnet_step.copy_with(
                ramps=[RampPattern(span=ValuePattern(greater=20e-3))]
            ),
            y_magnet_fixed,
            gate4um_fixed.copy_with(default="NC"),
            gateXum_fixed.copy_with(default="NC"),
            temp_fixed,
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="fraunhofer without curvetracer",
        filename_pattern=filename_pattern,
        patterns=[
            yoko_step.copy_with(ramps=[RampPattern()]),
            x_magnet_step.copy_with(
                ramps=[RampPattern(span=ValuePattern(greater=20e-3))]
            ),
            y_magnet_fixed,
            gate4um_fixed.copy_with(default="NC"),
            gateXum_fixed.copy_with(default="NC"),
            temp_fixed,
            dmm_log,
            lockin_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="in-plane vs bias",
        filename_pattern=filename_pattern,
        patterns=[
            y_magnet_step.copy_with(
                ramps=[RampPattern(points=ValuePattern(greater=20))]
            ),
            gate4um_fixed,
            gateXum_fixed,
            temp_fixed,
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="in-plane vs bias without curvetracer",
        filename_pattern=filename_pattern,
        patterns=[
            yoko_step.copy_with(ramps=[RampPattern()]),
            y_magnet_step.copy_with(
                ramps=[RampPattern(points=ValuePattern(greater=20))]
            ),
            gate4um_fixed,
            gateXum_fixed,
            temp_fixed,
            dmm_log,
            lockin_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="cpr",
        filename_pattern=filename_pattern,
        patterns=[
            x_magnet_step.copy_with(
                ramps=[RampPattern(span=ValuePattern(smaller=20e-3))]
            ),
            y_magnet_fixed,
            gate4um_fixed,
            gateXum_fixed,
            temp_fixed,
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="cpr without curvetracer",
        filename_pattern=filename_pattern,
        patterns=[
            yoko_step.copy_with(ramps=[RampPattern()]),
            x_magnet_step.copy_with(
                ramps=[RampPattern(span=ValuePattern(smaller=20e-3))]
            ),
            y_magnet_fixed,
            gate4um_fixed,
            gateXum_fixed,
            temp_fixed,
            dmm_log,
            lockin_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="frequency vs amplitude",
        filename_pattern=filename_pattern,
        patterns=[
            StepPattern(
                name="lock-in frequency",
                name_pattern=NamePattern(regex="vicurve - Lock-in: frequency"),
            ),
            StepPattern(
                name="lock-in amplitude",
                name_pattern=NamePattern(regex="vicurve - Lock-in: amplitude"),
            ),
            y_magnet_fixed,
            gate4um_fixed,
            gateXum_fixed,
            temp_fixed,
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="hysteresis",
        filename_pattern=filename_pattern,
        patterns=[
            yoko_step.copy_with(ramps=[RampPattern()]),
            y_magnet_fixed,
            gate4um_fixed,
            gateXum_fixed,
            temp_fixed,
            # used as dummy variable to alternate dc current bias sweep direction
            StepPattern(
                name="sweep direction", name_pattern=NamePattern(regex=".*Sensitivity$")
            ),
            dmm_log,
            lockin_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="Rn",
        filename_pattern=filename_pattern,
        patterns=[
            InstrumentConfigPattern(
                name="[Vv][Ii][Cc]urve.*",
                quantity="DMM: number of points",
                value=ValuePattern(greater=500),
            ),
            gate4um_fixed,
            gateXum_fixed,
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="Rn without curvetracer",
        filename_pattern=filename_pattern,
        patterns=[
            yoko_step.copy_with(ramps=[RampPattern(points=ValuePattern(greater=500))]),
            gate4um_fixed,
            gateXum_fixed,
            dmm_log,
            lockin_log,
            temp_log,
        ],
    ),
    MeasurementPattern(
        name="Bc",
        filename_pattern=filename_pattern,
        patterns=[
            StepPattern(
                name="z magnet field",
                name_pattern=NamePattern(regex="^vector magnet - Field Z$"),
                ramps=[RampPattern(stop=ValuePattern(greater=1))],
            ),
            lockin_log,
        ],
    ),
    MeasurementPattern(
        name="fluxline fraunhofer",
        filename_pattern=filename_pattern,
        patterns=[
            fluxline_current_step.copy_with(ramps=[RampPattern()]),
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
            fluxline_log,
            gate4um_fixed,
            gateXum_fixed,
        ],
    ),
    MeasurementPattern(
        name="fluxline (voltage source) fraunhofer",
        filename_pattern=filename_pattern,
        patterns=[
            fluxline_voltage_step.copy_with(ramps=[RampPattern()]),
            dmm_vec_log,
            lockin_vec_log,
            temp_log,
            fluxline_log,
            gate4um_fixed,
            gateXum_fixed,
        ],
    ),
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dc = DataClassifier(
            patterns=[m for m in measurements if m.name in sys.argv[1:]]
        )
    else:
        dc = DataClassifier(patterns=measurements)

    dc.identify_datasets(SEARCH_PATHS)
    dc.classify_datasets()
    dc.consolidate_dataset(OUTPUT_FILEPATH)
