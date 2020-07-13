# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to automate analysing large set of data.

Data are expected to have been aggregated using DataClassifier.

"""
import logging
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from h5py import File, Group

from .data_exploring import DataExplorer


LOGGER = logging.getLogger(__name__)


@dataclass
class AnalysisStep:
    """Description of an analysis step to perform on data."""

    #: Name of the step identifying it.
    name: str

    #: Names of the quantities expected as input.
    input_quantities: List[str]

    #: External parameters. If the value of an external parameter should be extracted
    #: from a classifier value it should be specified as a string with the following
    #: format: "classifiers@{level}@{name}"
    parameters: Dict[str, Any]

    #: Callable responsible of carrying out the analysis. Should expect input
    #: quantities as positional args, parameters as keywords and also debug as keyword
    routine: Callable

    def __post_init__(self):
        # Validate the parameters format.
        for k, v in self.parameters:
            if isinstance(v, str) and v.startswith("classifiers@"):
                parts = v.split("@")[1:]
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid classifier value access. Got {v} expected "
                        "'classifiers@{{level}}@{{name}}' "
                        "with level the classifier level."
                    )

    def populate_parameters(self, classifiers):
        """Generate the parameters to pass to the routine."""
        p = parameters.copy()
        for k, v in self.parameters:
            if isinstance(v, str) and v.startswith("classifiers@"):
                parts = v.split("@")[1:]
                p[k] = classifiers[parts[0]][parts[1]]
        return p


# XXX Clean up the data and write in the duplicate (Cannot overwrite)
@dataclass
class PreProcessingStep(AnalysisStep):
    """Step responsible for cleaning up existing data (offset, smoothing, conversion)

    """

    #: List of measurement names on which this step should be applied to.
    measurements: List[str]

    #: Names of the output quantities, that will be saved next to the existing data.
    output_quantities: List[str]

    # Data are read and written in the same group
    def run(
        self, group: Group, classifiers: Dict[int, Dict[str, Any]], debug: bool = False
    ) -> None:
        """Run the routine and save the produce data.

        The name of the step and the parameters values are saved in the datasets attrs.

        """
        data = [group[k] for k in self.input_quantities]
        p = self.populate_parameters(classifiers)
        out = self.routine(*data, **p, debug=True)
        if len(out) != self.output_quantities:
            raise ValueError(
                f"Got {len(out)} output but {len(self.output_quantities)}."
            )
        for n, d in zip(self.output_quantities, out):
            dset = group.create_dataset(n, data=d)
            dset.attrs["__step_name__"] = self.name
            dset.attrs.update(self.parameters)


# XXX extract relevant quantities and write in new files
# (different measurement can be combined on the same level)
# Organized in tiers (example)
# - tier 0: extracted data
# - tier 1: results of fitting
@dataclass
class ProcessingStep(AnalysisStep):
    """Step processing data and saving teh result in a new file per tier.

    Data may require multiple processing step (extraction, fit, ...) which can be stored
    under different tiers.

    """

    #: Tier under which the output of the routine should be stored.
    tier: str

    #: From where should the input data be pulled.
    #: Currently support raw@{measurement} for data stored in the original data
    #: aggregate, processed@{tier} for data created by a previous processing step.
    input_origin: str

    #: Names of the output quantities, that will be saved in the specified tier under
    #: the appropriate classifiers.
    output_quantities: List[str]

    def run(
        self,
        origin: Group,
        classifiers: Dict[int, Dict[str, Any]],
        out_explorer: DataExplorer,
        debug: bool = False,
    ) -> None:
        """Run the routine and save the produce data.

        The name of the step and the parameters values are saved in the datasets attrs.

        """
        data = [origin[k] for k in self.input_quantities]
        p = self.populate_parameters(classifiers)
        out = self.routine(*data, **p, debug=True)
        if len(out) != self.output_quantities:
            raise ValueError(
                f"Got {len(out)} output but {len(self.output_quantities)}."
            )

        group = out_explorer.require_group(self.tier, classifiers)
        for n, d in zip(self.output_quantities, out):
            dset = group.create_dataset(n, data=d)
            dset.attrs["__step_name__"] = self.name
            dset.attrs.update(self.parameters)


@dataclass
class SummarizingStep(AnalysisStep):
    """Step producing plots or other summary.

    It is in charge of saving its output and as a consequence should expect an
    extra directory parameter in which it can save its output, and it will also
    directly be passed the classifiers.

    """

    #: From where should the input data be pulled.
    #: Currently support raw@{measurement} for data stored in the original data
    #: aggregate, processed@{tier} for data created by a previous processing step.
    input_origin: str

    def run(
        self,
        origin: Group,
        classifiers: Dict[int, Dict[str, Any]],
        directory: str,
        debug: bool = False,
    ) -> None:
        """Run the routine"""
        data = [origin[k] for k in self.input_quantities]
        p = self.populate_parameters(classifiers)
        # XXX could we use a nicer format.
        p["directory"] = directory
        p["classifiers"] = classifiers
        self.routine(*data, **p, debug=True)


# XXX add logging for debugging
@dataclass
class ProcessCoordinator:
    """Coordinate multiple steps of processing."""

    #: Path to the aggregated data to be used as input.
    archive_path: str

    #: Path into which to copy the original data to avoid corrupting them.
    duplicate_path: str

    #: Path in which to save the results of the processing steps.
    #: The directory should exist.
    processing_path: str

    #: Path to the directory in which summary steps can save their outputs.
    summary_directory: str

    #: List of pre-processing steps to run.
    preprocessing_steps: List[PreProcessingStep]

    #: List of processing steps to run.
    processing_steps: List[ProcessingStep]

    #: List of summary steps to run.
    summary_steps: List[SummarizingStep]

    def run_preprocess(self, debug: bool = False) -> None:
        """Run the pre-processing steps."""
        # Duplicate the data to avoid corrupting the original dataset
        shutil.copyfile(self.archive_path, self.duplicate_path)
        LOGGER.debug(f"Copied {self.archive_path} to {self.duplicate_path}")

        # Open new dataset
        with DataExplorer(self.duplicate_path, allow_edits=True) as data:

            # Iterate on pre-processing steps
            for step in self.preprocessing_steps:
                LOGGER.debug(f"Running pre-processing: {step.name}")

                for meas in step.measurements:
                    LOGGER.debug(f"    on {meas}")

                    # Walk data pertaining to the right measurement
                    for classifiers, group in data.walk_data(meas):
                        LOGGER.debug(f"        for {classifiers}")

                        # Apply pre-processing to each dataset
                        # The step takes care of saving data.
                        step.run(group, classifiers, debug)

    def run_process(self, debug: bool = False) -> None:
        """Run the processing steps."""
        # Open duplicate dataset and processing path
        with DataExplorer(self.duplicate_path, allow_edits=True) as data, DataExplorer(
            self.processing_path, create_new=True
        ) as f:

            # Iterate on processing steps
            for step in self.processing_steps:
                LOGGER.debug(f"Running processing: {step.name}")

                # Walk data pertaining to the right dataset and measurement/tier
                d, mt = step.input_origin.split("@")
                origin = data if d == "raw" else f

                # Apply processing to each dataset
                for classifiers, dset in origin.walk_data(mt):
                    LOGGER.debug(f"    for {classifiers}")
                    step.run(origin, classifiers, f, debug)

    def run_summary(self):
        pass
