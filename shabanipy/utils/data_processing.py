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
import os
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from h5py import Dataset, Group

from .data_exploring import DataExplorer

logger = logging.getLogger(__name__)


@dataclass
class AnalysisStep:
    """Description of an analysis step to perform on data."""

    #: Name of the step identifying it.
    name: str

    #: Names of the quantities expected as input.
    input_quantities: List[str]

    #: External parameters.
    #: If the value of an external parameter should be extracted from a classifier
    #: value it should be specified as a string with the following format:
    #: "classifiers@{level}@{name}"
    #: If a parameter should be read from data stored in a parent group with a more
    #: generic classifier it should be indicated as: "data@{name}" and the first
    #: matching name will be used.
    # If the parameter should be obtained from an attribute on an hdf5 dataset, use
    # "attrs@{name}".
    parameters: Dict[str, Any]

    #: Callable responsible of carrying out the analysis. Should expect input
    #: quantities as positional args, parameters as keyword args.
    #: The returned value should always be a tuple or a list even when a single value
    #: is returned
    routine: Callable

    def __post_init__(self):
        # Validate the parameters format.
        for k, v in self.parameters.items():
            if isinstance(v, str):
                if v.startswith("classifiers@"):
                    parts = v.split("@")[1:]
                    if len(parts) != 2:
                        raise ValueError(
                            f"Invalid classifier value access. Got {v} expected "
                            "'classifiers@{{level}}@{{name}}' "
                            "with level the classifier level."
                        )
                    try:
                        int(parts[0])
                    except Exception as e:
                        raise ValueError(
                            "Invalid classifier value access. Could not convert "
                            f"the provided level: {parts[0]} to an integer"
                        )
                if v.startswith("data@"):
                    parts = v.split("@")[1:]
                    if len(parts) != 1:
                        raise ValueError(
                            f"Invalid generic data value access. Got {v} expected "
                            "'data@{{name}}'."
                        )
                if v.startswith("attrs@"):
                    parts = v.split("@")[1:]
                    if len(parts) != 1:
                        raise ValueError(
                            f"Malformed attribute parameter. Got '{v}', expected 'attrs@{{name}}'"
                        )

    def populate_parameters(self, group, classifiers):
        """Generate the parameters to pass to the routine."""
        p = self.parameters.copy()
        for k, v in self.parameters.items():
            if isinstance(v, str):
                if v.startswith("classifiers@"):
                    parts = v.split("@")[1:]
                    p[k] = classifiers[int(parts[0])][parts[1]]
                if v.startswith("data@"):
                    name = v.split("@")[1]
                    g = group
                    while True:
                        if name in g:
                            p[k] = g[name]
                            break
                        if g.parent.name == g.name:
                            raise RuntimeError(
                                f"Could not find any dataset named {name}"
                            )
                        g = g.parent
                if v.startswith("attrs@"):
                    name = v.split("@")[1]
                    # for now we assume all input quantity datasets have the requested
                    # attribute
                    params = [
                        group[dset].attrs[name]
                        for dset in self.input_quantities
                        if name in group[dset].attrs
                    ]
                    if len(set(params)) > 1:
                        raise RuntimeError(
                            f"Multiple unique values found for attribute '{name}' of datasets in group ({group.name})"
                        )
                    p[k] = params[0]

        return p


@dataclass
class PreProcessingStep(AnalysisStep):
    """Step responsible for cleaning up existing data (offset, smoothing, conversion)

    """

    #: List of measurement names on which this step should be applied to.
    measurements: List[str]

    #: Names of the output quantities, that will be saved next to the existing data.
    output_quantities: List[str]

    # Data are read and written in the same group
    def run(self, group: Group, classifiers: Dict[int, Dict[str, Any]]) -> None:
        """Run the routine and save the produce data.

        The name of the step and the parameters values are saved in the datasets attrs.

        """
        # Get actual numpy arrays as otherwise the differences may be slightly surprising
        data = [group[k][...] for k in self.input_quantities]
        p = self.populate_parameters(group, classifiers)
        out = self.routine(*data, **p)
        if len(self.output_quantities) > 1 and len(out) != len(self.output_quantities):
            raise ValueError(
                f"Got {len(out)} output but {len(self.output_quantities)}."
            )
        elif len(self.output_quantities) == 1:
            out = (out,)
        for n, d in zip(self.output_quantities, out):
            dset = group.create_dataset(n, data=d, compression="gzip")
            dset.attrs["__step_name__"] = self.name
            dset.attrs.update(p)


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
    ) -> None:
        """Run the routine and save the produce data.

        The name of the step and the parameters values are saved in the datasets attrs.

        """
        data = [origin[k][...] for k in self.input_quantities]
        p = self.populate_parameters(origin, classifiers)
        out = self.routine(*data, **p)
        if len(self.output_quantities) > 1 and len(out) != len(self.output_quantities):
            raise ValueError(
                f"Got {len(out)} outputs but expected {self.output_quantities}."
            )
        elif len(self.output_quantities) == 1:
            out = (out,)

        group = out_explorer.require_group(self.tier, classifiers)

        for n, d in zip(self.output_quantities, out):
            dset = group.create_dataset(n, data=d)
            """If running with 2d array (bias in forward and backward directions) you 
            can use compression="gzip" """
            #dset = group.create_dataset(n, data=d, compression="gzip")
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

    #: Whether to create a subdirectory matching the step name in the overall
    #: subdirectory and pass it to the routine.
    use_named_subdirectory: bool = True

    def run(
        self, origin: Group, classifiers: Dict[int, Dict[str, Any]], directory: str,
    ) -> None:
        """Run the routine"""
        data = [origin[k][...] for k in self.input_quantities]
        p = self.populate_parameters(origin, classifiers)
        dir_path = (
            os.path.join(directory, self.name)
            if self.use_named_subdirectory
            else directory
        )
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        p["directory"] = dir_path
        p["classifiers"] = classifiers
        self.routine(*data, **p)


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

    # XXX add ways to run by names, tiers
    def run_preprocess(self) -> None:
        """Run the pre-processing steps."""
        logger.info("preprocessing...")
        # Duplicate the data to avoid corrupting the original dataset
        logger.debug(f"Copying {self.archive_path} to {self.duplicate_path}")
        shutil.copyfile(self.archive_path, self.duplicate_path)
        logger.debug(f"Copied {self.archive_path} to {self.duplicate_path}")

        # Open new dataset
        with DataExplorer(self.duplicate_path, allow_edits=True) as data:

            # Iterate on pre-processing steps
            for step in self.preprocessing_steps:
                logger.debug(f"Running pre-processing: {step.name}")

                for meas in step.measurements:
                    logger.debug(f"    on {meas}")

                    # Walk data pertaining to the right measurement
                    for classifiers, group in data.walk_data(meas):
                        logger.debug(f"        for {classifiers}")

                        # Apply pre-processing to each dataset
                        # The step takes care of saving data.
                        step.run(group, classifiers)
        logger.info(f"...preprocessed data saved to {self.duplicate_path}")

    def run_process(self) -> None:
        """Run the processing steps."""
        logger.info("processing...")
        # Open duplicate dataset and processing path
        with DataExplorer(self.duplicate_path, allow_edits=True) as data, DataExplorer(
            self.processing_path, create_new=True
        ) as f:

            # Iterate on processing steps
            for step in self.processing_steps:
                logger.debug(f"Running processing: {step.name}")

                # Walk data pertaining to the right dataset and measurement/tier
                d, mt = step.input_origin.split("@")
                origin = data if d == "raw" else f

                # Apply processing to each dataset
                for classifiers, group in origin.walk_data(mt):
                    # Skip data set that do not contain the relevant quantities
                    # Possible when merging different measurement producing data
                    # at different classifiers level.
                    if not all(
                        bool(in_name in group) for in_name in step.input_quantities
                    ):
                        logger.debug(f"    skipped for {classifiers}")
                        continue
                    logger.debug(f"    for {classifiers}")
                    step.run(group, classifiers, f)
        logger.info(f"processed data saved to {self.processing_path}")

    def run_summary(self) -> None:
        """Run the summary steps."""
        logger.info("summarizing...")
        # Open duplicate dataset and processing path
        with DataExplorer(self.processing_path) as f, DataExplorer(
            self.duplicate_path
        ) as data:

            # Iterate on summary steps
            for step in self.summary_steps:
                logger.debug(f"Running summary: {step.name}")

                # Walk data pertaining to the right dataset and measurement/tier
                d, mt = step.input_origin.split("@")
                origin = data if d == "raw" else f

                # Apply summary to each dataset
                for classifiers, group in origin.walk_data(mt):
                    # Skip data set that do not contain the relevant quantities
                    # Possible when merging different measurement producing data
                    # at different classifiers level.
                    if not all(
                        bool(in_name in group) for in_name in step.input_quantities
                    ):
                        logger.debug(
                            f"    skipped for {classifiers}, "
                            f"looking for {step.input_quantities}, found "
                            f"{[n for n in group if isinstance(group[n], Dataset)]}."
                        )
                        continue
                    logger.debug(f"    for {classifiers}")
                    step.run(group, classifiers, self.summary_directory)
        logger.info(f"summarized data saved to directory {self.summary_directory}")
