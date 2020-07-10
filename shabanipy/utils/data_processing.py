# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines .

"""
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from h5py import File, Group

from .data_classifying import DataExplorer


@dataclass
class AnalysisStep:
    """
    """

    #:
    name: str

    #:
    input_quantities: List[str]

    #: External parameters
    parameters: Dict[str, Any]

    #:
    # Expect input quantities as positional args, parameters as keywords and
    # also debug as keyword
    routine: Callable

    def __post_init__(self):
        for k, v in self.parameters:
            if isinstance(v, str) and v.startswith("classifiers@"):
                parts = v.split("@")[1:]
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid classifier value access. Got {v} expected "
                        "'classifiers@level@name' with level is the classifier level."
                    )

    def populate_parameters(self, classifiers):
        """
        """
        p = parameters.copy()
        for k, v in self.parameters:
            if isinstance(v, str) and v.startswith("classifiers@"):
                parts = v.split("@")[1:]
                p[k] = classifiers[parts[0]][parts[1]]
        return p


# XXX Clean up the data and write in the duplicate (Cannot overwrite)
@dataclass
class PreProcessingStep(AnalysisStep):
    """
    """

    #:
    measurements: List[str]

    #:
    output_quantities: List[str]

    # Data are read and written in the same group
    def run(
        self, group: Group, classifiers: Dict[int, Dict[str, Any]], debug: bool = False
    ) -> None:
        """
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
    """
    """

    #:
    tier: str

    #:
    # raw@measurement, processed@tier
    input_origin: str

    #:
    output_quantities: List[str]

    def run(
        self,
        origin: Group,
        classifiers: Dict[int, Dict[str, Any]],
        out_explorer: DataExplorer,
        debug: bool = False,
    ) -> None:
        """
        """
        # XXX need to access the classifiers (need frequency for Shapiro)
        data = [origin[k] for k in self.input_quantities]
        p = self.populate_parameters(classifiers)
        out = self.routine(*data, **p, debug=True)
        if len(out) != self.output_quantities:
            raise ValueError(
                f"Got {len(out)} output but {len(self.output_quantities)}."
            )

        group = out_explorer.create_group(self.tier, classifiers)
        for n, d in zip(self.output_quantities, out):
            dset = group.create_dataset(n, data=d)
            dset.attrs["__step_name__"] = self.name
            dset.attrs.update(self.parameters)


# XXX generation of plots from extracted quantities
@dataclass
class SummarizingStep(AnalysisStep):
    """
    """

    pass


# XXX add logging for debugging
@dataclass
class ProcessCoordinator:

    #:
    archive_path: str

    #:
    duplicate_path: str

    #:
    processing_path: str

    #:
    preprocessing_steps: List[PreProcessingStep]

    #:
    processing_steps: List[ProcessingStep]

    #:
    summary_steps: List[SummarizingStep]

    def run_preprocess(self, debug: bool = False) -> None:
        """
        """
        # Duplicate the data to avoid corrupting the original dataset
        shutil.copyfile(self.archive_path, self.duplicate_path)

        # Open new dataset
        with DataExplorer(self.duplicate_path, allow_edits=True) as data:

            # Iterate on pre-processing steps
            for step in self.preprocessing_steps:

                for meas in step.measurements:

                    # Walk data pertaining to the right measurement
                    for classifiers, group in data.walk_data(meas):

                        # Apply pre-processing to each dataset
                        # The step takes care of saving data.
                        step.run(group, classifiers, debug)

    def run_process(self, debug: bool = False) -> None:
        """
        """
        # Open duplicate dataset and processing path
        with DataExplorer(self.duplicate_path, allow_edits=True) as data, DataExplorer(
            self.processing_path, create_new=True
        ) as f:

            # Iterate on processing steps
            for step in self.processing_steps:

                # Walk data pertaining to the right dataset and measurement/tier
                d, mt = step.input_origin.split("@")
                origin = data if d == "raw" else f

                # Apply processing to each dataset
                for classifiers, dset in origin.walk_data(mt):
                    step.run(origin, classifiers, f, debug)

    def run_summary(self):
        pass
