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
import logging
import os
import re
from collections import defaultdict, namedtuple
from dataclasses import astuple, dataclass, field, fields
from itertools import product
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from h5py import File, Group

from .labber_io import LabberData, LogEntry, StepConfig

logger = logging.getLogger(__name__)


class Classifier(NamedTuple):
    """
    """

    #:
    column_name: Optional[str]

    #:
    values: tuple

    #:
    requires_filtering: bool = False


@dataclass
class NamePattern:
    """[summary]
    """

    #:
    names: Optional[List[str]] = None

    #:
    regex: Optional[str] = None

    def match(self, name: str) -> bool:
        match = False
        if self.names is not None:
            match |= name in self.names
        if self.regex is not None:
            match |= bool(re.match(self.regex, name))

        return match


@dataclass
class FilenamePattern(NamePattern):
    """[summary]
    """

    #:
    use_in_classification: bool = False

    #:
    classifier_name: str = ""

    #:
    classifier_level: int = 0

    def match(self, name: str) -> bool:
        match = False
        if self.names is not None:
            match |= name in self.names
        if self.regex is not None:
            match |= bool(re.match(self.regex, name))

        return match

    def extract(self, name: str) -> Optional[Dict[str, Classifier]]:
        """
        """
        if self.regex is None:
            return None

        match = re.match(self.regex, name)
        if not match:
            raise ValueError(f"The provided name: {name}, does not match: {self.regex}")

        d = match.groupdict()
        if d:
            return {k: Classifier(None, (v,)) for k, v in d.items()}

        try:
            return {self.classifier_name: Classifier(None, (match.group(1),))}
        except IndexError:
            return None


@dataclass(init=True)
class ValuePattern:
    """
    """

    #:
    value: Optional[float] = None

    #:
    value_set: Optional[Set[float]] = None

    #:
    greater: Optional[float] = None

    #:
    smaller: Optional[float] = None

    #:
    strict_comparisons: bool = False

    def __post_init__(self):
        if all(getattr(self, f.name) is None for f in fields(self)):
            raise ValueError("At least one field of ValuePattern has to be non None")

    def match(self, value: float) -> bool:
        match = False
        if self.value:
            match |= value == self.value
        if self.value_set:
            match |= value in self.value_set
        if self.smaller:
            match |= (
                value < self.smaller
                if self.strict_comparisons
                else value <= self.smaller
            )
        if self.greater:
            match |= (
                value > self.greater
                if self.strict_comparisons
                else value >= self.greater
            )
        return match


@dataclass(init=True)
class RampPattern:
    """[summary]

    """

    #:
    start: Optional[ValuePattern] = None

    #:
    stop: Optional[ValuePattern] = None

    #:
    span: Optional[ValuePattern] = None

    #:
    points: Optional[ValuePattern] = None

    def __post_init__(self):
        # Turn dict into the proper class
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, dict):
                setattr(self, f.name, f.type.__args__[0](**value))

    @property
    def is_generic(self):
        """ """
        return not any(getattr(self, f.name) for f in fields(self))

    def match(self, start: float, stop: float, points: int) -> bool:
        """
        """
        # Allow to match against any ramp.
        if self.is_generic:
            return True

        match = False
        for name, value in zip(("start", "stop", "points"), (start, stop, points)):
            pattern = getattr(self, name)
            if pattern is not None:
                match |= pattern.match(value)

        if self.span is not None:
            match |= self.span.match(abs(stop - start))

        return match


@dataclass(init=True)
class StepPattern:
    """[summary]
    """

    #:
    name: str

    #:
    name_pattern: Union[NamePattern, int]

    #:
    ramps: Optional[List[RampPattern]] = None

    #:
    use_in_classification: bool = False

    #:
    classifier_level: int = 0

    def __post_init__(self):
        if isinstance(self.ramps, list) and isinstance(self.ramps[0], dict):
            self.ramps = [RampPattern(**rp) for rp in self.ramps]
        if isinstance(self.name, dict):
            self.ramps = NamePattern(**self.name)

    def match(self, index: int, config: StepConfig) -> bool:
        """
        """
        if isinstance(self.name_pattern, int) and self.name_pattern != index:
            return False
        elif isinstance(self.name_pattern, NamePattern) and not self.name_pattern.match(
            config.name
        ):
            return False

        # If ramps are specified:
        # - check that the config is ramped
        # - that we have a single generic ramp or that the number of ramps match
        # - finally that all all patterns match
        if self.ramps:
            if not config.ramps:
                return False
            if len(self.ramps) == 1 and self.ramps[0].is_generic:
                return True
            if len(config.ramps) != len(self.ramps):
                return False
            if any(
                not rp.match(*astuple(rc)) for rc, rp in zip(config.ramps, self.ramps)
            ):
                return False

        return True

    def extract(self, config: StepConfig) -> tuple:
        """
        """
        if self.ramps and not config.is_ramped:
            raise ValueError(
                "Step is ramped but pattern does not expect a ramp."
                f"Step: {config}, pattern: {self}"
            )
        if config.ramps:
            return tuple(
                np.hstack([np.linspace(*astuple(ramp)) for ramp in config.ramps])
            )
        else:
            return (config.value,)


@dataclass(init=True)
class LogPattern:
    """[summary]
    """

    #:
    name: str

    #:
    # XXX The index refers to the index in the log list not in the channels
    pattern: Union[NamePattern, int]

    #:
    x_name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.pattern, dict):
            self.pattern = NamePattern(**self.pattern)

    def match(self, index: int, entry: LogEntry) -> bool:
        """
        """
        if isinstance(self.name, int) and self.name != index:
            return False
        elif isinstance(self.name, NamePattern) and not self.name.match(entry.name):
            return False

        if self.x_name and (not entry.is_vector):
            return False

        return True


@dataclass(init=True)
class MeasurementPattern:
    """[summary]
    """

    #:
    name: str

    #:
    filename_pattern: Optional[FilenamePattern] = None

    #:
    steps: List[StepPattern] = field(default_factory=list)

    #:
    logs: List[LogPattern] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.filename_pattern, dict):
            self.filename_pattern = FilenamePattern(**self.filename_pattern)
        if isinstance(self.steps[0], dict):
            self.steps = [StepPattern(**v) for v in self.steps]  # type:ignore
        if isinstance(self.logs[0], dict):
            self.logs = [LogPattern(**v) for v in self.logs]  # type:ignore

        if not self.logs:
            raise ValueError(
                "No declared logs of interest, data consolidation would be empty."
            )

    def match(self, dataset: LabberData) -> bool:
        """

        """
        # Check the filename and exit early if it fails
        if self.filename_pattern and not self.filename_pattern.match(dataset.filename):
            return False

        # Check all step patterns against the existing steps, if one fails exit early
        steps = dataset.list_steps()
        for pattern in self.steps:
            if not any(pattern.match(i, step) for i, step in enumerate(steps)):
                return False

        # Check all log patterns.
        logs = dataset.list_logs()
        for lpattern in self.logs:
            if not any(lpattern.match(i, l) for i, l in enumerate(logs)):
                return False

        return True

    def extract_classifiers(
        self, dataset: LabberData
    ) -> Dict[int, Dict[str, Classifier]]:
        """

        """
        # Classifiers are stored by level, and each classifier can have multiple
        # values associated.
        classifiers: Dict[int, Dict[str, Classifier]] = defaultdict(dict)
        if self.filename_pattern and self.filename_pattern.use_in_classification:
            value = self.filename_pattern.extract(dataset.filename)
            if value is None:
                raise RuntimeError(
                    f"Expected a classifier value from {self.filename_pattern}"
                    f"({dataset.filename})"
                )
            classifiers[self.filename_pattern.classifier_level] = value

        for pattern in (p for p in self.steps if p.use_in_classification):
            for i, step in enumerate(dataset.list_steps()):
                if pattern.match(i, step):
                    classifiers[pattern.classifier_level][pattern.name] = Classifier(
                        step.name, pattern.extract(step), bool(step.ramps)
                    )
                    continue

        return classifiers


@dataclass(init=True)
class DataClassifier:
    """[summary]
    """

    patterns: List[MeasurementPattern]

    _datasets: Dict[str, List[str]] = field(init=False)

    #:
    _classified_datasets: Dict[
        str, Dict[str, Dict[int, Dict[str, Classifier]]]
    ] = field(init=False)

    def identify_datasets(self, folder):
        """
        """
        datasets = {p.name: [] for p in self.patterns}
        logger.debug(f"Walking {folder}")
        for root, dirs, files in os.walk(folder):
            for datafile in (f for f in files if f.endswith(".hdf5")):
                path = os.path.join(root, datafile)
                with LabberData(path) as f:
                    for p in self.patterns:
                        if p.match(f):
                            datasets[p.name].append(path)
                            logger.debug(
                                f"Accepted {datafile} for measurement pattern {p.name}"
                            )
                            break
                    else:
                        logger.debug(f"Rejected {datafile} for all patterns")

        self._datasets = datasets

    # XXX debugging utility
    def match_dataset(self, path):
        """
        """
        with LabberData(path) as f:
            for p in self.patterns:
                if p.match(f):
                    return True
            else:
                return False

    def prune_identified_datasets(self, black_list: Set[str]):
        """
        """
        for p in list(self._datasets):
            if p.rsplit(os.sep, 1)[-1] in black_list:
                del self._datasets[p]

    def dump_dataset_list(self):
        """
        """
        pass

    def load_dataset_list(self):
        """
        """
        pass

    def classify_datasets(self):
        """
        """
        if not self._datasets:
            raise RuntimeError(
                "No identified datasets to work on. Run `identify_datasets` or"
                " load an existing list of datasets using `load_dataset_list`."
            )

        classified_datasets = {p.name: {} for p in self.patterns}
        patterns = {p.name: p for p in self.patterns}
        for name, datafiles in self._datasets.items():
            classified = classified_datasets[name]
            for path in datafiles:
                with LabberData(path) as f:
                    classifiers = patterns[name].extract_classifiers(f)
                classified[path] = classifiers

            for path, clsf in classified.items():
                for p, c in classified.items():
                    if p != path and clsf == c:
                        raise RuntimeError(f"{path} and {p} have identical classifiers")

        self._classified_datasets = classified_datasets

    def dump_dataset_classification(self):
        """
        """
        pass

    def load_dataset_classification(self):
        """
        """
        pass

    def consolidate_dataset(self, path: str) -> None:
        """
        """
        if not self._classified_datasets:
            raise RuntimeError(
                "No classified datasets to work on. Run `classify_datasets` or"
                " load an existing list of datasets using "
                "`load_dataset_classification`."
            )

        with File(path, "w") as f:

            for name, classified in self._classified_datasets.items():

                # Create a group for that kind of measurement
                group = f.create_group(name)
                # Extract the measurement pattern
                meas_pattern = [p for p in self.patterns if p.name == name][0]

                for path, classifiers in classified.items():
                    clfs = [classifiers[k] for k in sorted(classifiers)]
                    self._create_entries(group, path, meas_pattern, clfs)

    def _create_entries(
        self,
        storage: Group,
        path: str,
        meas_pattern: MeasurementPattern,
        classifiers: List[Dict[str, Classifier]],
        filters: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        """
        if not classifiers:
            assert filters is not None
            self._extract_datasets(storage, path, meas_pattern, filters)
            return

        if filters is None:
            filters = {}

        # Work on the first level of classifiers provided
        # Extract names in order, values and column_names for filtering
        clf = classifiers[0]
        names = sorted(clf)
        require_filtering = [bool(clf[k].requires_filtering) for k in names]
        column_names = [
            clf[k].column_name for k, mask in zip(names, require_filtering) if mask
        ]
        all_values = [clf[k].values for k in names]

        # Create a conventional, easy to parse name for subgroup, & delimits
        # different classifiers, :: separates the classifier name from the value
        fmt_str = "&".join(f"{c_name}" "::{}" for c_name in names)

        for values in product(*all_values):
            # Create a new group and store classifiers values on attrs
            group = storage.require_group(fmt_str.format(*values))
            group.attrs.update(dict(zip(names, values)))

            # Update the filters to allow extraction of the data
            # We copy since we will use the previously known values for all possible
            # values of the classifiers
            f = filters.copy()
            f.update(
                dict(
                    zip(
                        # oddly bad inference from Mypy
                        column_names,  # type: ignore
                        [v for i, v in enumerate(values) if require_filtering[i]],
                    )
                )
            )

            # We do a recursive call passing one less level of classifiers
            self._create_entries(group, path, meas_pattern, classifiers[1:], f)

    def _extract_datasets(
        self,
        storage: Group,
        path: str,
        meas_pattern: MeasurementPattern,
        filters: Dict[str, float],
    ):
        """

        """
        step_dims = []
        vector_data_names = []
        to_store = {}

        # XXX enforce that the fast scan occurs on the last axis.
        # XXX can be done using np.transpose

        with LabberData(path) as f:
            # Find and exctract the relevant step channels (ie not used in classifying)
            for i, stepcf in [
                (i, s) for i, s in enumerate(f.list_steps()) if s.is_ramped
            ]:
                # Collect all ramps except those used for classification
                should_skip = False
                name = stepcf.name

                # Check if a pattern match and if yes determine if we need to
                # extract this parameter and if yes under what name
                for pattern in meas_pattern.steps:
                    if pattern.match(i, stepcf):
                        if pattern.use_in_classification:
                            should_skip = True
                            break
                        else:
                            name = pattern.name

                # Skip steps used in classification
                if should_skip:
                    continue

                # Determine the number of points for this step
                last_stop = None
                dim = 0
                for r in stepcf.ramps:
                    dim += r.steps
                    # For multiple ramps whose the start match the previous stop Labber
                    # saves a single point.
                    if last_stop and r.start == last_stop:
                        dim -= 1
                    last_stop = r.stop

                step_dims.append(dim)
                to_store[name] = f.get_data(stepcf.name, filters=filters)

            # Find and exctract the relevant log channels
            for i, entry in enumerate(f.list_logs()):
                # Collect only requested log entries.
                should_skip = True
                name = entry.name
                x_name = None

                # Check if a pattern match and if yes determine if we need to
                # extract this log entry and if yes under what name
                for lpattern in meas_pattern.logs:
                    if lpattern.match(i, entry):
                        should_skip = False
                        name = lpattern.name
                        x_name = lpattern.x_name

                # Skip steps used in classification
                if should_skip:
                    continue

                data = f.get_data(entry.name, filters=filters, get_x=x_name is not None)
                if x_name:
                    vector_data_names.append(name)
                    to_store[x_name] = data[0]
                    to_store[name] = data[1]
                else:
                    to_store[name] = data
                log_sizes = len(to_store[name])

            # In the presence of vector data do the following
            # - one vector, add a dummy dimension to all data sets to get something
            #   reminiscent of a normal scan
            # - two vector of more, do not do anything special
            has_many_vec = len(vector_data_names) > 1
            if len(vector_data_names) == 1:
                vec_dim = (
                    to_store[vector_data_names[0]].reshape(step_dims + [-1]).shape[0]
                )
                step_dims += [vec_dim]
                for n, d in list(to_store.items()):
                    if n not in vector_data_names:
                        new_data = np.empty(step_dims, dtype=d.dtype)
                        v = np.moveaxis(new_data, -1, 0)
                        v[:] = d
                        to_store[n] = new_data

            # Store the data in the most relevant shape
            # If no vector data or a single vector, step_dims has the right shape
            # If there is more than one vector, add a -1 to vector data shape
            for n, d in to_store.items():
                if has_many_vec and n in vector_data_names:
                    storage.create_dataset(n, data=d.reshape(step_dims + [-1]))
                else:
                    storage.create_dataset(n, data=d.reshape(step_dims))


@dataclass
class DataExplorer:
    """
    """

    #:
    path: str

    #:
    allow_edits: bool = False

    #:
    create_new: bool = False

    def open(self) -> None:
        """ Open the underlying HDF5 file.

        """
        mode = "w" if self.create_new else ("r+" if self.allow_edits else "r")
        self._file = File(self.path, mode)

    def close(self) -> None:
        """ Close the underlying HDF5 file.

        """
        if self._file:
            self._file.close()
        self._file = None

    def __enter__(self):
        """ Open the underlying HDF5 file when used as a context manager.

        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Close the underlying HDF5 file when used as a context manager.

        """
        self.close()

    def list_measurements(self) -> List[str]:
        """

        """
        if not self._file:
            raise RuntimeError("No opened datafile")
        return list(self._file.keys())

    def list_classifiers(self, measurement) -> Dict[int, List[str]]:
        """

        """
        if not self._file:
            raise RuntimeError("No opened datafile")
        if measurement not in self._file:
            raise ValueError(
                f"No measurement {measurement} in opened datafile, "
                f"existing measurements are {self.list_measurements()}"
            )

        def extract_classifiers(
            group: Group, classifiers: Dict[int, List[str]], level: int
        ) -> Dict[int, List[str]]:
            # By construction the classifiers are the same on each level
            # so we only visit one level of each
            for entry in group:
                if isinstance(entry, Group):
                    classifiers[level] = list(entry.attrs)
                    extract_classifiers(entry, classifiers, level + 1)
                    break
            return classifiers

        return extract_classifiers(self._file[measurement], dict(), 0)

    def walk_data(
        self, measurement: str
    ) -> Iterator[Tuple[Dict[int, Dict[str, Any]], Group]]:
        """

        """
        # Maximal depth of classifiers
        max_depth = len(self.list_classifiers(measurement))

        def yield_classifier_and_data(
            group: Group, depth: int, classifiers: Dict[int, Dict[str, Any]]
        ) -> Iterator[Tuple[Dict[int, Dict[str, Any]], Group]]:
            if depth == max_depth - 1:
                for g in group:
                    clfs = classifiers.copy()
                    clfs[depth] = dict(g.attrs)
                    yield clfs, g
            else:
                for g in group:
                    clfs = classifiers.copy()
                    clfs[depth] = dict(g.attrs)
                    yield from yield_classifier_and_data(g, depth + 1, clfs)

        yield from yield_classifier_and_data(self._file[measurement], 0, dict())

    def get_data(
        self, measurement: str, classifiers: Dict[int, Dict[str, Any]]
    ) -> Group:
        """
        """
        known = self.list_classifiers(measurement)
        if not {k: list(v) for k, v in classifiers.items()} == known:
            raise ValueError(
                f"Unknown classifiers used ({classifiers}),"
                f" known classifiers are {known}"
            )

        group = self._file[measurement]
        for level, values in classifiers.items():
            key = "&".join(f"{k}::{values[k]}" for k in sorted(values))
            if key not in group:
                raise ValueError(
                    f"No entry of level {level} found for {values}, "
                    f"at this level known entries are {[dict(g.attrs) for g in group]}."
                )
            group = group[key]

        return group

    def create_group(
        self, measurement: str, classifiers: Dict[int, Dict[str, Any]]
    ) -> Group:
        """
        """
        group = self._file.require_group(measurement)
        for level, values in classifiers.items():
            key = "&".join(f"{k}::{values[k]}" for k in sorted(values))
            group = group.require_group(key)

        return group
