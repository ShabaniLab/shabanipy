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
import pprint
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field, fields
from itertools import product
from pathlib import Path
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
import toml
from h5py import File, Group

from .data_exploring import make_group_name
from shabanipy.labber import LabberData
from shabanipy.labber.labber_io import InstrumentConfig, LogEntry, StepConfig

logger = logging.getLogger(__name__)


class LabberDataPattern(ABC):
    """A pattern to match against a Labber datafile."""

    @abstractmethod
    def match_file(self, datafile: LabberData) -> bool:
        """Match self against the given (opened) LabberData."""
        raise NotImplementedError(
            "Your subclass must implement this match_file() method."
        )


class Copyable:
    """An object that can be copied with some new attributes."""

    def copy_with(self, **attrs):
        """Copy self with updated attributes."""
        for attr in attrs:
            if not hasattr(self, attr):
                raise ValueError(f"{type(self)} has no attribute {attr}")
        cp = copy(self)
        cp.__dict__.update(attrs)
        return cp


class Classifier(NamedTuple):
    """Column identified as a classifying value for the collected data"""

    #: Name of the column in the dataset if relevant (classifiers based on measurement
    #: names do not set this value)
    column_name: Optional[str]

    #: Classifying values identified in the data.
    values: tuple

    #: Whether this classifier requires to pass a filter value to get_data (ie is
    #: associated with a step that contains ramps rather than setting a unique value).
    requires_filtering: bool = False


@dataclass
class NamePattern(Copyable):
    """Pattern use to match on a name (str)."""

    #: List of names that can be matched
    # (allow to aggregate similar data acquired using a different instrument).
    names: Optional[List[str]] = None

    #: Regular expression that the name should match.
    regex: Optional[str] = None

    def match(self, name: str) -> bool:
        """Match a name against the names and regex of the pattern."""
        match = False
        if self.names is not None:
            match |= name in self.names
            if not match:
                logger.debug(f"- '{name}' not in '{self.names}'")
        if self.regex is not None:
            match |= bool(re.match(self.regex, name))
            if not match:
                logger.debug(f"- '{name}' does not match '{self.regex}'")

        return match


@dataclass
class FilenamePattern(NamePattern):
    """Name pattern allowing to extract information from a measurement name."""

    #: Should the extracted information be used to classify the data.
    use_in_classification: bool = False

    #: Name of the classifier if a single value is extracted for classification.
    #: If multiple classifiers must be extracted use named fields in the regular
    #: expression.
    #: "^.+-(?P<sample>JJ-100nm-2)-.+$" will capture JJ-100nm-2 under sample.
    classifier_name: str = ""

    # Level of the classifier if used for classification.
    # If multiple classifiers will be extracted, the level of each named field in the
    # regex can be specified in a dictionary as {"field_name": level}
    classifier_level: Union[int, Dict[str, int]] = 0

    def extract(self, filename: str) -> Optional[Dict[int, Dict[str, Classifier]]]:
        """Extract classification information from the name.

        Parameters
        ----------
        filename
            The filename to match against self.

        Returns
        -------
        A dictionary of {level, {classifier_name: classifier}} where `level` is the
        integer level of the classifier in the hdf5 hierarchy, `classifier_name` is the
        name of the classifier (possibly specified in the regex), and `classifier` is
        the value of the classifier.
        """
        if self.regex is None:
            return None

        match = re.match(self.regex, filename)
        if not match:
            raise ValueError(f"The provided name: {name}, does not match: {self.regex}")

        d = match.groupdict()
        if d:
            out_dict = defaultdict(dict)
            if isinstance(self.classifier_level, int):
                levels = {clsf_name: self.classifier_level for clsf_name in d}
            else:
                levels = self.classifier_level

            for clsf_name, clsf_value in d.items():
                out_dict[levels[clsf_name]].update(
                    {clsf_name: Classifier(None, (clsf_value,))}
                )
            return out_dict

        try:
            return {
                self.classifier_level,
                {self.classifier_name: Classifier(None, (match.group(1),))},
            }
        except IndexError:
            return None


@dataclass
class ValuePattern(Copyable):
    """Pattern matching against a scalar value.

    If any condition is true the match will be valid.

    """

    #: Value against which to match.
    value: Optional[float] = None

    #: Set of possible values.
    value_set: Optional[Set[float]] = None

    #: Minimal value of the scalar.
    greater: Optional[float] = None

    #: Maximal value of the scalar.
    smaller: Optional[float] = None

    #: Should comparison be strict (<) or not (<=)
    strict_comparisons: bool = False

    def __post_init__(self):
        if all(getattr(self, f.name) is None for f in fields(self)):
            raise ValueError("At least one field of ValuePattern has to be non None")

    def match(self, value: Optional[float]) -> bool:
        """Match the pattern against a scalar.

        If any condition is met the pattern is considered matched.

        """
        if value is None:
            return False
        match = False
        if self.value is not None:
            match |= value == self.value
            if not match:
                logger.debug(f"- {value} != {self.value}")
        if self.value_set is not None:
            match |= value in self.value_set
            if not match:
                logger.debug(f"- {value} not in {self.value_set}")
        if self.smaller is not None:
            match |= (
                value < self.smaller
                if self.strict_comparisons
                else value <= self.smaller
            )
            if not match:
                logger.debug(
                    f"- {value} {'>=' if self.strict_comparisons else '>'}"
                    f"{self.smaller}"
                )
        if self.greater is not None:
            match |= (
                value > self.greater
                if self.strict_comparisons
                else value >= self.greater
            )
            if not match:
                logger.debug(
                    f"- {value} {'<=' if self.strict_comparisons else '<'}"
                    f"{self.smaller}"
                )
        return match


@dataclass
class RampPattern(Copyable):
    """Pattern used to identify a ramp.

    Note that Labber can have multiple ramps for a single step.

    """

    #: Value pattern for the starting point of the ramp.
    start: Optional[Union[ValuePattern, float]] = None

    #: Value pattern for the end point of the ramp.
    stop: Optional[Union[ValuePattern, float]] = None

    #: Value pattern for the span of the ramp.
    span: Optional[Union[ValuePattern, float]] = None

    #: Value pattern for the number of points of the ramp.
    points: Optional[Union[ValuePattern, int]] = None

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (int, float)):
                setattr(self, f.name, ValuePattern(value=value))

    @property
    def is_generic(self):
        """A ramp is considered generic if no specific pattern are provided."""
        return not any(getattr(self, f.name) for f in fields(self))

    def match(self, start: float, stop: float, points: int) -> bool:
        """Determine if a ramp match the pattern."""
        # Allow to match against any ramp.
        if self.is_generic:
            return True

        match = True
        for name, value in zip(("start", "stop", "points"), (start, stop, points)):
            pattern = getattr(self, name)
            if pattern is not None:
                match &= pattern.match(value)

        if self.span is not None:
            match &= self.span.match(abs(stop - start))

        return match


@dataclass
class InstrumentConfigPattern(Copyable):
    """Pattern used to identify an instrument configuration."""

    # user-specified name of the instrument/driver appearing in the Name/Address column
    # of the Measurement Editor
    name: Union[NamePattern, str]

    # name of the quantity to look for in the "Quantity" column of the "Show config"
    # side panel in the Log Browser/Viewer
    quantity: str

    # value to compare against "Value" column of the "Show config" side panel in the Log
    # Browser/Viewer; None matches any value
    value: Optional[Union[ValuePattern, float]] = None

    def __post_init__(self) -> None:
        """Convert literals to patterns."""
        if isinstance(self.name, str):
            self.name = NamePattern(regex=self.name)
        if isinstance(self.value, (int, float)):
            self.value = ValuePattern(value=self.value)

    def match(self, config: InstrumentConfig) -> bool:
        return self.name.match(config.name) and (
            self.value is None or self.value.match(config.get(self.quantity))
        )

    def extract(self, configs: List[InstrumentConfig], default) -> bool:
        """Extract the first classifying value found."""
        for config in configs:
            if self.match(config):
                return (config.get(self.quantity),)
        logger.debug("no instrument config found, using default")
        return (default,)


@dataclass
class StepPattern(Copyable, LabberDataPattern):
    """A pattern to match against stepped channels."""

    #: Name of the step to use when retrieving data or classifying using that step
    name: str

    #: Name pattern for the step or index of the step.
    #: Only positive indexes are supported
    name_pattern: Union[NamePattern, int]

    #: Single value of the step.
    value: Optional[ValuePattern] = None

    #: List of ramp pattern for the step.
    ramps: Optional[List[RampPattern]] = None

    #: InstrumentConfigPattern to fall back on if this StepPattern doesn't match any steps
    fallback: Optional[InstrumentConfigPattern] = None

    # Default value if this StepPattern and fallback InstrumentConfigPattern both fail
    # (e.g. the channel may not have been added to labber yet)
    default: Any = None

    #: Should this step be used to classify datasets.
    #: Steps used for classification should not contain overlapping ramps (both forward
    #: and backward ramp).
    use_in_classification: bool = False

    #: Level of the classifier.
    classifier_level: int = 0

    def match(self, index: int, config: StepConfig) -> bool:
        """Match a step that meet all the specify pattern."""
        if isinstance(self.name_pattern, int) and self.name_pattern != index:
            logger.debug(f"- step index {index} != {self.name_pattern}")
            return False
        elif isinstance(self.name_pattern, NamePattern) and not self.name_pattern.match(
            config.name
        ):
            return False

        # Check for a single value.
        if (
            self.value
            and config.value is not None
            and not self.value.match(config.value)
        ):
            return False

        # If ramps are specified:
        # - check that the config is ramped
        # - that we have a single generic ramp or that the number of ramps match
        # - finally that all all patterns match
        if self.ramps:
            if not config.ramps:
                logger.debug(f"- step does not have ramps")
                return False
            if len(self.ramps) == 1 and self.ramps[0].is_generic:
                return True
            if len(config.ramps) != len(self.ramps):
                logger.debug(
                    f"- step has {len(config.ramps)} ramps,"
                    f" expected {len(self.ramps)}"
                )
                return False
            if any(
                not rp.match(*(rc.start, rc.stop, rc.steps))
                for rc, rp in zip(config.ramps, self.ramps)
            ):
                return False

        logger.debug("- successful match")
        return True

    def match_file(self, datafile: LabberData) -> bool:
        """Match self against the stepped channels in the datafile.

        If no step channels match and a `fallback` pattern is specified, the instrument
        configs are checked.

        If a `default` value is specified, the pattern always matches regardless of the
        step channels and instrument configs.
        """
        if self.default is not None:
            logger.debug("default value specified, always matches")
            return True
        if any(self.match(i, step) for i, step in enumerate(datafile.steps)):
            return True
        if self.fallback:
            logger.debug("falling back to instrument config")
            return any(
                self.fallback.match(instr_config)
                for instr_config in datafile.instrument_configs
            )

    def extract(self, dataset: LabberData, config: StepConfig) -> tuple:
        """Extract the classification values associated with that step."""
        if self.ramps and not config.is_ramped:
            raise ValueError(
                "Step is ramped but pattern does not expect a ramp."
                f"Step: {config}, pattern: {self}"
            )
        if config.is_ramped:
            # Retrieve the classifier data directly from the log file to avoid
            # considering values that were not acquired because the measurement
            # was stopped.
            data = np.unique(dataset.get_data(config.name))
            # Remove nan
            return tuple(data[~np.isnan(data)])
        else:
            if config.relation:
                steps = {
                    "Step values" if s.name == config.name else s.name: s.value
                    for s in dataset.steps
                }
                locs = {k: steps[v] for k, v in config.relation[1].items()}
                # XXX should provide some math functions
                return (eval(config.relation[0], locs),)
            else:
                return (config.value,)


@dataclass
class LogPattern(Copyable, LabberDataPattern):
    """A pattern to match against logged channels."""

    #: Name to use when extracting the data.
    name: str

    #: Name pattern or index in the log list entries
    #: Only positive indexes are supported.
    pattern: Union[NamePattern, int]

    #: Name to use for storing the x data of vector data. If multiple log data are
    #: vectorial but refer to the same x, either specify the x_name for a single
    #: one or use the same name for all. This will ensure that the extracted data
    #: will be properly shaped to look like normal scans.
    x_name: Optional[str] = None

    # A datafile will only be rejected if it doesn't match a required LogPattern.
    # If the LogPattern is not required, any matching data will be included in the
    # aggregated data file if available.
    is_required: Optional[bool] = True

    def match(self, index: int, entry: LogEntry) -> bool:
        """Match a log entry on its index or name.

        If a x_name is specified the data have to be vectorial.

        """
        logger.debug(f"matching log entry: {entry.name}")
        if isinstance(self.pattern, int) and self.pattern != index:
            logger.debug(f"- log index {index} != {self.pattern}")
            return False
        elif isinstance(self.pattern, NamePattern) and not self.pattern.match(
            entry.name
        ):
            return False

        if self.x_name and (not entry.is_vector):
            logger.debug(f"- x_name specified but entry is not vectorial")
            return False

        logger.debug("- successful match")
        return True

    def match_file(self, datafile: LabberData) -> bool:
        """Match self against the logged channels in the datafile."""
        if not self.is_required:
            return True
        return any(self.match(idx, log) for idx, log in enumerate(datafile.logs))


@dataclass
class MeasurementPattern(Copyable, LabberDataPattern):
    """Pattern used to identify relevant measurements."""

    #: Name used to identify this kind of measurement. Used in post-processing.
    name: str

    #: Pattern that the filename of the measurement must match.
    filename_pattern: Optional[FilenamePattern] = None

    # list of patterns describing the measurement
    patterns: List[LabberDataPattern] = field(default_factory=list)

    #: Steps that should be excluded from the consolidated file. This applies to
    #: to steps ramped due to relations but that are not relevant. Un-ramped steps
    #: are always omitted.
    excluded_steps: List[Union[NamePattern, int]] = field(default_factory=list)

    def match_file(self, datafile: LabberData) -> bool:
        """Match self against the given Labber datafile."""
        if self.filename_pattern is not None and not self.filename_pattern.match(
            datafile.filename
        ):
            return False
        return all(p.match_file(datafile) for p in self.patterns)

    def match_excluded_steps(self, index: int, config: StepConfig) -> bool:
        """Determine if a config match any of the excluded channels."""
        for p in self.excluded_steps:
            if isinstance(p, int) and p == index:
                return True
            elif isinstance(p, NamePattern) and p.match(config.name):
                return True

        return False

    def extract_classifiers(
        self, dataset: LabberData
    ) -> Dict[int, Dict[str, Classifier]]:
        """Extract the relevant quantities to classify the data."""
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
            logger.debug(f"Classifiers extracted from {dataset.filename}: {value}")
            classifiers.update(value)

        for pattern in (
            p
            for p in self.patterns
            if isinstance(p, StepPattern) and p.use_in_classification
        ):
            found_match = False
            for i, step in enumerate(dataset.steps):
                if pattern.match(i, step):
                    found_match = True
                    classifiers[pattern.classifier_level][pattern.name] = Classifier(
                        step.name, pattern.extract(dataset, step), step.is_ramped
                    )
                    break

            # get from instrument config if nothing in step list matched
            if not found_match and pattern.fallback:
                logger.debug("Extract classifiers: Falling back to instrument config")
                classifiers[pattern.classifier_level][pattern.name] = Classifier(
                    step.name,
                    pattern.fallback.extract(
                        dataset.instrument_configs, pattern.default
                    ),
                )

        return classifiers


@dataclass
class DataClassifier:
    """Object used to identify and consolidate measurements in a single location.

    Measurements are identified using patterns matching on the filename, the steps
    involved in the measurement, the logged channels. In order to consolidate the data
    we first classify the data based on values extracted from the filename and some
    of the steps. Classifiers can have different levels, allowing for example to
    distinguish first by sample and then by gate voltage. It is also possible to
    do the contrary and have both those information on the same level. This becomes
    relevant for data analysis.
    Finally we can rewrite all the relevant data into the single HDF5 file. The file
    has a nested structure with the first level being the kind of measurement and
    each lower level corresponding to a classifier level. The resulting files can easily
    be explored and used through the `data_exploring.DataExplorer` class

    """

    #: List of measurement patterns to identify. Having more than one can allow to
    #: group together characterization measurement with the measurements of interest.
    patterns: List[MeasurementPattern]

    #: Identified dataset as a list of path for each measurement pattern.
    _datasets: Dict[str, List[str]] = field(init=False)

    #: Classifiers found for each dataset previously identified. Organized per
    #: measurement pattern/dataset path/classifier level
    _classified_datasets: Dict[
        str, Dict[str, Dict[int, Dict[str, Classifier]]]
    ] = field(init=False)

    def identify_datasets(self, folders):
        """Identify the relevant datasets by scanning the content of a folder."""
        logger.info("identifying datasets...")
        datasets = {p.name: [] for p in self.patterns}
        for folder in folders:
            if not Path(folder).exists():
                logger.warning(f"{folder} does not exist")
                continue
            logger.debug(f"Walking {folder}")
            for root, dirs, files in os.walk(folder):
                for datafile in (f for f in files if f.endswith(".hdf5")):
                    path = os.path.join(root, datafile)
                    logger.debug(f"matching file {datafile}")
                    try:
                        with LabberData(path) as f:
                            logger.debug(f"steps: {[s.name for s in f.steps]}")
                            logger.debug(f"logs:  {[l.name for l in f.logs]}")
                            logger.debug(
                                f"instrument configs: {[c.name for c in f.instrument_configs]}"
                            )
                            for p in self.patterns:
                                logger.debug(f"matching measurement pattern '{p.name}'")
                                if p.match_file(f):
                                    datasets[p.name].append(path)
                                    logger.debug(
                                        f"- accepted {datafile} "
                                        f"for measurement pattern '{p.name}'"
                                    )
                    except OSError:
                        logger.debug(f"- rejected {datafile}: file is corrupted")

        self._datasets = datasets
        logger.info(
            f"identified datasets:\n{pprint.pformat({k: str(len(v)) + ' files' for k, v in datasets.items()})}"
        )
        logger.debug(f"identified datasets:\n{pprint.pformat(datasets)}")

    def match_dataset(self, path: str) -> Optional[MeasurementPattern]:
        """Match a single file and return the pattern.

        This function is mostly included for debugging purposes.

        """
        with LabberData(path) as f:
            for p in self.patterns:
                if p.match(f):
                    return p
            else:
                return None

    def prune_identified_datasets(self, deny_list: List[str]) -> None:
        """Prune the identified datasets from bad values."""
        for p in list(self._datasets):
            if p.rsplit(os.sep, 1)[-1] in deny_list:
                del self._datasets[p]

    def dump_dataset_list(self, path: str) -> None:
        """Dump the list of identified files into a TOML file."""
        with open(path, "w") as f:
            toml.dump(self._datasets, f)

    def load_dataset_list(self, path: str) -> None:
        """Load a list of identified files from a TOML file.

        The measurement names are validated but the files are not matched.

        """
        d = toml.load(path)
        m_names = [m.name for m in self.patterns]
        for k in d:
            if k not in m_names:
                raise ValueError(
                    f"Loaded file contains pattern for {k} which is unknown. "
                    f"Known values are {m_names}"
                )
        self._datasets = d  # type: ignore

    def classify_datasets(self):
        """Find the classifiers values in each identified dataset."""
        logger.info(f"classifying datasets...")
        if not self._datasets:
            raise RuntimeError(
                "No identified datasets to work on. Run `identify_datasets` or"
                " load an existing list of datasets using `load_dataset_list`."
            )

        classified_datasets = {p.name: {} for p in self.patterns}
        patterns = {p.name: p for p in self.patterns}
        for name, datafiles in self._datasets.items():
            logger.debug(f"  classifying '{name}' measurements")
            classified = classified_datasets[name]
            for path in datafiles:
                logger.debug(f"    classifying {path}")
                with LabberData(path) as f:
                    classifiers = patterns[name].extract_classifiers(f)
                classified[path] = classifiers

            for path, clsf in classified.items():
                for p, c in classified.items():
                    if p != path and clsf == c:
                        raise RuntimeError(f"{path} and {p} have identical classifiers")

        self._classified_datasets = classified_datasets
        logger.debug(f"Classified datasets:\n{pprint.pformat(classified_datasets)}")

    def dump_dataset_classification(self):
        """"""
        raise NotImplementedError

    def load_dataset_classification(self):
        """"""
        raise NotImplementedError

    def consolidate_dataset(self, path: str) -> None:
        """Consolidate all the relevant data into a single file."""
        logger.info(f"consolidating data...")
        if not self._classified_datasets:
            raise RuntimeError(
                "No classified datasets to work on. Run `classify_datasets` or"
                " load an existing list of datasets using "
                "`load_dataset_classification`."
            )

        with File(path, "w") as f:

            for name, classified in self._classified_datasets.items():
                logger.debug(f"  Processing measurements under: {name}")

                # Create a group for that kind of measurement
                group = f.create_group(name)
                group.attrs["files"] = self._datasets[name]
                # Extract the measurement pattern
                meas_pattern = [p for p in self.patterns if p.name == name][0]

                for filepath, classifiers in classified.items():
                    logger.debug(f"    Processing: {filepath}")
                    clfs = [classifiers[k] for k in sorted(classifiers)]
                    self._create_entries(group, filepath, meas_pattern, clfs)
        logger.info(f"...data consolidated into {path}")

    def _create_entries(
        self,
        storage: Group,
        path: str,
        meas_pattern: MeasurementPattern,
        classifiers: List[Dict[str, Classifier]],
        filters: Optional[Dict[str, float]] = None,
    ) -> None:
        """Create group for each level of classifiers and each values."""

        # If we exhausted the classifiers we are ready to pull the data we care about
        # and store them.
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

        # This assumes that all combination of classifiers exist which may not be true
        # if the measurement was interrupted.
        # XXX this could be optimized for relation that enforce equality
        for values in product(*all_values):
            # Create a new group and store classifiers values on attrs
            group_name = make_group_name(dict(zip(names, values)))
            group = storage.require_group(group_name)
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

            # If we did not create any content and the group is empty (no group or
            # dataset), meaning that this combination of classifiers is invalid due
            # to the measurement being interrupted we delete the group.
            if not group.keys():
                del storage[group.name]

    def _extract_datasets(
        self,
        storage: Group,
        path: str,
        meas_pattern: MeasurementPattern,
        filters: Dict[str, float],
    ):
        """Extract the data corresponding to the specified filters."""
        vector_data_names = []
        x_vector_data_names = []
        to_store = {}

        with LabberData(path) as f:
            # Find and extract the relevant step channels (ie not used in classifying)
            for i, stepcf in [
                (i, s) for i, s in enumerate(f.steps) if s.is_ramped
            ]:
                # Collect all ramps except those used for classification
                should_skip = False
                name = stepcf.name

                # Check if a pattern match and if yes determine if we need to
                # extract this parameter and if yes under what name
                for pattern in (
                    p for p in meas_pattern.patterns if isinstance(p, StepPattern)
                ):
                    if pattern.match(i, stepcf):
                        if pattern.use_in_classification:
                            should_skip = True
                            break
                        else:
                            name = pattern.name

                # Skip steps used in classification and explicitely excluded
                if should_skip or meas_pattern.match_excluded_steps(i, stepcf):
                    continue

                # Get the data which are already in a meaningful shape (see
                # LabberData.get_data)
                to_store[name] = f.get_data(stepcf.name, filters=filters)

            # Find and extract the relevant log channels
            n_matched_logs = 0
            log_patterns = [
                p for p in meas_pattern.patterns if isinstance(p, LogPattern)
            ]
            for i, entry in enumerate(f.logs):
                # Collect only requested log entries.
                should_skip = True
                name = entry.name
                x_name = None

                # Check if a pattern match and if yes determine if we need to
                # extract this log entry and if yes under what name
                for lpattern in log_patterns:
                    if lpattern.match(i, entry):
                        n_matched_logs += 1
                        should_skip = False
                        name = lpattern.name
                        x_name = lpattern.x_name

                # Log entry was not requested
                if should_skip:
                    continue

                data = f.get_data(entry.name, filters=filters, get_x=x_name is not None)  # type: ignore

                if entry.is_vector:
                    vector_data_names.append(name)
                    if x_name:
                        x_vector_data_names.append(x_name)
                        to_store[x_name] = data[0]
                        to_store[name] = data[1]
                    else:
                        to_store[name] = data
                else:
                    to_store[name] = data

            if n_matched_logs > len(log_patterns):
                log_names = [log.name for log in log_patterns]
                raise RuntimeError(
                    "More logs were matched than there is log patterns. "
                    f"The matched logs are {[l for l in to_store if l in log_names]}"
                )

            # In the presence of vector data do the following
            # - one vector or vectors with the same length and a single x_name,
            #   add a dummy dimension to all data sets to get something reminiscent
            #   of a normal scan
            # - two vectors or more with different x, do not do anything special
            if len(vector_data_names) == 1 or (
                len(vector_data_names) > 1
                and all(
                    to_store[vector_data_names[0]].shape[-1] == to_store[n].shape[-1]
                    for n in vector_data_names[1:]
                )
                and len(set(x_vector_data_names)) < 2
            ):
                vec_dim = to_store[vector_data_names[0]].shape[-1]

                for n, d in list(to_store.items()):
                    if n not in vector_data_names and n not in x_vector_data_names:
                        # Create a new array with an extra dimension
                        new_data = np.empty(list(d.shape) + [vec_dim], dtype=d.dtype)

                        # Create a view allowing to easily assign the same value
                        # on all elements of the last dimension.
                        v = np.moveaxis(new_data, -1, 0)
                        v[:] = d

                        # Store the data
                        to_store[n] = new_data

            # If the data are empty, do not store anything.
            if any(v.shape == (0,) for v in to_store.values()):
                return

            # Store the data
            for n, d in to_store.items():
                # If data with similar classifers are already there check,
                # if there are the same shape. If yes and one of them contains less nan
                # than the other keep the most complete set, otherwise log the issue
                # and do not store the new one.
                if n in storage:
                    dset = storage[n]
                    if dset.shape == d.shape:
                        ex_count = np.count_nonzero(np.isnan(dset))
                        new_count = np.count_nonzero(np.isnan(d))
                        if new_count < ex_count:
                            dset = storage[n] = d
                        else:
                            logger.info(
                                f"Ignoring {n} in {path} since more complete "
                                "data already exists (less nans)"
                            )
                    elif dset.shape[1:] == d.shape[1:]:
                        if dset.shape[0] < d.shape[0]:
                            # Delete the existing dataset and use teh more complete one.
                            del storage[n]
                            dset = storage.create_dataset(
                                n, data=d, compression="gzip", compression_opts=6,
                            )
                        else:
                            logger.info(
                                f"Ignoring {n} in {path} since more complete "
                                "data already exists, larger outer dimension"
                            )
                    else:
                        logger.info(
                            f"Ignoring {n} in {path} since data of a different "
                            "shape already exists. "
                            f"Existing {dset.shape}, new {d.shape}"
                        )
                else:
                    dset = storage.create_dataset(
                        n, data=d, compression="gzip", compression_opts=6
                    )

                # Store the origin of the data by the data.
                dset.attrs["__file__"] = f.filename
