# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2018-2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
""" IO tools to interact with Labber saved data.

"""
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import cached_property, reduce
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
from h5py import File, Group

logger = logging.getLogger(__name__)

# XXX currently only linear sweeps are supported
@dataclass
class RampConfig:
    """Description of single ramp in a step.

    For ramps with a single point start and stop are equal and steps is one,
    np.linspace(start, stop, points) produces the expected result.

    """

    #: Start of the ramp
    start: float

    #: Stop of the ramp
    stop: float

    #: Number of points in the ramp
    steps: int

    #: Is that ramp the first of a measurement (used to discrimate similar ramps)
    #: taken from appended logs
    new_log: bool

    def create_array(self):
        """Create an array representing the ramp."""
        return np.linspace(self.start, self.stop, self.points)


@dataclass
class StepConfig:
    """Configuration describing a single step in a Labber measurement."""

    #: Name of the step. Match the name of the dataset in the log file.
    name: str

    #: Does this step contain more than a single value.
    is_ramped: bool

    #: Relation this step has to other other steps. The tuple contains a
    #: string and a dictionary mapping placeholder in the format string to step
    #: names. The relation can be evaluated by replacing the step name by the value
    #: and using the dictionary as local when evaluating the string.
    relation: Optional[Tuple[str, Dict[str, str]]]

    #: Set value for step with a single set point.
    value: Optional[float]

    #: List of ramps configuration describing the set points of the step.
    ramps: Optional[List[RampConfig]]

    #: Is the direction of the ramp alternating
    alternate_direction: bool

    #: Total number of set points for this step
    points: int = field(default=0, init=False)

    #: Number of points per log file
    points_per_log: Tuple[int, ...] = field(default=(0,), init=False)

    def __post_init__(self):
        if not self.ramps:
            return 1

        points_per_log = []
        points = 0
        last_stop = None
        for r in self.ramps:
            points += r.steps
            if last_stop is not None:
                # If this not the first ramp and the ramp is part of a separate
                # measurement that was appended reset the last stop and store the
                # number of points
                if r.new_log:
                    last_stop = None
                    points_per_log.append(points)
                    points = 0
                # For multiple ramps whose the start match the previous stop Labber
                # saves a single point.
                if r.start == last_stop:
                    points -= 1
            last_stop = r.stop
        points_per_log.append(points)
        self.points_per_log = tuple(points_per_log)
        self.points = sum(points_per_log)


@dataclass
class LogEntry:
    """Description of a log entry."""

    #: Name of the entry in the dataset
    name: str

    #: Are the data stored in that entry vectorial
    is_vector: bool = False

    #: Are the data stored in that entry complex
    is_complex: bool = False

    #: Name of the x axis for vectorial data with a provided x.
    x_name: str = ""


@dataclass
class InstrumentConfig:
    """Labber's description of an instrument configuration.

    These are contained under the HDF5 group "/Instrument config/" and list the
    instrument driver settings for the measurement.

    The names of the HDF5 subgroups have the following format:
        <driver> - IP: <ip_address>, <name> at localhost
    A concrete example:
        Keithley 2450 SourceMeter [Shabani] - IP: 192.168.0.13, x magnet at localhost
    """

    group: Group

    @cached_property
    def driver(self) -> str:
        """The name of the instrument driver."""
        assert self.group.name.startswith(
            "/Instrument config/"
        ), "Unknown group name format in file {self.group.file.filename}: {group.name} doesn't start with '/Instrument config/'"
        return self.group.name[19:].split(" - ", 1)[0]

    @cached_property
    def ip_address(self) -> str:
        """The IP address of the instrument driver. Possibly empty."""
        ip_address = self.group.name.split(" - ", 1)[1].split(", ", 1)[0]
        return ip_address

    @cached_property
    def name(self) -> str:
        """The name of the instrument defined by the user in Labber.

        This appears in the 'Name/Address' column of the Measurement Editor.
        """
        name = self.group.name.split(", ", 1)[1]
        assert name.endswith(
            " at localhost"
        ), "Unknown group name format in file {self.group.file.filename}: {name} doesn't end with ' at localhost'"
        return name[:-13]

    def get(self, quantity: str) -> Optional[Any]:
        """Get the value of the given quantity from the instrument config.

        This checks the {Quantity: Value} pairs as seen in the "Show config" side panel
        of the Log Browser/Viewer.
        """
        if quantity not in self.group.attrs:
            raise ValueError(
                f"'{quantity}' does not exist in '{self.name}'; "
                f"available quantities are {self.group.attrs.keys()}"
            )
        return self.group.attrs.get(quantity)


def maybe_decode(bytes_or_str: Union[str, bytes]) -> str:
    """H5py return some string as bytes in 3.10 so convert them."""
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.decode("utf-8")
    return bytes_or_str


@dataclass
class LabberData:
    """Labber saves data in HDF5 files and organizes them by channel.

    A channel is either a swept variable or a measured quantity. We use either
    strings or integers to identify channels.
    """

    #: Path to the HDF5 file containing the data.
    path: Union[str, Path]

    #: Name of the file (ie no directories)
    filename: str = field(init=False)

    __file: Optional[File] = field(default=None, init=False)

    @property
    def _file(self):
        """The underlying HDF5 file in which the data are stored."""
        if self.__file is None:
            raise RuntimeError("No HDF5 file is currently opened.")
        return self.__file

    @_file.setter
    def _file(self, value):
        self.__file = value

    #: Groups in which the data of appended measurements are stored.
    _nested: List[Group] = field(default_factory=list, init=False)

    #: Steps performed in the measurement.
    _steps: Optional[List[StepConfig]] = field(default=None, init=False)

    #: Log entries of the measurement.
    _logs: Optional[List[LogEntry]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.path, Path):
            self.path = str(self.path)
        self.filename = self.path.rsplit(os.sep, 1)[-1]

    def open(self) -> None:
        """ Open the underlying HDF5 file."""
        self._file = File(self.path, "r")

        # Identify nested dataset
        i = 2
        nested = []
        while f"Log_{i}" in self._file:
            nested.append(self._file[f"Log_{i}"])
            i += 1

        if nested:
            self._nested = nested

    def close(self) -> None:
        """ Close the underlying HDF5 file and clean cached values."""
        if self._file:
            self._file.close()
        self._file = None
        self._nested = []
        self._steps = None
        # clear the @cached_properties if they've been cached
        for attribute in [
            "instrument_configs",
            "channels",
            "channel_names",
            "logs",
            "steps",
        ]:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass

    @cached_property
    def steps(self) -> List[StepConfig]:
        """The step configurations in the Labber file.

        These are listed in the Measurement Editor under "Step sequence" and in the Log
        Browser under "Active channels" with a step list.  The are listed in the
        "/Step list" HDF5 dataset with further information in the "/Step config" group.
        """
        steps = []
        for (
            channel_name,
            _,  # unknown
            _,  # unknown
            _,  # 0 no sweep (ie direct update), 1 between points, 2 continuous
            _,  # unknown
            has_relation,
            relation,
            _,  # unknown
            _,  # unknown
            alternate,
            *_,
        ) in self._file["Step list"]:

            # Decode byte string from the hdf5 file
            channel_name = maybe_decode(channel_name)
            relation = maybe_decode(relation)

            step_configs = [
                f["Step config"][channel_name]["Step items"]
                for f in [self._file] + self._nested
            ]

            # A step is considered ramped if it has more than one config in any log,
            # if the first ramp of any log is ramped, if there is more than one
            # value for a given constant accross different logs
            # The format describing a single config is:
            # ramped, unknown, set value, min, max, center, span, step,
            # number of points, kind(ie linear, log, etc), sweep rate
            is_ramped = (
                any(len(configs) > 1 for configs in step_configs)
                or any(bool(configs[0][0]) for configs in step_configs)
                or len({configs[0][2] for configs in step_configs}) > 1
            )

            # We assume that if we have relations in one log we have them in all
            if has_relation:
                rel_params = {
                    maybe_decode(k): maybe_decode(v)
                    for k, v, _ in self._file["Step config"][channel_name][
                        "Relation parameters"
                    ]
                }
                relation = (
                    relation,
                    {
                        k: v
                        for k, v in rel_params.items()
                        # Preserve only the parameters useful to the relation
                        # \W is a non word character (no letter no digit)
                        if re.match(r"(.*\W+" + f"{k})|{k}" + r"(\W+.*|$)", relation)
                    },
                )
            else:
                relation = None

            steps.append(
                StepConfig(
                    name=channel_name,
                    is_ramped=is_ramped,
                    relation=relation,
                    value=None if is_ramped else step_configs[0][0][2],
                    alternate_direction=alternate,
                    ramps=[
                        (
                            RampConfig(
                                start=cfg[3],
                                stop=cfg[4],
                                steps=cfg[8],
                                new_log=bool(i == 0),
                            )
                            if cfg[0]
                            else RampConfig(
                                start=cfg[2],
                                stop=cfg[2],
                                steps=1,
                                new_log=bool(i == 0),
                            )
                        )
                        for configs in step_configs
                        for i, cfg in enumerate(configs)
                    ]
                    if is_ramped
                    else None,
                )
            )

        # Mark all channels with relation to a ramped channel as ramped.
        # One can inspect ramps to know if a step is ramped outside of a relation.
        for step in steps:
            if step.relation is not None:
                step.is_ramped |= any(
                    s.is_ramped for s in steps if s.name in step.relation[1].values()
                )

        return steps

    def get_step(self, name: str) -> StepConfig:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        raise ValueError(
            f"The requested step `{name}` does not exist in `{self.filename}`."
            f"Available steps are: {[step.name for step in self.steps]}"
        )

    @cached_property
    def logs(self) -> List[LogEntry]:
        """The logged channels in the Labber file.

        These are listed in the Measurement Editor under "Log channels" and in the Log
        Browser under "Active channels" with no step list.  They are listed in the
        "/Log list" HDF5 dataset with further information in the "/Data" group (for
        scalar data) and "/Traces" group (for vectorial data).
        """
        log_names = [maybe_decode(e[0]) for e in self._file["Log list"]]

        # identify scalar complex data
        complex_scalars = [
            maybe_decode(n)
            for n, v in self._file.get("Data/Channel names", ())
            if maybe_decode(v) == "Real"
        ]

        # identify vector data
        vectors = self._file.get("Traces", ())

        logs = []
        for name in log_names:
            if name in vectors:
                logs.append(
                    LogEntry(
                        name=name,
                        is_vector=True,
                        is_complex=self._file[f"Traces/{name}"].attrs.get("complex"),
                        x_name=self._file[f"Traces/{name}"].attrs.get("x, name"),
                    )
                )
            else:
                logs.append(LogEntry(name=name, is_complex=name in complex_scalars))

        return logs

    @cached_property
    def channels(self) -> List[Union[LogEntry, StepConfig]]:
        """Channels that are stepped or logged in the Labber file.

        Channels include step channels and log channels as viewed in the Measurement
        Editor under "Step sequence" and "Log channels" and are listed under "Active
        channels" in the Log Browser.

        Channel data can be retrieved using `get_data`.

        Returns
        -------
        Ramped step channels and all log channels.
        """
        return [s for s in self.steps if s.is_ramped] + [l for l in self.logs]

    @cached_property
    def channel_names(self) -> List[str]:
        """Names of the channels available in the Labber file."""
        return [c.name for c in self.channels]

    @cached_property
    def instrument_configs(self) -> List[InstrumentConfig]:
        """Instrument configurations for the measurement."""
        return [
            InstrumentConfig(group)
            for group in self._file["Instrument config"].values()
        ]

    @overload
    def get_data(
        self,
        name_or_index: Union[str, int],
        filters: Optional[dict] = None,
        filter_precision: float = 1e-10,
        get_x: Literal[False] = False,
    ) -> np.ndarray:
        pass

    @overload
    def get_data(
        self,
        name_or_index: Union[str, int],
        filters: Optional[dict] = None,
        filter_precision: float = 1e-10,
        get_x: Literal[True] = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_data(
        self, name_or_index, filters=None, filter_precision=1e-10, get_x=False
    ):
        """Retrieve data base on channel name or index

        Parameters
        ----------
        name_or_index : str | int
            Name or index of the channel whose data should be loaded.

        filters : dict, optional
            Dictionary specifying channel (as str or int), value pairs to use
            to filter the returned array. For example, passing {0: 1.0} will
            ensure that only the data for which the value of channel 0 is 1.0
            are returned.

        filter_precision : float, optional
            Precision used when comparing the data to the mask value.

        get_x : bool
            Specify for vector data whether the x-data should be returned along with y-data

        Returns
        -------
        data : Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Requested data (or x and requested data for vector data when x is required).
            The data are formatted such as the last axis corresponds the the most inner
            loop of the measurement and the shape match the non filtered steps of the
            measurement.

        """
        # Convert the name into a channel index.
        index = self._name_or_index_to_index(name_or_index)

        # Get the channel description that contains important details
        # (is_vector, is_complex, )
        channel = self.channels[index]  # type: ignore

        x_data: List[np.ndarray] = []
        vectorial_data = bool(isinstance(channel, LogEntry) and channel.is_vector)
        vec_dim = 0
        if vectorial_data:
            aux = self._get_traces_data(
                channel.name, is_complex=channel.is_complex, get_x=get_x  # type: ignore
            )
            # Always transpose vector to be able to properly filter them
            if not get_x:
                data = [a.T for a in aux[0]]
            else:
                x_data, data = [a.T for a in aux[0]], [a.T for a in aux[1]]
            vec_dim = data[0].shape[-1]
        else:
            data = self._get_data_data(
                channel.name, is_complex=getattr(channel, "is_complex", False)
            )

        # Filter the data based on the provided filters.
        filters = filters if filters is not None else {}

        # Only use positive indexes to describe filtering
        filters = {self._name_or_index_to_index(k): v for k, v in filters.items()}
        if filters:
            datasets = [self._file] + self._nested[:]
            results = []
            x_results = []
            for i, d in enumerate(datasets):
                masks = []
                for k, v in filters.items():
                    index = self._name_or_index_to_index(k)
                    filter_data = d["Data"]["Data"][:, index]
                    # Create the mask filtering the data
                    mask = np.less(np.abs(filter_data - v), filter_precision)
                    # If the mask is not empty, ensure we do not eliminate nans
                    # added by Labber to have complete sweeps
                    if np.any(mask):
                        mask |= np.isnan(filter_data)
                    masks.append(mask)

                mask = reduce(lambda x, y: x & y, masks)
                # For unclear reason vector data are not index in the same order
                # as other data and require the mask to be transposed before being
                # raveled
                if vectorial_data:
                    mask = np.ravel(mask.T)
                    if len(mask) > len(data[i]):
                        # if the scan was aborted, the mask (which is derived from the
                        # NaN-padded "/Data/Data" HDF5 dataset) may be larger than the
                        # vectorial data (which is derived from unpadded HDF5 datasets
                        # in "/Traces/")
                        mask = mask[: data[i].shape[0]]
                    elif len(mask) < len(data[i]):
                        # sometimes Labber inexplicably fails to record an incomplete
                        # scan in the "/Data/Data" HDF5 dataset, but still records the
                        # vectorial data points
                        logger.warning(
                            f"There are more traces ({len(data[i])}) recorded in "
                            f"'/Traces/' than there are step values ({len(mask)}) "
                            f"recorded in '/Data/Data'.  You might not be getting all "
                            f"the data from {self.filename} for {name_or_index=} with "
                            f"{filters=} and {get_x=}."
                        )
                        mask = np.append(mask, [False] * (len(data[i]) - len(mask)))

                # Filter
                results.append(data[i][mask])
                if vectorial_data and get_x:
                    x_results.append(x_data[i][mask])

                # If the filtering produces an empty output return early
                if not any(len(r) for r in results):
                    if get_x:
                        return np.empty(0), np.empty(0)
                    else:
                        return np.empty(0)

        else:
            results = data
            x_results = x_data

        # Identify the ramped steps not used for filtering
        steps_points = []
        first_step_is_used = False
        for i, s in reversed(list(enumerate(self.steps))):

            # Use ramps rather than is_ramped to consider only steps manually
            # ramped and exclude steps with multiple values because they
            # they have a relation to a ramped step but no ramp of their own
            if s.ramps is not None and i not in filters:

                # Labber stores scalar data as 3D:
                # - the first dimension is the first step number of points
                # - the second one refer to the channel
                # - the third refer to all other steps but in reverse order
                # For vector data it is the same except that the first is not special.
                if i == 0 and not vectorial_data:
                    steps_points.insert(0, s.points_per_log)
                    first_step_is_used = True
                else:
                    steps_points.append(s.points_per_log)

        # For vectorial data we add to the dimension of the vector at the end
        # of the step points. We take as many as their are logs.
        if vectorial_data:
            steps_points.append(
                (vec_dim,) * (len(steps_points[0]) if steps_points else 1)
            )

        # If we get a single value because we are accessing a value defined through
        # relations to channels we are filtering upon we can stop there and exit
        # early.
        if not steps_points:
            return results[0]  # [np.array([value])]

        # Get expected shape per log
        shape_per_logs = np.array(steps_points).T
        shaped_results = []
        shaped_x = []
        for i, shape in enumerate(shape_per_logs):
            # If the filtering produced an empty array skip it
            if results[i].shape == (0,):
                continue

            # The outer most dimension of the scan corresponds either to the first
            # index if the first step was filtered on (not first_step_is_used) or,
            # otherwise, to the second.
            points_inner_dimensions = (
                np.prod(shape[1:])
                if not first_step_is_used
                else shape[0] * np.prod(shape[2:])
            )
            padding = np.prod(results[i].shape) % (points_inner_dimensions)
            # Pad the data to ensure that only the last axis shrinks
            if padding:
                # Compute the number of points to add
                to_add = points_inner_dimensions - padding
                results[i] = np.concatenate(
                    (results[i], np.nan * np.ones(to_add)), None
                )
                if vectorial_data and get_x:
                    x_results[i] = np.concatenate(
                        (x_results[i], np.nan * np.ones(to_add)), None
                    )

            # Allow the most outer shape to shrink
            new_shape = list(shape)
            new_shape[1 if first_step_is_used and len(shape) > 1 else 0] = -1
            shaped_results.append(results[i].reshape(new_shape))
            if vectorial_data and get_x:
                shaped_x.append(x_results[i].reshape(new_shape))

        # Create the complete data
        full_data = np.concatenate(shaped_results, axis=-1)
        if vectorial_data and get_x:
            full_x = np.concatenate(shaped_x, axis=-1)

        # Move the first axis to last position so that the dimensions are in the reverse
        # order of the steps.
        if first_step_is_used:
            full_data = np.moveaxis(full_data, 0, -1)

        if vectorial_data and get_x:
            return full_x, full_data
        else:
            return full_data

    def get_axis(self, step_name: str) -> int:
        """Get axis of `step_name` as it appears in the array returned by `get_data`.

        Note: `get_data` reverses the order of the steps such that the outer-most loop
        appears first (axis 0), whereas in Labber's Measurement Editor and hdf5 files
        the inner-most loop appears first.

        Parameters
        ----------
        step_name
            Name of the step channel.

        Returns
        -------
        The axis of the requested step channel, ordered consistently with the data array
        returned by `get_data` (i.e. outer-most loop first).
        """
        n_steps = len([s for s in self.steps if s.is_ramped])
        index = self._name_or_index_to_index(step_name)
        return n_steps - index - 1

    def get_config_value(self, instrument: str, quantity: str) -> Union[float, str]:
        """Get the value of the given quantity from the instrument config."""
        instrument_names = [instr.name for instr in self.instrument_configs]
        try:
            index = instrument_names.index(instrument)
        except ValueError as e:
            raise ValueError(
                f"'{instrument}' does not exist in {self.filename}; "
                f"available instruments are {instrument_names}"
            ) from e
        return self.instrument_configs[index].get(quantity)

    def warn_not_constant(
        self, name_or_index: Union[str, int], max_deviation: Optional[float] = None
    ):
        """Issue a warning if `name_or_index` data is not roughly constant.

        Parameters
        ----------
        name_or_index
            The name or index of the channel whose data will be checked.
        max_deviation : optional
            The largest deviation from the mean that will not issue a warning.
            If None, defaults to 1% of the mean.
        """
        data = self.get_data(name_or_index)
        mean = np.mean(data)
        if max_deviation is None:
            max_deviation = 0.01 * mean
        abs_deviation = np.abs(data - mean)
        if np.any(abs_deviation > max_deviation):
            warnings.warn(
                f"Channel `{name_or_index}` deviates from mean {mean} "
                f"by {np.max(abs_deviation)} > {max_deviation}"
            )

    def _name_or_index_to_index(self, name_or_index: Union[str, int]) -> int:
        """Provide the index of a channel from its name."""
        ch_names = self.channel_names

        if isinstance(name_or_index, str):
            if name_or_index not in ch_names:
                msg = (
                    f"The specified name ({name_or_index}) does not exist "
                    f"in the dataset. Existing names are {ch_names}"
                )
                raise ValueError(msg)
            return ch_names.index(name_or_index)
        elif name_or_index >= len(ch_names):
            msg = (
                f"The specified index ({name_or_index}) "
                f"exceeds the number of channel: {len(ch_names)}"
            )
            raise ValueError(msg)
        else:
            return name_or_index

    def _get_traces_data(
        self, channel_name: str, is_complex: bool = False, get_x: bool = False
    ) -> Tuple[List[np.ndarray], ...]:
        """Get data stored in the "/Traces/" HDF5 group of the Labber file.

        This is where vectorial data are stored."""
        if channel_name not in self._file["Traces"]:
            raise ValueError(f"Unknown traces data {channel_name}")

        x_data = []
        data = []
        # Traces dimensions are (sweep, real/imag/x, steps)
        for storage in [self._file] + self._nested:
            if is_complex:
                real = storage["Traces"][channel_name][:, 0, :]
                imag = storage["Traces"][channel_name][:, 1, :]
                data.append(real + 1j * imag)
            else:
                data.append(storage["Traces"][channel_name][:, 0, :])
            if get_x:
                x_data.append(storage["Traces"][channel_name][:, -1, :])

        if get_x:
            return x_data, data
        else:
            return (data,)

    def _get_data_data(
        self, channel_name: str, is_complex: bool = False
    ) -> List[np.ndarray]:
        """Get data stored in the "/Data/" HDF5 group of the Labber file.

        This is where ordinary (i.e. non-vectorial) data are stored."""
        names = [maybe_decode(n) for n, _ in self._file["Data"]["Channel names"]]
        if is_complex:
            re_index = names.index(channel_name)
            im_index = re_index + 1
            real = self._pull_nested_data(re_index)
            imag = self._pull_nested_data(im_index)
            return [r + 1j * i for r, i in zip(real, imag)]
        else:
            return self._pull_nested_data(names.index(channel_name))

    def _pull_nested_data(self, index: int) -> List[np.ndarray]:
        """Pull data stored in the data segmentfrom all nested logs."""
        data = [self._file["Data"]["Data"][:, index]]
        for internal in self._nested:
            data.append(internal["Data"]["Data"][:, index])
        return data

    def __enter__(self) -> "LabberData":
        """ Open the underlying HDF5 file when used as a context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the underlying HDF5 file when used as a context manager."""
        self.close()
