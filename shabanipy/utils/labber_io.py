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
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, overload
from typing_extensions import Literal

import numpy as np

from h5py import File, Group


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

    #: Does that step contains more than a single value.
    is_ramped: bool

    #: Set value for step with a single set point.
    value: Optional[float]

    #: List of ramps configuration describing the set points of the step.
    ramps: Optional[List[RampConfig]]

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
class LabberData:
    """Labber save data in HDF5 files and organize them by channel.

    A channel is either a swept variable or a measured quantities. We use
    either string or integers to identify channels.

    """

    #: Path to the HDF5 file containing the data.
    path: str

    #: Name of the file (ie no directories)
    filename: str = field(init=False)

    #: Private reference to the underlying HDF5 file in which the data are stored
    _file: Optional[File] = field(default=None, init=False)

    #: Groups in which the data of appended measurements are stored.
    _nested: List[Group] = field(default_factory=list, init=False)

    #: Names of the channels accessible in the Data or Traces segments of the file
    _channel_names: Optional[List[str]] = field(default=None, init=False)

    #: Detailed informations about the channels.
    _channels: Optional[List[Union[LogEntry, StepConfig]]] = field(
        default=None, init=False
    )

    #: Steps performed in the measurement.
    _steps: Optional[List[StepConfig]] = field(default=None, init=False)

    #: Log entries of the measurement.
    _logs: Optional[List[LogEntry]] = field(default=None, init=False)

    def __post_init__(self) -> None:
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
        self._channel_names = None
        self._channels = None
        self._axis_dimensions = None
        self._nested = []
        self._steps = None
        self._logs = None

    def list_steps(self) -> List[StepConfig]:
        """List the different steps of a measurement."""
        if not self._file:
            raise RuntimeError("No file currently opened")

        if not self._steps:
            steps = []
            for (step, *_) in self._file["Step list"]:
                configs = [
                    f["Step config"][step]["Step items"]
                    for f in [self._file] + self._nested
                ]
                is_ramped = (
                    any(len(config) > 1 for config in configs)
                    or any(bool(config[0][0]) for config in configs)
                    or len({config[0][2] for config in configs}) > 1
                )
                steps.append(
                    StepConfig(
                        name=step,
                        is_ramped=is_ramped,
                        value=configs[0][0][2] if not is_ramped else None,
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
                            for config in configs
                            for i, cfg in enumerate(config)
                        ]
                        if is_ramped
                        else None,
                    )
                )
            self._steps = steps
        return self._steps

    def list_logs(self) -> List[LogEntry]:
        """List the existing log entries in the datafile."""
        if not self._file:
            raise RuntimeError("No file currently opened")

        if not self._logs:
            # Collect all logs names
            names = [e[0] for e in self._file["Log list"]]

            # Identify scalar complex data
            complex_scalars = [
                n for n, v in self._file["Data"]["Channel names"] if v == "Real"
            ]

            # Identify vector data
            vectors = self._file.get("Traces", ())

            logs = []
            for n in names:
                if n in vectors:
                    logs.append(
                        LogEntry(
                            name=n,
                            is_vector=True,
                            is_complex=self._file["Traces"][n].attrs.get("complex"),
                            x_name=self._file["Traces"][n].attrs.get("x, name"),
                        )
                    )
                else:
                    logs.append(LogEntry(name=n, is_complex=bool(n in complex_scalars)))
            self._logs = logs

        return self._logs

    def list_channels(self):
        """Identify the channel availables in the Labber file.

        Channels data can be retieved using get_data.

        """
        if self._channel_names is None:
            self._channel_names = [s.name for s in self.list_steps() if s.is_ramped] + [
                l.name for l in self.list_logs()
            ]
            self._channels = [s for s in self.list_steps() if s.is_ramped] + [
                l for l in self.list_logs()
            ]

        return self._channel_names

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
        if not self._file:
            msg = (
                "The underlying file needs to be opened before accessing "
                "data. Either call open or better use a context manager."
            )
            raise RuntimeError(msg)

        # Convert the name into a channel index.
        index = self._name_or_index_to_index(name_or_index)

        # Get the channel description that contains important details
        # (is_vector, is_complex, )
        if self._channels is None:
            self.list_channels()
        channel = self._channels[index]  # type: ignore

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
            vec_dim = data[0].shape[0]
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
                        np.logical_or(mask, np.isnan(filter_data), mask)
                    masks.append(mask)

                mask = masks.pop()
                for m in masks:
                    np.logical_and(mask, m, mask)
                # For unclear reason vector data are not index in the same order
                # as other data and require the mask to be transposed before being
                # raveled
                if vectorial_data:
                    mask = np.ravel(mask.T)

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
        steps_points = [
            s.points_per_log
            for i, s in enumerate(self.list_steps())
            if s.is_ramped and i not in filters
        ]

        if vectorial_data:
            steps_points.append((vec_dim,) * len(steps_points[0]))

        # Get expected shape per log
        shape_per_logs = np.array(steps_points).T
        shaped_results = []
        shaped_x = []
        for i, shape in enumerate(shape_per_logs):
            # If the filtering produced an empty array skip it
            if results[i].shape == (0,):
                continue

            padding = np.prod(results[i].shape) % np.prod(shape[:-1])
            # Pad the data to ensure that only the last axis shrinks
            if padding:
                # Compute the number of points to add
                to_add = np.prod(shape[:-1]) - padding
                results[i] = np.concatenate(
                    (results[i], np.nan * np.ones(to_add)), None
                )
                if vectorial_data and get_x:
                    x_results[i] = np.concatenate(
                        (x_results[i], np.nan * np.ones(to_add)), None
                    )

            shaped_results.append(results[i].reshape(tuple(shape)[:-1] + (-1,)))
            if vectorial_data and get_x:
                shaped_x.append(x_results[i].reshape(tuple(shape)[:-1] + (-1,)))

        # Create the complete data
        full_data = np.concatenate(shaped_results, axis=-1)
        if vectorial_data and get_x:
            full_x = np.concatenate(shaped_x, axis=-1)

        # Transpose the axis so that the inner most loops occupy the last dimensions
        full_data = np.transpose(full_data)

        # Due to the earlier manipulation to allow filtering we need to move
        # the now first axis to last position
        if vectorial_data:
            full_data = np.moveaxis(full_data, 0, -1)
        if vectorial_data and get_x:
            full_x = np.moveaxis(np.transpose(full_x), 0, -1)
            return full_x, full_data
        else:
            return full_data

    def _name_or_index_to_index(self, name_or_index: Union[str, int]) -> int:
        """Provide the index of a channel from its name."""
        ch_names = self.list_channels()

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
        """Get data stored as traces ie vector."""
        if not self._file:
            raise RuntimeError("No file currently opened")

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
        """Pull data stored in the data segment of the log file."""
        if not self._file:
            raise RuntimeError("No file currently opened")

        names = [n for n, _ in self._file["Data"]["Channel names"]]
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
        if not self._file:
            raise RuntimeError("No file currently opened")
        data = [self._file["Data"]["Data"][:, index]]
        for internal in self._nested:
            data.append(internal["Data"]["Data"][:, index])
        return data

    def __enter__(self) -> "LabberData":
        """ Open the underlying HDF5 file when used as a context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """ Close the underlying HDF5 file when used as a context manager.

        """
        self.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    FILE = "/Users/mdartiailh/Downloads/JS314_CD1_att40_006.hdf5"

    with LabberData(FILE) as ld:
        print(ld.list_channels())
        for ch in ld.list_channels():
            data = ld.get_data(ch, filters={0: 0}, get_x=True)
            print(
                ch,
                data.shape
                if isinstance(data, np.ndarray)
                else (data[0].shape, data[1].shape),
            )

        # plt.plot(np.absolute(cdata))
        # plt.show()
