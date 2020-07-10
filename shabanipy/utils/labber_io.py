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
from typing import Dict, List, Optional, Tuple, Union

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
    points: int = field(init=False)

    #: Number of points per log file
    points_per_log: Tuple[int, ...] = field(init=False)

    def __post_init__(self):
        if not self.ramps:
            return 1

        points_per_log = []
        points = 0
        last_stop = None
        for r in self.ramps:
            points += r.steps
            # If the ramp is part of a separate measurement that was appended
            # reset the last stop and store the number of points
            if r.new_log:
                last_stop = None
                points_per_log.append(points)
                points = 0
            # For multiple ramps whose the start match the previous stop Labber
            # saves a single point.
            if last_stop is not None and r.start == last_stop:
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

    #: Private
    _file: Optional[File] = field(default=None, init=False)

    #:
    _nested: List[Group] = field(default_factory=list, init=False)

    #:
    _channel_names: Optional[List[str]] = field(default=None, init=False)

    #:
    _axis_dimensions: Optional[tuple] = field(default=None, init=False)

    #:
    _steps: Optional[List[StepConfig]] = field(default=None, init=False)

    #:
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

    def get_data(
        self,
        name_or_index: Union[str, int],
        filters: Optional[dict] = None,
        filter_precision: float = 1e-10,
        get_x: bool = False,
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
        data : np.ndarray
            1D numpy array containing the requested data.

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
        channel = self._channels[index]

        x_data: List[np.ndarray] = []
        vectorial_data = bool(isinstance(channel, LogEntry) and channel.is_vector)
        vec_dim = 0
        if vectorial_data:
            aux = self._get_traces_data(
                channel.name, is_complex=channel.is_complex, get_x=get_x
            )
            if not get_x:
                data = aux[0]
            else:
                x_data, data = aux
            vec_dim = data[0].shape[0]
        else:
            data = self._get_data_data(
                channel.name, is_complex=getattr(channel, "is_complex", False)
            )

        # Filter the data based on the provided filters.
        filters = filters if filters is not None else {}
        if filters:
            datasets = [self._file] + self._nested[:]
            results = []
            x_results = []
            for i, d in enumerate(datasets):
                masks = []
                for k, v in filters.items():
                    index = self._name_or_index_to_index(k)
                    d = d["Data"]["Data"][:, index]
                    # Create the mask filtering the data
                    mask = np.less(np.abs(d - v), filter_precision)
                    # If the mask is not empty, ensure we do not eliminate nans
                    # added by labber to have complete sweeps
                    if np.any(mask):
                        np.logical_or(mask, np.isnan(d), mask)
                    masks.append(mask)

                mask = masks.pop()
                for m in masks:
                    np.logical_and(mask, m, mask)

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
            s.points_per_log for s in self.list_steps() if s.name not in filters
        ]

        if vectorial_data:
            steps_points.insert(0, (vec_dim,) * len(steps_points[0]))

        # Get expected shape per log
        shape_per_logs = np.array(steps_points).T
        shaped_results = []
        shaped_x = []
        for i, shape in shape_per_logs:
            # If the filtering produced an empty array skip it
            if results[i].shape == (0,):
                continue

            padding = len(results[i]) % np.prod(shape[:-1])
            # Pad the data to ensure that only the last axis shrinks
            if padding:
                results[i] = np.concatenate(
                    (results[i], np.nan * np.ones(padding)), None
                )
                if vectorial_data and get_x:
                    x_results[i] = np.concatenate(
                        (x_results[i], np.nan * np.ones(padding)), None
                    )
            shaped_results.append(results[i].reshape(shape[:-1] + (-1,)))
            if vectorial_data and get_x:
                shaped_x.append(x_results[i].reshape(shape[:-1] + (-1,)))

        # Create the complete data
        full_data = np.concatenate(results, axis=-1)
        if vectorial_data and get_x:
            full_x = np.concatenate(x_results, axis=-1)

        # Transpose the axis so that the inner most loops occupy the last dimensions
        full_data = np.transpose(full_data)
        if vectorial_data and get_x:
            full_x = np.transpose(full_x)
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

        names = list(self._file["Data"]["Channel names"])
        if is_complex:
            re_index = names.index(channel_name + " - Real")
            im_index = names.index(channel_name + " - Imag")
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

    FILE = "/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper/data/JS314_CD1_att60_007.hdf5"

    with LabberData(FILE) as ld:
        print(ld.list_channels())
        for ch in ld.list_channels():
            data = ld.get_data(ch)
            print(ch, data.shape)
        cdata = ld.get_data("VNA - S21")

        plt.plot(np.absolute(cdata))
        plt.show()
