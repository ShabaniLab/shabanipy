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
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from h5py import File, Group


@dataclass
class RampConfig:

    #:
    start: float

    #:
    stop: float

    #:
    steps: int


@dataclass
class StepConfig:
    """
    """

    #:
    name: str

    #:
    is_ramped: bool

    #:
    value: Optional[float]

    #:
    ramps: Optional[List[RampConfig]]


@dataclass
class LogEntry:

    #:
    name: str

    #:
    is_vector: bool = False

    #:
    x_name: str = ""


class LabberData:
    """Labber save data in HDF5 files and organize them by channel.

    A channel is either a swept variable or a measured quantities. We use
    either string or integers to identify channels.

    """

    #:
    path: str

    #:
    filename: str

    #:
    _file: Optional[File]

    #:
    _nested: List[Group]

    def __init__(self, path: str) -> None:
        self.path = path
        self.filename = path.rsplit(os.sep, 1)[-1]
        self._file: Optional[File] = None
        self._channel_names = None
        self._axis_dimensions = None
        self._nested = []

    def open(self) -> None:
        """ Open the underlying HDF5 file.

        """
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
        """ Close the underlying HDF5 file.

        """
        if self._file:
            self._file.close()
        self._file = None
        self._channel_names = None
        self._axis_dimensions = None
        self._nested = []

    def get_data(
        self,
        name_or_index: Union[str, int],
        filters: Optional[dict] = None,
        filter_precision: float = 1e-10,
        get_x: bool = False
        # XXX specify rather or not we want x data for vector
    ):
        """ Retrieve data base on channel name or index

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

        index = self.name_or_index_to_index(name_or_index)

        # Copy the data to an array to get a more efficient masking.
        data = [self._file["Data"]["Data"][:, index]]
        for internal in self._nested:
            data.append(internal["Data"]["Data"][:, index])

        if not filters:
            return np.hstack([np.ravel(d) for d in data])

        # Filter the data based on the provided filters.
        datasets = [self._file] + self._nested[:]
        results = []
        for i, d in enumerate(datasets):
            masks = []
            for k, v in filters.items():
                index = self.name_or_index_to_index(k)
                mask = np.less(
                    np.abs(d["Data"]["Data"][:, index] - v), filter_precision
                )
                masks.append(mask)

            mask = masks.pop()
            for m in masks:
                np.logical_and(mask, m, mask)

            results.append(data[i][mask])

        return np.hstack(results)

    # XXX issue if axis are not in the order of the measurement
    def compute_shape(self, sweeps_indexes_or_names):
        """ Compute the expected shape of data based on sweep axis.

        """
        shape = []
        for sw in sorted(
            self.name_or_index_to_index(i) for i in sweeps_indexes_or_names
        ):
            shape.append(self.get_axis_dimension(sw))
        return shape

    # XXX issue if axis are not in the order of the measurement
    def reshape_data(self, sweeps_indexes_or_names, data):
        """ Reshape data based on the swept quantities during the acquisition.

        """
        return data.reshape(self.compute_shape(sweeps_indexes_or_names))

    def get_axis_dimension(self, name_or_index):
        """ Get the dimension of sweeping channel.

        """
        if not self._axis_dimensions:
            data_attrs = self._file["Data"].attrs
            dims = {
                k: v
                for k, v in zip(data_attrs["Step index"], data_attrs["Step dimensions"])
            }
            self._axis_dimensions = dims

        dims = self._axis_dimensions
        index = self.name_or_index_to_index(name_or_index)
        if index not in dims:
            msg = (
                f"The specified axis {name_or_index} is not a stepped one. "
                f"Stepped axis are {list(dims)}."
            )
            raise ValueError(msg)
        return dims[index]

    # XXX should be private
    # XXX index in either data or traces
    # XXX can return is_complex
    # XXX track all complex field in data space
    def name_or_index_to_index(self, name_or_index):
        """Helper raising a nice error when a channel does not exist.

        """
        if self._channel_names is None:
            _ch_names = self._file["Data"]["Channel names"]
            self._channel_names = [n for (n, _) in list(_ch_names)]
        ch_names = self._channel_names

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

    def list_channels(self):
        """Identify the channel availables in the Labber file.

        """
        if self._channel_names is None:
            self._channel_names = [s.name for s in self.list_steps if s.is_ramped] + [
                l.name for l in self.list_logs()
            ]
            # XXX cache data data, vector data
            # XXX cache complex data data
        return self._channel_names

        # _ch_names = self._file["Data"]["Channel names"]
        #  = [n for (n, _) in list(_ch_names)]

    # XXX  document
    # XXX
    def list_steps(self) -> List[StepConfig]:
        """
        """
        if not self._file:
            raise RuntimeError("No file currently opened")

        steps = []
        for (step, *_) in self._file["Step list"]:
            config = self._file["Step config"][step]["Step items"]
            is_ramped = len(config) > 1 or bool(config[0][0])
            steps.append(
                StepConfig(
                    name=step,
                    is_ramped=is_ramped,
                    value=config[0][2] if not is_ramped else None,
                    ramps=[
                        (
                            RampConfig(start=cfg[3], stop=cfg[4], steps=cfg[8])
                            if cfg[0]
                            else RampConfig(start=cfg[2], stop=cfg[2], steps=1)
                        )
                        for cfg in config
                    ]
                    if is_ramped
                    else None,
                )
            )

        return steps

    # XXX provide more structure (in particular indicate vector data)
    def list_logs(self) -> List[LogEntry]:
        """
        """
        if not self._file:
            raise RuntimeError("No file currently opened")

        names = [e[0] for e in self._file["Log list"]]
        if "Traces" in self._file:
            for i, n in enumerate(names[:]):
                if n in self._file["Traces"]:
                    names[i] = LogEntry(
                        name=n,
                        is_vector=True,
                        x_name=self._file["Traces"][n].attrs["x, name"],
                    )
                else:
                    names[i] = LogEntry(name=n)

        return [LogEntry(name=e[0]) for e in self._file["Log list"]]

    def __enter__(self):
        """ Open the underlying HDF5 file when used as a context manager.

        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Close the underlying HDF5 file when used as a context manager.

        """
        self.close()
