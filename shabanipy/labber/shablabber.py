"""Labber hdf5 datafiles.

This is a rewrite of `labber_io.py` with an emphasis on simplicity and maintainability.

The central class in this module is `ShaBlabberFile`, which encapsulates an hdf5
datafile created by Labber.  It is a replacement for `labber_io.LabberData`.

Implementation details are most easily understood by looking at a Labber datafile in an
hdf reader like HDFView.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cached_property
from os import environ
from pathlib import Path
from pprint import pformat
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from h5py import File
from packaging.version import Version
from packaging.version import parse as parse_version


class ShaBlabberFile(File):
    """An hdf5 file created by Labber."""

    def __init__(self, path: Union[str, Path]):
        """
        The `path` to the datafile should be absolute, or relative to the path returned
        by `shablabber.get_data_dir()`.  If the datafile still can't be found, the
        entire directory tree below `shablabber.get_data_dir()` will be searched for
        `path`.
        """
        p = Path(path).expanduser()
        if not p.is_absolute():
            root = get_data_dir()
            p = root / path
            if not p.exists():
                print(f"Searching for {path} in {root}...")
                try:
                    p = next(root.rglob(path))
                except StopIteration as e:
                    raise ValueError(f"Can't find {path} in {root}") from e
        self.path = p
        super().__init__(str(p), "r")

    @property
    def log_name(self) -> str:
        """The internal name of the hdf5 log file."""
        return self.attrs["log_name"]

    @cached_property
    def _creation_time(self) -> datetime:
        """The date and time the hdf5 file was created."""
        return datetime.fromtimestamp(self.attrs["creation_time"])

    @property
    def _step_dims(self) -> Tuple[int]:
        """Step dimensions, i.e. number of points for each step channel."""
        return tuple(self.attrs["Step dimensions"])

    @cached_property
    def _shape(self) -> Tuple[int]:
        """The shape of the data."""
        data_shape = list(self["Data/Data"].shape)
        del data_shape[1]  # number of channels/columns
        data_shape = [dim for dim in data_shape if dim != 1]
        return self._trace_dims + tuple(data_shape)

    @property
    def comment(self) -> str:
        """The file comment."""
        return self.attrs["comment"]

    @property
    def star(self) -> bool:
        """Whether or not the file is starred."""
        return self.attrs["star"]

    @cached_property
    def _version(self) -> Version:
        """A version number presumably related to the Labber version or file spec."""
        return parse_version(self.attrs["version"])

    @cached_property
    def projects(self) -> List[str]:
        """Projects the file belongs to."""
        return _check_empty_list(list(self["Tags"].attrs["Project"]))

    @cached_property
    def tags(self) -> List[str]:
        """Tags assigned to the file."""
        return _check_empty_list(list(self["Tags"].attrs["Tags"]))

    @cached_property
    def _channels(self) -> List[Channel]:
        """All channels in the hdf5 file."""
        return [Channel(self, self["Channels"].dtype, c) for c in self["Channels"]]

    @cached_property
    def _channel_names(self) -> List[str]:
        """Names of all channels in the hdf5 file."""
        return [c.name for c in self._channels]

    def get_channel(self, name: str) -> Union[Channel, XChannel]:
        """Get the channel with `name`."""
        if name in self._channel_names:
            return self._channels[self._channel_names.index(name)]
        elif name in self._x_channel_names:
            return self._x_channels[self._x_channel_names.index(name)]
        raise ValueError(
            f"'{name}' does not exist.  Gettable channels are:\n{pformat(self._channel_names + self._x_channel_names)}"
        )

    @cached_property
    def _data_channel_names(self) -> List[str]:
        """Names of channels that recorded data."""
        return (
            list({name for name, _ in self._data_channel_infos})
            + self._x_channel_names
            + self._trace_channel_names
        )

    @cached_property
    def _data_channel_infos(self) -> List[Tuple[str, str]]:
        """Names of data channels with their associated info."""
        return [
            (name.decode("utf-8"), info.decode("utf-8"))
            for name, info in self["Data/Channel names"]
        ]

    @cached_property
    def _step_channel_names(self) -> List[str]:
        """Names of channels that are added to the 'Step sequence'."""
        return [sc.channel_name for sc in self._step_configs]

    @cached_property
    def _step_configs(self) -> List[StepConfig]:
        """Step configurations for channels that are added to the 'Step sequence'."""
        return [StepConfig(self, self["Step list"].dtype, s) for s in self["Step list"]]

    @cached_property
    def _sweep_channels(self) -> List[Union[XChannel, Channel]]:
        """Channels in the 'Step sequence' that are swept."""
        sweep_channels = [
            self.get_channel(name)
            for name in np.array(self._step_channel_names)[self._sweep_idxs]
        ]
        sweep_channels = [c for c in sweep_channels if not c._use_relations]
        return self._x_channels + sweep_channels

    @cached_property
    def _sweep_channel_names(self) -> List[str]:
        """Names of sweep channels."""
        return [c.name for c in self._sweep_channels]

    @property
    def _sweep_idxs(self) -> List[int]:
        """Indexes of channels in the 'Step sequence' that are swept."""
        return self["Data"].attrs["Step index"]

    @cached_property
    def _fixed_channels(self) -> List[Channel]:
        """Channels in the 'Step sequence' that are fixed at a single value."""
        return [
            self.get_channel(name)
            for name in np.array(self._step_channel_names)[self._fixed_idxs]
        ]

    @cached_property
    def _fixed_channel_names(self) -> List[str]:
        """Names of fixed channels."""
        return [c.name for c in self._fixed_channels]

    @property
    def _fixed_idxs(self) -> List[int]:
        """Indexes of channels in the 'Step sequence' that are fixed."""
        return self["Data"].attrs["Fixed step index"]

    def get_fixed_value(self, channel_name):
        """Get the fixed value of channel_name."""
        return self.get_channel(channel_name).get_fixed_value()

    @cached_property
    def _log_channel_names(self) -> List[str]:
        """Names of channels that are added to the 'Log list'."""
        return [name.decode("utf-8") for name, in self["Log list"]]

    @cached_property
    def _trace_channels(self) -> List[Channel]:
        """Channels that recorded traces (arrays) as a single data point."""
        return [self.get_channel(name) for name in self._trace_channel_names]

    @cached_property
    def _trace_channel_names(self) -> List[str]:
        """Names of channels that recorded traces (arrays) as a single data point."""
        try:
            return list(
                {
                    name.removesuffix("_N").removesuffix("_t0dt")
                    for name in self["Traces"]
                    if name != "Time stamp"
                }
            )
        except KeyError:
            return []

    @cached_property
    def _trace_dims(self) -> Tuple[int]:
        """The dimensions of the trace data."""
        return tuple(c.npoints for c in self._x_channels)

    @cached_property
    def _x_channels(self) -> List[XChannel]:
        """Virtual channels containing the trace channels' step data."""
        # assume all trace channels share the same x channel
        deduped = [self._trace_channels[0]] if self._trace_channels else []
        return [c._x_channel for c in deduped]

    @cached_property
    def _x_channel_names(self) -> List[str]:
        """Names of the trace channels' swept variables."""
        return [c.name for c in self._x_channels]

    @cached_property
    def _instruments(self) -> List[Instrument]:
        """All instruments in the hdf5 file."""
        instruments = self["Instruments"]
        return [Instrument(self, instruments.dtype, i) for i in instruments]

    @cached_property
    def _instrument_ids(self) -> List[str]:
        """IDs of all instruments in the hdf5 file."""
        return [i.id for i in self._instruments]

    def get_data(
        self,
        *channel_names: str,
        sort: bool = True,
        filters: Optional[Iterable[Tuple[str, Callable, float]]] = None,
        order: Optional[Iterable[str]] = None,
        slices: Optional[Iterable[Union[int, slice, Ellipsis]]] = None,
    ) -> Tuple[np.ndarray]:
        """Get the data from `channel_names`.

        Parameters
        ----------
        *channel_names
            Names of channels to get data from.
            If empty, data from all sweep and log channels are returned, with the
            sweep variable of any trace channels first.
        sort
            Sort the data so all stepped channels are monotonically increasing
            (default).  Otherwise, data remain in the order they were recorded.
        filters
            List of (channel_name, callable, value) used to filter the data.
            E.g. ("magnet", np.less, 5) will return data for which B < 5.  This will
            change the shape of the data and affect subsequent `slices`.
        order
            List of stepped channel names defining how the data axes should be ordered.
            If None, axes are ordered according to Labber (i.e. inner loop first).
        slices
            List of slices to take along each axis of the data.
            Axes are ordered according to `order`.  Ellipsis (...) is supported.
        """
        not_data_channels = set(channel_names) - set(self._data_channel_names)
        if not_data_channels:
            raise ValueError(
                f"{not_data_channels} are not data channels. Available data channels are:\n{pformat(self._data_channel_names)}"
            )
        if len(channel_names) == 0:
            channel_names = self._sweep_channel_names + self._log_channel_names
        data = tuple(self.get_channel(name).get_data() for name in channel_names)
        if sort:
            sweep_data = tuple(c.get_data() for c in self._sweep_channels)
            sweep_axes = tuple(c.axis for c in self._sweep_channels)
            sort_idxs = tuple(
                np.argsort(d, axis=ax) for d, ax in zip(sweep_data, sweep_axes)
            )
            for index, axis in zip(sort_idxs, sweep_axes):
                data = tuple(np.take_along_axis(d, index, axis) for d in data)
        if filters:
            sort = np.sort if sort else lambda a, *_: a
            masks, shapes = [], []
            for filt in filters:
                channel = self.get_channel(filt[0])
                axis = channel.axis
                mask = filt[1](sort(channel.get_data(), axis), filt[2])
                shape = list(data[0].shape)
                shape[axis] = -1
                for m, s in zip(masks, shapes):
                    mask = mask[m].reshape(s)
                masks.append(mask)
                shapes.append(shape)
                data = tuple(d[mask].reshape(shape) for d in data)
        if order:
            data = tuple(
                np.moveaxis(
                    d,
                    [self.get_channel(c).axis for c in order],
                    np.arange(len(order)),
                )
                for d in data
            )
        if slices:
            data = tuple(d[_expand_ellipsis(slices, d.ndim)] for d in data)

        return tuple(d.squeeze() for d in data)


class _DatasetRow:
    """A row in an hdf5 Dataset with named columns."""

    def __init__(self, file_, dtype, values):
        self._file = file_
        for name, value in zip(dtype.names, values):
            setattr(self, *_parse_field(name, value))

    def __repr__(self):
        v = vars(self)
        return (
            f"{type(self).__name__}("
            + ", ".join(
                f"{name}={value}"
                for name, value in vars(self).items()
                if name != "_file"
            )
            + ")"
        )


class Channel(_DatasetRow):
    """A quantity controlled or measured by an instrument."""

    @cached_property
    def instrument(self) -> Instrument:
        f = self._file
        return f._instruments[f._instrument_ids.index(self._instrument)]

    @cached_property
    def _step_config(self) -> StepConfig:
        f = self._file
        return f._step_configs[f._step_channel_names.index(self.name)]

    @property
    def _use_relations(self) -> bool:
        return self._step_config.use_relations

    @cached_property
    def _is_complex(self) -> bool:
        if self._is_trace_channel:
            return self._file[f"Traces/{self.name}"].attrs["complex"]
        else:
            return isinstance(self.instrument.config[self.quantity], complex)

    @property
    def _is_trace_channel(self) -> bool:
        return self.name in self._file._trace_channel_names

    @cached_property
    def _x_channel(self) -> XChannel:
        return XChannel(self)

    @cached_property
    def axis(self) -> int:
        """Axis along which the channel is stepped/swept."""
        # assume all trace channels share a single x channel
        if self._is_trace_channel:
            return 0
        else:
            return len(self._file._trace_dims) + self._step_config.axis

    def get_data(self) -> np.ndarray:
        if self.name not in self._file._data_channel_names:
            raise ValueError(
                f"'{self.name}' is not a data channel.  Available data channels are:\n"
                f"{pformat(self._file._data_channel_names)}"
            )
        if self._is_trace_channel:
            data = self._get_trace_data()
        else:
            data = self._get_data()
        data = data.squeeze()
        if self._file._x_channels and not self._is_trace_channel:
            expand = (np.newaxis,) * len(self._file._trace_dims) + (...,)
            data = np.broadcast_to(data[expand], self._file._trace_dims + data.shape)
        return data.reshape(self._file._shape, order="F")

    def _get_data(self) -> np.ndarray:
        def _data_column(info):
            column = self._file._data_channel_infos.index((self.name, info))
            return self._file["Data/Data"][:, column, ...]

        if self._is_complex:
            return _data_column("Real") + 1j * _data_column("Imaginary")
        else:
            return _data_column("")

    def _get_trace_data(self) -> np.ndarray:
        dset = self._file[f"Traces/{self.name}"]
        if self._is_complex:
            return dset[:, 0, ...] + 1j * dset[:, 1, ...]
        else:
            return dset[:, 0, ...]

    def get_fixed_value(self):
        if self.name in self._file._fixed_channel_names:
            return self._file["Data"].attrs["Fixed step values"][
                self._file._fixed_channel_names.index(self.name)
            ]
        else:
            return self.instrument.config[self.name.split(" - ")[-1]]


@dataclass
class XChannel:
    """A "virtual" channel containing a trace channel's step data."""

    _channel: Channel

    @property
    def name(self) -> str:
        return self._channel._file[f"Traces/{self._channel.name}"].attrs["x, name"]

    @property
    def unit(self) -> str:
        return self._channel._file[f"Traces/{self._channel.name}"].attrs["x, unit"]

    @property
    def npoints(self) -> int:
        (_,) = self._channel._file[f"Traces/{self._channel.name}_N"]
        return _

    @property
    def axis(self) -> int:
        return self._channel.axis

    def get_data(self) -> np.ndarray:
        return self._channel._file[f"Traces/{self._channel.name}"][:, -1, ...].reshape(
            self._channel._file._shape, order="F"
        )


class Instrument(_DatasetRow):
    """A driver controlling one or more pieces of hardware."""

    @property
    def config(self) -> dict:
        return dict(self._file[f"Instrument config/{self.id}"].attrs)


class StepConfig(_DatasetRow):
    """A specification of how a channel is stepped/swept."""

    @cached_property
    def step_items(self) -> List[StepItem]:
        step_items = self._file[f"Step config/{self.channel_name}/Step items"]
        return [StepItem(self._file, step_items.dtype, s) for s in step_items]

    @cached_property
    def axis(self) -> int:
        """Axis along which the channel is stepped/swept."""
        return self._file._step_channel_names.index(self.channel_name)


class AfterLast(Enum):
    GO_TO_FIRST = 0
    STAY_AT_FINAL = 1
    GO_TO_VALUE = 2


class SweepMode(Enum):
    OFF = 0
    BETWEEN_POINTS = 1
    CONTINUOUS = 2


class StepItem(_DatasetRow):
    """A specification of a range of values to step through."""


class RangeType(Enum):
    SINGLE = 0
    START_STOP = 1
    CENTER_SPAN = 2


class StepType(Enum):
    STEP_SIZE = 0
    N_POINTS = 1


class InterpType(Enum):
    LINEAR = 0
    LOG = 1
    LOG_NUM_PER_DECADE = 2
    LORENTZIAN = 3


def get_data_dir():
    """Get the path to the Labber Data directory.

    Returns
    -------
    Path
        The value of the environment variable LABBERDATA_DIR if set.
        Defaults to ~/Labber/Data.
    """
    if "LABBERDATA_DIR" in environ:
        path = Path(environ["LABBERDATA_DIR"])
    else:
        path = Path("~/Labber/Data").expanduser()
    if not path.exists():
        raise ValueError(f"{path} does not exist.")
    return path


def _parse_field(name, value):
    """Parse `name` and `value` into a more useful format and type."""
    name = name.replace(" ", "_")
    # `instrument` is reserved for the Instrument object
    if name == "instrument":
        name = "_instrument"
    if type(value) is bytes:
        value = value.decode("utf-8")
    value = _parsers.get(name, lambda _: _)(value)
    return name, value


_parsers = {
    "version": parse_version,
    "after_last": AfterLast,
    "sweep_mode": SweepMode,
    "range_type": RangeType,
    "step_type": StepType,
    "interp": InterpType,
}


def _expand_ellipsis(tup: Tuple, n: int) -> Tuple:
    """Expand ...  into :'s so that `tup` has length `n`."""
    if ... not in tup:
        return tup
    i = tup.index(...)
    list_ = list(tup)
    list_[i : i + 1] = (slice(None),) * (n - len(tup) + 1)
    return tuple(list_)


def _check_empty_list(self, list_) -> List[str]:
    if len(list_) == 1 and list_[0] == "":
        return []
    else:
        return list_
