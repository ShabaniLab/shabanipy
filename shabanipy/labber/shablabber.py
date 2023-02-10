"""Labber hdf5 datafiles.

This is a rewrite of `labber_io.py` with an emphasis on simplicity and maintainability.

The central class in this module is `ShaBlabberFile`, which encapsulates an hdf5
datafile created by Labber.  It is a replacement for `labber_io.LabberData`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from functools import cached_property
from os import environ
from pathlib import Path
from pprint import pformat
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
from h5py import File
from packaging.version import Version
from packaging.version import parse as parse_version


class ShaBlabberFile(File):
    """An hdf5 file created by Labber."""

    def __init__(self, path: Union[str, Path]):
        """
        The `path` to the datafile should be absolute, or relative to the path returned
        by `shablabber.get_data_dir()`.
        """
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = get_data_dir() / path
        self.path = path
        super().__init__(str(path), "r")

    @property
    def log_name(self) -> str:
        """The internal name of the hdf5 log file."""
        return self.attrs["log_name"]

    @cached_property
    def _creation_time(self) -> datetime:
        """The date and time the hdf5 file was created."""
        return datetime.fromtimestamp(self.attrs["creation_time"])

    @property
    def _step_dims(self) -> np.ndarray:
        """Step dimensions, i.e. the shape of the data."""
        return self.attrs["Step dimensions"]

    @cached_property
    def _shape(self) -> tuple:
        """The shape of the data, with trivial dimensions removed."""
        return tuple(d for d in self._step_dims if d != 1)

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
        return self._check_empty_list(list(self["Tags"].attrs["Project"]))

    @cached_property
    def tags(self) -> List[str]:
        """Tags assigned to the file."""
        return self._check_empty_list(list(self["Tags"].attrs["Tags"]))

    def _check_empty_list(self, listt) -> List[str]:
        if len(listt) == 1 and listt[0] == "":
            return []
        else:
            return listt

    @cached_property
    def _channels(self) -> List[Channel]:
        """All channels in the hdf5 file."""
        return [Channel(self, *c) for c in self["Channels"]]

    @cached_property
    def _channel_names(self) -> List[str]:
        """Names of all channels in the hdf5 file."""
        return [c.name for c in self._channels]

    def get_channel(self, channel_name: str) -> Channel:
        """Get the channel with name `channel_name`."""
        if channel_name not in self._channel_names:
            raise ValueError(
                f"'{channel_name}' does not exist.  Available channels are:\n{pformat(self._channel_names)}"
            )
        return self._channels[self._channel_names.index(channel_name)]

    @cached_property
    def _data_channels(self) -> List[Channel]:
        """Channels that are stepped/swept or logged/measured."""
        return [self.get_channel(name) for name in self._data_channel_names]

    @cached_property
    def _data_channel_names(self) -> List[str]:
        """Names of channels that are stepped/swept or logged/measured."""
        return [name.decode("utf-8") for name, _ in self["Data/Channel names"]]

    @cached_property
    def _data_channel_infos(self) -> List[Tuple[str, str]]:
        """Names of data channels with their associated info."""
        return [
            (name.decode("utf-8"), info.decode("utf-8"))
            for name, info in self["Data/Channel names"]
        ]

    @cached_property
    def _step_channels(self) -> List[Channel]:
        """Channels that are added to the 'Step sequence'."""
        return [self.get_channel(name) for name in self._step_channel_names]

    @cached_property
    def _step_channel_names(self) -> List[str]:
        """Names of channels that are added to the 'Step sequence'."""
        return [sc.channel_name for sc in self._step_configs]

    @cached_property
    def _step_configs(self) -> List[StepConfig]:
        """Step configurations for channels that are added to the 'Step sequence'."""
        return [StepConfig(self, *sc) for sc in self["Step list"]]

    def _get_step_config(self, channel_name) -> StepConfig:
        """Get the step config for `channel_name`."""
        if channel_name not in self._step_channel_names:
            raise ValueError(
                f"'{channel_name}' is not in the step sequence.  Available channels in the step sequence are:\n{pformat(self._step_channel_names)}"
            )
        return self._step_configs[self._step_channel_names.index(channel_name)]

    @cached_property
    def _stepped_step_channels(self) -> List[Channel]:
        """Channels in the 'Step sequence' that are actually stepped/swept."""
        return [
            self.get_channel(name)
            for name in np.array(self._step_channel_names)[self._step_idxs]
        ]

    @cached_property
    def _stepped_step_channel_names(self) -> List[str]:
        """Names of channels in the 'Step sequence' that are actually stepped/swept."""
        return [c.name for c in self._stepped_step_channels]

    @cached_property
    def _fixed_step_channels(self) -> List[Channel]:
        """Channels in the 'Step sequence' that are fixed at a single value."""
        return [
            self.get_channel(name)
            for name in np.array(self._step_channel_names)[self._fixed_idxs]
        ]

    @property
    def _step_idxs(self) -> List[int]:
        """Indexes of channels in the 'Step sequence' that are actually stepped/swept."""
        return self["Data"].attrs["Step index"]

    @property
    def _fixed_idxs(self) -> List[int]:
        """Indexes of channels in the 'Step sequence' that are fixed at a single value."""
        return self["Data"].attrs["Fixed step index"]

    @cached_property
    def _log_channels(self) -> List[Channel]:
        """Channels that are logged/measured."""
        return [self.get_channel(name) for name in self._log_channel_names]

    @cached_property
    def _log_channel_names(self) -> List[str]:
        """Names of channels that are logged/measured."""
        return [name.decode("utf-8") for name, in self["Log list"]]

    @cached_property
    def _instruments(self) -> List[Instrument]:
        """All instruments in the hdf5 file."""
        return [Instrument(self, *i) for i in self["Instruments"]]

    @cached_property
    def _instrument_ids(self) -> List[str]:
        """IDs of all instruments in the hdf5 file."""
        return [i.id for i in self._instruments]

    def _get_instrument_by_id(self, instrument_id: str) -> Instrument:
        """Get the instrument with ID `instrument_id`."""
        if instrument_id not in self._instrument_ids:
            raise ValueError(
                f"'{instrument_id}' does not exist.  Available instrument IDs are:\n{pformat(self._instrument_ids)}"
            )
        return self._instruments[self._instrument_ids.index(instrument_id)]

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
            If empty, data from all step and log channels are returned.
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
        if len(channel_names) == 0:
            channel_names = self._stepped_step_channel_names + self._log_channel_names
        data = tuple(self.get_channel(name).get_data() for name in channel_names)
        if sort:
            step_data = tuple(c.get_data() for c in self._stepped_step_channels)
            step_axes = tuple(c._step_config.axis for c in self._stepped_step_channels)
            sort_idxs = tuple(
                np.argsort(sd, axis=ax) for sd, ax in zip(step_data, step_axes)
            )
            for index, axis in zip(sort_idxs, step_axes):
                data = tuple(np.take_along_axis(d, index, axis) for d in data)
        if filters:
            masks, shapes = [], []
            for filt in filters:
                channel = self.get_channel(filt[0])
                axis = channel._step_config.axis
                mask = filt[1](np.sort(channel.get_data(), axis), filt[2])
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
                    [self.get_channel(c)._step_config.axis for c in order],
                    np.arange(len(order)),
                )
                for d in data
            )
        if slices:
            data = tuple(d[_expand_ellipsis(slices, d.ndim)] for d in data)

        return data


@dataclass
class Channel:
    """A quantity controlled or measured by an instrument."""

    # hdf5 file this channel belongs to
    _file: ShaBlabberFile

    # channel properties defined by Labber in the "Channels" hdf5 dataset
    name: str
    instrument_id: str
    quantity: str
    unitPhys: str
    unitInstr: str
    gain: float
    offset: float
    amp: float
    highLim: float
    lowLim: float
    outputChannel: str
    limit_action: str
    limit_run_script: bool
    limit_script: str
    use_log_interval: bool
    log_interval: float
    limit_run_always: bool

    def __post_init__(self):
        _bytes_to_str(self)

    @cached_property
    def instrument(self) -> Instrument:
        return self._file._get_instrument_by_id(self.instrument_id)

    @cached_property
    def _step_config(self) -> StepConfig:
        return self._file._get_step_config(self.name)

    @property
    def _is_complex(self) -> bool:
        return isinstance(self.instrument.config[self.quantity], complex)

    def get_data(self) -> np.ndarray:
        if self.name not in self._file._data_channel_names:
            raise ValueError(
                f"'{self.name}' is not a data channel.  Available data channels are:\n"
                f"{pformat(list(dict.fromkeys(self._file._data_channel_names)))}"
            )
        if self._is_complex:
            return self._get_complex_data()
        else:
            f = self._file
            return f["Data/Data"][
                :, f._data_channel_names.index(self.name), ...
            ].reshape(f._shape, order="F")

    def _get_complex_data(self) -> np.ndarray:
        f = self._file
        real = f["Data/Data"][:, f._data_channel_infos.index((self.name, "Real")), ...]
        imag = f["Data/Data"][
            :, f._data_channel_infos.index((self.name, "Imaginary")), ...
        ]
        return (real + 1j * imag).reshape(f._shape, order="F")


@dataclass
class Instrument:
    """A driver controlling one or more pieces of hardware."""

    # hdf5 file this instrument belongs to
    _file: ShaBlabberFile

    # instrument properties defined by Labber in the "Instruments" hdf5 dataset
    hardware: str  # really the driver
    version: Version
    id: str
    model: str
    name: str
    interface: int
    address: str
    server: str
    startup: int
    lock: bool
    show_advanced: bool
    timeout: float
    term_char: str
    send_end_on_write: bool
    lock_visa_resource: bool
    suppress_end_bit_termination_on_read: bool
    use_specific_tcp_port: bool
    tcp_port: str
    use_vicp_protocol: bool
    baud_rate: float
    data_bits: float
    stop_bits: float
    parity: str
    gpib_board_number: float
    send_gpib_go_to_local_at_close: bool
    pxi_chassis: float
    run_in_32_bit_mode: bool

    def __post_init__(self):
        _bytes_to_str(self)
        self.version = parse_version(self.version)

    @property
    def config(self) -> dict:
        return dict(self._file[f"Instrument config/{self.id}"].attrs)


@dataclass
class StepConfig:
    """A specification of how a channel is stepped/swept."""

    # hdf5 file this step config belongs to
    _file: ShaBlabberFile

    # step config specification defined by Labber in the "Step list" hdf5 dataset
    channel_name: str
    step_unit: int
    wait_after: float
    after_last: AfterLast
    final_value: float
    use_relations: bool
    equation: str
    show_advanced: bool
    sweep_mode: SweepMode
    use_outside_sweep_rate: bool
    sweep_rate_outside: float
    alternate_direction: bool

    def __post_init__(self):
        # here and elsewhere it would be nice to use Converters from the `attrs` package
        _bytes_to_str(self)
        self.after_last = AfterLast(self.after_last)
        self.sweep_mode = SweepMode(self.sweep_mode)

    @cached_property
    def step_items(self) -> List[StepItem]:
        return [
            StepItem(*si)
            for si in self._file[f"Step config/{self.channel_name}/Step items"]
        ]

    @cached_property
    def axis(self) -> int:
        """Axis along which `channel_name` is stepped/swept."""
        return self._file._step_channel_names.index(self.channel_name)


class AfterLast(Enum):
    GO_TO_FIRST = 0
    STAY_AT_FINAL = 1
    GO_TO_VALUE = 2


class SweepMode(Enum):
    OFF = 0
    BETWEEN_POINTS = 1
    CONTINUOUS = 2


@dataclass
class StepItem:
    """A specification of a range of values to step through."""

    range_type: RangeType
    step_type: StepType
    single: float
    start: float
    stop: float
    center: float
    span: float
    step_size: float
    n_points: int
    interp_type: InterpType
    sweep_rate: float

    def __post_init__(self):
        self.range_type = RangeType(self.range_type)
        self.step_type = StepType(self.step_type)
        self.interp_type = InterpType(self.interp_type)


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


def _bytes_to_str(dataclass):
    """Convert all fields in `dataclass` of type `bytes` to type `str`."""
    for field in fields(dataclass):
        value = getattr(dataclass, field.name)
        if type(value) is bytes:
            setattr(dataclass, field.name, value.decode("utf-8"))


def _expand_ellipsis(tup: Tuple, n: int) -> Tuple:
    """Expand ...  into :'s so that `tup` has length `n`."""
    if ... not in tup:
        return tup
    i = tup.index(...)
    listt = list(tup)
    listt[i : i + 1] = (slice(None),) * (n - len(tup) + 1)
    return tuple(listt)
