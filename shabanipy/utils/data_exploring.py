# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to navigate HDF5 file created by data aggregation or analysis routines.

"""
import logging
import os
import re
from collections import defaultdict, namedtuple
from dataclasses import astuple, dataclass, field, fields
from itertools import product, chain
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
from h5py import File, Group, Dataset

from .labber_io import LabberData, LogEntry, StepConfig


def format_classifiers(classifiers: Dict[int, Dict[str, Any]], separator: str) -> str:
    """Format the classifiers in a nice str format.

    Parameters
    ----------
    classifiers: Dict[int, Dict[str, Any]]
        Classifiers stored per level
    separator: str
        Separator to use between different classifiers

    """
    fmt = ""
    for name, value in chain.from_iterable(c.items() for c in classifiers.values()):
        if isinstance(value, float):
            fmt += f"{name}={value:g}"
        else:
            fmt += f"{name}={value}"
        fmt += separator
    return fmt


@dataclass
class DataExplorer:
    """Navigate datafile created through aggregation or processing.

    Valid datafile are those created by DataClassifier.consolidate_data and
    DataProcessor.run_process

    In those files the top-level refers either to:
    - a kind of measurement (DataClassifier.consolidate_data)
    - a tier of analysis (DataProcessor.run_process)

    """

    #: Path to the file to open.
    path: str

    #: Should the file be open such as to allow to edit it.
    allow_edits: bool = False

    #: Should a brand new file be created.
    create_new: bool = False

    def open(self) -> None:
        """Open the underlying HDF5 file."""
        mode = "w" if self.create_new else ("r+" if self.allow_edits else "r")
        self._file = File(self.path, mode)

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        if self._file:
            self._file.close()
        self._file = None

    def __enter__(self) -> "DataExplorer":
        """Open the underlying HDF5 file when used as a context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the underlying HDF5 file when used as a context manager."""
        self.close()

    def list_top_level(self) -> List[str]:
        """List the top level groups (measurement or tier)."""
        if not self._file:
            raise RuntimeError("No opened datafile")
        return list(self._file.keys())

    def list_classifiers(self, measurement) -> Dict[int, List[str]]:
        """List the classifiers name by level."""
        if not self._file:
            raise RuntimeError("No opened datafile")
        if measurement not in self._file:
            raise ValueError(
                f"No measurement {measurement} in opened datafile, "
                f"existing measurements are {self.list_top_level()}"
            )

        def extract_classifiers(
            group: Group, classifiers: Dict[int, List[str]], level: int
        ) -> Dict[int, List[str]]:
            # By construction the classifiers are the same on each level
            # so we only visit one level of each
            for entry in group.values():
                if isinstance(entry, Group):
                    classifiers[level] = list(entry.attrs)
                    extract_classifiers(entry, classifiers, level + 1)
                    break
            return classifiers

        return extract_classifiers(self._file[measurement], dict(), 0)

    def walk_data(
        self, measurement: str
    ) -> Iterator[Tuple[Dict[int, Dict[str, Any]], Group]]:
        """Iterate over all the data found under one top level entry.

        This function provides the classifiers and the group containing the
        datasets of interest.

        """
        # Maximal depth of classifiers
        max_depth = len(self.list_classifiers(measurement))

        def yield_classifier_and_data(
            group: Group, depth: int, classifiers: Dict[int, Dict[str, Any]]
        ) -> Iterator[Tuple[Dict[int, Dict[str, Any]], Group]]:
            # If the group has any dataset yield it and then keep going
            # This is relevant for processed data merged from different measurements
            if any(isinstance(k, Dataset) for k in group.values()):
                yield classifiers, group
            if depth == max_depth - 1:
                for g in [g for g in group.values() if isinstance(g, Group)]:
                    clfs = classifiers.copy()
                    clfs[depth] = dict(g.attrs)
                    yield clfs, g
            else:
                for g in group.values():
                    clfs = classifiers.copy()
                    clfs[depth] = dict(g.attrs)
                    yield from yield_classifier_and_data(g, depth + 1, clfs)

        yield from yield_classifier_and_data(self._file[measurement], 0, dict())

    def get_data(
        self, measurement: str, classifiers: Dict[int, Dict[str, Any]]
    ) -> Group:
        """Retrieve the group containing the datasets corresponding to the classifiers.

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

    def require_group(
        self, toplevel: str, classifiers: Dict[int, Dict[str, Any]]
    ) -> Group:
        """Access the group matching the toplevel and classifiers.

        If any group does not exist it is created.

        """
        # Ensure the top group is present
        group = self._file.require_group(toplevel)

        # At each classifier level check if the group exist, create it if necessary
        for level, values in classifiers.items():
            key = "&".join(f"{k}::{values[k]}" for k in sorted(values))
            if key not in group:
                group = group.create_group(key)
                group.attrs.update(values)
            else:
                group = group[key]

        return group
