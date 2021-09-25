# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Helper routines to plot data.

To be used in routines for summary steps in datanalysis.

"""
import os
from pickle import dump
from typing import Any, Dict, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .data_exploring import format_classifiers


def add_title_and_save(
    figure: Figure,
    directory: Optional[str],
    classifiers: Optional[Dict[int, Dict[str, Any]]],
    dir_per_plot_format: bool = True,
) -> None:
    """Add a meaningful title to a figure and save it.

    Parameters
    ----------
    figure : Figure
        Figure to be altered and saved.
    directory : Optional[str]
        Main directory in which to save
    classifiers : Optional[Dict[int, Dict[str, Any]]]
        Classifiers if the figure was generated as part of a summary data analysis step.
    dir_per_plot_format : bool, optional
        Should each plot format be saved in a separate subdirectory.

    """
    if classifiers:
        figure.suptitle(format_classifiers(classifiers, " "))
    if directory and classifiers:
        filename = format_classifiers(classifiers, "-")
        if dir_per_plot_format:
            png_dir = os.path.join(directory, "png")
            pdf_dir = os.path.join(directory, "pdf")
            pickle_dir = os.path.join(directory, "pickle")
        else:
            png_dir = pdf_dir = pickle_dir = directory
        for p in [png_dir, pdf_dir, pickle_dir]:
            if not os.path.isdir(p):
                os.makedirs(p)
        figure.savefig(os.path.join(png_dir, filename + ".png"))
        figure.savefig(os.path.join(pdf_dir, filename + ".pdf"))
        with open(os.path.join(pickle_dir, filename + ".pickle"), "wb") as f:
            dump(figure, f)
        figure.clf()
        plt.close(figure)
