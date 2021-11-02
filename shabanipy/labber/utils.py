"""Helper functions for dealing with Labber."""
import logging
from os import environ
from pathlib import Path

logger = logging.getLogger(__name__)


def get_data_dir():
    """Get the path to the Labber Data directory.

    Returns
    -------
    pathlib.Path
        The value of the environment variable LABBERDATA_DIR if set.
        Defaults to ~/Labber/Data.
    """
    LABBERDATA_DIR = "LABBERDATA_DIR"
    path = (
        Path(environ[LABBERDATA_DIR])
        if LABBERDATA_DIR in environ
        else Path("~/Labber/Data").expanduser()
    )
    if not path.exists():
        logger.warning(f"{path} does not exist")
    return path
