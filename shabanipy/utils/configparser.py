"""Utilities for dealing with configparser config files."""

from configparser import ConfigParser, ExtendedInterpolation, SectionProxy
from pathlib import Path
from pprint import pformat
from typing import Optional, Tuple, Union


def load_config(
    path: Union[str, Path], section: Optional[str] = None
) -> Tuple[ConfigParser, Optional[SectionProxy]]:
    """Load a configuration file.

    Parameters
    ----------
    path
        The path to the config file, i.e. an .ini file in configparser format.
    section (optional)
        A section of the config file to access.

    Returns
    -------
    ini, config
        The root of the .ini file dictionary, and the config section specified by
        `section`.  If `section` is None, `config` is None.
    """
    with open(Path(path)) as f:
        ini = ConfigParser(interpolation=ExtendedInterpolation())
        ini.read_file(f)
        if section is not None:
            try:
                config = ini[section]
            except KeyError as e:
                raise ValueError(
                    f"'{section}' not found. Available sections are:\n{pformat(sorted(ini.sections()))}"
                ) from e
        else:
            config = None
    return ini, config
