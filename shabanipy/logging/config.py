"""Configure logging with .ini files."""
import logging.config
from os import path


def configure_logging(filename=None):
    """Load the logging configuration in `filename`.

    Parameters
    ----------
    filename : str (optional)
        A .ini configuration file to load.  The file should be located in `./ini/`.  If
        None, `default.ini` will be used.
    """
    filename = "default.ini" if filename is None else filename
    filename += ".ini" if not filename.endswith(".ini") else ""
    filepath = path.join(path.dirname(path.abspath(__file__)), "ini", filename)
    logging.config.fileConfig(filepath, disable_existing_loggers=False)
