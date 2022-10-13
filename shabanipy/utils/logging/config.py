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
    if filename is None:
        filename = "default.ini"
    if not filename.endswith(".ini"):
        filename += ".ini"
    filepath = path.join(path.dirname(path.abspath(__file__)), "ini", filename)
    logging.config.fileConfig(filepath, disable_existing_loggers=False)
