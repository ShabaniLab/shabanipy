"""Input/output utilities."""
from os import environ
from pathlib import Path


def get_output_dir():
    """Get the path to the output directory.

    Returns
    -------
    Path
        The value of the environment variable SHABANIPY_OUTPUT_DIR if set.
        Defaults to "./output".
    """
    if "SHABANIPY_OUTPUT_DIR" in environ:
        path = Path(environ["SHABANIPY_OUTPUT_DIR"])
    else:
        path = Path("./output")
        path.mkdir(exist_ok=True)
    if not path.exists():
        raise ValueError(f"{path} does not exist.")
    return path
