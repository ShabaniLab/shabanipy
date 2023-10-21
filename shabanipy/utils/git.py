"""Unadvisable hacks for interacting with git from our scripts."""
import subprocess


def git_hash() -> str:
    """Get the hash of the current commit."""
    return (
        subprocess.check_output(["git", "describe", "--always"]).decode("ascii").strip()
    )
