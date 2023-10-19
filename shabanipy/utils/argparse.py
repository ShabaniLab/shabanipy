"""Utilities for building command line interfaces.

Script arguments are fetched in the following order of precedence:
    1. command line (argparse)
    2. config file (configparser)
    3. default value (argparse)

Positional and list arguments are not yet supported.

The ConfigArgParse module exists (https://pypi.org/project/ConfigArgParse/) but may not
do everything/exactly what we want.  For now the below is a lightweight solution without
external dependencies.
"""
import argparse
import sys

from .configparser import load_config


class ConfArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(**kwargs)

        group = self.add_argument_group(
            title="Configuration files",
            description="Script arguments can be provided in a configparser .ini file.",
        )
        group.add_argument(
            "--config_path",
            "-cp",
            help="path to config file containing additional script arguments",
        )
        group.add_argument(
            "--config_section", "-cs", help="section of the config file to use"
        )

        self.args, _ = super().parse_known_args(
            [a for a in sys.argv[1:] if a not in ("-h", "--help")]
        )

    def parse_args(self) -> argparse.Namespace:
        configargs = []
        if self.args.config_path is not None:
            _, config = load_config(self.args.config_path, self.args.config_section)
            # n.b. private variable _actions may break if argparse is updated
            argsdict = {action.dest: action for action in self._actions}
            # parse config arguments as if they were passed via the command line
            for k, v in config.items():
                if k in argsdict:
                    action = argsdict[k]
                    try:
                        configargs += [action.option_strings[0], v]
                    except IndexError:
                        raise NotImplementedError(
                            f"Setting positional arguments ({k}) in a config is not yet supported by {self.__class__}."
                        )
        # redundant arguments override earlier ones
        return super().parse_args(configargs + sys.argv[1:], namespace=self.args)
