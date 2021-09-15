"""Logging formatters."""
from logging import Formatter


class InformativeFormatter(Formatter):
    def format(self, record):
        return (
            f"[{record.levelname}]".ljust(11)
            + f"{record.filename}:{record.lineno}".ljust(25)
            + " "
            + record.getMessage()
            + (
                ("\n" + self.formatException(record.exc_info))
                if record.exc_info
                else ""
            )
        )
