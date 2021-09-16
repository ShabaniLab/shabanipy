"""Logging formatters."""
from logging import INFO, Formatter


class InformativeFormatter(Formatter):
    def format(self, record):
        return (
            f"[{record.levelname}]".ljust(11)
            + f"{record.filename}:{record.lineno}".ljust(25)
            + " "
            + record.getMessage()
            + (f"\n{self.formatException(record.exc_info)}" if record.exc_info else "")
        )


class ConsoleFormatter(Formatter):
    def format(self, record):
        return (
            f"[{record.levelname}]".ljust(11)
            + (f"{record.filename}:{record.lineno} " if record.levelno > INFO else "")
            + record.getMessage()
            + (f"\n{self.formatException(record.exc_info)}" if record.exc_info else "")
        )
