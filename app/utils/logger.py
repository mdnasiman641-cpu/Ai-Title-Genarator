"""Application logging with automatic secret redaction."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from app.config.constants import APP_NAME, APP_VERSION
from app.utils.helpers import get_app_data_dir
from app.utils.security import redact

LOGGER_NAME = "aivfa"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class RedactingFilter(logging.Filter):
    """Scrubs API keys from every record before it is written anywhere."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def get_logger(name: str = "") -> logging.Logger:
    """Return a child of the application logger (``aivfa.<name>``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def get_log_dir() -> Path:
    """Directory holding the rotating log files."""
    path = get_app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level: str = "INFO", *, to_file: bool = True) -> logging.Logger:
    """Configure the root application logger (idempotent; safe to call again to change level)."""
    logger = logging.getLogger(LOGGER_NAME)
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    redactor = RedactingFilter()
    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.addFilter(redactor)
    logger.addHandler(stream)

    if to_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                get_log_dir() / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
        except OSError:  # unwritable profile: keep logging to the console only
            file_handler = None
        if file_handler is not None:
            file_handler.setFormatter(formatter)
            file_handler.addFilter(redactor)
            logger.addHandler(file_handler)
    logger.info("%s %s starting (log level %s)", APP_NAME, APP_VERSION, logging.getLevelName(numeric))
    return logger
