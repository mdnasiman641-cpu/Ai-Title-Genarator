"""Last-resort error handling: log unexpected exceptions and show a friendly message.

Workers already convert their own failures into messages. This hook catches
anything that slips through (a bug in a slot, for example) so the application
reports it instead of silently dying, and never prints a raw traceback to the user.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from types import TracebackType

from PySide6.QtWidgets import QApplication, QMessageBox

from app.config.constants import APP_NAME
from app.i18n import tr
from app.utils.logger import get_log_dir, get_logger

log = get_logger("crash")
_MIN_SECONDS_BETWEEN_DIALOGS = 5.0
_last_dialog = 0.0


def _report(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None, *, show_dialog: bool) -> None:
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    log.error("Unhandled exception:\n%s", text)
    global _last_dialog
    now = time.monotonic()
    if not show_dialog or now - _last_dialog < _MIN_SECONDS_BETWEEN_DIALOGS:
        return
    _last_dialog = now
    if QApplication.instance() is None or threading.current_thread() is not threading.main_thread():
        return
    try:
        QMessageBox.critical(
            None, APP_NAME, tr("error.unexpected", error=f"{exc_type.__name__}: {exc}", logs=str(get_log_dir()))
        )
    except Exception:  # noqa: BLE001 - the error reporter must never raise
        log.exception("could not show the error dialog")


def install_exception_hooks() -> None:
    """Install ``sys.excepthook`` / ``threading.excepthook`` handlers."""

    def hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        _report(exc_type, exc, tb, show_dialog=True)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_value is not None and args.exc_type is not SystemExit:
            _report(args.exc_type, args.exc_value, args.exc_traceback, show_dialog=False)

    sys.excepthook = hook
    threading.excepthook = thread_hook
