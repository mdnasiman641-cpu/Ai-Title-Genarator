"""Shared plumbing for background workers (``QRunnable`` + signals).

Workers do the slow things - scanning, AI calls, file operations, ffprobe - on
the global ``QThreadPool`` so the UI never blocks. Every worker reports through
:class:`WorkerSignals`; nothing here touches widgets.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

from app.utils.logger import get_logger

log = get_logger("worker")


class CancelledError(Exception):
    """Raised inside a worker when the user cancelled the job."""


class WorkerSignals(QObject):
    """Signals emitted by every worker."""

    progress = Signal(int, int, str)  # done, total, message
    status = Signal(str)  # free-form status line
    result = Signal(object)
    error = Signal(str, object)  # user message, original exception
    cancelled = Signal()
    finished = Signal()


class BaseWorker(QRunnable):
    """Runs :meth:`execute` off the UI thread and relays its outcome via signals."""

    signals_class: type[WorkerSignals] = WorkerSignals

    def __init__(self) -> None:
        super().__init__()
        self.signals = self.signals_class()
        self.cancel_event = threading.Event()
        # The thread pool owns the runnable (autoDelete): the C++ object is freed only after
        # run() has fully returned, so dropping the Python reference in a ``finished`` slot is safe.

    # -- API for subclasses ------------------------------------------------
    def execute(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def friendly_error(self, exc: BaseException) -> str:
        """Hook: convert an exception to a user-facing message."""
        return str(exc) or exc.__class__.__name__

    # -- helpers -----------------------------------------------------------
    def cancel(self) -> None:
        self.cancel_event.set()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError

    def emit_progress(self, done: int, total: int, message: str = "") -> None:
        self.signals.progress.emit(done, total, message)

    # -- QRunnable ---------------------------------------------------------
    def run(self) -> None:
        try:
            outcome = self.execute()
        except CancelledError:
            self.signals.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - a worker must never crash the app
            log.error("%s failed: %s\n%s", type(self).__name__, exc, traceback.format_exc())
            self.signals.error.emit(self.friendly_error(exc), exc)
        else:
            self.signals.result.emit(outcome)
        finally:
            self.signals.finished.emit()
