"""Keep Python's cyclic garbage collector on the GUI thread.

PySide wrappers that end up in reference cycles are destroyed when the cyclic
GC runs - and the GC can run on *any* thread that allocates memory, including
our scan / AI worker threads. Deleting a ``QObject`` from a non-GUI thread
crashes Qt ("QBasicTimer::stop: Failed ..."), so automatic collection is turned
off and replaced by a periodic collection driven by a timer on the GUI thread.
"""

from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QTimer

_INTERVAL_MS = 4000


class MainThreadGC(QObject):
    """Runs ``gc.collect()`` periodically from the thread that owns this object."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        gc.disable()
        self._timer = QTimer(self)
        self._timer.setInterval(_INTERVAL_MS)
        self._timer.timeout.connect(self.collect)
        self._timer.start()

    @staticmethod
    def collect() -> None:
        gc.collect()

    def stop(self) -> None:
        self._timer.stop()
        gc.collect()
        gc.enable()
