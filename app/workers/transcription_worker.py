"""Background transcription for *Analyze Selected Videos* (the UI never waits on ffmpeg or the API)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal

from app.ai.transcription import TranscriptResult, TranscriptService
from app.files.scanner import FileEntry
from app.i18n import tr
from app.workers.base_worker import BaseWorker, WorkerSignals


class TranscriptionSignals(WorkerSignals):
    item = Signal(str, object)  # rel_path, TranscriptResult - one per video, as soon as it is done
    stage = Signal(str, str)  # rel_path, stage (download / load / extract / transcribe) - live status per video


class TranscriptionWorker(BaseWorker):
    """Transcribes the first N seconds of every selected video, one after another.

    A failure only affects its own video; Cancel stops before the next video (the current
    one finishes, so no half-written cache entries or temporary files are left behind).
    """

    signals_class = TranscriptionSignals

    def __init__(self, entries: list[FileEntry], workspace: Path, service: TranscriptService, seconds: int, language: str) -> None:
        super().__init__()
        self.entries = entries
        self.workspace = workspace
        self.service = service
        self.seconds = seconds
        self.language = language

    def execute(self) -> list[tuple[str, TranscriptResult]]:
        results: list[tuple[str, TranscriptResult]] = []
        total = len(self.entries)
        for index, entry in enumerate(self.entries):
            self.check_cancelled()

            def step(stage: str, index: int = index, rel_path: str = entry.rel_path) -> None:
                self.signals.stage.emit(rel_path, stage)  # type: ignore[attr-defined]
                self.emit_progress(index, total, tr(f"transcript.step_{stage}", index=index + 1, total=total))

            self.emit_progress(index, total, tr("transcript.step_extract", index=index + 1, total=total))
            result = self.service.get(self.workspace / entry.rel_path, self.seconds, self.language, step=step)
            results.append((entry.rel_path, result))
            self.signals.item.emit(entry.rel_path, result)  # type: ignore[attr-defined]
        ok = sum(r.ok for _p, r in results)
        self.emit_progress(total, total, tr("transcript.step_done", count=ok, total=total))
        return results
