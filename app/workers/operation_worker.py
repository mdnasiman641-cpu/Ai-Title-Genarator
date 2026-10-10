"""Workers that change (or restore) files."""

from __future__ import annotations

from app.files.move_engine import FsError
from app.files.operation_manager import ExecutionResult, OperationManager, UndoResult
from app.files.plan import Plan
from app.workers.base_worker import BaseWorker


class OperationWorker(BaseWorker):
    """Applies a plan: journaled, all-or-nothing, with progress."""

    def __init__(self, manager: OperationManager, plan: Plan, *, command: str, provider: str, keep_history: bool) -> None:
        super().__init__()
        self.manager = manager
        self.plan = plan
        self.command = command
        self.provider = provider
        self.keep_history = keep_history

    def execute(self) -> ExecutionResult:
        return self.manager.execute(
            self.plan, command=self.command, provider=self.provider, keep_history=self.keep_history,
            cancel=self.cancel_event, progress=self.emit_progress,
        )  # fmt: skip

    def friendly_error(self, exc: BaseException) -> str:
        return exc.message if isinstance(exc, FsError) else super().friendly_error(exc)


class UndoWorker(BaseWorker):
    """Reverses a recorded operation after the safety pre-check."""

    def __init__(self, manager: OperationManager, operation_id: int) -> None:
        super().__init__()
        self.manager = manager
        self.operation_id = operation_id

    def execute(self) -> UndoResult:
        return self.manager.undo(self.operation_id, cancel=self.cancel_event, progress=self.emit_progress)
