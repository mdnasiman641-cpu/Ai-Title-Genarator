"""Classifies a command into a :class:`TaskType` so the router can pick a suitable model.

Pure heuristics (keywords in English and Bangla + the number of files); no AI call.
Simple deterministic commands are recognised by the offline parser and never reach
an AI model at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config.constants import Capability, TaskType
from app.files.local_commands import parse_local_command

LARGE_BATCH_FILES = 200

_PATTERNS: tuple[tuple[TaskType, re.Pattern[str]], ...] = (
    (TaskType.DESCRIPTION_GENERATION, re.compile(r"description|বিবরণ|বর্ণনা", re.IGNORECASE)),
    (TaskType.TITLE_GENERATION, re.compile(r"\btitles?\b|youtube|শিরোনাম|টাইটেল", re.IGNORECASE)),
    (TaskType.FILE_ORGANIZATION, re.compile(r"\bfolders?\b|\bmove\b|\borgani[sz]e|\bgroup\b|\bcopy\b|ফোল্ডার|সরাও|গুছিয়ে", re.IGNORECASE)),
    (TaskType.COMPLEX_RENAME, re.compile(r"\brename\b|\bclean\b|\bepisode|\bseason\b|নাম|পর্ব|সিজন", re.IGNORECASE)),
    (TaskType.STRUCTURED_JSON, re.compile(r"\bjson\b|\bstructured\b", re.IGNORECASE)),
)


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """What the request needs from a model."""

    task_type: TaskType
    file_count: int
    required: frozenset[str]  # every candidate should have these (relaxed if none has them)
    preferred: frozenset[str]  # nice to have: ranks candidates higher

    @property
    def needs_ai(self) -> bool:
        return self.task_type != TaskType.SIMPLE_DETERMINISTIC


def analyze_task(command: str, file_count: int) -> TaskProfile:
    """Classify ``command`` for ``file_count`` selected files."""
    base = frozenset({Capability.TEXT.value, Capability.JSON.value})
    if parse_local_command(command) is not None:
        return TaskProfile(TaskType.SIMPLE_DETERMINISTIC, file_count, frozenset(), frozenset())
    task = TaskType.GENERAL
    for candidate, pattern in _PATTERNS:
        if pattern.search(command):
            task = candidate
            break
    if file_count > LARGE_BATCH_FILES and task in (TaskType.GENERAL, TaskType.COMPLEX_RENAME, TaskType.TITLE_GENERATION):
        task = TaskType.LARGE_BATCH
    preferred: set[str] = set()
    if task == TaskType.LARGE_BATCH or file_count > LARGE_BATCH_FILES:
        preferred.add(Capability.LONG_CONTEXT.value)
    if task in (TaskType.FILE_ORGANIZATION, TaskType.STRUCTURED_JSON, TaskType.GENERAL) and file_count <= 50:
        preferred.add(Capability.FAST.value)
    return TaskProfile(task, file_count, base, frozenset(preferred))
