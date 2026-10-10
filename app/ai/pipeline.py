"""From a natural-language command to a validated plan.

1. *Offline first*: simple commands are recognised locally and cost nothing.
2. Otherwise the task is classified and the model router picks a suitable model.
   The AI is asked **once per batch** (batches respect both "files per request"
   and a payload-size budget). If its answer to the first batch consists of rule
   actions, those are applied to *all* selected files locally and no further
   requests are made; only per-file (explicit) answers need follow-up batches.
3. Every answer is validated strictly (invalid JSON -> retry / next model; unsafe ->
   stop), then turned into a plan by the planner.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.ai.ai_router import (
    InvalidAnswer,
    ModelRouter,
    NoModelsAvailable,
    RouteResult,
    UnsafeAnswer,
)
from app.ai.base_provider import AIResponse
from app.ai.errors import ErrorKind
from app.ai.prompt_builder import (
    BatchInfo,
    build_request,
    estimate_request_chars,
    split_batches,
)
from app.ai.response_parser import ResponseError, parse_response
from app.ai.schemas import Action, ParsedResponse, RejectedAction
from app.ai.task_analyzer import TaskProfile, analyze_task
from app.ai.title_config import TitleConfig
from app.ai.web_search import SearchService
from app.config.constants import ProcessingMode, TaskType
from app.config.settings import AppSettings
from app.files.local_commands import parse_local_command
from app.files.plan import Plan
from app.files.planner import Planner, PlannerOptions
from app.files.scanner import FileEntry
from app.i18n import tr
from app.utils.logger import get_logger

log = get_logger("pipeline")
ProgressFn = Callable[[int, int, str], None]


class PipelineError(Exception):
    """A user-presentable failure while turning a command into a plan."""

    def __init__(self, message: str, *, kind: ErrorKind | None = None, unsafe: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.unsafe = unsafe


class PipelineCancelled(Exception):
    """The user cancelled while the pipeline was waiting on the AI."""


@dataclass(slots=True)
class PipelineRequest:
    """Everything needed to plan one command."""

    command: str
    files: list[FileEntry]
    workspace: Path
    processing: ProcessingMode
    settings: AppSettings
    title: TitleConfig | None = None  # Title Generator settings (used when ``title.enabled``)
    search: SearchService | None = None  # Google Search context for the Title Generator


@dataclass(slots=True)
class PipelineResult:
    """A ready-to-preview plan plus how it was produced."""

    plan: Plan
    source: str  # "offline" | "ai"
    provider_text: str  # e.g. "Gemini / gemini-2.5-flash ✓"
    provider_used: str  # model label(s) or "offline"
    ai_requests: int = 0
    fallbacks: int = 0
    summary: str = ""
    task_type: str = ""
    warnings: list[str] = field(default_factory=list)
    route_details: list[str] = field(default_factory=list)
    files: int = 0

    @property
    def completion_text(self) -> str:
        """``Completed · Model: Gemini / gemini-2.5-flash · Requests: 5 · Fallbacks: 1 · Files: 500``."""
        if self.source == "offline":
            return tr("pipeline.completed_offline", files=len({op.source for op in self.plan.ops if op.source}))
        return tr(
            "pipeline.completed", model=self.provider_used, requests=self.ai_requests, fallbacks=self.fallbacks,
            files=self.files,
        )  # fmt: skip


def planner_options(settings: AppSettings) -> PlannerOptions:
    """Planner switches derived from the File Operations settings."""
    return PlannerOptions(
        prevent_overwrite=settings.prevent_overwrite,
        duplicate_policy=settings.duplicate_policy,
        delete_mode=settings.delete_mode,
    )


def _validate(response: AIResponse) -> ParsedResponse:
    """Router validator: structured output or a reason to retry / stop."""
    try:
        return parse_response(response.text)
    except ResponseError as exc:
        if exc.unsafe:
            raise UnsafeAnswer(str(exc)) from exc
        raise InvalidAnswer(str(exc)) from exc


def run_pipeline(
    request: PipelineRequest,
    router: ModelRouter,
    *,
    cancel: threading.Event | None = None,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    """Plan ``request.command`` for ``request.files`` (offline when possible, otherwise via the AI)."""
    cancel = cancel or threading.Event()
    command = request.command.strip()
    if request.title is not None and request.title.enabled:
        return _run_title_generator(request, router, cancel, progress)
    if not command:
        raise PipelineError(tr("pipeline.no_command"))
    if not request.files:
        raise PipelineError(tr("pipeline.no_files"))

    use_local = request.processing == ProcessingMode.OFFLINE_ONLY or (
        request.processing == ProcessingMode.AUTO and request.settings.use_offline_parser
    )
    local = parse_local_command(command) if use_local else None
    profile = analyze_task(command, len(request.files))
    if local is not None:
        parsed = ParsedResponse(actions=list(local.actions), summary=local.description)
        log.info("Command handled offline (%d actions, %d files)", len(parsed.actions), len(request.files))
        result_meta = {"source": "offline", "provider_text": tr("pipeline.offline_used"), "provider_used": "offline",
                       "ai_requests": 0, "fallbacks": 0, "route_details": []}  # fmt: skip
    elif request.processing == ProcessingMode.OFFLINE_ONLY:
        raise PipelineError(tr("pipeline.offline_cannot"))
    else:
        parsed, result_meta = _ask_ai(request, command, router, profile, cancel, progress)

    plan = Planner(request.workspace, request.files, planner_options(request.settings)).build(
        parsed.actions, parsed.rejected, summary=parsed.summary
    )
    log.info(
        "Pipeline done: source=%s model=%s task=%s requests=%s fallbacks=%s ops=%d",
        result_meta["source"], result_meta["provider_used"], profile.task_type.value, result_meta["ai_requests"],
        result_meta["fallbacks"], len(plan.ops),
    )  # fmt: skip
    return PipelineResult(
        plan=plan, summary=parsed.summary, warnings=list(parsed.warnings), task_type=profile.task_type.value,
        files=len(request.files), **result_meta,  # type: ignore[arg-type]
    )  # fmt: skip


def _ask_ai(
    request: PipelineRequest,
    command: str,
    router: ModelRouter,
    profile: TaskProfile,
    cancel: threading.Event,
    progress: ProgressFn | None,
) -> tuple[ParsedResponse, dict[str, object]]:
    files = request.files
    batches = split_batches(files, request.settings.batch_size, request.settings.max_request_chars)
    log.info(
        "AI task %s: %d files in %d batch(es), ~%d chars", profile.task_type.value, len(files), len(batches),
        estimate_request_chars(files),
    )  # fmt: skip
    actions: list[Action] = []
    rejected: list[RejectedAction] = []
    warnings: list[str] = []
    summaries: list[str] = []
    routes: list[RouteResult] = []
    requests = 0
    explicit_only = False
    first_item = 1
    for index, batch in enumerate(batches, start=1):
        if cancel.is_set():
            raise PipelineCancelled
        info = BatchInfo(index, len(batches), first_item, first_item + len(batch) - 1)
        first_item += len(batch)
        if progress is not None:
            progress(index - 1, len(batches), tr("pipeline.asking", index=index, count=len(batches)))
        ai_request = build_request(
            command, batch, info,
            folder_name=request.workspace.name or "folder",
            total_selected=len(files),
            explicit_only=explicit_only,
            include_metadata=request.settings.send_metadata_to_ai,
        )  # fmt: skip
        try:
            route = router.generate(ai_request, profile=profile, validate=_validate, cancel=cancel)
        except UnsafeAnswer as exc:
            raise PipelineError(tr("pipeline.unsafe", detail=str(exc)), unsafe=True) from exc
        except NoModelsAvailable as exc:
            if exc.kind == ErrorKind.CANCELLED or cancel.is_set():
                raise PipelineCancelled from exc
            raise PipelineError(exc.user_message, kind=exc.kind) from exc
        if cancel.is_set():
            raise PipelineCancelled
        routes.append(route)
        requests += len(route.attempts)
        parsed: ParsedResponse = route.value
        if explicit_only:
            kept = [a for a in parsed.actions if not a.is_rule]
            if len(kept) != len(parsed.actions):
                warnings.append(tr("pipeline.ignored_rules", index=index))
            parsed.actions = kept
        actions.extend(parsed.actions)
        rejected.extend(parsed.rejected)
        warnings.extend(parsed.warnings)
        if parsed.summary:
            summaries.append(parsed.summary)
        if len(batches) == 1 or (index == 1 and (not parsed.actions or any(a.is_rule for a in parsed.actions))):
            break  # rules (or nothing to do) cover every file - no further requests needed
        explicit_only = True
    if progress is not None:
        progress(len(batches), len(batches), tr("pipeline.planning"))
    texts: list[str] = []
    for route in routes:
        if route.status_text not in texts:
            texts.append(route.status_text)
    used = list(dict.fromkeys(route.model.label for route in routes))
    fallbacks = sum(route.fallbacks for route in routes)
    merged = ParsedResponse(actions=actions, summary=summaries[0] if summaries else "", rejected=rejected, warnings=warnings)
    meta: dict[str, object] = {
        "source": "ai",
        "provider_text": " · ".join(texts),
        "provider_used": ", ".join(used),
        "ai_requests": requests,
        "fallbacks": fallbacks,
        "route_details": routes[-1].detail_lines() if routes else [],
    }
    return merged, meta


def _run_title_generator(
    request: PipelineRequest, router: ModelRouter, cancel: threading.Event, progress: ProgressFn | None
) -> PipelineResult:
    """Title Generator path: unique titles -> rename actions -> the usual planner, validation and preview."""
    from app.ai.title_generator import TitleGenerator

    if not request.files:
        raise PipelineError(tr("pipeline.no_files"))
    if request.processing == ProcessingMode.OFFLINE_ONLY:
        raise PipelineError(tr("title.offline_only"))
    generator = TitleGenerator(
        request.files, request.title or TitleConfig(enabled=True), router, command=request.command, search=request.search,
        include_metadata=request.settings.send_metadata_to_ai, batch_size=request.settings.batch_size, cancel=cancel,
        progress=progress,
    )  # fmt: skip
    outcome = generator.run()
    if progress is not None:
        progress(1, 1, tr("pipeline.planning"))
    parsed = outcome.parsed
    plan = Planner(request.workspace, request.files, planner_options(request.settings)).build(parsed.actions, summary=parsed.summary)
    for op in plan.ops:
        if op.source in outcome.infos:
            op.info = outcome.infos[op.source]
    log.info(
        "Title Generator done: model=%s requests=%s fallbacks=%s titles=%d",
        outcome.meta["provider_used"], outcome.meta["ai_requests"], outcome.meta["fallbacks"], len(parsed.actions),
    )  # fmt: skip
    return PipelineResult(
        plan=plan, summary=parsed.summary, warnings=list(parsed.warnings), task_type=TaskType.TITLE_GENERATION.value,
        files=len(request.files), **outcome.meta,
    )  # fmt: skip

