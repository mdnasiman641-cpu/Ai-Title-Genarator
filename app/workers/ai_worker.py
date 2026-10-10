"""Workers that talk to AI endpoints (never on the UI thread)."""

from __future__ import annotations

from app.ai.ai_router import ModelRouter
from app.ai.base_provider import ConnectionResult
from app.ai.errors import AIError, ErrorKind
from app.ai.model_config import ModelConfig, validate_base_url
from app.ai.pipeline import (
    PipelineCancelled,
    PipelineError,
    PipelineRequest,
    PipelineResult,
    run_pipeline,
)
from app.ai.providers import build_for_listing, build_provider
from app.i18n import tr
from app.workers.base_worker import BaseWorker, CancelledError


class ConnectionTestWorker(BaseWorker):
    """*Test Connection*: validates URL + key, sends one tiny request and times it."""

    def __init__(self, config: ModelConfig, api_key: str | None) -> None:
        super().__init__()
        self.config = config
        self.api_key = api_key

    def execute(self) -> ConnectionResult:
        problem = validate_base_url(self.config.base_url)
        if problem:
            return ConnectionResult(False, problem)
        try:
            provider = build_provider(self.config, self.api_key)
        except AIError as exc:
            return ConnectionResult(False, exc.user_message)
        return provider.test_connection()


class LoadModelsWorker(BaseWorker):
    """*Load Models*: asks the endpoint for its model list.

    Result: ``list[str]``. Providers without a list endpoint produce a friendly
    "enter the model ID manually" error instead of a failure.
    """

    def __init__(self, provider_type: str, base_url: str, api_key: str | None) -> None:
        super().__init__()
        self.provider_type = provider_type
        self.base_url = base_url
        self.api_key = api_key

    def execute(self) -> list[str]:
        problem = validate_base_url(self.base_url)
        if problem:
            raise AIError(ErrorKind.BAD_REQUEST, "", problem)
        provider = build_for_listing(self.provider_type, self.base_url, self.api_key)
        models = provider.list_models()
        if not models:
            raise AIError(ErrorKind.LISTING_UNSUPPORTED, provider.error_name, "The server returned an empty list.")
        return models

    def friendly_error(self, exc: BaseException) -> str:
        if isinstance(exc, AIError):
            if exc.kind == ErrorKind.LISTING_UNSUPPORTED:
                return tr("models.listing_unsupported")
            if exc.kind == ErrorKind.BAD_REQUEST and not exc.provider:
                return exc.detail
            return exc.user_message
        return super().friendly_error(exc)


class PlanWorker(BaseWorker):
    """Turns a command into a plan (offline parse or AI) without blocking the UI."""

    def __init__(self, request: PipelineRequest, router: ModelRouter) -> None:
        super().__init__()
        self.request = request
        self.router = router

    def execute(self) -> PipelineResult:
        try:
            return run_pipeline(self.request, self.router, cancel=self.cancel_event, progress=self.emit_progress)
        except PipelineCancelled:
            raise CancelledError from None

    def friendly_error(self, exc: BaseException) -> str:
        if isinstance(exc, PipelineError):
            return exc.message
        return super().friendly_error(exc)
