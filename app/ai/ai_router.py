"""Model router: picks, calls, validates, retries and falls back across any number of models.

    request ─► task profile ─► registry (enabled + key) ─► capability filter
            ─► health (cooldown, rate limits) ─► strategy ordering
            ─► call model ─► validate answer ─► success
                              └─ failure ─► retry same model (transient errors)
                                         └─► next model (strategies with fallback)

There is no provider-specific code here: adapters come from :mod:`app.ai.providers`.
Attempts are bounded (``max_fallback_attempts`` models x ``max_retries`` each), so a
request can never loop forever.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from app.ai.base_provider import AIProvider, AIRequest, AIResponse
from app.ai.errors import RETRYABLE_SAME_MODEL, AIError, ErrorKind
from app.ai.model_config import ModelConfig
from app.ai.model_health import HealthTracker
from app.ai.model_registry import ModelRegistry
from app.ai.providers import build_provider
from app.ai.task_analyzer import TaskProfile
from app.config.constants import (
    DEFAULT_MAX_FALLBACK_ATTEMPTS,
    Capability,
    RoutingStrategy,
    TaskType,
)
from app.i18n import tr
from app.utils.logger import get_logger

log = get_logger("ai.router")

ProviderFactory = Callable[[ModelConfig], AIProvider]
Validator = Callable[[AIResponse], Any]  # raises InvalidAnswer / UnsafeAnswer
_RETRY_NOTE = (
    "\n\nYour previous reply could not be used ({reason}). Reply again with ONLY the JSON object "
    'described in the instructions: {{"actions": [...], "summary": "..."}}.'
)


class InvalidAnswer(Exception):
    """The model answered, but not with usable structured output (retry / fall back)."""


class UnsafeAnswer(Exception):
    """The answer tried something forbidden: stop immediately, never try other models."""


@dataclass(slots=True)
class Attempt:
    """One call to one model."""

    model: ModelConfig
    error: AIError | None = None
    latency_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class RouteResult:
    """A successful, validated answer and how it was obtained."""

    response: AIResponse
    value: Any  # whatever the validator returned (e.g. ParsedResponse)
    model: ModelConfig
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def fallbacks(self) -> int:
        """How many times we had to move on to a different model."""
        models = [a.model.id for a in self.attempts]
        return sum(1 for prev, cur in pairwise(models) if prev != cur)

    @property
    def status_text(self) -> str:
        """``Gemini / gemini-2.5-flash ✓`` or ``OpenAI / a (HTTP 429) → Gemini / b ✓``."""
        failed = []
        for attempt in self.attempts:
            if attempt.error is not None and attempt.model.id != self.model.id:
                entry = f"{attempt.model.label} ({attempt.error.category})"
                if entry not in failed:
                    failed.append(entry)
        if failed:
            return tr("ai.route.fallback", failed=" → ".join(failed), used=self.model.label)
        return tr("ai.route.ok", used=self.model.label)

    def detail_lines(self) -> list[str]:
        """Primary / Failed / Fallback / Success lines for the UI."""
        lines: list[str] = []
        first = self.attempts[0].model if self.attempts else self.model
        lines.append(tr("ai.route.primary", model=first.label))
        for attempt in self.attempts:
            if attempt.error is not None:
                line = tr("ai.route.failed", model=attempt.model.label, reason=attempt.error.category)
                if line not in lines:  # retries of the same failure are shown once
                    lines.append(line)
        if first.id != self.model.id:
            lines.append(tr("ai.route.fallback_to", model=self.model.label))
        lines.append(tr("ai.route.success"))
        return lines


class NoModelsAvailable(Exception):
    """No configured/usable model, or every attempt failed. ``user_message`` explains which."""

    def __init__(self, message: str, attempts: list[Attempt] | None = None, kind: ErrorKind = ErrorKind.NO_MODELS) -> None:
        super().__init__(message)
        self.user_message = message
        self.attempts = attempts or []
        self.kind = kind


@dataclass(slots=True)
class RouterOptions:
    strategy: RoutingStrategy = RoutingStrategy.AUTO_FALLBACK
    preferred_model_id: str | None = None  # Manual: the model; other modes: tried first
    cost_aware: bool = False
    max_fallback_attempts: int = DEFAULT_MAX_FALLBACK_ATTEMPTS


class ModelRouter:
    """Chooses among the configured models and performs requests with retry/fallback."""

    def __init__(
        self,
        registry: ModelRegistry,
        health: HealthTracker,
        options: RouterOptions | None = None,
        *,
        factory: ProviderFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.registry = registry
        self.health = health
        self.options = options or RouterOptions()
        self._factory = factory or (lambda cfg: build_provider(cfg, registry.api_key(cfg.id)))
        self._sleep = sleep

    # ============================================================ selection
    def candidates(self, profile: TaskProfile | None = None) -> list[ModelConfig]:
        """Models to try, best first (Manual/Auto use only the first)."""
        opts = self.options
        usable = [m for m in self.registry.list() if self.registry.is_usable(m)]
        if not usable:
            raise NoModelsAvailable(tr("ai.router.no_models"))
        if opts.strategy == RoutingStrategy.MANUAL:
            chosen = next((m for m in usable if m.id == opts.preferred_model_id), None)
            if chosen is None and opts.preferred_model_id:
                missing = self.registry.get(opts.preferred_model_id)
                name = missing.label if missing else "?"
                raise NoModelsAvailable(tr("ai.router.manual_unavailable", model=name))
            return [chosen or usable[0]]

        required = profile.required if profile else frozenset()
        capable = [m for m in usable if required <= set(m.capabilities)] or usable  # relax rather than fail
        preferred = profile.preferred if profile else frozenset()

        def key(model: ModelConfig) -> tuple[Any, ...]:
            # Health is handled by the cooldown partition below: a model that failed is skipped
            # while cooling down and simply gets its normal place back afterwards.
            matches = -len(preferred & set(model.capabilities))
            cost = model.known_cost if model.known_cost is not None else float("inf")
            latency = self.health.stats(model.id).average_latency_s
            if opts.strategy == RoutingStrategy.HIGHEST_PRIORITY:
                return (model.priority,)
            if opts.strategy == RoutingStrategy.CHEAPEST:
                return (cost, matches, model.priority)
            if opts.strategy == RoutingStrategy.FASTEST:
                fast = 0 if model.has(Capability.FAST) else 1
                return (0 if latency is not None else 1, latency or 0.0, fast, model.priority)
            if opts.cost_aware:  # correctness (capability match) first, then price, then priority
                return (matches, cost, model.priority)
            return (matches, model.priority)

        ranked = sorted(capable, key=key)
        if opts.preferred_model_id:
            ranked.sort(key=lambda m: 0 if m.id == opts.preferred_model_id else 1)  # stable: keeps the rest
        ready = [m for m in ranked if not self.health.in_cooldown(m.id) and self.health.within_limits(m.id, m.rpm_limit, m.rpd_limit)]
        waiting = [m for m in ranked if m not in ready]  # cooling down / over limit: last resort only
        return ready + sorted(waiting, key=lambda m: self.health.cooldown_remaining(m.id))

    def preview(self, profile: TaskProfile | None = None) -> ModelConfig | None:
        """The model a request would start with (for the "AI: …" label); ``None`` if there is none."""
        try:
            return self.candidates(profile)[0]
        except NoModelsAvailable:
            return None

    # ============================================================== request
    def generate(
        self,
        request: AIRequest,
        *,
        profile: TaskProfile | None = None,
        validate: Validator | None = None,
        cancel: threading.Event | None = None,
    ) -> RouteResult:
        """Send ``request``; validate the answer; retry/fall back according to the strategy."""
        cancel = cancel or threading.Event()
        order = self.candidates(profile)
        if not self.options.strategy.falls_back:
            order = order[:1]
        order = order[: max(1, self.options.max_fallback_attempts)]
        attempts: list[Attempt] = []
        task = profile.task_type.value if profile else TaskType.GENERAL.value
        for index, model in enumerate(order):
            current = request
            for try_number in range(1 + max(0, model.max_retries)):
                if cancel.is_set():
                    raise NoModelsAvailable(tr("ai.err.cancelled"), attempts, ErrorKind.CANCELLED)
                start = time.perf_counter()
                response: AIResponse | None = None
                try:
                    provider = self._factory(model)
                    response = provider.generate(current)
                    value = validate(response) if validate else response
                except UnsafeAnswer:
                    self.health.record_failure(model.id, "unsafe_answer")
                    log.warning("Unsafe answer from %s - stopping (no fallback)", model.label)
                    raise
                except InvalidAnswer as exc:
                    elapsed = time.perf_counter() - start
                    truncated = response is not None and response.truncated
                    error = AIError(ErrorKind.INVALID_RESPONSE, model.label, ("truncated: " if truncated else "") + str(exc))
                    attempts.append(Attempt(model, error, elapsed))
                    self.health.record_failure(model.id, error.kind.value, str(exc)[:120])
                    log.warning("%s gave an unusable answer (%s) [task=%s]", model.label, exc, task)
                    current = AIRequest(request.system_prompt, request.user_prompt + _RETRY_NOTE.format(reason=exc))
                    continue
                except AIError as exc:
                    elapsed = time.perf_counter() - start
                    attempts.append(Attempt(model, exc, elapsed))
                    self.health.record_failure(model.id, exc.kind.value, exc.category)
                    log.warning("%s failed: %s (%.2fs) [task=%s]", model.label, exc.category, elapsed, task)
                    if exc.kind == ErrorKind.CANCELLED:
                        raise NoModelsAvailable(exc.user_message, attempts, ErrorKind.CANCELLED) from None
                    if exc.kind in RETRYABLE_SAME_MODEL and try_number < model.max_retries:
                        self._sleep(min(2.0, 0.5 * (try_number + 1)))
                        continue
                    break  # 429 / auth / not found / out of retries -> next model
                assert response is not None
                elapsed = time.perf_counter() - start
                attempts.append(Attempt(model, None, elapsed))
                self.health.record_success(model.id, elapsed, response.usage)
                result = RouteResult(response, value, model, attempts)
                log.info(
                    "AI request ok: %s in %.2fs (task=%s, attempts=%d, fallbacks=%d)",
                    model.label, elapsed, task, len(attempts), result.fallbacks,
                )  # fmt: skip
                return result
            if index + 1 < len(order):
                log.info("Falling back from %s to %s", model.label, order[index + 1].label)
        raise NoModelsAvailable(self._failure_message(attempts), attempts, attempts[-1].error.kind if attempts and attempts[-1].error else ErrorKind.NO_MODELS)

    @staticmethod
    def _failure_message(attempts: list[Attempt]) -> str:
        lines: list[str] = []
        for attempt in attempts:
            if attempt.error is not None:
                line = f"{attempt.model.label}: {attempt.error.reason}"
                if line not in lines:
                    lines.append(line)
        if not lines:
            return tr("ai.router.no_models")
        return tr("ai.router.all_failed") + "\n" + "\n".join(f"• {line}" for line in lines)


def make_router(registry: ModelRegistry, health: HealthTracker, settings: Any, *, strategy: RoutingStrategy | None = None,
                preferred_model_id: str | None = None) -> ModelRouter:  # fmt: skip
    """Router configured from saved settings (``strategy``/``preferred`` override them, e.g. from the command bar)."""
    health.configure(
        cooldown_s=settings.cooldown_seconds,
        cooldown_after_failures=settings.cooldown_after_failures,
        enabled=settings.health_tracking,
    )
    options = RouterOptions(
        strategy=strategy or RoutingStrategy(settings.routing_strategy),
        preferred_model_id=preferred_model_id if preferred_model_id is not None else (settings.preferred_model_id or None),
        cost_aware=settings.cost_aware,
        max_fallback_attempts=settings.max_fallback_attempts,
    )
    return ModelRouter(registry, health, options)
