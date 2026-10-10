"""Per-model health (cooldowns) and performance statistics, persisted locally.

The router reports every call here. A model that fails repeatedly - or answers
"rate limit / quota" - is put in a temporary *cooldown* and skipped; when the
cooldown expires it is simply tried again by the next request. Nothing sensitive
is stored: only counters, timings and an error *category*.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime

from app.ai.errors import ErrorKind
from app.config.constants import DEFAULT_COOLDOWN_AFTER_FAILURES, DEFAULT_COOLDOWN_S
from app.database.database import Database
from app.utils.logger import get_logger

log = get_logger("ai.health")
_IMMEDIATE_COOLDOWN = {ErrorKind.RATE_LIMIT.value}  # quota / 429: back off at once
_MAX_COOLDOWN_FACTOR = 8


@dataclass(slots=True)
class ModelStats:
    """Everything tracked for one model."""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    total_latency_s: float = 0.0
    last_latency_s: float | None = None
    last_used: float | None = None
    last_success: float | None = None
    last_failure: float | None = None
    last_error: str = ""  # error category + short reason, never secrets
    cooldown_until: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    minute_window: list[float] = field(default_factory=list)  # request times in the last 60 s
    day: str = ""
    day_count: int = 0

    @property
    def success_rate(self) -> float | None:
        done = self.successes + self.failures
        return self.successes / done if done else None

    @property
    def average_latency_s(self) -> float | None:
        return self.total_latency_s / self.successes if self.successes else None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelStats:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)  # type: ignore[arg-type]


class HealthTracker:
    """Thread-safe health book-keeping shared by the router and the Settings UI."""

    def __init__(
        self,
        db: Database | None = None,
        *,
        cooldown_s: int = DEFAULT_COOLDOWN_S,
        cooldown_after_failures: int = DEFAULT_COOLDOWN_AFTER_FAILURES,
        enabled: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db = db
        self._lock = threading.Lock()
        self.cooldown_s = cooldown_s
        self.cooldown_after_failures = max(1, cooldown_after_failures)
        self.enabled = enabled
        self._clock = clock
        self._stats: dict[str, ModelStats] = {}
        if db is not None:
            for model_id, data in db.get_health_rows().items():
                try:
                    self._stats[model_id] = ModelStats.from_dict(data)
                except TypeError:
                    continue

    def configure(self, *, cooldown_s: int, cooldown_after_failures: int, enabled: bool) -> None:
        self.cooldown_s = cooldown_s
        self.cooldown_after_failures = max(1, cooldown_after_failures)
        self.enabled = enabled

    # -------------------------------------------------------------- queries
    def stats(self, model_id: str) -> ModelStats:
        with self._lock:
            return ModelStats.from_dict(self._stats.get(model_id, ModelStats()).to_dict())

    def cooldown_remaining(self, model_id: str) -> float:
        if not self.enabled:
            return 0.0
        with self._lock:
            stats = self._stats.get(model_id)
            return max(0.0, stats.cooldown_until - self._clock()) if stats else 0.0

    def in_cooldown(self, model_id: str) -> bool:
        return self.cooldown_remaining(model_id) > 0

    def within_limits(self, model_id: str, rpm: int | None, rpd: int | None) -> bool:
        """False if the user-configured requests-per-minute / per-day limit is reached."""
        if rpm is None and rpd is None:
            return True
        now = self._clock()
        with self._lock:
            stats = self._stats.get(model_id)
            if stats is None:
                return True
            recent = [t for t in stats.minute_window if now - t < 60]
            if rpm is not None and len(recent) >= rpm:
                return False
            today = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
            return not (rpd is not None and stats.day == today and stats.day_count >= rpd)

    # ------------------------------------------------------------- updates
    def _bump_request(self, stats: ModelStats, now: float) -> None:
        stats.requests += 1
        stats.last_used = now
        stats.minute_window = [t for t in stats.minute_window if now - t < 60] + [now]
        today = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if stats.day != today:
            stats.day, stats.day_count = today, 0
        stats.day_count += 1

    def record_success(self, model_id: str, latency_s: float, usage: dict[str, int] | None = None) -> None:
        now = self._clock()
        with self._lock:
            stats = self._stats.setdefault(model_id, ModelStats())
            self._bump_request(stats, now)
            stats.successes += 1
            stats.consecutive_failures = 0
            stats.cooldown_until = 0.0
            stats.last_success = now
            stats.last_latency_s = latency_s
            stats.total_latency_s += latency_s
            if usage:
                stats.input_tokens += int(usage.get("input_tokens", 0) or 0)
                stats.output_tokens += int(usage.get("output_tokens", 0) or 0)
            self._save(model_id, stats)

    def record_failure(self, model_id: str, kind: str, message: str = "") -> None:
        now = self._clock()
        with self._lock:
            stats = self._stats.setdefault(model_id, ModelStats())
            self._bump_request(stats, now)
            stats.failures += 1
            stats.consecutive_failures += 1
            stats.last_failure = now
            stats.last_error = f"{kind}: {message}"[:200] if message else kind
            if self.enabled and (kind in _IMMEDIATE_COOLDOWN or stats.consecutive_failures >= self.cooldown_after_failures):
                steps = max(0, stats.consecutive_failures - self.cooldown_after_failures)
                factor = min(_MAX_COOLDOWN_FACTOR, 2**steps)  # back off longer if it keeps failing
                stats.cooldown_until = now + self.cooldown_s * factor
                log.info("Model %s in cooldown for %ds after %s", model_id[:8], int(self.cooldown_s * factor), kind)
            self._save(model_id, stats)

    def reset(self, model_id: str | None = None) -> None:
        """Forget statistics (one model, or all)."""
        with self._lock:
            if model_id is None:
                self._stats.clear()
            else:
                self._stats.pop(model_id, None)
            if self._db is not None:
                self._db.delete_health_row(model_id)

    def clear_cooldown(self, model_id: str) -> None:
        with self._lock:
            stats = self._stats.get(model_id)
            if stats is not None:
                stats.cooldown_until = 0.0
                stats.consecutive_failures = 0
                self._save(model_id, stats)

    def _save(self, model_id: str, stats: ModelStats) -> None:
        if self._db is not None:
            try:
                self._db.put_health_row(model_id, stats.to_dict())
            except Exception:  # noqa: BLE001 - statistics must never break a request
                log.debug("could not persist model health", exc_info=True)

    # ---------------------------------------------------------- UI helpers
    def summary(self, model_id: str) -> str:
        """``✓ 98% success · avg 1.8s · used 43×`` (or ``Not used yet``)."""
        stats = self.stats(model_id)
        if stats.requests == 0:
            return "Not used yet"
        parts = []
        rate = stats.success_rate
        if rate is not None:
            parts.append(f"✓ {rate * 100:.0f}% success")
        if stats.average_latency_s is not None:
            parts.append(f"avg {stats.average_latency_s:.1f}s")
        parts.append(f"used {stats.requests}×")
        if stats.input_tokens or stats.output_tokens:
            parts.append(f"~{stats.input_tokens + stats.output_tokens:,} tokens")
        return " · ".join(parts)
