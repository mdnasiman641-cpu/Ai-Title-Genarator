"""Model router: strategies, priority, capability filter, retry/fallback, health, cooldown and limits.

Models are real registry entries; the provider *factory* is replaced by scripted fakes so
every scenario (429, 5xx, timeout, invalid JSON, unsafe output ...) is deterministic.
"""

from __future__ import annotations

import pytest

from app.ai.ai_router import (
    InvalidAnswer,
    ModelRouter,
    NoModelsAvailable,
    RouterOptions,
    UnsafeAnswer,
    make_router,
)
from app.ai.base_provider import AIRequest, AIResponse
from app.ai.errors import AIError, ErrorKind
from app.ai.model_config import ModelConfig
from app.ai.model_health import HealthTracker
from app.ai.task_analyzer import analyze_task
from app.config.constants import RoutingStrategy, TaskType
from app.context import AppContext

REQUEST = AIRequest("SYSTEM", "USER")


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class Script:
    """Per-model queue of outcomes: an ``ErrorKind`` (raised), or answer text."""

    def __init__(self) -> None:
        self.plans: dict[str, list[object]] = {}
        self.calls: list[str] = []
        self.prompts: list[str] = []

    def factory(self, config: ModelConfig):
        script = self

        class Fake:
            def generate(self, request: AIRequest) -> AIResponse:
                script.calls.append(config.model_name)
                script.prompts.append(request.user_prompt)
                queue = script.plans.get(config.model_name) or ['{"ok": true}']
                outcome = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(outcome, ErrorKind):
                    status = {ErrorKind.RATE_LIMIT: 429, ErrorKind.SERVER: 503}.get(outcome)
                    raise AIError(outcome, config.label, "simulated", status)
                return AIResponse(str(outcome), config.provider_type, config.model_name, usage={"input_tokens": 7, "output_tokens": 3})

        return Fake()


def validate(response: AIResponse) -> str:
    if response.text == "UNSAFE":
        raise UnsafeAnswer("shell command")
    if not response.text.startswith("{"):
        raise InvalidAnswer("not JSON")
    return response.text


@pytest.fixture
def ctx(tmp_path):
    return AppContext.create(tmp_path / "data")


@pytest.fixture
def clock():
    return Clock()


def add(ctx, name: str, provider: str = "openai_compatible", **kw) -> ModelConfig:
    kw.setdefault("max_retries", 0)
    base = "http://localhost:9" if provider in ("openai_compatible", "custom") else ""
    return ctx.registry.add(ModelConfig.new(provider, name, base_url=base, **kw))


def router(ctx, clock, script: Script, **opts) -> ModelRouter:
    health = HealthTracker(ctx.db, cooldown_s=60, cooldown_after_failures=2, clock=clock)
    return ModelRouter(ctx.registry, health, RouterOptions(**opts), factory=script.factory, sleep=lambda _s: None)


# ------------------------------------------------------------------ selection
def test_no_models_gives_actionable_message(ctx, clock):
    with pytest.raises(NoModelsAvailable) as err:
        router(ctx, clock, Script()).generate(REQUEST)
    assert "Settings" in err.value.user_message and err.value.kind == ErrorKind.NO_MODELS


def test_priority_order_and_preferred_model(ctx, clock):
    a, b = add(ctx, "a"), add(ctx, "b")
    script = Script()
    result = router(ctx, clock, script).generate(REQUEST, validate=validate)
    assert result.model.id == a.id and script.calls == ["a"]
    result = router(ctx, clock, script, preferred_model_id=b.id).generate(REQUEST, validate=validate)
    assert result.model.id == b.id


def test_models_without_required_key_or_disabled_are_skipped(ctx, clock):
    gem = ctx.registry.add(ModelConfig.new("gemini", "g"))  # no key -> unusable
    off = add(ctx, "off")
    ctx.registry.set_enabled(off.id, False)
    ok = add(ctx, "ok")
    candidates = router(ctx, clock, Script()).candidates()
    assert [m.id for m in candidates] == [ok.id]
    ctx.registry.set_key(gem.id, "AIzaSy-test-key-0000")
    assert router(ctx, clock, Script()).candidates()[0].id == gem.id


def test_manual_uses_only_the_chosen_model(ctx, clock):
    add(ctx, "a")
    b = add(ctx, "b")
    script = Script()
    script.plans["b"] = [ErrorKind.SERVER]
    with pytest.raises(NoModelsAvailable):
        router(ctx, clock, script, strategy=RoutingStrategy.MANUAL, preferred_model_id=b.id).generate(REQUEST)
    assert script.calls == ["b"]  # never falls back in Manual mode


def test_manual_with_unusable_model_says_so(ctx, clock):
    gem = ctx.registry.add(ModelConfig.new("gemini", "gemini-x"))
    add(ctx, "a")
    with pytest.raises(NoModelsAvailable) as err:
        router(ctx, clock, Script(), strategy=RoutingStrategy.MANUAL, preferred_model_id=gem.id).generate(REQUEST)
    assert "gemini-x" in err.value.user_message


def test_capability_filter_and_preference(ctx, clock):
    plain = add(ctx, "plain", capabilities=("text",))  # no JSON capability
    json_model = add(ctx, "json", capabilities=("text", "json"))
    big = add(ctx, "big", capabilities=("text", "json", "long_context"))
    profile = analyze_task("Rename every file with a clean title", 500)
    assert profile.task_type == TaskType.LARGE_BATCH
    order = [m.id for m in router(ctx, clock, Script()).candidates(profile)]
    assert order[0] == big.id and plain.id not in order and json_model.id in order


def test_capability_filter_is_relaxed_rather_than_failing(ctx, clock):
    only = add(ctx, "only", capabilities=("text",))
    profile = analyze_task("Rename every file with a clean title", 5)
    assert router(ctx, clock, Script()).candidates(profile)[0].id == only.id


def test_cheapest_uses_only_user_supplied_prices(ctx, clock):
    unknown = add(ctx, "unknown")
    pricey = add(ctx, "pricey", cost_input_per_mtok=5.0, cost_output_per_mtok=15.0)
    cheap = add(ctx, "cheap", cost_input_per_mtok=0.1, cost_output_per_mtok=0.4)
    order = [m.id for m in router(ctx, clock, Script(), strategy=RoutingStrategy.CHEAPEST).candidates()]
    assert order == [cheap.id, pricey.id, unknown.id]  # unknown price is never treated as free


def test_cost_aware_keeps_capability_first(ctx, clock):
    cheap_small = add(ctx, "cheap", capabilities=("text", "json"), cost_input_per_mtok=0.1)
    pricey_big = add(ctx, "big", capabilities=("text", "json", "long_context"), cost_input_per_mtok=3.0)
    profile = analyze_task("Rename every file with a clean title", 500)
    order = [m.id for m in router(ctx, clock, Script(), cost_aware=True).candidates(profile)]
    assert order[0] == pricey_big.id  # correctness (long context) before price
    small = analyze_task("Rename every file with a clean title", 5)
    order = [m.id for m in router(ctx, clock, Script(), cost_aware=True).candidates(small)]
    assert order[0] == cheap_small.id


def test_fastest_prefers_measured_latency(ctx, clock):
    slow, fast = add(ctx, "slow"), add(ctx, "fast")
    r = router(ctx, clock, Script(), strategy=RoutingStrategy.FASTEST)
    r.health.record_success(slow.id, 4.0)
    r.health.record_success(fast.id, 0.5)
    assert r.candidates()[0].id == fast.id


def test_highest_priority_is_strict(ctx, clock):
    first = add(ctx, "first")
    add(ctx, "second", cost_input_per_mtok=0.01, capabilities=("text", "json", "fast"))
    assert router(ctx, clock, Script(), strategy=RoutingStrategy.HIGHEST_PRIORITY).candidates()[0].id == first.id


# ------------------------------------------------------------------- fallback
@pytest.mark.parametrize("kind", [ErrorKind.RATE_LIMIT, ErrorKind.SERVER, ErrorKind.TIMEOUT, ErrorKind.NETWORK, ErrorKind.AUTH])
def test_auto_fallback_moves_to_next_model(ctx, clock, kind):
    a, b = add(ctx, "a"), add(ctx, "b")
    script = Script()
    script.plans["a"] = [kind]
    result = router(ctx, clock, script).generate(REQUEST, validate=validate)
    assert result.model.id == b.id and result.fallbacks == 1
    assert [x.model.id for x in result.attempts] == [a.id, b.id]
    lines = result.detail_lines()
    assert lines[0] == f"Primary: {a.label}" and lines[-1] == "Success ✓"
    assert any(line.startswith("Failed:") for line in lines) and f"Fallback: {b.label}" in lines


def test_status_text_names_the_http_status(ctx, clock):
    add(ctx, "a")
    add(ctx, "b")
    script = Script()
    script.plans["a"] = [ErrorKind.RATE_LIMIT]
    result = router(ctx, clock, script).generate(REQUEST, validate=validate)
    assert result.status_text == "OpenAI-compatible / a (HTTP 429) → OpenAI-compatible / b ✓"


def test_auto_without_fallback_stops_after_first_model(ctx, clock):
    add(ctx, "a")
    add(ctx, "b")
    script = Script()
    script.plans["a"] = [ErrorKind.SERVER]
    with pytest.raises(NoModelsAvailable):
        router(ctx, clock, script, strategy=RoutingStrategy.AUTO).generate(REQUEST)
    assert script.calls == ["a"]


def test_transient_errors_retry_same_model_within_limit(ctx, clock):
    add(ctx, "a", max_retries=2)
    script = Script()
    script.plans["a"] = [ErrorKind.TIMEOUT, ErrorKind.SERVER, '{"done": 1}']
    result = router(ctx, clock, script).generate(REQUEST, validate=validate)
    assert script.calls == ["a", "a", "a"] and result.value == '{"done": 1}' and result.fallbacks == 0


def test_rate_limit_is_not_retried_on_same_model(ctx, clock):
    add(ctx, "a", max_retries=3)
    add(ctx, "b")
    script = Script()
    script.plans["a"] = [ErrorKind.RATE_LIMIT]
    router(ctx, clock, script).generate(REQUEST, validate=validate)
    assert script.calls == ["a", "b"]


def test_invalid_json_retries_with_correction_then_falls_back(ctx, clock):
    add(ctx, "a", max_retries=1)
    add(ctx, "b")
    script = Script()
    script.plans["a"] = ["Sure! Here is the plan."]  # always prose
    result = router(ctx, clock, script).generate(REQUEST, validate=validate)
    assert script.calls == ["a", "a", "b"] and result.model.model_name == "b"
    assert "not JSON" in script.prompts[1] and script.prompts[0] == "USER"  # correction appended on retry


def test_unsafe_answer_stops_immediately(ctx, clock):
    add(ctx, "a")
    add(ctx, "b")
    script = Script()
    script.plans["a"] = ["UNSAFE"]
    with pytest.raises(UnsafeAnswer):
        router(ctx, clock, script).generate(REQUEST, validate=validate)
    assert script.calls == ["a"]  # never "shops around" for a model that agrees


def test_attempts_are_bounded(ctx, clock):
    for i in range(8):
        add(ctx, f"m{i}", max_retries=5)
    script = Script()
    for i in range(8):
        script.plans[f"m{i}"] = [ErrorKind.SERVER]
    with pytest.raises(NoModelsAvailable) as err:
        router(ctx, clock, script, max_fallback_attempts=3).generate(REQUEST)
    assert len(set(script.calls)) == 3 and len(script.calls) == 3 * 6  # 3 models x (1 + 5 retries), then stop
    assert "Every suitable AI model failed" in err.value.user_message


def test_cancel_stops_before_next_attempt(ctx, clock):
    import threading

    add(ctx, "a")
    event = threading.Event()
    event.set()
    with pytest.raises(NoModelsAvailable) as err:
        router(ctx, clock, Script()).generate(REQUEST, cancel=event)
    assert err.value.kind == ErrorKind.CANCELLED


# --------------------------------------------------------------- health/cooldown
def test_rate_limit_puts_model_in_cooldown_and_it_recovers(ctx, clock):
    a, b = add(ctx, "a"), add(ctx, "b")
    script = Script()
    script.plans["a"] = [ErrorKind.RATE_LIMIT, '{"ok": 1}']
    r = router(ctx, clock, script)
    r.generate(REQUEST, validate=validate)
    assert r.health.in_cooldown(a.id) and [m.id for m in r.candidates()] == [b.id, a.id]
    script.calls.clear()
    r.generate(REQUEST, validate=validate)
    assert script.calls == ["b"]  # model in cooldown is skipped
    clock.now += 61
    assert not r.health.in_cooldown(a.id)
    script.calls.clear()
    result = r.generate(REQUEST, validate=validate)
    assert script.calls == ["a"] and result.model.id == a.id


def test_cooldown_after_consecutive_failures_with_backoff(ctx, clock):
    health = HealthTracker(None, cooldown_s=10, cooldown_after_failures=2, clock=clock)
    health.record_failure("m", "server")
    assert not health.in_cooldown("m")
    health.record_failure("m", "server")
    assert 9 < health.cooldown_remaining("m") <= 10
    health.record_failure("m", "server")
    assert 19 < health.cooldown_remaining("m") <= 20  # exponential backoff
    health.record_success("m", 0.5)
    assert not health.in_cooldown("m") and health.stats("m").consecutive_failures == 0


def test_health_disabled_never_cools_down(clock):
    health = HealthTracker(None, enabled=False, clock=clock)
    health.record_failure("m", "rate_limit")
    assert not health.in_cooldown("m")


def test_statistics_and_summary(ctx, clock):
    a = add(ctx, "a")
    r = router(ctx, clock, Script())
    for _ in range(3):
        r.generate(REQUEST, validate=validate)
    r.health.record_failure(a.id, "timeout", "timeout")
    stats = r.health.stats(a.id)
    assert (stats.requests, stats.successes, stats.failures) == (4, 3, 1)
    assert stats.input_tokens == 21 and stats.output_tokens == 9 and stats.last_used and stats.last_failure
    assert r.health.summary(a.id).startswith("✓ 75% success") and "used 4×" in r.health.summary(a.id)
    assert "timeout" in stats.last_error


def test_health_persists_across_restarts(ctx, clock, tmp_path):
    a = add(ctx, "a")
    HealthTracker(ctx.db, clock=clock).record_success(a.id, 1.5)
    again = AppContext.create(tmp_path / "data")
    assert again.health.stats(a.id).successes == 1 and again.health.stats(a.id).average_latency_s == 1.5


def test_user_rate_limits_are_respected(ctx, clock):
    limited = add(ctx, "limited", rpm_limit=2)
    other = add(ctx, "other")
    r = router(ctx, clock, Script())
    r.generate(REQUEST)
    r.generate(REQUEST)
    assert [m.id for m in r.candidates()] == [other.id, limited.id]  # over its per-minute limit
    clock.now += 61
    assert r.candidates()[0].id == limited.id
    day = add(ctx, "day", rpd_limit=1)
    ctx.registry.move(day.id, -5)
    r.generate(REQUEST)
    assert r.candidates()[-1].id == day.id


def test_make_router_reads_settings(ctx):
    a = add(ctx, "a")
    ctx.settings.update(routing_strategy="cheapest", cost_aware=True, max_fallback_attempts=2, cooldown_seconds=30)
    r = make_router(ctx.registry, ctx.health, ctx.settings.load())
    assert r.options.strategy == RoutingStrategy.CHEAPEST and r.options.cost_aware and r.options.max_fallback_attempts == 2
    assert ctx.health.cooldown_s == 30
    r2 = make_router(ctx.registry, ctx.health, ctx.settings.load(), strategy=RoutingStrategy.MANUAL, preferred_model_id=a.id)
    assert r2.options.strategy == RoutingStrategy.MANUAL and r2.preview().id == a.id


# ------------------------------------------------------------------ task types
@pytest.mark.parametrize(
    ("command", "files", "task"),
    [
        ("Remove 1080p from all filenames", 10, TaskType.SIMPLE_DETERMINISTIC),
        ("Rename these with clean episode names", 10, TaskType.COMPLEX_RENAME),
        ("Create YouTube titles for these videos", 10, TaskType.TITLE_GENERATION),
        ("Write a description for each video", 10, TaskType.DESCRIPTION_GENERATION),
        ("Season 2-এর সব ভিডিও আলাদা folder-এ রাখো", 10, TaskType.FILE_ORGANIZATION),
        ("সব ভিডিওর নাম clean করে দাও", 10, TaskType.COMPLEX_RENAME),
        ("Rename these with clean episode titles", 500, TaskType.LARGE_BATCH),
        ("Do something clever", 3, TaskType.GENERAL),
    ],
)
def test_task_analyzer(command, files, task):
    assert analyze_task(command, files).task_type == task
