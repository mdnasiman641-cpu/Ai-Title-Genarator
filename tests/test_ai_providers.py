"""Provider adapters (Gemini, OpenAI, OpenAI-compatible, Custom) against a real local HTTP server.

No real API keys are used: every adapter is pointed at :class:`FakeAIServer` through its
configurable base URL.
"""

from __future__ import annotations

import json

import pytest

from app.ai.base_provider import AIRequest
from app.ai.errors import AIError, ErrorKind
from app.ai.gemini_provider import GeminiProvider as _Gemini
from app.ai.model_config import ModelConfig
from app.ai.openai_provider import CustomProvider, OpenAICompatibleProvider
from app.ai.openai_provider import OpenAIProvider as _OpenAI
from app.ai.providers import build_for_listing, build_provider
from app.workers.ai_worker import ConnectionTestWorker, LoadModelsWorker
from tests.fake_ai_server import FakeAIServer, gemini_ok, openai_ok

REQUEST = AIRequest(system_prompt="SYSTEM JSON", user_prompt="USER {…}")
SECRET = "AIzaSySECRETKEY1234567890abcdefghijklm"


BASE: dict[str, str] = {}


@pytest.fixture
def server():
    with FakeAIServer() as srv:
        BASE["url"] = srv.base
        yield srv


def GeminiProvider(key, model, timeout=5, base_url=None):
    return _Gemini(key, model, timeout, base_url or BASE["url"])


def OpenAIProvider(key, model, timeout=5, base_url=None):
    return _OpenAI(key, model, timeout, base_url or BASE["url"])


# ----------------------------------------------------------------- Gemini
def test_gemini_success_sends_key_in_header_not_url(server):
    server.queue(200, gemini_ok('{"actions": []}'))
    result = GeminiProvider(SECRET, "my-model", 5).generate(REQUEST)
    assert json.loads(result.text) == {"actions": []}
    assert result.provider == "gemini" and result.model == "my-model" and not result.truncated
    sent = server.requests[0]
    assert sent["path"] == "/models/my-model:generateContent"
    assert SECRET not in sent["path"]  # never in the URL
    assert sent["headers"]["x-goog-api-key"] == SECRET
    body = json.loads(sent["body"])
    assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM JSON"
    assert body["contents"][0]["parts"][0]["text"] == "USER {…}"
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_model_comes_from_config_only(server):
    server.queue(200, gemini_ok("{}"))
    GeminiProvider("k", "gemini-future-9", 5).generate(REQUEST)
    assert "gemini-future-9" in server.requests[0]["path"]


def test_gemini_skips_thought_parts_and_flags_truncation(server):
    server.queue(200, {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": '{"actions":'}]}, "finishReason": "MAX_TOKENS"}]})
    result = GeminiProvider("k", "m", 5).generate(REQUEST)
    assert result.text == '{"actions":' and result.truncated


def test_gemini_blocked_prompt(server):
    server.queue(200, {"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(AIError) as err:
        GeminiProvider("k", "m", 5).generate(REQUEST)
    assert err.value.kind == ErrorKind.BLOCKED


def test_gemini_invalid_key_is_auth_error(server):
    server.queue(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}})
    with pytest.raises(AIError) as err:
        GeminiProvider("bad", "m", 5).generate(REQUEST)
    assert err.value.kind == ErrorKind.AUTH
    assert "Gemini request failed." in err.value.user_message and "Authentication" in err.value.user_message


def test_gemini_list_models_filters_to_generate_content(server):
    server.queue(200, {"models": [
        {"name": "models/gemini-a", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
    ]})
    assert GeminiProvider("k", "gemini-a", 5).list_models() == ["gemini-a"]


# ----------------------------------------------------------------- OpenAI
def test_openai_success_and_request_shape(server):
    server.queue(200, openai_ok('{"actions": []}'))
    result = OpenAIProvider("sk-test-1234567890abcdef", "my-gpt", 5).generate(REQUEST)
    assert json.loads(result.text) == {"actions": []} and result.provider == "openai"
    sent = server.requests[0]
    assert sent["path"] == "/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer sk-test-1234567890abcdef"
    body = json.loads(sent["body"])
    assert body["model"] == "my-gpt"
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_openai_refusal_and_empty(server):
    server.queue(200, {"choices": [{"message": {"content": None, "refusal": "no"}, "finish_reason": "stop"}]})
    with pytest.raises(AIError) as err:
        OpenAIProvider("k", "m", 5).generate(REQUEST)
    assert err.value.kind == ErrorKind.BLOCKED
    server.queue(200, {"choices": []})
    with pytest.raises(AIError) as err2:
        OpenAIProvider("k", "m", 5).generate(REQUEST)
    assert err2.value.kind == ErrorKind.INVALID_RESPONSE


def test_openai_list_models_keeps_chat_models(server):
    server.queue(200, {"data": [{"id": "gpt-4o-mini"}, {"id": "text-embedding-3-small"}, {"id": "whisper-1"}, {"id": "o3-mini"}, {"id": "gpt-4o-audio-preview"}]})
    assert OpenAIProvider("k", "gpt-4o-mini", 5).list_models() == ["gpt-4o-mini", "o3-mini"]


# ------------------------------------------------------- failure handling
@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (401, {"error": {"message": "Incorrect API key provided"}}, ErrorKind.AUTH),
        (403, {"error": {"message": "forbidden"}}, ErrorKind.AUTH),
        (429, {"error": {"message": "Rate limit reached"}}, ErrorKind.RATE_LIMIT),
        (404, {"error": {"message": "The model `x` does not exist"}}, ErrorKind.MODEL_NOT_FOUND),
        (500, {"error": {"message": "boom"}}, ErrorKind.SERVER),
        (503, {"error": {"message": "overloaded"}}, ErrorKind.SERVER),
        (400, {"error": {"message": "bad param"}}, ErrorKind.BAD_REQUEST),
    ],
)
def test_http_errors_map_to_friendly_kinds(server, status, body, kind):
    server.queue(status, body)
    with pytest.raises(AIError) as err:
        OpenAIProvider("k", "m", 5).generate(REQUEST)
    assert err.value.kind == kind
    assert err.value.user_message  # always something to show the user


def test_timeout_is_reported(server):
    server.queue(200, openai_ok("{}"), delay=1.5)
    with pytest.raises(AIError) as err:
        OpenAIProvider("k", "m", timeout=0.3).generate(REQUEST)
    assert err.value.kind == ErrorKind.TIMEOUT


def test_network_unreachable():
    with pytest.raises(AIError) as err:
        OpenAIProvider("k", "m", 3, base_url="http://127.0.0.1:1").generate(REQUEST)  # nothing listens
    assert err.value.kind == ErrorKind.NETWORK


def test_non_json_body_is_invalid_response(server):
    server.queue(200, b"<html>captive portal</html>")
    with pytest.raises(AIError) as err:
        GeminiProvider("k", "m", 5).generate(REQUEST)
    assert err.value.kind == ErrorKind.INVALID_RESPONSE


def test_missing_key_is_rejected_up_front():
    with pytest.raises(AIError) as err:
        GeminiProvider("", "m")
    assert err.value.kind == ErrorKind.NO_KEY


def test_error_details_never_leak_the_key(server):
    server.queue(400, {"error": {"message": f"bad key {SECRET}"}})
    from app.utils.security import register_secret

    register_secret(SECRET)
    with pytest.raises(AIError) as err:
        GeminiProvider(SECRET, "m", 5).generate(REQUEST)
    assert SECRET not in str(err.value) and SECRET not in err.value.user_message


def test_test_connection_sends_a_minimal_request_and_times_it(server):
    server.queue(200, {"data": [{"id": "gpt-4o-mini"}]})
    server.queue(200, openai_ok('{"ok": true}'))
    ok = OpenAIProvider("k", "gpt-4o-mini", 5).test_connection()
    assert ok.ok and ok.models == ["gpt-4o-mini"] and ok.latency_s is not None
    assert "Connection successful" in ok.message and "gpt-4o-mini" in ok.message
    assert json.loads(server.requests[1]["body"])["model"] == "gpt-4o-mini"


def test_test_connection_failures(server):
    server.queue(401, {"error": {"message": "Incorrect API key"}})
    bad = OpenAIProvider("k", "m", 5).test_connection()
    assert not bad.ok and "Authentication" in bad.message
    server.queue(200, {"data": []})
    server.queue(200, openai_ok("Sure! Here you go."))  # answer is not JSON
    prose = OpenAIProvider("k", "m", 5).test_connection()
    assert not prose.ok and "JSON" in prose.message
    server.queue(200, {"data": []})
    server.queue(404, {"error": {"message": "The model `nope` does not exist"}})
    missing = OpenAIProvider("k", "nope", 5).test_connection()
    assert not missing.ok


# ------------------------------------------------- base URL configuration
def test_gemini_uses_configured_base_url(server):
    server.queue(200, gemini_ok("{}"))
    provider = _Gemini("k", "m", 5, server.base + "/v1beta/")  # trailing slash is normalised
    provider.generate(REQUEST)
    assert server.requests[0]["path"] == "/v1beta/models/m:generateContent"


def test_default_base_urls_are_official_endpoints():
    assert _Gemini("k", "m").base_url.startswith("https://generativelanguage.googleapis.com")
    assert _OpenAI("k", "m").base_url == "https://api.openai.com/v1"


def test_missing_base_url_for_compatible_provider_is_rejected():
    with pytest.raises(AIError) as err:
        OpenAICompatibleProvider(None, "m", 5, "")
    assert err.value.kind == ErrorKind.BAD_REQUEST


# ------------------------------------------------------ OpenAI-compatible
def test_openai_compatible_works_without_key(server):
    server.queue(200, openai_ok('{"actions": []}'))
    result = OpenAICompatibleProvider(None, "llama3.1:8b", 5, server.base + "/v1").generate(REQUEST)
    assert json.loads(result.text) == {"actions": []}
    sent = server.requests[0]
    assert sent["path"] == "/v1/chat/completions" and "authorization" not in sent["headers"]


def test_openai_compatible_sends_bearer_key_when_given(server):
    server.queue(200, openai_ok("{}"))
    OpenAICompatibleProvider("or-key-123", "m", 5, server.base).generate(REQUEST)
    assert server.requests[0]["headers"]["authorization"] == "Bearer or-key-123"


def test_openai_compatible_retries_without_json_mode(server):
    server.queue(400, {"error": {"message": "Unsupported parameter: response_format"}})
    server.queue(200, openai_ok('{"actions": []}'))
    result = OpenAICompatibleProvider(None, "m", 5, server.base).generate(REQUEST)
    assert json.loads(result.text) == {"actions": []}
    first, second = (json.loads(r["body"]) for r in server.requests)
    assert "response_format" in first and "response_format" not in second


def test_openai_compatible_content_parts_and_unfiltered_listing(server):
    server.queue(200, {"choices": [{"message": {"content": [{"type": "text", "text": '{"a":'}, {"type": "text", "text": "1}"}]}}]})
    assert OpenAICompatibleProvider(None, "m", 5, server.base).generate(REQUEST).text == '{"a":1}'
    server.queue(200, {"data": [{"id": "mistral-large"}, {"id": "llama-3.1-70b"}, {"id": "text-embedding-x"}]})
    assert OpenAICompatibleProvider(None, "m", 5, server.base).list_models() == ["llama-3.1-70b", "mistral-large", "text-embedding-x"]


def test_listing_404_means_unsupported(server):
    server.queue(404, {"error": {"message": "Not found"}})
    with pytest.raises(AIError) as err:
        OpenAICompatibleProvider(None, "m", 5, server.base).list_models()
    assert err.value.kind == ErrorKind.LISTING_UNSUPPORTED


def test_custom_provider_has_no_json_mode_and_no_listing(server):
    server.queue(200, openai_ok('{"actions": []}'))
    provider = CustomProvider(None, "my-model", 5, server.base)
    provider.generate(REQUEST)
    assert "response_format" not in json.loads(server.requests[0]["body"])
    with pytest.raises(AIError) as err:
        provider.list_models()
    assert err.value.kind == ErrorKind.LISTING_UNSUPPORTED


# ------------------------------------------------------- factory/workers
def test_build_provider_uses_the_configuration(server):
    config = ModelConfig.new("openai_compatible", "qwen2.5", base_url=server.base + "/v1", timeout_s=17)
    provider = build_provider(config, None)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "qwen2.5" and provider.timeout == 17 and provider.base_url == server.base + "/v1"
    with pytest.raises(AIError) as err:
        build_provider(ModelConfig.new("gemini", "m"), None)  # Gemini needs a key
    assert err.value.kind == ErrorKind.NO_KEY


def test_load_models_worker(server):
    server.queue(200, {"models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]})
    assert LoadModelsWorker("gemini", server.base, "k").execute() == ["gemini-2.5-flash"]
    assert server.requests[0]["headers"]["x-goog-api-key"] == "k"
    server.queue(404, {"error": {"message": "no"}})
    worker = LoadModelsWorker("openai_compatible", server.base, None)
    with pytest.raises(AIError) as err:
        worker.execute()
    assert worker.friendly_error(err.value) == "Model listing is not available for this provider. Enter the model ID manually."
    custom = LoadModelsWorker("custom", server.base, None)
    with pytest.raises(AIError) as err2:
        custom.execute()
    assert "Enter the model ID manually" in custom.friendly_error(err2.value)
    assert build_for_listing("openai", server.base, "k").base_url == server.base


def test_load_models_worker_rejects_bad_urls():
    worker = LoadModelsWorker("openai_compatible", "http://example.com/v1", None)  # plain http to a public host
    with pytest.raises(AIError) as err:
        worker.execute()
    assert "https" in worker.friendly_error(err.value).lower()


def test_connection_worker_end_to_end(server):
    server.queue(200, {"data": [{"id": "m"}]})
    server.queue(200, openai_ok('{"ok": true}'))
    config = ModelConfig.new("openai_compatible", "m", base_url=server.base)
    result = ConnectionTestWorker(config, None).execute()
    assert result.ok and result.latency_s is not None
    missing_key = ConnectionTestWorker(ModelConfig.new("openai", "gpt-4o-mini"), None).execute()
    assert not missing_key.ok


