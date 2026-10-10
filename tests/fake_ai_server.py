"""A tiny local HTTP server that impersonates the Gemini / OpenAI APIs in tests."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeAIServer:
    """Serves queued responses and records every request it receives."""

    def __init__(self) -> None:
        self.responses: list[tuple[int, object, float]] = []  # (status, body, delay_seconds)
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode("utf-8") if length else ""
                outer.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},  # HTTP names are case-insensitive
                        "body": body,
                    }
                )
                status, payload, delay = outer.responses.pop(0) if outer.responses else (500, {"error": {"message": "no response queued"}}, 0)
                if delay:
                    time.sleep(delay)
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            do_GET = _serve  # noqa: N815
            do_POST = _serve  # noqa: N815

            def log_message(self, *args: object) -> None:  # silence
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def queue(self, status: int, payload: object, delay: float = 0) -> None:
        self.responses.append((status, payload, delay))

    def __enter__(self) -> FakeAIServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def gemini_ok(text: str) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }


def openai_ok(text: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
