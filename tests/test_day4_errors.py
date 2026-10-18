"""Visible errors, bounded retries, a classifier timeout, and thinking shown then kept out of history."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

import anyio  # noqa: E402
import httpx  # noqa: E402
import pytest  # noqa: E402
import requests  # noqa: E402

import jarvis.gateway as gateway  # noqa: E402
from jarvis import classify  # noqa: E402
from jarvis.gateway import device_semaphore, stream_tokens  # noqa: E402
from tests.test_day3_policy import _client, _post, auth_headers  # noqa: E402,F401

SECRET = "sk-test-secret-value"
LABELS = json.dumps({"intent": "code", "complexity": "simple", "needs_rag": False})


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    gateway._semaphores.clear()
    monkeypatch.setattr(gateway, "LITELLM_KEY", SECRET)
    monkeypatch.setattr(gateway.time, "sleep", lambda _s: None)
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    yield
    gateway._semaphores.clear()


# ── helpers ───────────────────────────────────────────────────────────────────
def _sse(delta: dict) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()


def _mock_stream(monkeypatch, handler):
    monkeypatch.setattr(
        gateway, "_stream_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def _tokens(deployment="reasoning-large-mbp", device="macbook-pro"):
    async def run():
        out = [t async for t in stream_tokens(deployment, device, [], 32)]
        assert device_semaphore(device).locked() is False
        return out

    return anyio.run(run)


def _reply(status: int, body: dict) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    r.headers["Content-Type"] = "application/json"
    return r


def _ok(text: str) -> requests.Response:
    return _reply(200, {"choices": [{"message": {"content": text}}]})


class FakePost:
    """Stands in for requests.post. script maps a model alias to a list of replies (the last one repeats)."""

    def __init__(self, script: dict[str, list]):
        self.script = script
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        model = json["model"]
        self.calls.append((model, {"payload": json, "timeout": timeout}))
        seen = sum(1 for m, _ in self.calls if m == model)
        replies = self.script[model]
        reply = replies[min(seen, len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def models(self):
        return [m for m, _ in self.calls]


# ── streaming errors ──────────────────────────────────────────────────────────
def test_stream_500_ends_with_a_jarvis_line(monkeypatch):
    body = {"error": {"message": "model crashed"}}
    _mock_stream(monkeypatch, lambda _r: httpx.Response(500, json=body))
    tokens = _tokens()
    assert len(tokens) == 1
    assert tokens[0].startswith("\n\n[Jarvis] reasoning-large-mbp: ")
    assert "500" in tokens[0] and "model crashed" in tokens[0]


def test_stream_error_reason_is_short_and_never_has_the_key(monkeypatch):
    body = {"error": {"message": f"bad key {SECRET} " + "x" * 1000}}
    _mock_stream(monkeypatch, lambda _r: httpx.Response(401, json=body))
    (line,) = _tokens()
    assert SECRET not in line
    assert len(line.split("[Jarvis] reasoning-large-mbp: ", 1)[1]) <= 300


def test_stream_transport_error_mid_stream_keeps_tokens_then_reports(monkeypatch):
    class Dies(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _sse({"content": "partial"})
            raise httpx.ReadError("connection reset")

    _mock_stream(monkeypatch, lambda _r: httpx.Response(200, stream=Dies()))
    tokens = _tokens()
    assert tokens[0] == "partial"
    assert tokens[-1].startswith("\n\n[Jarvis] reasoning-large-mbp: ")


def test_stream_500_through_the_route_ends_with_done_and_frees_the_lock(monkeypatch, auth_headers):  # noqa: F811
    _mock_stream(monkeypatch, lambda _r: httpx.Response(500, json={"error": {"message": "boom"}}))
    body = {"model": "reasoning", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    with _client(monkeypatch) as client:
        response = client.post("/v1/chat/completions", json=body, headers=auth_headers)
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    text = "".join(json.loads(ln[6:])["choices"][0]["delta"]["content"] for ln in lines[:-1])
    assert "[Jarvis] reasoning-large-mbp:" in text and "boom" in text
    assert device_semaphore("macbook-pro").locked() is False


# ── non-streaming errors and retries ──────────────────────────────────────────
def _chat_request(monkeypatch, headers, model="general", content="write a thing"):
    with _client(monkeypatch) as client:
        return _post(client, content, model=model, headers=headers)


def test_400_on_the_answer_is_a_502_with_the_message_and_no_retry(monkeypatch, auth_headers):  # noqa: F811
    err = {"error": {"message": f"context too long {SECRET}"}}
    post = FakePost({"general-small-mini": [_reply(400, err)]})
    monkeypatch.setattr(gateway.requests, "post", post)
    response = _chat_request(monkeypatch, auth_headers)
    assert response.status_code == 502
    message = response.json()["error"]["message"]
    assert "context too long" in message and "400" in message
    assert SECRET not in response.text
    assert len(message) <= 340
    assert post.models() == ["general-small-mini"]


def test_503_on_the_answer_is_retried_once(monkeypatch, auth_headers):  # noqa: F811
    post = FakePost({"general-small-mini": [_reply(503, {"error": {"message": "loading"}}), _ok("done")]})
    monkeypatch.setattr(gateway.requests, "post", post)
    response = _chat_request(monkeypatch, auth_headers)
    assert response.status_code == 200
    assert "done" in response.json()["choices"][0]["message"]["content"]
    assert post.models() == ["general-small-mini"] * 2


def test_a_connection_error_is_retried_once_then_reported(monkeypatch, auth_headers):  # noqa: F811
    post = FakePost({"general-small-mini": [requests.ConnectionError("refused")]})
    monkeypatch.setattr(gateway.requests, "post", post)
    response = _chat_request(monkeypatch, auth_headers)
    assert response.status_code == 502
    assert post.models() == ["general-small-mini"] * 2


def test_classifier_failure_falls_back_but_answer_failure_is_a_502(monkeypatch, auth_headers):  # noqa: F811
    post = FakePost(
        {
            "classify-mini": [_reply(500, {"error": {"message": "classifier down"}})],
            "general-small-mini": [_reply(400, {"error": {"message": "bad request"}})],
        }
    )
    monkeypatch.setattr(gateway.requests, "post", post)
    response = _chat_request(monkeypatch, auth_headers, model="jarvis", content="hello")
    assert response.status_code == 502
    assert "bad request" in response.json()["error"]["message"]
    assert post.models() == ["classify-mini", "general-small-mini"]


def test_classifier_failure_still_answers_with_intent_general(monkeypatch, auth_headers):  # noqa: F811
    post = FakePost(
        {"classify-mini": [_reply(500, {"error": {"message": "down"}})], "general-small-mini": [_ok("hi!")]}
    )
    monkeypatch.setattr(gateway.requests, "post", post)
    response = _chat_request(monkeypatch, auth_headers, model="jarvis-trace", content="hello")
    text = response.json()["choices"][0]["message"]["content"]
    assert response.status_code == 200
    assert "hi!" in text and "Intent: `general`" in text and "fallback" in text
    assert post.models().count("classify-mini") == 1


# ── classifier timeout ────────────────────────────────────────────────────────
class _Slow(BaseHTTPRequestHandler):
    classifier_calls = 0

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if payload["model"] == "classify-mini":
            type(self).classifier_calls += 1
            threading.Event().wait(1.0)
            text = LABELS
        else:
            text = "plain answer"
        data = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass  # the client already gave up

    def log_message(self, *_args):
        pass


def test_a_hung_classifier_times_out_once_and_falls_back(monkeypatch, auth_headers):  # noqa: F811
    _Slow.classifier_calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Slow)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "LITELLM_BASE", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setattr(classify, "CLASSIFY_TIMEOUT", 0.3)
    try:
        response = _chat_request(monkeypatch, auth_headers, model="jarvis-trace", content="hello")
    finally:
        server.shutdown()
    text = response.json()["choices"][0]["message"]["content"]
    assert response.status_code == 200
    assert "plain answer" in text and "Intent: `general`" in text
    assert "fallback: classifier timeout" in text
    assert _Slow.classifier_calls == 1


def test_chat_takes_a_timeout_and_defaults_to_600(monkeypatch):
    post = FakePost({"m": [_ok("x")]})
    monkeypatch.setattr(gateway.requests, "post", post)
    gateway.chat("m", [])
    gateway.chat("m", [], timeout=15)
    assert [c["timeout"] for _m, c in post.calls] == [600, 15]
