"""Every gateway route except GET /health requires JARVIS_API_KEY. CORS lists origins."""

import asyncio
import logging
import os

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

from tests.test_day3_policy import _client  # noqa: E402

KEY = "test-key"
BEARER = {"Authorization": f"Bearer {KEY}"}
GETS = ("/devices", "/v1/models", "/v1/openapi.json", "/v1/api/config")
CHAT = {
    "model": "general",
    "messages": [{"role": "user", "content": "hi"}],
}


def _chat(client, headers=None):
    return client.post("/v1/chat/completions", json=CHAT, headers=headers)


def test_protected_routes_reject_a_missing_key(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", KEY)
    with _client(monkeypatch) as client:
        for path in GETS:
            assert client.get(path).status_code == 401
        assert _chat(client).status_code == 401


def test_protected_routes_reject_a_wrong_key(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", KEY)
    wrong = {"Authorization": "Bearer wrong"}
    with _client(monkeypatch) as client:
        for path in GETS:
            assert client.get(path, headers=wrong).status_code == 401
        assert _chat(client, wrong).status_code == 401


def test_protected_routes_allow_the_right_key(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", KEY)
    monkeypatch.setattr("jarvis.gateway.chat", lambda model, *_a, **_k: f"from {model}")
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    with _client(monkeypatch) as client:
        for path in GETS:
            assert client.get(path, headers=BEARER).status_code == 200
        response = _chat(client, BEARER)
    assert response.status_code == 200
    assert "from general-small-mini" in response.json()["choices"][0]["message"]["content"]


def test_health_needs_no_key(monkeypatch):
    monkeypatch.delenv("JARVIS_API_KEY", raising=False)
    with _client(monkeypatch) as client:
        assert client.get("/health").status_code == 200


def test_empty_key_rejects_a_presented_header(monkeypatch):
    monkeypatch.delenv("JARVIS_API_KEY", raising=False)
    with _client(monkeypatch) as client:
        for path in GETS:
            assert client.get(path, headers=BEARER).status_code == 401
        assert _chat(client, BEARER).status_code == 401


def test_startup_logs_when_the_api_key_is_empty(monkeypatch, caplog):
    monkeypatch.delenv("JARVIS_API_KEY", raising=False)
    with caplog.at_level(logging.ERROR, logger="jarvis.gateway"):
        with _client(monkeypatch):
            pass
    assert "every protected route will return 401" in caplog.text


def test_cors_origins_split_on_commas(monkeypatch):
    from jarvis.config import cors_origins

    monkeypatch.setenv("JARVIS_CORS_ORIGINS", " http://openwebui.local:8080 , http://continue.local ")
    assert cors_origins() == ["http://openwebui.local:8080", "http://continue.local"]
    monkeypatch.setenv("JARVIS_CORS_ORIGINS", "")
    assert cors_origins() == []


def test_cors_preflight_only_echoes_listed_origins(monkeypatch):
    monkeypatch.setenv("JARVIS_CORS_ORIGINS", "http://openwebui.local:8080")
    monkeypatch.delenv("JARVIS_API_KEY", raising=False)
    from jarvis.gateway import create_app

    with _client(monkeypatch, app=create_app()) as client:
        blocked = client.options(
            "/v1/models",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
        )
        allowed = client.options(
            "/v1/models",
            headers={"Origin": "http://openwebui.local:8080", "Access-Control-Request-Method": "GET"},
        )
    assert "access-control-allow-origin" not in blocked.headers
    assert allowed.headers.get("access-control-allow-origin") == "http://openwebui.local:8080"
    assert allowed.headers.get("access-control-allow-credentials") is None


def test_builtin_docs_are_not_mounted(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", KEY)
    with _client(monkeypatch) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
            assert client.get(path, headers=BEARER).status_code == 404


def test_bearer_scheme_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", KEY)
    with _client(monkeypatch) as client:
        assert client.get("/v1/models", headers={"Authorization": f"bearer {KEY}"}).status_code == 200
        assert client.get("/v1/models", headers={"Authorization": f"BEARER {KEY}"}).status_code == 200
        assert client.get("/v1/models", headers={"Authorization": f"Basic {KEY}"}).status_code == 401


def test_non_ascii_token_is_401(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", KEY)
    from jarvis.gateway import app

    async def status() -> int:
        seen = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            seen.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/v1/models",
                "raw_path": b"/v1/models",
                "query_string": b"",
                "headers": [(b"authorization", b"Bearer caff\xff")],
                "client": ("127.0.0.1", 50000),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )
        return seen[0]["status"]

    assert asyncio.run(status()) == 401
