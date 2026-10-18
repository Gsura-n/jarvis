"""OpenAI message content is a string or a list of parts. Continue sends the list form."""

import os

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

import pytest  # noqa: E402

from jarvis.messages import text_of  # noqa: E402
from tests.test_day3_policy import _client, auth_headers  # noqa: E402,F401


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain", "plain"),
        ("", ""),
        ([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], "one\ntwo"),
        ([{"type": "text", "text": "keep"}, {"type": "image_url", "image_url": {"url": "x"}}], "keep"),
        ([{"type": "image_url", "image_url": {"url": "x"}}], ""),
        ([], ""),
        (["not a part", 3, None, {"type": "text", "text": "ok"}], "ok"),
        (None, ""),
        ({"type": "text", "text": "a dict is not a list of parts"}, ""),
        (42, ""),
    ],
)
def test_text_of(content, expected):
    assert text_of(content) == expected


def test_list_content_request_returns_200(monkeypatch, auth_headers):  # noqa: F811
    monkeypatch.setattr("jarvis.gateway.chat", lambda model, *_a, **_k: f"from {model}")
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    body = {
        "model": "general",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "explain this"}]}],
    }
    with _client(monkeypatch) as client:
        response = client.post("/v1/chat/completions", json=body, headers=auth_headers)
    assert response.status_code == 200
    assert "from general-small-mini" in response.json()["choices"][0]["message"]["content"]


def test_earlier_list_turns_reach_the_model_as_text(monkeypatch, auth_headers):  # noqa: F811
    sent = []

    def fake_chat(model, messages, *_a, **_k):
        sent.append(messages)
        return "ok"

    monkeypatch.setattr("jarvis.gateway.chat", fake_chat)
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    body = {
        "model": "general",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "first question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "follow up"}]},
        ],
    }
    with _client(monkeypatch) as client:
        response = client.post("/v1/chat/completions", json=body, headers=auth_headers)
    assert response.status_code == 200
    history = [m for m in sent[0] if m["role"] in ("user", "assistant")][:-1]
    assert history == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
