"""Thinking is shown to the user as a <think> block while a reasoning model streams."""

import os

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

import json  # noqa: E402

import anyio  # noqa: E402
import httpx  # noqa: E402
import pytest  # noqa: E402

import jarvis.gateway as gateway  # noqa: E402
from jarvis.gateway import device_semaphore, stream_tokens  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_semaphores():
    gateway._semaphores.clear()
    yield
    gateway._semaphores.clear()


def _sse(delta: dict) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()


def _tokens(monkeypatch, *deltas: dict) -> list[str]:
    body = b"".join(_sse(d) for d in deltas)
    monkeypatch.setattr(
        gateway,
        "_stream_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body))
        ),
    )

    async def run():
        out = [t async for t in stream_tokens("reasoning-large-mbp", "macbook-pro", [], 32)]
        assert device_semaphore("macbook-pro").locked() is False
        return out

    return anyio.run(run)


def test_reasoning_deltas_are_wrapped_in_a_think_block(monkeypatch):
    tokens = _tokens(
        monkeypatch,
        {"reasoning_content": "let me "},
        {"reasoning_content": "think"},
        {"content": "The answer"},
        {"content": " is 4."},
    )
    assert "".join(tokens) == "<think>\nlet me think\n</think>\n\nThe answer is 4."


def test_the_reasoning_key_is_read_too(monkeypatch):
    tokens = _tokens(monkeypatch, {"reasoning": "hmm"}, {"content": "ok"})
    assert "".join(tokens) == "<think>\nhmm\n</think>\n\nok"


def test_a_stream_that_ends_while_thinking_closes_the_block(monkeypatch):
    assert "".join(_tokens(monkeypatch, {"reasoning_content": "still going"})) == (
        "<think>\nstill going\n</think>\n\n"
    )
