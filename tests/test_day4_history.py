"""History hygiene through the gateway: no replayed trace appendix, client system messages after ours."""

import json
import os

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

import httpx  # noqa: E402
import pytest  # noqa: E402

import jarvis.gateway as gateway  # noqa: E402
from jarvis.budget import (  # noqa: E402
    answer_token_budget,
    assemble_messages,
    context_window,
    fit_input,
    token_caps,
)
from jarvis.messages import TRACE_MARK  # noqa: E402
from tests.test_day3_budget import _turn_text, _yaml  # noqa: E402
from tests.test_day3_policy import _client  # noqa: E402
from tests.test_day4_errors import _mock_stream, _sse  # noqa: E402

APPENDIX = f"{TRACE_MARK}\n- Intent: `general`\n- Labels from: `explicit model`"
LABELS = ({"intent": "general", "complexity": "simple", "needs_rag": False}, "classifier")


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", "test-key")
    return {"Authorization": "Bearer test-key"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    gateway._semaphores.clear()
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    monkeypatch.setattr("jarvis.gateway.classify", lambda *_a, **_k: LABELS)
    yield
    gateway._semaphores.clear()


def _record(monkeypatch, answer="the answer"):
    sent = []

    def fake(model, messages, *_a, **_k):
        sent.append((model, messages))
        return answer

    monkeypatch.setattr("jarvis.gateway.chat", fake)
    return sent


def _ask(client, messages, auth, model="general", stream=False, extra=None):
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": messages, "stream": stream},
        headers={**auth, **(extra or {})},
    )


def _text(response):
    return response.json()["choices"][0]["message"]["content"]


def _upstream(*deltas, status=200):
    def handler(_request):
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "boom"}})
        return httpx.Response(200, content=b"".join(_sse(d) for d in deltas) + b"data: [DONE]\n\n")

    return handler


# ── history ───────────────────────────────────────────────────────────────────
def test_assistant_turns_reach_the_model_without_the_trace_appendix(monkeypatch, auth):
    sent = _record(monkeypatch)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1" + APPENDIX},
        {"role": "user", "content": "q2"},
    ]
    with _client(monkeypatch) as client:
        assert _ask(client, messages, auth).status_code == 200
    history = sent[-1][1][1:-1]
    assert history == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]


@pytest.mark.parametrize("content", ["Answer in French.", [{"type": "text", "text": "Answer in French."}]])
def test_client_system_message_follows_ours(monkeypatch, auth, content):
    sent = _record(monkeypatch)
    messages = [{"role": "system", "content": content}, {"role": "user", "content": "hi"}]
    with _client(monkeypatch) as client:
        assert _ask(client, messages, auth).status_code == 200
    system = sent[-1][1][0]
    assert system["role"] == "system"
    assert system["content"].startswith("You are a helpful expert assistant.")
    assert system["content"].endswith("\n\nAnswer in French.")
    assert sent[-1][1].count(system) == 1


def test_streamed_answers_get_the_client_system_message_too(monkeypatch, auth):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _upstream({"content": "ok"})(request)

    _mock_stream(monkeypatch, handler)
    messages = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]
    with _client(monkeypatch) as client:
        _ask(client, messages, auth, stream=True)
    system = captured["body"]["messages"][0]["content"]
    assert system.startswith("You are a helpful expert assistant.") and system.endswith("\n\nBe terse.")


# ── block trimming with trace appendices in the replay ────────────────────────
def test_history_prefix_stays_stable_when_assistant_turns_carry_a_trace():
    config = _yaml("jarvis.yaml")
    deployment = "general-small-mini"
    block = int(config["budgets"].get("history_trim_block", 4))
    kept_by_turn = {}
    for traced in (False, True):
        conversation, previous, misses = [], None, []
        for turn in range(1, 21):
            question = _turn_text(f"q{turn:02d}", 150)
            history = gateway.prior_turns(conversation + [{"role": "user", "content": question}], question)
            fitted = fit_input(
                system="You are a helper.",
                history=history,
                rag="",
                handoff="",
                user=question,
                max_tokens=answer_token_budget(deployment, config),
                num_ctx=context_window(config, deployment),
                caps=token_caps(config),
                block=block,
            )
            kept = list(fitted.history)
            kept_by_turn[(traced, turn)] = kept
            if previous is not None and kept[: len(previous)] != previous:
                misses.append(turn)
            previous = kept
            answer = _turn_text(f"a{turn:02d}", 300) + (APPENDIX if traced else "")
            conversation += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        assert misses
        for begin in range(misses[0], 21, block):
            assert len([t for t in misses if begin <= t < begin + block]) <= 1, misses
    assert all(kept_by_turn[(True, t)] == kept_by_turn[(False, t)] for t in range(1, 21))


def test_assemble_messages_is_untouched_by_the_appendix():
    clean = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    common = dict(system="s", rag="", handoff="", user="q2", max_tokens=50, num_ctx=4000)
    traced = gateway.prior_turns(
        [clean[0], {"role": "assistant", "content": "a" + APPENDIX}, {"role": "user", "content": "q2"}], "q2"
    )
    assert assemble_messages(history=traced, **common) == assemble_messages(history=clean, **common)
