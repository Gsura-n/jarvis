"""History hygiene and traces through the gateway (ADR 0008 decision item 3)."""

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
from jarvis.traces import TraceStore  # noqa: E402
from tests.test_day3_budget import _turn_text, _yaml  # noqa: E402
from tests.test_day3_policy import _client, _refuse  # noqa: E402
from tests.test_day4_errors import _mock_stream, _sse  # noqa: E402

APPENDIX = f"{TRACE_MARK}\n- Intent: `general`\n- Labels from: `explicit model`"
LABELS = ({"intent": "general", "complexity": "simple", "needs_rag": False}, "classifier")
TRACE_ON = {"X-Jarvis-Trace": "1"}


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setenv("JARVIS_API_KEY", "test-key")
    return {"Authorization": "Bearer test-key"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    gateway._semaphores.clear()
    monkeypatch.setattr(gateway, "TRACES", TraceStore())
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


def _sse_contents(response):
    lines = [x[6:] for x in response.text.splitlines() if x.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(x)["choices"][0]["delta"]["content"] for x in lines[:-1]]


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


# ── follow-ups ────────────────────────────────────────────────────────────────
def _followup(answer, ask="which model answered?"):
    return [
        {"role": "user", "content": "What is a DAG?"},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": ask},
    ]


def test_follow_up_returns_the_stored_trace_with_no_model_call(monkeypatch, auth):
    sent = _record(monkeypatch, answer="A directed acyclic graph.")
    with _client(monkeypatch) as client:
        first = _ask(client, [{"role": "user", "content": "What is a DAG?"}], auth)
        assert "Model trace" not in _text(first)
        monkeypatch.setattr("jarvis.gateway.chat", _refuse)
        monkeypatch.setattr("jarvis.gateway.classify", _refuse)
        again = _ask(client, _followup(_text(first)), auth, model="jarvis")
    assert len(sent) == 1
    assert again.status_code == 200
    assert "Labels from: `explicit model`" in _text(again)
    assert "Deployment: `general-small-mini`" in _text(again)


def test_follow_up_is_served_as_a_stream_when_the_client_streams(monkeypatch, auth):
    _record(monkeypatch, answer="A directed acyclic graph.")
    with _client(monkeypatch) as client:
        first = _ask(client, [{"role": "user", "content": "What is a DAG?"}], auth)
        monkeypatch.setattr("jarvis.gateway.chat", _refuse)
        again = _ask(client, _followup(_text(first), "show trace"), auth, stream=True)
    assert "Labels from:" in "".join(_sse_contents(again))


def test_same_phrase_with_no_stored_trace_is_a_normal_question(monkeypatch, auth):
    sent = _record(monkeypatch, answer="I am a model.")
    with _client(monkeypatch) as client:
        response = _ask(client, _followup("Something Jarvis never said"), auth)
    assert [m for m, _ in sent] == ["general-small-mini"]
    assert sent[-1][1][-1] == {"role": "user", "content": "which model answered?"}
    assert _text(response) == "I am a model."


def test_which_model_should_i_use_is_a_normal_question_with_no_trace(monkeypatch, auth):
    _record(monkeypatch, answer="Use nomic-embed-text.")
    ask = [{"role": "user", "content": "which model should I use for embeddings?"}]
    with _client(monkeypatch) as client:
        response = _ask(client, ask, auth)
    assert _text(response) == "Use nomic-embed-text."


# ── trace on the current answer ───────────────────────────────────────────────
@pytest.mark.parametrize(("model", "extra"), [("jarvis-trace", None), ("general", TRACE_ON)])
def test_trace_is_appended_to_a_non_streamed_answer(monkeypatch, auth, model, extra):
    _record(monkeypatch)
    with _client(monkeypatch) as client:
        response = _ask(client, [{"role": "user", "content": "hi"}], auth, model, extra=extra)
    text = _text(response)
    assert text.startswith("the answer" + TRACE_MARK)
    assert "Labels from:" in text


@pytest.mark.parametrize(("model", "extra"), [("jarvis-trace", None), ("general", TRACE_ON)])
def test_trace_is_the_last_chunk_of_a_streamed_answer(monkeypatch, auth, model, extra):
    _mock_stream(monkeypatch, _upstream({"content": "hel"}, {"content": "lo"}))
    with _client(monkeypatch) as client:
        response = _ask(client, [{"role": "user", "content": "hi"}], auth, model, True, extra)
    chunks = _sse_contents(response)
    assert "".join(chunks[:-1]) == "hello"
    assert chunks[-1].startswith(TRACE_MARK) and "Labels from:" in chunks[-1]
    assert "llm_stream" in chunks[-1]  # built after the stream ended, so the timing is there


def test_no_trace_unless_asked(monkeypatch, auth):
    _mock_stream(monkeypatch, _upstream({"content": "hello"}))
    with _client(monkeypatch) as client:
        response = _ask(client, [{"role": "user", "content": "which model answered?"}], auth, "general", True)
    assert _sse_contents(response) == ["hello"]


def test_the_old_keyword_triggers_are_gone(monkeypatch, auth):
    assert not hasattr(gateway, "TRACE_TRIGGERS")
    _record(monkeypatch)
    with _client(monkeypatch) as client:
        response = _ask(client, [{"role": "user", "content": "why did you pick that?"}], auth)
    assert "Model trace" not in _text(response)


def test_a_streamed_answer_is_stored_for_a_later_follow_up(monkeypatch, auth):
    _mock_stream(monkeypatch, _upstream({"reasoning_content": "hmm"}, {"content": "hello"}))
    with _client(monkeypatch) as client:
        first = _ask(client, [{"role": "user", "content": "hi"}], auth, "general", True)
        answer = "".join(_sse_contents(first))
        monkeypatch.setattr("jarvis.gateway.chat", _refuse)
        again = _ask(client, _followup(answer), auth)
    assert "Labels from: `explicit model`" in _text(again)


# ── failures store nothing ────────────────────────────────────────────────────
def test_a_failed_answer_stores_no_trace(monkeypatch, auth):
    def broken(*_a, **_k):
        raise RuntimeError("down")

    monkeypatch.setattr("jarvis.gateway.chat", broken)
    with _client(monkeypatch) as client:
        response = _ask(client, [{"role": "user", "content": "hi"}], auth, extra=TRACE_ON)
    assert response.status_code == 502
    assert len(gateway.TRACES) == 0


def test_a_failed_stream_stores_and_sends_no_trace(monkeypatch, auth):
    _mock_stream(monkeypatch, _upstream(status=500))
    with _client(monkeypatch) as client:
        response = _ask(client, [{"role": "user", "content": "hi"}], auth, "general", True, TRACE_ON)
    chunks = _sse_contents(response)
    assert any("[Jarvis]" in c for c in chunks)
    assert not any("Model trace" in c for c in chunks)
    assert len(gateway.TRACES) == 0


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


def test_show_trace_twice_in_a_row_returns_the_same_trace(monkeypatch, auth):
    _record(monkeypatch, answer="A directed acyclic graph.")
    with _client(monkeypatch) as client:
        first = _ask(client, [{"role": "user", "content": "What is a DAG?"}], auth)
        monkeypatch.setattr("jarvis.gateway.chat", _refuse)
        monkeypatch.setattr("jarvis.gateway.classify", _refuse)
        convo = _followup(_text(first), "show trace")
        second = _ask(client, convo, auth, model="jarvis")
        convo += [{"role": "assistant", "content": _text(second)}, {"role": "user", "content": "show trace"}]
        third = _ask(client, convo, auth, model="jarvis")
    assert "Labels from:" in _text(second)
    assert _text(third) == _text(second)
