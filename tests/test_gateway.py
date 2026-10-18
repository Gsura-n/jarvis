"""
Gateway behaviour with the models mocked out: auth, routing, history, traces, streaming.
No network, no Ollama, no secrets.
"""

import json
import os

import pytest

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from jarvis import classify, gateway, llm  # noqa: E402

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeModels:
    """Records every call; the classifier alias returns labels, everything else returns an answer."""

    def __init__(self, labels=None, answer="Here is a long enough answer from the fake model."):
        self.labels = labels or {"intent": "general", "complexity": "simple", "needs_rag": False}
        self.answer = answer
        self.calls = []

    async def complete(self, alias, messages, **params):
        self.calls.append((alias, messages, params))
        if alias == "classifier":
            return json.dumps(self.labels)
        return self.answer

    async def stream(self, alias, messages, **params):
        self.calls.append((alias, messages, params))
        yield "reasoning", "thinking about it"
        yield "content", "Hello "
        yield "content", "there."

    def aliases(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def models(monkeypatch):
    fake = FakeModels()
    monkeypatch.setattr(gateway, "API_KEY", KEY)
    monkeypatch.setattr(llm, "complete", fake.complete)
    monkeypatch.setattr(llm, "stream", fake.stream)
    monkeypatch.setattr(gateway, "rag_query", lambda q, top_k=4: "")
    monkeypatch.setattr(gateway, "TRACES", gateway.TraceStore())
    return fake


@pytest.fixture
def client(models):
    return TestClient(gateway.app)


def chat(client, messages, model="jarvis", stream=False, headers=AUTH):
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": messages, "stream": stream},
        headers=headers,
    )


def content(r):
    return r.json()["choices"][0]["message"]["content"]


# ── auth ──────────────────────────────────────────────────────────────────────
def test_health_needs_no_key(client):
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY}])
def test_v1_rejects_missing_or_wrong_key(client, headers):
    assert client.get("/v1/models", headers=headers).status_code == 401
    assert chat(client, [{"role": "user", "content": "hi"}], headers=headers).status_code == 401


def test_v1_accepts_the_key(client):
    ids = [m["id"] for m in client.get("/v1/models", headers=AUTH).json()["data"]]
    assert {"jarvis", "coding", "reasoning", "general"} <= set(ids)


def test_unset_key_fails_closed(client, monkeypatch):
    monkeypatch.setattr(gateway, "API_KEY", "")
    assert client.get("/v1/models", headers=AUTH).status_code == 503


def test_no_wildcard_cors():
    assert not any("CORSMiddleware" in str(m.cls) for m in gateway.app.user_middleware)


# ── routing ───────────────────────────────────────────────────────────────────
def test_explicit_model_skips_classifier(client, models):
    r = chat(client, [{"role": "user", "content": "reverse a list"}], model="coding")
    assert r.status_code == 200
    assert models.aliases() == ["coding"]


def test_auto_model_uses_classifier_labels(client, models):
    models.labels = {"intent": "reasoning", "complexity": "medium", "needs_rag": False}
    chat(client, [{"role": "user", "content": "should I use Kafka or SQS?"}])
    assert models.aliases() == ["classifier", "reasoning"]
    params = models.calls[0][2]
    assert params["response_format"]["type"] == "json_schema"


def test_downstream_model_gets_the_users_words_not_a_rewrite(client, models):
    chat(client, [{"role": "user", "content": "exact words, keep the constraint: no pandas"}])
    downstream = models.calls[-1][1]
    assert downstream[-1] == {"role": "user", "content": "exact words, keep the constraint: no pandas"}


def test_bad_classifier_reply_falls_back_to_general(client, models):
    models.labels = {"intent": "poetry", "complexity": "simple", "needs_rag": False}
    r = chat(client, [{"role": "user", "content": "hi"}])
    assert r.status_code == 200
    assert models.aliases() == ["classifier", "general"]


def test_upstream_error_is_a_502_not_an_empty_answer(client, monkeypatch):
    async def boom(alias, messages, **params):
        raise llm.LLMError("coding: HTTP 500: model crashed")

    monkeypatch.setattr(llm, "complete", boom)
    r = chat(client, [{"role": "user", "content": "x"}], model="coding")
    assert r.status_code == 502
    assert "model crashed" in r.json()["error"]["message"]


# ── history ───────────────────────────────────────────────────────────────────
def test_history_is_passed_and_traces_are_stripped(client, models):
    messages = [
        {"role": "system", "content": "Client system prompt."},
        {"role": "user", "content": "What is a DAG?"},
        {
            "role": "assistant",
            "content": "<think>hmm</think>A directed acyclic graph." + gateway.TRACE_MARK + "\nx",
        },
        {"role": "user", "content": [{"type": "text", "text": "Give an example."}]},
    ]
    chat(client, messages, model="general")
    sent = models.calls[-1][1]
    assert sent[0]["role"] == "system" and "Client system prompt." in sent[0]["content"]
    assert sent[1:] == [
        {"role": "user", "content": "What is a DAG?"},
        {"role": "assistant", "content": "A directed acyclic graph."},
        {"role": "user", "content": "Give an example."},
    ]


def test_history_budget_drops_oldest_turns_but_keeps_latest():
    msgs = [{"role": "user", "content": "a" * 50}, {"role": "assistant", "content": "b" * 50}] * 5
    msgs.append({"role": "user", "content": "latest " * 100})
    convo = gateway.build_conversation(msgs, "sys", budget_chars=120)
    assert convo[-1]["content"].startswith("latest")
    assert convo[1]["role"] == "user"
    assert sum(len(m["content"]) for m in convo[1:-1]) <= 120


def test_classifier_sees_earlier_turns():
    msgs = [
        {"role": "user", "content": "Write a Python function to parse dates."},
        {"role": "assistant", "content": "def parse(): ..."},
        {"role": "user", "content": "now make it faster"},
    ]
    body = classify.build_messages(msgs)[1]["content"]
    assert "parse dates" in body and body.rstrip().endswith("now make it faster")


@pytest.mark.parametrize(
    "raw,ok",
    [
        ('{"intent": "code", "complexity": "simple", "needs_rag": true}', True),
        ('Sure: {"intent": "code", "complexity": "simple", "needs_rag": false}', True),
        ('{"intent": "code", "complexity": "simple"}', False),
        ('{"intent": "code", "complexity": "huge", "needs_rag": true}', False),
        ('{"intent": "code", "complexity": "simple", "needs_rag": "yes"}', False),
        ("not json", False),
    ],
)
def test_parse_labels(raw, ok):
    assert (classify.parse_labels(raw) is not None) is ok


# ── traces ────────────────────────────────────────────────────────────────────
def test_trace_followup_returns_previous_trace_without_a_model_call(client, models):
    first = [{"role": "user", "content": "What is a DAG?"}]
    answer = content(chat(client, first, model="general"))
    models.calls.clear()

    follow = first + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "Which model answered?"},
    ]
    r = chat(client, follow)
    assert models.calls == []
    text = content(r)
    assert "general (llama3.1:8b)" in text and "explicit model" in text


def test_trace_followup_without_stored_trace_is_a_normal_question(client, models):
    follow = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "an answer the gateway never produced"},
        {"role": "user", "content": "which model answered?"},
    ]
    chat(client, follow)
    assert models.aliases() == ["classifier", "general"]


def test_jarvis_trace_model_appends_trace(client, models):
    text = content(chat(client, [{"role": "user", "content": "hi"}], model="jarvis-trace"))
    assert "**Model trace:**" in text


# ── streaming ─────────────────────────────────────────────────────────────────
def test_stream_wraps_thinking_and_ends_with_done(client, models):
    r = chat(client, [{"role": "user", "content": "hi"}], model="reasoning", stream=True)
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    text = "".join(
        json.loads(ln[6:])["choices"][0]["delta"].get("content", "")
        for ln in lines[:-1]
        if ln != "data: [DONE]"
    )
    assert text.startswith("<think>\nthinking about it\n</think>")
    assert text.endswith("Hello there.")


def test_stream_error_is_visible(client, monkeypatch):
    async def broken(alias, messages, **params):
        raise llm.LLMError("reasoning: HTTP 404: model not found")
        yield  # pragma: no cover

    monkeypatch.setattr(llm, "stream", broken)
    r = chat(client, [{"role": "user", "content": "hi"}], model="reasoning", stream=True)
    assert "model not found" in r.text and r.text.rstrip().endswith("data: [DONE]")


def test_decomposer_path_streams_as_one_chunk(client, models, monkeypatch):
    monkeypatch.setattr(
        gateway,
        "decompose_and_execute",
        lambda prompt, rag: {"answer": "Built it.", "subtasks": [1, 2], "models_used": ["general"]},
    )
    r = chat(
        client, [{"role": "user", "content": "design and build X"}], model="jarvis-decompose", stream=True
    )
    assert "Built it." in r.text and r.text.rstrip().endswith("data: [DONE]")


# ── event loop ────────────────────────────────────────────────────────────────
def test_health_answers_while_a_slow_model_call_is_running(models, monkeypatch):
    import asyncio

    import httpx

    async def slow(alias, messages, **params):
        await asyncio.sleep(0.5)
        return "slow answer that is long enough"

    monkeypatch.setattr(llm, "complete", slow)

    async def run():
        transport = httpx.ASGITransport(app=gateway.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            req = asyncio.create_task(
                c.post(
                    "/v1/chat/completions",
                    json={"model": "general", "messages": [{"role": "user", "content": "x"}]},
                    headers=AUTH,
                )
            )
            await asyncio.sleep(0.05)  # let the slow request get going first
            h = await c.get("/health")
            health_s = loop.time() - t0  # a blocked loop would make this >= 0.5
            await req
            return h.status_code, health_s

    status, health_s = asyncio.run(run())
    assert status == 200 and health_s < 0.3
