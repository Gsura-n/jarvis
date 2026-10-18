"""Routing policy: tier, then the first candidate whose device is up and has the model."""

import os

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

import pytest  # noqa: E402

from jarvis.config import settings  # noqa: E402
from jarvis.policy import decide, prefer_decomposer, walk_tier  # noqa: E402

MBP_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"
MINI_MODELS = (
    "llama3.2:3b",
    "nomic-embed-text",
    "nomic-embed-text:latest",
    "llama3.1:8b",
    "qwen2.5-coder:7b",
    "deepseek-r1:8b",
)


def snap(online: set[str], pulled: dict[str, tuple]) -> dict:
    return {
        "devices": {
            name: {"online": name in online, "pulled": pulled.get(name, ())}
            for name in ("mini", "macbook-pro")
        }
    }


BOTH = snap({"mini", "macbook-pro"}, {"mini": MINI_MODELS, "macbook-pro": (MBP_MODEL,)})
MINI = snap({"mini"}, {"mini": MINI_MODELS, "macbook-pro": (MBP_MODEL,)})


def test_laptop_online_serves_reasoning_large():
    decision = decide("reasoning", "", None, BOTH, settings.topology)
    assert decision.tier == "reasoning-large"
    assert decision.device == "macbook-pro"
    assert decision.deployment == "reasoning-large-mbp"
    assert decision.degraded is False
    design = decide("code+reasoning", "large", None, BOTH, settings.topology)
    assert design.deployment == "reasoning-large-mbp"
    assert prefer_decomposer("code+reasoning", None, design, force=False) is False


def test_laptop_offline_falls_back_to_mini():
    decision = decide("reasoning", "large", None, MINI, settings.topology)
    assert decision.device == "mini"
    assert decision.deployment == "reasoning-large-mini"
    assert decision.degraded is True
    assert prefer_decomposer("code+reasoning", None, decision, force=False) is True


def test_model_not_pulled_skips_the_laptop():
    missing = snap({"mini", "macbook-pro"}, {"mini": MINI_MODELS, "macbook-pro": ()})
    decision = decide("code+reasoning", "", None, missing, settings.topology)
    assert decision.tier == "reasoning-large"
    assert decision.device == "mini"
    assert decision.degraded is True
    assert "not pulled" in decision.reason


def test_explicit_model_beats_intent():
    decision = decide("reasoning", "large", "coding", BOTH, settings.topology)
    assert decision.tier == "code-small"
    assert decision.deployment == "code-small-mini"
    assert decision.device == "mini"
    assert decision.degraded is False
    assert prefer_decomposer("code", "coding", decision, force=False) is False
    forced = decide("general", "", None, BOTH, settings.topology)
    assert prefer_decomposer("general", None, forced, force=True) is True


def test_latest_tag_counts_as_the_untagged_embed_model():
    snapshot = snap({"mini"}, {"mini": ("nomic-embed-text:latest",)})
    decision = walk_tier("embed", snapshot, settings.topology, "embed")
    assert decision.deployment == "embed-mini"
    assert decision.device == "mini"
    assert decision.degraded is False


def test_normalize_model_keeps_an_existing_tag_and_marks_a_bare_name():
    from jarvis.registry import normalize_model

    assert normalize_model("qwen2.5-coder:7b") == "qwen2.5-coder:7b"
    assert normalize_model("nomic-embed-text") == "nomic-embed-text:latest"


def test_nothing_available_still_names_the_mini_fallback():
    decision = decide("general", "", None, {"devices": {}}, settings.topology)
    assert decision.degraded is True
    assert decision.device == "mini"
    assert decision.deployment == "general-small-mini"
    assert decision.reason


def _client(monkeypatch, offline=False, app=None):
    """Lifespan polls through this stub, so the test never touches the LAN."""

    async def fetch(url: str, timeout: float) -> dict:
        if "10.0.0.47" in url and offline:
            raise TimeoutError("asleep")
        if "10.0.0.47" in url:
            return {"models": [{"name": MBP_MODEL}]}
        return {"models": [{"name": name} for name in MINI_MODELS]}

    monkeypatch.setattr("jarvis.registry.fetch_json", fetch)
    from fastapi.testclient import TestClient

    if app is None:
        from jarvis.gateway import app

    return TestClient(app)


@pytest.fixture
def auth_headers(monkeypatch):
    """Shared key the gateway tests send. The route reads JARVIS_API_KEY on each request."""
    monkeypatch.setenv("JARVIS_API_KEY", "test-key")
    return {"Authorization": "Bearer test-key"}


def _post(client, content, model="jarvis", headers=None):
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}]},
        headers=headers,
    )


TRACE_HEADERS = {"X-Jarvis-Trace": "1"}


def _refuse(*_args, **_kwargs):
    raise AssertionError("this call should not happen")


def test_explicit_coding_skips_the_classifier(monkeypatch, auth_headers):
    monkeypatch.setattr("jarvis.gateway.classify", _refuse)
    monkeypatch.setattr("jarvis.gateway.chat", lambda model, *_a, **_k: f"from {model}")
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    with _client(monkeypatch) as client:
        response = _post(client, "print hello", model="coding", headers=auth_headers)
    assert response.status_code == 200
    assert "from code-small-mini" in response.json()["choices"][0]["message"]["content"]


def test_reasoning_trace_names_the_laptop_when_it_is_online(monkeypatch, auth_headers):
    monkeypatch.setattr("jarvis.gateway.chat", lambda *_a, **_k: "done")
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    with _client(monkeypatch) as client:
        response = _post(
            client, "which model answered that?", model="reasoning", headers=TRACE_HEADERS | auth_headers
        )
    text = response.json()["choices"][0]["message"]["content"]
    assert "Device: `macbook-pro`" in text
    assert "Deployment: `reasoning-large-mbp`" in text
    assert "Degraded: `false`" in text


def test_offline_laptop_answers_from_the_mini(monkeypatch, auth_headers):
    monkeypatch.setattr("jarvis.gateway.chat", lambda model, *_a, **_k: f"from {model}")
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    with _client(monkeypatch, offline=True) as client:
        response = _post(
            client, "which model answered that?", model="reasoning", headers=TRACE_HEADERS | auth_headers
        )
    text = response.json()["choices"][0]["message"]["content"]
    assert "from reasoning-large-mini" in text
    assert "Device: `mini`" in text
    assert "Degraded: `true`" in text


def test_degraded_design_and_build_uses_the_decomposer(monkeypatch, auth_headers):
    monkeypatch.setattr("jarvis.gateway.rag_query", lambda *_a, **_k: "")
    monkeypatch.setattr(
        "jarvis.gateway.classify",
        lambda *_a, **_k: (
            {"intent": "code+reasoning", "complexity": "complex", "needs_rag": False},
            "classifier",
        ),
    )
    monkeypatch.setattr(
        "jarvis.gateway.decompose_and_execute",
        lambda *_a, **_k: {"answer": "partial", "models_used": ["code-small-mini"], "subtasks": [{"id": 1}]},
    )
    monkeypatch.setattr("jarvis.gateway.chat", _refuse)
    with _client(monkeypatch, offline=True) as client:
        response = _post(client, "design and build a todo api", headers=auth_headers)
    text = response.json()["choices"][0]["message"]["content"]
    assert "partial" in text
    assert "Decomposed into 1 subtasks" in text


def test_model_list_starts_with_jarvis_and_includes_trace_and_decompose(monkeypatch, auth_headers):
    with _client(monkeypatch) as client:
        ids = [item["id"] for item in client.get("/v1/models", headers=auth_headers).json()["data"]]
    assert ids[0] == "jarvis"
    assert {"general", "coding", "reasoning", "jarvis-trace", "jarvis-decompose"} <= set(ids)


def test_parallel_subtasks_keep_the_mini_only_snapshot(monkeypatch):
    """Fails before the fix: ThreadPoolExecutor does not copy the pinned snapshot."""
    from types import MappingProxyType

    from jarvis.registry import (
        DeviceView,
        Registry,
        Snapshot,
        bind_snapshot,
        mini_only,
        reset_snapshot,
        set_registry,
    )
    from jarvis.task_decomposer import execute_tasks_parallel, routed

    mbp = "qwen3:30b-a3b-instruct-2507-q4_K_M"
    live = Snapshot(
        MappingProxyType(
            {
                "mini": DeviceView("mini", True, ("deepseek-r1:8b", "llama3.1:8b"), (), 0),
                "macbook-pro": DeviceView("macbook-pro", True, (mbp,), (), 0),
            }
        )
    )
    registry = Registry(settings.topology)
    registry.snapshot = live
    set_registry(registry)
    token = bind_snapshot(mini_only(live))
    seen = []

    def record(subtask, _completed, _rag):
        seen.append(routed("reasoning").deployment)
        return {"id": subtask["id"], "task": subtask["task"], "intent": "reasoning", "result": "ok"}

    monkeypatch.setattr("jarvis.task_decomposer.run_single_subtask", record)
    state = {
        "subtasks": [
            {"id": 1, "task": "think a", "intent": "reasoning", "depends_on": []},
            {"id": 2, "task": "think b", "intent": "reasoning", "depends_on": []},
        ],
        "rag_context": "",
    }
    try:
        execute_tasks_parallel(state)
    finally:
        reset_snapshot(token)
        set_registry(None)
    assert seen == ["reasoning-large-mini", "reasoning-large-mini"]
