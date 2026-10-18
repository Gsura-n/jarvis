"""ADR 0008: classifier module, chat() extra params and the complexity mapping."""

import json
import os

os.environ.setdefault("JARVIS_ENV_FILE", "/nonexistent/.jarvis.env")

import pytest  # noqa: E402

from jarvis import classify  # noqa: E402
from jarvis.config import settings  # noqa: E402
from jarvis.policy import decide, pick_tier  # noqa: E402
from tests.test_day3_policy import BOTH  # noqa: E402

GOOD = {"intent": "code", "complexity": "simple", "needs_rag": True}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (json.dumps(GOOD), GOOD),
        ("Sure! " + json.dumps(GOOD) + " Hope that helps.", GOOD),
        ("```json\n" + json.dumps(GOOD) + "\n```", GOOD),
        (json.dumps({**GOOD, "intent": "poetry"}), None),
        (json.dumps({**GOOD, "complexity": "huge"}), None),
        (json.dumps({**GOOD, "complexity": "large"}), None),
        (json.dumps({**GOOD, "needs_rag": "yes"}), None),
        (json.dumps({**GOOD, "needs_rag": 1}), None),
        (json.dumps({"intent": "code", "complexity": "simple"}), None),
        (json.dumps({**GOOD, "rag_reason": "x", "enhanced_prompt": "rewrite"}), GOOD),
        (json.dumps([GOOD]), None),
        ("not json at all", None),
        ("{broken", None),
        ("", None),
    ],
)
def test_parse_labels(raw, expected):
    assert classify.parse_labels(raw) == expected


def _body(messages):
    return classify.build_messages(messages)[1]["content"]


def test_build_messages_shows_earlier_turns_as_a_transcript():
    messages = [
        {"role": "user", "content": "Write a function to parse dates."},
        {"role": "assistant", "content": "def parse(): ..."},
        {"role": "user", "content": "now make it faster"},
    ]
    built = classify.build_messages(messages)
    assert built[0] == {"role": "system", "content": classify.SYSTEM}
    body = built[1]["content"]
    assert "User: Write a function to parse dates." in body
    assert "Assistant: def parse(): ..." in body
    assert body.endswith("Latest user request:\nnow make it faster")


def test_build_messages_clips_earlier_turns_and_the_latest():
    messages = [
        {"role": "user", "content": "e" * 1000},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "l" * 5000},
    ]
    body = _body(messages)
    assert "e" * 400 + " ..." in body
    assert "e" * 401 not in body
    assert body.endswith("l" * 2000)
    assert "l" * 2001 not in body


def test_build_messages_keeps_only_the_last_six_turns():
    messages = [{"role": "user", "content": f"turn-{i}"} for i in range(10)]
    body = _body(messages)
    assert "turn-3" not in body and "turn-4" in body and body.endswith("turn-9")


def test_build_messages_reads_list_content():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "first part"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "an answer"}]},
        {"role": "user", "content": [{"type": "text", "text": "explain this"}]},
    ]
    body = _body(messages)
    assert "User: first part" in body and "Assistant: an answer" in body
    assert body.endswith("explain this")


def test_build_messages_with_no_turns_is_empty_not_an_error():
    assert classify.build_messages([{"role": "system", "content": "s"}])[1]["content"] == ""


def test_classify_sends_schema_and_returns_labels():
    seen = {}

    def complete(alias, messages, **params):
        seen.update(alias=alias, params=params)
        return json.dumps(GOOD)

    labels, source = classify.classify([{"role": "user", "content": "hi"}], complete, "classify-mini")
    assert (labels, source) == (GOOD, "classifier")
    assert seen["alias"] == "classify-mini"
    assert seen["params"]["temperature"] == 0 and seen["params"]["max_tokens"] == 40
    fmt = seen["params"]["response_format"]
    assert fmt == {"type": "json_schema", "json_schema": {"name": "route", "schema": classify.SCHEMA}}


def test_classify_falls_back_when_the_model_call_raises():
    def complete(*_a, **_k):
        raise RuntimeError("model crashed")

    labels, source = classify.classify([{"role": "user", "content": "hi"}], complete)
    assert labels == classify.DEFAULT
    assert source == "fallback: model crashed"


def test_chat_passes_extra_params_to_litellm(monkeypatch):
    import jarvis.gateway as gateway

    captured = {}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": " ok "}}]}

    def post(url, headers, json, timeout):
        captured.update(json)
        return Reply()

    monkeypatch.setattr(gateway.requests, "post", post)
    fmt = {"type": "json_schema", "json_schema": {"name": "route", "schema": classify.SCHEMA}}
    assert gateway.chat("classify-mini", [], temperature=0, max_tokens=40, response_format=fmt) == "ok"
    assert captured["response_format"] == fmt
    assert captured["max_tokens"] == 40 and captured["temperature"] == 0


def test_complex_code_picks_the_large_tier_and_medium_stays_small():
    assert pick_tier("code", "complex", None)[0] == "code-large"
    assert pick_tier("code", "medium", None)[0] == "code-small"
    assert pick_tier("code", "simple", None)[0] == "code-small"
    assert decide("code", "complex", None, BOTH, settings.topology).tier == "code-large"
    assert decide("code", "medium", None, BOTH, settings.topology).tier == "code-small"
