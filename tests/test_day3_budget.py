"""Context window: num_ctx is configured, and oversized prompts are trimmed."""

import logging
from pathlib import Path

import yaml

from jarvis.budget import (
    _cost,
    answer_token_budget,
    assemble_messages,
    context_window,
    estimate_tokens,
    fit_input,
)

ROOT = Path(__file__).resolve().parents[1]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def test_every_litellm_deployment_has_num_ctx():
    for item in _yaml("configs/litellm.yaml")["model_list"]:
        num_ctx = item["litellm_params"].get("num_ctx")
        assert isinstance(num_ctx, int) and num_ctx >= 8192, item["model_name"]


def test_every_jarvis_deployment_has_num_ctx_and_a_litellm_entry():
    lite = {item["model_name"]: item for item in _yaml("configs/litellm.yaml")["model_list"]}
    for tier, candidates in _yaml("jarvis.yaml")["tiers"].items():
        for cand in candidates:
            assert isinstance(cand.get("num_ctx"), int) and cand["num_ctx"] >= 8192, tier
            assert cand["litellm"] in lite, cand["litellm"]
            assert lite[cand["litellm"]]["litellm_params"]["num_ctx"] == cand["num_ctx"]


def _chars(tokens: int) -> str:
    return "x" * (tokens * 4 - 3)


def test_answer_budget_is_at_most_half_the_context_window():
    config = _yaml("jarvis.yaml")
    for _tier, candidates in config["tiers"].items():
        for cand in candidates:
            budget = answer_token_budget(cand["litellm"], config)
            assert budget <= cand["num_ctx"] // 2, cand["litellm"]
    assert answer_token_budget("reasoning-large-mbp", config) == 6000
    assert context_window(config, "reasoning-large-mbp") == 32768
    assert answer_token_budget("reasoning-large-mini", config) == 6000
    assert context_window(config, "reasoning-large-mini") == 16384
    assert answer_token_budget("code-large-mini", config) == 4000
    assert context_window(config, "code-large-mini") == 8192
    assert answer_token_budget("code-small-mini", config) == 4000
    assert context_window(config, "code-small-mini") == 8192


def test_code_large_mini_keeps_history_beside_a_full_rag_block():
    config = _yaml("jarvis.yaml")
    num_ctx = context_window(config, "code-large-mini")
    max_tokens = answer_token_budget("code-large-mini", config)
    rag = _chars(1500)
    history = [
        {"role": "user", "content": _chars(1000)},
        {"role": "assistant", "content": _chars(1000)},
    ]
    assert estimate_tokens(rag) == 1500
    assert sum(estimate_tokens(item["content"]) for item in history) == 2000
    fitted = fit_input(
        system="s",
        history=history,
        rag=rag,
        handoff="",
        user="q",
        max_tokens=max_tokens,
        num_ctx=num_ctx,
    )
    assert fitted.trimmed == ()
    assert [item["content"] for item in fitted.history] == [item["content"] for item in history]


def test_oldest_history_is_dropped_before_newer_history():
    fitted = fit_input(
        system="sys",
        history=[
            {"role": "user", "content": "OLD" * 200},
            {"role": "assistant", "content": "NEW"},
        ],
        rag="R" * 80,
        handoff="",
        user="question",
        max_tokens=20,
        num_ctx=80,
    )
    # Pair trimming drops the assistant turn together with its user turn.
    assert fitted.history == ()
    assert fitted.trimmed == ("history",)
    assert _cost(fitted.system, list(fitted.history), fitted.rag, fitted.handoff, fitted.user) <= int(
        (80 - 20) * 0.9
    )


def test_oversized_input_is_trimmed_and_logged(caplog):
    caplog.set_level(logging.WARNING)
    messages = assemble_messages(
        system="You are a helper.",
        history=[{"role": "user", "content": "old " * 300}, {"role": "assistant", "content": "newer"}],
        rag="context " * 400,
        handoff="design " * 400,
        user="do the thing",
        max_tokens=40,
        num_ctx=160,
    )
    assert "trimmed input" in caplog.text
    assert messages[0]["content"] == "You are a helper."
    assert "<question>\ndo the thing\n</question>" in messages[-1]["content"]
    assert "context" not in messages[0]["content"]
    assert sum(estimate_tokens(item["content"]) for item in messages) <= 160


def test_system_message_does_not_change_when_rag_changes():
    common = dict(system="You are a helper.", history=[], handoff="", user="q", max_tokens=100, num_ctx=8000)
    first = assemble_messages(rag="alpha docs", **common)
    second = assemble_messages(rag="beta docs", **common)
    assert first[0]["content"] == second[0]["content"] == "You are a helper."
    assert "alpha docs" in first[-1]["content"]
    assert "beta docs" in second[-1]["content"]
    assert "<reference>" in first[-1]["content"]


def test_history_prefix_stays_stable_across_turns():
    common = dict(system="sys", rag="", handoff="", max_tokens=50, num_ctx=4000)
    first = assemble_messages(history=[], user="q1", **common)
    turn_one = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    second = assemble_messages(history=turn_one, user="q2", **common)
    turn_two = turn_one + [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    third = assemble_messages(history=turn_two, user="q3", **common)
    assert second[1:-1][: len(first[1:-1])] == first[1:-1]
    assert third[1:-1][: len(second[1:-1])] == second[1:-1]


def test_caps_cut_rag_and_halve_history_in_pairs():
    caps = {"system": 50, "history": 200, "rag": 30, "handoff": 30}
    history = []
    for _ in range(6):
        history.append({"role": "user", "content": _chars(20)})
        history.append({"role": "assistant", "content": _chars(20)})
    fitted = fit_input(
        system="s",
        history=history,
        rag=_chars(100),
        handoff="",
        user="q",
        max_tokens=10,
        num_ctx=10000,
        caps=caps,
    )
    assert estimate_tokens(fitted.rag) <= 30
    history_cost = sum(estimate_tokens(item["content"]) + 4 for item in fitted.history)
    assert history_cost <= caps["history"] // 2
    assert fitted.history[0]["role"] == "user"
    assert [item["role"] for item in fitted.history] == ["user", "assistant"] * (len(fitted.history) // 2)


def test_plain_question_when_there_is_no_context():
    messages = assemble_messages(
        system="sys", history=[], rag="", handoff="", user="just this", max_tokens=10, num_ctx=500
    )
    assert messages[-1]["content"] == "just this"


def test_input_stays_within_the_safety_margin_or_is_marked_over_budget():
    fitted = fit_input(
        system="sys",
        history=[
            {"role": "user", "content": _chars(100)},
            {"role": "assistant", "content": _chars(100)},
        ],
        rag=_chars(80),
        handoff=_chars(80),
        user="q",
        max_tokens=40,
        num_ctx=200,
    )
    limit = int((200 - 40) * 0.9)
    cost = _cost(fitted.system, list(fitted.history), fitted.rag, fitted.handoff, fitted.user)
    if fitted.history or fitted.rag or fitted.handoff:
        assert cost <= limit
    else:
        assert "over_budget" in fitted.trimmed

    only_prompt = fit_input(
        system=_chars(500),
        history=[],
        rag="",
        handoff="",
        user=_chars(500),
        max_tokens=10,
        num_ctx=100,
    )
    assert only_prompt.history == ()
    assert only_prompt.rag == ""
    assert only_prompt.handoff == ""
    assert "over_budget" in only_prompt.trimmed


def test_decomposer_handoff_and_rag_reach_the_user_message(monkeypatch):
    captured = {}

    def fake(_model, system, user, **_kwargs):
        captured["system"] = system
        captured["user"] = user
        return "done " * 20

    monkeypatch.setattr("jarvis.task_decomposer.traced_stream_call", fake)
    from jarvis.task_decomposer import run_single_subtask

    run_single_subtask(
        {"id": 2, "task": "write the function", "intent": "code", "implements": 1, "depends_on": [1]},
        {1: {"task": "design", "result": "Recommend a small module. " * 4}},
        "src/app.py defines main",
    )
    assert "<handoff>" in captured["user"]
    assert "Recommend a small module." in captured["user"]
    assert "<reference>" in captured["user"]
    assert "src/app.py defines main" in captured["user"]
    assert "Recommend a small module." not in captured["system"]
    assert "reference material from the codebase" in captured["system"]
